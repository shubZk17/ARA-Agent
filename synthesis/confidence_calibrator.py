"""
synthesis/confidence_calibrator.py — Evidence-Weighted Confidence Scoring
==========================================================================

WHY THIS EXISTS:
    An investment recommendation without a CALIBRATED confidence score
    is dangerous. "Buy NVDA" means nothing without knowing:
    - How much evidence do we have?
    - How reliable are the sources?
    - Are sources contradicting each other?
    - Is the data fresh or stale?
    - Did we gather all the data we needed?

    This calibrator explicitly answers all of those questions and
    produces a transparent confidence score that EXPLAINS itself.

CALIBRATION PHILOSOPHY:
    - START at 0.5 (baseline: moderate confidence).
    - BOOST for: multi-source agreement, high-tier sources, complete data.
    - PENALIZE for: conflicts, missing data, stale evidence, low-tier sources.
    - NEVER exceed 0.95 (epistemic humility — markets are uncertain).
    - NEVER go below 0.10 (we always have SOME information).

HOW IT CONNECTS:
    - Consumes: FinancialSnapshot, SentimentProfile, RiskAssessment,
                MisalignmentSignal, + Phase 2 state fields (conflicts,
                evidence counts, etc.)
    - Produces: ConfidenceScore (synthesis/schemas.py).
    - Used by: engine.py, report_generator.py.
"""

from __future__ import annotations

from synthesis.schemas import (
    ConfidenceScore,
    FinancialSnapshot,
    MisalignmentSignal,
    RiskAssessment,
    RiskSeverity,
    SentimentProfile,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceCalibrator:
    """
    Produces calibrated confidence scores based on evidence quality,
    source reliability, data completeness, and internal consistency.
    """

    def calibrate(
        self,
        financial: FinancialSnapshot,
        sentiment: SentimentProfile,
        risk: RiskAssessment,
        misalignment: MisalignmentSignal,
        agent_state: dict,
    ) -> ConfidenceScore:
        """
        Compute a calibrated confidence score for the analysis.

        Args:
            financial: Financial analysis output.
            sentiment: Sentiment analysis output.
            risk: Risk assessment output.
            misalignment: Misalignment detection output.
            agent_state: Full agent state (for Phase 2 metadata).

        Returns:
            ConfidenceScore with component breakdowns and explanations.
        """
        penalties = []
        boosts = []

        # 1. Evidence Quality (from financial data completeness)
        eq = self._score_evidence_quality(financial, penalties, boosts)

        # 2. Source Reliability (from Phase 2 evidence governance)
        sr = self._score_source_reliability(agent_state, penalties, boosts)

        # 3. Data Completeness (tools called vs. available)
        dc = self._score_data_completeness(agent_state, penalties, boosts)

        # 4. Consistency (conflicts and misalignments)
        cs = self._score_consistency(agent_state, misalignment, penalties, boosts)

        # 5. Recency (freshness of data)
        rc = self._score_recency(agent_state, penalties, boosts)

        # --- Weighted composite ---
        # Evidence quality and consistency matter most
        weights = {
            "evidence_quality": 0.25,
            "source_reliability": 0.15,
            "data_completeness": 0.25,
            "consistency": 0.20,
            "recency": 0.15,
        }
        overall = (
            eq * weights["evidence_quality"]
            + sr * weights["source_reliability"]
            + dc * weights["data_completeness"]
            + cs * weights["consistency"]
            + rc * weights["recency"]
        )

        # Clamp to [0.10, 0.95]
        overall = max(0.10, min(0.95, overall))
        label = self._score_to_label(overall)

        explanation = self._build_explanation(
            overall, label, eq, sr, dc, cs, rc, penalties, boosts
        )

        score = ConfidenceScore(
            overall=round(overall, 2),
            label=label,
            evidence_quality=round(eq, 2),
            source_reliability=round(sr, 2),
            data_completeness=round(dc, 2),
            consistency=round(cs, 2),
            recency=round(rc, 2),
            penalties=penalties,
            boosts=boosts,
            explanation=explanation,
        )

        logger.info(
            f"Confidence calibrated: {overall:.2f} ({label}) — "
            f"EQ={eq:.2f}, SR={sr:.2f}, DC={dc:.2f}, "
            f"CS={cs:.2f}, RC={rc:.2f}"
        )
        return score

    # ---------------------------------------------------------------
    # Component Scorers
    # ---------------------------------------------------------------

    def _score_evidence_quality(
        self,
        financial: FinancialSnapshot,
        penalties: list,
        boosts: list,
    ) -> float:
        """How good is the financial evidence we have?"""
        eq = financial.evidence_quality  # 0-1 from financial engine

        if eq >= 0.7:
            boosts.append(
                f"High evidence quality ({financial.metrics_available} "
                f"of {financial.metrics_available + financial.metrics_missing} metrics available)"
            )
        elif eq < 0.4:
            penalties.append(
                f"Low evidence quality (only {financial.metrics_available} metrics available, "
                f"{financial.metrics_missing} missing)"
            )

        return eq

    def _score_source_reliability(
        self,
        state: dict,
        penalties: list,
        boosts: list,
    ) -> float:
        """How reliable are our data sources?"""
        # Phase 2 evidence confidence scores
        evidence_confidence = state.get("evidence_confidence", {})
        if evidence_confidence:
            avg = sum(evidence_confidence.values()) / len(evidence_confidence)
            if avg >= 0.7:
                boosts.append(f"High source reliability (avg: {avg:.2f})")
            return avg

        # If no Phase 2 data, assume moderate (tool outputs are Tier 1)
        tool_calls = state.get("tool_calls", [])
        if tool_calls:
            # Tool outputs are generally reliable (direct API data)
            succeeded = sum(
                1 for tc in tool_calls
                if hasattr(tc, "success") and tc.success
            )
            return min(0.8, 0.5 + succeeded * 0.1)

        return 0.5

    def _score_data_completeness(
        self,
        state: dict,
        penalties: list,
        boosts: list,
    ) -> float:
        """Did we gather enough types of data?"""
        tool_calls = state.get("tool_calls", [])
        tool_names = set()
        for tc in tool_calls:
            if hasattr(tc, "tool_name"):
                tool_names.add(tc.tool_name)

        # The ideal analysis uses all 4 tools
        ideal_tools = {"get_stock_price", "get_financial_metrics", "get_company_info", "get_news"}
        covered = len(tool_names & ideal_tools)
        score = covered / len(ideal_tools)

        if score >= 0.75:
            boosts.append(f"Comprehensive data gathering ({covered}/{len(ideal_tools)} key tools used)")
        elif score < 0.5:
            missing = ideal_tools - tool_names
            penalties.append(f"Incomplete data: missing {', '.join(missing)}")

        return score

    def _score_consistency(
        self,
        state: dict,
        misalignment: MisalignmentSignal,
        penalties: list,
        boosts: list,
    ) -> float:
        """Are sources agreeing or contradicting each other?"""
        score = 1.0  # Start perfect, deduct for conflicts

        # Phase 2 conflict reports
        conflicts = state.get("conflict_reports", [])
        if conflicts:
            score -= min(0.4, len(conflicts) * 0.1)
            penalties.append(f"{len(conflicts)} evidence conflict(s) detected")

        # Misalignment is a form of inconsistency
        if misalignment.detected:
            severity_penalty = {
                RiskSeverity.LOW: 0.05,
                RiskSeverity.MEDIUM: 0.10,
                RiskSeverity.HIGH: 0.15,
                RiskSeverity.CRITICAL: 0.25,
            }
            score -= severity_penalty.get(misalignment.severity, 0.1)
            penalties.append(
                f"Sentiment-financial misalignment detected "
                f"({misalignment.misalignment_type.value})"
            )

        if score >= 0.9 and not conflicts and not misalignment.detected:
            boosts.append("Evidence is internally consistent — no conflicts detected")

        return max(0.1, score)

    def _score_recency(
        self,
        state: dict,
        penalties: list,
        boosts: list,
    ) -> float:
        """How fresh is our data?"""
        # Tool data is fetched live, so it's generally fresh
        tool_calls = state.get("tool_calls", [])
        if tool_calls:
            boosts.append("Data fetched in real-time from live APIs")
            return 0.85

        return 0.5

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def _score_to_label(self, score: float) -> str:
        """Convert numeric score to human label."""
        if score >= 0.80:
            return "High"
        elif score >= 0.60:
            return "Moderate-High"
        elif score >= 0.45:
            return "Moderate"
        elif score >= 0.30:
            return "Low-Moderate"
        else:
            return "Low"

    def _build_explanation(
        self,
        overall: float,
        label: str,
        eq: float, sr: float, dc: float, cs: float, rc: float,
        penalties: list,
        boosts: list,
    ) -> str:
        """Build a human-readable confidence explanation."""
        parts = [
            f"Analysis confidence: {label} ({overall:.0%}).",
            f"Components: Evidence Quality={eq:.0%}, "
            f"Source Reliability={sr:.0%}, "
            f"Data Completeness={dc:.0%}, "
            f"Consistency={cs:.0%}, "
            f"Recency={rc:.0%}.",
        ]

        if boosts:
            parts.append(f"Strengths: {'; '.join(boosts[:3])}.")
        if penalties:
            parts.append(f"Concerns: {'; '.join(penalties[:3])}.")

        return " ".join(parts)
