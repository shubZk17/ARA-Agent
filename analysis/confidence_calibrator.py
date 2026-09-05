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

from analysis.schemas import (
    ConfidenceScore,
    FinancialSnapshot,
    MisalignmentSignal,
    RiskAssessment,
    RiskSeverity,
    SentimentProfile,
)
from config.logging import get_logger

logger = get_logger(__name__)


# Which upstream source each tool actually reads. Declarative on purpose —
# a new tool is one row here, and Phase 6/7 tools (market_context, EDGAR,
# transcripts) are what will finally make this set bigger than one.
SOURCE_FAMILIES: dict[str, str] = {
    "get_stock_price": "yfinance",
    "get_financial_metrics": "yfinance",
    "get_company_info": "yfinance",
    "get_news": "yfinance",
    # Phase 6. Both are still yfinance underneath and are declared as such
    # DELIBERATELY: market_context computes beta from a return series instead
    # of reading info["beta"], which is a second DERIVATION, not a second
    # SOURCE. Claiming diversity here would inflate confidence on exactly the
    # basis the D4 fix removed. Phase 7's EDGAR is the first real second family.
    "get_price_history": "yfinance",
    "get_market_context": "yfinance",
    # Phase 7.1 — the first genuinely independent family. SEC EDGAR reads
    # filed XBRL data, not yfinance, so this is where SOURCE_DIVERSITY_FACTOR
    # can finally reach 2 rather than being permanently capped at 1.
    "get_sec_filings": "sec_edgar",
    # Phase 7.4 — the third family. Tagged distinctly from "sec_edgar" on
    # purpose, not just because it's a different EDGAR endpoint: Form 4s are
    # filed by the INSIDER, not the company. A company misstating its
    # financials doesn't imply an insider misfiles a Form 4 — different
    # filer, different obligation, genuinely independent failure modes.
    "get_insider_transactions": "sec_edgar_ownership",
}

# distinct source families -> multiplier on data completeness
SOURCE_DIVERSITY_FACTOR: dict[int, float] = {
    0: 0.50,
    1: 0.75,   # everything from one provider — no way to catch its errors
    2: 0.90,
    3: 1.00,
}


# Phase 8.4: below this many graded outcomes, a measured hit rate is noise,
# not signal — stay on the heuristic constant instead of chasing it.
MIN_GRADED_SAMPLE = 10


def _empirical_reliability_baseline(default: float) -> float:
    """
    Phase 8.4: replace a heuristic fallback constant with the system's OWN
    measured hit rate, once Phase 8.1 has graded enough recommendations to
    trust it. A 50% hit rate (coin flip) reproduces `default` unchanged;
    measured skill above or below that shifts the baseline directly.

    Defensive on purpose: this only ever touches the two "we have no actual
    measurement" fallbacks below. A missing/empty/corrupt outcome log must
    never break confidence scoring, so any failure here just returns
    `default`, exactly as if Phase 8 had not shipped.
    """
    try:
        from analysis.outcome_scorer import summarize
        from analysis.recommendation_store import RecommendationStore

        summary = summarize(RecommendationStore().load_all())
        if summary.get("graded_count", 0) < MIN_GRADED_SAMPLE:
            return default
        return max(0.1, min(0.95, default + (summary["hit_rate"] - 0.5)))
    except Exception:
        return default


def _embeddings_degraded() -> bool:
    """
    Whether retrieval ran on meaningless hash vectors this process.

    Imported lazily and defensively: analysis/ must stay runnable when the
    knowledge layer is absent (a synthesis-only replay, or a test that never
    builds a vector store).
    """
    try:
        from knowledge.retrieval.embeddings import embeddings_degraded
        return embeddings_degraded()
    except Exception:
        return False


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

        # Retrieval degradation is a hard cap, not a nudge. If embeddings fell
        # back to hash vectors, every retrieved "supporting document" was
        # picked at random, and no amount of complete financial data makes
        # that analysis high-confidence (D6).
        if _embeddings_degraded():
            overall = min(overall, 0.50)
            penalties.append(
                "Embeddings degraded to hash fallback — retrieved evidence is "
                "not semantically related to the query; confidence capped at 50%"
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
        # Measured per-output reliability, written by agent/nodes.py's
        # tool_node. Before that write existed this branch was unreachable
        # and execution always fell to the count-based estimate below (D4).
        evidence_confidence = state.get("evidence_confidence", {})
        if evidence_confidence:
            avg = sum(evidence_confidence.values()) / len(evidence_confidence)
            failed = sum(1 for v in evidence_confidence.values() if v == 0.0)
            if avg >= 0.7:
                boosts.append(
                    f"High source reliability (measured avg {avg:.2f} "
                    f"across {len(evidence_confidence)} outputs)"
                )
            elif avg < 0.5:
                penalties.append(
                    f"Low source reliability (measured avg {avg:.2f})"
                )
            if failed:
                penalties.append(f"{failed} tool call(s) returned no usable evidence")
            return avg

        # Fallback: no measurement available (e.g. a replayed episodic state).
        # Counting successful calls measures effort, not reliability, so this
        # is capped well below what a measured score can reach.
        tool_calls = state.get("tool_calls", [])
        if tool_calls:
            succeeded = sum(
                1 for tc in tool_calls
                if hasattr(tc, "success") and tc.success
            )
            penalties.append("Source reliability estimated, not measured")
            return min(0.7, _empirical_reliability_baseline(0.4) + succeeded * 0.1)

        return _empirical_reliability_baseline(0.5)

    def _score_data_completeness(
        self,
        state: dict,
        penalties: list,
        boosts: list,
    ) -> float:
        """Did we gather enough types of data — and from enough sources?"""
        tool_calls = state.get("tool_calls", [])
        tool_names = set()
        for tc in tool_calls:
            if hasattr(tc, "tool_name"):
                tool_names.add(tc.tool_name)

        # What a complete analysis gathers. get_price_history joined this in
        # Phase 6: with a horizon attached to every verdict, a thesis with no
        # price series genuinely is missing evidence, not merely lacking a
        # nice-to-have. get_market_context stays out — it enriches, it is not
        # required to reach a defensible conclusion.
        ideal_tools = {
            "get_stock_price", "get_financial_metrics", "get_company_info",
            "get_news", "get_price_history",
        }
        covered = len(tool_names & ideal_tools)
        coverage = covered / len(ideal_tools)

        # Tool count is not source count. All four tools above read the same
        # yfinance endpoint, so "4/4 tools used" was scoring a single source
        # as complete data gathering. Independent sources are what make data
        # complete — a second opinion, not a fourth phrasing of the first.
        families = {SOURCE_FAMILIES.get(name, "unknown") for name in tool_names}
        diversity = SOURCE_DIVERSITY_FACTOR.get(
            len(families), max(SOURCE_DIVERSITY_FACTOR.values())
        )
        score = coverage * diversity

        if coverage >= 0.75:
            boosts.append(
                f"Broad data gathering ({covered}/{len(ideal_tools)} tool types used)"
            )
        elif coverage < 0.5:
            missing = ideal_tools - tool_names
            penalties.append(f"Incomplete data: missing {', '.join(missing)}")

        if len(families) <= 1:
            penalties.append(
                f"Single data source ({', '.join(sorted(families)) or 'none'}) — "
                f"no independent corroboration of any figure"
            )

        return score

    def _score_consistency(
        self,
        state: dict,
        misalignment: MisalignmentSignal,
        penalties: list,
        boosts: list,
    ) -> float:
        """Are sources agreeing or contradicting each other?"""
        conflicts = state.get("conflict_reports", [])

        if conflicts:
            score = 1.0 - min(0.4, len(conflicts) * 0.1)
            penalties.append(f"{len(conflicts)} evidence conflict(s) detected")
        else:
            # POLARITY FIX (plan §5.4). This used to start at 1.0 — "perfect
            # consistency" — whenever no conflicts were reported. But no
            # conflict has ever been reported: all four tools read the same
            # yfinance endpoint, so cross-source agreement is unmeasurable by
            # construction. Scoring an unmeasured quantity as perfect is how
            # a single-source run earned a "no conflicts detected" boost.
            #
            # Absence of evidence is not evidence of absence. Until Phase 6
            # adds a source that can genuinely disagree, this stays neutral.
            score = _empirical_reliability_baseline(0.6)
            penalties.append(
                "Consistency unmeasured — all evidence comes from a single "
                "data source, so cross-source agreement cannot be assessed"
            )

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
