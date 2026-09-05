"""
synthesis/engine.py — Central Synthesis Orchestrator
======================================================

WHY THIS EXISTS:
    This is the CONDUCTOR. Individual engines (financial, sentiment,
    risk, misalignment, confidence) each analyze one dimension.
    The synthesis engine:

    1. ORCHESTRATES all engines in the correct dependency order.
    2. PASSES outputs between engines (misalignment needs financial + sentiment).
    3. SYNTHESIZES a unified investment thesis from all outputs.
    4. DETERMINES the final investment outlook (buy/hold/sell).
    5. PRODUCES a SynthesisReport ready for the report generator.

DEPENDENCY ORDER:
    1. Financial Engine (no dependencies)
    2. Sentiment Analyzer (no dependencies)
    3. Misalignment Detector (needs financial + sentiment)
    4. Risk Analyzer (needs financial + sentiment + misalignment)
    5. Confidence Calibrator (needs ALL above + agent state)
    6. Investment Thesis (synthesizes everything)

WHY NOT just call engines independently?
    - Misalignment detection REQUIRES both financial and sentiment outputs.
    - Risk analysis REQUIRES misalignment detection output.
    - Confidence REQUIRES knowing about all risks and conflicts.
    - The thesis REQUIRES all of the above.
    This is a DAG, not independent parallel tasks.

HOW IT CONNECTS:
    - Consumes: AgentState (from LangGraph, post-execution).
    - Uses: All 5 synthesis engines.
    - Produces: SynthesisReport (synthesis/schemas.py).
    - Used by: main.py (after graph.invoke()), report_generator.py.

DESIGN PRINCIPLE:
    This module is a THIN orchestrator — it delegates to engines
    and assembles their outputs. It does NOT contain analysis logic.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from datetime import date, timedelta
from typing import Any

from config.settings import settings
from analysis.observability.collector import EventType
from config.horizons import (
    DEFAULT_HORIZON,
    DEFAULT_RISK_PROFILE,
    FINANCIAL_CATEGORIES,
    HorizonProfile,
    InvestmentHorizon,
    RiskProfileConfig,
    get_horizon_profile,
    get_risk_profile,
)
from analysis.confidence_calibrator import ConfidenceCalibrator
from analysis.financial_engine import FinancialAnalysisEngine
from analysis.misalignment_detector import MisalignmentDetector
from analysis.risk_analyzer import RiskAnalyzer
from analysis.schemas import (
    HorizonRecommendation,
    InvestmentOutlook,
    RiskSeverity,
    SentimentDirection,
    SynthesisReport,
    TechnicalSnapshot,
)
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.technical_engine import TechnicalAnalysisEngine
from config.logging import get_logger

logger = get_logger(__name__)

# Below this many populated metrics, the pipeline abstains instead of issuing
# a recommendation. A full yfinance fetch yields 21 of 22; single digits means
# the tool failed or the ticker is thinly covered, and either way there is not
# enough here to reason from.
MIN_METRICS_FOR_RECOMMENDATION = 8


class SynthesisEngine:
    """
    Central orchestrator for Phase 3 synthesis pipeline.

    Takes raw agent state (from Phase 1+2 graph execution) and
    produces a structured SynthesisReport.
    """

    def __init__(self) -> None:
        # Initialize all sub-engines
        self._financial = FinancialAnalysisEngine()
        self._technical = TechnicalAnalysisEngine()
        self._sentiment = SentimentAnalyzer()
        self._misalignment = MisalignmentDetector()
        self._risk = RiskAnalyzer()
        self._confidence = ConfidenceCalibrator()

        logger.info("Synthesis engine initialized with all sub-engines")

    def synthesize(self, agent_state: dict, collector=None) -> SynthesisReport:
        """
        Run the full synthesis pipeline on completed agent state.

        Args:
            agent_state: Final state dict from graph.invoke().
                Must contain: tool_calls, reasoning_trace, query, etc.
            collector: Phase 4 — TelemetryCollector, optional. None is a no-op.

        Returns:
            SynthesisReport with all engine outputs and investment thesis.
        """
        start = time.time()

        def _stage(name):
            return collector.track(EventType.SYNTHESIS_STEP, name) if collector else nullcontext()

        # Extract inputs from state
        tool_calls = agent_state.get("tool_calls", [])
        reasoning_trace = agent_state.get("reasoning_trace", [])
        observations = [
            step.observation
            for step in reasoning_trace
            if hasattr(step, "observation") and step.observation
        ]

        # The horizon decides how every stage below weighs its evidence, so
        # it is resolved once, first, and threaded down. An unknown value
        # falls back to long_term rather than raising — see get_horizon_profile.
        profile = get_horizon_profile(agent_state.get("horizon", DEFAULT_HORIZON))
        risk_profile = get_risk_profile(
            agent_state.get("risk_profile", DEFAULT_RISK_PROFILE)
        )

        logger.info(
            f"Starting synthesis pipeline (horizon={profile.horizon}, "
            f"risk_profile={risk_profile.profile})..."
        )

        # --- Step 1: Financial Analysis ---
        logger.info("[1/7] Running financial analysis engine...")
        with _stage("financial_analysis"):
            financial = self._financial.analyze(tool_calls, observations, profile)

        # --- Step 2: Technical Analysis (Phase 6) ---
        logger.info("[2/7] Running technical analysis engine...")
        with _stage("technical_analysis"):
            technical = self._technical.analyze(tool_calls, profile)

        # --- Step 3: Sentiment Analysis ---
        logger.info("[3/7] Running sentiment analyzer...")
        with _stage("sentiment_analysis"):
            sentiment = self._sentiment.analyze(tool_calls, observations)

        # --- Step 4: Misalignment Detection ---
        logger.info("[4/7] Running misalignment detector...")
        with _stage("misalignment_detection"):
            misalignment = self._misalignment.detect(financial, sentiment)

        # --- Step 5: Risk Analysis ---
        logger.info("[5/7] Running risk analyzer...")
        with _stage("risk_analysis"):
            risk = self._risk.analyze(financial, sentiment, misalignment, technical)

        # --- Step 6: Confidence Calibration ---
        logger.info("[6/7] Calibrating confidence...")
        with _stage("confidence_calibration"):
            confidence = self._confidence.calibrate(
                financial=financial,
                sentiment=sentiment,
                risk=risk,
                misalignment=misalignment,
                agent_state=agent_state,
            )

        # --- Step 7: Investment Thesis & Dated Recommendation ---
        logger.info("[7/7] Generating investment thesis...")
        with _stage("investment_thesis"):
            outlook = self._determine_outlook(
                financial, sentiment, risk, misalignment, confidence,
                technical, profile, risk_profile,
            )
            recommendation = self._build_recommendation(
                outlook, financial, technical, profile, risk_profile, confidence,
                price_now=self._latest_quote(tool_calls),
            )
            thesis = self._generate_thesis(
                financial, sentiment, risk, misalignment, confidence, outlook,
                technical, profile, recommendation,
            )
            key_findings = self._extract_key_findings(
                financial, sentiment, misalignment, risk, technical
            )
            contradictions = self._extract_contradictions(agent_state, misalignment)

        # --- Assemble Report ---
        elapsed = time.time() - start

        tools_used = list(set(
            tc.tool_name for tc in tool_calls
            if hasattr(tc, "tool_name")
        ))

        report = SynthesisReport(
            ticker=financial.ticker or technical.ticker,
            company_name=financial.company_name,
            query=agent_state.get("query", ""),
            horizon=profile.horizon,
            risk_profile=risk_profile.profile,
            target_holding_period=profile.target_holding_period_text,
            financial=financial,
            technical=technical,
            sentiment=sentiment,
            misalignment=misalignment,
            risk=risk,
            confidence=confidence,
            investment_thesis=thesis,
            outlook=outlook,
            recommendation=recommendation,
            key_findings=key_findings,
            contradictions=contradictions,
            tools_used=tools_used,
            iterations_used=agent_state.get("iteration_count", 0),
            evidence_count=len(agent_state.get("retrieved_evidence", [])),
            execution_time_seconds=round(elapsed, 2),
            model_used=settings.active_model,
        )

        logger.info(
            f"Synthesis complete: {report.ticker} [{profile.horizon}] → "
            f"{outlook.value} (confidence={confidence.overall:.0%}) "
            f"in {elapsed:.1f}s"
        )
        return report

    def _determine_outlook(
        self,
        financial,
        sentiment,
        risk,
        misalignment,
        confidence,
        technical: TechnicalSnapshot = None,
        profile: HorizonProfile = None,
        risk_profile: RiskProfileConfig = None,
    ) -> InvestmentOutlook:
        """
        Determine the investment outlook based on all analysis dimensions.

        This is the FINAL decision. Phase 6 changed how it is reached: instead
        of financial health as a base with fixed nudges for sentiment and
        risk, the three pillars are blended using the HORIZON's own weights.

            SHORT_TERM   technical 0.45, sentiment 0.20, fundamentals 0.35
            LONG_TERM    fundamentals 0.84, sentiment 0.10, technical 0.06

        That is the entire point of the phase: the same evidence, weighted by
        holding period, must be able to reach different verdicts. A company
        with excellent fundamentals in a broken downtrend is a long-term BUY
        and a short-term HOLD or SELL, and both statements are correct.

        Risk severity and the holder's risk profile then adjust the blend,
        and confidence gates it: below the profile's min_confidence_to_act,
        a directional call is downgraded to HOLD rather than issued at low
        conviction.
        """
        # --- Abstain gate (plan §5.4) ---
        # A recommendation built on almost no metrics is not a cautious HOLD,
        # it is a guess wearing a HOLD's clothes. Refusing to answer is a
        # legitimate — and for a real investing tool, necessary — output.
        #
        # Deliberately reuses the existing INSUFFICIENT_DATA rather than
        # adding a near-synonym INSUFFICIENT_EVIDENCE: report_generator and
        # the 22 evaluation metrics already handle this value, and two enum
        # members meaning "we don't know" would just need disambiguating
        # everywhere they are read.
        if financial.metrics_available < MIN_METRICS_FOR_RECOMMENDATION:
            logger.warning(
                f"Abstaining: only {financial.metrics_available} metrics available "
                f"(need {MIN_METRICS_FOR_RECOMMENDATION})"
            )
            return InvestmentOutlook.INSUFFICIENT_DATA

        # Insufficient data → can't recommend
        if confidence.overall < 0.25:
            return InvestmentOutlook.INSUFFICIENT_DATA

        profile = profile or get_horizon_profile(DEFAULT_HORIZON)
        risk_profile = risk_profile or get_risk_profile(DEFAULT_RISK_PROFILE)

        score = self._blend_pillars(financial, sentiment, technical, profile)

        # Risk penalty, scaled by how much drawdown this holder tolerates.
        risk_penalty = {
            RiskSeverity.LOW: 0.0,
            RiskSeverity.MEDIUM: -0.05,
            RiskSeverity.HIGH: -0.10,
            RiskSeverity.CRITICAL: -0.20,
        }
        score += (
            risk_penalty.get(risk.overall_risk_level, 0.0)
            * risk_profile.risk_penalty_multiplier
        )

        # Misalignment penalty
        if misalignment.detected:
            score -= 0.05

        # Map score to outlook
        if score >= 0.75:
            outlook = InvestmentOutlook.STRONG_BUY
        elif score >= 0.60:
            outlook = InvestmentOutlook.BUY
        elif score >= 0.40:
            outlook = InvestmentOutlook.HOLD
        elif score >= 0.25:
            outlook = InvestmentOutlook.SELL
        else:
            outlook = InvestmentOutlook.STRONG_SELL

        # Conviction gate. A directional call made at confidence the holder's
        # own profile says is too thin to act on is not a recommendation, it
        # is noise with an arrow drawn on it.
        directional = {
            InvestmentOutlook.STRONG_BUY, InvestmentOutlook.BUY,
            InvestmentOutlook.SELL, InvestmentOutlook.STRONG_SELL,
        }
        if outlook in directional and confidence.overall < risk_profile.min_confidence_to_act:
            logger.info(
                f"Downgrading {outlook.value} to HOLD: confidence "
                f"{confidence.overall:.0%} is below the "
                f"{risk_profile.label.lower()} threshold of "
                f"{risk_profile.min_confidence_to_act:.0%}"
            )
            return InvestmentOutlook.HOLD

        return outlook

    def _blend_pillars(
        self,
        financial,
        sentiment,
        technical: TechnicalSnapshot,
        profile: HorizonProfile,
    ) -> float:
        """
        Combine fundamentals, technicals and sentiment using the horizon's
        category weights. Returns 0.0–1.0.

        Pillars with no evidence drop out and the remaining weights
        renormalize. That matters most for the short horizon: without price
        history, 45% of its weight is missing, and treating the absent trend
        as 0.5 "neutral" would drag every short-term verdict toward HOLD
        while looking like a real assessment.
        """
        fundamental_weight = sum(profile.weight(c) for c in FINANCIAL_CATEGORIES)

        pillars: list[tuple[float, float]] = [
            (fundamental_weight, financial.financial_health_score),
        ]

        if technical is not None and technical.available:
            pillars.append((profile.weight("technical"), technical.trend_score))

        sentiment_score = {
            SentimentDirection.BULLISH: 0.75,
            SentimentDirection.BEARISH: 0.25,
            SentimentDirection.MIXED: 0.5,
            SentimentDirection.NEUTRAL: 0.5,
        }.get(sentiment.overall_direction, 0.5)
        if sentiment.signals:
            pillars.append((profile.weight("sentiment"), sentiment_score))

        total_weight = sum(weight for weight, _ in pillars)
        if not total_weight:
            return financial.financial_health_score

        return sum(weight * value for weight, value in pillars) / total_weight

    @staticmethod
    def _latest_quote(tool_calls: list) -> float | None:
        """Most recent live price from get_stock_price, if it ran."""
        price = None
        for call in tool_calls:
            if getattr(call, "tool_name", "") != "get_stock_price":
                continue
            payload = getattr(call, "tool_output_structured", None)
            if isinstance(payload, dict):
                value = payload.get("current_price")
                if isinstance(value, (int, float)):
                    price = float(value)
        return price

    def _build_recommendation(
        self,
        outlook: InvestmentOutlook,
        financial,
        technical: TechnicalSnapshot,
        profile: HorizonProfile,
        risk_profile: RiskProfileConfig,
        confidence,
        price_now: float = None,
    ) -> HorizonRecommendation:
        """
        Attach a holding period, an entry condition, an INVALIDATION condition
        and a review date to the verdict.

        The invalidation condition is the load-bearing field: it is what makes
        the recommendation falsifiable, and therefore what Phase 8 can score.
        It is written against real levels from the price series whenever one
        was gathered, and falls back to a fundamental trigger when it was not
        — never to a platitude, because a condition nobody can check is the
        same as having none.
        """
        review_by = (date.today() + timedelta(days=profile.review_days)).isoformat()
        levels = technical.key_levels if technical else {}
        # The price at recommendation is the one field Phase 8 cannot work
        # without, so it must not depend on a single tool having run. Price
        # history first (it is a settled close), the live quote second.
        last_close = (technical.last_close if technical else None) or price_now

        return HorizonRecommendation(
            outlook=outlook,
            horizon=profile.horizon,
            holding_period=profile.target_holding_period_text,
            entry_condition=self._entry_condition(outlook, profile, levels, last_close),
            invalidation_condition=self._invalidation_condition(
                outlook, profile, financial, levels, last_close
            ),
            review_by_date=review_by,
            price_at_recommendation=last_close,
            risk_profile=risk_profile.profile,
            position_note=risk_profile.position_note,
            rationale=(
                f"{profile.label}: fundamentals weighted "
                f"{sum(profile.weight(c) for c in FINANCIAL_CATEGORIES):.0%}, "
                f"technicals {profile.weight('technical'):.0%}, "
                f"sentiment {profile.weight('sentiment'):.0%}. "
                f"Confidence {confidence.overall:.0%} ({confidence.label})."
            ),
        )

    @staticmethod
    def _entry_condition(
        outlook: InvestmentOutlook,
        profile: HorizonProfile,
        levels: dict,
        last_close: float | None,
    ) -> str:
        """What has to be true to open the position."""
        if outlook in (InvestmentOutlook.HOLD, InvestmentOutlook.INSUFFICIENT_DATA):
            return "No new position. Existing holders: no action required."

        if outlook in (InvestmentOutlook.SELL, InvestmentOutlook.STRONG_SELL):
            return (
                "Reduce or exit on strength rather than at market; avoid selling "
                "into an already-extended decline."
            )

        sma_50, sma_200 = levels.get("sma_50"), levels.get("sma_200")

        if profile.horizon == InvestmentHorizon.SHORT_TERM.value:
            if sma_50 and last_close:
                if last_close >= sma_50:
                    return (
                        f"Enter while price holds above the 50-day SMA "
                        f"({sma_50:,.2f}); scale in rather than committing at once."
                    )
                return (
                    f"Wait for a daily close back above the 50-day SMA "
                    f"({sma_50:,.2f}) before entering — the trade is not yet confirmed."
                )
            return "Enter on confirmation of the short-term trend; scale in."

        if sma_200 and last_close and last_close < sma_200:
            return (
                f"Accumulate gradually. Price is below the 200-day SMA "
                f"({sma_200:,.2f}), so average in over several tranches rather "
                f"than sizing up at once."
            )
        return (
            "Accumulate on a schedule rather than in a single entry; the "
            "long-horizon case does not depend on timing the entry precisely."
        )

    @staticmethod
    def _invalidation_condition(
        outlook: InvestmentOutlook,
        profile: HorizonProfile,
        financial,
        levels: dict,
        last_close: float | None,
    ) -> str:
        """What would prove this recommendation wrong. Must be checkable."""
        if outlook == InvestmentOutlook.INSUFFICIENT_DATA:
            return (
                "Not applicable — no recommendation was issued. Re-run once "
                "the missing evidence can be gathered."
            )

        # A bearish call is refuted by the OPPOSITE events to a bullish one:
        # a SELL is not invalidated by the price falling, it is confirmed by it.
        # Every clause below is therefore built from the recommendation's own
        # direction, not from a fixed "things got worse" template.
        bearish = outlook in (InvestmentOutlook.SELL, InvestmentOutlook.STRONG_SELL)
        side = "above" if bearish else "below"
        clauses: list[str] = []

        if profile.horizon == InvestmentHorizon.SHORT_TERM.value:
            sma_50 = levels.get("sma_50")
            atr_14 = levels.get("atr_14")
            if sma_50:
                clauses.append(f"a daily close {side} the 50-day SMA ({sma_50:,.2f})")
            if atr_14 and last_close:
                bound = last_close + (2 * atr_14 if bearish else -2 * atr_14)
                clauses.append(
                    f"a close {side} {bound:,.2f} (2x the 14-day ATR of "
                    f"{atr_14:,.2f} from {last_close:,.2f})"
                )
        else:
            sma_200 = levels.get("sma_200")
            if sma_200:
                clauses.append(
                    f"a sustained (>1 month) close {side} the 200-day SMA "
                    f"({sma_200:,.2f})"
                )
            growth = next(
                (m for m in financial.growth_metrics
                 if m.name.startswith("Revenue Growth") and m.value is not None),
                None,
            )
            if growth is not None:
                bound = growth.value + 10.0 if bearish else max(0.0, growth.value - 10.0)
                verb = "recovering above" if bearish else "falling below"
                clauses.append(
                    f"revenue growth {verb} {bound:.1f}% year over year "
                    f"(currently {growth.value:.1f}%)"
                )
            margin = next(
                (m for m in financial.profitability_metrics
                 if m.name.startswith("Net Profit Margin") and m.value is not None),
                None,
            )
            if margin is not None:
                bound = margin.value + 5.0 if bearish else max(0.0, margin.value - 5.0)
                verb = "expanding above" if bearish else "compressing below"
                clauses.append(
                    f"net margin {verb} {bound:.1f}% (currently {margin.value:.1f}%)"
                )

        if not clauses:
            # No price series and no usable fundamentals: say so plainly rather
            # than inventing a level. An uncheckable condition is worse than an
            # admitted absence, because it reads as rigour.
            return (
                "NOT SPECIFIABLE from the evidence gathered — no price levels and "
                "no growth or margin baseline were available. Treat this "
                "recommendation as ungraded until one of them is."
            )

        label = "This bearish call is invalidated by" if bearish else "Invalidated by"
        return f"{label} {', or '.join(clauses)}."

    def _generate_thesis(
        self,
        financial,
        sentiment,
        risk,
        misalignment,
        confidence,
        outlook,
        technical: TechnicalSnapshot = None,
        profile: HorizonProfile = None,
        recommendation: HorizonRecommendation = None,
    ) -> str:
        """Generate a multi-paragraph investment thesis."""
        paragraphs = []

        # Opening — State the conclusion
        outlook_text = {
            InvestmentOutlook.STRONG_BUY: "represents a compelling investment opportunity",
            InvestmentOutlook.BUY: "presents a favorable risk-reward profile",
            InvestmentOutlook.HOLD: "warrants a hold position pending further developments",
            InvestmentOutlook.SELL: "faces headwinds that warrant caution",
            InvestmentOutlook.STRONG_SELL: "presents significant downside risk",
            InvestmentOutlook.INSUFFICIENT_DATA: "cannot be fully assessed due to limited data",
        }
        horizon_clause = (
            f" over a {profile.target_holding_period_text} holding period"
            if profile else ""
        )
        paragraphs.append(
            f"{financial.company_name or financial.ticker} "
            f"({financial.ticker or (technical.ticker if technical else '')}) "
            f"{outlook_text.get(outlook, 'requires further analysis')}"
            f"{horizon_clause}. "
            f"This assessment is based on analysis of financial fundamentals, "
            f"price action, market sentiment, and risk factors, with an overall "
            f"confidence of {confidence.overall:.0%} ({confidence.label})."
        )

        # Financial summary
        if financial.key_strengths or financial.key_weaknesses:
            fin_parts = []
            if financial.key_strengths:
                fin_parts.append(
                    f"Financial strengths include {', '.join(financial.key_strengths[:3])}"
                )
            if financial.key_weaknesses:
                fin_parts.append(
                    f"areas of concern include {', '.join(financial.key_weaknesses[:3])}"
                )
            paragraphs.append(
                f"From a fundamental perspective, {financial.overall_assessment} "
                + "; ".join(fin_parts) + "."
            )

        # Technical picture — placed before sentiment because on the short
        # horizon it carries the most weight of anything in the report.
        if technical is not None and technical.available and technical.summary:
            paragraphs.append(technical.summary)

        # Sentiment
        paragraphs.append(sentiment.summary)

        # Misalignment warning
        if misalignment.detected:
            paragraphs.append(
                f"⚠ MISALIGNMENT ALERT: {misalignment.explanation} "
                f"{misalignment.recommendation}"
            )

        # Risk summary
        if risk.risks:
            paragraphs.append(risk.summary)

        # Closing — the dated form, so the thesis ends with something that can
        # actually be acted on and later graded.
        if recommendation is not None:
            paragraphs.append(
                f"Given the analysis, the recommendation is "
                f"{outlook.value.replace('_', ' ').upper()} over "
                f"{recommendation.holding_period}, with "
                f"{confidence.label.lower()} confidence. "
                f"Entry: {recommendation.entry_condition} "
                f"{recommendation.invalidation_condition} "
                f"Reassess by {recommendation.review_by_date}."
            )
        else:
            paragraphs.append(
                f"Given the analysis, the recommendation is "
                f"{outlook.value.replace('_', ' ').upper()} with "
                f"{confidence.label.lower()} confidence. "
                f"Investors should monitor key risks and reassess "
                f"as new data becomes available."
            )

        return "\n\n".join(paragraphs)

    def _extract_key_findings(
        self, financial, sentiment, misalignment, risk, technical=None
    ) -> list[str]:
        """Extract the most important findings for the executive summary."""
        findings = []

        if financial.key_strengths:
            findings.append(f"Financial strength: {financial.key_strengths[0]}")
        if financial.key_weaknesses:
            findings.append(f"Financial concern: {financial.key_weaknesses[0]}")

        if technical is not None and technical.available:
            findings.append(
                f"Price trend: {technical.trend_label} "
                f"(score {technical.trend_score:.2f})"
            )

        findings.append(f"Market sentiment: {sentiment.overall_direction.value}")

        if misalignment.detected:
            findings.append(
                f"Misalignment detected: {misalignment.misalignment_type.value}"
            )

        if risk.high_risks:
            findings.append(f"Key risk: {risk.high_risks[0].title}")

        if sentiment.dominant_themes:
            findings.append(f"Key themes: {', '.join(sentiment.dominant_themes[:3])}")

        return findings

    def _extract_contradictions(self, state: dict, misalignment) -> list[str]:
        """Extract all contradictions for the report."""
        contradictions = []

        # Phase 2 conflict reports
        for cr in state.get("conflict_reports", []):
            if isinstance(cr, dict):
                contradictions.append(cr.get("resolution", str(cr)))
            else:
                contradictions.append(str(cr))

        # Misalignment
        if misalignment.detected:
            contradictions.append(misalignment.explanation)

        return contradictions
