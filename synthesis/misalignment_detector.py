"""
synthesis/misalignment_detector.py — Sentiment↔Financial Divergence Detection
================================================================================

WHY THIS EXISTS:
    This is the MANDATORY architectural feature for Phase 3.

    Financial markets frequently exhibit misalignment:
    - NVDA has a P/E of 46 but everyone is "bullish" → overvaluation risk
    - A company's margins are declining but news is about "growth" → narrative risk
    - Financials are strong but sentiment is bearish → potential opportunity

    Most financial tools either:
    1. Show you numbers (and ignore sentiment), OR
    2. Show you sentiment (and ignore numbers).

    ARA-1 does BOTH and then COMPARES them. This module is the comparator.

MISALIGNMENT TYPES:
    1. BULLISH_WEAK_FUNDAMENTALS — "Hype vs Reality"
       Sentiment is bullish, but financials show weakness.
       Risk: Potential overvaluation / bubble.

    2. BEARISH_STRONG_FUNDAMENTALS — "Hidden Gem"
       Sentiment is bearish, but financials are solid.
       Opportunity: Potential undervaluation.

    3. OVERVALUATION_VS_OPTIMISM — "Priced for Perfection"
       Valuation metrics are stretched despite optimism.
       Risk: Any disappointment will cause outsized drop.

    4. GROWTH_NARRATIVE_DECLINING_MARGINS — "Narrative Risk"
       News talks about growth, but margins are compressing.
       Risk: Growth may not translate to profits.

HOW IT CONNECTS:
    - Consumes: FinancialSnapshot (from financial_engine.py),
                SentimentProfile (from sentiment_analyzer.py).
    - Produces: MisalignmentSignal (synthesis/schemas.py).
    - Used by: engine.py → feeds into risk analysis and report.

DESIGN PRINCIPLE:
    Each misalignment check is a DETERMINISTIC comparison between
    financial metrics and sentiment signals. No LLM calls.
    The output EXPLAINS the misalignment so the report can reason about it.
"""

from __future__ import annotations

from synthesis.schemas import (
    FinancialSnapshot,
    MisalignmentSignal,
    MisalignmentType,
    RiskSeverity,
    SentimentDirection,
    SentimentProfile,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class MisalignmentDetector:
    """
    Detects divergence between market sentiment and financial fundamentals.

    Runs 4 misalignment checks and returns the most severe signal.
    If multiple misalignments exist, the highest-severity one is returned
    as the primary signal (all are logged for the report).
    """

    def detect(
        self,
        financial: FinancialSnapshot,
        sentiment: SentimentProfile,
    ) -> MisalignmentSignal:
        """
        Compare financial fundamentals against sentiment and detect misalignment.

        Args:
            financial: Assessed financial metrics from financial_engine.
            sentiment: Aggregated sentiment from sentiment_analyzer.

        Returns:
            MisalignmentSignal — detected=True if a divergence exists.
        """
        signals = []

        # Run all 4 checks
        signals.append(self._check_bullish_weak_fundamentals(financial, sentiment))
        signals.append(self._check_bearish_strong_fundamentals(financial, sentiment))
        signals.append(self._check_overvaluation_vs_optimism(financial, sentiment))
        signals.append(self._check_growth_narrative_declining_margins(financial, sentiment))

        # Filter to detected signals
        detected = [s for s in signals if s.detected]

        if not detected:
            logger.info("No sentiment-financial misalignment detected")
            return MisalignmentSignal(
                detected=False,
                misalignment_type=MisalignmentType.NONE,
                explanation="Sentiment and financial fundamentals appear broadly aligned.",
            )

        # Return the highest-severity signal
        severity_order = {
            RiskSeverity.CRITICAL: 4,
            RiskSeverity.HIGH: 3,
            RiskSeverity.MEDIUM: 2,
            RiskSeverity.LOW: 1,
        }
        primary = max(detected, key=lambda s: severity_order.get(s.severity, 0))

        logger.warning(
            f"MISALIGNMENT DETECTED: {primary.misalignment_type.value} "
            f"(severity={primary.severity.value}, "
            f"confidence={primary.confidence:.2f})"
        )
        return primary

    # ---------------------------------------------------------------
    # Check 1: Bullish sentiment + weak fundamentals
    # ---------------------------------------------------------------
    def _check_bullish_weak_fundamentals(
        self,
        financial: FinancialSnapshot,
        sentiment: SentimentProfile,
    ) -> MisalignmentSignal:
        """
        Bullish market sentiment despite weak financial fundamentals.

        Example: Everyone is buying the stock, but earnings are declining,
        margins are shrinking, and debt is rising.
        """
        is_bullish = sentiment.overall_direction in (
            SentimentDirection.BULLISH,
        )
        is_weak = financial.financial_health_score < 0.45

        if not (is_bullish and is_weak):
            return MisalignmentSignal(detected=False)

        severity = RiskSeverity.HIGH if financial.financial_health_score < 0.35 else RiskSeverity.MEDIUM
        confidence = min(0.9, sentiment.overall_confidence * 0.8 + 0.2)

        return MisalignmentSignal(
            detected=True,
            misalignment_type=MisalignmentType.BULLISH_WEAK_FUNDAMENTALS,
            severity=severity,
            confidence=confidence,
            explanation=(
                f"Market sentiment is bullish ({sentiment.bullish_count} positive signals), "
                f"but financial health is weak (score: {financial.financial_health_score:.2f}). "
                f"Weaknesses: {', '.join(financial.key_weaknesses[:3]) or 'multiple metrics below thresholds'}. "
                f"This divergence suggests potential overvaluation or unsustainable momentum."
            ),
            sentiment_evidence=sentiment.summary,
            financial_evidence=financial.overall_assessment,
            recommendation=(
                "Exercise caution. The bullish sentiment may be driven by narrative "
                "rather than fundamentals. Consider whether the market is pricing in "
                "future improvements that may not materialize."
            ),
        )

    # ---------------------------------------------------------------
    # Check 2: Bearish sentiment + strong fundamentals
    # ---------------------------------------------------------------
    def _check_bearish_strong_fundamentals(
        self,
        financial: FinancialSnapshot,
        sentiment: SentimentProfile,
    ) -> MisalignmentSignal:
        """
        Bearish market sentiment despite strong financial fundamentals.

        Example: Stock is being sold off, but the company has strong
        margins, low debt, high growth, and solid cash flow.
        """
        is_bearish = sentiment.overall_direction in (
            SentimentDirection.BEARISH,
        )
        is_strong = financial.financial_health_score > 0.65

        if not (is_bearish and is_strong):
            return MisalignmentSignal(detected=False)

        severity = RiskSeverity.MEDIUM  # This is actually an OPPORTUNITY
        confidence = min(0.9, sentiment.overall_confidence * 0.8 + 0.2)

        return MisalignmentSignal(
            detected=True,
            misalignment_type=MisalignmentType.BEARISH_STRONG_FUNDAMENTALS,
            severity=severity,
            confidence=confidence,
            explanation=(
                f"Market sentiment is bearish ({sentiment.bearish_count} negative signals), "
                f"but financial fundamentals are strong (health score: {financial.financial_health_score:.2f}). "
                f"Strengths: {', '.join(financial.key_strengths[:3]) or 'multiple metrics above thresholds'}. "
                f"This divergence may represent a buying opportunity if the bearish "
                f"sentiment is driven by temporary factors."
            ),
            sentiment_evidence=sentiment.summary,
            financial_evidence=financial.overall_assessment,
            recommendation=(
                "Investigate the SOURCE of bearish sentiment. If driven by "
                "macro factors or sector rotation (not company-specific issues), "
                "the fundamentals may eventually prevail."
            ),
        )

    # ---------------------------------------------------------------
    # Check 3: Overvaluation despite optimism
    # ---------------------------------------------------------------
    def _check_overvaluation_vs_optimism(
        self,
        financial: FinancialSnapshot,
        sentiment: SentimentProfile,
    ) -> MisalignmentSignal:
        """
        Extreme valuation metrics (high P/E, P/B) combined with bullish sentiment.

        This is the "priced for perfection" scenario — any earnings miss
        or guidance cut will trigger outsized drops.
        """
        is_bullish = sentiment.overall_direction in (
            SentimentDirection.BULLISH, SentimentDirection.MIXED,
        )

        # Check valuation metrics
        pe_value = None
        pb_value = None
        for m in financial.valuation_metrics:
            if "P/E" in m.name and "Trailing" in m.name and m.value is not None:
                pe_value = m.value
            if "Book" in m.name and m.value is not None:
                pb_value = m.value

        is_overvalued = False
        valuation_evidence = []

        if pe_value is not None and pe_value > 40:
            is_overvalued = True
            valuation_evidence.append(f"Trailing P/E of {pe_value:.1f}x (above 40x)")
        if pb_value is not None and pb_value > 15:
            is_overvalued = True
            valuation_evidence.append(f"Price/Book of {pb_value:.1f}x (above 15x)")

        if not (is_bullish and is_overvalued):
            return MisalignmentSignal(detected=False)

        return MisalignmentSignal(
            detected=True,
            misalignment_type=MisalignmentType.OVERVALUATION_VS_OPTIMISM,
            severity=RiskSeverity.MEDIUM,
            confidence=0.7,
            explanation=(
                f"Valuation metrics suggest premium pricing: {', '.join(valuation_evidence)}. "
                f"Combined with bullish sentiment, the stock is 'priced for perfection' — "
                f"any negative surprise could trigger a significant correction."
            ),
            sentiment_evidence=sentiment.summary,
            financial_evidence=f"Stretched valuations: {', '.join(valuation_evidence)}",
            recommendation=(
                "Consider whether the premium valuation is justified by growth rates. "
                "Check the Forward P/E and PEG ratio for a growth-adjusted view. "
                "High valuation + high growth may be reasonable; high valuation + "
                "slowing growth is a red flag."
            ),
        )

    # ---------------------------------------------------------------
    # Check 4: Growth narrative + declining margins
    # ---------------------------------------------------------------
    def _check_growth_narrative_declining_margins(
        self,
        financial: FinancialSnapshot,
        sentiment: SentimentProfile,
    ) -> MisalignmentSignal:
        """
        News focuses on "growth" but margins are actually declining.

        This is subtle and dangerous — revenue growth can mask
        profitability erosion if costs are growing faster.
        """
        # Check if "growth" is a theme in sentiment
        growth_in_sentiment = any(
            "growth" in theme.lower()
            for theme in sentiment.dominant_themes
        )
        if not growth_in_sentiment:
            # Also check for growth keywords in signals
            growth_in_sentiment = any(
                "growth" in " ".join(s.positive_keywords).lower()
                for s in sentiment.signals
            )

        # Check margins
        operating_margin = None
        profit_margin = None
        for m in financial.profitability_metrics:
            if "Operating" in m.name and m.value is not None:
                operating_margin = m.value
            if "Net Profit" in m.name and m.value is not None:
                profit_margin = m.value

        margins_weak = False
        margin_evidence = []
        if operating_margin is not None and operating_margin < 12:
            margins_weak = True
            margin_evidence.append(f"Operating margin: {operating_margin:.1f}%")
        if profit_margin is not None and profit_margin < 8:
            margins_weak = True
            margin_evidence.append(f"Net margin: {profit_margin:.1f}%")

        if not (growth_in_sentiment and margins_weak):
            return MisalignmentSignal(detected=False)

        return MisalignmentSignal(
            detected=True,
            misalignment_type=MisalignmentType.GROWTH_NARRATIVE_DECLINING_MARGINS,
            severity=RiskSeverity.HIGH,
            confidence=0.65,
            explanation=(
                f"Market narrative emphasizes 'growth', but margins are under pressure: "
                f"{', '.join(margin_evidence)}. Revenue growth may not translate to "
                f"earnings growth if cost structure is deteriorating."
            ),
            sentiment_evidence="Growth narrative present in news/sentiment analysis.",
            financial_evidence=f"Margin pressure: {', '.join(margin_evidence)}",
            recommendation=(
                "Analyze the QUALITY of growth. Revenue growth with margin compression "
                "suggests the company is buying growth at the expense of profitability. "
                "Investigate whether margins are expected to expand or compress further."
            ),
        )
