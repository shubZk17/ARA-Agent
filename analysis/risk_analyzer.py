"""
synthesis/risk_analyzer.py — Multi-Category Risk Assessment Engine
===================================================================

WHY THIS EXISTS:
    Investment decisions require understanding not just the upside
    (covered by financial analysis) but the DOWNSIDE. This engine
    systematically identifies and categorizes risks:

    - Valuation risk: Is the stock overpriced?
    - Financial risk: Is the balance sheet fragile?
    - Execution risk: Can the company deliver on promises?
    - Sector/competitive risk: External threats from industry.
    - Macroeconomic risk: Interest rates, recession, trade policy.
    - Concentration risk: Over-dependence on one product/market.

    Each risk is:
    - Identified from evidence (not fabricated),
    - Assigned a severity,
    - Explained with supporting data,
    - Paired with potential mitigation.

HOW IT CONNECTS:
    - Consumes: FinancialSnapshot, SentimentProfile, MisalignmentSignal.
    - Produces: RiskAssessment (synthesis/schemas.py).
    - Used by: engine.py (central orchestrator), report_generator.py.

DESIGN PRINCIPLE:
    Risks are identified by RULE-BASED checks on financial metrics
    and sentiment signals. Each check is independent and composable.
    Adding a new risk check = adding one method.
"""

from __future__ import annotations

from analysis.schemas import (
    FinancialSnapshot,
    MisalignmentSignal,
    RiskAssessment,
    RiskItem,
    RiskSeverity,
    SentimentDirection,
    SentimentProfile,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class RiskAnalyzer:
    """
    Systematic risk identification across 6 categories.

    Each risk check is independent. The analyzer runs ALL checks
    and aggregates results into a RiskAssessment.
    """

    def analyze(
        self,
        financial: FinancialSnapshot,
        sentiment: SentimentProfile,
        misalignment: MisalignmentSignal,
    ) -> RiskAssessment:
        """
        Run all risk checks and produce a comprehensive assessment.

        Args:
            financial: Financial metrics and assessments.
            sentiment: Sentiment profile from news analysis.
            misalignment: Detected sentiment-financial divergence.

        Returns:
            RiskAssessment with categorized risk items.
        """
        risks = []

        # Run each category of risk checks
        risks.extend(self._check_valuation_risk(financial))
        risks.extend(self._check_financial_risk(financial))
        risks.extend(self._check_volatility_risk(financial))
        risks.extend(self._check_sentiment_risk(sentiment))
        risks.extend(self._check_misalignment_risk(misalignment))
        risks.extend(self._check_concentration_risk(financial, sentiment))

        # Calculate overall risk score and level
        risk_score = self._calculate_risk_score(risks)
        risk_level = self._classify_risk_level(risk_score)

        summary = self._build_summary(risks, risk_level, risk_score)

        assessment = RiskAssessment(
            overall_risk_level=risk_level,
            risk_score=risk_score,
            risks=risks,
            summary=summary,
        )

        logger.info(
            f"Risk analysis: {risk_level.value} "
            f"(score={risk_score:.2f}, {len(risks)} risks identified)"
        )
        return assessment

    # ---------------------------------------------------------------
    # Valuation Risk
    # ---------------------------------------------------------------
    def _check_valuation_risk(self, financial: FinancialSnapshot) -> list[RiskItem]:
        """Check for overvaluation signals."""
        risks = []

        for m in financial.valuation_metrics:
            if m.value is None:
                continue

            if "Trailing P/E" in m.name and m.value > 50:
                risks.append(RiskItem(
                    category="valuation",
                    title="Elevated Trailing P/E",
                    description=(
                        f"Trailing P/E of {m.value:.1f}x significantly exceeds "
                        f"market averages (~20-25x). The stock is priced for "
                        f"sustained high growth that may not materialize."
                    ),
                    severity=RiskSeverity.MEDIUM if m.value < 80 else RiskSeverity.HIGH,
                    probability="moderate",
                    evidence=f"Trailing P/E: {m.formatted}",
                    mitigation="Check Forward P/E and PEG ratio for growth-adjusted valuation.",
                ))
            elif "Price/Book" in m.name and m.value > 20:
                risks.append(RiskItem(
                    category="valuation",
                    title="High Price-to-Book Ratio",
                    description=(
                        f"Price/Book of {m.value:.1f}x indicates the market "
                        f"is paying a significant premium over book value."
                    ),
                    severity=RiskSeverity.LOW if m.value < 30 else RiskSeverity.MEDIUM,
                    probability="low",
                    evidence=f"Price/Book: {m.formatted}",
                    mitigation="Common for tech/IP-heavy companies. Assess intangible asset value.",
                ))
            elif "Price/Sales" in m.name and m.value > 20:
                risks.append(RiskItem(
                    category="valuation",
                    title="Stretched Price-to-Sales",
                    description=(
                        f"Price/Sales of {m.value:.1f}x is well above typical "
                        f"ranges, suggesting the market has very high revenue expectations."
                    ),
                    severity=RiskSeverity.MEDIUM,
                    probability="moderate",
                    evidence=f"Price/Sales: {m.formatted}",
                    mitigation="Evaluate revenue growth trajectory and market size.",
                ))

        return risks

    # ---------------------------------------------------------------
    # Financial Risk
    # ---------------------------------------------------------------
    def _check_financial_risk(self, financial: FinancialSnapshot) -> list[RiskItem]:
        """Check for balance sheet and profitability risks."""
        risks = []

        for m in financial.leverage_metrics:
            if m.value is None:
                continue
            if "Debt-to-Equity" in m.name and m.value > 1.5:
                risks.append(RiskItem(
                    category="financial",
                    title="High Leverage",
                    description=(
                        f"Debt-to-Equity of {m.value:.2f} indicates significant "
                        f"financial leverage. Interest rate sensitivity is elevated."
                    ),
                    severity=RiskSeverity.HIGH if m.value > 2.5 else RiskSeverity.MEDIUM,
                    probability="moderate",
                    evidence=f"D/E Ratio: {m.formatted}",
                    mitigation="Review interest coverage ratio and debt maturity schedule.",
                ))

        for m in financial.liquidity_metrics:
            if m.value is None:
                continue
            if "Current Ratio" in m.name and m.value < 1.0:
                risks.append(RiskItem(
                    category="financial",
                    title="Liquidity Concern",
                    description=(
                        f"Current ratio of {m.value:.2f} is below 1.0, meaning "
                        f"short-term liabilities exceed short-term assets."
                    ),
                    severity=RiskSeverity.HIGH,
                    probability="moderate",
                    evidence=f"Current Ratio: {m.formatted}",
                    mitigation="Check credit facilities and upcoming debt maturities.",
                ))

        # Margin pressure
        for m in financial.profitability_metrics:
            if m.value is None:
                continue
            if "Net Profit Margin" in m.name and m.value < 5:
                risks.append(RiskItem(
                    category="financial",
                    title="Thin Profit Margins",
                    description=(
                        f"Net profit margin of {m.value:.1f}% leaves little "
                        f"room for error. Any cost increase or revenue shortfall "
                        f"could push the company into losses."
                    ),
                    severity=RiskSeverity.MEDIUM,
                    probability="moderate",
                    evidence=f"Net Margin: {m.formatted}",
                    mitigation="Evaluate cost structure and pricing power.",
                ))

        return risks

    # ---------------------------------------------------------------
    # Volatility Risk
    # ---------------------------------------------------------------
    def _check_volatility_risk(self, financial: FinancialSnapshot) -> list[RiskItem]:
        """Check for stock volatility risks."""
        risks = []

        for m in financial.leverage_metrics:
            if m.value is None:
                continue
            if "Beta" in m.name and m.value > 1.5:
                risks.append(RiskItem(
                    category="market",
                    title="High Stock Volatility",
                    description=(
                        f"Beta of {m.value:.2f} means the stock is "
                        f"{m.value:.0%} more volatile than the S&P 500. "
                        f"Expect amplified moves in both directions."
                    ),
                    severity=RiskSeverity.MEDIUM if m.value < 2.0 else RiskSeverity.HIGH,
                    probability="high",
                    evidence=f"Beta: {m.formatted}",
                    mitigation="Consider position sizing relative to portfolio risk tolerance.",
                ))

        return risks

    # ---------------------------------------------------------------
    # Sentiment Risk
    # ---------------------------------------------------------------
    def _check_sentiment_risk(self, sentiment: SentimentProfile) -> list[RiskItem]:
        """Check for sentiment-driven risks."""
        risks = []

        if sentiment.overall_direction == SentimentDirection.BEARISH:
            risks.append(RiskItem(
                category="sentiment",
                title="Negative Market Sentiment",
                description=(
                    f"Market sentiment is predominantly bearish "
                    f"({sentiment.bearish_count} negative signals). "
                    f"Bearish sentiment can create selling pressure "
                    f"regardless of fundamentals."
                ),
                severity=RiskSeverity.MEDIUM,
                probability="moderate",
                evidence=sentiment.summary,
                mitigation="Monitor for sentiment reversal catalysts (e.g., earnings beat).",
            ))

        if sentiment.overall_direction == SentimentDirection.MIXED:
            risks.append(RiskItem(
                category="sentiment",
                title="Uncertain Market Sentiment",
                description=(
                    "Sentiment signals are mixed, suggesting market uncertainty "
                    "about the company's direction. Mixed sentiment often "
                    "precedes increased volatility."
                ),
                severity=RiskSeverity.LOW,
                probability="moderate",
                evidence=sentiment.summary,
                mitigation="Wait for clarity from upcoming earnings or catalysts.",
            ))

        # Geopolitical theme risk
        geo_themes = [t for t in sentiment.dominant_themes if "Geopolit" in t]
        if geo_themes:
            risks.append(RiskItem(
                category="geopolitical",
                title="Geopolitical Risk Exposure",
                description=(
                    "Geopolitical themes detected in news coverage. "
                    "Trade tensions, export restrictions, or regional "
                    "instability may impact operations or supply chain."
                ),
                severity=RiskSeverity.MEDIUM,
                probability="moderate",
                evidence="Geopolitical themes in news analysis.",
                mitigation="Assess geographic revenue diversification and supply chain redundancy.",
            ))

        return risks

    # ---------------------------------------------------------------
    # Misalignment Risk
    # ---------------------------------------------------------------
    def _check_misalignment_risk(self, misalignment: MisalignmentSignal) -> list[RiskItem]:
        """Convert misalignment signals into risk items."""
        if not misalignment.detected:
            return []

        return [RiskItem(
            category="misalignment",
            title=f"Sentiment-Financial Misalignment: {misalignment.misalignment_type.value}",
            description=misalignment.explanation,
            severity=misalignment.severity,
            probability="moderate",
            evidence=(
                f"Sentiment: {misalignment.sentiment_evidence[:150]}. "
                f"Financials: {misalignment.financial_evidence[:150]}"
            ),
            mitigation=misalignment.recommendation,
        )]

    # ---------------------------------------------------------------
    # Concentration Risk
    # ---------------------------------------------------------------
    def _check_concentration_risk(
        self,
        financial: FinancialSnapshot,
        sentiment: SentimentProfile,
    ) -> list[RiskItem]:
        """Check for AI/technology concentration risk."""
        risks = []

        ai_themes = [t for t in sentiment.dominant_themes if "AI" in t.upper()]
        if ai_themes:
            risks.append(RiskItem(
                category="concentration",
                title="AI Industry Concentration",
                description=(
                    "Significant revenue or narrative dependence on AI/ML markets. "
                    "The AI investment cycle may slow if enterprise ROI on AI "
                    "spending is questioned or if regulation increases."
                ),
                severity=RiskSeverity.MEDIUM,
                probability="low",
                evidence="AI theme prominence in news and business description.",
                mitigation="Assess revenue diversification beyond AI/data center segment.",
            ))

        return risks

    # ---------------------------------------------------------------
    # Scoring & Aggregation
    # ---------------------------------------------------------------
    def _calculate_risk_score(self, risks: list[RiskItem]) -> float:
        """
        Calculate an overall risk score from individual risk items.

        Score: 0.0 (very safe) to 1.0 (very risky).

        DEFECT D12, fixed 2026-08-17. The previous formula was:

            min(1.0, total / max(len(risks) * 0.6, 1))

        which for two or more risks is just `mean_severity / 0.6` — the count
        cancels out, so it never produced the "diminishing returns" its own
        comment claimed. Concretely: ANY set of two or more MEDIUM risks
        scored 0.833, and 0.8 is the CRITICAL threshold. Two moderate concerns
        about Apple were therefore reported as CRITICAL risk, which is how
        that verdict survived the D2 fix that was supposed to remove it.

        The replacement has three explicit terms:
          - never softer than the worst single risk, less one notch
          - never softer than the average across all risks
          - a small, capped uplift for sheer volume of findings
        """
        if not risks:
            return 0.2  # Baseline — no risks found is not zero risk

        severity_scores = {
            RiskSeverity.LOW: 0.2,
            RiskSeverity.MEDIUM: 0.5,
            RiskSeverity.HIGH: 0.8,
            RiskSeverity.CRITICAL: 1.0,
        }
        scores = [severity_scores.get(r.severity, 0.3) for r in risks]

        mean = sum(scores) / len(scores)
        worst = max(scores)
        # Capped below the width of one severity band, so volume alone can
        # never promote a pile of trivial findings into a higher band.
        volume_uplift = min(0.10, 0.05 * (len(risks) - 1))

        score = max(mean, worst - 0.15) + volume_uplift
        return round(min(1.0, score), 2)

    def _classify_risk_level(self, score: float) -> RiskSeverity:
        """
        Classify overall risk level from score.

        Thresholds sit BETWEEN the per-severity scores (LOW 0.2, MEDIUM 0.5,
        HIGH 0.8, CRITICAL 1.0), not on top of them. CRITICAL was previously
        `>= 0.8`, exactly the score of a single HIGH risk — so one high risk
        was always escalated a band, and nothing could ever report as HIGH.
        """
        if score >= 0.90:
            return RiskSeverity.CRITICAL
        elif score >= 0.65:
            return RiskSeverity.HIGH
        elif score >= 0.35:
            return RiskSeverity.MEDIUM
        else:
            return RiskSeverity.LOW

    def _build_summary(
        self,
        risks: list[RiskItem],
        level: RiskSeverity,
        score: float,
    ) -> str:
        """Generate a human-readable risk summary."""
        if not risks:
            return "No significant risks identified from available evidence."

        categories = set(r.category for r in risks)
        high_risks = [r for r in risks if r.severity in (RiskSeverity.HIGH, RiskSeverity.CRITICAL)]

        parts = [
            f"Overall risk level: {level.value} (score: {score:.2f}). "
            f"Identified {len(risks)} risk factors across "
            f"{len(categories)} categories."
        ]

        if high_risks:
            high_names = [r.title for r in high_risks[:3]]
            parts.append(f"Key concerns: {'; '.join(high_names)}.")

        return " ".join(parts)
