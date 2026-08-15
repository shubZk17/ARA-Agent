"""
synthesis/financial_engine.py — Programmatic Financial Analysis
================================================================

WHY THIS EXISTS:
    Phase 1 tools retrieve RAW financial data. But raw numbers are
    meaningless without context:
    - "P/E of 46" — Is that good? Compared to WHAT?
    - "Revenue growth of 73%" — Sustainable? Or a one-time spike?
    - "Debt-to-Equity of 0.41" — For tech, that's low. For a bank, high.

    This engine transforms raw metrics into ASSESSED insights:
    - Categorizes each metric (valuation, profitability, growth, etc.)
    - Provides human-readable assessment ("Strong", "Concerning", etc.)
    - Identifies strengths and weaknesses
    - Produces an overall financial health score

WHAT IT IS NOT:
    - NOT an LLM call — these are deterministic, programmatic rules.
    - NOT a database lookup — benchmarks are hardcoded (sector-aware in Phase 4).
    - NOT a recommendation — it states FACTS with CONTEXT.

HOW IT CONNECTS:
    - Consumes: tool_calls from AgentState (specifically get_financial_metrics,
      get_stock_price, get_company_info outputs).
    - Produces: FinancialSnapshot (synthesis/schemas.py).
    - Used by: synthesis/engine.py (central orchestrator).
    - Also feeds: misalignment_detector.py, risk_analyzer.py.

ARCHITECTURAL TRADEOFF:
    Hardcoded benchmarks vs. dynamic sector benchmarks:
    - Current: Uses general-purpose thresholds (good enough for Phase 3).
    - Future: Sector-specific benchmarks from a DB (Phase 4).
    - Why: Building a full benchmark DB is scope creep. The current
      approach provides value NOW and is easy to upgrade later.

SCALABILITY:
    Adding a new metric = adding one entry to METRIC_DEFINITIONS.
    No other code changes needed.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from analysis.schemas import (
    FinancialSnapshot,
    MetricInsight,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ===================================================================
# Metric Definitions — The Knowledge Base
# ===================================================================
# Each definition specifies:
#   - key: The field name from tool output
#   - name: Human-readable metric name
#   - category: Grouping for the report
#   - format: How to display the value
#   - thresholds: (low, high) — below low is weak, above high is strong
#   - higher_is_better: Direction of quality
#   - assessment_fn: Optional custom assessment function

METRIC_DEFINITIONS = [
    # --- Valuation ---
    {
        "key": "trailing_pe",
        "name": "Trailing P/E Ratio",
        "category": "valuation",
        "format": "{:.2f}x",
        "thresholds": (15, 35),
        "higher_is_better": False,
        "context": "Price relative to past earnings. High P/E may indicate overvaluation or high growth expectations.",
    },
    {
        "key": "forward_pe",
        "name": "Forward P/E Ratio",
        "category": "valuation",
        "format": "{:.2f}x",
        "thresholds": (12, 30),
        "higher_is_better": False,
        "context": "Price relative to expected earnings. Lower suggests better value.",
    },
    {
        "key": "price_to_book",
        "name": "Price/Book Ratio",
        "category": "valuation",
        "format": "{:.2f}x",
        "thresholds": (1.5, 10),
        "higher_is_better": False,
        "context": "Price relative to book value. High P/B common in tech, may signal overvaluation.",
    },
    {
        "key": "price_to_sales",
        "name": "Price/Sales Ratio",
        "category": "valuation",
        "format": "{:.2f}x",
        "thresholds": (2, 15),
        "higher_is_better": False,
        "context": "Price relative to revenue. Useful for companies with volatile earnings.",
    },
    {
        "key": "peg_ratio",
        "name": "PEG Ratio",
        "category": "valuation",
        "format": "{:.2f}",
        "thresholds": (0.8, 2.0),
        "higher_is_better": False,
        "context": "P/E adjusted for growth. Below 1.0 may indicate undervaluation.",
    },
    {
        "key": "enterprise_to_ebitda",
        "name": "EV/EBITDA",
        "category": "valuation",
        "format": "{:.2f}x",
        "thresholds": (8, 25),
        "higher_is_better": False,
        "context": "Enterprise value to EBITDA. Core valuation metric, debt-adjusted.",
    },
    # --- Profitability ---
    {
        "key": "profit_margin",
        "name": "Net Profit Margin",
        "category": "profitability",
        "format": "{:.1f}%",
        "thresholds": (10, 25),
        "higher_is_better": True,
        "context": "Percentage of revenue retained as profit after all expenses.",
    },
    {
        "key": "gross_margin",
        "name": "Gross Margin",
        "category": "profitability",
        "format": "{:.1f}%",
        "thresholds": (30, 60),
        "higher_is_better": True,
        "context": "Revenue minus cost of goods sold. Indicates pricing power.",
    },
    {
        "key": "operating_margin",
        "name": "Operating Margin",
        "category": "profitability",
        "format": "{:.1f}%",
        "thresholds": (10, 25),
        "higher_is_better": True,
        "context": "Profitability from core operations, excluding interest and taxes.",
    },
    {
        "key": "roe",
        "name": "Return on Equity (ROE)",
        "category": "profitability",
        "format": "{:.1f}%",
        "thresholds": (10, 25),
        "higher_is_better": True,
        "context": "How efficiently management uses shareholders' equity to generate profit.",
    },
    {
        "key": "roa",
        "name": "Return on Assets (ROA)",
        "category": "profitability",
        "format": "{:.1f}%",
        "thresholds": (5, 15),
        "higher_is_better": True,
        "context": "How efficiently the company uses its assets to generate profit.",
    },
    {
        "key": "eps_trailing",
        "name": "Trailing EPS",
        "category": "profitability",
        "format": "${:.2f}",
        "thresholds": (1, 5),
        "higher_is_better": True,
        "context": "Earnings per share over the trailing 12 months.",
    },
    # --- Growth ---
    {
        "key": "revenue_growth",
        "name": "Revenue Growth (YoY)",
        "category": "growth",
        "format": "{:.1f}%",
        "thresholds": (5, 25),
        "higher_is_better": True,
        "context": "Year-over-year revenue growth rate. High growth attracts premium valuations.",
    },
    {
        "key": "earnings_growth",
        "name": "Earnings Growth (YoY)",
        "category": "growth",
        "format": "{:.1f}%",
        "thresholds": (5, 25),
        "higher_is_better": True,
        "context": "Year-over-year earnings growth. Sustainable growth is key.",
    },
    {
        "key": "free_cash_flow_growth",
        "name": "Free Cash Flow Growth",
        "category": "growth",
        "format": "{:.1f}%",
        "thresholds": (0, 20),
        "higher_is_better": True,
        "context": "Growth in cash available after capital expenditures.",
    },
    # --- Liquidity ---
    {
        "key": "current_ratio",
        "name": "Current Ratio",
        "category": "liquidity",
        "format": "{:.2f}",
        "thresholds": (1.0, 2.5),
        "higher_is_better": True,
        "context": "Ability to pay short-term obligations. Below 1.0 is concerning.",
    },
    {
        "key": "quick_ratio",
        "name": "Quick Ratio",
        "category": "liquidity",
        "format": "{:.2f}",
        "thresholds": (0.8, 2.0),
        "higher_is_better": True,
        "context": "Like current ratio but excludes inventory. More conservative measure.",
    },
    {
        "key": "free_cash_flow",
        "name": "Free Cash Flow",
        "category": "liquidity",
        "format": "${:,.0f}",
        "thresholds": (0, 1e9),
        "higher_is_better": True,
        "context": "Cash available after capital expenditures for dividends, buybacks, or reinvestment.",
    },
    # --- Leverage ---
    {
        "key": "debt_to_equity",
        "name": "Debt-to-Equity Ratio",
        "category": "leverage",
        "format": "{:.2f}",
        "thresholds": (0.3, 1.5),
        "higher_is_better": False,
        "context": "Total debt relative to shareholders' equity. Lower = less leveraged.",
    },
    {
        "key": "total_debt",
        "name": "Total Debt",
        "category": "leverage",
        "format": "${:,.0f}",
        "thresholds": (1e8, 1e10),
        "higher_is_better": False,
        "context": "Absolute debt level. Must be compared to cash and earnings.",
    },
    {
        "key": "total_cash",
        "name": "Total Cash",
        "category": "leverage",
        "format": "${:,.0f}",
        "thresholds": (1e8, 1e10),
        "higher_is_better": True,
        "context": "Cash and short-term investments. Provides financial flexibility.",
    },
    {
        "key": "beta",
        "name": "Beta (Volatility)",
        "category": "leverage",
        "format": "{:.2f}",
        "thresholds": (0.8, 1.5),
        "higher_is_better": False,
        "context": "Stock volatility relative to market. Above 1.0 = more volatile than S&P 500.",
    },
]


class FinancialAnalysisEngine:
    """
    Transforms raw financial data into assessed, categorized insights.

    This is a DETERMINISTIC engine — no LLM calls, no randomness.
    Given the same input, it always produces the same output.
    """

    def analyze(self, tool_calls: list, tool_observations: list[str]) -> FinancialSnapshot:
        """
        Analyze financial data from agent tool calls.

        Args:
            tool_calls: List of ToolCall objects from agent state.
            tool_observations: Raw observation strings from reasoning trace.

        Returns:
            FinancialSnapshot with categorized, assessed metrics.
        """
        # 1. Extract raw metrics from tool outputs
        raw_metrics = self._extract_metrics_from_observations(
            tool_calls, tool_observations
        )

        # 2. Extract company identity
        ticker, company_name = self._extract_identity(tool_calls, tool_observations)

        # 3. Calculate metric insights
        insights = self._calculate_insights(raw_metrics)

        # 4. Categorize
        valuation = [m for m in insights if m.category == "valuation"]
        profitability = [m for m in insights if m.category == "profitability"]
        growth = [m for m in insights if m.category == "growth"]
        liquidity = [m for m in insights if m.category == "liquidity"]
        leverage = [m for m in insights if m.category == "leverage"]

        # 5. Compute overall health score
        health_score = self._compute_health_score(insights)

        # 6. Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(insights)

        # 7. Compute data quality
        total_defined = len(METRIC_DEFINITIONS)
        available = len([m for m in insights if m.value is not None])

        snapshot = FinancialSnapshot(
            ticker=ticker,
            company_name=company_name,
            valuation_metrics=valuation,
            profitability_metrics=profitability,
            growth_metrics=growth,
            liquidity_metrics=liquidity,
            leverage_metrics=leverage,
            financial_health_score=health_score,
            overall_assessment=self._build_overall_assessment(health_score, strengths, weaknesses),
            key_strengths=strengths,
            key_weaknesses=weaknesses,
            metrics_available=available,
            metrics_missing=total_defined - available,
            evidence_quality=available / total_defined if total_defined > 0 else 0,
        )

        logger.info(
            f"Financial analysis complete: {ticker} — "
            f"{available}/{total_defined} metrics, "
            f"health={health_score:.2f}"
        )
        return snapshot

    def _extract_metrics_from_observations(
        self, tool_calls: list, observations: list[str]
    ) -> dict[str, float]:
        """
        Parse numeric metrics from tool output text.

        Uses regex patterns to extract known financial metrics
        from the free-text tool outputs.
        """
        raw = {}
        combined_text = " ".join(observations)

        # --- Regex extraction patterns ---
        patterns = {
            "trailing_pe": [r"(?:Trailing\s+)?P/E[:=\s]+(\d+\.?\d*)"],
            "forward_pe": [r"Forward\s+P/E[:=\s]+(\d+\.?\d*)"],
            "price_to_book": [r"Price[/\s]Book[:=\s]+(\d+\.?\d*)"],
            "price_to_sales": [r"Price[/\s]Sales[:=\s]+(\d+\.?\d*)"],
            "peg_ratio": [r"PEG\s+(?:Ratio)?[:=\s]+(\d+\.?\d*)"],
            "enterprise_to_ebitda": [r"EV/EBITDA[:=\s]+(\d+\.?\d*)"],
            "profit_margin": [r"(?:Net\s+)?Profit\s+Margin[:=\s]+([-\d]+\.?\d*)%?"],
            "gross_margin": [r"Gross\s+Margin[:=\s]+([-\d]+\.?\d*)%?"],
            "operating_margin": [r"Operating\s+Margin[:=\s]+([-\d]+\.?\d*)%?"],
            "roe": [r"ROE[:=\s]+([-\d]+\.?\d*)%?", r"Return\s+on\s+Equity[:=\s]+([-\d]+\.?\d*)%?"],
            "roa": [r"ROA[:=\s]+([-\d]+\.?\d*)%?", r"Return\s+on\s+Assets[:=\s]+([-\d]+\.?\d*)%?"],
            "eps_trailing": [r"(?:Trailing\s+)?EPS[:=\s]+\$?([-\d]+\.?\d*)"],
            "revenue_growth": [r"Revenue\s+Growth[:=\s]+([-\d]+\.?\d*)%?"],
            "earnings_growth": [r"Earnings\s+Growth[:=\s]+([-\d]+\.?\d*)%?"],
            "current_ratio": [r"Current\s+Ratio[:=\s]+(\d+\.?\d*)"],
            "quick_ratio": [r"Quick\s+Ratio[:=\s]+(\d+\.?\d*)"],
            "debt_to_equity": [r"Debt[/-](?:to[/-])?Equity[:=\s]+(\d+\.?\d*)"],
            "beta": [r"Beta[:=\s]+(\d+\.?\d*)"],
            "total_debt": [r"Total\s+Debt[:=\s]+\$?([\d,]+\.?\d*)"],
            "total_cash": [r"Total\s+Cash[:=\s]+\$?([\d,]+\.?\d*)"],
            "free_cash_flow": [r"Free\s+Cash\s+Flow[:=\s]+\$?([-\d,]+\.?\d*)"],
        }

        for metric_key, regex_list in patterns.items():
            for regex in regex_list:
                match = re.search(regex, combined_text, re.IGNORECASE)
                if match:
                    try:
                        val_str = match.group(1).replace(",", "")
                        raw[metric_key] = float(val_str)
                        break
                    except (ValueError, IndexError):
                        continue

        logger.debug(f"Extracted {len(raw)} raw metrics from tool outputs")
        return raw

    def _extract_identity(
        self, tool_calls: list, observations: list[str]
    ) -> tuple[str, str]:
        """Extract ticker and company name from tool outputs."""
        ticker = ""
        company_name = ""
        combined = " ".join(observations)

        # Get ticker from tool inputs
        for tc in tool_calls:
            if hasattr(tc, "tool_input"):
                t = tc.tool_input.get("ticker", "")
                if t:
                    ticker = t.upper()
                    break

        # Get company name from observations
        name_match = re.search(
            r"Company\s+Profile:\s+(.+?)\s*\(", combined, re.IGNORECASE
        )
        if name_match:
            company_name = name_match.group(1).strip()
        else:
            name_match = re.search(
                r"Financial\s+Metrics\s+for\s+(.+?)\s*\(", combined, re.IGNORECASE
            )
            if name_match:
                company_name = name_match.group(1).strip()

        return ticker, company_name

    def _calculate_insights(self, raw_metrics: dict[str, float]) -> list[MetricInsight]:
        """Convert raw metrics into assessed MetricInsight objects."""
        insights = []

        for defn in METRIC_DEFINITIONS:
            key = defn["key"]
            value = raw_metrics.get(key)

            if value is None:
                insights.append(MetricInsight(
                    name=defn["name"],
                    value=None,
                    formatted="N/A",
                    assessment="Data not available",
                    category=defn["category"],
                    evidence_sources=[],
                ))
                continue

            try:
                formatted = defn["format"].format(value)
            except (ValueError, KeyError):
                formatted = str(value)

            assessment = self._assess_metric(
                value=value,
                thresholds=defn["thresholds"],
                higher_is_better=defn["higher_is_better"],
                context=defn.get("context", ""),
            )

            insights.append(MetricInsight(
                name=defn["name"],
                value=value,
                formatted=formatted,
                assessment=assessment,
                category=defn["category"],
                evidence_sources=["get_financial_metrics", "get_stock_price"],
            ))

        return insights

    def _assess_metric(
        self,
        value: float,
        thresholds: tuple[float, float],
        higher_is_better: bool,
        context: str = "",
    ) -> str:
        """
        Generate a contextual assessment for a metric.

        Uses threshold-based classification with directional awareness.
        """
        low, high = thresholds

        if higher_is_better:
            if value >= high:
                strength = "Strong"
            elif value >= low:
                strength = "Moderate"
            else:
                strength = "Weak"
        else:
            if value <= low:
                strength = "Attractive"
            elif value <= high:
                strength = "Moderate"
            else:
                strength = "Elevated"

        return f"{strength} — {context}" if context else strength

    def _compute_health_score(self, insights: list[MetricInsight]) -> float:
        """
        Compute a 0.0–1.0 financial health score.

        Methodology:
        - Each available metric contributes a score (0, 0.5, or 1.0)
          based on where it falls relative to thresholds.
        - The final score is the average across all available metrics.
        - More missing metrics = lower ceiling (penalizes incomplete data).
        """
        scores = []

        for defn in METRIC_DEFINITIONS:
            matching = [i for i in insights if i.name == defn["name"] and i.value is not None]
            if not matching:
                continue

            value = matching[0].value
            low, high = defn["thresholds"]

            if defn["higher_is_better"]:
                if value >= high:
                    scores.append(1.0)
                elif value >= low:
                    scores.append(0.6)
                else:
                    scores.append(0.2)
            else:
                if value <= low:
                    scores.append(1.0)
                elif value <= high:
                    scores.append(0.6)
                else:
                    scores.append(0.2)

        if not scores:
            return 0.5  # No data — neutral

        return sum(scores) / len(scores)

    def _identify_strengths_weaknesses(
        self, insights: list[MetricInsight]
    ) -> tuple[list[str], list[str]]:
        """Extract key strengths and weaknesses from assessed metrics."""
        strengths = []
        weaknesses = []

        for m in insights:
            if m.value is None:
                continue
            assessment_lower = m.assessment.lower()
            if assessment_lower.startswith("strong") or assessment_lower.startswith("attractive"):
                strengths.append(f"{m.name}: {m.formatted}")
            elif assessment_lower.startswith("weak") or assessment_lower.startswith("elevated"):
                weaknesses.append(f"{m.name}: {m.formatted}")

        return strengths[:6], weaknesses[:6]

    def _build_overall_assessment(
        self,
        health_score: float,
        strengths: list[str],
        weaknesses: list[str],
    ) -> str:
        """Generate a human-readable overall assessment."""
        if health_score >= 0.8:
            quality = "excellent"
        elif health_score >= 0.65:
            quality = "strong"
        elif health_score >= 0.5:
            quality = "moderate"
        elif health_score >= 0.35:
            quality = "mixed"
        else:
            quality = "concerning"

        parts = [f"Overall financial health is {quality} (score: {health_score:.2f})."]

        if strengths:
            parts.append(f"Key strengths: {', '.join(strengths[:3])}.")
        if weaknesses:
            parts.append(f"Areas of concern: {', '.join(weaknesses[:3])}.")

        return " ".join(parts)
