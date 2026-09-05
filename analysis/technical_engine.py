"""
analysis/technical_engine.py — Deterministic grading of the price series
=========================================================================

WHY THIS EXISTS:
    tools/price_history.py produces ~25 raw numbers. A number is not a
    finding: RSI 71 is "overbought" only if you know 70 is the line, and
    -18% off the 52-week high means something different in a bull market
    than in a crash.

    This engine is the technical twin of FinancialAnalysisEngine — same
    shape, same idiom, same output type:

        TECHNICAL_DEFINITIONS   mirrors METRIC_DEFINITIONS
        MetricInsight           the identical output object
        deterministic           no LLM, no randomness, same input → same output

    Because it emits MetricInsight, report_generator.py renders technicals
    with no new rendering code at all.

THREE-WAY ASSESSMENT — the one thing this adds over the financial engine:
    Some technical readings are best in the MIDDLE, not at an extreme. RSI 90
    is not "very strong", it is stretched. Definitions can therefore set
    `higher_is_better: None`, meaning "inside the band is healthy, outside it
    in either direction is not". Financial metrics never need this, which is
    why the mode lives here rather than being retrofitted upstream.

HOW IT CONNECTS:
    - Consumes: the get_price_history payload + a HorizonProfile.
    - Produces: TechnicalSnapshot (analysis/schemas.py).
    - Used by: analysis/engine.py, as pipeline stage 2 of 7.
"""

from __future__ import annotations

from typing import Any, Optional

from analysis.schemas import MetricInsight, TechnicalSnapshot
from config.horizons import HorizonProfile
from config.logging import get_logger

logger = get_logger(__name__)

# The tool whose structured payload this engine reads.
PRICE_HISTORY_TOOL = "get_price_history"

# Same shape as METRIC_DEFINITIONS (analysis/financial_engine.py:71), with one
# addition: higher_is_better may be None, meaning "the band is the good place".
#
#   key / name / category / format / thresholds (low, high) /
#   higher_is_better (True | False | None) / context
TECHNICAL_DEFINITIONS: list[dict[str, Any]] = [
    # --- Trend ---
    {
        "key": "price_vs_sma_50",
        "name": "Price vs 50-Day SMA",
        "category": "technical",
        "format": "{:+.2f}%",
        "thresholds": (-2.0, 5.0),
        "higher_is_better": True,
        "context": "Position relative to the intermediate trend. Above it is constructive.",
    },
    {
        "key": "price_vs_sma_200",
        "name": "Price vs 200-Day SMA",
        "category": "technical",
        "format": "{:+.2f}%",
        "thresholds": (-5.0, 5.0),
        "higher_is_better": True,
        "context": "The primary trend line. Below it, most systematic strategies are flat.",
    },
    {
        "key": "pct_from_52w_high",
        "name": "Distance from 52-Week High",
        "category": "technical",
        "format": "{:+.2f}%",
        "thresholds": (-25.0, -7.0),
        "higher_is_better": True,
        "context": "Near the high is momentum; deeply below it is a broken trend.",
    },
    # --- Momentum ---
    {
        "key": "rsi_14",
        "name": "RSI (14)",
        "category": "technical",
        "format": "{:.1f}",
        "thresholds": (35.0, 68.0),
        "higher_is_better": None,   # band metric — extremes are bad in BOTH directions
        "context": "Below 35 is oversold, above 68 is stretched. The middle is healthy.",
    },
    {
        "key": "macd_hist_pct",
        "name": "MACD Histogram (% of price)",
        "category": "technical",
        "format": "{:+.3f}%",
        "thresholds": (0.0, 0.30),
        "higher_is_better": True,
        "context": "Positive means the short trend is accelerating away from the long one.",
    },
    {
        "key": "return_3m",
        "name": "3-Month Return",
        "category": "technical",
        "format": "{:+.2f}%",
        "thresholds": (0.0, 12.0),
        "higher_is_better": True,
        "context": "The swing-horizon momentum window.",
    },
    {
        "key": "return_1y",
        "name": "1-Year Return",
        "category": "technical",
        "format": "{:+.2f}%",
        "thresholds": (0.0, 20.0),
        "higher_is_better": True,
        "context": "Long-horizon price trend, independent of what fundamentals claim.",
    },
    # --- Risk ---
    {
        "key": "realized_vol_30d",
        "name": "Realized Volatility (30d, annualized)",
        "category": "technical",
        "format": "{:.1f}%",
        "thresholds": (25.0, 45.0),
        "higher_is_better": False,
        "context": "How violently the stock actually moves. Sizes the position, not the verdict.",
    },
    {
        "key": "max_drawdown_1y",
        "name": "Max Drawdown (1y)",
        "category": "technical",
        "format": "{:.1f}%",
        "thresholds": (-35.0, -15.0),
        "higher_is_better": True,   # closer to zero is better; values are negative
        "context": "Worst peak-to-trough fall in the last year — the pain of holding it.",
    },
    {
        "key": "relative_volume",
        "name": "Relative Volume",
        "category": "technical",
        "format": "{:.2f}x",
        "thresholds": (0.8, 1.5),
        "higher_is_better": None,   # unusual volume in either direction is a flag
        "context": "Today's volume against its 30-day average. Conviction, or a lack of it.",
    },
]

# Which readings define "the trend", and how much each says about it. Kept
# separate from the grading table above because trend is a directional
# question and volatility has no direction.
TREND_COMPONENTS: dict[str, float] = {
    "price_vs_sma_200": 0.35,
    "price_vs_sma_50": 0.30,
    "macd_hist_pct": 0.20,
    "pct_from_52w_high": 0.15,
}

KEY_LEVEL_FIELDS = ("sma_20", "sma_50", "sma_200", "high_52w", "low_52w", "atr_14")


class TechnicalAnalysisEngine:
    """Grades a price-history payload against horizon-aware thresholds."""

    def analyze(
        self,
        tool_calls: list,
        profile: Optional[HorizonProfile] = None,
    ) -> TechnicalSnapshot:
        """
        Build a TechnicalSnapshot from whatever price history the agent gathered.

        Returns an `available=False` snapshot when no price-history tool ran —
        which is a legitimate outcome, not an error. The agent may have been
        asked a purely fundamental question, and pretending to a technical read
        without a series is exactly the kind of confident emptiness Phase 5
        was about removing.
        """
        payload = self._latest_payload(tool_calls)
        if not payload:
            logger.info(
                "No get_price_history payload in this run — technical analysis "
                "unavailable (not an error)."
            )
            return TechnicalSnapshot(available=False, summary="No price history gathered.")

        insights = self._calculate_insights(payload, profile)
        trend_score = self._compute_trend_score(payload)

        snapshot = TechnicalSnapshot(
            ticker=str(payload.get("ticker", "")),
            as_of=str(payload.get("as_of", "")),
            available=True,
            metrics=insights,
            trend_score=trend_score,
            trend_label=self._trend_label(trend_score),
            last_close=self._number(payload.get("last_close")),
            key_levels={
                field: float(payload[field])
                for field in KEY_LEVEL_FIELDS
                if isinstance(payload.get(field), (int, float))
            },
            realized_volatility=self._number(payload.get("realized_vol_30d")),
            max_drawdown=self._number(payload.get("max_drawdown_1y")),
        )
        snapshot.summary = self._build_summary(snapshot, payload)

        logger.info(
            f"Technical analysis complete: {snapshot.ticker} — "
            f"trend={snapshot.trend_label} ({trend_score:.2f}), "
            f"{len([m for m in insights if m.value is not None])}/"
            f"{len(TECHNICAL_DEFINITIONS)} readings"
        )
        return snapshot

    # -----------------------------------------------------------------
    # Extraction
    # -----------------------------------------------------------------

    def _latest_payload(self, tool_calls: list) -> dict[str, Any]:
        """
        The most recent successful price-history payload.

        Later calls win: if the agent re-fetched, the second fetch is the
        current one. Structured payloads only — there is deliberately no regex
        fallback here, because unlike the financial metrics these numbers have
        never existed in a formatted string that could be scraped back.
        """
        payload: dict[str, Any] = {}
        for call in tool_calls:
            if getattr(call, "tool_name", "") != PRICE_HISTORY_TOOL:
                continue
            if not getattr(call, "success", True):
                continue
            candidate = getattr(call, "tool_output_structured", None)
            if isinstance(candidate, dict) and candidate:
                payload = candidate
        return payload

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    # -----------------------------------------------------------------
    # Grading
    # -----------------------------------------------------------------

    def _calculate_insights(
        self,
        payload: dict[str, Any],
        profile: Optional[HorizonProfile],
    ) -> list[MetricInsight]:
        insights = []

        for defn in TECHNICAL_DEFINITIONS:
            value = self._number(payload.get(defn["key"]))

            if value is None:
                insights.append(MetricInsight(
                    name=defn["name"],
                    value=None,
                    formatted="N/A",
                    assessment="Data not available",
                    category="technical",
                    evidence_sources=[],
                ))
                continue

            thresholds = defn["thresholds"]
            if profile is not None:
                thresholds = profile.thresholds_for(defn["key"], thresholds)

            insights.append(MetricInsight(
                name=defn["name"],
                value=value,
                formatted=self._format(defn["format"], value),
                assessment=self._assess(
                    value, thresholds, defn["higher_is_better"], defn.get("context", "")
                ),
                category="technical",
                evidence_sources=[PRICE_HISTORY_TOOL],
            ))

        return insights

    @staticmethod
    def _format(spec: str, value: float) -> str:
        try:
            return spec.format(value)
        except (ValueError, KeyError):
            return str(value)

    def _assess(
        self,
        value: float,
        thresholds: tuple[float, float],
        higher_is_better: Optional[bool],
        context: str,
    ) -> str:
        """
        Grade one reading. Wording matches the financial engine exactly
        ("Strong" / "Moderate" / "Weak" / "Attractive" / "Elevated") because
        _identify_strengths_weaknesses keys off those first words.
        """
        low, high = thresholds

        if higher_is_better is None:
            # Band metric — the middle is the good place.
            if value < low:
                strength = "Weak"
            elif value > high:
                strength = "Elevated"
            else:
                strength = "Strong"
        elif higher_is_better:
            strength = "Strong" if value >= high else "Moderate" if value >= low else "Weak"
        else:
            strength = "Attractive" if value <= low else "Moderate" if value <= high else "Elevated"

        return f"{strength} — {context}" if context else strength

    # -----------------------------------------------------------------
    # Trend
    # -----------------------------------------------------------------

    def _compute_trend_score(self, payload: dict[str, Any]) -> float:
        """
        0.0–1.0 directional read, weighted over TREND_COMPONENTS.

        Each component is squashed independently to 0–1 so that one wild
        reading cannot dominate: being 40% above the 200-day is not four times
        as bullish as being 10% above it.

        Missing components drop out and the remaining weights renormalize —
        a stock with under 200 bars of history still gets a trend score from
        what it does have, rather than a misleading 0.5.
        """
        total_weight = 0.0
        total = 0.0

        for key, weight in TREND_COMPONENTS.items():
            value = self._number(payload.get(key))
            if value is None:
                continue
            total += weight * self._squash(key, value)
            total_weight += weight

        if not total_weight:
            return 0.5
        return round(total / total_weight, 4)

    @staticmethod
    def _squash(key: str, value: float) -> float:
        """Map one raw reading onto 0–1, saturating at its practical extremes."""
        # (neutral point, half-range) — value at neutral scores 0.5, and
        # neutral ± half_range saturates at 1.0 / 0.0.
        scales = {
            "price_vs_sma_50": (0.0, 10.0),
            "price_vs_sma_200": (0.0, 20.0),
            "macd_hist_pct": (0.0, 0.6),
            "pct_from_52w_high": (-15.0, 15.0),
        }
        neutral, half_range = scales.get(key, (0.0, 10.0))
        return max(0.0, min(1.0, 0.5 + (value - neutral) / (2 * half_range)))

    @staticmethod
    def _trend_label(score: float) -> str:
        if score >= 0.70:
            return "uptrend"
        if score >= 0.55:
            return "mild uptrend"
        if score >= 0.45:
            return "sideways"
        if score >= 0.30:
            return "mild downtrend"
        return "downtrend"

    def _build_summary(self, snapshot: TechnicalSnapshot, payload: dict[str, Any]) -> str:
        parts = [
            f"{snapshot.ticker or 'The stock'} is in a {snapshot.trend_label} "
            f"(trend score {snapshot.trend_score:.2f}) as of {snapshot.as_of}."
        ]

        rsi_value = self._number(payload.get("rsi_14"))
        if rsi_value is not None:
            state = (
                "oversold" if rsi_value < 35
                else "overbought" if rsi_value > 68
                else "neutral"
            )
            parts.append(f"RSI is {rsi_value:.0f} ({state}).")

        if snapshot.realized_volatility is not None:
            parts.append(
                f"Realized volatility is {snapshot.realized_volatility:.0f}% annualized"
                + (
                    f", with a worst drawdown of {snapshot.max_drawdown:.0f}% "
                    f"over the past year."
                    if snapshot.max_drawdown is not None else "."
                )
            )

        sma_200 = snapshot.key_levels.get("sma_200")
        if sma_200 and snapshot.last_close:
            side = "above" if snapshot.last_close >= sma_200 else "below"
            parts.append(
                f"Price is {side} the 200-day moving average ({sma_200:,.2f})."
            )

        return " ".join(parts)
