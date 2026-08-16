"""
tools/units.py — Explicit unit declarations for every yfinance field we read
============================================================================

WHY THIS EXISTS:
    yfinance's unit conventions are genuinely mixed, and guessing wrong is
    what produced defects D2 and D3:

        profitMargins  = 0.27619  -> a FRACTION  -> 27.62%
        dividendYield  = 0.35     -> already a PERCENT -> 0.35%
        debtToEquity   = 78.445   -> a PERCENT   -> ratio 0.78
        totalDebt      = 84343996416 -> raw DOLLARS

    Two fields, both floats near zero, opposite meanings. There is no
    heuristic that separates them — only a declaration does.

DELIBERATELY NOT A HEURISTIC:
    An earlier idea was "if debt/equity > 20 it must be a percent". That
    fails on genuinely distressed companies (a real D/E of 25 exists) and is
    the same guessing that caused D2 in the first place. Every field below is
    declared by hand against a live API response.

    Verified against live yfinance 1.3.0 responses on 2026-08-16/17
    (AAPL and KO — KO's 2.42 dividend yield is the case that makes the
    percent-vs-fraction distinction unambiguous).

CANONICAL UNITS — what the rest of the system expects:
    dollars : raw, unscaled (84343996416, never "$84.34B")
    ratio   : 0.78 means 78%
    percent : 27.62 means 27.62%
    raw     : dimensionless (P/E, beta)

HOW IT CONNECTS:
    - tools/financial_metrics.py fetch() converts here, at the trust boundary
      where external data enters.
    - analysis/financial_engine.py consumes the already-canonical payload.

NOTE ON PLACEMENT (deviation from plan.md §5.3, which said analysis/units.py):
    This lives in tools/ because it describes *yfinance's* conventions, and
    because tools must not depend on analysis/ — the dependency runs the
    other way. The behaviour is identical.
"""

from __future__ import annotations

from typing import Any, Optional

# Conversion kinds
DOLLARS = "dollars"                        # already canonical
RAW = "raw"                                # dimensionless, already canonical
PERCENT = "percent"                        # already a percent — do NOT scale
FRACTION_TO_PERCENT = "fraction_to_percent"  # 0.276 -> 27.6
PERCENT_TO_RATIO = "percent_to_ratio"      # 78.445 -> 0.78445


# yfinance field -> (canonical metric key, conversion, canonical unit)
#
# The metric key matches METRIC_DEFINITIONS in analysis/financial_engine.py,
# so the payload drops straight into the engine with no name mapping.
YF_FIELD_UNITS: dict[str, tuple[str, str, str]] = {
    # --- Valuation (dimensionless) ---
    "trailingPE": ("trailing_pe", RAW, "raw"),
    "forwardPE": ("forward_pe", RAW, "raw"),
    "priceToBook": ("price_to_book", RAW, "raw"),
    "priceToSalesTrailing12Months": ("price_to_sales", RAW, "raw"),
    "pegRatio": ("peg_ratio", RAW, "raw"),
    "enterpriseToEbitda": ("enterprise_to_ebitda", RAW, "raw"),
    "beta": ("beta", RAW, "raw"),

    # --- Profitability (yfinance returns FRACTIONS here) ---
    "profitMargins": ("profit_margin", FRACTION_TO_PERCENT, "percent"),
    "grossMargins": ("gross_margin", FRACTION_TO_PERCENT, "percent"),
    "operatingMargins": ("operating_margin", FRACTION_TO_PERCENT, "percent"),
    "returnOnEquity": ("roe", FRACTION_TO_PERCENT, "percent"),
    "returnOnAssets": ("roa", FRACTION_TO_PERCENT, "percent"),
    "trailingEps": ("eps_trailing", DOLLARS, "dollars"),

    # --- Growth (fractions) ---
    "revenueGrowth": ("revenue_growth", FRACTION_TO_PERCENT, "percent"),
    "earningsGrowth": ("earnings_growth", FRACTION_TO_PERCENT, "percent"),

    # --- Liquidity ---
    "currentRatio": ("current_ratio", RAW, "raw"),
    "quickRatio": ("quick_ratio", RAW, "raw"),
    "freeCashflow": ("free_cash_flow", DOLLARS, "dollars"),

    # --- Leverage ---
    # THE D2 FIELD. yfinance reports a percent; thresholds are ratios.
    "debtToEquity": ("debt_to_equity", PERCENT_TO_RATIO, "ratio"),
    "totalDebt": ("total_debt", DOLLARS, "dollars"),
    "totalCash": ("total_cash", DOLLARS, "dollars"),

    # --- Display-only (no METRIC_DEFINITIONS entry, still needs correct units) ---
    # THE D3 FIELD. Already a percent — scaling it by 100 gave KO a 242% yield.
    "dividendYield": ("dividend_yield", PERCENT, "percent"),
    "payoutRatio": ("payout_ratio", FRACTION_TO_PERCENT, "percent"),
    "totalRevenue": ("total_revenue", DOLLARS, "dollars"),
    "marketCap": ("market_cap", DOLLARS, "dollars"),
}


def convert(kind: str, value: Any) -> Optional[float]:
    """Apply one conversion. Returns None for anything non-numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if kind == FRACTION_TO_PERCENT:
        return value * 100.0
    if kind == PERCENT_TO_RATIO:
        return value / 100.0
    return float(value)  # DOLLARS, RAW, PERCENT are already canonical


def canonical_payload(info: dict[str, Any]) -> dict[str, float]:
    """
    Convert a raw yfinance `.info` dict into canonical-unit values.

    Keys are metric keys (snake_case), not yfinance field names. Fields that
    are missing or non-numeric are omitted entirely rather than defaulted —
    a missing metric must stay missing so evidence quality reflects reality.
    """
    payload: dict[str, float] = {}
    for yf_field, (key, kind, _unit) in YF_FIELD_UNITS.items():
        converted = convert(kind, info.get(yf_field))
        if converted is not None:
            payload[key] = converted
    return payload


def unit_of(metric_key: str) -> str:
    """Canonical unit for a metric key — used by tests and by report rendering."""
    for key, kind, unit in YF_FIELD_UNITS.values():
        if key == metric_key:
            return unit
    return "unknown"
