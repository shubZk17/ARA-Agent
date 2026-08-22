"""
Tier 1 — golden-value tests. The highest-ROI tests in the plan.

Every one of D1, D2 and D3 was a number arriving in the wrong unit, and all
three survived four months and a 98.81%-scoring evaluation because nothing
ever asserted a value. These assert values.

Written to assert CORRECT behaviour, not observed behaviour: snapshotting
what the code did in August would have enshrined `total_debt == 84.34` as
expected and made the bug permanent.
"""

from __future__ import annotations

import pytest

from analysis.financial_engine import FinancialAnalysisEngine
from analysis.risk_analyzer import RiskAnalyzer
from tools.financial_metrics import FinancialMetricsTool
from tools.units import canonical_payload


# ===================================================================
# Unit conversion — the exact values, from frozen real responses
# ===================================================================

def test_dollar_amounts_keep_every_digit(aapl_info):
    """D1: no suffix, no rounding, no 10^9 error."""
    p = canonical_payload(aapl_info)
    assert p["total_debt"] == 84_343_996_416
    assert p["total_cash"] == 62_399_000_576
    assert p["free_cash_flow"] == 107_721_875_456


def test_debt_to_equity_is_a_ratio_not_a_percent(aapl_info, ko_info, nvda_info):
    """D2: yfinance reports a percent; every threshold in the system is a ratio."""
    assert canonical_payload(aapl_info)["debt_to_equity"] == pytest.approx(0.78445)
    assert canonical_payload(ko_info)["debt_to_equity"] == pytest.approx(1.15519)
    assert canonical_payload(nvda_info)["debt_to_equity"] == pytest.approx(0.06555)


def test_dividend_yield_is_not_multiplied_by_100(ko_info, aapl_info):
    """D3: KO yields 2.42%, not 242%."""
    assert canonical_payload(ko_info)["dividend_yield"] == pytest.approx(2.42)
    assert canonical_payload(aapl_info)["dividend_yield"] == pytest.approx(0.35)


def test_fractions_still_become_percents(aapl_info):
    """The other half of D3: margins ARE fractions and must be scaled."""
    p = canonical_payload(aapl_info)
    assert p["profit_margin"] == pytest.approx(27.618998)
    assert p["payout_ratio"] == pytest.approx(12.04, abs=0.01)
    # Apple's ROE really is ~149% — a large number here is not a unit bug.
    assert p["roe"] > 100


def test_missing_fields_stay_missing(aapl_info):
    """A metric with no data must be absent, not defaulted to zero."""
    info = dict(aapl_info)
    info.pop("totalDebt", None)
    info["totalCash"] = None
    p = canonical_payload(info)
    assert "total_debt" not in p
    assert "total_cash" not in p


# ===================================================================
# Grading — the assessments those units drive
# ===================================================================

def _insights(info: dict) -> dict:
    engine = FinancialAnalysisEngine()
    return {i.name: i for i in engine._calculate_insights(canonical_payload(info))}


def test_leverage_grading_matches_reality(aapl_info, nvda_info, ko_info):
    """
    Before D2 was fixed, all three of these graded "Elevated" and fired a
    high-severity leverage risk. Only KO is genuinely above 1.0.
    """
    assert _insights(aapl_info)["Debt-to-Equity Ratio"].assessment.startswith("Moderate")
    assert _insights(nvda_info)["Debt-to-Equity Ratio"].assessment.startswith("Attractive")
    assert _insights(ko_info)["Debt-to-Equity Ratio"].assessment.startswith("Moderate")


def test_balance_sheet_no_longer_inverted(aapl_info):
    """$84B of debt is not "Attractive" and $62B of cash is not "Weak"."""
    ins = _insights(aapl_info)
    assert ins["Total Cash"].assessment.startswith("Strong")
    assert ins["Total Debt"].formatted == "$84,343,996,416"


def test_no_false_high_leverage_risk(aapl_info, nvda_info):
    """
    The exact failure from the 2026-08-16 AAPL run: a spurious "High Leverage"
    risk drove overall risk to CRITICAL on two of the least-levered
    megacaps in existence.
    """
    engine = FinancialAnalysisEngine()
    for info in (aapl_info, nvda_info):
        snapshot = engine.analyze([], [])
        snapshot.leverage_metrics = [
            i for i in engine._calculate_insights(canonical_payload(info))
            if i.category == "leverage"
        ]
        risks = RiskAnalyzer()._check_financial_risk(snapshot)
        assert not [r for r in risks if r.title == "High Leverage"]


def test_liquidity_metrics_can_populate(aapl_info):
    """
    D8 was undercounted. free_cash_flow, current_ratio and quick_ratio were
    all defined with regexes that no tool output ever matched.
    """
    p = canonical_payload(aapl_info)
    for key in ("free_cash_flow", "current_ratio", "quick_ratio"):
        assert key in p, f"{key} still cannot populate"


# ===================================================================
# render() — the string the LLM sees, built from the same payload
# ===================================================================

def test_render_is_derived_from_the_payload(aapl_info):
    payload = canonical_payload(aapl_info)
    payload.update(ticker="AAPL", company_name="Apple Inc.")
    text = FinancialMetricsTool().render(payload)

    assert "Total Debt: $84.34B" in text     # human-readable, for the LLM
    assert "Debt/Equity: 0.78" in text       # a ratio, not 78.44
    assert "Dividend Yield: 0.35%" in text   # not 35.00%
    assert "Free Cash Flow: $107.72B" in text


def test_regex_fallback_still_understands_suffixes():
    """
    Replayed episodic states carry no structured payload, so the regex path
    stays live. It must not silently drop the B again.
    """
    engine = FinancialAnalysisEngine()
    raw = engine._extract_metrics_from_observations(
        [], ["Total Debt: $84.34B\n  Total Cash: $412,500.00\n  Debt/Equity: 0.78"]
    )
    assert raw["total_debt"] == pytest.approx(84.34e9)
    assert raw["total_cash"] == 412_500.0
    assert raw["debt_to_equity"] == pytest.approx(0.78)


def test_structured_path_is_preferred_over_regex(aapl_info):
    """
    analysis/engine.py used to pass tool_calls in and never look at them.
    Structured values must win, and must be exact where regex is rounded.
    """
    from agent.state import ToolCall

    payload = canonical_payload(aapl_info)
    payload.update(ticker="AAPL", company_name="Apple Inc.")
    call = ToolCall(
        tool_name="get_financial_metrics",
        tool_output=FinancialMetricsTool().render(payload),
        tool_output_structured=payload,
    )

    snapshot = FinancialAnalysisEngine().analyze([call], [call.tool_output])
    debt = [m for m in snapshot.leverage_metrics if m.name == "Total Debt"][0]
    assert debt.value == 84_343_996_416  # exact, not the 84.34e9 regex gives
    assert snapshot.ticker == "AAPL"
    assert snapshot.company_name == "Apple Inc."


@pytest.mark.network
def test_live_yfinance_units_still_match_fixtures():
    """
    Fixtures freeze a moment in time; yfinance could change conventions in a
    future release. This is the canary. Excluded from the default run.
    """
    import yfinance as yf

    info = yf.Ticker("KO").info
    # KO's dividend yield is ~3%. If yfinance switched to fractions this
    # would be ~0.03 and every yield in every report would be 100x wrong.
    assert 0.5 < info["dividendYield"] < 20, "dividendYield convention changed!"
    assert info["debtToEquity"] > 20, "debtToEquity convention changed!"
