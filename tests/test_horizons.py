"""
Phase 6 gate, as a test.

plan.md's gate for Phase 6: "same ticker, two horizons -> two materially
different theses, each justified by different evidence. If they come out the
same, the weighting isn't doing anything."

That is a claim about code, so it belongs here rather than in a one-off
manual run. The scenario below is deliberately the interesting one: a company
with solid fundamentals whose price is in a downtrend. A long-horizon reader
should see an opportunity; a short-horizon one should see a falling knife.
Anything that collapses those into the same verdict has broken the phase.

No network, no LLM — ToolCall objects are constructed directly.
"""

from __future__ import annotations

import pytest

from agent.state import ToolCall, create_initial_state
from analysis.engine import SynthesisEngine
from analysis.technical_engine import TechnicalAnalysisEngine
from config.horizons import (
    CATEGORIES,
    HORIZON_PROFILES,
    RISK_PROFILES,
    get_horizon_profile,
    get_risk_profile,
)


# ===================================================================
# The table itself
# ===================================================================

@pytest.mark.parametrize("horizon", sorted(HORIZON_PROFILES))
def test_category_weights_are_complete_and_normalized(horizon):
    """
    Every profile must weight every category and sum to 1.0. A missing
    category would silently weigh zero — the metric would still appear in the
    report while contributing nothing, which is the worst of both.
    """
    profile = HORIZON_PROFILES[horizon]

    assert set(profile.category_weights) == set(CATEGORIES)
    assert sum(profile.category_weights.values()) == pytest.approx(1.0)
    assert all(w >= 0 for w in profile.category_weights.values())


def test_the_two_horizons_disagree_about_what_matters():
    """If both profiles weighted evidence the same way, the phase is a no-op."""
    short = HORIZON_PROFILES["short_term"]
    long_term = HORIZON_PROFILES["long_term"]

    assert short.weight("technical") > long_term.weight("technical") * 3
    assert long_term.weight("valuation") > short.weight("valuation") * 2
    assert short.review_days < long_term.review_days


@pytest.mark.parametrize("value", ["", None, "medium_term", "LONG_TERM  "])
def test_unknown_horizons_fall_back_rather_than_raise(value):
    """
    Horizons arrive from CLI flags, HTTP bodies and replayed episodic state.
    An unrecognised one must not take down an otherwise valid analysis.
    """
    assert get_horizon_profile(value).horizon in HORIZON_PROFILES
    assert get_risk_profile(value).profile in RISK_PROFILES


def test_horizon_strings_round_trip_through_json():
    """State is serialized to JSON for episodic memory — enums would not survive."""
    import json

    state = create_initial_state("test", horizon="short_term")
    restored = json.loads(json.dumps({"horizon": state["horizon"]}))
    assert restored["horizon"] == "short_term"
    assert get_horizon_profile(restored["horizon"]).horizon == "short_term"


# ===================================================================
# The gate
# ===================================================================

# Solid fundamentals: profitable, growing, low leverage, fair multiple.
STRONG_FUNDAMENTALS = {
    "ticker": "TEST",
    "company_name": "Test Corp",
    "trailing_pe": 22.0,
    "forward_pe": 18.0,
    "price_to_book": 4.0,
    "price_to_sales": 5.0,
    "profit_margin": 24.0,
    "gross_margin": 58.0,
    "operating_margin": 26.0,
    "roe": 31.0,
    "roa": 14.0,
    "eps_trailing": 6.4,
    "revenue_growth": 18.0,
    "earnings_growth": 21.0,
    "current_ratio": 2.1,
    "quick_ratio": 1.7,
    "free_cash_flow": 22_000_000_000.0,
    "debt_to_equity": 0.35,
    "total_debt": 9_000_000_000.0,
    "total_cash": 31_000_000_000.0,
    "beta": 1.1,
}

# ...and a price series that has been falling all year.
BROKEN_TREND = {
    "ticker": "TEST",
    "as_of": "2026-08-17",
    "bars": 500,
    "last_close": 62.0,
    "return_1w": -4.0,
    "return_1m": -11.0,
    "return_3m": -24.0,
    "return_1y": -38.0,
    "high_52w": 118.0,
    "low_52w": 60.0,
    "pct_from_52w_high": -47.5,
    "pct_from_52w_low": 3.3,
    "realized_vol_30d": 52.0,
    "max_drawdown_1y": -47.5,
    "sma_20": 68.0,
    "price_vs_sma_20": -8.8,
    "sma_50": 76.0,
    "price_vs_sma_50": -18.4,
    "sma_200": 92.0,
    "price_vs_sma_200": -32.6,
    "rsi_14": 28.0,
    "macd_hist_pct": -0.45,
    "atr_14": 2.4,
    "atr_pct": 3.9,
    "relative_volume": 1.4,
}


def _state(horizon: str) -> dict:
    state = create_initial_state("Analyze TEST", horizon=horizon)
    state["tool_calls"] = [
        ToolCall(
            tool_name="get_financial_metrics",
            tool_input={"ticker": "TEST"},
            tool_output="(rendered elsewhere)",
            tool_output_structured=STRONG_FUNDAMENTALS,
        ),
        ToolCall(
            tool_name="get_price_history",
            tool_input={"ticker": "TEST"},
            tool_output="(rendered elsewhere)",
            tool_output_structured=BROKEN_TREND,
        ),
    ]
    return state


@pytest.fixture(scope="module")
def reports():
    engine = SynthesisEngine()
    return {h: engine.synthesize(_state(h)) for h in ("short_term", "long_term")}


def test_gate_two_horizons_reach_materially_different_verdicts(reports):
    """
    THE PHASE 6 GATE. Strong fundamentals in a broken downtrend must not
    produce one answer. If this ever passes trivially — both sides HOLD — the
    weighting has stopped doing work.
    """
    short, long_term = reports["short_term"], reports["long_term"]

    assert short.outlook != long_term.outlook, (
        f"both horizons returned {short.outlook.value} — the category weights "
        f"are not changing the verdict"
    )
    # The downtrend should weigh on the short view, not on the long one.
    assert short.financial.financial_health_score != long_term.financial.financial_health_score


def test_each_horizon_reports_its_own_holding_period(reports):
    assert reports["short_term"].target_holding_period == "1 week to 3 months"
    assert reports["long_term"].target_holding_period == "1 to 5 years"
    assert reports["short_term"].recommendation.horizon == "short_term"


def test_every_recommendation_states_what_would_refute_it(reports):
    """
    The invalidation condition is what makes Phase 8 scoring possible at all.
    It must reference a real level, not a platitude.
    """
    for horizon, report in reports.items():
        condition = report.recommendation.invalidation_condition
        assert condition, f"{horizon} produced no invalidation condition"
        assert any(char.isdigit() for char in condition), (
            f"{horizon} invalidation condition cites no level: {condition!r}"
        )
        assert report.recommendation.review_by_date


def test_review_dates_differ_by_horizon(reports):
    assert (
        reports["short_term"].recommendation.review_by_date
        < reports["long_term"].recommendation.review_by_date
    )


def test_report_filenames_do_not_collide_across_horizons(tmp_path, reports):
    """Two horizons on one ticker on one day must not overwrite each other."""
    from analysis.report_generator import ReportGenerator

    generator = ReportGenerator(output_dir=str(tmp_path))
    paths = {
        horizon: generator.generate(report, formats=["markdown"])["markdown"]
        for horizon, report in reports.items()
    }

    assert paths["short_term"] != paths["long_term"]
    assert len(list(tmp_path.glob("*.md"))) == 2


# ===================================================================
# Technical engine behaviour
# ===================================================================

def test_technical_engine_abstains_without_a_price_series():
    """
    No price history is a legitimate outcome, not an error — but it must be
    reported as unavailable rather than as a neutral reading, or a missing
    series looks exactly like a sideways trend.
    """
    snapshot = TechnicalAnalysisEngine().analyze([])

    assert snapshot.available is False
    assert snapshot.metrics == []


def test_technical_engine_grades_a_downtrend_as_one():
    snapshot = TechnicalAnalysisEngine().analyze([
        ToolCall(
            tool_name="get_price_history",
            tool_input={"ticker": "TEST"},
            tool_output_structured=BROKEN_TREND,
        )
    ])

    assert snapshot.available is True
    assert snapshot.trend_score < 0.35
    assert "downtrend" in snapshot.trend_label
    assert snapshot.key_levels["sma_200"] == 92.0


def test_band_metrics_penalize_both_extremes():
    """
    RSI 90 is not four times as healthy as RSI 60. A band metric must grade
    "Elevated" above its range and "Weak" below it — the one assessment mode
    the financial engine has no need for.
    """
    engine = TechnicalAnalysisEngine()

    assert engine._assess(90.0, (35.0, 68.0), None, "").startswith("Elevated")
    assert engine._assess(20.0, (35.0, 68.0), None, "").startswith("Weak")
    assert engine._assess(50.0, (35.0, 68.0), None, "").startswith("Strong")


def test_ignores_a_failed_price_history_call():
    """A failed fetch carries no payload; its remains must not be graded."""
    snapshot = TechnicalAnalysisEngine().analyze([
        ToolCall(
            tool_name="get_price_history",
            tool_input={"ticker": "TEST"},
            tool_output_structured=BROKEN_TREND,
            success=False,
        )
    ])

    assert snapshot.available is False
