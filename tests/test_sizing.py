"""
tests/test_sizing.py — Phase 8.3 volatility-scaled position sizing.
"""

from __future__ import annotations

from portfolio.sizing import MAX_POSITION_PCT, MIN_GRADED_SAMPLE, position_size


def test_zero_or_missing_volatility_refuses_to_size():
    assert position_size(0, 0.8).position_pct == 0.0
    assert position_size(None, 0.8).position_pct == 0.0


def test_confidence_below_risk_floor_sizes_zero():
    result = position_size(15.0, 0.1, risk_profile="conservative")
    assert result.position_pct == 0.0
    assert "floor" in result.rationale


def test_unvalidated_signal_is_floored_relative_to_validated():
    unvalidated = position_size(15.0, 0.8, risk_profile="balanced")
    validated = position_size(
        15.0, 0.8, risk_profile="balanced", hit_rate=0.75, graded_sample_size=MIN_GRADED_SAMPLE
    )
    assert 0 < unvalidated.position_pct < validated.position_pct


def test_measured_coinflip_or_worse_throttles_below_unvalidated():
    coinflip = position_size(15.0, 0.8, hit_rate=0.5, graded_sample_size=MIN_GRADED_SAMPLE)
    losing = position_size(15.0, 0.8, hit_rate=0.3, graded_sample_size=MIN_GRADED_SAMPLE)
    assert losing.position_pct <= coinflip.position_pct


def test_position_never_exceeds_cap():
    result = position_size(1.0, 0.95, risk_profile="aggressive", hit_rate=0.95, graded_sample_size=100)
    assert result.position_pct == MAX_POSITION_PCT


def test_correlation_penalty_reduces_size():
    base = position_size(15.0, 0.8, hit_rate=0.75, graded_sample_size=MIN_GRADED_SAMPLE)
    correlated = position_size(
        15.0, 0.8, hit_rate=0.75, graded_sample_size=MIN_GRADED_SAMPLE, max_correlation_to_existing=0.9
    )
    assert correlated.position_pct < base.position_pct
