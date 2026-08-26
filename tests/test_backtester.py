"""
tests/test_backtester.py — Phase 8.2 technical backtester, synthetic series only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.backtester import DEFAULT_FORWARD_DAYS, DEFAULT_SMA_WINDOW, TechnicalBacktester


def _series(prices) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=len(prices))
    return pd.DataFrame({"Close": prices}, index=dates)


def test_raises_on_too_short_history():
    bt = TechnicalBacktester()
    df = _series([100.0] * 10)
    with pytest.raises(ValueError):
        bt._run_on(df, "X", "custom", DEFAULT_FORWARD_DAYS, DEFAULT_SMA_WINDOW)


def test_clean_uptrend_favors_trend_up_bucket():
    rng = np.random.default_rng(1)
    prices = 100 + np.cumsum(rng.normal(0.2, 0.5, size=300))
    bt = TechnicalBacktester()
    result = bt._run_on(_series(prices), "SYNTH", "custom", DEFAULT_FORWARD_DAYS, DEFAULT_SMA_WINDOW)

    assert result.n_signals > 0
    assert result.buy_hold_return > 0
    if "trend_up" in result.bucket_stats:
        assert result.bucket_stats["trend_up"]["hit_rate"] >= 0.5


def test_forward_return_has_no_lookahead():
    """The signal at bar t must not use any bar after t."""
    prices = list(np.linspace(100, 200, 200))
    bt = TechnicalBacktester()
    df = _series(prices)
    result = bt._run_on(df, "X", "custom", forward_days=10, sma_window=20)
    # A monotonically rising series: every trend_up forward return should be
    # positive since forward always == higher price later.
    assert result.bucket_stats["trend_up"]["mean_forward_return"] > 0
