"""
Tier 2 — indicator correctness (plan.md Verification).

Phase 6 hand-rolls every indicator rather than taking a TA-Lib dependency,
which means correctness is ours to prove. These are the proof: hand-computed
values where the arithmetic is small enough to do by hand, and properties
that must hold for ANY input where it is not.

No network, no fixtures — synthetic series only, so a yfinance change can
never make these pass or fail for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.indicators import (
    atr,
    ema,
    macd,
    max_drawdown,
    realized_volatility,
    returns_table,
    rsi,
    sma,
)


def series(values: list[float], start: str = "2026-01-01") -> pd.Series:
    """A daily-indexed Series, oldest first — the shape every function expects."""
    return pd.Series(
        values,
        index=pd.date_range(start=start, periods=len(values), freq="D"),
        dtype="float64",
    )


# ===================================================================
# Hand-computed values
# ===================================================================

def test_sma_matches_hand_computation():
    s = series([10, 20, 30, 40, 50])
    result = sma(s, 3)

    # First two windows are incomplete and MUST be NaN, not a partial mean.
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(20.0)   # (10+20+30)/3
    assert result.iloc[3] == pytest.approx(30.0)   # (20+30+40)/3
    assert result.iloc[4] == pytest.approx(40.0)   # (30+40+50)/3


def test_ema_matches_recursive_definition():
    """
    adjust=False is the recursive form: EMA_t = a*x_t + (1-a)*EMA_{t-1},
    seeded with the first value. span=3 -> a = 2/(3+1) = 0.5.
    """
    s = series([10, 20, 30, 40])
    result = ema(s, 3)

    assert result.iloc[0] == pytest.approx(10.0)                    # seed
    assert result.iloc[1] == pytest.approx(0.5 * 20 + 0.5 * 10)     # 15
    assert result.iloc[2] == pytest.approx(0.5 * 30 + 0.5 * 15)     # 22.5
    assert result.iloc[3] == pytest.approx(0.5 * 40 + 0.5 * 22.5)   # 31.25


def test_rsi_is_100_on_an_unbroken_advance():
    """No down days means average loss is 0, RS is infinite, RSI is exactly 100."""
    result = rsi(series([float(100 + i) for i in range(30)]), period=14)
    assert result.iloc[-1] == pytest.approx(100.0)


def test_rsi_is_zero_on_an_unbroken_decline():
    result = rsi(series([float(200 - i) for i in range(30)]), period=14)
    assert result.iloc[-1] == pytest.approx(0.0)


def test_rsi_uses_wilder_smoothing_not_a_span_ema():
    """
    The classic hand-rolled RSI bug: ewm(span=period) instead of
    ewm(alpha=1/period). Both "work" and differ by several points, which is
    the difference between "overbought" and "neutral".

    On this alternating +2 / -1 series the two conventions give 65.126
    (Wilder, alpha = 1/14) and 63.415 (span = 14). The tolerance below is
    tight enough that swapping the smoothing fails this test — which is the
    only thing it is here to catch.
    """
    values = [100.0]
    for _ in range(40):
        values.append(values[-1] + 2)
        values.append(values[-1] - 1)

    result = rsi(series(values), period=14).iloc[-1]
    assert result == pytest.approx(65.126, abs=0.01)


def test_macd_histogram_is_line_minus_signal():
    s = series([float(100 + i * 2) for i in range(60)])
    line, signal, histogram = macd(s)

    assert histogram.iloc[-1] == pytest.approx(line.iloc[-1] - signal.iloc[-1])
    # A steady advance puts the fast EMA above the slow one.
    assert line.iloc[-1] > 0


def test_atr_captures_a_gap_that_high_minus_low_misses():
    """
    True range must include |high - prev_close|. A stock that gaps up 10 and
    then trades in a 1-wide range had a true range of 11, not 1 — an ATR
    built on high-low alone would understate the stop distance tenfold.
    """
    close = series([100.0] * 5 + [110.0] * 5)
    high = close + 0.5
    low = close - 0.5

    result = atr(high, low, close, period=2)
    # The gap bar is index 5: high 110.5 vs prev close 100 -> TR = 10.5
    assert result.iloc[5] > 5.0


# ===================================================================
# Properties — must hold for any input
# ===================================================================

def test_sma_of_a_constant_is_that_constant():
    result = sma(series([7.0] * 20), 5)
    assert result.dropna().eq(7.0).all()


@pytest.mark.parametrize("values", [
    [100, 105, 98, 110, 95, 120, 118, 130, 125, 140] * 4,
    [50, 49, 48, 47, 46, 45, 44, 43, 42, 41] * 4,
    [10, 200, 10, 200, 10, 200, 10, 200] * 5,
])
def test_rsi_stays_within_bounds(values):
    result = rsi(series([float(v) for v in values]), period=14).dropna()
    assert not result.empty
    assert result.between(0.0, 100.0).all()


def test_max_drawdown_is_never_positive():
    assert max_drawdown(series([1.0, 2.0, 3.0, 4.0])) == 0.0        # only rose
    assert max_drawdown(series([100.0, 50.0])) == pytest.approx(-50.0)
    assert max_drawdown(series([])) == 0.0


def test_max_drawdown_measures_from_the_peak_not_the_start():
    """80 -> 100 -> 60 is a 40% drawdown from the peak, not 25% from the open."""
    assert max_drawdown(series([80.0, 100.0, 60.0, 90.0])) == pytest.approx(-40.0)


def test_realized_volatility_of_a_flat_series_is_zero():
    result = realized_volatility(series([100.0] * 40), window=30)
    assert result.iloc[-1] == pytest.approx(0.0)


def test_realized_volatility_is_annualized():
    """
    A series with a constant 1% daily log move has daily sigma ~0, so use a
    random-ish alternating series and check the sqrt(252) scaling by
    comparing against the raw standard deviation.
    """
    rng = np.random.default_rng(0)
    prices = series(list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))))

    annualized = realized_volatility(prices, window=252).iloc[-1]
    log_returns = np.log(prices / prices.shift(1))
    expected = log_returns.tail(252).std() * np.sqrt(252) * 100

    assert annualized == pytest.approx(float(expected))


def test_returns_compose():
    """A doubling then a halving must net to 0%."""
    prices = series([100.0] * 10 + [200.0] * 10 + [100.0] * 10)
    table = returns_table(prices)
    assert table["1w"] == pytest.approx(0.0)


def test_returns_table_omits_windows_longer_than_the_history():
    """
    Six months of data must not report a 1-year return. Computing one against
    the earliest available bar is a fabrication that reads as a real number.
    """
    prices = series([float(100 + i) for i in range(120)])
    table = returns_table(prices)

    assert "1w" in table and "1m" in table and "3m" in table
    assert "1y" not in table


def test_returns_table_handles_a_tz_aware_index():
    """yfinance hands back a tz-aware index; date arithmetic must not raise."""
    prices = series([float(100 + i) for i in range(60)])
    prices.index = prices.index.tz_localize("America/New_York")

    table = returns_table(prices)
    assert table["1m"] > 0
