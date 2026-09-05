"""
analysis/indicators.py — technical indicators in plain pandas
===============================================================

WHY NOT TA-Lib:
    TA-Lib is a C extension whose Windows build is a recurring source of
    "works on my machine". Everything below is ~15 lines of pandas each, and
    pandas is already a dependency. The trade-off is that correctness is
    OURS to prove — hence tests/test_indicators.py, which pins each function
    against hand-computed values and against properties that must always
    hold (SMA of a constant is that constant; RSI stays in [0, 100];
    drawdown is never positive).

CONVENTIONS — every function here obeys these, so callers never have to ask:
    - Input is a pandas Series indexed by date, oldest first.
    - Output is a Series of the same length, NaN-padded at the front where
      the window has not filled yet. Never silently shortened.
    - Percentages are returned as percents (5.0 means 5%), matching the
      canonical units in tools/units.py.
    - Nothing here reaches for the network, a config, or a logger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Trading days in a year — the standard annualization factor for daily bars.
TRADING_DAYS_PER_YEAR = 252

# Calendar-day offsets for the standard return windows. Calendar days, not
# bars, because "3-month return" means three months of wall clock; using bar
# counts silently stretches every window across holidays.
RETURN_WINDOWS: dict[str, int] = {
    "1w": 7,
    "1m": 30,
    "3m": 91,
    "6m": 182,
    "1y": 365,
}


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over `window` bars."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average. `adjust=False` gives the recursive form
    every charting package uses, so values match what a user sees."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's Relative Strength Index, 0–100.

    Wilder smoothing is an EMA with alpha = 1/period (NOT span = period);
    using span here is the single most common way a hand-rolled RSI comes
    out subtly wrong.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means an unbroken run of gains: RS is infinite, RSI is 100.
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range — the average size of a daily move, in price units.

    True range is the widest of (high−low), (high−prev close), (prev close−low),
    so it captures gaps that high−low alone misses. Used here to express an
    invalidation level as a distance the stock actually travels, rather than a
    round number picked out of the air.
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def realized_volatility(series: pd.Series, window: int = 30) -> pd.Series:
    """
    Annualized realized volatility as a PERCENT (28.0 means 28%/yr).

    Standard deviation of daily log returns, scaled by sqrt(252). Log returns
    rather than simple ones so the measure is symmetric: a +10% day and the
    -9.09% day that undoes it have equal magnitude.
    """
    log_returns = np.log(series / series.shift(1))
    return log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    ) * 100.0


def max_drawdown(series: pd.Series) -> float:
    """
    Worst peak-to-trough decline over the whole series, as a NEGATIVE percent
    (-32.4 means the series fell 32.4% below a prior peak). Returns 0.0 for a
    series that only ever rose.
    """
    clean = series.dropna()
    if clean.empty:
        return 0.0
    running_peak = clean.cummax()
    drawdowns = (clean / running_peak - 1.0) * 100.0
    return float(min(drawdowns.min(), 0.0))


def returns_table(series: pd.Series) -> dict[str, float]:
    """
    Trailing returns as percents, keyed by window ("1w", "1m", …) plus "ytd".

    Each window looks back by CALENDAR days from the last observation and
    takes the first bar at or after that date, so weekends and holidays do
    not shift the window. Windows longer than the available history are
    omitted rather than computed against the earliest bar — a "1y return"
    from six months of data is a fabrication.
    """
    clean = series.dropna()
    if len(clean) < 2:
        return {}

    # Date arithmetic against a tz-aware index raises on comparison with the
    # naive Timestamp built for YTD below. Drop the tz rather than trust the
    # caller to have done it.
    if getattr(clean.index, "tz", None) is not None:
        clean = clean.tz_localize(None)

    last_date = clean.index[-1]
    last_price = float(clean.iloc[-1])
    first_date = clean.index[0]
    out: dict[str, float] = {}

    for label, days in RETURN_WINDOWS.items():
        target = last_date - pd.Timedelta(days=days)
        if target < first_date:
            continue  # not enough history for this window
        prior = clean[clean.index <= target]
        if prior.empty:
            continue
        start = float(prior.iloc[-1])
        if start:
            out[label] = (last_price / start - 1.0) * 100.0

    # Year to date: the last close of the previous calendar year.
    ytd_start = clean[clean.index < pd.Timestamp(year=last_date.year, month=1, day=1)]
    if not ytd_start.empty:
        start = float(ytd_start.iloc[-1])
        if start:
            out["ytd"] = (last_price / start - 1.0) * 100.0

    return out
