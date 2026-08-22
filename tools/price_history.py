"""
tools/price_history.py — OHLCV price series and its technical summary
=======================================================================

WHY THIS EXISTS:
    Before Phase 6, `Ticker.history()` appeared nowhere in the codebase. The
    agent could tell you what a company IS, never what its price has been
    DOING — which is most of a short-horizon thesis and all of an entry or
    invalidation level.

WHAT fetch() RETURNS — a SUMMARY, not 500 rows:
    The LLM cannot reason over a table of daily bars and the payload has to
    survive into episodic memory as JSON. So the series is reduced here, at
    the trust boundary, to ~25 scalars: trailing returns, 52-week position,
    realized volatility, drawdown, moving averages, RSI, MACD and ATR.

THREE THINGS THAT SILENTLY CORRUPT EVERY RETURN IF MISSED:
    1. auto_adjust=True, passed EXPLICITLY. yfinance's default has flipped
       between versions, and an unadjusted series shows a fake -50% crash on
       every split date. AAPL would read as having halved in 2020.
    2. Empty frames. Delisted, halted and mistyped tickers return a 0-row
       DataFrame rather than raising, and every downstream statistic on it
       is a NaN wearing a number's clothes.
    3. The index is timezone-aware. Comparing it to a naive Timestamp raises,
       and mixing the two shifts date boundaries by up to a day.

WHY A FIXED 2-YEAR WINDOW rather than one sized from the horizon:
    The 200-day SMA needs ~200 bars whatever the holding period, and one
    request costs the same at 6 months as at 2 years. Sizing the fetch from
    the horizon would also mean trusting the LLM to pass the right period —
    a decision it has no reason to get right. The horizon selects which of
    these numbers MATTER (analysis/technical_engine.py), not which exist.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from analytics.indicators import (
    atr,
    macd,
    max_drawdown,
    realized_volatility,
    returns_table,
    rsi,
    sma,
)
from tools.base import BaseTool, ToolParameter

# Long enough for a 200-day SMA plus a full year of trailing returns.
HISTORY_PERIOD = "2y"

MIN_BARS = 30  # below this, nothing here is meaningful


def _last(series: pd.Series) -> float | None:
    """Last non-NaN value of a series as a float, or None if there is none."""
    if series is None or len(series) == 0:
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


class PriceHistoryTool(BaseTool):
    """Fetches daily OHLCV history and reduces it to a technical summary."""

    @property
    def name(self) -> str:
        return "get_price_history"

    @property
    def description(self) -> str:
        return (
            "Retrieves 2 years of daily price history and returns a technical "
            "summary: trailing returns (1w/1m/3m/6m/1y/YTD), 52-week range and "
            "distance from it, annualized realized volatility, maximum drawdown, "
            "20/50/200-day moving averages, RSI, MACD, ATR and relative volume. "
            "Use this for trend, momentum and entry/exit levels."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="ticker",
                type="string",
                description="Stock ticker symbol (e.g., 'TSLA', 'AAPL', 'MSFT')",
                required=True,
            )
        ]

    def fetch(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Download the series and reduce it. Every value is canonical-unit."""
        ticker_symbol = tool_input.get("ticker", "").upper().strip()
        if not ticker_symbol:
            raise ValueError("'ticker' parameter is required.")

        frame = yf.Ticker(ticker_symbol).history(
            period=HISTORY_PERIOD,
            interval="1d",
            auto_adjust=True,   # explicit: see module docstring, hazard 1
        )

        if frame is None or frame.empty:
            raise ValueError(
                f"No price history for '{ticker_symbol}' — the ticker may be "
                f"delisted, halted, or misspelled."
            )
        if len(frame) < MIN_BARS:
            raise ValueError(
                f"Only {len(frame)} price bars available for '{ticker_symbol}'; "
                f"at least {MIN_BARS} are needed for a technical read."
            )

        # Hazard 3: normalize the tz-aware index once, here, so no downstream
        # date arithmetic has to think about it.
        if getattr(frame.index, "tz", None) is not None:
            frame.index = frame.index.tz_localize(None)

        return self._summarize(ticker_symbol, frame)

    def _summarize(self, ticker: str, frame: pd.DataFrame) -> dict[str, Any]:
        """Reduce an OHLCV frame to scalars. No formatting, no rounding."""
        close = frame["Close"]
        high, low, volume = frame["High"], frame["Low"], frame["Volume"]
        last_close = float(close.iloc[-1])

        payload: dict[str, Any] = {
            "ticker": ticker,
            "as_of": frame.index[-1].strftime("%Y-%m-%d"),
            "bars": int(len(frame)),
            "last_close": last_close,
        }

        # --- Trailing returns (percent) ---
        for label, value in returns_table(close).items():
            payload[f"return_{label}"] = value

        # --- 52-week position ---
        window = close.tail(252)
        high_52w, low_52w = float(window.max()), float(window.min())
        payload["high_52w"] = high_52w
        payload["low_52w"] = low_52w
        if high_52w:
            payload["pct_from_52w_high"] = (last_close / high_52w - 1.0) * 100.0
        if low_52w:
            payload["pct_from_52w_low"] = (last_close / low_52w - 1.0) * 100.0

        # --- Risk ---
        vol = _last(realized_volatility(close, window=30))
        if vol is not None:
            payload["realized_vol_30d"] = vol
        payload["max_drawdown_1y"] = max_drawdown(close.tail(252))

        # --- Trend ---
        for period in (20, 50, 200):
            value = _last(sma(close, period))
            if value is not None:
                payload[f"sma_{period}"] = value
                payload[f"price_vs_sma_{period}"] = (last_close / value - 1.0) * 100.0

        # --- Momentum ---
        rsi_value = _last(rsi(close, 14))
        if rsi_value is not None:
            payload["rsi_14"] = rsi_value

        macd_line, signal_line, histogram = macd(close)
        macd_hist = _last(histogram)
        if macd_hist is not None and last_close:
            payload["macd"] = _last(macd_line)
            payload["macd_signal"] = _last(signal_line)
            # Normalized by price so the threshold means the same thing on a
            # $12 stock and a $900 one. Raw MACD is in price units and is not
            # comparable across tickers.
            payload["macd_hist_pct"] = (macd_hist / last_close) * 100.0

        # --- Range / participation ---
        atr_value = _last(atr(high, low, close, 14))
        if atr_value is not None and last_close:
            payload["atr_14"] = atr_value
            payload["atr_pct"] = (atr_value / last_close) * 100.0

        avg_volume = _last(volume.rolling(30, min_periods=30).mean())
        if avg_volume:
            payload["avg_volume_30d"] = avg_volume
            payload["relative_volume"] = float(volume.iloc[-1]) / avg_volume

        return payload

    def render(self, payload: dict[str, Any]) -> str:
        """Format a fetch() payload for the LLM."""
        g = payload.get

        def num(key: str, suffix: str = "", places: int = 2) -> str:
            value = g(key)
            if not isinstance(value, (int, float)):
                return "N/A"
            return f"{value:,.{places}f}{suffix}"

        returns = " | ".join(
            f"{label.upper()}: {num(f'return_{label}', '%')}"
            for label in ("1w", "1m", "3m", "6m", "1y", "ytd")
            if f"return_{label}" in payload
        )

        return (
            f"Price History for {g('ticker')} (as of {g('as_of')}, "
            f"{g('bars')} daily bars, split/dividend adjusted):\n"
            f"  Last Close: {num('last_close')}\n"
            f"\n  === Trailing Returns ===\n"
            f"  {returns or 'N/A'}\n"
            f"\n  === Trend ===\n"
            f"  20-day SMA: {num('sma_20')} ({num('price_vs_sma_20', '%')} vs price)\n"
            f"  50-day SMA: {num('sma_50')} ({num('price_vs_sma_50', '%')} vs price)\n"
            f"  200-day SMA: {num('sma_200')} ({num('price_vs_sma_200', '%')} vs price)\n"
            f"\n  === Momentum ===\n"
            f"  RSI (14): {num('rsi_14', '', 1)}\n"
            f"  MACD Histogram: {num('macd_hist_pct', '% of price')}\n"
            f"\n  === Range & Risk ===\n"
            f"  52-Week High: {num('high_52w')} ({num('pct_from_52w_high', '%')} away)\n"
            f"  52-Week Low: {num('low_52w')} ({num('pct_from_52w_low', '%')} away)\n"
            f"  Realized Volatility (30d, annualized): {num('realized_vol_30d', '%', 1)}\n"
            f"  Max Drawdown (1y): {num('max_drawdown_1y', '%', 1)}\n"
            f"  ATR (14): {num('atr_14')} ({num('atr_pct', '% of price')})\n"
            f"  Relative Volume: {num('relative_volume', 'x')}"
        )
