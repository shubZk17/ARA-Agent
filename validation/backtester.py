"""
validation/backtester.py — Phase 8.2: technical-only rolling backtest
==========================================================================

WHY TECHNICAL-ONLY, NOT FUNDAMENTALS:
    `Ticker.info` is a snapshot with no history — NVDA's trailing P/E as of
    2024-03-01 is unrecoverable from it. `Ticker.history()`, by contrast, IS
    point-in-time by construction: the close on any past date is exactly
    what it was that day. That makes the technical engine backtestable today
    with zero new dependencies, while fundamentals are not.

WHY NO LOOKAHEAD BIAS HERE SPECIFICALLY:
    The signal at bar t (price vs. its trailing SMA) is computed only from
    bars <= t; the forward return is `close[t+N]/close[t] - 1`, shifted
    FORWARD, never back. There is no fundamentals-vs-filing-date problem to
    solve in this module — that problem is real (plan.md §8.2) but belongs
    to a fundamentals backtester, which needs 7.1's EDGAR `filed` dates and
    is NOT built here. Skipped deliberately: building it now, without a real
    need driving the design, would be exactly the speculative feature this
    project's own conventions warn against. Add it when a concrete fundamentals
    signal (e.g. a valuation-vs-growth score) needs regression-testing.

WHAT THIS ANSWERS, AND WHAT IT DOESN'T:
    A rolling backtest over a handful of tickers and a few years is
    REGRESSION DETECTION — "did a change to the trend signal make it worse?"
    — not evidence of a tradeable edge. Treat n_signals and the buy-and-hold
    comparison as the sanity checks they are, not a Sharpe ratio to publish.

NO API KEYS: yfinance is keyless. No LLM involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import yfinance as yf

from analytics.indicators import sma
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_FORWARD_DAYS = 20   # ~1 trading month
DEFAULT_SMA_WINDOW = 50
MIN_BARS = DEFAULT_SMA_WINDOW + DEFAULT_FORWARD_DAYS + 10


@dataclass(frozen=True)
class BacktestResult:
    ticker: str
    period: str
    forward_days: int
    n_signals: int
    bucket_stats: dict = field(default_factory=dict)   # "trend_up"/"trend_down" -> stats
    buy_hold_return: Optional[float] = None


class TechnicalBacktester:
    """Rolling trend-vs-forward-return backtest over daily bars."""

    def run(
        self,
        ticker: str,
        period: str = "5y",
        forward_days: int = DEFAULT_FORWARD_DAYS,
        sma_window: int = DEFAULT_SMA_WINDOW,
    ) -> BacktestResult:
        history = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        return self._run_on(history, ticker, period, forward_days, sma_window)

    def _run_on(
        self,
        history: pd.DataFrame,
        ticker: str,
        period: str,
        forward_days: int,
        sma_window: int,
    ) -> BacktestResult:
        if history is None or history.empty or len(history) < sma_window + forward_days + 1:
            raise ValueError(
                f"Not enough bars for {ticker} ({0 if history is None else len(history)}) "
                f"— need at least {sma_window + forward_days + 1}"
            )

        close = history["Close"]
        trend = sma(close, sma_window)
        forward_return = close.shift(-forward_days) / close - 1.0

        signal = pd.Series(
            ["trend_up" if c > t else "trend_down" for c, t in zip(close, trend)],
            index=close.index,
        )

        frame = pd.DataFrame({"signal": signal, "forward_return": forward_return}).dropna()

        buckets = {}
        for name, group in frame.groupby("signal"):
            buckets[name] = {
                "n": int(len(group)),
                "hit_rate": round(float((group["forward_return"] > 0).mean()), 4),
                "mean_forward_return": round(float(group["forward_return"].mean()), 4),
            }

        buy_hold_return = round(float(close.iloc[-1] / close.iloc[0] - 1.0), 4)

        return BacktestResult(
            ticker=ticker,
            period=period,
            forward_days=forward_days,
            n_signals=int(len(frame)),
            bucket_stats=buckets,
            buy_hold_return=buy_hold_return,
        )


def demo() -> None:
    """Self-check on a synthetic series, no network."""
    import numpy as np

    dates = pd.bdate_range("2020-01-01", periods=300)
    # Clean uptrend with noise — should show trend_up beating trend_down.
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0.15, 1.0, size=len(dates)))
    df = pd.DataFrame({"Close": prices}, index=dates)

    bt = TechnicalBacktester()
    result = bt._run_on(df, "SYNTH", "custom", DEFAULT_FORWARD_DAYS, DEFAULT_SMA_WINDOW)

    assert result.n_signals > 0
    assert "trend_up" in result.bucket_stats or "trend_down" in result.bucket_stats
    assert result.buy_hold_return is not None

    try:
        bt._run_on(df.iloc[:10], "TOO_SHORT", "custom", DEFAULT_FORWARD_DAYS, DEFAULT_SMA_WINDOW)
        raise AssertionError("expected ValueError on too-short history")
    except ValueError:
        pass

    print("validation.backtester self-check OK")


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Technical-only rolling backtest (Phase 8.2).")
    parser.add_argument("ticker")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--forward-days", type=int, default=DEFAULT_FORWARD_DAYS)
    parser.add_argument("--sma-window", type=int, default=DEFAULT_SMA_WINDOW)
    args = parser.parse_args()

    result = TechnicalBacktester().run(args.ticker, args.period, args.forward_days, args.sma_window)
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
