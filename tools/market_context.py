"""
tools/market_context.py — benchmark-relative performance and measured beta
===========================================================================

WHY THIS EXISTS — the strategic point of Phase 6:
    Every prior ARA tool reads `Ticker.info` for ONE ticker. Four tools, one
    endpoint, one opinion. The reliability tiers and conflict-detection
    machinery built in Phase 2 were scoring a set with a single member, which
    is why `conflict_reports` had never been written by anything.

    This tool produces the first figure that can genuinely DISAGREE with the
    primary source: beta computed from two years of actual return series,
    next to the beta yfinance reports. They routinely differ — different
    window, different benchmark, different update cadence — and when they
    differ materially the disagreement is emitted as a conflict rather than
    silently resolved in favour of whichever number was fetched last.

WHAT IT ANSWERS:
    "Is this stock outperforming because the business is doing well, or
    because everything went up?" A 20% gain is a different fact when the
    S&P did 22%.

NOTE ON SOURCE DIVERSITY:
    This is still yfinance underneath, and it is deliberately declared as the
    yfinance family in analysis/confidence_calibrator.py. A second DERIVATION
    is not a second SOURCE, and inflating the diversity multiplier here would
    undo exactly the honesty the D4 fix bought. Phase 7's EDGAR is the first
    genuinely independent provider.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from tools.base import BaseTool, ToolParameter

BENCHMARKS = {"SPY": "S&P 500", "QQQ": "Nasdaq 100"}

# yfinance sector strings -> SPDR sector ETF. Sectors absent here simply get
# no sector comparison; the benchmarks above still apply.
SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

PERIOD = "2y"

# How far the computed beta may sit from the reported one before it is worth
# flagging. 0.30 is roughly the spread you get purely from choosing a 2-year
# rather than a 5-year window — below that it is methodology, not conflict.
BETA_CONFLICT_THRESHOLD = 0.30

RETURN_WINDOWS = {"1m": 21, "3m": 63, "1y": 252}


def _period_return(series: pd.Series, bars: int) -> float | None:
    """Percent return over the last `bars` observations, or None if too short."""
    clean = series.dropna()
    if len(clean) <= bars:
        return None
    start = float(clean.iloc[-(bars + 1)])
    if not start:
        return None
    return (float(clean.iloc[-1]) / start - 1.0) * 100.0


class MarketContextTool(BaseTool):
    """Compares a stock to its benchmarks and measures its beta from returns."""

    @property
    def name(self) -> str:
        return "get_market_context"

    @property
    def description(self) -> str:
        return (
            "Compares a stock against the S&P 500, the Nasdaq 100 and its own "
            "sector ETF: relative strength over 1 month, 3 months and 1 year, "
            "plus beta and correlation measured from two years of actual daily "
            "returns. Use this to tell stock-specific movement apart from a "
            "market-wide move."
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
        ticker_symbol = tool_input.get("ticker", "").upper().strip()
        if not ticker_symbol:
            raise ValueError("'ticker' parameter is required.")

        info = yf.Ticker(ticker_symbol).info or {}
        sector = info.get("sector") or ""
        sector_etf = SECTOR_ETFS.get(sector, "")

        symbols = [ticker_symbol] + list(BENCHMARKS)
        if sector_etf and sector_etf not in symbols:
            symbols.append(sector_etf)

        closes = self._download_closes(symbols)
        if ticker_symbol not in closes or closes[ticker_symbol].dropna().empty:
            raise ValueError(
                f"No price history for '{ticker_symbol}' — cannot compute "
                f"market context."
            )

        subject = closes[ticker_symbol]
        payload: dict[str, Any] = {
            "ticker": ticker_symbol,
            "sector": sector,
            "sector_etf": sector_etf,
            "benchmarks": list(BENCHMARKS),
        }

        # --- Relative strength: subject return minus benchmark return ---
        for label, bars in RETURN_WINDOWS.items():
            own = _period_return(subject, bars)
            if own is None:
                continue
            payload[f"return_{label}"] = own
            for bench in [*BENCHMARKS, sector_etf]:
                if not bench or bench not in closes:
                    continue
                bench_return = _period_return(closes[bench], bars)
                if bench_return is None:
                    continue
                payload[f"{bench.lower()}_return_{label}"] = bench_return
                payload[f"relative_strength_{bench.lower()}_{label}"] = (
                    own - bench_return
                )

        # --- Beta and correlation, measured rather than looked up ---
        if "SPY" in closes:
            measured = self._beta(subject, closes["SPY"])
            if measured:
                beta_value, correlation = measured
                payload["beta_computed"] = beta_value
                payload["correlation_spy"] = correlation

                reported = info.get("beta")
                if isinstance(reported, (int, float)):
                    payload["beta_reported"] = float(reported)
                    divergence = abs(beta_value - float(reported))
                    payload["beta_divergence"] = divergence
                    if divergence > BETA_CONFLICT_THRESHOLD:
                        # Surfaced, not resolved. tool_node lifts anything under
                        # "conflicts" into AgentState.conflict_reports, which is
                        # what the confidence calibrator reads.
                        payload["conflicts"] = [{
                            "field": "beta",
                            "source_a": "yfinance info.beta",
                            "value_a": float(reported),
                            "source_b": f"computed from {PERIOD} of daily returns vs SPY",
                            "value_b": beta_value,
                            "divergence": divergence,
                            "resolution": (
                                f"Beta disagreement on {ticker_symbol}: yfinance "
                                f"reports {reported:.2f}, {PERIOD} of daily returns "
                                f"against SPY give {beta_value:.2f} "
                                f"(difference {divergence:.2f}). Neither is taken as "
                                f"truth; volatility-sensitive conclusions should be "
                                f"treated as uncertain."
                            ),
                        }]

        return payload

    def _download_closes(self, symbols: list[str]) -> dict[str, pd.Series]:
        """One batched request for every symbol. Missing ones are just absent."""
        data = yf.download(
            tickers=symbols,
            period=PERIOD,
            interval="1d",
            auto_adjust=True,   # explicit — unadjusted series fake a crash on splits
            progress=False,
            group_by="column",
        )
        if data is None or data.empty:
            raise ValueError(f"No market data returned for {', '.join(symbols)}.")

        close = data["Close"]
        if isinstance(close, pd.Series):  # single symbol collapses the frame
            close = close.to_frame(symbols[0])
        if getattr(close.index, "tz", None) is not None:
            close.index = close.index.tz_localize(None)

        return {
            symbol: close[symbol]
            for symbol in symbols
            if symbol in close.columns and not close[symbol].dropna().empty
        }

    def _beta(
        self, subject: pd.Series, benchmark: pd.Series
    ) -> tuple[float, float] | None:
        """
        Ordinary least-squares beta of subject vs benchmark daily returns,
        plus their correlation. Returns None if the overlap is too short.

        Aligned on the intersection of dates, deliberately: an outer join
        would pad one side with NaN across the other's holidays and quietly
        bias the covariance.
        """
        returns = pd.concat(
            [subject.pct_change(), benchmark.pct_change()], axis=1, join="inner"
        ).dropna()
        if len(returns) < 60:
            return None

        own, bench = returns.iloc[:, 0], returns.iloc[:, 1]
        variance = float(bench.var())
        if not variance:
            return None

        beta_value = float(np.cov(own, bench)[0][1] / variance)
        return round(beta_value, 4), round(float(own.corr(bench)), 4)

    def render(self, payload: dict[str, Any]) -> str:
        g = payload.get

        def pct(key: str) -> str:
            value = g(key)
            return f"{value:+.2f}%" if isinstance(value, (int, float)) else "N/A"

        lines = [
            f"Market Context for {g('ticker')} "
            f"(sector: {g('sector') or 'unknown'}"
            + (f", ETF: {g('sector_etf')}" if g("sector_etf") else "")
            + "):",
            "",
            "  === Relative Strength (stock return minus benchmark) ===",
        ]

        benchmarks = [b for b in [*BENCHMARKS, g("sector_etf")] if b]
        for label in RETURN_WINDOWS:
            if f"return_{label}" not in payload:
                continue
            parts = [f"{label.upper()}: stock {pct(f'return_{label}')}"]
            for bench in benchmarks:
                key = f"relative_strength_{bench.lower()}_{label}"
                if key in payload:
                    parts.append(f"vs {bench} {pct(key)}")
            lines.append("  " + " | ".join(parts))

        lines.append("")
        lines.append("  === Measured Risk ===")
        beta_computed = g("beta_computed")
        if isinstance(beta_computed, (int, float)):
            lines.append(
                f"  Beta (computed, {PERIOD} daily vs SPY): {beta_computed:.2f}"
            )
        reported = g("beta_reported")
        if isinstance(reported, (int, float)):
            lines.append(f"  Beta (reported by yfinance): {reported:.2f}")
        correlation = g("correlation_spy")
        if isinstance(correlation, (int, float)):
            lines.append(f"  Correlation with SPY: {correlation:.2f}")

        for conflict in payload.get("conflicts", []):
            lines.append(f"\n  [!] CONFLICT — {conflict['resolution']}")

        return "\n".join(lines)
