"""
tools/stock_price.py — Stock Price Retrieval Tool
===================================================

Uses yfinance to fetch real-time stock price data.
This is the most fundamental financial tool — almost every
financial analysis starts with "what's the current price?"

WHY yfinance:
    - Free (no API key required),
    - Reliable (backed by Yahoo Finance),
    - Rich data (price, volume, 52-week range, etc.),
    - Good for prototyping (upgrade to Bloomberg/Refinitiv in production).

EDGE CASES HANDLED:
    - Invalid ticker symbols,
    - Market closed / no data available,
    - Network errors (caught by BaseTool.execute).
"""

from __future__ import annotations

from typing import Any

import yfinance as yf

from tools.base import BaseTool, ToolParameter


class StockPriceTool(BaseTool):
    """Retrieves current and recent stock price data for a given ticker."""

    @property
    def name(self) -> str:
        return "get_stock_price"

    @property
    def description(self) -> str:
        return (
            "Retrieves current stock price, daily change, volume, "
            "52-week high/low, and recent price history for a stock ticker symbol."
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
        """Retrieve raw price data. Every number stays unscaled."""
        ticker_symbol = tool_input.get("ticker", "").upper().strip()
        if not ticker_symbol:
            raise ValueError("'ticker' parameter is required.")

        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        # yfinance returns an empty or minimal dict for invalid tickers
        current_price = info.get("regularMarketPrice") or info.get("currentPrice")
        if current_price is None:
            # Try fast_info as fallback (yfinance 0.2.x+)
            try:
                current_price = getattr(stock.fast_info, "last_price", None)
            except Exception:
                current_price = None
            if current_price is None:
                raise ValueError(
                    f"No price data found for ticker '{ticker_symbol}'. "
                    f"Verify the symbol is correct."
                )

        prev_close = info.get("regularMarketPreviousClose")

        payload: dict[str, Any] = {
            "ticker": ticker_symbol,
            "currency": info.get("currency", "USD"),
            "current_price": current_price,
            "previous_close": prev_close,
            "open": info.get("regularMarketOpen") or info.get("open"),
            "day_high": info.get("regularMarketDayHigh") or info.get("dayHigh"),
            "day_low": info.get("regularMarketDayLow") or info.get("dayLow"),
            "volume": info.get("regularMarketVolume") or info.get("volume"),
            "market_cap": info.get("marketCap"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        }

        if isinstance(current_price, (int, float)) and isinstance(prev_close, (int, float)) and prev_close:
            payload["daily_change"] = current_price - prev_close
            payload["daily_change_percent"] = ((current_price - prev_close) / prev_close) * 100

        return {k: v for k, v in payload.items() if v is not None}

    def render(self, payload: dict[str, Any]) -> str:
        """Format a fetch() payload for the LLM."""
        g = payload.get
        currency = g("currency", "USD")

        change_str = "N/A"
        if "daily_change" in payload:
            change_str = (
                f"{payload['daily_change']:+.2f} "
                f"({payload['daily_change_percent']:+.2f}%)"
            )

        market_cap = g("market_cap")
        if isinstance(market_cap, (int, float)):
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap / 1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap / 1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap / 1e6:.2f}M"
            else:
                market_cap_str = f"${market_cap:,.0f}"
        else:
            market_cap_str = "N/A"

        volume = g("volume")
        volume_str = f"{volume:,.0f}" if isinstance(volume, (int, float)) else "N/A"

        return (
            f"Stock Price Data for {g('ticker')}:\n"
            f"  Current Price: {currency} {g('current_price', 'N/A')}\n"
            f"  Daily Change: {change_str}\n"
            f"  Open: {currency} {g('open', 'N/A')}\n"
            f"  Day Range: {currency} {g('day_low', 'N/A')} - {currency} {g('day_high', 'N/A')}\n"
            f"  52-Week Range: {currency} {g('fifty_two_week_low', 'N/A')} - "
            f"{currency} {g('fifty_two_week_high', 'N/A')}\n"
            f"  Volume: {volume_str}\n"
            f"  Market Cap: {market_cap_str}"
        )
