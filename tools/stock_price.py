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

    def _execute(self, tool_input: dict[str, Any]) -> str:
        ticker_symbol = tool_input.get("ticker", "").upper().strip()
        if not ticker_symbol:
            return "Error: 'ticker' parameter is required."

        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        # yfinance returns an empty or minimal dict for invalid tickers
        if not info or info.get("regularMarketPrice") is None:
            # Try fast_info as fallback (yfinance 0.2.x+)
            try:
                fast = stock.fast_info
                current_price = getattr(fast, "last_price", None)
                if current_price is None:
                    return f"Error: No price data found for ticker '{ticker_symbol}'. Verify the symbol is correct."
            except Exception:
                return f"Error: No data found for ticker '{ticker_symbol}'. Verify the symbol is correct."

        # Extract key price data with safe defaults
        current_price = info.get("regularMarketPrice") or info.get("currentPrice", "N/A")
        prev_close = info.get("regularMarketPreviousClose", "N/A")
        open_price = info.get("regularMarketOpen") or info.get("open", "N/A")
        day_high = info.get("regularMarketDayHigh") or info.get("dayHigh", "N/A")
        day_low = info.get("regularMarketDayLow") or info.get("dayLow", "N/A")
        volume = info.get("regularMarketVolume") or info.get("volume", "N/A")
        market_cap = info.get("marketCap", "N/A")
        fifty_two_high = info.get("fiftyTwoWeekHigh", "N/A")
        fifty_two_low = info.get("fiftyTwoWeekLow", "N/A")
        currency = info.get("currency", "USD")

        # Calculate daily change
        change_str = "N/A"
        if isinstance(current_price, (int, float)) and isinstance(prev_close, (int, float)):
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            change_str = f"{change:+.2f} ({change_pct:+.2f}%)"

        # Format market cap
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

        # Format volume
        if isinstance(volume, (int, float)):
            volume_str = f"{volume:,.0f}"
        else:
            volume_str = "N/A"

        return (
            f"Stock Price Data for {ticker_symbol}:\n"
            f"  Current Price: {currency} {current_price}\n"
            f"  Daily Change: {change_str}\n"
            f"  Open: {currency} {open_price}\n"
            f"  Day Range: {currency} {day_low} - {currency} {day_high}\n"
            f"  52-Week Range: {currency} {fifty_two_low} - {currency} {fifty_two_high}\n"
            f"  Volume: {volume_str}\n"
            f"  Market Cap: {market_cap_str}"
        )
