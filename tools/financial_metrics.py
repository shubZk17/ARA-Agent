"""
tools/financial_metrics.py — Financial Metrics Tool
=====================================================

Retrieves key financial ratios and valuation metrics:
P/E, P/B, EPS, revenue, margins, ROE, debt-to-equity, etc.

These are the numbers that drive investment decisions.
An analyst without these is like a doctor without vitals.

DESIGN NOTE:
    This tool implements the fetch()/render() contract (tools/base.py):

      fetch()  -> canonical-unit numbers, via tools/units.py
      render() -> "$84.34B" style text, built FROM those numbers

    Formatting is for the LLM only. Analysis reads the payload, so a value
    never has to be parsed back out of a formatted string — that round trip
    was defect D1, which turned $84,343,996,416 of Apple debt into "$84".
"""

from __future__ import annotations

from typing import Any

import yfinance as yf

from tools.base import BaseTool, ToolParameter
from tools.units import canonical_payload


def _format_large_number(value: Any, prefix: str = "$") -> str:
    """Format large numbers with appropriate suffixes."""
    if not isinstance(value, (int, float)):
        return "N/A"
    if abs(value) >= 1e12:
        return f"{prefix}{value / 1e12:.2f}T"
    if abs(value) >= 1e9:
        return f"{prefix}{value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"{prefix}{value / 1e6:.2f}M"
    return f"{prefix}{value:,.2f}"


def _format_ratio(value: Any) -> str:
    """Format a financial ratio."""
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "N/A"


def _format_percentage(value: Any) -> str:
    """Format a decimal fraction (0.276) as a percentage."""
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
    return "N/A"


def _format_percent_value(value: Any) -> str:
    """
    Format a value that is ALREADY a percentage.

    Which yfinance fields are which is declared in tools/units.py — do not
    guess here. Verified live 2026-08-16: AAPL 0.35 and KO 2.42 dividend
    yields are percents; profitMargins and payoutRatio are fractions.
    """
    if isinstance(value, (int, float)):
        return f"{value:.2f}%"
    return "N/A"


class FinancialMetricsTool(BaseTool):
    """Retrieves key financial ratios and valuation metrics."""

    @property
    def name(self) -> str:
        return "get_financial_metrics"

    @property
    def description(self) -> str:
        return (
            "Retrieves key financial metrics including P/E ratio, EPS, "
            "revenue, profit margins, ROE, debt-to-equity, free cash flow, "
            "liquidity ratios, dividend yield, and other fundamental "
            "valuation metrics."
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
        """Retrieve canonical-unit metrics. No formatting happens here."""
        ticker_symbol = tool_input.get("ticker", "").upper().strip()
        if not ticker_symbol:
            raise ValueError("'ticker' parameter is required.")

        info = yf.Ticker(ticker_symbol).info
        if not info or not info.get("longName"):
            raise ValueError(f"No financial data found for '{ticker_symbol}'.")

        payload: dict[str, Any] = canonical_payload(info)
        payload["ticker"] = ticker_symbol
        payload["company_name"] = info.get("longName", ticker_symbol)
        return payload

    def render(self, payload: dict[str, Any]) -> str:
        """Format a fetch() payload for the LLM."""
        g = payload.get

        return (
            f"Financial Metrics for {g('company_name')} ({g('ticker')}):\n"
            f"\n  === Valuation ===\n"
            f"  Trailing P/E: {_format_ratio(g('trailing_pe'))}\n"
            f"  Forward P/E: {_format_ratio(g('forward_pe'))}\n"
            f"  Price/Book: {_format_ratio(g('price_to_book'))}\n"
            f"  Price/Sales: {_format_ratio(g('price_to_sales'))}\n"
            f"  PEG Ratio: {_format_ratio(g('peg_ratio'))}\n"
            f"  EV/EBITDA: {_format_ratio(g('enterprise_to_ebitda'))}\n"
            f"\n  === Profitability ===\n"
            f"  Trailing EPS: {_format_ratio(g('eps_trailing'))}\n"
            f"  Profit Margin: {_format_percent_value(g('profit_margin'))}\n"
            f"  Operating Margin: {_format_percent_value(g('operating_margin'))}\n"
            f"  Gross Margin: {_format_percent_value(g('gross_margin'))}\n"
            f"  Return on Equity: {_format_percent_value(g('roe'))}\n"
            f"  Return on Assets: {_format_percent_value(g('roa'))}\n"
            f"\n  === Growth ===\n"
            f"  Revenue: {_format_large_number(g('total_revenue'))}\n"
            f"  Revenue Growth: {_format_percent_value(g('revenue_growth'))}\n"
            f"  Earnings Growth: {_format_percent_value(g('earnings_growth'))}\n"
            f"\n  === Liquidity ===\n"
            f"  Current Ratio: {_format_ratio(g('current_ratio'))}\n"
            f"  Quick Ratio: {_format_ratio(g('quick_ratio'))}\n"
            f"  Free Cash Flow: {_format_large_number(g('free_cash_flow'))}\n"
            f"\n  === Balance Sheet ===\n"
            f"  Total Debt: {_format_large_number(g('total_debt'))}\n"
            f"  Total Cash: {_format_large_number(g('total_cash'))}\n"
            f"  Debt/Equity: {_format_ratio(g('debt_to_equity'))}\n"
            f"\n  === Dividends & Risk ===\n"
            f"  Dividend Yield: {_format_percent_value(g('dividend_yield'))}\n"
            f"  Payout Ratio: {_format_percent_value(g('payout_ratio'))}\n"
            f"  Beta: {_format_ratio(g('beta'))}"
        )
