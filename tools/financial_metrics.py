"""
tools/financial_metrics.py — Financial Metrics Tool
=====================================================

Retrieves key financial ratios and valuation metrics:
P/E, P/B, EPS, revenue, margins, ROE, debt-to-equity, etc.

These are the numbers that drive investment decisions.
An analyst without these is like a doctor without vitals.

DESIGN NOTE:
    We format numbers with appropriate precision and units
    because raw numbers like 394328000000 are meaningless to
    both humans and LLMs. "$394.33B" is instantly interpretable.
"""

from __future__ import annotations

from typing import Any

import yfinance as yf

from tools.base import BaseTool, ToolParameter


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
    """Format a decimal as percentage."""
    if isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"
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
            "revenue, profit margins, ROE, debt-to-equity, dividend yield, "
            "and other fundamental valuation metrics."
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

        if not info or not info.get("longName"):
            return f"Error: No financial data found for '{ticker_symbol}'."

        company_name = info.get("longName", ticker_symbol)

        # Valuation Metrics
        pe_trailing = _format_ratio(info.get("trailingPE"))
        pe_forward = _format_ratio(info.get("forwardPE"))
        pb_ratio = _format_ratio(info.get("priceToBook"))
        ps_ratio = _format_ratio(info.get("priceToSalesTrailing12Months"))
        peg_ratio = _format_ratio(info.get("pegRatio"))
        ev_ebitda = _format_ratio(info.get("enterpriseToEbitda"))

        # Profitability
        eps_trailing = _format_ratio(info.get("trailingEps"))
        eps_forward = _format_ratio(info.get("forwardEps"))
        profit_margin = _format_percentage(info.get("profitMargins"))
        operating_margin = _format_percentage(info.get("operatingMargins"))
        gross_margin = _format_percentage(info.get("grossMargins"))
        roe = _format_percentage(info.get("returnOnEquity"))
        roa = _format_percentage(info.get("returnOnAssets"))

        # Revenue & Earnings
        revenue = _format_large_number(info.get("totalRevenue"))
        revenue_growth = _format_percentage(info.get("revenueGrowth"))
        earnings_growth = _format_percentage(info.get("earningsGrowth"))

        # Balance Sheet
        total_debt = _format_large_number(info.get("totalDebt"))
        total_cash = _format_large_number(info.get("totalCash"))
        debt_to_equity = _format_ratio(info.get("debtToEquity"))

        # Dividends
        dividend_yield = _format_percentage(info.get("dividendYield"))
        payout_ratio = _format_percentage(info.get("payoutRatio"))

        # Beta
        beta = _format_ratio(info.get("beta"))

        return (
            f"Financial Metrics for {company_name} ({ticker_symbol}):\n"
            f"\n  === Valuation ===\n"
            f"  Trailing P/E: {pe_trailing}\n"
            f"  Forward P/E: {pe_forward}\n"
            f"  Price/Book: {pb_ratio}\n"
            f"  Price/Sales: {ps_ratio}\n"
            f"  PEG Ratio: {peg_ratio}\n"
            f"  EV/EBITDA: {ev_ebitda}\n"
            f"\n  === Profitability ===\n"
            f"  Trailing EPS: {eps_trailing}\n"
            f"  Forward EPS: {eps_forward}\n"
            f"  Profit Margin: {profit_margin}\n"
            f"  Operating Margin: {operating_margin}\n"
            f"  Gross Margin: {gross_margin}\n"
            f"  Return on Equity: {roe}\n"
            f"  Return on Assets: {roa}\n"
            f"\n  === Growth ===\n"
            f"  Revenue: {revenue}\n"
            f"  Revenue Growth: {revenue_growth}\n"
            f"  Earnings Growth: {earnings_growth}\n"
            f"\n  === Balance Sheet ===\n"
            f"  Total Debt: {total_debt}\n"
            f"  Total Cash: {total_cash}\n"
            f"  Debt/Equity: {debt_to_equity}\n"
            f"\n  === Dividends & Risk ===\n"
            f"  Dividend Yield: {dividend_yield}\n"
            f"  Payout Ratio: {payout_ratio}\n"
            f"  Beta: {beta}"
        )
