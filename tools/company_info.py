"""
tools/company_info.py — Company Information Tool
==================================================

Retrieves fundamental company information: sector, industry,
description, headquarters, employee count, website, etc.

This provides the QUALITATIVE context that complements the
quantitative data from the price and metrics tools.
"""

from __future__ import annotations

from typing import Any

import yfinance as yf

from tools.base import BaseTool, ToolParameter


class CompanyInfoTool(BaseTool):
    """Retrieves company profile and fundamental information."""

    @property
    def name(self) -> str:
        return "get_company_info"

    @property
    def description(self) -> str:
        return (
            "Retrieves company profile information including sector, industry, "
            "business description, headquarters, employee count, and website."
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
        """Retrieve the company profile. Mostly identity, not numbers."""
        ticker_symbol = tool_input.get("ticker", "").upper().strip()
        if not ticker_symbol:
            raise ValueError("'ticker' parameter is required.")

        info = yf.Ticker(ticker_symbol).info
        if not info or not info.get("longName"):
            raise ValueError(f"No company information found for '{ticker_symbol}'.")

        # Location is assembled here rather than in render() so that the
        # payload carries the finished value, not formatting instructions.
        location_parts = [
            info.get("city", ""),
            info.get("state", ""),
            info.get("country", ""),
        ]
        location = ", ".join(p for p in location_parts if p and p != "N/A")

        return {
            "ticker": ticker_symbol,
            "company_name": info.get("longName", ticker_symbol),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "headquarters": location or "N/A",
            "employees": info.get("fullTimeEmployees"),
            "website": info.get("website", "N/A"),
            "description": info.get("longBusinessSummary", "No description available."),
        }

    def render(self, payload: dict[str, Any]) -> str:
        """Format a fetch() payload for the LLM."""
        employees = payload.get("employees")
        employees_str = f"{employees:,}" if isinstance(employees, (int, float)) else "N/A"

        # Truncate description to avoid overwhelming the LLM context
        description = payload.get("description", "")
        if len(description) > 500:
            description = description[:497] + "..."

        return (
            f"Company Profile: {payload.get('company_name')} ({payload.get('ticker')})\n"
            f"  Sector: {payload.get('sector')}\n"
            f"  Industry: {payload.get('industry')}\n"
            f"  Headquarters: {payload.get('headquarters')}\n"
            f"  Employees: {employees_str}\n"
            f"  Website: {payload.get('website')}\n"
            f"  Description: {description}"
        )
