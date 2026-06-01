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

    def _execute(self, tool_input: dict[str, Any]) -> str:
        ticker_symbol = tool_input.get("ticker", "").upper().strip()
        if not ticker_symbol:
            return "Error: 'ticker' parameter is required."

        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        if not info or not info.get("longName"):
            return f"Error: No company information found for '{ticker_symbol}'."

        company_name = info.get("longName", ticker_symbol)
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        description = info.get("longBusinessSummary", "No description available.")
        country = info.get("country", "N/A")
        city = info.get("city", "N/A")
        state = info.get("state", "")
        website = info.get("website", "N/A")
        employees = info.get("fullTimeEmployees", "N/A")
        ceo = info.get("companyOfficers", [{}])

        # Format employee count
        if isinstance(employees, (int, float)):
            employees_str = f"{employees:,}"
        else:
            employees_str = "N/A"

        # Format location
        location_parts = [city]
        if state:
            location_parts.append(state)
        location_parts.append(country)
        location = ", ".join(p for p in location_parts if p and p != "N/A")

        # Truncate description to avoid overwhelming the LLM context
        if len(description) > 500:
            description = description[:497] + "..."

        return (
            f"Company Profile: {company_name} ({ticker_symbol})\n"
            f"  Sector: {sector}\n"
            f"  Industry: {industry}\n"
            f"  Headquarters: {location}\n"
            f"  Employees: {employees_str}\n"
            f"  Website: {website}\n"
            f"  Description: {description}"
        )
