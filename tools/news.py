"""
tools/news.py — News Retrieval Tool
=====================================

Retrieves recent news headlines for a given stock ticker or company.

IMPLEMENTATION NOTE:
    For Phase 1, we use yfinance's built-in news feed. It's limited but
    requires no additional API keys — keeping the setup friction minimal.

    Phase 2+ can upgrade to:
    - NewsAPI (newsapi.org) — broader coverage, requires API key
    - Alpha Vantage News — financial-specific
    - Custom RSS aggregation
    - Tavily / Serper for web search

    The tool interface stays the same regardless of the backend.
    That's the power of the BaseTool abstraction.
"""

from __future__ import annotations

from typing import Any

import yfinance as yf

from tools.base import BaseTool, ToolParameter


class NewsRetrievalTool(BaseTool):
    """Retrieves recent news headlines related to a stock ticker."""

    @property
    def name(self) -> str:
        return "get_news"

    @property
    def description(self) -> str:
        return (
            "Retrieves recent news headlines and article summaries "
            "related to a stock ticker symbol. Returns up to 5 recent articles."
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

        try:
            news = stock.news
        except Exception:
            return f"Error: Could not retrieve news for '{ticker_symbol}'."

        if not news:
            return f"No recent news found for '{ticker_symbol}'."

        # Format up to 5 articles
        articles = []
        for i, article in enumerate(news[:5], 1):
            # yfinance news format may vary between versions
            # Handle both dict and nested content formats
            if isinstance(article, dict):
                # Try newer yfinance format first
                content = article.get("content", article)
                if isinstance(content, dict):
                    title = content.get("title", "No title")
                    publisher = content.get("provider", {})
                    if isinstance(publisher, dict):
                        publisher = publisher.get("displayName", "Unknown source")
                    pub_date = content.get("pubDate", "Unknown date")
                    link = content.get("canonicalUrl", {})
                    if isinstance(link, dict):
                        link = link.get("url", "No link")
                else:
                    title = article.get("title", "No title")
                    publisher = article.get("publisher", "Unknown source")
                    pub_date = article.get("providerPublishTime", "Unknown date")
                    link = article.get("link", "No link")
            else:
                title = str(article)
                publisher = "Unknown source"
                pub_date = "Unknown date"
                link = "No link"

            articles.append(
                f"  {i}. {title}\n"
                f"     Source: {publisher}\n"
                f"     Date: {pub_date}\n"
                f"     Link: {link}"
            )

        return (
            f"Recent News for {ticker_symbol}:\n\n"
            + "\n\n".join(articles)
        )
