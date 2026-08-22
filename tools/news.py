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

    def fetch(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """
        Retrieve up to 5 recent articles as structured records.

        Article fields stay separate (title / publisher / date / link) rather
        than pre-joined, so Phase 7's per-source reliability scoring has
        something to score.
        """
        ticker_symbol = tool_input.get("ticker", "").upper().strip()
        if not ticker_symbol:
            raise ValueError("'ticker' parameter is required.")

        try:
            news = yf.Ticker(ticker_symbol).news
        except Exception as e:
            raise ValueError(
                f"Could not retrieve news for '{ticker_symbol}': {e}"
            ) from e

        articles = []
        for article in (news or [])[:5]:
            # yfinance news format varies between versions — handle both the
            # newer nested "content" shape and the older flat one.
            if isinstance(article, dict):
                content = article.get("content", article)
                if isinstance(content, dict) and "content" in article:
                    publisher = content.get("provider", {})
                    if isinstance(publisher, dict):
                        publisher = publisher.get("displayName", "Unknown source")
                    link = content.get("canonicalUrl", {})
                    if isinstance(link, dict):
                        link = link.get("url", "No link")
                    articles.append({
                        "title": content.get("title", "No title"),
                        "publisher": publisher,
                        "published": content.get("pubDate", "Unknown date"),
                        "link": link,
                    })
                else:
                    articles.append({
                        "title": article.get("title", "No title"),
                        "publisher": article.get("publisher", "Unknown source"),
                        "published": article.get("providerPublishTime", "Unknown date"),
                        "link": article.get("link", "No link"),
                    })
            else:
                articles.append({
                    "title": str(article),
                    "publisher": "Unknown source",
                    "published": "Unknown date",
                    "link": "No link",
                })

        # Always a non-empty payload so execute() takes the fetch/render path
        # even when a ticker genuinely has no news.
        return {"ticker": ticker_symbol, "articles": articles}

    def render(self, payload: dict[str, Any]) -> str:
        """Format a fetch() payload for the LLM."""
        ticker = payload.get("ticker", "")
        articles = payload.get("articles", [])

        if not articles:
            return f"No recent news found for '{ticker}'."

        formatted = [
            f"  {i}. {a.get('title')}\n"
            f"     Source: {a.get('publisher')}\n"
            f"     Date: {a.get('published')}\n"
            f"     Link: {a.get('link')}"
            for i, a in enumerate(articles, 1)
        ]
        return f"Recent News for {ticker}:\n\n" + "\n\n".join(formatted)
