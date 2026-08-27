"""
tools/sec_insider.py — SEC EDGAR Form 4 insider filing activity (Phase 7.4)
==============================================================================

The second EDGAR source, but a genuinely independent SIGNAL from
tools/sec_filings.py: Form 4s are filed by the INSIDER (officer, director,
10%+ owner), not the company. A company misstating its financials doesn't
imply an insider misfiles a Form 4 — different filer, different obligation,
different failure mode. That's why it gets its own SOURCE_FAMILIES tag
("sec_edgar_ownership") rather than being folded into "sec_edgar".

Deliberately scoped to FILING ACTIVITY, not per-transaction buy/sell amounts:
each Form 4 is itself an XML document that would need fetching and parsing
individually to extract shares/price/transaction-code, which is a much
heavier fetch (one request per filing) for a tool that isn't in any horizon's
required_tools. Filing count and recency is real, verifiable evidence on its
own — a cluster of insider filings is itself a signal worth surfacing — and
render() says plainly that direction/size isn't included.

No API key, but SEC requires the same descriptive User-Agent as
tools/sec_filings.py — reuses its CIK lookup and headers rather than
duplicating them.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import requests

from tools.base import BaseTool, ToolParameter
from tools.sec_filings import _headers, _load_ticker_cik_map
from utils.logger import get_logger

logger = get_logger(__name__)

OWNERSHIP_FEED_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count=40&output=atom"
)


def _local_tag(tag: str) -> str:
    """Strip the Atom namespace so entries can be read without a namespace map."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _parse_entries(xml_text: str) -> list[dict[str, str]]:
    root = ET.fromstring(xml_text)
    entries = []
    for entry_el in root:
        if _local_tag(entry_el.tag) != "entry":
            continue
        fields = {_local_tag(child.tag): (child.text or "").strip() for child in entry_el}
        entries.append({
            "title": fields.get("title", ""),
            "filed": fields.get("filing-date") or fields.get("updated", "")[:10],
        })
    return entries


class SecInsiderTool(BaseTool):
    """Recent SEC Form 4 (insider transaction) filing activity for a ticker."""

    @property
    def name(self) -> str:
        return "get_insider_transactions"

    @property
    def description(self) -> str:
        return (
            "Retrieves recent SEC Form 4 filing activity (insider transaction "
            "reports) for a ticker directly from EDGAR — filing count and "
            "dates only, not transaction direction or size. Optional; use "
            "when insider activity is relevant, not required for every run."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="ticker",
                type="string",
                description="Stock ticker symbol (e.g., 'AAPL')",
                required=True,
            )
        ]

    def fetch(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        ticker_symbol = tool_input.get("ticker", "").upper().strip()
        if not ticker_symbol:
            raise ValueError("'ticker' parameter is required.")

        try:
            cik = _load_ticker_cik_map().get(ticker_symbol)
        except requests.RequestException as e:
            raise ValueError(f"SEC EDGAR ticker lookup failed: {e}") from e
        if not cik:
            raise ValueError(f"'{ticker_symbol}' is not in the SEC's ticker list.")

        try:
            resp = requests.get(
                OWNERSHIP_FEED_URL.format(cik=cik), headers=_headers(), timeout=20
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(
                f"SEC EDGAR ownership feed request failed for {ticker_symbol} "
                f"(CIK {cik}): {e}."
            ) from e

        entries = _parse_entries(resp.text)
        filed_dates = sorted((e["filed"] for e in entries if e["filed"]), reverse=True)

        return {
            "ticker": ticker_symbol,
            "cik": cik,
            "form4_count": len(entries),
            "most_recent_filed": filed_dates[0] if filed_dates else "",
            "recent_filings": entries[:10],
        }

    def render(self, payload: dict[str, Any]) -> str:
        ticker = payload.get("ticker", "")
        count = payload.get("form4_count", 0)

        if count == 0:
            return f"No recent SEC Form 4 (insider) filings found for {ticker}."

        lines = [
            f"SEC Form 4 insider filing activity for {ticker}: "
            f"{count} filing(s), most recent {payload.get('most_recent_filed', 'unknown')}.",
            "(Filing count and dates only — not transaction direction or size.)",
            "",
        ]
        for f in payload.get("recent_filings", [])[:5]:
            lines.append(f"  {f.get('filed', '?')}: {f.get('title', '')}")
        return "\n".join(lines)
