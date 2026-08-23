"""
tools/sec_filings.py — SEC EDGAR company facts (Phase 7.1)
=============================================================

The first genuinely independent source family. Every other tool reads
yfinance's `Ticker.info`; this reads SEC EDGAR's `companyfacts` XBRL API
directly — figures as *filed*, with a `filed` date distinct from the fiscal
period they describe. That distinction is what makes point-in-time
fundamentals possible later (plan.md §8.2's lookahead-bias note).

No API key, but SEC requires a descriptive `User-Agent` with a real contact
or every request 403s (config/settings.py `sec_edgar_user_agent`).

Two network calls, cheap to cache:
  1. ticker -> CIK, from the SEC's own ticker list (cached to disk, refreshed
     monthly — the mapping changes rarely).
  2. CIK -> companyfacts (a few MB of JSON; we pull ~6 concepts out of it).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from config.settings import settings
from tools.base import BaseTool, ToolParameter
from utils.logger import get_logger

logger = get_logger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
TICKERS_CACHE_MAX_AGE_SECONDS = 30 * 86400  # the CIK map changes rarely

# us-gaap concept -> our label. Kept small and deliberately: this is
# cross-check evidence, not a replacement for financial_engine's yfinance
# path — see plan.md 7.4, which pins that reroute to when 3 families exist.
CONCEPTS: dict[str, str] = {
    "Revenues": "revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenues",
    "NetIncomeLoss": "net_income",
    "Assets": "total_assets",
    "Liabilities": "total_liabilities",
    "StockholdersEquity": "stockholders_equity",
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
}

# yfinance metric this EDGAR fact can be cross-checked against, and the
# relative-difference threshold above which it's a genuine disagreement
# rather than GAAP-vs-non-GAAP noise.
CROSS_CHECK: dict[str, tuple[str, float]] = {
    "revenues": ("totalRevenue", 0.05),
}


def _headers() -> dict[str, str]:
    return {"User-Agent": settings.sec_edgar_user_agent, "Accept": "application/json"}


def _load_ticker_cik_map() -> dict[str, str]:
    """Ticker -> zero-padded 10-digit CIK, cached to disk for ~30 days."""
    cache_path = settings.sec_edgar_cache_dir / "company_tickers.json"
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < TICKERS_CACHE_MAX_AGE_SECONDS:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return _index_by_ticker(raw)

    resp = requests.get(TICKERS_URL, headers=_headers(), timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(raw), encoding="utf-8")
    return _index_by_ticker(raw)


def _index_by_ticker(raw: dict[str, Any]) -> dict[str, str]:
    return {
        row["ticker"].upper(): str(row["cik_str"]).zfill(10)
        for row in raw.values()
    }


def _duration_penalty(fact: dict[str, Any]) -> float:
    """
    How far this fact's reporting period is from what its form implies.

    A single 10-Q reports the same concept for multiple spans in one filing
    (the quarter, year-to-date, and the same quarter a year ago) — all with
    the same `filed` date. Without this, `max()` on `filed` alone ties and
    picks whichever the API happened to list first, which silently mixed a
    prior-year comparative figure into what looked like the current quarter.

    Instant concepts (balance-sheet items: Assets, Liabilities, equity) have
    no `start`, so they get a perfect score and are decided by end/filed alone.
    """
    start, end = fact.get("start"), fact.get("end")
    if not start or not end:
        return 0.0
    try:
        days = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
    except ValueError:
        return 0.0
    expected = 365 if str(fact.get("form", "")).startswith("10-K") else 91
    return -abs(days - expected)


def _latest_fact(facts: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """
    The value for one concept as of the most recent filing.

    Ranked by (filed date, period end, duration match) in that order — filed
    date first because a later filing can restate an earlier one, period end
    second to prefer the freshest period a filing reports, duration match
    last to break the same-filing multi-span tie described above.
    """
    candidates = [
        f for f in facts
        if f.get("val") is not None and str(f.get("form", "")).startswith(("10-K", "10-Q"))
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda f: (f.get("filed", ""), f.get("end", ""), _duration_penalty(f)),
    )


class SecFilingsTool(BaseTool):
    """Point-in-time fundamentals as actually filed with the SEC."""

    @property
    def name(self) -> str:
        return "get_sec_filings"

    @property
    def description(self) -> str:
        return (
            "Retrieves key financial facts (revenue, net income, assets, "
            "liabilities, equity, operating cash flow) directly from SEC "
            "EDGAR's XBRL filings, with the actual filing date — the only "
            "source in this system independent of yfinance."
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
                COMPANYFACTS_URL.format(cik=cik), headers=_headers(), timeout=20
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(
                f"SEC EDGAR companyfacts request failed for {ticker_symbol} "
                f"(CIK {cik}): {e}. If this is a 403, set a real "
                f"SEC_EDGAR_USER_AGENT in .env."
            ) from e

        payload = resp.json()
        us_gaap = payload.get("facts", {}).get("us-gaap", {})

        # Some labels have more than one possible XBRL tag (companies changed
        # concepts, e.g. after ASU 606). Pool every alias's facts together
        # before picking the latest — picking per-alias and taking the first
        # populated one silently locked onto a retired tag's stale history.
        facts_by_label: dict[str, list[dict[str, Any]]] = {}
        for concept, label in CONCEPTS.items():
            units = us_gaap.get(concept, {}).get("units", {}).get("USD", [])
            facts_by_label.setdefault(label, []).extend(units)

        extracted: dict[str, dict[str, Any]] = {}
        for label, units in facts_by_label.items():
            fact = _latest_fact(units)
            if fact:
                extracted[label] = {
                    "value": fact["val"],
                    "fiscal_year": fact.get("fy"),
                    "fiscal_period": fact.get("fp"),
                    "form": fact.get("form"),
                    "period_end": fact.get("end"),
                    "filed": fact.get("filed"),
                }

        if not extracted:
            raise ValueError(
                f"No usable us-gaap facts found for {ticker_symbol} (CIK {cik})."
            )

        latest_filed = max(
            (f["filed"] for f in extracted.values() if f.get("filed")), default=""
        )

        return {
            "ticker": ticker_symbol,
            "cik": cik,
            "entity_name": payload.get("entityName", ""),
            "facts": extracted,
            "latest_filed": latest_filed,
            "conflicts": _cross_check(ticker_symbol, extracted),
        }

    def render(self, payload: dict[str, Any]) -> str:
        ticker = payload.get("ticker", "")
        entity = payload.get("entity_name", "")
        facts = payload.get("facts", {})

        if not facts:
            return f"No SEC EDGAR facts available for {ticker}."

        lines = [f"SEC EDGAR filed financials for {entity or ticker} ({ticker}):", ""]
        for label, fact in facts.items():
            lines.append(
                f"  {label.replace('_', ' ').title()}: ${fact['value']:,} "
                f"(FY{fact.get('fiscal_year')} {fact.get('fiscal_period')}, "
                f"{fact.get('form')}, period ending {fact.get('period_end')}, "
                f"filed {fact.get('filed')})"
            )
        lines.append(
            "\nDates above are FILING dates, not period-end dates — use them, "
            "not the period end, to reason about what was actually knowable "
            "on a given day."
        )
        return "\n".join(lines)


def _cross_check(ticker: str, facts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Compare an EDGAR-filed figure against yfinance's `.info` for the same
    concept. A real disagreement here is the first case in this project of
    two INDEPENDENT sources, not two derivations of one (see market_context's
    beta check, and 6.5's note on why that one doesn't count).
    """
    conflicts: list[dict[str, Any]] = []
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        logger.debug(f"Cross-check skipped, yfinance lookup failed: {e}")
        return conflicts

    for label, (yf_field, threshold) in CROSS_CHECK.items():
        fact = facts.get(label)
        yf_value = info.get(yf_field)
        if not fact or not isinstance(yf_value, (int, float)) or yf_value == 0:
            continue
        # yfinance's totalRevenue is trailing-twelve-months. A quarterly
        # EDGAR figure is a different time base by construction (~4x off)
        # and would flag every single run — only compare annual to annual.
        if not str(fact.get("form", "")).startswith("10-K"):
            continue
        edgar_value = fact["value"]
        rel_diff = abs(edgar_value - yf_value) / abs(yf_value)
        if rel_diff > threshold:
            conflicts.append({
                "metric": label,
                "source_a": "sec_edgar",
                "value_a": edgar_value,
                "source_b": "yfinance",
                "value_b": yf_value,
                "relative_difference": round(rel_diff, 4),
                "detail": (
                    f"EDGAR {label}=${edgar_value:,} (filed {fact.get('filed')}) "
                    f"vs yfinance {yf_field}=${yf_value:,.0f} "
                    f"({rel_diff:.1%} apart)"
                ),
            })
    return conflicts
