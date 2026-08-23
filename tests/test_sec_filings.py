"""
tests/test_sec_filings.py — Phase 7.1 SEC EDGAR tool

No network by default: requests.get is monkeypatched with a small synthetic
companyfacts payload shaped like the real API (a few concepts, two filings
each) rather than a full multi-megabyte fixture — the parsing logic doesn't
care about the other ~150 concepts a real filer reports.
"""

from __future__ import annotations

import dataclasses

import pytest

from tools.sec_filings import SecFilingsTool, _cross_check, _latest_fact


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


TICKERS_PAYLOAD = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

COMPANYFACTS_PAYLOAD = {
    "entityName": "Apple Inc.",
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"val": 90000000000, "fy": 2023, "fp": "FY", "form": "10-K",
                         "end": "2023-09-30", "filed": "2023-11-01"},
                        {"val": 100000000000, "fy": 2024, "fp": "FY", "form": "10-K",
                         "end": "2024-09-28", "filed": "2024-11-01"},
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {"val": 25000000000, "fy": 2024, "fp": "FY", "form": "10-K",
                         "end": "2024-09-28", "filed": "2024-11-01"},
                    ]
                }
            },
        }
    },
}


@pytest.fixture(autouse=True)
def _no_disk_cache(tmp_path, monkeypatch):
    """Redirect the ticker-map cache to a throwaway dir for every test."""
    from config.settings import settings

    patched = dataclasses.replace(settings, sec_edgar_cache_dir=tmp_path)
    monkeypatch.setattr("tools.sec_filings.settings", patched)


def test_latest_fact_picks_most_recently_filed():
    facts = COMPANYFACTS_PAYLOAD["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    latest = _latest_fact(facts)
    assert latest["val"] == 100000000000
    assert latest["filed"] == "2024-11-01"


def test_latest_fact_ignores_non_10k_10q_forms():
    facts = [
        {"val": 1, "form": "8-K", "filed": "2025-01-01"},
        {"val": 2, "form": "10-Q", "filed": "2024-06-01"},
    ]
    assert _latest_fact(facts)["val"] == 2


def test_latest_fact_empty_list_returns_none():
    assert _latest_fact([]) is None


def test_fetch_extracts_facts_and_uses_filed_date_not_period_end(monkeypatch):
    def fake_get(url, headers, timeout):
        if "company_tickers" in url:
            return _FakeResponse(TICKERS_PAYLOAD)
        return _FakeResponse(COMPANYFACTS_PAYLOAD)

    monkeypatch.setattr("tools.sec_filings.requests.get", fake_get)
    monkeypatch.setattr("tools.sec_filings._cross_check", lambda *a, **k: [])

    payload = SecFilingsTool().fetch({"ticker": "aapl"})

    assert payload["cik"] == "0000320193"
    assert payload["facts"]["revenues"]["value"] == 100000000000
    assert payload["facts"]["revenues"]["filed"] == "2024-11-01"
    assert payload["latest_filed"] == "2024-11-01"


def test_render_includes_filed_date_disclaimer(monkeypatch):
    monkeypatch.setattr("tools.sec_filings._cross_check", lambda *a, **k: [])

    def fake_get(url, headers, timeout):
        if "company_tickers" in url:
            return _FakeResponse(TICKERS_PAYLOAD)
        return _FakeResponse(COMPANYFACTS_PAYLOAD)

    monkeypatch.setattr("tools.sec_filings.requests.get", fake_get)

    tool = SecFilingsTool()
    result = tool.execute({"ticker": "AAPL"})
    assert result.success
    assert "filed" in result.data.lower()
    assert "$100,000,000,000" in result.data


def test_unknown_ticker_raises_and_becomes_failed_result(monkeypatch):
    monkeypatch.setattr(
        "tools.sec_filings.requests.get",
        lambda url, headers, timeout: _FakeResponse(TICKERS_PAYLOAD),
    )
    result = SecFilingsTool().execute({"ticker": "NOTREAL"})
    assert not result.success
    assert "not in the SEC" in result.error


def test_cross_check_flags_material_disagreement(monkeypatch):
    class _FakeTicker:
        info = {"totalRevenue": 100_000_000_000 * 1.20}  # 20% apart

    monkeypatch.setattr("yfinance.Ticker", lambda ticker: _FakeTicker())

    facts = {"revenues": {"value": 100_000_000_000, "filed": "2024-11-01", "form": "10-K"}}
    conflicts = _cross_check("AAPL", facts)

    assert len(conflicts) == 1
    assert conflicts[0]["source_a"] == "sec_edgar"
    assert conflicts[0]["source_b"] == "yfinance"


def test_cross_check_silent_when_within_threshold(monkeypatch):
    class _FakeTicker:
        info = {"totalRevenue": 100_000_000_000 * 1.01}  # 1% apart

    monkeypatch.setattr("yfinance.Ticker", lambda ticker: _FakeTicker())

    facts = {"revenues": {"value": 100_000_000_000, "filed": "2024-11-01", "form": "10-K"}}
    assert _cross_check("AAPL", facts) == []


def test_cross_check_skips_quarterly_facts(monkeypatch):
    """A 10-Q revenue figure vs yfinance's TTM totalRevenue is always ~4x
    apart by construction — must not be reported as a conflict."""
    class _FakeTicker:
        info = {"totalRevenue": 400_000_000_000}

    monkeypatch.setattr("yfinance.Ticker", lambda ticker: _FakeTicker())

    facts = {"revenues": {"value": 100_000_000_000, "filed": "2026-07-31", "form": "10-Q"}}
    assert _cross_check("AAPL", facts) == []


@pytest.mark.network
def test_live_edgar_returns_apple_facts():
    """Live canary — catches SEC changing the companyfacts schema."""
    result = SecFilingsTool().execute({"ticker": "AAPL"})
    assert result.success, result.error
    assert result.structured["facts"]["revenues"]["value"] > 0
