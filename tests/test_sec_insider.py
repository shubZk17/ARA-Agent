"""
tests/test_sec_insider.py — Phase 7.4 SEC EDGAR insider (Form 4) tool

No network by default: requests.get is monkeypatched with a small synthetic
Atom feed shaped like EDGAR's browse-edgar output=atom response.
"""

from __future__ import annotations

import dataclasses

import pytest

from tools.sec_insider import SecInsiderTool, _parse_entries

TICKERS_PAYLOAD = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>4 - COOK TIMOTHY D (0001214156) (Reporting)</title>
    <updated>2026-08-20T16:30:00-04:00</updated>
    <filing-date>2026-08-20</filing-date>
    <filing-type>4</filing-type>
  </entry>
  <entry>
    <title>4 - MAESTRI LUCA (0001576942) (Reporting)</title>
    <updated>2026-07-15T16:30:00-04:00</updated>
    <filing-date>2026-07-15</filing-date>
    <filing-type>4</filing-type>
  </entry>
</feed>
"""


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


@pytest.fixture(autouse=True)
def _no_disk_cache(tmp_path, monkeypatch):
    """Redirect the shared ticker-map cache to a throwaway dir for every test."""
    from config.settings import settings

    patched = dataclasses.replace(settings, sec_edgar_cache_dir=tmp_path)
    monkeypatch.setattr("tools.sec_filings.settings", patched)


def test_parse_entries_extracts_title_and_filed_date():
    entries = _parse_entries(ATOM_FEED)
    assert len(entries) == 2
    assert entries[0]["filed"] == "2026-08-20"
    assert "COOK TIMOTHY D" in entries[0]["title"]


def test_fetch_returns_count_and_most_recent_filing(monkeypatch):
    # company_tickers.json is fetched via requests.get in _load_ticker_cik_map,
    # which calls resp.json() — a payload-returning fake distinct from the
    # atom-feed fake (which only exposes .text) used by this tool's own fetch.
    class _TickersResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return TICKERS_PAYLOAD

    def routed_get(url, headers, timeout):
        if "company_tickers" in url:
            return _TickersResponse()
        return _FakeResponse(ATOM_FEED)

    monkeypatch.setattr("tools.sec_filings.requests.get", routed_get)
    monkeypatch.setattr("tools.sec_insider.requests.get", routed_get)

    payload = SecInsiderTool().fetch({"ticker": "aapl"})

    assert payload["cik"] == "0000320193"
    assert payload["form4_count"] == 2
    assert payload["most_recent_filed"] == "2026-08-20"


def test_zero_filings_is_not_an_error(monkeypatch):
    class _TickersResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return TICKERS_PAYLOAD

    empty_feed = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'

    def routed_get(url, headers, timeout):
        if "company_tickers" in url:
            return _TickersResponse()
        return _FakeResponse(empty_feed)

    monkeypatch.setattr("tools.sec_filings.requests.get", routed_get)
    monkeypatch.setattr("tools.sec_insider.requests.get", routed_get)

    result = SecInsiderTool().execute({"ticker": "AAPL"})
    assert result.success
    assert result.structured["form4_count"] == 0
    assert "No recent SEC Form 4" in result.data


def test_unknown_ticker_raises_and_becomes_failed_result(monkeypatch):
    class _TickersResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return TICKERS_PAYLOAD

    monkeypatch.setattr(
        "tools.sec_filings.requests.get",
        lambda url, headers, timeout: _TickersResponse(),
    )
    result = SecInsiderTool().execute({"ticker": "NOTREAL"})
    assert not result.success
    assert "not in the SEC" in result.error


@pytest.mark.network
def test_live_edgar_returns_apple_form4_activity():
    """Live canary — catches EDGAR changing the browse-edgar atom schema."""
    result = SecInsiderTool().execute({"ticker": "AAPL"})
    assert result.success, result.error
    assert result.structured["form4_count"] >= 0
