"""
tests/test_document_loaders.py — Phase 7.2 (PDF) and 7.3 (transcript) loaders

No real PDF file needed: pdfplumber.open() is monkeypatched with a fake
context manager whose pages return canned extract_text() results. That
covers the two things this code actually branches on — "extracted fine",
"a page came back blank" — without depending on a binary fixture.
"""

from __future__ import annotations

import pytest

from knowledge.ingestion.loaders import load_earnings_transcript, load_pdf_document
from knowledge.ingestion.pipeline import IngestionPipeline
from knowledge.retrieval.schemas import DocumentType, SourceTier


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_pdfplumber(monkeypatch, pages):
    import pdfplumber

    monkeypatch.setattr(pdfplumber, "open", lambda path: _FakePdf(pages))


def test_load_pdf_document_extracts_all_pages(monkeypatch):
    _patch_pdfplumber(monkeypatch, [_FakePage("Page one text."), _FakePage("Page two text.")])

    doc = load_pdf_document("fake_10k.pdf", ticker="aapl")

    assert "Page one text." in doc.content
    assert "Page two text." in doc.content
    assert doc.ticker == "AAPL"
    assert doc.source_type == DocumentType.SEC_FILING.value
    assert doc.source_tier == SourceTier.TIER_1.value
    assert doc.extra_metadata["total_pages"] == 2
    assert doc.extra_metadata["empty_pages"] == 0


def test_load_pdf_document_raises_when_fully_scanned(monkeypatch):
    """All pages empty -> almost certainly a scanned doc; OCR isn't implemented."""
    _patch_pdfplumber(monkeypatch, [_FakePage(""), _FakePage(None)])

    with pytest.raises(ValueError, match="scanned"):
        load_pdf_document("scanned.pdf")


def test_load_pdf_document_partial_scan_still_ingests_available_text(monkeypatch, caplog):
    _patch_pdfplumber(monkeypatch, [_FakePage("Real text here."), _FakePage("")])

    doc = load_pdf_document("mixed.pdf")

    assert doc.content == "Real text here."
    assert doc.extra_metadata["empty_pages"] == 1


def test_load_pdf_document_open_failure_wraps_as_valueerror(monkeypatch):
    import pdfplumber

    def _raise(path):
        raise Exception("not a PDF")

    monkeypatch.setattr(pdfplumber, "open", _raise)

    with pytest.raises(ValueError, match="Could not read PDF"):
        load_pdf_document("garbage.pdf")


def test_investor_deck_gets_tier_2_not_tier_1(monkeypatch):
    _patch_pdfplumber(monkeypatch, [_FakePage("Q3 highlights.")])

    doc = load_pdf_document(
        "deck.pdf", source_type=DocumentType.ANALYST_REPORT, source_name="investor_deck"
    )
    assert doc.source_tier == SourceTier.TIER_2.value


def test_load_earnings_transcript_basic():
    doc = load_earnings_transcript(
        "Operator: Welcome to the call...", ticker="nvda", fiscal_period="Q2 FY2026"
    )
    assert doc.source_type == DocumentType.EARNINGS_TRANSCRIPT.value
    assert doc.ticker == "NVDA"
    assert doc.extra_metadata["fiscal_period"] == "Q2 FY2026"


def test_load_earnings_transcript_rejects_empty_content():
    with pytest.raises(ValueError, match="empty"):
        load_earnings_transcript("   ", ticker="NVDA")


def test_pipeline_ingest_pdf_delegates_to_ingest_document(monkeypatch):
    pipeline = IngestionPipeline.__new__(IngestionPipeline)  # skip __init__, no real store needed
    seen = {}

    def _fake_ingest(doc):
        seen["doc"] = doc
        return 3

    monkeypatch.setattr(pipeline, "ingest_document", _fake_ingest)
    _patch_pdfplumber(monkeypatch, [_FakePage("Some text.")])

    result = pipeline.ingest_pdf("f.pdf", ticker="AAPL")

    assert result == 3
    assert seen["doc"].ticker == "AAPL"


def test_pipeline_ingest_earnings_transcript_delegates_to_ingest_document(monkeypatch):
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    seen = {}

    def _fake_ingest(doc):
        seen["doc"] = doc
        return 1

    monkeypatch.setattr(pipeline, "ingest_document", _fake_ingest)

    result = pipeline.ingest_earnings_transcript("Hello.", ticker="NVDA")

    assert result == 1
    assert seen["doc"].source_type == DocumentType.EARNINGS_TRANSCRIPT.value
