"""
ingestion/loaders.py — Document Type Loaders
==============================================

WHY THIS EXISTS:
    Financial data arrives in different forms: tool outputs (strings),
    news articles (text), structured metrics (key-value pairs).
    Each needs to be converted into a standardized Document object
    before entering the ingestion pipeline.

    Loaders handle source-specific parsing and metadata extraction.

HOW IT CONNECTS:
    - ingestion/pipeline.py calls the appropriate loader.
    - Loaders return Document objects (from retrieval/schemas.py).
    - The pipeline then cleans, chunks, and embeds the document.

DESIGN DECISIONS:
    - One loader function per source type for clarity.
    - Each loader enriches metadata (source_type, tier, etc.).
    - Tool outputs get special handling — they're the most common
      source in Phase 2 (auto-ingesting Phase 1 tool results).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from knowledge.retrieval.schemas import Document, DocumentType, SourceTier
from config.logging import get_logger

logger = get_logger(__name__)


# ===================================================================
# Source → Tier Mapping
# ===================================================================

SOURCE_TIER_MAP: dict[str, SourceTier] = {
    # Tier 1 — Primary/Official sources
    "sec_edgar": SourceTier.TIER_1,
    "company_filing": SourceTier.TIER_1,
    "earnings_report": SourceTier.TIER_1,
    "get_financial_metrics": SourceTier.TIER_1,
    "get_stock_price": SourceTier.TIER_1,
    "get_company_info": SourceTier.TIER_1,
    "get_sec_filings": SourceTier.TIER_1,
    "sec_filing_pdf": SourceTier.TIER_1,
    "investor_deck": SourceTier.TIER_2,   # company-issued, not audited like a 10-K
    "earnings_transcript": SourceTier.TIER_1,

    # Tier 2 — Established media
    "reuters": SourceTier.TIER_2,
    "bloomberg": SourceTier.TIER_2,
    "wsj": SourceTier.TIER_2,
    "yahoo_finance": SourceTier.TIER_2,
    "get_news": SourceTier.TIER_2,

    # Tier 3 — Secondary sources
    "seeking_alpha": SourceTier.TIER_3,
    "reddit": SourceTier.TIER_3,
    "twitter": SourceTier.TIER_3,
    "blog": SourceTier.TIER_3,
}


def get_source_tier(source_name: str) -> SourceTier:
    """Look up the reliability tier for a source name."""
    return SOURCE_TIER_MAP.get(source_name.lower(), SourceTier.UNKNOWN)


# ===================================================================
# Tool Output Loader
# ===================================================================

def load_tool_output(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: str,
    ticker: str = "",
) -> Document:
    """
    Convert a Phase 1 tool output into a Document for ingestion.

    This is the most common loader — it automatically ingests
    every tool result into the vector store, building the agent's
    long-term memory as a side effect of normal operation.

    Args:
        tool_name: Name of the tool (e.g., "get_stock_price").
        tool_input: Arguments passed to the tool.
        tool_output: Raw output string from the tool.
        ticker: Stock ticker symbol (extracted from tool_input if not provided).

    Returns:
        Document ready for ingestion.
    """
    if not ticker:
        ticker = tool_input.get("ticker", "")

    # Map tool names to document types
    tool_doc_types = {
        "get_stock_price": DocumentType.FINANCIAL_DATA,
        "get_company_info": DocumentType.FINANCIAL_DATA,
        "get_financial_metrics": DocumentType.FINANCIAL_DATA,
        "get_news": DocumentType.NEWS_ARTICLE,
        "get_sec_filings": DocumentType.SEC_FILING,
    }

    doc = Document(
        content=tool_output,
        title=f"{tool_name} output for {ticker}",
        source_type=tool_doc_types.get(tool_name, DocumentType.TOOL_OUTPUT),
        source_name=tool_name,
        source_tier=get_source_tier(tool_name),
        ticker=ticker.upper(),
        document_date=datetime.now(timezone.utc).isoformat(),
        extra_metadata={
            "tool_name": tool_name,
            "tool_input": str(tool_input),
        },
    )

    logger.debug(f"Loaded tool output as document: {doc.id} ({tool_name}, {ticker})")
    return doc


# ===================================================================
# News Article Loader
# ===================================================================

def load_news_article(
    title: str,
    content: str,
    source: str = "",
    url: str = "",
    ticker: str = "",
    published_date: str = "",
) -> Document:
    """
    Load a news article as a Document.

    Args:
        title: Article headline.
        content: Article body text.
        source: Publisher name (e.g., "Reuters").
        url: Source URL.
        ticker: Related stock ticker.
        published_date: Publication date (ISO format).

    Returns:
        Document with news-specific metadata.
    """
    full_content = f"{title}\n\n{content}" if title else content

    doc = Document(
        content=full_content,
        title=title,
        source_type=DocumentType.NEWS_ARTICLE,
        source_name=source or "unknown_news",
        source_url=url,
        source_tier=get_source_tier(source),
        ticker=ticker.upper() if ticker else "",
        document_date=published_date or datetime.now(timezone.utc).isoformat(),
    )

    logger.debug(f"Loaded news article: {doc.id} ({title[:50]}...)")
    return doc


# ===================================================================
# PDF Loader (Phase 7.2)
# ===================================================================

def load_pdf_document(
    file_path: str,
    ticker: str = "",
    title: str = "",
    source_type: DocumentType = DocumentType.SEC_FILING,
    source_name: str = "sec_filing_pdf",
    document_date: str = "",
) -> Document:
    """
    Extract the embedded text layer of a PDF (10-K/10-Q, investor deck) as
    a Document.

    Text-layer extraction only — no OCR. That is the defensible slice of
    "multi-modal" plan.md 7.2 asks for (~95% of real filings and decks are
    born-digital), not a scanned-image pipeline. A page pdfplumber can't
    extract text from is either genuinely scanned or blank; either way we
    can't recover it here, so it's dropped and counted rather than silently
    producing a shorter document that looks complete.

    Raises:
        ValueError if the file can't be opened as a PDF, or if every page
        comes back empty (almost certainly a scanned document — flag it as
        an error rather than ingesting a documentation of nothing).
    """
    import pdfplumber

    try:
        with pdfplumber.open(file_path) as pdf:
            page_texts = [page.extract_text() or "" for page in pdf.pages]
    except Exception as e:
        raise ValueError(f"Could not read PDF '{file_path}': {e}") from e

    total_pages = len(page_texts)
    empty_pages = sum(1 for t in page_texts if not t.strip())
    if total_pages == 0 or empty_pages == total_pages:
        raise ValueError(
            f"'{file_path}' has no extractable text ({total_pages} pages, all "
            f"empty) — likely a scanned document. OCR is not implemented; "
            f"re-export or find a text-layer version."
        )

    if empty_pages > total_pages * 0.3:
        logger.warning(
            f"'{file_path}': {empty_pages}/{total_pages} pages had no "
            f"extractable text — likely partially scanned. Extraction is "
            f"incomplete, not wrong; treat coverage as partial."
        )

    content = "\n\n".join(t for t in page_texts if t.strip())

    doc = Document(
        content=content,
        title=title or Path(file_path).stem,
        source_type=source_type,
        source_name=source_name,
        source_tier=get_source_tier(source_name),
        ticker=ticker.upper() if ticker else "",
        document_date=document_date or datetime.now(timezone.utc).isoformat(),
        extra_metadata={
            "total_pages": total_pages,
            "empty_pages": empty_pages,
            "original_path": str(file_path),
        },
    )

    logger.info(
        f"Loaded PDF as document: {doc.id} ({total_pages} pages, "
        f"{empty_pages} empty, {source_type})"
    )
    return doc


# ===================================================================
# Earnings Transcript Loader (Phase 7.3)
# ===================================================================

def load_earnings_transcript(
    content: str,
    ticker: str = "",
    title: str = "",
    fiscal_period: str = "",
    published_date: str = "",
) -> Document:
    """
    Load an earnings call transcript (plain text) as a Document.

    Text-only — no audio/video processing. Transcripts are typically pasted
    or downloaded as plain text (investor relations pages, transcript
    services); the loader doesn't fetch them, it only standardizes whatever
    text is handed to it into the same Document shape as every other source.
    """
    if not content or not content.strip():
        raise ValueError("Transcript content is empty.")

    doc = Document(
        content=content,
        title=title or f"Earnings call transcript — {ticker} {fiscal_period}".strip(),
        source_type=DocumentType.EARNINGS_TRANSCRIPT,
        source_name="earnings_transcript",
        source_tier=get_source_tier("earnings_transcript"),
        ticker=ticker.upper() if ticker else "",
        document_date=published_date or datetime.now(timezone.utc).isoformat(),
        extra_metadata={"fiscal_period": fiscal_period} if fiscal_period else {},
    )

    logger.info(f"Loaded earnings transcript: {doc.id} ({ticker} {fiscal_period})")
    return doc


# ===================================================================
# Research Note Loader
# ===================================================================

def load_research_note(
    content: str,
    ticker: str = "",
    title: str = "",
) -> Document:
    """
    Load an agent-generated research note/summary as a Document.

    These are the agent's own synthesized analyses, stored for
    future reference. They get TIER_2 reliability (agent-generated,
    not primary source).

    Args:
        content: The research note text.
        ticker: Related stock ticker.
        title: Note title.

    Returns:
        Document with research note metadata.
    """
    doc = Document(
        content=content,
        title=title or f"Research note for {ticker}",
        source_type=DocumentType.RESEARCH_NOTE,
        source_name="ara_agent",
        source_tier=SourceTier.TIER_2,
        ticker=ticker.upper() if ticker else "",
        document_date=datetime.now(timezone.utc).isoformat(),
    )

    logger.debug(f"Loaded research note: {doc.id}")
    return doc
