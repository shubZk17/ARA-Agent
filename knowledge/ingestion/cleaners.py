"""
ingestion/cleaners.py — Text Cleaning & Preprocessing
=======================================================

WHY THIS EXISTS:
    Financial documents are messy. SEC filings contain HTML artifacts,
    earnings transcripts have speaker tags and timestamps, and news
    articles include boilerplate headers/footers.

    Dirty text produces noisy embeddings → poor retrieval quality.
    This module normalizes text BEFORE chunking and embedding.

DESIGN DECISIONS:
    - Composable cleaners: each function handles one type of noise.
    - Pipeline pattern: cleaners are chained, easy to add/remove.
    - Non-destructive: we preserve meaningful content, only removing noise.

HOW IT CONNECTS:
    - ingestion/pipeline.py calls clean_text() before chunking.
    - Input is raw document text from loaders.
    - Output is cleaned text ready for chunking.
"""

from __future__ import annotations

import re
import unicodedata

from config.logging import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """
    Master cleaning pipeline. Chains all individual cleaners.

    Args:
        text: Raw document text.

    Returns:
        Cleaned, normalized text.
    """
    if not text:
        return ""

    original_len = len(text)

    text = remove_html_tags(text)
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    text = remove_boilerplate(text)
    text = normalize_financial_text(text)

    cleaned_len = len(text)
    if original_len > 0:
        reduction = ((original_len - cleaned_len) / original_len) * 100
        if reduction > 5:
            logger.debug(
                f"Text cleaned: {original_len} → {cleaned_len} chars "
                f"({reduction:.1f}% reduction)"
            )

    return text.strip()


def remove_html_tags(text: str) -> str:
    """Remove HTML tags while preserving text content."""
    # Remove script and style blocks entirely
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Replace <br> and <p> tags with newlines
    text = re.sub(r"<br\s*/?>|</?p\s*/?>", "\n", text, flags=re.IGNORECASE)
    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&nbsp;", " ")
    text = text.replace("&#39;", "'")
    return text


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to their standard forms."""
    # NFKD normalization: decomposes characters (é → e + combining accent)
    # then we re-compose to NFC for consistent storage
    text = unicodedata.normalize("NFKC", text)
    # Replace common Unicode quotation marks with ASCII equivalents
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # Smart double quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # Smart single quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")  # En/em dashes
    text = text.replace("\u2026", "...")  # Ellipsis
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph structure.

    - Collapse multiple spaces into one.
    - Collapse 3+ newlines into double newline (paragraph break).
    - Remove trailing whitespace from lines.
    """
    # Collapse multiple spaces (but not newlines) into one
    text = re.sub(r"[^\S\n]+", " ", text)
    # Remove trailing whitespace from each line
    text = re.sub(r" +\n", "\n", text)
    # Collapse 3+ consecutive newlines into 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def remove_boilerplate(text: str) -> str:
    """
    Remove common boilerplate text from financial documents.

    Targets:
    - Copyright notices
    - "Safe harbor" disclaimers (partial — preserves if substantive)
    - Page numbers and headers
    - Repeated separator lines
    """
    # Remove lines that are just page numbers
    text = re.sub(r"^\s*Page\s+\d+\s*(of\s+\d+)?\s*$", "", text, flags=re.MULTILINE)
    # Remove lines that are just dashes or equals signs (separators)
    text = re.sub(r"^[-=]{10,}\s*$", "", text, flags=re.MULTILINE)
    # Remove copyright lines
    text = re.sub(
        r"^.*(?:Copyright|©|All rights reserved).*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return text


def normalize_financial_text(text: str) -> str:
    """
    Financial-domain-specific text normalization.

    - Standardize currency formats.
    - Normalize percentage representations.
    - Clean up table-like formatting.
    """
    # Standardize dollar amounts: $ 1,234.56 → $1,234.56
    text = re.sub(r"\$\s+", "$", text)
    # Normalize percentage: 25 % → 25%
    text = re.sub(r"(\d)\s+%", r"\1%", text)
    # Clean up excessive tab-based formatting (from tables)
    text = re.sub(r"\t+", "  ", text)
    return text
