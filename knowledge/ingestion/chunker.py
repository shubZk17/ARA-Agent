"""
ingestion/chunker.py — Recursive Text Chunking Engine
======================================================

WHY THIS EXISTS:
    Embedding models and LLM context windows have token limits.
    A 50-page SEC filing can't be embedded as a single vector —
    the embedding would be too diluted to capture specific details.

    Chunking splits documents into smaller, semantically meaningful
    segments that can be individually embedded and retrieved.

CHUNKING TRADEOFFS:
    ┌──────────────────────────────────────────────────────┐
    │ Smaller chunks (200-300 tokens):                     │
    │  ✅ More precise retrieval                           │
    │  ✅ Better embedding quality per chunk               │
    │  ❌ Loss of surrounding context                      │
    │  ❌ More chunks to store and search                   │
    │                                                      │
    │ Larger chunks (800-1500 tokens):                     │
    │  ✅ More context preserved                           │
    │  ✅ Fewer chunks to manage                           │
    │  ❌ More diluted embeddings                           │
    │  ❌ May include irrelevant text in results            │
    └──────────────────────────────────────────────────────┘

    ARA-1 DEFAULT: 500 tokens with 100-token overlap.
    This is a well-tested sweet spot for financial text where:
    - Paragraphs are typically 100-300 tokens.
    - Key metrics often span 2-3 sentences.
    - Overlap prevents splitting a sentence between chunks.

CHUNK OVERLAP STRATEGY:
    Overlap ensures that information at chunk boundaries isn't lost.
    With 100-token overlap, the last 100 tokens of chunk N are also
    the first 100 tokens of chunk N+1.

    WHY overlap matters:
    - "Revenue was $30B" at end of chunk 1
    - "representing 25% YoY growth" at start of chunk 2
    - Without overlap, a query about "revenue growth" might miss
      this relationship.

RECURSIVE SPLITTING:
    We don't just split at token boundaries. We split at natural
    text boundaries, in order of preference:
    1. Double newlines (paragraph breaks)
    2. Single newlines (line breaks)
    3. Sentences (period + space)
    4. Words (space characters)
    5. Characters (last resort)

    This produces chunks that are semantically coherent, not
    arbitrary token-boundary slices.

CONTEXT WINDOW OPTIMIZATION:
    The agent's LLM context window is limited (4096-128K tokens).
    Retrieved chunks must fit within this budget. Token counting
    during chunking ensures we know exactly how many tokens
    each chunk consumes, enabling budget-aware retrieval.

HOW IT CONNECTS:
    - ingestion/pipeline.py feeds documents into chunk_document().
    - Output Chunk objects go to retrieval/embeddings.py for embedding.
    - retrieval/schemas.py defines the Chunk model.

SCALABILITY:
    - Chunk size and overlap are configurable per document type.
    - Financial tables could use a specialized table-aware chunker.
    - Code/structured data could use AST-based chunking.
"""

from __future__ import annotations

from typing import Optional

from knowledge.retrieval.schemas import Chunk, Document
from config.logging import get_logger

logger = get_logger(__name__)


# ===================================================================
# Default Configuration
# ===================================================================

DEFAULT_CHUNK_SIZE = 500      # Target tokens per chunk
DEFAULT_CHUNK_OVERLAP = 100   # Overlap tokens between chunks
MIN_CHUNK_SIZE = 50           # Minimum viable chunk (skip tiny fragments)


# ===================================================================
# Separator Hierarchy
# ===================================================================

# Ordered from most preferred to least preferred split points
SEPARATORS = [
    "\n\n",     # Paragraph break (best — preserves paragraph coherence)
    "\n",       # Line break
    ". ",       # Sentence boundary
    "! ",       # Exclamation
    "? ",       # Question
    "; ",       # Semicolon
    ", ",       # Comma (last structural separator)
    " ",        # Word boundary
]


class TextChunker:
    """
    Recursive text chunker that splits documents into embedding-ready segments.

    ALGORITHM:
    1. Try to split text at the most preferred separator (paragraph break).
    2. If resulting segments are small enough, they become chunks.
    3. If a segment is still too large, recursively split with the next separator.
    4. Apply overlap between consecutive chunks.
    5. Attach metadata from the parent document.

    COMMON BUGS:
    1. Infinite recursion on text without any separators → base case at character split.
    2. Off-by-one in overlap calculation → explicit unit tests needed.
    3. Unicode text with unusual whitespace → normalize before splitting.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE,
    ) -> None:
        """
        Args:
            chunk_size: Target size in approximate tokens.
            chunk_overlap: Number of overlapping tokens between chunks.
            min_chunk_size: Chunks smaller than this are merged or discarded.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size}). "
                f"Otherwise chunks never advance through the text."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size

    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Split a Document into a list of Chunks.

        Each chunk inherits metadata from the parent document
        for retrieval filtering.

        Args:
            document: The source document to chunk.

        Returns:
            List of Chunk objects with inherited metadata.
        """
        if not document.content or not document.content.strip():
            logger.warning(f"Document {document.id} has no content — skipping")
            return []

        # Perform recursive splitting
        text_chunks = self._recursive_split(
            text=document.content.strip(),
            separators=list(SEPARATORS),
        )

        # Apply overlap
        text_chunks = self._apply_overlap(text_chunks)

        # Filter tiny chunks
        text_chunks = [
            c for c in text_chunks
            if self._estimate_tokens(c) >= self._min_chunk_size
        ]

        if not text_chunks:
            # If nothing survived filtering, keep the whole text as one chunk
            text_chunks = [document.content.strip()]

        # Convert to Chunk objects with metadata
        chunks = []
        for i, text in enumerate(text_chunks):
            chunk = Chunk(
                content=text,
                document_id=document.id,
                chunk_index=i,
                total_chunks=len(text_chunks),
                source_type=document.source_type,
                source_name=document.source_name,
                source_tier=document.source_tier,
                ticker=document.ticker,
                company_name=document.company_name,
                document_date=document.document_date,
                ingestion_date=document.ingestion_date,
                token_count=self._estimate_tokens(text),
            )
            chunks.append(chunk)

        logger.info(
            f"Chunked document '{document.id}' into {len(chunks)} chunks "
            f"(avg {sum(c.token_count for c in chunks) // max(len(chunks), 1)} tokens/chunk)"
        )
        return chunks

    def _recursive_split(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """
        Recursively split text using the separator hierarchy.

        ALGORITHM:
        1. Try splitting at the current separator.
        2. Merge consecutive small segments until they reach chunk_size.
        3. If any merged segment is still too large, recurse with the next separator.
        4. Base case: if no separators left, split by character count.
        """
        # Base case: text fits in one chunk
        if self._estimate_tokens(text) <= self._chunk_size:
            return [text] if text.strip() else []

        # Base case: no separators left — hard split
        if not separators:
            return self._hard_split(text)

        current_sep = separators[0]
        remaining_seps = separators[1:]

        # Split at current separator
        parts = text.split(current_sep)

        if len(parts) == 1:
            # Separator not found in text — try next separator
            return self._recursive_split(text, remaining_seps)

        # Merge small parts into chunk-sized groups
        chunks = []
        current_group = ""

        for part in parts:
            candidate = (current_group + current_sep + part).strip() if current_group else part.strip()

            if self._estimate_tokens(candidate) <= self._chunk_size:
                current_group = candidate
            else:
                # Current group is full — save it
                if current_group.strip():
                    chunks.append(current_group.strip())
                # Start new group with this part
                if self._estimate_tokens(part) > self._chunk_size:
                    # Part itself is too large — recurse with finer separators
                    sub_chunks = self._recursive_split(part, remaining_seps)
                    chunks.extend(sub_chunks)
                    current_group = ""
                else:
                    current_group = part.strip()

        # Don't forget the last group
        if current_group.strip():
            chunks.append(current_group.strip())

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """
        Last-resort splitting by approximate token boundaries.

        Used when no semantic separators produce viable chunks.
        This produces less coherent chunks but prevents infinite recursion.
        """
        chars_per_token = 4  # Rough approximation for English
        chunk_chars = self._chunk_size * chars_per_token
        chunks = []

        for i in range(0, len(text), chunk_chars):
            chunk = text[i:i + chunk_chars].strip()
            if chunk:
                chunks.append(chunk)

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """
        Add overlap between consecutive chunks.

        The last N tokens of chunk[i] are prepended to chunk[i+1].
        This ensures information at boundaries isn't lost.
        """
        if len(chunks) <= 1 or self._chunk_overlap <= 0:
            return chunks

        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):
            # Get the tail of the previous chunk for overlap
            prev_words = chunks[i - 1].split()
            overlap_words = prev_words[-self._chunk_overlap:] if len(prev_words) > self._chunk_overlap else prev_words

            # Prepend overlap to current chunk
            overlap_text = " ".join(overlap_words)
            overlapped.append(f"{overlap_text} {chunks[i]}")

        return overlapped

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Estimate token count from text length.

        WHY estimate instead of exact count?
        - tiktoken import is expensive on first call.
        - For chunking decisions, a rough estimate is sufficient.
        - Exact counts are done later in the embedding pipeline.

        Rule of thumb: ~1.3 tokens per whitespace-separated word for English.
        """
        if not text:
            return 0
        return int(len(text.split()) * 1.3)
