"""
retrieval/schemas.py — Document & Chunk Data Models
=====================================================

WHY THIS EXISTS:
    Financial documents come in many forms: SEC filings, earnings transcripts,
    news articles, tool outputs. Before they can be embedded and stored in a
    vector database, they need a consistent data model that captures:

    1. The content itself (text).
    2. The source (where did this come from?).
    3. The context (what company? what time period?).
    4. The reliability tier (how trustworthy is this source?).

    Without standardized schemas, metadata is ad-hoc and unreliable,
    making retrieval filtering impossible and reliability scoring meaningless.

DESIGN DECISIONS:
    - Document = the full original piece of content (e.g., an entire SEC filing).
    - Chunk = a section of a Document, sized for embedding (typically 500-1000 tokens).
    - Each Chunk retains a reference to its parent Document and inherits metadata.
    - Metadata is structured (not free-form dict) to enable consistent filtering.

HOW IT CONNECTS:
    - ingestion/pipeline.py creates Documents from raw inputs.
    - ingestion/chunker.py splits Documents into Chunks.
    - retrieval/embeddings.py embeds Chunks into vectors.
    - retrieval/vector_store.py stores embedded Chunks.
    - reliability/scorer.py uses source_type and source_name for scoring.

SCALABILITY:
    Adding new document types (10-K filings, analyst PDFs) means adding
    a new DocumentType enum value and potentially a new loader in
    ingestion/loaders.py — no changes to the core pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ===================================================================
# Enums — Document Classification
# ===================================================================

class DocumentType(str, Enum):
    """
    Classification of financial documents.

    WHY an enum?
    - Prevents free-form strings that lead to inconsistent categorization.
    - Each type maps to a reliability tier in reliability/tiers.py.
    - Enables type-specific processing in the ingestion pipeline.
    """
    SEC_FILING = "sec_filing"           # 10-K, 10-Q, 8-K filings
    EARNINGS_TRANSCRIPT = "earnings_transcript"
    NEWS_ARTICLE = "news_article"
    ANALYST_REPORT = "analyst_report"
    TOOL_OUTPUT = "tool_output"         # Output from Phase 1 tools
    FINANCIAL_DATA = "financial_data"   # Structured metrics/data
    RESEARCH_NOTE = "research_note"     # Agent-generated summaries
    SOCIAL_MEDIA = "social_media"       # Twitter, Reddit, etc.
    UNKNOWN = "unknown"


class SourceTier(str, Enum):
    """
    Reliability tier for information sources.

    Tier 1: Primary sources — official filings, company reports.
    Tier 2: Established financial media — Reuters, Bloomberg, WSJ.
    Tier 3: Secondary sources — blogs, social media, forums.
    """
    TIER_1 = "tier_1"  # Highest reliability
    TIER_2 = "tier_2"  # Medium reliability
    TIER_3 = "tier_3"  # Lower reliability
    UNKNOWN = "unknown"


# ===================================================================
# Document Model — Full Source Document
# ===================================================================

class Document(BaseModel):
    """
    A complete financial document before chunking.

    Represents the original source material — SEC filing, news article,
    tool output, etc. This is the INPUT to the ingestion pipeline.
    """
    id: str = Field(
        default_factory=lambda: f"doc_{uuid.uuid4().hex[:12]}",
        description="Unique document identifier"
    )
    content: str = Field(description="Full text content of the document")
    title: str = Field(default="", description="Document title or headline")

    # --- Source Metadata ---
    source_type: DocumentType = Field(
        default=DocumentType.UNKNOWN,
        description="Classification of document type"
    )
    source_name: str = Field(
        default="",
        description="Name of the source (e.g., 'Reuters', 'SEC EDGAR', 'get_stock_price')"
    )
    source_url: str = Field(default="", description="URL if available")
    source_tier: SourceTier = Field(
        default=SourceTier.UNKNOWN,
        description="Reliability tier of the source"
    )

    # --- Financial Context ---
    ticker: str = Field(default="", description="Stock ticker symbol (e.g., 'NVDA')")
    company_name: str = Field(default="", description="Full company name")

    # --- Temporal Metadata ---
    document_date: str = Field(
        default="",
        description="When the document was published/created (ISO format)"
    )
    ingestion_date: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When we ingested this document"
    )

    # --- Additional Metadata ---
    extra_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata not covered by structured fields"
    )

    class Config:
        use_enum_values = True


# ===================================================================
# Chunk Model — Embedding-Ready Text Segment
# ===================================================================

class Chunk(BaseModel):
    """
    A text segment from a Document, sized for embedding.

    WHY chunk instead of embedding whole documents?
    1. Embedding models have token limits (8191 for text-embedding-3-small).
    2. Shorter texts produce more focused embeddings (better retrieval precision).
    3. Retrieval returns the specific relevant SECTION, not the entire document.

    Each chunk retains a reference to its parent document and inherits
    relevant metadata for filtering during retrieval.
    """
    id: str = Field(
        default_factory=lambda: f"chunk_{uuid.uuid4().hex[:12]}",
        description="Unique chunk identifier"
    )
    content: str = Field(description="The chunk text content")
    document_id: str = Field(description="Parent document ID")

    # --- Position in Document ---
    chunk_index: int = Field(
        default=0,
        description="Position of this chunk within the document (0-indexed)"
    )
    total_chunks: int = Field(
        default=1,
        description="Total number of chunks from the parent document"
    )

    # --- Inherited Metadata (from parent Document) ---
    source_type: DocumentType = Field(default=DocumentType.UNKNOWN)
    source_name: str = Field(default="")
    source_tier: SourceTier = Field(default=SourceTier.UNKNOWN)
    ticker: str = Field(default="")
    company_name: str = Field(default="")
    document_date: str = Field(default="")
    ingestion_date: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --- Embedding ---
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Vector embedding (set after embedding pipeline)"
    )
    token_count: int = Field(
        default=0,
        description="Number of tokens in this chunk"
    )

    class Config:
        use_enum_values = True

    def to_vector_metadata(self) -> dict[str, Any]:
        """
        Convert to metadata dict for vector store storage.

        WHY a separate method?
        - Vector stores have constraints on metadata types (strings, numbers, booleans).
        - We need to exclude the embedding and content (stored separately).
        - Chroma and Pinecone have different metadata schemas — this normalizes them.
        """
        return {
            "chunk_id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "source_tier": self.source_tier,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "document_date": self.document_date,
            "ingestion_date": self.ingestion_date,
            "token_count": self.token_count,
        }


# ===================================================================
# Retrieved Evidence Model — Retrieval Output
# ===================================================================

class RetrievedEvidence(BaseModel):
    """
    A piece of evidence retrieved from the vector store.

    This is what the retrieval pipeline returns to the agent.
    It bundles the content with relevance and reliability metadata
    so the agent can reason about evidence quality.
    """
    chunk_id: str = Field(description="ID of the retrieved chunk")
    content: str = Field(description="The evidence text")

    # --- Retrieval Metadata ---
    similarity_score: float = Field(
        default=0.0,
        description="Cosine similarity score (0.0-1.0, higher = more relevant)"
    )
    reliability_score: float = Field(
        default=0.0,
        description="Source reliability score (0.0-1.0, higher = more trustworthy)"
    )
    combined_score: float = Field(
        default=0.0,
        description="Weighted combination of similarity and reliability"
    )

    # --- Source Context ---
    source_type: str = Field(default="")
    source_name: str = Field(default="")
    source_tier: str = Field(default="")
    ticker: str = Field(default="")
    document_date: str = Field(default="")

    # --- Flags ---
    is_stale: bool = Field(
        default=False,
        description="True if the evidence may be outdated"
    )
    has_conflict: bool = Field(
        default=False,
        description="True if conflicting evidence exists"
    )
    conflict_details: str = Field(
        default="",
        description="Description of the conflict (if any)"
    )

    def format_for_prompt(self) -> str:
        """
        Format this evidence for injection into the LLM prompt.

        Includes source attribution and confidence metadata so the
        LLM can reason about evidence quality.
        """
        lines = [
            f"[Evidence from {self.source_name or 'Unknown'} "
            f"(Tier: {self.source_tier}, Relevance: {self.similarity_score:.2f}, "
            f"Reliability: {self.reliability_score:.2f})]",
            self.content,
        ]
        if self.is_stale:
            lines.append(f"⚠ STALE: This evidence may be outdated (dated {self.document_date})")
        if self.has_conflict:
            lines.append(f"⚠ CONFLICT: {self.conflict_details}")
        return "\n".join(lines)


# ===================================================================
# Conflict Report Model
# ===================================================================

class ConflictReport(BaseModel):
    """
    Report of conflicting evidence detected during retrieval.

    WHY surface conflicts explicitly?
    - Silently picking one source over another hides uncertainty.
    - The LLM should KNOW when evidence conflicts and reason about it.
    - Transparency in evidence governance reduces hallucination risk.
    """
    id: str = Field(
        default_factory=lambda: f"conflict_{uuid.uuid4().hex[:8]}",
        description="Unique conflict identifier"
    )
    evidence_a_id: str = Field(description="First conflicting evidence")
    evidence_b_id: str = Field(description="Second conflicting evidence")
    evidence_a_summary: str = Field(default="", description="Summary of evidence A")
    evidence_b_summary: str = Field(default="", description="Summary of evidence B")
    conflict_type: str = Field(
        default="contradictory_claim",
        description="Type of conflict: contradictory_claim, stale_data, inconsistent_metric"
    )
    resolution: str = Field(
        default="",
        description="How the conflict was resolved (if at all)"
    )
    confidence_in_resolution: float = Field(
        default=0.0,
        description="Confidence in the resolution (0.0-1.0)"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
