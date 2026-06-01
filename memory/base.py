"""
memory/base.py — Abstract Memory Interface
=============================================

WHY THIS EXISTS:
    ARA-1 Phase 2 introduces THREE types of memory, each with different
    lifecycles, storage strategies, and access patterns. Without a common
    interface, the agent would need to know the implementation details
    of each memory system — violating dependency inversion.

    This base class defines the CONTRACT that all memory systems fulfil:
    - store(key, value, metadata)  — persist information
    - retrieve(query, top_k)       — find relevant information
    - clear()                      — reset memory

MEMORY SEPARATION — WHY IT MATTERS:
    1. SHORT-TERM MEMORY (in-session):
       - Holds the current reasoning context.
       - Cleared when the agent run ends.
       - Problem it solves: prevents context window overflow by
         selectively including only relevant prior observations.

    2. LONG-TERM MEMORY (persistent vector store):
       - Stores financial documents, tool outputs, and research data.
       - Persists across sessions.
       - Problem it solves: the agent can recall information from
         previous research without re-fetching from APIs.

    3. EPISODIC MEMORY (run experiences):
       - Stores WHAT HAPPENED during prior agent runs.
       - Not the data itself, but the reasoning patterns and outcomes.
       - Problem it solves: the agent learns from past successes and
         failures, improving future analysis strategies.

WHAT BELONGS IN VECTOR MEMORY:
    ✅ Tool outputs (stock prices, financial metrics, news articles)
    ✅ Document chunks (SEC filings, earnings transcripts)
    ✅ Synthesized analysis summaries
    ✅ Structured financial data with metadata

WHAT SHOULD NOT GO INTO VECTOR MEMORY:
    ❌ Raw reasoning traces (too noisy, poor embedding quality)
    ❌ System prompts (static, not useful for retrieval)
    ❌ Error messages (not semantically useful)
    ❌ Configuration data (structured, not unstructured)

SCALABILITY:
    The abstract interface allows swapping implementations:
    - Development: in-memory dict, local Chroma
    - Production: Redis for short-term, Pinecone for long-term,
      PostgreSQL for episodic

HOW IT CONNECTS:
    - memory/short_term.py and memory/long_term.py inherit from this.
    - memory/episodic.py uses a specialized version of this interface.
    - agent/nodes.py interacts with memory through these abstractions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ===================================================================
# Memory Types Enum
# ===================================================================

class MemoryType(str, Enum):
    """
    Classification of memory systems.

    Each type has distinct lifecycle, storage, and retrieval characteristics.
    """
    SHORT_TERM = "short_term"    # Single session, in-memory
    LONG_TERM = "long_term"      # Persistent, vector-based
    EPISODIC = "episodic"        # Run history, file-based


# ===================================================================
# Memory Entry Model
# ===================================================================

class MemoryEntry(BaseModel):
    """
    A single unit of information stored in memory.

    WHY Pydantic and not a plain dict?
    - Type validation prevents silent data corruption.
    - Serialization is built-in (JSON, dict).
    - Schema evolution is explicit (add fields with defaults).
    """
    id: str = Field(description="Unique identifier for this entry")
    content: str = Field(description="The actual information content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured metadata (source, timestamp, ticker, etc.)"
    )
    memory_type: MemoryType = Field(description="Which memory system owns this")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When this entry was created"
    )
    relevance_score: float = Field(
        default=0.0,
        description="Retrieval relevance (0.0-1.0), set during search"
    )

    class Config:
        use_enum_values = True


# ===================================================================
# Retrieval Result Model
# ===================================================================

class RetrievalResult(BaseModel):
    """
    Result from a memory retrieval operation.

    Bundles the matching entries with metadata about the search itself.
    """
    entries: list[MemoryEntry] = Field(
        default_factory=list,
        description="Matching memory entries, ordered by relevance"
    )
    query: str = Field(default="", description="The search query used")
    total_searched: int = Field(
        default=0,
        description="Total entries in the memory at search time"
    )
    search_time_ms: float = Field(
        default=0.0,
        description="Search latency in milliseconds"
    )


# ===================================================================
# Abstract Base Memory
# ===================================================================

class BaseMemory(ABC):
    """
    Abstract interface for all memory systems.

    Concrete implementations MUST implement:
        - store()    — persist an entry
        - retrieve() — find relevant entries
        - clear()    — reset the memory
        - count      — number of entries

    Concrete implementations SHOULD implement:
        - delete()   — remove specific entries
        - exists()   — check if an entry exists
    """

    @property
    @abstractmethod
    def memory_type(self) -> MemoryType:
        """Which type of memory this is."""
        ...

    @abstractmethod
    def store(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        entry_id: Optional[str] = None,
    ) -> str:
        """
        Store information in memory.

        Args:
            content: The text content to store.
            metadata: Optional structured metadata.
            entry_id: Optional custom ID (auto-generated if None).

        Returns:
            The ID of the stored entry.
        """
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant entries from memory.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.
            metadata_filter: Optional metadata constraints.

        Returns:
            RetrievalResult with matching entries.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from this memory."""
        ...

    @property
    @abstractmethod
    def count(self) -> int:
        """Number of entries currently in memory."""
        ...

    def delete(self, entry_id: str) -> bool:
        """
        Delete a specific entry. Default implementation returns False.
        Override in subclasses that support deletion.
        """
        return False

    def exists(self, entry_id: str) -> bool:
        """
        Check if an entry exists. Default implementation returns False.
        Override in subclasses that support existence checks.
        """
        return False
"""
Description:
    Abstract base for all memory systems with MemoryEntry/RetrievalResult data models.
    Defines the interface contract (store/retrieve/clear) that short-term, long-term,
    and episodic memory implementations must fulfil.
"""
