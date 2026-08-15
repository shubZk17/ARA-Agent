"""
memory/long_term.py — Persistent Vector-Based Long-Term Memory
===============================================================

WHY THIS EXISTS:
    Long-term memory bridges Phase 1 tools and Phase 2 retrieval.
    Every tool output is stored here, building an ever-growing knowledge
    base that the agent can search semantically.

    Without long-term memory, the agent:
    - Can't recall prior research (re-fetches everything each run).
    - Can't cross-reference historical data with current data.
    - Loses all accumulated knowledge when the session ends.

    With long-term memory:
    - "What was NVDA's P/E ratio last time I checked?" → semantic search.
    - Prior tool outputs enrich future analyses.
    - The agent builds institutional knowledge over time.

STORAGE STRATEGY:
    Backed by the vector store (Chroma in dev, Pinecone in prod).
    Content is chunked, embedded, and stored with rich metadata.
    Retrieval is semantic (not keyword-based).

LIFECYCLE:
    - Persistent across agent sessions.
    - Grows with every tool execution and analysis.
    - Can be selectively cleared by ticker or time range.

HOW IT CONNECTS:
    - Wraps retrieval/vector_store.py with the BaseMemory interface.
    - Uses ingestion/pipeline.py for storage.
    - Uses retrieval/retriever.py for retrieval.
    - Agent nodes interact with this via the BaseMemory interface.
"""

from __future__ import annotations

from typing import Any, Optional

from knowledge.memory.base import BaseMemory, MemoryEntry, MemoryType, RetrievalResult
from knowledge.retrieval.embeddings import EmbeddingPipeline
from knowledge.retrieval.vector_store import VectorStoreBase
from utils.logger import get_logger

logger = get_logger(__name__)


class LongTermMemory(BaseMemory):
    """
    Persistent memory backed by a vector database.

    Implements the BaseMemory interface using the vector store
    and embedding pipeline from the retrieval layer.
    """

    def __init__(
        self,
        vector_store: VectorStoreBase,
        embedding_pipeline: EmbeddingPipeline,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedding_pipeline

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.LONG_TERM

    def store(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        entry_id: Optional[str] = None,
    ) -> str:
        """
        Store content in vector memory with embedding.

        The content is embedded and stored directly (no chunking here —
        use IngestionPipeline for full document processing).
        """
        import uuid

        entry_id = entry_id or f"ltm_{uuid.uuid4().hex[:12]}"
        metadata = metadata or {}

        # Generate embedding
        embedding = self._embedder.embed_text(content)

        # Store in vector DB
        self._vector_store.add(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )

        logger.debug(f"Long-term memory stored: {entry_id} ({len(content)} chars)")
        return entry_id

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Semantic search over long-term memory.
        """
        import time

        start = time.time()

        # Embed query
        query_embedding = self._embedder.embed_text(query)

        # Search
        results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        # Convert to MemoryEntry objects
        entries = []
        for result in results:
            entry = MemoryEntry(
                id=result.get("id", ""),
                content=result.get("document", ""),
                metadata=result.get("metadata", {}),
                memory_type=MemoryType.LONG_TERM,
                relevance_score=result.get("score", 0.0),
            )
            entries.append(entry)

        elapsed_ms = (time.time() - start) * 1000

        return RetrievalResult(
            entries=entries,
            query=query,
            total_searched=self._vector_store.count(),
            search_time_ms=round(elapsed_ms, 2),
        )

    def clear(self) -> None:
        """Clear all long-term memory."""
        self._vector_store.clear()
        logger.info("Long-term memory cleared")

    @property
    def count(self) -> int:
        return self._vector_store.count()

    def delete(self, entry_id: str) -> bool:
        """Delete a specific entry from long-term memory."""
        try:
            self._vector_store.delete(ids=[entry_id])
            return True
        except Exception:
            return False
