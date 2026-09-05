"""
retrieval/vector_store.py — Vector Database Abstraction Layer
==============================================================

WHY THIS EXISTS:
    Vector databases store high-dimensional embeddings and enable
    semantic similarity search. This module provides a UNIFIED interface
    over Chroma (local development) and Pinecone (future production).

    WHY an abstraction instead of using Chroma directly?
    1. MIGRATION PATH — When you outgrow Chroma's single-node limits,
       switching to Pinecone should be a config change, not a rewrite.
    2. TESTING — You can swap in a mock vector store for unit tests.
    3. VENDOR LOCK-IN PREVENTION — The agent logic never imports Chroma
       directly. It only knows about VectorStoreBase.

VECTOR INDEXING STRATEGY:
    Chroma uses HNSW (Hierarchical Navigable Small World) indexing by default.
    - Good for: datasets up to ~1M vectors, low-latency queries.
    - Trade-off: approximate nearest neighbors (not exact), but 95%+ recall.
    - For ARA-1's scale (thousands of financial document chunks), HNSW is optimal.

EMBEDDING LIFECYCLE:
    1. Document → chunks (ingestion/chunker.py)
    2. Chunks → embeddings (retrieval/embeddings.py)
    3. Embeddings + metadata → vector store (THIS MODULE)
    4. Query → query embedding → similarity search → ranked results

SIMILARITY SEARCH MECHANICS:
    Cosine similarity: measures angle between vectors.
    - Score 1.0 = identical meaning
    - Score 0.0 = completely unrelated
    - Score -1.0 = opposite meaning (rare in practice)
    Chroma returns distance (lower = more similar), which we convert to
    similarity score for consistent interface.

HOW IT CONNECTS:
    - retrieval/embeddings.py generates vectors and calls store_chunks().
    - retrieval/retriever.py calls search() to find relevant evidence.
    - config/settings.py provides vector DB configuration.
    - memory/long_term.py wraps this as the long-term memory backend.

COMMON FAILURE MODES:
    1. Chroma DB corruption → catch, log, offer re-initialization.
    2. Embedding dimension mismatch → validate on first insert.
    3. Collection not found → auto-create with configured settings.
    4. Metadata type errors → Chroma only accepts str/int/float/bool.

SCALABILITY:
    - Chroma: local dev, up to ~100K chunks.
    - Pinecone: production, millions of vectors, serverless scaling.
    - The abstraction makes this switch transparent to the agent.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from config.logging import get_logger

logger = get_logger(__name__)


# ===================================================================
# Abstract Vector Store Interface
# ===================================================================

class VectorStoreBase(ABC):
    """
    Abstract interface for vector database operations.

    All vector store implementations (Chroma, Pinecone, etc.) must
    implement this interface.
    """

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Store embeddings with associated documents and metadata."""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Find the top-k most similar documents.

        Returns list of dicts with keys:
            - id: chunk ID
            - document: text content
            - metadata: associated metadata
            - score: similarity score (0.0-1.0, higher = more similar)
        """
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored documents."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all documents from the store."""
        ...


# ===================================================================
# Chroma Vector Store — Local Development
# ===================================================================

class ChromaVectorStore(VectorStoreBase):
    """
    Chroma-backed vector store for local development.

    WHY Chroma for development:
    - Zero external dependencies (runs in-process).
    - Persistent storage to disk (survives restarts).
    - Built-in HNSW indexing (no manual index management).
    - Free and open source.

    WHEN to migrate away from Chroma:
    - Dataset exceeds ~100K chunks.
    - Multi-user concurrent access needed.
    - Sub-10ms query latency required at scale.
    """

    def __init__(
        self,
        collection_name: str = "ara_financial_docs",
        persist_directory: Optional[str] = None,
    ) -> None:
        """
        Initialize Chroma vector store.

        Args:
            collection_name: Name of the Chroma collection.
            persist_directory: Path for persistent storage.
                If None, defaults to ./data/chroma.
        """
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb is required for vector storage. "
                "Install it with: pip install chromadb"
            )

        if persist_directory is None:
            persist_directory = str(
                Path(__file__).resolve().parent.parent / "data" / "chroma"
            )

        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",  # Use cosine similarity
                "description": "ARA-1 financial document embeddings",
            },
        )

        self._collection_name = collection_name
        logger.info(
            f"ChromaVectorStore initialized: collection='{collection_name}', "
            f"persist='{persist_directory}', "
            f"existing_docs={self._collection.count()}"
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Add embeddings to Chroma.

        Chroma handles deduplication by ID — if an ID already exists,
        it will be updated (upsert behavior).

        IMPORTANT: Chroma metadata values must be str, int, float, or bool.
        Lists, dicts, and None are NOT allowed. We sanitize metadata here.
        """
        # Sanitize metadata for Chroma compatibility
        clean_metadatas = [self._sanitize_metadata(m) for m in metadatas]

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=clean_metadatas,
            )
            logger.info(f"Stored {len(ids)} chunks in ChromaVectorStore")
        except Exception as e:
            logger.error(f"ChromaVectorStore.add failed: {type(e).__name__}: {e}")
            raise

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic similarity search.

        Chroma returns results sorted by distance (ascending).
        We convert distance to similarity score:
            cosine_distance ranges from 0 (identical) to 2 (opposite).
            similarity = 1 - (distance / 2)

        Args:
            query_embedding: Vector representation of the search query.
            top_k: Number of results to return.
            metadata_filter: Chroma-compatible where clause.

        Returns:
            List of result dicts with id, document, metadata, score.
        """
        if self._collection.count() == 0:
            logger.debug("Vector store is empty — no results")
            return []

        # Build Chroma query
        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }

        # Add metadata filter if provided
        if metadata_filter:
            query_params["where"] = metadata_filter

        try:
            results = self._collection.query(**query_params)
        except Exception as e:
            logger.error(f"ChromaVectorStore.search failed: {type(e).__name__}: {e}")
            return []

        # Parse Chroma results into standard format
        output = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # Convert cosine distance to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0.0
                similarity = 1.0 - (distance / 2.0)

                output.append({
                    "id": doc_id,
                    "document": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": round(max(0.0, min(1.0, similarity)), 4),
                })

        logger.debug(f"Vector search returned {len(output)} results (top_k={top_k})")
        return output

    def delete(self, ids: list[str]) -> None:
        """Delete specific documents by ID."""
        try:
            self._collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks from ChromaVectorStore")
        except Exception as e:
            logger.error(f"ChromaVectorStore.delete failed: {e}")

    def count(self) -> int:
        """Total documents in the collection."""
        return self._collection.count()

    def clear(self) -> None:
        """
        Remove all documents from the collection.

        WHY delete and recreate instead of collection.delete()?
        Chroma's delete-all is done by deleting the collection entirely
        and recreating it. This is cleaner than trying to delete by ID.
        """
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaVectorStore cleared")
        except Exception as e:
            logger.error(f"ChromaVectorStore.clear failed: {e}")

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize metadata for Chroma compatibility.

        Chroma only accepts str, int, float, bool values.
        Everything else gets converted to string.
        """
        clean = {}
        for key, value in metadata.items():
            if value is None:
                clean[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (list, dict)):
                clean[key] = str(value)
            else:
                clean[key] = str(value)
        return clean


# ===================================================================
# Factory Function
# ===================================================================

def create_vector_store(
    backend: str = "chroma",
    collection_name: str = "ara_financial_docs",
    persist_directory: Optional[str] = None,
    **kwargs: Any,
) -> VectorStoreBase:
    """
    Factory function to create the appropriate vector store backend.

    Args:
        backend: "chroma" or "pinecone" (pinecone is a future stub).
        collection_name: Name for the collection/index.
        persist_directory: Storage path (Chroma only).

    Returns:
        Configured VectorStoreBase instance.

    WHY a factory?
    - main.py calls create_vector_store(settings.vector_backend).
    - Switching backends is a config change, not a code change.
    """
    if backend == "chroma":
        return ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
    elif backend == "pinecone":
        raise NotImplementedError(
            "Pinecone backend is planned for production deployment. "
            "Use 'chroma' for local development."
        )
    else:
        raise ValueError(f"Unknown vector store backend: '{backend}'. Use 'chroma' or 'pinecone'.")
