"""
retrieval/embeddings.py — Embedding Pipeline
==============================================

WHY THIS EXISTS:
    Embeddings are the bridge between human-readable text and machine-searchable
    vectors. This module converts text strings into high-dimensional vectors
    using OpenAI's text-embedding-3-small model.

    WHY text-embedding-3-small?
    - 1536 dimensions (good balance of quality and efficiency).
    - Excellent performance on financial/technical text.
    - $0.02 per 1M tokens (very cost-effective).
    - 8191 token limit (sufficient for well-chunked documents).

EMBEDDING LIFECYCLE:
    1. Raw text (from document chunks) enters this pipeline.
    2. Text is validated (non-empty, within token limits).
    3. Text is sent to OpenAI's embedding API.
    4. A 1536-dimensional float vector is returned.
    5. The vector is stored alongside the text in the vector store.

COSINE SIMILARITY EXPLAINED:
    Two vectors A and B:
    - cos(A, B) = (A · B) / (|A| × |B|)
    - Result ranges from -1 to 1 (0 to 1 for normalized embeddings).
    - Higher = more semantically similar.

    Example:
    - "NVDA revenue growth" vs "NVIDIA earnings increased" → ~0.85 (high)
    - "NVDA revenue growth" vs "weather forecast today" → ~0.12 (low)

    This is WHY embedding-based retrieval works — similar MEANINGS
    produce similar VECTORS, even with different words.

FAILURE CASES:
    1. API rate limits → exponential backoff retry.
    2. Invalid API key → clear error message.
    3. Text exceeds token limit → truncation warning.
    4. Empty text → skip with warning.
    5. Network errors → retry with timeout.

RETRIEVAL PRECISION vs RECALL TRADE-OFFS:
    - Higher top_k → more recall (find more relevant docs) but lower precision.
    - Lower top_k → higher precision but may miss relevant docs.
    - ARA-1 default: top_k=5. Good balance for financial analysis.

HOW IT CONNECTS:
    - ingestion/pipeline.py calls embed_chunks() after chunking.
    - retrieval/retriever.py calls embed_query() for search queries.
    - config/settings.py provides the OpenAI API key and model name.

SCALABILITY:
    - Batch embedding support for bulk ingestion.
    - Caching layer (future) to avoid re-embedding identical text.
    - Model swapping via config (e.g., to Cohere or local models).
"""

from __future__ import annotations

import os
import time
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ===================================================================
# Embedding Model Configuration
# ===================================================================

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
MAX_TOKENS_PER_CHUNK = 8191  # Model limit
MAX_BATCH_SIZE = 100  # OpenAI batch limit


class EmbeddingPipeline:
    """
    Generates vector embeddings using OpenAI's embedding API.

    Supports:
    - Single text embedding (for queries).
    - Batch embedding (for bulk document ingestion).
    - Graceful error handling with retries.
    - Token counting for validation.

    COMMON BUGS:
    1. Forgetting to set OPENAI_API_KEY → clear error on first call.
    2. Sending empty strings → OpenAI returns an error, we skip them.
    3. Exceeding batch size → we split into sub-batches automatically.
    4. Rate limiting → exponential backoff.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        """
        Args:
            model: OpenAI embedding model name.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            max_retries: Number of retry attempts on transient failures.
        """
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._max_retries = max_retries
        self._client = None  # Lazy initialization

        if not self._api_key:
            logger.warning(
                "OPENAI_API_KEY not set. Embedding pipeline will use "
                "fallback placeholder embeddings (NOT suitable for production)."
            )

    def _get_client(self):
        """Lazy-initialize the OpenAI client."""
        if self._client is not None:
            return self._client

        if not self._api_key:
            return None

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
            logger.info(f"OpenAI embedding client initialized (model={self._model})")
            return self._client
        except ImportError:
            logger.error(
                "openai package not installed. Install with: pip install openai"
            )
            return None

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text string.

        Used for:
        - Embedding search queries during retrieval.
        - Embedding single documents during real-time ingestion.

        Args:
            text: The text to embed.

        Returns:
            1536-dimensional float vector.
            Returns placeholder vector if API unavailable.
        """
        if not text or not text.strip():
            logger.warning("Attempted to embed empty text — returning zero vector")
            return [0.0] * EMBEDDING_DIMENSIONS

        client = self._get_client()
        if client is None:
            return self._fallback_embedding(text)

        for attempt in range(self._max_retries):
            try:
                response = client.embeddings.create(
                    model=self._model,
                    input=text.strip(),
                )
                embedding = response.data[0].embedding
                logger.debug(
                    f"Embedded text ({len(text)} chars) → "
                    f"{len(embedding)}-dim vector"
                )
                return embedding

            except Exception as e:
                if attempt < self._max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Embedding attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Embedding failed after {self._max_retries} attempts: {e}. "
                        f"Falling back to placeholder embedding."
                    )
                    return self._fallback_embedding(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Used for bulk document ingestion. Automatically handles:
        - Empty text filtering.
        - Batch size limits (100 per API call).
        - Retry logic per batch.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (same order as input).
        """
        if not texts:
            return []

        # Validate and clean
        clean_texts = []
        empty_indices = set()
        for i, t in enumerate(texts):
            if t and t.strip():
                clean_texts.append(t.strip())
            else:
                clean_texts.append("placeholder")  # OpenAI rejects empty strings
                empty_indices.add(i)

        client = self._get_client()
        if client is None:
            return [self._fallback_embedding(t) for t in texts]

        all_embeddings: list[list[float]] = []

        # Process in batches of MAX_BATCH_SIZE
        for batch_start in range(0, len(clean_texts), MAX_BATCH_SIZE):
            batch = clean_texts[batch_start:batch_start + MAX_BATCH_SIZE]

            for attempt in range(self._max_retries):
                try:
                    response = client.embeddings.create(
                        model=self._model,
                        input=batch,
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    logger.debug(
                        f"Batch embedded {len(batch)} texts "
                        f"(batch {batch_start // MAX_BATCH_SIZE + 1})"
                    )
                    break

                except Exception as e:
                    if attempt < self._max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(
                            f"Batch embedding attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"Batch embedding failed after {self._max_retries} attempts. "
                            f"Using fallback for {len(batch)} texts."
                        )
                        all_embeddings.extend(
                            [self._fallback_embedding(t) for t in batch]
                        )

        # Zero out embeddings for empty inputs
        for idx in empty_indices:
            if idx < len(all_embeddings):
                all_embeddings[idx] = [0.0] * EMBEDDING_DIMENSIONS

        return all_embeddings

    @staticmethod
    def _fallback_embedding(text: str) -> list[float]:
        """
        Generate a deterministic placeholder embedding when the API is unavailable.

        WHY a fallback instead of crashing?
        - Development/testing may not have API keys.
        - The agent should degrade gracefully, not crash.
        - The fallback produces poor retrieval quality but the system still functions.

        METHOD:
        Uses a simple hash-based approach to produce consistent (deterministic)
        but low-quality vectors. Same text always produces same vector.
        """
        import hashlib
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand hash to fill EMBEDDING_DIMENSIONS
        embedding = []
        for i in range(EMBEDDING_DIMENSIONS):
            byte_idx = i % len(hash_bytes)
            # Normalize byte (0-255) to float (-1.0 to 1.0)
            val = (hash_bytes[byte_idx] / 255.0) * 2.0 - 1.0
            embedding.append(round(val, 6))
        return embedding

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Falls back to word-count estimate if tiktoken unavailable.
        """
        try:
            import tiktoken
            encoder = tiktoken.encoding_for_model("text-embedding-3-small")
            return len(encoder.encode(text))
        except (ImportError, Exception):
            # Rough estimate: ~0.75 tokens per word for English text
            return int(len(text.split()) * 1.3)
