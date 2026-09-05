"""
retrieval/retriever.py — Semantic Retrieval Orchestrator
=========================================================

WHY THIS EXISTS:
    This is the BRAIN of the retrieval system. It coordinates:
    1. Embedding the search query.
    2. Querying the vector store.
    3. Enriching results with reliability scores.
    4. Detecting conflicts between retrieved evidence.
    5. Formatting evidence for injection into the agent's reasoning.

    Without this orchestrator, retrieval would be a series of
    disconnected function calls scattered across the agent nodes.

RETRIEVAL FLOW:
    Query string
      → embed_query() → query vector
      → vector_store.search() → raw results
      → _enrich_with_reliability() → scored results
      → _check_conflicts() → conflict-annotated results
      → format_for_context() → prompt-ready text

RETRIEVAL PRECISION vs RECALL:
    - top_k=3: High precision, may miss relevant evidence.
    - top_k=5: Balanced (ARA-1 default).
    - top_k=10: High recall, but more noise in context.

    We also apply a MINIMUM SIMILARITY THRESHOLD (0.3 by default)
    to filter out irrelevant results even within top_k.

HOW IT CONNECTS:
    - agent/nodes.py calls retrieve_evidence() when the agent
      decides it needs more context.
    - reliability/scorer.py provides source scores.
    - reliability/conflict_resolver.py detects contradictions.
    - prompts/system.py uses formatted evidence in the prompt.

COMMON FAILURE MODES:
    1. Empty vector store → returns empty results gracefully.
    2. Embedding API failure → fallback embeddings (poor quality but functional).
    3. All results below similarity threshold → returns empty with explanation.
    4. Conflicting evidence → flags conflicts, doesn't suppress either side.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from knowledge.retrieval.embeddings import EmbeddingPipeline
from knowledge.retrieval.schemas import RetrievedEvidence, SourceTier
from knowledge.retrieval.vector_store import VectorStoreBase
from config.logging import get_logger

logger = get_logger(__name__)

# Default configuration
DEFAULT_TOP_K = 5
MIN_SIMILARITY_THRESHOLD = 0.3  # Below this, results are considered irrelevant


class SemanticRetriever:
    """
    Orchestrates semantic search over the vector store.

    Combines embedding generation, similarity search, reliability
    scoring, and conflict detection into a single retrieval interface.
    """

    def __init__(
        self,
        vector_store: VectorStoreBase,
        embedding_pipeline: EmbeddingPipeline,
        reliability_scorer: Optional[Any] = None,  # Lazy import to avoid circular deps
        conflict_resolver: Optional[Any] = None,
    ) -> None:
        """
        Args:
            vector_store: The vector database to search.
            embedding_pipeline: For generating query embeddings.
            reliability_scorer: Optional source reliability scorer.
            conflict_resolver: Optional conflict detection engine.
        """
        self._vector_store = vector_store
        self._embedder = embedding_pipeline
        self._reliability_scorer = reliability_scorer
        self._conflict_resolver = conflict_resolver

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        ticker_filter: Optional[str] = None,
        source_type_filter: Optional[str] = None,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
    ) -> list[RetrievedEvidence]:
        """
        Perform semantic retrieval with full evidence governance.

        Args:
            query: Natural language search query.
            top_k: Maximum results to return.
            ticker_filter: Filter by stock ticker.
            source_type_filter: Filter by document type.
            min_similarity: Minimum similarity score to include.

        Returns:
            List of RetrievedEvidence objects, sorted by combined score.
        """
        start_time = time.time()

        if self._vector_store.count() == 0:
            logger.info("Vector store is empty — no retrieval results")
            return []

        # 1. Embed the query
        query_embedding = self._embedder.embed_text(query)

        # 2. Build metadata filter
        metadata_filter = self._build_metadata_filter(
            ticker=ticker_filter,
            source_type=source_type_filter,
        )

        # 3. Search vector store
        raw_results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Fetch extra, then filter by threshold
            metadata_filter=metadata_filter,
        )

        if not raw_results:
            logger.info(f"No results for query: '{query[:80]}...'")
            return []

        # 4. Convert to RetrievedEvidence and apply similarity threshold
        evidence_list: list[RetrievedEvidence] = []

        for result in raw_results:
            sim_score = result.get("score", 0.0)

            if sim_score < min_similarity:
                continue

            metadata = result.get("metadata", {})

            evidence = RetrievedEvidence(
                chunk_id=result.get("id", ""),
                content=result.get("document", ""),
                similarity_score=sim_score,
                source_type=metadata.get("source_type", ""),
                source_name=metadata.get("source_name", ""),
                source_tier=metadata.get("source_tier", ""),
                ticker=metadata.get("ticker", ""),
                document_date=metadata.get("document_date", ""),
            )

            # 5. Apply reliability scoring
            if self._reliability_scorer:
                try:
                    evidence.reliability_score = self._reliability_scorer.score(
                        source_name=evidence.source_name,
                        source_tier=evidence.source_tier,
                        document_date=evidence.document_date,
                    )
                except Exception as e:
                    logger.warning(f"Reliability scoring failed: {e}")
                    evidence.reliability_score = 0.5  # Neutral default
            else:
                # Default reliability based on tier
                evidence.reliability_score = self._default_reliability(
                    evidence.source_tier
                )

            # 6. Compute combined score
            # Weighted: 60% semantic relevance + 40% source reliability
            evidence.combined_score = round(
                evidence.similarity_score * 0.6 + evidence.reliability_score * 0.4,
                4,
            )

            evidence_list.append(evidence)

        # 7. Sort by combined score (descending)
        evidence_list.sort(key=lambda e: e.combined_score, reverse=True)

        # 8. Trim to top_k
        evidence_list = evidence_list[:top_k]

        # 9. Check for conflicts
        if self._conflict_resolver and len(evidence_list) > 1:
            try:
                evidence_list = self._conflict_resolver.check_conflicts(evidence_list)
            except Exception as e:
                logger.warning(f"Conflict resolution failed: {e}")

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Retrieved {len(evidence_list)} evidence items for '{query[:60]}...' "
            f"in {elapsed_ms:.1f}ms"
        )

        return evidence_list

    def format_evidence_for_prompt(
        self,
        evidence_list: list[RetrievedEvidence],
        max_tokens: int = 2000,
    ) -> str:
        """
        Format retrieved evidence for injection into the LLM prompt.

        Respects a token budget to prevent context overflow.
        Evidence is presented with source attribution and confidence
        scores so the LLM can reason about evidence quality.

        Args:
            evidence_list: Retrieved evidence to format.
            max_tokens: Maximum token budget for the evidence section.

        Returns:
            Formatted evidence string for prompt injection.
        """
        if not evidence_list:
            return ""

        lines = [
            "═══════════════════════════════════════════════════════════════════",
            "RETRIEVED EVIDENCE (from vector memory)",
            "═══════════════════════════════════════════════════════════════════",
            "",
        ]

        total_tokens = 0
        included_count = 0

        for i, evidence in enumerate(evidence_list, 1):
            formatted = evidence.format_for_prompt()

            # Estimate tokens
            est_tokens = len(formatted.split()) * 1.3
            if total_tokens + est_tokens > max_tokens:
                lines.append(
                    f"\n[{len(evidence_list) - included_count} additional results "
                    f"omitted due to context budget]"
                )
                break

            lines.append(f"--- Evidence {i}/{len(evidence_list)} ---")
            lines.append(formatted)
            lines.append("")
            total_tokens += est_tokens
            included_count += 1

        if any(e.has_conflict for e in evidence_list):
            lines.append(
                "\n⚠ CONFLICTING EVIDENCE DETECTED: "
                "Some retrieved sources contain contradictory information. "
                "Reason about these conflicts explicitly in your analysis."
            )

        lines.append(
            "═══════════════════════════════════════════════════════════════════"
        )

        return "\n".join(lines)

    @staticmethod
    def _build_metadata_filter(
        ticker: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Build Chroma-compatible metadata filter."""
        conditions = []

        if ticker:
            conditions.append({"ticker": ticker.upper()})
        if source_type:
            conditions.append({"source_type": source_type})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _default_reliability(source_tier: str) -> float:
        """Default reliability scores by tier."""
        tier_scores = {
            "tier_1": 0.9,
            "tier_2": 0.7,
            "tier_3": 0.4,
            "unknown": 0.3,
        }
        return tier_scores.get(source_tier, 0.3)
