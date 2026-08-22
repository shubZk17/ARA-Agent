"""
memory/short_term.py — In-Session Context Memory
==================================================

WHY THIS EXISTS:
    During a single agent run, the LLM accumulates observations from
    tool calls. As the number of iterations grows, the conversation
    history can exceed the context window or drown useful information
    in noise.

    Short-term memory solves this by:
    1. Storing all observations in a searchable structure.
    2. Allowing selective retrieval of RELEVANT prior observations.
    3. Preventing context window overflow.

    Without short-term memory, the agent either:
    - Includes ALL prior observations (context overflow), or
    - Includes NONE (loses useful context).

    With short-term memory, the agent includes the MOST RELEVANT
    prior observations, maintaining a focused reasoning context.

LIFECYCLE:
    - Created at the start of an agent run.
    - Populated during tool execution (observations stored here).
    - Queried during reasoning (relevant context retrieved).
    - Destroyed when the agent run completes.

STORAGE STRATEGY:
    In-memory Python dict. No persistence needed — this memory
    exists only for the duration of one agent session.

RETRIEVAL ACCESS PATTERN:
    Simple keyword + recency-based matching. Not vector-based —
    that would be overkill for a single session's observations
    (typically 3–10 entries).

HOW IT CONNECTS:
    - agent/nodes.py stores tool outputs here after execution.
    - agent/nodes.py queries here before reasoning to enrich context.
    - Cleared automatically when the agent run ends.

SCALABILITY:
    For single-session use, in-memory is fine. If we later support
    multi-turn conversations, this could be backed by Redis with
    session-scoped TTL keys.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from knowledge.memory.base import (
    BaseMemory,
    MemoryEntry,
    MemoryType,
    RetrievalResult,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class ShortTermMemory(BaseMemory):
    """
    In-session memory for the current agent run.

    Stores observations and intermediate results. Supports simple
    keyword-based retrieval and recency weighting.

    Thread-safe for future async usage (GIL protects dict operations).
    """

    def __init__(self, max_entries: int = 100) -> None:
        """
        Args:
            max_entries: Maximum entries before oldest are evicted.
                         100 is generous — most runs produce 3-10 entries.
        """
        self._entries: dict[str, MemoryEntry] = {}
        self._max_entries = max_entries
        self._insertion_order: list[str] = []  # For recency tracking

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.SHORT_TERM

    def store(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        entry_id: Optional[str] = None,
    ) -> str:
        """
        Store an observation or intermediate result.

        Args:
            content: Text content (e.g., tool output, observation).
            metadata: Optional context (tool_name, ticker, iteration, etc.).
            entry_id: Custom ID or auto-generated UUID.

        Returns:
            ID of the stored entry.
        """
        entry_id = entry_id or f"stm_{uuid.uuid4().hex[:12]}"

        entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata or {},
            memory_type=MemoryType.SHORT_TERM,
        )

        # Evict oldest if at capacity (FIFO eviction)
        if len(self._entries) >= self._max_entries:
            oldest_id = self._insertion_order.pop(0)
            self._entries.pop(oldest_id, None)
            logger.debug(f"Short-term memory evicted oldest entry: {oldest_id}")

        self._entries[entry_id] = entry
        self._insertion_order.append(entry_id)

        logger.debug(
            f"Short-term memory stored: {entry_id} "
            f"({len(content)} chars, metadata={list((metadata or {}).keys())})"
        )
        return entry_id

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant entries using keyword matching + recency weighting.

        Scoring formula:
            score = keyword_overlap_ratio * 0.6 + recency_score * 0.4

        WHY this formula?
        - Keyword overlap captures topical relevance.
        - Recency ensures recent observations are preferred (more likely
          to be contextually relevant in a sequential reasoning process).
        - 60/40 split gives relevance slight priority over recency.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results.
            metadata_filter: Optional metadata constraints.

        Returns:
            RetrievalResult with scored entries.
        """
        start = time.time()
        query_words = set(query.lower().split())
        candidates: list[MemoryEntry] = []

        for entry in self._entries.values():
            # Apply metadata filter
            if metadata_filter:
                if not self._matches_filter(entry.metadata, metadata_filter):
                    continue

            # Calculate keyword overlap score
            content_words = set(entry.content.lower().split())
            overlap = query_words & content_words
            keyword_score = len(overlap) / max(len(query_words), 1)

            # Calculate recency score (position in insertion order)
            try:
                position = self._insertion_order.index(entry.id)
                recency_score = position / max(len(self._insertion_order) - 1, 1)
            except ValueError:
                recency_score = 0.0

            # Combined score
            combined_score = keyword_score * 0.6 + recency_score * 0.4

            scored_entry = entry.model_copy(
                update={"relevance_score": round(combined_score, 4)}
            )
            candidates.append(scored_entry)

        # Sort by relevance (descending)
        candidates.sort(key=lambda e: e.relevance_score, reverse=True)

        elapsed_ms = (time.time() - start) * 1000

        return RetrievalResult(
            entries=candidates[:top_k],
            query=query,
            total_searched=len(self._entries),
            search_time_ms=round(elapsed_ms, 2),
        )

    def clear(self) -> None:
        """Reset all short-term memory."""
        count = len(self._entries)
        self._entries.clear()
        self._insertion_order.clear()
        logger.info(f"Short-term memory cleared ({count} entries removed)")

    @property
    def count(self) -> int:
        return len(self._entries)

    def delete(self, entry_id: str) -> bool:
        """Remove a specific entry."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._insertion_order.remove(entry_id)
            return True
        return False

    def exists(self, entry_id: str) -> bool:
        return entry_id in self._entries

    def get_recent(self, n: int = 3) -> list[MemoryEntry]:
        """
        Get the N most recent entries.

        Useful for building context without a specific query —
        e.g., "what did I just observe?"
        """
        recent_ids = self._insertion_order[-n:]
        return [self._entries[eid] for eid in reversed(recent_ids) if eid in self._entries]

    @staticmethod
    def _matches_filter(
        metadata: dict[str, Any],
        filter_dict: dict[str, Any],
    ) -> bool:
        """Check if metadata matches all filter conditions."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
