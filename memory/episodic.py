"""
memory/episodic.py — Episodic Memory (Prior Run Experiences)
=============================================================

WHY THIS EXISTS:
    Episodic memory stores WHAT HAPPENED during prior agent runs —
    not the financial data itself (that's in long-term memory),
    but the EXPERIENCE of analyzing it:

    - "I successfully analyzed NVDA using 4 tools in 5 iterations."
    - "The get_news tool failed for XYZINVALID — it's not a valid ticker."
    - "TSLA analysis required checking both stock price and news."

    This is the foundation for AUTONOMOUS LEARNING:
    - The agent can recall that certain tickers require more tools.
    - Failed analyses provide lessons for future attempts.
    - Recurring patterns (e.g., always check news last) can be reinforced.

WHAT IS STORED:
    - Query text and ticker symbol.
    - Tools used and their success/failure.
    - Number of iterations taken.
    - Final status (completed, error, max_iterations).
    - Key observations summary.
    - Timestamps.

WHAT IS NOT STORED:
    - Full tool outputs (those go in long-term memory).
    - Raw reasoning traces (too verbose).
    - LLM response text (not useful for future retrieval).

LIFECYCLE:
    - Persistent across agent sessions (stored as JSON files).
    - Grows with every agent run.
    - Can be queried by ticker, recency, or outcome.

ARCHITECTURE:
    File-based (JSON) rather than vector-based because:
    1. Episodic entries are structured (not free-text).
    2. Queries are by field (ticker, status), not semantic similarity.
    3. The number of episodes is small (100s, not 1000s).
    4. File-based is simpler and doesn't need embeddings.

AUTONOMOUS LEARNING IMPLICATIONS:
    Phase 3+ could use episodic memory for:
    - Tool selection optimization (prefer tools that succeeded before).
    - Iteration estimation (predict how many iterations a query needs).
    - Error avoidance (skip known-bad tool/ticker combinations).

LONG-TERM SCALING:
    - Current: JSON files on disk.
    - Future: SQLite for structured queries.
    - Production: PostgreSQL with indexed queries.

HOW IT CONNECTS:
    - main.py saves episodes after each agent run.
    - main.py queries episodes at startup for prior context.
    - agent/state.py carries episodic_context through the graph.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger(__name__)


# ===================================================================
# Episode Data Model
# ===================================================================

class Episode(BaseModel):
    """
    Record of a single agent run experience.

    Captures the high-level outcome and strategy, not the raw data.
    """
    id: str = Field(
        default_factory=lambda: f"ep_{uuid.uuid4().hex[:10]}",
        description="Unique episode identifier"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When this episode occurred"
    )

    # --- Query Context ---
    query: str = Field(description="The user's original query")
    ticker: str = Field(default="", description="Primary ticker analyzed")

    # --- Execution Summary ---
    status: str = Field(default="", description="Final status: completed, error, etc.")
    iterations_used: int = Field(default=0, description="How many iterations were needed")
    max_iterations: int = Field(default=10, description="Maximum allowed iterations")
    execution_time_seconds: float = Field(default=0.0)

    # --- Tool Usage ---
    tools_used: list[str] = Field(
        default_factory=list,
        description="List of tool names called (in order)"
    )
    tools_succeeded: int = Field(default=0)
    tools_failed: int = Field(default=0)
    failed_tools: list[str] = Field(
        default_factory=list,
        description="Tools that returned errors"
    )

    # --- Key Observations ---
    key_findings: str = Field(
        default="",
        description="Brief summary of the main findings"
    )
    errors_encountered: list[str] = Field(
        default_factory=list,
        description="Error messages from the run"
    )

    # --- Memory Stats ---
    evidence_retrieved: int = Field(
        default=0,
        description="Number of evidence items retrieved from vector memory"
    )
    conflicts_detected: int = Field(
        default=0,
        description="Number of evidence conflicts found"
    )

    def to_context_string(self) -> str:
        """
        Format this episode as context for the agent's prompt.

        Provides a concise summary of the prior experience.
        """
        lines = [
            f"Prior Analysis: {self.query}",
            f"  Ticker: {self.ticker or 'N/A'}",
            f"  Status: {self.status}",
            f"  Tools Used: {', '.join(self.tools_used) or 'none'}",
            f"  Iterations: {self.iterations_used}/{self.max_iterations}",
        ]
        if self.key_findings:
            lines.append(f"  Key Findings: {self.key_findings[:200]}")
        if self.failed_tools:
            lines.append(f"  Failed Tools: {', '.join(self.failed_tools)}")
        if self.errors_encountered:
            lines.append(f"  Errors: {'; '.join(self.errors_encountered[-2:])}")
        return "\n".join(lines)


# ===================================================================
# Episodic Memory Store
# ===================================================================

class EpisodicMemory:
    """
    File-based episodic memory for agent run experiences.

    Stores and retrieves Episode records as JSON files.
    Each episode is one JSON file in the data/episodic/ directory.
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        max_episodes: int = 500,
    ) -> None:
        """
        Args:
            storage_dir: Directory for episode JSON files.
            max_episodes: Maximum episodes to retain (oldest evicted).
        """
        if storage_dir is None:
            storage_dir = str(
                Path(__file__).resolve().parent.parent / "data" / "episodic"
            )

        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._max_episodes = max_episodes

        logger.info(
            f"EpisodicMemory initialized: {self._storage_dir} "
            f"({self.count} existing episodes)"
        )

    def save_episode(self, episode: Episode) -> str:
        """
        Save an episode to disk.

        Args:
            episode: The episode to persist.

        Returns:
            Episode ID.
        """
        filepath = self._storage_dir / f"{episode.id}.json"

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(episode.model_dump(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved episode: {episode.id} (query='{episode.query[:50]}...')")
        except Exception as e:
            logger.error(f"Failed to save episode {episode.id}: {e}")
            return ""

        # Evict oldest if over limit
        self._evict_if_needed()

        return episode.id

    def get_recent_episodes(
        self,
        n: int = 5,
        ticker: Optional[str] = None,
    ) -> list[Episode]:
        """
        Get the N most recent episodes, optionally filtered by ticker.

        Args:
            n: Maximum episodes to return.
            ticker: Filter by ticker symbol (case-insensitive).

        Returns:
            List of Episode objects, most recent first.
        """
        episodes = self._load_all_episodes()

        if ticker:
            ticker_upper = ticker.upper()
            episodes = [e for e in episodes if e.ticker.upper() == ticker_upper]

        # Sort by timestamp (most recent first)
        episodes.sort(key=lambda e: e.timestamp, reverse=True)

        return episodes[:n]

    def get_context_for_query(
        self,
        query: str,
        ticker: Optional[str] = None,
        max_episodes: int = 3,
    ) -> str:
        """
        Build episodic context string for the agent's prompt.

        Searches for relevant prior episodes and formats them
        as context. This is injected into the system prompt so
        the agent can learn from past experiences.

        Args:
            query: Current query (for relevance matching).
            ticker: Ticker to filter by.
            max_episodes: Maximum prior episodes to include.

        Returns:
            Formatted context string (empty if no relevant episodes).
        """
        recent = self.get_recent_episodes(n=max_episodes, ticker=ticker)

        if not recent:
            return ""

        lines = [
            "═══════════════════════════════════════════════════════════════════",
            "PRIOR ANALYSIS HISTORY (from episodic memory)",
            "═══════════════════════════════════════════════════════════════════",
            "",
        ]

        for ep in recent:
            lines.append(ep.to_context_string())
            lines.append("")

        lines.append(
            "Use these prior experiences to inform your strategy. "
            "Avoid repeating past failures."
        )
        lines.append(
            "═══════════════════════════════════════════════════════════════════"
        )

        return "\n".join(lines)

    @property
    def count(self) -> int:
        """Number of stored episodes."""
        return len(list(self._storage_dir.glob("*.json")))

    def clear(self) -> None:
        """Delete all stored episodes."""
        for f in self._storage_dir.glob("*.json"):
            f.unlink()
        logger.info("Episodic memory cleared")

    def _load_all_episodes(self) -> list[Episode]:
        """Load all episode files from disk."""
        episodes = []
        for filepath in self._storage_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                episodes.append(Episode(**data))
            except Exception as e:
                logger.warning(f"Failed to load episode {filepath.name}: {e}")
        return episodes

    def _evict_if_needed(self) -> None:
        """Remove oldest episodes if we exceed max_episodes."""
        episodes = self._load_all_episodes()
        if len(episodes) <= self._max_episodes:
            return

        # Sort by timestamp (oldest first)
        episodes.sort(key=lambda e: e.timestamp)

        # Delete oldest until under limit
        to_delete = len(episodes) - self._max_episodes
        for ep in episodes[:to_delete]:
            filepath = self._storage_dir / f"{ep.id}.json"
            try:
                filepath.unlink()
                logger.debug(f"Evicted old episode: {ep.id}")
            except Exception:
                pass
