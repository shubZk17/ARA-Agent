"""
agent/checkpoint_manager.py — Execution Checkpoint & Recovery System
======================================================================

WHY THIS EXISTS:
    Agent runs are EXPENSIVE. An analysis that took 8 iterations
    and 45 seconds of LLM calls should not be lost because of a
    rate limit at iteration 9.

    The checkpoint manager:
    1. SAVES state snapshots at each iteration boundary.
    2. ENABLES resumption from any checkpoint.
    3. PERSISTS to disk for crash recovery.
    4. MANAGES checkpoint lifecycle (create, restore, cleanup).

WHAT PROBLEM IT SOLVES:
    "The API timed out at iteration 5. Can I resume from there?"
    → Yes. Load checkpoint_iter_5.json and continue.

HOW IT INTEGRATES:
    - Called by agent/graph.py at each iteration boundary.
    - Reads/writes AgentState (serialized to JSON).
    - Observability collector records checkpoint events.
    - Evaluation metrics track recovery success rate.

DESIGN DECISIONS:
    - JSON serialization (human-readable, debuggable).
    - One file per checkpoint (easy to list and select).
    - Automatic cleanup of old checkpoints (disk hygiene).
    - Pydantic models serialized via .model_dump().

SCALABILITY:
    - Replace filesystem with Redis for distributed agents.
    - Add compression for large states.
    - Add encryption for sensitive financial data.

PRODUCTION TRADEOFFS:
    - Checkpointing adds ~5ms per iteration (negligible vs LLM calls).
    - Disk usage: ~10KB per checkpoint × 10 iterations = 100KB.
    - Benefits far outweigh costs for runs > 30 seconds.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """
    Manages execution checkpoints for agent state recovery.

    Usage:
        # During execution
        manager = CheckpointManager(run_id="run_001")
        manager.save_checkpoint(state, iteration=3)

        # After interruption
        manager = CheckpointManager(run_id="run_001")
        state, iteration = manager.restore_latest()
        # Resume from iteration 3
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "data/checkpoints",
        run_id: str = "",
        max_checkpoints: int = 20,
    ) -> None:
        self._run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._checkpoint_dir = Path(checkpoint_dir) / self._run_id
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._max_checkpoints = max_checkpoints

    @property
    def run_id(self) -> str:
        return self._run_id

    def save_checkpoint(
        self,
        state: dict,
        iteration: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save a checkpoint of the current agent state.

        Args:
            state: Current AgentState dict.
            iteration: Current iteration number.
            metadata: Optional metadata (timing, etc.).

        Returns:
            Path to the saved checkpoint file.
        """
        checkpoint_path = self._checkpoint_dir / f"checkpoint_iter_{iteration:03d}.json"

        # Serialize state (handle Pydantic models and enums)
        serializable_state = self._make_serializable(state)

        checkpoint_data = {
            "run_id": self._run_id,
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": serializable_state,
            "metadata": metadata or {},
        }

        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            logger.debug(f"Checkpoint saved: iteration {iteration} → {checkpoint_path}")

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to save checkpoint at iteration {iteration}: {e}")
            return ""

    def restore_latest(self) -> tuple[dict | None, int]:
        """
        Restore the most recent checkpoint.

        Returns:
            Tuple of (state_dict, iteration) or (None, 0) if no checkpoint.
        """
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            logger.info("No checkpoints found")
            return None, 0

        latest = checkpoints[-1]
        return self.restore_checkpoint(latest)

    def restore_checkpoint(self, checkpoint_path: str | Path) -> tuple[dict | None, int]:
        """
        Restore a specific checkpoint.

        Returns:
            Tuple of (state_dict, iteration) or (None, 0) on failure.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None, 0

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            state = data.get("state", {})
            iteration = data.get("iteration", 0)

            # Reconstruct Pydantic models
            state = self._reconstruct_state(state)

            logger.info(
                f"Checkpoint restored: iteration {iteration} "
                f"from {checkpoint_path.name}"
            )

            return state, iteration

        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return None, 0

    def list_available_checkpoints(self) -> list[dict]:
        """List all available checkpoints with metadata."""
        checkpoints = []
        for path in self._list_checkpoints():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                checkpoints.append({
                    "path": str(path),
                    "iteration": data.get("iteration", 0),
                    "timestamp": data.get("timestamp", ""),
                    "run_id": data.get("run_id", ""),
                })
            except Exception:
                pass
        return checkpoints

    def has_checkpoints(self) -> bool:
        """Check if any checkpoints exist for this run."""
        return len(self._list_checkpoints()) > 0

    def clear_checkpoints(self) -> None:
        """Remove all checkpoints for this run."""
        if self._checkpoint_dir.exists():
            shutil.rmtree(self._checkpoint_dir, ignore_errors=True)
            logger.info(f"Cleared checkpoints for run {self._run_id}")

    def _list_checkpoints(self) -> list[Path]:
        """List checkpoint files sorted by iteration."""
        if not self._checkpoint_dir.exists():
            return []
        return sorted(self._checkpoint_dir.glob("checkpoint_iter_*.json"))

    def _cleanup_old_checkpoints(self) -> None:
        """Remove excess checkpoints beyond max_checkpoints."""
        checkpoints = self._list_checkpoints()
        if len(checkpoints) > self._max_checkpoints:
            to_remove = checkpoints[:-self._max_checkpoints]
            for path in to_remove:
                try:
                    path.unlink()
                except Exception:
                    pass

    def _make_serializable(self, state: dict) -> dict:
        """
        Convert agent state to JSON-serializable format.

        Handles:
        - Pydantic BaseModel objects (via .model_dump())
        - Enum values (via .value)
        - Nested dicts and lists
        """
        result = {}
        for key, value in state.items():
            result[key] = self._serialize_value(value)
        return result

    def _serialize_value(self, value: Any) -> Any:
        """Recursively serialize a value."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]

        # Pydantic model
        if hasattr(value, "model_dump"):
            return value.model_dump()

        # Enum
        if hasattr(value, "value"):
            return value.value

        # Fallback
        return str(value)

    def _reconstruct_state(self, state: dict) -> dict:
        """
        Reconstruct complex objects from serialized state.

        Note: We reconstruct ToolCall and ReasoningStep from dicts.
        AgentStatus is kept as string (nodes.py handles both).
        """
        from agent.state import ToolCall, ReasoningStep

        # Reconstruct tool_calls
        if "tool_calls" in state:
            reconstructed_calls = []
            for tc in state["tool_calls"]:
                if isinstance(tc, dict):
                    reconstructed_calls.append(ToolCall(**tc))
                else:
                    reconstructed_calls.append(tc)
            state["tool_calls"] = reconstructed_calls

        # Reconstruct reasoning_trace
        if "reasoning_trace" in state:
            reconstructed_trace = []
            for step in state["reasoning_trace"]:
                if isinstance(step, dict):
                    reconstructed_trace.append(ReasoningStep(**step))
                else:
                    reconstructed_trace.append(step)
            state["reasoning_trace"] = reconstructed_trace

        return state


def find_latest_run(checkpoint_base_dir: str | Path = "data/checkpoints") -> str | None:
    """
    Find the most recent run ID with checkpoints.

    Returns the run_id or None if no checkpoints exist.
    """
    base = Path(checkpoint_base_dir)
    if not base.exists():
        return None

    run_dirs = sorted(base.iterdir(), reverse=True)
    for run_dir in run_dirs:
        if run_dir.is_dir() and list(run_dir.glob("checkpoint_iter_*.json")):
            return run_dir.name

    return None
