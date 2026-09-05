"""
observability/collector.py — Centralized Telemetry Collector
==============================================================

WHY THIS EXISTS:
    Production systems are OPAQUE by default. Without telemetry:
    - You can't answer "why did iteration 4 take 12 seconds?"
    - You can't answer "how many tokens did we spend on that analysis?"
    - You can't answer "which tool failed and how did the agent recover?"

    The collector is a PASSIVE OBSERVER. It sits alongside the execution
    pipeline and records everything without affecting execution flow.
    This is the Observer Pattern applied to agent telemetry.

WHAT PROBLEM IT SOLVES:
    1. Performance attribution — which step is the bottleneck?
    2. Cost tracking — how many tokens per query?
    3. Error forensics — what happened before the crash?
    4. Capacity planning — what's our p95 latency?
    5. Quality assurance — are we getting worse over time?

HOW IT INTEGRATES WITH PHASES 1-3:
    - Phase 1 (agent/nodes.py): records LLM calls, tool executions, parse events.
    - Phase 2 (retrieval/): records embedding calls, vector queries, memory ops.
    - Phase 3 (synthesis/): records engine latency, report generation timing.
    - Phase 4 (evaluation/): the evaluator READS from the collector.

SCALABILITY:
    The in-memory collector works for single-run. For production:
    - Swap to Redis/TimescaleDB backend.
    - Add async flushing for high-throughput.
    - Add sampling for cost control.
    The interface stays the same — only the storage backend changes.

PRODUCTION TRADEOFFS:
    - In-memory = fast but volatile (lost on crash).
    - We mitigate by flushing to JSON on run completion.
    - For distributed: replace with OpenTelemetry SDK.
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from config.logging import get_logger

logger = get_logger(__name__)


class EventType(str, Enum):
    """Categorizes telemetry events for filtering and aggregation."""
    LLM_CALL = "llm_call"
    TOOL_EXECUTION = "tool_execution"
    RETRIEVAL_QUERY = "retrieval_query"
    EMBEDDING_CALL = "embedding_call"
    PARSE_ATTEMPT = "parse_attempt"
    MEMORY_OPERATION = "memory_operation"
    SYNTHESIS_STEP = "synthesis_step"
    ERROR = "error"
    RECOVERY = "recovery"
    CHECKPOINT = "checkpoint"
    RETRY = "retry"
    FALLBACK = "fallback"
    FAILURE_INJECTION = "failure_injection"


@dataclass
class TelemetryEvent:
    """
    A single telemetry event with timing and metadata.

    Every observable action in the system generates one of these.
    They're lightweight by design — a full run might generate 50-200 events.
    """
    event_type: str
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    duration_ms: float = 0.0
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    iteration: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def complete(self, success: bool = True, error: str = "") -> None:
        """Mark this event as completed and compute duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error = error

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunTelemetry:
    """
    Aggregated telemetry for a complete agent run.

    This is the SUMMARY that the evaluation framework reads.
    It's computed from the raw events after the run completes.
    """
    run_id: str = ""
    query: str = ""
    ticker: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    total_duration_ms: float = 0.0

    # LLM stats
    llm_calls: int = 0
    llm_total_ms: float = 0.0
    llm_avg_ms: float = 0.0
    llm_failures: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Tool stats
    tool_calls: int = 0
    tool_total_ms: float = 0.0
    tool_avg_ms: float = 0.0
    tool_failures: int = 0
    unique_tools_used: int = 0
    redundant_tool_calls: int = 0

    # Retrieval stats
    retrieval_queries: int = 0
    retrieval_total_ms: float = 0.0
    embedding_calls: int = 0

    # Memory stats
    memory_operations: int = 0

    # Synthesis stats
    synthesis_total_ms: float = 0.0
    synthesis_steps: int = 0

    # Error / recovery stats
    total_errors: int = 0
    total_recoveries: int = 0
    retries: int = 0
    fallbacks: int = 0

    # Parse stats
    parse_attempts: int = 0
    parse_failures: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class TelemetryCollector:
    """
    Thread-safe telemetry collector for the ARA-1 agent.

    Usage:
        collector = TelemetryCollector(run_id="run_001")

        # Record an event
        event = collector.start_event(EventType.LLM_CALL, "reasoning_llm")
        # ... do the LLM call ...
        collector.end_event(event, success=True, metadata={"tokens": 1500})

        # Or use the context manager
        with collector.track(EventType.TOOL_EXECUTION, "get_stock_price") as event:
            result = tool.execute(input)
            event.metadata["result_size"] = len(result)

        # Get summary
        summary = collector.get_run_telemetry()
    """

    def __init__(self, run_id: str = "") -> None:
        self._run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._events: list[TelemetryEvent] = []
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._query = ""
        self._ticker = ""

    @property
    def run_id(self) -> str:
        return self._run_id

    def set_context(self, query: str = "", ticker: str = "") -> None:
        """Set run-level context (query, ticker)."""
        self._query = query
        self._ticker = ticker

    def start_event(
        self,
        event_type: EventType,
        name: str,
        iteration: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> TelemetryEvent:
        """Start tracking a new event. Returns the event for later completion."""
        event = TelemetryEvent(
            event_type=event_type.value,
            name=name,
            iteration=iteration,
            metadata=metadata or {},
        )
        return event

    def end_event(
        self,
        event: TelemetryEvent,
        success: bool = True,
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Complete an event and record it."""
        if metadata:
            event.metadata.update(metadata)
        event.complete(success=success, error=error)

        with self._lock:
            self._events.append(event)

    def record_event(
        self,
        event_type: EventType,
        name: str,
        duration_ms: float = 0,
        success: bool = True,
        error: str = "",
        iteration: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a completed event in one call (for events already timed)."""
        event = TelemetryEvent(
            event_type=event_type.value,
            name=name,
            duration_ms=duration_ms,
            success=success,
            error=error,
            iteration=iteration,
            metadata=metadata or {},
        )
        event.end_time = event.start_time + (duration_ms / 1000)

        with self._lock:
            self._events.append(event)

    class _EventContext:
        """Context manager for tracking events with automatic timing."""

        def __init__(self, collector: "TelemetryCollector", event: TelemetryEvent):
            self._collector = collector
            self.event = event

        def __enter__(self) -> TelemetryEvent:
            return self.event

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            success = exc_type is None
            error = str(exc_val) if exc_val else ""
            self._collector.end_event(self.event, success=success, error=error)
            return False  # Don't suppress exceptions

    def track(
        self,
        event_type: EventType,
        name: str,
        iteration: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> _EventContext:
        """
        Context manager for tracking events with automatic timing.

        Usage:
            with collector.track(EventType.TOOL_EXECUTION, "get_stock_price") as event:
                result = tool.execute(input)
                event.metadata["result_size"] = len(result.data)
        """
        event = self.start_event(event_type, name, iteration, metadata)
        return self._EventContext(self, event)

    def get_events(
        self,
        event_type: EventType | None = None,
        success_only: bool = False,
    ) -> list[TelemetryEvent]:
        """Get filtered events."""
        with self._lock:
            events = list(self._events)

        if event_type:
            events = [e for e in events if e.event_type == event_type.value]
        if success_only:
            events = [e for e in events if e.success]
        return events

    def get_run_telemetry(self) -> RunTelemetry:
        """
        Compute aggregated telemetry for the entire run.

        This is the primary interface for the evaluation framework.
        Called after graph.invoke() completes.
        """
        with self._lock:
            events = list(self._events)

        now = time.time()
        telemetry = RunTelemetry(
            run_id=self._run_id,
            query=self._query,
            ticker=self._ticker,
            start_time=self._start_time,
            end_time=now,
            total_duration_ms=(now - self._start_time) * 1000,
        )

        # Track unique tools for redundancy detection
        tool_call_keys = set()
        all_tool_keys = []

        for event in events:
            et = event.event_type

            if et == EventType.LLM_CALL.value:
                telemetry.llm_calls += 1
                telemetry.llm_total_ms += event.duration_ms
                if not event.success:
                    telemetry.llm_failures += 1
                telemetry.total_input_tokens += event.metadata.get("input_tokens", 0)
                telemetry.total_output_tokens += event.metadata.get("output_tokens", 0)

            elif et == EventType.TOOL_EXECUTION.value:
                telemetry.tool_calls += 1
                telemetry.tool_total_ms += event.duration_ms
                if not event.success:
                    telemetry.tool_failures += 1
                # Track for redundancy
                tool_key = f"{event.name}:{json.dumps(event.metadata.get('input', {}), sort_keys=True)}"
                all_tool_keys.append(tool_key)
                tool_call_keys.add(event.name)

            elif et == EventType.RETRIEVAL_QUERY.value:
                telemetry.retrieval_queries += 1
                telemetry.retrieval_total_ms += event.duration_ms

            elif et == EventType.EMBEDDING_CALL.value:
                telemetry.embedding_calls += 1

            elif et == EventType.MEMORY_OPERATION.value:
                telemetry.memory_operations += 1

            elif et == EventType.SYNTHESIS_STEP.value:
                telemetry.synthesis_steps += 1
                telemetry.synthesis_total_ms += event.duration_ms

            elif et == EventType.ERROR.value:
                telemetry.total_errors += 1

            elif et == EventType.RECOVERY.value:
                telemetry.total_recoveries += 1

            elif et == EventType.RETRY.value:
                telemetry.retries += 1

            elif et == EventType.FALLBACK.value:
                telemetry.fallbacks += 1

            elif et == EventType.PARSE_ATTEMPT.value:
                telemetry.parse_attempts += 1
                if not event.success:
                    telemetry.parse_failures += 1

        # Compute averages
        if telemetry.llm_calls > 0:
            telemetry.llm_avg_ms = telemetry.llm_total_ms / telemetry.llm_calls
        if telemetry.tool_calls > 0:
            telemetry.tool_avg_ms = telemetry.tool_total_ms / telemetry.tool_calls

        telemetry.unique_tools_used = len(tool_call_keys)
        telemetry.redundant_tool_calls = len(all_tool_keys) - len(set(all_tool_keys))

        return telemetry

    def export_to_json(self, output_path: str | Path) -> str:
        """Export all events and summary to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "run_id": self._run_id,
            "query": self._query,
            "ticker": self._ticker,
            "summary": self.get_run_telemetry().to_dict(),
            "events": [e.to_dict() for e in self._events],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Telemetry exported to {output_path}")
        return str(output_path)

    def reset(self) -> None:
        """Reset the collector for a new run."""
        with self._lock:
            self._events.clear()
        self._start_time = time.time()
        self._query = ""
        self._ticker = ""

    def __len__(self) -> int:
        return len(self._events)

    def __bool__(self) -> bool:
        """
        A collector is always truthy — even with zero events.

        Without this, Python derives truthiness from __len__, so `if collector:`
        was False for an empty collector. Since nothing instruments the run yet,
        the collector is ALWAYS empty, so every `if collector:` guard in main.py
        and app.py silently skipped telemetry export and set_context. That is
        why data/telemetry/ was never created (defect D9).
        """
        return True
