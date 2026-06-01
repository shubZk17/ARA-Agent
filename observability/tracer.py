"""
observability/tracer.py — Execution & Reasoning Trace Recorder
================================================================

WHY THIS EXISTS:
    The collector records EVENTS. The tracer records NARRATIVE —
    the human-readable story of what the agent did and why.

    Think of it like this:
    - Collector = structured metrics (for dashboards)
    - Tracer = structured logs (for debugging)

    When something goes wrong, you don't want to read raw events.
    You want: "At iteration 3, the agent called get_stock_price,
    got a timeout, retried once, succeeded, then reasoned about
    the price data."

WHAT PROBLEM IT SOLVES:
    1. Post-mortem debugging — reconstruct what happened.
    2. Quality review — was the reasoning sound?
    3. Compliance — audit trail for financial recommendations.
    4. Evaluation — provides data for hallucination detection.

HOW IT INTEGRATES:
    - Wraps around agent/nodes.py execution.
    - Consumed by evaluation/hallucination_detector.py.
    - Exported alongside reports for audit.

PRODUCTION TRADEOFFS:
    - Full tracing = complete but verbose.
    - In production, you might sample or filter by severity.
    - The tracer is write-only during execution, read-only after.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TraceSpan:
    """
    A single span in the execution trace.

    Spans form a flat timeline (not a tree — we keep it simple).
    Each span corresponds to one meaningful action.
    """
    span_id: str
    name: str
    span_type: str  # "llm", "tool", "retrieval", "synthesis", "decision"
    iteration: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    duration_ms: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    success: bool = True
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[str] = field(default_factory=list)

    def complete(
        self,
        output_summary: str = "",
        success: bool = True,
        error: str = "",
    ) -> None:
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.output_summary = output_summary
        self.success = success
        self.error = error


@dataclass
class ExecutionTrace:
    """
    Complete execution trace for one agent run.

    Contains all spans in chronological order,
    plus metadata about the run.
    """
    run_id: str = ""
    query: str = ""
    ticker: str = ""
    start_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    spans: list[TraceSpan] = field(default_factory=list)
    total_iterations: int = 0
    final_status: str = ""
    total_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "ticker": self.ticker,
            "start_time": self.start_time,
            "total_iterations": self.total_iterations,
            "final_status": self.final_status,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "span_count": len(self.spans),
            "spans": [asdict(s) for s in self.spans],
        }

    def get_timeline_summary(self) -> str:
        """Generate a human-readable timeline."""
        lines = [
            f"=== Execution Trace: {self.run_id} ===",
            f"Query: {self.query}",
            f"Duration: {self.total_duration_ms:.0f}ms",
            f"Iterations: {self.total_iterations}",
            f"Status: {self.final_status}",
            "",
        ]

        for span in self.spans:
            status = "✓" if span.success else "✗"
            lines.append(
                f"  [{status}] {span.name} "
                f"({span.span_type}, {span.duration_ms:.0f}ms) "
                f"iter={span.iteration}"
            )
            if span.input_summary:
                lines.append(f"      Input: {span.input_summary[:100]}")
            if span.output_summary:
                lines.append(f"      Output: {span.output_summary[:100]}")
            if span.error:
                lines.append(f"      Error: {span.error[:100]}")

        return "\n".join(lines)


class ExecutionTracer:
    """
    Records the execution narrative of an agent run.

    Usage:
        tracer = ExecutionTracer(run_id="run_001")

        span = tracer.start_span("reasoning_node", "llm", iteration=1)
        # ... do work ...
        tracer.end_span(span, output_summary="Decided to call get_stock_price")

        trace = tracer.get_trace()
    """

    def __init__(self, run_id: str = "") -> None:
        self._run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._trace = ExecutionTrace(run_id=self._run_id)
        self._span_counter = 0
        self._start_time = time.time()

    def set_context(self, query: str = "", ticker: str = "") -> None:
        self._trace.query = query
        self._trace.ticker = ticker

    def start_span(
        self,
        name: str,
        span_type: str,
        iteration: int = 0,
        input_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:
        """Start a new trace span."""
        self._span_counter += 1
        span = TraceSpan(
            span_id=f"span_{self._span_counter:04d}",
            name=name,
            span_type=span_type,
            iteration=iteration,
            input_summary=input_summary,
            metadata=metadata or {},
        )
        return span

    def end_span(
        self,
        span: TraceSpan,
        output_summary: str = "",
        success: bool = True,
        error: str = "",
    ) -> None:
        """Complete a span and add it to the trace."""
        span.complete(output_summary, success, error)
        self._trace.spans.append(span)

    def record_span(
        self,
        name: str,
        span_type: str,
        duration_ms: float = 0,
        iteration: int = 0,
        input_summary: str = "",
        output_summary: str = "",
        success: bool = True,
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a completed span in one call."""
        self._span_counter += 1
        now = time.time()
        span = TraceSpan(
            span_id=f"span_{self._span_counter:04d}",
            name=name,
            span_type=span_type,
            iteration=iteration,
            start_time=now - (duration_ms / 1000),
            end_time=now,
            duration_ms=duration_ms,
            input_summary=input_summary,
            output_summary=output_summary,
            success=success,
            error=error,
            metadata=metadata or {},
        )
        self._trace.spans.append(span)

    def get_trace(self) -> ExecutionTrace:
        """Get the complete execution trace."""
        self._trace.total_duration_ms = (time.time() - self._start_time) * 1000
        return self._trace

    def finalize(
        self,
        total_iterations: int = 0,
        final_status: str = "",
    ) -> ExecutionTrace:
        """Finalize the trace with run-level metadata."""
        self._trace.total_iterations = total_iterations
        self._trace.final_status = final_status
        self._trace.total_duration_ms = (time.time() - self._start_time) * 1000
        return self._trace

    def export_to_json(self, output_path: str | Path) -> str:
        """Export trace to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.get_trace().to_dict(), f, indent=2, default=str)

        logger.info(f"Trace exported to {output_path}")
        return str(output_path)
