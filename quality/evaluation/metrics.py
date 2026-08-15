"""
evaluation/metrics.py — 20+ Evaluation Metrics Framework
==========================================================

WHY THIS EXISTS:
    "If you can't measure it, you can't improve it." — Peter Drucker

    ARA-1 needs to QUANTIFY its own performance. Without metrics:
    - How do you know if retrieval is useful or wasted?
    - How do you know if the agent hallucinates?
    - How do you know if tools are used efficiently?
    - How do you know if confidence scores are calibrated?

    This module defines 20+ metrics that cover every dimension
    of agent performance: reasoning, tools, retrieval, synthesis,
    reliability, and production readiness.

DESIGN DECISIONS:
    1. Every metric is a self-contained dataclass with compute() logic.
    2. Metrics return a value (0.0-1.0 or count) + a human-readable label.
    3. Metrics are COMPOSABLE — the evaluator runs all of them.
    4. Each metric carries a threshold (what counts as "passing").
    5. Results are serializable to JSON for dashboards.

HOW IT CONNECTS:
    - evaluation/evaluator.py computes ALL metrics on a completed run.
    - evaluation/hallucination_detector.py feeds the hallucination metric.
    - evaluation/tool_efficiency.py feeds tool efficiency metrics.
    - observability/collector.py provides latency/timing data.
    - synthesis/schemas.py provides SynthesisReport data.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class MetricCategory(str, Enum):
    """Categories for organizing metrics."""
    REASONING = "reasoning"
    TOOLS = "tools"
    RETRIEVAL = "retrieval"
    SYNTHESIS = "synthesis"
    RELIABILITY = "reliability"
    PERFORMANCE = "performance"


class MetricStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class MetricResult:
    """Result of computing a single metric."""
    name: str
    category: str
    value: float
    formatted: str  # Human-readable value
    threshold: float  # Passing threshold
    status: str  # pass / warn / fail
    description: str = ""
    details: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Complete evaluation report across all metrics."""
    run_id: str = ""
    ticker: str = ""
    query: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metrics: list[MetricResult] = field(default_factory=list)
    overall_score: float = 0.0
    pass_rate: float = 0.0
    total_metrics: int = 0
    passed_metrics: int = 0
    warned_metrics: int = 0
    failed_metrics: int = 0
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "ticker": self.ticker,
            "query": self.query,
            "timestamp": self.timestamp,
            "overall_score": round(self.overall_score, 4),
            "pass_rate": round(self.pass_rate, 4),
            "total_metrics": self.total_metrics,
            "passed_metrics": self.passed_metrics,
            "warned_metrics": self.warned_metrics,
            "failed_metrics": self.failed_metrics,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "metrics": [m.to_dict() for m in self.metrics],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ===================================================================
# Metric Computation Functions
# ===================================================================

def compute_hallucination_rate(
    state: dict,
    hallucination_flags: list[dict] = None,
) -> MetricResult:
    """
    M1: Hallucination Rate — % of claims not backed by tool evidence.

    Target: < 2%.
    """
    flags = hallucination_flags or []
    total_claims = max(len(flags), 1)
    hallucinated = sum(1 for f in flags if f.get("is_hallucination", False))
    rate = hallucinated / total_claims

    return MetricResult(
        name="Hallucination Rate",
        category=MetricCategory.RELIABILITY,
        value=rate,
        formatted=f"{rate:.1%}",
        threshold=0.02,
        status=MetricStatus.PASS if rate <= 0.02 else (
            MetricStatus.WARN if rate <= 0.05 else MetricStatus.FAIL
        ),
        description="Percentage of claims not supported by tool evidence",
        details=f"{hallucinated}/{total_claims} claims flagged",
    )


def compute_tool_efficiency(state: dict) -> MetricResult:
    """
    M2: Tool Efficiency — ratio of useful vs total tool calls.

    A call is "useful" if: succeeded AND output was used in final answer.
    Target: > 70%.
    """
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return MetricResult(
            name="Tool Efficiency",
            category=MetricCategory.TOOLS,
            value=0.0,
            formatted="N/A (no tools)",
            threshold=0.70,
            status=MetricStatus.WARN,
            description="Ratio of successful, non-redundant tool calls",
        )

    succeeded = sum(1 for tc in tool_calls if hasattr(tc, "success") and tc.success)
    total = len(tool_calls)

    # Check for redundant calls (same tool+input called twice)
    seen = set()
    unique = 0
    for tc in tool_calls:
        key = f"{tc.tool_name}:{json.dumps(tc.tool_input, sort_keys=True)}" if hasattr(tc, "tool_name") else str(tc)
        if key not in seen:
            seen.add(key)
            unique += 1

    redundant = total - unique
    useful = succeeded - redundant
    efficiency = useful / total if total > 0 else 0

    return MetricResult(
        name="Tool Efficiency",
        category=MetricCategory.TOOLS,
        value=efficiency,
        formatted=f"{efficiency:.0%}",
        threshold=0.70,
        status=MetricStatus.PASS if efficiency >= 0.70 else (
            MetricStatus.WARN if efficiency >= 0.50 else MetricStatus.FAIL
        ),
        description="Ratio of successful, non-redundant tool calls",
        details=f"{useful} useful / {total} total ({redundant} redundant, {total - succeeded} failed)",
    )


def compute_tool_success_rate(state: dict) -> MetricResult:
    """M3: Tool Success Rate — % of tool calls that succeeded."""
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return MetricResult(
            name="Tool Success Rate", category=MetricCategory.TOOLS,
            value=1.0, formatted="N/A", threshold=0.80,
            status=MetricStatus.PASS, description="% of tool calls that succeeded",
        )
    succeeded = sum(1 for tc in tool_calls if hasattr(tc, "success") and tc.success)
    rate = succeeded / len(tool_calls)
    return MetricResult(
        name="Tool Success Rate", category=MetricCategory.TOOLS,
        value=rate, formatted=f"{rate:.0%}", threshold=0.80,
        status=MetricStatus.PASS if rate >= 0.80 else MetricStatus.FAIL,
        description="% of tool calls that succeeded",
        details=f"{succeeded}/{len(tool_calls)} succeeded",
    )


def compute_iteration_efficiency(state: dict) -> MetricResult:
    """M4: Iteration Efficiency — how close to optimal iteration count."""
    used = state.get("iteration_count", 0)
    maximum = state.get("max_iterations", 10)
    tool_calls = state.get("tool_calls", [])
    unique_tools = len(set(
        tc.tool_name for tc in tool_calls if hasattr(tc, "tool_name")
    ))

    # Optimal = unique_tools + 1 (for final answer)
    optimal = unique_tools + 1 if unique_tools > 0 else 2
    efficiency = min(1.0, optimal / max(used, 1))

    return MetricResult(
        name="Iteration Efficiency", category=MetricCategory.REASONING,
        value=efficiency, formatted=f"{efficiency:.0%}", threshold=0.60,
        status=MetricStatus.PASS if efficiency >= 0.60 else MetricStatus.WARN,
        description="How close to optimal iteration count",
        details=f"Used {used} iterations, optimal ~{optimal}",
    )


def compute_reasoning_depth(state: dict) -> MetricResult:
    """M5: Reasoning Depth — average thought length (proxy for reasoning quality)."""
    trace = state.get("reasoning_trace", [])
    if not trace:
        return MetricResult(
            name="Reasoning Depth", category=MetricCategory.REASONING,
            value=0.0, formatted="0 chars", threshold=50,
            status=MetricStatus.FAIL, description="Average thought length in characters",
        )

    lengths = [len(s.thought) for s in trace if hasattr(s, "thought") and s.thought]
    avg = sum(lengths) / len(lengths) if lengths else 0

    return MetricResult(
        name="Reasoning Depth", category=MetricCategory.REASONING,
        value=avg, formatted=f"{avg:.0f} chars", threshold=50,
        status=MetricStatus.PASS if avg >= 50 else MetricStatus.WARN,
        description="Average thought length in characters",
        details=f"Across {len(lengths)} reasoning steps",
    )


def compute_source_diversity(state: dict) -> MetricResult:
    """M6: Source Diversity — how many distinct tools were used."""
    tool_calls = state.get("tool_calls", [])
    unique = set(tc.tool_name for tc in tool_calls if hasattr(tc, "tool_name"))
    available = 4  # get_stock_price, get_financial_metrics, get_company_info, get_news
    diversity = len(unique) / available if available > 0 else 0

    return MetricResult(
        name="Source Diversity", category=MetricCategory.RETRIEVAL,
        value=diversity, formatted=f"{len(unique)}/{available} tools",
        threshold=0.75,
        status=MetricStatus.PASS if diversity >= 0.75 else MetricStatus.WARN,
        description="Ratio of distinct tools used vs available",
    )


def compute_retrieval_usage(state: dict) -> MetricResult:
    """M7: Retrieval Usage — whether vector retrieval was used."""
    retrieved = state.get("retrieved_evidence", [])
    queries = state.get("retrieval_queries", [])
    count = len(retrieved)
    used = count > 0

    return MetricResult(
        name="Retrieval Usage", category=MetricCategory.RETRIEVAL,
        value=1.0 if used else 0.0,
        formatted=f"{count} items from {len(queries)} queries",
        threshold=0.5,
        status=MetricStatus.PASS if used else MetricStatus.WARN,
        description="Whether vector memory retrieval was utilized",
    )


def compute_conflict_detection(state: dict) -> MetricResult:
    """M8: Conflict Detection — whether the system detected conflicts."""
    conflicts = state.get("conflict_reports", [])
    count = len(conflicts)

    return MetricResult(
        name="Conflict Detection", category=MetricCategory.RELIABILITY,
        value=1.0 if count > 0 else 0.5,
        formatted=f"{count} conflicts detected",
        threshold=0.0,  # Having 0 conflicts is fine
        status=MetricStatus.PASS,
        description="Number of evidence conflicts detected and reported",
    )


def compute_error_recovery_rate(state: dict) -> MetricResult:
    """M9: Error Recovery Rate — % of errors that were handled gracefully."""
    errors = state.get("errors", [])
    status = state.get("status", "")
    completed = status in ("completed", "max_iterations_reached")

    if not errors:
        return MetricResult(
            name="Error Recovery Rate", category=MetricCategory.RELIABILITY,
            value=1.0, formatted="100% (no errors)", threshold=0.80,
            status=MetricStatus.PASS,
            description="% of errors that were handled without crashing",
        )

    recovery_rate = 1.0 if completed else 0.0
    return MetricResult(
        name="Error Recovery Rate", category=MetricCategory.RELIABILITY,
        value=recovery_rate,
        formatted=f"{recovery_rate:.0%} ({len(errors)} errors, {'recovered' if completed else 'failed'})",
        threshold=0.80,
        status=MetricStatus.PASS if recovery_rate >= 0.80 else MetricStatus.FAIL,
        description="% of errors that were handled without crashing",
    )


def compute_completion_status(state: dict) -> MetricResult:
    """M10: Completion Status — did the agent produce a final answer?"""
    has_answer = bool(state.get("final_answer"))
    return MetricResult(
        name="Completion Status", category=MetricCategory.REASONING,
        value=1.0 if has_answer else 0.0,
        formatted="Complete" if has_answer else "Incomplete",
        threshold=1.0,
        status=MetricStatus.PASS if has_answer else MetricStatus.FAIL,
        description="Whether the agent produced a final answer",
    )


def compute_answer_length(state: dict) -> MetricResult:
    """M11: Answer Comprehensiveness — length of the final answer."""
    answer = state.get("final_answer", "")
    length = len(answer)
    # A good financial analysis is typically 500-3000 chars
    quality = min(1.0, length / 1000) if length > 100 else 0.0

    return MetricResult(
        name="Answer Comprehensiveness", category=MetricCategory.SYNTHESIS,
        value=quality, formatted=f"{length} chars", threshold=0.5,
        status=MetricStatus.PASS if quality >= 0.5 else MetricStatus.WARN,
        description="Length and quality proxy of the final answer",
    )


def compute_evidence_grounding(state: dict) -> MetricResult:
    """M12: Evidence Grounding — ratio of evidence actually retrieved."""
    retrieved = state.get("retrieved_evidence", [])
    tool_calls = state.get("tool_calls", [])
    succeeded = sum(1 for tc in tool_calls if hasattr(tc, "success") and tc.success)

    if not succeeded:
        return MetricResult(
            name="Evidence Grounding", category=MetricCategory.RETRIEVAL,
            value=0.0, formatted="No evidence", threshold=0.5,
            status=MetricStatus.FAIL, description="How well evidence supports the analysis",
        )

    grounding = min(1.0, (len(retrieved) + succeeded) / max(succeeded * 2, 1))
    return MetricResult(
        name="Evidence Grounding", category=MetricCategory.RETRIEVAL,
        value=grounding, formatted=f"{grounding:.0%}", threshold=0.5,
        status=MetricStatus.PASS if grounding >= 0.5 else MetricStatus.WARN,
        description="How well evidence supports the analysis",
    )


def compute_latency(state: dict, elapsed_seconds: float = 0) -> MetricResult:
    """M13: Total Latency — wall-clock execution time."""
    return MetricResult(
        name="Total Latency", category=MetricCategory.PERFORMANCE,
        value=elapsed_seconds, formatted=f"{elapsed_seconds:.1f}s",
        threshold=120.0,  # 2 minutes is acceptable
        status=MetricStatus.PASS if elapsed_seconds <= 120 else MetricStatus.WARN,
        description="Total wall-clock execution time",
    )


def compute_avg_iteration_time(state: dict, elapsed_seconds: float = 0) -> MetricResult:
    """M14: Average Iteration Time."""
    iterations = max(state.get("iteration_count", 1), 1)
    avg = elapsed_seconds / iterations

    return MetricResult(
        name="Avg Iteration Time", category=MetricCategory.PERFORMANCE,
        value=avg, formatted=f"{avg:.1f}s", threshold=30.0,
        status=MetricStatus.PASS if avg <= 30 else MetricStatus.WARN,
        description="Average time per reasoning iteration",
    )


def compute_redundant_calls(state: dict) -> MetricResult:
    """M15: Redundant Tool Calls — same tool+input called multiple times."""
    tool_calls = state.get("tool_calls", [])
    seen = set()
    redundant = 0
    for tc in tool_calls:
        if hasattr(tc, "tool_name"):
            key = f"{tc.tool_name}:{json.dumps(tc.tool_input, sort_keys=True)}"
            if key in seen:
                redundant += 1
            seen.add(key)

    return MetricResult(
        name="Redundant Calls", category=MetricCategory.TOOLS,
        value=redundant, formatted=f"{redundant} redundant",
        threshold=1,  # Allow 1 redundant call
        status=MetricStatus.PASS if redundant <= 1 else MetricStatus.WARN,
        description="Number of duplicate tool calls (same tool + same input)",
    )


def compute_episodic_memory_usage(state: dict) -> MetricResult:
    """M16: Episodic Memory Usage — whether prior context was loaded."""
    context = state.get("episodic_context", "")
    used = len(context) > 10

    return MetricResult(
        name="Episodic Memory Usage", category=MetricCategory.RETRIEVAL,
        value=1.0 if used else 0.0,
        formatted="Active" if used else "Not used",
        threshold=0.0,
        status=MetricStatus.PASS,
        description="Whether episodic memory context was loaded",
    )


def compute_parse_error_rate(state: dict) -> MetricResult:
    """M17: Parse Error Rate — % of iterations with JSON parse errors."""
    errors = state.get("errors", [])
    iterations = max(state.get("iteration_count", 1), 1)
    parse_errors = sum(1 for e in errors if "parse" in e.lower() or "json" in e.lower())
    rate = parse_errors / iterations

    return MetricResult(
        name="Parse Error Rate", category=MetricCategory.RELIABILITY,
        value=rate, formatted=f"{rate:.0%} ({parse_errors} errors)",
        threshold=0.20,
        status=MetricStatus.PASS if rate <= 0.20 else MetricStatus.WARN,
        description="Percentage of iterations with JSON parse errors",
    )


def compute_financial_metrics_coverage(
    synthesis_report: Optional[dict] = None,
) -> MetricResult:
    """M18: Financial Metrics Coverage — from Phase 3 synthesis."""
    if not synthesis_report:
        return MetricResult(
            name="Financial Metrics Coverage", category=MetricCategory.SYNTHESIS,
            value=0.0, formatted="No synthesis data", threshold=0.5,
            status=MetricStatus.WARN,
            description="% of financial metrics successfully extracted",
        )

    financial = synthesis_report.get("financial", {})
    available = financial.get("metrics_available", 0)
    missing = financial.get("metrics_missing", 0)
    total = available + missing
    coverage = available / total if total > 0 else 0

    return MetricResult(
        name="Financial Metrics Coverage", category=MetricCategory.SYNTHESIS,
        value=coverage, formatted=f"{coverage:.0%} ({available}/{total})",
        threshold=0.50,
        status=MetricStatus.PASS if coverage >= 0.50 else MetricStatus.WARN,
        description="% of financial metrics successfully extracted",
    )


def compute_confidence_calibration(
    synthesis_report: Optional[dict] = None,
) -> MetricResult:
    """M19: Confidence Calibration Quality — from Phase 3."""
    if not synthesis_report:
        return MetricResult(
            name="Confidence Calibration", category=MetricCategory.SYNTHESIS,
            value=0.5, formatted="No data", threshold=0.3,
            status=MetricStatus.WARN,
            description="Quality of confidence score calibration",
        )

    confidence = synthesis_report.get("confidence", {})
    overall = confidence.get("overall", 0.5)
    # Good calibration = confidence is neither too high nor too low
    # Penalize extreme values (>0.95 or <0.15) as likely miscalibrated
    calibration = 1.0 - abs(overall - 0.6) * 1.5
    calibration = max(0.0, min(1.0, calibration))

    return MetricResult(
        name="Confidence Calibration", category=MetricCategory.SYNTHESIS,
        value=calibration,
        formatted=f"{calibration:.0%} (confidence={overall:.0%})",
        threshold=0.30,
        status=MetricStatus.PASS if calibration >= 0.30 else MetricStatus.WARN,
        description="Quality of confidence score calibration",
    )


def compute_misalignment_detection(
    synthesis_report: Optional[dict] = None,
) -> MetricResult:
    """M20: Misalignment Detection — Phase 3 feature validation."""
    if not synthesis_report:
        return MetricResult(
            name="Misalignment Detection", category=MetricCategory.SYNTHESIS,
            value=0.0, formatted="Not available", threshold=0.0,
            status=MetricStatus.PASS,
            description="Whether sentiment-financial misalignment was checked",
        )

    misalignment = synthesis_report.get("misalignment", {})
    ran = "misalignment_type" in misalignment
    detected = misalignment.get("detected", False)

    return MetricResult(
        name="Misalignment Detection", category=MetricCategory.SYNTHESIS,
        value=1.0 if ran else 0.0,
        formatted=f"{'Detected' if detected else 'None found'}" if ran else "Not run",
        threshold=0.0,
        status=MetricStatus.PASS if ran else MetricStatus.WARN,
        description="Whether sentiment-financial misalignment was checked",
    )


def compute_risk_assessment_quality(
    synthesis_report: Optional[dict] = None,
) -> MetricResult:
    """M21: Risk Assessment Quality — from Phase 3."""
    if not synthesis_report:
        return MetricResult(
            name="Risk Assessment", category=MetricCategory.SYNTHESIS,
            value=0.0, formatted="No data", threshold=0.0,
            status=MetricStatus.WARN,
            description="Number and quality of identified risks",
        )

    risk = synthesis_report.get("risk", {})
    risks = risk.get("risks", [])
    categories = set(r.get("category", "") for r in risks)

    return MetricResult(
        name="Risk Assessment", category=MetricCategory.SYNTHESIS,
        value=len(risks), formatted=f"{len(risks)} risks, {len(categories)} categories",
        threshold=0.0,
        status=MetricStatus.PASS if len(risks) >= 1 else MetricStatus.WARN,
        description="Number and quality of identified risks",
    )


def compute_report_generation(
    synthesis_report: Optional[dict] = None,
    report_paths: Optional[dict] = None,
) -> MetricResult:
    """M22: Report Generation — whether reports were produced."""
    generated = bool(report_paths)
    formats = list(report_paths.keys()) if report_paths else []

    return MetricResult(
        name="Report Generation", category=MetricCategory.SYNTHESIS,
        value=1.0 if generated else 0.0,
        formatted=f"Generated: {', '.join(formats)}" if generated else "Not generated",
        threshold=1.0,
        status=MetricStatus.PASS if generated else MetricStatus.FAIL,
        description="Whether investment reports were successfully generated",
    )


# ===================================================================
# Metric Registry — All metrics in one place
# ===================================================================

ALL_METRIC_FUNCTIONS = [
    compute_hallucination_rate,
    compute_tool_efficiency,
    compute_tool_success_rate,
    compute_iteration_efficiency,
    compute_reasoning_depth,
    compute_source_diversity,
    compute_retrieval_usage,
    compute_conflict_detection,
    compute_error_recovery_rate,
    compute_completion_status,
    compute_answer_length,
    compute_evidence_grounding,
    compute_latency,
    compute_avg_iteration_time,
    compute_redundant_calls,
    compute_episodic_memory_usage,
    compute_parse_error_rate,
    compute_financial_metrics_coverage,
    compute_confidence_calibration,
    compute_misalignment_detection,
    compute_risk_assessment_quality,
    compute_report_generation,
]
