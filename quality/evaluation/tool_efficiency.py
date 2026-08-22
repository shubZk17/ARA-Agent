"""
evaluation/tool_efficiency.py — Tool Usage Analytics & Optimization
=====================================================================

WHY THIS EXISTS:
    Tools are the MOST EXPENSIVE part of agent execution (API calls,
    network I/O, rate limits). Inefficient tool usage means:
    - Higher latency (each unnecessary call adds 1-5 seconds).
    - Higher cost (each call counts against API rate limits).
    - Lower quality (redundant data doesn't add insight).

    This module tracks:
    1. REDUNDANT calls — same tool + same input called multiple times.
    2. FAILED calls — calls that returned errors.
    3. RETRIEVAL WASTE — data retrieved but never used.
    4. UNNECESSARY iterations — loops that didn't add information.

WHAT PROBLEM IT SOLVES:
    Answers "Is the agent using tools efficiently?" with a single
    number: tool_efficiency_score (target > 70%).

HOW IT INTEGRATES:
    - Reads from: agent state (tool_calls, reasoning_trace).
    - Reads from: observability/collector.py (timing data).
    - Outputs to: evaluation/metrics.py (tool efficiency metric).
    - Used by: evaluation/evaluator.py.

PRODUCTION TRADEOFFS:
    - Static analysis (post-run) vs dynamic optimization (during run).
    - We do both: post-run scoring + runtime recommendations.
    - Dynamic optimization would require modifying the LLM prompt,
      which risks destabilizing reasoning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ToolCallAnalysis:
    """Analysis of a single tool call."""
    tool_name: str
    tool_input: dict[str, Any]
    succeeded: bool
    is_redundant: bool = False
    is_useful: bool = True
    output_used_in_answer: bool = False
    latency_ms: float = 0.0
    output_size: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolEfficiencyReport:
    """Complete tool efficiency analysis for one run."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    redundant_calls: int = 0
    useful_calls: int = 0
    unique_tools_used: int = 0
    available_tools: int = 4  # Default Phase 1 tools

    # Scores (0-1)
    efficiency_score: float = 0.0
    success_rate: float = 0.0
    diversity_score: float = 0.0
    redundancy_rate: float = 0.0

    # Timing
    total_tool_time_ms: float = 0.0
    avg_tool_time_ms: float = 0.0
    slowest_call_ms: float = 0.0
    slowest_tool: str = ""

    # Analysis details
    call_analyses: list[ToolCallAnalysis] = field(default_factory=list)
    optimization_suggestions: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "redundant_calls": self.redundant_calls,
            "useful_calls": self.useful_calls,
            "unique_tools_used": self.unique_tools_used,
            "efficiency_score": round(self.efficiency_score, 4),
            "success_rate": round(self.success_rate, 4),
            "diversity_score": round(self.diversity_score, 4),
            "redundancy_rate": round(self.redundancy_rate, 4),
            "total_tool_time_ms": round(self.total_tool_time_ms, 2),
            "avg_tool_time_ms": round(self.avg_tool_time_ms, 2),
            "slowest_call_ms": round(self.slowest_call_ms, 2),
            "slowest_tool": self.slowest_tool,
            "optimization_suggestions": self.optimization_suggestions,
            "call_analyses": [a.to_dict() for a in self.call_analyses],
        }


class ToolEfficiencyAnalyzer:
    """
    Analyzes tool usage patterns and computes efficiency scores.

    The efficiency formula:
        efficiency = useful_calls / total_calls
        where useful = succeeded AND not redundant

    Target: efficiency > 70%.

    Usage:
        analyzer = ToolEfficiencyAnalyzer()
        report = analyzer.analyze(agent_state)
        print(f"Tool efficiency: {report.efficiency_score:.0%}")
    """

    def __init__(self, available_tools: int = 4) -> None:
        self._available_tools = available_tools

    def analyze(
        self,
        state: dict,
        telemetry_events: list | None = None,
    ) -> ToolEfficiencyReport:
        """
        Run full tool efficiency analysis.

        Args:
            state: Completed agent state with tool_calls.
            telemetry_events: Optional timing data from collector.

        Returns:
            ToolEfficiencyReport with scores and recommendations.
        """
        report = ToolEfficiencyReport(available_tools=self._available_tools)
        tool_calls = state.get("tool_calls", [])

        if not tool_calls:
            report.optimization_suggestions.append(
                "No tool calls were made. The agent may not be using available tools."
            )
            return report

        # Build timing map from telemetry
        timing_map = self._build_timing_map(telemetry_events)

        # Analyze each call
        seen_keys = set()
        analyses = []
        unique_tools = set()
        final_answer = state.get("final_answer", "").lower()

        for i, tc in enumerate(tool_calls):
            tool_name = tc.tool_name if hasattr(tc, "tool_name") else str(tc)
            tool_input = tc.tool_input if hasattr(tc, "tool_input") else {}
            succeeded = tc.success if hasattr(tc, "success") else True
            output = tc.tool_output if hasattr(tc, "tool_output") else ""

            # Check redundancy
            call_key = f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"
            is_redundant = call_key in seen_keys
            seen_keys.add(call_key)

            # Check if output was used in final answer (heuristic)
            output_used = self._check_output_used(output, final_answer)

            # Determine usefulness
            is_useful = succeeded and not is_redundant

            # Get timing
            latency = timing_map.get(i, 0.0)

            analysis = ToolCallAnalysis(
                tool_name=tool_name,
                tool_input=tool_input,
                succeeded=succeeded,
                is_redundant=is_redundant,
                is_useful=is_useful,
                output_used_in_answer=output_used,
                latency_ms=latency,
                output_size=len(output) if output else 0,
                reason=self._get_call_reason(tool_name, is_redundant, succeeded, output_used),
            )
            analyses.append(analysis)
            unique_tools.add(tool_name)

        # Compute metrics
        report.call_analyses = analyses
        report.total_calls = len(analyses)
        report.successful_calls = sum(1 for a in analyses if a.succeeded)
        report.failed_calls = report.total_calls - report.successful_calls
        report.redundant_calls = sum(1 for a in analyses if a.is_redundant)
        report.useful_calls = sum(1 for a in analyses if a.is_useful)
        report.unique_tools_used = len(unique_tools)

        # Scores
        report.efficiency_score = (
            report.useful_calls / report.total_calls
            if report.total_calls > 0 else 0
        )
        report.success_rate = (
            report.successful_calls / report.total_calls
            if report.total_calls > 0 else 0
        )
        report.diversity_score = (
            report.unique_tools_used / report.available_tools
            if report.available_tools > 0 else 0
        )
        report.redundancy_rate = (
            report.redundant_calls / report.total_calls
            if report.total_calls > 0 else 0
        )

        # Timing
        latencies = [a.latency_ms for a in analyses if a.latency_ms > 0]
        if latencies:
            report.total_tool_time_ms = sum(latencies)
            report.avg_tool_time_ms = sum(latencies) / len(latencies)
            max_idx = latencies.index(max(latencies))
            report.slowest_call_ms = max(latencies)
            report.slowest_tool = analyses[max_idx].tool_name if max_idx < len(analyses) else ""

        # Optimization suggestions
        report.optimization_suggestions = self._generate_suggestions(report)

        logger.info(
            f"Tool efficiency: {report.efficiency_score:.0%} "
            f"({report.useful_calls}/{report.total_calls} useful, "
            f"{report.redundant_calls} redundant, {report.failed_calls} failed)"
        )

        return report

    def _build_timing_map(self, events: list | None) -> dict[int, float]:
        """Build a map of tool call index to latency from telemetry events."""
        if not events:
            return {}

        timing_map = {}
        tool_idx = 0
        for event in events:
            if hasattr(event, "event_type") and event.event_type == "tool_execution":
                timing_map[tool_idx] = event.duration_ms
                tool_idx += 1
        return timing_map

    def _check_output_used(self, output: str, answer: str) -> bool:
        """Heuristic check if tool output influenced the final answer."""
        if not output or not answer:
            return False

        # Extract key tokens from output
        output_lower = output.lower()
        # Look for numbers from output in answer
        numbers = set()
        import re
        for match in re.finditer(r'[0-9]+(?:\.[0-9]+)?', output):
            num = match.group()
            if len(num) > 2:  # Skip tiny numbers
                numbers.add(num)

        # If any significant number from output appears in answer
        for num in numbers:
            if num in answer:
                return True

        # Check for key word overlap
        output_words = set(output_lower.split())
        answer_words = set(answer.split())
        overlap = output_words & answer_words
        # Filter common words
        significant = {w for w in overlap if len(w) > 5}
        return len(significant) > 3

    def _get_call_reason(
        self,
        tool_name: str,
        is_redundant: bool,
        succeeded: bool,
        output_used: bool,
    ) -> str:
        """Generate a human-readable reason for the call's classification."""
        if is_redundant:
            return f"Redundant: {tool_name} called with same input previously"
        if not succeeded:
            return f"Failed: {tool_name} returned an error"
        if not output_used:
            return f"Unused: {tool_name} output not reflected in final answer"
        return f"Useful: {tool_name} contributed to analysis"

    def _generate_suggestions(self, report: ToolEfficiencyReport) -> list[str]:
        """Generate optimization suggestions based on analysis."""
        suggestions = []

        if report.redundant_calls > 0:
            suggestions.append(
                f"Eliminate {report.redundant_calls} redundant call(s). "
                f"Consider caching tool results within the reasoning loop."
            )

        if report.failed_calls > 0:
            suggestions.append(
                f"{report.failed_calls} tool call(s) failed. "
                f"Review error handling and input validation."
            )

        if report.diversity_score < 0.5:
            unused_count = report.available_tools - report.unique_tools_used
            suggestions.append(
                f"Only {report.unique_tools_used}/{report.available_tools} tools used. "
                f"Consider using {unused_count} more tool(s) for broader analysis."
            )

        if report.efficiency_score < 0.70:
            suggestions.append(
                f"Tool efficiency ({report.efficiency_score:.0%}) below 70% target. "
                f"Focus on reducing redundant calls and improving success rate."
            )

        if report.avg_tool_time_ms > 5000:
            suggestions.append(
                f"Average tool latency ({report.avg_tool_time_ms:.0f}ms) is high. "
                f"Consider parallel execution or caching."
            )

        if not suggestions:
            suggestions.append("Tool usage is efficient. No optimizations needed.")

        return suggestions
