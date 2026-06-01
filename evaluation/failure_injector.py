"""
evaluation/failure_injector.py — Deliberate Failure Injection System
======================================================================

WHY THIS EXISTS:
    You can't trust a system that's only been tested in ideal conditions.
    Chaos engineering principles say: BREAK IT ON PURPOSE to learn how
    it breaks in production.

    This module injects 7 DELIBERATE, CONTROLLED failures:
    1. Malformed JSON — tests parser resilience.
    2. Contradictory Metrics — tests conflict detection.
    3. Retrieval Corruption — tests evidence validation.
    4. Stale Evidence — tests temporal reasoning.
    5. API Timeout — tests retry/fallback system.
    6. Vector DB Failure — tests degraded mode.
    7. Hallucinated Retrieval Chunks — tests hallucination detection.

WHAT PROBLEM IT SOLVES:
    Proves that ARA-1 can SURVIVE real-world failure modes.
    The final report MUST document all 7 failures, their impact,
    and recovery behavior.

HOW IT INTEGRATES:
    - Wraps tool execution with failure injection.
    - Wraps retrieval with corrupted data.
    - Wraps LLM responses with malformed output.
    - Records everything to observability collector.
    - Results are included in the evaluation report.

IMPORTANT:
    Failure injection is CONTROLLED and REVERSIBLE.
    It is activated explicitly, never in production mode.
    Every injected failure is logged and documented.

DESIGN DECISIONS:
    - Each failure is a self-contained scenario with:
      - injection function
      - expected behavior
      - recovery criteria
      - documentation template
    - Failures are injected DURING execution (not simulated after).
    - The agent must handle them WITHOUT special knowledge.
"""

from __future__ import annotations

import json
import time
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FailureScenario:
    """Definition of a deliberate failure scenario."""
    id: int
    name: str
    description: str
    failure_type: str
    expected_behavior: str
    recovery_criteria: str
    severity: str  # "low", "medium", "high", "critical"


@dataclass
class FailureResult:
    """Result of executing a failure scenario."""
    scenario: FailureScenario
    injected: bool = False
    detected: bool = False
    recovered: bool = False
    recovery_time_ms: float = 0.0
    impact: str = ""
    agent_behavior: str = ""
    lessons_learned: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario.id,
            "scenario_name": self.scenario.name,
            "description": self.scenario.description,
            "failure_type": self.scenario.failure_type,
            "severity": self.scenario.severity,
            "injected": self.injected,
            "detected": self.detected,
            "recovered": self.recovered,
            "recovery_time_ms": round(self.recovery_time_ms, 2),
            "impact": self.impact,
            "agent_behavior": self.agent_behavior,
            "expected_behavior": self.scenario.expected_behavior,
            "recovery_criteria": self.scenario.recovery_criteria,
            "lessons_learned": self.lessons_learned,
            "timestamp": self.timestamp,
            "details": self.details,
        }


@dataclass
class FailureInjectionReport:
    """Complete report of all failure injection tests."""
    total_scenarios: int = 7
    injected: int = 0
    detected: int = 0
    recovered: int = 0
    recovery_rate: float = 0.0
    results: list[FailureResult] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "total_scenarios": self.total_scenarios,
            "injected": self.injected,
            "detected": self.detected,
            "recovered": self.recovered,
            "recovery_rate": round(self.recovery_rate, 4),
            "results": [r.to_dict() for r in self.results],
            "timestamp": self.timestamp,
        }

    def to_markdown(self) -> str:
        """Generate markdown documentation for the final report."""
        lines = [
            "## Deliberate Failure Injection Report",
            "",
            f"**Total Scenarios:** {self.total_scenarios}",
            f"**Injected:** {self.injected}",
            f"**Detected:** {self.detected}",
            f"**Recovered:** {self.recovered}",
            f"**Recovery Rate:** {self.recovery_rate:.0%}",
            "",
        ]

        for result in self.results:
            status = "✅ RECOVERED" if result.recovered else "❌ FAILED"
            lines.extend([
                f"### Failure {result.scenario.id}: {result.scenario.name}",
                f"**Status:** {status}",
                f"**Type:** {result.scenario.failure_type}",
                f"**Severity:** {result.scenario.severity}",
                f"**Description:** {result.scenario.description}",
                f"**Impact:** {result.impact}",
                f"**Agent Behavior:** {result.agent_behavior}",
                f"**Recovery Time:** {result.recovery_time_ms:.0f}ms",
                f"**Lessons Learned:** {result.lessons_learned}",
                "",
            ])

        return "\n".join(lines)


# ===================================================================
# The 7 Failure Scenarios
# ===================================================================

FAILURE_SCENARIOS = [
    FailureScenario(
        id=1,
        name="Malformed JSON Response",
        description="LLM returns invalid JSON that can't be parsed as a ReAct response",
        failure_type="parse_error",
        expected_behavior="Parser detects invalid JSON, returns error, agent retries",
        recovery_criteria="Agent continues reasoning after encountering parse error",
        severity="medium",
    ),
    FailureScenario(
        id=2,
        name="Contradictory Financial Metrics",
        description="Tool returns P/E ratio = 15 while another observation says P/E = 45",
        failure_type="data_conflict",
        expected_behavior="Conflict resolver detects contradiction, flags in report",
        recovery_criteria="Final report acknowledges conflicting data",
        severity="high",
    ),
    FailureScenario(
        id=3,
        name="Corrupted Retrieval Results",
        description="Vector store returns garbled/irrelevant chunks mixed with valid data",
        failure_type="retrieval_corruption",
        expected_behavior="Reliability scorer filters low-quality evidence",
        recovery_criteria="Agent doesn't use corrupted evidence in final answer",
        severity="high",
    ),
    FailureScenario(
        id=4,
        name="Stale Evidence Injection",
        description="Evidence from 2 years ago presented as current data",
        failure_type="temporal_staleness",
        expected_behavior="System flags temporal mismatch or uses with caveat",
        recovery_criteria="Output notes data staleness or uses current tool data",
        severity="medium",
    ),
    FailureScenario(
        id=5,
        name="API Timeout Simulation",
        description="Tool execution simulates a network timeout after 5 seconds",
        failure_type="api_timeout",
        expected_behavior="Retry handler retries, then continues with available data",
        recovery_criteria="Agent completes analysis despite tool timeout",
        severity="high",
    ),
    FailureScenario(
        id=6,
        name="Vector Database Failure",
        description="Vector store raises connection error on query",
        failure_type="infrastructure_failure",
        expected_behavior="Agent degrades to Phase 1 mode (tool-only, no retrieval)",
        recovery_criteria="Agent produces valid output without vector memory",
        severity="critical",
    ),
    FailureScenario(
        id=7,
        name="Hallucinated Retrieval Chunks",
        description="Vector store returns fabricated financial data that looks real",
        failure_type="hallucinated_evidence",
        expected_behavior="Hallucination detector catches inconsistencies",
        recovery_criteria="Fabricated data doesn't appear in final report",
        severity="critical",
    ),
]


class FailureInjector:
    """
    Injects controlled failures into the ARA-1 pipeline.

    Usage:
        injector = FailureInjector()
        
        # Run all scenarios
        report = injector.run_all_scenarios(agent_state)
        
        # Or inject specific failures during execution
        if injector.should_inject(iteration=3, scenario_id=1):
            response = injector.inject_malformed_json(response)
    """

    def __init__(self, collector: Any = None) -> None:
        self._collector = collector
        self._active_scenarios: set[int] = set()
        self._results: list[FailureResult] = []

    def activate_scenario(self, scenario_id: int) -> None:
        """Activate a specific failure scenario."""
        self._active_scenarios.add(scenario_id)
        logger.info(f"Activated failure scenario {scenario_id}")

    def activate_all(self) -> None:
        """Activate all failure scenarios."""
        self._active_scenarios = {s.id for s in FAILURE_SCENARIOS}
        logger.info("Activated ALL failure scenarios")

    def is_active(self, scenario_id: int) -> bool:
        """Check if a scenario is active."""
        return scenario_id in self._active_scenarios

    # -------------------------------------------------------------------
    # Failure 1: Malformed JSON
    # -------------------------------------------------------------------
    def inject_malformed_json(self, valid_response: str) -> str:
        """
        Corrupt a valid LLM response into malformed JSON.

        Simulates: LLM returns partial/broken JSON.
        """
        scenario = FAILURE_SCENARIOS[0]
        start = time.time()

        # Intentionally break the JSON
        corrupted = valid_response[:len(valid_response)//2] + '{"broken": true, unclosed'

        result = FailureResult(
            scenario=scenario,
            injected=True,
            impact="LLM response cannot be parsed as ReAct JSON",
            agent_behavior="Parser returns error, agent should retry",
            details={"original_length": len(valid_response), "corrupted_length": len(corrupted)},
        )

        self._record_injection(scenario, result, start)
        return corrupted

    # -------------------------------------------------------------------
    # Failure 2: Contradictory Metrics
    # -------------------------------------------------------------------
    def inject_contradictory_metrics(
        self,
        tool_output: str,
        tool_name: str,
    ) -> str:
        """
        Inject contradictory financial metrics into tool output.

        Simulates: Two data sources disagree on a key metric.
        """
        scenario = FAILURE_SCENARIOS[1]
        start = time.time()

        contradiction = (
            "\n\n[CONFLICTING DATA SOURCE] "
            "Alternative data source reports: P/E Ratio = 999.99, "
            "Revenue Growth = -50.0%, Market Cap = $1.00. "
            "This conflicts with the primary data above."
        )

        result = FailureResult(
            scenario=scenario,
            injected=True,
            impact="Tool output contains contradictory financial metrics",
            agent_behavior="Conflict resolver should detect and flag",
            details={"tool_name": tool_name},
        )

        self._record_injection(scenario, result, start)
        return tool_output + contradiction

    # -------------------------------------------------------------------
    # Failure 3: Corrupted Retrieval
    # -------------------------------------------------------------------
    def inject_corrupted_retrieval(
        self,
        valid_chunks: list[dict],
    ) -> list[dict]:
        """
        Mix corrupted chunks into retrieval results.

        Simulates: Vector store returns garbage alongside valid data.
        """
        scenario = FAILURE_SCENARIOS[2]
        start = time.time()

        corrupted_chunks = [
            {
                "content": "CORRUPTED: ☠️ §§§ binary garbage 0x4F2A %%%",
                "source": "corrupted_source",
                "score": 0.1,
                "metadata": {"corrupted": True},
            },
            {
                "content": "ERROR: Database connection reset. Partial data follows: NULL NULL NULL",
                "source": "error_chunk",
                "score": 0.05,
                "metadata": {"corrupted": True},
            },
        ]

        mixed = valid_chunks + corrupted_chunks

        result = FailureResult(
            scenario=scenario,
            injected=True,
            impact=f"2 corrupted chunks mixed with {len(valid_chunks)} valid chunks",
            agent_behavior="Reliability scorer should filter low-quality chunks",
            details={"valid_count": len(valid_chunks), "corrupted_count": 2},
        )

        self._record_injection(scenario, result, start)
        return mixed

    # -------------------------------------------------------------------
    # Failure 4: Stale Evidence
    # -------------------------------------------------------------------
    def inject_stale_evidence(self, tool_output: str) -> str:
        """
        Add outdated evidence to tool output.

        Simulates: Evidence from years ago mixed with current data.
        """
        scenario = FAILURE_SCENARIOS[3]
        start = time.time()

        stale_data = (
            "\n\n[HISTORICAL DATA - January 2022] "
            "Stock price: $45.00. P/E Ratio: 12.5. "
            "Revenue: $2.1B. Market sentiment: Bearish due to COVID recovery concerns. "
            "Note: This data is from 2 years ago and may not reflect current conditions."
        )

        result = FailureResult(
            scenario=scenario,
            injected=True,
            impact="Historical data mixed with current analysis",
            agent_behavior="Agent should note temporal mismatch or prioritize current data",
            details={"stale_date": "2022-01"},
        )

        self._record_injection(scenario, result, start)
        return tool_output + stale_data

    # -------------------------------------------------------------------
    # Failure 5: API Timeout
    # -------------------------------------------------------------------
    def inject_api_timeout(self, delay_seconds: float = 0.5) -> None:
        """
        Simulate an API timeout.

        In real injection, this is called BEFORE a tool execution
        to simulate the tool hanging.

        Note: Using a shorter delay for testing (0.5s instead of 5s).
        """
        scenario = FAILURE_SCENARIOS[4]
        start = time.time()

        # Simulate delay (short for testing)
        time.sleep(delay_seconds)

        result = FailureResult(
            scenario=scenario,
            injected=True,
            impact=f"Tool execution delayed by {delay_seconds}s",
            agent_behavior="Retry handler should catch timeout and retry",
            details={"delay_seconds": delay_seconds},
        )

        self._record_injection(scenario, result, start)
        raise TimeoutError(f"Simulated API timeout after {delay_seconds}s")

    # -------------------------------------------------------------------
    # Failure 6: Vector DB Failure
    # -------------------------------------------------------------------
    def inject_vector_db_failure(self) -> None:
        """
        Simulate a vector database connection failure.

        Simulates: ChromaDB is unreachable.
        """
        scenario = FAILURE_SCENARIOS[5]
        start = time.time()

        result = FailureResult(
            scenario=scenario,
            injected=True,
            impact="Vector memory unavailable, retrieval disabled",
            agent_behavior="Agent degrades to Phase 1 mode (no retrieval context)",
            details={"error": "ConnectionError: ChromaDB unreachable"},
        )

        self._record_injection(scenario, result, start)
        raise ConnectionError("Simulated vector DB failure: ChromaDB unreachable")

    # -------------------------------------------------------------------
    # Failure 7: Hallucinated Retrieval Chunks
    # -------------------------------------------------------------------
    def inject_hallucinated_chunks(self) -> list[dict]:
        """
        Return fabricated but realistic-looking financial data.

        Simulates: Vector store returns hallucinated evidence.
        """
        scenario = FAILURE_SCENARIOS[6]
        start = time.time()

        hallucinated = [
            {
                "content": (
                    "NVIDIA reported Q4 2024 earnings of $0.15 per share, "
                    "missing analyst estimates by 85%. Revenue declined 40% "
                    "year-over-year to $3.2 billion. CEO announced plans to "
                    "exit the AI chip market entirely."
                ),
                "source": "fabricated_earnings_report",
                "score": 0.92,  # High score to make it convincing
                "metadata": {
                    "hallucinated": True,
                    "ticker": "NVDA",
                    "date": "2024-12-01",
                },
            },
        ]

        result = FailureResult(
            scenario=scenario,
            injected=True,
            impact="Fabricated financial data with high confidence score",
            agent_behavior="Hallucination detector should catch impossible claims",
            details={"fabricated_claims": ["$0.15 EPS", "-40% revenue", "exit AI market"]},
        )

        self._record_injection(scenario, result, start)
        return hallucinated

    # -------------------------------------------------------------------
    # Scenario Execution
    # -------------------------------------------------------------------
    def run_all_scenarios(self, agent_state: dict) -> FailureInjectionReport:
        """
        Execute all failure scenarios and assess recovery.

        This is the POST-RUN analysis that checks whether
        the agent properly handled injected failures.

        Args:
            agent_state: Final agent state after execution with failures.

        Returns:
            FailureInjectionReport documenting all 7 failures.
        """
        report = FailureInjectionReport()

        # Assess each scenario based on agent state
        for result in self._results:
            # Check detection
            result.detected = self._assess_detection(result, agent_state)

            # Check recovery
            result.recovered = self._assess_recovery(result, agent_state)

            # Generate lessons
            result.lessons_learned = self._generate_lessons(result)

        report.results = self._results
        report.injected = sum(1 for r in self._results if r.injected)
        report.detected = sum(1 for r in self._results if r.detected)
        report.recovered = sum(1 for r in self._results if r.recovered)
        report.recovery_rate = (
            report.recovered / max(report.injected, 1)
        )

        logger.info(
            f"Failure injection report: {report.recovered}/{report.injected} recovered "
            f"({report.recovery_rate:.0%})"
        )

        return report

    def _assess_detection(self, result: FailureResult, state: dict) -> bool:
        """Check if the failure was detected by the system."""
        errors = state.get("errors", [])
        final_answer = state.get("final_answer", "")

        scenario_type = result.scenario.failure_type

        if scenario_type == "parse_error":
            return any("parse" in e.lower() or "json" in e.lower() for e in errors)

        elif scenario_type == "data_conflict":
            conflicts = state.get("conflict_reports", [])
            return len(conflicts) > 0 or "conflict" in final_answer.lower()

        elif scenario_type == "retrieval_corruption":
            return any("corrupt" in e.lower() or "invalid" in e.lower() for e in errors)

        elif scenario_type == "temporal_staleness":
            return "historical" in final_answer.lower() or "stale" in final_answer.lower()

        elif scenario_type == "api_timeout":
            return any("timeout" in e.lower() or "retry" in e.lower() for e in errors)

        elif scenario_type == "infrastructure_failure":
            return any("unavailable" in e.lower() or "connection" in e.lower() for e in errors)

        elif scenario_type == "hallucinated_evidence":
            return "exit" not in final_answer.lower()  # Didn't use the fake "exit AI" claim

        return False

    def _assess_recovery(self, result: FailureResult, state: dict) -> bool:
        """Check if the system recovered from the failure."""
        status = str(state.get("status", ""))
        has_answer = bool(state.get("final_answer"))

        # Basic recovery = agent completed with a final answer
        if status in ("completed", "max_iterations_reached") and has_answer:
            return True

        return False

    def _generate_lessons(self, result: FailureResult) -> str:
        """Generate lessons learned from each failure scenario."""
        lessons = {
            "parse_error": (
                "JSON parse errors are common with LLMs. The parser should "
                "handle partial/malformed JSON gracefully and give the LLM "
                "another chance to respond correctly."
            ),
            "data_conflict": (
                "Contradictory data sources are inevitable in financial analysis. "
                "The conflict resolver should surface contradictions rather than "
                "silently choosing one source."
            ),
            "retrieval_corruption": (
                "Vector retrieval can return low-quality chunks, especially "
                "with noisy embeddings. Reliability scoring is essential "
                "for filtering evidence quality."
            ),
            "temporal_staleness": (
                "Historical data mixed with current analysis can mislead "
                "investors. Temporal awareness is critical for financial systems."
            ),
            "api_timeout": (
                "API timeouts are the most common production failure. "
                "Retry with backoff + fallback providers ensures continuity."
            ),
            "infrastructure_failure": (
                "Infrastructure failures should trigger graceful degradation, "
                "not total system failure. Phase 1 capabilities should always work."
            ),
            "hallucinated_evidence": (
                "Fabricated evidence with high confidence scores is the most "
                "dangerous failure mode. Cross-validation against tool outputs "
                "is the primary defense."
            ),
        }

        return lessons.get(
            result.scenario.failure_type,
            "System should handle this failure gracefully."
        )

    def _record_injection(
        self,
        scenario: FailureScenario,
        result: FailureResult,
        start_time: float,
    ) -> None:
        """Record the injection event."""
        result.recovery_time_ms = (time.time() - start_time) * 1000
        self._results.append(result)

        if self._collector:
            from observability.collector import EventType
            self._collector.record_event(
                EventType.FAILURE_INJECTION,
                f"failure_{scenario.id}_{scenario.failure_type}",
                success=True,
                metadata={
                    "scenario_id": scenario.id,
                    "scenario_name": scenario.name,
                    "failure_type": scenario.failure_type,
                },
            )

        logger.warning(
            f"[FAILURE INJECTION] Scenario {scenario.id}: {scenario.name} — INJECTED"
        )

    def get_results(self) -> list[FailureResult]:
        """Get all failure injection results."""
        return self._results

    def reset(self) -> None:
        """Reset for a new test run."""
        self._active_scenarios.clear()
        self._results.clear()
