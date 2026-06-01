"""
evaluation/benchmarks.py — Performance Benchmarking Suite
===========================================================

WHY THIS EXISTS:
    Metrics measure ONE run. Benchmarks measure the SYSTEM across
    multiple runs and configurations. They answer:

    - "Is ARA-1 getting faster or slower over time?"
    - "How does performance change with different LLM providers?"
    - "What's the p95 latency for a full analysis?"
    - "What's the minimum quality bar we can guarantee?"

WHAT PROBLEM IT SOLVES:
    Provides reproducible, comparable performance measurements
    for CI/CD gates, regression detection, and optimization.

HOW IT INTEGRATES:
    - Runs the full agent pipeline (main.run_agent).
    - Uses evaluation/evaluator.py for quality scoring.
    - Uses observability/collector.py for timing.
    - Outputs benchmark results for dashboards.

PRODUCTION TRADEOFFS:
    - Full benchmarks are SLOW (each run takes 30-120s).
    - Run nightly or on release, not on every commit.
    - Use cached results for quick regression checks.
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
class BenchmarkCase:
    """A single benchmark test case."""
    name: str
    query: str
    expected_ticker: str = ""
    min_quality_score: float = 0.5
    max_latency_seconds: float = 120
    description: str = ""


@dataclass
class BenchmarkResult:
    """Result of running one benchmark case."""
    case_name: str
    query: str
    passed: bool = False
    quality_score: float = 0.0
    latency_seconds: float = 0.0
    tool_efficiency: float = 0.0
    hallucination_rate: float = 0.0
    iterations_used: int = 0
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    pass_rate: float = 0.0
    avg_quality_score: float = 0.0
    avg_latency_seconds: float = 0.0
    avg_tool_efficiency: float = 0.0
    avg_hallucination_rate: float = 0.0
    p95_latency_seconds: float = 0.0
    results: list[BenchmarkResult] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "pass_rate": round(self.pass_rate, 4),
            "avg_quality_score": round(self.avg_quality_score, 4),
            "avg_latency_seconds": round(self.avg_latency_seconds, 2),
            "avg_tool_efficiency": round(self.avg_tool_efficiency, 4),
            "avg_hallucination_rate": round(self.avg_hallucination_rate, 4),
            "p95_latency_seconds": round(self.p95_latency_seconds, 2),
            "results": [r.to_dict() for r in self.results],
            "timestamp": self.timestamp,
        }


# ===================================================================
# Standard Benchmark Cases
# ===================================================================

STANDARD_BENCHMARKS = [
    BenchmarkCase(
        name="blue_chip_analysis",
        query="Analyze AAPL stock - provide a comprehensive investment analysis",
        expected_ticker="AAPL",
        min_quality_score=0.6,
        max_latency_seconds=120,
        description="Standard blue-chip stock analysis",
    ),
    BenchmarkCase(
        name="growth_stock_analysis",
        query="Analyze NVDA stock for investment potential",
        expected_ticker="NVDA",
        min_quality_score=0.5,
        max_latency_seconds=120,
        description="High-growth tech stock analysis",
    ),
    BenchmarkCase(
        name="value_stock_analysis",
        query="Provide investment analysis for JNJ stock",
        expected_ticker="JNJ",
        min_quality_score=0.5,
        max_latency_seconds=120,
        description="Value/defensive stock analysis",
    ),
]


class BenchmarkRunner:
    """
    Runs benchmark suites against the ARA-1 system.

    Usage:
        runner = BenchmarkRunner()
        report = runner.run_standard_benchmarks()
        print(f"Pass rate: {report.pass_rate:.0%}")
    """

    def __init__(
        self,
        output_dir: str | Path = "data/benchmarks",
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def run_single_benchmark(
        self,
        case: BenchmarkCase,
        run_agent_func: Any = None,
        evaluator: Any = None,
    ) -> BenchmarkResult:
        """
        Run a single benchmark case.

        Args:
            case: The benchmark test case.
            run_agent_func: Function to run the agent (default: main.run_agent).
            evaluator: SystemEvaluator instance.

        Returns:
            BenchmarkResult with pass/fail and scores.
        """
        result = BenchmarkResult(
            case_name=case.name,
            query=case.query,
        )

        logger.info(f"Running benchmark: {case.name}")

        try:
            # Run the agent
            start = time.time()

            if run_agent_func:
                final_state = run_agent_func(case.query)
            else:
                # Import dynamically to avoid circular imports
                from main import run_agent, initialize_phase2_systems
                phase2 = initialize_phase2_systems()
                final_state = run_agent(case.query, phase2)

            elapsed = time.time() - start
            result.latency_seconds = elapsed

            # Run evaluation
            if evaluator:
                eval_report = evaluator.evaluate(
                    agent_state=final_state,
                    elapsed_seconds=elapsed,
                )
                result.quality_score = eval_report.overall_score

                # Extract specific metrics
                for m in eval_report.metrics:
                    if m.name == "Tool Efficiency":
                        result.tool_efficiency = m.value
                    elif m.name == "Hallucination Rate":
                        result.hallucination_rate = m.value

            # Extract info from state
            result.iterations_used = final_state.get("iteration_count", 0)
            result.errors = final_state.get("errors", [])

            # Determine pass/fail
            result.passed = (
                result.quality_score >= case.min_quality_score
                and result.latency_seconds <= case.max_latency_seconds
            )

        except Exception as e:
            result.passed = False
            result.errors.append(f"Benchmark failed: {type(e).__name__}: {str(e)}")
            logger.error(f"Benchmark {case.name} failed: {e}")

        return result

    def run_benchmark_suite(
        self,
        cases: list[BenchmarkCase],
        run_agent_func: Any = None,
        evaluator: Any = None,
    ) -> BenchmarkReport:
        """
        Run a suite of benchmark cases.

        Args:
            cases: List of benchmark cases to run.
            run_agent_func: Function to run the agent.
            evaluator: SystemEvaluator instance.

        Returns:
            BenchmarkReport with aggregated results.
        """
        report = BenchmarkReport()
        results = []

        for case in cases:
            result = self.run_single_benchmark(case, run_agent_func, evaluator)
            results.append(result)

        report.results = results
        report.total_cases = len(results)
        report.passed_cases = sum(1 for r in results if r.passed)
        report.failed_cases = report.total_cases - report.passed_cases
        report.pass_rate = report.passed_cases / max(report.total_cases, 1)

        # Averages
        if results:
            report.avg_quality_score = sum(r.quality_score for r in results) / len(results)
            report.avg_latency_seconds = sum(r.latency_seconds for r in results) / len(results)
            report.avg_tool_efficiency = sum(r.tool_efficiency for r in results) / len(results)
            report.avg_hallucination_rate = sum(r.hallucination_rate for r in results) / len(results)

            # P95 latency
            latencies = sorted(r.latency_seconds for r in results)
            p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
            report.p95_latency_seconds = latencies[p95_idx]

        return report

    def run_standard_benchmarks(
        self,
        run_agent_func: Any = None,
        evaluator: Any = None,
    ) -> BenchmarkReport:
        """Run the standard benchmark suite."""
        return self.run_benchmark_suite(
            STANDARD_BENCHMARKS,
            run_agent_func,
            evaluator,
        )

    def export_report(
        self,
        report: BenchmarkReport,
        filename: str = "",
    ) -> str:
        """Export benchmark report to JSON."""
        if not filename:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{ts}.json"

        output_path = self._output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        logger.info(f"Benchmark report exported to {output_path}")
        return str(output_path)
