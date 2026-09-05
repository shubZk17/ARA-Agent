"""
evaluation/evaluator.py — Master Evaluation Orchestrator
==========================================================

WHY THIS EXISTS:
    Individual metrics (metrics.py) and detectors (hallucination_detector.py,
    tool_efficiency.py) each compute one dimension. The evaluator:

    1. ORCHESTRATES all metric computations in correct order.
    2. AGGREGATES results into a single EvaluationReport.
    3. COMPUTES the overall system score.
    4. GENERATES human-readable evaluation summaries.
    5. EXPORTS results for dashboards and compliance.

WHAT PROBLEM IT SOLVES:
    Answers "How good was this agent run?" with a single number
    (0-100%) plus detailed breakdowns by category.

HOW IT INTEGRATES:
    - Reads from: agent state, hallucination report, tool efficiency report.
    - Reads from: observability/collector.py (telemetry data).
    - Uses: evaluation/metrics.py (all 22 metric functions).
    - Outputs to: reports, dashboards, JSON exports.
    - Called by: main.py after agent + synthesis complete.

DESIGN DECISIONS:
    - Metrics are run AFTER the agent completes (not during).
    - This avoids coupling evaluation with execution.
    - Evaluation should NEVER affect agent behavior.
    - Exception: failure injection actively tests behavior (separate module).

SCALABILITY:
    - Add custom metrics by extending ALL_METRIC_FUNCTIONS.
    - Add historical comparison by reading prior evaluation reports.
    - Add regression detection (score dropped from last run).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional

from analysis.evaluation.metrics import (
    EvaluationReport,
    MetricResult,
    MetricStatus,
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
)
from analysis.evaluation.hallucination_detector import HallucinationDetector, HallucinationReport
from analysis.evaluation.tool_efficiency import ToolEfficiencyAnalyzer, ToolEfficiencyReport
from config.logging import get_logger

logger = get_logger(__name__)


class SystemEvaluator:
    """
    Master evaluation orchestrator for ARA-1.

    Computes all 22+ metrics, runs hallucination detection,
    analyzes tool efficiency, and produces a comprehensive report.

    Usage:
        evaluator = SystemEvaluator()
        eval_report = evaluator.evaluate(
            agent_state=final_state,
            elapsed_seconds=elapsed,
            synthesis_report=report.to_dict() if report else None,
            report_paths=paths,
        )
        print(f"Overall: {eval_report.overall_score:.0%}")
    """

    def __init__(self) -> None:
        self._hallucination_detector = HallucinationDetector()
        self._tool_analyzer = ToolEfficiencyAnalyzer()

    def evaluate(
        self,
        agent_state: dict,
        elapsed_seconds: float = 0,
        synthesis_report: dict | None = None,
        report_paths: dict | None = None,
        telemetry_events: list | None = None,
        failure_injection_results: list | None = None,
    ) -> EvaluationReport:
        """
        Run the complete evaluation pipeline.

        Args:
            agent_state: Final state from graph.invoke().
            elapsed_seconds: Total wall-clock time.
            synthesis_report: Phase 3 synthesis report (dict form).
            report_paths: Dict of format → file path for generated reports.
            telemetry_events: Raw telemetry events from collector.
            failure_injection_results: Results from deliberate failure tests.

        Returns:
            EvaluationReport with all metrics and overall score.
        """
        start = time.time()

        # Extract context
        ticker = self._extract_ticker(agent_state)
        query = agent_state.get("query", "")

        logger.info(f"Starting evaluation for: {ticker or query[:50]}")

        # --- Phase A: Run sub-analyzers ---
        hallucination_report = self._hallucination_detector.analyze(agent_state)
        tool_report = self._tool_analyzer.analyze(agent_state, telemetry_events)

        # --- Phase B: Compute all 22 metrics ---
        metrics: list[MetricResult] = []

        # M1: Hallucination Rate
        metrics.append(compute_hallucination_rate(
            agent_state,
            [f.to_dict() for f in hallucination_report.flags],
        ))

        # M2: Tool Efficiency
        metrics.append(compute_tool_efficiency(agent_state))

        # M3: Tool Success Rate
        metrics.append(compute_tool_success_rate(agent_state))

        # M4: Iteration Efficiency
        metrics.append(compute_iteration_efficiency(agent_state))

        # M5: Reasoning Depth
        metrics.append(compute_reasoning_depth(agent_state))

        # M6: Source Diversity
        metrics.append(compute_source_diversity(agent_state))

        # M7: Retrieval Usage
        metrics.append(compute_retrieval_usage(agent_state))

        # M8: Conflict Detection
        metrics.append(compute_conflict_detection(agent_state))

        # M9: Error Recovery Rate
        metrics.append(compute_error_recovery_rate(agent_state))

        # M10: Completion Status
        metrics.append(compute_completion_status(agent_state))

        # M11: Answer Comprehensiveness
        metrics.append(compute_answer_length(agent_state))

        # M12: Evidence Grounding
        metrics.append(compute_evidence_grounding(agent_state))

        # M13: Total Latency
        metrics.append(compute_latency(agent_state, elapsed_seconds))

        # M14: Average Iteration Time
        metrics.append(compute_avg_iteration_time(agent_state, elapsed_seconds))

        # M15: Redundant Tool Calls
        metrics.append(compute_redundant_calls(agent_state))

        # M16: Episodic Memory Usage
        metrics.append(compute_episodic_memory_usage(agent_state))

        # M17: Parse Error Rate
        metrics.append(compute_parse_error_rate(agent_state))

        # M18: Financial Metrics Coverage
        metrics.append(compute_financial_metrics_coverage(synthesis_report))

        # M19: Confidence Calibration
        metrics.append(compute_confidence_calibration(synthesis_report))

        # M20: Misalignment Detection
        metrics.append(compute_misalignment_detection(synthesis_report))

        # M21: Risk Assessment Quality
        metrics.append(compute_risk_assessment_quality(synthesis_report))

        # M22: Report Generation
        metrics.append(compute_report_generation(synthesis_report, report_paths))

        # --- Phase C: Compute aggregate scores ---
        eval_time = (time.time() - start) * 1000

        passed = sum(1 for m in metrics if m.status == MetricStatus.PASS)
        warned = sum(1 for m in metrics if m.status == MetricStatus.WARN)
        failed = sum(1 for m in metrics if m.status == MetricStatus.FAIL)
        total = len(metrics)
        pass_rate = passed / total if total > 0 else 0

        # Weighted overall score (categories have different importance)
        overall_score = self._compute_weighted_score(metrics)

        report = EvaluationReport(
            run_id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            ticker=ticker,
            query=query,
            metrics=metrics,
            overall_score=overall_score,
            pass_rate=pass_rate,
            total_metrics=total,
            passed_metrics=passed,
            warned_metrics=warned,
            failed_metrics=failed,
            execution_time_ms=eval_time,
        )

        logger.info(
            f"Evaluation complete: {overall_score:.0%} overall, "
            f"{passed}/{total} passed, {warned} warned, {failed} failed"
        )

        return report

    def _compute_weighted_score(self, metrics: list[MetricResult]) -> float:
        """
        Compute a weighted overall score.

        Category weights reflect business importance:
        - Reliability (hallucination, recovery): 30%
        - Synthesis (financial coverage, reports): 25%
        - Tools (efficiency, success rate): 20%
        - Reasoning (depth, completion): 15%
        - Performance (latency): 10%
        """
        category_weights = {
            "reliability": 0.30,
            "synthesis": 0.25,
            "tools": 0.20,
            "reasoning": 0.15,
            "retrieval": 0.10,
            "performance": 0.05,
        }

        category_scores: dict[str, list[float]] = {}
        for m in metrics:
            cat = m.category.lower() if isinstance(m.category, str) else m.category.value
            if cat not in category_scores:
                category_scores[cat] = []

            # Normalize: pass=1.0, warn=0.5, fail=0.0
            if m.status == MetricStatus.PASS:
                category_scores[cat].append(1.0)
            elif m.status == MetricStatus.WARN:
                category_scores[cat].append(0.5)
            else:
                category_scores[cat].append(0.0)

        # Weighted average
        total_score = 0.0
        total_weight = 0.0
        for cat, weight in category_weights.items():
            if cat in category_scores:
                avg = sum(category_scores[cat]) / len(category_scores[cat])
                total_score += avg * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0

    def _extract_ticker(self, state: dict) -> str:
        """Extract ticker from agent state."""
        for tc in state.get("tool_calls", []):
            if hasattr(tc, "tool_input"):
                ticker = tc.tool_input.get("ticker", "")
                if ticker:
                    return ticker.upper()
        return ""

    def generate_summary(self, report: EvaluationReport) -> str:
        """Generate a human-readable evaluation summary."""
        lines = [
            "=" * 60,
            "ARA-1 EVALUATION REPORT",
            "=" * 60,
            f"Run ID:    {report.run_id}",
            f"Ticker:    {report.ticker}",
            f"Query:     {report.query[:60]}",
            f"Timestamp: {report.timestamp}",
            "",
            f"OVERALL SCORE: {report.overall_score:.0%}",
            f"Pass Rate:     {report.pass_rate:.0%} ({report.passed_metrics}/{report.total_metrics})",
            f"Warnings:      {report.warned_metrics}",
            f"Failures:      {report.failed_metrics}",
            "",
            "-" * 60,
            "METRIC DETAILS:",
            "-" * 60,
        ]

        # Group by category
        by_category: dict[str, list[MetricResult]] = {}
        for m in report.metrics:
            cat = m.category if isinstance(m.category, str) else m.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(m)

        status_icons = {
            MetricStatus.PASS: "[PASS]",
            MetricStatus.WARN: "[WARN]",
            MetricStatus.FAIL: "[FAIL]",
            "pass": "[PASS]",
            "warn": "[WARN]",
            "fail": "[FAIL]",
        }

        for category, metrics in sorted(by_category.items()):
            lines.append(f"\n  {category.upper()}:")
            for m in metrics:
                icon = status_icons.get(m.status, "[????]")
                lines.append(
                    f"    {icon} {m.name}: {m.formatted}"
                )
                if m.details:
                    lines.append(f"           {m.details}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def export_to_json(
        self,
        report: EvaluationReport,
        output_path: str | Path,
    ) -> str:
        """Export evaluation report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        logger.info(f"Evaluation report exported to {output_path}")
        return str(output_path)
