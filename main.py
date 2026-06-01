"""
main.py — ARA-1 Entry Point
==============================

This is the assembly point that wires all components together.

STARTUP SEQUENCE:
    1. Validate configuration (API keys, settings)
    2. Initialize tool registry and register all tools
    3. Initialize Phase 2 systems (vector store, embeddings, retrieval, memory)
    4. Initialize Phase 3 systems (synthesis engine, report generator)
    5. Initialize Phase 4 systems (observability, evaluation, checkpointing)
    6. Build the LangGraph agent graph
    7. Load episodic context from prior runs
    8. Accept user query
    9. Create initial state (with episodic context)
    10. Run the graph
    11. Save episode to episodic memory
    12. Run Phase 3 synthesis
    13. Run Phase 4 evaluation
    14. Display results

DESIGN PRINCIPLE:
    main.py is a THIN orchestration layer. It should contain:
    - component initialization,
    - dependency wiring,
    - CLI interaction,
    - result display.

    It should NOT contain:
    - business logic,
    - tool implementations,
    - parsing logic,
    - prompt templates.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown

from agent.graph import build_graph
from agent.state import AgentStatus, create_initial_state
from config.settings import settings
from tools.company_info import CompanyInfoTool
from tools.financial_metrics import FinancialMetricsTool
from tools.news import NewsRetrievalTool
from tools.registry import ToolRegistry
from tools.stock_price import StockPriceTool
from utils.logger import get_logger

logger = get_logger(__name__)
console = Console(force_terminal=True)


# ===================================================================
# Startup Validation
# ===================================================================

def validate_configuration() -> bool:
    """
    Validate all configuration before starting the agent.

    Returns True if valid, False with error display if not.
    """
    errors = settings.validate()
    if errors:
        console.print("\n[bold red]Configuration Errors:[/bold red]")
        for error in errors:
            console.print(f"  [red]x[/red] {error}")
        console.print(
            "\n[yellow]Fix these in your .env file. "
            "See .env.example for reference.[/yellow]\n"
        )
        return False
    return True


# ===================================================================
# Tool Registry Setup
# ===================================================================

def create_tool_registry() -> ToolRegistry:
    """
    Create and populate the tool registry with all Phase 1 tools.

    Adding a new tool to ARA-1 is as simple as:
    1. Create a new file in tools/ inheriting from BaseTool
    2. Add one line here: registry.register(MyNewTool())

    That's it. No other code changes needed.
    """
    registry = ToolRegistry()

    registry.register(StockPriceTool())
    registry.register(CompanyInfoTool())
    registry.register(FinancialMetricsTool())
    registry.register(NewsRetrievalTool())

    logger.info(f"Registered {len(registry)} tools: {registry.list_tools()}")
    return registry


# ===================================================================
# Phase 2: System Initialization
# ===================================================================

def initialize_phase2_systems():
    """
    Initialize Phase 2 subsystems: vector store, embeddings,
    ingestion pipeline, retriever, reliability scorer, conflict resolver,
    and episodic memory.

    Returns a dict of initialized components, or empty dict if
    Phase 2 initialization fails (agent falls back to Phase 1 behavior).

    WHY a single initialization function?
    - Dependencies between Phase 2 components are complex.
    - Initialization order matters (vector store before retriever).
    - If ANY component fails, the agent should still work with Phase 1.
    """
    components = {}

    try:
        # 1. Vector Store
        from retrieval.vector_store import create_vector_store
        vector_store = create_vector_store(
            backend=settings.vector_backend,
            collection_name=settings.vector_collection_name,
            persist_directory=str(settings.chroma_persist_dir),
        )
        components["vector_store"] = vector_store

        # 2. Embedding Pipeline
        from retrieval.embeddings import EmbeddingPipeline
        embedding_pipeline = EmbeddingPipeline(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
        )
        components["embedding_pipeline"] = embedding_pipeline

        # 3. Ingestion Pipeline
        from ingestion.pipeline import IngestionPipeline
        ingestion_pipeline = IngestionPipeline(
            vector_store=vector_store,
            embedding_pipeline=embedding_pipeline,
        )
        components["ingestion_pipeline"] = ingestion_pipeline

        # 4. Reliability Scorer
        from reliability.scorer import ReliabilityScorer
        reliability_scorer = ReliabilityScorer()
        components["reliability_scorer"] = reliability_scorer

        # 5. Conflict Resolver
        from reliability.conflict_resolver import ConflictResolver
        conflict_resolver = ConflictResolver()
        components["conflict_resolver"] = conflict_resolver

        # 6. Semantic Retriever
        from retrieval.retriever import SemanticRetriever
        semantic_retriever = SemanticRetriever(
            vector_store=vector_store,
            embedding_pipeline=embedding_pipeline,
            reliability_scorer=reliability_scorer,
            conflict_resolver=conflict_resolver,
        )
        components["semantic_retriever"] = semantic_retriever

        # 7. Episodic Memory
        from memory.episodic import EpisodicMemory
        episodic_memory = EpisodicMemory(
            storage_dir=str(settings.episodic_memory_dir),
        )
        components["episodic_memory"] = episodic_memory

        console.print("[green][OK][/green] Phase 2 systems initialized:")
        console.print(f"     [cyan]Vector Store:[/cyan] {settings.vector_backend} ({vector_store.count()} existing docs)")
        console.print(f"     [cyan]Embeddings:[/cyan] {settings.embedding_model}")
        console.print(f"     [cyan]Episodic Memory:[/cyan] {episodic_memory.count} prior episodes")

    except Exception as e:
        logger.warning(f"Phase 2 initialization failed (falling back to Phase 1): {e}")
        console.print(
            f"[yellow][!][/yellow] Phase 2 unavailable: {e}\n"
            f"    [dim]Agent will run in Phase 1 mode (tool-only).[/dim]"
        )
        return {}

    return components


# ===================================================================
# Result Display
# ===================================================================

def display_results(state: dict) -> None:
    """
    Display the agent's results in a formatted, readable way.

    Shows:
    1. Final analysis (the answer)
    2. Reasoning trace (how it got there)
    3. Tool usage summary
    4. Phase 2 memory stats
    5. Execution metadata
    """
    console.print()

    # --- Final Answer ---
    status = state.get("status", "unknown")
    if status == AgentStatus.COMPLETED:
        status_color = "green"
        status_text = "[OK] Analysis Complete"
    elif status == AgentStatus.MAX_ITERATIONS_REACHED:
        status_color = "yellow"
        status_text = "[!] Partial Analysis (iteration limit)"
    else:
        status_color = "red"
        status_text = f"[FAIL] {status}"

    answer = state.get("final_answer", "No answer produced.")
    console.print(Panel(
        Markdown(answer),
        title=f"[bold {status_color}]{status_text}[/bold {status_color}]",
        border_style=status_color,
        padding=(1, 2),
    ))

    # --- Reasoning Trace ---
    trace = state.get("reasoning_trace", [])
    if trace:
        console.print(f"\n[bold cyan]Reasoning Trace ({len(trace)} steps):[/bold cyan]")
        for step in trace:
            iteration = step.iteration if hasattr(step, "iteration") else "?"
            thought = step.thought if hasattr(step, "thought") else str(step)
            action = step.action if hasattr(step, "action") else ""

            console.print(f"\n  [dim]── Step {iteration} ──[/dim]")
            if thought:
                console.print(f"  [green]Thought:[/green] {thought[:200]}{'...' if len(thought) > 200 else ''}")
            if action and action != "final_answer":
                action_input = step.action_input if hasattr(step, "action_input") else {}
                console.print(f"  [blue]Action:[/blue] {action}({action_input})")
            if hasattr(step, "observation") and step.observation and action != "final_answer":
                obs = step.observation
                console.print(f"  [yellow]Observation:[/yellow] {obs[:150]}{'...' if len(obs) > 150 else ''}")

    # --- Tool Usage Summary ---
    tool_calls = state.get("tool_calls", [])
    if tool_calls:
        console.print(f"\n[bold cyan]Tool Usage Summary:[/bold cyan]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Tool", style="cyan")
        table.add_column("Input", style="white")
        table.add_column("Status", style="green")

        for i, tc in enumerate(tool_calls, 1):
            name = tc.tool_name if hasattr(tc, "tool_name") else str(tc)
            inp = str(tc.tool_input)[:40] if hasattr(tc, "tool_input") else ""
            status = "[green]OK[/green]" if (hasattr(tc, "success") and tc.success) else "[red]FAIL[/red]"
            table.add_row(str(i), name, inp, status)

        console.print(table)

    # --- Phase 2: Memory Stats ---
    memory_ops = state.get("memory_retrievals", [])
    if memory_ops:
        console.print(f"\n[bold cyan]Memory Operations:[/bold cyan]")
        for op in memory_ops:
            console.print(f"  [dim]📦[/dim] {op}")

    # --- Execution Metadata ---
    console.print(f"\n[dim]Iterations: {state.get('iteration_count', 0)}/{state.get('max_iterations', 0)}[/dim]")
    console.print(f"[dim]Model: {settings.active_model}[/dim]")

    errors = state.get("errors", [])
    if errors:
        console.print(f"\n[bold yellow]Warnings/Errors ({len(errors)}):[/bold yellow]")
        for err in errors[-5:]:  # Show last 5
            console.print(f"  [yellow]![/yellow] {err}")


# ===================================================================
# Main Entry Point
# ===================================================================

def run_agent(query: str, phase2_components: dict = None) -> dict:
    """
    Run the ARA-1 agent with the given query.

    Args:
        query: Financial research question (e.g., "Analyze Tesla stock")
        phase2_components: Dict of Phase 2 systems (or None for Phase 1 only)

    Returns:
        Final agent state dict.
    """
    phase2_components = phase2_components or {}

    # 1. Initialize tools
    registry = create_tool_registry()

    # 2. Build graph (with Phase 2 systems if available)
    graph = build_graph(
        tool_registry=registry,
        ingestion_pipeline=phase2_components.get("ingestion_pipeline"),
        semantic_retriever=phase2_components.get("semantic_retriever"),
    )

    # 3. Load episodic context (Phase 2)
    episodic_context = ""
    episodic_memory = phase2_components.get("episodic_memory")
    if episodic_memory:
        try:
            episodic_context = episodic_memory.get_context_for_query(
                query=query,
                max_episodes=3,
            )
            if episodic_context:
                logger.info("Loaded episodic context from prior analyses")
        except Exception as e:
            logger.warning(f"Episodic context loading failed: {e}")

    # 4. Create initial state
    initial_state = create_initial_state(
        query=query,
        max_iterations=settings.max_iterations,
    )
    # Inject episodic context
    if episodic_context:
        initial_state["episodic_context"] = episodic_context

    # 5. Run the graph
    console.print(f"\n[bold cyan]Starting analysis...[/bold cyan]")
    console.print(f"[dim]Query: {query}[/dim]")
    console.print(f"[dim]Max iterations: {settings.max_iterations}[/dim]\n")

    start_time = time.time()

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"Agent execution failed: {type(e).__name__}: {str(e)}")
        console.print(f"\n[bold red]Agent execution failed:[/bold red] {str(e)}")
        raise

    elapsed = time.time() - start_time
    console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")

    # 6. Save episode to episodic memory (Phase 2)
    if episodic_memory and final_state:
        _save_episode(episodic_memory, query, final_state, elapsed)

    # 7. Save final analysis to vector memory (Phase 2)
    ingestion = phase2_components.get("ingestion_pipeline")
    if ingestion and final_state.get("final_answer"):
        try:
            ingestion.ingest_research_note(
                content=final_state["final_answer"],
                ticker=_extract_ticker(final_state),
                title=f"Analysis: {query[:80]}",
            )
            logger.info("Final analysis saved to vector memory")
        except Exception as e:
            logger.warning(f"Failed to save analysis to memory: {e}")

    return final_state


# ===================================================================
# Phase 4: System Initialization
# ===================================================================

def initialize_phase4_systems() -> dict:
    """
    Initialize Phase 4 subsystems: observability, evaluation,
    checkpointing, retry handler, and failure injector.

    Returns a dict of initialized components.

    WHY separate from Phase 1-3 init?
    - Phase 4 is OBSERVATIONAL — it wraps existing behavior.
    - It should never prevent the agent from running.
    - Every Phase 4 failure is non-fatal by design.
    """
    components = {}

    try:
        # 1. Telemetry Collector
        from observability.collector import TelemetryCollector
        collector = TelemetryCollector()
        components["collector"] = collector

        # 2. Execution Tracer
        from observability.tracer import ExecutionTracer
        tracer = ExecutionTracer(run_id=collector.run_id)
        components["tracer"] = tracer

        # 3. Checkpoint Manager
        if settings.enable_checkpoints:
            from agent.checkpoint_manager import CheckpointManager
            checkpoint_mgr = CheckpointManager(
                checkpoint_dir=str(settings.checkpoint_dir),
                run_id=collector.run_id,
            )
            components["checkpoint_manager"] = checkpoint_mgr

        # 4. Retry Handler
        from agent.retry_handler import RetryHandler, RetryConfig
        retry_handler = RetryHandler(
            config=RetryConfig(
                max_retries=settings.max_retries,
                initial_delay_seconds=settings.retry_backoff_seconds,
            ),
            collector=collector,
        )
        if settings.enable_fallback:
            retry_handler.configure_fallback_chain()
        components["retry_handler"] = retry_handler

        # 5. System Evaluator
        if settings.enable_evaluation:
            from evaluation.evaluator import SystemEvaluator
            evaluator = SystemEvaluator()
            components["evaluator"] = evaluator

        # 6. Failure Injector (only if explicitly enabled)
        if settings.enable_failure_injection:
            from evaluation.failure_injector import FailureInjector
            injector = FailureInjector(collector=collector)
            components["failure_injector"] = injector

        console.print("[green][OK][/green] Phase 4 systems initialized:")
        console.print(f"     [cyan]Telemetry:[/cyan] collector + tracer active")
        console.print(f"     [cyan]Checkpoints:[/cyan] {'enabled' if settings.enable_checkpoints else 'disabled'}")
        console.print(f"     [cyan]Retry/Fallback:[/cyan] max {settings.max_retries} retries")
        console.print(f"     [cyan]Evaluation:[/cyan] {'enabled' if settings.enable_evaluation else 'disabled'}")
        console.print(f"     [cyan]Failure Injection:[/cyan] {'ACTIVE' if settings.enable_failure_injection else 'disabled'}")

    except Exception as e:
        logger.warning(f"Phase 4 initialization failed (non-fatal): {e}")
        console.print(
            f"[yellow][!][/yellow] Phase 4 unavailable: {e}\n"
            f"    [dim]Agent will run without evaluation/observability.[/dim]"
        )

    return components


def run_phase4_evaluation(
    final_state: dict,
    phase4_components: dict,
    elapsed_seconds: float = 0,
    synthesis_report: dict | None = None,
    report_paths: dict | None = None,
) -> None:
    """
    Run Phase 4 evaluation pipeline on completed agent state.

    This is the POST-GRAPH, POST-SYNTHESIS evaluation:
    1. Compute all 22+ metrics.
    2. Run hallucination detection.
    3. Run tool efficiency analysis.
    4. Export results.
    5. Display to console.
    """
    evaluator = phase4_components.get("evaluator")
    collector = phase4_components.get("collector")

    if not evaluator:
        return

    try:
        console.print("\n[bold cyan]Phase 4: Running evaluation pipeline...[/bold cyan]")

        # Run evaluation
        eval_report = evaluator.evaluate(
            agent_state=final_state,
            elapsed_seconds=elapsed_seconds,
            synthesis_report=synthesis_report,
            report_paths=report_paths,
        )

        # Display results
        _display_evaluation_results(eval_report)

        # Export evaluation JSON
        try:
            settings.evaluation_dir.mkdir(parents=True, exist_ok=True)
            eval_path = settings.evaluation_dir / f"eval_{collector.run_id if collector else 'latest'}.json"
            evaluator.export_to_json(eval_report, eval_path)
            console.print(f"  [dim]📊 Evaluation: {eval_path}[/dim]")
        except Exception as e:
            logger.warning(f"Failed to export evaluation: {e}")

        # Export telemetry
        if collector:
            try:
                settings.telemetry_dir.mkdir(parents=True, exist_ok=True)
                tel_path = settings.telemetry_dir / f"telemetry_{collector.run_id}.json"
                collector.export_to_json(tel_path)
                console.print(f"  [dim]📈 Telemetry: {tel_path}[/dim]")
            except Exception as e:
                logger.warning(f"Failed to export telemetry: {e}")

    except Exception as e:
        logger.error(f"Phase 4 evaluation failed: {e}")
        console.print(f"\n[yellow]Phase 4 evaluation failed (non-fatal):[/yellow] {e}")


def _display_evaluation_results(eval_report) -> None:
    """Display Phase 4 evaluation results to the console."""
    from evaluation.metrics import MetricStatus

    # Overall score panel
    score = eval_report.overall_score
    if score >= 0.8:
        score_color = "green"
    elif score >= 0.6:
        score_color = "yellow"
    else:
        score_color = "red"

    console.print(Panel(
        Text.from_markup(
            f"[bold]Overall Score:[/bold] [{score_color}]{score:.0%}[/{score_color}]\n"
            f"[bold]Pass Rate:[/bold] {eval_report.pass_rate:.0%} "
            f"({eval_report.passed_metrics}/{eval_report.total_metrics})\n"
            f"[bold]Warnings:[/bold] {eval_report.warned_metrics}\n"
            f"[bold]Failures:[/bold] {eval_report.failed_metrics}"
        ),
        title="[bold cyan]Phase 4: Evaluation Results[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # Metric details table
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Status", width=8)
    table.add_column("Category", style="dim")

    status_styles = {
        MetricStatus.PASS: "[green]PASS[/green]",
        MetricStatus.WARN: "[yellow]WARN[/yellow]",
        MetricStatus.FAIL: "[red]FAIL[/red]",
        "pass": "[green]PASS[/green]",
        "warn": "[yellow]WARN[/yellow]",
        "fail": "[red]FAIL[/red]",
    }

    for i, m in enumerate(eval_report.metrics, 1):
        status_display = status_styles.get(m.status, str(m.status))
        cat = m.category if isinstance(m.category, str) else m.category.value
        table.add_row(
            str(i),
            m.name,
            m.formatted,
            status_display,
            cat,
        )

    console.print(table)


def _save_episode(episodic_memory, query: str, state: dict, elapsed: float):
    """Save the current run as an episode in episodic memory."""
    try:
        from memory.episodic import Episode

        tool_calls = state.get("tool_calls", [])
        tools_used = [
            tc.tool_name if hasattr(tc, "tool_name") else str(tc)
            for tc in tool_calls
        ]
        succeeded = sum(1 for tc in tool_calls if hasattr(tc, "success") and tc.success)
        failed_tools = [
            tc.tool_name for tc in tool_calls
            if hasattr(tc, "success") and not tc.success
        ]

        episode = Episode(
            query=query,
            ticker=_extract_ticker(state),
            status=str(state.get("status", "")),
            iterations_used=state.get("iteration_count", 0),
            max_iterations=state.get("max_iterations", 10),
            execution_time_seconds=round(elapsed, 1),
            tools_used=tools_used,
            tools_succeeded=succeeded,
            tools_failed=len(failed_tools),
            failed_tools=failed_tools,
            key_findings=state.get("final_answer", "")[:300],
            errors_encountered=state.get("errors", [])[-3:],
            evidence_retrieved=len(state.get("retrieved_evidence", [])),
            conflicts_detected=len(state.get("conflict_reports", [])),
        )

        episodic_memory.save_episode(episode)
    except Exception as e:
        logger.warning(f"Failed to save episode: {e}")


def _extract_ticker(state: dict) -> str:
    """Extract the primary ticker from the agent state."""
    for tc in state.get("tool_calls", []):
        if hasattr(tc, "tool_input"):
            ticker = tc.tool_input.get("ticker", "")
            if ticker:
                return ticker.upper()
    return ""


# ===================================================================
# Phase 3: Synthesis Initialization
# ===================================================================

def initialize_phase3_systems() -> dict:
    """
    Initialize Phase 3 subsystems: synthesis engine and report generator.

    Returns a dict of initialized components, or empty dict if
    Phase 3 initialization fails (agent still works with Phase 1+2).
    """
    components = {}

    try:
        from synthesis.engine import SynthesisEngine
        from synthesis.report_generator import ReportGenerator

        synthesis_engine = SynthesisEngine()
        components["synthesis_engine"] = synthesis_engine

        report_generator = ReportGenerator(
            output_dir=str(settings.report_output_dir),
        )
        components["report_generator"] = report_generator

        console.print("[green][OK][/green] Phase 3 systems initialized:")
        console.print(f"     [cyan]Synthesis Engine:[/cyan] all sub-engines loaded")
        console.print(f"     [cyan]Report Generator:[/cyan] output → {settings.report_output_dir}")

    except Exception as e:
        logger.warning(f"Phase 3 initialization failed (synthesis unavailable): {e}")
        console.print(
            f"[yellow][!][/yellow] Phase 3 unavailable: {e}\n"
            f"    [dim]Agent will run without synthesis/reports.[/dim]"
        )
        return {}

    return components


def run_phase3_synthesis(final_state: dict, phase3_components: dict) -> None:
    """
    Run Phase 3 synthesis pipeline on completed agent state.

    This is the POST-GRAPH pipeline:
    1. Synthesis Engine processes all tool outputs and evidence.
    2. Report Generator produces Markdown (and optionally PDF) reports.
    3. Results are displayed to the console.
    """
    synthesis_engine = phase3_components.get("synthesis_engine")
    report_generator = phase3_components.get("report_generator")

    if not synthesis_engine:
        return

    try:
        console.print("\n[bold cyan]Phase 3: Running synthesis pipeline...[/bold cyan]")

        # Run synthesis
        report = synthesis_engine.synthesize(final_state)

        # Generate reports
        formats = ["markdown"]
        try:
            import fpdf
            formats.append("pdf")
        except ImportError:
            pass

        paths = {}
        if report_generator:
            paths = report_generator.generate(report, formats=formats)

        # Display synthesis results
        _display_synthesis_results(report, paths)

    except Exception as e:
        logger.error(f"Phase 3 synthesis failed: {e}")
        console.print(f"\n[yellow]Phase 3 synthesis failed (non-fatal):[/yellow] {e}")


def _display_synthesis_results(report, paths: dict) -> None:
    """Display Phase 3 synthesis results to the console."""
    from synthesis.schemas import InvestmentOutlook, RiskSeverity

    # Investment Outlook
    outlook_colors = {
        InvestmentOutlook.STRONG_BUY: "bold green",
        InvestmentOutlook.BUY: "green",
        InvestmentOutlook.HOLD: "yellow",
        InvestmentOutlook.SELL: "red",
        InvestmentOutlook.STRONG_SELL: "bold red",
        InvestmentOutlook.INSUFFICIENT_DATA: "dim",
    }
    color = outlook_colors.get(report.outlook, "white")

    console.print(Panel(
        Text.from_markup(
            f"[bold]Investment Outlook:[/bold] [{color}]"
            f"{report.outlook.value.replace('_', ' ').upper()}[/{color}]\n"
            f"[bold]Confidence:[/bold] {report.confidence.overall:.0%} "
            f"({report.confidence.label})\n"
            f"[bold]Financial Health:[/bold] {report.financial.financial_health_score:.2f}\n"
            f"[bold]Sentiment:[/bold] {report.sentiment.overall_direction.value}\n"
            f"[bold]Risk Level:[/bold] {report.risk.overall_risk_level.value}"
        ),
        title="[bold cyan]Phase 3: Synthesis Results[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # Key Findings
    if report.key_findings:
        console.print("\n[bold cyan]Key Findings:[/bold cyan]")
        for finding in report.key_findings:
            console.print(f"  [dim]•[/dim] {finding}")

    # Misalignment Alert
    if report.misalignment.detected:
        console.print(Panel(
            Text.from_markup(
                f"[bold]Type:[/bold] {report.misalignment.misalignment_type.value}\n"
                f"[bold]Severity:[/bold] {report.misalignment.severity.value}\n"
                f"{report.misalignment.explanation}"
            ),
            title="[bold yellow]⚠ Misalignment Detected[/bold yellow]",
            border_style="yellow",
        ))

    # Report paths
    if paths:
        console.print("\n[bold cyan]Generated Reports:[/bold cyan]")
        for fmt, path in paths.items():
            console.print(f"  [dim]📄[/dim] {fmt.upper()}: {path}")


def main():
    """CLI entry point."""
    # Banner
    console.print(Panel(
        Text.from_markup(
            "[bold cyan]ARA-1[/bold cyan] - [dim]Autonomous Research Agent[/dim]\n"
            "[dim]Phase 4: Production-Grade Financial Intelligence System[/dim]"
        ),
        border_style="cyan",
        padding=(1, 2),
    ))

    # Validate config
    if not validate_configuration():
        sys.exit(1)

    console.print(f"[green][OK][/green] Configuration valid")
    console.print(f"[green][OK][/green] LLM Provider: [cyan]{settings.llm_provider}[/cyan] ({settings.active_model})")

    # Initialize Phase 2 systems
    phase2_components = initialize_phase2_systems()
    console.print()

    # Initialize Phase 3 systems
    phase3_components = initialize_phase3_systems()
    console.print()

    # Initialize Phase 4 systems
    phase4_components = initialize_phase4_systems()
    console.print()

    # Get query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = console.input("[bold]Enter your financial research query:[/bold] ")

    if not query.strip():
        console.print("[red]No query provided. Exiting.[/red]")
        sys.exit(1)

    # Set telemetry context
    collector = phase4_components.get("collector")
    tracer = phase4_components.get("tracer")
    if collector:
        collector.set_context(query=query.strip())
    if tracer:
        tracer.set_context(query=query.strip())

    # Run agent
    try:
        start_time = time.time()
        final_state = run_agent(query.strip(), phase2_components)
        elapsed = time.time() - start_time
        display_results(final_state)

        # Phase 3: Synthesis & Report Generation
        synthesis_report_dict = None
        report_paths = None
        if phase3_components and final_state:
            synthesis_report_dict, report_paths = _run_phase3_with_capture(
                final_state, phase3_components
            )

        # Phase 4: Evaluation
        if phase4_components and final_state:
            run_phase4_evaluation(
                final_state=final_state,
                phase4_components=phase4_components,
                elapsed_seconds=elapsed,
                synthesis_report=synthesis_report_dict,
                report_paths=report_paths,
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]Agent interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        logger.exception("Fatal error in main")
        sys.exit(1)


def _run_phase3_with_capture(
    final_state: dict,
    phase3_components: dict,
) -> tuple[dict | None, dict | None]:
    """
    Run Phase 3 synthesis and capture the report + paths for Phase 4.

    Returns (synthesis_report_dict, report_paths) for evaluation.
    """
    synthesis_engine = phase3_components.get("synthesis_engine")
    report_generator = phase3_components.get("report_generator")

    if not synthesis_engine:
        return None, None

    try:
        console.print("\n[bold cyan]Phase 3: Running synthesis pipeline...[/bold cyan]")

        # Run synthesis
        report = synthesis_engine.synthesize(final_state)

        # Generate reports
        formats = ["markdown"]
        try:
            import fpdf
            formats.append("pdf")
        except ImportError:
            pass

        paths = {}
        if report_generator:
            paths = report_generator.generate(report, formats=formats)

        # Display synthesis results
        _display_synthesis_results(report, paths)

        # Convert report to dict for evaluation
        report_dict = None
        if hasattr(report, "to_dict"):
            report_dict = report.to_dict()
        elif hasattr(report, "__dict__"):
            report_dict = {
                "financial": {
                    "metrics_available": len(getattr(report.financial, 'key_strengths', [])),
                    "metrics_missing": len(getattr(report.financial, 'key_weaknesses', [])),
                },
                "confidence": {"overall": getattr(report.confidence, 'overall', 0.5)},
                "misalignment": {
                    "detected": getattr(report.misalignment, 'detected', False),
                    "misalignment_type": getattr(report.misalignment, 'misalignment_type', None),
                },
                "risk": {
                    "risks": [
                        {"category": getattr(r, 'category', 'unknown')}
                        for r in getattr(report.risk, 'risks', [])
                    ],
                },
            }

        return report_dict, paths

    except Exception as e:
        logger.error(f"Phase 3 synthesis failed: {e}")
        console.print(f"\n[yellow]Phase 3 synthesis failed (non-fatal):[/yellow] {e}")
        return None, None


if __name__ == "__main__":
    main()

