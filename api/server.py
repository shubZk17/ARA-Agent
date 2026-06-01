"""
api/server.py — FastAPI Production Backend
=============================================

WHY THIS EXISTS:
    ARA-1 Phases 1-3 run as a CLI application. For production use,
    we need:
    - HTTP API endpoints for programmatic access.
    - Concurrent request handling.
    - Health checks for monitoring.
    - Structured JSON responses.
    - Async execution for long-running analyses.

WHAT PROBLEM IT SOLVES:
    Transforms ARA-1 from "run in terminal" to "call via API."
    Other systems (dashboards, trading bots, portfolio managers)
    can now consume ARA-1 intelligence.

HOW IT INTEGRATES:
    - Wraps main.run_agent() in HTTP endpoints.
    - Exposes evaluation results via /evaluate endpoint.
    - Provides telemetry via /telemetry endpoint.
    - Health check at /health.
    - Serves report files at /reports.

PRODUCTION TRADEOFFS:
    - Sync execution = simpler but blocks the worker.
    - For production: add Celery/background tasks for long analyses.
    - Single worker fine for dev; use Gunicorn/Uvicorn workers for scale.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Force UTF-8 on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ===================================================================
# FastAPI Application
# ===================================================================

app = FastAPI(
    title="ARA-1 Financial Intelligence API",
    description=(
        "Autonomous Research Agent — Production API for financial "
        "analysis, evaluation, and report generation."
    ),
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================
# Request / Response Models
# ===================================================================

class AnalysisRequest(BaseModel):
    """Request to run a financial analysis."""
    query: str = Field(description="Financial research question")
    max_iterations: int = Field(default=10, description="Max reasoning iterations")
    enable_evaluation: bool = Field(default=True, description="Run evaluation after analysis")
    enable_failure_injection: bool = Field(default=False, description="Enable failure injection testing")


class AnalysisResponse(BaseModel):
    """Response from a financial analysis."""
    status: str
    query: str
    ticker: str = ""
    final_answer: str = ""
    iterations_used: int = 0
    execution_time_seconds: float = 0.0
    evaluation_score: float = 0.0
    tool_calls: int = 0
    errors: list[str] = []


class HealthResponse(BaseModel):
    """System health check response."""
    status: str
    version: str
    llm_provider: str
    model: str
    timestamp: str
    uptime_seconds: float = 0.0


class EvaluationResponse(BaseModel):
    """Evaluation results response."""
    overall_score: float
    pass_rate: float
    total_metrics: int
    passed_metrics: int
    warned_metrics: int
    failed_metrics: int
    metrics: list[dict] = []


# ===================================================================
# Global state
# ===================================================================
_startup_time = time.time()
_phase2_components: dict = {}
_phase3_components: dict = {}
_last_telemetry: dict = {}


# ===================================================================
# Endpoints
# ===================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="4.0.0",
        llm_provider=settings.llm_provider,
        model=settings.active_model,
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=time.time() - _startup_time,
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def run_analysis(request: AnalysisRequest):
    """
    Run a financial analysis.

    This is the primary endpoint. It:
    1. Runs the agent with the given query.
    2. Optionally runs Phase 3 synthesis.
    3. Optionally runs Phase 4 evaluation.
    4. Returns structured results.
    """
    try:
        from main import run_agent, initialize_phase2_systems, initialize_phase3_systems

        start = time.time()

        # Initialize systems (cached after first call)
        global _phase2_components, _phase3_components
        if not _phase2_components:
            _phase2_components = initialize_phase2_systems()
        if not _phase3_components:
            _phase3_components = initialize_phase3_systems()

        # Run agent
        final_state = run_agent(request.query, _phase2_components)
        elapsed = time.time() - start

        # Extract results
        ticker = ""
        for tc in final_state.get("tool_calls", []):
            if hasattr(tc, "tool_input"):
                t = tc.tool_input.get("ticker", "")
                if t:
                    ticker = t.upper()
                    break

        response = AnalysisResponse(
            status=str(final_state.get("status", "completed")),
            query=request.query,
            ticker=ticker,
            final_answer=final_state.get("final_answer", ""),
            iterations_used=final_state.get("iteration_count", 0),
            execution_time_seconds=round(elapsed, 2),
            tool_calls=len(final_state.get("tool_calls", [])),
            errors=final_state.get("errors", []),
        )

        # Run evaluation if requested
        if request.enable_evaluation:
            try:
                from evaluation.evaluator import SystemEvaluator
                evaluator = SystemEvaluator()
                eval_report = evaluator.evaluate(
                    agent_state=final_state,
                    elapsed_seconds=elapsed,
                )
                response.evaluation_score = eval_report.overall_score
            except Exception as e:
                logger.warning(f"Evaluation failed (non-fatal): {e}")

        return response

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse)
async def run_evaluation(request: AnalysisRequest):
    """
    Run analysis + full evaluation pipeline.

    Returns detailed evaluation metrics.
    """
    try:
        from main import run_agent, initialize_phase2_systems
        from evaluation.evaluator import SystemEvaluator

        # Initialize
        global _phase2_components
        if not _phase2_components:
            _phase2_components = initialize_phase2_systems()

        # Run agent
        start = time.time()
        final_state = run_agent(request.query, _phase2_components)
        elapsed = time.time() - start

        # Run evaluation
        evaluator = SystemEvaluator()
        eval_report = evaluator.evaluate(
            agent_state=final_state,
            elapsed_seconds=elapsed,
        )

        return EvaluationResponse(
            overall_score=eval_report.overall_score,
            pass_rate=eval_report.pass_rate,
            total_metrics=eval_report.total_metrics,
            passed_metrics=eval_report.passed_metrics,
            warned_metrics=eval_report.warned_metrics,
            failed_metrics=eval_report.failed_metrics,
            metrics=[m.to_dict() for m in eval_report.metrics],
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports")
async def list_reports():
    """List all generated reports."""
    report_dir = settings.report_output_dir
    if not report_dir.exists():
        return {"reports": []}

    reports = []
    for f in sorted(report_dir.iterdir(), reverse=True):
        if f.suffix in (".md", ".pdf"):
            reports.append({
                "filename": f.name,
                "format": f.suffix[1:],
                "size_bytes": f.stat().st_size,
                "created": datetime.fromtimestamp(
                    f.stat().st_ctime, tz=timezone.utc
                ).isoformat(),
            })

    return {"reports": reports}


@app.get("/reports/{filename}")
async def get_report(filename: str):
    """Download a specific report file."""
    report_path = settings.report_output_dir / filename
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report not found: {filename}")

    media_types = {
        ".md": "text/markdown",
        ".pdf": "application/pdf",
    }
    media_type = media_types.get(report_path.suffix, "application/octet-stream")

    return FileResponse(
        path=str(report_path),
        media_type=media_type,
        filename=filename,
    )


@app.get("/telemetry")
async def get_telemetry():
    """Get the latest telemetry data."""
    return {"telemetry": _last_telemetry}


@app.get("/config")
async def get_config():
    """Get current system configuration (safe subset)."""
    return {
        "llm_provider": settings.llm_provider,
        "model": settings.active_model,
        "max_iterations": settings.max_iterations,
        "vector_backend": settings.vector_backend,
        "log_level": settings.log_level,
    }
