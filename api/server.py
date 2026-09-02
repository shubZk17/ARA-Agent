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
from fastapi.staticfiles import StaticFiles
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

# CORS middleware for dashboard access.
#
# `allow_origins=["*"]` together with `allow_credentials=True` is the
# combination browsers refuse to honour anyway, and it invites any site the
# user visits to call this API with their cookies. Origins are now an explicit
# allowlist (the local Streamlit dashboard by default); override with
# CORS_ORIGINS as a comma-separated list.
_cors_origins = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
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
    # Phase 6 — optional with a default, so clients written before this
    # existed keep working and keep getting the long-term view they got before.
    horizon: str = Field(
        default="long_term",
        description=(
            "Investment horizon: 'short_term' (1w-3m), 'medium_term' (3mo-1y), "
            "or 'long_term' (1-5y)"
        ),
    )
    risk_profile: str = Field(
        default="balanced",
        description="Risk appetite: 'conservative', 'balanced' or 'aggressive'",
    )
    # Bring-your-own key. Passed straight through to the agent per-request and
    # never persisted or logged. If unset, the server's configured key is used.
    api_key: Optional[str] = Field(default=None, description="Your own LLM API key")
    llm_provider: Optional[str] = Field(
        default=None, description="'groq', 'openai' or 'claude' — defaults to server config"
    )
    llm_model: Optional[str] = Field(
        default=None, description="Model name for the chosen provider — defaults to server config"
    )
    # `enable_failure_injection` removed — quality/evaluation/failure_injector.py
    # was deleted 2026-08-15, so the flag advertised a capability that no
    # longer existed anywhere in the codebase.


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
    # Phase 3 synthesis, when available — confidence breakdown, the dated
    # HorizonRecommendation (invalidation condition, review date), risk
    # detail. None if synthesis wasn't run or failed (non-fatal).
    report: Optional[dict] = None


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
_phase4_components: dict = {}
_last_telemetry: dict = {}


# ===================================================================
# Endpoints
# ===================================================================

def _llm_override_from(request: AnalysisRequest) -> Optional[dict]:
    """Build the per-request LLM override, or None to use server config.

    Raises 400 if neither the request nor the server has a key for the
    resolved provider — clearer than letting the agent abstain with an
    opaque auth error.
    """
    provider = request.llm_provider or settings.llm_provider
    server_key = {
        "groq": settings.groq_api_key,
        "openai": settings.openai_api_key,
        "claude": settings.anthropic_api_key,
    }.get(provider, "")

    if not request.api_key and not server_key:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No {provider} API key available. This deployment has no server "
                f"key configured — enter your own key to run an analysis."
            ),
        )
    if not request.api_key:
        return None
    return {
        "api_key": request.api_key,
        "provider": request.llm_provider,
        "model": request.llm_model,
    }


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
        from main import (
            run_agent, initialize_phase2_systems, initialize_phase3_systems,
            initialize_phase4_systems,
        )

        llm_override = _llm_override_from(request)
        start = time.time()

        # Initialize systems (cached after first call)
        global _phase2_components, _phase3_components, _phase4_components
        if not _phase2_components:
            _phase2_components = initialize_phase2_systems()
        if not _phase3_components:
            _phase3_components = initialize_phase3_systems()
        if not _phase4_components:
            try:
                _phase4_components = initialize_phase4_systems()
            except Exception as e:
                logger.warning(f"Phase 4 (observability) init failed (non-fatal): {e}")

        collector = _phase4_components.get("collector")
        tracer = _phase4_components.get("tracer")

        # Run agent
        final_state = run_agent(
            request.query,
            _phase2_components,
            horizon=request.horizon,
            risk_profile=request.risk_profile,
            collector=collector,
            tracer=tracer,
            llm_override=llm_override,
        )
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

        # Run Phase 3 synthesis, non-fatal — the docstring above has promised
        # this since the endpoint was written, but nothing ever called it.
        synthesis_engine = _phase3_components.get("synthesis_engine")
        if synthesis_engine:
            try:
                from validation import record_recommendation
                report = synthesis_engine.synthesize(final_state, collector=collector)
                record_recommendation(report)
                response.report = report.model_dump(mode="json")
            except Exception as e:
                logger.warning(f"Synthesis failed (non-fatal): {e}")

        # Run evaluation if requested
        if request.enable_evaluation:
            try:
                from quality.evaluation.evaluator import SystemEvaluator
                evaluator = SystemEvaluator()
                eval_report = evaluator.evaluate(
                    agent_state=final_state,
                    elapsed_seconds=elapsed,
                )
                response.evaluation_score = eval_report.overall_score
            except Exception as e:
                logger.warning(f"Evaluation failed (non-fatal): {e}")

        return response

    except HTTPException:
        raise  # a deliberate 4xx (e.g. missing API key) — don't mask as 500
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
        from quality.evaluation.evaluator import SystemEvaluator

        # Initialize
        global _phase2_components
        if not _phase2_components:
            _phase2_components = initialize_phase2_systems()

        # Run agent
        llm_override = _llm_override_from(request)
        start = time.time()
        final_state = run_agent(
            request.query, _phase2_components, llm_override=llm_override
        )
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

    except HTTPException:
        raise
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
    # Path traversal guard. `settings.report_output_dir / filename` happily
    # resolves "../../.env" — this endpoint would serve any file the process
    # can read. Resolve first, then require the result to stay inside the
    # reports directory; comparing the resolved paths is what makes ".." and
    # symlinks and absolute paths all fail the same way.
    reports_root = settings.report_output_dir.resolve()
    report_path = (reports_root / filename).resolve()
    if not report_path.is_relative_to(reports_root):
        raise HTTPException(status_code=400, detail="Invalid report filename")

    if report_path.suffix not in (".md", ".pdf"):
        raise HTTPException(status_code=400, detail="Invalid report filename")

    if not report_path.is_file():
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
        # True when the deployment ships its own key — the UI can then make the
        # key field optional instead of required.
        "server_key_configured": bool(
            {"groq": settings.groq_api_key, "openai": settings.openai_api_key,
             "claude": settings.anthropic_api_key}.get(settings.llm_provider, "")
        ),
    }


# ===================================================================
# Static frontend — served same-origin so the browser calls /analyze,
# /reports, /health directly (no gateway, no CORS). Mounted last so it
# never shadows an API route.
# ===================================================================
_STATIC_DIR = _PROJECT_ROOT / "webui" / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
