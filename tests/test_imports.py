"""
Import smoke checks, as assertions.

Replaces the root-level test_phase4_imports.py script. These would not have
caught a single one of D1–D11 — but they DO catch the failure mode this
codebase is genuinely prone to: initialize_phase2/3/4_systems() swallow
ImportError, so a broken import shows up as a silently missing subsystem
rather than a crash. That is exactly how the 15→8 package restructure left
the old smoke script failing 4/10 for a day without anyone noticing.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    # The ReAct loop
    "agent.graph", "agent.nodes", "agent.state", "agent.prompts", "agent.react_parser",
    # Tools
    "tools.base", "tools.registry", "tools.units",
    "tools.stock_price", "tools.financial_metrics", "tools.company_info", "tools.news",
    "tools.price_history", "tools.market_context",
    # Analytics (leaf — must import with no other package loaded)
    "analytics.indicators",
    # Knowledge
    "knowledge.retrieval.embeddings", "knowledge.retrieval.retriever",
    "knowledge.retrieval.vector_store", "knowledge.ingestion.pipeline",
    "knowledge.memory.episodic", "knowledge.reliability.scorer",
    # Analysis
    "analysis.engine", "analysis.financial_engine", "analysis.confidence_calibrator",
    "analysis.risk_analyzer", "analysis.sentiment_analyzer",
    "analysis.misalignment_detector", "analysis.report_generator", "analysis.schemas",
    "analysis.technical_engine",
    # Phase 6 — horizons and the recommendation log
    "config.horizons", "validation.recommendation_store",
    # Quality
    "quality.evaluation.evaluator", "quality.evaluation.metrics",
    "quality.evaluation.hallucination_detector", "quality.evaluation.tool_efficiency",
    "quality.observability.collector", "quality.observability.tracer",
    # Interfaces / config
    "api.server", "config.settings", "utils.logger",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_deleted_modules_stay_deleted():
    """
    retry_handler and checkpoint_manager were constructed and never invoked.
    If they come back, it should be LangGraph's native checkpointer instead.
    """
    for dead in ("agent.retry_handler", "agent.checkpoint_manager",
                 "quality.evaluation.failure_injector", "quality.evaluation.benchmarks"):
        with pytest.raises(ImportError):
            importlib.import_module(dead)


def test_empty_telemetry_collector_is_truthy():
    """
    D9: __len__ without __bool__ made an event-less collector falsy, so every
    `if collector:` guard skipped telemetry export. The collector is always
    event-less today, so this was always skipped.
    """
    from quality.observability.collector import TelemetryCollector

    collector = TelemetryCollector()
    assert len(collector) == 0
    assert collector, "empty collector is falsy again — telemetry export will be skipped"


def test_settings_report_a_healthy_config():
    from config.settings import settings

    assert settings.validate() == [] or settings.llm_provider in ("groq", "openai", "claude")
    assert isinstance(settings.warnings(), list)
    # The collection name must encode the embedding width — 384-dim MiniLM
    # vectors and 1536-dim OpenAI vectors cannot share a Chroma collection.
    assert settings.vector_collection_name.endswith(("_384", "_1536"))
