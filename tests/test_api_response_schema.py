"""
api/server.py's AnalysisResponse must accept and serialize the optional
Phase 3 `report` field added 2026-08-27 — schema-level only, no network/LLM
(the endpoint itself calls run_agent, which needs a live LLM to exercise).
"""

from __future__ import annotations

from api.server import AnalysisResponse


def test_report_field_defaults_to_none():
    response = AnalysisResponse(status="completed", query="q")
    assert response.report is None


def test_report_field_accepts_and_serializes_a_dict():
    response = AnalysisResponse(
        status="completed",
        query="q",
        report={"confidence": {"overall": 0.74}, "outlook": "buy"},
    )
    dumped = response.model_dump(mode="json")
    assert dumped["report"]["confidence"]["overall"] == 0.74
    assert dumped["report"]["outlook"] == "buy"
