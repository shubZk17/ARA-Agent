"""
D4 — confidence must respond to evidence.

Every successful run in the project's history reported ~90% "High", including
the run that graded Apple as CRITICAL risk on a unit error. These tests pin
the behaviour that makes the number move.
"""

from __future__ import annotations

import pytest

from analysis.confidence_calibrator import ConfidenceCalibrator
from analysis.schemas import (
    FinancialSnapshot,
    MisalignmentSignal,
    RiskAssessment,
    SentimentProfile,
)

ALL_TOOLS = [
    "get_stock_price", "get_financial_metrics", "get_company_info", "get_news",
]


class _Call:
    def __init__(self, tool_name: str, success: bool = True):
        self.tool_name = tool_name
        self.success = success


def _calibrate(**state):
    state.setdefault("tool_calls", [_Call(t) for t in ALL_TOOLS])
    state.setdefault("evidence_confidence", {})
    state.setdefault("conflict_reports", [])
    return ConfidenceCalibrator().calibrate(
        financial=FinancialSnapshot(
            metrics_available=state.pop("metrics", 21),
            metrics_missing=1,
            evidence_quality=state.pop("evidence_quality", 0.95),
        ),
        sentiment=SentimentProfile(),
        risk=RiskAssessment(),
        misalignment=MisalignmentSignal(),
        agent_state=state,
    )


def test_measured_reliability_beats_the_estimate():
    """The measured branch was unreachable before tool_node wrote to it."""
    measured = _calibrate(evidence_confidence={f"{t}:{i}": 0.9 for i, t in enumerate(ALL_TOOLS)})
    estimated = _calibrate(evidence_confidence={})

    assert measured.source_reliability == 0.9
    assert estimated.source_reliability < measured.source_reliability
    assert any("estimated, not measured" in p for p in estimated.penalties)


def test_failed_evidence_drags_confidence_down():
    good = _calibrate(evidence_confidence={"a:1": 0.9, "b:2": 0.9})
    bad = _calibrate(evidence_confidence={"a:1": 0.9, "b:2": 0.0})
    assert bad.overall < good.overall
    assert any("no usable evidence" in p for p in bad.penalties)


def test_unmeasured_consistency_is_not_scored_as_perfect():
    """
    The polarity fix. No conflict reports means nothing was measured — all
    four tools read one endpoint — not that everything agreed.
    """
    score = _calibrate()
    assert score.consistency < 1.0
    assert any("Consistency unmeasured" in p for p in score.penalties)
    assert not any("no conflicts detected" in b for b in score.boosts)


def test_single_source_costs_data_completeness():
    """Four tools reading one API is not four sources of data."""
    score = _calibrate()
    assert score.data_completeness < 1.0
    assert any("Single data source" in p for p in score.penalties)


def test_three_source_families_reach_full_diversity_factor():
    """
    Phase 7.4 — get_insider_transactions is tagged a genuinely distinct
    family ("sec_edgar_ownership") from get_sec_filings ("sec_edgar"), so
    3 independent families are now reachable and SOURCE_DIVERSITY_FACTOR[3]
    (1.00) applies instead of being capped at [2] (0.90).
    """
    calibrator = ConfidenceCalibrator()
    ideal_tools = [
        "get_stock_price", "get_financial_metrics", "get_company_info",
        "get_news", "get_price_history",
    ]

    two_families = calibrator._score_data_completeness(
        {"tool_calls": [_Call(t) for t in ideal_tools] + [_Call("get_sec_filings")]},
        [], [],
    )
    three_families = calibrator._score_data_completeness(
        {
            "tool_calls": [_Call(t) for t in ideal_tools]
            + [_Call("get_sec_filings"), _Call("get_insider_transactions")]
        },
        [], [],
    )

    assert three_families > two_families
    assert three_families == pytest.approx(1.0)


def test_thin_evidence_scores_far_below_complete_evidence():
    """
    The headline symptom of D4: a run with one tool and two metrics used to
    land within a few points of a run with four tools and 21 metrics.
    """
    rich = _calibrate(
        evidence_confidence={f"{t}:{i}": 0.9 for i, t in enumerate(ALL_TOOLS)},
    )
    thin = _calibrate(
        tool_calls=[_Call("get_stock_price")],
        evidence_confidence={"get_stock_price:1": 0.9},
        metrics=2,
        evidence_quality=0.09,
    )
    assert rich.overall - thin.overall > 0.25, (
        f"thin evidence scored {thin.overall} vs {rich.overall} — "
        f"confidence still is not tracking evidence"
    )


def test_confidence_stays_inside_its_declared_bounds():
    score = _calibrate(evidence_confidence={"a:1": 1.0}, evidence_quality=1.0)
    assert 0.10 <= score.overall <= 0.95


def test_degraded_embeddings_cap_confidence(monkeypatch):
    """D6: an analysis whose retrieval was random cannot be high-confidence."""
    monkeypatch.setattr(
        "analysis.confidence_calibrator._embeddings_degraded", lambda: True
    )
    score = _calibrate(evidence_confidence={f"{t}:{i}": 0.95 for i, t in enumerate(ALL_TOOLS)})
    assert score.overall <= 0.50
    assert any("degraded" in p.lower() for p in score.penalties)
