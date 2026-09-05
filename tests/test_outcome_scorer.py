"""
tests/test_outcome_scorer.py — Phase 8.1 grading logic and store round-trip.

No network: yfinance is monkeypatched with a tiny synthetic price history.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from analysis.outcome_scorer import OutcomeScorer, summarize
from analysis.recommendation_store import RecommendationStore


@pytest.fixture
def store(tmp_path) -> RecommendationStore:
    return RecommendationStore(store_dir=str(tmp_path))


def _write_record(store: RecommendationStore, **overrides) -> dict:
    record = {
        "id": "rec_test0001",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "ticker": "AAPL",
        "horizon": "short_term",
        "outlook": "buy",
        "confidence": 0.7,
        "price_at_recommendation": 100.0,
        "invalidation_condition": "close below 90",
        "review_by_date": "2026-01-15",
        "outcome": None,
    }
    record.update(overrides)
    path = store._path_for(pd.Timestamp(record["timestamp"]))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def test_mark_outcome_updates_in_place(store):
    _write_record(store)
    found = store.mark_outcome("rec_test0001", {"correct": True, "realized_return": 0.1})
    assert found is True
    (loaded,) = store.load_all()
    assert loaded["outcome"] == {"correct": True, "realized_return": 0.1}


def test_mark_outcome_missing_id_returns_false(store):
    _write_record(store)
    assert store.mark_outcome("does_not_exist", {"correct": True}) is False


def test_score_due_grades_a_buy_and_writes_outcome(store, monkeypatch):
    _write_record(store)

    def fake_fetch_close_near(ticker, target, window_days=5):
        # Everything closes at 120 — a winning BUY from an entry of 100.
        return 120.0, str(target.date())

    monkeypatch.setattr("analysis.outcome_scorer._fetch_close_near", fake_fetch_close_near)

    scorer = OutcomeScorer(store)
    graded = scorer.score_due(as_of="2026-02-01")

    assert len(graded) == 1
    outcome = graded[0]["outcome"]
    assert outcome["correct"] is True
    assert outcome["realized_return"] == pytest.approx(0.2)

    (persisted,) = store.load_all()
    assert persisted["outcome"]["correct"] is True


def test_score_due_grades_a_losing_sell(store, monkeypatch):
    _write_record(store, id="rec_test0002", outlook="sell", ticker="MSFT")

    def fake_fetch_close_near(ticker, target, window_days=5):
        return 130.0, str(target.date())  # up 30% — bad for a SELL call

    monkeypatch.setattr("analysis.outcome_scorer._fetch_close_near", fake_fetch_close_near)

    scorer = OutcomeScorer(store)
    graded = scorer.score_due(as_of="2026-02-01")
    assert graded[0]["outcome"]["correct"] is False


def test_hold_graded_within_band(store, monkeypatch):
    _write_record(store, id="rec_test0003", outlook="hold")

    def fake_fetch_close_near(ticker, target, window_days=5):
        return 102.0, str(target.date())  # +2%, inside the 5% HOLD band

    monkeypatch.setattr("analysis.outcome_scorer._fetch_close_near", fake_fetch_close_near)

    scorer = OutcomeScorer(store)
    graded = scorer.score_due(as_of="2026-02-01")
    assert graded[0]["outcome"]["correct"] is True


def test_score_due_skips_records_missing_price(store, monkeypatch):
    _write_record(store, id="rec_test0004", price_at_recommendation=None)
    scorer = OutcomeScorer(store)
    graded = scorer.score_due(as_of="2026-02-01")
    assert graded == []


def test_summarize_empty():
    assert summarize([]) == {"graded_count": 0}


def test_summarize_hit_rate_and_brier():
    records = [
        {"outlook": "buy", "outcome": {"correct": True, "brier_component": 0.09}},
        {"outlook": "buy", "outcome": {"correct": False, "brier_component": 0.49}},
        {"outlook": "sell", "outcome": {"correct": True, "brier_component": 0.04}},
        {"outlook": "hold", "outcome": None},  # not yet graded — excluded
    ]
    summary = summarize(records)
    assert summary["graded_count"] == 3
    assert summary["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["hit_rate_by_outlook"]["buy"] == pytest.approx(0.5)
    assert summary["hit_rate_by_outlook"]["sell"] == pytest.approx(1.0)
