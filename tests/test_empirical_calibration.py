"""
tests/test_empirical_calibration.py — Phase 8.4: empirical fallback baseline.
"""

from __future__ import annotations

from analysis.confidence_calibrator import MIN_GRADED_SAMPLE, _empirical_reliability_baseline
from analysis.recommendation_store import RecommendationStore


def _graded_records(hit_rate: float, n: int) -> list[dict]:
    hits = round(hit_rate * n)
    records = []
    for i in range(n):
        correct = i < hits
        records.append({
            "id": f"rec_{i}",
            "outlook": "buy",
            "outcome": {"correct": correct, "brier_component": 0.0},
        })
    return records


def test_below_sample_threshold_returns_default(monkeypatch):
    monkeypatch.setattr(
        RecommendationStore, "load_all", lambda self: _graded_records(0.9, MIN_GRADED_SAMPLE - 1)
    )
    assert _empirical_reliability_baseline(0.5) == 0.5


def test_measured_hit_rate_shifts_baseline(monkeypatch):
    monkeypatch.setattr(
        RecommendationStore, "load_all", lambda self: _graded_records(0.8, 20)
    )
    # 80% hit rate vs. the implicit 50% coin-flip reference -> default + 0.3
    assert _empirical_reliability_baseline(0.5) == 0.8


def test_broken_store_falls_back_to_default(monkeypatch):
    def _raise(self):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(RecommendationStore, "load_all", _raise)
    assert _empirical_reliability_baseline(0.6) == 0.6


def test_baseline_clamped_to_valid_range(monkeypatch):
    monkeypatch.setattr(RecommendationStore, "load_all", lambda self: _graded_records(1.0, 20))
    assert _empirical_reliability_baseline(0.9) == 0.95  # clamped, not 1.4
