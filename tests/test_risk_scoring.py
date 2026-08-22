"""
D12 — risk aggregation must track severity, not risk count.

Found on 2026-08-17 only because fixing D2 removed the false "High Leverage"
risk and Apple STILL came out CRITICAL. The old formula reduced to
`mean_severity / 0.6` for any n >= 2, so two medium concerns pinned the score
at 0.833 — above the 0.8 CRITICAL threshold — no matter what they were.
"""

from __future__ import annotations

import pytest

from analysis.risk_analyzer import RiskAnalyzer
from analysis.schemas import RiskItem, RiskSeverity


def _risks(*severities: RiskSeverity) -> list[RiskItem]:
    return [
        RiskItem(
            category="valuation",
            title=f"risk {i}",
            description="d",
            severity=s,
            probability="moderate",
        )
        for i, s in enumerate(severities)
    ]


def _level(*severities: RiskSeverity) -> RiskSeverity:
    analyzer = RiskAnalyzer()
    return analyzer._classify_risk_level(analyzer._calculate_risk_score(_risks(*severities)))


M, L, H, C = (
    RiskSeverity.MEDIUM, RiskSeverity.LOW, RiskSeverity.HIGH, RiskSeverity.CRITICAL,
)


def test_two_medium_risks_are_not_critical():
    """The exact regression: the AAPL report's CRITICAL verdict."""
    assert _level(M, M) == RiskSeverity.MEDIUM


def test_a_single_risk_maps_to_its_own_severity():
    assert _level(L) == RiskSeverity.LOW
    assert _level(M) == RiskSeverity.MEDIUM
    assert _level(H) == RiskSeverity.HIGH
    assert _level(C) == RiskSeverity.CRITICAL


def test_low_risks_stay_low_however_many():
    assert _level(L, L, L, L) == RiskSeverity.LOW


def test_one_critical_risk_is_not_diluted_by_mild_ones():
    """Averaging alone would bury a critical finding under four low ones."""
    assert _level(C, L, L, L, L) == RiskSeverity.CRITICAL


def test_severe_risks_still_reach_critical():
    assert _level(H, H, H) == RiskSeverity.CRITICAL


def test_a_lone_high_risk_is_high_not_critical():
    """
    HIGH scores 0.8 and CRITICAL used to trigger at >= 0.8, so a single high
    risk was always escalated and the HIGH band was unreachable in practice.
    """
    assert _level(H) == RiskSeverity.HIGH
    assert _level(H, M) == RiskSeverity.HIGH


def test_more_risks_never_lower_the_score():
    """Monotonicity: adding a finding cannot make a company look safer."""
    analyzer = RiskAnalyzer()
    previous = 0.0
    for n in range(1, 8):
        score = analyzer._calculate_risk_score(_risks(*([M] * n)))
        assert score >= previous, f"score dropped at n={n}"
        previous = score


def test_score_stays_in_range():
    analyzer = RiskAnalyzer()
    assert analyzer._calculate_risk_score([]) == pytest.approx(0.2)
    assert analyzer._calculate_risk_score(_risks(*([C] * 10))) <= 1.0
