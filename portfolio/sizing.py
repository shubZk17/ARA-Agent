"""
portfolio/sizing.py — volatility-scaled position sizing (Phase 8.3)
======================================================================

WHY THIS WAITS ON 8.1:
    A position size is a bet on the recommendation's confidence being
    correct. Sizing off a confidence score that has never been checked
    against a real outcome is just leveraging an unverified signal — so
    below MIN_GRADED_SAMPLE graded recommendations, sizing is deliberately
    floored rather than trusting the number.

WHAT DRIVES THE SIZE:
    - realized volatility (6.3's technical engine already computes this;
      pass it straight through — no re-derivation here)
    - the risk profile's confidence floor (config/horizons.RISK_PROFILES) —
      below it, size is zero, not "small"
    - Phase 8.1's measured hit rate, once there is enough of it
    - a correlation penalty against the rest of the book, if given

NO NEW DEPENDENCIES, NO NETWORK: pure arithmetic on numbers the caller
already has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config.horizons import get_risk_profile

# Below this many graded outcomes, a measured hit rate is noise (matches
# analysis/confidence_calibrator.py's MIN_GRADED_SAMPLE — same reasoning).
MIN_GRADED_SAMPLE = 10

# Annualized vol (%) treated as "1x" sizing — a position at this vol gets
# the base sleeve with no vol adjustment either way.
TARGET_VOL_PCT = 15.0

BASE_SLEEVE_PCT = 0.10          # base position size at target vol, multiplier 1.0
MAX_POSITION_PCT = 0.25         # single-name cap regardless of signal strength
CORRELATION_PENALTY_THRESHOLD = 0.7

_RISK_MULTIPLIER = {"conservative": 0.6, "balanced": 1.0, "aggressive": 1.4}


@dataclass(frozen=True)
class SizingResult:
    position_pct: float          # fraction of capital, e.g. 0.08 = 8%
    rationale: str


def position_size(
    annualized_volatility_pct: float,
    confidence: float,
    risk_profile: str = "balanced",
    hit_rate: Optional[float] = None,
    graded_sample_size: int = 0,
    max_correlation_to_existing: float = 0.0,
) -> SizingResult:
    """
    Volatility-scaled position size as a fraction of capital.

    `hit_rate`/`graded_sample_size` should come from
    validation.outcome_scorer.summarize()'s "hit_rate"/"graded_count" for
    the relevant outlook class — pass None/0 if nothing has been graded yet.
    """
    if annualized_volatility_pct is None or annualized_volatility_pct <= 0:
        return SizingResult(0.0, "No usable volatility estimate — refusing to size")

    profile_cfg = get_risk_profile(risk_profile)
    if confidence < profile_cfg.min_confidence_to_act:
        return SizingResult(
            0.0,
            f"Confidence {confidence:.0%} below {risk_profile}'s "
            f"{profile_cfg.min_confidence_to_act:.0%} floor to act",
        )

    base_pct = TARGET_VOL_PCT / annualized_volatility_pct * BASE_SLEEVE_PCT
    multiplier = _RISK_MULTIPLIER.get(risk_profile, 1.0)

    if hit_rate is None or graded_sample_size < MIN_GRADED_SAMPLE:
        multiplier *= 0.5
        note = (
            f" (hit-rate sample too small — {graded_sample_size}/"
            f"{MIN_GRADED_SAMPLE} graded — size floored until Phase 8.1 has more data)"
        )
    else:
        # 0 at a 50% (coin-flip) hit rate, 1.0 at a perfect 100%. A measured
        # edge earns more size; a measured coin-flip or worse gets throttled.
        edge = max(0.0, hit_rate - 0.5) * 2
        multiplier *= (0.5 + edge)
        note = f" (measured hit rate {hit_rate:.0%} over {graded_sample_size} calls)"

    pct = base_pct * multiplier

    if max_correlation_to_existing > CORRELATION_PENALTY_THRESHOLD:
        pct *= (1 - max_correlation_to_existing)
        note += f"; correlation {max_correlation_to_existing:.2f} to existing book reduced size"

    pct = max(0.0, min(MAX_POSITION_PCT, pct))
    return SizingResult(round(pct, 4), f"{pct:.1%} of capital" + note)


def demo() -> None:
    """Self-check: no network, no LLM."""
    zero_vol = position_size(0, 0.7)
    assert zero_vol.position_pct == 0.0

    below_floor = position_size(20.0, 0.2, risk_profile="conservative")
    assert below_floor.position_pct == 0.0

    unvalidated = position_size(15.0, 0.8, risk_profile="balanced")
    assert 0 < unvalidated.position_pct <= MAX_POSITION_PCT
    assert "floored" in unvalidated.rationale

    validated_strong = position_size(15.0, 0.8, risk_profile="balanced", hit_rate=0.75, graded_sample_size=20)
    validated_weak = position_size(15.0, 0.8, risk_profile="balanced", hit_rate=0.40, graded_sample_size=20)
    assert validated_strong.position_pct > unvalidated.position_pct
    assert validated_weak.position_pct < validated_strong.position_pct

    capped = position_size(1.0, 0.9, risk_profile="aggressive", hit_rate=0.9, graded_sample_size=50)
    assert capped.position_pct == MAX_POSITION_PCT

    correlated = position_size(
        15.0, 0.8, hit_rate=0.75, graded_sample_size=20, max_correlation_to_existing=0.9
    )
    assert correlated.position_pct < validated_strong.position_pct

    print("portfolio.sizing self-check OK")


if __name__ == "__main__":
    demo()
