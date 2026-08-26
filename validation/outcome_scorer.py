"""
validation/outcome_scorer.py — Phase 8.1: grade matured recommendations
==========================================================================

WHY THIS EXISTS:
    Every one of the 22 evaluation metrics in quality/evaluation/metrics.py
    measures PROCESS (tool efficiency, parse-error rate). None of them can
    tell a right recommendation from a wrong one. This module is the first
    thing in the project that can: it re-fetches price history for a
    recommendation whose review_by_date has passed, compares the realized
    return to what the call implied, and produces a Brier score — the first
    genuine calibration measurement ARA has.

WHY A SEPARATE CLI, NOT PART OF A RUN:
    A recommendation cannot be graded until its review window has elapsed —
    weeks or months after it was made. This has to run on its own schedule,
    detached from any single analysis. `python -m validation.score`.

GRADING RULE:
    BUY/STRONG_BUY is "correct" if the realized return over the window is
    positive; SELL/STRONG_SELL if negative; HOLD is "correct" if the move
    stayed within HOLD_BAND (flat, as HOLD implies). Confidence is treated
    as the forecast probability of being correct, so Brier score is just
    (confidence - outcome)^2, averaged.

NO API KEYS: yfinance is keyless. This module never touches an LLM.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from utils.logger import get_logger
from validation.recommendation_store import RecommendationStore

logger = get_logger(__name__)

# A HOLD call is "correct" if price moved less than this in either direction.
HOLD_BAND = 0.05

# +1 for a bullish call, -1 bearish, 0 = flat (HOLD graded against HOLD_BAND).
_DIRECTION = {
    "strong_buy": 1, "buy": 1,
    "hold": 0,
    "sell": -1, "strong_sell": -1,
}

_BENCHMARK = "SPY"


def _closest_close(history: pd.DataFrame, target: pd.Timestamp) -> Optional[tuple[float, str]]:
    """Close price on the trading day nearest `target`, or None if empty."""
    if history is None or history.empty:
        return None
    idx = history.index.tz_localize(None) if history.index.tz is not None else history.index
    pos = idx.get_indexer([target], method="nearest")[0]
    if pos == -1:
        return None
    return float(history["Close"].iloc[pos]), idx[pos].date().isoformat()


def _fetch_close_near(ticker: str, target: pd.Timestamp, window_days: int = 5) -> Optional[tuple[float, str]]:
    """Best-effort close price near `target`. Never raises — a fetch failure
    just leaves the recommendation ungraded for this run."""
    try:
        start = target - timedelta(days=window_days)
        end = target + timedelta(days=window_days + 1)
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        return _closest_close(hist, target)
    except Exception as e:
        logger.warning(f"Price fetch failed for {ticker} near {target.date()}: {e}")
        return None


class OutcomeScorer:
    """Grades due recommendations against realized price outcomes."""

    def __init__(self, store: Optional[RecommendationStore] = None) -> None:
        self.store = store or RecommendationStore()

    def score_due(self, as_of: Optional[str] = None) -> list[dict[str, Any]]:
        """Grade every ungraded recommendation whose review date has passed.
        Returns the outcome dicts written. Never raises."""
        graded = []
        for record in self.store.due_for_review(as_of=as_of):
            outcome = self._grade(record)
            if outcome is None:
                continue
            if self.store.mark_outcome(record["id"], outcome):
                graded.append({**record, "outcome": outcome})
                logger.info(
                    f"Graded {record['id']} {record['ticker']} {record['outlook']}: "
                    f"return={outcome['realized_return']:+.2%} correct={outcome['correct']}"
                )
        return graded

    def _grade(self, record: dict[str, Any]) -> Optional[dict[str, Any]]:
        ticker = record.get("ticker")
        entry_price = record.get("price_at_recommendation")
        outlook = (record.get("outlook") or "").lower()
        if not ticker or entry_price in (None, 0) or outlook not in _DIRECTION:
            logger.warning(f"Skipping ungradeable record {record.get('id')}: missing ticker/price/outlook")
            return None

        review_date = pd.Timestamp(record.get("review_by_date") or record["timestamp"][:10])
        exit_data = _fetch_close_near(ticker, review_date)
        if exit_data is None:
            return None
        exit_price, exit_date = exit_data

        realized_return = (exit_price - entry_price) / entry_price

        entry_date = pd.Timestamp(record["timestamp"][:10])
        spy_entry = _fetch_close_near(_BENCHMARK, entry_date)
        spy_exit = _fetch_close_near(_BENCHMARK, review_date)
        spy_return = None
        if spy_entry and spy_exit and spy_entry[0]:
            spy_return = (spy_exit[0] - spy_entry[0]) / spy_entry[0]

        direction = _DIRECTION[outlook]
        if direction == 0:
            correct = abs(realized_return) <= HOLD_BAND
        else:
            correct = (realized_return > 0) == (direction > 0)

        confidence = record.get("confidence", 0.5)
        brier_component = (confidence - (1.0 if correct else 0.0)) ** 2

        return {
            "exit_price": round(exit_price, 4),
            "exit_date": exit_date,
            "realized_return": round(realized_return, 4),
            "spy_return": round(spy_return, 4) if spy_return is not None else None,
            "excess_return": round(realized_return - spy_return, 4) if spy_return is not None else None,
            "correct": correct,
            "brier_component": round(brier_component, 4),
            "graded_at": datetime.now(timezone.utc).isoformat(),
        }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate stats over any list of already-graded records (from
    `store.load_all()`, not just a single run's newly-graded batch)."""
    graded = [r for r in records if r.get("outcome")]
    if not graded:
        return {"graded_count": 0}

    hits = sum(1 for r in graded if r["outcome"]["correct"])
    brier = sum(r["outcome"]["brier_component"] for r in graded) / len(graded)

    by_outlook: dict[str, list[bool]] = {}
    for r in graded:
        by_outlook.setdefault(r["outlook"], []).append(r["outcome"]["correct"])

    return {
        "graded_count": len(graded),
        "hit_rate": round(hits / len(graded), 4),
        "brier_score": round(brier, 4),
        "hit_rate_by_outlook": {
            k: round(sum(v) / len(v), 4) for k, v in by_outlook.items()
        },
    }


def demo() -> None:
    """Self-check: grading logic against synthetic records, no network."""
    scorer = OutcomeScorer.__new__(OutcomeScorer)  # bypass __init__, no store needed

    buy = {"ticker": "X", "price_at_recommendation": 100.0, "outlook": "buy", "confidence": 0.7}
    hold = {"ticker": "X", "price_at_recommendation": 100.0, "outlook": "hold", "confidence": 0.6}

    def fake_grade(entry_price, exit_price, outlook, confidence):
        realized_return = (exit_price - entry_price) / entry_price
        direction = _DIRECTION[outlook]
        correct = abs(realized_return) <= HOLD_BAND if direction == 0 else (realized_return > 0) == (direction > 0)
        return correct, (confidence - (1.0 if correct else 0.0)) ** 2

    assert fake_grade(100, 110, "buy", 0.7)[0] is True
    assert fake_grade(100, 90, "buy", 0.7)[0] is False
    assert fake_grade(100, 101, "hold", 0.6)[0] is True
    assert fake_grade(100, 120, "hold", 0.6)[0] is False
    assert fake_grade(100, 90, "sell", 0.7)[0] is True

    summary = summarize([
        {"outlook": "buy", "outcome": {"correct": True, "brier_component": 0.09}},
        {"outlook": "buy", "outcome": {"correct": False, "brier_component": 0.49}},
        {"outlook": "sell", "outcome": {"correct": True, "brier_component": 0.04}},
    ])
    assert summary["graded_count"] == 3
    assert summary["hit_rate"] == round(2 / 3, 4)
    print("outcome_scorer self-check OK")


if __name__ == "__main__":
    demo()
