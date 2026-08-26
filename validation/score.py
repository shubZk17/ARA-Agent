"""
validation/score.py — CLI: grade matured recommendations (Phase 8.1)

    .venv/Scripts/python.exe -m validation.score
    .venv/Scripts/python.exe -m validation.score --as-of 2026-09-01   # testing
    .venv/Scripts/python.exe -m validation.score --summary-only       # skip grading, just report

Grades every ungraded recommendation whose review_by_date has passed, then
prints hit rate, Brier score, and hit rate by outlook class over ALL graded
recommendations (not just this run's batch).
"""

from __future__ import annotations

import argparse
import json

from validation.outcome_scorer import OutcomeScorer, summarize
from validation.recommendation_store import RecommendationStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade matured ARA recommendations against realized outcomes.")
    parser.add_argument("--as-of", default=None, help="Grade as of this date (YYYY-MM-DD) instead of today.")
    parser.add_argument("--summary-only", action="store_true", help="Skip grading, just summarize what's already graded.")
    args = parser.parse_args()

    store = RecommendationStore()

    if not args.summary_only:
        scorer = OutcomeScorer(store)
        newly_graded = scorer.score_due(as_of=args.as_of)
        print(f"Graded {len(newly_graded)} recommendation(s) this run.")

    summary = summarize(store.load_all())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
