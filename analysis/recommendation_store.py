"""
analysis/recommendation_store.py — append-only log of every call made
========================================================================

WHY IT EXISTS NOW, IN PHASE 6, RATHER THAN IN PHASE 8 WHERE IT IS USED:
    The 22 evaluation metrics measure PROCESS — tool efficiency, parse-error
    rate, reasoning depth. Not one of them can tell a right recommendation
    from a wrong one. Phase 8 fixes that by scoring calls against realized
    returns, but a realized return needs two things this file captures and
    nothing else does:

        1. The price at the moment of the recommendation.
        2. The invalidation condition, stated BEFORE the outcome was known.

    Neither can be reconstructed later. Backfilling price is possible;
    backfilling "what we said would prove us wrong" is not, and without it
    a graded outcome is just hindsight.

WHY JSONL AND NOT A DATABASE:
    Append-only, one JSON object per line, no schema migration, readable with
    `tail`, and safe to concurrently append to on every platform that matters
    here. A recommendation is never updated in place — a revised view is a new
    line, so the record of what was believed when stays intact.

WHY THE CODE VERSION IS STORED:
    An outcome is evidence about the version of the system that produced it.
    Without the git SHA, a batch of scored recommendations spanning a change
    to the weighting tables is uninterpretable.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.logging import get_logger

logger = get_logger(__name__)

DEFAULT_STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "recommendations"


def _git_revision() -> str:
    """Short git SHA of the running code, or "" if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        # Not a git checkout, git not installed, or too slow. A missing SHA is
        # a small loss; a crashed analysis run over one is not acceptable.
        return ""


class RecommendationStore:
    """Append-only JSONL store, one file per month."""

    def __init__(self, store_dir: Optional[str] = None) -> None:
        self._dir = Path(store_dir) if store_dir else DEFAULT_STORE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, when: datetime) -> Path:
        """One file per month — small enough to read, few enough to list."""
        return self._dir / f"recommendations_{when.strftime('%Y%m')}.jsonl"

    def record(self, report) -> Optional[dict[str, Any]]:
        """
        Append one SynthesisReport's recommendation. Returns the written
        record, or None if there was nothing worth recording.

        Abstentions are deliberately NOT recorded: INSUFFICIENT_DATA is the
        system declining to make a call, and scoring a non-call as a miss
        would punish exactly the behaviour Phase 5 was built to encourage.
        """
        outlook = getattr(report.outlook, "value", str(report.outlook))
        if outlook == "insufficient_data":
            logger.info("Not logging an abstention — no recommendation was made.")
            return None

        if not report.ticker:
            logger.warning("Not logging a recommendation with no ticker.")
            return None

        now = datetime.now(timezone.utc)
        rec = report.recommendation

        record = {
            "id": f"rec_{uuid.uuid4().hex[:12]}",
            "timestamp": now.isoformat(),
            "ticker": report.ticker,
            "company_name": report.company_name,
            "query": report.query,

            # --- The call ---
            "horizon": report.horizon,
            "risk_profile": report.risk_profile,
            "outlook": outlook,
            "confidence": round(report.confidence.overall, 4),
            "holding_period": rec.holding_period,

            # --- What makes it gradable later ---
            "price_at_recommendation": rec.price_at_recommendation,
            "entry_condition": rec.entry_condition,
            "invalidation_condition": rec.invalidation_condition,
            "review_by_date": rec.review_by_date,

            # --- Supporting state, for attributing an outcome to a cause ---
            "financial_health_score": round(report.financial.financial_health_score, 4),
            "trend_score": (
                round(report.technical.trend_score, 4)
                if report.technical.available else None
            ),
            "risk_level": getattr(report.risk.overall_risk_level, "value", ""),
            "metrics_available": report.financial.metrics_available,

            # --- Provenance: which version of the system said this ---
            "report_id": report.id,
            "model_used": report.model_used,
            "git_revision": _git_revision(),

            # Filled in by Phase 8's scorer; present from the start so the
            # schema does not change once rows exist.
            "outcome": None,
        }

        path = self._path_for(now)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            f"Recommendation logged: {record['ticker']} {outlook} "
            f"[{record['horizon']}] @ {record['price_at_recommendation']} "
            f"→ review by {record['review_by_date']} ({path.name})"
        )
        return record

    def load_all(self) -> list[dict[str, Any]]:
        """Every recorded recommendation, oldest file first. Bad lines skipped."""
        records = []
        for path in sorted(self._dir.glob("recommendations_*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line in {path.name}")
        return records

    def due_for_review(self, as_of: Optional[str] = None) -> list[dict[str, Any]]:
        """Ungraded recommendations whose review date has passed."""
        today = as_of or datetime.now(timezone.utc).date().isoformat()
        return [
            r for r in self.load_all()
            if r.get("outcome") is None and (r.get("review_by_date") or "9999") <= today
        ]

    def mark_outcome(self, record_id: str, outcome: dict[str, Any]) -> bool:
        """
        Fill in one record's `outcome` field in place (Phase 8.1's scorer).

        Not a new JSONL line: `outcome` was reserved as null from the start
        for exactly this update (see the module docstring). Only the month
        file containing the id is rewritten. Returns True if the id was found.
        """
        found = False
        for path in sorted(self._dir.glob("recommendations_*.jsonl")):
            lines = path.read_text(encoding="utf-8").splitlines()
            changed = False
            new_lines = []
            for line in lines:
                if not line.strip():
                    new_lines.append(line)
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    new_lines.append(line)
                    continue
                if rec.get("id") == record_id:
                    rec["outcome"] = outcome
                    changed = True
                    found = True
                new_lines.append(json.dumps(rec, ensure_ascii=False))
            if changed:
                path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                return True
        return found


def record_recommendation(report) -> Optional[dict[str, Any]]:
    """
    Convenience wrapper used by the analysis pipeline.

    Never raises: logging a recommendation is bookkeeping, and bookkeeping
    must not be able to fail an analysis that already succeeded.
    """
    try:
        return RecommendationStore().record(report)
    except Exception as e:
        logger.warning(f"Recommendation logging failed (non-fatal): {e}")
        return None
