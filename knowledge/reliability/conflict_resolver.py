"""
reliability/conflict_resolver.py — Contradiction Detection & Resolution Engine
================================================================================

WHY THIS EXISTS:
    Financial data frequently contradicts itself:
    - An SEC filing says revenue was $10B, but a news article says $9.8B.
    - Tool output shows stock up 5%, but news says market is down.
    - A fresh analyst report disagrees with a stale earnings transcript.

    The WORST thing an agent can do is silently pick one source and ignore
    the other. This module EXPLICITLY detects conflicts and reports them
    so the LLM can reason about uncertainty.

CONFLICT TYPES:
    1. CONTRADICTORY_CLAIM — Two sources make opposing factual claims.
       Example: "Revenue grew 15%" vs "Revenue declined 3%"

    2. INCONSISTENT_METRIC — Numeric values differ significantly.
       Example: P/E ratio of 25 vs P/E ratio of 45

    3. STALE_CONFLICT — A newer source contradicts an older one.
       Example: "Q3 earnings beat expectations" vs "Q2 earnings missed"

    4. SENTIMENT_CONFLICT — Sources disagree on outlook.
       Example: "Bullish on NVDA" vs "Bearish outlook for NVDA"

RESOLUTION STRATEGY:
    The resolver does NOT pick a winner. It:
    1. Detects that a conflict exists.
    2. Reports the conflict with both sides.
    3. Suggests which source is more reliable (based on tiers).
    4. Lets the LLM make the final judgment.

    WHY NOT auto-resolve?
    - Financial analysis requires nuance that rules can't capture.
    - The LLM might have additional context that helps resolution.
    - Transparency is more valuable than false certainty.

EVIDENCE ARBITRATION:
    When conflicts exist, the resolver provides a RECOMMENDED source
    based on: tier > recency > specificity.
    But it's a recommendation, not an override.

HOW IT CONNECTS:
    - retrieval/retriever.py calls check_conflicts() on retrieved evidence.
    - retrieval/schemas.py defines ConflictReport model.
    - The ConflictReport is surfaced in the prompt for LLM reasoning.
"""

from __future__ import annotations

import re
from typing import Optional

from knowledge.retrieval.schemas import ConflictReport, RetrievedEvidence
from config.logging import get_logger

logger = get_logger(__name__)


# ===================================================================
# Conflict Detection Patterns
# ===================================================================

# Financial metrics that should be compared numerically
NUMERIC_METRIC_PATTERNS = [
    r"P/E\s*(?:ratio)?[:=]?\s*(\d+(?:\.\d+)?)",
    r"(?:revenue|sales)\s*(?:of|was|is|:)?\s*\$?([\d,.]+)\s*(?:billion|million|B|M)?",
    r"(?:growth|change|return)\s*(?:of|was|is|:)?\s*([-+]?\d+(?:\.\d+)?)\s*%",
    r"(?:price|stock)\s*(?:at|is|was|:)?\s*\$?([\d,.]+)",
    r"(?:EPS|earnings per share)\s*(?:of|was|is|:)?\s*\$?([\d,.]+)",
    r"(?:margin|yield)\s*(?:of|was|is|:)?\s*([\d,.]+)\s*%",
]

# Sentiment keywords for sentiment conflict detection
POSITIVE_SENTIMENT = {
    "bullish", "positive", "growth", "beat", "exceeded", "strong",
    "optimistic", "upgrade", "outperform", "buy", "increase", "rise",
    "up", "gained", "surged", "rally",
}

NEGATIVE_SENTIMENT = {
    "bearish", "negative", "decline", "miss", "missed", "weak",
    "pessimistic", "downgrade", "underperform", "sell", "decrease",
    "drop", "fell", "plunged", "crash", "down",
}


class ConflictResolver:
    """
    Detects and reports conflicts between retrieved evidence.

    Does NOT silently resolve conflicts — instead surfaces them
    so the LLM can reason about uncertainty explicitly.
    """

    def __init__(
        self,
        numeric_tolerance: float = 0.15,  # 15% difference = conflict
        sentiment_threshold: int = 3,     # Need 3+ keywords for sentiment signal
    ) -> None:
        """
        Args:
            numeric_tolerance: Percentage difference threshold for numeric conflicts.
                A 15% tolerance means P/E of 45 and 52 would NOT conflict,
                but P/E of 25 and 45 WOULD conflict.
            sentiment_threshold: Minimum sentiment keywords to consider
                the text as expressing a sentiment direction.
        """
        self._numeric_tolerance = numeric_tolerance
        self._sentiment_threshold = sentiment_threshold

    def check_conflicts(
        self,
        evidence_list: list[RetrievedEvidence],
    ) -> list[RetrievedEvidence]:
        """
        Scan evidence list for pairwise conflicts.

        Modifies evidence items in-place to set has_conflict and
        conflict_details flags.

        Args:
            evidence_list: List of retrieved evidence to check.

        Returns:
            The same list with conflict flags set.
        """
        if len(evidence_list) < 2:
            return evidence_list

        conflicts_found = 0

        for i in range(len(evidence_list)):
            for j in range(i + 1, len(evidence_list)):
                conflict = self._detect_pairwise_conflict(
                    evidence_list[i],
                    evidence_list[j],
                )
                if conflict:
                    conflicts_found += 1
                    evidence_list[i].has_conflict = True
                    evidence_list[j].has_conflict = True

                    # Add conflict details to the less reliable source
                    if evidence_list[i].reliability_score >= evidence_list[j].reliability_score:
                        evidence_list[j].conflict_details = conflict.resolution
                    else:
                        evidence_list[i].conflict_details = conflict.resolution

        if conflicts_found > 0:
            logger.info(
                f"Detected {conflicts_found} conflict(s) "
                f"across {len(evidence_list)} evidence items"
            )

        return evidence_list

    def _detect_pairwise_conflict(
        self,
        evidence_a: RetrievedEvidence,
        evidence_b: RetrievedEvidence,
    ) -> Optional[ConflictReport]:
        """
        Check two pieces of evidence for conflicts.

        Checks in order:
        1. Numeric metric conflicts (strongest signal).
        2. Sentiment conflicts (weaker signal).
        3. Staleness-based conflicts.

        Returns ConflictReport if conflict found, None otherwise.
        """
        # 1. Numeric conflict
        numeric_conflict = self._check_numeric_conflict(evidence_a, evidence_b)
        if numeric_conflict:
            return numeric_conflict

        # 2. Sentiment conflict
        sentiment_conflict = self._check_sentiment_conflict(evidence_a, evidence_b)
        if sentiment_conflict:
            return sentiment_conflict

        # 3. Staleness conflict (if one is stale and the other isn't)
        if evidence_a.is_stale != evidence_b.is_stale:
            stale_ev = evidence_a if evidence_a.is_stale else evidence_b
            fresh_ev = evidence_b if evidence_a.is_stale else evidence_a

            # Only flag if they discuss similar content (same ticker)
            if stale_ev.ticker and stale_ev.ticker == fresh_ev.ticker:
                return ConflictReport(
                    evidence_a_id=evidence_a.chunk_id,
                    evidence_b_id=evidence_b.chunk_id,
                    evidence_a_summary=evidence_a.content[:100],
                    evidence_b_summary=evidence_b.content[:100],
                    conflict_type="stale_data",
                    resolution=(
                        f"Stale evidence from {stale_ev.source_name} "
                        f"(dated {stale_ev.document_date}) may conflict with "
                        f"fresher data from {fresh_ev.source_name}. "
                        f"Prefer the more recent source."
                    ),
                    confidence_in_resolution=0.7,
                )

        return None

    def _check_numeric_conflict(
        self,
        evidence_a: RetrievedEvidence,
        evidence_b: RetrievedEvidence,
    ) -> Optional[ConflictReport]:
        """
        Compare numeric metrics extracted from two evidence items.

        If the same metric appears in both but differs by more than
        the tolerance threshold, it's flagged as a conflict.
        """
        for pattern in NUMERIC_METRIC_PATTERNS:
            matches_a = re.findall(pattern, evidence_a.content, re.IGNORECASE)
            matches_b = re.findall(pattern, evidence_b.content, re.IGNORECASE)

            if not matches_a or not matches_b:
                continue

            try:
                # Take the first match from each
                val_a = float(matches_a[0].replace(",", ""))
                val_b = float(matches_b[0].replace(",", ""))

                if val_a == 0 and val_b == 0:
                    continue

                # Calculate relative difference
                avg = (abs(val_a) + abs(val_b)) / 2
                if avg == 0:
                    continue

                diff_ratio = abs(val_a - val_b) / avg

                if diff_ratio > self._numeric_tolerance:
                    # Determine which source to prefer
                    preferred = "neither"
                    if evidence_a.reliability_score > evidence_b.reliability_score:
                        preferred = evidence_a.source_name
                    elif evidence_b.reliability_score > evidence_a.reliability_score:
                        preferred = evidence_b.source_name

                    return ConflictReport(
                        evidence_a_id=evidence_a.chunk_id,
                        evidence_b_id=evidence_b.chunk_id,
                        evidence_a_summary=f"{evidence_a.source_name}: value={val_a}",
                        evidence_b_summary=f"{evidence_b.source_name}: value={val_b}",
                        conflict_type="inconsistent_metric",
                        resolution=(
                            f"Numeric discrepancy: {val_a} vs {val_b} "
                            f"({diff_ratio:.0%} difference). "
                            f"Preferred source: {preferred} "
                            f"(higher reliability score)."
                        ),
                        confidence_in_resolution=0.6,
                    )
            except (ValueError, IndexError):
                continue

        return None

    def _check_sentiment_conflict(
        self,
        evidence_a: RetrievedEvidence,
        evidence_b: RetrievedEvidence,
    ) -> Optional[ConflictReport]:
        """
        Detect opposing sentiment between two evidence items.

        Counts positive and negative sentiment keywords. If one is
        predominantly positive and the other predominantly negative,
        flag it as a sentiment conflict.
        """
        sentiment_a = self._classify_sentiment(evidence_a.content)
        sentiment_b = self._classify_sentiment(evidence_b.content)

        # Conflict only if sentiments are opposite
        if sentiment_a and sentiment_b and sentiment_a != sentiment_b:
            return ConflictReport(
                evidence_a_id=evidence_a.chunk_id,
                evidence_b_id=evidence_b.chunk_id,
                evidence_a_summary=f"{evidence_a.source_name}: {sentiment_a} sentiment",
                evidence_b_summary=f"{evidence_b.source_name}: {sentiment_b} sentiment",
                conflict_type="sentiment_conflict",
                resolution=(
                    f"Conflicting sentiment: {evidence_a.source_name} is {sentiment_a}, "
                    f"{evidence_b.source_name} is {sentiment_b}. "
                    f"Consider both perspectives in your analysis."
                ),
                confidence_in_resolution=0.4,  # Low — sentiment is subjective
            )

        return None

    def _classify_sentiment(self, text: str) -> Optional[str]:
        """
        Simple keyword-based sentiment classification.

        Returns "positive", "negative", or None (neutral/unclear).
        """
        words = set(text.lower().split())
        pos_count = len(words & POSITIVE_SENTIMENT)
        neg_count = len(words & NEGATIVE_SENTIMENT)

        if pos_count >= self._sentiment_threshold and pos_count > neg_count * 2:
            return "positive"
        if neg_count >= self._sentiment_threshold and neg_count > pos_count * 2:
            return "negative"
        return None
