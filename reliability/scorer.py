"""
reliability/scorer.py — Source Reliability Scoring Engine
==========================================================

WHY THIS EXISTS:
    Every piece of retrieved evidence needs a reliability score.
    This score determines how much the agent should TRUST the evidence
    when synthesizing its analysis.

    The scorer considers:
    1. SOURCE TIER — Is this an SEC filing or a Reddit post?
    2. STALENESS — Is this data from today or 3 months ago?
    3. SOURCE REPUTATION — Specific source name scoring within a tier.

SCORING FORMULA:
    reliability_score = base_tier_score × staleness_decay × source_modifier

    Where:
    - base_tier_score: Default score for the source's tier (0.0-1.0).
    - staleness_decay: Time-based decay factor (1.0 = fresh, 0.5 = very old).
    - source_modifier: Fine-tuning within a tier (1.0 = average for tier).

STALENESS HANDLING:
    Financial data has a shelf life:
    - Stock prices: stale after 1 day.
    - News: stale after 7 days.
    - Financial metrics: stale after 90 days.
    - Company info: stale after 365 days.

    Staleness doesn't make data WRONG, but it makes it LESS RELIABLE
    for current analysis. The decay function reflects this nuance.

HOW IT CONNECTS:
    - retrieval/retriever.py calls scorer.score() for each retrieved evidence.
    - reliability/tiers.py provides tier definitions and score ranges.
    - retrieval/schemas.py stores the reliability_score in RetrievedEvidence.

COMMON FAILURE MODES:
    1. Missing document_date → assume moderate staleness (not zero).
    2. Unknown source → assign lowest tier (conservative).
    3. Future dates → treat as current (clock sync issues).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from reliability.tiers import (
    TierConfig,
    get_tier_config,
    get_tier_for_key,
    TIER_UNKNOWN,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ===================================================================
# Staleness Configuration
# ===================================================================

# Maximum age (in days) before staleness decay begins, by document type
STALENESS_THRESHOLDS: dict[str, int] = {
    "financial_data": 1,        # Stock prices go stale fast
    "news_article": 7,          # News loses relevance within a week
    "tool_output": 1,           # Tool outputs are point-in-time
    "sec_filing": 90,           # Quarterly filings — 3 months
    "earnings_transcript": 90,
    "analyst_report": 30,
    "research_note": 30,
    "social_media": 3,          # Social media is very ephemeral
}

DEFAULT_STALENESS_THRESHOLD = 30  # Default: 30 days

# How aggressively staleness decays the score
# score_decay = max(0.5, 1.0 - (days_over_threshold / decay_period))
STALENESS_DECAY_PERIOD = 90  # Days over threshold to reach minimum decay
MINIMUM_STALENESS_MULTIPLIER = 0.5  # Floor — even stale data retains some value


class ReliabilityScorer:
    """
    Computes reliability scores for evidence sources.

    Combines tier-based base scores with staleness decay
    to produce a final reliability score (0.0-1.0).
    """

    def score(
        self,
        source_name: str,
        source_tier: str = "",
        document_date: str = "",
        source_type: str = "",
    ) -> float:
        """
        Compute reliability score for a piece of evidence.

        Args:
            source_name: Name of the source (e.g., "reuters", "get_stock_price").
            source_tier: Tier key (e.g., "tier_1"). Used as fallback if
                         source_name isn't in the registry.
            document_date: ISO timestamp of the document. Used for staleness.
            source_type: Document type (for staleness threshold lookup).

        Returns:
            Reliability score between 0.0 and 1.0.
        """
        # 1. Get tier config
        tier_config = get_tier_config(source_name)

        # If source_name wasn't found but we have a tier key, use that
        if tier_config == TIER_UNKNOWN and source_tier:
            tier_config = get_tier_for_key(source_tier)

        # 2. Base score from tier
        base_score = tier_config.default_score

        # 3. Staleness decay
        staleness_multiplier = self._compute_staleness_decay(
            document_date=document_date,
            source_type=source_type,
        )

        # 4. Final score
        final_score = round(base_score * staleness_multiplier, 4)

        # Clamp to [0.0, 1.0]
        final_score = max(0.0, min(1.0, final_score))

        logger.debug(
            f"Reliability score: {source_name} "
            f"(tier={tier_config.tier_key}, base={base_score}, "
            f"staleness={staleness_multiplier:.2f}) = {final_score}"
        )

        return final_score

    def is_stale(
        self,
        document_date: str,
        source_type: str = "",
    ) -> bool:
        """
        Determine if a document is considered stale.

        Returns True if the document's age exceeds the staleness
        threshold for its source type.
        """
        if not document_date:
            return False  # Can't determine — assume not stale

        age_days = self._compute_age_days(document_date)
        if age_days is None:
            return False

        threshold = STALENESS_THRESHOLDS.get(
            source_type, DEFAULT_STALENESS_THRESHOLD
        )

        return age_days > threshold

    def _compute_staleness_decay(
        self,
        document_date: str,
        source_type: str = "",
    ) -> float:
        """
        Compute the staleness decay multiplier.

        Returns:
            1.0 for fresh data, down to MINIMUM_STALENESS_MULTIPLIER for very old data.
        """
        if not document_date:
            # No date — assume moderate staleness (not zero, not fresh)
            return 0.8

        age_days = self._compute_age_days(document_date)
        if age_days is None:
            return 0.8

        threshold = STALENESS_THRESHOLDS.get(
            source_type, DEFAULT_STALENESS_THRESHOLD
        )

        if age_days <= threshold:
            return 1.0  # Fresh — no decay

        # Linear decay from 1.0 to MINIMUM_STALENESS_MULTIPLIER
        days_over = age_days - threshold
        decay = 1.0 - (days_over / STALENESS_DECAY_PERIOD)
        return max(MINIMUM_STALENESS_MULTIPLIER, decay)

    @staticmethod
    def _compute_age_days(document_date: str) -> Optional[float]:
        """Parse document date and compute age in days."""
        try:
            doc_dt = datetime.fromisoformat(document_date.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta = now - doc_dt

            # Handle future dates (clock sync issues)
            if delta.total_seconds() < 0:
                return 0.0

            return delta.total_seconds() / 86400  # Convert to days
        except (ValueError, TypeError):
            return None
