"""
validation/ — recording recommendations now so they can be graded later.

Phase 8 scores recommendations against realized returns. Those outcomes
mature on wall-clock time and cannot be reconstructed after the fact: the
price at the moment of the call, and the invalidation level stated in
advance, exist only if something wrote them down at the time.

Hence this package ships in Phase 6, ahead of the scorer that will read it.
Every week it does not exist is a permanently missing week of ground truth.
"""

from validation.recommendation_store import (  # noqa: F401
    RecommendationStore,
    record_recommendation,
)
