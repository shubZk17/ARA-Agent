"""
reliability/tiers.py — Source Reliability Tier Definitions
===========================================================

WHY THIS EXISTS:
    Not all financial information is equally trustworthy.
    An SEC filing is more reliable than a Reddit post.
    Without explicit reliability ranking, the agent treats all
    information as equally valid, which leads to hallucination
    when low-quality sources contradict high-quality ones.

TIER DEFINITIONS:

    TIER 1 — Primary Official Sources (0.85-1.0)
    ─────────────────────────────────────────────
    Sources that are legally required to be accurate or are
    directly from the company/exchange.
    - SEC filings (10-K, 10-Q, 8-K)
    - Official earnings reports
    - Exchange data (real-time stock prices)
    - Company press releases
    - ARA-1 tool outputs (direct API data)

    TIER 2 — Established Financial Media (0.60-0.84)
    ──────────────────────────────────────────────────
    Professional financial journalism with editorial standards.
    - Reuters, Bloomberg, WSJ, Financial Times
    - Yahoo Finance (curated content)
    - Analyst reports from major firms
    - Agent-generated research notes

    TIER 3 — Secondary/Informal Sources (0.30-0.59)
    ──────────────────────────────────────────────────
    Community-generated or unverified content.
    - Financial blogs (Seeking Alpha, Motley Fool)
    - Social media (Twitter, Reddit, StockTwits)
    - Forum discussions
    - Unverified news aggregators

WHY RELIABILITY SCORING MATTERS:
    1. HALLUCINATION MITIGATION — When conflicting information exists,
       higher-tier sources take precedence.
    2. TRUST CALIBRATION — The agent can express lower confidence
       when evidence comes from low-tier sources.
    3. EVIDENCE WEIGHTING — Combined relevance + reliability scores
       determine which evidence to prioritize.

HOW IT CONNECTS:
    - reliability/scorer.py uses these definitions to compute scores.
    - retrieval/retriever.py weights results by reliability.
    - retrieval/schemas.py stores tier information in evidence metadata.
"""

from __future__ import annotations

from dataclasses import dataclass


# ===================================================================
# Tier Configuration
# ===================================================================

@dataclass(frozen=True)
class TierConfig:
    """Configuration for a reliability tier."""
    name: str
    tier_key: str
    min_score: float
    max_score: float
    default_score: float
    description: str


TIER_1 = TierConfig(
    name="Tier 1 — Primary Official Sources",
    tier_key="tier_1",
    min_score=0.85,
    max_score=1.0,
    default_score=0.90,
    description="Legally mandated filings, exchange data, official company reports.",
)

TIER_2 = TierConfig(
    name="Tier 2 — Established Financial Media",
    tier_key="tier_2",
    min_score=0.60,
    max_score=0.84,
    default_score=0.70,
    description="Professional financial journalism with editorial standards.",
)

TIER_3 = TierConfig(
    name="Tier 3 — Secondary/Informal Sources",
    tier_key="tier_3",
    min_score=0.30,
    max_score=0.59,
    default_score=0.40,
    description="Community-generated or unverified content.",
)

TIER_UNKNOWN = TierConfig(
    name="Unknown Source",
    tier_key="unknown",
    min_score=0.10,
    max_score=0.29,
    default_score=0.25,
    description="Unrecognized source — treated as low reliability.",
)


# ===================================================================
# Source → Tier Mapping
# ===================================================================

# Maps specific source names to their tier configuration.
# Extend this as new sources are added.

SOURCE_TIER_REGISTRY: dict[str, TierConfig] = {
    # --- Tier 1 ---
    "sec_edgar": TIER_1,
    "company_filing": TIER_1,
    "10-k": TIER_1,
    "10-q": TIER_1,
    "8-k": TIER_1,
    "earnings_report": TIER_1,
    "get_stock_price": TIER_1,
    "get_company_info": TIER_1,
    "get_financial_metrics": TIER_1,

    # --- Tier 2 ---
    "reuters": TIER_2,
    "bloomberg": TIER_2,
    "wsj": TIER_2,
    "wall_street_journal": TIER_2,
    "financial_times": TIER_2,
    "yahoo_finance": TIER_2,
    "get_news": TIER_2,
    "cnbc": TIER_2,
    "barrons": TIER_2,
    "ara_agent": TIER_2,

    # --- Tier 3 ---
    "seeking_alpha": TIER_3,
    "motley_fool": TIER_3,
    "reddit": TIER_3,
    "twitter": TIER_3,
    "stocktwits": TIER_3,
    "blog": TIER_3,
    "investopedia": TIER_3,
}


def get_tier_config(source_name: str) -> TierConfig:
    """
    Look up the tier configuration for a given source name.

    Falls back to TIER_UNKNOWN for unrecognized sources.
    """
    return SOURCE_TIER_REGISTRY.get(source_name.lower(), TIER_UNKNOWN)


def get_tier_for_key(tier_key: str) -> TierConfig:
    """Look up tier config by tier key (tier_1, tier_2, tier_3, unknown)."""
    tier_map = {
        "tier_1": TIER_1,
        "tier_2": TIER_2,
        "tier_3": TIER_3,
        "unknown": TIER_UNKNOWN,
    }
    return tier_map.get(tier_key, TIER_UNKNOWN)
