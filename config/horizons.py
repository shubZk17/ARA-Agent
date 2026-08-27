"""
config/horizons.py — Investment horizons and risk profiles, as data
====================================================================

WHY THIS EXISTS:
    Until Phase 6 every ARA report answered an undated question. "BUY" with
    no holding period cannot be acted on and cannot be graded — the same
    verdict is right at 5 years and wrong at 5 weeks. A horizon is what turns
    a description of a company into a strategy.

    The two horizons weight the SAME evidence differently:

        SHORT_TERM  trend, momentum, realized volatility, news flow
        LONG_TERM   valuation vs. durable growth, margins, balance sheet

DESIGN — a declarative table, not an abstraction:
    Deliberately mirrors METRIC_DEFINITIONS (analysis/financial_engine.py:71).
    Adding MEDIUM_TERM later is a data addition to HORIZON_PROFILES; no code
    anywhere needs to change.

WHY .value STRINGS EVERYWHERE DOWNSTREAM:
    AgentState carries `horizon` as the enum's *string* value, never the enum
    object. Episodic memory serializes state to JSON and LangGraph
    checkpointers pickle it; a bare string survives both round trips.

HOW IT CONNECTS:
    - agent/prompts.py injects profile.prompt_directive into the system prompt.
    - analysis/financial_engine.py applies category_weights and
      metric_threshold_overrides.
    - analysis/technical_engine.py sizes its lookback from lookback_days.
    - analysis/engine.py resolves the profile once and threads it everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InvestmentHorizon(str, Enum):
    """How long the recommendation is meant to be held."""
    SHORT_TERM = "short_term"   # 1 week – 3 months (swing)
    MEDIUM_TERM = "medium_term" # 3 months – 1 year (earnings-cycle)
    LONG_TERM = "long_term"     # 1 – 5 years


class RiskProfile(str, Enum):
    """How much drawdown the holder is willing to sit through."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


# The seven evidence categories a thesis can be built from. The first five
# match METRIC_DEFINITIONS' `category` values exactly, so financial weights
# can be looked up by an insight's own category with no mapping table.
CATEGORIES = (
    "valuation",
    "profitability",
    "growth",
    "liquidity",
    "leverage",
    "technical",
    "sentiment",
)

FINANCIAL_CATEGORIES = CATEGORIES[:5]


@dataclass(frozen=True)
class HorizonProfile:
    """Everything that changes when the holding period changes."""

    horizon: str
    label: str
    lookback_days: int                       # of price history the technicals need
    category_weights: dict[str, float]       # must cover CATEGORIES, sums to 1.0
    metric_threshold_overrides: dict[str, tuple[float, float]]
    required_tools: tuple[str, ...]
    sentiment_decay_days: int                # news older than this stops counting
    target_holding_period_text: str
    review_days: int                         # when the recommendation must be revisited
    prompt_directive: str                    # injected verbatim into the system prompt

    def weight(self, category: str) -> float:
        return self.category_weights.get(category, 0.0)

    def thresholds_for(
        self, metric_key: str, default: tuple[float, float]
    ) -> tuple[float, float]:
        """Horizon-specific thresholds, falling back to the metric's own."""
        return self.metric_threshold_overrides.get(metric_key, default)


HORIZON_PROFILES: dict[str, HorizonProfile] = {
    InvestmentHorizon.SHORT_TERM.value: HorizonProfile(
        horizon=InvestmentHorizon.SHORT_TERM.value,
        label="Short term (1 week – 3 months)",
        lookback_days=180,
        # Over weeks, price pays almost no attention to book value. Trend and
        # flow dominate; fundamentals act as a floor, not a driver.
        category_weights={
            "valuation": 0.08,
            "profitability": 0.07,
            "growth": 0.08,
            "liquidity": 0.05,
            "leverage": 0.07,
            "technical": 0.45,
            "sentiment": 0.20,
        },
        # A rich multiple is not a short-term sell signal — momentum names
        # stay expensive for quarters. Loosened so P/E stops dominating the
        # weaknesses list on a 6-week call.
        metric_threshold_overrides={
            "trailing_pe": (25.0, 60.0),
            "forward_pe": (20.0, 50.0),
            "price_to_book": (3.0, 20.0),
            "price_to_sales": (4.0, 25.0),
        },
        # get_financial_metrics is required on BOTH horizons even though the
        # short one barely weights fundamentals: analysis/engine.py abstains
        # below MIN_METRICS_FOR_RECOMMENDATION regardless of horizon, so
        # omitting it here produced a run that gathered exactly what this
        # table asked for and then refused to issue a recommendation.
        required_tools=(
            "get_stock_price", "get_price_history", "get_news",
            "get_financial_metrics",
        ),
        sentiment_decay_days=14,
        target_holding_period_text="1 week to 3 months",
        review_days=14,
        prompt_directive=(
            "HORIZON: SHORT TERM (1 week to 3 months, swing trade).\n"
            "   Weight recent price action, trend, momentum, realized volatility and "
            "current news flow most heavily. Valuation matters only as a guard against "
            "extremes. You MUST call get_price_history and get_news for this horizon. "
            "State a specific entry condition and the level at which the trade is wrong."
        ),
    ),
    InvestmentHorizon.MEDIUM_TERM.value: HorizonProfile(
        horizon=InvestmentHorizon.MEDIUM_TERM.value,
        label="Medium term (3 months – 1 year)",
        lookback_days=365,
        # Earnings-cycle driven: neither pure trend (short) nor pure
        # multi-year compounding (long). Technical still matters — a name
        # can be fundamentally sound and still be mid-drawdown for a year —
        # but fundamentals now carry real weight because a quarter or two of
        # results will actually land inside the holding period.
        category_weights={
            "valuation": 0.15,
            "profitability": 0.15,
            "growth": 0.15,
            "liquidity": 0.07,
            "leverage": 0.08,
            "technical": 0.25,
            "sentiment": 0.15,
        },
        metric_threshold_overrides={
            "trailing_pe": (22.0, 50.0),
            "price_to_book": (3.5, 15.0),
        },
        # Deliberately not a superset/subset of either neighbor's set: swaps
        # short_term's get_news for get_market_context (relative performance
        # and beta matter more over a multi-quarter hold than one news cycle)
        # and swaps long_term's get_company_info for the same tool (a year
        # is too short for competitive-position writeups to matter, but long
        # enough that how the stock has moved against its sector does).
        required_tools=(
            "get_stock_price", "get_price_history",
            "get_financial_metrics", "get_market_context",
        ),
        sentiment_decay_days=45,
        target_holding_period_text="3 months to 1 year",
        review_days=60,
        prompt_directive=(
            "HORIZON: MEDIUM TERM (3 months to 1 year, earnings-cycle hold).\n"
            "   Weight technical positioning and near-term fundamentals roughly "
            "equally — a name can be cheap and still be in a drawdown, and a "
            "strong chart does not survive a bad quarter. At least one earnings "
            "cycle will land inside the holding period, so weigh recent results "
            "and guidance more than a single news event or a five-year growth "
            "story. You MUST call get_financial_metrics and get_market_context "
            "for this horizon."
        ),
    ),
    InvestmentHorizon.LONG_TERM.value: HorizonProfile(
        horizon=InvestmentHorizon.LONG_TERM.value,
        label="Long term (1 – 5 years)",
        lookback_days=730,
        # Over years the entry chart is noise; what compounds is margin,
        # growth durability and the price paid for them.
        category_weights={
            "valuation": 0.22,
            "profitability": 0.22,
            "growth": 0.20,
            "liquidity": 0.08,
            "leverage": 0.12,
            "technical": 0.06,
            "sentiment": 0.10,
        },
        # A high trailing P/E is tolerable over five years IF growth is
        # durable — the growth weight above is what pays for it.
        metric_threshold_overrides={
            "trailing_pe": (20.0, 45.0),
        },
        # Price history is required here too. A five-year thesis does not
        # depend on the entry chart, but it should still know whether it is
        # recommending a purchase at the 52-week high or in a 40% drawdown —
        # and the long-horizon invalidation condition is written against the
        # 200-day SMA, which does not exist without it.
        required_tools=(
            "get_financial_metrics",
            "get_stock_price",
            "get_company_info",
            "get_price_history",
        ),
        sentiment_decay_days=90,
        target_holding_period_text="1 to 5 years",
        review_days=90,
        prompt_directive=(
            "HORIZON: LONG TERM (1 to 5 years).\n"
            "   Weight valuation relative to durable growth, margin quality, balance "
            "sheet strength and competitive position most heavily. Recent price swings "
            "and single news cycles are largely noise. You MUST call "
            "get_financial_metrics and get_company_info for this horizon."
        ),
    ),
}


@dataclass(frozen=True)
class RiskProfileConfig:
    """How the holder's risk appetite bends the final verdict."""

    profile: str
    label: str
    risk_penalty_multiplier: float   # scales the outlook's risk deduction
    min_confidence_to_act: float     # below this, downgrade to HOLD
    position_note: str


RISK_PROFILES: dict[str, RiskProfileConfig] = {
    RiskProfile.CONSERVATIVE.value: RiskProfileConfig(
        profile=RiskProfile.CONSERVATIVE.value,
        label="Conservative",
        risk_penalty_multiplier=1.5,
        min_confidence_to_act=0.65,
        position_note="Size small; prefer capital preservation over participation.",
    ),
    RiskProfile.BALANCED.value: RiskProfileConfig(
        profile=RiskProfile.BALANCED.value,
        label="Balanced",
        risk_penalty_multiplier=1.0,
        min_confidence_to_act=0.50,
        position_note="Standard position sizing.",
    ),
    RiskProfile.AGGRESSIVE.value: RiskProfileConfig(
        profile=RiskProfile.AGGRESSIVE.value,
        label="Aggressive",
        risk_penalty_multiplier=0.6,
        min_confidence_to_act=0.35,
        position_note="Tolerates drawdown for upside; still respects invalidation levels.",
    ),
}


DEFAULT_HORIZON = InvestmentHorizon.LONG_TERM.value
DEFAULT_RISK_PROFILE = RiskProfile.BALANCED.value


def get_horizon_profile(horizon: str | None) -> HorizonProfile:
    """
    Resolve a horizon string to its profile, falling back to the default.

    Unknown values fall back rather than raise: horizon arrives from CLI
    flags, HTTP bodies and replayed episodic state, and an unrecognised one
    must not take down an otherwise valid analysis.
    """
    return HORIZON_PROFILES.get(
        (horizon or "").strip().lower(), HORIZON_PROFILES[DEFAULT_HORIZON]
    )


def get_risk_profile(risk_profile: str | None) -> RiskProfileConfig:
    """Resolve a risk-profile string to its config, falling back to balanced."""
    return RISK_PROFILES.get(
        (risk_profile or "").strip().lower(), RISK_PROFILES[DEFAULT_RISK_PROFILE]
    )
