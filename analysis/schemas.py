"""
synthesis/schemas.py — Phase 3 Data Models
============================================

WHY THIS EXISTS:
    Every Phase 3 engine (financial, sentiment, risk, misalignment,
    confidence) produces structured output. Without shared schemas:
    - Engines pass ad-hoc dicts → fragile, untestable.
    - The report generator guesses field names → runtime errors.
    - Confidence calibration can't introspect engine outputs.

    These Pydantic models are the CONTRACT between engines and the
    report generator. If an engine produces it, the schema validates it.

DESIGN DECISIONS:
    1. Each model is self-serializing (to_dict, to_markdown).
    2. All scores are 0.0–1.0 normalized for consistent calibration.
    3. Every insight carries an evidence_sources list for citation.
    4. Models are IMMUTABLE after creation (frozen=False for Pydantic
       compat, but treated as immutable by convention).

HOW IT CONNECTS:
    - synthesis/financial_engine.py → FinancialSnapshot
    - synthesis/sentiment_analyzer.py → SentimentProfile
    - synthesis/misalignment_detector.py → MisalignmentSignal
    - synthesis/risk_analyzer.py → RiskAssessment
    - synthesis/confidence_calibrator.py → ConfidenceScore
    - synthesis/engine.py → SynthesisReport (aggregates all above)
    - synthesis/report_generator.py consumes SynthesisReport
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ===================================================================
# Enums
# ===================================================================

class SentimentDirection(str, Enum):
    """Sentiment polarity classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class RiskSeverity(str, Enum):
    """How severe a risk factor is."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MisalignmentType(str, Enum):
    """Categories of sentiment-financial misalignment."""
    BULLISH_WEAK_FUNDAMENTALS = "bullish_sentiment_weak_fundamentals"
    BEARISH_STRONG_FUNDAMENTALS = "bearish_sentiment_strong_fundamentals"
    OVERVALUATION_VS_OPTIMISM = "overvaluation_despite_optimism"
    GROWTH_NARRATIVE_DECLINING_MARGINS = "growth_narrative_declining_margins"
    NONE = "none"


class InvestmentOutlook(str, Enum):
    """Final investment recommendation."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    INSUFFICIENT_DATA = "insufficient_data"


# ===================================================================
# Financial Snapshot — Output of Financial Analysis Engine
# ===================================================================

class MetricInsight(BaseModel):
    """A single financial metric with context and assessment."""
    name: str = Field(description="Metric name (e.g., 'Revenue Growth')")
    value: Optional[float] = Field(default=None, description="Numeric value")
    formatted: str = Field(default="N/A", description="Human-readable value")
    assessment: str = Field(
        default="",
        description="Contextual interpretation (e.g., 'Strong — above industry avg')"
    )
    category: str = Field(
        default="general",
        description="Category: valuation, profitability, growth, liquidity, leverage"
    )
    evidence_sources: list[str] = Field(
        default_factory=list,
        description="Which tools/sources provided this data"
    )


class FinancialSnapshot(BaseModel):
    """
    Comprehensive financial profile of a company.

    WHY this structure?
    - Metrics are grouped by category for the report.
    - Each metric carries its own assessment (not just a number).
    - The overall_assessment synthesizes across categories.
    - evidence_quality reflects how complete the data is.
    """
    ticker: str = Field(default="")
    company_name: str = Field(default="")

    # --- Categorized Metrics ---
    valuation_metrics: list[MetricInsight] = Field(default_factory=list)
    profitability_metrics: list[MetricInsight] = Field(default_factory=list)
    growth_metrics: list[MetricInsight] = Field(default_factory=list)
    liquidity_metrics: list[MetricInsight] = Field(default_factory=list)
    leverage_metrics: list[MetricInsight] = Field(default_factory=list)

    # --- Overall Assessments ---
    financial_health_score: float = Field(
        default=0.5, description="0.0 (very weak) to 1.0 (very strong)"
    )
    overall_assessment: str = Field(default="")
    key_strengths: list[str] = Field(default_factory=list)
    key_weaknesses: list[str] = Field(default_factory=list)

    # --- Data Quality ---
    metrics_available: int = Field(default=0)
    metrics_missing: int = Field(default=0)
    evidence_quality: float = Field(
        default=0.5, description="0.0 (no data) to 1.0 (complete data)"
    )

    @property
    def all_metrics(self) -> list[MetricInsight]:
        return (
            self.valuation_metrics
            + self.profitability_metrics
            + self.growth_metrics
            + self.liquidity_metrics
            + self.leverage_metrics
        )


# ===================================================================
# Sentiment Profile — Output of Sentiment Analyzer
# ===================================================================

class SentimentSignal(BaseModel):
    """A single sentiment signal from a text source."""
    source: str = Field(description="Source name (e.g., 'Yahoo Finance')")
    text_snippet: str = Field(default="", description="Relevant excerpt")
    direction: SentimentDirection = Field(default=SentimentDirection.NEUTRAL)
    confidence: float = Field(
        default=0.5, description="0.0–1.0 confidence in classification"
    )
    positive_keywords: list[str] = Field(default_factory=list)
    negative_keywords: list[str] = Field(default_factory=list)


class SentimentProfile(BaseModel):
    """
    Aggregated sentiment analysis across all text sources.

    WHY aggregate?
    - Individual news headlines are noisy.
    - The agent needs the overall sentiment DIRECTION plus
      how CONFIDENT we are in that direction.
    - Divergent signals (mixed sentiment) are themselves
      valuable information for risk analysis.
    """
    overall_direction: SentimentDirection = Field(
        default=SentimentDirection.NEUTRAL
    )
    overall_confidence: float = Field(
        default=0.5, description="0.0–1.0, how sure we are"
    )

    signals: list[SentimentSignal] = Field(default_factory=list)
    bullish_count: int = Field(default=0)
    bearish_count: int = Field(default=0)
    neutral_count: int = Field(default=0)

    dominant_themes: list[str] = Field(
        default_factory=list,
        description="Key themes from news (e.g., 'AI spending', 'tariff risk')"
    )
    summary: str = Field(default="")


# ===================================================================
# Misalignment Signal — Output of Misalignment Detector
# ===================================================================

class MisalignmentSignal(BaseModel):
    """
    Detected divergence between sentiment and financial fundamentals.

    THIS IS THE MANDATORY FEATURE:
    The system must explicitly detect when market sentiment
    contradicts financial reality:
    - Bullish sentiment + weak fundamentals → potential bubble risk
    - Bearish sentiment + strong fundamentals → potential undervaluation
    - Growth narrative + declining margins → narrative risk
    """
    detected: bool = Field(
        default=False, description="Whether a misalignment exists"
    )
    misalignment_type: MisalignmentType = Field(
        default=MisalignmentType.NONE
    )
    severity: RiskSeverity = Field(default=RiskSeverity.LOW)
    confidence: float = Field(
        default=0.0, description="How confident we are in the detection"
    )
    explanation: str = Field(
        default="No misalignment detected.",
        description="Human-readable explanation"
    )
    sentiment_evidence: str = Field(
        default="", description="What the sentiment says"
    )
    financial_evidence: str = Field(
        default="", description="What the financials say"
    )
    recommendation: str = Field(
        default="", description="What the analyst should consider"
    )


# ===================================================================
# Risk Assessment — Output of Risk Analyzer
# ===================================================================

class RiskItem(BaseModel):
    """A single identified risk factor."""
    category: str = Field(description="e.g., valuation, execution, macro")
    title: str = Field(description="Short risk name")
    description: str = Field(description="Detailed explanation")
    severity: RiskSeverity = Field(default=RiskSeverity.MEDIUM)
    probability: str = Field(default="moderate", description="low/moderate/high")
    evidence: str = Field(default="", description="Supporting evidence")
    mitigation: str = Field(default="", description="Possible mitigation")


class RiskAssessment(BaseModel):
    """
    Comprehensive risk profile.

    WHY separate risk categories?
    - Different risks require different responses.
    - Institutional reports separate financial risk from macro risk.
    - The agent can highlight which risks are evidence-backed
      vs. speculative.
    """
    overall_risk_level: RiskSeverity = Field(default=RiskSeverity.MEDIUM)
    risk_score: float = Field(
        default=0.5, description="0.0 (very safe) to 1.0 (very risky)"
    )
    risks: list[RiskItem] = Field(default_factory=list)
    summary: str = Field(default="")

    @property
    def critical_risks(self) -> list[RiskItem]:
        return [r for r in self.risks if r.severity == RiskSeverity.CRITICAL]

    @property
    def high_risks(self) -> list[RiskItem]:
        return [r for r in self.risks if r.severity == RiskSeverity.HIGH]


# ===================================================================
# Confidence Score — Output of Confidence Calibrator
# ===================================================================

class ConfidenceScore(BaseModel):
    """
    Calibrated confidence in the overall analysis.

    WHY calibrate?
    - Prevents overconfident recommendations when evidence is thin.
    - Reflects uncertainty explicitly.
    - Adjusts DOWN for conflicts, missing data, stale evidence.
    - Adjusts UP for consistent multi-source evidence.
    """
    overall: float = Field(
        default=0.5, description="0.0–1.0 overall confidence"
    )
    label: str = Field(
        default="Moderate",
        description="Human label: Very Low / Low / Moderate / High / Very High"
    )

    # --- Component Scores ---
    evidence_quality: float = Field(default=0.5)
    source_reliability: float = Field(default=0.5)
    data_completeness: float = Field(default=0.5)
    consistency: float = Field(
        default=0.5, description="1.0 = no conflicts, 0.0 = many conflicts"
    )
    recency: float = Field(
        default=0.5, description="1.0 = all fresh, 0.0 = all stale"
    )

    # --- Adjustment Factors ---
    penalties: list[str] = Field(
        default_factory=list,
        description="Reasons confidence was reduced"
    )
    boosts: list[str] = Field(
        default_factory=list,
        description="Reasons confidence was increased"
    )

    explanation: str = Field(default="")


# ===================================================================
# Synthesis Report — Unified Output of Synthesis Engine
# ===================================================================

class SynthesisReport(BaseModel):
    """
    The FINAL output of Phase 3: everything the report generator
    needs to produce an institutional-quality investment report.

    This is the single object that aggregates all engine outputs.
    """
    id: str = Field(
        default_factory=lambda: f"report_{uuid.uuid4().hex[:10]}"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --- Subject ---
    ticker: str = Field(default="")
    company_name: str = Field(default="")
    query: str = Field(default="")

    # --- Engine Outputs ---
    financial: FinancialSnapshot = Field(default_factory=FinancialSnapshot)
    sentiment: SentimentProfile = Field(default_factory=SentimentProfile)
    misalignment: MisalignmentSignal = Field(default_factory=MisalignmentSignal)
    risk: RiskAssessment = Field(default_factory=RiskAssessment)
    confidence: ConfidenceScore = Field(default_factory=ConfidenceScore)

    # --- Final Synthesis ---
    investment_thesis: str = Field(
        default="", description="Multi-paragraph investment argument"
    )
    outlook: InvestmentOutlook = Field(
        default=InvestmentOutlook.INSUFFICIENT_DATA
    )
    key_findings: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)

    # --- Metadata ---
    tools_used: list[str] = Field(default_factory=list)
    iterations_used: int = Field(default=0)
    evidence_count: int = Field(default=0)
    execution_time_seconds: float = Field(default=0.0)
    model_used: str = Field(default="")
