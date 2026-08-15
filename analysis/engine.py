"""
synthesis/engine.py — Central Synthesis Orchestrator
======================================================

WHY THIS EXISTS:
    This is the CONDUCTOR. Individual engines (financial, sentiment,
    risk, misalignment, confidence) each analyze one dimension.
    The synthesis engine:

    1. ORCHESTRATES all engines in the correct dependency order.
    2. PASSES outputs between engines (misalignment needs financial + sentiment).
    3. SYNTHESIZES a unified investment thesis from all outputs.
    4. DETERMINES the final investment outlook (buy/hold/sell).
    5. PRODUCES a SynthesisReport ready for the report generator.

DEPENDENCY ORDER:
    1. Financial Engine (no dependencies)
    2. Sentiment Analyzer (no dependencies)
    3. Misalignment Detector (needs financial + sentiment)
    4. Risk Analyzer (needs financial + sentiment + misalignment)
    5. Confidence Calibrator (needs ALL above + agent state)
    6. Investment Thesis (synthesizes everything)

WHY NOT just call engines independently?
    - Misalignment detection REQUIRES both financial and sentiment outputs.
    - Risk analysis REQUIRES misalignment detection output.
    - Confidence REQUIRES knowing about all risks and conflicts.
    - The thesis REQUIRES all of the above.
    This is a DAG, not independent parallel tasks.

HOW IT CONNECTS:
    - Consumes: AgentState (from LangGraph, post-execution).
    - Uses: All 5 synthesis engines.
    - Produces: SynthesisReport (synthesis/schemas.py).
    - Used by: main.py (after graph.invoke()), report_generator.py.

DESIGN PRINCIPLE:
    This module is a THIN orchestrator — it delegates to engines
    and assembles their outputs. It does NOT contain analysis logic.
"""

from __future__ import annotations

import time
from typing import Any

from config.settings import settings
from analysis.confidence_calibrator import ConfidenceCalibrator
from analysis.financial_engine import FinancialAnalysisEngine
from analysis.misalignment_detector import MisalignmentDetector
from analysis.risk_analyzer import RiskAnalyzer
from analysis.schemas import (
    InvestmentOutlook,
    RiskSeverity,
    SentimentDirection,
    SynthesisReport,
)
from analysis.sentiment_analyzer import SentimentAnalyzer
from utils.logger import get_logger

logger = get_logger(__name__)


class SynthesisEngine:
    """
    Central orchestrator for Phase 3 synthesis pipeline.

    Takes raw agent state (from Phase 1+2 graph execution) and
    produces a structured SynthesisReport.
    """

    def __init__(self) -> None:
        # Initialize all sub-engines
        self._financial = FinancialAnalysisEngine()
        self._sentiment = SentimentAnalyzer()
        self._misalignment = MisalignmentDetector()
        self._risk = RiskAnalyzer()
        self._confidence = ConfidenceCalibrator()

        logger.info("Synthesis engine initialized with all sub-engines")

    def synthesize(self, agent_state: dict) -> SynthesisReport:
        """
        Run the full synthesis pipeline on completed agent state.

        Args:
            agent_state: Final state dict from graph.invoke().
                Must contain: tool_calls, reasoning_trace, query, etc.

        Returns:
            SynthesisReport with all engine outputs and investment thesis.
        """
        start = time.time()

        # Extract inputs from state
        tool_calls = agent_state.get("tool_calls", [])
        reasoning_trace = agent_state.get("reasoning_trace", [])
        observations = [
            step.observation
            for step in reasoning_trace
            if hasattr(step, "observation") and step.observation
        ]

        logger.info("Starting synthesis pipeline...")

        # --- Step 1: Financial Analysis ---
        logger.info("[1/6] Running financial analysis engine...")
        financial = self._financial.analyze(tool_calls, observations)

        # --- Step 2: Sentiment Analysis ---
        logger.info("[2/6] Running sentiment analyzer...")
        sentiment = self._sentiment.analyze(tool_calls, observations)

        # --- Step 3: Misalignment Detection ---
        logger.info("[3/6] Running misalignment detector...")
        misalignment = self._misalignment.detect(financial, sentiment)

        # --- Step 4: Risk Analysis ---
        logger.info("[4/6] Running risk analyzer...")
        risk = self._risk.analyze(financial, sentiment, misalignment)

        # --- Step 5: Confidence Calibration ---
        logger.info("[5/6] Calibrating confidence...")
        confidence = self._confidence.calibrate(
            financial=financial,
            sentiment=sentiment,
            risk=risk,
            misalignment=misalignment,
            agent_state=agent_state,
        )

        # --- Step 6: Investment Thesis & Outlook ---
        logger.info("[6/6] Generating investment thesis...")
        outlook = self._determine_outlook(financial, sentiment, risk, misalignment, confidence)
        thesis = self._generate_thesis(financial, sentiment, risk, misalignment, confidence, outlook)
        key_findings = self._extract_key_findings(financial, sentiment, misalignment, risk)
        contradictions = self._extract_contradictions(agent_state, misalignment)

        # --- Assemble Report ---
        elapsed = time.time() - start

        tools_used = list(set(
            tc.tool_name for tc in tool_calls
            if hasattr(tc, "tool_name")
        ))

        report = SynthesisReport(
            ticker=financial.ticker,
            company_name=financial.company_name,
            query=agent_state.get("query", ""),
            financial=financial,
            sentiment=sentiment,
            misalignment=misalignment,
            risk=risk,
            confidence=confidence,
            investment_thesis=thesis,
            outlook=outlook,
            key_findings=key_findings,
            contradictions=contradictions,
            tools_used=tools_used,
            iterations_used=agent_state.get("iteration_count", 0),
            evidence_count=len(agent_state.get("retrieved_evidence", [])),
            execution_time_seconds=round(elapsed, 2),
            model_used=settings.active_model,
        )

        logger.info(
            f"Synthesis complete: {financial.ticker} → "
            f"{outlook.value} (confidence={confidence.overall:.0%}) "
            f"in {elapsed:.1f}s"
        )
        return report

    def _determine_outlook(
        self,
        financial,
        sentiment,
        risk,
        misalignment,
        confidence,
    ) -> InvestmentOutlook:
        """
        Determine the investment outlook based on all analysis dimensions.

        This is the FINAL decision. It's a weighted consideration of:
        - Financial health (strongest signal)
        - Sentiment alignment
        - Risk severity
        - Confidence level
        """
        # Insufficient data → can't recommend
        if confidence.overall < 0.25:
            return InvestmentOutlook.INSUFFICIENT_DATA

        # Score-based determination
        # Start with financial health as base
        score = financial.financial_health_score  # 0-1

        # Sentiment alignment bonus/penalty
        if sentiment.overall_direction == SentimentDirection.BULLISH:
            score += 0.05
        elif sentiment.overall_direction == SentimentDirection.BEARISH:
            score -= 0.05

        # Risk penalty
        risk_penalty = {
            RiskSeverity.LOW: 0.0,
            RiskSeverity.MEDIUM: -0.05,
            RiskSeverity.HIGH: -0.10,
            RiskSeverity.CRITICAL: -0.20,
        }
        score += risk_penalty.get(risk.overall_risk_level, 0)

        # Misalignment penalty
        if misalignment.detected:
            score -= 0.05

        # Map score to outlook
        if score >= 0.75:
            return InvestmentOutlook.STRONG_BUY
        elif score >= 0.60:
            return InvestmentOutlook.BUY
        elif score >= 0.40:
            return InvestmentOutlook.HOLD
        elif score >= 0.25:
            return InvestmentOutlook.SELL
        else:
            return InvestmentOutlook.STRONG_SELL

    def _generate_thesis(
        self,
        financial,
        sentiment,
        risk,
        misalignment,
        confidence,
        outlook,
    ) -> str:
        """Generate a multi-paragraph investment thesis."""
        paragraphs = []

        # Opening — State the conclusion
        outlook_text = {
            InvestmentOutlook.STRONG_BUY: "represents a compelling investment opportunity",
            InvestmentOutlook.BUY: "presents a favorable risk-reward profile",
            InvestmentOutlook.HOLD: "warrants a hold position pending further developments",
            InvestmentOutlook.SELL: "faces headwinds that warrant caution",
            InvestmentOutlook.STRONG_SELL: "presents significant downside risk",
            InvestmentOutlook.INSUFFICIENT_DATA: "cannot be fully assessed due to limited data",
        }
        paragraphs.append(
            f"{financial.company_name or financial.ticker} "
            f"({financial.ticker}) {outlook_text.get(outlook, 'requires further analysis')}. "
            f"This assessment is based on analysis of financial fundamentals, "
            f"market sentiment, and risk factors, with an overall confidence "
            f"of {confidence.overall:.0%} ({confidence.label})."
        )

        # Financial summary
        if financial.key_strengths or financial.key_weaknesses:
            fin_parts = []
            if financial.key_strengths:
                fin_parts.append(
                    f"Financial strengths include {', '.join(financial.key_strengths[:3])}"
                )
            if financial.key_weaknesses:
                fin_parts.append(
                    f"areas of concern include {', '.join(financial.key_weaknesses[:3])}"
                )
            paragraphs.append(
                f"From a fundamental perspective, {financial.overall_assessment} "
                + "; ".join(fin_parts) + "."
            )

        # Sentiment
        paragraphs.append(sentiment.summary)

        # Misalignment warning
        if misalignment.detected:
            paragraphs.append(
                f"⚠ MISALIGNMENT ALERT: {misalignment.explanation} "
                f"{misalignment.recommendation}"
            )

        # Risk summary
        if risk.risks:
            paragraphs.append(risk.summary)

        # Closing
        paragraphs.append(
            f"Given the analysis, the recommendation is "
            f"{outlook.value.replace('_', ' ').upper()} with "
            f"{confidence.label.lower()} confidence. "
            f"Investors should monitor key risks and reassess "
            f"as new data becomes available."
        )

        return "\n\n".join(paragraphs)

    def _extract_key_findings(self, financial, sentiment, misalignment, risk) -> list[str]:
        """Extract the most important findings for the executive summary."""
        findings = []

        if financial.key_strengths:
            findings.append(f"Financial strength: {financial.key_strengths[0]}")
        if financial.key_weaknesses:
            findings.append(f"Financial concern: {financial.key_weaknesses[0]}")

        findings.append(f"Market sentiment: {sentiment.overall_direction.value}")

        if misalignment.detected:
            findings.append(
                f"Misalignment detected: {misalignment.misalignment_type.value}"
            )

        if risk.high_risks:
            findings.append(f"Key risk: {risk.high_risks[0].title}")

        if sentiment.dominant_themes:
            findings.append(f"Key themes: {', '.join(sentiment.dominant_themes[:3])}")

        return findings

    def _extract_contradictions(self, state: dict, misalignment) -> list[str]:
        """Extract all contradictions for the report."""
        contradictions = []

        # Phase 2 conflict reports
        for cr in state.get("conflict_reports", []):
            if isinstance(cr, dict):
                contradictions.append(cr.get("resolution", str(cr)))
            else:
                contradictions.append(str(cr))

        # Misalignment
        if misalignment.detected:
            contradictions.append(misalignment.explanation)

        return contradictions
