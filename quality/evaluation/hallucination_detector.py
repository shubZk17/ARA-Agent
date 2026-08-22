"""
evaluation/hallucination_detector.py — Hallucination Detection Engine
=======================================================================

WHY THIS EXISTS:
    LLMs hallucinate. In financial analysis, a hallucinated metric
    (e.g., "P/E ratio of 15.2" when the actual data says 28.4) can
    lead to catastrophic investment decisions.

    This module detects four types of hallucinations:
    1. FABRICATED METRICS — Numbers not present in any tool output.
    2. UNSUPPORTED CLAIMS — Assertions with no evidence backing.
    3. FAKE CITATIONS — References to sources that don't exist.
    4. UNJUSTIFIED CONFIDENCE — High confidence with thin evidence.

WHAT PROBLEM IT SOLVES:
    Allows ARA-1 to SELF-AUDIT its outputs before presenting them
    to users. Target: hallucination rate < 2%.

HOW IT INTEGRATES:
    - Reads from: agent state (tool_calls, reasoning_trace, final_answer).
    - Reads from: observability/tracer.py (execution trace).
    - Outputs to: evaluation/metrics.py (hallucination_rate metric).
    - Used by: evaluation/evaluator.py (master evaluator).

DESIGN DECISIONS:
    - Pattern-matching based (not LLM-based) to avoid "using the
      problem to solve the problem."
    - Conservative: better to flag a false positive than miss
      a real hallucination.
    - Each detector returns structured flags for the evaluation report.

SCALABILITY:
    - Add LLM-based cross-checking for higher accuracy (Phase 5).
    - Add external fact-checking APIs for market data.
    - Add historical comparison ("was this metric in range last quarter?").

PRODUCTION TRADEOFFS:
    - Pattern matching is fast but has false positives.
    - LLM cross-check is accurate but doubles token cost.
    - We default to pattern matching and flag for review.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HallucinationFlag:
    """
    A single detected (or suspected) hallucination.

    is_hallucination = True means CONFIRMED fabrication.
    is_hallucination = False means SUSPECTED but uncertain.
    """
    claim: str                      # The claim being checked
    claim_type: str                 # "metric", "assertion", "citation", "confidence"
    is_hallucination: bool          # Confirmed?
    confidence: float               # 0-1, how sure we are about the flag
    evidence_found: str             # What evidence we DID find (or "none")
    explanation: str                # Why we flagged this
    source_location: str = ""       # Where in the output the claim appeared
    severity: str = "medium"        # "low", "medium", "high", "critical"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HallucinationReport:
    """Complete hallucination analysis for one run."""
    total_claims_checked: int = 0
    hallucinations_found: int = 0
    suspected_hallucinations: int = 0
    hallucination_rate: float = 0.0
    flags: list[HallucinationFlag] = field(default_factory=list)
    clean_claims: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "total_claims_checked": self.total_claims_checked,
            "hallucinations_found": self.hallucinations_found,
            "suspected_hallucinations": self.suspected_hallucinations,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "clean_claims": self.clean_claims,
            "flags": [f.to_dict() for f in self.flags],
            "timestamp": self.timestamp,
        }


class HallucinationDetector:
    """
    Detects hallucinations in agent outputs by cross-referencing
    claims against actual tool evidence.

    Usage:
        detector = HallucinationDetector()
        report = detector.analyze(agent_state)
        print(f"Hallucination rate: {report.hallucination_rate:.1%}")
    """

    def __init__(self) -> None:
        # Patterns for extracting numeric claims from text
        self._number_pattern = re.compile(
            r'(?:(?:P/E|PE|EPS|revenue|market\s*cap|price|profit|margin|ratio|'
            r'growth|debt|ROE|ROA|EBITDA|dividend|yield|beta|volume|shares)'
            r'[^0-9$%]*?)'
            r'[\$]?\s*([0-9]+[,.]?[0-9]*(?:\.[0-9]+)?)\s*[%BMTKx]?',
            re.IGNORECASE
        )
        self._price_pattern = re.compile(
            r'\$\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
        )
        self._percentage_pattern = re.compile(
            r'([0-9]+(?:\.[0-9]+)?)\s*%',
        )

    def analyze(self, state: dict) -> HallucinationReport:
        """
        Run full hallucination analysis on completed agent state.

        Checks:
        1. Numeric claims in final_answer vs tool outputs
        2. Assertion coverage
        3. Citation validity
        4. Confidence justification
        """
        report = HallucinationReport()
        flags = []

        # Gather all evidence (tool outputs)
        evidence_text = self._gather_evidence(state)

        # Get the final answer
        final_answer = state.get("final_answer", "")
        if not final_answer:
            return report

        # --- Check 1: Fabricated Metrics ---
        metric_flags = self._check_fabricated_metrics(final_answer, evidence_text)
        flags.extend(metric_flags)

        # --- Check 2: Unsupported Claims ---
        claim_flags = self._check_unsupported_claims(final_answer, evidence_text)
        flags.extend(claim_flags)

        # --- Check 3: Fake Citations ---
        citation_flags = self._check_fake_citations(final_answer, state)
        flags.extend(citation_flags)

        # --- Check 4: Unjustified Confidence ---
        confidence_flags = self._check_unjustified_confidence(final_answer, state)
        flags.extend(confidence_flags)

        # Compute report
        report.flags = flags
        report.total_claims_checked = len(flags) if flags else 1
        report.hallucinations_found = sum(
            1 for f in flags if f.is_hallucination
        )
        report.suspected_hallucinations = sum(
            1 for f in flags if not f.is_hallucination and f.confidence > 0.5
        )
        report.clean_claims = report.total_claims_checked - report.hallucinations_found
        report.hallucination_rate = (
            report.hallucinations_found / max(report.total_claims_checked, 1)
        )

        logger.info(
            f"Hallucination analysis: {report.hallucinations_found} confirmed, "
            f"{report.suspected_hallucinations} suspected, "
            f"rate={report.hallucination_rate:.1%}"
        )

        return report

    def _gather_evidence(self, state: dict) -> str:
        """Collect all tool outputs as evidence text."""
        evidence_parts = []

        # From tool calls
        for tc in state.get("tool_calls", []):
            if hasattr(tc, "tool_output") and tc.tool_output:
                evidence_parts.append(tc.tool_output)
            elif hasattr(tc, "success") and tc.success:
                evidence_parts.append(str(tc))

        # From reasoning trace observations
        for step in state.get("reasoning_trace", []):
            if hasattr(step, "observation") and step.observation:
                evidence_parts.append(step.observation)

        return "\n".join(evidence_parts)

    def _extract_numbers(self, text: str) -> list[tuple[str, float]]:
        """Extract numeric values with their context from text."""
        results = []

        # Extract dollar amounts
        for match in self._price_pattern.finditer(text):
            raw = match.group(1).replace(",", "")
            try:
                value = float(raw)
                context = text[max(0, match.start() - 30):match.end() + 10]
                results.append((context.strip(), value))
            except ValueError:
                pass

        # Extract percentages
        for match in self._percentage_pattern.finditer(text):
            try:
                value = float(match.group(1))
                context = text[max(0, match.start() - 30):match.end() + 10]
                results.append((context.strip(), value))
            except ValueError:
                pass

        return results

    def _check_fabricated_metrics(
        self,
        answer: str,
        evidence: str,
    ) -> list[HallucinationFlag]:
        """Check if numeric claims in the answer appear in evidence."""
        flags = []

        answer_numbers = self._extract_numbers(answer)
        evidence_numbers = self._extract_numbers(evidence)
        evidence_values = {v for _, v in evidence_numbers}

        # Also extract all raw numbers from evidence for fuzzy matching
        raw_evidence_nums = set()
        for match in re.finditer(r'[0-9]+(?:\.[0-9]+)?', evidence):
            try:
                raw_evidence_nums.add(float(match.group()))
            except ValueError:
                pass

        for context, value in answer_numbers:
            # Check exact match
            if value in evidence_values or value in raw_evidence_nums:
                flags.append(HallucinationFlag(
                    claim=context,
                    claim_type="metric",
                    is_hallucination=False,
                    confidence=0.0,
                    evidence_found="Exact match in evidence",
                    explanation="Numeric value found in tool evidence",
                ))
                continue

            # Check approximate match (within 5% tolerance)
            found_approx = False
            for ev in evidence_values | raw_evidence_nums:
                if ev > 0 and abs(value - ev) / ev < 0.05:
                    found_approx = True
                    break

            if found_approx:
                flags.append(HallucinationFlag(
                    claim=context,
                    claim_type="metric",
                    is_hallucination=False,
                    confidence=0.2,
                    evidence_found="Approximate match in evidence",
                    explanation="Numeric value approximately matches evidence (within 5%)",
                ))
            else:
                # Value not found — potential fabrication
                # But some numbers (like dates, common values) are OK
                if self._is_common_number(value):
                    continue

                flags.append(HallucinationFlag(
                    claim=context,
                    claim_type="metric",
                    is_hallucination=True,
                    confidence=0.7,
                    evidence_found="No match found",
                    explanation=f"Value {value} not found in any tool evidence",
                    severity="high" if value > 1 else "medium",
                ))

        return flags

    def _is_common_number(self, value: float) -> bool:
        """Check if a number is too common to be a meaningful hallucination."""
        common = {0, 1, 2, 3, 4, 5, 10, 12, 20, 30, 50, 100, 52, 365, 2024, 2025, 2026}
        return value in common or (value > 1900 and value < 2100)  # Year

    def _check_unsupported_claims(
        self,
        answer: str,
        evidence: str,
    ) -> list[HallucinationFlag]:
        """Check for strong claims not supported by evidence."""
        flags = []

        # Patterns indicating strong claims
        strong_claim_patterns = [
            (r'significantly\s+(?:increased|decreased|outperformed|underperformed)',
             "Significance claim"),
            (r'(?:best|worst|highest|lowest|leading|top)\s+(?:in|among|performer)',
             "Superlative claim"),
            (r'(?:guaranteed|certain|definitely|absolutely|always|never)\s+',
             "Certainty claim"),
        ]

        evidence_lower = evidence.lower()
        for pattern, claim_type in strong_claim_patterns:
            for match in re.finditer(pattern, answer, re.IGNORECASE):
                claim_text = answer[max(0, match.start() - 20):match.end() + 30]
                # Check if similar language exists in evidence
                claim_words = set(match.group().lower().split())
                evidence_support = any(
                    word in evidence_lower for word in claim_words
                )

                if not evidence_support:
                    flags.append(HallucinationFlag(
                        claim=claim_text.strip(),
                        claim_type="assertion",
                        is_hallucination=False,
                        confidence=0.5,
                        evidence_found="No supporting language in evidence",
                        explanation=f"Strong claim ({claim_type}) without clear evidence support",
                        severity="medium",
                    ))

        return flags

    def _check_fake_citations(
        self,
        answer: str,
        state: dict,
    ) -> list[HallucinationFlag]:
        """Check for references to tools or sources that weren't actually used."""
        flags = []

        # Tools that were actually used
        actual_tools = set()
        for tc in state.get("tool_calls", []):
            if hasattr(tc, "tool_name"):
                actual_tools.add(tc.tool_name)

        # Check for tool name mentions in answer
        all_tools = {
            "get_stock_price", "get_company_info",
            "get_financial_metrics", "get_news",
        }

        for tool in all_tools - actual_tools:
            # Convert tool name to natural language for pattern matching
            tool_phrases = {
                "get_stock_price": ["stock price data", "price analysis"],
                "get_company_info": ["company profile", "company information"],
                "get_financial_metrics": ["financial metrics", "financial data", "financial statements"],
                "get_news": ["news analysis", "recent news", "news sentiment"],
            }

            for phrase in tool_phrases.get(tool, []):
                if phrase.lower() in answer.lower():
                    flags.append(HallucinationFlag(
                        claim=phrase,
                        claim_type="citation",
                        is_hallucination=False,
                        confidence=0.4,
                        evidence_found=f"Tool '{tool}' was not called",
                        explanation=f"Answer references '{phrase}' but {tool} was never called",
                        severity="low",
                    ))

        return flags

    def _check_unjustified_confidence(
        self,
        answer: str,
        state: dict,
    ) -> list[HallucinationFlag]:
        """Check if the answer expresses high confidence with insufficient evidence."""
        flags = []

        # Count evidence strength
        tool_calls = state.get("tool_calls", [])
        successful_calls = sum(
            1 for tc in tool_calls
            if hasattr(tc, "success") and tc.success
        )
        unique_tools = len(set(
            tc.tool_name for tc in tool_calls
            if hasattr(tc, "tool_name") and hasattr(tc, "success") and tc.success
        ))

        # High confidence patterns
        high_confidence_patterns = [
            r'(?:strongly|highly)\s+recommend',
            r'(?:strong\s+buy|strong\s+sell)',
            r'(?:very\s+confident|high\s+confidence)',
            r'(?:clear|obvious|evident)\s+(?:opportunity|risk)',
        ]

        # If we have thin evidence but strong language
        if successful_calls <= 1 or unique_tools <= 1:
            for pattern in high_confidence_patterns:
                match = re.search(pattern, answer, re.IGNORECASE)
                if match:
                    flags.append(HallucinationFlag(
                        claim=answer[max(0, match.start() - 20):match.end() + 20],
                        claim_type="confidence",
                        is_hallucination=False,
                        confidence=0.6,
                        evidence_found=f"Only {successful_calls} successful tool calls, {unique_tools} unique tools",
                        explanation="High-confidence language with limited evidence base",
                        severity="medium",
                    ))

        return flags
