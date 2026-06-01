"""
synthesis/sentiment_analyzer.py — News & Text Sentiment Analysis
================================================================

WHY THIS EXISTS:
    Financial analysis isn't just numbers — market SENTIMENT drives
    short-term price action and reveals information not in the metrics:
    - A company with great financials but terrible news may drop.
    - A company with weak earnings but "AI hype" may soar.
    - DIVERGENCE between sentiment and fundamentals is a signal.

    This module analyzes text from news, tool outputs, and retrieved
    evidence to produce a structured SentimentProfile.

WHAT THIS IS:
    A keyword + pattern based sentiment analyzer with confidence scoring.
    It's fast, deterministic, and good enough for financial text.

WHAT THIS IS NOT:
    - Not an LLM-based sentiment classifier (too expensive per-call).
    - Not a fine-tuned model (Phase 4 could add FinBERT integration).
    - Not a social media scraper (we analyze what tools provide).

DESIGN DECISIONS:
    1. Financial-domain keyword lexicon (not generic sentiment).
    2. Headline-level analysis (each news item scored independently).
    3. Theme extraction (identifies recurring topics like "AI", "tariffs").
    4. Confidence based on signal strength and consistency.

HOW IT CONNECTS:
    - Consumes: news/text from tool_calls, reasoning_trace observations.
    - Produces: SentimentProfile (synthesis/schemas.py).
    - Used by: misalignment_detector.py, engine.py, report_generator.py.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from synthesis.schemas import (
    SentimentDirection,
    SentimentProfile,
    SentimentSignal,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ===================================================================
# Financial Sentiment Lexicon
# ===================================================================
# These are FINANCIAL-DOMAIN specific, not generic sentiment words.
# "beat" is positive in earnings context, neutral elsewhere.
# "short" is negative in investment context, neutral elsewhere.

POSITIVE_TERMS = {
    # Earnings / Performance
    "beat", "beats", "exceeded", "surpassed", "outperformed", "record",
    "strong", "robust", "solid", "impressive", "stellar",
    # Outlook
    "upgrade", "upgraded", "buy", "outperform", "overweight",
    "bullish", "optimistic", "positive", "upside", "rally",
    # Growth
    "growth", "grew", "expanding", "acceleration", "momentum",
    "surge", "surged", "soared", "jumped", "gained",
    # Innovation / Strategy
    "breakthrough", "innovation", "launch", "partnership", "deal",
    "expansion", "investment", "opportunity",
    # Market
    "highs", "high", "rise", "rising", "climbed", "advanced",
}

NEGATIVE_TERMS = {
    # Earnings / Performance
    "missed", "miss", "disappointing", "weak", "decline", "fell",
    "dropped", "plunged", "slumped", "loss", "losses",
    # Outlook
    "downgrade", "downgraded", "sell", "underperform", "underweight",
    "bearish", "pessimistic", "negative", "downside", "crash",
    # Risk
    "risk", "risks", "concern", "concerns", "warning", "caution",
    "uncertainty", "volatile", "volatility", "threat", "threatens",
    # Business Issues
    "lawsuit", "investigation", "recall", "layoffs", "restructuring",
    "deficit", "debt", "default", "bankruptcy",
    # Market
    "lows", "low", "fall", "falling", "tumbled", "selloff",
    "correction", "bear",
}

# Domain-specific themes to extract
THEME_PATTERNS = {
    "AI / Machine Learning": [
        r"\bAI\b", r"artificial intelligence", r"machine learning",
        r"deep learning", r"GPU", r"data center", r"generative AI",
    ],
    "Earnings": [
        r"earnings", r"revenue", r"profit", r"EPS", r"quarter",
        r"quarterly", r"fiscal",
    ],
    "Regulation / Policy": [
        r"regulation", r"regulatory", r"tariff", r"sanction",
        r"antitrust", r"ban", r"restriction", r"policy",
    ],
    "Competition": [
        r"competition", r"competitor", r"rival", r"market share",
        r"versus", r"\bvs\b",
    ],
    "Macroeconomic": [
        r"interest rate", r"inflation", r"recession", r"Fed\b",
        r"economic", r"GDP", r"unemployment",
    ],
    "Product / Launch": [
        r"launch", r"release", r"product", r"chip", r"processor",
        r"platform", r"architecture",
    ],
    "Geopolitical": [
        r"China", r"Taiwan", r"trade war", r"export", r"geopolit",
    ],
}


class SentimentAnalyzer:
    """
    Analyzes financial text to produce structured sentiment profiles.

    Deterministic, fast, and finance-domain specific.
    """

    def __init__(
        self,
        positive_weight: float = 1.0,
        negative_weight: float = 1.2,  # Negative news has outsized market impact
    ) -> None:
        """
        Args:
            positive_weight: Weight for positive keywords.
            negative_weight: Weight for negative keywords (slightly higher
                because negative news has more impact on markets).
        """
        self._pos_weight = positive_weight
        self._neg_weight = negative_weight

    def analyze(
        self,
        tool_calls: list,
        observations: list[str],
    ) -> SentimentProfile:
        """
        Analyze sentiment from tool outputs and observations.

        Args:
            tool_calls: ToolCall objects from agent state.
            observations: Raw observation text strings.

        Returns:
            SentimentProfile with direction, confidence, themes.
        """
        # 1. Extract text segments to analyze
        text_segments = self._extract_text_segments(tool_calls, observations)

        if not text_segments:
            logger.info("No text segments found for sentiment analysis")
            return SentimentProfile(
                summary="Insufficient text data for sentiment analysis."
            )

        # 2. Analyze each segment
        signals = []
        for source, text in text_segments:
            signal = self._analyze_segment(source, text)
            signals.append(signal)

        # 3. Aggregate signals
        profile = self._aggregate_signals(signals)

        # 4. Extract themes
        combined_text = " ".join(text for _, text in text_segments)
        profile.dominant_themes = self._extract_themes(combined_text)

        logger.info(
            f"Sentiment analysis: {profile.overall_direction.value} "
            f"(confidence={profile.overall_confidence:.2f}, "
            f"signals={len(signals)})"
        )
        return profile

    def _extract_text_segments(
        self,
        tool_calls: list,
        observations: list[str],
    ) -> list[tuple[str, str]]:
        """
        Extract analyzable text segments from tool outputs.

        Focuses on news content and descriptive text, NOT numeric data.
        Returns: list of (source_name, text) tuples.
        """
        segments = []

        for tc in tool_calls:
            tool_name = tc.tool_name if hasattr(tc, "tool_name") else ""
            output = tc.tool_output if hasattr(tc, "tool_output") else ""

            if not output:
                continue

            # News tool outputs are the richest sentiment source
            if "news" in tool_name.lower():
                # Split into individual headlines/snippets
                headlines = self._split_news_items(output)
                for i, headline in enumerate(headlines):
                    segments.append((f"News Item {i+1}", headline))

            # Company info contains qualitative description
            elif "company" in tool_name.lower():
                segments.append(("Company Profile", output))

        return segments

    def _split_news_items(self, news_text: str) -> list[str]:
        """Split multi-item news output into individual items."""
        items = []
        # Try splitting by numbered items (1., 2., etc.)
        parts = re.split(r'\n\s*\d+\.\s+', news_text)
        for part in parts:
            cleaned = part.strip()
            if cleaned and len(cleaned) > 20:
                items.append(cleaned)

        # If no numbered items, use the whole text
        if not items and len(news_text) > 20:
            items.append(news_text)

        return items

    def _analyze_segment(self, source: str, text: str) -> SentimentSignal:
        """Analyze sentiment of a single text segment."""
        words = set(re.findall(r'\b\w+\b', text.lower()))

        pos_matches = words & POSITIVE_TERMS
        neg_matches = words & NEGATIVE_TERMS

        pos_score = len(pos_matches) * self._pos_weight
        neg_score = len(neg_matches) * self._neg_weight

        total = pos_score + neg_score
        if total == 0:
            direction = SentimentDirection.NEUTRAL
            confidence = 0.3
        elif pos_score > neg_score * 1.5:
            direction = SentimentDirection.BULLISH
            confidence = min(0.95, 0.4 + (pos_score - neg_score) / max(total, 1) * 0.5)
        elif neg_score > pos_score * 1.5:
            direction = SentimentDirection.BEARISH
            confidence = min(0.95, 0.4 + (neg_score - pos_score) / max(total, 1) * 0.5)
        else:
            direction = SentimentDirection.MIXED
            confidence = 0.4

        return SentimentSignal(
            source=source,
            text_snippet=text[:200],
            direction=direction,
            confidence=confidence,
            positive_keywords=list(pos_matches)[:5],
            negative_keywords=list(neg_matches)[:5],
        )

    def _aggregate_signals(self, signals: list[SentimentSignal]) -> SentimentProfile:
        """Aggregate individual signals into an overall profile."""
        if not signals:
            return SentimentProfile()

        bullish = [s for s in signals if s.direction == SentimentDirection.BULLISH]
        bearish = [s for s in signals if s.direction == SentimentDirection.BEARISH]
        neutral = [s for s in signals if s.direction == SentimentDirection.NEUTRAL]
        mixed = [s for s in signals if s.direction == SentimentDirection.MIXED]

        # Overall direction by majority
        counts = {
            SentimentDirection.BULLISH: len(bullish),
            SentimentDirection.BEARISH: len(bearish),
            SentimentDirection.NEUTRAL: len(neutral),
            SentimentDirection.MIXED: len(mixed),
        }
        dominant = max(counts, key=counts.get)

        # If close between bullish and bearish, it's MIXED
        if (
            counts[SentimentDirection.BULLISH] > 0
            and counts[SentimentDirection.BEARISH] > 0
            and abs(counts[SentimentDirection.BULLISH] - counts[SentimentDirection.BEARISH]) <= 1
        ):
            dominant = SentimentDirection.MIXED

        # Overall confidence: average of individual confidences,
        # penalized for inconsistency
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        consistency_penalty = 0.0
        if counts[SentimentDirection.BULLISH] > 0 and counts[SentimentDirection.BEARISH] > 0:
            # Conflicting signals reduce confidence
            consistency_penalty = 0.15
        overall_confidence = max(0.1, avg_confidence - consistency_penalty)

        summary = self._build_summary(dominant, bullish, bearish, signals)

        return SentimentProfile(
            overall_direction=dominant,
            overall_confidence=overall_confidence,
            signals=signals,
            bullish_count=len(bullish),
            bearish_count=len(bearish),
            neutral_count=len(neutral) + len(mixed),
            summary=summary,
        )

    def _extract_themes(self, text: str) -> list[str]:
        """Identify dominant themes in the text."""
        themes = []
        for theme_name, patterns in THEME_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    themes.append(theme_name)
                    break
        return themes

    def _build_summary(
        self,
        direction: SentimentDirection,
        bullish: list[SentimentSignal],
        bearish: list[SentimentSignal],
        all_signals: list[SentimentSignal],
    ) -> str:
        """Generate a human-readable sentiment summary."""
        total = len(all_signals)
        parts = []

        if direction == SentimentDirection.BULLISH:
            parts.append(
                f"Market sentiment is predominantly bullish "
                f"({len(bullish)}/{total} signals positive)."
            )
        elif direction == SentimentDirection.BEARISH:
            parts.append(
                f"Market sentiment is predominantly bearish "
                f"({len(bearish)}/{total} signals negative)."
            )
        elif direction == SentimentDirection.MIXED:
            parts.append(
                f"Market sentiment is mixed — "
                f"{len(bullish)} bullish vs {len(bearish)} bearish signals."
            )
        else:
            parts.append(
                f"Market sentiment is neutral with limited directional signals."
            )

        # Add color from top keywords
        all_pos = []
        all_neg = []
        for s in all_signals:
            all_pos.extend(s.positive_keywords)
            all_neg.extend(s.negative_keywords)

        if all_pos:
            top_pos = [w for w, _ in Counter(all_pos).most_common(3)]
            parts.append(f"Positive drivers: {', '.join(top_pos)}.")
        if all_neg:
            top_neg = [w for w, _ in Counter(all_neg).most_common(3)]
            parts.append(f"Negative drivers: {', '.join(top_neg)}.")

        return " ".join(parts)
