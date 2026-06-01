"""
synthesis/report_generator.py — Professional Report Generation
================================================================

WHY THIS EXISTS:
    The SynthesisReport contains structured data. This module transforms
    that data into human-readable professional documents:

    1. MARKDOWN — For display, GitHub, and portability.
    2. PDF — For institutional-quality deliverables (via fpdf2).

    Reports follow a standard investment report structure:
    - Executive Summary
    - Company Overview
    - Financial Analysis (with tables)
    - Sentiment Analysis
    - Risk Analysis
    - Contradictions & Misalignments
    - Investment Thesis
    - Confidence Score
    - Evidence Quality Summary

HOW IT CONNECTS:
    - Consumes: SynthesisReport (from synthesis/engine.py).
    - Produces: .md and .pdf files in the reports output directory.
    - Called by: main.py after synthesis completes.

DESIGN DECISIONS:
    1. Markdown first — it's the universal format, previews in terminals.
    2. PDF from Markdown — fpdf2 renders the same content as PDF.
    3. Reports are SAVED to disk, not just printed. This creates
       a historical record of analyses.
    4. Every claim has a citation (evidence source traceable to tools).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from synthesis.schemas import (
    InvestmentOutlook,
    MisalignmentType,
    RiskSeverity,
    SynthesisReport,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generates institutional-quality investment reports from SynthesisReport.

    Supports Markdown and PDF output formats.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        """
        Args:
            output_dir: Directory to save reports. Defaults to ./reports/
        """
        if output_dir is None:
            output_dir = str(
                Path(__file__).resolve().parent.parent / "reports"
            )
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        report: SynthesisReport,
        formats: list[str] = None,
    ) -> dict[str, str]:
        """
        Generate report files from a SynthesisReport.

        Args:
            report: The synthesis report to render.
            formats: List of formats to generate ("markdown", "pdf").
                     Defaults to ["markdown"].

        Returns:
            Dict mapping format -> file path.
        """
        if formats is None:
            formats = ["markdown"]

        paths = {}

        # Markdown is always generated first (PDF depends on it)
        md_content = self._render_markdown(report)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker = report.ticker or "UNKNOWN"

        if "markdown" in formats:
            md_path = self._output_dir / f"{ticker}_{timestamp}_report.md"
            md_path.write_text(md_content, encoding="utf-8")
            paths["markdown"] = str(md_path)
            logger.info(f"Markdown report saved: {md_path}")

        if "pdf" in formats:
            pdf_path = self._generate_pdf(report, ticker, timestamp)
            if pdf_path:
                paths["pdf"] = pdf_path

        return paths

    # ===================================================================
    # Markdown Rendering
    # ===================================================================

    def _render_markdown(self, r: SynthesisReport) -> str:
        """Render the full report as Markdown."""
        sections = [
            self._render_header(r),
            self._render_executive_summary(r),
            self._render_company_overview(r),
            self._render_financial_analysis(r),
            self._render_sentiment_analysis(r),
            self._render_risk_analysis(r),
            self._render_contradictions(r),
            self._render_investment_thesis(r),
            self._render_confidence_score(r),
            self._render_evidence_quality(r),
            self._render_footer(r),
        ]
        return "\n\n".join(s for s in sections if s)

    def _render_header(self, r: SynthesisReport) -> str:
        """Report header with title and metadata."""
        outlook_emoji = {
            InvestmentOutlook.STRONG_BUY: "🟢🟢",
            InvestmentOutlook.BUY: "🟢",
            InvestmentOutlook.HOLD: "🟡",
            InvestmentOutlook.SELL: "🔴",
            InvestmentOutlook.STRONG_SELL: "🔴🔴",
            InvestmentOutlook.INSUFFICIENT_DATA: "⚪",
        }
        emoji = outlook_emoji.get(r.outlook, "")

        return f"""# {r.company_name or r.ticker} ({r.ticker}) — Investment Analysis Report

{emoji} **Recommendation: {r.outlook.value.replace('_', ' ').upper()}** | Confidence: {r.confidence.overall:.0%} ({r.confidence.label})

---

| Field | Value |
|-------|-------|
| **Date** | {datetime.now(timezone.utc).strftime('%B %d, %Y')} |
| **Ticker** | {r.ticker} |
| **Model** | {r.model_used} |
| **Analysis Time** | {r.execution_time_seconds:.1f}s |
| **Iterations** | {r.iterations_used} |
| **Tools Used** | {', '.join(r.tools_used) or 'N/A'} |"""

    def _render_executive_summary(self, r: SynthesisReport) -> str:
        """One-paragraph executive summary."""
        findings = "\n".join(f"- {f}" for f in r.key_findings) if r.key_findings else "- No key findings available."

        return f"""## 1. Executive Summary

{r.investment_thesis.split(chr(10) + chr(10))[0] if r.investment_thesis else 'Analysis pending.'}

### Key Findings
{findings}"""

    def _render_company_overview(self, r: SynthesisReport) -> str:
        """Company overview section."""
        return f"""## 2. Company Overview

| Field | Value |
|-------|-------|
| **Company** | {r.company_name or 'N/A'} |
| **Ticker** | {r.ticker or 'N/A'} |
| **Financial Health Score** | {r.financial.financial_health_score:.2f} / 1.00 |
| **Overall Assessment** | {r.financial.overall_assessment or 'N/A'} |"""

    def _render_financial_analysis(self, r: SynthesisReport) -> str:
        """Detailed financial metrics tables."""
        sections = []
        sections.append("## 3. Financial Analysis")

        categories = [
            ("Valuation Metrics", r.financial.valuation_metrics),
            ("Profitability Metrics", r.financial.profitability_metrics),
            ("Growth Metrics", r.financial.growth_metrics),
            ("Liquidity Metrics", r.financial.liquidity_metrics),
            ("Leverage Metrics", r.financial.leverage_metrics),
        ]

        for cat_name, metrics in categories:
            available = [m for m in metrics if m.value is not None]
            if not available:
                continue

            sections.append(f"\n### {cat_name}\n")
            sections.append("| Metric | Value | Assessment |")
            sections.append("|--------|-------|------------|")
            for m in available:
                # Truncate assessment for table readability
                assessment_short = m.assessment.split(" — ")[0] if " — " in m.assessment else m.assessment[:40]
                sections.append(f"| {m.name} | {m.formatted} | {assessment_short} |")

        # Strengths & Weaknesses
        if r.financial.key_strengths or r.financial.key_weaknesses:
            sections.append("\n### Strengths & Weaknesses\n")
            if r.financial.key_strengths:
                sections.append("**✅ Strengths:**")
                for s in r.financial.key_strengths:
                    sections.append(f"- {s}")
            if r.financial.key_weaknesses:
                sections.append("\n**⚠️ Weaknesses:**")
                for w in r.financial.key_weaknesses:
                    sections.append(f"- {w}")

        return "\n".join(sections)

    def _render_sentiment_analysis(self, r: SynthesisReport) -> str:
        """Sentiment analysis section."""
        direction_emoji = {
            "bullish": "📈",
            "bearish": "📉",
            "neutral": "➡️",
            "mixed": "🔀",
        }
        emoji = direction_emoji.get(r.sentiment.overall_direction.value, "")

        lines = [
            f"## 4. Sentiment Analysis",
            "",
            f"{emoji} **Overall Direction:** {r.sentiment.overall_direction.value.upper()} "
            f"(Confidence: {r.sentiment.overall_confidence:.0%})",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Bullish Signals | {r.sentiment.bullish_count} |",
            f"| Bearish Signals | {r.sentiment.bearish_count} |",
            f"| Neutral Signals | {r.sentiment.neutral_count} |",
        ]

        if r.sentiment.dominant_themes:
            lines.append(f"\n**Dominant Themes:** {', '.join(r.sentiment.dominant_themes)}")

        if r.sentiment.summary:
            lines.append(f"\n{r.sentiment.summary}")

        return "\n".join(lines)

    def _render_risk_analysis(self, r: SynthesisReport) -> str:
        """Risk analysis section with severity indicators."""
        severity_emoji = {
            RiskSeverity.LOW: "🟢",
            RiskSeverity.MEDIUM: "🟡",
            RiskSeverity.HIGH: "🟠",
            RiskSeverity.CRITICAL: "🔴",
        }

        lines = [
            f"## 5. Risk Analysis",
            "",
            f"**Overall Risk Level:** {severity_emoji.get(r.risk.overall_risk_level, '')} "
            f"{r.risk.overall_risk_level.value.upper()} "
            f"(Score: {r.risk.risk_score:.2f})",
            "",
        ]

        if r.risk.risks:
            lines.append("| # | Risk | Category | Severity | Description |")
            lines.append("|---|------|----------|----------|-------------|")
            for i, risk in enumerate(r.risk.risks, 1):
                emoji = severity_emoji.get(risk.severity, "")
                desc = risk.description[:80] + "..." if len(risk.description) > 80 else risk.description
                lines.append(
                    f"| {i} | {risk.title} | {risk.category} | "
                    f"{emoji} {risk.severity.value} | {desc} |"
                )
        else:
            lines.append("No significant risks identified from available evidence.")

        if r.risk.summary:
            lines.append(f"\n{r.risk.summary}")

        return "\n".join(lines)

    def _render_contradictions(self, r: SynthesisReport) -> str:
        """Contradictions and misalignment section."""
        lines = ["## 6. Contradictions & Misalignments"]

        if r.misalignment.detected:
            misalign_emoji = {
                MisalignmentType.BULLISH_WEAK_FUNDAMENTALS: "⚠️ HYPE RISK",
                MisalignmentType.BEARISH_STRONG_FUNDAMENTALS: "💡 POTENTIAL OPPORTUNITY",
                MisalignmentType.OVERVALUATION_VS_OPTIMISM: "⚠️ PRICED FOR PERFECTION",
                MisalignmentType.GROWTH_NARRATIVE_DECLINING_MARGINS: "⚠️ NARRATIVE RISK",
            }
            tag = misalign_emoji.get(r.misalignment.misalignment_type, "⚠️ MISALIGNMENT")

            lines.append(f"\n### {tag}")
            lines.append(f"\n**Type:** {r.misalignment.misalignment_type.value}")
            lines.append(f"\n**Severity:** {r.misalignment.severity.value}")
            lines.append(f"\n{r.misalignment.explanation}")
            lines.append(f"\n**Recommendation:** {r.misalignment.recommendation}")
        else:
            lines.append("\nNo significant misalignment between sentiment and fundamentals detected.")

        if r.contradictions:
            lines.append("\n### Evidence Contradictions")
            for c in r.contradictions:
                lines.append(f"- {c}")

        return "\n".join(lines)

    def _render_investment_thesis(self, r: SynthesisReport) -> str:
        """Investment thesis section."""
        outlook_bar = {
            InvestmentOutlook.STRONG_BUY: "█████████████████████ STRONG BUY",
            InvestmentOutlook.BUY: "████████████████░░░░░ BUY",
            InvestmentOutlook.HOLD: "██████████░░░░░░░░░░░ HOLD",
            InvestmentOutlook.SELL: "█████░░░░░░░░░░░░░░░░ SELL",
            InvestmentOutlook.STRONG_SELL: "██░░░░░░░░░░░░░░░░░░░ STRONG SELL",
            InvestmentOutlook.INSUFFICIENT_DATA: "░░░░░░░░░░░░░░░░░░░░░ INSUFFICIENT DATA",
        }

        bar = outlook_bar.get(r.outlook, "")

        return f"""## 7. Investment Thesis

### Final Recommendation

```
{bar}
```

**Outlook:** {r.outlook.value.replace('_', ' ').upper()}
**Confidence:** {r.confidence.overall:.0%} ({r.confidence.label})

---

{r.investment_thesis}"""

    def _render_confidence_score(self, r: SynthesisReport) -> str:
        """Confidence score breakdown."""
        c = r.confidence

        lines = [
            "## 8. Confidence Score",
            "",
            f"**Overall: {c.overall:.0%} ({c.label})**",
            "",
            "| Component | Score |",
            "|-----------|-------|",
            f"| Evidence Quality | {c.evidence_quality:.0%} |",
            f"| Source Reliability | {c.source_reliability:.0%} |",
            f"| Data Completeness | {c.data_completeness:.0%} |",
            f"| Consistency | {c.consistency:.0%} |",
            f"| Recency | {c.recency:.0%} |",
        ]

        if c.boosts:
            lines.append("\n**Confidence Boosts:**")
            for b in c.boosts:
                lines.append(f"- ✅ {b}")

        if c.penalties:
            lines.append("\n**Confidence Penalties:**")
            for p in c.penalties:
                lines.append(f"- ⚠️ {p}")

        return "\n".join(lines)

    def _render_evidence_quality(self, r: SynthesisReport) -> str:
        """Evidence quality summary."""
        fin = r.financial
        return f"""## 9. Evidence Quality Summary

| Metric | Value |
|--------|-------|
| Financial Metrics Available | {fin.metrics_available} |
| Financial Metrics Missing | {fin.metrics_missing} |
| Evidence Quality Score | {fin.evidence_quality:.0%} |
| Sentiment Signals Analyzed | {len(r.sentiment.signals)} |
| Risks Identified | {len(r.risk.risks)} |
| Contradictions Found | {len(r.contradictions)} |
| Tools Used | {len(r.tools_used)} |
| Agent Iterations | {r.iterations_used} |"""

    def _render_footer(self, r: SynthesisReport) -> str:
        """Report footer with disclaimers."""
        return f"""---

*Report generated by ARA-1 (Autonomous Research Agent) on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.*
*Model: {r.model_used} | Report ID: {r.id}*

> **Disclaimer:** This report is generated by an AI system for informational purposes only.
> It does not constitute financial advice. Always consult a qualified financial advisor
> before making investment decisions. Past performance does not guarantee future results.*"""

    # ===================================================================
    # PDF Generation
    # ===================================================================

    def _generate_pdf(
        self,
        report: SynthesisReport,
        ticker: str,
        timestamp: str,
    ) -> Optional[str]:
        """Generate a PDF report using fpdf2."""
        try:
            from fpdf import FPDF
        except ImportError:
            logger.warning(
                "fpdf2 not installed — skipping PDF generation. "
                "Install with: pip install fpdf2"
            )
            return None

        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # Title
            pdf.set_font("Helvetica", "B", 20)
            title = f"{report.company_name or ticker} ({ticker})"
            pdf.cell(0, 15, title, new_x="LMARGIN", new_y="NEXT", align="C")

            pdf.set_font("Helvetica", "", 12)
            pdf.cell(
                0, 8,
                f"Investment Analysis Report - {report.outlook.value.replace('_', ' ').upper()}",
                new_x="LMARGIN", new_y="NEXT", align="C",
            )
            pdf.cell(
                0, 8,
                f"Confidence: {report.confidence.overall:.0%} ({report.confidence.label})",
                new_x="LMARGIN", new_y="NEXT", align="C",
            )
            pdf.ln(10)

            # Sections
            self._pdf_section(pdf, "Executive Summary", report.investment_thesis.split("\n\n")[0] if report.investment_thesis else "N/A")
            self._pdf_section(pdf, "Financial Health", report.financial.overall_assessment)
            self._pdf_section(pdf, "Sentiment", report.sentiment.summary)

            if report.misalignment.detected:
                self._pdf_section(pdf, "Misalignment Warning", report.misalignment.explanation)

            self._pdf_section(pdf, "Risk Assessment", report.risk.summary)

            # Key Findings
            if report.key_findings:
                self._pdf_section(pdf, "Key Findings", "\n".join(f"• {f}" for f in report.key_findings))

            # Confidence
            self._pdf_section(pdf, "Confidence Score", report.confidence.explanation)

            # Disclaimer
            pdf.ln(10)
            pdf.set_font("Helvetica", "I", 8)
            pdf.multi_cell(
                0, 4,
                "Disclaimer: This report is generated by ARA-1 for informational purposes only. "
                "It does not constitute financial advice.",
            )

            # Save
            pdf_path = self._output_dir / f"{ticker}_{timestamp}_report.pdf"
            pdf.output(str(pdf_path))
            logger.info(f"PDF report saved: {pdf_path}")
            return str(pdf_path)

        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return None

    def _pdf_section(self, pdf, title: str, content: str) -> None:
        """Add a section to the PDF."""
        pdf.set_font("Helvetica", "B", 14)
        safe_title = self._sanitize_for_pdf(title)
        pdf.cell(0, 10, safe_title, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 10)
        safe_content = self._sanitize_for_pdf(content)
        pdf.multi_cell(0, 5, safe_content)
        pdf.ln(5)

    @staticmethod
    def _sanitize_for_pdf(text: str) -> str:
        """
        Sanitize text for PDF rendering with Helvetica font.

        Replaces common Unicode characters with ASCII equivalents,
        then falls back to latin-1 encoding for anything remaining.
        """
        replacements = {
            "\u2014": "--",   # em-dash
            "\u2013": "-",    # en-dash
            "\u2018": "'",    # left single quote
            "\u2019": "'",    # right single quote
            "\u201c": '"',    # left double quote
            "\u201d": '"',    # right double quote
            "\u2022": "*",    # bullet
            "\u2026": "...",  # ellipsis
            "\u00a0": " ",   # non-breaking space
            "\u2192": "->",   # right arrow
            "\u2190": "<-",   # left arrow
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        # Remove emoji and other non-latin chars
        return text.encode("latin-1", errors="replace").decode("latin-1")
