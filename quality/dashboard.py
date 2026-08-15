"""
dashboard/app.py — Streamlit Monitoring Dashboard
====================================================

WHY THIS EXISTS:
    Operators need visual insight into ARA-1's behavior.
    Raw JSON logs and metrics are insufficient for:
    - Real-time monitoring
    - Quick anomaly detection
    - Non-technical stakeholder reporting

WHAT PROBLEM IT SOLVES:
    Provides a single-page dashboard showing:
    - System health
    - Latest evaluation scores
    - Metric breakdowns by category
    - Failure injection results
    - Historical performance trends

HOW IT INTEGRATES:
    - Reads evaluation JSONs from data/evaluations/
    - Reads telemetry JSONs from data/telemetry/
    - Reads reports from reports/
    - Optionally calls the FastAPI backend for live data

RUN:
    streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)


# ===================================================================
# Page Configuration
# ===================================================================

st.set_page_config(
    page_title="ARA-1 Monitor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===================================================================
# Dashboard Layout
# ===================================================================

def main():
    """Main dashboard entry point."""

    # Header
    st.title("🤖 ARA-1 Monitoring Dashboard")
    st.markdown("*Phase 4: Evaluation, Reliability & Observability*")
    st.divider()

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        try:
            from config.settings import settings
            st.success(f"Provider: {settings.llm_provider}")
            st.info(f"Model: {settings.active_model}")
            st.info(f"Max Iterations: {settings.max_iterations}")
        except Exception:
            st.warning("Could not load settings")

        st.divider()
        st.header("📂 Data Sources")

        eval_dir = _PROJECT_ROOT / "data" / "evaluations"
        telemetry_dir = _PROJECT_ROOT / "data" / "telemetry"
        report_dir = _PROJECT_ROOT / "reports"

        st.text(f"Evaluations: {eval_dir}")
        st.text(f"Telemetry: {telemetry_dir}")
        st.text(f"Reports: {report_dir}")

    # Main content - tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Evaluation", "🔍 Telemetry", "⚠️ Failures", "📄 Reports"
    ])

    with tab1:
        render_evaluation_tab()

    with tab2:
        render_telemetry_tab()

    with tab3:
        render_failures_tab()

    with tab4:
        render_reports_tab()


# ===================================================================
# Evaluation Tab
# ===================================================================

def render_evaluation_tab():
    """Render the evaluation metrics tab."""
    st.header("📊 Evaluation Metrics")

    eval_dir = _PROJECT_ROOT / "data" / "evaluations"

    if not eval_dir.exists() or not list(eval_dir.glob("*.json")):
        st.info(
            "No evaluation data found. Run an analysis with evaluation enabled "
            "to generate data."
        )

        # Show placeholder with metric definitions
        st.subheader("Available Metrics (22)")
        metrics = [
            ("M1", "Hallucination Rate", "< 2%", "reliability"),
            ("M2", "Tool Efficiency", "> 70%", "tools"),
            ("M3", "Tool Success Rate", "> 80%", "tools"),
            ("M4", "Iteration Efficiency", "> 60%", "reasoning"),
            ("M5", "Reasoning Depth", "> 50 chars", "reasoning"),
            ("M6", "Source Diversity", "> 75%", "retrieval"),
            ("M7", "Retrieval Usage", "Active", "retrieval"),
            ("M8", "Conflict Detection", "Active", "reliability"),
            ("M9", "Error Recovery Rate", "> 80%", "reliability"),
            ("M10", "Completion Status", "Complete", "reasoning"),
            ("M11", "Answer Comprehensiveness", "> 500 chars", "synthesis"),
            ("M12", "Evidence Grounding", "> 50%", "retrieval"),
            ("M13", "Total Latency", "< 120s", "performance"),
            ("M14", "Avg Iteration Time", "< 30s", "performance"),
            ("M15", "Redundant Calls", "≤ 1", "tools"),
            ("M16", "Episodic Memory Usage", "Active", "retrieval"),
            ("M17", "Parse Error Rate", "< 20%", "reliability"),
            ("M18", "Financial Metrics Coverage", "> 50%", "synthesis"),
            ("M19", "Confidence Calibration", "> 30%", "synthesis"),
            ("M20", "Misalignment Detection", "Active", "synthesis"),
            ("M21", "Risk Assessment", "≥ 1 risk", "synthesis"),
            ("M22", "Report Generation", "Generated", "synthesis"),
        ]

        cols = st.columns(3)
        for i, (mid, name, target, category) in enumerate(metrics):
            with cols[i % 3]:
                st.metric(
                    label=f"{mid}: {name}",
                    value=target,
                    help=f"Category: {category}",
                )

        return

    # Load latest evaluation
    eval_files = sorted(eval_dir.glob("*.json"), reverse=True)
    latest = eval_files[0]

    with open(latest, "r") as f:
        eval_data = json.load(f)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Score", f"{eval_data.get('overall_score', 0):.0%}")
    with col2:
        st.metric("Pass Rate", f"{eval_data.get('pass_rate', 0):.0%}")
    with col3:
        st.metric("Passed", eval_data.get("passed_metrics", 0))
    with col4:
        st.metric("Failed", eval_data.get("failed_metrics", 0))

    # Metric details
    st.subheader("Metric Details")
    metrics = eval_data.get("metrics", [])
    if metrics:
        for m in metrics:
            status_emoji = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(m.get("status", ""), "❓")
            with st.expander(f"{status_emoji} {m.get('name', 'Unknown')}: {m.get('formatted', '')}"):
                st.write(f"**Category:** {m.get('category', '')}")
                st.write(f"**Value:** {m.get('value', '')}")
                st.write(f"**Threshold:** {m.get('threshold', '')}")
                st.write(f"**Description:** {m.get('description', '')}")
                if m.get("details"):
                    st.write(f"**Details:** {m.get('details', '')}")


# ===================================================================
# Telemetry Tab
# ===================================================================

def render_telemetry_tab():
    """Render the telemetry tab."""
    st.header("🔍 Runtime Telemetry")

    telemetry_dir = _PROJECT_ROOT / "data" / "telemetry"

    if not telemetry_dir.exists() or not list(telemetry_dir.glob("*.json")):
        st.info("No telemetry data found. Run an analysis to generate telemetry.")
        return

    # Load latest telemetry
    tel_files = sorted(telemetry_dir.glob("*.json"), reverse=True)
    latest = tel_files[0]

    with open(latest, "r") as f:
        tel_data = json.load(f)

    summary = tel_data.get("summary", {})

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("LLM Calls", summary.get("llm_calls", 0))
        st.metric("LLM Avg (ms)", f"{summary.get('llm_avg_ms', 0):.0f}")
    with col2:
        st.metric("Tool Calls", summary.get("tool_calls", 0))
        st.metric("Tool Avg (ms)", f"{summary.get('tool_avg_ms', 0):.0f}")
    with col3:
        st.metric("Total Duration", f"{summary.get('total_duration_ms', 0)/1000:.1f}s")
        st.metric("Errors", summary.get("total_errors", 0))

    # Events timeline
    events = tel_data.get("events", [])
    if events:
        st.subheader("Event Timeline")
        for event in events[-20:]:  # Last 20 events
            emoji = "✅" if event.get("success") else "❌"
            st.text(
                f"{emoji} [{event.get('event_type', '')}] "
                f"{event.get('name', '')} "
                f"({event.get('duration_ms', 0):.0f}ms)"
            )


# ===================================================================
# Failures Tab
# ===================================================================

def render_failures_tab():
    """Render the failure injection tab."""
    st.header("⚠️ Failure Injection Results")

    failure_dir = _PROJECT_ROOT / "data" / "failures"

    if not failure_dir.exists() or not list(failure_dir.glob("*.json")):
        st.info(
            "No failure injection data found. Run with failure injection "
            "enabled to generate results."
        )

        # Show scenario definitions
        st.subheader("7 Failure Scenarios")
        scenarios = [
            ("1", "Malformed JSON", "medium", "Tests parser resilience"),
            ("2", "Contradictory Metrics", "high", "Tests conflict detection"),
            ("3", "Retrieval Corruption", "high", "Tests evidence validation"),
            ("4", "Stale Evidence", "medium", "Tests temporal reasoning"),
            ("5", "API Timeout", "high", "Tests retry/fallback system"),
            ("6", "Vector DB Failure", "critical", "Tests degraded mode"),
            ("7", "Hallucinated Chunks", "critical", "Tests hallucination detection"),
        ]

        for sid, name, severity, desc in scenarios:
            severity_color = {
                "medium": "🟡", "high": "🟠", "critical": "🔴"
            }.get(severity, "⚪")
            st.markdown(f"{severity_color} **Failure {sid}: {name}** — {desc}")

        return

    # Load results
    failure_files = sorted(failure_dir.glob("*.json"), reverse=True)
    latest = failure_files[0]

    with open(latest, "r") as f:
        failure_data = json.load(f)

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Injected", failure_data.get("injected", 0))
    with col2:
        st.metric("Recovered", failure_data.get("recovered", 0))
    with col3:
        st.metric("Recovery Rate", f"{failure_data.get('recovery_rate', 0):.0%}")

    # Results
    for result in failure_data.get("results", []):
        status = "✅" if result.get("recovered") else "❌"
        with st.expander(f"{status} {result.get('scenario_name', 'Unknown')}"):
            st.write(f"**Type:** {result.get('failure_type', '')}")
            st.write(f"**Severity:** {result.get('severity', '')}")
            st.write(f"**Impact:** {result.get('impact', '')}")
            st.write(f"**Recovery:** {result.get('agent_behavior', '')}")
            st.write(f"**Lessons:** {result.get('lessons_learned', '')}")


# ===================================================================
# Reports Tab
# ===================================================================

def render_reports_tab():
    """Render the reports tab."""
    st.header("📄 Generated Reports")

    report_dir = _PROJECT_ROOT / "reports"

    if not report_dir.exists():
        st.info("No reports directory found.")
        return

    reports = sorted(report_dir.iterdir(), reverse=True)
    if not reports:
        st.info("No reports generated yet.")
        return

    for report in reports:
        if report.suffix in (".md", ".pdf"):
            size_kb = report.stat().st_size / 1024
            st.markdown(f"📄 **{report.name}** ({size_kb:.1f} KB)")

            if report.suffix == ".md":
                with st.expander("Preview"):
                    content = report.read_text(encoding="utf-8", errors="replace")
                    st.markdown(content[:3000])


if __name__ == "__main__":
    main()
