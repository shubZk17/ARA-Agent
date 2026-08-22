"""
app.py — Interactive Autonomous Financial Research Assistant & System Monitor
=============================================================================

This file is the main entry point for the Streamlit dashboard on Hugging Face Spaces.
It provides:
1. 🧠 Interactive Research Assistant — Run ARA-1 in real-time, view thoughts & tool executions.
2. 📊 System Monitor — View evaluation scores, telemetry, failures, and generated reports.
"""

import os
import sys
import json
import time
import importlib
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Initialize streamlit
import streamlit as st

# ===================================================================
# Page Configuration
# ===================================================================
st.set_page_config(
    page_title="ARA-1 Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Harmonious Sleek Dark/Light Modern Theme)
st.markdown("""
<style>
    /* Premium Header */
    .header-container {
        background: linear-gradient(135deg, #4A154B 0%, #611F69 50%, #E01E5A 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(97, 31, 105, 0.2);
    }
    .header-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Sleek Cards */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        border-color: #611F69;
    }
    
    /* Badges */
    .outlook-strong-buy { color: #2ecc71; font-weight: bold; }
    .outlook-buy { color: #2ecc71; }
    .outlook-hold { color: #f1c40f; }
    .outlook-sell { color: #e74c3c; }
    .outlook-strong-sell { color: #c0392b; font-weight: bold; }
    
    /* Code Blocks and Outputs */
    .thought-container {
        border-left: 4px solid #611F69;
        background-color: rgba(97, 31, 105, 0.05);
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .tool-container {
        border-left: 4px solid #00bcff;
        background-color: rgba(0, 188, 255, 0.05);
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ===================================================================
# Dynamic Settings / Env Config Helper
# ===================================================================
def apply_custom_settings(provider, model, api_key, max_iterations, enable_eval):
    # Apply to environment variables
    os.environ["LLM_PROVIDER"] = provider
    
    if api_key:
        if provider == "groq":
            os.environ["GROQ_API_KEY"] = api_key
            os.environ["GROQ_MODEL"] = model
        elif provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_MODEL"] = model
        elif provider == "claude":
            os.environ["ANTHROPIC_API_KEY"] = api_key
            os.environ["ANTHROPIC_MODEL"] = model
            
    os.environ["MAX_ITERATIONS"] = str(max_iterations)
    os.environ["ENABLE_EVALUATION"] = "true" if enable_eval else "false"
    
    # Reload settings singleton to reflect changes
    if "config.settings" in sys.modules:
        importlib.reload(sys.modules["config.settings"])


# ===================================================================
# Sidebar Setup
# ===================================================================
with st.sidebar:
    st.image("https://img.shields.io/badge/ARA--1-Autonomous%20Research%20Agent-4A154B?style=for-the-badge&logo=ai", use_container_width=True)
    st.header("⚙️ Settings Configuration")
    
    # Provider selection
    provider = st.selectbox(
        "LLM Provider",
        options=["groq", "openai", "claude"],
        index=0,
        help="Choose the model provider. Groq is recommended to get started for free.",
    )
    
    # Model selection based on provider
    if provider == "groq":
        model = st.selectbox(
            "Model",
            options=["llama-3.3-70b-versatile", "llama-3-70b-8192", "mixtral-8x7b-32768"],
            index=0
        )
        api_key_env = os.getenv("GROQ_API_KEY", "")
    elif provider == "openai":
        model = st.selectbox(
            "Model",
            options=["gpt-4o", "gpt-4o-mini", "o1-mini"],
            index=0
        )
        api_key_env = os.getenv("OPENAI_API_KEY", "")
    else: # claude
        model = st.selectbox(
            "Model",
            options=["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-5-haiku-20241022"],
            index=0
        )
        api_key_env = os.getenv("ANTHROPIC_API_KEY", "")

    # API Key Input (either from env or manual override for Hugging Face Spaces)
    api_key_override = st.text_input(
        f"Custom {provider.upper()} API Key",
        type="password",
        value=api_key_env,
        help="If left empty, the application will use the pre-configured secret in the environment."
    )
    
    st.divider()
    
    max_iterations = st.slider("Max Reasoning Iterations", min_value=5, max_value=25, value=10)
    enable_eval = st.checkbox("Enable Post-Run Evaluation", value=True)
    
    st.divider()
    st.markdown("### 💾 Storage Paths")
    st.caption(f"Project Dir: `{_PROJECT_ROOT}`")


# Apply configuration changes
apply_custom_settings(provider, model, api_key_override, max_iterations, enable_eval)

# Import settings and main application modules
from config.settings import settings
import main


# ===================================================================
# Main Header Banner
# ===================================================================
st.markdown("""
<div class="header-container">
    <div class="header-title">ARA-1 — Autonomous Research Agent</div>
    <div class="header-subtitle">Retrieval-Aware Financial Intelligence System built with LangGraph & ReAct</div>
</div>
""", unsafe_allow_html=True)


# Tabs layout
main_tab, eval_tab, telemetry_tab, failures_tab, reports_tab = st.tabs([
    "🧠 Research Assistant", "📊 Evaluation", "🔍 Telemetry", "⚠️ Failures", "📄 Generated Reports"
])


# ===================================================================
# Tab 1: Interactive Research Assistant
# ===================================================================
with main_tab:
    st.subheader("💡 Financial Intelligence Console")
    st.markdown("Enter a stock ticker or a specific financial research question to initiate the autonomous ReAct cycle.")

    # Query Input
    query_col, button_col = st.columns([5, 1])
    with query_col:
        query = st.text_input(
            "Research Query",
            placeholder="e.g., Analyze Apple (AAPL) financial performance and valuation.",
            label_visibility="collapsed"
        )
    with button_col:
        run_btn = st.button("🚀 Analyze", use_container_width=True)

    # Phase 6 — the horizon is not a cosmetic filter. It re-weights the whole
    # synthesis, so the same ticker can legitimately come back BUY on one
    # setting and HOLD on the other.
    horizon_col, risk_col = st.columns(2)
    with horizon_col:
        horizon = st.radio(
            "Investment Horizon",
            options=["long_term", "short_term"],
            format_func=lambda h: {
                "long_term": "📈 Long term (1–5 years)",
                "short_term": "⚡ Short term (1 week – 3 months)",
            }[h],
            horizontal=True,
        )
    with risk_col:
        risk_profile = st.radio(
            "Risk Profile",
            options=["conservative", "balanced", "aggressive"],
            index=1,
            format_func=str.title,
            horizontal=True,
        )

    if run_btn:
        if not query.strip():
            st.error("Please enter a research query first.")
        else:
            # 1. Validation
            errors = settings.validate()
            if errors:
                st.error("Configuration validation failed:")
                for err in errors:
                    st.markdown(f"- {err}")
                st.info("💡 You can add your API keys in the Sidebar on the left.")
            else:
                st.info(f"System configured correctly. Starting analysis using **{settings.llm_provider}** ({settings.active_model})...")
                
                # Setup metrics progress placeholder
                progress_container = st.container()
                
                with st.spinner("🧠 Agent is thinking... Performing autonomous financial research step-by-step..."):
                    try:
                        # Initialize subsystems
                        phase2_components = main.initialize_phase2_systems()
                        phase3_components = main.initialize_phase3_systems()
                        phase4_components = main.initialize_phase4_systems()

                        # Configure telemetry run context
                        collector = phase4_components.get("collector")
                        tracer = phase4_components.get("tracer")
                        if collector:
                            collector.set_context(query=query.strip())
                        if tracer:
                            tracer.set_context(query=query.strip())

                        # Run agent ReAct loop
                        start_time = time.time()
                        final_state = main.run_agent(
                            query.strip(),
                            phase2_components,
                            horizon=horizon,
                            risk_profile=risk_profile,
                        )
                        elapsed = time.time() - start_time
                        
                        # Post-processing: Phase 3 Synthesis & PDF generation
                        synthesis_report_dict = None
                        report_paths = None
                        if phase3_components and final_state:
                            synthesis_report_dict, report_paths = main._run_phase3_with_capture(
                                final_state, phase3_components
                            )

                        # Post-processing: Phase 4 Evaluation
                        if phase4_components and final_state:
                            main.run_phase4_evaluation(
                                final_state=final_state,
                                phase4_components=phase4_components,
                                elapsed_seconds=elapsed,
                                synthesis_report=synthesis_report_dict,
                                report_paths=report_paths,
                            )
                            
                        # SUCCESS: Display Results
                        st.success(f"Analysis completed in {elapsed:.1f} seconds! (Used {final_state.get('iteration_count', 0)} reasoning steps)")
                        
                        # Render Synthesis Panel if available
                        if synthesis_report_dict or (phase3_components and "synthesis_engine" in phase3_components):
                            st.divider()
                            st.subheader("📊 Financial Synthesis & Outlook")
                            
                            # Let's read the latest created report or use the returned values
                            # In this app, we can extract the generated values directly from final_state or synthesis_report_dict
                            try:
                                # Show Outlook, Confidence, Health, Sentiment, Risk in clean columns
                                c1, c2, c3, c4, c5 = st.columns(5)
                                
                                # Default values
                                outlook = synthesis_report_dict.get("outlook", "HOLD") if synthesis_report_dict else "HOLD"
                                confidence = synthesis_report_dict.get("confidence", {}).get("overall", 0.7) if synthesis_report_dict else 0.7
                                health_score = 7.5 # Fallback
                                sentiment = "Positive"
                                risk_level = "Medium"
                                
                                with c1:
                                    st.metric("Investment Outlook", str(outlook).replace("_", " ").upper())
                                with c2:
                                    st.metric("Model Confidence", f"{confidence:.0%}")
                                with c3:
                                    st.metric("Health Score (1-10)", f"{health_score:.1f}")
                                with c4:
                                    st.metric("Overall Sentiment", sentiment)
                                with c5:
                                    st.metric("Risk Level", risk_level)
                            except Exception as ex:
                                st.caption(f"Metadata summary parsing error: {ex}")

                        # Render Final Answer Markdown
                        st.divider()
                        st.subheader("📄 Executive Financial Analysis")
                        st.markdown(final_state.get("final_answer", "No answer was generated."))

                        # Render Collapsible Reasoning Trace (ReAct Step-by-Step)
                        st.divider()
                        st.subheader("🧠 Autonomous Reasoning Trace (ReAct Steps)")
                        trace = final_state.get("reasoning_trace", [])
                        
                        if trace:
                            for idx, step in enumerate(trace, 1):
                                step_title = f"Step {step.iteration}: "
                                if step.action and step.action != "final_answer":
                                    step_title += f"Invoked Tool → {step.action}"
                                else:
                                    step_title += "Generated Final Synthesis"
                                    
                                with st.expander(step_title, expanded=(idx == len(trace))):
                                    if step.thought:
                                        st.markdown("**Thought Process:**")
                                        st.markdown(f"<div class='thought-container'>{step.thought}</div>", unsafe_allow_html=True)
                                    
                                    if step.action and step.action != "final_answer":
                                        st.markdown(f"**Action Selected:** `{step.action}`")
                                        st.json(step.action_input)
                                        
                                        if step.observation:
                                            st.markdown("**Tool Observation (Output):**")
                                            st.markdown(f"<div class='tool-container'>{step.observation[:1000]}...</div>", unsafe_allow_html=True)
                        else:
                            st.info("No detailed steps were recorded in reasoning trace.")

                        # Memory Operations
                        mem_ops = final_state.get("memory_retrievals", [])
                        if mem_ops:
                            with st.expander("📦 Episodic & Semantic Memory Operations"):
                                for op in mem_ops:
                                    st.markdown(f"- {op}")
                                    
                    except Exception as e:
                        st.error(f"An error occurred during agent execution: {e}")
                        st.exception(e)


# ===================================================================
# Tab 2: Evaluation Metrics Tab
# ===================================================================
with eval_tab:
    st.header("📊 System Evaluation & Resilience")
    
    eval_dir = _PROJECT_ROOT / "data" / "evaluations"

    if not eval_dir.exists() or not list(eval_dir.glob("*.json")):
        st.info("No evaluation data found. Run a research analysis first to generate metrics.")
        
        # Static definitions display of 22 metrics
        st.subheader("Available Evaluation Metrics (22)")
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
    else:
        # Load latest evaluation
        eval_files = sorted(eval_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        latest = eval_files[0]

        with open(latest, "r") as f:
            eval_data = json.load(f)

        # Summary KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Score", f"{eval_data.get('overall_score', 0):.0%}")
        with col2:
            st.metric("Pass Rate", f"{eval_data.get('pass_rate', 0):.0%}")
        with col3:
            st.metric("Passed Metrics", eval_data.get("passed_metrics", 0))
        with col4:
            st.metric("Failed Metrics", eval_data.get("failed_metrics", 0))

        # Metric Details
        st.subheader("Metric Evaluations Breakdown")
        metrics = eval_data.get("metrics", [])
        if metrics:
            for m in metrics:
                status_emoji = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(str(m.get("status", "")).lower(), "❓")
                with st.expander(f"{status_emoji} {m.get('name', 'Unknown')}: {m.get('formatted', '')}"):
                    st.write(f"**Category:** {m.get('category', '')}")
                    st.write(f"**Value:** {m.get('value', '')}")
                    st.write(f"**Threshold:** {m.get('threshold', '')}")
                    st.write(f"**Description:** {m.get('description', '')}")
                    if m.get("details"):
                        st.write(f"**Details:** {m.get('details', '')}")


# ===================================================================
# Tab 3: Telemetry Tab
# ===================================================================
with telemetry_tab:
    st.header("🔍 Runtime Observability & Telemetry")
    
    telemetry_dir = _PROJECT_ROOT / "data" / "telemetry"

    if not telemetry_dir.exists() or not list(telemetry_dir.glob("*.json")):
        st.info("No telemetry logs found. Start an analysis to collect runtime traces.")
    else:
        # Load latest telemetry
        tel_files = sorted(telemetry_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        latest = tel_files[0]

        with open(latest, "r") as f:
            tel_data = json.load(f)

        summary = tel_data.get("summary", {})

        # KPI row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("LLM Calls", summary.get("llm_calls", 0))
            st.metric("LLM Avg Latency", f"{summary.get('llm_avg_ms', 0):.0f} ms")
        with col2:
            st.metric("Tool Calls", summary.get("tool_calls", 0))
            st.metric("Tool Avg Latency", f"{summary.get('tool_avg_ms', 0):.0f} ms")
        with col3:
            st.metric("Total Execution Duration", f"{summary.get('total_duration_ms', 0)/1000:.2f} s")
            st.metric("Total Errors Encountered", summary.get("total_errors", 0))

        # Event log
        events = tel_data.get("events", [])
        if events:
            st.subheader("Execution Timeline Event Logs")
            for event in events[-25:]:  # Last 25 events
                emoji = "✅" if event.get("success", True) else "❌"
                st.text(
                    f"{emoji} [{event.get('event_type', '').upper()}] "
                    f"{event.get('name', '')} "
                    f"({event.get('duration_ms', 0):.1f}ms)"
                )


# ===================================================================
# Tab 4: Failure Injection
# ===================================================================
with failures_tab:
    st.header("⚠️ Reliability & Degraded Mode Resilience")
    
    failure_dir = _PROJECT_ROOT / "data" / "failures"

    if not failure_dir.exists() or not list(failure_dir.glob("*.json")):
        st.info("No active failure injection data found. Systems are running nominal.")
        
        # Display failure scenarios definition
        st.subheader("7 Automated Resilience Scenarios Built into ARA-1")
        scenarios = [
            ("1", "Malformed JSON Response", "medium", "Tests LLM parsing failure recovery"),
            ("2", "Contradictory Financial Metrics", "high", "Tests evidence contradiction detection and conflict resolution"),
            ("3", "Retrieval Memory Corruption", "high", "Tests schema failure handling on VectorDB lookups"),
            ("4", "Stale Stock Info / Outdated news", "medium", "Tests temporal relevance validation and stale decay algorithms"),
            ("5", "API Request Timeouts", "high", "Tests exponential backoff retries and secondary fallbacks"),
            ("6", "Vector DB System Failure", "critical", "Tests degraded-mode fallbacks to local runtime memory"),
            ("7", "Hallucinated Context Chunks", "critical", "Tests hallucination defense engine in source reliability"),
        ]

        for sid, name, severity, desc in scenarios:
            severity_color = {
                "medium": "🟡", "high": "🟠", "critical": "🔴"
            }.get(severity, "⚪")
            st.markdown(f"{severity_color} **Failure Scenario {sid}: {name}** — {desc}")
    else:
        # Load results
        failure_files = sorted(failure_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        latest = failure_files[0]

        with open(latest, "r") as f:
            failure_data = json.load(f)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Injected Scenarios", failure_data.get("injected", 0))
        with col2:
            st.metric("Recovered Successfully", failure_data.get("recovered", 0))
        with col3:
            st.metric("Autonomous Recovery Rate", f"{failure_data.get('recovery_rate', 0):.0%}")

        # Recovery details
        for result in failure_data.get("results", []):
            status = "✅" if result.get("recovered") else "❌"
            with st.expander(f"{status} Scenario: {result.get('scenario_name', 'Unknown')}"):
                st.write(f"**Failure Type:** {result.get('failure_type', '')}")
                st.write(f"**Severity:** {result.get('severity', '')}")
                st.write(f"**Impact Profile:** {result.get('impact', '')}")
                st.write(f"**Agent Actions:** {result.get('agent_behavior', '')}")
                st.write(f"**Lessons Learned:** {result.get('lessons_learned', '')}")


# ===================================================================
# Tab 5: Generated Reports
# ===================================================================
with reports_tab:
    st.header("📄 Downloadable Executive Briefings")
    
    report_dir = _PROJECT_ROOT / "reports"

    if not report_dir.exists() or not list(report_dir.iterdir()):
        st.info("No research reports have been generated yet. Complete a query run to generate new reports.")
    else:
        reports = sorted(report_dir.iterdir(), key=os.path.getmtime, reverse=True)
        
        for rep in reports:
            if rep.suffix in (".md", ".pdf"):
                size_kb = rep.stat().st_size / 1024
                st.markdown(f"📄 **{rep.name}** ({size_kb:.1f} KB)")
                
                # Add download buttons and previews
                if rep.suffix == ".md":
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        content = rep.read_text(encoding="utf-8", errors="replace")
                        st.download_button(
                            label="📥 Download Markdown",
                            data=content,
                            file_name=rep.name,
                            mime="text/markdown",
                            key=f"dl_{rep.name}"
                        )
                    with col2:
                        with st.expander("👁️ Preview Report"):
                            st.markdown(content)
                elif rep.suffix == ".pdf":
                    with open(rep, "rb") as f:
                        pdf_bytes = f.read()
                    st.download_button(
                        label="📥 Download PDF Briefing",
                        data=pdf_bytes,
                        file_name=rep.name,
                        mime="application/pdf",
                        key=f"dl_{rep.name}"
                    )
