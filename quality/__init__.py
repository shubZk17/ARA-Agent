"""
quality/ — How well did the agent do, and can we see what it did?

    evaluation/      22 metrics scoring a completed run: reasoning depth,
                     tool efficiency, retrieval quality, hallucination
                     detection. Runs after synthesis, writes JSON to
                     data/evaluations/.
    observability/   Telemetry collector and execution tracer. Built, but
                     currently has zero instrumentation call sites — see
                     CLAUDE.md, defect D9.
    dashboard.py     Read-only Streamlit monitor over the JSON these write.
                     Superseded by the root app.py, which does the same
                     plus live analysis.

Important limitation to keep in mind: every metric here scores the
*process*, not the *outcome*. Nothing in this package can tell you whether
a BUY recommendation was correct — that needs the validation/ package
planned in plan.md Phase 8.

Nothing in quality/ is allowed to break a run. It observes and reports;
failures here are non-fatal by design.
"""
