"""
Tier 4 — contract invariants.

The 2026-08-15 audit found nine AgentState fields that no node ever wrote.
That finding was true for three months and nothing noticed, because nothing
could notice. These tests turn both halves of the contract into something
that fails loudly instead of rotting quietly.
"""

from __future__ import annotations

from pathlib import Path

from agent.state import AgentState, create_initial_state

ROOT = Path(__file__).resolve().parent.parent

# Fields no node ever touches, each with a reason. Adding a name here is a
# deliberate act; a field that lands here by accident fails the test below.
KNOWN_UNUSED: dict[str, str] = {
    "start_time": "set once by create_initial_state; immutable for the run",
    "retrieved_evidence": "Phase 6/7 — retrieval writes to the prompt, not to state",
    "retrieval_queries": "Phase 6/7 — as above",
    "conflict_reports": "Phase 6.5/7.4 — needs a second source that can disagree",
    "telemetry_events": "plan §5.7 — observability is to be wired, not cut",
}


def test_initial_state_covers_every_declared_field():
    """
    TypedDict enforces nothing at runtime, so a field added to AgentState but
    forgotten in create_initial_state() is a KeyError waiting for the one
    code path that reads it without .get().
    """
    declared = set(AgentState.__annotations__)
    created = set(create_initial_state("test query"))
    assert declared == created, (
        f"missing from create_initial_state: {declared - created} · "
        f"unexpected extras: {created - declared}"
    )


def test_every_state_field_is_written_or_declared_unused():
    """
    Every AgentState field must either be referenced by the agent package or
    be listed in KNOWN_UNUSED with a reason. Dead state is a lie about what
    the system does — `retry_count` claimed retries existed for months.
    """
    sources = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (ROOT / "agent").glob("*.py")
        if p.name != "state.py"
    )

    unwritten = [
        field for field in AgentState.__annotations__
        if f'"{field}"' not in sources and f"'{field}'" not in sources
    ]

    undeclared = set(unwritten) - set(KNOWN_UNUSED)
    assert not undeclared, (
        f"These AgentState fields are never written by any node and are not "
        f"listed in KNOWN_UNUSED: {sorted(undeclared)}. Either write them or "
        f"declare them dead."
    )

    stale = set(KNOWN_UNUSED) - set(unwritten)
    assert not stale, (
        f"These are listed in KNOWN_UNUSED but ARE now written — remove them "
        f"from the list: {sorted(stale)}"
    )


def test_accumulating_fields_have_reducers():
    """
    A list or dict field without an Annotated reducer is REPLACED on every
    node update rather than merged. That was defect D5: evidence_confidence
    would have silently discarded all but the last tool call's scores.
    """
    from typing import Annotated, get_args, get_origin, get_type_hints

    # state.py uses `from __future__ import annotations`, so the raw
    # __annotations__ are strings. get_type_hints resolves them — with
    # include_extras, or the Annotated wrapper is stripped and every field
    # would look reducer-less. LangGraph resolves them the same way.
    hints = get_type_hints(AgentState, include_extras=True)

    must_accumulate = {
        "reasoning_trace", "tool_calls", "errors", "retrieved_evidence",
        "retrieval_queries", "evidence_confidence", "conflict_reports",
        "memory_retrievals", "telemetry_events",
    }

    for field in must_accumulate:
        annotation = hints[field]
        assert get_origin(annotation) is Annotated, (
            f"{field} accumulates across iterations but has no reducer — "
            f"LangGraph will replace it instead of merging"
        )
        assert len(get_args(annotation)) >= 2, f"{field} has no reducer function"
