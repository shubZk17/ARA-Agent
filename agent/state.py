"""
agent/state.py — Agent State Design
=====================================

WHY THIS EXISTS:
    The agent state is the SINGLE SOURCE OF TRUTH that flows through
    every node in the LangGraph graph. It carries:
    - what the user asked,
    - what the agent has thought so far,
    - what tools were called and what they returned,
    - how many iterations have elapsed,
    - whether we have a final answer.

    Without a well-designed state object, you end up passing dozens of
    loose variables between functions, which is:
    - impossible to debug (what was the state at step 3?),
    - impossible to extend (adding a field means touching every function),
    - impossible to serialize (can't save/restore agent sessions).

HOW LANGGRAPH USES STATE:
    LangGraph passes a state dict to every node function. Each node
    receives the current state, does its work, and returns a PARTIAL
    state update (only the fields that changed). LangGraph merges the
    update into the current state automatically.

    This is why we use TypedDict — it gives us type safety while
    remaining dict-compatible for LangGraph.

DESIGN DECISIONS:
    - TypedDict for LangGraph compatibility (it expects dict-like objects).
    - Pydantic models for NESTED structured data (ToolCall, ReasoningStep)
      because these need validation and serialization.
    - Operator.add annotation on list fields so LangGraph APPENDS
      rather than REPLACES when merging updates.

SCALABILITY:
    Phase 2 adds: retrieved_evidence, retrieval_queries, evidence_confidence,
    conflict_reports, episodic_context, memory_retrievals.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ===================================================================
# Enums — Agent lifecycle states
# ===================================================================

class AgentStatus(str, Enum):
    """
    Tracks where the agent is in its lifecycle.

    Why an enum instead of a string?
    - Prevents typos (status = "runnng" won't compile).
    - IDE autocomplete.
    - Exhaustive match in conditional logic.
    """
    IDLE = "idle"
    REASONING = "reasoning"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETED = "completed"
    ERROR = "error"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"


# ===================================================================
# Pydantic Models — Structured nested data
# ===================================================================

class ToolCall(BaseModel):
    """
    Record of a single tool invocation.

    We store both input and output so the full execution trace
    is available for debugging and evaluation.
    """
    tool_name: str = Field(description="Name of the tool invoked")
    tool_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool"
    )
    tool_output: str = Field(default="", description="Raw tool output")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of invocation"
    )
    success: bool = Field(default=True, description="Whether the tool executed without error")
    error_message: str = Field(default="", description="Error details if success=False")


class ReasoningStep(BaseModel):
    """
    One iteration of the ReAct loop: Thought → Action → Observation.

    This is the atomic unit of agent reasoning. A complete agent run
    consists of a list of these steps, forming the reasoning trace.
    """
    iteration: int = Field(description="1-indexed iteration number")
    thought: str = Field(default="", description="Agent's reasoning about what to do next")
    action: str = Field(default="", description="Tool name selected (empty if final answer)")
    action_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the selected tool"
    )
    observation: str = Field(default="", description="Result returned by the tool")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ===================================================================
# LangGraph State — TypedDict for graph compatibility
# ===================================================================

class AgentState(TypedDict):
    """
    The core state object that flows through the LangGraph graph.

    IMPORTANT LANGGRAPH CONVENTION:
    Fields annotated with `Annotated[list[X], operator.add]` use the
    ADD reducer — when a node returns {"reasoning_trace": [new_step]},
    LangGraph APPENDS new_step to the existing list rather than
    replacing the entire list. This is critical for accumulating
    history across iterations.

    Fields WITHOUT the annotation are REPLACED on each update.
    """

    # --- User Input ---
    query: str  # Original user question

    # --- Reasoning History (append-only) ---
    reasoning_trace: Annotated[list[ReasoningStep], operator.add]
    tool_calls: Annotated[list[ToolCall], operator.add]

    # --- Current Iteration State ---
    current_thought: str          # Latest thought from the LLM
    current_action: str           # Latest tool name selected
    current_action_input: dict[str, Any]  # Latest tool arguments
    current_observation: str      # Latest tool result

    # --- Loop Control ---
    iteration_count: int          # Current iteration number
    max_iterations: int           # Safety limit

    # --- Output ---
    final_answer: str             # Set when agent decides to stop
    status: AgentStatus           # Current lifecycle phase

    # --- Metadata ---
    start_time: str               # ISO timestamp of agent start
    errors: Annotated[list[str], operator.add]  # Accumulated error messages

    # --- Phase 2: Retrieval Context ---
    retrieved_evidence: Annotated[list[dict], operator.add]  # Evidence from vector memory
    retrieval_queries: Annotated[list[str], operator.add]    # Queries sent to retrieval

    # --- Phase 2: Evidence Governance ---
    evidence_confidence: dict[str, float]    # evidence_id → confidence score
    conflict_reports: Annotated[list[dict], operator.add]  # Detected conflicts

    # --- Phase 2: Memory ---
    episodic_context: str   # Prior analysis context from episodic memory
    memory_retrievals: Annotated[list[str], operator.add]  # Memory retrieval log

    # --- Phase 4: Observability ---
    telemetry_events: Annotated[list[dict], operator.add]  # Raw telemetry events
    checkpoint_path: str  # Path to last checkpoint file

    # --- Phase 4: Recovery ---
    retry_count: int  # Total retries in this run
    fallback_provider: str  # Provider used if fallback was triggered
    failure_injections: Annotated[list[str], operator.add]  # Injected failure IDs


def create_initial_state(query: str, max_iterations: int = 10) -> AgentState:
    """
    Factory function to create a properly initialized agent state.

    WHY a factory instead of inline dict construction?
    - Guarantees all fields are present (TypedDict doesn't enforce at runtime).
    - Sets sensible defaults.
    - Single place to modify initialization logic.
    """
    return AgentState(
        query=query,
        reasoning_trace=[],
        tool_calls=[],
        current_thought="",
        current_action="",
        current_action_input={},
        current_observation="",
        iteration_count=0,
        max_iterations=max_iterations,
        final_answer="",
        status=AgentStatus.IDLE,
        start_time=datetime.now(timezone.utc).isoformat(),
        errors=[],
        # Phase 2 fields
        retrieved_evidence=[],
        retrieval_queries=[],
        evidence_confidence={},
        conflict_reports=[],
        episodic_context="",
        memory_retrievals=[],
        # Phase 4 fields
        telemetry_events=[],
        checkpoint_path="",
        retry_count=0,
        fallback_provider="",
        failure_injections=[],
    )
