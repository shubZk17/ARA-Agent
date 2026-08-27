"""
agent/graph.py — LangGraph StateGraph Definition
==================================================

WHY THIS EXISTS:
    This module defines the agent's execution graph — the state machine
    that orchestrates the ReAct loop. It wires together the nodes from
    nodes.py into a directed graph with conditional edges.

    WHY LANGGRAPH instead of a plain while loop?

    1. VISIBILITY — The graph structure is explicit and inspectable.
       You can visualize it, test individual nodes, and reason about
       control flow without reading procedural code.

    2. STATE MANAGEMENT — LangGraph handles state merging, reducer
       functions (operator.add), and state validation automatically.

    3. EXTENSIBILITY — Adding new nodes (memory retrieval, conflict
       resolution, human-in-the-loop) is adding a node + edge,
       not refactoring a monolithic loop.

    4. CHECKPOINTING (Phase 2+) — LangGraph can persist state at each
       node boundary, enabling pause/resume, time-travel debugging,
       and crash recovery.

    5. STREAMING (Phase 2+) — LangGraph natively supports streaming
       intermediate results to the UI as nodes execute.

HOW IT CONNECTS:
    - main.py calls build_graph() to get a compiled graph.
    - main.py calls graph.invoke(initial_state) to run the agent.
    - The graph calls nodes from agent/nodes.py.
    - Nodes use tools/registry.py and parsers/react_parser.py.

GRAPH STRUCTURE (Phase 2):

    START → reasoning_node → should_continue?
                                ├── "tool_node" → tool_node → reasoning_node
                                └── "output_node" → output_node → END

    Phase 2 additions:
    - tool_node now auto-ingests outputs into vector memory.
    - reasoning_node now retrieves evidence from vector memory.
    - Both are handled within the existing nodes (no new graph nodes needed).
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, StateGraph

from agent.nodes import (
    output_node,
    reasoning_node,
    set_tool_registry,
    set_ingestion_pipeline,
    set_semantic_retriever,
    set_observability,
    should_continue,
    tool_node,
)
from agent.state import AgentState
from tools.registry import ToolRegistry
from utils.logger import get_logger

logger = get_logger(__name__)


def build_graph(
    tool_registry: ToolRegistry,
    ingestion_pipeline=None,
    semantic_retriever=None,
    collector=None,
    tracer=None,
) -> StateGraph:
    """
    Build and compile the ReAct agent graph.

    Args:
        tool_registry: Registry of available tools. Injected here
                       so the graph and its nodes have access to tools.
        ingestion_pipeline: Phase 2 — Pipeline for auto-ingesting tool outputs.
        semantic_retriever: Phase 2 — Retriever for evidence retrieval.
        collector: Phase 4 — TelemetryCollector, optional. None is a no-op.
        tracer: Phase 4 — ExecutionTracer, optional. None is a no-op.

    Returns:
        Compiled LangGraph StateGraph ready for .invoke().

    Architecture Note:
        We inject dependencies at build time rather than passing
        them through state because:
        1. These don't change during a single agent run.
        2. They contain callable objects (not serializable in state).
        3. It keeps the state focused on DATA, not infrastructure.
    """
    # Inject dependencies into the nodes module
    set_tool_registry(tool_registry)

    # Phase 2: Inject ingestion and retrieval systems
    if ingestion_pipeline is not None:
        set_ingestion_pipeline(ingestion_pipeline)
        logger.info("Phase 2: Ingestion pipeline connected to graph")

    if semantic_retriever is not None:
        set_semantic_retriever(semantic_retriever)
        logger.info("Phase 2: Semantic retriever connected to graph")

    # Phase 4: Inject observability (no-op inside nodes.py if both are None)
    set_observability(collector, tracer)

    # --- Define the graph ---
    graph = StateGraph(AgentState)

    # --- Add nodes ---
    # Each node is a function: (state) -> partial_state_update
    graph.add_node("reasoning_node", reasoning_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("output_node", output_node)

    # --- Set entry point ---
    graph.set_entry_point("reasoning_node")

    # --- Add conditional edge from reasoning ---
    # After reasoning, should_continue() decides the next node
    graph.add_conditional_edges(
        source="reasoning_node",
        path=should_continue,  # function that returns node name
        path_map={
            "tool_node": "tool_node",
            "output_node": "output_node",
            # Self-loop for unparseable responses (D7). Bounded by
            # MAX_PARSE_RETRIES and by the iteration limit.
            "reasoning_node": "reasoning_node",
        },
    )

    # --- After tool execution, always go back to reasoning ---
    # This creates the loop: reason → act → observe → reason → ...
    graph.add_edge("tool_node", "reasoning_node")

    # --- Output node is terminal ---
    graph.add_edge("output_node", END)

    # --- Compile ---
    compiled = graph.compile()
    logger.info("Agent graph compiled successfully")

    return compiled
