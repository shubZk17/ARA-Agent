"""
agent/nodes.py — LangGraph Node Functions
===========================================

WHY THIS EXISTS:
    In LangGraph, a "node" is a function that:
    1. Receives the current state,
    2. Performs some computation,
    3. Returns a PARTIAL state update.

    LangGraph automatically merges the returned dict into the state.

    This file defines the THREE core nodes of the ReAct loop:

    ┌─────────────────┐
    │  reasoning_node  │ ← Calls LLM, parses response
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  should_continue │ ← Conditional edge (not a node)
    └────────┬────────┘
         ┌───┴───┐
    ┌────▼───┐ ┌─▼───────┐
    │ tool_  │ │ output_ │
    │ node   │ │ node    │
    └────┬───┘ └─────────┘
         │
         └──→ back to reasoning_node

HOW IT CONNECTS:
    - agent/graph.py wires these nodes into a StateGraph.
    - Each node reads from AgentState and returns partial updates.
    - config/settings.py provides LLM configuration.
    - tools/registry.py provides tool execution.
    - parsers/react_parser.py parses LLM output.
    - prompts/system.py builds the system prompt.

DESIGN DECISIONS:
    - Nodes are PURE FUNCTIONS (state in → partial state out) for testability.
    - LLM client is initialized once at module level (not per-call).
    - Error handling at every boundary prevents cascade failures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState, AgentStatus, ReasoningStep, ToolCall
from config.settings import settings
from agent.react_parser import parse_react_response
from agent.prompts import build_observation_message, build_system_prompt
from config.horizons import (
    DEFAULT_HORIZON,
    DEFAULT_RISK_PROFILE,
    get_horizon_profile,
)
from tools.registry import ToolRegistry
from quality.observability.collector import EventType
from utils.logger import get_logger

logger = get_logger(__name__)

# How many consecutive unparseable LLM responses to tolerate before giving up
# and producing a partial answer (D7). Low on purpose: if the model can't emit
# JSON twice in a row, a third attempt rarely helps and burns tokens.
MAX_PARSE_RETRIES = 2

# ---------------------------------------------------------------------------
# Module-level LLM client
# ---------------------------------------------------------------------------
# _llm caches ONLY the default (config/env) client. A per-run override — a
# user-supplied API key from the web UI — is built fresh each call and never
# cached, so one visitor's key can't be reused for the next. Set by
# build_graph() each run, same DI pattern as set_observability().
_llm = None
_llm_override: dict | None = None


def set_llm_override(override: dict | None) -> None:
    """Per-run LLM credentials, or None to use config/env.

    Keys (all optional): 'api_key', 'provider' ('groq'|'openai'|'claude'),
    'model'. Anything absent falls back to config.settings.
    """
    global _llm_override
    _llm_override = override or None


def _resolve_llm_config() -> tuple[str, str, str]:
    """(provider, model, api_key) — override wins field-by-field over settings."""
    o = _llm_override or {}
    provider = o.get("provider") or settings.llm_provider
    if provider == "claude":
        return provider, o.get("model") or settings.anthropic_model, o.get("api_key") or settings.anthropic_api_key
    if provider == "groq":
        return provider, o.get("model") or settings.groq_model, o.get("api_key") or settings.groq_api_key
    return provider, o.get("model") or settings.openai_model, o.get("api_key") or settings.openai_api_key


def _build_llm(provider: str, model: str, api_key: str):
    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=api_key, temperature=0.1, max_tokens=4096)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=api_key, temperature=0.1, max_tokens=4096)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, api_key=api_key, temperature=0.1, max_tokens=4096)


def _get_llm():
    """Lazy LLM client. Lazy so importing this module needs no API key."""
    global _llm
    provider, model, api_key = _resolve_llm_config()

    if _llm_override:
        # Don't retain a user's key in a module global across runs.
        return _build_llm(provider, model, api_key)

    if _llm is None:
        _llm = _build_llm(provider, model, api_key)
        logger.info(f"Initialized LLM: {model}")
    return _llm


# ===================================================================
# We need a module-level reference to the tool registry.
# This is set by the graph builder in graph.py before execution.
# ===================================================================
_tool_registry: ToolRegistry | None = None


def set_tool_registry(registry: ToolRegistry) -> None:
    """Set the tool registry for node functions to use."""
    global _tool_registry
    _tool_registry = registry


# ===================================================================
# Phase 2: Module-level references for ingestion and retrieval.
# Set by graph.py at build time, alongside the tool registry.
# ===================================================================
_ingestion_pipeline = None
_semantic_retriever = None


def set_ingestion_pipeline(pipeline) -> None:
    """Set the ingestion pipeline for auto-storing tool outputs."""
    global _ingestion_pipeline
    _ingestion_pipeline = pipeline


def set_semantic_retriever(retriever) -> None:
    """Set the semantic retriever for evidence retrieval."""
    global _semantic_retriever
    _semantic_retriever = retriever


# ===================================================================
# Phase 4 (wired 2026-08-27): Module-level references for observability.
# Both default to None so an unset collector/tracer is a pure no-op —
# nodes must behave identically whether or not a caller wires telemetry.
# ===================================================================
_collector = None
_tracer = None


def set_observability(collector, tracer) -> None:
    """Set the telemetry collector and execution tracer for node functions."""
    global _collector, _tracer
    _collector = collector
    _tracer = tracer


# ===================================================================
# NODE 1: Reasoning Node
# ===================================================================

def reasoning_node(state: AgentState) -> dict[str, Any]:
    """
    Call the LLM to generate the next thought + action.

    This is the BRAIN of the agent. It:
    1. Builds the system prompt with current context,
    2. Constructs the message history,
    3. Calls the LLM,
    4. Parses the response,
    5. Returns a state update.

    Returns:
        Partial state update with current_thought, current_action,
        current_action_input, and updated iteration_count.
    """
    iteration = state["iteration_count"] + 1
    logger.info(f"[bold cyan]=== Iteration {iteration}/{state['max_iterations']} ===[/bold cyan]")

    # Build system prompt with tool descriptions
    tool_descriptions = ""
    if _tool_registry:
        tool_descriptions = _tool_registry.get_tool_descriptions()

    system_prompt = build_system_prompt(
        query=state["query"],
        tool_descriptions=tool_descriptions,
        iteration=iteration,
        max_iterations=state["max_iterations"],
        retrieval_context=_get_retrieval_context(state),
        episodic_context=state.get("episodic_context", ""),
        horizon=state.get("horizon", DEFAULT_HORIZON),
        risk_profile=state.get("risk_profile", DEFAULT_RISK_PROFILE),
    )

    # Build message list
    messages = [SystemMessage(content=system_prompt)]

    # Add previous reasoning as conversation history
    for step in state.get("reasoning_trace", []):
        if step.thought:
            # The agent's previous response
            prev_response = {
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
            }
            messages.append(
                HumanMessage(content=f"Your previous response:\n{prev_response}")
            )
        if step.observation:
            # The observation from the tool
            observation_msg = build_observation_message(
                action=step.action,
                action_input=step.action_input,
                observation=step.observation,
            )
            messages.append(HumanMessage(content=observation_msg))

    # If there's a current observation (from the last tool execution),
    # add it as the latest message
    if state.get("current_observation"):
        observation_msg = build_observation_message(
            action=state.get("current_action", ""),
            action_input=state.get("current_action_input", {}),
            observation=state["current_observation"],
        )
        messages.append(HumanMessage(content=observation_msg))
    elif iteration == 1:
        # First iteration — just send the query
        messages.append(HumanMessage(content=f"Please analyze: {state['query']}"))

    # The model offered a final answer without gathering the evidence this
    # horizon requires, and should_continue sent it back here. Tell it exactly
    # what is missing — a repeat of the general rule it already ignored would
    # not change anything.
    missing = (
        _missing_required_tools(state)
        if state.get("current_action", "").lower().strip() == "final_answer"
        else set()
    )
    if missing:
        messages.append(HumanMessage(content=(
            f"REJECTED: you produced a final answer without calling "
            f"{', '.join(sorted(missing))} in this run.\n"
            f"Any figure you stated therefore came from retrieved notes about "
            f"EARLIER runs, not from current data — which is exactly what rule 5 "
            f"forbids. Call {sorted(missing)[0]} now and respond with the tool "
            f"action JSON only."
        )))

    # Call LLM
    span = _tracer.start_span("reasoning_llm", "llm", iteration=iteration) if _tracer else None
    try:
        logger.debug(f"Calling LLM with {len(messages)} messages")
        llm = _get_llm()
        if _collector:
            with _collector.track(EventType.LLM_CALL, "reasoning_llm", iteration=iteration):
                response = llm.invoke(messages)
        else:
            response = llm.invoke(messages)
        raw_text = response.content
        logger.debug(f"LLM response: {raw_text[:300]}...")
        if span:
            _tracer.end_span(span, output_summary=raw_text[:200])
    except Exception as e:
        if span:
            _tracer.end_span(span, success=False, error=str(e))
        error_msg = f"LLM call failed: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        return {
            "current_thought": f"Error calling LLM: {str(e)}",
            "current_action": "final_answer",
            "current_action_input": {"answer": f"I encountered an error communicating with the LLM: {str(e)}"},
            "iteration_count": iteration,
            "status": AgentStatus.ERROR,
            "errors": [error_msg],
        }

    # Parse response
    parsed = parse_react_response(raw_text)

    if parsed.parse_error and not parsed.is_valid:
        retries = state.get("parse_retry_count", 0) + 1
        logger.warning(f"Parse error (retry {retries}/{MAX_PARSE_RETRIES}): {parsed.parse_error}")
        # Feed the error back as an observation and let should_continue route
        # us to reasoning again. Before D7 was fixed this fell through to
        # output_node, so a single malformed response ended the whole run.
        return {
            "current_thought": f"Parse error occurred: {parsed.parse_error}",
            "current_action": "",
            "current_action_input": {},
            "current_observation": f"Your previous response was not valid JSON. Error: {parsed.parse_error}. Please respond with ONLY a valid JSON object.",
            "iteration_count": iteration,
            "parse_retry_count": retries,
            "status": AgentStatus.REASONING,
            "errors": [f"Parse error at iteration {iteration}: {parsed.parse_error}"],
        }

    logger.info(f"[bold green]Thought:[/bold green] {parsed.thought[:150]}...")
    if parsed.is_final_answer:
        logger.info("[bold yellow]>> Final Answer[/bold yellow]")
    else:
        logger.info(f"[bold blue]>> Action:[/bold blue] {parsed.action}({parsed.action_input})")

    return {
        "current_thought": parsed.thought,
        "current_action": parsed.action,
        "current_action_input": parsed.action_input,
        "iteration_count": iteration,
        "parse_retry_count": 0,  # A good response clears the retry budget
        "status": AgentStatus.REASONING,
    }


# ===================================================================
# NODE 2: Tool Execution Node
# ===================================================================

def tool_node(state: AgentState) -> dict[str, Any]:
    """
    Execute the tool selected by the reasoning node.

    This node:
    1. Looks up the tool in the registry,
    2. Validates it exists,
    3. Calls tool.execute() (which handles its own errors),
    4. Records the tool call and observation,
    5. Returns a state update.

    Returns:
        Partial state update with current_observation, tool_calls
        (appended), and reasoning_trace (appended).
    """
    action = state["current_action"]
    action_input = state["current_action_input"]
    iteration = state["iteration_count"]

    logger.info(f"[bold magenta]Executing tool:[/bold magenta] {action}")

    if not _tool_registry:
        error_msg = "Tool registry not initialized"
        logger.error(error_msg)
        return _build_tool_error_update(action, action_input, error_msg, iteration)

    # Look up tool
    tool = _tool_registry.get(action)
    if tool is None:
        available = _tool_registry.list_tools()
        error_msg = (
            f"Tool '{action}' not found. Available tools: {', '.join(available)}"
        )
        logger.warning(error_msg)
        return _build_tool_error_update(action, action_input, error_msg, iteration)

    # Execute tool
    span = _tracer.start_span(action, "tool", iteration=iteration) if _tracer else None
    if _collector:
        with _collector.track(
            EventType.TOOL_EXECUTION, action, iteration=iteration,
            metadata={"input": action_input},
        ):
            result = tool.execute(action_input)
    else:
        result = tool.execute(action_input)
    if span:
        _tracer.end_span(
            span,
            output_summary=(result.data[:200] if result.success else result.error),
            success=result.success,
            error="" if result.success else result.error,
        )

    if result.success:
        logger.info(f"[bold green]Tool succeeded[/bold green] ({len(result.data)} chars)")
        logger.debug(f"Tool output: {result.data[:200]}...")
    else:
        logger.warning(f"Tool failed: {result.error}")

    # Build observation string
    observation = result.data if result.success else f"Tool Error: {result.error}"

    # Record tool call
    tool_call = ToolCall(
        tool_name=action,
        tool_input=action_input,
        tool_output=observation,
        tool_output_structured=result.structured,
        success=result.success,
        error_message=result.error if not result.success else "",
    )

    # Record reasoning step
    reasoning_step = ReasoningStep(
        iteration=iteration,
        thought=state.get("current_thought", ""),
        action=action,
        action_input=action_input,
        observation=observation,
    )

    return {
        "current_observation": observation,
        "tool_calls": [tool_call],
        "reasoning_trace": [reasoning_step],
        "status": AgentStatus.OBSERVING,
        "evidence_confidence": _score_evidence(action, iteration, result.success),
        "conflict_reports": _extract_conflicts(action, result),
        **_auto_ingest_tool_output(action, action_input, observation, result.success),
    }


def _extract_conflicts(tool_name: str, result) -> list[dict]:
    """
    Lift any disagreements a tool reported into AgentState.conflict_reports.

    Generic on purpose: a tool that can check a figure against a second
    derivation puts them under `conflicts` in its structured payload, and this
    one line carries them into state — no per-tool branch here, ever.

    Until Phase 6 nothing wrote this field (every tool read the same endpoint
    and had nothing to disagree with), which is why the confidence
    calibrator's conflict branch had never executed. tools/market_context.py
    is the first writer.
    """
    if not result.success:
        return []

    conflicts = result.structured.get("conflicts") if result.structured else None
    if not isinstance(conflicts, list) or not conflicts:
        return []

    tagged = []
    for conflict in conflicts:
        if isinstance(conflict, dict):
            tagged.append({**conflict, "detected_by": tool_name})

    if tagged:
        logger.warning(
            f"{tool_name} reported {len(tagged)} evidence conflict(s) — "
            f"flagged, not resolved"
        )
    return tagged


def _score_evidence(tool_name: str, iteration: int, success: bool) -> dict[str, float]:
    """
    Score this tool output's reliability and return it as an evidence_confidence
    fragment, merged into state by the operator.or_ reducer.

    WHY THIS EXISTS (defect D4): ReliabilityScorer was built in Phase 2 but
    nothing ever called it during a run, so `evidence_confidence` stayed empty
    and confidence_calibrator._score_source_reliability fell through to a
    constant. Every successful run reported ~90% confidence no matter what
    evidence it had. This is the write that makes that branch reachable.
    """
    if not success:
        # A failed call is evidence of nothing. Recording it as 0.0 rather
        # than omitting it is deliberate: it drags the average down, which is
        # the honest signal.
        return {f"{tool_name}:{iteration}": 0.0}

    try:
        from knowledge.reliability.scorer import ReliabilityScorer

        score = ReliabilityScorer().score(
            source_name=tool_name,
            source_type="tool_output",
            document_date=datetime.now(timezone.utc).isoformat(),
        )
        return {f"{tool_name}:{iteration}": score}
    except Exception as e:
        logger.warning(f"Reliability scoring failed (non-fatal): {e}")
        return {}


def _build_tool_error_update(
    action: str,
    action_input: dict,
    error_msg: str,
    iteration: int,
) -> dict[str, Any]:
    """Helper to build a consistent error state update for tool failures."""
    tool_call = ToolCall(
        tool_name=action,
        tool_input=action_input,
        tool_output="",
        success=False,
        error_message=error_msg,
    )
    reasoning_step = ReasoningStep(
        iteration=iteration,
        thought="",
        action=action,
        action_input=action_input,
        observation=f"Error: {error_msg}",
    )
    return {
        "current_observation": f"Error: {error_msg}",
        "tool_calls": [tool_call],
        "reasoning_trace": [reasoning_step],
        "status": AgentStatus.OBSERVING,
        "errors": [error_msg],
    }


# ===================================================================
# NODE 3: Output Node
# ===================================================================

def output_node(state: AgentState) -> dict[str, Any]:
    """
    Produce the final output.

    This node runs when:
    - The agent decided to give a final answer, OR
    - Max iterations were reached.

    It extracts the final answer and sets the terminal status.
    """
    action_input = state.get("current_action_input", {})

    if state["current_action"] == "final_answer":
        # Agent chose to stop — extract the answer
        final_answer = action_input.get("answer", "")
        if not final_answer:
            # Try other common keys
            final_answer = action_input.get("response", "")
            if not final_answer:
                final_answer = str(action_input) if action_input else "No answer provided."

        # Record the final reasoning step
        reasoning_step = ReasoningStep(
            iteration=state["iteration_count"],
            thought=state.get("current_thought", ""),
            action="final_answer",
            action_input=action_input,
            observation="[Final answer produced]",
        )

        logger.info("[bold green][OK] Agent completed with final answer[/bold green]")
        return {
            "final_answer": final_answer,
            "status": AgentStatus.COMPLETED,
            "reasoning_trace": [reasoning_step],
        }
    else:
        # Max iterations reached — synthesize from what we have
        logger.warning(
            f"Max iterations ({state['max_iterations']}) reached. "
            "Producing partial analysis."
        )
        # Compile what we learned from tool calls
        observations = [
            step.observation
            for step in state.get("reasoning_trace", [])
            if step.observation
        ]
        partial = (
            f"[Analysis incomplete - reached {state['max_iterations']} iteration limit]\n\n"
            f"Based on gathered information:\n\n"
            + "\n\n".join(observations[-3:])  # Last 3 observations
        )

        return {
            "final_answer": partial,
            "status": AgentStatus.MAX_ITERATIONS_REACHED,
        }


# ===================================================================
# CONDITIONAL EDGE: Should Continue?
# ===================================================================

def should_continue(state: AgentState) -> str:
    """
    Conditional routing function for LangGraph.

    Returns a string that maps to a node name via the graph edges.
    This is the DECISION POINT of the ReAct loop.

    Returns:
        "tool_node" — if the agent wants to execute a tool
        "reasoning_node" — if the response was unparseable and retries remain,
                           or a final answer was offered without the evidence
        "output_node" — if the agent has a final answer or hit limits
    """
    # Check for error status FIRST. reasoning_node's LLM-failure path sets
    # current_action="final_answer" to carry the error message out, so testing
    # for a final answer before testing for an error would treat a dead LLM as
    # a real answer and send it round the evidence guard below — burning
    # iterations on a call that cannot succeed.
    if state.get("status") == AgentStatus.ERROR:
        return "output_node"

    # Check for final answer
    if state.get("current_action", "").lower().strip() == "final_answer":
        missing = _missing_required_tools(state)
        if missing and state["iteration_count"] < state["max_iterations"]:
            logger.warning(
                f"Final answer offered without calling {', '.join(sorted(missing))} "
                f"— sending the model back to gather evidence"
            )
            return "reasoning_node"
        return "output_node"

    # Check iteration limit
    if state["iteration_count"] >= state["max_iterations"]:
        logger.warning("Iteration limit reached - routing to output")
        return "output_node"

    # Check if there's an action to execute
    if state.get("current_action"):
        return "tool_node"

    # No action and no final answer — a parse error. Retry reasoning while
    # the budget lasts (D7). Bounded twice over: by MAX_PARSE_RETRIES and by
    # the iteration limit above, which reasoning_node increments every pass.
    retries = state.get("parse_retry_count", 0)
    if retries < MAX_PARSE_RETRIES:
        logger.info(f"Unparseable response — retrying reasoning ({retries}/{MAX_PARSE_RETRIES})")
        return "reasoning_node"

    logger.warning(f"Giving up after {retries} unparseable responses")
    return "output_node"


def _missing_required_tools(state: AgentState) -> set[str]:
    """
    Which of this horizon's required tools have not run successfully yet.

    WHY THIS IS ENFORCED IN CODE RATHER THAN IN THE PROMPT:
        agent/prompts.py has said "you MUST call get_financial_metrics and
        get_stock_price" since Phase 5, and models still hand back a fully
        confident answer having called nothing — assembled out of the
        retrieved notes from EARLIER runs that the retriever helpfully put in
        front of them. Phase 5 documented that regression and answered it with
        stronger wording; it came back the moment the corpus got richer, which
        is what a prose rule is worth against a model that thinks it already
        knows the answer.

        The horizon profile already declares required_tools. Checking it here
        makes the requirement structural: the answer is not accepted until the
        evidence exists, whatever the model is and however the prompt is
        phrased. Bounded by max_iterations, so it cannot loop forever.

    Only SUCCESSFUL calls count — a failed fetch gathered nothing, and
    accepting it would let one bad ticker unlock an evidence-free answer.
    """
    profile = get_horizon_profile(state.get("horizon", DEFAULT_HORIZON))
    available = set(_tool_registry.list_tools()) if _tool_registry else set()

    called = {
        call.tool_name for call in state.get("tool_calls", [])
        if getattr(call, "success", False)
    }

    # Never demand a tool this deployment does not have registered — that
    # would loop to the iteration limit and produce nothing.
    return (set(profile.required_tools) & available) - called


# ===================================================================
# Phase 2: Helper Functions
# ===================================================================

def _get_retrieval_context(state: AgentState) -> str:
    """
    Retrieve relevant evidence from vector memory for the current query.

    Called by reasoning_node to inject evidence context into the prompt.
    Returns empty string if retrieval is unavailable or vector store is empty.
    """
    if _semantic_retriever is None:
        return ""

    try:
        # Extract ticker from query if possible (simple heuristic)
        query = state.get("query", "")
        ticker = _extract_ticker_from_state(state)

        evidence_list = _semantic_retriever.retrieve(
            query=query,
            top_k=5,
            ticker_filter=ticker,
        )

        if not evidence_list:
            return ""

        formatted = _semantic_retriever.format_evidence_for_prompt(evidence_list)
        return formatted

    except Exception as e:
        logger.warning(f"Retrieval context failed (non-fatal): {e}")
        return ""


def _auto_ingest_tool_output(
    tool_name: str,
    tool_input: dict,
    observation: str,
    success: bool,
) -> dict:
    """
    Auto-ingest successful tool outputs into vector memory.

    Returns additional state fields to merge (retrieved_evidence, etc.).
    Returns empty dict if ingestion is unavailable or fails.
    """
    if _ingestion_pipeline is None:
        return {}

    if not success:
        return {}

    try:
        ticker = tool_input.get("ticker", "")
        chunks_stored = _ingestion_pipeline.ingest_tool_output(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=observation,
            ticker=ticker,
        )
        if chunks_stored > 0:
            logger.debug(
                f"Auto-ingested {chunks_stored} chunks from {tool_name} "
                f"into vector memory"
            )
            return {"memory_retrievals": [f"ingested:{tool_name}:{ticker}:{chunks_stored}chunks"]}
    except Exception as e:
        logger.warning(f"Auto-ingestion failed (non-fatal): {e}")

    return {}


def _extract_ticker_from_state(state: AgentState) -> str:
    """
    Extract the primary ticker symbol from the agent state.

    Looks at tool calls and query for ticker references.
    """
    # Check tool calls first (most reliable)
    for tc in state.get("tool_calls", []):
        if hasattr(tc, "tool_input"):
            ticker = tc.tool_input.get("ticker", "")
            if ticker:
                return ticker.upper()

    # Fall back to query keyword extraction (simple heuristic)
    query = state.get("query", "").upper()
    # Common ticker patterns: 1-5 uppercase letters
    import re
    # Look for known ticker mentions (all caps, 1-5 chars)
    candidates = re.findall(r'\b([A-Z]{1,5})\b', query)
    # Filter out common non-ticker words
    noise_words = {
        "A", "I", "THE", "AND", "OR", "FOR", "OF", "IN", "ON", "AT",
        "TO", "IS", "IT", "AN", "AS", "BY", "BE", "DO", "IF", "MY",
        "UP", "SO", "NO", "NOT", "ARE", "WAS", "HAS", "HAD", "CAN",
        "ALL", "BUT", "OUT", "HER", "HIS", "ITS", "OUR", "WHO",
        "NOW", "OLD", "NEW", "MAY", "WAY", "DAY", "TOO", "USE",
    }
    for c in candidates:
        if c not in noise_words and len(c) >= 2:
            return c

    return ""
