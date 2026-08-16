"""
Tier 3 — graph and state behaviour, with a stubbed LLM and a fake registry.

Never calls a real LLM or a real API.
"""

from __future__ import annotations

import operator

import pytest

from agent import nodes
from agent.state import AgentStatus, create_initial_state
from tools.base import BaseTool, ToolParameter


# ===================================================================
# Fakes
# ===================================================================

class _StructuredTool(BaseTool):
    """A tool on the new fetch()/render() contract."""

    @property
    def name(self) -> str:
        return "get_financial_metrics"

    @property
    def description(self) -> str:
        return "fake"

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    def fetch(self, tool_input):
        return {"ticker": tool_input.get("ticker", ""), "total_debt": 1_234_567_890.0}

    def render(self, payload):
        return f"Total Debt: ${payload['total_debt'] / 1e9:.2f}B"


class _LegacyTool(BaseTool):
    """A tool that only implements the old single-method contract."""

    @property
    def name(self) -> str:
        return "legacy"

    @property
    def description(self) -> str:
        return "fake"

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    def _execute(self, tool_input):
        return "plain string output"


class _ExplodingTool(_LegacyTool):
    @property
    def name(self) -> str:
        return "exploding"

    def _execute(self, tool_input):
        raise RuntimeError("upstream API is down")


class _FakeRegistry:
    def __init__(self, *tools):
        self._tools = {t.name: t for t in tools}

    def get(self, name):
        return self._tools.get(name)

    def list_tools(self):
        return list(self._tools)

    def get_tool_descriptions(self):
        return ""


@pytest.fixture
def registry():
    reg = _FakeRegistry(_StructuredTool(), _LegacyTool(), _ExplodingTool())
    nodes.set_tool_registry(reg)
    yield reg
    nodes.set_tool_registry(None)


# ===================================================================
# Tool contract
# ===================================================================

def test_structured_tool_returns_both_shapes():
    result = _StructuredTool().execute({"ticker": "AAPL"})
    assert result.success
    assert result.structured["total_debt"] == 1_234_567_890.0   # for analysis
    assert result.data == "Total Debt: $1.23B"                  # for the LLM


def test_legacy_tool_still_works_with_empty_payload():
    """The refactor had to be additive. Old-contract tools keep running."""
    result = _LegacyTool().execute({})
    assert result.success
    assert result.data == "plain string output"
    assert result.structured == {}


def test_tool_exceptions_never_escape():
    result = _ExplodingTool().execute({})
    assert result.success is False
    assert "upstream API is down" in result.error


def test_tool_node_records_the_structured_payload(registry):
    state = create_initial_state("q")
    state["current_action"] = "get_financial_metrics"
    state["current_action_input"] = {"ticker": "AAPL"}
    state["iteration_count"] = 1

    update = nodes.tool_node(state)
    assert update["tool_calls"][0].tool_output_structured["total_debt"] == 1_234_567_890.0


def test_tool_failures_do_not_crash_the_run(registry):
    state = create_initial_state("q")
    state["current_action"] = "exploding"
    state["current_action_input"] = {}
    state["iteration_count"] = 1

    update = nodes.tool_node(state)
    assert update["tool_calls"][0].success is False
    assert update["status"] == AgentStatus.OBSERVING


# ===================================================================
# D4/D5 — evidence confidence is written, and merges
# ===================================================================

def test_tool_node_writes_evidence_confidence(registry):
    state = create_initial_state("q")
    state["current_action"] = "get_financial_metrics"
    state["current_action_input"] = {"ticker": "AAPL"}
    state["iteration_count"] = 1

    scores = nodes.tool_node(state)["evidence_confidence"]
    assert scores, "evidence_confidence is still never written (D4)"
    assert 0.0 < next(iter(scores.values())) <= 1.0


def test_failed_tools_score_zero_rather_than_being_omitted(registry):
    scores = nodes._score_evidence("get_news", 2, success=False)
    assert scores == {"get_news:2": 0.0}


def test_evidence_confidence_merges_across_iterations():
    """
    D5: with no reducer LangGraph replaces the dict, so iteration 2 would
    erase iteration 1 and confidence would only ever see the last tool.
    """
    first = nodes._score_evidence("get_stock_price", 1, success=True)
    second = nodes._score_evidence("get_news", 2, success=True)
    merged = operator.or_(first, second)
    assert set(merged) == {"get_stock_price:1", "get_news:2"}


# ===================================================================
# D7 — a parse error retries instead of ending the run
# ===================================================================

def test_parse_error_routes_back_to_reasoning():
    state = create_initial_state("q")
    state["current_action"] = ""
    state["iteration_count"] = 1
    state["parse_retry_count"] = 1
    assert nodes.should_continue(state) == "reasoning_node"


def test_parse_retries_are_bounded():
    state = create_initial_state("q")
    state["current_action"] = ""
    state["iteration_count"] = 1
    state["parse_retry_count"] = nodes.MAX_PARSE_RETRIES
    assert nodes.should_continue(state) == "output_node"


def test_retry_loop_cannot_outlive_the_iteration_limit():
    """The second bound: even mid-retry, max_iterations still ends the run."""
    state = create_initial_state("q", max_iterations=3)
    state["current_action"] = ""
    state["iteration_count"] = 3
    state["parse_retry_count"] = 0
    assert nodes.should_continue(state) == "output_node"


def test_normal_routing_is_unchanged():
    state = create_initial_state("q")
    state["iteration_count"] = 1

    state["current_action"] = "get_stock_price"
    assert nodes.should_continue(state) == "tool_node"

    state["current_action"] = "final_answer"
    assert nodes.should_continue(state) == "output_node"


def test_graph_declares_the_retry_edge():
    """
    should_continue can return "reasoning_node", and LangGraph raises at
    runtime if a returned key is missing from the path_map.
    """
    import inspect
    from agent import graph as graph_module

    source = inspect.getsource(graph_module.build_graph)
    assert '"reasoning_node": "reasoning_node"' in source
