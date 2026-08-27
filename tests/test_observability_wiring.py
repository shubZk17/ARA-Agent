"""
Phase 4 (wired 2026-08-27) — collector/tracer are threaded into reasoning_node
and tool_node. Never calls a real LLM or a real API.
"""

from __future__ import annotations

import pytest

from agent import nodes
from agent.state import create_initial_state
from quality.observability.collector import EventType, TelemetryCollector
from quality.observability.tracer import ExecutionTracer
from tools.base import BaseTool, ToolParameter


class _StubTool(BaseTool):
    @property
    def name(self) -> str:
        return "get_stock_price"

    @property
    def description(self) -> str:
        return "fake"

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    def _execute(self, tool_input):
        return "price: $100"


class _FakeRegistry:
    def __init__(self, *tools):
        self._tools = {t.name: t for t in tools}

    def get(self, name):
        return self._tools.get(name)

    def list_tools(self):
        return list(self._tools)

    def get_tool_descriptions(self):
        return ""


class _StubResponse:
    content = '{"thought": "t", "action": "final_answer", "action_input": {"answer": "done"}}'


class _StubLLM:
    def invoke(self, messages):
        return _StubResponse()


@pytest.fixture
def registry():
    reg = _FakeRegistry(_StubTool())
    nodes.set_tool_registry(reg)
    yield reg
    nodes.set_tool_registry(None)


@pytest.fixture(autouse=True)
def _reset_observability():
    yield
    nodes.set_observability(None, None)


def test_tool_node_records_telemetry_when_observability_is_wired(registry):
    collector = TelemetryCollector()
    tracer = ExecutionTracer()
    nodes.set_observability(collector, tracer)

    state = create_initial_state("q")
    state["current_action"] = "get_stock_price"
    state["current_action_input"] = {"ticker": "AAPL"}
    state["iteration_count"] = 1

    nodes.tool_node(state)

    assert len(collector.get_events(EventType.TOOL_EXECUTION)) == 1
    assert len(tracer.get_trace().spans) == 1


def test_reasoning_node_records_telemetry_when_observability_is_wired(monkeypatch, registry):
    collector = TelemetryCollector()
    tracer = ExecutionTracer()
    nodes.set_observability(collector, tracer)
    monkeypatch.setattr(nodes, "_get_llm", lambda: _StubLLM())

    state = create_initial_state("q")
    state["iteration_count"] = 0

    nodes.reasoning_node(state)

    assert len(collector.get_events(EventType.LLM_CALL)) == 1
    assert len(tracer.get_trace().spans) == 1


def test_unset_observability_is_a_no_op(monkeypatch, registry):
    """The default (None, None) must leave node behavior byte-identical."""
    nodes.set_observability(None, None)
    monkeypatch.setattr(nodes, "_get_llm", lambda: _StubLLM())

    state = create_initial_state("q")
    state["iteration_count"] = 0
    reasoning_update = nodes.reasoning_node(state)
    assert reasoning_update["current_action"] == "final_answer"

    state2 = create_initial_state("q")
    state2["current_action"] = "get_stock_price"
    state2["current_action_input"] = {"ticker": "AAPL"}
    state2["iteration_count"] = 1
    tool_update = nodes.tool_node(state2)
    assert tool_update["tool_calls"][0].success is True
