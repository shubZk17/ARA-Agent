"""
Per-run LLM key override (BYO-key web UI). No real LLM, no real keys.

The bug this guards: nodes._get_llm() used to memoize the first client for the
life of the process, so on a shared server every visitor after the first got
the first visitor's key. The override path must build fresh each call and never
touch the module cache.
"""

from __future__ import annotations

import pytest

from agent import nodes


@pytest.fixture(autouse=True)
def _capture_build(monkeypatch):
    calls = []

    def fake_build(provider, model, api_key):
        calls.append((provider, model, api_key))
        return f"client::{provider}::{model}::{api_key}"

    monkeypatch.setattr(nodes, "_build_llm", fake_build)
    monkeypatch.setattr(nodes, "_llm", None)
    monkeypatch.setattr(nodes, "_llm_override", None)
    return calls


def test_override_routes_to_supplied_provider_and_key(_capture_build):
    nodes.set_llm_override({"provider": "openai", "api_key": "sk-user", "model": "gpt-4o"})
    assert nodes._get_llm() == "client::openai::gpt-4o::sk-user"
    assert _capture_build[-1] == ("openai", "gpt-4o", "sk-user")


def test_override_is_not_cached_across_runs(_capture_build):
    nodes.set_llm_override({"provider": "groq", "api_key": "key-A"})
    nodes._get_llm()
    nodes.set_llm_override({"provider": "groq", "api_key": "key-B"})
    nodes._get_llm()
    # Second visitor gets their own key, not the first's.
    assert _capture_build[-1][2] == "key-B"
    assert len(_capture_build) == 2  # built fresh both times, nothing memoized


def test_partial_override_falls_back_to_settings(_capture_build):
    nodes.set_llm_override({"provider": "groq", "api_key": "only-key"})  # no model
    nodes._get_llm()
    prov, mod, k = _capture_build[-1]
    assert (prov, k) == ("groq", "only-key")     # supplied fields used
    assert mod == nodes.settings.groq_model      # absent field from settings


def test_no_override_uses_cached_default(_capture_build):
    nodes.set_llm_override(None)
    a = nodes._get_llm()
    b = nodes._get_llm()
    assert a is b or a == b
    assert len(_capture_build) == 1  # default client built once and reused
