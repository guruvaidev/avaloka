"""Swarm as an optional capability of the conversational agent.

The conversational agent came from develop-1.7 where swarm is always present.
On develop-1.6 it must run without it, because swarm is a commercial capability
and conversational analysis is core. These tests pin that the absence is a
normal, total state — not an error path.
"""

from __future__ import annotations

import sys
import types

import pytest

from app.agents.avaloka_agent import capabilities as caps


@pytest.fixture(autouse=True)
def _clear_cache():
    caps.reset_cache()
    yield
    caps.reset_cache()


@pytest.fixture
def without_swarm(monkeypatch):
    """Simulate an open-source build: the module is not importable."""
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "app.agents.avaloka_agent" and args and args[2] and "swarm" in args[2]:
            raise ImportError("No module named 'app.agents.avaloka_agent.swarm'")
        if name.endswith("avaloka_agent.swarm"):
            raise ImportError("No module named swarm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    caps.reset_cache()
    yield


@pytest.fixture
def with_swarm(monkeypatch):
    """Simulate a commercial build with swarm installed."""
    module = types.ModuleType("app.agents.avaloka_agent.swarm")
    module.plan_swarm = lambda intent: [{"agent": "sampler", "intent": intent}]
    module.announce = lambda intent: f"Coordinating specialists for: {intent}"
    monkeypatch.setitem(sys.modules, "app.agents.avaloka_agent.swarm", module)
    caps.reset_cache()
    yield module


# --------------------------------------------------------------------------- #
# The open-source build: absence is normal, not an error
# --------------------------------------------------------------------------- #

def test_conversational_agent_imports_without_swarm(without_swarm):
    """The whole point: no ImportError when the module is absent."""
    assert caps.swarm_available() is False


def test_plan_is_empty_rather_than_raising(without_swarm):
    assert caps.plan_swarm("segment my customers") == []


def test_announce_is_empty_rather_than_raising(without_swarm):
    assert caps.announce("segment my customers") == ""


def test_absence_is_reported_honestly(without_swarm):
    described = caps.describe()
    assert described["swarm"]["available"] is False
    assert "not present" in described["swarm"]["reason"]


# --------------------------------------------------------------------------- #
# The commercial build: the real module is used
# --------------------------------------------------------------------------- #

def test_installed_swarm_is_used(with_swarm):
    assert caps.swarm_available() is True
    plan = caps.plan_swarm("forecast churn")
    assert plan and plan[0]["intent"] == "forecast churn"


def test_installed_announce_is_used(with_swarm):
    assert "Coordinating specialists" in caps.announce("forecast churn")


# --------------------------------------------------------------------------- #
# A broken or partial swarm must not break a reply
# --------------------------------------------------------------------------- #

def test_a_swarm_module_missing_functions_is_treated_as_absent(monkeypatch):
    partial = types.ModuleType("app.agents.avaloka_agent.swarm")
    partial.plan_swarm = lambda intent: []          # announce missing
    monkeypatch.setitem(sys.modules, "app.agents.avaloka_agent.swarm", partial)
    caps.reset_cache()
    assert caps.swarm_available() is False


def test_a_raising_swarm_degrades_instead_of_breaking_the_reply(monkeypatch):
    """Narration failing must never cost the user their answer."""
    broken = types.ModuleType("app.agents.avaloka_agent.swarm")
    def boom(intent):
        raise RuntimeError("swarm backend down")
    broken.plan_swarm = boom
    broken.announce = boom
    monkeypatch.setitem(sys.modules, "app.agents.avaloka_agent.swarm", broken)
    caps.reset_cache()

    assert caps.plan_swarm("x") == []
    assert caps.announce("x") == ""


def test_a_swarm_returning_none_is_normalised(monkeypatch):
    odd = types.ModuleType("app.agents.avaloka_agent.swarm")
    odd.plan_swarm = lambda intent: None
    odd.announce = lambda intent: None
    monkeypatch.setitem(sys.modules, "app.agents.avaloka_agent.swarm", odd)
    caps.reset_cache()

    assert caps.plan_swarm("x") == []
    assert caps.announce("x") == ""


# --------------------------------------------------------------------------- #
# Resolution is by import probe, not a flag
# --------------------------------------------------------------------------- #

def test_availability_cannot_be_claimed_by_configuration(without_swarm, monkeypatch):
    """An OSS deployment must not be able to assert a capability it lacks.

    This is the edition boundary: capability follows installed code, never an
    environment variable someone can set.
    """
    monkeypatch.setenv("AVALOKA_EDITION", "enterprise")
    monkeypatch.setenv("AVALOKA_SWARM_ENABLED", "1")
    assert caps.swarm_available() is False


def test_resolution_is_cached(with_swarm):
    first = caps.get_swarm()
    assert caps.get_swarm() is first


def test_reset_cache_reresolves(with_swarm):
    first = caps.get_swarm()
    caps.reset_cache()
    assert caps.get_swarm() is not first


def test_describe_is_json_serialisable(without_swarm):
    import json
    json.dumps(caps.describe())


# --------------------------------------------------------------------------- #
# The ported agent uses a live model
# --------------------------------------------------------------------------- #

def test_conversational_model_is_registered_and_live():
    """The port hardcoded llama-3.3-70b-versatile, which Groq has removed."""
    from app.core.model_config import DEFAULT_MODELS
    assert "conversational" in DEFAULT_MODELS
    assert not DEFAULT_MODELS["conversational"].startswith("llama-")


def test_conversational_model_passes_the_backend_guard():
    from app.core.model_config import DEFAULT_MODELS, check_model_supported
    check_model_supported("conversational", DEFAULT_MODELS["conversational"])


def test_conversational_models_honor_local_provider_without_groq_key(monkeypatch):
    from app.agents.avaloka_agent import agent, intent_classifier

    monkeypatch.setenv("INFERENCE_PROVIDER", "local")
    monkeypatch.delenv("GROQ_API_KEY_PLANNING_AGENT", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    agent._reply_llm = None
    intent_classifier._llm = None
    reply_llm = object()
    intent_llm = object()
    monkeypatch.setattr(agent, "build_chat_model", lambda **_: reply_llm)
    monkeypatch.setattr(intent_classifier, "build_chat_model", lambda **_: intent_llm)
    try:
        assert agent._get_reply_llm() is reply_llm
        assert intent_classifier._get_llm() is intent_llm
    finally:
        agent._reply_llm = None
        intent_classifier._llm = None
