"""Tests for the top-level conversational Avaloka agent.

Covers:
- Heuristic intent classification (no LLM key required)
- Size → fidelity → execution-mode reasoning
- Infra preference resolution (explicit signal, prior context, URI, env)
- Connection-event-driven onboarding mode
- Delegation routing (`route_avaloka`)
- Openclaw client fallback when Gateway is unreachable
"""

from __future__ import annotations

import os
from typing import Any, Dict
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.avaloka_agent import avaloka_agent_node, route_avaloka
from app.agents.avaloka_agent.execution_profile import (
    LARGE_THRESHOLD,
    MEDIUM_THRESHOLD,
    SMALL_THRESHOLD,
    decide_execution_profile,
)
from app.agents.avaloka_agent.infra_preference import resolve_infra_preference
from app.agents.avaloka_agent.intent_classifier import (
    IntentResult,
    _heuristic_intent,
    classify_intent,
)
from app.agents.avaloka_agent.openclaw_client import (
    OpenclawDiscoveryClient,
    OpenclawUnavailable,
)


# ---------------------------------------------------------------------------
# Intent classification (heuristic / Groq-free)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("show me a histogram of age", "visualization"),
        ("train a model on this data", "ml_training"),
        ("create a training plan for churn", "ml_training"),
        ("predict the next 5 rows", "ml_inference"),
        ("convert the file to parquet", "data_transfer"),
        ("compute the correlation between price and demand", "statistical_analysis"),
        ("provision a gke cluster", "infrastructure"),
        ("schedule this every day at 2am", "schedule"),
        ("did the training finish?", "status"),
        ("show me the schema", "exploration"),
        ("hello", "chit_chat"),
        # develop-1.6 (eceebaea) moved sampling switches out of "exploration":
        # they are instructions to the planner, and exploration does not
        # delegate, so the mode silently never changed. Pinned on 1.6 by
        # tests/test_intent_routing.py::test_sampling_switches_reach_the_planner.
        ("use quick sample", "sampling_mode"),
        ("use portfolio samples", "sampling_mode"),
    ],
)
def test_heuristic_intent_routes_keywords(text: str, expected: str):
    result = _heuristic_intent(text, has_connection_event=False)
    assert result.intent == expected, (
        f"expected {expected!r} for {text!r}, got {result.intent!r}"
    )


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT: develop-1.6 made the keyword classifier longest-match-wins "
        "(intent_classifier._heuristic_intent). 'distribution' (statistical_analysis) "
        "is longer than 'plot ' (visualization), so an explicit request to plot a "
        "distribution is classified as statistical analysis. Found when the 1.6 "
        "merge ran this 1.7 expectation. Remove this xfail when fixed."
    ),
)
def test_plot_the_distribution_is_a_visualization_request():
    result = _heuristic_intent("plot the distribution of price", has_connection_event=False)
    assert result.intent == "visualization"


def test_heuristic_intent_picks_onboarding_after_connection():
    result = _heuristic_intent("hi", has_connection_event=True)
    assert result.intent == "onboarding"


def test_heuristic_intent_falls_back_to_clarification():
    result = _heuristic_intent("frobnicate the widget gizmo", has_connection_event=False)
    assert result.intent == "clarification"


def test_classify_intent_uses_heuristic_when_no_groq_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY_PLANNING_AGENT", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Force re-init so the cached _llm doesn't carry a key from another test
    import app.agents.avaloka_agent.intent_classifier as ic
    ic._llm = None

    msgs = [HumanMessage(content="train a model on this data")]
    result = classify_intent(msgs, has_connection_event=False)
    assert result.intent == "ml_training"


# ---------------------------------------------------------------------------
# Size → fidelity → execution mode
# ---------------------------------------------------------------------------


def test_small_local_dataset_is_read_whole_locally():
    # develop-1.6 (f83b88eb): a file at or under SMALL_THRESHOLD is read whole.
    # Sampling it saved nothing and told the user a 12-row CSV had been
    # "computed on a 12-row sample".
    state = {"dataset_size_bytes": SMALL_THRESHOLD - 1}
    out = decide_execution_profile(state, intent="statistical_analysis")
    assert out["analysis_fidelity"] == "entire_dataset"
    assert out["execution_mode"] == "local"
    assert "task_schedule" not in out


def test_medium_cloud_dataset_picks_portfolio_on_ray():
    state = {
        "dataset_size_bytes": MEDIUM_THRESHOLD + 1,
        "data_source_location_cloud": "gs://bucket/file.parquet",
        "connection_id": "conn-1",
    }
    out = decide_execution_profile(state, intent="statistical_analysis")
    assert out["analysis_fidelity"] == "portfolio_samples"
    assert out["execution_mode"] == "k8s-ray"


def test_huge_cloud_dataset_schedules_entire_dataset_run():
    state = {
        "dataset_size_bytes": LARGE_THRESHOLD * 5,
        "data_source_location_cloud": "s3://bucket/big.parquet",
        "connection_id": "conn-2",
    }
    out = decide_execution_profile(state, intent="statistical_analysis")
    assert out["analysis_fidelity"] == "entire_dataset"
    assert out["execution_mode"] == "k8s-ray"
    assert out["task_schedule"]["task_type"] == "execute"


def test_user_fidelity_override_is_respected():
    state = {
        "dataset_size_bytes": LARGE_THRESHOLD * 5,
        "data_source_location_cloud": "s3://bucket/big.parquet",
        "connection_id": "conn-3",
        "analysis_fidelity": "quick_sample",
    }
    out = decide_execution_profile(state, intent="statistical_analysis")
    assert out["analysis_fidelity"] == "quick_sample"
    assert out["execution_mode"] == "local"


def test_ml_training_on_large_cloud_forces_k8s_ray():
    state = {
        "dataset_size_bytes": MEDIUM_THRESHOLD + 1,
        "data_source_location_cloud": "gs://bucket/file.parquet",
        "connection_id": "conn-4",
    }
    out = decide_execution_profile(state, intent="ml_training")
    assert out["execution_mode"] == "k8s-ray"


# ---------------------------------------------------------------------------
# Infra preference
# ---------------------------------------------------------------------------


def test_infra_explicit_aws_signal():
    state = {"messages": [HumanMessage(content="use aws for this run")]}
    out = resolve_infra_preference(state)
    assert out.get("default_infra_platform") == "aws"


def test_infra_explicit_local_signal():
    state = {"messages": [HumanMessage(content="run this locally")]}
    out = resolve_infra_preference(state)
    assert out.get("execution_mode") == "local"
    assert out.get("default_infra_platform") == "local"


def test_infra_falls_back_to_uri_inference():
    state = {
        "messages": [HumanMessage(content="analyze this")],
        "data_source_location_cloud": "gs://bucket/file.parquet",
    }
    out = resolve_infra_preference(state)
    assert out.get("default_infra_platform") == "gcp"


def test_infra_uses_redis_recall(monkeypatch):
    monkeypatch.delenv("DEFAULT_INFRA_PLATFORM", raising=False)
    state = {"messages": [HumanMessage(content="continue please")]}
    out = resolve_infra_preference(state, redis_context={"last_infra_platform": "aws"})
    assert out.get("default_infra_platform") == "aws"


# ---------------------------------------------------------------------------
# avaloka_agent_node end-to-end (no LLM)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_groq(monkeypatch):
    """Run agent tests without Groq so heuristics drive intent."""
    monkeypatch.delenv("GROQ_API_KEY_PLANNING_AGENT", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import app.agents.avaloka_agent.intent_classifier as ic
    import app.agents.avaloka_agent.agent as ag
    ic._llm = None
    ag._reply_llm = None
    yield


@pytest.fixture(autouse=True)
def _empty_redis_context(monkeypatch):
    async def _empty(_thread_id):
        return {}
    monkeypatch.setattr(
        "app.agents.avaloka_agent.agent.load_thread_context",
        _empty,
    )


def test_node_delegates_for_data_touching_intent():
    state: Dict[str, Any] = {
        "messages": [HumanMessage(content="train a model on this data")],
        "dataset_size_bytes": SMALL_THRESHOLD - 1,
    }
    out = avaloka_agent_node(state)
    assert out["delegate_to_planner"] is True
    assert out["avaloka_intent"] == "ml_training"
    assert out["enable_training"] is True
    assert out["avaloka_handoff_message"]
    assert out["analysis_fidelity"] == "entire_dataset"   # small file: read whole (f83b88eb)
    assert out["execution_mode"] == "local"


def test_node_replies_directly_for_chitchat():
    state: Dict[str, Any] = {
        "messages": [HumanMessage(content="hello")],
    }
    out = avaloka_agent_node(state)
    assert out["delegate_to_planner"] is False
    assert any(isinstance(m, AIMessage) for m in out["messages"])


def test_node_onboarding_when_connection_event_present():
    state: Dict[str, Any] = {
        "messages": [HumanMessage(content="hi")],
        "connection_event": {
            "kind": "upload",
            "dataset_id": "ds-123",
            "filename": "titanic.csv",
            "schema_columns": ["age", "fare", "survived"],
        },
    }
    out = avaloka_agent_node(state)
    assert out["avaloka_mode"] == "onboarding"
    assert out["delegate_to_planner"] is False
    # Connection event is consumed (cleared) so we don't re-onboard next turn.
    assert out.get("connection_event") is None

def test_node_adapts_novice_guidance_without_lowering_rigor():
    state = {"messages": [HumanMessage(content="I am new to data science. Train a model for churn.")]}
    out = avaloka_agent_node(state)
    assert out["avaloka_expertise"]["level"] == "novice"
    assert out["avaloka_conversation_policy"]["max_options"] == 2
    assert "plain language" in out["conversation_guidance"]
    assert "lower statistical rigor" in out["conversation_guidance"]


def test_node_allows_explicit_expert_override_on_later_turn():
    state = {
        "messages": [HumanMessage(content="Expert mode; skip the basics and include confidence interval diagnostics.")],
        "avaloka_expertise": {"level": "novice", "confidence": 1.0, "explicit": True},
    }
    out = avaloka_agent_node(state)
    assert out["avaloka_expertise"]["level"] == "expert"
    assert out["avaloka_conversation_policy"]["expose_method_details"] is True
    assert "estimand" in out["conversation_guidance"]



def test_route_avaloka_uses_delegate_flag():
    assert route_avaloka({"delegate_to_planner": True}) == "delegate_to_planner"
    assert route_avaloka({"delegate_to_planner": False}) == "end"
    assert route_avaloka({}) == "end"


# ---------------------------------------------------------------------------
# Openclaw client
# ---------------------------------------------------------------------------


def test_openclaw_disabled_by_default(monkeypatch):
    monkeypatch.delenv("OPENCLAW_ENABLED", raising=False)
    assert OpenclawDiscoveryClient.is_enabled() is False


def test_openclaw_unavailable_when_gateway_absent(monkeypatch):
    monkeypatch.setenv("OPENCLAW_ENABLED", "1")
    # Point at a port that nothing is listening on
    monkeypatch.setenv("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("OPENCLAW_TIMEOUT_S", "0.2")
    client = OpenclawDiscoveryClient()
    assert client.is_available() is False
    with pytest.raises(OpenclawUnavailable):
        client.discover_database("conn-x")


def test_node_falls_back_when_openclaw_enabled_but_unreachable(monkeypatch):
    monkeypatch.setenv("OPENCLAW_ENABLED", "1")
    monkeypatch.setenv("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("OPENCLAW_TIMEOUT_S", "0.2")
    state: Dict[str, Any] = {
        "messages": [HumanMessage(content="show me the schema")],
        "connection_id": None,
    }
    out = avaloka_agent_node(state)
    # discovery_result is populated via native fallback, agent still replies.
    assert out["delegate_to_planner"] is False
    assert any(isinstance(m, AIMessage) for m in out["messages"])
