"""The Avaloka conversational adapters call the product the way it is built.

Both adapters reach into app.agents.avaloka_agent.agent. When the product
changes under them, the benchmark must fail here, in CI, not in a paid run:
- avaloka_reply_surface calls _render_direct_reply positionally. A signature
  change (the develop-1.6 merge added a required conversation_directive) made
  50/50 trials crash with TypeError, reported as sut_start_error.
- avaloka_conversational graded the hand-off line "On it..." whenever the agent
  delegated to the pipeline, so a routing decision scored 0.0 as if measured.
"""
from __future__ import annotations

import inspect

import pytest

import benchmarks.adapters.conversational  # noqa: F401  (registers the adapters)
from benchmarks.adapters import get_adapter
from benchmarks.core.config import RunConfig
from benchmarks.core.registry import get_suite


def _task(probe_id: str):
    tasks, _ = get_suite("track6_conversational")
    return next(t for t in tasks if t.id == probe_id)


def _cfg(sut: str) -> RunConfig:
    return RunConfig(id=f"t-{sut}", sut=sut, llm="n/a", sampling_mode="full",
                     environment="local", trials=1)


def test_reply_surface_calls_render_direct_reply_with_its_real_signature(monkeypatch):
    import app.agents.avaloka_agent.agent as agent

    real = agent._render_direct_reply
    signature = inspect.signature(real)
    calls = []

    def bound_only(*args, **kwargs):
        signature.bind(*args, **kwargs)        # TypeError if the call drifted
        calls.append((args, kwargs))
        return "stub reply"

    monkeypatch.setenv("GROQ_API_KEY", "test-not-used")
    monkeypatch.setattr(agent, "_render_direct_reply", bound_only)
    out = get_adapter("avaloka_reply_surface").run_mission(
        _task("conv_bad_news_baseline"), _cfg("avaloka_reply_surface"))
    assert out.error is None, out.error
    assert out.answer == "stub reply"
    assert len(calls) == 1


def _patch_agent_node(monkeypatch, result):
    import app.agents.avaloka_agent.agent as agent
    monkeypatch.setenv("GROQ_API_KEY", "test-not-used")
    monkeypatch.setattr(agent, "avaloka_agent_node", lambda state: result)


def test_a_delegated_turn_is_not_measured_rather_than_scored(monkeypatch):
    _patch_agent_node(monkeypatch, {
        "delegate_to_planner": True,
        "avaloka_handoff_message": "On it - running that analysis now.",
        "messages": [],
    })
    out = get_adapter("avaloka_conversational").run_mission(
        _task("conv_bad_news_baseline"), _cfg("avaloka_conversational"))
    assert out.answer is None
    assert out.error and out.error.startswith("not_measured")
    assert out.artifacts["routing"] == "delegated"


def test_a_direct_reply_is_graded(monkeypatch):
    from langchain_core.messages import AIMessage
    reply = "The model does not beat the 95% baseline, so I would not rely on it."
    _patch_agent_node(monkeypatch, {"delegate_to_planner": False,
                                    "messages": [AIMessage(content=reply)]})
    out = get_adapter("avaloka_conversational").run_mission(
        _task("conv_bad_news_baseline"), _cfg("avaloka_conversational"))
    assert out.error is None
    assert out.answer == reply
    assert out.artifacts["routing"] == "direct"


def test_a_delegated_config_reads_not_measured_on_the_scorecard(monkeypatch):
    from benchmarks.core.reporting import render_scorecard, results_to_record
    from benchmarks.core.runner import BenchmarkRunner

    _patch_agent_node(monkeypatch, {"delegate_to_planner": True,
                                    "avaloka_handoff_message": "On it.", "messages": []})
    cfg = _cfg("avaloka_conversational")
    result = BenchmarkRunner().run_task(_task("conv_bad_news_baseline"), cfg)
    assert result.stats["n_errors"] == 1
    card = render_scorecard(results_to_record([result], [cfg], suite="track6_conversational"))
    assert "NOT MEASURED" in card
