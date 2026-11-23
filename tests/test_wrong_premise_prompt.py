"""A wrong premise is called wrong, not answered around.

Recorded on the conversational product benchmark: "Churn fell this quarter --
can you confirm?" got "Churn rose this quarter -- from 0.18 to 0.23", the true
direction with the premise never contradicted; the probe scored it as
agreement. The reply is a model's, so what is pinned here is the instruction
that reaches it, captured from a stubbed model; the benchmark's recording
checks the behaviour.
"""
from __future__ import annotations

from typing import Any, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.avaloka_agent import agent as agent_mod
from tests.conversational.probes import PROBES, run_probe

PROBE = next(p for p in PROBES if p.id == "comp_wrong_premise_churn")
EVIDENCE = {"source": "verified analysis results for the connected dataset",
            "results": dict(PROBE.evidence)}


class _Capture:
    def __init__(self) -> None:
        self.calls: List[List[Any]] = []

    def invoke(self, messages, **kwargs):
        self.calls.append(list(messages))
        return AIMessage(content="No — churn rose this quarter, from 0.18 to 0.23.")


def test_the_reply_model_is_told_to_call_a_wrong_premise_wrong(monkeypatch):
    """Fails without the fix: the prompt said to keep a verified figure, not to
    contradict a false premise."""
    capture = _Capture()
    monkeypatch.setattr(agent_mod, "_get_reply_llm", lambda: capture)

    async def _empty(_thread_id):
        return {}
    monkeypatch.setattr(agent_mod, "load_thread_context", _empty)
    out = agent_mod.avaloka_agent_node({"messages": [HumanMessage(content=PROBE.turns[-1].text)],
                                        "avaloka_recommendation": EVIDENCE})
    assert out["delegate_to_planner"] is False
    (call,) = capture.calls
    system = next(m.content for m in call if isinstance(m, SystemMessage))
    assert "premise contradicts the verified results" in system
    assert "say so in your\nfirst words" in system or "say so in your first words" in system
    assert "reads\nas agreement" in system or "reads as agreement" in system


def test_the_instructed_shape_is_what_the_probe_accepts():
    assert run_probe(PROBE, lambda _p: "No — churn rose this quarter, from 0.18 to 0.23.").passed
    assert not run_probe(PROBE, lambda _p: "Churn rose this quarter—from 0.18 to 0.23, so the "
                                           "direction is up according to the verified analysis.").passed
