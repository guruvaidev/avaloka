"""An ambiguous question is asked about, not answered twice.

The conversational product benchmark recorded the reply model answering "What
was revenue last quarter?" with both readings' figures ("Gross revenue:
$1,240,000 - Net revenue: $1,118,000. Which of these do you need?"). The prompt
said to ask which reading was meant; it never said to withhold the candidate
figures, so the model showed both and treated that as not picking. A user handed
two numbers quotes one.

The reply is a model's, so what is pinned here is the instruction that reaches
it on this path, captured from a stubbed model; the behaviour itself is checked
by the benchmark's next recording.
"""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.avaloka_agent import agent as agent_mod

EVIDENCE = {"source": "verified analysis results for the connected dataset",
            "results": {"options": ["gross", "net"], "values": [1240000.0, 1118000.0]}}


class _Capture:
    def __init__(self) -> None:
        self.calls: List[List[Any]] = []

    def invoke(self, messages, **kwargs):
        self.calls.append(list(messages))
        return AIMessage(content="Do you mean gross revenue or net revenue?")


def _system_prompt_for(monkeypatch, text: str) -> str:
    capture = _Capture()
    monkeypatch.setattr(agent_mod, "_get_reply_llm", lambda: capture)

    async def _empty(_thread_id):
        return {}
    monkeypatch.setattr(agent_mod, "load_thread_context", _empty)
    state: Dict[str, Any] = {"messages": [HumanMessage(content=text)],
                             "avaloka_recommendation": EVIDENCE}
    out = agent_mod.avaloka_agent_node(state)
    assert out["delegate_to_planner"] is False, "the question must reach the reply step"
    (call,) = capture.calls
    return next(m.content for m in call if isinstance(m, SystemMessage))


def test_the_reply_model_is_told_to_withhold_both_figures(monkeypatch):
    """Fails on the base: the instruction to ask was there, the one to withhold was not."""
    prompt = _system_prompt_for(monkeypatch, "What was revenue last quarter?")
    assert "ASK WHICH ONE" in prompt
    assert "state\nNEITHER reading's figure" in prompt or "state NEITHER reading's figure" in prompt
    assert "not even both side by side" in prompt


def test_the_reply_still_reaches_the_user(monkeypatch):
    capture = _Capture()
    monkeypatch.setattr(agent_mod, "_get_reply_llm", lambda: capture)

    async def _empty(_thread_id):
        return {}
    monkeypatch.setattr(agent_mod, "load_thread_context", _empty)
    out = agent_mod.avaloka_agent_node({"messages": [HumanMessage(content="What was revenue last quarter?")],
                                        "avaloka_recommendation": EVIDENCE})
    assert out["messages"][-1].content == "Do you mean gross revenue or net revenue?"
