"""An ambiguous ask about results in hand is asked in code's words.

Recorded on gpt-oss with the prompt rule in place, "What was revenue last
quarter?" still got both figures ("Gross revenue: $1,240,000 - Net revenue:
$1,118,000. Which of these do you need?"). So the question is now written by
code: when the planner's measure check finds two measures the user's word could
mean, the reply is that check's question and the model is not asked at all.
The stub model below would leak both figures if it were called.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.avaloka_agent import agent as agent_mod

RESULTS = {"source": "verified analysis results for the connected dataset",
           "results": {"gross_revenue": 1240000.0, "net_revenue": 1118000.0}}
LEAK = ("I see two revenue figures for the last quarter: - **Gross revenue:** $1,240,000 "
        "- **Net revenue:** $1,118,000 Which of these do you need?")


class _Leaky:
    def __init__(self) -> None:
        self.calls: List[Any] = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        return AIMessage(content=LEAK)


def _run(monkeypatch, text: str, results=RESULTS) -> Dict[str, Any]:
    model = _Leaky()
    monkeypatch.setattr(agent_mod, "_get_reply_llm", lambda: model)

    async def _empty(_thread_id):
        return {}
    monkeypatch.setattr(agent_mod, "load_thread_context", _empty)
    out = agent_mod.avaloka_agent_node({"messages": [HumanMessage(content=text)],
                                        "avaloka_recommendation": results})
    out["_model_calls"] = len(model.calls)
    return out


def _figures(text: str) -> List[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", text)]


def test_the_ambiguous_ask_carries_no_candidate_figure(monkeypatch):
    """Fails on d36c4e70: the model wrote the reply, and both figures came through."""
    out = _run(monkeypatch, "What was revenue last quarter?")
    reply = out["messages"][-1].content
    assert not any(abs(n - v) < 1 for n in _figures(reply) for v in (1240000.0, 1118000.0)), reply
    assert reply == "Which revenue measure do you mean: gross revenue or net revenue?"
    assert out["_model_calls"] == 0
    assert out["pending_clarification"]["type"] == "metric_choice"
    assert out["pending_clarification"]["columns"] == ["gross_revenue", "net_revenue"]


def test_a_named_reading_is_answered_by_the_model(monkeypatch):
    out = _run(monkeypatch, "What was net revenue last quarter?")
    assert out["_model_calls"] == 1


def test_one_measure_is_not_ambiguous(monkeypatch):
    out = _run(monkeypatch, "What was revenue last quarter?",
               {"results": {"revenue": 1240000.0}})
    assert out["_model_calls"] == 1


def test_a_question_about_other_results_is_not_held_up(monkeypatch):
    out = _run(monkeypatch, "How many rows are in this dataset?",
               {"results": {"rows": 20640.0, "gross_revenue": 1.0, "net_revenue": 2.0}})
    assert out["_model_calls"] == 1
