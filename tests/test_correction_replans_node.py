"""The agent node re-plans a correction and says why, from facts code wrote.

Uses only the node's entry point, so it runs on the base and fails there by
assertion: the base delegated the correction with nothing said and kept no
facts. The rule itself is pinned in tests/test_correction_replans.py.
"""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.avaloka_agent import agent as agent_mod

PRIOR = {"source": "verified analysis results for the connected dataset",
         "results": {"mean_order_value": 37.31}}
TURNS = [HumanMessage(content="What's the mean order value?"),
         HumanMessage(content="You excluded refunds. Include them and recompute.")]


def _node(monkeypatch, model) -> Dict[str, Any]:
    monkeypatch.setattr(agent_mod, "_get_reply_llm", lambda: model)

    async def _empty(_thread_id):
        return {}
    monkeypatch.setattr(agent_mod, "load_thread_context", _empty)
    # A dataset is bound: verified results come from one, and with none bound
    # the turn is stopped before the planner (tests/test_no_dataset_new_work.py).
    return agent_mod.avaloka_agent_node({"messages": list(TURNS), "avaloka_recommendation": PRIOR,
                                         "data_source_location": "/data/orders.csv"})


def test_the_node_re_plans_and_says_why(monkeypatch):
    """Fails on the base: the turn was delegated with nothing said, and no facts kept."""
    out = _node(monkeypatch, None)
    assert out["delegate_to_planner"] is True
    assert out.get("correction", {}).get("original") == "excluding refunds"
    said = out["messages"][-1].content if isinstance(out.get("messages", [None])[-1], AIMessage) else ""
    assert "excluding refunds" in said and "including refunds" in said
    assert "because" in said and "alternative" in said


