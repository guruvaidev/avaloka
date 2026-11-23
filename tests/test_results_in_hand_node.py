"""The agent node answers a question about verified results in hand.

Uses only the node's public entry point, so it runs on the base unchanged and
fails there by assertion: the base delegated every one of these to the planner,
which never reads ``avaloka_recommendation``. The routing rule itself is pinned
in tests/test_intent_results_in_hand.py.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.avaloka_agent.agent import avaloka_agent_node

RESULTS = {"source": "verified analysis results for the connected dataset",
           "results": {"series": [0.18, 0.23], "direction": "up"}}


@pytest.fixture
def _no_redis(monkeypatch):
    async def _empty(_thread_id):
        return {}
    monkeypatch.setattr("app.agents.avaloka_agent.agent.load_thread_context", _empty)


def test_the_node_answers_a_question_about_results_in_hand(_no_redis) -> None:
    """Fails on the base: the node delegated this to the planner."""
    state: Dict[str, Any] = {
        "messages": [HumanMessage(content="Churn fell this quarter — can you confirm?")],
        "avaloka_recommendation": RESULTS,
    }
    out = avaloka_agent_node(state)
    assert out["delegate_to_planner"] is False
    assert out["avaloka_intent"] == "discussion"
    assert isinstance(out["messages"][-1], AIMessage)
    assert out["discovery_result"] is None


def test_the_node_still_delegates_new_work_with_results_in_hand(_no_redis) -> None:
    state: Dict[str, Any] = {
        "messages": [HumanMessage(content="You excluded refunds. Include them and recompute.")],
        "avaloka_recommendation": RESULTS,
        # Bound: with no data source the turn is stopped before the planner.
        "data_source_location": "/data/orders.csv",
    }
    assert avaloka_agent_node(state)["delegate_to_planner"] is True


def test_an_execution_profile_alone_is_not_results(_no_redis) -> None:
    """``avaloka_recommendation`` also carries the execution profile; that is
    not something to answer a question from."""
    state: Dict[str, Any] = {
        "messages": [HumanMessage(content="Churn fell this quarter — can you confirm?")],
        "avaloka_recommendation": {"fidelity": "entire_dataset", "execution_mode": "local"},
    }
    assert avaloka_agent_node(state)["delegate_to_planner"] is True
