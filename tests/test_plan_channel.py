"""Regression tests: planner state keys must be declared ETLState channels.

The planner's `plan` (and its data-transfer keys) were dropped by LangGraph at
the plan_etl node boundary because they were not declared channels — code_etl
always saw plan=None and fell back to a generic 3-step plan, the planner's
early-exit paths never fired, and chat-registered databases vanished after
the turn.
"""
import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

import app.api.workflow as wf
from app.agents.planner import plan_etl_job
from app.graph.etl_state import ETLState

PLAN = (
    "1. Load sales.csv into a pandas DataFrame.\n"
    "2. Group by region and compute the average of the sales column.\n"
    "3. Save the result to output.csv."
)


# -------------------------
# Channel survival: node RETURNS of plan / dta_database_registry persist
# -------------------------

def test_plan_returned_by_a_node_survives_the_boundary():
    def planner(state):
        return {"plan": PLAN, "ready_to_code": True}

    seen = {}

    def coder(state):
        seen["plan"] = state.get("plan")
        return {}

    g = StateGraph(ETLState)
    g.add_node("planner", planner)
    g.add_node("coder", coder)
    g.set_entry_point("planner")
    g.add_edge("planner", "coder")
    g.add_edge("coder", END)
    final = g.compile().invoke({"user_id": "u", "session_id": "s"})

    assert seen["plan"] == PLAN
    assert final.get("plan") == PLAN


def test_dta_state_keys_survive_the_boundary():
    registry = {"prod_db": {"db_type": "mysql", "host": "h", "port": 3306}}
    transfer = {"source": "prod_db", "destination": "warehouse", "success": True}

    def register(state):
        return {
            "dta_database_registry": registry,
            "dta_injection_script": "import daft",
            "dta_last_transfer": transfer,
            "is_dta_request": True,
        }

    g = StateGraph(ETLState)
    g.add_node("register", register)
    g.set_entry_point("register")
    g.add_edge("register", END)
    final = g.compile().invoke({"user_id": "u", "session_id": "s"})
    assert final.get("dta_database_registry") == registry
    assert final.get("dta_injection_script") == "import daft"
    assert final.get("dta_last_transfer") == transfer
    assert final.get("is_dta_request") is True


def test_registered_database_survives_across_turns():
    # Mirrors the server setup: one checkpointer, one thread_id, two invokes.
    from langgraph.checkpoint.memory import MemorySaver

    def planner(state):
        registry = dict(state.get("dta_database_registry") or {})
        if not registry:
            registry["prod_db"] = {"db_type": "mysql", "host": "h"}
        return {"dta_database_registry": registry}

    g = StateGraph(ETLState)
    g.add_node("planner", planner)
    g.set_entry_point("planner")
    g.add_edge("planner", END)
    graph = g.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "ui:t1"}}

    graph.invoke({"user_id": "u", "session_id": "s"}, config=config)
    turn2 = graph.invoke({"user_id": "u", "session_id": "s"}, config=config)
    assert "prod_db" in (turn2.get("dta_database_registry") or {})


# -------------------------
# Planner early exits fire once plan persists
# -------------------------

def test_planner_early_exit_keeps_existing_plan_when_ready_to_code():
    state = {
        "messages": [HumanMessage(content="average sales per region")],
        "plan": PLAN,
        "ready_to_code": True,
    }
    result = plan_etl_job(state)
    assert result.get("plan") == PLAN
    assert result.get("ready_to_code") is True


def test_planner_early_exit_keeps_existing_plan_when_ready_to_summarize():
    state = {
        "messages": [HumanMessage(content="average sales per region")],
        "plan": PLAN,
        "ready_to_summarize": True,
    }
    result = plan_etl_job(state)
    assert result.get("plan") == PLAN
    assert result.get("ready_to_summarize") is True


# -------------------------
# code_etl uses the planner's plan, not the generic fallback
# -------------------------

@pytest.fixture
def capture_subgraph(monkeypatch):
    captured = {}

    class FakeGraph:
        def invoke(self, initial):
            captured.update(initial)
            return {"generated_code": "x = 1", "llm_raw_response": None}

    monkeypatch.setattr(wf, "build_coding_graph", lambda *a, **k: FakeGraph())
    monkeypatch.setattr(wf, "coding_graph", None, raising=False)
    monkeypatch.setattr(wf, "coder_node_is_mock", False, raising=False)
    monkeypatch.setattr(wf, "_detect_ambiguous_prompt_details", lambda prompt: None)
    return captured


def test_coding_subgraph_receives_planner_plan(capture_subgraph):
    wf.coding_subgraph_node({
        "user_prompt": "average sales per region",
        "messages": [],
        "schema": {},
        "plan": PLAN,
    })
    assert capture_subgraph["plan"] == PLAN


def test_coding_subgraph_falls_back_only_when_plan_missing(capture_subgraph):
    wf.coding_subgraph_node({
        "user_prompt": "average sales per region",
        "messages": [],
        "schema": {},
    })
    assert "Fulfill this request" in capture_subgraph["plan"]
