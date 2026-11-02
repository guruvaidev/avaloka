"""1.6 release suite: planner routing (plan case AG-01).

The planner runs its deterministic path (no LLM), so the routes asserted here are
the ones an install without a model key takes, and the ones the LLM path must not
contradict. See docs/test-reports/1.6-traceability.csv.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

import app.agents.planner as planner_mod
import app.api.workflow as wf

_SCHEMA = {"region": "object", "revenue": "float64", "churn": "int64"}


def _plan(prompt: str) -> dict:
    state = {
        "messages": [HumanMessage(content=prompt)],
        "user_id": "user-1",
        "session_id": "session-1",
        "schema": dict(_SCHEMA),
        "data_source_location": "/tmp/release16-routing.csv",
        "input_data_type": "csv",
    }
    out = planner_mod.plan_etl_job(state)
    return {**state, **out}


@pytest.fixture(autouse=True)
def _no_planner_llm(monkeypatch):
    # Hermetic: the deterministic planner path, whatever keys the host has.
    monkeypatch.setattr(planner_mod, "llm", None, raising=False)


def test_ag01_chart_training_and_transfer_prompts_reach_their_specialists():
    """AG-01: a chart prompt reaches the coder and then the visualization agent,
    a training prompt reaches the model-training agent, and a transfer prompt is
    handled by the data-transfer agent."""
    viz = _plan("plot revenue by region as a bar chart")
    assert wf.route_planner_output(viz) == "prepare_code"
    # After a successful execution the chart turn is handed to the visualizer.
    executed = {**viz, "execution_result": {"status": "success"}}
    assert wf.route_after_execution(executed) == "visualize"

    train = _plan("train a model to predict churn")
    assert train.get("enable_training") is True
    assert wf.route_planner_output(train) == "train_models"

    transfer = _plan("transfer the orders table from srcdb to dstdb")
    assert transfer.get("is_dta_request") is True
    assert not transfer.get("enable_training")
    # The DTA runs inside the planner turn; nothing else may run after it.
    assert wf.route_planner_output(transfer) == "end"
