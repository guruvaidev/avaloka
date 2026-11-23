"""New work asked with no dataset bound says what it needs; no code runs.

Recorded on the conversational product benchmark: "Compute distinct aisles on
the entire dataset" got "I'm unable to compute distinct aisles because no
dataset has been provided", and "chart median_house_value by ocean_proximity on
the cleaned dataset" got "I couldn't complete this step. Reason: the analysis
code stopped with an error" -- the planner wrote code against no input and ran
it. The stop is upstream, in the agent node: with no data source bound and none
named in the request, data work is not delegated; the reply says which data is
needed and how to provide it. And a delegated turn's hand-off line ("On it —
writing the analysis now… Expect seconds to minutes.") now reaches the chat.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.avaloka_agent import agent as agent_mod
from app.agents.avaloka_agent.execution_profile import SMALL_THRESHOLD


@pytest.fixture(autouse=True)
def _no_redis_no_model(monkeypatch):
    async def _empty(_thread_id):
        return {}
    monkeypatch.setattr(agent_mod, "load_thread_context", _empty)
    monkeypatch.setattr(agent_mod, "_get_reply_llm", lambda: None)


def _run(*texts: str, **state: Any) -> Dict[str, Any]:
    return agent_mod.avaloka_agent_node(
        {"messages": [HumanMessage(content=t) for t in texts], **state})


def _said(out: Dict[str, Any]) -> str:
    return "\n".join(m.content for m in out.get("messages", []) if isinstance(m, AIMessage))


def test_new_work_with_no_dataset_says_what_it_needs():
    """Fails on the base: the turn went to the planner, which ran code on nothing."""
    out = _run("Compute distinct aisles on the entire dataset.")
    assert out["delegate_to_planner"] is False
    said = _said(out)
    assert "no dataset connected" in said
    assert "uploading a file" in said and "connecting a database" in said
    assert "error" not in said.lower()


def test_it_names_the_columns_the_user_asked_about():
    out = _run("Impute the missing total_bedrooms with the median.",
               "Now chart median_house_value by ocean_proximity on the cleaned dataset.")
    assert out["delegate_to_planner"] is False
    assert ("a table with the columns total_bedrooms, median_house_value and "
            "ocean_proximity") in _said(out)


def test_the_hand_off_reaches_the_chat_with_data_bound():
    """Fails on the base: the hand-off line stayed in a state field."""
    out = _run("train a model on this data", dataset_size_bytes=SMALL_THRESHOLD - 1)
    assert out["delegate_to_planner"] is True
    said = _said(out)
    assert said == out["avaloka_handoff_message"]
    assert "Expect seconds to minutes." in said
    assert isinstance(out["messages"][-2], HumanMessage), "the user's request stays the last human turn"


@pytest.mark.parametrize("state", [
    {"data_source_location": "/data/orders.csv"},
    {"connection_id": "conn-1"},
    {"active_dataset_id": "ds-1"},
    {"data_source_location_cloud": "s3://bucket/orders.parquet"},
])
def test_any_bound_source_is_delegated_as_before(state):
    assert _run("Compute distinct aisles on the entire dataset.", **state)["delegate_to_planner"] is True


@pytest.mark.parametrize("text", [
    "Load s3://bucket/orders.csv and compute the mean order value",
    "compute the churn rate from the customers table",
    "download the titanic dataset from kaggle and train a model",
    "SELECT region, SUM(sales) FROM orders GROUP BY region",
])
def test_a_request_that_names_its_source_is_the_planners(text):
    assert _run(text)["delegate_to_planner"] is True


@pytest.mark.parametrize("text", [
    "schedule the export every day at 2am",
    "provision a ray cluster on gke",
    "list tasks",
])
def test_work_with_its_own_target_is_not_stopped(text):
    out = _run(text)
    assert "no dataset connected" not in _said(out)


def test_a_correction_with_no_data_still_re_plans_in_words():
    out = _run("What's the mean order value?", "You excluded refunds. Include them and recompute.")
    assert out["delegate_to_planner"] is False
    said = _said(out)
    assert "excluding refunds" in said and "including refunds" in said and "alternative" in said
    assert "no dataset connected" in said
    assert out["correction"]["corrected"] == "including refunds"


def test_a_prior_artifact_the_request_refers_to_is_said_not_to_exist():
    """Fails without the fix: "the cleaned dataset" was left hanging, and the
    prior-reference probe scored the reply as not resolving the reference."""
    from tests.conversational.probes import PROBES, run_probe
    probe = next(p for p in PROBES if p.id == "int_prior_reference")
    out = _run(*(t.text for t in probe.turns))
    said = _said(out)
    assert "The cleaned dataset you refer to does not exist in this conversation" in said
    assert run_probe(probe, lambda _p: said).passed, said


def test_the_reply_gives_the_products_own_sense_of_scale():
    """Fails without the fix: the reply said it would say how long, and never how long."""
    from app.agents.avaloka_agent.execution_profile import LOCAL_ETA_TEXT
    from tests.conversational.probes import PROBES, eval_narrates_before_long_work
    probe = next(p for p in PROBES if p.id == "resp_long_job_narrated")
    said = _said(_run(probe.turns[-1].text))
    assert LOCAL_ETA_TEXT in said
    metrics = eval_narrates_before_long_work(said, probe.grading_context)
    assert metrics["gave_a_sense_of_scale"] == 1.0 and metrics["pass"] == 1.0, metrics

