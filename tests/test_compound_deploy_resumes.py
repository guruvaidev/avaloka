"""A compound request ("deploy this and then analyze X") is not cut in half by the confirmation.

Before the confirmation step, the provision_infra -> plan_etl loop-back planned
the analysis half of the same message. With the confirmation step the last
message is "yes", so the analysis half was dropped. The rest of the request now
rides on the pending confirmation and is planned once the provisioning outcome
has been reported, or straight after a "no".
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

import app.api.workflow as wf
from app.agents.planner import plan_etl_job

COMPOUND = "deploy this on GCP and then analyze average sales by region"
REST = "analyze average sales by region"


def _turn(messages, **extra):
    return plan_etl_job({"messages": messages, "user_id": "u", "session_id": "s", **extra})


def _ai(result):
    return [m.content for m in result.get("messages", []) if isinstance(m, AIMessage)]


def _humans(result):
    return [m.content for m in result.get("messages", []) if isinstance(m, HumanMessage)]


def test_the_rest_of_the_request_rides_on_the_confirmation():
    first = _turn([HumanMessage(content=COMPOUND)])
    assert not first.get("infrastructure_request")
    pending = first["pending_clarification"]
    assert pending["type"] == "infrastructure_confirmation"
    assert pending.get("resume") == REST, pending
    assert REST in _ai(first)[-1], "the confirmation does not say the analysis will follow"


def test_yes_provisions_and_keeps_the_rest_for_after():
    pending = {"type": "infrastructure_confirmation",
               "infrastructure_request": {"type": "gcp", "app_type": "python-docker"}, "resume": REST}
    result = _turn([HumanMessage(content=COMPOUND), AIMessage(content="confirm?"), HumanMessage(content="yes")],
                   pending_clarification=pending)
    assert wf.route_planner_output(result) == "provision_infra"
    assert result["pending_clarification"].get("resume") == REST
    assert REST in _ai(result)[-1]


def test_after_provisioning_the_outcome_is_reported_and_the_analysis_is_planned():
    result = _turn(
        [HumanMessage(content=COMPOUND), AIMessage(content="confirm?"), HumanMessage(content="yes"),
         AIMessage(content="starting")],
        pending_clarification={"type": "infrastructure_report", "resume": REST},
        infrastructure_request={"type": "gcp", "app_type": "python-docker", "confirmed": True},
        infrastructure_provisioned={"type": "gcp", "status": "failed", "details": []},
    )
    assert any("finished with status **failed**" in c for c in _ai(result)), "the outcome was not reported"
    assert _humans(result)[-1] == REST, "the analysis half was not resumed"
    assert result.get("plan"), "the analysis half was not planned"
    assert wf.route_planner_output(result) != "provision_infra"
    assert not result.get("pending_clarification") or result["pending_clarification"].get("type") not in (
        "infrastructure_report", "infrastructure_confirmation")


def test_no_cancels_the_deploy_but_still_plans_the_analysis():
    pending = {"type": "infrastructure_confirmation",
               "infrastructure_request": {"type": "gcp", "app_type": "python-docker"}, "resume": REST}
    result = _turn([HumanMessage(content=COMPOUND), AIMessage(content="confirm?"), HumanMessage(content="no")],
                   pending_clarification=pending)
    assert not result.get("infrastructure_request")
    assert wf.route_planner_output(result) != "provision_infra"
    assert any("nothing was provisioned" in c.lower() for c in _ai(result))
    assert _humans(result)[-1] == REST and result.get("plan"), "the analysis half was dropped"


def test_a_plain_deploy_still_just_reports():
    result = _turn(
        [HumanMessage(content="deploy this on GCP"), HumanMessage(content="yes")],
        pending_clarification={"type": "infrastructure_report"},
        infrastructure_request={"type": "gcp", "app_type": "python-docker", "confirmed": True},
        infrastructure_provisioned={"type": "gcp", "status": "provisioned", "details": []},
    )
    assert "finished with status **provisioned**" in _ai(result)[-1]
    assert not result.get("plan")
