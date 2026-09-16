"""Regression tests: infrastructure provisioning must be reachable from chat.

route_planner_output used to require execution_mode == "cloud" before routing
to provision_infra, but the server only ever seeds "local"/"k8s-ray", so a
pending infrastructure_request always fell through to continue_planning and
the deploy keyword branch returned without any AI reply.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.api.workflow as wf
from app.agents.planner import plan_etl_job


# -------------------------
# Router: pending infrastructure_request reaches provision_infra
# -------------------------

@pytest.mark.parametrize("mode", ["local", "k8s-ray", "cloud", None])
def test_pending_infra_request_routes_to_provision(mode):
    state = {"infrastructure_request": {"type": "gcp", "app_type": "python-docker"}}
    if mode is not None:
        state["execution_mode"] = mode
    assert wf.route_planner_output(state) == "provision_infra"


def test_provisioned_infra_does_not_reroute():
    state = {
        "infrastructure_request": {"type": "gcp"},
        "infrastructure_provisioned": {"status": "provisioned"},
    }
    assert wf.route_planner_output(state) != "provision_infra"


def test_ready_flags_take_precedence_over_infra_request():
    state = {
        "infrastructure_request": {"type": "gcp"},
        "ready_to_code": True,
        "plan": "1. do things",
    }
    assert wf.route_planner_output(state) == "prepare_code"

    state = {
        "infrastructure_request": {"type": "gcp"},
        "ready_to_summarize": True,
    }
    assert wf.route_planner_output(state) == "summarize"


# -------------------------
# Planner deploy keyword branch: request set AND user gets a reply
# -------------------------

def _deploy_turn(prompt):
    return plan_etl_job({
        "messages": [HumanMessage(content=prompt)],
        "user_id": "u",
        "session_id": "s",
    })


def test_deploy_on_gcp_sets_request_and_replies():
    result = _deploy_turn("deploy this on GCP")
    assert (result.get("infrastructure_request") or {}).get("type") == "gcp"
    ai_replies = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
    assert ai_replies, "deploy request must produce a user-visible reply"
    assert "gcp" in ai_replies[-1].content.lower()


def test_deploy_on_aws_sets_request_and_replies():
    result = _deploy_turn("deploy this on AWS please")
    assert (result.get("infrastructure_request") or {}).get("type") == "aws"
    ai_replies = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
    assert ai_replies


def test_deploy_turn_routes_to_provision_infra():
    result = _deploy_turn("deploy this on GCP")
    assert wf.route_planner_output(result) == "provision_infra"


def test_keyword_branch_does_not_refire_after_provisioning():
    result = plan_etl_job({
        "messages": [HumanMessage(content="deploy this on GCP")],
        "user_id": "u",
        "session_id": "s",
        "infrastructure_request": {"type": "gcp", "app_type": "python-docker"},
        "infrastructure_provisioned": {"status": "provisioned"},
    })
    assert wf.route_planner_output(result) != "provision_infra"


# -------------------------
# Keyword matching must not fire on substrings of ordinary words
# -------------------------

@pytest.mark.parametrize("prompt", [
    "find flaws in the data",
    "sum the drawspeed column",          # contains "aws" as a substring
    "compute the jigcpu metric",         # contains "gcp" as a substring
    "list the sawstops per machine",
])
def test_ordinary_prompts_are_not_hijacked_by_infra_keywords(prompt):
    result = _deploy_turn(prompt)
    assert not result.get("infrastructure_request")
    assert result.get("plan"), "prompt should get a normal analysis plan"
    assert wf.route_planner_output(result) != "provision_infra"


def test_whole_word_infra_keywords_still_fire():
    result = _deploy_turn("deployment to kubernetes please")
    assert (result.get("infrastructure_request") or {}).get("type") == "gcp"


# -------------------------
# infra_agent_node honors the requested platform for either request key
# -------------------------

@pytest.fixture
def stub_invoker(monkeypatch):
    import app.agents.infra_agent as ia

    calls = {}

    def ok(step):
        return {"step_name": step, "status": "SUCCESS", "message": "", "details": ""}

    def fail_kubectl(*a, **k):
        raise RuntimeError("kubectl unavailable")

    def record_provision(platform):
        calls["platform"] = platform
        return ok("provision")

    monkeypatch.setattr(ia.subprocess, "run", fail_kubectl)
    monkeypatch.setattr(ia.k8s_invoker, "provision_cluster", record_provision)
    monkeypatch.setattr(ia.k8s_invoker, "configure_kubectl", lambda platform: ok("kubectl"))
    monkeypatch.setattr(ia.k8s_invoker, "run_cluster_health_checks", lambda: ok("health"))
    monkeypatch.setattr(ia.k8s_invoker, "deploy_application", lambda app, platform: ok("deploy"))
    monkeypatch.setattr(ia.k8s_invoker, "verify_application_status", lambda app: ok("verify"))
    monkeypatch.setattr(ia.k8s_invoker, "get_service_ip_and_port", lambda svc: ok("service"))
    return calls


@pytest.mark.parametrize("request_dict,expected", [
    ({"type": "aws", "app_type": "python-docker"}, "aws"),
    ({"platform": "aws", "app_type": "python-docker"}, "aws"),
    ({"platform": "AWS", "app_type": "python-docker"}, "aws"),
    ({"type": "gcp", "app_type": "python-docker"}, "gcp"),
])
def test_infra_agent_provisions_requested_platform(stub_invoker, request_dict, expected):
    from app.agents.infra_agent import infra_agent_node

    result = infra_agent_node({"infrastructure_request": request_dict})
    assert stub_invoker["platform"] == expected
    assert result["infrastructure_provisioned"]["type"] == expected
    assert result["infrastructure_provisioned"]["status"] == "provisioned"
