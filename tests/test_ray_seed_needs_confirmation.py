"""No provisioning without the user's confirmation, on the k8s-ray path too.

_seed_infra_request_if_needed wrote a Ray infrastructure_request (namespace,
KubeRay; provision_cluster when kubectl was not configured) into every
k8s-ray turn on cloud data, unasked, and any turn that neither coded nor
summarized went through provision_infra. The owner's ruling: explicit intent
plus confirmation for ANY provisioning. Now the seed only returns the Ray
fields, the confirmation path adds them to the request it confirms, and the
router provisions only a request marked confirmed.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

import app.api.workflow as wf
from app.agents import planner

RAY_STATE = {
    "user_id": "u",
    "session_id": "s",
    "execution_mode": "k8s-ray",
    "data_source_location_cloud": "gs://bucket/big/events.parquet",
    "connection_id": "conn-1",
    "file_size_bytes": 10 * 1024 ** 3,
}


def _turn(prompt, **extra):
    return planner.plan_etl_job({"messages": [HumanMessage(content=prompt)], **RAY_STATE, **extra})


def test_a_k8s_ray_turn_seeds_no_infrastructure_request():
    result = _turn("which columns does this dataset have?")
    assert not result.get("infrastructure_request"), (
        f"the planner requested infrastructure unasked: {result.get('infrastructure_request')}")
    assert wf.route_planner_output(result) != "provision_infra"


def test_the_seed_returns_the_ray_fields_without_writing_them():
    state = {**RAY_STATE, "messages": []}
    ray = planner._seed_infra_request_if_needed(state)
    assert ray is not None and ray["use_ray"] is True and ray["type"] == "gcp"
    assert "infrastructure_request" not in state


def test_an_unconfirmed_request_does_not_route_to_provisioning():
    for request in ({"type": "gcp", "app_type": "python-docker", "use_ray": True},
                    {"type": "gcp", "app_type": "python-docker"},
                    {"type": "aws", "confirmed": "yes"}):
        assert wf.route_planner_output({"infrastructure_request": request}) != "provision_infra", request


def test_a_confirmed_request_routes_to_provisioning():
    request = {"type": "gcp", "app_type": "python-docker", "confirmed": True}
    assert wf.route_planner_output({"infrastructure_request": request}) == "provision_infra"


def test_an_explicit_request_in_k8s_ray_mode_confirms_the_ray_prerequisites():
    first = _turn("provision a GKE cluster")
    assert not first.get("infrastructure_request")
    pending = first.get("pending_clarification") or {}
    assert pending.get("type") == "infrastructure_confirmation"
    assert pending["infrastructure_request"].get("use_ray") is True, "the confirmation lost the Ray fields"
    result = _turn("yes", pending_clarification=pending)
    assert result["infrastructure_request"]["confirmed"] is True
    assert result["infrastructure_request"]["use_ray"] is True
    assert wf.route_planner_output(result) == "provision_infra"
    assert "starting gcp provisioning now" in [m for m in result["messages"]
                                               if isinstance(m, AIMessage)][-1].content.lower()
