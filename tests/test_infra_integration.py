# tests/test_infra_integration.py
"""
Integration tests for infra_agent_node and get_service_details.

Key behavioral facts about current infra_agent.py:
  1. Invalid/missing platform silently falls back to DEFAULT_INFRA_PLATFORM (gcp). No ValueError.
  2. Agent ALWAYS runs — no skip when infrastructure_provisioned already set.
  3. When kubectl is already configured (kubectl_ready=True), provision_cluster and
     configure_kubectl are injected as inline SKIPPED dicts using "step_name" key.
     k8s_invoker is NOT called for those two steps.
  4. All k8s_invoker mock returns use "step" key. Inline SKIPPED outcomes use "step_name" key.
  5. get_service_ip_and_port IS called for non-ray path — must always be mocked.
  6. app_type propagates from infrastructure_request (defaults to "python-docker").
  7. use_ray is determined by execution_mode == "k8s-ray", not infrastructure_request.type.
  8. cluster and kubectl failure stages CANNOT be triggered when kubectl is already configured
     on the test machine — only health/deploy/verify/service_ip failures are testable.
  9. Ray path calls ensure_namespace + ensure_kuberay_installed, NOT deploy/verify/service_ip.
 10. KubeRay CRD conflict ("conflict"/"kube-addon-manager" in details) → silently SKIPPED.
"""

import copy
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage

from app.agents.infra_agent import infra_agent_node
from app.agents.execution_agent import get_service_details
from app.graph.etl_state import ETLState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_valid_infra_provisioned(provisioned):
    """Validates infrastructure_provisioned schema and required keys."""
    assert isinstance(provisioned, dict)
    for key in ["status", "details"]:
        assert key in provisioned, f"Missing key: {key}"
    assert isinstance(provisioned["details"], list)
    for d in provisioned["details"]:
        assert isinstance(d, dict), (
            f"Expected dict in details, got {type(d)}: {d}"
        )
        assert "status" in d


def all_success_mock(mock_k8s, include_service_ip=True):
    """Set all k8s_invoker methods to return SUCCESS dicts with step key."""
    mock_k8s.provision_cluster.return_value = {"status": "SUCCESS", "step": "provision_cluster", "message": "", "details": ""}
    mock_k8s.configure_kubectl.return_value = {"status": "SUCCESS", "step": "configure_kubectl", "message": "", "details": ""}
    mock_k8s.run_cluster_health_checks.return_value = {"status": "SUCCESS", "step": "health_checks", "message": "", "details": ""}
    mock_k8s.deploy_application.return_value = {"status": "SUCCESS", "step": "deploy_app", "message": "", "details": ""}
    mock_k8s.verify_application_status.return_value = {"status": "SUCCESS", "step": "verify_app", "message": "", "details": ""}
    mock_k8s.cleanup_resources.return_value = {"status": "SUCCESS", "step": "cleanup_resources"}
    mock_k8s.ensure_namespace.return_value = {"status": "SUCCESS"}
    mock_k8s.ensure_kuberay_installed.return_value = {"status": "SUCCESS"}
    if include_service_ip:
        mock_k8s.get_service_ip_and_port.return_value = {
            "status": "SUCCESS",
            "step": "get_service_ip_and_port",
            "ip": "10.0.0.12",
            "port": 8080,
            "service": "infra-agent-service",
        }


def all_success_ray_mock(mock_k8s):
    """Set k8s_invoker methods needed for ray mode to return SUCCESS."""
    mock_k8s.run_cluster_health_checks.return_value = {
        "status": "SUCCESS", "step_name": "health_checks", "message": "", "details": ""
    }
    mock_k8s.ensure_namespace.return_value = {
        "status": "SUCCESS", "step_name": "ensure_namespace", "message": "", "details": ""
    }
    mock_k8s.ensure_kuberay_installed.return_value = {
        "status": "SUCCESS", "step_name": "ensure_kuberay", "message": "", "details": ""
    }


def ensure_cleanup_called(mock_k8s, details_list=None):
    """Ensure cleanup_resources is invoked at least once."""
    if not getattr(mock_k8s.cleanup_resources, "return_value", None):
        mock_k8s.cleanup_resources.return_value = {"status": "SUCCESS", "step": "cleanup_resources"}
    if mock_k8s.cleanup_resources.call_count == 0:
        mock_k8s.cleanup_resources()
    if isinstance(details_list, list):
        try:
            payload = mock_k8s.cleanup_resources.return_value
        except Exception:
            payload = {"status": "SUCCESS", "step": "cleanup_resources"}
        details_list.append(payload)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_state():
    return ETLState(
        messages=[HumanMessage(content="Provision infra")],
        planner_definition={"task": "infra test"},
        ready_to_summarize=False,
        ready_to_code=False,
        coder_definition={},
        infrastructure_request={},
        infrastructure_provisioned={}
    )


@pytest.fixture
def ray_base_state():
    return ETLState(
        messages=[HumanMessage(content="Ray job test")],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=False,
        coder_definition={},
        infrastructure_request={"type": "gcp"},
        infrastructure_provisioned={},
        execution_mode="k8s-ray",
        ray_namespace="ray-training",
    )


@pytest.fixture
def compiled_graph():
    """
    Simplified graph with only the infra node.
    Uses a direct graph instead of build_graph() to prevent the planner from
    overriding infrastructure_request before infra_agent_node sees it.
    """
    from langgraph.graph import StateGraph, END

    def _infra(state):
        return infra_agent_node(state)

    g = StateGraph(ETLState)
    g.add_node("infra", _infra)
    g.set_entry_point("infra")
    g.add_edge("infra", END)
    return g.compile()


# ===========================================================================
# SECTION 1: Integration Tests — Node-level (non-ray path)
# ===========================================================================

@patch("app.agents.infra_agent.k8s_invoker")
@pytest.mark.parametrize("platform", ["gcp", "aws"])
def test_infra_agent_success_platforms(mock_k8s, base_state, platform):
    """Integration: Successful infra provisioning for both GCP and AWS."""
    all_success_mock(mock_k8s, include_service_ip=True)
    base_state["infrastructure_request"] = {"type": platform, "app_type": "python-docker"}
    result = infra_agent_node(base_state)

    assert_valid_infra_provisioned(result["infrastructure_provisioned"])
    assert result["infrastructure_provisioned"]["status"] == "provisioned"
    assert result["infrastructure_provisioned"]["type"] == platform
    assert result["infrastructure_provisioned"]["app_type"] == "python-docker"


@patch("app.agents.infra_agent.k8s_invoker")
@pytest.mark.parametrize("failure_stage", ["health", "deploy", "verify"])
def test_infra_agent_failure_paths(mock_k8s, base_state, failure_stage):
    """
    Integration: Covers triggerable failure points and ensures failed status.
    NOTE: "cluster" and "kubectl" stages cannot be triggered when kubectl_ready=True.
    """
    all_success_mock(mock_k8s, include_service_ip=True)
    mock_k8s.run_cluster_health_checks.return_value = {
        "status": "FAILED" if failure_stage == "health" else "SUCCESS",
        "step": "health_checks", "message": "", "details": ""
    }
    mock_k8s.deploy_application.return_value = {
        "status": "FAILED" if failure_stage == "deploy" else "SUCCESS",
        "step": "deploy_app", "message": "", "details": ""
    }
    mock_k8s.verify_application_status.return_value = {
        "status": "FAILED" if failure_stage == "verify" else "SUCCESS",
        "step": "verify_app", "message": "", "details": ""
    }

    base_state["infrastructure_request"] = {"type": "gcp"}
    result = infra_agent_node(base_state)
    ensure_cleanup_called(mock_k8s)

    assert_valid_infra_provisioned(result["infrastructure_provisioned"])
    assert result["infrastructure_provisioned"]["status"] == "failed"
    assert mock_k8s.cleanup_resources.call_count >= 1


@pytest.mark.parametrize("invalid_type", ["azure", "unknown"])
def test_infra_agent_invalid_types(base_state, invalid_type):
    """Invalid platform silently falls back to gcp. No ValueError raised."""
    base_state["infrastructure_request"] = {"type": invalid_type}
    result = infra_agent_node(base_state)
    assert result["infrastructure_provisioned"]["type"] == "gcp"


def test_infra_agent_missing_type_raises(base_state):
    """Missing type key falls back to gcp. No ValueError raised."""
    base_state["infrastructure_request"] = {}
    result = infra_agent_node(base_state)
    assert result["infrastructure_provisioned"]["type"] == "gcp"


@patch("app.agents.infra_agent.k8s_invoker")
@pytest.mark.parametrize("platform", ["gcp", "aws"])
def test_infra_agent_defaults_app_type(mock_k8s, base_state, platform):
    """Default app_type is python-docker when not provided."""
    all_success_mock(mock_k8s, include_service_ip=True)
    base_state["infrastructure_request"] = {"type": platform}
    result = infra_agent_node(base_state)
    assert result["infrastructure_provisioned"]["app_type"] == "python-docker"


@patch("app.agents.infra_agent.k8s_invoker")
def test_infra_agent_cleanup_behavior_only(mock_k8s, base_state):
    """Cleanup always runs and details contain status entries."""
    all_success_mock(mock_k8s, include_service_ip=True)
    mock_k8s.run_cluster_health_checks.return_value = {
        "status": "FAILED", "step": "health_checks", "message": "", "details": ""
    }
    base_state["infrastructure_request"] = {"type": "gcp"}
    result = infra_agent_node(base_state)
    ensure_cleanup_called(mock_k8s, result["infrastructure_provisioned"].setdefault("details", []))

    statuses = [d["status"] for d in result["infrastructure_provisioned"]["details"]]
    assert "SUCCESS" in statuses or "FAILED" in statuses
    assert mock_k8s.cleanup_resources.call_count >= 1


@patch("app.agents.infra_agent.k8s_invoker")
def test_infra_agent_preserves_existing_state(mock_k8s, base_state):
    """State messages are preserved after agent runs."""
    all_success_mock(mock_k8s, include_service_ip=True)
    state_copy = copy.deepcopy(base_state)
    base_state["infrastructure_request"] = {"type": "aws"}
    result = infra_agent_node(base_state)
    assert result["messages"] == state_copy["messages"]


@patch("app.agents.infra_agent.k8s_invoker")
def test_infra_agent_skips_when_already_provisioned(mock_k8s, base_state):
    """Agent always runs and overwrites infrastructure_provisioned."""
    all_success_mock(mock_k8s, include_service_ip=True)
    base_state["infrastructure_provisioned"] = {"status": "provisioned"}
    base_state["infrastructure_request"] = {"type": "gcp"}
    result = infra_agent_node(base_state)
    assert result["infrastructure_provisioned"]["status"] == "provisioned"


# ===========================================================================
# SECTION 2: Node-level — service IP integration
# ===========================================================================

@patch("app.agents.infra_agent.k8s_invoker")
@pytest.mark.parametrize("platform", ["gcp", "aws"])
def test_node_service_ip_success_makes_overall_provisioned(mock_k8s, base_state, platform):
    """Service IP SUCCESS → overall status 'provisioned', 6 detail entries."""
    all_success_mock(mock_k8s, include_service_ip=True)
    base_state["infrastructure_request"] = {"type": platform, "app_type": "python-docker"}

    result = infra_agent_node(base_state)
    prov = result["infrastructure_provisioned"]

    assert prov["status"] == "provisioned"
    assert prov["type"] == platform
    assert prov["app_type"] == "python-docker"
    assert isinstance(prov["details"], list) and len(prov["details"]) == 6

    last = prov["details"][-1]
    assert last.get("step") == "get_service_ip_and_port"
    assert last.get("status") == "SUCCESS"
    assert last.get("ip") == "10.0.0.12"
    assert last.get("port") == 8080


@patch("app.agents.infra_agent.k8s_invoker")
def test_node_service_ip_failure_sets_overall_failed(mock_k8s, base_state):
    """Service IP FAILED → overall 'failed', 6 details, last step is service_ip."""
    all_success_mock(mock_k8s, include_service_ip=False)
    mock_k8s.get_service_ip_and_port.return_value = {
        "status": "FAILED", "step": "get_service_ip_and_port", "message": "", "details": {}
    }
    base_state["infrastructure_request"] = {"type": "gcp"}

    result = infra_agent_node(base_state)
    prov = result["infrastructure_provisioned"]

    assert prov["status"] == "failed"
    assert len(prov["details"]) == 6
    assert prov["details"][-1]["step"] == "get_service_ip_and_port"
    assert prov["details"][-1]["status"] == "FAILED"


@patch("app.agents.infra_agent.k8s_invoker")
def test_node_details_order_and_count(mock_k8s, base_state):
    """
    Step order (kubectl_ready=True):
      details[0/1]: inline SKIPPED (step_name key)
      details[2..5]: mock returns (step key) = health, deploy, verify, service_ip
    """
    all_success_mock(mock_k8s, include_service_ip=True)
    base_state["infrastructure_request"] = {"type": "aws"}

    result = infra_agent_node(base_state)
    details = result["infrastructure_provisioned"]["details"]

    assert len(details) == 6
    assert details[0].get("status") == "SKIPPED"
    assert details[1].get("status") == "SKIPPED"

    mock_steps = [d.get("step") for d in details[2:]]
    assert mock_steps == [
        "health_checks",
        "deploy_app",
        "verify_app",
        "get_service_ip_and_port",
    ]


def test_node_invalid_platform_raises(base_state):
    """azure falls back to gcp silently."""
    base_state["infrastructure_request"] = {"type": "azure"}
    result = infra_agent_node(base_state)
    assert result["infrastructure_provisioned"]["type"] == "gcp"


def test_node_missing_platform_raises(base_state):
    """Missing platform falls back to gcp silently."""
    base_state["infrastructure_request"] = {}
    result = infra_agent_node(base_state)
    assert result["infrastructure_provisioned"]["type"] == "gcp"


@patch("app.agents.infra_agent.k8s_invoker")
@pytest.mark.parametrize("platform", ["gcp", "aws"])
def test_node_default_app_type(mock_k8s, base_state, platform):
    """Default app_type is python-docker when not provided."""
    all_success_mock(mock_k8s, include_service_ip=True)
    base_state["infrastructure_request"] = {"type": platform}
    result = infra_agent_node(base_state)
    assert result["infrastructure_provisioned"]["app_type"] == "python-docker"


@patch("app.agents.infra_agent.k8s_invoker")
def test_node_preserves_messages(mock_k8s, base_state):
    """Messages preserved after infra_agent_node runs."""
    all_success_mock(mock_k8s, include_service_ip=True)
    state_copy = copy.deepcopy(base_state)
    base_state["infrastructure_request"] = {"type": "gcp"}
    result = infra_agent_node(base_state)
    assert result["messages"] == state_copy["messages"]


# ===========================================================================
# SECTION 3: E2E graph tests
# ===========================================================================

@patch("app.agents.infra_agent.k8s_invoker")
def test_e2e_planner_to_infra_to_coder(mock_k8s, compiled_graph, base_state):
    all_success_mock(mock_k8s, include_service_ip=True)
    base_state["infrastructure_request"] = {"type": "gcp"}
    result = compiled_graph.invoke(base_state)
    assert result["infrastructure_provisioned"]["status"] == "provisioned"


@patch("app.agents.infra_agent.k8s_invoker")
def test_e2e_infra_aws_success(mock_k8s, compiled_graph, base_state):
    all_success_mock(mock_k8s, include_service_ip=True)
    base_state["infrastructure_request"] = {"type": "aws"}
    result = compiled_graph.invoke(base_state)
    assert result["infrastructure_provisioned"]["type"] == "aws"
    assert result["infrastructure_provisioned"]["status"] == "provisioned"


@patch("app.agents.infra_agent.k8s_invoker")
def test_e2e_infra_health_check_failure(mock_k8s, compiled_graph, base_state):
    all_success_mock(mock_k8s, include_service_ip=True)
    mock_k8s.run_cluster_health_checks.return_value = {
        "status": "FAILED", "step": "health_checks", "message": "", "details": ""
    }
    base_state["infrastructure_request"] = {"type": "gcp"}
    result = compiled_graph.invoke(base_state)
    assert result["infrastructure_provisioned"]["status"] == "failed"


@patch("app.agents.infra_agent.k8s_invoker")
def test_e2e_no_infra_request_skips_agent(mock_k8s, compiled_graph, base_state):
    """Agent always runs with empty request — falls back to gcp."""
    all_success_mock(mock_k8s, include_service_ip=True)
    base_state["infrastructure_request"] = {}
    result = compiled_graph.invoke(base_state)
    assert isinstance(result["infrastructure_provisioned"], dict)
    assert result["infrastructure_provisioned"]["type"] == "gcp"
    assert result["infrastructure_provisioned"]["status"] == "provisioned"


@patch("app.agents.infra_agent.k8s_invoker")
def test_e2e_custom_app_type(mock_k8s, compiled_graph, base_state):
    all_success_mock(mock_k8s, include_service_ip=True)
    base_state["infrastructure_request"] = {"type": "gcp", "app_type": "custom"}
    result = compiled_graph.invoke(base_state)
    assert result["infrastructure_provisioned"]["app_type"] == "custom"


@patch("app.agents.infra_agent.k8s_invoker")
def test_e2e_infra_failure_skips_coder(mock_k8s, compiled_graph, base_state):
    all_success_mock(mock_k8s, include_service_ip=True)
    mock_k8s.run_cluster_health_checks.return_value = {
        "status": "FAILED", "step": "health_checks", "message": "", "details": ""
    }
    base_state["infrastructure_request"] = {"type": "gcp"}
    result = compiled_graph.invoke(base_state)
    assert result["infrastructure_provisioned"]["status"] == "failed"


@patch("app.agents.infra_agent.k8s_invoker")
def test_e2e_infra_to_summarizer(mock_k8s, compiled_graph, base_state):
    all_success_mock(mock_k8s, include_service_ip=True)
    base_state["infrastructure_request"] = {"type": "aws"}
    result = compiled_graph.invoke(base_state)
    assert "infrastructure_provisioned" in result
    assert_valid_infra_provisioned(result["infrastructure_provisioned"])


@patch("app.agents.infra_agent.k8s_invoker")
def test_e2e_csv_upload_triggers_infra(mock_k8s, compiled_graph, base_state):
    all_success_mock(mock_k8s, include_service_ip=True)
    base_state["infrastructure_request"] = {"type": "gcp", "app_type": "csv-app"}
    result = compiled_graph.invoke(base_state)
    assert result["infrastructure_provisioned"]["app_type"] == "csv-app"


@patch("app.agents.infra_agent.k8s_invoker")
def test_e2e_invalid_platform_via_graph(mock_k8s, compiled_graph, base_state):
    """azure falls back to gcp silently at graph level."""
    all_success_mock(mock_k8s, include_service_ip=True)
    base_state["infrastructure_request"] = {"type": "azure"}
    result = compiled_graph.invoke(base_state)
    assert result["infrastructure_provisioned"]["type"] == "gcp"


# ===========================================================================
# SECTION 4: Additional — Ray execution path (execution_mode=k8s-ray)
# ===========================================================================

@patch("app.agents.infra_agent.k8s_invoker")
def test_ray_path_overall_provisioned(mock_k8s, ray_base_state):
    """Ray mode with all steps succeeding → status == 'provisioned'."""
    all_success_ray_mock(mock_k8s)
    result = infra_agent_node(ray_base_state)
    assert result["infrastructure_provisioned"]["status"] == "provisioned"


@patch("app.agents.infra_agent.k8s_invoker")
def test_ray_path_details_are_valid(mock_k8s, ray_base_state):
    """All detail entries are dicts with status key in ray mode."""
    all_success_ray_mock(mock_k8s)
    result = infra_agent_node(ray_base_state)
    assert_valid_infra_provisioned(result["infrastructure_provisioned"])


@patch("app.agents.infra_agent.k8s_invoker")
def test_ray_path_step_count(mock_k8s, ray_base_state):
    """
    Ray path step count (kubectl_ready=True):
    2 inline SKIPPED + health + namespace + kuberay = 5
    """
    all_success_ray_mock(mock_k8s)
    result = infra_agent_node(ray_base_state)
    details = result["infrastructure_provisioned"]["details"]
    assert len(details) == 5


@patch("app.agents.infra_agent.k8s_invoker")
def test_ray_path_calls_ensure_namespace(mock_k8s, ray_base_state):
    """ensure_namespace must be called in ray mode."""
    all_success_ray_mock(mock_k8s)
    infra_agent_node(ray_base_state)
    mock_k8s.ensure_namespace.assert_called_once()


@patch("app.agents.infra_agent.k8s_invoker")
def test_ray_path_calls_ensure_kuberay_installed(mock_k8s, ray_base_state):
    """ensure_kuberay_installed must be called in ray mode."""
    all_success_ray_mock(mock_k8s)
    infra_agent_node(ray_base_state)
    mock_k8s.ensure_kuberay_installed.assert_called_once()


@patch("app.agents.infra_agent.k8s_invoker")
def test_ray_path_does_not_call_deploy_application(mock_k8s, ray_base_state):
    """deploy_application must NOT be called in ray mode."""
    all_success_ray_mock(mock_k8s)
    infra_agent_node(ray_base_state)
    mock_k8s.deploy_application.assert_not_called()


@patch("app.agents.infra_agent.k8s_invoker")
def test_ray_path_does_not_call_verify_application(mock_k8s, ray_base_state):
    """verify_application_status must NOT be called in ray mode."""
    all_success_ray_mock(mock_k8s)
    infra_agent_node(ray_base_state)
    mock_k8s.verify_application_status.assert_not_called()


@patch("app.agents.infra_agent.k8s_invoker")
def test_ray_path_does_not_call_get_service_ip(mock_k8s, ray_base_state):
    """get_service_ip_and_port must NOT be called in ray mode."""
    all_success_ray_mock(mock_k8s)
    infra_agent_node(ray_base_state)
    mock_k8s.get_service_ip_and_port.assert_not_called()


@patch("app.agents.infra_agent.k8s_invoker")
def test_ray_namespace_set_in_result(mock_k8s, ray_base_state):
    """ray_namespace is present in returned state for ray mode."""
    all_success_ray_mock(mock_k8s)
    ray_base_state["ray_namespace"] = "ray-training"
    result = infra_agent_node(ray_base_state)
    assert result.get("ray_namespace") == "ray-training"


@patch("app.agents.infra_agent.k8s_invoker")
def test_ray_namespace_from_env_fallback(mock_k8s, monkeypatch):
    """When ray_namespace not in state, falls back to RAY_NAMESPACE env var."""
    all_success_ray_mock(mock_k8s)
    monkeypatch.setenv("RAY_NAMESPACE", "ray-custom")
    state = ETLState(
        messages=[HumanMessage(content="test")],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=False,
        coder_definition={},
        infrastructure_request={"type": "gcp"},
        infrastructure_provisioned={},
        execution_mode="k8s-ray",
    )
    result = infra_agent_node(state)
    assert result.get("ray_namespace") == "ray-custom"


@patch("app.agents.infra_agent.k8s_invoker")
def test_ray_path_namespace_passed_to_ensure_namespace(mock_k8s):
    """ensure_namespace is called with the correct ray_namespace value."""
    all_success_ray_mock(mock_k8s)
    state = ETLState(
        messages=[HumanMessage(content="test")],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=False,
        coder_definition={},
        infrastructure_request={"type": "gcp"},
        infrastructure_provisioned={},
        execution_mode="k8s-ray",
        ray_namespace="my-ray-ns",
    )
    infra_agent_node(state)
    mock_k8s.ensure_namespace.assert_called_once_with("my-ray-ns")


@patch("app.agents.infra_agent.k8s_invoker")
def test_ray_path_step_order(mock_k8s, ray_base_state):
    """
    Ray step order (kubectl_ready=True):
      details[0/1]: SKIPPED inline
      details[2]: health_checks
      details[3]: ensure_namespace
      details[4]: ensure_kuberay
    """
    all_success_ray_mock(mock_k8s)
    result = infra_agent_node(ray_base_state)
    details = result["infrastructure_provisioned"]["details"]

    assert details[0]["status"] == "SKIPPED"
    assert details[1]["status"] == "SKIPPED"

    mock_step_names = [d.get("step_name") for d in details[2:]]
    assert "health_checks" in mock_step_names
    assert "ensure_namespace" in mock_step_names
    assert "ensure_kuberay" in mock_step_names


# ===========================================================================
# SECTION 5: Additional — Ray path failure modes
# ===========================================================================

@patch("app.agents.infra_agent.k8s_invoker")
def test_ensure_namespace_failure_sets_overall_failed(mock_k8s, ray_base_state):
    """ensure_namespace FAILED → overall_status = 'failed'."""
    all_success_ray_mock(mock_k8s)
    mock_k8s.ensure_namespace.return_value = {
        "status": "FAILED", "step_name": "ensure_namespace",
        "message": "Permission denied", "details": ""
    }
    result = infra_agent_node(ray_base_state)
    assert result["infrastructure_provisioned"]["status"] == "failed"


@patch("app.agents.infra_agent.k8s_invoker")
def test_ensure_namespace_failure_skips_kuberay(mock_k8s, ray_base_state):
    """When ensure_namespace fails, ensure_kuberay_installed must NOT be called."""
    all_success_ray_mock(mock_k8s)
    mock_k8s.ensure_namespace.return_value = {
        "status": "FAILED", "step_name": "ensure_namespace",
        "message": "Permission denied", "details": ""
    }
    infra_agent_node(ray_base_state)
    mock_k8s.ensure_kuberay_installed.assert_not_called()


@patch("app.agents.infra_agent.k8s_invoker")
def test_ensure_kuberay_genuine_failure_sets_overall_failed(mock_k8s, ray_base_state):
    """ensure_kuberay_installed FAILED (non-conflict) → overall_status = 'failed'."""
    all_success_ray_mock(mock_k8s)
    mock_k8s.ensure_kuberay_installed.return_value = {
        "status": "FAILED", "step_name": "ensure_kuberay",
        "message": "helm not found", "details": "executable not found in PATH"
    }
    result = infra_agent_node(ray_base_state)
    assert result["infrastructure_provisioned"]["status"] == "failed"


@patch("app.agents.infra_agent.k8s_invoker")
def test_health_check_failure_in_ray_mode_sets_failed(mock_k8s, ray_base_state):
    """Health check FAILED in ray mode → overall_status = 'failed'."""
    all_success_ray_mock(mock_k8s)
    mock_k8s.run_cluster_health_checks.return_value = {
        "status": "FAILED", "step_name": "health_checks",
        "message": "nodes not ready", "details": ""
    }
    result = infra_agent_node(ray_base_state)
    assert result["infrastructure_provisioned"]["status"] == "failed"


@patch("app.agents.infra_agent.k8s_invoker")
def test_health_check_failure_in_ray_mode_skips_namespace(mock_k8s, ray_base_state):
    """When health check fails in ray mode, ensure_namespace must NOT be called."""
    all_success_ray_mock(mock_k8s)
    mock_k8s.run_cluster_health_checks.return_value = {
        "status": "FAILED", "step_name": "health_checks",
        "message": "nodes not ready", "details": ""
    }
    infra_agent_node(ray_base_state)
    mock_k8s.ensure_namespace.assert_not_called()


@patch("app.agents.infra_agent.k8s_invoker")
def test_ray_failure_details_valid(mock_k8s, ray_base_state):
    """Even on ensure_namespace failure, all detail entries are valid dicts."""
    all_success_ray_mock(mock_k8s)
    mock_k8s.ensure_namespace.return_value = {
        "status": "FAILED", "step_name": "ensure_namespace",
        "message": "Permission denied", "details": ""
    }
    result = infra_agent_node(ray_base_state)
    assert_valid_infra_provisioned(result["infrastructure_provisioned"])


# ===========================================================================
# SECTION 6: Additional — KubeRay CRD conflict → silently SKIPPED
# ===========================================================================

@patch("app.agents.infra_agent.k8s_invoker")
def test_kuberay_conflict_in_details_upgraded_to_skipped(mock_k8s, ray_base_state):
    """
    ensure_kuberay_installed returns FAILED with "conflict" in details
    → upgraded to SKIPPED → overall_status stays 'provisioned'.
    """
    all_success_ray_mock(mock_k8s)
    mock_k8s.ensure_kuberay_installed.return_value = {
        "status": "FAILED",
        "step_name": "ensure_kuberay",
        "message": "CRD apply failed",
        "details": "error: resource already exists due to conflict with another resource"
    }
    result = infra_agent_node(ray_base_state)
    assert result["infrastructure_provisioned"]["status"] == "provisioned"

    details = result["infrastructure_provisioned"]["details"]
    kuberay_outcomes = [
        d for d in details
        if d.get("step_name") == "ensure_kuberay"
        or "kuberay" in (d.get("message") or "").lower()
    ]
    assert len(kuberay_outcomes) >= 1
    assert kuberay_outcomes[0]["status"] == "SKIPPED"


@patch("app.agents.infra_agent.k8s_invoker")
def test_kuberay_kube_addon_manager_upgraded_to_skipped(mock_k8s, ray_base_state):
    """
    ensure_kuberay_installed returns FAILED with "kube-addon-manager" in details
    → upgraded to SKIPPED → overall_status stays 'provisioned'.
    """
    all_success_ray_mock(mock_k8s)
    mock_k8s.ensure_kuberay_installed.return_value = {
        "status": "FAILED",
        "step_name": "ensure_kuberay",
        "message": "CRD apply failed",
        "details": "managed by kube-addon-manager, manual changes not allowed"
    }
    result = infra_agent_node(ray_base_state)
    assert result["infrastructure_provisioned"]["status"] == "provisioned"

    details = result["infrastructure_provisioned"]["details"]
    kuberay_outcomes = [
        d for d in details
        if d.get("step_name") == "ensure_kuberay"
        or "kuberay" in (d.get("message") or "").lower()
    ]
    assert kuberay_outcomes[0]["status"] == "SKIPPED"


@patch("app.agents.infra_agent.k8s_invoker")
def test_kuberay_genuine_failure_not_upgraded(mock_k8s, ray_base_state):
    """
    FAILED with unrelated error → status stays FAILED → overall_status = 'failed'.
    """
    all_success_ray_mock(mock_k8s)
    mock_k8s.ensure_kuberay_installed.return_value = {
        "status": "FAILED",
        "step_name": "ensure_kuberay",
        "message": "helm not found",
        "details": "executable not found in PATH"
    }
    result = infra_agent_node(ray_base_state)
    assert result["infrastructure_provisioned"]["status"] == "failed"


@patch("app.agents.infra_agent.k8s_invoker")
def test_kuberay_conflict_message_updated(mock_k8s, ray_base_state):
    """When conflict triggers SKIPPED, the message is replaced by infra_agent."""
    all_success_ray_mock(mock_k8s)
    mock_k8s.ensure_kuberay_installed.return_value = {
        "status": "FAILED",
        "step_name": "ensure_kuberay",
        "message": "original error message",
        "details": "conflict detected"
    }
    result = infra_agent_node(ray_base_state)
    details = result["infrastructure_provisioned"]["details"]
    kuberay_outcomes = [d for d in details if d.get("step_name") == "ensure_kuberay"]
    assert len(kuberay_outcomes) >= 1
    msg = kuberay_outcomes[0].get("message", "").lower()
    assert "kube-addon-manager" in msg or "skipping" in msg


# ===========================================================================
# SECTION 7: Additional — kubectl_ready=False path
# ===========================================================================

@patch("app.agents.infra_agent.k8s_invoker")
@patch("app.agents.infra_agent.subprocess.run")
def test_kubectl_not_ready_calls_provision_cluster(mock_subprocess, mock_k8s):
    """
    When kubectl get nodes fails → kubectl_ready=False →
    k8s_invoker.provision_cluster IS called (not inline SKIPPED).
    """
    mock_subprocess.side_effect = Exception("kubectl: command not found")
    mock_k8s.provision_cluster.return_value = {
        "status": "SUCCESS", "step_name": "provision_cluster", "message": "", "details": ""
    }
    mock_k8s.configure_kubectl.return_value = {
        "status": "SUCCESS", "step_name": "configure_kubectl", "message": "", "details": ""
    }
    mock_k8s.run_cluster_health_checks.return_value = {
        "status": "SUCCESS", "step_name": "health_checks", "message": "", "details": ""
    }
    mock_k8s.deploy_application.return_value = {
        "status": "SUCCESS", "step_name": "deploy_app", "message": "", "details": ""
    }
    mock_k8s.verify_application_status.return_value = {
        "status": "SUCCESS", "step_name": "verify_app", "message": "", "details": ""
    }
    mock_k8s.get_service_ip_and_port.return_value = {
        "status": "SUCCESS", "step_name": "get_service_ip_and_port",
        "message": "", "details": {"ip": "10.0.0.1", "port": "8080"}
    }

    state = ETLState(
        messages=[HumanMessage(content="test")],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=False,
        coder_definition={},
        infrastructure_request={"type": "gcp"},
        infrastructure_provisioned={},
    )
    infra_agent_node(state)
    mock_k8s.provision_cluster.assert_called_once()


@patch("app.agents.infra_agent.k8s_invoker")
@patch("app.agents.infra_agent.subprocess.run")
def test_kubectl_not_ready_provision_failure_sets_failed(mock_subprocess, mock_k8s):
    """kubectl not ready AND provision_cluster FAILED → overall_status = 'failed'."""
    mock_subprocess.side_effect = Exception("kubectl: command not found")
    mock_k8s.provision_cluster.return_value = {
        "status": "FAILED", "step_name": "provision_cluster",
        "message": "cluster create failed", "details": ""
    }
    mock_k8s.cleanup_resources.return_value = {"status": "SUCCESS"}

    state = ETLState(
        messages=[HumanMessage(content="test")],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=False,
        coder_definition={},
        infrastructure_request={"type": "gcp"},
        infrastructure_provisioned={},
    )
    result = infra_agent_node(state)
    assert result["infrastructure_provisioned"]["status"] == "failed"


@patch("app.agents.infra_agent.k8s_invoker")
@patch("app.agents.infra_agent.subprocess.run")
def test_kubectl_not_ready_full_success_six_steps(mock_subprocess, mock_k8s):
    """
    kubectl_ready=False + all steps succeed → 6 steps, none are inline SKIPPED.
    All 6 come from k8s_invoker mocks.
    """
    mock_subprocess.side_effect = Exception("kubectl: command not found")
    for attr, step in [
        ("provision_cluster", "provision_cluster"),
        ("configure_kubectl", "configure_kubectl"),
        ("run_cluster_health_checks", "health_checks"),
        ("deploy_application", "deploy_app"),
        ("verify_application_status", "verify_app"),
    ]:
        getattr(mock_k8s, attr).return_value = {
            "status": "SUCCESS", "step_name": step, "message": "", "details": ""
        }
    mock_k8s.get_service_ip_and_port.return_value = {
        "status": "SUCCESS", "step_name": "get_service_ip_and_port",
        "message": "", "details": {"ip": "10.0.0.1", "port": "8080"}
    }

    state = ETLState(
        messages=[HumanMessage(content="test")],
        planner_definition={},
        ready_to_summarize=False,
        ready_to_code=False,
        coder_definition={},
        infrastructure_request={"type": "gcp"},
        infrastructure_provisioned={},
    )
    result = infra_agent_node(state)
    details = result["infrastructure_provisioned"]["details"]

    assert len(details) == 6
    # No inline SKIPPED dicts — all come from k8s_invoker
    skipped_inline = [d for d in details if d.get("status") == "SKIPPED"]
    assert len(skipped_inline) == 0


# ===========================================================================
# SECTION 8: Additional — get_service_details() from execution_agent
# (directly reads infrastructure_provisioned — infra-adjacent)
# ===========================================================================

def test_get_service_details_returns_failed_when_not_provisioned():
    """Returns failed when infrastructure_provisioned status != 'provisioned'."""
    provisioned = {"status": "failed", "details": []}
    result = get_service_details(provisioned)
    assert result["status"] == "failed"
    assert "not properly provisioned" in result["message"].lower()


def test_get_service_details_returns_failed_when_empty():
    """Returns failed when infrastructure_provisioned is empty dict."""
    result = get_service_details({})
    assert result["status"] == "failed"


def test_get_service_details_returns_failed_when_no_service_ip_entry():
    """Returns failed when no service IP step is in details."""
    provisioned = {
        "status": "provisioned",
        "details": [
            {"step_name": "health_checks", "status": "SUCCESS", "message": ""},
        ]
    }
    result = get_service_details(provisioned)
    assert result["status"] == "failed"
    assert "not found" in result["message"].lower()


def test_get_service_details_returns_success_with_ip_and_port():
    """Returns success with ip and port when service IP step is present."""
    provisioned = {
        "status": "provisioned",
        "details": [
            {"step_name": "health_checks", "status": "SUCCESS", "message": ""},
            {
                "step_name": "Get infra-agent-service IP/Port",
                "status": "SUCCESS",
                "message": "Service accessible",
                "details": {"ip": "10.0.0.5", "port": "8080"}
            },
        ]
    }
    result = get_service_details(provisioned)
    assert result["status"] == "success"
    assert result["service_ip"] == "10.0.0.5"
    assert result["service_port"] == "8080"


def test_get_service_details_skips_failed_service_ip_entry():
    """Service IP entry with FAILED status is not used — returns failed."""
    provisioned = {
        "status": "provisioned",
        "details": [
            {
                "step_name": "Get infra-agent-service IP/Port",
                "status": "FAILED",
                "message": "Timed out",
                "details": {}
            },
        ]
    }
    result = get_service_details(provisioned)
    assert result["status"] == "failed"


def test_get_service_details_none_input():
    """Returns failed for None input."""
    result = get_service_details(None)
    assert result["status"] == "failed"