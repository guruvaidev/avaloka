# tests/infra/test_ray_manager.py
"""Unit tests for ray_manager, install_k8s, cloud_provisioner and the provider
factory. No real cluster / Ray is touched — subprocess and helm calls are mocked."""
import os
from unittest.mock import patch

import pytest

from app.infra import cloud_provisioner, deploy_stack, install_k8s, ray_manager
from app.infra.providers import SUPPORTED_PLATFORMS, get_provider
from app.infra.providers.local_kind import LocalKindProvider
from app.infra.providers.gcp_gke import GcpGkeProvider
from app.infra.providers.aws_eks import AwsEksProvider


def _ok(step="step"):
    return {"step_name": step, "status": "SUCCESS", "message": "", "details": ""}


# ---------------- provider factory ---------------- #

def test_factory_returns_expected_providers():
    from app.infra.providers.azure_aks import AzureAksProvider
    assert isinstance(get_provider("local"), LocalKindProvider)
    assert isinstance(get_provider("gcp"), GcpGkeProvider)
    assert isinstance(get_provider("aws"), AwsEksProvider)
    assert isinstance(get_provider("azure"), AzureAksProvider)  # D5


@pytest.mark.parametrize("bad", ["", "k3s-unknown"])
def test_factory_rejects_unknown(bad):
    with pytest.raises(ValueError):
        get_provider(bad)


def test_supported_platforms():
    assert SUPPORTED_PLATFORMS == ("local", "gcp", "aws", "azure")  # D5 adds azure


@patch("app.infra.providers.gcp_gke.run_command")
@patch("app.infra.providers.gcp_gke.subprocess.run")
def test_gke_create_enables_workload_identity(mock_subprocess, mock_run, monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "dev-project")
    mock_subprocess.return_value.returncode = 1
    mock_run.return_value = _ok()

    GcpGkeProvider().provision_cluster()

    command = mock_run.call_args.args[0]
    assert "--workload-pool" in command
    assert "dev-project.svc.id.goog" in command


# ---------------- ray_manager ---------------- #

@patch("app.infra.ray_manager.run_command")
def test_install_kuberay_operator_uses_helm(mock_run):
    mock_run.return_value = _ok()
    ray_manager.install_kuberay_operator(namespace="ns")
    cmds = [c.args[0] for c in mock_run.call_args_list]
    assert ["helm", "repo", "add", "kuberay", ray_manager.KUBERAY_HELM_REPO] in cmds
    assert any(c[:3] == ["helm", "upgrade", "--install"] and "kuberay-operator" in c for c in cmds)


@patch("app.infra.ray_manager.run_command")
def test_install_kuberay_operator_aborts_on_repo_failure(mock_run):
    mock_run.return_value = {"status": "FAILED", "step_name": "Add KubeRay Helm repo", "message": "x", "details": ""}
    out = ray_manager.install_kuberay_operator()
    assert out["status"] == "FAILED"
    # only the repo-add call happened; no upgrade attempted
    assert mock_run.call_count == 1


@patch("app.infra.ray_manager.run_command")
def test_apply_ray_cluster_waits_for_head(mock_run):
    mock_run.return_value = _ok()
    ray_manager.apply_ray_cluster(namespace="ns")
    cmds = [c.args[0] for c in mock_run.call_args_list]
    assert any(c[:2] == ["kubectl", "apply"] for c in cmds)
    assert any("wait" in c and "ray.io/node-type=head" in c for c in cmds)


def test_render_manifest_keeps_side_loaded_kind_image(monkeypatch):
    monkeypatch.setattr(ray_manager, "RAY_IMAGE", "avaloka-ray:branch-test")
    monkeypatch.delenv("AVALOKA_RAY_IMAGE_PULL_POLICY", raising=False)

    rendered = ray_manager._render_manifest(ray_manager.RAYSERVICE_MANIFEST)
    try:
        text = open(rendered).read()
        assert "image: avaloka-ray:branch-test" in text
        assert "imagePullPolicy: IfNotPresent" in text
    finally:
        os.unlink(rendered)


def test_render_manifest_pulls_registry_image(monkeypatch):
    monkeypatch.setattr(ray_manager, "RAY_IMAGE", "us-central1-docker.pkg.dev/p/ray:v1")
    monkeypatch.delenv("AVALOKA_RAY_IMAGE_PULL_POLICY", raising=False)

    rendered = ray_manager._render_manifest(ray_manager.RAYSERVICE_MANIFEST)
    try:
        text = open(rendered).read()
        assert "image: us-central1-docker.pkg.dev/p/ray:v1" in text
        assert "imagePullPolicy: Always" in text
    finally:
        os.unlink(rendered)


def test_resolve_ray_address_prefers_explicit(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "ray://env:10001")
    assert ray_manager.resolve_ray_address("ray://explicit:10001") == "ray://explicit:10001"
    assert ray_manager.resolve_ray_address(None) == "ray://env:10001"


def test_connect_without_address_fails(monkeypatch):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    out = ray_manager.connect(None)
    assert out["status"] == "FAILED"


# ---------------- install_k8s preflight ---------------- #

def test_check_tools_reports_missing():
    with patch("app.infra.install_k8s.shutil.which", return_value=None):
        out = install_k8s.check_tools(["kubectl", "helm"])
    assert out["status"] == "FAILED"
    assert "kubectl" in out["message"]


def test_check_tools_all_present():
    with patch("app.infra.install_k8s.shutil.which", return_value="/usr/bin/x"):
        out = install_k8s.check_tools_for("local")
    assert out["status"] == "SUCCESS"


# ---------------- cloud_provisioner delegation ---------------- #

def test_cloud_provisioner_stops_on_provision_failure():
    class FakeProvider:
        def provision_cluster(self):
            return {"step_name": "p", "status": "FAILED", "message": "", "details": ""}
        def configure_kubectl(self):  # should not be reached
            raise AssertionError("configure_kubectl should not run after provision failure")
        def health_check(self):
            raise AssertionError("health_check should not run after provision failure")

    with patch("app.infra.cloud_provisioner.get_provider", return_value=FakeProvider()):
        outcomes = cloud_provisioner.provision("local")
    assert len(outcomes) == 1 and outcomes[0]["status"] == "FAILED"


# ---------------- deploy_stack helm command shape ---------------- #

@patch("app.infra.deploy_stack.run_command")
def test_deploy_avaloka_connect_sets_address(mock_run):
    mock_run.return_value = _ok()
    deploy_stack.deploy_avaloka(connect_existing=True, ray_address="ray://h:10001",
                                groq_planning_key="pk", groq_coding_key="ck")
    cmd = mock_run.call_args.args[0]
    assert "ray.connectExisting=true" in cmd
    assert "ray.address=ray://h:10001" in cmd
    assert "secrets.groqPlanningKey=pk" in cmd


@patch("app.infra.deploy_stack.run_command")
def test_deploy_data_stack_skips_unknown(mock_run):
    mock_run.return_value = _ok()
    outcomes = deploy_stack.deploy_data_stack(["not-a-real-component"])
    assert outcomes[0]["status"] == "SKIPPED"
    mock_run.assert_not_called()
