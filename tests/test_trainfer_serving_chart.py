"""Serving topology invariants (T-17, T-33): one ClusterIP model-server Deployment at
zero replicas with an HPA and a NetworkPolicy; scoped RBAC; the RayService opt-in
with Serve autoscaling; no public LoadBalancer anywhere on the Ray/serving path.

Templates are scanned textually (``{{ }}`` is not YAML), as tests/contract does;
values.yaml is parsed. Base (develop-1.6 @ 9cd9e122): the model-server template
and values are absent (A on the file checks), rayservice.yaml pins
``num_replicas: 1`` (A), and the legacy code creates LoadBalancers (A).
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CHART = REPO / "deploy" / "helm" / "avaloka"
TEMPLATE = CHART / "templates" / "model-server.yaml"
RBAC = CHART / "templates" / "rbac.yaml"
RAYSERVICE = REPO / "deploy" / "helm" / "ray" / "rayservice.yaml"
LEGACY = REPO / "app" / "agents" / "mta_v2" / "gcp"


def _values():
    return yaml.safe_load(CHART.joinpath("values.yaml").read_text())


def test_model_server_is_one_clusterip_deployment_at_zero_replicas_with_hpa_and_networkpolicy():
    assert TEMPLATE.is_file(), "deploy/helm/avaloka/templates/model-server.yaml is missing"
    text = TEMPLATE.read_text()
    kinds = re.findall(r"^kind:\s*(\S+)", text, re.M)
    assert kinds.count("Deployment") >= 1 and "Service" in kinds and "HorizontalPodAutoscaler" in kinds and "NetworkPolicy" in kinds
    assert re.search(r"replicas:\s*\{\{\s*\.Values\.modelServer\.initialReplicas\s*\}\}", text)
    assert "type: ClusterIP" in text and "LoadBalancer" not in text
    assert re.search(r"minReplicas:\s*1\b", text), "the HPA owns 1..N; the reaper owns 0<->1"
    assert "app.serve.model_server" in text
    assert "values: [api, langgraph, celery]" in text, "only the proxying pods may reach the server"


def test_values_default_to_zero_replicas_two_hour_idle_and_controller_scale_to_zero():
    ms = _values()["modelServer"]
    assert ms["enabled"] is True
    assert ms["initialReplicas"] == 0
    assert ms["idleTimeoutS"] == 7200
    assert ms["scaleToZero"] == "controller"
    assert ms["gpu"]["enabled"] is False, "GPU is on request only"
    assert ms["maxReplicas"] >= 2


def test_rbac_scopes_scale_rights_to_the_model_server_deployments_only():
    text = RBAC.read_text()
    blocks = re.findall(r"resources:\s*\[\"deployments/scale\"\]\s*\n\s*resourceNames:\s*\[([^\]]+)\]\s*\n\s*verbs:\s*\[([^\]]+)\]", text)
    assert len(blocks) >= 2, "both the app and the celery Role need a scoped deployments/scale rule"
    for names, verbs in blocks:
        assert "model-server" in names and "model-server-gpu" in names
        assert set(v.strip().strip('"') for v in verbs.split(",")) == {"get", "patch"}
    assert "ClusterRole" not in text, "no cluster-wide rights"


def test_rayservice_is_opt_in_with_serve_autoscaling_and_no_fixed_replica():
    text = RAYSERVICE.read_text()
    assert "num_replicas: 1" not in text
    assert re.search(r"autoscaling_config:\s*\n\s*min_replicas:\s*0", text)
    assert "enableInTreeAutoscaling: true" in text
    assert re.search(r"groupName: serve-workers\s*\n\s*replicas:\s*0\s*\n\s*minReplicas:\s*0", text)
    assert "OPT-IN" in text and "INFERENCE_BACKEND=rayserve" in text
    assert "LoadBalancer" not in text


def test_no_public_loadbalancer_on_any_ray_or_serving_path():
    offenders = []
    for path in list((REPO / "deploy" / "helm" / "ray").glob("*.yaml")) + list(LEGACY.glob("*.py")) + [TEMPLATE]:
        text = path.read_text()
        if re.search(r"type:\s*LoadBalancer|type=\"LoadBalancer\"|serviceType\":\s*\"LoadBalancer", text):
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == [], offenders
    assert "_wait_for_external_ip" not in (LEGACY / "run_inference_service.py").read_text()
    assert "load_balancer.ingress" not in (LEGACY / "get_ray_dashboard_url.py").read_text()


def test_bootstrap_applies_the_rayservice_only_for_the_rayserve_backend():
    text = (REPO / "app" / "infra" / "cluster_bootstrap.py").read_text()
    assert re.search(r'INFERENCE_BACKEND.*rayserve', text), "the RayService step must be gated on the backend"
