"""T-15: the per-job RayCluster path is off by default and ClusterIP-only when on.

Base (develop-1.6 @ 9cd9e122): RayTrainer enters the owns-cluster branch
unconditionally (A: the sentinel from the patched create_ray_cluster fires),
and get_ray_dashboard_url returns the LoadBalancer ingress IP (A).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class _Sentinel(Exception):
    pass


def _plan():
    return {"model_type": "classification", "model_name": "m", "model_description": "", "model_version": "1",
            "hyperparameter_config": {}, "ray_config": {"num_workers": 1, "cpu_per_worker": 1, "memory_per_worker": "1Gi"},
            "data_config": {"dataset_uri": "s3://b/x.csv", "feature_columns": ["a"], "target_column": "t"}}


def test_without_a_shared_cluster_and_the_flag_off_training_fails_plainly(monkeypatch):
    from app.agents.mta_v2 import ray_trainer as rt
    monkeypatch.delenv("MTA_RAY_DASHBOARD_URL", raising=False)
    monkeypatch.delenv("AVALOKA_ALLOW_LEGACY_RAY_CLUSTER", raising=False)
    monkeypatch.setenv("MTA_RAY_SUBMISSION_MODE", "direct")   # the dashboard-API path; kuberay is the default now
    with patch.object(rt, "create_ray_cluster", side_effect=_Sentinel("a per-job cluster was provisioned")), \
         patch.object(rt, "get_k8s_api", return_value=MagicMock()), \
         patch.object(rt.RayTrainer, "_read_gcp_sa_json", return_value="{}", create=True):
        result = rt.RayTrainer(user_id="u", session_id="s").train(_plan())
    assert result["status"] == "error"
    assert "per-job Ray clusters are disabled" in result["error"] or "AVALOKA_ALLOW_LEGACY_RAY_CLUSTER" in result["error"]


def test_with_the_flag_on_the_legacy_path_runs_and_uses_a_cluster_local_url(monkeypatch):
    from app.agents.mta_v2 import ray_trainer as rt
    from app.agents.mta_v2.gcp import get_ray_dashboard_url as gurl
    monkeypatch.delenv("MTA_RAY_DASHBOARD_URL", raising=False)
    monkeypatch.setenv("AVALOKA_ALLOW_LEGACY_RAY_CLUSTER", "1")
    monkeypatch.setenv("MTA_RAY_SUBMISSION_MODE", "direct")
    seen = {}

    def fake_submit(dashboard_url, env, entry_point=None):
        seen["url"] = dashboard_url
        raise _Sentinel("stop after submit")

    core = MagicMock()
    core.read_namespaced_service.return_value = SimpleNamespace(
        status=SimpleNamespace(load_balancer=SimpleNamespace(ingress=[SimpleNamespace(ip="203.0.113.7", hostname=None)])))
    with patch.object(rt, "create_ray_cluster"), patch.object(rt, "create_namespace"), \
         patch.object(rt, "get_k8s_api", return_value=MagicMock()), \
         patch.object(rt, "get_ray_head_svc", return_value="ray-head-svc"), \
         patch.object(rt, "submit_ray_job", side_effect=fake_submit), \
         patch.object(rt.RayTrainer, "_read_gcp_sa_json", return_value="{}", create=True), \
         patch.object(gurl.client, "CoreV1Api", return_value=core):
        rt.RayTrainer(user_id="u", session_id="s").train(_plan())
    assert "url" in seen, "the legacy path must run when the flag is on"
    assert seen["url"].endswith(".svc.cluster.local:8265")
    assert "203.0.113.7" not in seen["url"], "the public ingress IP is never used"


def test_cluster_local_url_helper():
    from app.agents.mta_v2.gcp.get_ray_dashboard_url import cluster_local_dashboard_url
    assert cluster_local_dashboard_url("ray-head-svc", "ray-trainer-ab12") == "http://ray-head-svc.ray-trainer-ab12.svc.cluster.local:8265"
