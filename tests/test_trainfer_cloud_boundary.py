"""T-34 / TI-60: the core serving/infra path imports no cloud SDK; only the Kubernetes API.

E on the base (modules absent). Also pins the upgrade path (TI-52) and the opt-in
dashboard NodePort (TI-51) at the source level.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CORE = (
    "app/serve/model_server.py",
    "app/infra/training_tasks.py",
    "app/infra/model_server_scaler.py",
    "app/infra/ray_reaper.py",
    "app/api/models_routes.py",
    "app/core/ml_registry.py",
    "app/core/ml_store.py",
)
CLOUD = ("google", "boto3", "botocore", "azure", "googleapiclient", "gcsfs", "s3fs")


def _imports(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


@pytest.mark.parametrize("rel", CORE)
def test_core_module_imports_no_cloud_sdk(rel):
    path = REPO / rel
    assert path.is_file(), rel
    offenders = sorted({m for m in _imports(path) if m.split(".")[0] in CLOUD})
    assert offenders == [], f"{rel} imports cloud SDK(s) {offenders}"
    text = path.read_text()
    assert "gcloud" not in text and "container_v1" not in text, f"{rel} calls a GKE-specific client"


def test_only_the_kubernetes_client_touches_the_cluster():
    for rel in ("app/infra/model_server_scaler.py", "app/infra/ray_reaper.py"):
        mods = set(_imports(REPO / rel))
        assert "kubernetes" in {m.split(".")[0] for m in mods}, rel
        assert not any(m.startswith("subprocess") for m in mods), f"{rel} shells out instead of using the API"


def test_upgrade_removes_the_stale_rayservice_and_nodeport_is_opt_in():
    boot = (REPO / "app/infra/cluster_bootstrap.py").read_text()
    assert "remove_ray_serve" in boot, "an upgrade must remove the pre-2026-11 RayService when the backend is not rayserve"
    assert "apply_ray_dashboard_nodeport" in boot
    raycluster = (REPO / "deploy/helm/ray/raycluster.yaml").read_text()
    assert "NodePort" not in raycluster, "the kind-only NodePort moved to ray_dashboard_nodeport.yaml (opt-in)"
    assert (REPO / "deploy/helm/ray/ray_dashboard_nodeport.yaml").is_file()
    reaper = (REPO / "app/infra/ray_reaper.py").read_text()
    assert '"rayservices"' in reaper and "stale RayService" in reaper
