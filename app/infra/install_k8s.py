# app/infra/install_k8s.py
"""Host-tool preflight checks and KubeRay operator installation.

Used by ``cluster_bootstrap`` before provisioning so failures surface early with
actionable messages rather than deep in a Helm/kubectl call.
"""
from __future__ import annotations

import shutil
from typing import List

from app.infra import ray_manager

# Tools required per provider for the provision flow.
TOOL_REQUIREMENTS = {
    "local": ["docker", "kind", "kubectl", "helm"],
    "gcp": ["gcloud", "docker", "kubectl", "helm"],
    "aws": ["aws", "docker", "kubectl", "helm"],
    "azure": ["az", "docker", "kubectl", "helm"],
    "connect": ["kubectl", "helm"],
}


def check_tools(required: List[str]) -> dict:
    """Verify the required CLI tools are present on PATH."""
    missing = [t for t in required if shutil.which(t) is None]
    if missing:
        return {
            "step_name": "Preflight tool check",
            "status": "FAILED",
            "message": f"Missing required tools: {', '.join(missing)}.",
            "details": "Install them and re-run. (e.g. `brew install kind`)",
        }
    return {
        "step_name": "Preflight tool check",
        "status": "SUCCESS",
        "message": f"All required tools present: {', '.join(required)}.",
        "details": "",
    }


def check_tools_for(mode_or_provider: str) -> dict:
    return check_tools(TOOL_REQUIREMENTS.get(mode_or_provider, ["kubectl", "helm"]))


def install_kuberay(namespace: str = "default") -> dict:
    """Install/upgrade the KubeRay operator (delegates to ray_manager)."""
    return ray_manager.install_kuberay_operator(namespace=namespace)


if __name__ == "__main__":  # pragma: no cover
    print(check_tools_for("local"))
