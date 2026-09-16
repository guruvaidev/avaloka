# app/infra/providers/local_kind.py
"""Local Kubernetes provider backed by `kind` (Kubernetes-in-Docker).

Makes avaloka deployable end-to-end on any laptop/CI with Docker, no cloud cost.
"""
from __future__ import annotations

import os
import subprocess
import time

from app.infra.providers.base import ClusterProvider, run_command

# Repo-root-relative path to the default kind cluster config.
_KIND_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "deploy",
    "clusters",
    "kind-cluster.yaml",
)


class LocalKindProvider(ClusterProvider):
    name = "local"

    def __init__(self, cluster_name: str | None = None, config_path: str | None = None):
        self.cluster_name = cluster_name or os.environ.get("KIND_CLUSTER_NAME", "avaloka")
        self.config_path = config_path or os.environ.get("KIND_CONFIG", _KIND_CONFIG)

    def _exists(self) -> bool:
        result = subprocess.run(["kind", "get", "clusters"], capture_output=True, text=True)
        if result.returncode != 0:
            return False
        return self.cluster_name in result.stdout.split()

    def provision_cluster(self) -> dict:
        if self._exists():
            return {
                "step_name": "Provision local Cluster",
                "status": "SKIPPED",
                "message": f"kind cluster '{self.cluster_name}' already exists.",
                "details": "",
            }
        cmd = ["kind", "create", "cluster", "--name", self.cluster_name, "--wait", "120s"]
        if self.config_path and os.path.exists(self.config_path):
            cmd += ["--config", self.config_path]
        return run_command(cmd, "Create local Cluster")

    def configure_kubectl(self) -> dict:
        # `kind create` already sets the kubeconfig context; make it explicit + verify.
        outcome = run_command(
            ["kubectl", "config", "use-context", f"kind-{self.cluster_name}"],
            "Configure Kubectl",
        )
        if outcome["status"] != "SUCCESS":
            return outcome
        for _ in range(24):  # ~2 minutes
            try:
                subprocess.run(["kubectl", "get", "nodes"], check=True, capture_output=True)
                outcome["details"] += "\nkubectl connected to cluster."
                return outcome
            except subprocess.CalledProcessError:
                time.sleep(5)
        outcome["status"] = "FAILED"
        outcome["message"] = "kubectl could not connect to the kind cluster."
        return outcome

    def load_image(self, image: str) -> dict:
        """Load a locally-built image into the kind cluster's node(s)."""
        return run_command(
            ["kind", "load", "docker-image", image, "--name", self.cluster_name],
            f"Load image {image} into kind",
        )

    def teardown(self) -> dict:
        if not self._exists():
            return {
                "step_name": "Cleanup local Cluster",
                "status": "SKIPPED",
                "message": f"kind cluster '{self.cluster_name}' does not exist.",
                "details": "",
            }
        return run_command(
            ["kind", "delete", "cluster", "--name", self.cluster_name],
            "Delete local Cluster",
        )
