# app/infra/providers/gcp_gke.py
"""GKE (Google Kubernetes Engine) cluster provider.

Refactored out of ``k8s_invoker`` so the GKE flow is reusable by the new
provider abstraction while keeping identical behavior.
"""
from __future__ import annotations

import os
import subprocess
import time

from app.infra.providers.base import ClusterProvider, run_command


class GcpGkeProvider(ClusterProvider):
    name = "gcp"

    def __init__(self):
        self.project_id = os.environ.get("GCP_PROJECT_ID")
        self.cluster_name = os.environ.get("GCP_CLUSTER_NAME", "test-gke-cluster")
        self.zone = os.environ.get("GCP_ZONE", "us-central1-c")

    def provision_cluster(self) -> dict:
        if not self.project_id:
            return {
                "step_name": "Provision gcp Cluster",
                "status": "FAILED",
                "message": "GCP_PROJECT_ID environment variable not set.",
                "details": "",
            }
        check_cmd = [
            "gcloud", "container", "clusters", "describe", self.cluster_name,
            "--project", self.project_id, "--zone", self.zone,
        ]
        check_result = subprocess.run(check_cmd, capture_output=True, text=True)
        if check_result.returncode == 0:
            return {
                "step_name": "Provision gcp Cluster",
                "status": "SKIPPED",
                "message": f"GKE cluster {self.cluster_name} already exists.",
                "details": check_result.stdout.strip(),
            }
        create_cmd = [
            "gcloud", "container", "clusters", "create", self.cluster_name,
            "--project", self.project_id, "--zone", self.zone, "--num-nodes", "1",
            # Required for the chart's annotated Kubernetes service account to
            # obtain Application Default Credentials inside API, MLflow and Ray.
            "--workload-pool", f"{self.project_id}.svc.id.goog",
        ]
        return run_command(create_cmd, "Create gcp Cluster")

    def configure_kubectl(self) -> dict:
        cmd = [
            "gcloud", "container", "clusters", "get-credentials", self.cluster_name,
            "--project", self.project_id, "--zone", self.zone,
        ]
        outcome = run_command(cmd, "Configure Kubectl")
        if outcome["status"] == "SUCCESS":
            for _ in range(60):  # 5 minute timeout
                try:
                    subprocess.run(["kubectl", "get", "nodes"], check=True, capture_output=True)
                    outcome["details"] += "\nkubectl connected to cluster."
                    break
                except subprocess.CalledProcessError:
                    time.sleep(5)
            else:
                outcome["status"] = "FAILED"
                outcome["message"] = "kubectl could not connect to the cluster after multiple retries."
                outcome["details"] = "Timed out waiting for kubectl to connect."
        return outcome

    def teardown(self) -> dict:
        if not self.project_id:
            return {
                "step_name": "Cleanup gcp Cluster",
                "status": "SKIPPED",
                "message": "GCP_PROJECT_ID environment variable not set, skipping cleanup.",
                "details": "",
            }
        check_cmd = [
            "gcloud", "container", "clusters", "describe", self.cluster_name,
            "--project", self.project_id, "--zone", self.zone,
        ]
        if subprocess.run(check_cmd, capture_output=True, text=True).returncode != 0:
            return {
                "step_name": "Cleanup gcp Cluster",
                "status": "SKIPPED",
                "message": f"GKE cluster {self.cluster_name} does not exist, skipping deletion.",
                "details": "",
            }
        cmd = [
            "gcloud", "container", "clusters", "delete", self.cluster_name,
            "--project", self.project_id, "--zone", self.zone, "--quiet",
        ]
        return run_command(cmd, "Delete gcp Cluster")
