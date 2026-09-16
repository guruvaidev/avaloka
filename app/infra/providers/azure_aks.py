# app/infra/providers/azure_aks.py
"""AKS (Azure Kubernetes Service) cluster provider.

Mirrors ``gcp_gke.GcpGkeProvider``'s shape so the provider abstraction covers
Azure (R7). Uses the ``az`` CLI. Config comes from the environment:

    AZURE_RESOURCE_GROUP   resource group holding the cluster (required)
    AZURE_CLUSTER_NAME     AKS cluster name (default: avaloka-aks)
    AZURE_LOCATION         region for creation (default: eastus)
    AZURE_NODE_COUNT       node count on create (default: 1)
    AZURE_ACR_NAME         optional ACR registry for load_image() pushes
"""
from __future__ import annotations

import os
import subprocess
import time

from app.infra.providers.base import ClusterProvider, run_command


class AzureAksProvider(ClusterProvider):
    name = "azure"

    def __init__(self):
        self.resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
        self.cluster_name = os.environ.get("AZURE_CLUSTER_NAME", "avaloka-aks")
        self.location = os.environ.get("AZURE_LOCATION", "eastus")
        self.node_count = os.environ.get("AZURE_NODE_COUNT", "1")
        self.acr_name = os.environ.get("AZURE_ACR_NAME", "")

    def provision_cluster(self) -> dict:
        if not self.resource_group:
            return {
                "step_name": "Provision azure Cluster",
                "status": "FAILED",
                "message": "AZURE_RESOURCE_GROUP environment variable not set.",
                "details": "",
            }
        check_cmd = [
            "az", "aks", "show", "--name", self.cluster_name,
            "--resource-group", self.resource_group,
        ]
        check_result = subprocess.run(check_cmd, capture_output=True, text=True)
        if check_result.returncode == 0:
            return {
                "step_name": "Provision azure Cluster",
                "status": "SKIPPED",
                "message": f"AKS cluster {self.cluster_name} already exists.",
                "details": check_result.stdout.strip(),
            }
        create_cmd = [
            "az", "aks", "create", "--name", self.cluster_name,
            "--resource-group", self.resource_group, "--location", self.location,
            "--node-count", self.node_count, "--generate-ssh-keys",
        ]
        if self.acr_name:
            create_cmd += ["--attach-acr", self.acr_name]
        return run_command(create_cmd, "Create azure Cluster")

    def configure_kubectl(self) -> dict:
        if not self.resource_group:
            return {
                "step_name": "Configure Kubectl",
                "status": "FAILED",
                "message": "AZURE_RESOURCE_GROUP environment variable not set.",
                "details": "",
            }
        cmd = [
            "az", "aks", "get-credentials", "--name", self.cluster_name,
            "--resource-group", self.resource_group, "--overwrite-existing",
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

    def load_image(self, image: str) -> dict:
        """Cloud nodes pull from a registry rather than side-loading. When an ACR is
        configured, push the local image to it; otherwise no-op (SKIPPED)."""
        if not self.acr_name:
            return {
                "step_name": "Load Image (azure)",
                "status": "SKIPPED",
                "message": "No AZURE_ACR_NAME set; cloud nodes pull from a registry. Nothing to side-load.",
                "details": "",
            }
        target = f"{self.acr_name}.azurecr.io/{image.split('/')[-1]}"
        tag = run_command(["docker", "tag", image, target], "Tag image for ACR")
        if tag["status"] != "SUCCESS":
            return tag
        login = run_command(["az", "acr", "login", "--name", self.acr_name], "ACR login")
        if login["status"] != "SUCCESS":
            return login
        return run_command(["docker", "push", target], "Push image to ACR")

    def teardown(self) -> dict:
        if not self.resource_group:
            return {
                "step_name": "Cleanup azure Cluster",
                "status": "SKIPPED",
                "message": "AZURE_RESOURCE_GROUP environment variable not set, skipping cleanup.",
                "details": "",
            }
        check_cmd = [
            "az", "aks", "show", "--name", self.cluster_name,
            "--resource-group", self.resource_group,
        ]
        if subprocess.run(check_cmd, capture_output=True, text=True).returncode != 0:
            return {
                "step_name": "Cleanup azure Cluster",
                "status": "SKIPPED",
                "message": f"AKS cluster {self.cluster_name} does not exist, skipping deletion.",
                "details": "",
            }
        cmd = [
            "az", "aks", "delete", "--name", self.cluster_name,
            "--resource-group", self.resource_group, "--yes",
        ]
        return run_command(cmd, "Delete azure Cluster")
