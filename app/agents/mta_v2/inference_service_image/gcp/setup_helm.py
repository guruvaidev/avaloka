"""
setup_helm.py
-------------
Install the Helm releases required for a fully autoscaling Ray cluster on GKE.

GKE Autopilot vs Standard
--------------------------
* **Autopilot** (this project's default) — GKE manages node provisioning
  automatically; you do NOT install cluster-autoscaler yourself.
  Only metrics-server is installed here.

* **Standard** — call `install_cluster_autoscaler()` explicitly after this
  module's `setup_helm()`, passing your node-pool and project details.

Helm releases installed by `setup_helm()`:
  1. metrics-server  (namespace: kube-system)
     Required for `kubectl top` and resource-based autoscaling decisions.
     The KubeRay autoscaler sidecar uses resource metrics to decide when to
     add / remove worker pods.
"""

import logging
import subprocess
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Helm chart coordinates
# ---------------------------------------------------------------------------
METRICS_SERVER_REPO = "https://kubernetes-sigs.github.io/metrics-server/"
METRICS_SERVER_CHART = "metrics-server/metrics-server"
METRICS_SERVER_RELEASE = "metrics-server"
METRICS_SERVER_VERSION = "3.12.1"   # latest stable as of Ray 2.10 era

CLUSTER_AUTOSCALER_REPO = "https://kubernetes.github.io/autoscaler"
CLUSTER_AUTOSCALER_CHART = "autoscaler/cluster-autoscaler"
CLUSTER_AUTOSCALER_RELEASE = "cluster-autoscaler"
CLUSTER_AUTOSCALER_VERSION = "9.37.0"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_helm(
    kubeconfig: Optional[str] = None,
    install_metrics_server: bool = True,
    namespace: str = "kube-system",
) -> None:
    """
    Add required Helm repos and install metrics-server.

    Args:
        kubeconfig: Optional path to a kubeconfig file.  When None the default
                    kubeconfig / in-cluster config is used.
        install_metrics_server: Set False to skip metrics-server (e.g. if your
                                 GKE version already bundles it).
    """
    env = _build_env(kubeconfig)

    _helm_repo_add("metrics-server", METRICS_SERVER_REPO, env)
    _helm_repo_update(env)

    if install_metrics_server:
        _install_or_upgrade(
            release=METRICS_SERVER_RELEASE,
            chart=METRICS_SERVER_CHART,
            namespace=namespace,
            version=METRICS_SERVER_VERSION,
            values={
                # Required on GKE Autopilot — kubelet serving certs are not
                # signed by the cluster CA, so skip TLS verification.
                "args[0]": "--kubelet-insecure-tls",
            },
            env=env,
        )
        _wait_for_deployment(METRICS_SERVER_RELEASE, namespace, env)


def install_cluster_autoscaler(
    cluster_name: str,
    project_id: str,
    location: str,
    node_group: str,
    min_nodes: int = 1,
    max_nodes: int = 110,
    namespace: str = "kube-system",
    kubeconfig: Optional[str] = None,
) -> None:
    """
    Install the Kubernetes cluster-autoscaler for GKE Standard clusters.

    **Do NOT call this for GKE Autopilot** — Autopilot manages nodes itself.

    Args:
        cluster_name:  GKE cluster name.
        project_id:    GCP project ID.
        location:      GKE cluster location (region or zone).
        node_group:    Node pool resource path, e.g.
                       "projects/my-proj/locations/us-central1/clusters/my-cluster/nodePools/default-pool"
        min_nodes:     Minimum number of nodes in the pool.
        max_nodes:     Maximum number of nodes (set ≥ max_worker_replicas + 10
                       for system overhead).
        kubeconfig:    Optional path to kubeconfig file.
    """
    env = _build_env(kubeconfig)

    _helm_repo_add("autoscaler", CLUSTER_AUTOSCALER_REPO, env)
    _helm_repo_update(env)

    _install_or_upgrade(
        release=CLUSTER_AUTOSCALER_RELEASE,
        chart=CLUSTER_AUTOSCALER_CHART,
        namespace=namespace,
        version=CLUSTER_AUTOSCALER_VERSION,
        values={
            "autoDiscovery.clusterName": cluster_name,
            "cloudProvider": "gce",
            "cloudConfigPath": "",          # use workload-identity, no key file
            "rbac.serviceAccount.annotations.iam\\.gke\\.io/gcp-service-account":
                f"cluster-autoscaler@{project_id}.iam.gserviceaccount.com",
            # Node group limits — must accommodate max Ray worker pods
            f"nodeGroups[0].name": node_group,
            f"nodeGroups[0].minSize": str(min_nodes),
            f"nodeGroups[0].maxSize": str(max_nodes),
            # Scale-down settings aligned with Ray's idleTimeoutSeconds (60 s)
            "extraArgs.scale-down-delay-after-add": "1m",
            "extraArgs.scale-down-unneeded-time": "1m",
            "extraArgs.scale-down-utilization-threshold": "0.5",
            "extraArgs.skip-nodes-with-system-pods": "false",
        },
        env=env,
    )
    _wait_for_deployment(CLUSTER_AUTOSCALER_RELEASE, namespace, env)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_env(kubeconfig: Optional[str]) -> dict:
    """Build subprocess env, injecting KUBECONFIG if provided."""
    import os
    env = os.environ.copy()
    if kubeconfig:
        env["KUBECONFIG"] = kubeconfig
    return env


def _helm_repo_add(name: str, url: str, env: dict) -> None:
    logging.info("Adding Helm repo '%s' → %s", name, url)
    _run(["helm", "repo", "add", name, url, "--force-update"], env)


def _helm_repo_update(env: dict) -> None:
    logging.info("Updating Helm repos …")
    _run(["helm", "repo", "update"], env)


def _install_or_upgrade(
    release: str,
    chart: str,
    namespace: str,
    version: str,
    values: dict,
    env: dict,
) -> None:
    cmd = [
        "helm", "upgrade", "--install", release, chart,
        "--namespace", namespace,
        "--create-namespace",
        "--version", version,
        "--wait",
        "--timeout", "5m",
    ]
    for k, v in values.items():
        cmd += ["--set", f"{k}={v}"]

    logging.info("helm upgrade --install %s (chart=%s, version=%s)", release, chart, version)
    _run(cmd, env)
    logging.info("Helm release '%s' installed/upgraded successfully.", release)


def _wait_for_deployment(name: str, namespace: str, env: dict, timeout: int = 120) -> None:
    """Poll until the named deployment is Available, or raise TimeoutError."""
    logging.info("Waiting for deployment '%s/%s' to become available …", namespace, name)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = subprocess.run(
                [
                    "kubectl", "rollout", "status",
                    f"deployment/{name}",
                    "-n", namespace,
                    "--timeout=10s",
                ],
                capture_output=True, text=True, env=env,
            )
            if result.returncode == 0:
                logging.info("Deployment '%s' is available.", name)
                return
        except FileNotFoundError:
            raise RuntimeError("kubectl not found — ensure it is installed and on PATH.")
        time.sleep(5)
    raise TimeoutError(
        f"Deployment '{name}' in namespace '{namespace}' did not become available "
        f"within {timeout} seconds."
    )


def _run(cmd: list, env: dict) -> None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(cmd)}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        if result.stdout.strip():
            logging.debug(result.stdout.strip())
    except FileNotFoundError:
        raise RuntimeError(
            f"'{cmd[0]}' not found — ensure Helm is installed and on PATH. "
            "Install: https://helm.sh/docs/intro/install/"
        )
