"""Training as a KubeRay RayJob on the shared cluster (submit-only).

The trainer used to call the Ray dashboard's job API and then block the Celery
worker for up to twelve hours waiting for the result. Here a training run is a
RayJob custom resource with ``clusterSelector`` pointing at the one shared
RayCluster: no cluster per job, no dashboard address, no public IP, and nothing
waits -- ``app/infra/training_tasks.poll_training_job`` reads the CR's status
on a countdown chain and collects the result from the submitter's logs.

Nothing secret is rendered into the CR: the shared Ray pods already carry the
named platform keys (deploy/helm/ray/raycluster.yaml); the job's env holds URIs
and hyperparameters only. No cloud SDK; kubectl through the existing runner.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import yaml

SHARED_RAY_CLUSTER = os.getenv("RAY_CLUSTER_NAME", "avaloka-raycluster")
RAY_NAMESPACE = os.getenv("MTA_RAY_NAMESPACE", os.getenv("RAY_NAMESPACE", "default"))
TTL_AFTER_FINISHED_S = int(os.getenv("RAY_TRAINING_JOB_TTL_S", "600"))
RESULT_BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "TERMINATED"}
#: Keys a job env may carry into the CR. Anything else (a credential by mistake) is dropped.
_FORBIDDEN_ENV = re.compile(r"(SECRET|PASSWORD|TOKEN|CREDENTIALS|_KEY$|SERVICE_ACCOUNT_JSON)", re.I)


def job_name(task_id: str) -> str:
    return f"avaloka-train-{re.sub(r'[^a-z0-9]', '', str(task_id).lower())[:12]}"


def scrub_env(env: Dict[str, str]) -> Dict[str, str]:
    """Only non-secret settings travel in the CR; secrets reach the pods from the cluster's own Secret refs."""
    return {k: str(v) for k, v in env.items() if not _FORBIDDEN_ENV.search(k) and v is not None}


def training_rayjob_manifest(name: str, *, entrypoint: str, env: Dict[str, str], job_id: Optional[str] = None,
                             namespace: str = RAY_NAMESPACE, cluster: str = SHARED_RAY_CLUSTER,
                             use_gpu: bool = False, ttl_s: int = TTL_AFTER_FINISHED_S) -> Dict[str, Any]:
    """A RayJob bound to the shared cluster; the cluster outlives the job."""
    spec: Dict[str, Any] = {
        "entrypoint": entrypoint,
        "clusterSelector": {"ray.io/cluster": cluster},
        # The cluster is shared: never shut it down with the job. The CR itself
        # is garbage-collected ttl_s after it finishes.
        "shutdownAfterJobFinishes": False,
        "ttlSecondsAfterFinished": int(ttl_s),
        "submissionMode": "K8sJobMode",
        "runtimeEnvYAML": yaml.safe_dump({"env_vars": scrub_env(env)}, sort_keys=True),
    }
    if use_gpu:
        spec["entrypointNumGpus"] = 1   # GPU only when the plan asked for it (R4)
    labels = {"app.kubernetes.io/part-of": "avaloka", "avaloka.ai/kind": "training"}
    if job_id:
        labels["avaloka.ai/ml-job"] = str(job_id)
    return {"apiVersion": "ray.io/v1", "kind": "RayJob",
            "metadata": {"name": name, "namespace": namespace, "labels": labels},
            "spec": spec}


def submit(manifest: Dict[str, Any], *, apply=None) -> str:
    """kubectl apply the CR; returns the job name."""
    if apply is None:
        from app.infra.ray_job_runner import apply_yaml_text
        apply = apply_yaml_text
    apply(yaml.safe_dump(manifest, sort_keys=False), namespace=manifest["metadata"]["namespace"])
    return manifest["metadata"]["name"]


def status(name: str, namespace: str = RAY_NAMESPACE, *, get=None) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """(jobStatus, jobDeploymentStatus, startTime, endTime) of the CR."""
    if get is None:
        from app.infra.ray_job_runner import get_rayjob_json
        get = get_rayjob_json
    from app.infra.ray_job_runner import _rayjob_status_fields
    return _rayjob_status_fields(get(name, namespace))


def is_terminal(job_status: Optional[str]) -> bool:
    return bool(job_status) and str(job_status).upper() in TERMINAL


def result_from_logs(logs: str) -> Optional[Dict[str, Any]]:
    """The training job prints its result as a ```json block (ray_job.py); the last one wins."""
    blocks = RESULT_BLOCK.findall(logs or "")
    for block in reversed(blocks):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def fetch_logs(name: str, namespace: str = RAY_NAMESPACE, *, fetch=None) -> str:
    if fetch is None:
        from app.infra.ray_job_runner import fetch_rayjob_logs
        fetch = fetch_rayjob_logs
    return fetch(name, namespace)


def delete(name: str, namespace: str = RAY_NAMESPACE, *, kubectl=None) -> bool:
    if kubectl is None:
        from app.infra.ray_job_runner import _kubectl
        kubectl = _kubectl
    rc, _, _ = kubectl("delete", "rayjob", name, "-n", namespace, "--ignore-not-found", timeout_s=60)
    return rc == 0


def backoff_s(attempt: int, first: int = 15, cap: int = 60) -> int:
    """15, 30, 60, 60, ... seconds between status polls."""
    return int(min(cap, first * (2 ** max(0, attempt))))
