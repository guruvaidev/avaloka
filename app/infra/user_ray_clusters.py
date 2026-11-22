"""Per-user Ray clusters (Professional plan on the hosted service).

Each user it applies to gets one RayCluster of their own, created on demand
and deleted after it has been idle for a while. Nothing is stored outside
Kubernetes: the RayCluster object itself is the record.

  name         ray-u-<first 12 hex of sha256(user_id)>  (stable, DNS-safe,
               and the raw user id never appears in a resource name)
  namespace    RAY_USER_CLUSTER_NAMESPACE (default ray-users); created by
               Helm, not here, so the app needs no cluster-wide rights
  labels       avaloka.ai/user-cluster=true, avaloka.ai/user-hash=<hash>
  annotations  avaloka.ai/user-id, avaloka.ai/last-active (UTC ISO time)
  dashboard    <name>-head-svc.<namespace>.svc.cluster.local:8265, a
               ClusterIP service: Ray's dashboard has no login and its Jobs
               API runs any code, so it is never given a LoadBalancer
               (mta_v2's create_ray_cluster uses one, which is why its
               manifest is not reused here)

Who gets one: RAY_PER_USER_CLUSTERS=1 and the user's plan is in
RAY_PER_USER_CLUSTER_PLANS (default "professional"; "*" = everyone, for local
testing). Everyone else, and every self-hosted / Enterprise deployment that
leaves the switch off, keeps using the shared cluster exactly as before.

Activity that keeps a cluster alive (last-active is moved to now):
  * a job submitted to it (ray_job_runner),
  * the /ray/ dashboard open and in use (POST /api/ray/heartbeat),
  * the reaper finding a job still PENDING or RUNNING on it.
reap_idle() (Celery beat, every RAY_USER_CLUSTER_REAP_EVERY_S) deletes a
cluster idle longer than RAY_USER_CLUSTER_IDLE_MINUTES (default 60).

All functions block (Kubernetes API, HTTP to the dashboard).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GROUP, VERSION, PLURAL = "ray.io", "v1", "rayclusters"
LABEL_MANAGED = "avaloka.ai/user-cluster"
LABEL_USER_HASH = "avaloka.ai/user-hash"
ANN_USER_ID = "avaloka.ai/user-id"
ANN_LAST_ACTIVE = "avaloka.ai/last-active"
DASHBOARD_PORT = 8265
ACTIVE_JOB_STATES = {"PENDING", "RUNNING"}

# Swapped in tests.
_sleep = time.sleep
_monotonic = time.monotonic
_API = None
_NET_API = None


# ------------------------------------------------------------------ settings

def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes")


def enabled() -> bool:
    return _truthy(os.getenv("RAY_PER_USER_CLUSTERS"))


def namespace() -> str:
    return (os.getenv("RAY_USER_CLUSTER_NAMESPACE") or "ray-users").strip() or "ray-users"


def idle_minutes() -> int:
    return max(1, int(os.getenv("RAY_USER_CLUSTER_IDLE_MINUTES", "60")))


def _plans() -> set:
    raw = os.getenv("RAY_PER_USER_CLUSTER_PLANS", "professional")
    return {p.strip().lower() for p in raw.split(",") if p.strip()}


# ------------------------------------------------------------------- naming

def user_hash(user_id: str) -> str:
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:12]


def cluster_name(user_id: str) -> str:
    return f"ray-u-{user_hash(user_id)}"


def upstream_for(name: str) -> str:
    """host:port of the cluster's dashboard, as the /ray/ proxy reaches it."""
    return f"{name}-head-svc.{namespace()}.svc.cluster.local:{DASHBOARD_PORT}"


def dashboard_url_for(name: str) -> str:
    return f"http://{upstream_for(name)}"


def applies_to(user_id: Optional[str]) -> bool:
    """Does this user get a cluster of their own?"""
    if not enabled() or not user_id:
        return False
    plans = _plans()
    if "*" in plans:
        return True
    try:
        from app.core import plan_limits  # lazy: keeps Supabase out of import time
        plan = plan_limits.try_resolve_plan(user_id)
    except Exception as exc:
        logger.warning("[user-ray] plan lookup failed; using the shared cluster: %s", exc)
        plan = None
    return bool(plan) and str(plan).strip().lower() in plans


# --------------------------------------------------------------- kubernetes

def _custom():
    global _API
    if _API is None:
        from kubernetes import client, config
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        _API = client.CustomObjectsApi()
    return _API


def _networking():
    global _NET_API
    if _NET_API is None:
        _custom()  # loads the Kubernetes config
        from kubernetes import client
        _NET_API = client.NetworkingV1Api()
    return _NET_API


def _http_status(exc: Exception) -> Optional[int]:
    return getattr(exc, "status", None)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _get(name: str) -> Optional[Dict[str, Any]]:
    try:
        return _custom().get_namespaced_custom_object(
            group=GROUP, version=VERSION, namespace=namespace(), plural=PLURAL, name=name,
        )
    except Exception as exc:
        if _http_status(exc) == 404:
            return None
        raise


def _state(obj: Optional[Dict[str, Any]]) -> str:
    """stopped | starting | running | stopping"""
    if not obj:
        return "stopped"
    if (obj.get("metadata") or {}).get("deletionTimestamp"):
        return "stopping"
    state = str(((obj.get("status") or {}).get("state")) or "").lower()
    return "running" if state == "ready" else "starting"


# ---------------------------------------------------------------- manifest

_VERSION_RE = re.compile(r":(\d+\.\d+\.\d+)")


def _image() -> str:
    return (os.getenv("RAY_USER_CLUSTER_IMAGE") or "rayproject/ray:2.10.0").strip()


def _ray_version(image: str) -> str:
    explicit = (os.getenv("RAY_USER_CLUSTER_RAY_VERSION") or "").strip()
    if explicit:
        return explicit
    match = _VERSION_RE.search(image)
    return match.group(1) if match else "2.10.0"


def _logical_cpus(cpu: str) -> str:
    """Ray's --num-cpus takes a whole number; Kubernetes takes "500m" too.
    Round a Kubernetes CPU value up to a whole number of logical CPUs."""
    import math
    text = str(cpu).strip()
    try:
        value = float(text[:-1]) / 1000 if text.endswith("m") else float(text)
    except ValueError:
        return "1"
    return str(max(1, math.ceil(value)))


def _resources(cpu: str, memory: str) -> Dict[str, Dict[str, str]]:
    return {"requests": {"cpu": cpu, "memory": memory}, "limits": {"cpu": cpu, "memory": memory}}


def build_manifest(user_id: str, now: Optional[datetime] = None) -> Dict[str, Any]:
    """The RayCluster for one user: a head plus 0..N autoscaled workers.

    Defaults give 1 to 3 nodes (head + up to 2 workers). Every size is a
    setting, so a local kind cluster can run it smaller.
    """
    now = now or _now()
    name = cluster_name(user_id)
    image = _image()
    head_cpu = os.getenv("RAY_USER_CLUSTER_HEAD_CPU", "2")
    head_mem = os.getenv("RAY_USER_CLUSTER_HEAD_MEMORY", "8Gi")
    worker_cpu = os.getenv("RAY_USER_CLUSTER_WORKER_CPU", "4")
    worker_mem = os.getenv("RAY_USER_CLUSTER_WORKER_MEMORY", "16Gi")
    min_workers = max(0, int(os.getenv("RAY_USER_CLUSTER_MIN_WORKERS", "0")))
    max_workers = max(min_workers, int(os.getenv("RAY_USER_CLUSTER_MAX_WORKERS", "2")))
    service_account = (os.getenv("RAY_USER_CLUSTER_SERVICE_ACCOUNT") or "").strip()
    # A locally built image loaded into kind (avaloka-ray:latest) is in no
    # registry: ":latest" defaults to Always and would fail to pull. Set
    # IfNotPresent (or Never) for it, the same as the shared cluster.
    pull_policy = (os.getenv("RAY_USER_CLUSTER_IMAGE_PULL_POLICY") or "").strip()

    def pod(container: Dict[str, Any]) -> Dict[str, Any]:
        if pull_policy:
            container["imagePullPolicy"] = pull_policy
        spec: Dict[str, Any] = {"containers": [container]}
        if service_account:
            spec["serviceAccountName"] = service_account
        return {"metadata": {"labels": {LABEL_MANAGED: "true", LABEL_USER_HASH: user_hash(user_id)}},
                "spec": spec}

    head = {
        "serviceType": "ClusterIP",
        "rayStartParams": {"dashboard-host": "0.0.0.0", "num-cpus": _logical_cpus(head_cpu)},
        "template": pod({
            "name": "ray-head",
            "image": image,
            "ports": [
                {"containerPort": 6379, "name": "gcs"},
                {"containerPort": DASHBOARD_PORT, "name": "dashboard"},
                {"containerPort": 10001, "name": "client"},
            ],
            "resources": _resources(head_cpu, head_mem),
        }),
    }
    workers = {
        "groupName": "workers",
        "replicas": min_workers,
        "minReplicas": min_workers,
        "maxReplicas": max_workers,
        "rayStartParams": {},
        "template": pod({
            "name": "ray-worker",
            "image": image,
            "resources": _resources(worker_cpu, worker_mem),
        }),
    }
    return {
        "apiVersion": f"{GROUP}/{VERSION}",
        "kind": "RayCluster",
        "metadata": {
            "name": name,
            "namespace": namespace(),
            "labels": {
                LABEL_MANAGED: "true",
                LABEL_USER_HASH: user_hash(user_id),
                "app.kubernetes.io/managed-by": "avaloka",
            },
            "annotations": {ANN_USER_ID: str(user_id), ANN_LAST_ACTIVE: _iso(now)},
        },
        "spec": {
            "rayVersion": _ray_version(image),
            "enableInTreeAutoscaling": True,
            "autoscalerOptions": {
                "upscalingMode": "Default",
                # Workers (not the cluster) scale down after this long unused.
                "idleTimeoutSeconds": int(os.getenv("RAY_USER_CLUSTER_WORKER_IDLE_S", "600")),
            },
            "headGroupSpec": head,
            "workerGroupSpecs": [workers],
        },
    }


# ------------------------------------------------------------ public calls

def describe(user_id: str) -> Dict[str, str]:
    name = cluster_name(user_id)
    return {"name": name, "state": _state(_get(name)), "upstream": upstream_for(name)}


# ------------------------------------------------------- network isolation
#
# deploy/helm/ray/user_ray_clusters.yaml denies all ingress in the namespace
# and lets the platform (backend, Celery, /ray proxy, KubeRay) reach each
# head's dashboard port. That alone would also cut a cluster's head off from
# its own workers, so every cluster gets one more policy: its pods may talk to
# each other, and to nothing else in the namespace. The policy is owned by the
# RayCluster, so Kubernetes deletes it together with the cluster.

def network_policy_enabled() -> bool:
    return (os.getenv("RAY_USER_CLUSTER_NETWORK_POLICY", "1").strip().lower()
            not in ("0", "false", "no"))


def build_network_policy(user_id: str, owner_uid: str) -> Dict[str, Any]:
    name = cluster_name(user_id)
    pods = {LABEL_USER_HASH: user_hash(user_id)}
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": f"{name}-same-cluster",
            "namespace": namespace(),
            "labels": {LABEL_MANAGED: "true", LABEL_USER_HASH: user_hash(user_id),
                       "app.kubernetes.io/managed-by": "avaloka"},
            "ownerReferences": [{
                "apiVersion": f"{GROUP}/{VERSION}", "kind": "RayCluster",
                "name": name, "uid": owner_uid,
            }],
        },
        "spec": {
            "podSelector": {"matchLabels": dict(pods)},
            "policyTypes": ["Ingress"],
            "ingress": [{"from": [{"podSelector": {"matchLabels": dict(pods)}}]}],
        },
    }


def _ensure_network_policy(user_id: str) -> None:
    """Create the cluster's same-cluster policy if it is missing.

    A failure is raised, not logged: under the namespace's default-deny, a
    cluster without this policy cannot start its workers anyway.
    """
    if not network_policy_enabled():
        return
    obj = _get(cluster_name(user_id))
    uid = ((obj or {}).get("metadata") or {}).get("uid")
    if not uid:
        return  # the cluster is gone; nothing to protect
    try:
        _networking().create_namespaced_network_policy(
            namespace=namespace(), body=build_network_policy(user_id, uid),
        )
    except Exception as exc:
        if _http_status(exc) != 409:  # 409: already there
            raise


def start(user_id: str) -> Dict[str, str]:
    """Create the user's cluster; returns at once (state 'starting')."""
    name = cluster_name(user_id)
    try:
        _custom().create_namespaced_custom_object(
            group=GROUP, version=VERSION, namespace=namespace(), plural=PLURAL,
            body=build_manifest(user_id),
        )
        logger.info("[user-ray] creating %s in %s", name, namespace())
    except Exception as exc:
        if _http_status(exc) != 409:  # 409: it exists already, which is fine
            raise
        touch(user_id)
    # Also on 409: a cluster created before this policy existed gets it now.
    _ensure_network_policy(user_id)
    return describe(user_id)


def _touch_name(name: str, now: Optional[datetime] = None) -> bool:
    body = {"metadata": {"annotations": {ANN_LAST_ACTIVE: _iso(now or _now())}}}
    try:
        _custom().patch_namespaced_custom_object(
            group=GROUP, version=VERSION, namespace=namespace(), plural=PLURAL, name=name, body=body,
        )
        return True
    except Exception as exc:
        if _http_status(exc) == 404:
            return False
        raise


def touch(user_id: str) -> bool:
    """Mark the user's cluster as in use now. False when it does not exist."""
    return _touch_name(cluster_name(user_id))


def _delete_name(name: str) -> None:
    try:
        _custom().delete_namespaced_custom_object(
            group=GROUP, version=VERSION, namespace=namespace(), plural=PLURAL, name=name,
            body={"propagationPolicy": "Background"},
        )
        logger.info("[user-ray] deleted idle cluster %s", name)
    except Exception as exc:
        if _http_status(exc) != 404:
            raise


def _dashboard_answers(url: str) -> bool:
    timeout = float(os.getenv("RAY_DASHBOARD_CONNECT_TIMEOUT_S", "5"))
    try:
        with urllib.request.urlopen(f"{url}/api/version", timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def ensure_running(user_id: str, timeout_s: Optional[float] = None,
                   poll_s: float = 10.0) -> Optional[str]:
    """Start the user's cluster if needed and wait until its dashboard answers.

    Returns the dashboard URL, or None if it is not ready within ``timeout_s``
    (RAY_USER_CLUSTER_START_TIMEOUT_S, default 600). Counts as activity.
    """
    if timeout_s is None:
        timeout_s = float(os.getenv("RAY_USER_CLUSTER_START_TIMEOUT_S", "600"))
    name = cluster_name(user_id)
    url = dashboard_url_for(name)
    deadline = _monotonic() + timeout_s
    started = False
    while True:
        state = _state(_get(name))
        if state == "stopped":
            start(user_id)
            started = True
        elif state == "running" and _dashboard_answers(url):
            if not started:
                _touch_name(name)
            return url
        if _monotonic() >= deadline:
            logger.warning("[user-ray] %s not ready after %.0fs (state=%s)", name, timeout_s, state)
            return None
        _sleep(poll_s)


def _active_jobs(url: str) -> Optional[int]:
    """Jobs PENDING or RUNNING on the cluster; None if it cannot be asked."""
    timeout = float(os.getenv("RAY_DASHBOARD_CONNECT_TIMEOUT_S", "5"))
    try:
        with urllib.request.urlopen(f"{url}/api/jobs/", timeout=timeout) as resp:
            jobs = json.loads(resp.read().decode() or "[]")
    except Exception:
        return None
    if not isinstance(jobs, list):
        return None
    return sum(1 for j in jobs if str((j or {}).get("status") or "").upper() in ACTIVE_JOB_STATES)


def reap_idle(now: Optional[datetime] = None) -> Dict[str, Any]:
    """Delete user clusters idle longer than the limit and running no job.

    * idle is measured from avaloka.ai/last-active (else the creation time);
    * a cluster with a PENDING/RUNNING job is kept and marked active;
    * a Ready cluster whose dashboard does not answer is kept until twice
      the limit, then deleted (a broken cluster must not bill forever);
    * one that never became Ready is deleted at the limit.
    """
    summary: Dict[str, Any] = {"deleted": [], "busy": [], "unreachable": [], "kept": 0}
    if not enabled():
        summary["enabled"] = False
        return summary
    now = now or _now()
    limit_s = idle_minutes() * 60
    items: List[Dict[str, Any]] = (_custom().list_namespaced_custom_object(
        group=GROUP, version=VERSION, namespace=namespace(), plural=PLURAL,
        label_selector=f"{LABEL_MANAGED}=true",
    ) or {}).get("items") or []

    for obj in items:
        meta = obj.get("metadata") or {}
        name = meta.get("name")
        if not name or meta.get("deletionTimestamp"):
            continue
        last = _parse((meta.get("annotations") or {}).get(ANN_LAST_ACTIVE)) \
            or _parse(meta.get("creationTimestamp")) or now
        idle_s = (now - last).total_seconds()
        if idle_s < limit_s:
            summary["kept"] += 1
            continue
        state = _state(obj)
        active = _active_jobs(dashboard_url_for(name)) if state == "running" else 0
        if active:
            _touch_name(name, now)
            summary["busy"].append(name)
            continue
        if active is None and idle_s < 2 * limit_s:
            summary["unreachable"].append(name)
            continue
        _delete_name(name)
        summary["deleted"].append(name)

    if summary["deleted"] or summary["busy"] or summary["unreachable"]:
        logger.info("[user-ray] reaper: %s", summary)
    return summary
