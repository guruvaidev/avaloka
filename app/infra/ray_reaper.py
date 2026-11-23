"""Reapers: the infra layer's cost and hygiene loop (Celery beat).

Three periodic tasks, all through the Kubernetes API (no cloud SDK), all
dry-runnable with ``AVALOKA_REAPER_DRY_RUN=1``:

``reap_idle_model_server``  every 5 min: the model-server Deployment(s) go to
                            zero replicas after ``MODEL_SERVER_IDLE_TIMEOUT_S``
                            (2 h) without a prediction (``ml_endpoints.last_request_at``).
``reap_idle_ray_cluster``   every 5 min: the shared RayCluster is deleted after
                            ``RAY_IDLE_TEARDOWN_S`` (2 h) with no RayJob running or
                            pending; ``ensure_shared_cluster`` re-applies it on the
                            next training job. ``0`` disables the teardown.
``reap_orphan_ray``         hourly: RayClusters / RayJobs / Services / Namespaces
                            that the legacy per-job path left behind
                            (``ray-trainer-*``, ``inference-service-*``, or labelled
                            ``app.kubernetes.io/part-of: avaloka``) and that are
                            terminal or older than ``RAY_ORPHAN_MAX_AGE_H`` (24) with
                            no ``ml_jobs`` row in ``queued|running``. The 77-day
                            leftover the owner found would have lived one hour.

Every deletion is recorded as an ``ml_jobs(kind=reap)`` row naming the Kubernetes
object (names of our own objects, never customer data).
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DRY_RUN = os.getenv("AVALOKA_REAPER_DRY_RUN", "").strip().lower() in {"1", "true", "yes"}
#: The 2 h idle rule, shared with the model server (coordinator condition 3).
RAY_IDLE_TIMEOUT_S = int(os.getenv("RAY_IDLE_TIMEOUT_S", os.getenv("RAY_IDLE_TEARDOWN_S", "7200")))
RAY_IDLE_TEARDOWN_S = RAY_IDLE_TIMEOUT_S  # old name, kept for callers
#: What an idle shared cluster gets. "scale-workers" (default) patches every worker
#: group to 0 replicas and leaves the head: safe for every Ray consumer (sampling,
#: data transfer, execute_on_ray, the dashboard panel) that connects to the head
#: directly. "delete" removes the CR and is refused unless every consumer goes
#: through ensure_shared_cluster() -- see RAY_CONSUMERS_BYPASSING_ENSURE.
RAY_IDLE_ACTION = os.getenv("RAY_IDLE_ACTION", "scale-workers").strip().lower()

#: Ray call sites on develop-1.6 that reach the cluster WITHOUT ensure_shared_cluster().
#: Inventory of 2026-11-15, kept exact by test_ray_consumer_inventory_matches_the_tree
#: (real call sites: ray.init / RAY_ADDRESS reads / JobSubmissionClient / submit_ray_job /
#: execute_on_ray / ray.data / run_ray_job / apply_yaml_text). While this is non-empty,
#: deleting the idle cluster would break them; the reaper may only scale workers.
RAY_CONSUMERS_BYPASSING_ENSURE = (
    "app/agents/execution_agent.py",          # execution_agent_node_ray / execute_on_ray
    "app/agents/sampling_agent_daft.py",      # Daft / portfolio sampling on Ray
    "app/agents/sampling_async.py",           # background sampling task
    "app/infra/ray_job_runner.py",            # scheduled analyses, data-transfer RayJobs
    "app/infra/ray_jobs_service.py",          # the dashboard panel (#731)
    "app/infra/rayjob_submitter.py",
    "app/infra/run_ray_smoketest.py",
)


def idle_action_allowed(action: str, bypassing: tuple = RAY_CONSUMERS_BYPASSING_ENSURE) -> str:
    """The idle action the reaper may take: 'delete' only when nothing bypasses ensure."""
    action = (action or "scale-workers").strip().lower()
    if action == "delete" and bypassing:
        logger.warning("[reaper] RAY_IDLE_ACTION=delete refused: %d Ray consumer(s) bypass ensure_shared_cluster(); "
                       "scaling workers to 0 instead", len(bypassing))
        return "scale-workers"
    return action if action in {"delete", "scale-workers", "none"} else "scale-workers"
RAY_ORPHAN_MAX_AGE_H = float(os.getenv("RAY_ORPHAN_MAX_AGE_H", "24"))
SHARED_RAY_CLUSTER = os.getenv("RAY_CLUSTER_NAME", "avaloka-raycluster")
RAY_NAMESPACE = os.getenv("MTA_RAY_NAMESPACE", os.getenv("RAY_NAMESPACE", "default"))
ORPHAN_NAME_RE = re.compile(r"^(ray-trainer-|inference-service-)")
PART_OF = "app.kubernetes.io/part-of"
RAYJOB_ACTIVE = {"PENDING", "RUNNING", "Initializing", "Running", "New", "Suspended"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _record(kind_obj: str, name: str, namespace: str, reason: str, *, dry_run: bool) -> None:
    try:
        from app.core.ml_store import MLJob, get_ml_store

        job = MLJob(user_id=os.getenv("AVALOKA_SYSTEM_USER_ID", "00000000-0000-0000-0000-000000000000"),
                    kind="reap", status="succeeded" if not dry_run else "cancelled",
                    error=f"{'DRY-RUN ' if dry_run else ''}{kind_obj} {namespace}/{name}: {reason}"[:300])
        job.finished_at = _now()
        get_ml_store().create_job(job)
    except Exception:  # noqa: BLE001 - the record must not stop the reap
        logger.warning("[reaper] could not record %s %s/%s", kind_obj, namespace, name, exc_info=True)


# --------------------------------------------------------------------------- #
# Kubernetes access (thin, injectable)
# --------------------------------------------------------------------------- #

class K8s:
    """The few calls the reapers need; a fake with the same methods drives the tests."""

    def __init__(self) -> None:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except Exception:  # noqa: BLE001
            config.load_kube_config()
        self.custom = client.CustomObjectsApi()
        self.core = client.CoreV1Api()

    def list_ray(self, plural: str, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        if namespace:
            out = self.custom.list_namespaced_custom_object("ray.io", "v1", namespace, plural)
        else:
            out = self.custom.list_cluster_custom_object("ray.io", "v1", plural)
        return list(out.get("items", []))

    def delete_ray(self, plural: str, namespace: str, name: str) -> None:
        self.custom.delete_namespaced_custom_object("ray.io", "v1", namespace, plural, name)

    def patch_ray(self, plural: str, namespace: str, name: str, body: Dict[str, Any]) -> None:
        self.custom.patch_namespaced_custom_object("ray.io", "v1", namespace, plural, name, body)

    def list_namespaces(self) -> List[Dict[str, Any]]:
        return [{"metadata": {"name": ns.metadata.name, "labels": ns.metadata.labels or {},
                              "creationTimestamp": ns.metadata.creation_timestamp}}
                for ns in self.core.list_namespace().items]

    def delete_namespace(self, name: str) -> None:
        self.core.delete_namespace(name)

    def list_services(self, namespace: str) -> List[Dict[str, Any]]:
        return [{"metadata": {"name": s.metadata.name, "labels": s.metadata.labels or {},
                              "creationTimestamp": s.metadata.creation_timestamp},
                 "spec": {"type": s.spec.type}}
                for s in self.core.list_namespaced_service(namespace).items]

    def delete_service(self, namespace: str, name: str) -> None:
        self.core.delete_namespaced_service(name, namespace)


# --------------------------------------------------------------------------- #
# Pure decisions
# --------------------------------------------------------------------------- #

def rayjob_is_active(item: Dict[str, Any]) -> bool:
    status = item.get("status") or {}
    js = str(status.get("jobStatus") or "")
    ds = str(status.get("jobDeploymentStatus") or "")
    if js in {"SUCCEEDED", "FAILED", "STOPPED"} or ds in {"Complete", "Failed"}:
        return False
    return js in RAYJOB_ACTIVE or ds in RAYJOB_ACTIVE or (not js and not ds)


def looks_orphan(meta: Dict[str, Any]) -> bool:
    name = str(meta.get("name") or "")
    labels = meta.get("labels") or {}
    return bool(ORPHAN_NAME_RE.match(name)) or labels.get(PART_OF) == "avaloka"


def older_than(meta: Dict[str, Any], hours: float, *, now: Optional[datetime] = None) -> bool:
    created = _parse(meta.get("creationTimestamp"))
    if created is None:
        return False
    return ((now or _now()) - created).total_seconds() >= hours * 3600


def active_job_names(store: Any) -> set:
    names = set()
    try:
        for job in store.active_jobs():
            if job.ray_job_name:
                names.add(job.ray_job_name)
    except Exception:  # noqa: BLE001
        logger.warning("[reaper] could not read active jobs", exc_info=True)
    return names


# --------------------------------------------------------------------------- #
# The three reapers
# --------------------------------------------------------------------------- #

def reap_idle_model_server(*, store: Any = None, scalers: Optional[Iterable[Any]] = None,
                           now: Optional[float] = None, dry_run: bool = DRY_RUN) -> Dict[str, bool]:
    """Sleep each model-server Deployment idle past the timeout. Returns {name: scaled}."""
    from app.core.ml_store import get_ml_store
    from app.infra.model_server_scaler import DEPLOYMENT, GPU_DEPLOYMENT, ModelServerScaler

    if os.getenv("MODEL_SERVER_SCALE_TO_ZERO", "controller").strip().lower() == "keda":
        return {}
    store = store or get_ml_store()
    clock = now if now is not None else time.time()
    last_by_server: Dict[str, Optional[float]] = {"cpu": None, "gpu": None}
    try:
        for ep in store.list_all_endpoints() if hasattr(store, "list_all_endpoints") else _all_endpoints(store):
            if ep.last_request_at is None:
                continue
            stamp = ep.last_request_at.timestamp()
            key = "gpu" if ep.server == "gpu" else "cpu"
            last_by_server[key] = max(last_by_server[key] or 0.0, stamp)
    except Exception:  # noqa: BLE001
        logger.warning("[reaper] could not read endpoint activity", exc_info=True)
    scalers = list(scalers) if scalers is not None else [ModelServerScaler(DEPLOYMENT), ModelServerScaler(GPU_DEPLOYMENT)]
    out: Dict[str, bool] = {}
    for scaler in scalers:
        key = "gpu" if "gpu" in scaler.name else "cpu"
        if dry_run:
            out[scaler.name] = False
            continue
        out[scaler.name] = bool(scaler.sleep_if_idle(last_by_server[key], now=clock))
        if out[scaler.name]:
            _record("Deployment", scaler.name, scaler.namespace, "idle past the timeout; scaled to 0", dry_run=False)
    return out


def _all_endpoints(store: Any) -> List[Any]:
    # InMemory/Supabase stores list per user; the reaper needs every row.
    try:
        return list(store.endpoints.values())  # InMemoryMLStore
    except AttributeError:
        pass
    try:
        rows = store._rows(store._c().table(os.getenv("AVALOKA_ML_ENDPOINTS_TABLE", "ml_endpoints"))
                           .select("*").not_.is_("last_request_at", "null").execute())
        from app.core.ml_store import MLEndpoint, _from_row
        return [_from_row(MLEndpoint, r) for r in rows]
    except Exception:  # noqa: BLE001
        logger.warning("[reaper] could not list endpoints", exc_info=True)
        return []


def reap_idle_ray_cluster(*, k8s: Any = None, store: Any = None, now: Optional[datetime] = None,
                          dry_run: bool = DRY_RUN, idle_s: int = RAY_IDLE_TIMEOUT_S,
                          name: str = SHARED_RAY_CLUSTER, namespace: str = RAY_NAMESPACE,
                          action: str = RAY_IDLE_ACTION) -> bool:
    """After ``idle_s`` (7200 s) with no active RayJob: scale the shared cluster's
    workers to 0 and keep the head (default), or delete the CR when
    ``RAY_IDLE_ACTION=delete`` is allowed. True when it acted. ``idle_s <= 0`` disables."""
    if idle_s <= 0:
        return False
    action = idle_action_allowed(action)
    if action == "none":
        return False
    k8s = k8s or K8s()
    clusters = [c for c in k8s.list_ray("rayclusters", namespace) if c["metadata"]["name"] == name]
    if not clusters:
        return False
    jobs = k8s.list_ray("rayjobs", namespace)
    if any(rayjob_is_active(j) for j in jobs):
        return False
    clock = now or _now()
    last_activity: Optional[datetime] = None
    for j in jobs:
        for key in ("endTime", "startTime"):
            stamp = _parse((j.get("status") or {}).get(key))
            if stamp and (last_activity is None or stamp > last_activity):
                last_activity = stamp
    created = _parse(clusters[0]["metadata"].get("creationTimestamp"))
    anchor = max([t for t in (last_activity, created) if t is not None], default=None)
    if anchor is not None and (clock - anchor).total_seconds() < idle_s:
        return False
    if dry_run:
        _record("RayCluster", name, namespace, f"idle; would {action}", dry_run=True)
        return False
    if action == "delete":
        k8s.delete_ray("rayclusters", namespace, name)
        _record("RayCluster", name, namespace, f"idle >= {idle_s}s with no RayJob; deleted", dry_run=False)
        logger.info("[reaper] deleted idle RayCluster %s/%s", namespace, name)
        return True
    groups = (clusters[0].get("spec") or {}).get("workerGroupSpecs") or []
    if all(int(g.get("replicas") or 0) == 0 for g in groups):
        return False  # already at zero workers; the head stays by design
    patch = {"spec": {"workerGroupSpecs": [{"groupName": g.get("groupName"), "replicas": 0} for g in groups]}}
    k8s.patch_ray("rayclusters", namespace, name, patch)
    _record("RayCluster", name, namespace, f"idle >= {idle_s}s with no RayJob; workers scaled to 0, head kept", dry_run=False)
    logger.info("[reaper] scaled idle RayCluster %s/%s workers to 0 (head kept)", namespace, name)
    return True


class SharedClusterUnavailable(RuntimeError):
    """The shared RayCluster could not be made ready in time (TI-44)."""


ENSURE_TIMEOUT_S = int(os.getenv("RAY_ENSURE_TIMEOUT_S", "600"))


def _cluster_ready(cluster: Dict[str, Any]) -> bool:
    st = cluster.get("status") or {}
    state = str(st.get("state") or "").lower()
    if state == "ready":
        return True
    # Older KubeRay: no `state`; a head pod with an IP means the head is up.
    return bool(st.get("head", {}).get("podIP")) if isinstance(st.get("head"), dict) else False


def ensure_shared_cluster(*, k8s: Any = None, name: str = SHARED_RAY_CLUSTER, namespace: str = RAY_NAMESPACE,
                          apply: Any = None, wait: bool = False, timeout_s: int = ENSURE_TIMEOUT_S,
                          poll_s: float = 5.0, sleep: Any = time.sleep, clock: Any = time.monotonic) -> bool:
    """Re-apply the RayCluster manifest when the CR is missing; with ``wait`` block until
    it reports ready or raise :class:`SharedClusterUnavailable` at ``timeout_s``.
    Returns True when the manifest was applied."""
    k8s = k8s or K8s()
    applied = False
    clusters = [c for c in k8s.list_ray("rayclusters", namespace) if c["metadata"]["name"] == name]
    if not clusters:
        if apply is None:
            from app.infra.ray_manager import apply_ray_cluster
            apply = apply_ray_cluster
        apply(namespace=namespace)
        applied = True
        logger.info("[reaper] re-applied RayCluster %s/%s for a new training job", namespace, name)
    if not wait:
        return applied
    deadline = clock() + timeout_s
    while True:
        clusters = [c for c in k8s.list_ray("rayclusters", namespace) if c["metadata"]["name"] == name]
        if clusters and _cluster_ready(clusters[0]):
            return applied
        if clock() >= deadline:
            raise SharedClusterUnavailable(
                f"The shared Ray cluster '{name}' in namespace '{namespace}' did not become ready "
                f"within {timeout_s} seconds; training was not started."
            )
        sleep(poll_s)


def reap_orphan_ray(*, k8s: Any = None, store: Any = None, now: Optional[datetime] = None,
                    dry_run: bool = DRY_RUN, max_age_h: float = RAY_ORPHAN_MAX_AGE_H,
                    shared_name: str = SHARED_RAY_CLUSTER) -> List[str]:
    """Delete leftovers of the legacy per-job path. Returns the names acted on (or would act on)."""
    from app.core.ml_store import get_ml_store

    k8s = k8s or K8s()
    store = store or get_ml_store()
    clock = now or _now()
    active = active_job_names(store)
    acted: List[str] = []

    def _act(kind: str, namespace: str, name: str, reason: str, delete) -> None:
        acted.append(f"{kind}/{namespace}/{name}")
        if dry_run:
            _record(kind, name, namespace, f"would delete: {reason}", dry_run=True)
            return
        try:
            delete()
            _record(kind, name, namespace, reason, dry_run=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[reaper] delete %s %s/%s failed: %s", kind, namespace, name, exc)

    for job in k8s.list_ray("rayjobs"):
        meta = job["metadata"]
        if meta["name"] in active:
            continue
        terminal = not rayjob_is_active(job)
        if (terminal and looks_orphan(meta)) or (looks_orphan(meta) and older_than(meta, max_age_h, now=clock)):
            _act("RayJob", meta["namespace"], meta["name"], "terminal or past max age with no active ml_job",
                 lambda m=meta: k8s.delete_ray("rayjobs", m["namespace"], m["name"]))

    # Upgrade path (TI-52): the pre-2026-11 RayService keeps a head running for
    # nothing once the model server is the backend. Remove it within the hour.
    if os.getenv("INFERENCE_BACKEND", "modelserver").strip().lower() != "rayserve":
        for svc in k8s.list_ray("rayservices"):
            meta = svc["metadata"]
            if meta["name"] == os.getenv("RAY_SERVICE_NAME", "avaloka-inference"):
                _act("RayService", meta["namespace"], meta["name"], "stale RayService: INFERENCE_BACKEND is not rayserve",
                     lambda m=meta: k8s.delete_ray("rayservices", m["namespace"], m["name"]))

    for cluster in k8s.list_ray("rayclusters"):
        meta = cluster["metadata"]
        if meta["name"] == shared_name:
            continue  # the shared cluster has its own idle rule
        if looks_orphan(meta) and older_than(meta, max_age_h, now=clock):
            _act("RayCluster", meta["namespace"], meta["name"], f"per-job cluster older than {max_age_h}h",
                 lambda m=meta: k8s.delete_ray("rayclusters", m["namespace"], m["name"]))

    for ns in k8s.list_namespaces():
        meta = ns["metadata"]
        if ORPHAN_NAME_RE.match(str(meta.get("name") or "")) and older_than(meta, max_age_h, now=clock):
            for svc in k8s.list_services(meta["name"]):
                if (svc.get("spec") or {}).get("type") == "LoadBalancer":
                    _act("Service", meta["name"], svc["metadata"]["name"], "public LoadBalancer of a legacy namespace",
                         lambda n=meta["name"], s=svc["metadata"]["name"]: k8s.delete_service(n, s))
            _act("Namespace", meta["name"], meta["name"], f"legacy per-job namespace older than {max_age_h}h",
                 lambda n=meta["name"]: k8s.delete_namespace(n))
    return acted
