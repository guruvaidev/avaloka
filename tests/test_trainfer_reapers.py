"""The infra layer's reapers (T-35, T-36, T-21 idle model server). All E on the base."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core import ml_store


@pytest.fixture
def store():
    s = ml_store.InMemoryMLStore()
    ml_store.set_ml_store(s)
    yield s
    ml_store.set_ml_store(None)


NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


def _meta(name, namespace="default", age_h=0.0, labels=None):
    return {"name": name, "namespace": namespace, "labels": labels or {},
            "creationTimestamp": (NOW - timedelta(hours=age_h)).isoformat()}


class FakeK8s:
    def __init__(self, rayjobs=(), rayclusters=(), namespaces=(), services=None):
        self.rayjobs, self.rayclusters, self.namespaces = list(rayjobs), list(rayclusters), list(namespaces)
        self.services = services or {}
        self.deleted = []
        self.patched = []

    def list_ray(self, plural, namespace=None):
        items = self.rayjobs if plural == "rayjobs" else self.rayclusters
        return [i for i in items if namespace is None or i["metadata"]["namespace"] == namespace]

    def delete_ray(self, plural, namespace, name):
        self.deleted.append((plural, namespace, name))

    def patch_ray(self, plural, namespace, name, body):
        self.patched.append((plural, namespace, name, body))

    def list_namespaces(self):
        return self.namespaces

    def delete_namespace(self, name):
        self.deleted.append(("namespace", name, name))

    def list_services(self, namespace):
        return self.services.get(namespace, [])

    def delete_service(self, namespace, name):
        self.deleted.append(("service", namespace, name))


# T-35 orphans -----------------------------------------------------------------------

def test_orphan_reaper_deletes_old_legacy_namespaces_and_their_public_lb(store):
    from app.infra import ray_reaper as rr
    k8s = FakeK8s(
        namespaces=[{"metadata": _meta("ray-trainer-5bd923f0", age_h=77 * 24)},
                    {"metadata": _meta("ray-trainer-fresh", age_h=1)},
                    {"metadata": _meta("default", age_h=1000)}],
        services={"ray-trainer-5bd923f0": [{"metadata": _meta("ray-head-svc", "ray-trainer-5bd923f0"), "spec": {"type": "LoadBalancer"}}]},
    )
    acted = rr.reap_orphan_ray(k8s=k8s, store=store, now=NOW, dry_run=False)
    assert ("service", "ray-trainer-5bd923f0", "ray-head-svc") in k8s.deleted
    assert ("namespace", "ray-trainer-5bd923f0", "ray-trainer-5bd923f0") in k8s.deleted
    assert not any(n == "ray-trainer-fresh" for _, n, _ in k8s.deleted), "a one-hour-old namespace is not an orphan"
    assert not any(n == "default" for _, n, _ in k8s.deleted)
    assert acted == ["Service/ray-trainer-5bd923f0/ray-head-svc", "Namespace/ray-trainer-5bd923f0/ray-trainer-5bd923f0"]
    kinds = [j.error for j in store.jobs.values() if j.kind == "reap"]
    assert len(kinds) == 2 and all("ray-trainer-5bd923f0" in e for e in kinds)


def test_orphan_reaper_keeps_a_rayjob_with_an_active_ml_job_and_the_shared_cluster(store):
    from app.infra import ray_reaper as rr
    store.create_job(ml_store.MLJob(user_id="u", kind="training", status="running", ray_job_name="train-keep"))
    k8s = FakeK8s(
        rayjobs=[{"metadata": _meta("train-keep", age_h=30, labels={"app.kubernetes.io/part-of": "avaloka"}), "status": {"jobStatus": "RUNNING"}},
                 {"metadata": _meta("train-done", age_h=2, labels={"app.kubernetes.io/part-of": "avaloka"}), "status": {"jobStatus": "SUCCEEDED"}}],
        rayclusters=[{"metadata": _meta("avaloka-raycluster", age_h=500)},
                     {"metadata": _meta("ray-trainer-old", "ray-trainer-old", age_h=30)}],
    )
    rr.reap_orphan_ray(k8s=k8s, store=store, now=NOW, dry_run=False)
    assert ("rayjobs", "default", "train-done") in k8s.deleted, "a terminal job of ours is cleaned up"
    assert ("rayjobs", "default", "train-keep") not in k8s.deleted
    assert ("rayclusters", "ray-trainer-old", "ray-trainer-old") in k8s.deleted
    assert ("rayclusters", "default", "avaloka-raycluster") not in k8s.deleted, "the shared cluster has its own idle rule"


def test_orphan_reaper_dry_run_deletes_nothing_but_records_intent(store):
    from app.infra import ray_reaper as rr
    k8s = FakeK8s(namespaces=[{"metadata": _meta("ray-trainer-x", age_h=48)}])
    acted = rr.reap_orphan_ray(k8s=k8s, store=store, now=NOW, dry_run=True)
    assert acted == ["Namespace/ray-trainer-x/ray-trainer-x"]
    assert k8s.deleted == []
    recorded = [j for j in store.jobs.values() if j.kind == "reap"]
    assert len(recorded) == 1 and recorded[0].error.startswith("DRY-RUN")


# T-36 idle shared cluster ---------------------------------------------------------------

def _cluster(age_h=10, worker_replicas=2):
    return {"metadata": _meta("avaloka-raycluster", age_h=age_h),
            "spec": {"workerGroupSpecs": [{"groupName": "cpu-workers", "replicas": worker_replicas}]}}


def test_idle_cluster_default_action_scales_workers_to_zero_and_keeps_the_head(store):
    """Coordinator condition 1: many Ray consumers connect to the head directly, so
    the idle reaper must not delete the CR; it scales the worker groups to 0."""
    from app.infra import ray_reaper as rr
    finished = {"metadata": _meta("j1", age_h=5), "status": {"jobStatus": "SUCCEEDED", "endTime": (NOW - timedelta(hours=3)).isoformat()}}
    k8s = FakeK8s(rayclusters=[_cluster()], rayjobs=[finished])
    assert rr.reap_idle_ray_cluster(k8s=k8s, store=store, now=NOW, dry_run=False, idle_s=7200) is True
    assert k8s.deleted == [], "the head (the CR) stays"
    assert k8s.patched == [("rayclusters", "default", "avaloka-raycluster",
                            {"spec": {"workerGroupSpecs": [{"groupName": "cpu-workers", "replicas": 0}]}})]

    running = {"metadata": _meta("j2", age_h=1), "status": {"jobStatus": "RUNNING"}}
    k8s2 = FakeK8s(rayclusters=[_cluster()], rayjobs=[finished, running])
    assert rr.reap_idle_ray_cluster(k8s=k8s2, store=store, now=NOW, dry_run=False, idle_s=7200) is False
    assert k8s2.deleted == [] and k8s2.patched == []

    k8s3 = FakeK8s(rayclusters=[_cluster(worker_replicas=0)], rayjobs=[finished])
    assert rr.reap_idle_ray_cluster(k8s=k8s3, store=store, now=NOW, dry_run=False, idle_s=7200) is False, "already at zero workers"


def test_delete_action_is_refused_while_any_consumer_bypasses_ensure(store):
    from app.infra import ray_reaper as rr
    assert rr.RAY_CONSUMERS_BYPASSING_ENSURE, "the inventory must list today's direct Ray consumers"
    assert rr.idle_action_allowed("delete") == "scale-workers"
    assert rr.idle_action_allowed("delete", bypassing=()) == "delete"
    assert rr.idle_action_allowed("garbage") == "scale-workers"
    finished = {"metadata": _meta("j1", age_h=5), "status": {"jobStatus": "SUCCEEDED", "endTime": (NOW - timedelta(hours=3)).isoformat()}}
    k8s = FakeK8s(rayclusters=[_cluster()], rayjobs=[finished])
    rr.reap_idle_ray_cluster(k8s=k8s, store=store, now=NOW, dry_run=False, idle_s=7200, action="delete")
    assert k8s.deleted == [], "delete requested but refused: workers scaled instead"
    assert k8s.patched


def test_ray_consumer_inventory_matches_the_tree():
    """Every file that reaches Ray directly is in the inventory, and nothing in the
    inventory has stopped reaching Ray (so RAY_IDLE_ACTION=delete is refused for
    the right reason, and the day the list empties the gate opens)."""
    import re
    from pathlib import Path
    from app.infra import ray_reaper as rr
    root = Path(rr.__file__).resolve().parents[2]
    # Calls that reach a Ray cluster, not mentions of the word (comments, type
    # names and error classes are not consumers).
    pattern = re.compile(r"ray\.init\(|os\.environ\[.RAY_ADDRESS|getenv\(.RAY_ADDRESS|JobSubmissionClient\(|submit_ray_job\(|execute_on_ray\(|execution_agent_node_ray\(|ray\.data\.|run_ray_job|apply_yaml_text\(")
    skip = ("training_docker_image", "inference_docker_image", "inference_service_image", "ray_reaper.py",
            "ray_manager.py", "ray_trainer.py", "cluster_bootstrap.py", "deploy_stack.py", "k8s_invoker.py",
            "user_ray_clusters.py", "rayjob_renderer.py", "ray_jobs_store.py", "container_file.py", "docker_run.py",
            "mta_v2/gcp/", "mta_v2/inference.py", "mta_v2/loader/", "output_contract.py", "planner.py", "celery_app.py")
    consumers = set()
    for path in (root / "app").rglob("*.py"):
        rel = str(path.relative_to(root))
        if any(part in rel for part in skip):
            continue
        if pattern.search(path.read_text(errors="ignore")):
            consumers.add(rel)
    assert consumers == set(rr.RAY_CONSUMERS_BYPASSING_ENSURE), {
        "unlisted": sorted(consumers - set(rr.RAY_CONSUMERS_BYPASSING_ENSURE)),
        "stale": sorted(set(rr.RAY_CONSUMERS_BYPASSING_ENSURE) - consumers)}


def test_idle_cluster_boundary_7199_holds_7200_acts_and_zero_disables(store):
    from app.infra import ray_reaper as rr
    recent = {"metadata": _meta("j1", age_h=5), "status": {"jobStatus": "SUCCEEDED", "endTime": (NOW - timedelta(seconds=7199)).isoformat()}}
    k8s = FakeK8s(rayclusters=[_cluster()], rayjobs=[recent])
    assert rr.reap_idle_ray_cluster(k8s=k8s, store=store, now=NOW, dry_run=False, idle_s=7200) is False
    exact = {"metadata": _meta("j1", age_h=5), "status": {"jobStatus": "SUCCEEDED", "endTime": (NOW - timedelta(seconds=7200)).isoformat()}}
    k8s2 = FakeK8s(rayclusters=[_cluster()], rayjobs=[exact])
    assert rr.reap_idle_ray_cluster(k8s=k8s2, store=store, now=NOW, dry_run=False, idle_s=7200) is True
    assert rr.reap_idle_ray_cluster(k8s=FakeK8s(rayclusters=[_cluster()]), store=store, now=NOW, dry_run=False, idle_s=0) is False


def test_both_idle_timeouts_default_to_two_hours(monkeypatch):
    """Coordinator condition 3: INFERENCE_IDLE_TIMEOUT_S and RAY_IDLE_TIMEOUT_S default to 7200."""
    import importlib
    for var in ("INFERENCE_IDLE_TIMEOUT_S", "RAY_IDLE_TIMEOUT_S", "RAY_IDLE_TEARDOWN_S", "MODEL_SERVER_IDLE_TIMEOUT_S"):
        monkeypatch.delenv(var, raising=False)
    from app.infra import model_server_scaler, ray_reaper
    rr = importlib.reload(ray_reaper); ms = importlib.reload(model_server_scaler)
    assert rr.RAY_IDLE_TIMEOUT_S == 7200
    assert ms.IDLE_TIMEOUT_S == 7200
    monkeypatch.setenv("RAY_IDLE_TIMEOUT_S", "60")
    assert importlib.reload(ray_reaper).RAY_IDLE_TIMEOUT_S == 60
    monkeypatch.delenv("RAY_IDLE_TIMEOUT_S")
    importlib.reload(ray_reaper); importlib.reload(model_server_scaler)


def test_model_server_scale_to_zero_disabled_never_patches(store):
    """TI-19: with the idle timeout 0 the floor is 1 and the reaper patches nothing."""
    from app.infra.model_server_scaler import ModelServerScaler
    api = MagicMock()
    s = ModelServerScaler("d", "ns", api=api, idle_timeout_s=0)
    assert s.scale_to_zero is False
    assert s.sleep_if_idle(0.0, now=1e9) is False
    api.patch_namespaced_deployment_scale.assert_not_called()


def test_ensure_shared_cluster_applies_only_when_missing():
    from app.infra import ray_reaper as rr
    apply = MagicMock()
    assert rr.ensure_shared_cluster(k8s=FakeK8s(), apply=apply) is True
    apply.assert_called_once()
    present = FakeK8s(rayclusters=[{"metadata": _meta("avaloka-raycluster")}])
    assert rr.ensure_shared_cluster(k8s=present, apply=apply) is False
    apply.assert_called_once()


# T-21 idle model server through the reaper ------------------------------------------------

def test_idle_model_server_reaper_scales_the_right_deployment_to_zero(store, monkeypatch):
    from app.infra import ray_reaper as rr
    monkeypatch.delenv("MODEL_SERVER_SCALE_TO_ZERO", raising=False)
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="r-cpu", user_id="u", status="ready", server="cpu",
                                              last_request_at=NOW - timedelta(hours=3)))
    store.upsert_endpoint(ml_store.MLEndpoint(mlflow_run_id="r-gpu", user_id="u", status="ready", server="gpu",
                                              last_request_at=NOW - timedelta(minutes=5)))
    cpu = MagicMock(name="cpu", namespace="ns"); cpu.name = "avaloka-model-server"; cpu.sleep_if_idle.return_value = True
    gpu = MagicMock(namespace="ns"); gpu.name = "avaloka-model-server-gpu"; gpu.sleep_if_idle.return_value = False
    out = rr.reap_idle_model_server(store=store, scalers=[cpu, gpu], now=NOW.timestamp(), dry_run=False)
    assert out == {"avaloka-model-server": True, "avaloka-model-server-gpu": False}
    cpu.sleep_if_idle.assert_called_once_with((NOW - timedelta(hours=3)).timestamp(), now=NOW.timestamp())
    gpu.sleep_if_idle.assert_called_once_with((NOW - timedelta(minutes=5)).timestamp(), now=NOW.timestamp())


def test_idle_model_server_reaper_is_a_noop_under_keda(store, monkeypatch):
    from app.infra import ray_reaper as rr
    monkeypatch.setenv("MODEL_SERVER_SCALE_TO_ZERO", "keda")
    scaler = MagicMock()
    assert rr.reap_idle_model_server(store=store, scalers=[scaler], dry_run=False) == {}
    scaler.sleep_if_idle.assert_not_called()


def test_rayjob_activity_classification():
    from app.infra.ray_reaper import rayjob_is_active
    assert rayjob_is_active({"status": {"jobStatus": "RUNNING"}})
    assert rayjob_is_active({"status": {}}), "a job with no status yet is treated as active (never reaped early)"
    assert not rayjob_is_active({"status": {"jobStatus": "SUCCEEDED"}})
    assert not rayjob_is_active({"status": {"jobDeploymentStatus": "Failed"}})
