"""1.6 release suite: Kubernetes, cloud provisioning and scaling (plan sheet 2).

Every test names the plan case it certifies. Live clusters and cloud accounts
are out of scope here: each test drives the product function that decides the
behaviour (the rendered manifest, the provider's command sequence, the failure
classifier, the scheduler wiring) with the outside world replaced by fakes.
The live legs stay in tests/k8s/ under the ``cluster``/``cloud`` markers.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml


# --------------------------------------------------------------------------- #
# K8-04: EKS provisioning must give the cluster somewhere to run pods.
# --------------------------------------------------------------------------- #

_EKS_ENV = {
    "AWS_REGION": "us-east-1",
    "AWS_EKS_CLUSTER_ROLE_ARN": "arn:aws:iam::000000000000:role/test-eks-role",
    "AWS_SUBNET_IDS": "subnet-aaa,subnet-bbb",
    "AWS_SECURITY_GROUP_IDS": "sg-aaa",
    "AWS_CLUSTER_NAME": "release-suite-eks",
}


class _FakeSts:
    def assume_role(self, **_kw):
        return {"Credentials": {"AccessKeyId": "test-akid",
                                "SecretAccessKey": "test-secret",
                                "SessionToken": "test-token"}}


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "K8-04 (MAJ-147): AwsEksProvider.provision_cluster creates the EKS control "
    "plane only; no managed nodegroup or Fargate profile, so no pod can schedule"))
def test_k804_eks_provision_creates_worker_capacity(monkeypatch):
    """K8-04: provisioning an EKS cluster also creates worker capacity (a nodegroup or Fargate profile)."""
    import app.infra.providers.aws_eks as aws_eks

    for key, value in _EKS_ENV.items():
        monkeypatch.setenv(key, value)
    # provision_cluster writes the assumed-role credentials straight into
    # os.environ (app/infra/providers/aws_eks.py). Register those keys with
    # monkeypatch (set, then delete: absent during the test, and teardown
    # removes whatever the product wrote) so they cannot leak into later tests,
    # where cloud_config would infer CLOUD_PROVIDER=aws from them.
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION"):
        monkeypatch.setenv(key, "placeholder")
        monkeypatch.delenv(key)
    commands: List[List[str]] = []

    class _Probe:  # describe-cluster: "does not exist yet"
        returncode = 1
        stdout = ""
        stderr = "ResourceNotFoundException"

    monkeypatch.setattr(aws_eks.boto3, "client", lambda *a, **k: _FakeSts())
    monkeypatch.setattr(aws_eks.subprocess, "run", lambda cmd, **kw: (commands.append(list(cmd)), _Probe())[1])

    def _fake_run_command(cmd, step_name):
        commands.append(list(cmd))
        return {"step_name": step_name, "status": "SUCCESS", "message": "", "details": ""}

    monkeypatch.setattr(aws_eks, "run_command", _fake_run_command)

    outcome = aws_eks.AwsEksProvider().provision_cluster()

    assert outcome["status"] != "FAILED", outcome
    verbs = [" ".join(c[:3]) for c in commands]
    assert "aws eks create-cluster" in verbs, verbs
    created_capacity = [v for v in verbs if v in ("aws eks create-nodegroup", "aws eks create-fargate-profile")]
    assert created_capacity, f"no worker capacity created; commands were: {verbs}"
    # Capacity must come after the control plane exists.
    assert verbs.index(created_capacity[0]) > verbs.index("aws eks create-cluster")


# --------------------------------------------------------------------------- #
# K8-10 / K8-11: Ray workers scale up under load and down to zero when idle.
# Owner decision (2026-10-31): the idle floor is 0. Every place that defines a
# Ray compute worker group is checked: the RayJob renderer and its shipped
# template, the interactive-profile override, and the deployed RayCluster chart.
# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parents[2]


def _strip_replica_comments(template: str) -> str:
    # render_rayjob_yaml's replica regexes only match a line that ends after the
    # number. The shipped template carries trailing comments, so on it they never
    # fire; this variant makes the renderer's own rewrite observable.
    return re.sub(r"(?m)^(\s*(?:replicas|minReplicas|maxReplicas):\s*\d+)\s*#.*$", r"\1", template)


def _render(workers: int, *, strip_comments: bool = False) -> str:
    from app.agents.execution_agent import BASE_RAYJOB_YAML, render_rayjob_yaml

    template = _strip_replica_comments(BASE_RAYJOB_YAML) if strip_comments else BASE_RAYJOB_YAML
    return render_rayjob_yaml(
        base_yaml=template,
        name="release-suite-job",
        namespace="ray-jobs",
        workers=workers,
        data_uri="s3://example-bucket/data.csv",
        script_text="print('hello')",
        cloud_secret_name="release-suite-secret",
    )


def _rayjob_cluster(text: str) -> Dict[str, Any]:
    docs = [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]
    return next(d for d in docs if d.get("kind") == "RayJob")["spec"]["rayClusterSpec"]


def _assert_room_to_scale(cluster: Dict[str, Any], workers: int) -> None:
    assert cluster.get("enableInTreeAutoscaling") is True
    groups = cluster["workerGroupSpecs"]
    assert groups, "no worker group"
    for group in groups:
        lo, hi = int(group["minReplicas"]), int(group["maxReplicas"])
        assert hi >= workers, group
        assert hi > lo, f"minReplicas {lo} == maxReplicas {hi}: the autoscaler has no room"


def test_k810_shipped_rayjob_manifest_leaves_room_to_scale_up():
    """K8-10: under load the worker count can rise toward maxReplicas (the RayJob rendered from the shipped template has autoscaling on and maxReplicas > minReplicas)."""
    _assert_room_to_scale(_rayjob_cluster(_render(workers=4)), workers=4)


def test_k810_renderer_does_not_pin_min_to_max():
    """K8-10: the renderer itself leaves autoscaling room: on a template it rewrites, maxReplicas is the requested worker count and minReplicas stays below it."""
    _assert_room_to_scale(_rayjob_cluster(_render(workers=4, strip_comments=True)), workers=4)


def _worker_floors(cluster: Dict[str, Any]) -> List[int]:
    return [int(g["minReplicas"]) for g in cluster["workerGroupSpecs"]]


def test_k811_shipped_rayjob_idle_floor_is_zero():
    """K8-11: an idle RayJob cluster releases every worker (the RayJob rendered from the shipped template has minReplicas 0 with autoscaling on)."""
    cluster = _rayjob_cluster(_render(workers=4))
    assert cluster.get("enableInTreeAutoscaling") is True
    assert _worker_floors(cluster) == [0]


def test_k811_renderer_sets_the_idle_floor_to_zero():
    """K8-11: whatever worker count is requested, the renderer writes minReplicas 0, so idle workers are released."""
    for workers in (1, 4, 10):
        assert _worker_floors(_rayjob_cluster(_render(workers=workers, strip_comments=True))) == [0], workers


def test_k811_interactive_profile_keeps_the_zero_floor():
    """K8-11: the interactive (small-footprint) profile override also leaves the idle floor at 0."""
    from app.agents.execution_agent import _apply_interactive_profile_to_yaml

    text = _apply_interactive_profile_to_yaml(_render(workers=4, strip_comments=True))
    assert _worker_floors(_rayjob_cluster(text)) == [0]


def test_k811_deployed_raycluster_idle_floor_is_zero():
    """K8-11: the deployed compute RayCluster (deploy/helm/ray/raycluster.yaml) autoscales and scales its workers to zero when idle."""
    docs = [d for d in yaml.safe_load_all((REPO / "deploy/helm/ray/raycluster.yaml").read_text()) if isinstance(d, dict)]
    spec = next(d for d in docs if d.get("kind") == "RayCluster")["spec"]
    assert spec.get("enableInTreeAutoscaling") is True
    for group in spec["workerGroupSpecs"]:
        assert int(group["minReplicas"]) == 0, group["groupName"]
        assert int(group["maxReplicas"]) > 0, group["groupName"]


# --------------------------------------------------------------------------- #
# K8-13: a failure that is not about memory must not be "fixed" with memory.
# --------------------------------------------------------------------------- #

_NON_MEMORY_FAILURES = [
    ("timeout", "Ray job did not complete within 1800s", {}),
    ("deadline", "context deadline exceeded while waiting for job", {}),
    ("image_pull", "Back-off pulling image", {"reason": "ImagePullBackOff"}),
    ("mlflow", "MLflow tracking uri unreachable: connection refused", {}),
    ("data", "target column 'price' not found in dataset", {}),
    ("generic", "Ray job failed with exit code 1", {"exit_code": 1}),
]


@pytest.mark.parametrize("label,error,kube", _NON_MEMORY_FAILURES, ids=[f[0] for f in _NON_MEMORY_FAILURES])
def test_k813_non_memory_failures_never_propose_more_memory(label, error, kube):
    """K8-13: a non-memory failure is classified as not-memory and no resource (memory) change is proposed."""
    from app.agents.mta_v2.failure_diagnostics import build_training_failure

    failure = build_training_failure(error, kubernetes=kube)

    assert failure["code"] != "OOM_KILLED", (label, failure["code"])
    memory_actions = [a for a in failure["actions"] if re.search(r"(?i)\bmemory\b", a)]
    assert memory_actions == [], (label, memory_actions)


def test_k813_an_oom_kill_is_still_classified_as_memory():
    """K8-13 (control): the classifier still recognises a real OOM kill, so the negative cases above are meaningful."""
    from app.agents.mta_v2.failure_diagnostics import build_training_failure

    failure = build_training_failure("container terminated", kubernetes={"reason": "OOMKilled", "exit_code": 137})

    assert failure["code"] == "OOM_KILLED"
    assert any(re.search(r"(?i)memory", a) for a in failure["actions"])


# --------------------------------------------------------------------------- #
# K8-14: something in production must call the idle reaper.
# --------------------------------------------------------------------------- #

@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "K8-14: RayServiceScaler.reap_if_idle (app/agents/mta_v2/inference_autoscale.py:288) "
    "has no production caller; no beat/CronJob entry runs it, so an idle service never scales to zero"))
def test_k814_idle_inference_reaper_runs_on_a_schedule(monkeypatch):
    """K8-14: a scheduled job (Celery beat entry) invokes the inference reaper, so an idle service scales to zero."""
    from app.agents.mta_v2.inference_autoscale import RayServiceScaler
    from app.core.celery_app import celery_app

    calls: List[str] = []
    monkeypatch.setattr(RayServiceScaler, "reap_if_idle",
                        lambda self, **kw: calls.append(self.name) or False)
    monkeypatch.setattr(RayServiceScaler, "current_replicas", lambda self: 0, raising=False)

    schedule = dict(celery_app.conf.beat_schedule or {})
    assert schedule, "celery beat_schedule is empty: nothing runs periodically"
    for entry in schedule.values():
        task = celery_app.tasks.get(entry["task"])
        assert task is not None, f"beat entry names an unregistered task: {entry['task']}"
        task.run(*entry.get("args", ()), **entry.get("kwargs", {}))

    assert calls, f"no scheduled task called reap_if_idle; beat entries: {sorted(schedule)}"


# --------------------------------------------------------------------------- #
# K8-15: the per-model namespace backend cannot run by accident.
# --------------------------------------------------------------------------- #

def test_k815_legacy_gateway_backend_is_refused_without_opt_in(monkeypatch):
    """K8-15: the legacy backend that minted an inference-service-{uuid} namespace per model is refused unless explicitly opted in, so teardown leaves no orphaned namespaces."""
    import app.agents.mta_v2.inference_service_manager as ism

    reached: List[str] = []
    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "gateway")
    monkeypatch.setattr(ism, "ALLOW_LEGACY_GATEWAY", False)
    monkeypatch.setattr(ism.InferenceServiceManager, "_configure_gateway",
                        lambda self, run_id: reached.append(run_id) or {})

    with pytest.raises(RuntimeError, match="AVALOKA_ALLOW_LEGACY_GATEWAY"):
        ism.InferenceServiceManager().configure_inference_service("run-1")
    assert reached == [], "the namespace-minting path ran without the opt-in"


def test_k815_default_backend_is_the_shared_rayservice(monkeypatch):
    """K8-15: the default backend is the shared RayService; the gateway path runs only with AVALOKA_ALLOW_LEGACY_GATEWAY."""
    import app.agents.mta_v2.inference_service_manager as ism

    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "rayserve")
    monkeypatch.setattr(ism, "ALLOW_LEGACY_GATEWAY", False)
    assert ism.InferenceServiceManager._backend() == "rayserve"

    monkeypatch.setattr(ism, "INFERENCE_BACKEND", "gateway")
    monkeypatch.setattr(ism, "ALLOW_LEGACY_GATEWAY", True)
    assert ism.InferenceServiceManager._backend() == "gateway"
