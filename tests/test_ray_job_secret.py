"""Ray pods carry no platform secrets beyond named keys; each job carries its own.

The shared RayCluster and RayService run user code. They used to load the
whole avaloka-secrets Secret (database, JWT, payment and model-provider keys)
through envFrom. They now get named keys only. An analysis job's connection
keys reach that job alone: a Secret owned by its RayJob (KubeRay mode), or the
job's env (direct mode).

kubectl is faked at subprocess.run, so every manifest the code sends is
captured from stdin. The credential values are dummies and never touch disk.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

import app.agents.execution_agent as ea
import app.infra.ray_job_runner as rjr

REPO = Path(__file__).resolve().parents[1]
RAY_DIR = REPO / "deploy" / "helm" / "ray"
CHART = REPO / "deploy" / "helm" / "avaloka"

PLATFORM_STORAGE_KEYS = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}
# MTA training reads the PII key in its driver, which Ray runs on the head.
ALLOWED = {
    ("raycluster", "ray-head"): PLATFORM_STORAGE_KEYS | {"AVALOKA_PII_KEY"},
    ("raycluster", "ray-worker"): PLATFORM_STORAGE_KEYS,
    ("rayservice", "ray-head"): PLATFORM_STORAGE_KEYS,
    ("rayservice", "ray-worker"): PLATFORM_STORAGE_KEYS,
}


def _ray_containers():
    for manifest in ("raycluster", "rayservice"):
        for doc in yaml.safe_load_all((RAY_DIR / f"{manifest}.yaml").read_text()):
            if not doc or doc.get("kind") not in ("RayCluster", "RayService"):
                continue
            spec = doc["spec"] if doc["kind"] == "RayCluster" else doc["spec"]["rayClusterConfig"]
            groups = [spec["headGroupSpec"], *spec.get("workerGroupSpecs", [])]
            for group in groups:
                for c in group["template"]["spec"]["containers"]:
                    yield manifest, c


def test_ray_pods_do_not_load_avaloka_secrets_whole():
    seen = set()
    for manifest, c in _ray_containers():
        seen.add((manifest, c["name"]))
        whole = [e["secretRef"]["name"] for e in c.get("envFrom") or [] if "secretRef" in e]
        assert not whole, f"{manifest}/{c['name']} loads whole Secrets: {whole}"
        keyed = {
            e["name"]: e["valueFrom"]["secretKeyRef"]
            for e in c.get("env") or []
            if "secretKeyRef" in (e.get("valueFrom") or {})
        }
        allowed = ALLOWED[(manifest, c["name"])]
        assert set(keyed) <= allowed, f"{manifest}/{c['name']} gets {sorted(set(keyed) - allowed)}"
        assert set(keyed) == allowed, f"{manifest}/{c['name']} lacks {sorted(allowed - set(keyed))}"
        for name, ref in keyed.items():
            assert ref.get("key") == name and ref.get("optional") is True, (manifest, c["name"], name)
    assert seen == set(ALLOWED)


# ── the analysis job's own Secret ────────────────────────────────────────────

CONN = {
    "provider": "s3",
    "access_key": "dummy-access-key",
    "secret_key": "dummy-secret-value",
    "region": "eu-west-1",
    # Fields a connection row also has; none of them belongs in the job.
    "user_id": "u-1", "bucket": "b", "name": "my conn", "id": "conn-abc",
}
CONN_KEYS = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"}


class _FakeKubectl:
    """Answers the kubectl calls a job makes; records them in order."""

    def __init__(self):
        self.events = []      # (verb, detail)
        self.secrets = []     # manifests sent on stdin

    def __call__(self, cmd, *a, input=None, **k):
        args = [str(x) for x in cmd[1:]] if cmd and "kubectl" in str(cmd[0]) else None
        if args is None:
            raise AssertionError(f"unexpected subprocess: {cmd!r}")
        out = ""
        if input:
            doc = yaml.safe_load(input)
            if isinstance(doc, dict) and doc.get("kind") == "Secret":
                self.secrets.append(doc)
                self.events.append(("create-secret", doc["metadata"]["name"]))
        if args[:2] == ["get", "rayjob"] and any("metadata.uid" in x for x in args):
            out = "uid-123"
        if "delete" in args and "secret" in args:
            self.events.append(("delete-secret", args[args.index("secret") + 1]))
        return subprocess.CompletedProcess(cmd, 0, out, "")


def _state():
    return {
        "ray_namespace": "ray-jobs", "ray_workers": 2, "ray_timeout_s": 60,
        "data_source_location_cloud": "s3://bucket/data.csv", "connection_id": "conn-abc",
        "dataset_id": "ds-001", "coder_definition": {"code": "def main(df):\n    return df\n"},
        "messages": [],
    }


def _run_kuberay(monkeypatch, ending):
    monkeypatch.delenv("RAY_DASHBOARD_URL", raising=False)
    monkeypatch.delenv("RAYJOB_DRYRUN", raising=False)
    fake = _FakeKubectl()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(rjr, "_which_kubectl", lambda: "kubectl", raising=False)
    applied = {}

    def _apply(yaml_text, namespace="default"):
        applied["yaml"] = yaml_text
        fake.events.append(("apply-rayjob", namespace))

    def _wait(rayjob_name, namespace="default", timeout_s=0, poll_interval_s=0):
        if ending == "timeout":
            raise TimeoutError("still running")
        if ending == "error":
            raise RuntimeError("kubectl lost the cluster")
        return {"status": {"jobStatus": "SUCCEEDED" if ending == "success" else "FAILED"}}

    monkeypatch.setattr(rjr, "apply_yaml_text", _apply)
    monkeypatch.setattr(rjr, "wait_for_rayjob", _wait)
    monkeypatch.setattr(rjr, "fetch_rayjob_logs", lambda *a, **k: "")
    monkeypatch.setattr(rjr, "cleanup_rayjob_code_configmap", lambda *a, **k: None)
    monkeypatch.setattr(rjr, "_get_current_context", lambda: "test")
    monkeypatch.setattr(ea, "get_cloud_connection", lambda cid: None)
    monkeypatch.setattr(ea, "_run_coro_sync", lambda coro: dict(CONN))
    monkeypatch.setattr(ea, "_dispatch_execution_memory_record", lambda *a, **k: None)
    out = ea.execution_agent_node_ray(_state())
    return out, fake, applied


@pytest.mark.parametrize("ending", ["success", "failed", "timeout", "error"])
def test_analysis_job_gets_its_own_secret_owned_by_the_rayjob(monkeypatch, ending):
    out, fake, applied = _run_kuberay(monkeypatch, ending)
    rayjob = out["execution_result"]["rayjob_name"]

    assert len(fake.secrets) == 1, f"expected one job Secret, got {len(fake.secrets)}"
    secret = fake.secrets[0]
    name = secret["metadata"]["name"]
    assert set(secret.get("stringData") or {}) == CONN_KEYS
    owners = secret["metadata"].get("ownerReferences") or [{}]
    assert (owners[0].get("kind"), owners[0].get("name"), owners[0].get("uid")) == ("RayJob", rayjob, "uid-123"), (
        "the job Secret must be owned by its RayJob, so it goes when the job goes"
    )
    verbs = [v for v, _ in fake.events]
    assert verbs.index("apply-rayjob") < verbs.index("create-secret"), (
        "the Secret is created after the RayJob, so the owner is in the create itself"
    )
    assert ("delete-secret", name) in fake.events[verbs.index("create-secret"):], (
        f"the job Secret was not deleted after a {ending} job"
    )
    # The RayJob's pods read that Secret; the CR itself holds no credential.
    assert name in applied["yaml"]
    assert CONN["secret_key"] not in applied["yaml"] and CONN["access_key"] not in applied["yaml"]


def test_direct_mode_passes_the_connection_keys_as_the_jobs_env(monkeypatch):
    # Direct mode submits to the shared cluster, whose pods no longer hold
    # storage keys for a tenant's bucket: the job must carry its own.
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://ray.invalid:8265")
    monkeypatch.delenv("RAYJOB_DRYRUN", raising=False)
    seen = {}

    def _direct(**kw):
        seen.update(kw)
        return rjr.RayJobOutcome(status="SUCCEEDED", rayjob_name=kw["job_id"], namespace="direct",
                                 job_status="SUCCEEDED", deployment_status=None, runtime_s=1.0,
                                 logs="", details={})

    monkeypatch.setattr(rjr, "_run_ray_job_direct", _direct)
    monkeypatch.setattr(ea, "get_cloud_connection", lambda cid: None)
    monkeypatch.setattr(ea, "_run_coro_sync", lambda coro: dict(CONN))
    monkeypatch.setattr(ea, "_dispatch_execution_memory_record", lambda *a, **k: None)
    ea.execution_agent_node_ray(_state())
    env = (seen.get("runtime_env") or {}).get("env_vars") or {}
    assert {k: env.get(k) for k in CONN_KEYS} == {
        "AWS_ACCESS_KEY_ID": "dummy-access-key",
        "AWS_SECRET_ACCESS_KEY": "dummy-secret-value",
        "AWS_DEFAULT_REGION": "eu-west-1",
    }


# ── RBAC: create/delete on Secrets, never get, only in the job namespace ─────

def _helm(*args):
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")
    run = subprocess.run([helm, "template", "avaloka", str(CHART), *args], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return [d for d in yaml.safe_load_all(run.stdout) if d]


def test_rbac_grants_job_secret_create_delete_without_get_in_the_job_namespace():
    docs = _helm("--set", "rbac.rayJobSecretsNamespace=ray-training")
    roles = [d for d in docs if d["kind"] == "Role" and d["metadata"].get("namespace") == "ray-training"]
    assert len(roles) == 1, "no Role in the RayJob namespace"
    secret_rules = [r for r in roles[0]["rules"] if "secrets" in r.get("resources", [])]
    assert [sorted(r["verbs"]) for r in secret_rules] == [["create", "delete"]]
    binding = next(d for d in docs if d["kind"] == "RoleBinding" and d["metadata"].get("namespace") == "ray-training")
    assert binding["roleRef"]["name"] == roles[0]["metadata"]["name"]
    # No Role anywhere may read Secrets.
    for d in docs:
        if d["kind"] in ("Role", "ClusterRole"):
            for r in d.get("rules", []):
                if "secrets" in r.get("resources", []):
                    assert not {"get", "list", "watch", "patch", "update"} & set(r["verbs"]), d["metadata"]["name"]
