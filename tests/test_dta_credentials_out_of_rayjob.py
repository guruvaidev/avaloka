"""A data transfer's credentials stay out of the RayJob, its ConfigMap, state, logs and replies.

A transfer carries its source and destination database URLs
(DTA_SOURCE_CONN_STR / DTA_DEST_CONN_STR) and any bucket keys. They used to be
rendered into the RayJob's runtimeEnvYAML, so the CR -- and with it the
Kubernetes API, etcd and anyone allowed to read RayJobs -- held them. Now:

  * KubeRay mode: a Secret <rayjob>-env per job, holding only that job's keys,
    created owned by the RayJob, read through secretKeyRef and deleted on every
    exit path. The app may create and delete Secrets there, never read them.
  * Direct mode: the keys go to the Ray Job Submission API as the job's env.
  * The planner's registry keeps a UI connection's handle, not its secret; the
    transfer opens the handle when it runs.

kubectl is faked at subprocess.run, so every manifest the code sends is
captured. The credential values are synthetic and recognisable, and never
touch disk. The last test runs a real copy against Postgres.
"""
import json
import os
import secrets
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote_plus, urlsplit

import pytest
import yaml

import app.data_transfer_docker.gke_run as gk
import app.infra.ray_job_runner as rjr

REPO = Path(__file__).resolve().parents[1]
CHART = REPO / "deploy" / "helm" / "avaloka"

SRC_PW = "PROBE-src-pw-7f3a1c"
DST_PW = "PROBE-dst-pw-9c1e55"
AWS_AK = "AKIAPROBEACCESS0001"
AWS_SK = "probe/secret+key=0001xyz"
CLOUD_ENV = {
    "DTA_SOURCE_CONN_STR": f"postgresql://src_user:{SRC_PW}@src-db.example:5432/src",
    "DTA_DEST_CONN_STR": f"postgresql://dst_user:{DST_PW}@dst-db.example:5432/dst",
    "DEST_AWS_ACCESS_KEY_ID": AWS_AK,
    "DEST_AWS_SECRET_ACCESS_KEY": AWS_SK,
}
PROBES = (SRC_PW, DST_PW, AWS_AK, AWS_SK, quote_plus(AWS_SK))


def _leaks(text):
    return [p for p in PROBES if p in (text or "")]


def _docs(yaml_text):
    return [d for d in yaml.safe_load_all(yaml_text) if d]


def _capture_launch(monkeypatch):
    """Run launch_gke_pipeline with the runner replaced; return what it was handed."""
    seen = {}

    def _runner(**kw):
        seen.update(kw)
        return rjr.RayJobOutcome(status="SUCCEEDED", rayjob_name=kw.get("rayjob_name") or "x",
                                 namespace="ns", job_status="SUCCEEDED", deployment_status=None,
                                 runtime_s=1.0, logs="AVALOKA_RESULT rows_inserted=3 table=t",
                                 details={})

    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://ray.invalid:8265")
    monkeypatch.setattr(gk, "run_rayjob_from_yaml", _runner)
    gk.launch_gke_pipeline("print('copy')", job_id="j1", cloud_env=dict(CLOUD_ENV))
    return seen


# ── 1. the rendered CR and ConfigMap ─────────────────────────────────────────

def test_rendered_rayjob_and_configmap_hold_no_credential(monkeypatch):
    seen = _capture_launch(monkeypatch)
    text = seen["yaml_text"]
    found = _leaks(text)
    assert not found, f"the rendered RayJob/ConfigMap carries {found}"
    docs = _docs(text)
    rayjob = next(d for d in docs if d["kind"] == "RayJob")
    configmap = next(d for d in docs if d["kind"] == "ConfigMap")
    found = _leaks(json.dumps(configmap))
    assert not found, f"the ConfigMap carries {found}"
    runtime_env = yaml.safe_load(rayjob["spec"]["runtimeEnvYAML"])
    assert not set(CLOUD_ENV) & set(runtime_env.get("env_vars") or {}), (
        "credential env names are still in the CR's runtimeEnvYAML"
    )
    # The pods read each key from the job's own Secret, by name.
    secret = f"{rayjob['metadata']['name']}-env"
    spec = rayjob["spec"]["rayClusterSpec"]
    containers = [spec["headGroupSpec"]["template"]["spec"]["containers"][0],
                  spec["workerGroupSpecs"][0]["template"]["spec"]["containers"][0]]
    for c in containers:
        refs = {e["name"]: e["valueFrom"]["secretKeyRef"] for e in c.get("env") or []
                if "secretKeyRef" in (e.get("valueFrom") or {})}
        assert refs == {k: {"name": secret, "key": k} for k in CLOUD_ENV}, c["name"]


# ── 2. the per-job Secret (KubeRay mode) ─────────────────────────────────────

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
            out = "uid-dta-1"
        if "delete" in args and "secret" in args:
            self.events.append(("delete-secret", args[args.index("secret") + 1]))
        return subprocess.CompletedProcess(cmd, 0, out, "")


def _kuberay(monkeypatch, ending, cloud_env, job_id):
    """launch_gke_pipeline, but the runner takes KubeRay mode.

    launch_gke_pipeline itself insists on RAY_DASHBOARD_URL (1.6 and 2.0 alike),
    so the variable is set for that check and removed for the runner, which
    then applies the RayJob with kubectl -- the mode the CR exists in.
    """
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://ray.invalid:8265")
    monkeypatch.delenv("RAYJOB_DRYRUN", raising=False)
    fake = _FakeKubectl()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(rjr, "_which_kubectl", lambda: "kubectl", raising=False)
    applied = {}

    def _apply(yaml_text, namespace="default"):
        applied["yaml"] = yaml_text
        applied["namespace"] = namespace
        fake.events.append(("apply-rayjob", namespace))

    def _wait(rayjob_name, namespace="default", timeout_s=0, poll_interval_s=0):
        applied["rayjob"] = rayjob_name
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
    real = rjr.run_rayjob_from_yaml

    def _runner(**kw):
        monkeypatch.delenv("RAY_DASHBOARD_URL", raising=False)
        return real(**{**kw, "dashboard_url": None})

    monkeypatch.setattr(gk, "run_rayjob_from_yaml", _runner)
    gk.launch_gke_pipeline("print('copy')", job_id=job_id, cloud_env=dict(cloud_env))
    return fake, applied


@pytest.mark.parametrize("ending", ["success", "failed", "timeout", "error"])
def test_job_secret_is_owned_by_the_rayjob_holds_only_its_keys_and_is_deleted(monkeypatch, ending):
    fake, applied = _kuberay(monkeypatch, ending, CLOUD_ENV, "j-" + ending)
    rayjob = applied["rayjob"]

    assert len(fake.secrets) == 1, f"expected one job Secret, got {len(fake.secrets)}"
    secret = fake.secrets[0]
    name = secret["metadata"]["name"]
    assert name == f"{rayjob}-env"
    assert secret["metadata"]["namespace"] == applied["namespace"]
    assert secret.get("stringData") == CLOUD_ENV, "the job Secret holds exactly this job's keys"
    owners = secret["metadata"].get("ownerReferences") or [{}]
    assert (owners[0].get("kind"), owners[0].get("name"), owners[0].get("uid")) == (
        "RayJob", rayjob, "uid-dta-1"), "the job Secret must be owned by its RayJob"
    verbs = [v for v, _ in fake.events]
    assert verbs.index("apply-rayjob") < verbs.index("create-secret"), (
        "the Secret is created after the RayJob, so the owner is in the create itself"
    )
    assert ("delete-secret", name) in fake.events[verbs.index("create-secret"):], (
        f"the job Secret was not deleted after a {ending} job"
    )
    found = _leaks(applied["yaml"])
    assert not found, "the applied RayJob/ConfigMap carries a credential"


def test_each_job_secret_holds_only_that_jobs_connection(monkeypatch):
    pw = "other-job-pw-1234"
    other = {"DTA_SOURCE_CONN_STR": f"postgresql://a:{pw}@h/a",
             "DTA_DEST_CONN_STR": f"postgresql://b:{pw}-dst@h/b"}
    first, _ = _kuberay(monkeypatch, "success", CLOUD_ENV, "j-first")
    second, _ = _kuberay(monkeypatch, "success", other, "j-second")
    assert [s.get("stringData") for s in first.secrets] == [CLOUD_ENV]
    assert [s.get("stringData") for s in second.secrets] == [other]
    assert first.secrets[0]["metadata"]["name"] != second.secrets[0]["metadata"]["name"]


# ── 3. RBAC: the namespace transfers go to; create/delete, never get ─────────

def _helm(*args):
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")
    run = subprocess.run([helm, "template", "avaloka", str(CHART), *args], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return [d for d in yaml.safe_load_all(run.stdout) if d]


def test_rbac_lets_transfers_create_and_delete_their_secret_without_get():
    docs = _helm("--set", "rbac.rayJobSecretsNamespace=dta-jobs")
    config = next((d for d in docs if d["kind"] == "ConfigMap" and "KUBE_NAMESPACE" in (d.get("data") or {})), None)
    assert config is not None, "no ConfigMap sets KUBE_NAMESPACE: transfers go to 'default', where no Role is"
    assert config["data"]["KUBE_NAMESPACE"] == "dta-jobs"
    roles = [d for d in docs if d["kind"] == "Role" and d["metadata"].get("namespace") == "dta-jobs"]
    assert len(roles) == 1, "no Role in the transfer namespace"
    secret_rules = [r for r in roles[0]["rules"] if "secrets" in r.get("resources", [])]
    assert [sorted(r["verbs"]) for r in secret_rules] == [["create", "delete"]]
    binding = next(d for d in docs if d["kind"] == "RoleBinding" and d["metadata"].get("namespace") == "dta-jobs")
    assert binding["roleRef"]["name"] == roles[0]["metadata"]["name"]
    assert any(s["kind"] == "ServiceAccount" for s in binding["subjects"])
    for d in docs:
        if d["kind"] in ("Role", "ClusterRole"):
            for r in d.get("rules", []):
                if "secrets" in r.get("resources", []):
                    assert not {"get", "list", "watch", "patch", "update"} & set(r["verbs"]), d["metadata"]["name"]


# ── 4. direct mode: the job's runtime env, and nothing persisted ─────────────

def test_direct_mode_passes_credentials_as_the_jobs_runtime_env_only(monkeypatch):
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://ray.invalid:8265")
    monkeypatch.delenv("RAYJOB_DRYRUN", raising=False)
    handed, submitted = {}, {}
    real = rjr.run_rayjob_from_yaml

    def _runner(**kw):
        handed.update(kw)
        return real(**kw)

    def _direct(**kw):
        submitted.update(kw)
        return rjr.RayJobOutcome(status="SUCCEEDED", rayjob_name=kw["job_id"], namespace="direct",
                                 job_status="SUCCEEDED", deployment_status=None, runtime_s=1.0,
                                 logs="", details={})

    def _no_kubectl(cmd, *a, **k):
        raise AssertionError(f"direct mode must not create any cluster object: {cmd!r}")

    monkeypatch.setattr(gk, "run_rayjob_from_yaml", _runner)
    monkeypatch.setattr(rjr, "_run_ray_job_direct", _direct)
    monkeypatch.setattr(subprocess, "run", _no_kubectl)
    gk.launch_gke_pipeline("print('copy')", job_id="jd", cloud_env=dict(CLOUD_ENV))

    env = (submitted.get("runtime_env") or {}).get("env_vars") or {}
    assert {k: env.get(k) for k in CLOUD_ENV} == CLOUD_ENV, "the job's runtime env lacks its credentials"
    found = _leaks(handed["yaml_text"])
    assert not found, "the rendered RayJob/ConfigMap carries a credential"
    found = _leaks(submitted.get("script_text"))
    assert not found, "the job's code carries a credential"


# ── 5. the planner: registry handles, replies ────────────────────────────────

CLOUD_ID = "6f1c2a3b-4d5e-4f60-8a7b-9c0d1e2f3a4b"


def _planner_harness(monkeypatch, pipeline=None):
    """The planner's transfer path with the onboarding API, cloud store and runner faked."""
    import app.agents.planner as P
    import app.api.cloud_connections as cc

    creds = {
        "src_db": {"host": "src-db.example", "port": 5432, "database": "src",
                   "username": "src_user", "password": SRC_PW, "table": "orders"},
        "dst_db": {"host": "dst-db.example", "port": 5432, "database": "dst",
                   "username": "dst_user", "password": DST_PW, "table": "orders_copy"},
    }

    class _Resp:
        def __init__(self, code, body):
            self.status_code, self._body = code, body

        def json(self):
            return self._body

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None, **_kw):
            tail = url[len(P.ONBOARDING_API_URL):]
            parts = tail.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "customers" and parts[2] == "credentials":
                body = creds.get(parts[1])
                return _Resp(200, {**body, "database_type": "postgresql"}) if body else _Resp(404, {})
            if tail.rstrip("/") == "/customers":
                return _Resp(200, [{"customer_id": k, "database_type": "postgresql"} for k in creds])
            return _Resp(404, {})

    async def _cloud(conn_id):
        if conn_id != CLOUD_ID:
            raise LookupError(conn_id)
        return {"id": CLOUD_ID, "provider": "s3", "name": "exports", "bucket_name": "probe-bucket",
                "region": "eu-west-1", "access_key": AWS_AK, "secret_key": AWS_SK}

    monkeypatch.setattr(P.requests, "Session", _Session)
    monkeypatch.setattr(cc, "get_cloud_connection", _cloud)
    monkeypatch.setattr(P, "_list_dest_tables", lambda *a, **k: ["orders_copy"])
    monkeypatch.setenv("EXECUTION_ENV", "gke")
    monkeypatch.setenv("DTA_GKE_REACHES_PRIVATE", "1")

    handed, launched = {}, {}

    def _pipeline(**kw):
        handed.update(kw)
        return {"error_message": None, "warnings": None, "runtime_secret_env": {}}, "# copy"

    def _launch(script, **kw):
        launched.update(kw)
        return SimpleNamespace(rows_inserted=3, table=kw.get("table") or "orders_copy")

    fake_dta = types.ModuleType("app.agents.data_transfer_agent.data_transfer_agent")
    fake_dta.data_transfer_pipeline = pipeline or _pipeline
    fake_gke = types.ModuleType("app.data_transfer_docker.gke_run")
    fake_gke.launch_gke_pipeline = _launch
    monkeypatch.setitem(sys.modules, fake_dta.__name__, fake_dta)
    monkeypatch.setitem(sys.modules, fake_gke.__name__, fake_gke)
    return SimpleNamespace(P=P, handed=handed, launched=launched)


def _transfer(h, source, destination, **extra):
    state = {"user_id": "user_a", "dta_database_registry": {}, "messages": []}
    msg = h.P._handle_initiate_transfer(
        state,
        h.P.InitiateTransferParams(source_alias=source, destination_alias=destination,
                                   user_prompt=f"copy orders from {source} to {destination}",
                                   write_mode="append", **extra),
    )
    return state, msg


def test_registry_keeps_a_ui_databases_handle_not_its_password(monkeypatch):
    h = _planner_harness(monkeypatch)
    state, msg = _transfer(h, "src_db", "dst_db", dest_table="orders_copy")
    assert h.handed["source_credentials"]["password"] == SRC_PW, msg
    assert h.handed["destination_credentials"]["password"] == DST_PW, msg
    dumped = json.dumps(state, default=str)
    found = _leaks(dumped)
    assert not found, f"graph state carries {found}"
    assert state["dta_database_registry"]["src_db"].get("conn_id") == "src_db"
    found = _leaks(msg)
    assert not found, f"the reply carries {found}"


def test_registry_keeps_a_cloud_connections_id_not_its_keys(monkeypatch):
    h = _planner_harness(monkeypatch)
    state, msg = _transfer(h, "src_db", CLOUD_ID, dest_object="exports/orders.csv")
    env = h.launched.get("cloud_env") or {}
    assert (env.get("DEST_AWS_ACCESS_KEY_ID"), env.get("DEST_AWS_SECRET_ACCESS_KEY")) == (AWS_AK, AWS_SK), msg
    dumped = json.dumps(state, default=str)
    found = _leaks(dumped)
    assert not found, f"graph state carries {found}"
    assert state["dta_database_registry"][CLOUD_ID].get("conn_id") == CLOUD_ID


def test_a_checkpoint_holding_a_ui_password_is_scrubbed_on_read(monkeypatch):
    h = _planner_harness(monkeypatch)
    state = {"user_id": "user_a", "messages": [], "dta_database_registry": {
        "src_db": {"db_type": "postgresql", "host": "src-db.example", "port": 5432, "database": "src",
                   "user": "src_user", "password": SRC_PW, "_source": "ui"},
    }}
    registry = h.P._get_db_registry(state)
    found = _leaks(json.dumps(registry))
    assert not found, "an older checkpoint's UI password survives a read"
    assert registry["src_db"]["host"] == "src-db.example"


def test_a_failed_pipeline_reply_and_log_carry_no_password(monkeypatch, caplog):
    def _boom(**kw):
        src = kw["source_credentials"]
        raise ValueError(f"Could not parse SQLAlchemy URL from string "
                         f"'postgresql://{src['user']}:{src['password']}@{src['host']}/{src['database']}'")

    h = _planner_harness(monkeypatch, pipeline=_boom)
    caplog.set_level("DEBUG")
    state, msg = _transfer(h, "src_db", "dst_db", dest_table="orders_copy")
    assert "Transfer failed" in msg
    found = _leaks(msg)
    assert not found, f"the reply carries {found}"
    found = _leaks(caplog.text)
    assert not found, f"the log carries {found}"


# ── 6. end to end: a real copy against Postgres ──────────────────────────────

ADMIN_URL = os.environ.get("AVALOKA_TEST_PG_ADMIN_URL", "").strip()
ROWS = 41

COPY_SCRIPT = '''
import os
from sqlalchemy import create_engine, text
src = create_engine(os.environ["DTA_SOURCE_CONN_STR"])
dst = create_engine(os.environ["DTA_DEST_CONN_STR"])
with src.connect() as s:
    rows = s.execute(text('SELECT id, amount FROM "{src_table}" ORDER BY id')).all()
with dst.begin() as d:
    d.execute(text('INSERT INTO "{dst_table}" (id, amount) VALUES (:id, :amount)'),
              [{{"id": r[0], "amount": r[1]}} for r in rows])
print(f"AVALOKA_RESULT rows_inserted={{len(rows)}} table={dst_table}")
'''


def _admin_url():
    if ADMIN_URL:
        return ADMIN_URL
    message = "AVALOKA_TEST_PG_ADMIN_URL is not set; the transfer was not exercised against Postgres"
    if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
        pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
    pytest.skip(message)


def _engine(url):
    from sqlalchemy import create_engine
    return create_engine(url, isolation_level="AUTOCOMMIT", pool_pre_ping=True)


@pytest.fixture(scope="module")
def pg():
    """A source and a destination database, each owned by its own login role."""
    from sqlalchemy import text

    admin = urlsplit(_admin_url())
    host, port = admin.hostname, admin.port or 5432
    run = secrets.token_hex(4)
    dbs = {}
    eng = _engine(_admin_url())
    try:
        with eng.connect() as conn:
            for n in ("src", "dst"):
                db = role = f"e2e_dtacred_{n}_{run}"
                # The probe: a password that must show up nowhere but the job's env.
                pw = f"PROBE-e2e-{n}-" + secrets.token_hex(8)
                conn.execute(text(f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{pw}'"))
                conn.execute(text(f'CREATE DATABASE "{db}" OWNER "{role}"'))
                dbs[n] = SimpleNamespace(db=db, role=role, pw=pw,
                                         url=f"postgresql://{role}:{quote_plus(pw)}@{host}:{port}/{db}")
        e = _engine(dbs["src"].url)
        with e.connect() as conn:
            conn.execute(text("CREATE TABLE orders (id integer PRIMARY KEY, amount numeric(10,2))"))
            conn.execute(text(f"INSERT INTO orders SELECT g, g * 1.5 FROM generate_series(1, {ROWS}) g"))
        e.dispose()
        e = _engine(dbs["dst"].url)
        with e.connect() as conn:
            conn.execute(text("CREATE TABLE orders_copy (id integer, amount numeric(10,2))"))
        e.dispose()
        yield SimpleNamespace(host=host, port=port, dbs=dbs)
    finally:
        with eng.connect() as conn:
            for d in dbs.values():
                conn.execute(text(f'DROP DATABASE IF EXISTS "{d.db}" WITH (FORCE)'))
                conn.execute(text(f'DROP ROLE IF EXISTS "{d.role}"'))
        eng.dispose()


class _FakeCluster:
    """kubectl against a pretend cluster that really runs the job.

    It stores what is applied (the RayJob and its ConfigMap) and created (the
    Secret), and when the RayJob is polled it does what the kubelet and Ray
    would: the head container's env, secretKeyRefs resolved against the stored
    Secret, plus the runtimeEnvYAML env_vars, then the ConfigMap's script.
    """

    def __init__(self, real_run):
        self.real_run = real_run
        self.objects = {}     # (kind, name) -> manifest
        self.texts = []       # every manifest text the cluster was sent
        self.deleted = []
        self.logs = ""
        self.status = None

    def _run_job(self, name):
        rayjob = self.objects[("RayJob", name)]
        script = self.objects[("ConfigMap", f"{name}-code")]["data"]["sample_code.py"]
        head = rayjob["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"][0]
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
        for e in head.get("env") or []:
            ref = (e.get("valueFrom") or {}).get("secretKeyRef")
            if ref:
                secret = self.objects.get(("Secret", ref["name"]))
                if secret is None or ref["key"] not in (secret.get("stringData") or {}):
                    self.status, self.logs = "FAILED", f"CreateContainerConfigError: {ref}"
                    return
                env[e["name"]] = secret["stringData"][ref["key"]]
            elif "value" in e:
                env[e["name"]] = str(e["value"])
        runtime = yaml.safe_load(rayjob["spec"].get("runtimeEnvYAML") or "") or {}
        env.update({k: str(v) for k, v in (runtime.get("env_vars") or {}).items()})
        proc = self.real_run([sys.executable, "-c", script], capture_output=True, text=True,
                             env=env, timeout=120)
        self.logs = proc.stdout + proc.stderr
        self.status = "SUCCEEDED" if proc.returncode == 0 else "FAILED"

    def __call__(self, cmd, *a, input=None, **k):
        if not (cmd and "kubectl" in str(cmd[0])):
            return self.real_run(cmd, *a, input=input, **k)
        args = [str(x) for x in cmd[1:]]
        out = ""
        if args[:1] == ["apply"]:
            text = Path(args[args.index("-f") + 1]).read_text()
            self.texts.append(text)
            for d in _docs(text):
                self.objects[(d["kind"], d["metadata"]["name"])] = d
        elif args[:1] == ["create"] and input:
            self.texts.append(input)
            d = yaml.safe_load(input)
            self.objects[(d["kind"], d["metadata"]["name"])] = d
        elif args[:2] == ["get", "rayjob"]:
            name = args[2]
            obj = self.objects[("RayJob", name)]
            if any("metadata.uid" in x for x in args):
                out = "uid-e2e-1"
            else:
                if self.status is None:
                    self._run_job(name)
                out = json.dumps({**obj, "status": {"jobStatus": self.status}})
        elif args[:1] == ["logs"]:
            out = self.logs
        elif args[:1] == ["delete"]:
            self.deleted.append((args[1], args[2]))
            kind = {"secret": "Secret", "configmap": "ConfigMap", "rayjob": "RayJob"}.get(args[1])
            self.objects.pop((kind, args[2]), None)
        elif args[:2] == ["config", "current-context"]:
            out = "fake"
        return subprocess.CompletedProcess(cmd, 0, out, "")


@pytest.mark.parametrize("mode", ["kuberay", "direct"])
def test_real_postgres_transfer_copies_and_the_probe_shows_up_nowhere(pg, monkeypatch, caplog, capsys,
                                                                     tmp_path, mode):
    import app.agents.planner as P
    import app.mcp_server.customer_dbs as cdb
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy import text

    caplog.set_level("DEBUG")
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("MCP_SERVICE_TOKEN", token)
    monkeypatch.setenv("EXECUTION_ENV", "gke")
    monkeypatch.setenv("DTA_GKE_REACHES_PRIVATE", "1")
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://ray.invalid:8265")
    monkeypatch.delenv("RAY_DASHBOARD_URL_INCLUSTER", raising=False)
    monkeypatch.delenv("RAYJOB_DRYRUN", raising=False)
    monkeypatch.setenv("KUBE_NAMESPACE", "dta-jobs")

    e = _engine(pg.dbs["dst"].url)
    with e.connect() as conn:
        conn.execute(text("TRUNCATE orders_copy"))
    e.dispose()

    # The connections live in the onboarding store, reached through its real router.
    records = {}
    for cid, n in (("cust_src", "src"), ("cust_dst", "dst")):
        records[cid] = {
            "customer_id": cid, "customer_name": cid, "database_type": "postgresql",
            "connection_string": pg.dbs[n].url, "environment": "production",
            "api_key": "ak_" + secrets.token_hex(16), "status": "active",
            "created_at": "2026-10-31T00:00:00", "owner_user_id": "user_a",
        }
    store = tmp_path / "customers.json"
    store.write_text(json.dumps(records))
    try:
        mgr = cdb.customer_manager
        monkeypatch.setattr(mgr, "customers_file", str(store))
        monkeypatch.setattr(mgr, "customers", {})
        mgr.load_customers()
        onboarding = FastAPI()
        onboarding.include_router(cdb.router)
        client = TestClient(onboarding)
        base = P.ONBOARDING_API_URL

        class _RoutedSession:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url, headers=None, **_kw):
                assert url.startswith(base)
                return client.get(url[len(base):], headers=headers or {})

        monkeypatch.setattr(P.requests, "Session", _RoutedSession)

        # Code generation is replaced (no model); everything after it is real:
        # the planner's env merge, launch_gke_pipeline, the runner, the Secret.
        def _pipeline(**kw):
            def url(c):
                return (f"postgresql://{c['user']}:{quote_plus(c['password'])}"
                        f"@{c['host']}:{c['port']}/{c['database']}")
            src, dst = kw["source_credentials"], kw["destination_credentials"]
            script = COPY_SCRIPT.format(src_table="orders", dst_table=dst["table"])
            return ({"error_message": None, "warnings": None,
                     "runtime_secret_env": {"DTA_SOURCE_CONN_STR": url(src), "DTA_DEST_CONN_STR": url(dst)}},
                    script)

        fake_dta = types.ModuleType("app.agents.data_transfer_agent.data_transfer_agent")
        fake_dta.data_transfer_pipeline = _pipeline
        monkeypatch.setitem(sys.modules, fake_dta.__name__, fake_dta)
        monkeypatch.setitem(sys.modules, "app.data_transfer_docker.gke_run", gk)

        real_run = subprocess.run
        cluster = _FakeCluster(real_run)
        seen = {}
        monkeypatch.setattr(rjr, "_which_kubectl", lambda: "kubectl", raising=False)
        real_runner = rjr.run_rayjob_from_yaml

        if mode == "kuberay":
            monkeypatch.setattr(subprocess, "run", cluster)

            def _runner(**kw):
                seen.update(kw)
                monkeypatch.delenv("RAY_DASHBOARD_URL", raising=False)
                return real_runner(**{**kw, "dashboard_url": None})
        else:
            def _direct(**kw):
                seen["runtime_env"] = kw["runtime_env"]
                env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                       **kw["runtime_env"].get("env_vars", {})}
                proc = real_run([sys.executable, "-c", kw["script_text"]], capture_output=True,
                                text=True, env=env, timeout=120)
                ok = proc.returncode == 0
                return rjr.RayJobOutcome(status="SUCCEEDED" if ok else "FAILED", rayjob_name=kw["job_id"],
                                         namespace="direct", job_status=None, deployment_status=None,
                                         runtime_s=1.0, logs=proc.stdout + proc.stderr, details={})

            monkeypatch.setattr(rjr, "_run_ray_job_direct", _direct)

            def _runner(**kw):
                seen.update(kw)
                return real_runner(**kw)

        monkeypatch.setattr(gk, "run_rayjob_from_yaml", _runner)

        state = {"user_id": "user_a", "dta_database_registry": {}, "messages": []}
        msg = P._handle_initiate_transfer(state, P.InitiateTransferParams(
            source_alias="cust_src", destination_alias="cust_dst",
            user_prompt="copy orders from cust_src to cust_dst",
            write_mode="append", dest_table="orders_copy"))
    finally:
        store.unlink(missing_ok=True)

    from sqlalchemy import text as _t
    e = _engine(pg.dbs["dst"].url)
    with e.connect() as conn:
        copied = conn.execute(_t('SELECT count(*) FROM "orders_copy"')).scalar_one()
    e.dispose()
    assert copied == ROWS, msg
    assert state["dta_last_transfer"]["rows_inserted"] == ROWS, msg

    probes = (pg.dbs["src"].pw, pg.dbs["dst"].pw)
    out = capsys.readouterr()
    places = {
        "the rendered RayJob/ConfigMap": seen.get("yaml_text", ""),
        "the reply": msg,
        "graph state": json.dumps(state, default=str),
        "the log": caplog.text,
        "stdout/stderr": out.out + out.err,
    }
    if mode == "kuberay":
        places["what the cluster was sent (RayJob, ConfigMap)"] = "\n".join(
            t for t in cluster.texts if '"kind": "Secret"' not in t)
    # Booleans only: a failing assertion must not print the text (or the probe).
    leaked = sorted(where for where, text_ in places.items()
                    if any(p in text_ or quote_plus(p) in text_ for p in probes))
    assert not leaked, f"the probe credential is in: {leaked}"
    if mode == "kuberay":
        rayjob = seen["rayjob_name"]
        assert ("secret", f"{rayjob}-env") in cluster.deleted, "the job Secret outlived the job"
        assert ("Secret", f"{rayjob}-env") not in cluster.objects
    else:
        env = seen["runtime_env"].get("env_vars", {})
        carries = pg.dbs["src"].pw in env.get("DTA_SOURCE_CONN_STR", "")
        assert carries, "direct mode: the job's env carries its source URL"


def test_a_db_probe_error_that_quotes_the_url_carries_no_password(monkeypatch, caplog):
    import sqlalchemy
    from app.agents.data_transfer_agent import data_transfer_agent as dta

    url = CLOUD_ENV["DTA_SOURCE_CONN_STR"]

    def _engine(u, **_kw):
        raise ValueError(f"Could not parse SQLAlchemy URL from string '{u}'")

    monkeypatch.setattr(sqlalchemy, "create_engine", _engine)
    caplog.set_level("DEBUG")
    with pytest.raises(dta.PipelineError) as err:
        dta.probe_db_connection(url, "source")
    found = _leaks(str(err.value))
    assert not found, f"the probe error (error_message, then the reply) carries {found}"
    found = _leaks(caplog.text)
    assert not found, f"the log carries {found}"
