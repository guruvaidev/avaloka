"""MLflow and Ray require authentication, and nothing outside the cluster reaches them.

The MLflow tracking server ran from the API image with no authentication (an
RCE advisory applies to unauthenticated servers), and the Ray dashboard, Jobs
API, client server and GCS listened on 0.0.0.0 with no RAY_AUTH_MODE, behind a
NodePort on kind. These tests read what a pod is actually handed -- the
rendered chart, the Ray manifests the app applies, the RayJob the runner
applies -- and what the app's clients actually send.

Every test imports only modules that exist before this change, so on the old
tree each one fails by assertion, not by an import error.
"""
from __future__ import annotations

import base64
import http.server
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import threading
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "deploy" / "helm" / "avaloka"
RAY_DIR = ROOT / "deploy" / "helm" / "ray"
RAYJOB = ROOT / "app" / "infra" / "rayjob.yaml"
RAYJOB_DTA = ROOT / "app" / "infra" / "rayjob_data_transfer.yaml"
CLIENTS = ("api", "langgraph", "celery", "celery-redbeat")
PUBLISHED = {"password", "password1234", "changeme", "secret", "admin"}
LOOKUP = 'lookup "v1" "Secret" $ctx.Release.Namespace (include "avaloka.secretName" $ctx)'


# --------------------------------------------------------------------- helpers

def _helm(*args: str, chart: Path = CHART, cmd: str = "template") -> str:
    helm = shutil.which("helm")
    if helm is None:
        if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
            pytest.fail("helm is not installed and CI is set: the rendered-chart checks did not run")
        pytest.skip("helm is not installed")
    argv = [helm, cmd, "avaloka", str(chart), *args]
    if cmd == "install":
        argv.insert(2, "--dry-run=client")
    run = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stderr[-800:]
    return run.stdout


def _docs(text: str) -> list:
    return [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]


def _render(*args: str, chart: Path = CHART) -> list:
    return _docs(_helm(*args, chart=chart))


def _find(docs, kind, name):
    found = [d for d in docs if d.get("kind") == kind and d["metadata"]["name"] == name]
    assert len(found) == 1, f"expected one {kind} {name}, got {len(found)}"
    return found[0]


def _release_secret(docs) -> dict:
    return _find(docs, "Secret", "avaloka-secrets").get("stringData") or {}


def _workload(docs, component):
    found = [d for d in docs if d.get("kind") in ("Deployment", "StatefulSet")
             and d["spec"]["template"]["metadata"].get("labels", {}).get("app.kubernetes.io/component") == component]
    assert len(found) == 1, f"expected one workload {component}, got {len(found)}"
    return found[0]


def _env(container) -> dict:
    return {e["name"]: e for e in container.get("env") or []}


def _secret_ref(entry) -> dict:
    return ((entry or {}).get("valueFrom") or {}).get("secretKeyRef") or {}


def _usable(value) -> bool:
    v = (value or "").strip()
    return len(v) >= 32 and v.lower() not in PUBLISHED


def _ray_manifest(name: str) -> dict:
    docs = _docs((RAY_DIR / name).read_text())
    assert docs, name
    return docs[0]


def _cluster_spec(doc) -> dict:
    spec = doc["spec"]
    return spec.get("rayClusterConfig") or spec.get("rayClusterSpec") or spec


def _ray_containers(cluster_spec) -> list:
    out = [("head", c) for c in cluster_spec["headGroupSpec"]["template"]["spec"]["containers"]]
    for group in cluster_spec.get("workerGroupSpecs") or []:
        out += [(group["groupName"], c) for c in group["template"]["spec"]["containers"]]
    return out


def _assert_ray_auth(cluster_spec, secret_name: str, where: str) -> None:
    opts = cluster_spec.get("authOptions") or {}
    assert opts == {"mode": "token", "secretName": secret_name}, (
        f"{where}: authOptions is {opts!r}; the KubeRay operator needs it to read job and "
        "Serve status from a token-authenticated dashboard"
    )
    targets = _ray_containers(cluster_spec)
    if cluster_spec.get("enableInTreeAutoscaling"):
        targets.append(("autoscaler", cluster_spec.get("autoscalerOptions") or {}))
    for group, container in targets:
        env = _env(container)
        assert (env.get("RAY_AUTH_MODE") or {}).get("value") == "token", (
            f"{where} {group}: RAY_AUTH_MODE=token is missing")
        ref = _secret_ref(env.get("RAY_AUTH_TOKEN"))
        assert ref.get("name") == secret_name and ref.get("key") == "auth_token", (
            f"{where} {group}: RAY_AUTH_TOKEN must come by secretKeyRef from {secret_name}/auth_token, "
            f"got {env.get('RAY_AUTH_TOKEN')!r}")
        assert not ref.get("optional"), f"{where} {group}: an optional token lets a pod start open"
        for source in container.get("envFrom") or []:
            assert "secretRef" not in source or where.startswith("rayjob"), (
                f"{where} {group}: a whole Secret by envFrom ({source})")


class _Recorder(http.server.BaseHTTPRequestHandler):
    status = 200
    body = b"{}"

    def _answer(self):
        type(self).seen.append((self.command, self.path, self.headers.get("Authorization")))
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(type(self).body)))
        self.end_headers()
        self.wfile.write(type(self).body)

    do_GET = do_POST = do_DELETE = _answer

    def log_message(self, *a):  # keep test output quiet
        pass


@pytest.fixture
def server():
    handler = type("H", (_Recorder,), {"seen": [], "status": 200, "body": b"{}"})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield handler, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


# ------------------------------------------------------ 1. nothing public

# The one Service in deploy/helm/ray that is not ClusterIP: the kind-only
# NodePort for the Ray dashboard. Allowed only because it is opt-in (applied by
# cluster_bootstrap when RAY_DASHBOARD_EXPOSE_NODEPORT=1, never by default) and
# the head behind it runs with RAY_AUTH_MODE=token, so the port is closed without
# the token. tests/test_trainfer_cloud_boundary.py keeps the file itself.
KIND_ONLY_NODEPORT = "ray_dashboard_nodeport.yaml"


def _assert_kind_only_nodeport_is_opt_in(path) -> None:
    text = path.read_text()
    assert "kind only" in text and "RAY_DASHBOARD_EXPOSE_NODEPORT" in text, (
        f"{path.name} must say it is the kind-only, opt-in NodePort")
    manager = (ROOT / "app/infra/ray_manager.py").read_text()
    gate = re.search(r"def dashboard_nodeport_requested\(\).*?return os\.environ\.get\(\"RAY_DASHBOARD_EXPOSE_NODEPORT\"", manager, re.S)
    assert gate, "ray_manager.dashboard_nodeport_requested() must gate the NodePort on RAY_DASHBOARD_EXPOSE_NODEPORT"
    assert re.search(r"def apply_ray_dashboard_nodeport\(.*?if not dashboard_nodeport_requested\(\):", manager, re.S), (
        "apply_ray_dashboard_nodeport() must skip unless the NodePort was requested")
    for doc in _docs(text):
        if doc.get("kind") == "Service":
            selector = doc["spec"].get("selector") or {}
            assert selector.get("ray.io/node-type") == "head" and selector.get("ray.io/cluster"), (
                f"{path.name}: the NodePort may select only the shared cluster's head")
            assert [p["port"] for p in doc["spec"]["ports"]] == [8265], "only the token-checked dashboard port"


def test_neither_mlflow_nor_ray_has_a_public_service() -> None:
    docs = _render()
    for svc in (d for d in docs if d.get("kind") == "Service"):
        selector = (svc.get("spec") or {}).get("selector") or {}
        if selector.get("app.kubernetes.io/component") == "mlflow" or any(k.startswith("ray.io/") for k in selector):
            assert svc["spec"].get("type", "ClusterIP") == "ClusterIP", f"{svc['metadata']['name']} is public"
    for path in sorted(RAY_DIR.glob("*.yaml")):
        if path.name == KIND_ONLY_NODEPORT:
            _assert_kind_only_nodeport_is_opt_in(path)
            continue
        for doc in _docs(path.read_text()):
            if doc.get("kind") == "Service":
                assert doc["spec"].get("type", "ClusterIP") == "ClusterIP", (
                    f"{path.name}: Service {doc['metadata']['name']} is {doc['spec'].get('type')}; "
                    "the Ray dashboard must be reached by port-forward, with the token")
            spec = _cluster_spec(doc) if doc.get("kind", "").startswith("Ray") else {}
            head = spec.get("headGroupSpec") or {}
            assert head.get("serviceType", "ClusterIP") == "ClusterIP", f"{path.name}: head serviceType"


# ------------------------------------------------- 2. NetworkPolicies

def _policy_for(docs, predicate):
    return [d for d in docs if d.get("kind") == "NetworkPolicy" and predicate(d["spec"]["podSelector"])]


def test_mlflow_networkpolicy_admits_only_its_clients() -> None:
    docs = _render()
    policies = _policy_for(docs, lambda s: (s.get("matchLabels") or {}).get("app.kubernetes.io/component") == "mlflow")
    assert len(policies) == 1, "no NetworkPolicy selects the MLflow pod"
    spec = policies[0]["spec"]
    assert spec["policyTypes"] == ["Ingress"]
    (rule,) = spec["ingress"]
    assert [p["port"] for p in rule["ports"]] == [5000], rule.get("ports")
    peers = rule["from"]
    assert peers and all(p for p in peers), "an empty peer admits everything"
    components = set()
    ray_peer = False
    for peer in peers:
        exprs = (peer.get("podSelector") or {}).get("matchExpressions") or []
        for e in exprs:
            if e["key"] == "app.kubernetes.io/component":
                assert e["operator"] == "In"
                components |= set(e["values"])
            if e["key"] == "ray.io/node-type":
                ray_peer = True
    assert components == set(CLIENTS), f"MLflow admits {sorted(components)}"
    assert ray_peer, "Ray training and Serve pods are MLflow clients and must be admitted"


def test_ray_networkpolicy_admits_ray_pods_clients_and_the_operator() -> None:
    docs = _render("--set-string", "rbac.rayJobSecretsNamespace=ray-training")
    selects_ray = lambda s: any(e["key"] == "ray.io/node-type" and e["operator"] == "Exists"
                                for e in s.get("matchExpressions") or [])
    policies = _policy_for(docs, selects_ray)
    namespaces = sorted(p["metadata"].get("namespace") for p in policies)
    assert namespaces == ["default", "ray-training"], f"Ray pod policies in {namespaces}"
    for policy in policies:
        (rule,) = policy["spec"]["ingress"]
        assert "ports" not in rule or rule["ports"], "ports: [] would admit nothing, [] vs absent matters"
        peers = rule["from"]
        assert all(p for p in peers), "an empty peer admits every pod in every namespace"
        operator = [p for p in peers
                    if (p.get("podSelector") or {}).get("matchLabels") == {"app.kubernetes.io/name": "kuberay-operator"}]
        assert operator, f"{policy['metadata']['name']}: the KubeRay operator is not admitted"
    shared = next(p for p in policies if p["metadata"]["namespace"] == "default")
    components = set()
    for peer in shared["spec"]["ingress"][0]["from"]:
        for e in (peer.get("podSelector") or {}).get("matchExpressions") or []:
            if e["key"] == "app.kubernetes.io/component":
                components |= set(e["values"])
    assert components == set(CLIENTS), f"the shared Ray cluster admits {sorted(components)}"


# ------------------------------------------------- 3. MLflow server auth

def _mlflow(docs):
    return _workload(docs, "mlflow")["spec"]["template"]["spec"]


def test_mlflow_server_runs_basic_auth_with_credentials_from_the_secret(tmp_path) -> None:
    pod = _mlflow(_render())
    (container,) = pod["containers"]
    script = container["args"][0]
    assert re.search(r"exec mlflow server\b[^\n]*(?:\\\n[^\n]*)*--app-name basic-auth", script), (
        "the tracking server does not run MLflow's basic-auth app")
    env = _env(container)
    for name, key in (("MLFLOW_AUTH_ADMIN_USERNAME", "MLFLOW_TRACKING_USERNAME"),
                      ("MLFLOW_AUTH_ADMIN_PASSWORD", "MLFLOW_TRACKING_PASSWORD"),
                      ("MLFLOW_FLASK_SERVER_SECRET_KEY", "MLFLOW_FLASK_SERVER_SECRET_KEY")):
        ref = _secret_ref(env.get(name))
        assert ref == {"name": "avaloka-secrets", "key": key}, f"{name} must come from the release Secret: {env.get(name)}"
    assert "envFrom" not in container, "the MLflow server gets named keys, not a whole Secret"

    # The script, run against a stub mlflow: upgrade, the auth ini (no password in
    # it, owner-only), then the basic-auth server.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    (bindir / "mlflow").write_text(f'#!/bin/sh\necho "$*" >> "{log}"\nexit 0\n')
    (bindir / "mlflow").chmod(0o755)
    ini = tmp_path / "auth" / "basic_auth.ini"
    ini.parent.mkdir()
    pw = "pw-" + secrets.token_hex(16)
    db_pw = "p%40" + secrets.token_hex(4)          # a URL-encoded "@": the % must reach configparser as %%
    run_env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "MLFLOW_BACKEND_STORE_URI": "postgresql://m@db/m",
        "MLFLOW_ARTIFACTS_DESTINATION": "s3://b/a",
        "MLFLOW_PORT": "5000", "MLFLOW_WORKERS": "1", "MLFLOW_ALLOWED_HOSTS": "x",
        "MLFLOW_AUTH_CONFIG_PATH": str(ini),
        "MLFLOW_AUTH_DEFAULT_PERMISSION": env["MLFLOW_AUTH_DEFAULT_PERMISSION"]["value"],
        "MLFLOW_AUTH_DATABASE_URI": f"postgresql://u:{db_pw}@db/auth",
        "MLFLOW_AUTH_ADMIN_USERNAME": "avaloka", "MLFLOW_AUTH_ADMIN_PASSWORD": pw,
    }
    run = subprocess.run(["/bin/sh", "-c", script], env=run_env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    calls = log.read_text().splitlines()
    assert calls[0] == "db upgrade postgresql://m@db/m" and calls[1].endswith("--app-name basic-auth"), calls
    text = ini.read_text()
    escaped = db_pw.replace("%", "%%")
    assert f"database_uri = postgresql://u:{escaped}@db/auth" in text, "a % in the URI must be escaped for configparser"
    # The admin password is in the ini (owner-only), so even an mlflow that ignores
    # MLFLOW_AUTH_ADMIN_PASSWORD (3.14's packaged ini says password1234) uses it.
    assert f"admin_password = {pw}" in text and "password1234" not in text, "the ini does not pin the admin password"
    assert pw not in run.stdout + run.stderr, "the admin password was printed"
    assert stat.S_IMODE(ini.stat().st_mode) & 0o077 == 0, "the auth ini is readable by others"


def test_the_image_the_mlflow_pod_runs_has_the_basic_auth_dependencies() -> None:
    """The server runs from the API image (values: mlflow.image defaults to image.*),
    and `--app-name basic-auth` imports Flask-WTF, which only mlflow's [auth] extra
    pulls in: without it the pod exits with ImportError on start. Found by running
    the rendered container script against a real mlflow 3.17.0."""
    pod = _mlflow(_render())
    (container,) = pod["containers"]
    values = yaml.safe_load((CHART / "values.yaml").read_text())
    assert container["image"] == f"{values['image']['repository']}:{values['image']['tag']}"
    dockerfile = (ROOT / "deploy" / "docker" / "Dockerfile.api").read_text()
    assert "requirements.txt" in dockerfile
    reqs = (ROOT / "requirements.txt").read_text()
    pin = re.search(r"^mlflow(\[[^\]]*\])?==([0-9.]+)\s*$", reqs, re.M)
    assert pin, "requirements.txt does not pin mlflow"
    extras = {e.strip() for e in (pin.group(1) or "[]")[1:-1].split(",") if e.strip()}
    assert "auth" in extras or re.search(r"(?im)^flask[-_]wtf\b", reqs), (
        "the API image has no Flask-WTF: mlflow's basic-auth app cannot start")


def test_mlflow_pod_is_hardened() -> None:
    pod = _mlflow(_render())
    (container,) = pod["containers"]
    assert pod.get("automountServiceAccountToken") is False
    sc = {**(pod.get("securityContext") or {}), **(container.get("securityContext") or {})}
    assert sc.get("runAsNonRoot") is True and sc.get("runAsUser") not in (None, 0)
    assert sc.get("readOnlyRootFilesystem") is True and sc.get("allowPrivilegeEscalation") is False
    assert (sc.get("capabilities") or {}).get("drop") == ["ALL"]


# ---------------------------------------- 4. no default, generated, kept

def test_no_default_mlflow_password_is_ever_rendered() -> None:
    a, b = _release_secret(_render()), _release_secret(_render())
    for key in ("MLFLOW_TRACKING_PASSWORD", "MLFLOW_FLASK_SERVER_SECRET_KEY", "RAY_AUTH_TOKEN"):
        assert _usable(a.get(key)), f"{key} renders empty, short or published"
        assert a[key] != b[key], f"{key} renders the same on every install: a default, not a secret"
    for published in ("password1234", "password"):
        given = _release_secret(_render("--set-string", f"secrets.mlflowAdminPassword={published}"))
        assert given["MLFLOW_TRACKING_PASSWORD"] != published, f"the published MLflow password {published!r} was kept"
    values = yaml.safe_load((CHART / "values.yaml").read_text())
    assert values["secrets"]["mlflowAdminPassword"] == "" and values["secrets"]["rayAuthToken"] == ""


def test_service_secrets_are_kept_on_upgrade(tmp_path) -> None:
    chart = tmp_path / "avaloka"
    shutil.copytree(CHART, chart)
    helpers = chart / "templates" / "_helpers.tpl"
    text = helpers.read_text()
    assert LOOKUP in text
    kept = {k: "kept-" + secrets.token_hex(20)
            for k in ("MLFLOW_TRACKING_PASSWORD", "MLFLOW_FLASK_SERVER_SECRET_KEY", "RAY_AUTH_TOKEN")}
    data = " ".join(f'"{k}" "{base64.b64encode(v.encode()).decode()}"' for k, v in kept.items())
    helpers.write_text(text.replace(LOOKUP, f'(dict "data" (dict {data}))'))
    docs = _render(chart=chart)
    rendered = _release_secret(docs)
    assert {k: rendered.get(k) for k in kept} == kept, "an upgrade rotated an MLflow or Ray secret"
    ray_auth = _find(docs, "Secret", "avaloka-ray-auth").get("stringData") or {}
    assert ray_auth == {"auth_token": kept["RAY_AUTH_TOKEN"]}, "the Ray pods' Secret differs from the clients' token"


def test_notes_name_the_secrets_but_never_print_them() -> None:
    pw, token = "pw-" + secrets.token_hex(16), "tok-" + secrets.token_hex(16)
    out = _helm("--set-string", f"secrets.mlflowAdminPassword={pw}",
                "--set-string", f"secrets.rayAuthToken={token}", cmd="install")
    notes = out.split("NOTES:", 1)[-1]
    assert "MLFLOW_TRACKING_PASSWORD" in notes and "avaloka-ray-auth" in notes, "NOTES do not say where they are"
    assert pw not in notes and token not in notes, "NOTES printed a secret"


# ------------------------------------------------ 5. Ray pods and jobs

def test_shared_ray_cluster_and_service_require_the_token() -> None:
    _assert_ray_auth(_cluster_spec(_ray_manifest("raycluster.yaml")), "avaloka-ray-auth", "raycluster")
    _assert_ray_auth(_cluster_spec(_ray_manifest("rayservice.yaml")), "avaloka-ray-auth", "rayservice")
    docs = _render()
    assert _find(docs, "ConfigMap", "avaloka-config")["data"].get("RAY_AUTH_MODE") == "token"
    ray_auth = _find(docs, "Secret", "avaloka-ray-auth")["stringData"]
    assert ray_auth["auth_token"] == _release_secret(docs)["RAY_AUTH_TOKEN"], (
        "one token for clients, pods and the operator")


def _render_job(path: Path, secret: str) -> dict:
    text = path.read_text()
    for placeholder, value in (("__CLOUD_SECRET_NAME__", secret), ("__RAYJOB_NAME__", "job1"),
                               ("__INJECTION_SCRIPT__", "print(1)"), ("__SECRET_ENV_REFS__", "")):
        text = text.replace(placeholder, value)
    text = re.sub(r"__[A-Z0-9_]+__", "x", text)
    return _docs(text)[0]


def test_rayjob_templates_require_the_token_from_the_jobs_own_secret() -> None:
    _assert_ray_auth(_cluster_spec(_render_job(RAYJOB, "job1-env")), "job1-env", "rayjob")
    _assert_ray_auth(_cluster_spec(_render_job(RAYJOB_DTA, "job1-env")), "job1-env", "rayjob-dta")
    for path in (RAYJOB, RAYJOB_DTA):
        spec = _cluster_spec(_render_job(path, "job1-env"))
        major_minor = tuple(int(x) for x in spec["rayVersion"].split(".")[:2])
        assert major_minor >= (2, 52), f"{path.name}: Ray {spec['rayVersion']} has no token auth"
        for _g, c in _ray_containers(spec):
            tag = c["image"].rsplit(":", 1)[-1]
            if re.match(r"^\d+\.\d+", tag):
                assert tuple(int(x) for x in tag.split(".")[:2]) >= (2, 52), f"{path.name}: image {c['image']}"


class _Kubectl:
    def __init__(self):
        self.secrets, self.applied = [], []

    def __call__(self, cmd, *a, **k):
        args = list(cmd[1:])
        stdin = k.get("input") or ""
        if args[:1] == ["create"] and stdin:
            self.secrets.append(json.loads(stdin))
        out = "uid-1" if args[:2] == ["get", "rayjob"] else ""
        return subprocess.CompletedProcess(cmd, 0, out, "")


def _kuberay_run(monkeypatch, env: dict):
    from app.infra import ray_job_runner as rjr

    for k in ("RAY_DASHBOARD_URL", "RAY_AUTH_MODE", "RAY_AUTH_TOKEN", "RAY_AUTH_TOKEN_PATH"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    fake = _Kubectl()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(rjr, "_which_kubectl", lambda: "kubectl", raising=False)
    applied = {}
    monkeypatch.setattr(rjr, "apply_yaml_text", lambda y, namespace="default": applied.setdefault("yaml", y))
    monkeypatch.setattr(rjr, "wait_for_rayjob", lambda *a, **k: {"status": {"jobStatus": "SUCCEEDED"}})
    monkeypatch.setattr(rjr, "fetch_rayjob_logs", lambda *a, **k: "")
    monkeypatch.setattr(rjr, "cleanup_rayjob_code_configmap", lambda *a, **k: None)
    monkeypatch.setattr(rjr, "_get_current_context", lambda: "test")
    yaml_text = RAYJOB.read_text().replace("__CLOUD_SECRET_NAME__", "job1-env")
    yaml_text = re.sub(r"__[A-Z0-9_]+__", "job1", yaml_text)
    outcome = rjr.run_rayjob_from_yaml(yaml_text=yaml_text, namespace="ray-training", rayjob_name="job1",
                                       secret_env={"AWS_ACCESS_KEY_ID": "AKIA-test"}, secret_name="job1-env")
    return outcome, fake, applied


def test_kuberay_rayjob_gets_the_token_in_its_own_secret_never_in_the_cr(monkeypatch, caplog) -> None:
    token = "tok-" + secrets.token_hex(16)
    caplog.set_level("DEBUG")
    outcome, fake, applied = _kuberay_run(monkeypatch, {"RAY_AUTH_MODE": "token", "RAY_AUTH_TOKEN": token})
    assert outcome.status == "SUCCEEDED", outcome.details
    assert len(fake.secrets) == 1
    data = fake.secrets[0]["stringData"]
    assert data.get("auth_token") == token, "the job's Secret does not carry the Ray token"
    assert data.get("AWS_ACCESS_KEY_ID") == "AKIA-test"
    assert token not in applied["yaml"], "the token was rendered into the RayJob CR"
    assert "authOptions" in applied["yaml"]
    assert token not in caplog.text, "the token was logged"


def test_kuberay_rayjob_refuses_to_start_without_a_token_when_auth_is_on(monkeypatch) -> None:
    outcome, fake, applied = _kuberay_run(monkeypatch, {"RAY_AUTH_MODE": "token"})
    assert outcome.status == "ERROR" and "RAY_AUTH_TOKEN" in str(outcome.details), outcome.details
    assert "yaml" not in applied and not fake.secrets, "a job was applied that could never authenticate"


def test_kuberay_rayjob_without_auth_mode_drops_the_auth_blocks(monkeypatch) -> None:
    outcome, fake, applied = _kuberay_run(monkeypatch, {})
    assert outcome.status == "SUCCEEDED", outcome.details
    assert "authOptions" not in applied["yaml"] and "RAY_AUTH" not in applied["yaml"], (
        "a job demanding a token its Secret does not have would never start")
    assert "auth_token" not in fake.secrets[0]["stringData"]


# ------------------------------------------------ 6. clients send them

def test_app_workloads_get_the_mlflow_and_ray_credentials() -> None:
    docs = _render()
    secret = _release_secret(docs)
    for key in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "RAY_AUTH_TOKEN"):
        assert secret.get(key), f"the release Secret has no {key}"
    config = _find(docs, "ConfigMap", "avaloka-config")["data"]
    assert config.get("RAY_AUTH_MODE") == "token" and config.get("MLFLOW_TRACKING_URI")
    for component in CLIENTS:
        (container, *_) = _workload(docs, component)["spec"]["template"]["spec"]["containers"]
        sources = {(s.get("secretRef") or s.get("configMapRef") or {}).get("name") for s in container.get("envFrom") or []}
        assert {"avaloka-secrets", "avaloka-config"} <= sources, f"{component} loads {sources}"
    for name in ("raycluster.yaml", "rayservice.yaml"):
        for group, c in _ray_containers(_cluster_spec(_ray_manifest(name))):
            env = _env(c)
            for key in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
                assert _secret_ref(env.get(key)).get("key") == key, f"{name} {group}: no {key}"


def test_the_apps_mlflow_client_sends_the_credentials(server, monkeypatch) -> None:
    """Passes before this change too: the mlflow client reads the variables itself.
    It pins the mechanism the chart relies on, not the fix."""
    handler, url = server
    handler.status, handler.body = 404, json.dumps({"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "x"}).encode()
    pw = "pw-" + secrets.token_hex(12)
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "avaloka")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", pw)
    from app.agents.mta_v2.mlflow_manager import MLflowManager

    MLflowManager(tracking_uri=url, experiment_name="e")
    expected = "Basic " + base64.b64encode(f"avaloka:{pw}".encode()).decode()
    assert handler.seen and all(h == expected for _m, _p, h in handler.seen), handler.seen


def test_the_apps_ray_dashboard_calls_send_the_token(server, monkeypatch) -> None:
    handler, url = server
    token = "tok-" + secrets.token_hex(12)
    monkeypatch.setenv("RAY_AUTH_MODE", "token")
    monkeypatch.setenv("RAY_AUTH_TOKEN", token)
    monkeypatch.setenv("RAY_DASHBOARD_URL", url)
    handler.body = json.dumps({"submission_id": "s1", "status": "SUCCEEDED", "logs": "",
                               "ray_version": "2.52.1", "data": {"summary": []}}).encode()
    from app.infra import ray_job_runner, ray_jobs_service
    from app.agents.mta_v2.gcp import cancel_ray_job, get_job_result, submit_ray_job, wait_for_ray_job_completion

    ray_jobs_service.cluster_status()
    ray_job_runner._preflight_dashboard(url)
    submit_ray_job.submit_ray_job(url, {})
    wait_for_ray_job_completion.get_ray_job_status(url, "s1")
    get_job_result.get_job_logs(url, "s1", retries=1, delay=0)
    handler.body = json.dumps({"status": "RUNNING"}).encode()
    cancel_ray_job.cancel_ray_job(url, "s1", max_retries=1)
    paths = sorted({p.split("?")[0] for _m, p, _h in handler.seen})
    assert {"/api/version", "/api/jobs/", "/api/jobs/s1", "/api/jobs/s1/logs"} <= set(paths), paths
    unauthenticated = [(m, p) for m, p, h in handler.seen if h != f"Bearer {token}"]
    assert not unauthenticated, f"sent without the Ray token: {unauthenticated}"


def test_a_dashboard_that_refuses_the_token_fails_the_preflight(server, monkeypatch) -> None:
    handler, url = server
    handler.status = 403
    monkeypatch.setenv("RAY_AUTH_MODE", "token")
    monkeypatch.setenv("RAY_AUTH_TOKEN", "wrong-" + secrets.token_hex(8))
    from app.infra import ray_job_runner

    with pytest.raises(PermissionError):
        ray_job_runner._preflight_dashboard(url)


def test_legacy_gke_ray_cluster_is_created_with_the_token(monkeypatch) -> None:
    from app.agents.mta_v2.gcp import create_ray_cluster as crc

    token = "tok-" + secrets.token_hex(12)
    monkeypatch.setenv("RAY_AUTH_MODE", "token")
    monkeypatch.setenv("RAY_AUTH_TOKEN", token)
    created = {}

    class _Custom:
        def __init__(self, *_a):
            pass

        def create_namespaced_custom_object(self, **kw):
            created["cluster"] = kw["body"]

        def get_namespaced_custom_object(self, **kw):
            return {"status": {"state": "ready"}}

    class _Core:
        def __init__(self, *_a):
            pass

        def create_namespaced_secret(self, namespace, body):
            created["secret"] = body

    monkeypatch.setattr(crc.client, "CustomObjectsApi", _Custom)
    monkeypatch.setattr(crc.client, "CoreV1Api", _Core, raising=False)
    crc.create_ray_cluster(object(), "rc", namespace="ns1")
    spec = created["cluster"]["spec"]
    secret = created.get("secret")
    assert secret is not None, "no token Secret was created for the LoadBalancer-published cluster"
    assert secret.string_data == {"auth_token": token}
    _assert_ray_auth(spec, secret.metadata.name, "legacy-gke")
    assert token not in json.dumps(spec), "the token was rendered into the RayCluster CR"


# ------------------------------------------------ 7. the operator can follow

def test_kuberay_operator_supports_token_auth_and_crds_are_upgraded(monkeypatch) -> None:
    from app.infra import ray_manager

    major, minor = (int(x) for x in ray_manager.KUBERAY_OPERATOR_VERSION.split(".")[:2])
    assert (major, minor) >= (1, 6), (
        f"KubeRay {ray_manager.KUBERAY_OPERATOR_VERSION} cannot read a token-authenticated dashboard "
        "from a named Secret: RayJobs and the RayService would never report status")
    calls = []
    monkeypatch.setattr(ray_manager, "run_command",
                        lambda cmd, step: calls.append(cmd) or {"status": "SUCCESS", "step_name": step,
                                                                 "message": "", "details": ""})
    ray_manager.install_kuberay_operator(namespace="ns")
    flat = [" ".join(c) for c in calls]
    crd = [i for i, c in enumerate(flat) if "show crds" in c and "--server-side" in c]
    upgrade = [i for i, c in enumerate(flat) if c.startswith("helm upgrade --install kuberay-operator")]
    assert crd and upgrade and crd[0] < upgrade[0], f"CRDs are not applied before the operator upgrade: {flat}"
