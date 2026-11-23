"""The LangGraph server is internal-only, and refuses calls without its token.

The server (`langgraph dev`, langgraph.json) has no authentication of its own.
The API is its only caller and uses it as a thread registry (app/api/server.py
``lg_request``). Before this change the avaloka-router chart routed the public
host ``ingress.hosts.langgraph`` to it, and so did the legacy backend chart's
own Ingress.

What these tests hold:

1. no chart in the repository renders a public route (Ingress, HTTPRoute,
   LoadBalancer or NodePort Service) to a LangGraph Service under its default
   values, its own overlays, or blue-green mode;
2. a NetworkPolicy admits only the api pods to the LangGraph port (main chart),
   and no other pod to the LangGraph container port (legacy chart, where the
   api container in the same pod is the caller);
3. LANGGRAPH_SERVICE_TOKEN is generated into a Secret and reaches the API and
   the server by secretKeyRef;
4. a real ``langgraph dev``, started with the repository's langgraph.json,
   answers 401 without the token (including the routes langgraph's own auth
   hook leaves open, and the Studio header that skips it) and works with it;
5. the API's LangGraph client sends the token.

Every token here is generated per run and never printed. On the tree without
the change every test fails on an assertion, not on an import: the new module
is reached only through langgraph.json, and helpers return None or {} instead
of raising.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import pathlib
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_CHART = ROOT / "deploy" / "helm" / "avaloka"
LEGACY_BACKEND = ROOT / "deploy" / "avaloka" / "charts" / "avaloka-backend"
TOKEN = "LANGGRAPH_SERVICE_TOKEN"


def _ci() -> bool:
    return os.environ.get("CI", "").strip().lower() not in ("", "0", "false")


def _need(ok: bool, what: str) -> None:
    """Skip on a laptop without the tool; fail on CI, where a skip would read as a pass."""
    if ok:
        return
    if _ci():
        pytest.fail(f"{what} is not available; CI is set, so this is a failure rather than a skip.")
    pytest.skip(f"{what} is not available")


# --------------------------------------------------------------------------- rendering
def _helm() -> str:
    helm = shutil.which("helm")
    _need(helm is not None, "helm")
    return helm


def _charts() -> list[pathlib.Path]:
    """Every Helm chart in the repository (a directory holding Chart.yaml)."""
    found = sorted(p.parent for p in ROOT.rglob("Chart.yaml")
                   if "node_modules" not in p.parts and ".git" not in p.parts)
    return found


def _render(chart: pathlib.Path, tmp: pathlib.Path, *args: str):
    """`helm template` of a copy of ``chart`` -> (rc, docs, stderr).

    Dependencies fetched from a remote repository (the umbrella's bitnami redis)
    are dropped from the copy: they are not in the repository and route nothing
    to LangGraph. The chart's own subcharts (file://, under charts/) stay.
    """
    import yaml

    helm = _helm()
    work = tmp / ("c-" + secrets.token_hex(4))
    shutil.copytree(chart, work)
    meta = yaml.safe_load((work / "Chart.yaml").read_text())
    deps = meta.get("dependencies") or []
    local = [d for d in deps if str(d.get("repository", "")).startswith("file://")]
    if len(local) != len(deps):
        meta["dependencies"] = local
        (work / "Chart.yaml").write_text(yaml.safe_dump(meta))
        (work / "Chart.lock").unlink(missing_ok=True)
    run = subprocess.run([helm, "template", "t", str(work), *args],
                         capture_output=True, text=True, timeout=180)
    docs = [d for d in yaml.safe_load_all(run.stdout) if d] if run.returncode == 0 else []
    return run.returncode, docs, run.stderr


def _variants(chart: pathlib.Path) -> list[tuple[str, list[str]]]:
    out = [("defaults", [])]
    for overlay in sorted((chart / "values").glob("*.yaml")) if (chart / "values").is_dir() else []:
        out.append((overlay.name, ["-f", str(overlay)]))
    values = (chart / "values.yaml").read_text() if (chart / "values.yaml").exists() else ""
    if re.search(r"^color:", values, re.M):
        out.append(("color=blue", ["--set", "color=blue"]))
    return out


def _is_langgraph_service(svc: dict) -> bool:
    meta = svc.get("metadata") or {}
    labels = meta.get("labels") or {}
    selector = (svc.get("spec") or {}).get("selector") or {}
    ports = (svc.get("spec") or {}).get("ports") or []
    return ("langgraph" in str(meta.get("name", "")).lower()
            or labels.get("app.kubernetes.io/component") == "langgraph"
            or labels.get("avaloka.io/port") == "langgraph"
            or selector.get("app.kubernetes.io/component") == "langgraph"
            or any("langgraph" in str(p.get("name", "")) or "langgraph" in str(p.get("targetPort", ""))
                   for p in ports))


def public_langgraph_routes(docs: list) -> list[str]:
    """Every object in a render that exposes a LangGraph Service outside the cluster."""
    services = {d["metadata"]["name"]: d for d in docs if d.get("kind") == "Service"}

    def targets_langgraph(name) -> bool:
        name = str(name or "")
        return "langgraph" in name.lower() or (name in services and _is_langgraph_service(services[name]))

    bad = []
    for d in docs:
        kind, name = d.get("kind"), (d.get("metadata") or {}).get("name")
        spec = d.get("spec") or {}
        if kind == "Ingress":
            backends = [spec.get("defaultBackend") or {}]
            for rule in spec.get("rules") or []:
                if "langgraph" in str(rule.get("host", "")).lower():
                    bad.append(f"Ingress {name}: host {rule.get('host')}")
                for path in ((rule.get("http") or {}).get("paths") or []):
                    backends.append(path.get("backend") or {})
            for b in backends:
                svc = (b.get("service") or {}).get("name") or b.get("serviceName")
                if svc and targets_langgraph(svc):
                    bad.append(f"Ingress {name} -> Service {svc}")
        elif kind in ("HTTPRoute", "GRPCRoute", "TLSRoute", "TCPRoute"):
            for rule in spec.get("rules") or []:
                for ref in rule.get("backendRefs") or []:
                    if targets_langgraph(ref.get("name")):
                        bad.append(f"{kind} {name} -> {ref.get('name')}")
        elif kind == "Service" and spec.get("type") in ("LoadBalancer", "NodePort"):
            if _is_langgraph_service(d):
                bad.append(f"Service {name} is type {spec.get('type')}")
    return bad


def test_the_route_detector_sees_a_route_to_langgraph() -> None:
    """The guard below is not vacuous: a route like the one the router shipped is caught."""
    svc = {"kind": "Service", "metadata": {"name": "avaloka-langgraph-active",
                                           "labels": {"avaloka.io/port": "langgraph"}},
           "spec": {"type": "ClusterIP", "ports": [{"port": 3000}]}}
    ing = {"kind": "Ingress", "metadata": {"name": "r"}, "spec": {"rules": [
        {"host": "x.example", "http": {"paths": [{"backend": {"service": {"name": "avaloka-langgraph-active"}}}]}}]}}
    assert public_langgraph_routes([svc, ing]) == ["Ingress r -> Service avaloka-langgraph-active"]
    lb = {"kind": "Service", "metadata": {"name": "lg", "labels": {"app.kubernetes.io/component": "langgraph"}},
          "spec": {"type": "LoadBalancer"}}
    assert public_langgraph_routes([lb]) == ["Service lg is type LoadBalancer"]


def test_every_chart_in_the_repository_is_rendered() -> None:
    names = {str(c.relative_to(ROOT)) for c in _charts()}
    assert {"deploy/helm/avaloka", "deploy/avaloka", "deploy/avaloka-router",
            "deploy/avaloka/charts/avaloka-backend"} <= names, sorted(names)


@pytest.mark.parametrize("chart", [str(c.relative_to(ROOT)) for c in _charts()])
def test_no_chart_renders_a_public_route_to_langgraph(chart, tmp_path) -> None:
    path = ROOT / chart
    rendered = 0
    problems = {}
    for label, args in _variants(path):
        code, docs, err = _render(path, tmp_path, *args)
        if label == "defaults":
            assert code == 0, f"{chart} does not render with its defaults: {err[-400:]}"
        if code != 0:
            continue  # an overlay that needs values it does not carry is not a default install
        rendered += 1
        found = public_langgraph_routes(docs)
        if found:
            problems[label] = found
    assert rendered >= 1
    assert not problems, f"{chart} routes the LangGraph server publicly: {problems}"


def test_the_router_ignores_a_langgraph_host_left_in_old_values(tmp_path) -> None:
    router = ROOT / "deploy" / "avaloka-router"
    code, docs, err = _render(router, tmp_path, "--set", "ingress.hosts.langgraph=lg.example.com")
    assert code == 0, err[-400:]
    assert public_langgraph_routes(docs) == []
    assert not [d for d in docs if d.get("kind") == "Service" and _is_langgraph_service(d)], \
        "the router still renders a Service for the LangGraph server"


# --------------------------------------------------------------------------- NetworkPolicy
def _pod_labels(docs, name: str) -> dict:
    for d in docs:
        if d.get("kind") == "Deployment" and d["metadata"]["name"] == name:
            return d["spec"]["template"]["metadata"].get("labels") or {}
    return {}


def _selects(selector: dict, labels: dict) -> bool:
    match = (selector or {}).get("matchLabels") or {}
    return bool(match) and all(labels.get(k) == v for k, v in match.items()) and \
        not (selector or {}).get("matchExpressions")


def test_main_chart_admits_only_the_api_pods_to_the_langgraph_port(tmp_path) -> None:
    code, docs, err = _render(MAIN_CHART, tmp_path)
    assert code == 0, err[-400:]
    lg = _pod_labels(docs, "avaloka-langgraph")
    api = _pod_labels(docs, "avaloka")
    assert lg and api, "the api or langgraph Deployment is missing"
    policies = [d for d in docs if d.get("kind") == "NetworkPolicy"
                and _selects(d["spec"].get("podSelector"), lg)]
    assert len(policies) == 1, f"expected one NetworkPolicy selecting the langgraph pods, got {len(policies)}"
    spec = policies[0]["spec"]
    assert "Ingress" in (spec.get("policyTypes") or [])
    rules = spec.get("ingress") or []
    assert rules, "an Ingress policy with no rules admits nothing; the API could not reach it"
    for rule in rules:
        peers = rule.get("from") or []
        assert peers, "a rule with no 'from' admits every pod"
        for peer in peers:
            assert set(peer) == {"podSelector"}, f"peer other than a pod selector: {peer}"
            assert _selects(peer["podSelector"], api), f"peer does not select the api pods: {peer}"
            assert peer["podSelector"]["matchLabels"].get("app.kubernetes.io/component") == "api", peer
        assert [p.get("port") for p in rule.get("ports") or []] == [2024], rule.get("ports")
    for other in ("avaloka-celery", "avaloka-celery-redbeat", "avaloka-celery-executor", "avaloka-mcp"):
        labels = _pod_labels(docs, other)
        if labels:
            assert not any(_selects(p["podSelector"], labels) for r in rules for p in r["from"]), other


@pytest.mark.parametrize("color", ["", "blue"])
def test_legacy_chart_admits_no_other_pod_to_the_langgraph_container(color, tmp_path) -> None:
    args = ["--set", f"color={color}"] if color else []
    code, docs, err = _render(LEGACY_BACKEND, tmp_path, *args)
    assert code == 0, err[-400:]
    deployments = [d for d in docs if d.get("kind") == "Deployment"]
    assert len(deployments) == 1
    pod = deployments[0]["spec"]["template"]
    labels = pod["metadata"]["labels"]
    containers = {c["name"]: c for c in pod["spec"]["containers"]}
    lg_ports = {p["containerPort"] for p in containers["langgraph"].get("ports") or []}
    api_ports = {p["containerPort"] for p in containers["api"].get("ports") or []}
    assert lg_ports and api_ports
    policies = [d for d in docs if d.get("kind") == "NetworkPolicy" and _selects(d["spec"].get("podSelector"), labels)]
    assert policies, "no NetworkPolicy selects the backend pod, so any pod reaches the LangGraph port"
    for pol in policies:
        assert "Ingress" in (pol["spec"].get("policyTypes") or [])
        for rule in pol["spec"].get("ingress") or []:
            ports = {p.get("port") for p in rule.get("ports") or []}
            assert ports, "a rule with no ports admits every port, the LangGraph one included"
            assert not ports & lg_ports, f"{pol['metadata']['name']} admits the LangGraph port"
    admitted = {p.get("port") for pol in policies for r in pol["spec"].get("ingress") or [] for p in r.get("ports") or []}
    assert api_ports <= admitted, "the API port must stay reachable"


# --------------------------------------------------------------------------- token wiring
def _secret_key_ref(container: dict, name: str):
    for e in container.get("env") or []:
        if e.get("name") == name:
            return ((e.get("valueFrom") or {}).get("secretKeyRef")) or {}
    return None


def test_main_chart_generates_the_token_and_both_ends_read_it_by_secret_key_ref(tmp_path) -> None:
    code, docs, err = _render(MAIN_CHART, tmp_path)
    assert code == 0, err[-400:]
    secret = next((d for d in docs if d.get("kind") == "Secret" and d["metadata"]["name"] == "avaloka-secrets"), {})
    value = (secret.get("stringData") or {}).get(TOKEN, "")
    assert len(value) >= 32, "the release Secret carries no generated LANGGRAPH_SERVICE_TOKEN"
    _, docs2, _ = _render(MAIN_CHART, tmp_path)
    secret2 = next((d for d in docs2 if d.get("kind") == "Secret" and d["metadata"]["name"] == "avaloka-secrets"), {})
    assert (secret2.get("stringData") or {}).get(TOKEN) != value, "the token renders the same on every install"
    for workload in ("avaloka", "avaloka-langgraph"):
        dep = next((d for d in docs if d.get("kind") == "Deployment" and d["metadata"]["name"] == workload), None)
        assert dep is not None, workload
        refs = [_secret_key_ref(c, TOKEN) for c in dep["spec"]["template"]["spec"]["containers"]]
        assert {"name": "avaloka-secrets", "key": TOKEN} in refs, f"{workload} does not read {TOKEN} by secretKeyRef"


def test_main_chart_takes_a_given_token_and_replaces_a_published_one(tmp_path) -> None:
    given = "given-" + secrets.token_hex(20)
    for value, kept in ((given, True), ("changeme", False)):
        code, docs, err = _render(MAIN_CHART, tmp_path, "--set-string", f"secrets.langgraphServiceToken={value}")
        assert code == 0, err[-400:]
        got = next((d["stringData"] for d in docs if d.get("kind") == "Secret"
                    and d["metadata"]["name"] == "avaloka-secrets"), {}).get(TOKEN, "")
        if kept:
            assert got == value, "the given token was not used"
        else:
            assert got != value and len(got) >= 32, "the chart rendered a published token"


def test_legacy_chart_generates_the_token_keeps_it_on_upgrade_and_both_containers_read_it(tmp_path) -> None:
    code, docs, err = _render(LEGACY_BACKEND, tmp_path, "--set", "color=green")
    assert code == 0, err[-400:]
    secrets_ = {d["metadata"]["name"]: d for d in docs if d.get("kind") == "Secret"}
    holder = [n for n, d in secrets_.items() if TOKEN in (d.get("stringData") or {})]
    assert len(holder) == 1, f"no Secret carries {TOKEN}: {sorted(secrets_)}"
    assert len(secrets_[holder[0]]["stringData"][TOKEN]) >= 32
    pod = next(d for d in docs if d.get("kind") == "Deployment")["spec"]["template"]["spec"]
    for c in pod["containers"]:
        assert _secret_key_ref(c, TOKEN) == {"name": holder[0], "key": TOKEN}, f"container {c['name']}"

    # helm template cannot reach a cluster: stub the lookup with the Secret an install left.
    chart = tmp_path / "backend"
    shutil.copytree(LEGACY_BACKEND, chart)
    tpl = chart / "templates" / "langgraph-internal.yaml"
    text = tpl.read_text()
    lookup = 'lookup "v1" "Secret" .Release.Namespace $name'
    assert lookup in text
    kept = "kept-" + secrets.token_hex(20)
    tpl.write_text(text.replace(lookup, f'(dict "data" (dict "{TOKEN}" "{base64.b64encode(kept.encode()).decode()}"))'))
    code, docs, err = _render(chart, tmp_path, "--set", "color=green")
    assert code == 0, err[-400:]
    got = next(d for d in docs if d.get("kind") == "Secret" and TOKEN in (d.get("stringData") or {}))
    assert got["stringData"][TOKEN] == kept, "an upgrade rotated the LangGraph token"


def test_the_install_notes_never_print_the_token(tmp_path) -> None:
    helm = _helm()
    run = subprocess.run([helm, "install", "t", str(MAIN_CHART), "--dry-run=client"],
                         capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stderr[-400:]
    manifest, _, notes = run.stdout.partition("NOTES:")
    m = re.search(rf"^\s*{TOKEN}:\s*\"?([A-Za-z0-9]+)\"?\s*$", manifest, re.M)
    assert m and len(m.group(1)) >= 32, "the release Secret carries no generated token"
    assert TOKEN in notes, "NOTES does not say the token was generated"
    assert m.group(1) not in notes
    assert run.stdout.count(m.group(1)) == 1, "the token appears outside the Secret (NOTES or another object)"


# --------------------------------------------------------------------------- the server
def _config() -> dict:
    return json.loads((ROOT / "langgraph.json").read_text())


def _gate_app_path() -> str:
    return str((_config().get("http") or {}).get("app") or "")


def _load_gate(monkeypatch, tmp_path):
    """The http.app langgraph.json names, loaded as langgraph-api loads it; None when there is none."""
    ref = _gate_app_path()
    if not ref:
        return None
    path, _, attr = ref.rpartition(":")
    monkeypatch.setenv("AVALOKA_AUTO_PROVISION", "0")
    monkeypatch.setenv("AVALOKA_ENV_FILE", str(tmp_path / ".env"))
    spec = importlib.util.spec_from_file_location("lg_gate_under_test", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, getattr(module, attr)


def test_langgraph_json_wires_a_gate_in_front_of_every_route() -> None:
    cfg = _config()
    assert _gate_app_path(), "langgraph.json names no http.app: the server has no authentication"
    assert (cfg.get("http") or {}).get("middleware_order") != "auth_first"
    assert not cfg.get("auth"), "langgraph's own auth hook leaves /internal/truncate open; do not rely on it"


@pytest.mark.parametrize("configured", ["", "changeme", "   "])
def test_with_no_usable_token_configured_the_gate_refuses_everything(configured, monkeypatch, tmp_path) -> None:
    from starlette.testclient import TestClient

    monkeypatch.setenv(TOKEN, configured)
    loaded = _load_gate(monkeypatch, tmp_path)
    assert loaded is not None, "langgraph.json names no http.app"
    _, app = loaded
    client = TestClient(app)
    for header in ({}, {"Authorization": f"Bearer {configured}"}, {"Authorization": "Bearer "}):
        assert client.post("/threads", headers=header).status_code == 401
    assert client.get("/ok").status_code == 404, "GET /ok should pass the gate (and find no route here)"
    assert not (tmp_path / ".env").exists(), "the gate provisioned with AVALOKA_AUTO_PROVISION=0"


def test_the_gate_compares_the_token_in_constant_time(monkeypatch, tmp_path) -> None:
    import hmac
    from starlette.testclient import TestClient

    token = secrets.token_hex(24)
    monkeypatch.setenv(TOKEN, token)
    loaded = _load_gate(monkeypatch, tmp_path)
    assert loaded is not None, "langgraph.json names no http.app"
    _, app = loaded
    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr(hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    client = TestClient(app)
    assert client.post("/threads", headers={"Authorization": f"Bearer {token}x"}).status_code == 401
    assert client.post("/threads", headers={"Authorization": f"Bearer {token}"}).status_code == 404
    assert len(calls) == 2, "the token was not compared with hmac.compare_digest"


def test_locally_the_gate_generates_the_token_into_dot_env(monkeypatch, tmp_path) -> None:
    """Outside Kubernetes `langgraph dev` and the API share .env; whichever starts first generates it."""
    ref = _gate_app_path()
    assert ref, "langgraph.json names no http.app"
    env = {k: v for k, v in os.environ.items() if k not in (TOKEN, "KUBERNETES_SERVICE_HOST", "AVALOKA_AUTO_PROVISION")}
    env.update(AVALOKA_ENV_FILE=str(tmp_path / ".env"), PYTHONPATH=str(ROOT))
    code = ("import importlib.util,sys; s=importlib.util.spec_from_file_location('g', sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)")
    run = subprocess.run([sys.executable, "-c", code, str(ROOT / ref.rpartition(":")[0])],
                         capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=120)
    assert run.returncode == 0, run.stderr[-400:]
    saved = dict(re.findall(r"^([A-Z_]+)=(.*)$", (tmp_path / ".env").read_text(), re.M)) \
        if (tmp_path / ".env").exists() else {}
    assert len(saved.get(TOKEN, "")) >= 32, "no token was generated into .env"
    assert saved[TOKEN] not in run.stdout + run.stderr, "the generated token was printed"


class _Server:
    def __init__(self, url, token, proc, log):
        self.url, self.token, self.proc, self.log = url, token, proc, log

    @property
    def up(self) -> bool:
        return self.url is not None

    def why(self) -> str:
        try:
            return self.log.read_text()[-1500:]
        except OSError:
            return "(no log)"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def langgraph_server(tmp_path_factory):
    """A real `langgraph dev` with the repository's langgraph.json.

    Only the graph is swapped for a two-line one (the real graph loads the
    embedding models) and the .env file dropped; every other key, ``http``
    included, is the repository's, with its paths made absolute. Its
    environment carries no inherited secret.
    """
    _need(importlib.util.find_spec("langgraph_api") is not None
          and importlib.util.find_spec("langgraph_cli") is not None, "langgraph-cli[inmem]")
    tmp = tmp_path_factory.mktemp("lgdev")
    (tmp / "graph.py").write_text(
        "from typing import TypedDict\nfrom langgraph.graph import StateGraph, END\n"
        "class S(TypedDict):\n    x: int\n"
        "g = StateGraph(S)\ng.add_node('n', lambda s: {'x': 1})\ng.set_entry_point('n')\ng.add_edge('n', END)\n"
        "graph = g.compile()\n")
    cfg = _config()
    cfg["graphs"] = {"avaloka": f"{tmp / 'graph.py'}:graph"}
    cfg.pop("env", None)
    http = cfg.get("http") or {}
    if http.get("app", "").startswith("./"):
        http["app"] = str(ROOT / http["app"][2:])
    (tmp / "langgraph.json").write_text(json.dumps(cfg))
    token = secrets.token_hex(24)
    port = _free_port()
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "TMPDIR") if k in os.environ}
    env.update({TOKEN: token, "PYTHONPATH": str(ROOT), "AVALOKA_AUTO_PROVISION": "0",
                "AVALOKA_ENV_FILE": str(tmp / ".env"), "LANGSMITH_TRACING": "false",
                "LANGGRAPH_CLI_NO_ANALYTICS": "1", "LANGGRAPH_NO_VERSION_CHECK": "true",
                "PYTHONDONTWRITEBYTECODE": "1"})
    log = tmp / "server.log"
    with log.open("w") as fh:
        proc = subprocess.Popen(
            [sys.executable, "-c", "from langgraph_cli.cli import cli; cli()", "dev",
             "--config", str(tmp / "langgraph.json"), "--host", "127.0.0.1", "--port", str(port),
             "--no-browser", "--no-reload"],
            cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    import httpx

    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 180
    ready = False
    while time.time() < deadline and proc.poll() is None:
        try:
            if httpx.get(url + "/ok", timeout=2).status_code == 200:
                ready = True
                break
        except httpx.HTTPError:
            pass
        time.sleep(1)
    server = _Server(url if ready else None, token, proc, log)
    yield server
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def _call(server, method, path, headers=None, body=None):
    import httpx

    return httpx.request(method, server.url + path, headers=headers or {}, json=body, timeout=15)


_REFUSED = [
    ("no header", "POST", "/threads", {}),
    ("search, no header", "POST", "/threads/search", {}),
    ("wrong token", "POST", "/threads", "wrong"),
    ("token as Basic", "POST", "/threads", "basic"),
    ("Studio header", "POST", "/threads", {"x-auth-scheme": "langsmith"}),
    ("truncate (open even to langgraph's auth hook)", "POST", "/internal/truncate", {}),
    ("info", "GET", "/info", {}),
    ("assistants", "POST", "/assistants/search", {}),
]


@pytest.mark.parametrize("label,method,path,headers", _REFUSED, ids=[r[0] for r in _REFUSED])
def test_the_server_refuses_a_call_without_the_token(langgraph_server, label, method, path, headers) -> None:
    assert langgraph_server.up, f"langgraph dev did not start: {langgraph_server.why()}"
    if headers == "wrong":
        headers = {"Authorization": f"Bearer {secrets.token_hex(24)}"}
    elif headers == "basic":
        headers = {"Authorization": f"Basic {langgraph_server.token}"}
    r = _call(langgraph_server, method, path, headers, {})
    assert r.status_code == 401, f"{label}: {method} {path} answered {r.status_code}"


def test_the_server_answers_ok_without_the_token(langgraph_server) -> None:
    assert langgraph_server.up, f"langgraph dev did not start: {langgraph_server.why()}"
    assert _call(langgraph_server, "GET", "/ok").status_code == 200


def test_the_server_creates_lists_and_deletes_threads_with_the_token(langgraph_server) -> None:
    assert langgraph_server.up, f"langgraph dev did not start: {langgraph_server.why()}"
    auth = {"Authorization": f"Bearer {langgraph_server.token}"}
    mark = secrets.token_hex(6)
    r = _call(langgraph_server, "POST", "/threads", auth, {"metadata": {"t": mark}})
    assert r.status_code == 200, r.status_code
    tid = r.json()["thread_id"]
    found = _call(langgraph_server, "POST", "/threads/search", auth, {"metadata": {"t": mark}})
    assert found.status_code == 200 and [t["thread_id"] for t in found.json()] == [tid]
    assert _call(langgraph_server, "DELETE", f"/threads/{tid}", auth).status_code in (200, 204)
    assert _call(langgraph_server, "POST", "/threads/search", auth, {"metadata": {"t": mark}}).json() == []


def test_the_api_client_creates_and_lists_threads_on_the_real_server(langgraph_server, monkeypatch) -> None:
    """lg_json, the path every /threads route of the API takes, against the server above."""
    import asyncio

    import httpx

    assert langgraph_server.up, f"langgraph dev did not start: {langgraph_server.why()}"
    import app.api.config as api_config
    import app.api.server as server

    monkeypatch.setenv(TOKEN, langgraph_server.token)
    mark = secrets.token_hex(6)

    async def go():
        client = httpx.AsyncClient(base_url=langgraph_server.url, timeout=15)
        monkeypatch.setattr(api_config, "http_client", client)
        try:
            created = await server.lg_json("POST", "/threads", json={"metadata": {"t": mark}})
            listed = await server.lg_json("POST", "/threads/search", json={"metadata": {"t": mark}})
            return created, listed
        except Exception as exc:  # noqa: BLE001 - the refusal is the answer on the old tree
            return exc, None
        finally:
            await client.aclose()

    created, listed = asyncio.run(go())
    assert isinstance(created, dict) and created.get("thread_id"), f"thread create failed: {created!r}"[:300]
    assert [t["thread_id"] for t in listed] == [created["thread_id"]]


@pytest.mark.parametrize("shared_client", [True, False])
def test_the_api_client_sends_the_token(shared_client, monkeypatch) -> None:
    import asyncio

    import httpx

    import app.api.config as api_config
    import app.api.server as server

    token = secrets.token_hex(24)
    monkeypatch.setenv(TOKEN, token)
    seen = []

    def handler(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"thread_id": "x"})

    transport = httpx.MockTransport(handler)
    if shared_client:
        async def go():
            client = httpx.AsyncClient(base_url="http://lg.invalid", transport=transport)
            monkeypatch.setattr(api_config, "http_client", client)
            try:
                await server.lg_request("POST", "/threads", json={}, headers={"X-Other": "1"})
            finally:
                await client.aclose()
    else:
        real = httpx.AsyncClient
        monkeypatch.setattr(api_config, "http_client", None)
        monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))

        async def go():
            await server.lg_request("POST", "/threads", json={})
    asyncio.run(go())
    assert seen == [f"Bearer {token}"], "the API's LangGraph client did not send the service token"
