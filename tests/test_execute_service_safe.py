"""The /execute service runs model-written code safely, and nothing re-exposes it.

It used to be ``app/infra/python_app``: Flask ``exec()`` on the request body,
no authentication, behind a ``type: LoadBalancer`` Service, with the default
ServiceAccount token mounted, docker-built and kubectl-applied at runtime by
the agent. Any chat message saying "deploy", "aws", "gcp" or "kubernetes"
started that provisioning. Every test here fails on that tree by assertion:

* the service: 401 without the token (before the body is read), and code it
  runs sees none of the server's secrets;
* the chart: ClusterIP, a NetworkPolicy with no egress, a locked-down pod;
* guards over every rendered chart and every manifest in the repository, and
  over every web handler in the code;
* the callers send the token, and nothing provisions without confirmation.
"""
from __future__ import annotations

import ast
import base64
import importlib.util
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from unittest import mock

import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage

REPO = pathlib.Path(__file__).resolve().parents[1]
CHART = REPO / "deploy" / "helm" / "avaloka"
TOKEN = "t0k-" + "x" * 40
PLANTED = {
    "DB_ENCRYPTION_KEY": "planted-db-encryption-key-5f1c",
    "OPENAI_API_KEY": "planted-openai-key-9a2b",
    "MCP_SERVICE_TOKEN": "planted-mcp-token-77d0",
    "SUPABASE_SERVICE_ROLE_KEY": "planted-supabase-role-3e4f",
}


# --------------------------------------------------------------------------- the service

def _service_spec():
    try:
        return importlib.util.find_spec("app.executor.execute_service")
    except ModuleNotFoundError:
        return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def service():
    """The hardened service, started the way the chart starts it, with secrets planted in its env.

    None when there is no such service; each test asserts on that, so the old
    tree fails by assertion rather than by a setup error.
    """
    if _service_spec() is None:
        yield None
        return
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if k not in PLANTED}
    env.update(PLANTED)
    env.update({"EXECUTE_SERVICE_TOKEN": TOKEN, "PYTHONPATH": str(REPO), "AVALOKA_EXECUTE_TIMEOUT": "60"})
    proc = subprocess.Popen([sys.executable, "-m", "app.executor.execute_service",
                             "--host", "127.0.0.1", "--port", str(port)],
                            cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(150):
            try:
                with urllib.request.urlopen(base + "/health", timeout=1):
                    break
            except OSError:
                if proc.poll() is not None:
                    raise AssertionError(f"execute service exited: {proc.stdout.read().decode()[-2000:]}")
                time.sleep(0.1)
        else:
            raise AssertionError("execute service never answered /health")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


NO_SERVICE = ("no hardened /execute service (app.executor.execute_service): the only /execute is "
              "app/infra/python_app's unauthenticated Flask exec()")


def _post(base, body, token=None, raw=None):
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(base + "/execute", data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=90) as res:
            return res.status, json.loads(res.read() or b"{}")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def _code(src: str) -> dict:
    return {"code": base64.b64encode(src.encode()).decode()}


@pytest.mark.parametrize("token", [None, "", "wrong-" + "y" * 40, TOKEN[:-1]])
def test_execute_without_the_right_token_is_refused(service, token, tmp_path):
    assert service, NO_SERVICE
    # The code would create a file if it ran; a 401 must mean it did not.
    marker = tmp_path / "execute-ran-unauthenticated"
    status, body = _post(service, _code(f"open({str(marker)!r}, 'w').write('ran')"), token=token)
    assert status == 401, (status, body)
    assert not marker.exists(), "unauthenticated code ran"


def test_the_401_comes_before_the_body_is_read(service):
    """Headers promising 10 MB and no body: a service that reads first would hang, not answer."""
    assert service, NO_SERVICE
    host, port = service.replace("http://", "").split(":")
    with socket.create_connection((host, int(port)), timeout=5) as s:
        s.sendall(b"POST /execute HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  b"Content-Length: 10000000\r\n\r\n")
        s.settimeout(5)
        reply = s.recv(4096).decode("latin-1")
    assert reply.startswith("HTTP/1.1 401"), reply[:200]


def test_execute_with_the_token_still_works(service):
    assert service, NO_SERVICE
    csv = "team,n\na,1\nb,2\na,3\n"
    src = ("df = input_df.groupby('team', as_index=False)['n'].sum()\n"
           "df.to_csv(output_file, index=False)\nresult = int(df['n'].sum())\nprint('ok')\n")
    body = _code(src)
    body["data"] = base64.b64encode(csv.encode()).decode()
    status, out = _post(service, body, token=TOKEN)
    assert status == 200, out
    assert out["status"] == "success" and out["result"] == 6 and out["stdout"].strip() == "ok"
    decoded = base64.b64decode(out["output"].split("base64,")[1]).decode()
    assert decoded.splitlines() == ["team,n", "a,4", "b,2"]


def test_planted_secrets_do_not_reach_executed_code(service):
    assert service, NO_SERVICE
    src = (
        "import os\n"
        "seen = dict(os.environ)\n"
        "try:\n"
        "    seen['_parent'] = open(f'/proc/{os.getppid()}/environ', 'rb').read().decode('latin-1')\n"
        "except OSError as exc:\n"
        "    seen['_parent'] = 'unreadable: ' + type(exc).__name__\n"
        "result = seen\n"
    )
    status, out = _post(service, _code(src), token=TOKEN)
    assert status == 200, out
    dumped = json.dumps(out)
    leaked = [k for k, v in PLANTED.items() if v in dumped]
    assert not leaked, f"executed code saw the server's {leaked}"
    assert TOKEN not in dumped, "executed code saw the service token in an environment"
    assert "EXECUTE_SERVICE_TOKEN" not in out["result"]


def test_a_failing_script_is_reported_not_raised(service):
    assert service, NO_SERVICE
    status, out = _post(service, _code("raise ValueError('boom')"), token=TOKEN)
    assert status == 500 and "boom" in out["error"] and "ValueError" in out["traceback"]


def test_the_service_refuses_to_start_without_a_token():
    assert _service_spec() is not None, "no hardened /execute service"
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXECUTE_SERVICE_TOKEN")}
    env["PYTHONPATH"] = str(REPO)
    run = subprocess.run([sys.executable, "-m", "app.executor.execute_service", "--host", "127.0.0.1",
                          "--port", str(_free_port())], cwd=str(REPO), env=env,
                         capture_output=True, text=True, timeout=60)
    assert run.returncode != 0 and "EXECUTE_SERVICE_TOKEN" in (run.stderr + run.stdout)


# --------------------------------------------------------------------------- the chart

def _render(*args: str) -> list:
    helm = shutil.which("helm")
    if helm is None:
        message = "helm is not installed; the rendered-chart checks did not run"
        if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
            pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
        pytest.skip(message)
    run = subprocess.run([helm, "template", "avaloka", str(CHART), *args], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return [d for d in yaml.safe_load_all(run.stdout) if d]


def _component(docs, kind, component):
    return [d for d in docs if d.get("kind") == kind
            and ((d.get("metadata") or {}).get("labels") or {}).get("app.kubernetes.io/component") == component]


def test_the_execute_service_is_off_by_default():
    docs = _render()
    assert not [d for d in docs if "execute-service" in d["metadata"]["name"]]
    config = next(d for d in docs if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "avaloka-config")
    assert "AVALOKA_EXECUTE_SERVICE_URL" not in config["data"]


def test_the_rendered_execute_service_is_cluster_internal_and_locked_down():
    docs = _render("--set", "executeService.enabled=true")
    deployments = _component(docs, "Deployment", "execute-service")
    assert len(deployments) == 1, "the chart renders no hardened execute service"
    services = _component(docs, "Service", "execute-service")
    assert [s["spec"].get("type", "ClusterIP") for s in services] == ["ClusterIP"]

    pod = deployments[0]["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
    (container,) = pod["containers"]
    sc = container["securityContext"]
    assert sc["readOnlyRootFilesystem"] is True and sc["allowPrivilegeEscalation"] is False
    assert sc["capabilities"]["drop"] == ["ALL"]
    assert "app.executor.execute_service" in container["command"]
    assert not container.get("envFrom"), "the whole ConfigMap/Secret would put server secrets in its env"
    assert not [e for e in container.get("env") or [] if "valueFrom" in e]
    tmp = [v for v in pod["volumes"] if v["name"] in {m["name"] for m in container["volumeMounts"]
                                                       if m["mountPath"] == "/tmp"}]
    assert tmp and "emptyDir" in tmp[0], "no writable /tmp on the read-only root"
    token_vols = [v for v in pod["volumes"] if "secret" in v]
    assert [i["key"] for v in token_vols for i in v["secret"]["items"]] == ["EXECUTE_SERVICE_TOKEN"]

    sa_name = pod.get("serviceAccountName")
    sas = [d for d in docs if d["kind"] == "ServiceAccount" and d["metadata"]["name"] == sa_name]
    assert len(sas) == 1, "the execute service has no dedicated ServiceAccount"
    (sa,) = sas
    assert sa.get("automountServiceAccountToken") is False
    bound = [d["metadata"]["name"] for d in docs if d["kind"] in ("RoleBinding", "ClusterRoleBinding")
             and any(s.get("name") == sa_name for s in d.get("subjects") or [])]
    assert not bound, f"the execute service's ServiceAccount has RBAC: {bound}"

    policies = _component(docs, "NetworkPolicy", "execute-service")
    assert len(policies) == 1, "no NetworkPolicy guards the execute service"
    (policy,) = policies
    spec = policy["spec"]
    assert spec["podSelector"]["matchLabels"]["app.kubernetes.io/component"] == "execute-service"
    assert sorted(spec["policyTypes"]) == ["Egress", "Ingress"] and spec["egress"] == []
    (rule,) = spec["ingress"]
    (peer,) = rule["from"]
    (expr,) = peer["podSelector"]["matchExpressions"]
    assert (expr["key"], expr["operator"]) == ("app.kubernetes.io/component", "In")
    assert sorted(expr["values"]) == ["api", "celery", "langgraph"]

    config = next(d for d in docs if d["kind"] == "ConfigMap" and d["metadata"]["name"] == "avaloka-config")
    assert config["data"]["AVALOKA_EXECUTE_SERVICE_URL"] == "http://avaloka-execute-service:8080"
    secret = next(d for d in docs if d["kind"] == "Secret" and d["metadata"]["name"] == "avaloka-secrets")
    assert len(secret["stringData"]["EXECUTE_SERVICE_TOKEN"]) >= 32


def test_the_graph_pods_get_the_token_and_the_executor_pods_cannot_reach_the_service():
    docs = _render("--set", "executeService.enabled=true")
    for name in ("avaloka", "avaloka-langgraph", "avaloka-celery"):
        deps = [d for d in docs if d["kind"] == "Deployment" and d["metadata"]["name"] == name]
        assert len(deps) == 1, f"no Deployment {name}"
        (dep,) = deps
        refs = {(s.get("secretRef") or {}).get("name") for c in dep["spec"]["template"]["spec"]["containers"]
                for s in c.get("envFrom") or []}
        assert "avaloka-secrets" in refs, f"{name} cannot send EXECUTE_SERVICE_TOKEN"
    policies = _component(docs, "NetworkPolicy", "execute-service")
    assert len(policies) == 1, "no NetworkPolicy guards the execute service"
    (policy,) = policies
    admitted = policy["spec"]["ingress"][0]["from"][0]["podSelector"]["matchExpressions"][0]["values"]
    assert "celery-executor" not in admitted, "scheduled-pod code could call the execute service"


# --------------------------------------------------------------------------- regression guards

#: What ran the unauthenticated service: its image, its Service, its entrypoint.
_EXEC_MARKERS = ("infra-agent-python-app", "python_app", "execute_service", "execute-service")


def _flag_exposed_exec(docs, where) -> list:
    """LoadBalancer/NodePort Services that front the execute service in one set of documents."""
    pods = []
    for d in docs:
        if d.get("kind") in ("Deployment", "StatefulSet", "DaemonSet", "Pod"):
            tpl = d if d["kind"] == "Pod" else (d.get("spec") or {}).get("template") or {}
            labels = (tpl.get("metadata") or {}).get("labels") or {}
            text = json.dumps((tpl.get("spec") or {}).get("containers") or [])
            pods.append((labels, any(m in text for m in _EXEC_MARKERS)))
    bad = []
    for d in docs:
        if d.get("kind") != "Service":
            continue
        spec = d.get("spec") or {}
        if spec.get("type") not in ("LoadBalancer", "NodePort"):
            continue
        name = (d.get("metadata") or {}).get("name") or ""
        selector = spec.get("selector") or {}
        fronts_exec = any(selector and all(labels.get(k) == v for k, v in selector.items()) and is_exec
                          for labels, is_exec in pods)
        if fronts_exec or any(m in name for m in ("infra-agent", "execute")):
            bad.append(f"{where}: Service {name} type {spec.get('type')}")
    return bad


def _repo_manifests():
    skip = {".git", "node_modules", ".venv", "venv"}
    for path in REPO.rglob("*"):
        if path.suffix not in (".yaml", ".yml") or skip & set(path.parts) or not path.is_file():
            continue
        try:
            docs = [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8", errors="replace"))
                    if isinstance(d, dict)]
        except yaml.YAMLError:
            continue  # a Helm template; the rendered chart covers those
        if docs:
            yield path.relative_to(REPO), docs


def test_no_manifest_in_the_repository_exposes_the_execute_service():
    bad = []
    for rel, docs in _repo_manifests():
        bad += _flag_exposed_exec(docs, str(rel))
    assert not bad, "an unauthenticated exec service is exposed outside the cluster:\n" + "\n".join(bad)


@pytest.mark.parametrize("args", [
    (),
    ("--set", "executeService.enabled=true"),
    # The API may legitimately be a LoadBalancer; the execute service never.
    ("--set", "executeService.enabled=true", "--set", "service.type=LoadBalancer"),
    ("--set", "executeService.enabled=true", "--set", "service.type=NodePort"),
], ids=["default", "enabled", "api-loadbalancer", "api-nodeport"])
def test_no_rendered_chart_exposes_the_execute_service(args):
    docs = _render(*args)
    bad = _flag_exposed_exec(docs, " ".join(args) or "default")
    assert not bad, bad
    if "service.type=LoadBalancer" in args:
        # The guard must not flag the legitimate API LoadBalancer it is meant to allow.
        assert [d["metadata"]["name"] for d in docs
                if d["kind"] == "Service" and (d.get("spec") or {}).get("type") == "LoadBalancer"] == ["avaloka"]


_HANDLER_DECORATORS = {"route", "post", "get", "put", "patch", "delete", "api_route", "websocket"}


def _exec_calls(node) -> list:
    return [n.lineno for n in ast.walk(node) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id in ("exec", "eval")]


def test_no_web_handler_executes_request_input_in_process():
    """A route handler (Flask/FastAPI decorator, or an http.server do_* method) never calls exec/eval.

    Model-written code runs in a child with a scrubbed environment
    (app/executor), never in the process that holds the request.
    """
    assert not (REPO / "app" / "infra" / "python_app").exists(), "app/infra/python_app is back"
    found = []
    for root in ("app", "avaloka"):
        for path in (REPO / root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names = {getattr(d.func if isinstance(d, ast.Call) else d, "attr", None)
                             for d in node.decorator_list}
                    if names & _HANDLER_DECORATORS and _exec_calls(node):
                        found.append(f"{path.relative_to(REPO)}:{node.lineno} {node.name}")
                if isinstance(node, ast.ClassDef) and any(
                        getattr(b, "id", getattr(b, "attr", "")) == "BaseHTTPRequestHandler" for b in node.bases):
                    for fn in node.body:
                        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("do_") and _exec_calls(fn):
                            found.append(f"{path.relative_to(REPO)}:{fn.lineno} {node.name}.{fn.name}")
    assert not found, f"web handlers exec() request input in process: {found}"


# --------------------------------------------------------------------------- callers

def test_the_k8s_caller_sends_the_token_to_the_configured_service(monkeypatch):
    from app.agents import execution_agent as ea

    monkeypatch.setenv("EXECUTE_SERVICE_TOKEN", TOKEN)
    monkeypatch.setenv("AVALOKA_EXECUTE_SERVICE_URL", "http://avaloka-execute-service:8080")
    details = ea.get_service_details({})
    assert (details.get("status"), details.get("service_ip"), str(details.get("service_port"))) == \
        ("success", "avaloka-execute-service", "8080")
    seen = {}

    def fake_post(url, **kw):
        seen.update(url=url, headers=kw.get("headers") or {})
        return mock.Mock(status_code=200, headers={}, raise_for_status=lambda: None,
                         json=lambda: {"status": "success"})

    monkeypatch.setattr(ea.requests, "post", fake_post)
    ea.execute_code_on_k8s(code="result = 1", service_ip=details["service_ip"], service_port=details["service_port"])
    assert seen["url"] == "http://avaloka-execute-service:8080/execute"
    assert seen["headers"].get("Authorization") == f"Bearer {TOKEN}"


def test_the_real_caller_against_the_real_service(service, monkeypatch):
    assert service, NO_SERVICE
    from app.agents import execution_agent as ea

    host, port = service.replace("http://", "").split(":")
    monkeypatch.setenv("EXECUTE_SERVICE_TOKEN", TOKEN)
    assert ea.execute_code_on_k8s(code="result = 41 + 1", service_ip=host, service_port=port)["result"] == 42
    monkeypatch.delenv("EXECUTE_SERVICE_TOKEN")
    out = ea.execute_code_on_k8s(code="result = 1", service_ip=host, service_port=port)
    assert out["status"] == "error" and "401" in out["message"]


def test_the_celery_path_sends_the_token(monkeypatch):
    from app.agents import execution_agent as ea
    from app.core import celery_app as celery_module

    monkeypatch.setenv("EXECUTE_SERVICE_TOKEN", TOKEN)
    monkeypatch.setenv("AVALOKA_EXECUTE_SERVICE_URL", "http://avaloka-execute-service:8080")
    seen = []

    def fake_post(url, **kw):
        seen.append((kw.get("headers") or {}).get("Authorization"))
        return mock.Mock(status_code=200, headers={}, raise_for_status=lambda: None,
                         json=lambda: {"status": "success", "stdout": "", "stderr": ""})

    monkeypatch.setattr(ea.requests, "post", fake_post)
    monkeypatch.setattr(ea, "infra_agent_node", mock.Mock(side_effect=AssertionError("provisioned")))
    celery_module._run_agent_task({
        "task_schedule": {"task_type": "execute"},
        "execution_mode": "local",
        "deploy_on_k8s": True,
        "coder_definition": {"code": "result = 1"},
        "messages": [],
    })
    assert seen == [f"Bearer {TOKEN}"]


def test_an_unconfigured_service_fails_without_provisioning_anything(monkeypatch):
    from app.agents import execution_agent as ea

    monkeypatch.delenv("AVALOKA_EXECUTE_SERVICE_URL", raising=False)
    infra = mock.Mock(side_effect=lambda s: s)
    monkeypatch.setattr(ea, "infra_agent_node", infra)
    out = ea.execution_agent_node({"messages": [], "coder_definition": {"code": "result = 1"},
                                   "infrastructure_provisioned": None})
    assert infra.call_count == 0, "running an analysis provisioned infrastructure"
    assert out["execution_result"]["status"] == "failed"
    assert "executeService.enabled" in out["execution_result"]["message"]


def test_python_docker_builds_and_applies_nothing_at_runtime(monkeypatch):
    from app.infra import k8s_invoker

    calls = []
    monkeypatch.setattr(k8s_invoker, "_run_command", lambda cmd, *a, **k: calls.append(cmd) or
                        {"step_name": "x", "status": "SUCCESS", "message": "", "details": ""})
    monkeypatch.setattr(k8s_invoker, "_probe", lambda cmd, **k: calls.append(cmd) or
                        mock.Mock(returncode=1, stdout="", stderr=""))
    monkeypatch.setattr(k8s_invoker.subprocess, "run", lambda cmd, *a, **k: calls.append(cmd) or
                        mock.Mock(returncode=0, stdout="", stderr=""))
    monkeypatch.delenv("AVALOKA_EXECUTE_SERVICE_URL", raising=False)
    out = k8s_invoker.deploy_application("python-docker", "gcp")
    assert calls == [], f"runtime build/apply: {calls}"
    assert out["status"] == "FAILED" and "executeService.enabled" in out["message"]


# --------------------------------------------------------------------------- keyword hijack

def _turn(prompt, **state):
    from app.agents.planner import plan_etl_job

    return plan_etl_job({"messages": [HumanMessage(content=prompt)], "user_id": "u", "session_id": "s", **state})


def _last_reply(result) -> str:
    replies = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
    return replies[-1].content if replies else ""


@pytest.mark.parametrize("prompt", [
    "analyze our AWS costs",
    "deployment frequency by team",
    "what is our kubernetes spend per cluster?",
    "how many deployments did each team ship on GCP last month",
    "set up a connection to our AWS bucket",
])
def test_an_analysis_question_naming_a_cloud_stays_analysis(prompt):
    import app.api.workflow as wf

    result = _turn(prompt)
    assert not result.get("infrastructure_request"), "an analysis question requested infrastructure"
    assert wf.route_planner_output(result) != "provision_infra"
    assert "provisioning" not in _last_reply(result).lower()


@pytest.mark.parametrize("prompt,platform", [("deploy this to GCP", "gcp"), ("please deploy on AWS", "aws"),
                                             ("spin up a GKE cluster", "gcp")])
def test_an_explicit_deploy_asks_for_confirmation_first(prompt, platform):
    import app.api.workflow as wf

    result = _turn(prompt)
    assert not result.get("infrastructure_request"), "provisioning started without confirmation"
    assert wf.route_planner_output(result) != "provision_infra"
    pending = result.get("pending_clarification") or {}
    assert pending.get("type") == "infrastructure_confirmation"
    assert pending["infrastructure_request"]["type"] == platform
    reply = _last_reply(result).lower()
    assert "nothing has been provisioned" in reply and "reply **yes**" in reply
    assert not re.search(r"provisioning\b.*\bnow", reply), "the reply claims provisioning is under way"


def test_yes_starts_provisioning_and_says_so():
    import app.api.workflow as wf

    pending = {"type": "infrastructure_confirmation",
               "infrastructure_request": {"type": "gcp", "app_type": "python-docker"}}
    result = _turn("yes", pending_clarification=pending)
    assert (result.get("infrastructure_request") or {}).get("type") == "gcp"
    assert wf.route_planner_output(result) == "provision_infra"
    assert "starting gcp provisioning now" in _last_reply(result).lower()


def test_no_cancels_and_says_nothing_was_provisioned():
    import app.api.workflow as wf

    pending = {"type": "infrastructure_confirmation",
               "infrastructure_request": {"type": "gcp", "app_type": "python-docker"}}
    result = _turn("no", pending_clarification=pending)
    assert not result.get("infrastructure_request")
    assert wf.route_planner_output(result) != "provision_infra"
    assert "nothing was provisioned" in _last_reply(result).lower()


def test_after_provisioning_the_reply_reports_what_happened():
    import app.api.workflow as wf

    result = _turn("yes", pending_clarification={"type": "infrastructure_report"},
                   infrastructure_request={"type": "gcp", "app_type": "python-docker"},
                   infrastructure_provisioned={"type": "gcp", "status": "failed", "details": [
                       {"step_name": "Locate execute service", "status": "FAILED", "message": "not configured"}]})
    assert wf.route_planner_output(result) != "provision_infra"
    reply = _last_reply(result)
    assert "finished with status **failed**" in reply and "Locate execute service: FAILED" in reply


def test_the_avaloka_agent_neither_provisions_nor_drops_a_confirmation(monkeypatch):
    from app.agents.avaloka_agent import agent

    monkeypatch.setattr(agent, "_load_redis_context", lambda state: {})
    monkeypatch.setattr(agent, "classify_intent", lambda *a, **k: agent.IntentResult(
        intent="infrastructure", confidence=0.9, reasoning="test"))
    # An "infrastructure" intent used to pre-set infrastructure_request, which
    # sent the turn straight to provision_infra with no confirmation.
    out = agent.avaloka_agent_node({"messages": [HumanMessage(content="provision a GKE cluster")]})
    assert out["delegate_to_planner"] is True
    assert not out.get("infrastructure_request"), "the agent requested infrastructure without confirmation"
    out = agent.avaloka_agent_node({
        "messages": [HumanMessage(content="yes")],
        "pending_clarification": {"type": "infrastructure_confirmation",
                                  "infrastructure_request": {"type": "gcp"}},
    })
    assert out["delegate_to_planner"] is True
    assert not out.get("infrastructure_request")
