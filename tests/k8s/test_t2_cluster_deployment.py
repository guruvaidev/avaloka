"""T2 — Backend on Kubernetes (cluster, slow).

Deploys the avaloka API chart to namespace ``avaloka-test`` on the kind context
from AVALOKA_TEST_KUBE_CONTEXT (default kind-avaloka) and asserts the deployed
backend is the real API (not Streamlit). Teardown runs even on failure.

SAFETY: destructive steps (helm uninstall / kubectl delete) run ONLY against a
kind- context unless AVALOKA_TEST_ALLOW_CLOUD=1.
"""
from __future__ import annotations

import json
import os
import time

import pytest
import requests

from tests.k8s.helpers import k8s
from tests.k8s.helpers.portforward import port_forward

pytestmark = [pytest.mark.cluster, pytest.mark.slow]

NS = k8s.NAMESPACE
SA = "avaloka"


def _cluster_reachable() -> bool:
    r = k8s.kubectl("get", "nodes", check=False, timeout=15)
    return r.returncode == 0


def _images_in_kind() -> bool:
    """avaloka-api + avaloka-ray loaded into the kind node?"""
    ctx = k8s.CONTEXT
    if not ctx.startswith("kind-"):
        return True  # cloud: images come from a registry, not side-loaded
    node = ctx.replace("kind-", "") + "-control-plane"
    got = __import__("subprocess").run(
        ["docker", "exec", node, "crictl", "images"],
        capture_output=True, text=True)
    return "avaloka-api" in got.stdout


@pytest.fixture(scope="module")
def deployed():
    # T2.1 / safety
    if not _cluster_reachable():
        pytest.skip(f"cluster context {k8s.CONTEXT} not reachable")
    if not k8s.is_kind_context() and os.getenv("AVALOKA_TEST_ALLOW_CLOUD") != "1":
        pytest.skip("refusing to run against a non-kind context without AVALOKA_TEST_ALLOW_CLOUD=1")
    if not _images_in_kind():
        pytest.skip("avaloka-api image not loaded into kind — build+load first "
                    "(deploy_stack.build_images(load_into_kind=True))")

    from app.infra import deploy_stack

    k8s.ensure_namespace()
    outcome = deploy_stack.deploy_avaloka(namespace=NS, service_type="ClusterIP")
    assert outcome["status"] in ("SUCCESS", "SKIPPED"), f"deploy failed: {outcome}"

    ok = k8s.wait_for_deployment("avaloka", timeout_s=300)
    if not ok:
        diag = k8s.diagnostics("app.kubernetes.io/component=api")
        pytest.fail(f"avaloka Deployment not available in 300s.\n{diag}")
    # /threads proxies to the LangGraph server — wait for it too so the auth-200 path works.
    if not k8s.wait_for_deployment("avaloka-langgraph", timeout_s=300):
        diag = k8s.diagnostics("app.kubernetes.io/component=langgraph")
        pytest.fail(f"avaloka-langgraph Deployment not available in 300s.\n{diag}")
    yield
    # T2.13 teardown — runs even on failure, kind-only unless cloud allowed.
    if k8s.is_kind_context() or os.getenv("AVALOKA_TEST_ALLOW_CLOUD") == "1":
        __import__("subprocess").run(
            ["helm", "--kube-context", k8s.CONTEXT, "-n", NS, "uninstall", "avaloka"],
            capture_output=True, text=True)
        k8s.kubectl("delete", "namespace", NS, "--wait=false", check=False)


# --------------------------------------------------------------------------- T2.5/2.6
def test_t2_health_endpoint(deployed):
    with port_forward("svc/avaloka", 9000) as lp:
        r = requests.get(f"http://127.0.0.1:{lp}/health", timeout=10)
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("status", "redis_connected", "graph_ready"):
        assert key in body, f"missing {key} in /health: {body}"
    assert body["redis_connected"] is True, f"Redis not connected: {body}"


# --------------------------------------------------------------------------- T2.7
def test_t2_docs_and_openapi_prove_real_api(deployed):
    with port_forward("svc/avaloka", 9000) as lp:
        docs = requests.get(f"http://127.0.0.1:{lp}/docs", timeout=10)
        spec = requests.get(f"http://127.0.0.1:{lp}/openapi.json", timeout=10)
    assert docs.status_code == 200
    paths = spec.json().get("paths", {})
    # The real FastAPI app exposes ~32 paths / ~36 routes incl. /health, /api/upload,
    # /threads, /api/models/{run_id}/inference, /api/missions/plan. Streamlit exposes
    # essentially none, so a high path count is the proof this is the API.
    assert len(paths) >= 30, f"expected the real API (>=30 paths), got {len(paths)} (Streamlit would have ~0)"
    assert "/health" in paths and "/api/upload" in paths


# --------------------------------------------------------------------------- T2.8
def test_t2_auth_contract(deployed):
    import jwt
    secret = os.getenv("SUPABASE_JWT_SECRET", "")
    with port_forward("svc/avaloka", 9000) as lp:
        no_tok = requests.get(f"http://127.0.0.1:{lp}/threads", timeout=10)
        assert no_tok.status_code == 401
        if not secret:
            pytest.skip("SUPABASE_JWT_SECRET unset — cannot mint a valid token for the 200 case")
        token = jwt.encode({"sub": "user-1", "iat": int(time.time())}, secret, algorithm="HS256")
        ok = requests.get(f"http://127.0.0.1:{lp}/threads",
                          headers={"Authorization": f"Bearer {token}"}, timeout=10)
        assert ok.status_code == 200, ok.text


# --------------------------------------------------------------------------- T2.10
def test_t2_rbac_grants_rayjobs_and_rayservices(deployed):
    """The chart's Role must grant the SA create-rayjobs + patch-rayservices. The
    live `kubectl auth can-i` needs the ray.io CRDs installed (that's T3, which
    validates it end-to-end); here we assert the RBAC rule the chart ships."""
    role = k8s.get_json("get", "role", "avaloka")
    ray_rules = [r for r in role["rules"] if "ray.io" in r.get("apiGroups", [])]
    assert ray_rules, "no ray.io rules in the chart Role"
    rule = ray_rules[0]
    assert "rayjobs" in rule["resources"] and "rayservices" in rule["resources"]
    assert {"create", "patch", "update"}.issubset(set(rule["verbs"]))
    # RoleBinding binds it to the API service account.
    rb = k8s.get_json("get", "rolebinding", "avaloka")
    assert any(s.get("name") == SA for s in rb["subjects"]), "Role not bound to the API SA"


# --------------------------------------------------------------------------- T2.11
def test_t2_configmap_has_ray_and_redis(deployed):
    cm = k8s.get_json("get", "configmap", "avaloka-config")
    data = cm["data"]
    assert data["RAY_ADDRESS"] == "ray://avaloka-raycluster-head-svc:10001"
    assert "RAY_SERVE_URL" in data
    assert "REDIS_URL" in data


# --------------------------------------------------------------------------- T2.12
def test_t2_upgrade_path_increments_revision(deployed):
    from app.infra import deploy_stack

    def _rev() -> int:
        r = __import__("subprocess").run(
            ["helm", "--kube-context", k8s.CONTEXT, "-n", NS, "list", "-o", "json"],
            capture_output=True, text=True)
        rels = {x["name"]: x for x in json.loads(r.stdout or "[]")}
        return int(rels["avaloka"]["revision"])

    before = _rev()
    out = deploy_stack.deploy_avaloka(namespace=NS, service_type="ClusterIP")
    assert out["status"] == "SUCCESS", out
    assert _rev() > before
