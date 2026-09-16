"""T3 — KubeRay execution (cluster, kuberay, slow).

Installs the KubeRay operator at the D2 version, brings up the RayCluster, proves
the cluster runs Ray 2.49.2 end to end, runs a Ray job, and exercises the D3
RayService inference path (real prediction / not a scaffold).

Resource note: a RayCluster and a RayService each run a head pod. On a 2-node kind
cluster, tear the RayCluster down before the RayService or expect Pending pods.
SAFETY: kind-context only unless AVALOKA_TEST_ALLOW_CLOUD=1.
"""
from __future__ import annotations

import json
import os

import pytest
import requests

from tests.k8s.helpers import k8s
from tests.k8s.helpers.portforward import port_forward

pytestmark = [pytest.mark.cluster, pytest.mark.kuberay, pytest.mark.slow]

NS = k8s.NAMESPACE
RAY_VERSION = "2.49.2"


def _reachable() -> bool:
    return k8s.kubectl("get", "nodes", check=False, timeout=15).returncode == 0


@pytest.fixture(scope="module")
def kuberay_operator():
    if not _reachable():
        pytest.skip(f"cluster {k8s.CONTEXT} not reachable")
    if not k8s.is_kind_context() and os.getenv("AVALOKA_TEST_ALLOW_CLOUD") != "1":
        pytest.skip("non-kind context without AVALOKA_TEST_ALLOW_CLOUD=1")
    from app.infra import install_k8s, ray_manager

    k8s.ensure_namespace()
    out = install_k8s.install_kuberay(namespace=NS)
    assert out["status"] in ("SUCCESS", "SKIPPED"), out
    # T3.1: CRDs present
    crds = k8s.kubectl("get", "crd", check=False).stdout
    for crd in ("rayclusters.ray.io", "rayjobs.ray.io", "rayservices.ray.io"):
        assert crd in crds, f"missing CRD {crd}"
    assert ray_manager.KUBERAY_OPERATOR_VERSION.startswith("1."), ray_manager.KUBERAY_OPERATOR_VERSION
    yield


# --------------------------------------------------------------------------- T3.1
def test_t3_1_operator_version_supports_ray_253(kuberay_operator):
    from app.infra import ray_manager
    major, minor, *_ = ray_manager.KUBERAY_OPERATOR_VERSION.split(".")
    # KubeRay >= 1.3 supports Ray 2.38.0+ (hence 2.53) per the compatibility matrix.
    assert (int(major), int(minor)) >= (1, 3), ray_manager.KUBERAY_OPERATOR_VERSION


# --------------------------------------------------------------------------- T3.2/3.3/3.5
@pytest.mark.slow
def test_t3_raycluster_runs_ray_253(kuberay_operator):
    from app.infra import ray_manager

    out = ray_manager.apply_ray_cluster(namespace=NS)
    # WARNING == "applied, but the head pod wasn't Ready within apply's own short,
    # racy wait" (the operator hadn't created the pod yet). We do our own robust wait.
    assert out["status"] in ("SUCCESS", "SKIPPED", "WARNING"), out
    # wait for head Running
    ok = False
    import time
    deadline = time.time() + 600
    while time.time() < deadline:
        pods = k8s.get_json("get", "pods", "-l", "ray.io/node-type=head")
        items = pods.get("items", [])
        if items and items[0].get("status", {}).get("phase") == "Running":
            ok = True
            break
        time.sleep(10)
    if not ok:
        pytest.fail("RayCluster head not Running in 600s\n" +
                    k8s.diagnostics("ray.io/node-type=head"))
    # T3.3: dashboard reports 2.49.2
    with port_forward("svc/avaloka-raycluster-head-svc", 8265) as lp:
        ver = requests.get(f"http://127.0.0.1:{lp}/api/version", timeout=15).json()
    assert ver.get("ray_version") == RAY_VERSION, ver


# --------------------------------------------------------------------------- T3.7/3.8
@pytest.mark.slow
def test_t3_rayservice_inference_real_prediction(kuberay_operator):
    """D3 proof: the RayService serves a real (non-scaffold) prediction.

    Tears down the RayCluster first (2-node kind capacity) before standing the
    RayService up.
    """
    from app.infra import ray_manager

    k8s.kubectl("delete", "raycluster", "avaloka-raycluster", "-n", NS,
                "--ignore-not-found", check=False)
    out = ray_manager.deploy_ray_serve(namespace=NS)
    assert out["status"] in ("SUCCESS", "SKIPPED", "WARNING"), out

    import time
    # The serve svc is created by KubeRay only once the Serve app is healthy; on a
    # 2-node kind cluster the worker pull+start is slow, so give it room.
    deadline = time.time() + 900
    ready = False
    while time.time() < deadline:
        svc = k8s.kubectl_ns("get", "svc", "avaloka-inference-serve-svc",
                             check=False).returncode == 0
        rs = k8s.kubectl_ns("get", "rayservice", "avaloka-inference",
                            "-o", "jsonpath={.status.serviceStatus}", check=False).stdout.strip()
        if svc and rs == "Running":
            ready = True
            break
        time.sleep(10)
    if not ready:
        pytest.fail("RayService serve svc not Running in 900s\n" +
                    k8s.diagnostics("ray.io/serve=true"))

    with port_forward("svc/avaloka-inference-serve-svc", 8000) as lp:
        # T3.7 readiness
        g = requests.get(f"http://127.0.0.1:{lp}/", timeout=30).json()
        assert g.get("status") == "ready", g
        assert "model_loaded" in g
        # T3.8: real prediction, not the scaffold note
        p = requests.post(f"http://127.0.0.1:{lp}/",
                          json={"features": {"a": 1, "b": 2, "c": 3}}, timeout=30).json()
    assert "scaffold response" not in json.dumps(p), p
    assert "prediction" in p and p["prediction"] is not None, p


# --------------------------------------------------------------------------- T3.10
def test_t3_teardown(kuberay_operator):
    for kind, name in (("rayservice", "avaloka-inference"),
                       ("raycluster", "avaloka-raycluster")):
        k8s.kubectl("delete", kind, name, "-n", NS, "--ignore-not-found", check=False)
    # operator/CRDs left for reuse across a test session; a full session finalizer
    # (test_t2/t3 teardown) removes the namespace.
