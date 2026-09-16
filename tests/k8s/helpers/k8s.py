"""Thin kubectl helpers for the cluster test tiers. All calls are namespaced and
never touch the `default` namespace. Uses the context from AVALOKA_TEST_KUBE_CONTEXT."""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

CONTEXT = os.getenv("AVALOKA_TEST_KUBE_CONTEXT", "kind-avaloka")
NAMESPACE = os.getenv("AVALOKA_TEST_NAMESPACE", "avaloka-test")


def kubectl(*args: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", "--context", CONTEXT, *args],
        capture_output=True, text=True, check=check, timeout=timeout,
    )


def kubectl_ns(*args: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    return kubectl("-n", NAMESPACE, *args, check=check, timeout=timeout)


def current_context() -> str:
    return subprocess.run(["kubectl", "config", "current-context"],
                          capture_output=True, text=True).stdout.strip()


def is_kind_context(ctx: Optional[str] = None) -> bool:
    return (ctx or CONTEXT).startswith("kind-")


def ensure_namespace() -> None:
    kubectl("create", "namespace", NAMESPACE, check=False)


def get_json(*args: str) -> Dict[str, Any]:
    return json.loads(kubectl_ns(*args, "-o", "json").stdout)


def wait_for_deployment(name: str, timeout_s: int = 300) -> bool:
    """Wait until a Deployment has >=1 availableReplicas. On failure, callers should
    dump logs/describe into the assertion message."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            obj = get_json("get", "deploy", name)
            if (obj.get("status", {}).get("availableReplicas") or 0) >= 1:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def diagnostics(selector: str) -> str:
    """kubectl logs + describe for pods matching a label selector — for assertion msgs."""
    out = []
    for verb in (["describe", "pods", "-l", selector], ["logs", "-l", selector, "--tail", "80"]):
        r = kubectl_ns(*verb, check=False)
        out.append(f"$ kubectl -n {NAMESPACE} {' '.join(verb)}\n{r.stdout}\n{r.stderr}")
    return "\n".join(out)


def can_i(verb: str, resource: str, sa: str) -> bool:
    r = kubectl_ns("auth", "can-i", verb, resource,
                   f"--as=system:serviceaccount:{NAMESPACE}:{sa}", check=False)
    return r.stdout.strip() == "yes"
