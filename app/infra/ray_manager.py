# app/infra/ray_manager.py
"""KubeRay lifecycle + Ray compute helpers.

Two modes, matching the agreed design:

* **provision** — install the KubeRay operator (Helm), apply a ``RayCluster`` CR,
  and (optionally) a ``RayService`` for inference-as-a-service.
* **connect**   — attach to an already-running Ray cluster given a ``RAY_ADDRESS``
  (e.g. ``ray://host:10001``); no operator/cluster creation.

The compute entry point ``run_code`` submits Python to Ray and is the new
canonical replacement for the legacy Flask ``/execute`` pod.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile

from app.infra.providers.base import run_command

# Repo root → deploy/helm/ray/*.yaml
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAY_DIR = os.path.join(_REPO_ROOT, "deploy", "helm", "ray")
RAYCLUSTER_MANIFEST = os.path.join(_RAY_DIR, "raycluster.yaml")
RAYSERVICE_MANIFEST = os.path.join(_RAY_DIR, "rayservice.yaml")

# Default Ray worker/head image baked into the manifests. Override (e.g. with a
# GCR image for GKE) via AVALOKA_RAY_IMAGE.
DEFAULT_RAY_IMAGE = "avaloka-ray:latest"
RAY_IMAGE = os.environ.get("AVALOKA_RAY_IMAGE", DEFAULT_RAY_IMAGE)

KUBERAY_HELM_REPO = "https://ray-project.github.io/kuberay-helm/"
# 1.6 is the first operator that can read a token-authenticated Ray dashboard
# from a Secret we name (authOptions.secretName, key auth_token): the RayCluster,
# RayService and RayJob manifests set RAY_AUTH_MODE=token, and an older operator
# is refused when it polls job and Serve status, so RayJobs never finish and the
# RayService never turns ready. Its CRDs must carry authOptions too, which
# `helm upgrade` never updates; upgrade_kuberay_crds() applies them first.
KUBERAY_OPERATOR_VERSION = os.environ.get("KUBERAY_OPERATOR_VERSION", "1.6.2")
KUBERAY_CHART = "kuberay/kuberay-operator"


def upgrade_kuberay_crds(version: str = None) -> dict:
    """Apply the operator chart's CRDs server-side: Helm installs CRDs once and never upgrades them."""
    version = version or KUBERAY_OPERATOR_VERSION
    if not re.fullmatch(r"[0-9A-Za-z.+-]+", version):
        raise ValueError(f"KUBERAY_OPERATOR_VERSION {version!r} is not a chart version")
    return run_command(
        [
            "bash", "-c",
            "set -euo pipefail; "
            f"helm show crds {KUBERAY_CHART} --version {version} "
            "| kubectl apply --server-side --force-conflicts -f -",
        ],
        "Upgrade KubeRay CRDs",
    )


def _render_manifest(path: str) -> str:
    """Return a manifest path with the Ray image substituted when overridden.

    When AVALOKA_RAY_IMAGE differs from the baked-in default, rewrite the image in
    a temp file. Registry images default to ``Always`` while a local image name
    (such as one side-loaded into kind) remains ``IfNotPresent``. Override either
    choice with AVALOKA_RAY_IMAGE_PULL_POLICY.
    """
    if RAY_IMAGE == DEFAULT_RAY_IMAGE:
        return path
    with open(path) as f:
        content = f.read()
    content = content.replace(DEFAULT_RAY_IMAGE, RAY_IMAGE)
    pull_policy = os.environ.get("AVALOKA_RAY_IMAGE_PULL_POLICY", "").strip()
    if not pull_policy:
        pull_policy = "Always" if "/" in RAY_IMAGE else "IfNotPresent"
    if pull_policy not in {"Always", "IfNotPresent", "Never"}:
        raise ValueError(
            "AVALOKA_RAY_IMAGE_PULL_POLICY must be Always, IfNotPresent, or Never"
        )
    content = content.replace(
        "imagePullPolicy: IfNotPresent", f"imagePullPolicy: {pull_policy}"
    )
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="ray-manifest-")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return tmp


def install_kuberay_operator(namespace: str = "default") -> dict:
    """Install/upgrade the KubeRay operator via Helm (idempotent)."""
    add = run_command(["helm", "repo", "add", "kuberay", KUBERAY_HELM_REPO], "Add KubeRay Helm repo")
    if add["status"] == "FAILED":
        return add
    run_command(["helm", "repo", "update", "kuberay"], "Update KubeRay Helm repo")
    crds = upgrade_kuberay_crds()
    if crds["status"] == "FAILED":
        return crds
    return run_command(
        [
            "helm", "upgrade", "--install", "kuberay-operator", KUBERAY_CHART,
            "--version", KUBERAY_OPERATOR_VERSION,
            "--namespace", namespace, "--create-namespace",
            "--wait", "--timeout", "5m",
        ],
        "Install KubeRay Operator",
    )


def apply_ray_cluster(manifest: str = RAYCLUSTER_MANIFEST, namespace: str = "default") -> dict:
    """Apply the RayCluster CR and wait for the head pod to become Ready."""
    manifest = _render_manifest(manifest)
    apply = run_command(["kubectl", "apply", "-n", namespace, "-f", manifest], "Apply RayCluster")
    if apply["status"] == "FAILED":
        return apply
    wait = run_command(
        [
            "kubectl", "wait", "-n", namespace, "--for=condition=Ready", "pod",
            "-l", "ray.io/node-type=head", "--timeout=300s",
        ],
        "Wait for Ray head Ready",
    )
    # head may take a moment to be scheduled; surface a warning rather than hard-fail
    if wait["status"] == "FAILED":
        wait["status"] = "WARNING"
        wait["message"] = "RayCluster applied but head pod not Ready yet; check `kubectl get pods`."
    return wait


def deploy_ray_serve(manifest: str = RAYSERVICE_MANIFEST, namespace: str = "default") -> dict:
    """Deploy the Ray Serve app (inference-as-a-service) via a RayService CR."""
    manifest = _render_manifest(manifest)
    return run_command(["kubectl", "apply", "-n", namespace, "-f", manifest], "Deploy Ray Serve (RayService)")


def remove_ray_serve(name: str = "avaloka-inference", namespace: str = "default") -> dict:
    """Delete the opt-in RayService if an older install left it running (idempotent)."""
    return run_command(
        ["kubectl", "delete", "-n", namespace, "rayservice", name, "--ignore-not-found"],
        "Remove the pre-2026-11 RayService (the model server serves its models now)",
    )


RAY_DASHBOARD_NODEPORT_MANIFEST = os.path.join(_RAY_DIR, "ray_dashboard_nodeport.yaml")


def dashboard_nodeport_requested() -> bool:
    """The kind-only NodePort for the Ray dashboard is opt-in (RAY_DASHBOARD_EXPOSE_NODEPORT=1)."""
    return os.environ.get("RAY_DASHBOARD_EXPOSE_NODEPORT", "").strip().lower() in {"1", "true", "yes"}


def apply_ray_dashboard_nodeport(namespace: str = "default") -> dict:
    if not dashboard_nodeport_requested():
        return {"step_name": "Expose Ray dashboard (NodePort)", "status": "SKIPPED",
                "message": "RAY_DASHBOARD_EXPOSE_NODEPORT not set; the dashboard is reached through the API proxy",
                "details": ""}
    return run_command(["kubectl", "apply", "-n", namespace, "-f", RAY_DASHBOARD_NODEPORT_MANIFEST],
                       "Expose Ray dashboard (NodePort, kind only)")


def resolve_ray_address(explicit: str | None = None) -> str | None:
    """Return the Ray address to connect to, if any (arg > env)."""
    return explicit or os.environ.get("RAY_ADDRESS") or None


def connect(ray_address: str | None = None) -> dict:
    """Validate connectivity to an existing Ray cluster.

    Uses the Ray client if the ``ray`` package is importable; otherwise reports
    the configured address so the caller can proceed (full validation happens at
    first ``run_code``).
    """
    address = resolve_ray_address(ray_address)
    if not address:
        return {
            "step_name": "Connect to Ray",
            "status": "FAILED",
            "message": "No Ray address provided (set RAY_ADDRESS or pass --ray-address).",
            "details": "",
        }
    try:
        import ray  # type: ignore

        ray.init(address=address, ignore_reinit_error=True)
        nodes = len(ray.nodes())
        ray.shutdown()
        return {
            "step_name": "Connect to Ray",
            "status": "SUCCESS",
            "message": f"Connected to existing Ray cluster at {address} ({nodes} node(s)).",
            "details": {"address": address, "nodes": nodes},
        }
    except ImportError:
        return {
            "step_name": "Connect to Ray",
            "status": "WARNING",
            "message": f"ray package not installed; will use address {address} at runtime.",
            "details": {"address": address},
        }
    except Exception as e:  # pragma: no cover - network dependent
        return {
            "step_name": "Connect to Ray",
            "status": "FAILED",
            "message": f"Could not connect to Ray at {address}: {e}",
            "details": str(e),
        }


def run_code(code: str, ray_address: str | None = None, num_cpus: int = 1) -> dict:
    """Execute a Python snippet on Ray as a remote task.

    This is the KubeRay-based replacement for the legacy Flask ``/execute`` pod.
    The snippet runs inside a Ray worker; anything it prints is captured and any
    ``result`` variable it sets is returned.
    """
    address = resolve_ray_address(ray_address)
    try:
        import ray  # type: ignore
    except ImportError:
        return {"status": "error", "message": "ray package not installed; add ray[default] to requirements."}

    try:
        ray.init(address=address, ignore_reinit_error=True)

        @ray.remote(num_cpus=num_cpus)
        def _exec(src: str) -> dict:
            import io
            import contextlib
            import traceback

            g: dict = {"result": None}
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(src, g)  # noqa: S102 - sandboxed in a Ray worker
                return {"status": "success", "stdout": buf.getvalue(), "result": repr(g.get("result"))}
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "error",
                    "message": str(exc),
                    "stdout": buf.getvalue(),
                    "traceback": traceback.format_exc(),
                }

        out = ray.get(_exec.remote(code))
        return out
    except Exception as e:  # pragma: no cover - network dependent
        return {"status": "error", "message": f"Ray execution failed: {e}"}
    finally:
        try:
            import ray  # type: ignore

            ray.shutdown()
        except Exception:
            pass
