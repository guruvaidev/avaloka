"""Avaloka Serve — a REST inference API on Kubernetes from an MLflow model.

The pipeline is exactly the spec's deployment promise made operational:

    MLflow model  ->  Docker image  ->  Kubernetes Deployment+Service  ->  REST API

Avaloka reuses the model the swarm already validated and packaged: the
``Dockerfile`` copies ``model/`` and the typed ``service.py`` (FastAPI ``/predict``
+ ``/health``); the MLflow descriptor under ``mlflow/`` makes the model
registry-ready. It reuses the existing cluster helpers in
``app.infra.k8s_invoker`` to read the service IP/port.

By default this is a *plan* (it prints every command). Pass ``apply=True`` to
actually build the image and apply the manifests — guarded by tool/cluster
availability so it degrades to the plan when they're absent.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class InferencePlan:
    package_dir: str
    image: str
    service_name: str
    deployment_name: str
    target: str
    manifest: str
    mlflow_uri: str | None
    steps: list[dict[str, Any]] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    endpoint: str | None = None
    applied: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _read_manifest(pkg: Path) -> tuple[str, str, str]:
    """Pull the image, Service and Deployment names from the k8s manifest."""
    manifest = pkg / "deployment" / "kubernetes.yaml"
    image = "avaloka-inference:latest"
    service = "avaloka-inference"
    deployment = "avaloka-inference"
    if manifest.exists():
        try:
            docs = list(yaml.safe_load_all(manifest.read_text()))
            for d in docs:
                if not isinstance(d, dict):
                    continue
                if d.get("kind") == "Deployment":
                    image = d["spec"]["template"]["spec"]["containers"][0]["image"]
                    deployment = d["metadata"]["name"]
                if d.get("kind") == "Service":
                    service = d["metadata"]["name"]
        except Exception:
            pass
    return image, service, deployment


def prepare(
    package_dir: str,
    *,
    target: str = "kubernetes",
    image: str | None = None,
    mlflow_uri: str | None = None,
    registry: str | None = None,
) -> InferencePlan:
    """Validate the package and assemble the MLflow→docker→k8s plan."""
    pkg = Path(package_dir).expanduser()
    if not pkg.exists():
        raise FileNotFoundError(f"Model package not found: {package_dir}")

    required = ["model/model.pkl", "service.py", "Dockerfile", "deployment/kubernetes.yaml"]
    missing = [r for r in required if not (pkg / r).exists()]
    if mlflow_uri and not (pkg / "mlflow" / "register.py").exists():
        missing.append("mlflow/register.py")
    if missing:
        raise ValueError(
            f"Package is not deployable — missing {missing}. Run "
            "`avaloka train ... --deployable` first.")

    default_image, service, deployment = _read_manifest(pkg)
    image = image or default_image
    if registry:
        image = f"{registry.rstrip('/')}/{image.split('/')[-1]}"
    manifest = str(pkg / "deployment" / "kubernetes.yaml")

    plan = InferencePlan(
        package_dir=str(pkg), image=image, service_name=service,
        deployment_name=deployment, target=target,
        manifest=manifest, mlflow_uri=mlflow_uri,
    )

    # Step 1 — MLflow: register/resolve the model.
    if mlflow_uri:
        plan.steps.append({"stage": "mlflow", "action": "register",
                           "detail": f"Log/resolve the model in MLflow at {mlflow_uri}."})
        plan.commands.append(
            f"cd {shlex.quote(str(pkg))} && "
            f"MLFLOW_TRACKING_URI={shlex.quote(mlflow_uri)} "
            "python mlflow/register.py"
        )
    else:
        plan.steps.append({"stage": "mlflow", "action": "local",
                           "detail": "Use the bundled MLflow descriptor (mlflow/MLmodel); "
                                     "pass --mlflow-uri to push to a tracking server."})

    # Step 2 — Docker: build the model container.
    plan.steps.append({"stage": "docker", "action": "build",
                       "detail": f"Build image {image} from the model Dockerfile."})
    plan.commands.append(f"docker build -t {image} {pkg}")
    if registry:
        plan.commands.append(f"docker push {image}")

    # Step 3 — Kubernetes: apply Deployment + Service.
    plan.steps.append({"stage": "kubernetes", "action": "apply",
                       "detail": f"Apply {manifest} (Deployment + Service)."})
    plan.commands.append(f"kubectl apply -f {manifest}")
    plan.commands.append(f"kubectl rollout status deploy/{deployment}")

    # Step 4 — Endpoint: resolve the REST URL.
    plan.steps.append({"stage": "endpoint", "action": "resolve",
                       "detail": "Read the service IP/port and expose the REST API."})
    return plan


def _have(tool: str) -> bool:
    return shutil.which(tool) is not None


def _run(
    cmd: list[str],
    timeout: int = 1800,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            env=env,
            input=input_text,
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, str(exc)


def _manifest_for_image(plan: InferencePlan) -> str:
    """Render the package manifest with every inference container using plan.image."""
    docs = list(yaml.safe_load_all(Path(plan.manifest).read_text()))
    deployments = 0
    for doc in docs:
        if not isinstance(doc, dict) or doc.get("kind") != "Deployment":
            continue
        containers = (
            doc.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        if not containers:
            raise ValueError(f"Deployment {doc.get('metadata', {}).get('name')!r} has no containers")
        containers[0]["image"] = plan.image
        deployments += 1
    if not deployments:
        raise ValueError(f"No Deployment found in {plan.manifest}")
    return yaml.safe_dump_all(docs, sort_keys=False)


def apply(plan: InferencePlan, *, push: bool = False) -> InferencePlan:
    """Actually build the image, apply the manifest and resolve the endpoint.

    Safe by construction: each external step is gated on tool availability and
    failures are recorded as notes rather than raised, so a missing cluster
    yields a partial result, not a crash.
    """
    pkg = Path(plan.package_dir)

    if plan.mlflow_uri:
        registration_env = os.environ.copy()
        registration_env["MLFLOW_TRACKING_URI"] = plan.mlflow_uri
        ok, out = _run(
            [sys.executable, "mlflow/register.py"],
            cwd=pkg,
            env=registration_env,
        )
        plan.notes.append(f"mlflow register: {'ok' if ok else 'failed'} — {out[-200:]}")
        if not ok:
            return plan

    if not _have("docker"):
        plan.notes.append("docker not found — skipped image build.")
    else:
        ok, out = _run(["docker", "build", "-t", plan.image, str(pkg)])
        plan.notes.append(f"docker build: {'ok' if ok else 'failed'} — {out[-200:]}")
        if not ok:
            return plan
        if push:
            pok, pout = _run(["docker", "push", plan.image])
            plan.notes.append(f"docker push: {'ok' if pok else 'failed'} — {pout[-160:]}")
            if not pok:
                return plan

    if not _have("kubectl"):
        plan.notes.append("kubectl not found — skipped k8s apply; endpoint not resolved.")
        return plan

    try:
        rendered_manifest = _manifest_for_image(plan)
    except Exception as exc:
        plan.notes.append(f"manifest render: failed — {exc}")
        return plan

    ok, out = _run(
        ["kubectl", "apply", "-f", "-"],
        input_text=rendered_manifest,
    )
    plan.notes.append(f"kubectl apply: {'ok' if ok else 'failed'} — {out[-200:]}")
    if not ok:
        return plan

    rollout_ok, rollout_out = _run(
        ["kubectl", "rollout", "status", f"deployment/{plan.deployment_name}"],
    )
    plan.notes.append(
        f"kubectl rollout: {'ok' if rollout_ok else 'failed'} — {rollout_out[-200:]}"
    )
    if not rollout_ok:
        return plan

    # Resolve the endpoint, reusing the existing cluster helper when possible.
    endpoint = _resolve_endpoint(plan.service_name)
    if endpoint:
        plan.endpoint = endpoint
        plan.applied = True
    else:
        plan.notes.append("Service applied but IP/port not yet assigned (LoadBalancer pending).")
    return plan


def _resolve_endpoint(service_name: str, retries: int = 6, delay: int = 10) -> str | None:
    try:
        from app.infra import k8s_invoker  # reuse existing cluster integration
    except Exception:
        k8s_invoker = None

    for _ in range(retries):
        if k8s_invoker is not None:
            try:
                res = k8s_invoker.get_service_ip_and_port(service_name)
                if res.get("status") == "SUCCESS":
                    ip = res["details"].get("ip")
                    port = res["details"].get("port")
                    if ip and port:
                        return f"http://{ip}:{port}"
            except Exception:
                pass
        # Fallback: kubectl jsonpath.
        ok, ip = _run(["kubectl", "get", "svc", service_name, "-o",
                       "jsonpath={.status.loadBalancer.ingress[0].ip}"], timeout=30)
        ok2, port = _run(["kubectl", "get", "svc", service_name, "-o",
                          "jsonpath={.spec.ports[0].port}"], timeout=30)
        if ok and ok2 and ip and port:
            return f"http://{ip}:{port}"
        time.sleep(delay)
    return None


def curl_example(endpoint: str, package_dir: str) -> str:
    example = Path(package_dir) / "tests" / "example_request.json"
    body = example.read_text().replace("\n", "") if example.exists() else '{"instances": [{}]}'
    return f"curl -s {endpoint}/predict -H 'content-type: application/json' -d '{body}'"
