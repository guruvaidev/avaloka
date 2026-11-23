"""Every place that starts app code switches the third-party usage beacons off.

daft pings daft.gateway.scarf.sh and osstelemetry.io on import and on every
query; mlflow, onnxruntime, Ray, langgraph-cli and others have beacons of
their own (app/core/thirdparty_optout.py says what each switch silences). No
customer data goes, but each is undisclosed third-party egress that tells the
library's vendor where a deployment runs.

The app package switches them off when it is imported. That is not enough on
its own: a beacon that fires before any app module loads -- ``langgraph dev``,
a Ray head, the mlflow server, a Ray worker running a submitted script -- only
sees the process environment. So the same switches are pinned here in every
deploy surface:

* every Dockerfile in the repository, in every build stage, unless it is
  named in EXEMPT_DOCKERFILES with the reason it runs none of these libraries;
* the rendered chart: every container that runs the app image, through the
  ConfigMap it loads or its own env; the vLLM and chroma servers' own switch;
* the umbrella chart's backend ConfigMap and the RayCluster manifest;
* every compose service that runs app code; chroma's own switch.

tests/test_thirdparty_optout.py proves at runtime that the image's values do
stop daft's beacon. This file only proves they are everywhere.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import pathlib
import re
import shutil
import subprocess
from typing import Dict, Iterator, List, Tuple

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: The switches, as the deployment sets them. DO_NOT_TRACK is "true", not "1":
#: daft accepts either, but mlflow 3.14 honours only the literal "true".
EXPECTED: Dict[str, str] = {
    "DO_NOT_TRACK": "true",
    "SCARF_NO_ANALYTICS": "true",
    "ORT_DISABLE_TELEMETRY": "1",
    "ANONYMIZED_TELEMETRY": "False",
    "MLFLOW_DISABLE_TELEMETRY": "true",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DVC_NO_ANALYTICS": "1",
    "RAY_USAGE_STATS_ENABLED": "0",
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    "LANGGRAPH_CLI_NO_ANALYTICS": "1",
    "LANGGRAPH_NO_VERSION_CHECK": "true",
}

#: Dockerfiles that need none of the switches, and why. A new Dockerfile is
#: checked by default; leaving it out of the switches is a decision made here.
EXEMPT_DOCKERFILES: Dict[str, str] = {
    "deploy/docker/Dockerfile.functions": "Supabase edge runtime (Deno); no Python",
    "ui/Dockerfile": "Node: TanStack Start and Vite, neither has usage telemetry",
    "deploy/avaloka/charts/avaloka-frontend/docker/Dockerfile.frontend": "Node dev server for the old frontend",
}

#: Images the repository builds that run app code. The discovery below must
#: find at least these, or it is not looking where the images are.
KNOWN_APP_DOCKERFILES = (
    "deploy/docker/Dockerfile.api",
    "deploy/docker/Dockerfile.ray",
    "app/data_transfer_docker/Dockerfile",
    "app/agents/mta_v2/training_docker_image/Dockerfile",
    "app/agents/mta_v2/inference_service_image/Dockerfile",
)

_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache"}


def _walk(pattern: str) -> List[str]:
    out = []
    for path in REPO_ROOT.rglob(pattern):
        rel = path.relative_to(REPO_ROOT)
        if path.is_file() and not _SKIP_DIRS.intersection(rel.parts):
            out.append(rel.as_posix())
    return sorted(out)


# --------------------------------------------------------------------------- Dockerfiles

def _instructions(text: str) -> Iterator[Tuple[str, str]]:
    """(INSTRUCTION, arguments) per Dockerfile instruction, continuations joined."""
    logical = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not logical and (not stripped or stripped.startswith("#")):
            continue
        if logical and stripped.startswith("#"):
            continue  # a comment inside a continued instruction
        if stripped.endswith("\\"):
            logical += stripped[:-1] + " "
            continue
        logical += stripped
        word, _, rest = logical.partition(" ")
        yield word.upper(), rest.strip()
        logical = ""


def dockerfile_stages(text: str) -> List[Dict[str, str]]:
    """The ENV each build stage sets, in order (one dict per FROM)."""
    stages: List[Dict[str, str]] = []
    for word, rest in _instructions(text):
        if word == "FROM":
            stages.append({})
        elif word == "ENV" and stages:
            for key, value in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|\S+)', rest):
                stages[-1][key] = value.strip('"')
    return stages


def _gaps(env: Dict[str, str]) -> List[str]:
    return sorted(f"{k}={v}" for k, v in EXPECTED.items() if env.get(k) != v)


def test_the_app_module_pins_the_same_switches():
    """app/core/thirdparty_optout.py is what the app, the test session and Ray jobs use."""
    spec = importlib.util.find_spec("app.core.thirdparty_optout")
    assert spec is not None, "app/core/thirdparty_optout.py is missing"
    module = importlib.import_module("app.core.thirdparty_optout")
    assert module.THIRDPARTY_TELEMETRY_OPTOUT_ENV == EXPECTED


def test_the_dockerfile_discovery_finds_the_app_images():
    found = _walk("*Dockerfile*")
    missing = [p for p in KNOWN_APP_DOCKERFILES if p not in found]
    assert not missing, f"discovery missed {missing}; found {found}"


def test_every_dockerfile_switches_the_beacons_off_or_is_exempt():
    missing = {}
    for rel in _walk("*Dockerfile*"):
        if rel in EXEMPT_DOCKERFILES:
            continue
        stages = dockerfile_stages((REPO_ROOT / rel).read_text("utf-8", errors="ignore"))
        assert stages, f"{rel}: no FROM found"
        for index, env in enumerate(stages):
            gaps = _gaps(env)
            if gaps:
                missing[f"{rel} (stage {index + 1} of {len(stages)})"] = gaps
    assert not missing, (
        "images that would ship with third-party usage beacons on (add the ENV block "
        f"from app/core/thirdparty_optout.py, or exempt the file with a reason): {missing}")


def test_the_exemptions_are_real():
    """Each exempt file exists and installs none of the libraries the switches are for."""
    libraries = re.compile(r"requirements|daft|mlflow|onnxruntime|chromadb|huggingface|"
                           r"transformers|\bdvc\b|kedro|\bray\b|streamlit|langgraph", re.I)
    for rel, reason in EXEMPT_DOCKERFILES.items():
        path = REPO_ROOT / rel
        assert path.is_file(), f"stale exemption: {rel} ({reason}) no longer exists"
        hits = sorted({m.group(0) for m in libraries.finditer(path.read_text("utf-8", errors="ignore"))})
        assert not hits, f"{rel} is exempt as '{reason}' but mentions {hits}"


def test_the_stage_parser_reads_a_continued_env():
    """Pins the parser the image checks rely on, so they cannot pass on a misread."""
    text = (
        "FROM python:3.11-slim AS builder\n"
        "# comment\n"
        "ENV DO_NOT_TRACK=true \\\n"
        "    # inline comment\n"
        "    SCARF_NO_ANALYTICS=\"true\"\n"
        "RUN echo hi\n"
        "FROM python:3.11-slim\n"
        "ENV OTHER=1\n"
    )
    assert dockerfile_stages(text) == [
        {"DO_NOT_TRACK": "true", "SCARF_NO_ANALYTICS": "true"}, {"OTHER": "1"}]


# --------------------------------------------------------------------------- charts

def _helm_template(chart: pathlib.Path, *args: str) -> list:
    helm = shutil.which("helm")
    if helm is None:
        # As in test_c2: a skip reads as a pass in a summary, so CI fails instead.
        message = "helm is not installed; the rendered-chart opt-out checks did not run"
        if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
            pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
        pytest.skip(message)
    run = subprocess.run([helm, "template", "avaloka", str(chart), *args],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]
    return [d for d in yaml.safe_load_all(run.stdout) if d]


def _pod_specs(docs) -> Iterator[Tuple[str, dict]]:
    for doc in docs:
        spec = doc.get("spec") or {}
        if doc.get("kind") == "CronJob":
            spec = spec.get("jobTemplate", {}).get("spec", {})
        pod = (spec.get("template") or {}).get("spec")
        if pod:
            yield f"{doc['kind']}/{doc['metadata']['name']}", pod


def _effective_env(container: dict, configmaps: Dict[str, dict]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for source in container.get("envFrom") or []:
        ref = source.get("configMapRef")
        if ref:
            env.update({k: str(v) for k, v in (configmaps.get(ref["name"]) or {}).items()})
    for item in container.get("env") or []:
        if "value" in item:
            env[item["name"]] = str(item["value"])
    return env


def test_every_app_container_in_the_chart_switches_the_beacons_off():
    chart = REPO_ROOT / "deploy" / "helm" / "avaloka"
    values = yaml.safe_load((chart / "values.yaml").read_text("utf-8"))
    app_image = values["image"]["repository"]
    docs = _helm_template(chart, "--set", "executeService.enabled=true",
                          "--set", "localLLM.enabled=true", "--set", "localLLM.engine=vllm")
    configmaps = {d["metadata"]["name"]: d.get("data") or {} for d in docs if d.get("kind") == "ConfigMap"}

    checked, missing = [], {}
    vllm = chroma = 0
    for workload, pod in _pod_specs(docs):
        for container in (pod.get("containers") or []) + (pod.get("initContainers") or []):
            env = _effective_env(container, configmaps)
            image = str(container.get("image", ""))
            name = f"{workload}/{container['name']}"
            if image.startswith(app_image):
                checked.append(name)
                if _gaps(env):
                    missing[name] = _gaps(env)
            elif image.startswith("vllm/"):
                vllm += 1
                if env.get("VLLM_NO_USAGE_STATS") != "1":
                    missing[name] = ["VLLM_NO_USAGE_STATS=1"]
            elif "chroma" in image:
                chroma += 1
                if env.get("ANONYMIZED_TELEMETRY") != "False":
                    missing[name] = ["ANONYMIZED_TELEMETRY=False"]
    # API, celery worker, redbeat, executor, langgraph, mcp, mlflow, execute-service.
    assert len(checked) >= 8, f"only {checked} run the app image; is the render complete?"
    assert vllm and chroma, "the vLLM and chroma servers were not rendered"
    assert not missing, f"chart containers with third-party usage beacons on: {missing}"


def test_the_umbrella_backend_configmap_switches_the_beacons_off():
    chart = REPO_ROOT / "deploy" / "avaloka" / "charts" / "avaloka-backend"
    docs = _helm_template(chart)
    configmaps = [d for d in docs if d.get("kind") == "ConfigMap" and d["metadata"]["name"].endswith("-config")]
    assert len(configmaps) == 1, [d["metadata"]["name"] for d in docs if d.get("kind") == "ConfigMap"]
    assert not _gaps(configmaps[0].get("data") or {}), _gaps(configmaps[0].get("data") or {})


def test_every_ray_pod_switches_the_beacons_off():
    path = REPO_ROOT / "deploy" / "helm" / "ray" / "raycluster.yaml"
    clusters = [d for d in yaml.safe_load_all(path.read_text("utf-8")) if d and d.get("kind") == "RayCluster"]
    assert clusters, f"no RayCluster in {path}"
    missing, seen = {}, 0
    for cluster in clusters:
        spec = cluster["spec"]
        groups = [("head", spec["headGroupSpec"])] + [
            (g.get("groupName", "worker"), g) for g in spec.get("workerGroupSpecs") or []]
        for group, group_spec in groups:
            for container in group_spec["template"]["spec"]["containers"]:
                seen += 1
                gaps = _gaps(_effective_env(container, {}))
                if gaps:
                    missing[f"{cluster['metadata']['name']}/{group}/{container['name']}"] = gaps
    assert seen >= 2, "expected a head and at least one worker group"
    assert not missing, f"Ray pods with third-party usage beacons on: {missing}"


# --------------------------------------------------------------------------- compose

def _compose_files() -> List[str]:
    return [p for p in _walk("*.y*ml")
            if re.search(r"(^|/)(docker-)?compose[^/]*\.ya?ml$", p)]


def _service_env(service: dict) -> Dict[str, str]:
    env = service.get("environment") or {}
    if isinstance(env, list):
        env = dict(item.split("=", 1) if "=" in item else (item, "") for item in env)
    return {k: str(v) for k, v in env.items()}


def _built_from_exempt_dockerfile(compose_rel: str, service: dict) -> bool:
    """A service whose image is built from a Dockerfile exempt above (the Node UI) runs no app code."""
    build = service.get("build")
    if not isinstance(build, dict):
        return False
    context = (REPO_ROOT / compose_rel).parent / build.get("context", ".")
    dockerfile = (context / build.get("dockerfile", "Dockerfile")).resolve()
    try:
        return dockerfile.relative_to(REPO_ROOT).as_posix() in EXEMPT_DOCKERFILES
    except ValueError:
        return False


def test_every_compose_service_that_runs_app_code_switches_the_beacons_off():
    """App code in compose: a python image, or one built from this repository."""
    missing, app_services, chroma = {}, [], 0
    for rel in _compose_files():
        doc = yaml.safe_load((REPO_ROOT / rel).read_text("utf-8")) or {}
        for name, service in (doc.get("services") or {}).items():
            image = str(service.get("image", ""))
            env = _service_env(service)
            where = f"{rel}:{name}"
            if _built_from_exempt_dockerfile(rel, service):
                continue
            if image.startswith("python:") or "build" in service:
                app_services.append(where)
                if _gaps(env):
                    missing[where] = _gaps(env)
            elif "chroma" in image:
                chroma += 1
                if env.get("ANONYMIZED_TELEMETRY") != "False":
                    missing[where] = ["ANONYMIZED_TELEMETRY=False"]
    assert len(app_services) >= 2, f"found only {app_services}; is the discovery looking in deploy/compose?"
    assert chroma >= 1
    assert not missing, f"compose services with third-party usage beacons on: {missing}"
