"""langgraph-cli lives in the LangGraph server's image, never in the API's.

langgraph-cli[inmem] installs langgraph-api, and the 0.5.x line that langgraph
0.6 allows caps ``cryptography<45``. While the CLI sat in requirements.txt, that
cap held the API process -- which only talks HTTP to the LangGraph server
(app/api/config.py) -- at cryptography 44.0.3 with four high advisories
(GHSA-537c-gmf6-5ccf, GHSA-g6cj-pr64-35w5, GHSA-jwv3-5hgf-82ww,
GHSA-r6ph-v2qm-q3c2). Moved out, the API resolves cryptography 50.

So the server needs its own image, and every path that runs `langgraph dev`
must install requirements-langgraph-server.txt or it has no `langgraph`
command. These checks pin both halves:

  * the API side: requirements.txt names no LangGraph server distribution;
  * the server side: the file, the Dockerfile argument, the published image,
    the chart's image for the langgraph pod (rendered), the local kind build,
    and the two other places that start `langgraph dev`.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
SERVER_REQS = "requirements-langgraph-server.txt"
SERVER_ONLY = re.compile(r"^(langgraph-cli|langgraph-api|langgraph-runtime-inmem)$")


def _names(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        assert m, f"unparsed line in {path.name}: {raw!r}"
        out[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = line
    return out


# ---------------------------------------------------------------- the API side

def test_api_requirements_carry_no_langgraph_server():
    names = _names(REPO / "requirements.txt")
    assert len(names) > 50, f"parsed only {len(names)} requirements"
    found = sorted(n for n in names if SERVER_ONLY.match(n))
    assert not found, (
        f"{found} in requirements.txt: the API image installs this file, and langgraph-cli"
        f"[inmem]'s langgraph-api caps cryptography <45. It belongs in {SERVER_REQS}."
    )


# ------------------------------------------------------------- the server side

def test_server_requirements_are_the_api_set_plus_the_cli():
    text = (REPO / SERVER_REQS).read_text()
    lines = [l.split("#", 1)[0].strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    assert "-r requirements.txt" in lines, "the server runs the same graph, so the same deps"
    clis = [l for l in lines if l.startswith("langgraph-cli")]
    assert any(l.startswith("langgraph-cli[inmem]") for l in clis), clis


def test_dockerfile_api_installs_the_requirements_file_it_is_given():
    text = (REPO / "deploy/docker/Dockerfile.api").read_text()
    assert re.search(r"^ARG REQUIREMENTS=requirements\.txt$", text, re.M), \
        "the API image must default to requirements.txt"
    copy = re.search(r"^COPY (.+) /tmp/\s*$", text, re.M)
    assert copy and {"requirements.txt", SERVER_REQS} <= set(copy.group(1).split())
    assert '-r "/tmp/${REQUIREMENTS}"' in text


def test_images_workflow_publishes_the_server_image():
    wf = yaml.safe_load((REPO / ".github/workflows/images.yml").read_text())
    include = wf["jobs"]["build"]["strategy"]["matrix"]["include"]
    by_name = {e["name"]: e for e in include}
    api, lg = by_name["avaloka-api"], by_name.get("avaloka-langgraph")
    assert lg, "nothing publishes the image the chart's langgraph pod runs"
    assert lg["dockerfile"] == api["dockerfile"] == "deploy/docker/Dockerfile.api"
    assert lg.get("build_args") == f"REQUIREMENTS={SERVER_REQS}"
    assert not api.get("build_args"), "the API image must build with the default requirements"
    steps = wf["jobs"]["build"]["steps"]
    push = next(s for s in steps if s.get("uses", "").startswith("docker/build-push-action"))
    assert "matrix.build_args" in str(push["with"].get("build-args", ""))
    on = wf.get("on", wf.get(True))  # PyYAML reads a bare `on:` key as True
    assert SERVER_REQS in on["push"]["paths"]


def test_image_tag_hashes_the_server_requirements():
    import importlib.util
    spec = importlib.util.spec_from_file_location("image_tag", REPO / "scripts/image_tag.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    deps = mod.DEPENDENCY_INPUTS["langgraph"]
    assert {"requirements.txt", SERVER_REQS, "deploy/docker/Dockerfile.api"} <= set(deps)


def _render(*args: str):
    helm = shutil.which("helm")
    if helm is None:
        message = "helm is not installed; the langgraph image checks did not run"
        if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
            pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
        pytest.skip(message)
    run = subprocess.run([helm, "template", "avaloka", str(REPO / "deploy/helm/avaloka"), *args],
                         capture_output=True, text=True)
    docs = [d for d in yaml.safe_load_all(run.stdout) if d] if run.returncode == 0 else []
    return run.returncode, docs, run.stderr


def _images(docs) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for d in docs:
        if d.get("kind") not in ("Deployment", "StatefulSet", "Job", "CronJob"):
            continue
        spec = d["spec"]
        pod = (spec.get("jobTemplate", {}).get("spec", spec))["template"]["spec"]
        for c in pod.get("containers", []) + pod.get("initContainers", []):
            out.setdefault(d["metadata"]["name"], set()).add(c["image"])
    return out


@pytest.mark.parametrize("overlay,expected", [
    (None, "ghcr.io/guruvaidev/avaloka-langgraph:main"),
    ("values/values-minikube.yaml", "avaloka-langgraph:latest"),
    ("values/values-gke.yaml", "gcr.io/PROJECT_ID/avaloka-langgraph:latest"),
])
def test_chart_runs_the_langgraph_pod_from_the_server_image(overlay, expected):
    args = ["-f", str(REPO / "deploy/helm/avaloka" / overlay)] if overlay else []
    code, docs, err = _render(*args)
    assert code == 0, err[:400]
    images = _images(docs)
    assert images.get("avaloka-langgraph") == {expected}, images.get("avaloka-langgraph")
    # And nothing else runs it: the API, MCP and celery pods keep the API image.
    others = {n: i for n, i in images.items() if n != "avaloka-langgraph" and any("avaloka-langgraph" in x for x in i)}
    assert not others, others


def test_chart_refuses_a_server_image_it_cannot_derive():
    code, _, err = _render("--set", "image.repository=registry.example/custom")
    assert code != 0 and "langgraph.image.repository must be set" in err, err[:400]
    code, docs, err = _render("--set", "image.repository=registry.example/custom",
                              "--set", "langgraph.image.repository=registry.example/lg",
                              "--set", "langgraph.image.tag=v7")
    assert code == 0, err[:400]
    assert _images(docs)["avaloka-langgraph"] == {"registry.example/lg:v7"}


def test_local_kind_builds_and_loads_the_server_image():
    import inspect

    from app.infra import cluster_bootstrap, deploy_stack
    assert deploy_stack.LANGGRAPH_IMAGE == "avaloka-langgraph:latest"
    # The chart derives the name from the API image's; keep them in step.
    assert deploy_stack.API_IMAGE.replace("avaloka-api", "avaloka-langgraph") == deploy_stack.LANGGRAPH_IMAGE
    src = inspect.getsource(deploy_stack.build_images)
    assert f"REQUIREMENTS={SERVER_REQS}" in src and "LANGGRAPH_IMAGE" in src
    assert "LANGGRAPH_IMAGE" in inspect.getsource(cluster_bootstrap.provision)


@pytest.mark.parametrize("path", [
    "bitbucket-pipelines.yml",
    "deploy/avaloka/charts/avaloka-backend/docker/Dockerfile.langgraph",
])
def test_every_other_langgraph_dev_path_installs_the_server_file(path):
    text = (REPO / path).read_text()
    assert "langgraph" in text
    assert re.search(r"pip install[^\n]*-r\s+\S*" + re.escape(SERVER_REQS), text), (
        f"{path} starts the LangGraph server but does not install {SERVER_REQS}, "
        "so it has no `langgraph` command"
    )
