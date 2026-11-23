"""The chart and the compose files declare who operates the install.

AVALOKA_DEPLOYMENT (app/core/editions.py deployment_state) decides billing, plan
caps and whether the API docs need a login. An install that does not declare it
is "undeclared" and keeps each check's old default, which is billed AND docs
public: right for nobody. So every way of running Avaloka declares it: the chart
renders config.deployment (self_hosted unless overridden) into the ConfigMap
that the API, Celery, LangGraph and UI pods read, and the compose services that
run Avaloka code set it too. Only the Avaloka-operated hosted sites override it.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHART_ROOT = REPO_ROOT / "deploy" / "helm" / "avaloka"
#: The app ConfigMap (configmap.yaml) for the release name used below.
APP_CONFIGMAP = "avaloka-config"


def _helm() -> str:
    helm = shutil.which("helm")
    if helm is None:
        message = "helm is not installed, so the chart's AVALOKA_DEPLOYMENT cannot be checked"
        if os.environ.get("CI") or os.environ.get("AVALOKA_REQUIRE_HELM"):
            pytest.fail(message + "; install helm in this pipeline")
        pytest.skip(message)
    return helm


def _render(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([_helm(), "template", "avaloka", str(CHART_ROOT), *args],
                          capture_output=True, text=True, cwd=str(REPO_ROOT))


def _deployment_in_configmap(*args: str) -> object:
    result = _render(*args)
    assert result.returncode == 0, f"helm template failed:\n{result.stderr}"
    maps = [d for d in yaml.safe_load_all(result.stdout)
            if isinstance(d, dict) and d.get("kind") == "ConfigMap"
            and d["metadata"]["name"] == APP_CONFIGMAP]
    assert len(maps) == 1, f"expected the one app ConfigMap ({APP_CONFIGMAP})"
    return (maps[0].get("data") or {}).get("AVALOKA_DEPLOYMENT")


def test_the_chart_declares_self_hosted_by_default() -> None:
    assert _deployment_in_configmap() == "self_hosted"


def test_a_values_override_declares_the_hosted_deployment(tmp_path: pathlib.Path) -> None:
    assert _deployment_in_configmap("--set", "config.deployment=avaloka_hosted") == "avaloka_hosted"
    values = tmp_path / "hosted.yaml"
    values.write_text("config:\n  deployment: avaloka_hosted\n")
    assert _deployment_in_configmap("-f", str(values)) == "avaloka_hosted"


@pytest.mark.parametrize("word", ["hosted", "avaloka-hosted", "selfhosted"])
def test_the_chart_refuses_a_word_the_app_would_read_as_undeclared(word: str) -> None:
    result = _render("--set", f"config.deployment={word}")
    assert result.returncode != 0, f"config.deployment={word} rendered; the app would read it as undeclared"
    assert "config.deployment must be self_hosted or avaloka_hosted" in result.stderr


def test_the_pods_that_run_avaloka_code_read_the_configmap() -> None:
    result = _render()
    assert result.returncode == 0, result.stderr
    readers = set()
    for doc in yaml.safe_load_all(result.stdout):
        if not isinstance(doc, dict) or doc.get("kind") != "Deployment":
            continue
        for container in doc["spec"]["template"]["spec"]["containers"]:
            refs = [e.get("configMapRef", {}).get("name", "") for e in container.get("envFrom") or []]
            if APP_CONFIGMAP in refs:
                readers.add(doc["metadata"]["name"])
    # The API, both Celery workers and LangGraph run Avaloka code.
    assert {"avaloka", "avaloka-celery", "avaloka-celery-redbeat", "avaloka-langgraph"} <= readers, sorted(readers)


def _compose(path: str) -> dict:
    return yaml.safe_load((REPO_ROOT / path).read_text())


@pytest.mark.parametrize("service", ["celery-worker", "celery-redbeat-worker"])
def test_compose_services_that_run_avaloka_code_declare_self_hosted(service: str) -> None:
    env = _compose("deploy/compose/docker-compose.scheduler.yml")["services"][service].get("environment") or {}
    assert env.get("AVALOKA_DEPLOYMENT") == "${AVALOKA_DEPLOYMENT:-self_hosted}", env


def test_the_quickstart_still_declares_self_hosted() -> None:
    text = (REPO_ROOT / "docker-compose.quickstart.yml").read_text()
    assert "AVALOKA_DEPLOYMENT: self_hosted" in text


# --------------------------------------------------------------------------- the hosted sites

OVERLAYS = CHART_ROOT / "values"
#: How each Avaloka-operated site deploys (deploy/README.md "Two GKE instances",
#: deploy/gcp-lb-prereqs.sh): test.avaloka.ai with values-gke.yaml alone,
#: app.avaloka.ai with values-gke.yaml then values-gke-app.yaml.
HOSTED_SITES = {
    "test.avaloka.ai": ["values-gke.yaml"],
    "app.avaloka.ai": ["values-gke.yaml", "values-gke-app.yaml"],
}


def _overlay_args(files: list[str]) -> list[str]:
    args: list[str] = []
    for name in files:
        args += ["-f", str(OVERLAYS / name)]
    return args


@pytest.mark.parametrize("site", sorted(HOSTED_SITES))
def test_each_hosted_site_renders_avaloka_hosted(site: str) -> None:
    assert _deployment_in_configmap(*_overlay_args(HOSTED_SITES[site])) == "avaloka_hosted"


@pytest.mark.parametrize("overlay", sorted({f for files in HOSTED_SITES.values() for f in files}))
def test_every_hosted_overlay_declares_it_itself(overlay: str) -> None:
    """Each file carries the value, so dropping it from one cannot hide behind another."""
    assert _deployment_in_configmap(*_overlay_args([overlay])) == "avaloka_hosted"


def test_the_self_hosted_gke_provisioner_declares_self_hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    """`make up PROVIDER=gcp` applies values-gke.yaml too; its install is the operator's own."""
    from unittest.mock import patch

    from app.infra import deploy_stack

    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    with patch.object(deploy_stack, "run_command") as run:
        run.return_value = {"step_name": "x", "status": "SUCCESS", "message": "", "details": ""}
        deploy_stack.deploy_avaloka(provider="gcp", object_store=False)
    argv = run.call_args.args[0]
    assert "values-gke.yaml" in " ".join(argv), "the GKE provisioner no longer applies values-gke.yaml"
    assert "config.deployment=self_hosted" in argv, f"the provisioner does not declare self_hosted: {argv}"
    # Render what it would install: its overlays and its deployment flag.
    keep: list[str] = []
    for flag, value in zip(argv, argv[1:]):
        if flag == "-f" or (flag == "--set-string" and value.startswith("config.deployment=")):
            keep += [flag, value]
    assert _deployment_in_configmap(*keep) == "self_hosted"


_HOSTED_OVERLAYS = {f for files in HOSTED_SITES.values() for f in files}
_OTHER_OVERLAYS = sorted(p.name for p in OVERLAYS.glob("*.yaml") if p.name not in _HOSTED_OVERLAYS)


def test_there_are_overlays_that_are_not_hosted() -> None:
    assert len(_OTHER_OVERLAYS) >= 4, f"overlay set collapsed: {_OTHER_OVERLAYS}"


@pytest.mark.parametrize("overlay", _OTHER_OVERLAYS)
def test_no_other_overlay_declares_the_hosted_deployment(overlay: str) -> None:
    """Only the hosted sites' overlays may say avaloka_hosted; a self-hosted user's overlay never does."""
    assert _deployment_in_configmap(*_overlay_args([overlay])) == "self_hosted"


def test_no_other_values_file_in_the_repo_declares_the_hosted_deployment() -> None:
    """Every values file under deploy/, the older deploy/avaloka/ copies included."""
    hosted = {(OVERLAYS / name).resolve() for name in _HOSTED_OVERLAYS}
    files = [p for p in sorted((REPO_ROOT / "deploy").rglob("values*.yaml")) if p.resolve() not in hosted]
    assert len(files) >= 8, f"values-file scan collapsed: {[str(p) for p in files]}"
    offenders = []
    for path in files:
        for doc in yaml.safe_load_all(path.read_text()):
            config = (doc or {}).get("config") if isinstance(doc, dict) else None
            if isinstance(config, dict) and str(config.get("deployment", "")).replace("-", "_") == "avaloka_hosted":
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"only the hosted sites' overlays may declare avaloka_hosted: {offenders}"
