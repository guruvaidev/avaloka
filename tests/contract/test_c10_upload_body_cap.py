"""Suite C10 -- the edge body cap is one chart value, finite, and above 300 MB.

Defect D5: a 300 MB upload to a self-hosted install was refused. The API's own
limit was one cause; the other was a hard-coded ``proxy-body-size: "100m"`` in
three ingress charts and ``client_max_body_size 100m`` in the legacy webui
nginx config, which refused it at the edge with a bare nginx 413.

Each of those is now rendered from ``upload.maxBodySize``. The API enforces the
real limit (plan or server resources); the edge value is the outer bound, so it
must stay finite: an upload without Content-Length skips the API's pre-body
check, and "0" (unlimited) here would leave such a request no bound at all.

These render the charts with the real helm binary: a value that a template
never reads would pass any check over the values file alone.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ANNOTATION = "nginx.ingress.kubernetes.io/proxy-body-size"
MB = 1024 * 1024

#: Charts whose Ingress carries the annotation (rendered standalone: the
#: deploy/avaloka umbrella pulls a remote redis chart).
INGRESS_CHARTS = (
    "deploy/avaloka-router",
    "deploy/avaloka/charts/avaloka-backend",
    "deploy/avaloka/charts/avaloka-frontend",
)
WEBUI_CHART = "deploy/helm/avaloka"


def _helm_template(chart: str, *extra: str) -> list[dict]:
    """`helm template` a chart. A missing helm fails under CI (a skipped check
    reads as a pass in a green pipeline); it may skip only on a developer box."""
    helm = shutil.which("helm")
    if helm is None:
        message = "helm is not installed, so the rendered body-cap contract cannot be checked"
        if os.environ.get("CI") or os.environ.get("AVALOKA_REQUIRE_HELM"):
            pytest.fail(message + "; install helm in this pipeline")
        pytest.skip(message)
    result = subprocess.run(
        [helm, "template", "c10", str(REPO_ROOT / chart), *extra],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"helm template {chart} failed:\n{result.stderr}"
    return [doc for doc in yaml.safe_load_all(result.stdout) if isinstance(doc, dict)]


def _default_value(chart: str) -> str:
    values = yaml.safe_load((REPO_ROOT / chart / "values.yaml").read_text(encoding="utf-8"))
    return str(values["upload"]["maxBodySize"])


def _nginx_bytes(size: str) -> int:
    """nginx size syntax: a number with an optional k/m/g suffix. 0 means unlimited."""
    match = re.fullmatch(r"(\d+)([kKmMgG]?)", size.strip())
    assert match, f"{size!r} is not an nginx size"
    number, unit = int(match.group(1)), match.group(2).lower()
    return number * {"": 1, "k": 1024, "m": MB, "g": 1024 * MB}[unit]


def _ingress_caps(docs: list[dict]) -> list[str]:
    ingresses = [d for d in docs if d.get("kind") == "Ingress"]
    assert ingresses, "the chart rendered no Ingress, so the cap was not checked"
    return [str((d["metadata"].get("annotations") or {}).get(ANNOTATION)) for d in ingresses]


def _webui_nginx_conf(docs: list[dict]) -> str:
    (conf,) = [
        d for d in docs
        if d.get("kind") == "ConfigMap" and d["metadata"]["name"].endswith("-webui-nginx-config")
    ]
    return conf["data"]["default.conf.template"]


@pytest.mark.parametrize("chart", INGRESS_CHARTS + (WEBUI_CHART,))
def test_c10_01_the_default_is_finite_and_above_300_mb(chart: str) -> None:
    size = _nginx_bytes(_default_value(chart))
    assert size != 0, f"{chart}: upload.maxBodySize 0 is unlimited at the edge"
    assert size > 300 * MB, f"{chart}: {size / MB:.0f} MB refuses the 300 MB self-hosted upload"


@pytest.mark.parametrize("chart", INGRESS_CHARTS)
def test_c10_02_the_rendered_annotation_is_the_value(chart: str) -> None:
    assert _ingress_caps(_helm_template(chart)) == [_default_value(chart)]
    assert _ingress_caps(_helm_template(chart, "--set-string", "upload.maxBodySize=2g")) == ["2g"]


@pytest.mark.parametrize("chart", INGRESS_CHARTS)
def test_c10_03_a_stray_annotation_cannot_override_the_value(chart: str) -> None:
    """One source of truth: a proxy-body-size left in ingress.annotations is dropped,
    not rendered as a duplicate key that nginx resolves unpredictably."""
    escaped = ANNOTATION.replace(".", r"\.")
    docs = _helm_template(chart, "--set-string", f"ingress.annotations.{escaped}=100m")
    assert _ingress_caps(docs) == [_default_value(chart)]


def test_c10_04_the_webui_nginx_config_reads_the_value() -> None:
    conf = _webui_nginx_conf(_helm_template(WEBUI_CHART))
    assert f"client_max_body_size {_default_value(WEBUI_CHART)};" in conf
    conf = _webui_nginx_conf(_helm_template(WEBUI_CHART, "--set-string", "upload.maxBodySize=2g"))
    assert "client_max_body_size 2g;" in conf
