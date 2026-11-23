"""The chart always declares AVALOKA_DEPLOYMENT.

app/core/editions.declared_deployment is the one signal the plan gates, billing
and the plan-lookup fail mode read. Unset it is "undeclared" (treated as
self-hosted, warned once). The chart renders it into the ConfigMap every app pod
loads: self_hosted by default, avaloka_hosted when config.deployment says so,
and self_hosted for an install upgraded with the values of an older chart that
has no config.deployment key.
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
OLD_VALUES = REPO_ROOT / "tests" / "fixtures" / "helm" / "values-2026-09-12.yaml"


def _helm() -> str:
    helm = shutil.which("helm")
    if helm is None:
        message = "helm is not installed, so the chart's deployment signal cannot be checked"
        if os.environ.get("CI") or os.environ.get("AVALOKA_REQUIRE_HELM"):
            pytest.fail(message + "; install helm in this pipeline")
        pytest.skip(message)
    return helm


def _config(chart: pathlib.Path, *args: str) -> dict:
    result = subprocess.run(
        [_helm(), "template", "avaloka", str(chart), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
    )
    assert result.returncode == 0, f"helm template failed:\n{result.stderr[-800:]}"
    # The app ConfigMap (templates/configmap.yaml), not the web UI's.
    maps = [d for d in yaml.safe_load_all(result.stdout)
            if isinstance(d, dict) and d.get("kind") == "ConfigMap"
            and d["metadata"]["name"] == "avaloka-config"]
    assert len(maps) == 1, "expected one avaloka-config ConfigMap"
    return maps[0].get("data") or {}


def test_default_render_declares_self_hosted():
    assert _config(CHART_ROOT).get("AVALOKA_DEPLOYMENT") == "self_hosted"


def test_override_renders_avaloka_hosted():
    data = _config(CHART_ROOT, "--set", "config.deployment=avaloka_hosted")
    assert data.get("AVALOKA_DEPLOYMENT") == "avaloka_hosted"


def test_values_of_an_older_install_declare_self_hosted(tmp_path):
    chart = tmp_path / "avaloka"
    shutil.copytree(CHART_ROOT, chart)
    shutil.copyfile(OLD_VALUES, chart / "values.yaml")
    assert _config(chart).get("AVALOKA_DEPLOYMENT") == "self_hosted"


def test_every_app_workload_loads_the_config_map():
    # The declaration only reaches a pod that loads the ConfigMap.
    for name in ("deployment.yaml", "celery.yaml", "langgraph.yaml"):
        text = (CHART_ROOT / "templates" / name).read_text()
        assert '-config' in text and "configMapRef" in text, name
