"""`helm upgrade --reuse-values` of an existing install renders.

`--reuse-values` hands the NEW chart's templates the values the release was
installed with, and none of the new chart's defaults. A template that walks into
a map the old values never had -- `.Values.upload.maxBodySize` when `upload` did
not exist -- is a nil-pointer error, and Helm aborts the whole upgrade.

Two stored-values fixtures, each a chart's values.yaml with every secret-like
value blanked (swapping one in for the chart's values.yaml is what
`--reuse-values` does: the old values, the new templates):

- values-2.0-rc.yaml      72f31eee, base/2.0-rc: the oldest 2.x release chart.
- values-2026-09-12.yaml  0827b7a7 (1.6): the kind release installed 12 Sep;
                          a 1.6 install upgraded straight to 1.7.

Ported from 1.6 (8ab77d69, tests/contract/test_chart_reuse_values.py).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "deploy" / "helm" / "avaloka"
FIXTURES = ROOT / "tests" / "fixtures" / "helm"
OLD_VALUES = ["values-2.0-rc.yaml", "values-2026-09-12.yaml"]


def _helm() -> str:
    helm = shutil.which("helm")
    if helm is None:
        if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
            pytest.fail("helm is not installed and CI is set: the --reuse-values check did not run")
        pytest.skip("helm is not installed")
    return helm


def _render_with_old_values(tmp_path: Path, fixture: str, *args: str):
    chart = tmp_path / "avaloka"
    shutil.copytree(CHART, chart)
    shutil.copyfile(FIXTURES / fixture, chart / "values.yaml")
    run = subprocess.run([_helm(), "template", "avaloka", str(chart), *args],
                         capture_output=True, text=True, timeout=180)
    docs = [d for d in yaml.safe_load_all(run.stdout) if isinstance(d, dict)] if run.returncode == 0 else []
    return run.returncode, docs, run.stderr


@pytest.mark.parametrize("fixture", OLD_VALUES)
@pytest.mark.parametrize("args", [
    (),
    # celery.serviceAccount did not exist in the old values (nil map on --reuse-values).
    ("--set", "celery.enabled=true"),
    ("--set", "celery.enabled=false"),
    # The credential guard reads lineage.databaseUrl only when existingSecret is set.
    ("--set", "celery.enabled=true", "--set", "existingSecret=customer-secret"),
], ids=["defaults", "celery", "no-celery", "existing-secret"])
def test_the_chart_renders_with_the_values_of_an_existing_install(tmp_path, fixture, args) -> None:
    code, docs, err = _render_with_old_values(tmp_path, fixture, *args)
    assert code == 0, (
        f"`helm upgrade --reuse-values` of an install made with {fixture} aborts; every existing "
        "install upgraded that way is blocked:\n" + err[-800:]
    )
    assert any(d.get("kind") == "Deployment" for d in docs)


@pytest.mark.parametrize("fixture", OLD_VALUES)
def test_keys_the_old_values_lack_behave_as_the_old_install_did(tmp_path, fixture) -> None:
    """Nil-safe is not enough: a missing key must not silently change behaviour.

    - lineage.enabled (default true) must not read as false: no AVALOKA_LINEAGE=off;
    - postgres/redis persistence: exactly what the old values said; where the key
      did not exist (1.6) both ran on emptyDir, and a missing key must not switch
      them to a volumeClaimTemplate, which a StatefulSet cannot take on upgrade;
    - executor and executeService did not exist: neither appears (no new
      workload, and no AVALOKA_EXECUTOR_URL pointing at one);
    - upload.maxBodySize did not exist: the webui nginx still gets a valid
      client_max_body_size (the chart default), not an empty directive.
    """
    code, docs, err = _render_with_old_values(tmp_path, fixture)
    assert code == 0, err[-800:]
    config = next(d for d in docs if d.get("kind") == "ConfigMap" and d["metadata"]["name"] == "avaloka-config")
    data = config.get("data") or {}
    assert "AVALOKA_LINEAGE" not in data, "a missing lineage.enabled turned lineage off"
    assert "AVALOKA_EXECUTOR_URL" not in data and "AVALOKA_EXECUTE_SERVICE_URL" not in data
    old = yaml.safe_load((FIXTURES / fixture).read_text())
    for component in ("postgres", "redis"):
        # As the old values had it: a PVC only where they said persistence.enabled.
        wanted = bool(((old.get(component) or {}).get("persistence") or {}).get("enabled"))
        for s in (d for d in docs if d.get("kind") == "StatefulSet" and d["metadata"]["name"] == f"avaloka-{component}"):
            assert bool(s["spec"].get("volumeClaimTemplates")) == wanted, (
                f"avaloka-{component}: PVC {'dropped' if wanted else 'added'} on --reuse-values")
    names = {d["metadata"]["name"] for d in docs}
    assert not {n for n in names if n.endswith(("execute-service", "code-executor"))}, names
    webui = [d for d in docs if d.get("kind") == "ConfigMap" and "default.conf.template" in (d.get("data") or {})]
    for cm in webui:
        line = next(l for l in cm["data"]["default.conf.template"].splitlines() if "client_max_body_size" in l)
        assert line.strip() == "client_max_body_size 1024m;", line
