"""`helm upgrade --reuse-values` of an existing install renders.

`--reuse-values` hands the NEW chart's templates the values the release was
installed with, and none of the new chart's defaults. A template that walks into
a map the old values never had -- `.Values.celery.serviceAccount.annotations`
when `celery.serviceAccount` did not exist -- is a nil-pointer error, and Helm
aborts the whole upgrade. That happened on the kind release (installed 12 Sep
from 0827b7a7): `templates/serviceaccount.yaml:21` (from ab90e6c5) stopped the
upgrade, so no Secret, ConfigMap or workload changed.

tests/fixtures/helm/values-2026-09-12.yaml is that chart's values.yaml (the
release's `helm get values --all` has exactly its keys), with every secret-like
value blanked. Swapping it in for the chart's values.yaml is what
`--reuse-values` does: the old values, the new templates.
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
OLD_VALUES = ROOT / "tests" / "fixtures" / "helm" / "values-2026-09-12.yaml"


def _helm() -> str:
    helm = shutil.which("helm")
    if helm is None:
        if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
            pytest.fail("helm is not installed and CI is set: the --reuse-values check did not run")
        pytest.skip("helm is not installed")
    return helm


def _render_with_old_values(tmp_path: Path, *args: str):
    chart = tmp_path / "avaloka"
    shutil.copytree(CHART, chart)
    shutil.copyfile(OLD_VALUES, chart / "values.yaml")
    run = subprocess.run([_helm(), "template", "avaloka", str(chart), *args],
                         capture_output=True, text=True, timeout=180)
    docs = [d for d in yaml.safe_load_all(run.stdout) if isinstance(d, dict)] if run.returncode == 0 else []
    return run.returncode, docs, run.stderr


@pytest.mark.parametrize("args", [
    ("--set", "celery.enabled=true"),
    ("--set", "celery.enabled=false"),
    # The credential guard reads lineage.databaseUrl only when existingSecret is set.
    ("--set", "celery.enabled=true", "--set", "existingSecret=customer-secret"),
], ids=["celery", "no-celery", "existing-secret"])
def test_the_chart_renders_with_the_values_of_an_existing_install(tmp_path, args) -> None:
    code, docs, err = _render_with_old_values(tmp_path, *args)
    assert code == 0, (
        "`helm upgrade --reuse-values` of an install from the 2026-09-12 chart aborts; every "
        "existing install upgraded that way is blocked:\n" + err[-800:]
    )
    assert any(d.get("kind") == "Deployment" for d in docs)


def test_keys_the_old_values_lack_behave_as_the_old_install_did(tmp_path) -> None:
    """Nil-safe is not enough: a missing key must not silently change behaviour.

    - lineage.enabled (default true) must not read as false: no AVALOKA_LINEAGE=off;
    - postgres/redis persistence did not exist and both ran on emptyDir: a
      missing key must not switch them to a volumeClaimTemplate, which a
      StatefulSet cannot take on an upgrade anyway;
    - no executeService (new, default off).
    """
    code, docs, err = _render_with_old_values(tmp_path, "--set", "celery.enabled=true")
    assert code == 0, err[-800:]
    config = next(d for d in docs if d.get("kind") == "ConfigMap" and d["metadata"]["name"] == "avaloka-config")
    assert "AVALOKA_LINEAGE" not in (config.get("data") or {}), "a missing lineage.enabled turned lineage off"
    for name in ("avaloka-postgres", "avaloka-redis"):
        sts = [d for d in docs if d.get("kind") == "StatefulSet" and d["metadata"]["name"] == name]
        for s in sts:
            assert not s["spec"].get("volumeClaimTemplates"), f"{name} switched to a PVC on --reuse-values"
    assert not [d for d in docs if d["metadata"]["name"].endswith("execute-service")]
