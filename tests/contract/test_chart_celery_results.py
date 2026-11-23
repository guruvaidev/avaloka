"""The chart sets the Celery result expiry and retention, nil-safe.

CELERY_RESULT_EXPIRES (seconds in Redis) and CELERY_RESULT_RETENTION_DAYS (days
in Postgres) reach every pod through the release ConfigMap. An install made
before these values existed and upgraded with --reuse-values has no
celery.resultExpires / resultRetentionDays: it must get the defaults, not a
nil-pointer abort and not an empty value. Retention outside 1-1095 fails the
render (above 365 also needs the Enterprise edition, which the app checks).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.contract.test_chart_reuse_values import CHART, OLD_VALUES, _helm


def _render(tmp_path: Path, *args: str, old_values: bool = False):
    chart = tmp_path / "avaloka"
    shutil.copytree(CHART, chart)
    if old_values:
        shutil.copyfile(OLD_VALUES, chart / "values.yaml")
    run = subprocess.run([_helm(), "template", "avaloka", str(chart), "--set", "celery.enabled=true", *args],
                         capture_output=True, text=True, timeout=180)
    docs = [d for d in yaml.safe_load_all(run.stdout) if isinstance(d, dict)] if run.returncode == 0 else []
    return run.returncode, docs, run.stderr


def _config(docs) -> dict:
    cm = next(d for d in docs if d.get("kind") == "ConfigMap" and d["metadata"]["name"] == "avaloka-config")
    return cm.get("data") or {}


def test_defaults_keep_results_seven_days_in_redis_and_a_year_in_postgres(tmp_path):
    code, docs, err = _render(tmp_path)
    assert code == 0, err[-800:]
    data = _config(docs)
    assert data.get("CELERY_RESULT_EXPIRES") == "604800", "the chart does not set the Redis result expiry"
    assert data.get("CELERY_RESULT_RETENTION_DAYS") == "365"


def test_values_override_both(tmp_path):
    code, docs, err = _render(tmp_path, "--set", "celery.resultExpires=3600",
                              "--set", "celery.resultRetentionDays=1095")
    assert code == 0, err[-800:]
    data = _config(docs)
    assert (data.get("CELERY_RESULT_EXPIRES"), data.get("CELERY_RESULT_RETENTION_DAYS")) == ("3600", "1095")


def test_an_install_without_the_values_gets_the_defaults(tmp_path):
    """--reuse-values: the 2026-09-12 values have neither key."""
    code, docs, err = _render(tmp_path, old_values=True)
    assert code == 0, err[-800:]
    data = _config(docs)
    assert (data.get("CELERY_RESULT_EXPIRES"), data.get("CELERY_RESULT_RETENTION_DAYS")) == ("604800", "365")


@pytest.mark.parametrize("days", ["0", "1096"])
def test_retention_out_of_range_fails_the_render(tmp_path, days):
    code, _docs, err = _render(tmp_path, "--set", f"celery.resultRetentionDays={days}")
    assert code != 0 and "resultRetentionDays must be between 1 and 1095" in err


# -- the one-shot backfill Job (ops.backfillCeleryResults) -------------------------

def _jobs(docs):
    return [d for d in docs if d.get("kind") == "Job" and "backfill-celery-results" in d["metadata"]["name"]]


def test_the_backfill_job_is_off_by_default(tmp_path):
    code, docs, err = _render(tmp_path)
    assert code == 0, err[-800:]
    assert not _jobs(docs)


def test_the_backfill_job_renders_nothing_for_an_install_without_the_values(tmp_path):
    code, docs, err = _render(tmp_path, old_values=True)
    assert code == 0, err[-800:]
    assert not _jobs(docs)


def test_enabling_the_backfill_job_runs_a_dry_run(tmp_path):
    code, docs, err = _render(tmp_path, "--set", "ops.backfillCeleryResults.enabled=true")
    assert code == 0, err[-800:]
    jobs = _jobs(docs)
    assert len(jobs) == 1, "enabling ops.backfillCeleryResults renders no Job"
    pod = jobs[0]["spec"]["template"]["spec"]
    c = pod["containers"][0]
    assert c["command"] == ["python", "/app/scripts/ops/backfill_celery_results.py"]
    assert c["args"] == ["--dry-run", "--batch", "200"]
    assert pod["restartPolicy"] == "Never" and jobs[0]["spec"]["backoffLimit"] == 0
    refs = [e.get("configMapRef", e.get("secretRef", {})).get("name") for e in c["envFrom"]]
    assert "avaloka-config" in refs and len(refs) == 2


def test_apply_runs_the_real_backfill(tmp_path):
    code, docs, err = _render(tmp_path, "--set", "ops.backfillCeleryResults.enabled=true",
                              "--set", "ops.backfillCeleryResults.apply=true",
                              "--set", "ops.backfillCeleryResults.batch=50")
    assert code == 0, err[-800:]
    jobs = _jobs(docs)
    assert jobs, "enabling ops.backfillCeleryResults renders no Job"
    assert jobs[0]["spec"]["template"]["spec"]["containers"][0]["args"] == ["--apply", "--batch", "50"]
