"""The rendered migrate hook waits, bounded, for gotrue's auth.jwt() before applying anything.

post-install only means the gotrue Deployment was created. A hook that applied the
baseline before gotrue had migrated the auth schema recorded 20260818080148 with
its support-ticket policies missing (tests/test_migrate_waits_for_auth.py runs
that against Postgres). This checks what the chart renders: the Job's timeout,
and that the script it mounts waits before the first migration and fails, rather
than proceeds, when the wait runs out.
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
CHART_ROOT = REPO_ROOT / "deploy" / "helm" / "avaloka"

# helm's default --timeout, which also bounds a post-install hook.
HELM_DEFAULT_TIMEOUT_S = 300


@pytest.fixture(scope="module")
def render() -> list[dict]:
    helm = shutil.which("helm")
    if helm is None:
        message = "helm is not installed, so the migrate hook's wait cannot be checked"
        if os.environ.get("CI") or os.environ.get("AVALOKA_REQUIRE_HELM"):
            pytest.fail(message + "; install helm in this pipeline")
        pytest.skip(message)
    result = subprocess.run(
        [helm, "template", "avaloka", str(CHART_ROOT),
         "--set", "supabase.enabled=true", "--set", "supabase.requireOwnCredentials=false",
         "--set-string", "supabase.db.password=render-only-placeholder",
         "--set-string", "supabase.jwtSecret=render-only-placeholder-render-only-placeholder"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"helm template failed:\n{result.stderr}"
    return [d for d in yaml.safe_load_all(result.stdout) if isinstance(d, dict)]


def _one(docs: list[dict], kind: str, suffix: str) -> dict:
    hits = [d for d in docs if d.get("kind") == kind and d["metadata"]["name"].endswith(suffix)]
    assert len(hits) == 1, f"expected one {kind} *{suffix}"
    return hits[0]


def test_the_job_sets_a_bounded_wait_under_helms_hook_timeout(render):
    job = _one(render, "Job", "-supabase-migrate")
    assert job["metadata"]["annotations"]["helm.sh/hook"] == "post-install,post-upgrade"
    container = next(c for c in job["spec"]["template"]["spec"]["containers"] if c["name"] == "migrate")
    env = {e["name"]: e.get("value") for e in container.get("env", [])}
    wait = env.get("AUTH_WAIT_SECONDS")
    assert wait is not None and wait.isdigit(), "the migrate Job does not bound its wait for gotrue"
    assert 0 < int(wait) < HELM_DEFAULT_TIMEOUT_S, f"AUTH_WAIT_SECONDS={wait} would outlast helm's hook timeout"
    assert job["spec"]["template"]["spec"]["restartPolicy"] == "OnFailure"


def test_the_mounted_script_waits_for_auth_jwt_before_the_first_migration_and_fails_on_timeout(render):
    script = _one(render, "ConfigMap", "-supabase-migrations")["data"]["migrate.sh"]
    probe = script.find("to_regprocedure('auth.jwt()')")
    first_apply = script.find("---- 1. version table")
    assert probe != -1, "the migrate script does not wait for auth.jwt()"
    assert first_apply != -1 and probe < first_apply, "the wait for auth.jwt() comes after migrations start"
    loop = script[script.rfind("until", 0, probe):first_apply]
    assert re.search(r'-ge "\$AUTH_WAIT_SECONDS"', loop), "the wait for auth.jwt() has no bound"
    assert re.search(r"auth\.jwt\(\) does not exist[^\n]*>&2\s*\n\s*exit 1", loop), (
        "the wait does not fail the Job, with a message, when it runs out"
    )
