"""T6 — Web UI + local Supabase on Kubernetes.

Proves the full application (React UI + avaloka backend + data plane + local
Supabase) deploys and wires up on kind:

  T6.1  (no cluster)  helm renders webui + supabase; UI image artifacts exist
  T6.2  (cluster)     UI pod serves the SPA; /env-config.js carries runtime config
  T6.3  (cluster)     UI nginx reverse-proxies /health to the in-cluster avaloka API
  T6.4  (cluster)     local Supabase up (db/auth/rest/kong); gotrue /health via kong;
                      migrations Job succeeded

Cluster tiers use AVALOKA_TEST_KUBE_CONTEXT (default kind-avaloka), namespace
avaloka-test. Supabase tests skip unless supabase is deployed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import requests
import yaml

from tests.k8s.conftest import CHART_DIR, has, helm, requires_helm
from tests.k8s.helpers import k8s
from tests.k8s.helpers.portforward import port_forward

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- T6.1
@requires_helm
def test_t6_1_chart_renders_ui_and_supabase():
    # UI build artifacts present
    for rel in ("ui/Dockerfile", "ui/src/lib/runtime-env.ts",
                "ui/src/routes/__root.tsx", "ui/package.json"):
        assert (REPO_ROOT / rel).exists(), f"missing {rel}"

    # webui renders by default; enable the bundled Supabase dev stack explicitly.
    rendered = helm(
        "template", "avaloka", str(CHART_DIR), "--set", "supabase.enabled=true"
    ).stdout
    docs = [d for d in yaml.safe_load_all(rendered) if d]
    webui = [d for d in docs if d.get("kind") == "Deployment"
             and d["metadata"]["labels"].get("app.kubernetes.io/component") == "webui"]
    assert webui, "webui Deployment not rendered"
    env_entries = webui[0]["spec"]["template"]["spec"]["containers"][0]["env"]
    env = {e["name"]: e.get("value") for e in env_entries}
    assert "AVALOKA_API_UPSTREAM" in env and "SUPABASE_URL" in env
    assert env["SUPABASE_INCLUSTER_URL"] == "http://avaloka-supabase-kong:8000"
    assert {"PRIMARY_SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SERVICE_ROLE_KEY"} <= set(env)

    # Supabase is deliberately opt-in; when enabled it renders db/auth/rest/kong/migrations.
    sb = helm("template", "avaloka", str(CHART_DIR), "--set", "supabase.enabled=true").stdout
    for comp in ("supabase-db", "supabase-auth", "supabase-rest", "supabase-kong"):
        assert f"component: {comp}" in sb, f"supabase {comp} not rendered"
    assert "kind: Job" in sb and "supabase-migrate" in sb


# --------------------------------------------------------------------------- T6.1b
# The chart ships its own copy of the Supabase migrations because Helm's
# .Files.Glob cannot read outside the chart directory — it can never reference
# ui/supabase/migrations directly. Nothing keeps the two in sync, and when they
# drifted the failure was SILENT: the migrations Job runs with ON_ERROR_STOP=0 so
# it always exits 0, and T6.4 only asserts the Job "succeeded". The plan schema
# (plan_type / selected_plan / customer_accounts, added 2025-10-26) was therefore
# missing in-cluster for months and every user silently resolved to the free plan.
# This test is the guard: it fails the moment the copy falls behind the source.
SOURCE_MIGRATIONS = REPO_ROOT / "ui" / "supabase" / "migrations"
CHART_MIGRATIONS = CHART_DIR / "files" / "supabase-migrations"


def _sql_by_name(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.glob("*.sql"))}


def test_t6_1b_chart_migrations_match_ui_source():
    """The chart bundle contains every source migration exactly once.

    Re-sync with:
        cp ui/supabase/migrations/*.sql deploy/helm/avaloka/files/supabase-migrations/

    A small number of chart copies intentionally omit policies installed by the
    separate Storage service, so byte equality is not a valid invariant.
    """
    assert SOURCE_MIGRATIONS.is_dir(), f"missing {SOURCE_MIGRATIONS}"
    assert CHART_MIGRATIONS.is_dir(), f"missing {CHART_MIGRATIONS}"

    source = _sql_by_name(SOURCE_MIGRATIONS)
    chart = _sql_by_name(CHART_MIGRATIONS)
    assert source, "no migrations found in ui/supabase/migrations"

    missing = sorted(set(source) - set(chart))
    extra = sorted(set(chart) - set(source))
    assert not missing, (
        f"{len(missing)} migration(s) missing from the chart bundle — the "
        f"in-cluster Supabase would not get them. First: {missing[:3]}"
    )
    assert not extra, f"chart bundle has migrations not in the UI source: {extra[:3]}"
    assert all(chart.values()), "chart bundle contains an empty migration"


# --------------------------------------------------------------------------- cluster
pytestmark_cluster = [pytest.mark.cluster, pytest.mark.slow]


def _reachable() -> bool:
    return k8s.kubectl("get", "nodes", check=False, timeout=15).returncode == 0


@pytest.fixture(scope="module")
def ui_ready():
    if not _reachable():
        pytest.skip(f"cluster {k8s.CONTEXT} not reachable")
    if not k8s.is_kind_context() and os.getenv("AVALOKA_TEST_ALLOW_CLOUD") != "1":
        pytest.skip("non-kind context without AVALOKA_TEST_ALLOW_CLOUD=1")
    if not k8s.wait_for_deployment("avaloka-webui", timeout_s=180):
        pytest.skip("avaloka-webui not deployed (deploy with webui.enabled)\n" +
                    k8s.diagnostics("app.kubernetes.io/component=webui"))
    yield


@pytest.mark.cluster
@pytest.mark.slow
def test_t6_2_ui_serves_spa_and_runtime_config(ui_ready):
    with port_forward("svc/avaloka-webui", 80) as lp:
        root = requests.get(f"http://127.0.0.1:{lp}/", timeout=10)
        env = requests.get(f"http://127.0.0.1:{lp}/env-config.js", timeout=10)
    assert root.status_code == 200 and "<div id=\"root\">" in root.text
    assert "window._env_" in env.text
    assert "API_BASE" in env.text and "SUPABASE_URL" in env.text


@pytest.mark.cluster
@pytest.mark.slow
def test_t6_3_ui_proxies_health_to_backend(ui_ready):
    # nginx in the UI pod reverse-proxies /health to the avaloka API.
    with port_forward("svc/avaloka-webui", 80) as lp:
        r = requests.get(f"http://127.0.0.1:{lp}/health", timeout=15)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "status" in body and "graph_ready" in body, body


# --------------------------------------------------------------------------- T6.4 supabase
def _supabase_present() -> bool:
    return k8s.kubectl_ns("get", "deploy", "avaloka-supabase-kong", check=False).returncode == 0


@pytest.mark.cluster
@pytest.mark.slow
def test_t6_4_local_supabase_up_and_migrated():
    if not _reachable():
        pytest.skip(f"cluster {k8s.CONTEXT} not reachable")
    if not _supabase_present():
        pytest.skip("local Supabase not deployed (deploy with supabase.enabled=true)")
    # all four core services available
    for comp in ("supabase-db", "supabase-auth", "supabase-rest", "supabase-kong"):
        name = f"avaloka-{comp}"
        assert k8s.wait_for_deployment(name, timeout_s=300), (
            f"{name} not available\n" + k8s.diagnostics(f"app.kubernetes.io/component={comp}"))
    # gotrue health via the kong gateway
    with port_forward("svc/avaloka-supabase-kong", 8000) as lp:
        h = requests.get(f"http://127.0.0.1:{lp}/auth/v1/health", timeout=15)
    assert h.status_code == 200, h.text
    # migrations Job completed
    job = k8s.get_json("get", "job", "avaloka-supabase-migrate")
    assert (job.get("status", {}).get("succeeded") or 0) >= 1, "supabase-migrate Job did not succeed"
