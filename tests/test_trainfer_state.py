"""Durable training/endpoint state: migration, endpoint lifecycle, store.

Base-class of each test (how it fails on develop-1.6 @ 9cd9e122 without this
feature) is recorded in the docstring: A = fails by assertion, E = errors
because the module does not exist yet.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MIGRATION = "20261117130000_ml_jobs_endpoints.sql"
TREES = (
    REPO / "deploy" / "helm" / "avaloka" / "files" / "supabase-migrations",
    REPO / "ui" / "supabase" / "migrations",
)
FLOOR = "20261115100000"


# --------------------------------------------------------------------------- #
# T-01 migration  (A: the file does not exist on base)
# --------------------------------------------------------------------------- #

def test_migration_exists_in_both_trees_identical_and_after_the_floor():
    """The ml_* migration is present in both trees, byte-identical, and sorts
    after the 1.6 floor. (Checking the name constant alone would pass on the
    base, so the version check is made on the files actually found.)"""
    found = []
    for tree in TREES:
        matches = sorted(p for p in tree.glob("*_ml_jobs_endpoints.sql"))
        assert matches, f"no ml_jobs_endpoints migration under {tree.relative_to(REPO)}"
        found.append(matches[-1])
    assert [p.name for p in found] == [MIGRATION, MIGRATION]
    assert found[0].read_text() == found[1].read_text(), "chart copy and ui/supabase copy must be byte-identical"
    for path in found:
        m = re.match(r"^(\d{14})_", path.name)
        assert m and m.group(1) > FLOOR, f"{path.name} must sort after {FLOOR}"


def test_migration_declares_the_endpoint_lifecycle_and_server_kinds():
    sql = (TREES[0] / MIGRATION).read_text()
    for state in ("pending", "deploying", "ready", "failed", "stopped"):
        assert f"'{state}'" in sql, f"endpoint state {state} missing from the check constraint"
    assert "server in ('cpu','gpu')" in sql.replace(" ", "").replace("'cpu','gpu'", "'cpu','gpu'") or \
        re.search(r"server\s+text[^,]*check\s*\(server in \('cpu','gpu'\)\)", sql), "server kind check missing"
    for table in ("ml_jobs", "ml_models", "ml_endpoints", "ml_cross_validation"):
        assert f"create table if not exists public.{table}" in sql, table
    assert "enable row level security" in sql


def test_migration_is_listed_in_merge_order():
    """On 1.7 the merged list holds stems (tests/contract/test_migration_versions.py) and a
    migration ported from 1.6 keeps its 1.6 name, so it is also in FROM_1_6, exempt from
    the version-order checks (like the Celery one)."""
    from tests.contract.test_migration_versions import FROM_1_6, MERGED_MIGRATIONS
    stem = MIGRATION[:-len(".sql")]
    assert stem in MERGED_MIGRATIONS
    assert stem in FROM_1_6


def test_migration_holds_no_customer_data_columns():
    sql = (TREES[0] / MIGRATION).read_text().lower()
    for forbidden in ("feature_columns", "dataset_uri", "filename", "file_name", "column_names"):
        assert forbidden not in sql, forbidden


# --------------------------------------------------------------------------- #
# T-12 endpoint state machine  (E: app.core.ml_endpoints does not exist on base)
# --------------------------------------------------------------------------- #

def test_endpoint_legal_path():
    from app.core.ml_endpoints import transition
    s = "pending"
    s = transition(s, "start_deploy"); assert s == "deploying"
    s = transition(s, "deployed");     assert s == "ready"
    s = transition(s, "stop");         assert s == "stopped"
    s = transition(s, "redeploy");     assert s == "pending"


def test_endpoint_failed_path_and_redeploy():
    from app.core.ml_endpoints import transition
    assert transition("deploying", "deploy_failed") == "failed"
    assert transition("failed", "redeploy") == "pending"


@pytest.mark.parametrize("current,event", [
    ("pending", "deployed"),       # cannot be ready without deploying
    ("ready", "start_deploy"),     # cannot redeploy a live endpoint without stopping
    ("ready", "deployed"),
    ("stopped", "stop"),
    ("failed", "deployed"),
    ("pending", "redeploy"),
])
def test_endpoint_illegal_transitions_raise(current, event):
    from app.core.ml_endpoints import IllegalTransition, transition
    with pytest.raises(IllegalTransition):
        transition(current, event)


def test_endpoint_unknown_state_or_event_raise():
    from app.core.ml_endpoints import IllegalTransition, transition
    with pytest.raises(IllegalTransition):
        transition("sleeping", "stop")
    with pytest.raises(IllegalTransition):
        transition("ready", "nap")


def test_only_ready_is_servable():
    from app.core.ml_endpoints import STATES, is_servable
    assert [s for s in STATES if is_servable(s)] == ["ready"]


# --------------------------------------------------------------------------- #
# Store  (E: app.core.ml_store does not exist on base)
# --------------------------------------------------------------------------- #

class _FakeQuery:
    """Records the Supabase builder chain and returns canned rows."""

    def __init__(self, log, table, rows=None):
        self.log, self.table, self.rows = log, table, rows or []
        self.ops = []

    def __getattr__(self, name):
        def _op(*args, **kwargs):
            self.ops.append((name, args, kwargs))
            return self
        return _op

    def execute(self):
        self.log.append((self.table, list(self.ops)))
        class R:  # noqa: D401 - minimal supabase result shape
            data = self.rows
        return R()


class _FakeClient:
    def __init__(self, rows_by_table=None):
        self.log = []
        self.rows_by_table = rows_by_table or {}

    def table(self, name):
        return _FakeQuery(self.log, name, self.rows_by_table.get(name))


def test_supabase_store_upserts_json_safe_rows_on_the_primary_key():
    from app.core.ml_store import MLEndpoint, MLJob, SupabaseMLStore
    client = _FakeClient()
    store = SupabaseMLStore(client)
    job = MLJob(user_id="u1", kind="training", mlflow_run_id="run-1")
    store.create_job(job)
    store.upsert_endpoint(MLEndpoint(mlflow_run_id="run-1", user_id="u1"))

    tables = [t for t, _ in client.log]
    assert tables == ["ml_jobs", "ml_endpoints"]
    (_, job_ops), (_, ep_ops) = client.log
    assert job_ops[0][0] == "upsert" and ep_ops[0][0] == "upsert"
    job_row = job_ops[0][1][0]
    assert job_row["id"] == job.id and job_row["status"] == "queued"
    # datetimes must already be strings: the REST client cannot serialise datetime
    for key in ("created_at", "updated_at"):
        assert isinstance(job_row[key], str) and job_row[key].endswith("+00:00")
    assert job_row["started_at"] is None


def test_supabase_store_update_job_stamps_updated_at_and_scopes_by_id():
    from app.core.ml_store import SupabaseMLStore
    client = _FakeClient({"ml_jobs": [{"id": "j1", "user_id": "u1", "kind": "training", "status": "running"}]})
    store = SupabaseMLStore(client)
    finished = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
    job = store.update_job("j1", status="running", started_at=finished)
    assert job is not None and job.status == "running"
    (_, ops), = client.log
    names = [op[0] for op in ops]
    assert names == ["update", "eq"]
    payload = ops[0][1][0]
    assert payload["started_at"] == finished.isoformat()
    assert "updated_at" in payload
    assert ops[1][1] == ("id", "j1")


def test_supabase_store_reads_fail_soft_and_writes_fail_loud():
    from app.core.ml_store import MLJob, SupabaseMLStore

    class Boom:
        def table(self, name):
            raise RuntimeError("postgres down")

    store = SupabaseMLStore(Boom())
    assert store.get_job("j1") is None
    assert store.list_jobs("u1") == []
    with pytest.raises(RuntimeError):
        store.create_job(MLJob(user_id="u1", kind="training"))


def test_job_row_validates_kind_and_status():
    from app.core.ml_store import MLJob
    with pytest.raises(ValueError):
        MLJob(user_id="u1", kind="bake")
    with pytest.raises(ValueError):
        MLJob(user_id="u1", kind="training", status="sleeping")


def test_memory_store_is_selected_by_env_and_swappable(monkeypatch):
    from app.core import ml_store
    monkeypatch.setenv("AVALOKA_ML_STORE", "memory")
    ml_store.set_ml_store(None)
    store = ml_store.get_ml_store()
    assert isinstance(store, ml_store.InMemoryMLStore)
    other = ml_store.InMemoryMLStore()
    ml_store.set_ml_store(other)
    assert ml_store.get_ml_store() is other
    ml_store.set_ml_store(None)


def test_memory_store_lists_newest_first_and_filters_by_run_and_kind():
    from app.core.ml_store import InMemoryMLStore, MLJob
    store = InMemoryMLStore()
    a = store.create_job(MLJob(user_id="u1", kind="training", mlflow_run_id="r1",
                               created_at=datetime(2026, 1, 1, tzinfo=timezone.utc)))
    b = store.create_job(MLJob(user_id="u1", kind="deploy_endpoint", mlflow_run_id="r1",
                               created_at=datetime(2026, 1, 2, tzinfo=timezone.utc)))
    store.create_job(MLJob(user_id="u2", kind="training", mlflow_run_id="r9"))
    assert [j.id for j in store.list_jobs("u1")] == [b.id, a.id]
    assert [j.id for j in store.list_jobs("u1", kind="training")] == [a.id]
    assert store.list_jobs("u1", mlflow_run_id="nope") == []
    store.update_job(a.id, status="succeeded")
    assert [j.id for j in store.active_jobs()] != [] and a.id not in [j.id for j in store.active_jobs()]
