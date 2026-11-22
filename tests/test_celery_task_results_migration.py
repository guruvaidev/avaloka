"""Migration 20261117120000_celery_task_results on real Postgres.

The durable copy of every Celery task result (app/core/celery_results.py).
Pinned:
  * the table is service-role only: anon and authenticated can neither read,
    write nor call the purge; service_role can;
  * applying the migration twice is a no-op;
  * purge_celery_task_results deletes finished rows past the cutoff only, at
    most batch_size per call, oldest first, and never an unfinished task;
  * the date_done index exists.

Needs AVALOKA_TEST_PG_ADMIN_URL (a role that may CREATE DATABASE) and psql.
Skips on a laptop without them and FAILS when CI is set.
"""
from __future__ import annotations

import secrets
import shutil

import pytest

from tests.test_plan_enforced_org_features import ADMIN_URL, REPO_ROOT, SUPABASE_STUB, Db, _drop, _require

pytestmark = pytest.mark.integration

MIGRATION = REPO_ROOT / "deploy" / "helm" / "avaloka" / "files" / "supabase-migrations" / "20261117120000_celery_task_results.sql"
DENIED = "permission denied"


@pytest.fixture(scope="module")
def db():
    _require(bool(ADMIN_URL), "AVALOKA_TEST_PG_ADMIN_URL is not set; the celery_task_results migration was not exercised")
    _require(shutil.which("psql") is not None, "psql is not installed")
    assert MIGRATION.is_file(), "the celery_task_results migration is missing"
    d = Db(ADMIN_URL, f"celeryres_{secrets.token_hex(6)}")
    d.psql(f'CREATE DATABASE "{d.name}"', db=d.admin_db)
    try:
        d.psql(SUPABASE_STUB)
        d.psql(MIGRATION.read_text())
        yield d
    finally:
        _drop(d)


def _as(d: Db, role: str, sql: str):
    return d.psql(f"BEGIN;\nSET LOCAL ROLE {role};\n{sql}\nCOMMIT;\n", check=False)


def _seed(d: Db, rows):
    values = ",\n".join(
        f"('{tid}', '{status}', '{{}}'::jsonb, now() - interval '{days} days')" for tid, status, days in rows)
    d.psql("TRUNCATE public.celery_task_results;\n"
           f"INSERT INTO public.celery_task_results (task_id, status, result, date_done) VALUES\n{values};")


def _ids(d: Db) -> set[str]:
    out = d.scalar("SELECT coalesce(string_agg(task_id, ',' ORDER BY task_id), '') FROM public.celery_task_results")
    return set(filter(None, out.split(",")))


def test_applying_it_twice_is_a_no_op(db):
    db.psql(MIGRATION.read_text())
    assert db.scalar("SELECT count(*) FROM pg_class WHERE relname = 'celery_task_results'") == "1"


@pytest.mark.parametrize("role", ["anon", "authenticated"])
@pytest.mark.parametrize("sql", [
    "SELECT * FROM public.celery_task_results;",
    "INSERT INTO public.celery_task_results (task_id, status) VALUES ('x', 'SUCCESS');",
    "SELECT public.purge_celery_task_results(now(), 10);",
], ids=["select", "insert", "purge"])
def test_browser_roles_have_no_access(db, role, sql):
    out = _as(db, role, sql)
    assert out.returncode != 0 and DENIED in out.stderr, f"{role} ran {sql!r}: {out.stdout} {out.stderr}"


def test_rls_is_on(db):
    assert db.scalar("SELECT relrowsecurity FROM pg_class WHERE relname = 'celery_task_results'") == "t"


def test_the_service_role_reads_writes_and_purges(db):
    out = _as(db, "service_role",
              "INSERT INTO public.celery_task_results (task_id, status, result, date_done) "
              "VALUES ('svc', 'SUCCESS', '{}'::jsonb, now()) ON CONFLICT (task_id) DO NOTHING;\n"
              "SELECT count(*) FROM public.celery_task_results WHERE task_id = 'svc';\n"
              "SELECT public.purge_celery_task_results(now() - interval '1 day', 10);")
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["1", "0"]


def test_an_unknown_status_is_refused(db):
    out = db.psql("INSERT INTO public.celery_task_results (task_id, status) VALUES ('bad', 'DONE');", check=False)
    assert out.returncode != 0 and "check constraint" in out.stderr


def test_past_retention_is_purged_and_inside_is_kept(db):
    _seed(db, [("old", "SUCCESS", 366), ("old-fail", "FAILURE", 400), ("old-revoked", "REVOKED", 500),
               ("recent", "SUCCESS", 364)])
    n = db.scalar("SELECT public.purge_celery_task_results(now() - interval '365 days', 100)")
    assert n == "3" and _ids(db) == {"recent"}


def test_an_unfinished_task_is_never_purged(db):
    _seed(db, [(f"open-{s}", s, 2000) for s in ("PENDING", "RECEIVED", "STARTED", "RETRY")])
    assert db.scalar("SELECT public.purge_celery_task_results(now(), 100)") == "0"
    assert len(_ids(db)) == 4


def test_one_call_deletes_at_most_batch_size_oldest_first(db):
    _seed(db, [(f"r{i:02d}", "SUCCESS", 400 + i) for i in range(25)])
    assert db.scalar("SELECT public.purge_celery_task_results(now() - interval '365 days', 10)") == "10"
    left = _ids(db)
    assert len(left) == 15 and {f"r{i:02d}" for i in range(15)} == left, "a batch did not take the oldest rows"


@pytest.mark.parametrize("batch", ["0", "10001", "NULL"])
def test_an_unbounded_batch_is_refused(db, batch):
    out = db.psql(f"SELECT public.purge_celery_task_results(now(), {batch});", check=False)
    assert out.returncode != 0 and "batch_size" in out.stderr


def test_the_purge_uses_the_date_done_index(db):
    assert db.scalar("SELECT count(*) FROM pg_indexes WHERE indexname = 'celery_task_results_date_done_idx'") == "1"
