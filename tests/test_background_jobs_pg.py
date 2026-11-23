"""Migration 20261121110000_background_jobs.sql on real Postgres.

Runs through the chart's own tracked migrate hook (tests/test_plan_enforced_org_features
harness: Supabase stub, psql, a throwaway database per module). Pinned:

  * the file applies on a fresh database (no ml_jobs yet) and again on one where
    train-infer's 20261117130000 ml_jobs table already exists, ending with the
    same columns and the widened kind check either way; applying it twice
    changes nothing;
  * the notified_at claim: two concurrent sessions race the UPDATE, exactly one
    gets a row back;
  * row-level security: another user reads none of a user's ml_jobs,
    job_chat_notes or notifications, and no authenticated user can insert into
    job_chat_notes (service role only);
  * the check constraints match app/core/job_ledger.py's vocabularies.

Needs AVALOKA_TEST_PG_ADMIN_URL, psql and sh. Skips on a laptop without them
and FAILS when CI is set. Marked `integration`: it belongs on the pipeline's
Postgres line.
"""

from __future__ import annotations

import os
import pathlib
import secrets
import subprocess
import threading

import pytest

from tests.test_plan_enforced_org_features import (
    MIGRATIONS,
    REPO_ROOT,
    Db,
    _baseline_versions,
    _chart_set,
    _drop,
    _new_db,
    _ok,
    _uuid,
)

pytestmark = pytest.mark.integration

VERSION = "20261121110000_background_jobs"
UI_COPY = REPO_ROOT / "ui" / "supabase" / "migrations"

#: train-infer's ml_jobs table exactly as 20261117130000_ml_jobs_endpoints.sql
#: creates it on feature/1.6-trainfer-async-registry @ 6d65c6a3. That branch is
#: not checked out in CI, so the statements are carried here; the test below
#: applies them BEFORE the chart set to prove the "their migration first" order.
TRAIN_INFER_ML_JOBS = """
create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;
create table if not exists public.ml_jobs (
    id              uuid primary key default gen_random_uuid(),
    user_id         uuid not null references auth.users(id) on delete cascade,
    session_id      text,
    kind            text not null check (kind in ('training','register','deploy_endpoint','stop_endpoint','cross_validate','reap')),
    status          text not null check (status in ('queued','running','succeeded','failed','cancelled')),
    mlflow_run_id   text,
    ray_job_name    text,
    celery_task_id  text,
    error           text,
    attempts        integer not null default 0,
    created_at      timestamptz not null default now(),
    started_at      timestamptz,
    finished_at     timestamptz,
    updated_at      timestamptz not null default now()
);
create index if not exists ml_jobs_user_created on public.ml_jobs (user_id, created_at desc);
create index if not exists ml_jobs_run on public.ml_jobs (mlflow_run_id);
alter table public.ml_jobs enable row level security;
drop policy if exists "ml_jobs_select_own" on public.ml_jobs;
create policy "ml_jobs_select_own" on public.ml_jobs for select using (user_id = auth.uid());
"""

LEDGER_COLUMNS = {"thread_id", "analysis_id", "resource_type", "resource_id", "label", "fail_reason",
                  "notified_at", "notification_id", "origin"}


def _columns(d: Db, table: str) -> set[str]:
    out = d.scalar(f"SELECT string_agg(column_name, ',') FROM information_schema.columns "
                   f"WHERE table_schema='public' AND table_name='{table}'")
    return set(out.split(",")) if out else set()


def _check_def(d: Db, name: str) -> str:
    return d.scalar(f"SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = '{name}'")


def _tokens(defn: str) -> set[str]:
    import re
    return set(re.findall(r"'([a-z_]+)'", defn))


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    """Order A: the chart set on a database that never had ml_jobs."""
    d = _new_db()
    try:
        mig = _chart_set(tmp_path_factory.mktemp("bgjobs_fresh"))
        first = _ok(d.run_hook(mig))
        second = _ok(d.run_hook(mig))
        yield d, first, second
    finally:
        _drop(d)


@pytest.fixture(scope="module")
def after_train_infer(tmp_path_factory):
    """Order B: train-infer's table first, then the chart set."""
    d = _new_db()
    try:
        d.psql(TRAIN_INFER_ML_JOBS)
        mig = _chart_set(tmp_path_factory.mktemp("bgjobs_after"))
        first = _ok(d.run_hook(mig))
        yield d, first
    finally:
        _drop(d)


# ---- the file itself ----------------------------------------------------------------------

def test_migration_is_new_and_both_copies_are_identical():
    assert VERSION not in _baseline_versions()
    a = (MIGRATIONS / f"{VERSION}.sql").read_bytes()
    b = (UI_COPY / f"{VERSION}.sql").read_bytes()
    assert a == b and len(a) > 1000


def test_check_constraints_match_the_python_vocabularies(fresh):
    from app.core import job_ledger as jl
    d = fresh[0]
    assert _tokens(_check_def(d, "ml_jobs_kind_check")) == jl.KINDS
    assert _tokens(_check_def(d, "ml_jobs_fail_reason_check")) == jl.FAIL_REASONS
    assert _tokens(_check_def(d, "ml_jobs_resource_type_check")) == jl.RESOURCE_TYPES
    assert _tokens(_check_def(d, "ml_jobs_origin_check")) == jl.ORIGINS


# ---- either apply order, idempotent --------------------------------------------------------

def test_applies_on_a_fresh_database_and_again_without_change(fresh):
    d, first, second = fresh
    assert f"{VERSION}" in first
    assert LEDGER_COLUMNS <= _columns(d, "ml_jobs")
    assert {"id", "job_id", "user_id", "thread_id", "text", "delivered_at", "batch_id"} <= _columns(d, "job_chat_notes")
    assert f"applying {VERSION}\n" in first
    assert f"applying {VERSION}" not in second                     # tracked: the second hook run skips it


def test_applies_after_train_infers_table_with_the_same_result(fresh, after_train_infer):
    a = fresh[0]
    b = after_train_infer[0]
    assert _columns(a, "ml_jobs") == _columns(b, "ml_jobs")
    assert _check_def(a, "ml_jobs_kind_check") == _check_def(b, "ml_jobs_kind_check")
    assert "'transfer'" in _check_def(b, "ml_jobs_kind_check") and "'training'" in _check_def(b, "ml_jobs_kind_check")
    # the pre-existing train-infer rows survive the widening
    uid = _uuid()
    b.admin(f"INSERT INTO auth.users (id, email) VALUES ('{uid}', 'ti@example.com');"
            f"INSERT INTO public.ml_jobs (user_id, kind, status) VALUES ('{uid}', 'training', 'queued');")
    assert b.scalar("SELECT count(*) FROM public.ml_jobs") == "1"


def test_the_sql_file_run_again_by_hand_changes_nothing(fresh):
    d = fresh[0]
    before = (_columns(d, "ml_jobs"), _check_def(d, "ml_jobs_kind_check"),
              d.scalar("SELECT count(*) FROM pg_policies WHERE tablename IN ('ml_jobs','job_chat_notes')"))
    d.psql((MIGRATIONS / f"{VERSION}.sql").read_text())
    after = (_columns(d, "ml_jobs"), _check_def(d, "ml_jobs_kind_check"),
             d.scalar("SELECT count(*) FROM pg_policies WHERE tablename IN ('ml_jobs','job_chat_notes')"))
    assert before == after


# ---- the claim ------------------------------------------------------------------------------

def _seed_two_users(d: Db):
    a, b = _uuid(), _uuid()
    pa, pb = _uuid(), _uuid()
    d.admin(
        f"INSERT INTO auth.users (id, email) VALUES ('{a}', 'a-{a[:6]}@example.com'), ('{b}', 'b-{b[:6]}@example.com');"
        f"INSERT INTO public.profiles (id, user_id, full_name) VALUES ('{pa}', '{a}', 'A'), ('{pb}', '{b}', 'B');"
    )
    return (a, pa), (b, pb)


def test_two_sessions_racing_the_claim_get_exactly_one_row(fresh):
    d = fresh[0]
    (a, _), _ = _seed_two_users(d)
    jid = _uuid()
    d.admin(f"INSERT INTO public.ml_jobs (id, user_id, kind, status) VALUES ('{jid}', '{a}', 'transfer', 'succeeded');")
    claim = (f"UPDATE public.ml_jobs SET notified_at = now() WHERE id = '{jid}' AND notified_at IS NULL RETURNING id;")
    results: list[str] = []
    barrier = threading.Barrier(2)

    def go():
        barrier.wait()
        # pg_sleep inside the transaction keeps the first writer's lock held while the second arrives
        r = d.psql(f"BEGIN;\n{claim}\nSELECT pg_sleep(0.3);\nCOMMIT;")
        results.append(r.stdout.strip())

    t1, t2 = threading.Thread(target=go), threading.Thread(target=go)
    t1.start(); t2.start(); t1.join(); t2.join()
    winners = [r for r in results if jid in r]
    assert len(winners) == 1, results
    assert d.scalar(f"SELECT notified_at IS NOT NULL FROM public.ml_jobs WHERE id = '{jid}'") == "t"


# ---- row-level security --------------------------------------------------------------------

def test_another_user_sees_none_of_a_users_jobs_notes_or_notifications(fresh):
    d = fresh[0]
    (a, pa), (b, pb) = _seed_two_users(d)
    jid = _uuid()
    d.admin(
        f"INSERT INTO public.ml_jobs (id, user_id, kind, status, thread_id) VALUES ('{jid}', '{a}', 'transfer', 'succeeded', 'thread-a');"
        f"INSERT INTO public.job_chat_notes (job_id, user_id, thread_id, text) VALUES ('{jid}', '{a}', 'thread-a', 'done');"
        f"INSERT INTO public.notifications (recipient_id, actor_id, type, title, metadata) "
        f"VALUES ('{pa}', '{pa}', 'transfer.completed', 'Data transfer completed', '{{\"job_id\": \"{jid}\"}}');"
    )
    for table in ("ml_jobs", "job_chat_notes", "notifications"):
        mine = d.as_user(a, f"SELECT count(*) FROM public.{table};")
        theirs = d.as_user(b, f"SELECT count(*) FROM public.{table};")
        assert mine.returncode == 0 and mine.stdout.strip() == "1", (table, mine.stdout, mine.stderr)
        assert theirs.returncode == 0 and theirs.stdout.strip() == "0", (table, theirs.stdout, theirs.stderr)


def test_no_authenticated_user_can_write_job_chat_notes_or_ml_jobs(fresh):
    d = fresh[0]
    (a, _), _ = _seed_two_users(d)
    jid = _uuid()
    d.admin(f"INSERT INTO public.ml_jobs (id, user_id, kind, status) VALUES ('{jid}', '{a}', 'transfer', 'running');")
    r = d.as_user(a, f"INSERT INTO public.job_chat_notes (job_id, user_id, thread_id, text) VALUES ('{jid}', '{a}', 't', 'x');")
    assert r.returncode != 0 and ("row-level security" in (r.stderr + r.stdout) or "permission denied" in (r.stderr + r.stdout))
    r = d.as_user(a, f"UPDATE public.ml_jobs SET status = 'succeeded' WHERE id = '{jid}';")
    out = r.stderr + r.stdout
    assert r.returncode != 0 or d.scalar(f"SELECT status FROM public.ml_jobs WHERE id = '{jid}'") == "running", out


def test_one_note_per_job_is_a_database_rule(fresh):
    d = fresh[0]
    (a, _), _ = _seed_two_users(d)
    jid = _uuid()
    d.admin(f"INSERT INTO public.ml_jobs (id, user_id, kind, status) VALUES ('{jid}', '{a}', 'sampling', 'succeeded');")
    d.admin(f"INSERT INTO public.job_chat_notes (job_id, user_id, thread_id, text) VALUES ('{jid}', '{a}', 't', 'x');")
    r = d.psql(f"INSERT INTO public.job_chat_notes (job_id, user_id, thread_id, text) VALUES ('{jid}', '{a}', 't', 'y');", check=False)
    assert r.returncode != 0 and "job_chat_notes_job_id_key" in (r.stderr + r.stdout)


def test_vocabulary_violations_are_refused_by_the_database(fresh):
    d = fresh[0]
    (a, _), _ = _seed_two_users(d)
    for kind, reason, origin in (("nope", "NULL", "'api'"), ("transfer", "'disk_full'", "'api'"), ("transfer", "NULL", "'cron'")):
        r = d.psql(f"INSERT INTO public.ml_jobs (user_id, kind, status, fail_reason, origin) "
                   f"VALUES ('{a}', '{kind}', 'failed', {reason}, {origin});", check=False)
        assert r.returncode != 0 and "check constraint" in (r.stderr + r.stdout), (kind, reason, origin)
