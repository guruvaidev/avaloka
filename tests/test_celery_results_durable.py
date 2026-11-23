"""Celery task results: Postgres is the record, Redis the cache (hermetic).

Before: ``result_expires=None`` and the plain Redis backend, so every result --
the whole final state, its conversation, its arguments and the output CSV as
base64 -- stayed in Redis for ever, and nowhere else. A result that did leave
Redis read as PENDING, which every reader showed as "running".

These tests drive the real backend (``celery_app.backend``) and the real
readers against an in-memory Redis and PostgREST (tests/celery_result_fakes.py).
The SQL side (table, RLS, purge function) is tests/test_celery_task_results_migration.py.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tests.celery_result_fakes import FakeRedis, FakeSupabase, iso_days_ago

import app.core.celery_app as celery_module
from app.core.celery_app import celery_app
from app.utils import convert_message_objects_to_dicts

REPO = Path(__file__).resolve().parents[1]
AGENT_TASK = "app.core.celery_app.agent_task"
PURGE_TASK = "app.core.celery_app.purge_celery_task_results"
META = "celery-task-meta-"

#: The four credential forms the chat-credential redaction covers.
# A Postgres DSN pasted into chat is the likeliest real case. Every credential-
# bearing URI is built at run time, so no URI with a literal user and password sits in a
# tracked file (the repo secret scans, test_c1 and test_c9).
PG_PASSWORD = "Pg" + "Dsn5ecret"
PG_DSN = "postgres" + "ql://" + "analyst:" + PG_PASSWORD + "@" + "warehouse.internal:5432/sales"
REDIS_PASSWORD = "S3cret" + "Pass"
REDIS_URI = "redis" + "://" + "admin:" + REDIS_PASSWORD + "@" + "cache.internal:6379/0"
SECRETS = ("Hunter2Secret", "S3cretPass", "Tr0ub4dor", "Zebra99Pw", PG_PASSWORD)
CSV_B64 = base64.b64encode(b"region,revenue\nnorth,10\n").decode()


def _messages():
    return convert_message_objects_to_dicts([
        HumanMessage(content="connect to postgres host db.internal user admin password=Hunter2Secret"),
        HumanMessage(content=f"use {REDIS_URI}"),
        HumanMessage(content=f"and the warehouse is {PG_DSN}"),
        HumanMessage(content="the password is Tr0ub4dor"),
        HumanMessage(content="pwd: Zebra99Pw"),
        AIMessage(content="Here is the revenue table."),
    ])


def _full_state():
    return {
        "messages": _messages(),
        "user_id": "u-1",
        "schedule_id": "sched-1",
        "output_location": "/tmp/work/out/result.csv",
        "output_file_data": {"filename": "result.csv", "content": "data:text/csv;base64," + CSV_B64, "size": 24},
        "execution_result": {"status": "success", "stdout": f"dsn {REDIS_URI}"},
        "generated_code": "df[df.pw == 'Tr0ub4dor']",
        "task_run_statistics": {"start": 1_700_000_000.0, "end": 1_700_000_060.0, "time_spent": 60.0},
        "available_samples": ["random_baseline"],
    }


def _request(args=None, task=AGENT_TASK):
    return SimpleNamespace(
        task=task, args=args if args is not None else [{"user_id": "u-1", "schedule_id": "sched-1",
                                                        "messages": _messages(),
                                                        "db_password": "Hunter2Secret"}],
        kwargs={}, hostname="worker-1", retries=0, delivery_info={"routing_key": "celery"},
        group=None, parent_id=None, children=[], stamps=None, ignore_result=False,
    )


@pytest.fixture
def stores(monkeypatch):
    from celery.backends.redis import RedisBackend
    import app.agents.sampling_persistence as sp

    r, db = FakeRedis(), FakeSupabase()
    monkeypatch.setattr(RedisBackend, "client", property(lambda self: r))
    monkeypatch.setattr(sp, "get_supabase_client", lambda: db)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    celery_app.backend._cache.clear()
    getattr(celery_app.backend, "_misses", {}).clear()
    yield SimpleNamespace(r=r, db=db)
    celery_app.backend._cache.clear()


def _store_success(stores, tid="run-1", state=None):
    celery_app.backend.store_result(tid, state if state is not None else _full_state(), "SUCCESS",
                                    request=_request())
    return tid


def _durable_rows(stores):
    return stores.db.rows("celery_task_results")


def _expire_from_redis(stores, tid):
    stores.r.delete(META + tid)
    celery_app.backend._cache.clear()


def _scheduled_run(stores, tid, status="success"):
    stores.db.tables.setdefault("scheduled_runs", {})[tid] = {
        "run_id": tid, "schedule_id": "sched-1", "status": status, "error": None,
        "result": {"logs": ["[execution] status=success", "[assistant] Revenue rose 4%."],
                   "output_json": [{"region": "north", "revenue": 10}],
                   "run_truth": {"outcome": "succeeded"}, "message": None},
    }


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------

def test_results_stay_in_redis_seven_days_by_default():
    assert celery_app.conf.result_expires == 7 * 24 * 3600, (
        "result_expires is not 7 days: results either never leave Redis or leave it too soon")
    assert celery_app.backend.expires == 7 * 24 * 3600


def test_result_expiry_is_read_from_the_environment():
    code = "from app.core.celery_app import celery_app; print(celery_app.conf.result_expires)"
    env = {**os.environ, "CELERY_RESULT_EXPIRES": "600", "PYTHONPATH": str(REPO)}
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-1500:]
    assert out.stdout.strip().splitlines()[-1] == "600", "CELERY_RESULT_EXPIRES is ignored"


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------

def test_a_finished_result_is_written_to_postgres_and_cached_with_a_ttl(stores):
    tid = _store_success(stores)
    rows = _durable_rows(stores)
    assert [r["task_id"] for r in rows] == [tid], "no durable row: the result exists only in Redis"
    row = rows[0]
    assert row["status"] == "SUCCESS" and row["user_id"] == "u-1" and row["schedule_id"] == "sched-1"
    assert row["name"] == AGENT_TASK and row["date_done"]
    assert row["result"]["output_location"] == "/tmp/work/out/result.csv"
    assert row["result"]["task_run_statistics"]["time_spent"] == 60.0
    assert stores.r.ttl(META + tid) == 7 * 24 * 3600


def test_nothing_from_the_conversation_reaches_the_stored_result(stores):
    """No messages, no task arguments, no output CSV, none of the four credential forms."""
    tid = _store_success(stores)
    rows = _durable_rows(stores)
    assert rows, "no durable row was written"
    stored = {"postgres": json.dumps(rows[0]), "redis": str(stores.r.get(META + tid))}
    for where, blob in stored.items():
        for secret in SECRETS:
            assert secret not in blob, f"a typed credential reached the {where} copy of the result"
        assert "base64," not in blob, f"the output CSV is inlined in the {where} copy"
        assert '"messages"' not in blob, f"the conversation is in the {where} copy"
        assert '"args"' not in blob and '"kwargs"' not in blob, f"the task arguments are in the {where} copy"
        assert "generated_code" not in blob and "stdout" not in blob


def test_a_failure_keeps_the_exception_class_and_nothing_it_said(stores):
    try:
        raise RuntimeError("could not connect: password=Hunter2Secret")
    except RuntimeError as exc:
        celery_app.backend.store_result("run-f", exc, "FAILURE", traceback="Traceback ... Hunter2Secret",
                                        request=_request())
    rows = _durable_rows(stores)
    assert rows and rows[0]["status"] == "FAILURE", "no durable row for the failure"
    assert rows[0]["result"]["exc_type"] == "RuntimeError"
    assert rows[0]["traceback"] is None
    assert "Hunter2Secret" not in json.dumps(rows[0])


def test_postgres_down_keeps_the_result_in_redis_without_expiry(stores, caplog):
    stores.db.down = True
    tid = _store_success(stores)
    assert stores.r.get(META + tid), "the result was lost"
    assert stores.r.ttl(META + tid) == -1, "a result that is not durable must not expire or be evictable"
    assert tid in stores.r.smembers("celery-durable-pending"), "the result is not queued for the backfill"
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert any(tid in m for m in errors), "a failed durable write must be logged loudly"
    assert not any(s in m for m in errors for s in SECRETS)


def test_the_run_now_inline_caller_still_gets_the_full_state(monkeypatch):
    """Summaries are what the backend stores; agent_task's own return value is unchanged."""
    monkeypatch.setattr(celery_module, "_run_agent_task", lambda state: _full_state())
    monkeypatch.setattr(celery_module, "_notify_job_finished", lambda *a, **k: None)
    out = celery_module.agent_task({"task_schedule": {"task_type": "execute"}})
    assert out["output_file_data"]["content"].startswith("data:text/csv;base64,")
    assert out["messages"]


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------

def test_a_result_gone_from_redis_is_loaded_back_from_postgres(stores):
    from celery.result import AsyncResult

    tid = _store_success(stores)
    _expire_from_redis(stores, tid)
    res = AsyncResult(tid, app=celery_app)
    assert res.state == "SUCCESS", "a finished result that left Redis reads as PENDING"
    assert res.result["output_location"] == "/tmp/work/out/result.csv"
    assert res.date_done is not None
    assert 0 < stores.r.ttl(META + tid) <= 7 * 24 * 3600, "the cache was not refilled with a TTL"


def test_a_result_seen_while_queued_still_loads_back_after_it_leaves_redis(stores):
    """A reader that polled the task while it was queued, then got the result
    (from Redis or a pub/sub message), still reads Postgres once the key is gone."""
    from celery.result import AsyncResult

    assert AsyncResult("run-q", app=celery_app).state == "PENDING"   # polled while queued
    _store_success(stores, tid="run-q")
    # The reader is another process (the API): the worker's store never cleared
    # the reader's own record of that miss.
    getattr(celery_app.backend, "_note_miss", lambda _tid: None)("run-q")
    celery_app.backend.decode_result(stores.r.get(META + "run-q"))  # the result consumer's message
    _expire_from_redis(stores, "run-q")
    assert AsyncResult("run-q", app=celery_app).state == "SUCCESS", (
        "a result seen while queued reads PENDING after it leaves Redis")


def test_redis_down_reads_a_finished_result_from_postgres(stores):
    from celery.result import AsyncResult
    import redis

    tid = _store_success(stores)
    celery_app.backend._cache.clear()
    stores.r.down = True
    try:
        state = AsyncResult(tid, app=celery_app).state
    except redis.ConnectionError:
        state = "raised ConnectionError"
    assert state == "SUCCESS", f"with Redis down a durable result read as {state}"


def test_a_queued_task_stays_pending(stores):
    from celery.result import AsyncResult
    assert AsyncResult("never-ran", app=celery_app).state == "PENDING"


# ---------------------------------------------------------------------------
# Readers, after the result left Redis
# ---------------------------------------------------------------------------

class _Schedule:
    run_every = timedelta(days=1)

    def remaining_estimate(self, _last):
        return timedelta(seconds=60)


def _entry(task_type="execute"):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(name="sched-1", max_runs=-1, total_run_count=1, last_run_at=now, due_at=now,
                           schedule=_Schedule(), args=[{"task_schedule": {"task_type": task_type}}])


@pytest.fixture
def expired_run(stores, monkeypatch):
    """A run that finished, whose result is in Postgres only."""
    tid = _store_success(stores)
    _expire_from_redis(stores, tid)
    _scheduled_run(stores, tid)
    from app.core.celery_app import AvalokaEntry, AvalokaScheduler
    monkeypatch.setattr(AvalokaEntry, "from_key", classmethod(lambda cls, key, app=None: _entry()))
    for name, value in (("get_task_ids", [tid]), ("get_last_run_task_id", tid),
                        ("get_last_run_at", datetime.now(timezone.utc)),
                        ("get_task_ids_with_timestamp", [(tid, datetime.now(timezone.utc))])):
        monkeypatch.setattr(AvalokaScheduler, name, staticmethod(lambda *a, _v=value, **k: _v))
    return tid


def _op_state(**extra):
    return {"messages": [], "task_operation": {"task_id": "sched-1", "result_index": 0}, **extra}


def test_scheduler_result_of_an_expired_run_is_not_running(expired_run):
    from app.agents import scheduler

    out = scheduler.handle_task_result(_op_state())
    text = out["messages"][-1].content
    assert "currently running" not in text, "a finished run whose result left Redis is reported running"
    assert "ran successfully" in text and "Revenue rose 4%." in text
    assert out["output_location"] == "/tmp/work/out/result.csv"
    assert out["output_file_data"]["content"].startswith("data:text/csv;base64,")


def test_scheduler_status_of_an_expired_run_is_not_running(expired_run):
    from app.agents import scheduler

    text = scheduler.handle_task_status(_op_state())["messages"][-1].content
    assert "currently running" not in text, "a finished run whose result left Redis is reported running"
    assert "ran successfully" in text and "/tmp/work/out/result.csv" in text


def test_scheduler_list_shows_an_expired_run_as_finished(expired_run):
    from app.agents import scheduler

    text = scheduler.handle_task_list({"messages": [], "task_list": ["sched-1"]})["messages"][-1].content
    assert "Running" not in text and "Success" in text


def test_a_run_whose_result_is_gone_everywhere_is_not_running(stores, expired_run):
    """No Celery result at all (lost or purged), but the run record says it ended."""
    from app.agents import scheduler

    stores.db.tables.setdefault("celery_task_results", {}).clear()
    celery_app.backend._misses.clear() if hasattr(celery_app.backend, "_misses") else None
    text = scheduler.handle_task_result(_op_state())["messages"][-1].content
    assert "currently running" not in text and "no longer available" in text


def test_an_expired_run_is_not_in_flight(expired_run):
    import app.api.server as server
    assert server._scheduled_task_is_in_flight("sched-1") is False, (
        "a finished run whose result left Redis still counts as in flight")


def test_a_lost_run_with_a_finished_record_is_not_in_flight(stores, expired_run):
    import app.api.server as server

    stores.db.tables.setdefault("celery_task_results", {}).clear()
    stores.db.tables["scheduled_runs"][expired_run]["status"] = "failure"
    if hasattr(celery_app.backend, "_misses"):
        celery_app.backend._misses.clear()
    assert server._scheduled_task_is_in_flight("sched-1") is False


def _drain(agen, timeout=8.0):
    async def run():
        return [e async for e in agen]
    try:
        return asyncio.run(asyncio.wait_for(run(), timeout))
    except asyncio.TimeoutError:
        return None


def test_the_status_stream_of_an_expired_run_ends(expired_run):
    import app.api.server as server

    events = _drain(server.task_sse_iter(expired_run, "redbeat:sched-1"))
    assert events is not None, "the status stream spins for ever on a result that left Redis"
    body = "".join(events)
    assert '"status": "SUCCESS"' in body and "Revenue rose 4%." in body
    assert "north" in json.dumps(json.loads(body.split("data: ", 1)[1].split("\n", 1)[0])["result"]["output_json"])


def test_the_status_stream_of_a_lost_run_with_a_finished_record_ends(stores, expired_run):
    import app.api.server as server

    stores.db.tables.setdefault("celery_task_results", {}).clear()
    if hasattr(celery_app.backend, "_misses"):
        celery_app.backend._misses.clear()
    events = _drain(server.task_sse_iter(expired_run, "redbeat:sched-1"))
    assert events is not None, "the status stream spins for ever on a run that ended without a result"
    assert "no longer available" in "".join(events)


# -- the HTTP readers, called directly -------------------------------------------

def _http_request(path="/", session="sess-1"):
    from starlette.requests import Request
    return Request({"type": "http", "method": "GET", "path": path, "query_string": b"",
                    "headers": [(b"x-avaloka-session", session.encode())]})


@pytest.fixture
def http(expired_run, monkeypatch):
    import app.api.server as server
    from app.services import session_service

    sess = {"user_id": "u-1", "tasks": ["sched-1"], "dataset_id": "ds-1", "background_task_id": "sched-1",
            "sample_status": "quick_sample"}

    async def _get_session(sid, bypass_circuit=False):
        return sess if sid == "sess-1" else None

    async def _update_session(sid, fn):
        fn(sess)
        return sess

    async def _save_session(sid, data):
        return True

    monkeypatch.setattr(server, "_resolve_user_id", lambda request: "u-1")
    monkeypatch.setattr(server, "get_session", _get_session)
    monkeypatch.setattr(server, "update_session", _update_session)
    monkeypatch.setattr(server, "save_session", _save_session)
    monkeypatch.setattr(session_service, "cache", object())

    async def _no_assets(**_kw):
        return None
    monkeypatch.setattr(server, "_persist_assets_background", _no_assets)
    return SimpleNamespace(server=server, sess=sess, tid=expired_run)


def test_the_task_result_endpoint_of_an_expired_run_is_not_running(http):
    out = asyncio.run(http.server.get_task_result(_http_request(), "sched-1", 0))
    assert out["status"] == "SUCCESS", f"a finished run whose result left Redis reads {out['status']}"
    assert out["assistant_message"]["content"] == "Revenue rose 4%."
    assert out["result"]["output_json"] == [{"region": "north", "revenue": 10}]


def test_the_task_status_endpoint_of_an_expired_run_does_not_stream_running(http):
    resp = asyncio.run(http.server.get_task_status(_http_request(), "sched-1"))
    events = _drain(resp.body_iterator)
    assert events is not None, "the status endpoint streams a finished run as running, for ever"
    body = "".join(events)
    assert '"status": "PENDING"' not in body and '"status": "SCHEDULED"' in body


def test_the_runs_fallback_shows_an_expired_run_as_finished(http, monkeypatch):
    async def _no_table(_task_id):
        return None
    monkeypatch.setattr(http.server, "_load_scheduled_runs", _no_table)
    runs = asyncio.run(http.server.get_task_runs(_http_request(), "sched-1"))
    assert runs[0]["status"] == "success", f"a finished run whose result left Redis is listed {runs[0]['status']}"
    assert runs[0]["result"]["message"]


def test_background_task_status_of_an_expired_run_is_not_running(http):
    out = asyncio.run(http.server.get_background_task_status(_http_request(), "ds-1"))
    assert out["state"] == "SUCCESS", f"a finished sampling run whose result left Redis reads {out['state']}"
    assert out["available_samples"] == ["random_baseline"]


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def _purge_task():
    task = celery_app.tasks.get(PURGE_TASK)
    assert task is not None, "there is no purge task: results would be kept for ever"
    return task


def _seed(db, task_id, days_ago, status="SUCCESS"):
    db.tables.setdefault("celery_task_results", {})[task_id] = {
        "task_id": task_id, "status": status, "date_done": iso_days_ago(days_ago), "result": {}}


@pytest.fixture
def retention_env(monkeypatch):
    for k in ("CELERY_RESULT_RETENTION_DAYS", "AVALOKA_EDITION", "AVALOKA_LICENSE_KEY",
              "CELERY_RESULT_PURGE_BATCH", "CELERY_RESULT_PURGE_MAX_BATCHES"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AVALOKA_HOME", "/nonexistent-avaloka-home")
    return monkeypatch


def test_a_row_past_retention_is_purged_and_one_inside_is_kept(stores, retention_env):
    _seed(stores.db, "old", 366)
    _seed(stores.db, "recent", 364)
    out = _purge_task().run()
    left = {r["task_id"] for r in _durable_rows(stores)}
    assert left == {"recent"}, f"after the purge: {sorted(left)}"
    assert out["deleted"] == 1 and out["retention_days"] == 365


def test_the_purge_honours_its_batch_limit(stores, retention_env):
    for i in range(25):
        _seed(stores.db, f"old-{i:02d}", 400 + i)
    retention_env.setenv("CELERY_RESULT_PURGE_BATCH", "10")
    retention_env.setenv("CELERY_RESULT_PURGE_MAX_BATCHES", "2")
    out = _purge_task().run()
    calls = [c for c in stores.db.calls if c[0] == "rpc"]
    assert [c[2]["batch_size"] for c in calls] == [10, 10], "the purge did not delete in bounded batches"
    assert out["deleted"] == 20 and len(_durable_rows(stores)) == 5


def test_an_unfinished_task_is_never_purged(stores, retention_env):
    for status in ("PENDING", "STARTED", "RETRY", "RECEIVED"):
        _seed(stores.db, f"open-{status}", 2000, status=status)
    _purge_task().run()
    assert len(_durable_rows(stores)) == 4


def test_three_years_without_enterprise_is_refused(stores, retention_env):
    retention_env.setenv("CELERY_RESULT_RETENTION_DAYS", "1095")
    retention_env.setenv("AVALOKA_EDITION", "professional")
    _seed(stores.db, "old", 500)
    task = _purge_task()
    with pytest.raises(Exception) as err:
        task.run()
    assert type(err.value).__name__ == "ResultStoreConfigError" and "Enterprise" in str(err.value)
    assert not [c for c in stores.db.calls if c[0] == "rpc"], "a refused setting must purge nothing"


def test_enterprise_edition_without_the_commercial_build_is_refused(stores, retention_env):
    retention_env.setenv("CELERY_RESULT_RETENTION_DAYS", "1095")
    retention_env.setenv("AVALOKA_EDITION", "enterprise")
    from app.core import editions
    retention_env.setattr(editions, "commercial_build", lambda: False)
    task = _purge_task()
    with pytest.raises(Exception, match="Enterprise"):
        task.run()


def test_enterprise_may_keep_results_three_years(stores, retention_env):
    retention_env.setenv("CELERY_RESULT_RETENTION_DAYS", "1095")
    retention_env.setenv("AVALOKA_EDITION", "enterprise")
    from app.core import editions
    retention_env.setattr(editions, "commercial_build", lambda: True)
    _seed(stores.db, "two-years", 730)
    _seed(stores.db, "four-years", 1460)
    out = _purge_task().run()
    assert out["retention_days"] == 1095
    assert {r["task_id"] for r in _durable_rows(stores)} == {"two-years"}


@pytest.mark.parametrize("value", ["1096", "0", "-1", "a year"])
def test_out_of_range_retention_is_refused_even_on_enterprise(stores, retention_env, value):
    retention_env.setenv("CELERY_RESULT_RETENTION_DAYS", value)
    retention_env.setenv("AVALOKA_EDITION", "enterprise")
    from app.core import editions
    retention_env.setattr(editions, "commercial_build", lambda: True)
    task = _purge_task()
    with pytest.raises(Exception) as err:
        task.run()
    assert type(err.value).__name__ == "ResultStoreConfigError"
    assert not [c for c in stores.db.calls if c[0] == "rpc"]


def test_the_purge_runs_daily_and_stores_no_result_of_its_own():
    task = _purge_task()
    assert task.ignore_result is True, "the purge would feed the table it purges"
    entry = (celery_app.conf.beat_schedule or {}).get("purge-celery-task-results")
    assert entry and entry["task"] == PURGE_TASK
    assert "purge-celery-task-results" in celery_module._UNTRACKED_BEAT_ENTRIES
    assert celery_app.tasks["app.core.celery_app.reap_idle_user_ray_clusters"].ignore_result is True


# ---------------------------------------------------------------------------
# The migration ships correctly (the SQL itself: tests/test_celery_task_results_migration.py)
# ---------------------------------------------------------------------------

MIGRATION = "20261117120000_celery_task_results.sql"


def test_the_migration_ships_byte_identical_in_both_directories():
    a = REPO / "ui" / "supabase" / "migrations" / MIGRATION
    b = REPO / "deploy" / "helm" / "avaloka" / "files" / "supabase-migrations" / MIGRATION
    assert a.is_file() and b.is_file(), "the celery_task_results migration is missing"
    assert a.read_bytes() == b.read_bytes()


def test_the_migration_is_in_ci_postgres_line_with_per_test_results():
    ci = (REPO / "bitbucket-pipelines.yml").read_text()
    line = next((ln for ln in ci.splitlines() if "-m integration tests/test_lineage.py" in ln), "")
    assert "tests/test_celery_task_results_migration.py" in line, (
        "the migration's real-Postgres tests are not on CI's Postgres line, so they never run")
    assert " -rA " in line, "the Postgres line does not name each test's result"


# -- /datasets/{id}/preview: the background-sampling upgrade ----------------------

from tests.server_harness import SESSIONS, _do_upload, client, make_auth_headers  # noqa: E402,F401


def test_preview_upgrades_to_the_full_sample_when_the_result_left_redis(client, expired_run):
    out = _do_upload(client, user_id="u-prev")
    sess = SESSIONS[out["session_id"]]
    sess.update(sample_status="quick_sample", background_task_id="sched-1")
    resp = client.get(f"/datasets/{out['dataset_id']}/preview", headers=make_auth_headers("u-prev"))
    assert resp.status_code == 200, resp.text[:400]
    assert resp.json().get("sample_status") == "full_sample", (
        "a finished sampling run whose result left Redis leaves the preview on the quick sample")
