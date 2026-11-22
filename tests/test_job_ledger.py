"""app/core/job_ledger.py: one row per job, one notification per terminal
transition, redacted labels, closed vocabularies, sync-safe emission.

Hermetic: the ledger and notification_service run on tests/job_fakes.FakeDB.

    pytest tests/test_job_ledger.py -q
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.core import job_ledger as jl
from app.core import job_notes as jn
from app.services import notification_service as ns
from tests.job_fakes import FakeDB

USER = FakeDB.AUTH_UID
THREAD = FakeDB.THREAD_ID


@pytest.fixture
def db(monkeypatch):
    fake = FakeDB()
    monkeypatch.setattr(ns, "_client", lambda: fake)
    ns._profile_cache.clear()
    jl.set_store(jl.SupabaseJobs(client=fake))
    jn.set_store(jn.SupabaseNotes(client=fake))
    yield fake
    jl.set_store(None)
    jn.set_store(None)
    ns._profile_cache.clear()


def _start(db, kind="transfer", **kw):
    kw.setdefault("thread_id", THREAD)
    return jl.start(USER, kind, label="orders ➡️ warehouse", **kw)


# ---- vocabularies ----------------------------------------------------------

def test_every_kind_has_a_family_a_title_and_user_text():
    assert set(jl.FAMILY) == jl.KINDS
    assert set(jl.KIND_TEXT) == jl.KINDS
    assert set(jl.TITLES) == jl.KINDS - {"reap"}          # reap is housekeeping: silent
    for kind, fam in jl.FAMILY.items():
        if fam is not None:
            assert f"{fam}.completed" in ns.TYPES and f"{fam}.failed" in ns.TYPES, kind
    assert "job.cancelled" in ns.TYPES


def test_train_infer_vocabulary_is_kept_verbatim():
    assert {"training", "register", "deploy_endpoint", "stop_endpoint", "cross_validate", "reap"} <= jl.KINDS
    assert jl.TITLES["training"] == {"succeeded": "Model training completed", "failed": "Model training failed"}
    assert jl.TITLES["deploy_endpoint"]["succeeded"] == "Inference endpoint ready"
    assert jl.TITLES["stop_endpoint"]["succeeded"] == "Inference endpoint stopped"


def test_migration_check_constraints_match_the_python_vocabularies():
    import pathlib, re
    sql = (pathlib.Path(__file__).resolve().parents[1] / "ui" / "supabase" / "migrations"
           / "20261121110000_background_jobs.sql").read_text()

    def tokens(name):
        m = re.search(name + r" check \((?:\w+ is null or )?\w+ in \(([^)]*)\)", sql, re.S)
        assert m, name
        return set(re.findall(r"'([a-z_]+)'", m.group(1)))

    assert tokens("ml_jobs_kind_check") == jl.KINDS
    assert tokens("ml_jobs_fail_reason_check") == jl.FAIL_REASONS
    assert tokens("ml_jobs_resource_type_check") == jl.RESOURCE_TYPES == ns.RESOURCE_TYPES
    assert tokens("ml_jobs_origin_check") == jl.ORIGINS


@pytest.mark.parametrize("bad", [dict(kind="nope"), dict(resource_type="file"), dict(origin="cron")])
def test_start_rejects_values_outside_the_vocabulary(db, bad):
    kw = {"kind": "transfer", **bad}
    kind = kw.pop("kind")
    with pytest.raises(ValueError):
        jl.start(USER, kind, **kw)
    assert db.jobs == []


def test_finish_rejects_a_non_terminal_status_and_unknown_reasons(db):
    jid = _start(db)
    with pytest.raises(ValueError):
        jl.finish(jid, "running")
    with pytest.raises(ValueError):
        jl.finish(jid, "failed", fail_reason="disk_full")
    assert db.jobs[0]["status"] == "queued"


# ---- transitions --------------------------------------------------------------

def test_start_then_running_then_finish_writes_one_row(db):
    jid = _start(db)
    row = db.jobs[0]
    assert (row["status"], row["kind"], row["thread_id"], row["origin"]) == ("queued", "transfer", THREAD, "api")
    jl.running(jid, ray_job_name="rayjob-1")
    assert db.jobs[0]["status"] == "running" and db.jobs[0]["started_at"] and db.jobs[0]["ray_job_name"] == "rayjob-1"
    res = jl.finish(jid, "succeeded")
    assert res.transitioned and res.claimed and len(db.jobs) == 1
    assert db.jobs[0]["status"] == "succeeded" and db.jobs[0]["finished_at"] and db.jobs[0]["notified_at"]


def test_running_never_reopens_a_terminal_row(db):
    jid = _start(db)
    jl.finish(jid, "failed", fail_reason="timed_out")
    jl.running(jid)
    assert db.jobs[0]["status"] == "failed"


def test_finish_twice_is_one_transition_one_notification_one_note(db):
    jid = _start(db)
    first = jl.finish(jid, "succeeded")
    second = jl.finish(jid, "failed", fail_reason="failed")   # a retry that disagrees still loses
    assert first.transitioned and first.claimed and first.notification_id and first.note_queued
    assert not second.transitioned and not second.claimed and second.notification_id is None and not second.note_queued
    assert db.jobs[0]["status"] == "succeeded"
    assert len(db.notifications) == 1 and len(db.notes) == 1


def test_concurrent_finishes_from_threads_claim_once(db):
    jid = _start(db)
    results = []
    barrier = threading.Barrier(8)

    def go():
        barrier.wait()
        results.append(jl.finish(jid, "succeeded"))

    with ThreadPoolExecutor(8) as pool:
        for _ in range(8):
            pool.submit(go)
    assert sum(1 for r in results if r.claimed) == 1
    assert len(db.notifications) == 1 and len(db.notes) == 1


def test_claim_is_one_update_filtered_on_notified_at_null(db):
    jid = _start(db)
    db.calls.clear()
    assert jl.SupabaseJobs(client=db).claim_notification(jid) is True
    assert jl.SupabaseJobs(client=db).claim_notification(jid) is False
    assert db.calls == [("ml_jobs", "update"), ("ml_jobs", "update")]


def test_a_second_net_the_notification_ref_is_the_job_id(db):
    jid = _start(db)
    jl.finish(jid, "succeeded")
    assert db.notifications[0]["metadata"]["ref"] == f"job:{jid}"


# ---- the bell -------------------------------------------------------------------

@pytest.mark.parametrize("kind,status,reason,ntype,title", [
    ("transfer", "succeeded", None, "transfer.completed", "Data transfer completed"),
    ("transfer", "failed", "timed_out", "transfer.failed", "Data transfer failed"),
    ("training", "succeeded", None, "training.completed", "Model training completed"),
    ("deploy_endpoint", "succeeded", None, "inference.completed", "Inference endpoint ready"),
    ("stop_endpoint", "succeeded", None, "inference.completed", "Inference endpoint stopped"),
    ("inference_batch", "failed", "cluster_error", "inference.failed", "Batch prediction failed"),
    ("sampling", "succeeded", None, "sampling.completed", "Full-data sample ready"),
    ("scheduled_run", "failed", "refused", "scheduled_run.failed", "Scheduled analysis failed"),
    ("swarm", "succeeded", None, "swarm.completed", "Swarm mission completed"),
    ("ray_job", "cancelled", None, "job.cancelled", "Analysis on the cluster cancelled"),
])
def test_type_and_title_per_kind_and_status(db, kind, status, reason, ntype, title):
    jid = _start(db, kind=kind)
    jl.finish(jid, status, fail_reason=reason)
    n = db.notifications[0]
    assert (n["type"], n["title"]) == (ntype, title)
    assert n["metadata"]["job_id"] == jid and n["metadata"]["kind"] == kind


def test_inference_family_carries_the_sub_kind_in_metadata(db):
    for kind, sub in (("deploy_endpoint", "deploy"), ("stop_endpoint", "stop"), ("inference_batch", "batch")):
        jl.finish(_start(db, kind=kind), "succeeded")
    assert [n["metadata"]["sub_kind"] for n in db.notifications] == ["deploy", "stop", "batch"]
    assert all(n["type"] == "inference.completed" for n in db.notifications)


def test_reap_is_silent_but_recorded(db):
    jid = _start(db, kind="reap", thread_id=None)
    res = jl.finish(jid, "succeeded")
    assert res.transitioned and res.claimed and db.jobs[0]["status"] == "succeeded"
    assert db.notifications == [] and db.notes == []


def test_failure_body_is_the_closed_reason_never_the_error_line(db):
    jid = _start(db)
    secret = "psycopg2.OperationalError: FATAL password authentication failed for user etl at 10.0.0.9:5432"
    jl.finish(jid, "failed", fail_reason="cluster_error", error_line=secret)
    n = db.notifications[0]
    assert "It failed on the cluster." in n["body"]
    for leak in ("psycopg2", "10.0.0.9", "etl", "FATAL"):
        assert leak not in n["body"] and leak not in n["title"]
    assert db.jobs[0]["error"].startswith("psycopg2.OperationalError")      # the row keeps the one-liner


def test_success_body_has_the_label_and_a_duration(db):
    jid = _start(db)
    db.jobs[0]["started_at"] = "2026-11-15T10:00:00+00:00"
    jl.finish(jid, "succeeded")
    body = db.notifications[0]["body"]
    assert body.startswith("orders ➡️ warehouse") and "finished in" in body


def test_resource_is_passed_through_to_the_notification(db):
    jid = _start(db, resource_type="transfer", resource_id="t-1")
    jl.finish(jid, "succeeded")
    n = db.notifications[0]
    assert n["resource_type"] == "transfer" and n["metadata"]["resource_id"] == "t-1"


def test_a_job_without_a_thread_rings_the_bell_but_queues_no_note(db):
    jid = _start(db, thread_id=None)
    res = jl.finish(jid, "succeeded")
    assert res.notification_id and not res.note_queued and db.notes == []


def test_notify_false_takes_the_claim_without_ringing(db):
    jid = _start(db)
    res = jl.finish(jid, "succeeded", notify=False)
    assert res.claimed and res.notification_id is None and db.notifications == []
    assert res.note_queued                       # the thread still learns of it
    assert not jl.finish(jid, "succeeded").claimed


# ---- redaction of the one free-text field -------------------------------------

@pytest.mark.parametrize("label,leak", [
    ("register warehouse host db.example:5432 user etl password FAKE-PW-7781", "FAKE-PW-7781"),
    # Assembled at run time: no tracked file carries scheme://user:password@ (E11.05, C9.01).
    ("copy postgres://etl:" + "FAKE-PW-22" + "@db.example/orders to lake", "FAKE-PW-22"),
    ("pwd: FAKE-PW-3 then run", "FAKE-PW-3"),
    ("load /Users/jane/Downloads/patients_2026.csv", "patients_2026"),
    ("from gs://acme-prod-bucket/finance/q3.parquet", "acme-prod-bucket"),
    ("token=FAKE-TOKEN-9 for the API", "FAKE-TOKEN-9"),
])
def test_labels_are_scrubbed_before_they_are_stored_or_shown(db, label, leak):
    jid = jl.start(USER, "transfer", thread_id=THREAD, label=label)
    assert leak not in (db.jobs[0]["label"] or "")
    jl.finish(jid, "succeeded")
    assert leak not in (db.notifications[0]["body"] or "")
    assert leak not in db.notes[0]["text"]


def test_label_is_clipped_and_whitespace_collapsed():
    out = jl.scrub_label("a   b\n\n" + "x" * 200)
    assert len(out) <= jl.LABEL_MAX and out.startswith("a b x") and out.endswith("…")
    assert jl.scrub_label("   ") is None and jl.scrub_label(None) is None


def test_ordinary_label_text_is_unchanged():
    assert jl.scrub_label("orders ➡️ warehouse") == "orders ➡️ warehouse"
    assert jl.scrub_label("rows where password is null") == "rows where password is [redacted]"  # over-strips: by design


# ---- never fails the producer -------------------------------------------------

def test_a_down_ledger_returns_none_and_raises_nothing(db, caplog):
    db.fail_tables.add("ml_jobs")
    with caplog.at_level(logging.WARNING):
        assert jl.start(USER, "transfer", thread_id=THREAD) is None
        res = jl.finish("missing-id", "succeeded")
    assert res.job is None and res.errors and db.notifications == []


def test_a_down_bell_still_records_the_transition_and_the_note(db):
    jid = _start(db)
    db.fail_tables.add("notifications")
    res = jl.finish(jid, "succeeded")
    assert res.transitioned and res.claimed and res.notification_id is None and res.note_queued
    assert db.jobs[0]["status"] == "succeeded"


def test_missing_ledger_columns_degrade_to_bell_only(db, caplog):
    """Migration not applied yet: writes drop the new columns once, the
    condition is remembered, the job still finishes and the bell still rings."""
    db.missing_columns_tables.add("ml_jobs")
    with caplog.at_level(logging.WARNING):
        jid = jl.start(USER, "transfer", thread_id=THREAD, label="x")
    assert jid and "thread_id" not in db.jobs[0] and "label" not in db.jobs[0]
    assert jl.get_store().columns_missing is True
    res = jl.finish(jid, "succeeded")
    assert res.transitioned and res.claimed and res.notification_id
    assert "fail_reason" not in db.jobs[0] and "notified_at" not in db.jobs[0]
    assert any("degrading to bell-only" in r.getMessage() for r in caplog.records)


# ---- sync-safe emission (the loop rule) -------------------------------------------

def test_finish_on_a_live_event_loop_rings_from_a_worker_thread_and_warns(db, caplog):
    """The API must use asyncio.to_thread; if a caller forgets, the job is not
    failed and the loop is not driven by a nested asyncio.run."""
    jid = _start(db)
    seen_loops = []
    real_run = asyncio.run

    def spy_run(coro):
        seen_loops.append(threading.get_ident())
        return real_run(coro)

    async def on_loop():
        with caplog.at_level(logging.WARNING, logger="avaloka.jobs"):
            import unittest.mock as um
            with um.patch.object(jl.asyncio, "run", spy_run):
                return jl.finish(jid, "succeeded")

    res = asyncio.run(on_loop())
    assert res.transitioned and res.claimed and res.notification_id
    assert seen_loops and seen_loops[0] != threading.get_ident()          # the private run happened off this thread
    assert any("running event loop" in r.getMessage() for r in caplog.records)


def test_finish_from_to_thread_rings_the_bell(db):
    jid = _start(db)

    async def main():
        return await asyncio.to_thread(jl.finish, jid, "succeeded")

    res = asyncio.run(main())
    assert res.notification_id and len(db.notifications) == 1


def test_private_loop_releases_its_redis_client_when_the_fix_is_present(db, monkeypatch):
    from app.services import session_service
    released = []

    class _Cache:
        async def release_loop_client(self):
            released.append(asyncio.get_running_loop())

    monkeypatch.setattr(session_service, "cache", _Cache(), raising=False)
    jl.finish(_start(db), "succeeded")
    assert len(released) == 1 and released[0].is_closed()         # released on the private loop, which then closed


def test_private_loop_without_the_fix_does_not_break(db, monkeypatch):
    from app.services import session_service
    monkeypatch.setattr(session_service, "cache", object(), raising=False)
    assert jl.finish(_start(db), "succeeded").notification_id


# ---- recurring schedules (ruling 3) ---------------------------------------------

@pytest.mark.parametrize("one_off,succeeded,prev,expect", [
    (True, True, True, True), (True, False, None, True),          # one-off: always
    (False, False, True, True), (False, False, False, True),      # recurring failure: always
    (False, True, None, True),                                    # first success
    (False, True, False, True),                                   # recovery
    (False, True, True, False),                                   # steady state: quiet
])
def test_recurring_schedule_notify_policy(one_off, succeeded, prev, expect):
    assert jl.scheduled_run_should_notify(one_off=one_off, succeeded=succeeded, previous_succeeded=prev) is expect


# ---- metadata-only telemetry --------------------------------------------------------

def test_log_lines_carry_ids_and_vocabulary_only(db, caplog):
    with caplog.at_level(logging.INFO, logger="avaloka.jobs"):
        jid = jl.start(USER, "transfer", thread_id=THREAD, label="orders ➡️ warehouse FAKE-PW-5")
        jl.finish(jid, "failed", fail_reason="timed_out", error_line="Traceback: boom at /srv/x.py")
    text = "\n".join(r.getMessage() for r in caplog.records)
    for leak in ("orders", "warehouse", "FAKE-PW-5", "Traceback", "/srv/x.py", USER):
        assert leak not in text
    assert "timed_out" in text and "transfer" in text


def test_ledger_modules_have_no_egress_imports():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "app" / "core"
    for name in ("job_ledger.py", "job_notes.py"):
        src = (root / name).read_text()
        for forbidden in ("import httpx", "import requests", "app.analytics", "urllib.request", "import socket"):
            assert forbidden not in src, (name, forbidden)
