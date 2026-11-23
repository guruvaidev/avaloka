"""celery_app.py changes from 29-30 Oct.

* Beat must use the RedBeat scheduler (default PersistentScheduler crashed with
  "'PersistentScheduler' object has no attribute 'lock_key'").
* ``_notify_job_finished``: a bell notification when a one-time job ends, and
  never for recurring schedules (an every-minute job would flood the bell).
"""
from __future__ import annotations

import pytest

ca = pytest.importorskip("app.core.celery_app")
notification_service = pytest.importorskip("app.services.notification_service")

RUN_NOW = {"task_type": "execute", "schedule_type": "relative", "second": 0, "max_runs": 1}
EVERY_MINUTE = {"task_type": "execute", "schedule_type": "crontab", "minute": "*", "max_runs": -1}


def _state(**overrides) -> dict:
    state = {
        "user_id": "user-1",
        "thread_id": "thread-1",
        "dataset_id": "ds-1",
        "schedule_id": "sched-1",
        "task_schedule": dict(RUN_NOW),
        "messages": [
            {
                "role": "human",
                "content": "Rank brands by number of views\n\n[Analysis context] Current fidelity: entire_dataset",
            }
        ],
    }
    state.update(overrides)
    return state


@pytest.fixture
def notifications(monkeypatch):
    """The ledger-backed notifier: a FakeDB behind job_ledger, job_notes and
    notification_service. Returns the notifications table."""
    from app.core import job_ledger as jl
    from app.core import job_notes as jn
    from tests.job_fakes import FakeDB
    fake = FakeDB()
    fake.tables["profiles"].append({"id": "profile-user-1", "user_id": "user-1"})
    monkeypatch.setattr(notification_service, "_client", lambda: fake)
    notification_service._profile_cache.clear()
    jl.set_store(jl.SupabaseJobs(client=fake))
    jn.set_store(jn.SupabaseNotes(client=fake))
    fake.notifications_db = fake
    yield fake.notifications
    jl.set_store(None)
    jn.set_store(None)
    notification_service._profile_cache.clear()


# ----------------------------------------------------------- config ---

def test_beat_uses_redbeat_scheduler():
    conf = ca.celery_app.conf
    assert conf.beat_scheduler == "app.core.celery_app:AvalokaScheduler"
    assert conf.redbeat_redis_url == ca.REDIS_URL
    assert conf.task_track_started is True


# ------------------------------------------------- _notify_job_finished ---
# The notifier is the job ledger's terminal transition (app/core/job_ledger.py):
# one row per run, one bell per terminal transition, the raw error on the row
# and never in the bell.

def _jobs():
    from app.core import job_ledger as jl
    return jl.get_store()._client.jobs


def test_successful_run_now_job_notifies_completed(notifications):
    final = {"execution_result": {"status": "success"}}
    ca._notify_job_finished(_state(), final, execution_id="11111111-1111-4111-8111-111111111111")

    assert len(notifications) == 1
    n = notifications[0]
    assert n["type"] == "analysis.completed" and n["title"] == "Analysis completed"
    assert n["metadata"]["thread_id"] == "thread-1"
    assert n["metadata"]["ref"] == "job:11111111-1111-4111-8111-111111111111"
    assert "Rank brands by number of views" in n["body"]
    assert "[Analysis context]" not in n["body"]
    row = _jobs()[0]
    assert (row["kind"], row["status"], row["origin"], row["thread_id"]) == ("analysis", "succeeded", "celery", "thread-1")


def test_failed_ray_job_notifies_failed_with_a_closed_reason_not_the_error(notifications):
    final = {"execution_result": {"status": "error", "error": "Ray cluster unreachable at http://127.0.0.1:28265"}}
    ca._notify_job_finished(_state(), final, execution_id="22222222-2222-4222-8222-222222222222")

    assert len(notifications) == 1
    n = notifications[0]
    assert n["type"] == "analysis.failed" and n["title"] == "Analysis failed"
    assert "127.0.0.1" not in n["body"] and "Ray cluster" not in n["body"]       # the raw error stays off the bell
    assert "It failed." in n["body"]
    assert "Ray cluster unreachable" in _jobs()[0]["error"]                      # and on the row


def test_refused_result_is_not_answered(notifications):
    final = {"execution_result": {"status": "refused", "error": "output contract: no answer"}}
    ca._notify_job_finished(_state(), final, execution_id="33333333-3333-4333-8333-333333333333")
    n = notifications[0]
    assert n["type"] == "analysis.failed" and n["metadata"]["fail_reason"] == "not_answered"
    assert "did not answer the question" in n["body"]


def test_exception_path_notifies_failed(notifications):
    ca._notify_job_finished(_state(), None, error="worker crashed", execution_id="44444444-4444-4444-8444-444444444444")
    assert notifications[0]["type"] == "analysis.failed"
    assert "worker crashed" not in notifications[0]["body"] and _jobs()[0]["error"] == "worker crashed"


def test_same_run_finishing_twice_rings_once(notifications):
    final = {"execution_result": {"status": "success"}}
    ca._notify_job_finished(_state(), final, execution_id="55555555-5555-4555-8555-555555555555")
    ca._notify_job_finished(_state(), final, execution_id="55555555-5555-4555-8555-555555555555")
    assert len(notifications) == 1 and len(_jobs()) == 1


def test_recurring_schedule_notifies_first_success_failures_and_recoveries_only(notifications, monkeypatch):
    history = []
    monkeypatch.setattr(ca, "_previous_run_succeeded", lambda schedule_id, run_id: history[-1] if history else None)

    def run(i, ok):
        final = {"execution_result": {"status": "success" if ok else "error", "error": None if ok else "x"}}
        ca._notify_job_finished(_state(task_schedule=dict(EVERY_MINUTE)), final,
                                execution_id=f"66666666-6666-4666-8666-{i:012d}")
        history.append(ok)

    run(1, True); assert len(notifications) == 1                      # first success
    run(2, True); assert len(notifications) == 1                      # steady state: quiet
    run(3, True); assert len(notifications) == 1
    run(4, False); assert len(notifications) == 2                     # every failure
    run(5, False); assert len(notifications) == 3
    run(6, True); assert len(notifications) == 4                      # recovery
    run(7, True); assert len(notifications) == 4
    assert notifications[0]["type"] == "scheduled_run.completed" and notifications[1]["type"] == "scheduled_run.failed"
    assert all(r["kind"] == "scheduled_run" and r["status"] in ("succeeded", "failed") for r in _jobs())
    assert all(r.get("thread_id") is None for r in _jobs())           # recurring runs never write into the chat


def test_previous_run_lookup_reads_scheduled_runs(notifications):
    from app.core import job_ledger as jl
    fake = jl.get_store()._client
    fake.tables["scheduled_runs"] = [
        {"run_id": "r1", "schedule_id": "sched-1", "status": "failure", "started_at": "2026-11-15T10:00:00+00:00"},
        {"run_id": "r2", "schedule_id": "sched-1", "status": "success", "started_at": "2026-11-15T10:05:00+00:00"},
        {"run_id": "r3", "schedule_id": "sched-1", "status": "running", "started_at": "2026-11-15T10:10:00+00:00"},
    ]
    import app.core.celery_app as mod
    orig = mod._supabase
    mod._supabase = lambda: fake
    try:
        assert ca._previous_run_succeeded("sched-1", "r3") is True       # r2, the newest other run
        assert ca._previous_run_succeeded("sched-1", "r2") is False      # r3 is running: not a success
        assert ca._previous_run_succeeded("sched-9", "r1") is None
        assert ca._previous_run_succeeded(None, "r1") is None
    finally:
        mod._supabase = orig


def test_training_task_notifies_with_the_training_title(notifications):
    ca._notify_job_finished(_state(task_schedule={**RUN_NOW, "task_type": "training"}), {},
                            execution_id="77777777-7777-4777-8777-777777777777")
    n = notifications[0]
    assert (n["type"], n["title"]) == ("training.failed", "Model training failed")   # an empty final is a failed run


def test_inference_tasks_use_one_family_with_the_sub_kind(notifications):
    ca._notify_job_finished(_state(task_schedule={**RUN_NOW, "task_type": "start_inference"}),
                            {"execution_result": {"status": "success"}}, execution_id="88888888-8888-4888-8888-888888888888")
    n = notifications[0]
    assert (n["type"], n["title"], n["metadata"]["sub_kind"]) == ("inference.completed", "Inference endpoint ready", "deploy")


def test_missing_user_does_not_notify(notifications):
    ca._notify_job_finished(_state(user_id=None), {"execution_result": {"status": "success"}}, execution_id="x")
    assert notifications == []


def test_long_question_is_shortened(notifications):
    long_q = "Rank brands " * 40
    ca._notify_job_finished(
        _state(messages=[{"role": "human", "content": long_q}]),
        {"execution_result": {"status": "success"}}, execution_id="99999999-9999-4999-8999-999999999999",
    )
    assert len(notifications[0]["body"]) < 120


def test_the_turns_row_is_reused_when_the_chat_handed_the_work_over(notifications):
    from app.core import job_ledger as jl
    turn_row = jl.start("user-1", "analysis", job_id="abababab-abab-4bab-8bab-abababababab", thread_id="thread-1",
                        status="running", origin="api", label="rank brands")
    state = _state(task_schedule=dict(RUN_NOW), ledger_job_id=turn_row)
    exec_id = "cdcdcdcd-cdcd-4dcd-8dcd-cdcdcdcdcdcd"
    assert ca._record_ledger_start(state, exec_id) == turn_row           # attached, not duplicated
    ca._notify_job_finished(state, {"execution_result": {"status": "success"}}, execution_id=exec_id)
    assert len(_jobs()) == 1 and _jobs()[0]["celery_task_id"] == exec_id and _jobs()[0]["status"] == "succeeded"
    assert len(notifications) == 1


def test_a_killed_task_ends_its_row_as_timed_out(notifications):
    state = _state()
    ca._finish_killed_job("dededede-dede-4ede-8ede-dededededede", state, "timed_out", "the 3600s hard time limit killed the task")
    assert _jobs()[0]["status"] == "failed" and _jobs()[0]["fail_reason"] == "timed_out"
    assert notifications[0]["type"] == "analysis.failed" and "timed out" in notifications[0]["body"]


def test_a_down_ledger_does_not_silence_the_bell(monkeypatch):
    """Supabase unreachable for the ledger: the bell still rings, with the
    ledger's text and ref, through notification_service alone."""
    from app.core import job_ledger as jl
    sent = []

    async def fake_create(user_id, n_type, title, body, **kw):
        sent.append({"user_id": user_id, "type": n_type, "title": title, "body": body, **kw})

    monkeypatch.setattr(notification_service, "create_notification", fake_create)

    class Down:
        def _c(self):
            raise RuntimeError("no supabase")
        get = lambda self, job_id: (_ for _ in ()).throw(RuntimeError("no supabase"))  # noqa: E731
        upsert = update = lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("no supabase"))  # noqa: E731
        claim_notification = lambda self, job_id: (_ for _ in ()).throw(RuntimeError("no supabase"))  # noqa: E731

    monkeypatch.setattr(ca, "_run_status_from_final", lambda final, task_type=None: ("success", None))
    jl.set_store(Down())
    try:
        ca._notify_job_finished(_state(task_schedule={**RUN_NOW, "task_type": "training"}),
                                {"training_result": {"mlflow_run_id": "r"}}, execution_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    finally:
        jl.set_store(None)
    assert len(sent) == 1
    assert (sent[0]["type"], sent[0]["title"]) == ("training.completed", "Model training completed")
    assert sent[0]["ref"] == "job:aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa" and sent[0]["thread_id"] == "thread-1"


def test_notification_failure_never_breaks_the_task(monkeypatch, notifications):
    async def broken(*args, **kwargs):
        raise RuntimeError("supabase down")

    monkeypatch.setattr(notification_service, "create_notification", broken)
    ca._notify_job_finished(_state(), {"execution_result": {"status": "success"}}, execution_id="x")  # must not raise


# ------------------------------------------------- agent_task wiring ---

def test_agent_task_notifies_after_success(monkeypatch):
    seen = []
    final = {"execution_result": {"status": "success"}}
    monkeypatch.setattr(ca, "_run_agent_task", lambda state: final)
    monkeypatch.setattr(ca, "_notify_job_finished", lambda state, fin, error=None, **kw: seen.append((fin, error)))

    result = ca.agent_task(_state(schedule_id=None))

    assert result is final
    assert seen == [(final, None)]


def test_agent_task_notifies_and_reraises_on_crash(monkeypatch):
    seen = []

    def boom(state):
        raise RuntimeError("executor exploded")

    monkeypatch.setattr(ca, "_run_agent_task", boom)
    monkeypatch.setattr(ca, "_notify_job_finished", lambda state, fin, error=None, **kw: seen.append((fin, error)))

    with pytest.raises(RuntimeError, match="executor exploded"):
        ca.agent_task(_state(schedule_id=None))

    assert seen == [(None, "executor exploded")]