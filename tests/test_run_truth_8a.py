"""Item 8a, the run-truth contract (docs/design/1.6-CAPABILITY-GAPS.md, item 8).

Every run ends in exactly one of succeeded / failed / partial, with a reason
and a scope written by the harness, never by the model. Each test here fails
on develop-1.6 at 49e9ab4d on an assertion:

- a scheduled run where nothing executed was recorded "success"
  (app/core/celery_app.py:337-361);
- task_soft_time_limit=1860 > task_time_limit=1800 (:33-35), so the soft
  limit never fired, and a hard kill left the row "running";
- a deferred turn whose analysis failed was notified ok=True
  (app/api/server.py:6982).

This file imports nothing from app.core.run_truth, so on the base it fails on
assertions rather than on an ImportError.
"""
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import app.api.server as server
from app.core import celery_app as celery_module

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
        SESSIONS, THREAD_TO_SESSION, make_auth_headers, client, shared_loop_client, _do_upload,
    )
except ImportError:
    from server_harness import (  # noqa: F401
        SESSIONS, THREAD_TO_SESSION, make_auth_headers, client, shared_loop_client, _do_upload,
    )

OUTCOMES = {"succeeded", "failed", "partial"}
POLL_TIMEOUT_S = 10.0


# ---------------------------------------------------------------------------
# Supabase stub: records every update and the filters it was sent with.
# ---------------------------------------------------------------------------

class _Query:
    def __init__(self, log, rows):
        self.log, self.rows, self.payload, self.filters = log, rows, None, {}

    def update(self, payload):
        self.payload = dict(payload)
        return self

    def upsert(self, payload, **_kw):
        self.rows[payload["run_id"]] = dict(payload)
        return self

    def select(self, *_a, **_kw):
        return self

    def eq(self, key, value):
        self.filters[key] = value
        return self

    def execute(self):
        if self.payload is None and self.filters.get("run_id") in self.rows:
            return type("R", (), {"count": 0, "data": [self.rows[self.filters["run_id"]]]})()
        if self.payload is not None:
            self.log.append((self.payload, dict(self.filters)))
            row = self.rows.get(self.filters.get("run_id"))
            if row is not None and all(row.get(k) == v for k, v in self.filters.items()):
                row.update(self.payload)
        return type("R", (), {"count": 0, "data": []})()


class _Supabase:
    def __init__(self):
        self.log, self.rows = [], {}

    def table(self, _name):
        return _Query(self.log, self.rows)


@pytest.fixture()
def sb(monkeypatch):
    fake = _Supabase()
    monkeypatch.setattr(celery_module, "_supabase", lambda: fake)
    return fake


def _truth(payload):
    result = payload.get("result") or {}
    truth = result.get("run_truth")
    assert isinstance(truth, dict), f"no run_truth in scheduled_runs.result: {sorted(result)}"
    return truth


def _assert_contract(truth):
    assert truth.get("outcome") in OUTCOMES
    assert truth.get("reason_code")
    assert truth.get("reason")
    scope = truth.get("scope")
    assert isinstance(scope, dict)
    assert {"rows_processed", "rows_total", "rows_total_kind", "sample", "engine", "fallback"} <= set(scope)
    assert truth.get("written_by") == "harness"


# ---------------------------------------------------------------------------
# 1. Nothing ran is never "success"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("final", [
    {"execution_result": {}},
    {"execution_result": {"status": ""}},
    {},
    None,
])
def test_a_scheduled_run_where_nothing_ran_is_not_success(final):
    status, error = celery_module._run_status_from_final(final, task_type="execute")
    assert status == "failure"
    assert error and "Nothing was executed" in error


def test_nothing_ran_is_recorded_failed_with_a_reason(sb):
    celery_module._record_run_finish(
        "sched-1", "run-1", {"execution_result": {}, "task_run_statistics": {}},
        time.perf_counter(), scheduled_state={"task_schedule": {"task_type": "execute"}},
    )
    payload, _ = sb.log[-1]
    assert payload["status"] == "failure"
    truth = _truth(payload)
    assert truth.get("outcome") == "failed"
    assert truth.get("reason_code") == "nothing_ran"
    _assert_contract(truth)


def test_non_execute_tasks_without_an_execution_result_still_succeed(sb):
    """Existing behaviour kept: sampling and inference actions never produce an
    execution_result, and they are not 'nothing ran'."""
    for task_type in ("sample_profile", "start_inference", "stop_inference"):
        status, _ = celery_module._run_status_from_final({"result": {}}, task_type=task_type)
        assert status == "success", task_type


# ---------------------------------------------------------------------------
# 2. Time limits: soft < hard, and a timed-out or killed run is failed
# ---------------------------------------------------------------------------

def test_the_soft_time_limit_fires_before_the_hard_one():
    conf = celery_module.celery_app.conf
    assert conf.task_soft_time_limit < conf.task_time_limit


def test_scheduled_training_limits_keep_soft_below_hard():
    from app.agents import scheduler
    opts = scheduler._celery_options_for_schedule({"task_type": "training"})
    assert opts["soft_time_limit"] < opts["time_limit"]


def test_a_soft_timeout_records_failed_timeout(sb, monkeypatch):
    from celery.exceptions import SoftTimeLimitExceeded

    def _slow(_state):
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr(celery_module, "_run_agent_task", _slow)
    monkeypatch.setattr(celery_module, "current_task",
                        type("T", (), {"request": type("Rq", (), {"id": "run-soft"})()})())
    with pytest.raises(SoftTimeLimitExceeded):
        celery_module.agent_task.run({"schedule_id": "sched-1", "task_schedule": {"task_type": "execute"}})

    final_payload, _ = sb.log[-1]
    assert final_payload["status"] == "failure"
    truth = _truth(final_payload)
    assert truth.get("outcome") == "failed"
    assert truth.get("reason_code") == "timeout"
    _assert_contract(truth)


def _request_for(task, args):
    from celery.contrib.testing.mocks import TaskMessage
    from celery.utils.imports import symbol_by_name
    cls = symbol_by_name(task.Request)
    return cls(TaskMessage(task.name, id="run-hard", args=args), app=celery_module.celery_app, task=task)


@pytest.mark.parametrize("task_name", ["agent_task", "execution_agent_task"])
def test_a_hard_kill_leaves_no_row_running(sb, monkeypatch, task_name):
    task = getattr(celery_module, task_name)
    monkeypatch.setattr(task.backend, "mark_as_failure", lambda *a, **k: None)
    sb.rows["run-hard"] = {"run_id": "run-hard", "status": "running"}

    req = _request_for(task, [{"schedule_id": "sched-1", "task_schedule": {"task_type": "execute"}}])
    req.on_timeout(soft=False, timeout=celery_module.celery_app.conf.task_time_limit)

    row = sb.rows["run-hard"]
    assert row["status"] != "running"
    assert row["status"] == "failure"
    truth = (row.get("result") or {}).get("run_truth") or {}
    assert truth.get("outcome") == "failed"
    assert truth.get("reason_code") == "timeout"


def test_a_lost_worker_leaves_no_row_running(sb, monkeypatch):
    from billiard.einfo import ExceptionInfo
    from celery.exceptions import WorkerLostError

    task = celery_module.agent_task
    monkeypatch.setattr(task.backend, "mark_as_failure", lambda *a, **k: None)
    sb.rows["run-hard"] = {"run_id": "run-hard", "status": "running"}
    req = _request_for(task, [{"schedule_id": "sched-1", "task_schedule": {"task_type": "execute"}}])
    try:
        raise WorkerLostError("Worker exited prematurely: signal 9 (SIGKILL)")
    except WorkerLostError:
        einfo = ExceptionInfo()
    req.on_failure(einfo, send_failed_event=False, return_ok=False)

    row = sb.rows["run-hard"]
    assert row["status"] == "failure"
    assert ((row.get("result") or {}).get("run_truth") or {}).get("reason_code") == "worker_lost"


def test_a_hard_kill_keeps_the_ray_job_identity(sb, monkeypatch):
    task = celery_module.agent_task
    monkeypatch.setattr(task.backend, "mark_as_failure", lambda *a, **k: None)
    sb.rows["run-hard"] = {"run_id": "run-hard", "status": "running",
                           "result": {"ray_job": {"job_id": "raysubmit_1"}}}
    req = _request_for(task, [{"schedule_id": "sched-1"}])
    req.on_timeout(soft=False, timeout=1860)
    row = sb.rows["run-hard"]
    assert row["status"] == "failure"
    assert row["result"].get("ray_job") == {"job_id": "raysubmit_1"}


def test_a_kill_after_the_finish_write_does_not_overwrite_it(sb, monkeypatch):
    """Exactly one outcome: the parent only moves a row still marked running."""
    task = celery_module.agent_task
    monkeypatch.setattr(task.backend, "mark_as_failure", lambda *a, **k: None)
    sb.rows["run-hard"] = {"run_id": "run-hard", "status": "success", "result": {"run_truth": {"outcome": "succeeded"}}}
    req = _request_for(task, [{"schedule_id": "sched-1"}])
    req.on_timeout(soft=False, timeout=1860)
    assert sb.rows["run-hard"]["status"] == "success"
    assert any(f.get("status") == "running" for _, f in sb.log), "the kill write must be conditional on running"


# ---------------------------------------------------------------------------
# 3. Every scheduled run kind writes run_truth with outcome, reason and scope
# ---------------------------------------------------------------------------

_SCHEDULED_CASES = {
    "executor_success": (
        {"execution_mode": "local", "execution_result": {"status": "success", "output_row_count": 3}},
        "succeeded", "ok", "executor",
    ),
    "executor_failure": (
        {"execution_mode": "local", "execution_result": {"status": "failed", "message": "KeyError: 'x'"}},
        "failed", "exception", "executor",
    ),
    "ray_success_full": (
        {"execution_mode": "k8s-ray", "execution_result": {
            "mode": "k8s-ray", "status": "succeeded",
            "data_scope": {"rows_used": 200000, "rows_total": 200000, "sample": False}}},
        "succeeded", "ok", "ray",
    ),
    "ray_sample": (
        {"execution_mode": "k8s-ray", "execution_result": {
            "mode": "k8s-ray", "status": "succeeded",
            "data_scope": {"rows_used": 50000, "rows_total": 200000, "sample": True}}},
        "partial", "sampled", "ray",
    ),
    "ray_timeout": (
        {"execution_mode": "k8s-ray", "execution_result": {"mode": "k8s-ray", "status": "timeout"}},
        "failed", "timeout", "ray",
    ),
    "ray_refused": (
        {"execution_mode": "k8s-ray", "execution_result": {
            "mode": "k8s-ray", "status": "error", "message": "no connection",
            "refused_reason": "no cloud storage connection is linked to this dataset"}},
        "failed", "refused", "ray",
    ),
}


@pytest.mark.parametrize("case", sorted(_SCHEDULED_CASES))
def test_every_scheduled_run_kind_writes_run_truth(sb, case):
    final, outcome, reason_code, engine = _SCHEDULED_CASES[case]
    celery_module._record_run_finish(
        "sched-1", f"run-{case}", final, time.perf_counter(),
        scheduled_state={"task_schedule": {"task_type": "execute"}},
    )
    payload, _ = sb.log[-1]
    truth = _truth(payload)
    _assert_contract(truth)
    assert truth["outcome"] == outcome
    assert truth["reason_code"] == reason_code
    assert truth["scope"]["engine"] == engine
    assert payload["status"] == ("failure" if outcome == "failed" else "success")


def test_ray_scope_comes_from_the_executor_not_an_estimate(sb):
    final, *_ = _SCHEDULED_CASES["ray_sample"]
    celery_module._record_run_finish("sched-1", "run-ray", final, time.perf_counter(),
                                     scheduled_state={"task_schedule": {"task_type": "execute"}})
    scope = _truth(sb.log[-1][0])["scope"]
    assert scope["rows_processed"] == 50000
    assert scope["rows_total"] == 200000
    assert scope["rows_total_kind"] == "exact"
    assert scope["sample"] is True


def test_an_uncaught_scheduled_failure_writes_run_truth(sb):
    try:
        raise RuntimeError("worker blew up")
    except RuntimeError as exc:
        celery_module._record_run_failure("sched-1", "run-x", exc, time.perf_counter(),
                                          scheduled_state={"task_schedule": {"task_type": "execute"}})
    truth = _truth(sb.log[-1][0])
    _assert_contract(truth)
    assert truth["outcome"] == "failed"
    assert truth["reason_code"] == "exception"


def test_scheduled_training_success_and_failure_write_run_truth(sb):
    ok = {"training_completed": True, "training_result": {"status": "success", "mlflow_run_id": "m1"}}
    celery_module._record_run_finish("s", "r1", ok, time.perf_counter(),
                                     scheduled_state={"task_schedule": {"task_type": "training"}})
    assert _truth(sb.log[-1][0])["outcome"] == "succeeded"
    bad = {"training_result": {"status": "error", "error": "OOM"}}
    celery_module._record_run_finish("s", "r2", bad, time.perf_counter(),
                                     scheduled_state={"task_schedule": {"task_type": "training"}})
    payload = sb.log[-1][0]
    assert payload["status"] == "failure"
    assert _truth(payload)["outcome"] == "failed"


# ---------------------------------------------------------------------------
# 4. The model cannot set run_truth
# ---------------------------------------------------------------------------

_FORGED = {"outcome": "succeeded", "reason_code": "ok", "reason": "All rows analysed.",
           "scope": {"rows_processed": 10**9, "rows_total": 10**9, "sample": False}, "written_by": "model"}


def test_a_model_supplied_run_truth_in_a_scheduled_state_is_overwritten(sb):
    final = {"run_truth": dict(_FORGED), "execution_result": {"run_truth": dict(_FORGED)}}
    celery_module._record_run_finish("sched-1", "run-forged", final, time.perf_counter(),
                                     scheduled_state={"task_schedule": {"task_type": "execute"}})
    payload, _ = sb.log[-1]
    truth = _truth(payload)
    assert truth["outcome"] == "failed"
    assert truth["reason_code"] == "nothing_ran"
    assert truth["written_by"] == "harness"
    assert truth["scope"]["rows_processed"] is None


# ---------------------------------------------------------------------------
# 5. Turns: interactive and deferred
# ---------------------------------------------------------------------------

def _graph(result_fields, delay=0.0):
    class _G:
        checkpointer = object()

        def invoke(self, state_in, config=None):
            if delay:
                time.sleep(delay)
            out = {
                "messages": state_in["messages"] + [AIMessage(content="done")],
                "planner_definition": {}, "ready_to_summarize": False, "ready_to_code": False,
                "coder_definition": {}, "visualization_config": {}, "visualization_status": "",
                "execution_result": {},
            }
            out.update(result_fields)
            return out
    return _G()


def _quiet(monkeypatch):
    monkeypatch.setattr(server, "_parse_fidelity_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_parse_selected_sample_from_text", lambda t: None, raising=False)
    monkeypatch.setattr(server, "_is_mode_switch_message", lambda t: False, raising=False)


_FAILED_ANALYSIS = {
    "execution_result": {"status": "failed", "message": "ZeroDivisionError: division by zero"},
    "execution_error": "ZeroDivisionError: division by zero",
}


def test_an_interactive_failed_turn_carries_failed_run_truth(client: TestClient, monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _graph({**_FAILED_ANALYSIS, "run_truth": dict(_FORGED)}), raising=False)
    up = _do_upload(client, user_id="rt-user")
    r = client.post(f"/threads/{up['thread_id']}/messages",
                    json={"content": "average of a", "metadata": {"dataset_id": up["dataset_id"]}},
                    headers=make_auth_headers("rt-user"))
    assert r.status_code == 200, r.text
    truth = r.json().get("run_truth") or {}
    assert truth.get("outcome") == "failed"
    assert truth.get("reason_code") == "exception"
    assert truth.get("written_by") == "harness"
    _assert_contract(truth)


def test_an_interactive_turn_that_ran_nothing_says_so(client: TestClient, monkeypatch):
    """A conversational reply is succeeded/reply_only with an empty scope, so it
    can never be read as an analysis that covered the data."""
    _quiet(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _graph({}), raising=False)
    up = _do_upload(client, user_id="rt-user2")
    r = client.post(f"/threads/{up['thread_id']}/messages",
                    json={"content": "hello", "metadata": {"dataset_id": up["dataset_id"]}},
                    headers=make_auth_headers("rt-user2"))
    truth = r.json().get("run_truth") or {}
    assert truth.get("outcome") == "succeeded"
    assert truth.get("reason_code") == "reply_only"
    assert truth["scope"]["rows_processed"] is None


def _run_deferred(client, monkeypatch, result_fields, user_id):
    _quiet(monkeypatch)
    monkeypatch.setattr(server, "TURN_SYNC_DEADLINE_S", 0.05, raising=False)
    monkeypatch.setattr(server, "GRAPH", _graph(result_fields, delay=0.4), raising=False)
    notes = []

    async def _capture(**kw):
        notes.append(kw)

    monkeypatch.setattr(server, "_notify_deferred_turn", _capture)
    up = _do_upload(client, user_id=user_id)
    thread_id = up["thread_id"]
    r = client.post(f"/threads/{thread_id}/messages",
                    json={"content": "average of a", "metadata": {"dataset_id": up["dataset_id"]}},
                    headers=make_auth_headers(user_id))
    assert r.status_code == 200, r.text
    deferred_id = r.json().get("analysis_task_id")
    assert deferred_id
    deadline = time.monotonic() + POLL_TIMEOUT_S
    body = {}
    while time.monotonic() < deadline:
        body = client.get(f"/threads/{thread_id}/pending-turn?deferred_id={deferred_id}",
                          headers=make_auth_headers(user_id)).json()
        if body.get("status") != "running" and notes:
            break
        time.sleep(0.05)
    return notes, body, thread_id


def test_a_deferred_turn_whose_analysis_failed_notifies_ok_false(shared_loop_client, monkeypatch):
    notes, _body, _ = _run_deferred(shared_loop_client, monkeypatch, _FAILED_ANALYSIS, "rt-def-fail")
    assert notes, "no notification was sent"
    assert notes[-1]["ok"] is False
    assert notes[-1].get("error_message")


def test_a_deferred_turn_records_its_outcome(shared_loop_client, monkeypatch):
    _notes, _body, thread_id = _run_deferred(shared_loop_client, monkeypatch, _FAILED_ANALYSIS, "rt-def-rec")
    pt = server._PENDING_TURN_MEM.get(thread_id) or {}
    assert pt.get("outcome") == "failed"
    truth = (pt.get("result") or {}).get("run_truth") or {}
    assert truth.get("kind") == "deferred"
    _assert_contract(truth)


def test_a_deferred_turn_that_succeeded_still_notifies_ok_true(shared_loop_client, monkeypatch):
    notes, _body, thread_id = _run_deferred(
        shared_loop_client, monkeypatch,
        {"execution_result": {"status": "success", "fresh_output": True}}, "rt-def-ok",
    )
    assert notes and notes[-1]["ok"] is True
    pt = server._PENDING_TURN_MEM.get(thread_id) or {}
    assert pt.get("outcome") in ("succeeded", "partial")
