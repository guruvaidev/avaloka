"""The replan loop's record reaches the run truth the server and the worker write.

``classify_turn_with_recovery`` existed before this, but nothing called it:
the chat response's ``run_truth`` and the scheduled run's ``result.run_truth``
were still ``classify_turn`` / ``classify_scheduled``, so a turn the loop
recovered by leaving rows out was recorded ``succeeded/ok`` with no
``recovery`` block, and a scheduled run that stopped on the measured-column
question was ``failed/exception``. Each test here fails on that tree on an
assertion (no ``recovery`` key; the old reason code).
"""

import time

from fastapi.testclient import TestClient

import app.api.server as server
from app.core import celery_app as celery_module

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
        SESSIONS, THREAD_TO_SESSION, make_auth_headers, client, _do_upload,
    )
    from tests.test_run_truth_8a import _Supabase, _graph, _quiet
except ImportError:
    from server_harness import (  # noqa: F401
        SESSIONS, THREAD_TO_SESSION, make_auth_headers, client, _do_upload,
    )
    from test_run_truth_8a import _Supabase, _graph, _quiet

#: A local run the loop recovered by leaving 12 rows out, as the graph leaves
#: the state: the last attempt is still "pending" (the node closes it only on
#: a further failure) and replan_final is unset.
_RECOVERED_BY_EXCLUSION = {
    "execution_mode": "local",
    "execution_result": {"status": "success", "output_row_count": 1},
    "replan_attempts": [{
        "n": 1, "error_class": "IntCastingNaNError", "error_kind": "data", "diagnosis": "missing_values",
        "diagnosed_by": "rule", "code_sha": "abc", "outcome": "pending", "action": "data_step",
        "code_changed": False, "duration_ms": None,
        "transforms": [{"op": "exclude_rows", "column": "revenue", "rows_total": 300, "rows_affected": 12,
                        "non_finite": 0, "chosen_by": "env", "measured": True}],
    }],
    "replan_transforms": [{"op": "exclude_rows", "column": "revenue", "rows_total": 300, "rows_affected": 12}],
}

#: A run that stopped on the measured-column question, nothing applied.
_STOPPED_ON_THE_QUESTION = {
    "execution_mode": "local",
    "execution_result": {"status": "failed", "message": "Script failed with exit status 1"},
    "execution_error": "Script failed with exit status 1",
    "replan_attempts": [],
    "replan_final": "needs_user",
    "replan_stopped": {"reason": "needs_user", "error_class": "IntCastingNaNError", "diagnosis": "missing_values",
                       "diagnosed_by": "rule", "error_kind": "data", "choice": "no_answer", "interactive": False},
}


def _scheduled_truth(monkeypatch, final):
    fake = _Supabase()
    monkeypatch.setattr(celery_module, "_supabase", lambda: fake)
    celery_module._record_run_finish("sched-1", "run-replan", final, time.perf_counter(),
                                     scheduled_state={"task_schedule": {"task_type": "execute"}})
    payload, _ = fake.log[-1]
    return payload, payload["result"]["run_truth"]


def test_a_scheduled_run_recovered_by_exclusion_is_recorded_partial_with_its_recovery(monkeypatch):
    payload, truth = _scheduled_truth(monkeypatch, dict(_RECOVERED_BY_EXCLUSION))
    assert "recovery" in truth, sorted(truth)
    assert (truth["outcome"], truth["reason_code"]) == ("partial", "rows_excluded")
    assert "12 of 300 rows" in truth["reason"]
    assert truth["recovery"]["final"] == "recovered"
    [attempt] = truth["recovery"]["attempts"]
    assert attempt["outcome"] == "recovered" and attempt["transforms"][0]["op"] == "exclude_rows"
    # The output was produced, so the row the UI reads keeps "success".
    assert payload["status"] == "success"
    assert truth["written_by"] == "harness"


def test_a_scheduled_run_that_stopped_on_the_question_is_recorded_needs_decision(monkeypatch):
    payload, truth = _scheduled_truth(monkeypatch, dict(_STOPPED_ON_THE_QUESTION))
    assert (truth["outcome"], truth["reason_code"]) == ("failed", "needs_decision")
    assert truth["recovery"]["stopped"]["choice"] == "no_answer"
    assert truth["recovery"]["attempts"] == []
    assert payload["status"] == "failure"
    # The error the row shows is the run truth's reason, not the exit status.
    assert "exit status" not in (payload.get("error") or "")


def test_the_chat_turn_run_truth_carries_the_recovery(client: TestClient, monkeypatch):
    _quiet(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _graph(dict(_RECOVERED_BY_EXCLUSION)), raising=False)
    up = _do_upload(client, user_id="rt-replan")
    r = client.post(f"/threads/{up['thread_id']}/messages",
                    json={"content": "total revenue", "metadata": {"dataset_id": up["dataset_id"]}},
                    headers=make_auth_headers("rt-replan"))
    assert r.status_code == 200, r.text
    truth = r.json().get("run_truth") or {}
    assert "recovery" in truth, sorted(truth)
    assert (truth["outcome"], truth["reason_code"]) == ("partial", "rows_excluded")
    assert truth["kind"] == "interactive"
    assert truth["recovery"]["attempts"][0]["transforms"][0]["column"] == "revenue"


def test_a_turn_the_loop_never_touched_is_unchanged(client: TestClient, monkeypatch):
    _quiet(monkeypatch)
    plain = {"execution_mode": "local", "execution_result": {"status": "success", "output_row_count": 3}}
    monkeypatch.setattr(server, "GRAPH", _graph(plain), raising=False)
    up = _do_upload(client, user_id="rt-replan2")
    r = client.post(f"/threads/{up['thread_id']}/messages",
                    json={"content": "total revenue", "metadata": {"dataset_id": up["dataset_id"]}},
                    headers=make_auth_headers("rt-replan2"))
    truth = r.json().get("run_truth") or {}
    assert (truth["outcome"], truth["reason_code"]) == ("succeeded", "ok")
    assert "recovery" not in truth
