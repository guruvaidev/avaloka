"""Guard: no API response body or CLI line carries raw error output.

Owner rule: the user sees a plain, code-written message; a traceback, a stack
frame or a raw exception stays in the server log. Each surface below is driven
through its real endpoint with a forced failure that puts raw error text in
its way, and every string in the response is swept for the shapes in
RAW_ERROR. A new leak on any of these surfaces trips the sweep whatever field
it arrives in.

The patterns are defined here, not imported from app.core.user_errors, so the
guard does not share a blind spot with the code it checks.
"""
from __future__ import annotations

import datetime
import json
import re
import traceback
from typing import Any, Iterator, List

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import app.api.server as server
from app.agents.contract import failure_record
from app.api.schemas import ChatResponse

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import (  # noqa: F401
        SESSIONS, FakeGraph, _do_upload, _install_sampler, _quiet_send, _rows, client, make_auth_headers,
    )
except ImportError:
    from server_harness import (  # noqa: F401
        SESSIONS, FakeGraph, _do_upload, _install_sampler, _quiet_send, _rows, client, make_auth_headers,
    )


RAW_ERROR = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r'File "[^"]*", line \d+'),
    re.compile(r"\b[A-Z]\w*(?:Error|Exception)\((?:'|\")"),
    re.compile(r"\b[A-Z]\w*(?:Error|Exception)\s*:\s*\S"),
]

STDERR = (
    "Traceback (most recent call last):\n"
    '  File "/tmp/avaloka/run_77/script.py", line 41, in <module>\n'
    "    result = main(df)\n"
    '  File "/tmp/avaloka/run_77/script.py", line 12, in main\n'
    "    return df['revenue_2026'].astype(float)\n"
    "ValueError: could not convert string to float: 'n/a'\n"
)


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def assert_no_raw_error(payload: Any, where: str) -> None:
    hits = [(rx.pattern, s[:160]) for s in _strings(payload) for rx in RAW_ERROR if rx.search(s)]
    assert not hits, f"{where} carries raw error output: {hits}"


def _sse(text: str) -> List[dict]:
    return [json.loads(line[len("data: "):]) for block in text.split("\n\n")
            for line in block.splitlines() if line.startswith("data: ")]


def test_the_patterns_catch_each_shape():
    """The sweep is not vacuous: each shape it must catch is caught."""
    for sample in (STDERR, "ValueError('bad')", "KeyError: 'revenue_2026'",
                   '  File "x.py", line 3, in main'):
        with pytest.raises(AssertionError):
            assert_no_raw_error({"x": [sample]}, "sample")
    assert_no_raw_error({"m": "Model training could not be completed. Error reference: `X`"}, "plain")


# --------------------------------------------------------------------------- #
# save-and-execute: validation_feedback
# --------------------------------------------------------------------------- #

def _save_and_execute(client, monkeypatch, code):
    async def _aid(analysis_id, user_id):
        return {"session_id": "s-guard"}

    monkeypatch.setattr(server, "_resolve_session_from_aid", _aid, raising=False)
    SESSIONS["s-guard"] = {"user_id": "guard-user", "schema": {"revenue_2026": "object"}}
    monkeypatch.setattr(server, "review_edited_code",
                        lambda state, code, user_intent="": {"code": code, "status": "approved"},
                        raising=False)
    return client.post("/analysis/aid-guard/save-and-execute",
                       json={"code": code, "run_validation": True},
                       headers=make_auth_headers("guard-user"))


def test_save_and_execute_syntax_error_is_plain(client: TestClient, monkeypatch):
    # The real validator, no stub: its syntax feedback is the compiler's error line.
    resp = _save_and_execute(client, monkeypatch, "def main(df)\n    return df\n")
    body = resp.json()
    assert body["status"] == "validation_failed", body
    assert_no_raw_error(body, "save-and-execute (syntax)")
    assert "syntax error" in body["validation_feedback"]


@pytest.mark.parametrize("feedback", [
    STDERR,
    "ValueError: could not convert string to float: 'n/a'",
    "could not convert string to float: 'n/a'",
])
def test_save_and_execute_sample_failure_is_plain(client: TestClient, monkeypatch, feedback):
    monkeypatch.setattr(server, "run_code_validation", lambda state: {
        "ok": False, "feedback": feedback, "code": state["generated_code"], "stage": "sample_execution",
    }, raising=False)
    body = _save_and_execute(client, monkeypatch, "def main(df):\n    return df\n").json()
    assert body["status"] == "validation_failed", body
    assert_no_raw_error(body, "save-and-execute (sample run)")
    # The sample run's text is the exception itself: data values never reach the user.
    assert "n/a" not in body["validation_feedback"]
    assert "small sample of your data" in body["validation_feedback"]


# --------------------------------------------------------------------------- #
# chat: agent_errors and the training-failure reply
# --------------------------------------------------------------------------- #

def _failure_record_with_traceback():
    try:
        {}["revenue_2026"]
    except KeyError as exc:
        return failure_record("coder", exc)


def test_chat_response_agent_errors_carry_no_traceback():
    record = _failure_record_with_traceback()
    assert "Traceback" in record["traceback"]          # the forced failure is real
    resp = ChatResponse(messages=[], agent_errors=[record])
    payload = resp.model_dump()
    assert_no_raw_error(payload, "ChatResponse")
    assert payload["agent_errors"] == [{"agent": "coder", "error_class": "KeyError"}]


def test_chat_training_failure_reply_is_plain(client: TestClient, monkeypatch):
    _quiet_send(monkeypatch)
    _install_sampler(monkeypatch, {"random_baseline": _rows(20)})
    record = _failure_record_with_traceback()

    class TrainingFailedGraph(FakeGraph):
        def invoke(self, state_in, config=None):
            return {
                "messages": list(state_in["messages"]),
                "training_result": {"status": "error", "error": STDERR},
                "agent_errors": [record],
                "execution_result": {},
            }

    monkeypatch.setattr(server, "GRAPH", TrainingFailedGraph(), raising=False)
    out = _do_upload(client, user_id="guard-chat")
    resp = client.post(
        f"/threads/{out['thread_id']}/messages",
        json={"content": "train a model to predict revenue", "metadata": {"dataset_id": out["dataset_id"]}},
        headers=make_auth_headers("guard-chat"),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("training_status") == "failed", body.get("training_status")
    replies = [m for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "assistant"]
    assert replies, body.get("messages")
    assert_no_raw_error(body, "chat response (training failure)")


# --------------------------------------------------------------------------- #
# scheduled runs: status (SSE), result, run history
# --------------------------------------------------------------------------- #

class _Schedule:
    def remaining_estimate(self, last_run_at):
        return datetime.timedelta(seconds=30)


class _Entry:
    args = [{"task_schedule": {"task_type": "execute"}}]
    schedule = _Schedule()
    max_runs = 1
    total_run_count = 1
    last_run_at = datetime.datetime(2026, 10, 18, tzinfo=datetime.timezone.utc)
    due_at = datetime.datetime(2026, 10, 19, tzinfo=datetime.timezone.utc)

    @staticmethod
    def generate_key(app, task_id):
        return f"redbeat:{task_id}"

    @classmethod
    def from_key(cls, key, app):
        return cls()


class _Scheduler:
    @staticmethod
    def get_task_ids(key, generate_key=False):
        return ["exec-1"]

    @staticmethod
    def get_task_ids_with_timestamp(key, generate_key=False):
        return [("exec-1", 1760000000.0)]

    @staticmethod
    def get_last_run_at(key, generate_key=False):
        return None

    @staticmethod
    def get_last_run_task_id(key, generate_key=False):
        return "exec-1"


def _failed_async_result(running_polls=0):
    lookups = {"n": 0}

    class _AsyncResult:
        traceback = STDERR
        date_done = None

        def __init__(self, task_id, app=None):
            lookups["n"] += 1
            self.state = "STARTED" if lookups["n"] <= running_polls else "FAILURE"
            self.result = ValueError("could not convert string to float: 'n/a'") \
                if self.state == "FAILURE" else None

        def failed(self):
            return self.state == "FAILURE"

    return _AsyncResult


@pytest.fixture
def scheduled(client: TestClient, monkeypatch):
    async def _no_sleep(*_a, **_k):
        return None

    upload = _do_upload(client, user_id="guard-sched")
    SESSIONS[upload["session_id"]]["tasks"] = ["sched-1"]
    monkeypatch.setattr(server, "AvalokaEntry", _Entry, raising=False)
    monkeypatch.setattr(server, "AvalokaScheduler", _Scheduler, raising=False)
    monkeypatch.setattr(server, "sleep", _no_sleep, raising=False)
    headers = make_auth_headers("guard-sched")
    headers["X-Avaloka-Session"] = upload["session_id"]
    return headers


def test_scheduled_status_stream_carries_no_traceback(client, monkeypatch, scheduled):
    monkeypatch.setattr(server, "AsyncResult", _failed_async_result(running_polls=2), raising=False)
    resp = client.get("/tasks/sched-1/status", headers=scheduled)
    assert resp.status_code == 200, resp.text
    events = [e for e in _sse(resp.text) if e.get("status")]
    assert events and events[-1]["status"] == "FAILED", events
    assert_no_raw_error(events, "task status stream")


def test_scheduled_result_carries_no_traceback(client, monkeypatch, scheduled):
    monkeypatch.setattr(server, "AsyncResult", _failed_async_result(), raising=False)
    body = client.get("/tasks/sched-1/result/0", headers=scheduled).json()
    assert body.get("status") == "FAILED", body
    assert_no_raw_error(body, "task result")


def test_run_history_from_the_result_backend_carries_no_traceback(client, monkeypatch, scheduled):
    async def _no_table(schedule_id):
        return None

    monkeypatch.setattr(server, "_load_scheduled_runs", _no_table, raising=False)
    monkeypatch.setattr(server, "AsyncResult", _failed_async_result(), raising=False)
    runs = client.get("/tasks/sched-1/runs", headers=scheduled).json()
    assert runs and runs[0]["status"] == "failure", runs
    assert_no_raw_error(runs, "run history (result backend)")


def test_run_history_from_the_durable_table_carries_no_traceback(client, monkeypatch, scheduled):
    # Rows as celery_app._record_run_failure / _record_run_finish write them.
    try:
        raise ValueError("could not convert string to float: 'n/a'")
    except ValueError as exc:
        rows = [
            {"run_id": "exec-2", "run_number": 2, "status": "failure", "error": str(exc),
             "result": {"logs": [f"[error] {exc}", traceback.format_exc()]}},
            {"run_id": "exec-1", "run_number": 1, "status": "failure", "error": "execution failed",
             "result": {"logs": ["[execution] status=failed", f"[stderr] {STDERR}",
                                 f"[message] Script failed with exit status 1\n\n{STDERR}"],
                        "message": None}},
        ]

    async def _table(schedule_id):
        return json.loads(json.dumps(rows))

    monkeypatch.setattr(server, "_load_scheduled_runs", _table, raising=False)
    runs = client.get("/tasks/sched-1/runs", headers=scheduled).json()
    assert len(runs) == 2, runs
    assert_no_raw_error(runs, "run history (durable table)")
    # The non-error log lines survive.
    assert "[execution] status=failed" in runs[1]["result"]["logs"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_crash_line_is_plain(monkeypatch, capsys):
    import avaloka.cli as cli

    def boom():
        raise ValueError("could not convert string to float: 'n/a'")

    monkeypatch.setattr(cli, "app", boom)
    monkeypatch.delenv("AVALOKA_TRACEBACK", raising=False)
    with pytest.raises(SystemExit):
        cli.main()
    out = " ".join(capsys.readouterr().out.split())
    assert "couldn't recover" in out
    assert_no_raw_error(out, "CLI")
    assert "n/a" not in out


# --------------------------------------------------------------------------- #
# chat: a failed execution step (1.6 surfaced the exception line as "Reason:")
# --------------------------------------------------------------------------- #

def _exec_state(tmp_path):
    return {
        "messages": [],
        "coder_definition": {"code": "print('x')"},
        "coder_raw_response": None,
        "generated_code": None,
        "data_source_location": None,
        "output_location": str(tmp_path / "result.csv"),
        "infrastructure_provisioned": {},
    }


def test_chat_reply_for_a_failed_local_run_is_plain(tmp_path, monkeypatch):
    import app.agents.execution_agent as exec_mod

    monkeypatch.setattr(exec_mod, "execute_code_on_local", lambda **k: {
        "status": "error", "execution_error": "Script failed with exit status 1",
        "execution_stderr": STDERR, "returncode": 1,
    })
    new_state = exec_mod.execution_agent_node_local(_exec_state(tmp_path))
    reply = new_state["messages"][-1].content
    assert "couldn't complete this step" in reply
    assert_no_raw_error(reply, "chat reply (failed run)")
    assert "n/a" not in reply


def test_chat_reply_when_the_output_cannot_be_prepared_is_plain(tmp_path, monkeypatch):
    import app.agents.execution_agent as exec_mod

    def fake_execute(**k):
        with open(k["output_location"], "w", encoding="utf-8") as f:
            f.write("a,b\n1,2\n")
        return {"status": "success"}

    def broken_read_csv(*a, **k):
        raise ValueError("could not convert string to float: 'n/a'")

    monkeypatch.setattr(exec_mod, "execute_code_on_local", fake_execute)
    monkeypatch.setattr(exec_mod.pd, "read_csv", broken_read_csv)
    new_state = exec_mod.execution_agent_node_local(_exec_state(tmp_path))
    reply = new_state["messages"][-1].content
    assert "couldn't prepare" in reply, reply
    assert_no_raw_error(reply, "chat reply (output preparation)")
    assert "n/a" not in reply


# --------------------------------------------------------------------------- #
# chat: a data transfer's "Reason:" line (planner's "Transfer Setup Failed" reply)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("detail", [
    STDERR,
    "KeyError: 'revenue_2026'",
    "sqlalchemy.exc.ProgrammingError: (psycopg.errors.UndefinedColumn) column \"revenue_2026\" does not exist",
])
def test_dta_reason_line_is_plain_for_a_raw_error(detail):
    from app.agents.data_transfer_agent.data_transfer_agent import _plain_reason

    line = _plain_reason(detail)
    assert_no_raw_error(line, "DTA reason line")
    assert "revenue_2026" not in line and "n/a" not in line


def test_dta_reason_line_keeps_a_code_written_message():
    from app.agents.data_transfer_agent.data_transfer_agent import _plain_reason

    plain = "The request references column(s) that don't exist in the source table."
    assert _plain_reason(plain) == plain


# --------------------------------------------------------------------------- #
# run_truth: the reason sentence on every run record and chat response
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("detail", [STDERR, "KeyError: 'revenue_2026'", "ValueError('n/a')"])
def test_run_truth_reason_carries_no_raw_error(detail):
    from app.core import run_truth

    for truth in (
        run_truth.failed("exception", kind="interactive", detail=detail),
        run_truth.classify_turn({"training_result": {"status": "error", "error": detail}}),
        run_truth.classify_scheduled({"execution_result": {"status": "failed", "error": detail}}),
    ):
        assert truth["outcome"] == "failed"
        assert_no_raw_error(truth, "run_truth")
        assert "revenue_2026" not in truth["reason"] and "n/a" not in truth["reason"]
    # A code-written detail is kept as it is.
    kept = run_truth.failed("refused", kind="interactive", detail="the output contract refused it")
    assert kept["reason"] == "The run was refused: the output contract refused it"
