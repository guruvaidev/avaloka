"""The develop-1.7 API wiring, exercised through 1.6's real server.

The develop-1.6 -> 1.7 merge kept 1.6's server.py, which silently dropped the
wiring PR #245 added: ``app/api/ui_contract.py`` and ``app/api/swarm_bridge.py``
survived the merge, but nothing imported them, and their own unit tests kept
passing because they never touch the server. Every test here goes through an
HTTP route, so removing a piece of wiring from server.py fails one of them.
"""
from __future__ import annotations

import datetime
import json
import sys
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

import app.api.server as server
from app.agents.avaloka_agent import capabilities as swarm_capabilities

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
        SESSIONS, FakeGraph, client, make_auth_headers, _do_upload, _quiet_send,
    )
except ImportError:
    from server_harness import (  # noqa: F401
        SESSIONS, FakeGraph, client, make_auth_headers, _do_upload, _quiet_send,
    )


_SWARM_PLAN = [{
    "clone": "scout", "name": "Ava-Scout", "role": "Data scout",
    "focus": "inspect schema", "status": "pending",
}]


class _RecordingGraph(FakeGraph):
    """FakeGraph that records each graph input and can play the agent's part.

    ``consume_event`` mimics avaloka_agent, which returns
    ``connection_event: None`` once it has onboarded the user.
    """

    def __init__(self, *, consume_event: bool = False, swarm: bool = False):
        self.inputs: List[Dict[str, Any]] = []
        self.consume_event = consume_event
        self.swarm = swarm

    def invoke(self, state_in, config=None):
        self.inputs.append(dict(state_in))
        final = super().invoke(state_in, config)
        if self.consume_event and state_in.get("connection_event"):
            final["connection_event"] = None
        if self.swarm:
            final["avaloka_swarm"] = _SWARM_PLAN
            final["avaloka_swarm_message"] = "Dispatching a scout"
        return final


def _send(client: TestClient, user: str, upload: Dict[str, Any], text: str):
    resp = client.post(
        f"/threads/{upload['thread_id']}/messages",
        json={"content": text, "metadata": {"dataset_id": upload["dataset_id"]}},
        headers=make_auth_headers(user),
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ======================================================================================
# GET /api/capabilities
# ======================================================================================


def test_capabilities_route_returns_documented_shape(client: TestClient):
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200, resp.text
    doc = resp.json()

    assert doc["api_contract"] == "1.5.2-compatible"
    assert doc["compatibility"] == "additive"
    features = doc["features"]
    swarm = features["swarm_execution_context"]
    assert swarm["path"] == "execution_context.swarm"
    assert swarm["optional"] is True
    assert swarm["routing_semantics"] is False
    assert swarm["enabled"] is swarm_capabilities.swarm_available()
    assert features["async_execution_context"] is True
    # Formats are what POST /api/upload accepts, not what the CLI loader reads.
    assert features["upload_formats"] == sorted(server.SUPPORTED_UPLOAD_EXTS)
    assert "mongodb" not in json.dumps(doc)
    # Beside the list, each format's status: accepted formats enabled, the
    # table layouts no route serves deferred with a reason.
    status = features["upload_format_status"]
    assert set(status) == set(server.SUPPORTED_UPLOAD_EXTS) | set(server.DEFERRED_UPLOAD_FORMATS)
    assert all(status[ext]["status"] == "enabled" for ext in server.SUPPORTED_UPLOAD_EXTS)
    assert all(status[ext]["status"] == "deferred" and status[ext]["reason"]
               for ext in server.DEFERRED_UPLOAD_FORMATS)


def test_capabilities_reports_swarm_absent_when_module_is_not_shipped(
    client: TestClient, monkeypatch
):
    # Simulate the open-source build, which does not ship swarm.py: a None
    # entry in sys.modules makes the import probe raise ImportError. The
    # package attribute goes too, or `from pkg import swarm` would find an
    # already-imported submodule without consulting sys.modules.
    import app.agents.avaloka_agent as avaloka_agent_pkg

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "app.agents.avaloka_agent.swarm", None)
        m.delattr(avaloka_agent_pkg, "swarm", raising=False)
        swarm_capabilities.reset_cache()
        try:
            doc = client.get("/api/capabilities").json()
        finally:
            swarm_capabilities.reset_cache()

    swarm = doc["features"]["swarm_execution_context"]
    assert swarm["enabled"] is False
    assert swarm["reason"] == "not present in this build"


# ======================================================================================
# Graph input: thread_id distinct from session_id; connection_event round trip
# ======================================================================================


def test_graph_input_carries_thread_id_distinct_from_session_id(
    client: TestClient, monkeypatch
):
    _quiet_send(monkeypatch)
    graph = _RecordingGraph()
    monkeypatch.setattr(server, "GRAPH", graph, raising=False)

    upload = _do_upload(client, user_id="wire-ids")
    _send(client, "wire-ids", upload, "hello")
    assert len(graph.inputs) == 1

    state_in = graph.inputs[-1]
    assert state_in["thread_id"] == upload["thread_id"]
    assert state_in["session_id"] == upload["session_id"]
    assert state_in["thread_id"] != state_in["session_id"]


def test_upload_connection_event_reaches_graph_and_is_consumed_once(
    client: TestClient, monkeypatch
):
    _quiet_send(monkeypatch)
    graph = _RecordingGraph(consume_event=True)
    monkeypatch.setattr(server, "GRAPH", graph, raising=False)

    upload = _do_upload(client, user_id="wire-event")
    sid = upload["session_id"]
    event = SESSIONS[sid].get("connection_event")
    assert event and event["kind"] == "upload"
    assert event["dataset_id"] == upload["dataset_id"]

    _send(client, "wire-event", upload, "hi")
    assert len(graph.inputs) == 1
    assert graph.inputs[-1]["connection_event"] == event
    # The agent consumed it; the stored session must forget it, or every later
    # turn re-reads it from Redis and onboards again.
    assert "connection_event" not in SESSIONS[sid]

    # Must reach the graph: a schema question ("what columns...") is answered
    # by send_message directly and would leave inputs[-1] at the first turn.
    _send(client, "wire-event", upload, "hello again")
    assert len(graph.inputs) == 2
    assert graph.inputs[-1]["connection_event"] is None


def test_unconsumed_connection_event_is_kept_for_the_next_turn(
    client: TestClient, monkeypatch
):
    _quiet_send(monkeypatch)
    graph = _RecordingGraph(consume_event=False)
    monkeypatch.setattr(server, "GRAPH", graph, raising=False)

    upload = _do_upload(client, user_id="wire-keep")
    event = SESSIONS[upload["session_id"]]["connection_event"]

    _send(client, "wire-keep", upload, "hello")
    _send(client, "wire-keep", upload, "hello again")
    assert len(graph.inputs) == 2
    assert graph.inputs[-1]["connection_event"] == event
    assert SESSIONS[upload["session_id"]]["connection_event"] == event


def test_register_existing_storage_sets_connection_event(client: TestClient):
    resp = client.post(
        "/api/register-existing-storage",
        json={
            "storage_uri": "gs://external-bucket/some-prefix",
            "key": "foo.csv",
            "connection_id": "conn-1",
            "schema_json": None,
        },
        headers=make_auth_headers("wire-reg"),
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    event = SESSIONS[out["session_id"]]["connection_event"]
    assert event["kind"] == "cloud_storage"
    assert event["dataset_id"] == out["dataset_id"]
    assert event["connection_id"] == "conn-1"


def test_database_connect_sets_connection_event_without_credentials(
    client: TestClient, monkeypatch
):
    async def list_tables_ok(customer_id: str, api_key: str) -> Dict[str, Any]:
        return {"tables": ["t1", "t2"], "meta": {"database_type": "postgres"}}

    monkeypatch.setattr(server, "list_database_tables", list_tables_ok, raising=False)
    resp = client.post(
        "/api/database/connect",
        json={"customer_id": "cust-1", "api_key": "not-a-real-key"},
        headers=make_auth_headers("wire-db"),
    )
    assert resp.status_code == 200, resp.text
    event = SESSIONS[resp.json()["session_id"]]["connection_event"]
    assert event["kind"] == "database"
    assert event["tables_available"] == 2
    assert "not-a-real-key" not in json.dumps(event)
    # Same id and type the connect response hands back (aligned with #248).
    assert event["connection_id"] == "cust-1" == resp.json()["customer_id"]
    assert event["database_type"] == "postgres" == resp.json()["database_type"]


def test_database_event_uses_the_saved_connection_id_and_type_hint():
    """The legacy saved-connection path (#248) supplies both; the event must
    carry the id the response returns and fall back to the saved type."""
    from app.api.swarm_bridge import database_connection_event

    event = database_connection_event(
        customer_id="cust-1", test_result={"tables": ["t1"], "meta": {}},
        created_at="2026-10-17T00:00:00Z",
        connection_id="saved-conn-9", database_type_hint="mysql",
    )
    assert event["connection_id"] == "saved-conn-9"
    assert event["database_type"] == "mysql"

    listed = database_connection_event(
        customer_id="cust-1",
        test_result={"tables": ["t1"], "meta": {"database_type": "postgres"}},
        created_at="2026-10-17T00:00:00Z",
        connection_id="  ", database_type_hint="mysql",
    )
    assert listed["connection_id"] == "cust-1"      # blank id falls back
    assert listed["database_type"] == "postgres"    # listing wins over the hint


# ======================================================================================
# execution_context on scheduled-task results and the task SSE stream
# ======================================================================================


class _FakeSchedule:
    def remaining_estimate(self, last_run_at):
        return datetime.timedelta(seconds=30)


class _FakeEntry:
    args = [{"task_schedule": {}}]
    schedule = _FakeSchedule()
    max_runs = 1
    total_run_count = 1
    last_run_at = None

    @staticmethod
    def generate_key(app, task_id):
        return f"redbeat:{task_id}"

    @classmethod
    def from_key(cls, key, app):
        return cls()


class _FakeScheduler:
    @staticmethod
    def get_task_ids(key, generate_key=False):
        return ["exec-1"]

    @staticmethod
    def get_last_run_at(key, generate_key=False):
        return None

    @staticmethod
    def get_last_run_task_id(key, generate_key=False):
        return "exec-1"


def _install_fake_scheduler(monkeypatch, final: Dict[str, Any], *, running_polls: int = 0):
    """Celery/RedBeat stand-ins. The first ``running_polls`` AsyncResult
    lookups report STARTED, every later one SUCCESS with ``final``."""
    lookups = {"n": 0}

    class _FakeAsyncResult:
        traceback = None

        def __init__(self, task_id, app=None):
            lookups["n"] += 1
            self.state = "STARTED" if lookups["n"] <= running_polls else "SUCCESS"
            self.result = json.loads(json.dumps(final)) if self.state == "SUCCESS" else None

        def failed(self):
            return self.state == "FAILURE"

    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(server, "AvalokaEntry", _FakeEntry, raising=False)
    monkeypatch.setattr(server, "AvalokaScheduler", _FakeScheduler, raising=False)
    monkeypatch.setattr(server, "AsyncResult", _FakeAsyncResult, raising=False)
    monkeypatch.setattr(server, "sleep", _no_sleep, raising=False)


def _scheduled_final(*, swarm: bool) -> Dict[str, Any]:
    final: Dict[str, Any] = {
        "messages": [],
        "analysis_fidelity": "entire_dataset",
        "execution_result": {"status": "success"},
    }
    if swarm:
        final["avaloka_swarm"] = _SWARM_PLAN
        final["avaloka_swarm_message"] = "Dispatching a scout"
    return final


def _sync_execution_context(client, monkeypatch, user: str, *, swarm: bool):
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _RecordingGraph(swarm=swarm), raising=False)
    upload = _do_upload(client, user_id=user)
    out = _send(client, user, upload, "summarise the data")
    return upload, out["execution_context"]


def _sse_events(text: str) -> List[Dict[str, Any]]:
    events = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


@pytest.mark.parametrize("swarm", [False, True], ids=["plain", "swarm"])
def test_task_result_carries_same_execution_context_keys_as_sync_response(
    client: TestClient, monkeypatch, swarm
):
    user = f"wire-result-{swarm}"
    upload, sync_ctx = _sync_execution_context(client, monkeypatch, user, swarm=swarm)
    assert ("swarm" in sync_ctx) is swarm

    SESSIONS[upload["session_id"]]["tasks"] = ["sched-1"]
    _install_fake_scheduler(monkeypatch, _scheduled_final(swarm=swarm))
    headers = make_auth_headers(user)
    headers["X-Avaloka-Session"] = upload["session_id"]
    resp = client.get("/tasks/sched-1/result/0", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "SUCCESS"

    ctx = body["result"]["execution_context"]
    assert set(ctx) == set(sync_ctx)
    assert ctx["mode"] == "entire_dataset"
    assert ctx["mode_label"] == "Entire Dataset"
    if swarm:
        assert ctx["swarm"] == _SWARM_PLAN


@pytest.mark.parametrize("swarm", [False, True], ids=["plain", "swarm"])
def test_task_sse_result_carries_same_execution_context_keys_as_sync_response(
    client: TestClient, monkeypatch, swarm
):
    user = f"wire-sse-{swarm}"
    upload, sync_ctx = _sync_execution_context(client, monkeypatch, user, swarm=swarm)

    SESSIONS[upload["session_id"]]["tasks"] = ["sched-1"]
    # Running at request time, so /status streams through task_sse_iter.
    _install_fake_scheduler(monkeypatch, _scheduled_final(swarm=swarm), running_polls=2)
    headers = make_auth_headers(user)
    headers["X-Avaloka-Session"] = upload["session_id"]
    resp = client.get("/tasks/sched-1/status", headers=headers)
    assert resp.status_code == 200, resp.text

    events = _sse_events(resp.text)
    success = [e for e in events if e.get("status") == "SUCCESS"]
    assert events and events[0]["status"] == "STARTED"
    assert len(success) == 1, events
    ctx = success[0]["result"]["execution_context"]
    assert set(ctx) == set(sync_ctx)
    assert ctx["mode"] == "entire_dataset"
    if swarm:
        assert ctx["swarm"] == _SWARM_PLAN
