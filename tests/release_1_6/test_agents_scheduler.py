"""1.6 release suite: scheduler (plan cases AG-16, AG-17).

Hermetic stand-ins, no Redis server or Celery worker:
- _FakeRedis implements only the RedBeat operations Avaloka's scheduler uses
  (pipeline hset/hsetnx/hget/zadd/execute, hget, zrange, zrank), installed as the
  app's cached RedBeat client, so AvalokaEntry.save/from_key run unmodified.
- _FakeSupabase implements the scheduled_runs table queries the worker writes
  and the API reads.
Real Celery execution of a run needs a worker and stays manual.
See docs/test-reports/1.6-traceability.csv.
"""
from __future__ import annotations

import base64
import time

import pytest
from langchain_core.messages import HumanMessage

# The API client fixture (fake session cache/blob store, test JWT secret).
from tests.server_harness import SESSIONS, client, make_auth_headers  # noqa: F401


class _FakeRedis:
    def __init__(self):
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    # direct commands
    def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value
        return 1

    def hsetnx(self, key, field, value):
        h = self.hashes.setdefault(key, {})
        if field in h:
            return 0
        h[field] = value
        return 1

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)
        return len(mapping)

    def zrange(self, key, start, end):
        items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1])
        keys = [k for k, _ in items]
        return keys[start:] if end == -1 else keys[start:end + 1]

    def zrank(self, key, member):
        keys = self.zrange(key, 0, -1)
        return keys.index(member) if member in keys else None

    def pipeline(self):
        return _FakePipeline(self)


class _FakePipeline:
    def __init__(self, redis):
        self._redis, self._ops = redis, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        def _queue(*args, **kwargs):
            self._ops.append((name, args, kwargs))
            return self
        return _queue

    def execute(self):
        ops, self._ops = self._ops, []
        return [getattr(self._redis, name)(*a, **k) for name, a, k in ops]


class _Result:
    def __init__(self, data, count=None):
        self.data, self.count = data, count


class _Query:
    def __init__(self, rows, op, payload=None, count=None):
        self._rows, self._op, self._payload, self._count = rows, op, payload, count
        self._filters, self._order = [], None

    def eq(self, col, val):
        self._filters.append((col, val))
        return self

    def order(self, col, desc=False):
        self._order = (col, desc)
        return self

    def _match(self):
        return [r for r in self._rows if all(r.get(c) == v for c, v in self._filters)]

    def execute(self):
        if self._op == "upsert":
            self._rows[:] = [r for r in self._rows if r.get("run_id") != self._payload.get("run_id")]
            self._rows.append(dict(self._payload))
            return _Result([dict(self._payload)])
        if self._op == "update":
            for row in self._match():
                row.update(self._payload)
            return _Result(self._match())
        rows = [dict(r) for r in self._match()]
        if self._order:
            col, desc = self._order
            rows.sort(key=lambda r: r.get(col) or "", reverse=desc)
        return _Result(rows, count=len(rows) if self._count else None)


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows

    def select(self, _cols="*", count=None):
        return _Query(self._rows, "select", count=count)

    def upsert(self, row):
        return _Query(self._rows, "upsert", payload=row)

    def update(self, values):
        return _Query(self._rows, "update", payload=values)


class _FakeSupabase:
    def __init__(self):
        self.tables: dict[str, list] = {}

    def table(self, name):
        return _FakeTable(self.tables.setdefault(name, []))


@pytest.fixture()
def scheduler_backends(monkeypatch):
    """Install the fake RedBeat client and fake Supabase on every reader/writer."""
    import redbeat.schedulers as redbeat_schedulers

    import app.agents.scheduler as scheduler
    import app.api.server as server
    import app.core.celery_app as celery_mod

    redis, supa = _FakeRedis(), _FakeSupabase()
    redbeat_schedulers.ensure_conf(celery_mod.celery_app)
    monkeypatch.setattr(celery_mod.celery_app, redbeat_schedulers.REDBEAT_REDIS_KEY, redis, raising=False)
    monkeypatch.setattr(scheduler, "is_celery_worker_running", lambda: True)
    monkeypatch.setattr(celery_mod, "_supabase", lambda: supa)
    monkeypatch.setattr(server, "get_supabase_client", lambda: supa)
    return redis, supa


def _schedule_from_prompt(prompt: str, session_id: str, user_id: str) -> str:
    """Prompt -> planner cadence parser -> scheduler node; returns the task id."""
    import app.agents.planner as planner
    from app.agents.scheduler import task_scheduler_node

    schedule, concrete = planner._parse_task_schedule_from_text(prompt.lower())
    assert concrete, f"cadence not parsed from {prompt!r}"
    out = task_scheduler_node({
        "messages": [HumanMessage(content=prompt)],
        "session_id": session_id,
        "user_id": user_id,
        "task_schedule": schedule,
    })
    task_id = (out.get("task_info") or {}).get("task_id")
    assert task_id, out.get("messages")
    assert task_id in (out.get("task_list") or [])
    return task_id


def test_ag16_schedule_prompt_creates_a_task_listed_by_the_tasks_api(scheduler_backends, client):
    """AG-16: a scheduling prompt creates a task, and GET /tasks lists it with
    the requested schedule."""
    SESSIONS["sess-ag16"] = {"session_id": "sess-ag16", "user_id": "user-1"}
    task_id = _schedule_from_prompt("Every Monday 9am recompute fraud rate by ProductCD",
                                    "sess-ag16", "user-1")
    SESSIONS["sess-ag16"]["tasks"] = [task_id]

    headers = {**make_auth_headers("user-1"), "X-Avaloka-Session": "sess-ag16"}
    resp = client.get("/tasks", headers=headers)

    assert resp.status_code == 200, resp.text
    tasks = {t["id"]: t for t in resp.json()}
    assert task_id in tasks, tasks
    sched = tasks[task_id]["schedule"]
    assert (str(sched["day_of_week"]), str(sched["hour"]), str(sched["minute"])) == ("1", "9", "0")
    assert tasks[task_id]["max_runs"] == -1   # a weekly cron repeats until cancelled


def _csv_payload(rows: list[dict]) -> dict:
    header = ",".join(rows[0])
    body = "\n".join(",".join(str(v) for v in r.values()) for r in rows)
    return {"content": "data:text/csv;base64," + base64.b64encode(f"{header}\n{body}\n".encode()).decode()}


def test_ag17_each_run_is_retained_with_status_and_output(scheduler_backends, client):
    """AG-17: every run of a scheduled job is retained, numbered, with its status
    and its own output, and GET /tasks/{id}/runs returns them newest first."""
    import app.core.celery_app as celery_mod

    SESSIONS["sess-ag17"] = {"session_id": "sess-ag17", "user_id": "user-1"}
    task_id = _schedule_from_prompt("recompute totals daily at 6am", "sess-ag17", "user-1")
    SESSIONS["sess-ag17"]["tasks"] = [task_id]

    run_state = {"session_id": "sess-ag17", "user_id": "user-1",
                 "task_schedule": {"task_type": "execute"}}
    outputs = {"exec-1": [{"total": "10"}], "exec-2": [{"total": "12"}]}
    for exec_id, rows in outputs.items():
        t0 = time.perf_counter()
        celery_mod._record_run_start(task_id, exec_id, run_state)
        celery_mod._record_run_finish(task_id, exec_id, {
            "messages": [],
            "execution_result": {"status": "success"},
            "output_file_data": _csv_payload(rows),
        }, t0, scheduled_state=run_state)

    headers = {**make_auth_headers("user-1"), "X-Avaloka-Session": "sess-ag17"}
    resp = client.get(f"/tasks/{task_id}/runs", headers=headers)

    assert resp.status_code == 200, resp.text
    runs = resp.json()
    assert [r["run_id"] for r in runs] == ["exec-2", "exec-1"]       # newest first
    assert [r["run_number"] for r in runs] == [2, 1]
    for run in runs:
        assert run["status"] == "success"
        assert run["finished_at"]
        assert run["result"]["output_json"] == outputs[run["run_id"]]
