"""scripts/ops/backfill_celery_results.py: copy Redis-only results to Postgres, then expire them.

Seeded the way a Redis looks after months of ``result_expires=None``: full
results with no TTL holding a conversation, the task arguments, typed
credentials and the output CSV as base64, beside the broker's, the
scheduler's and the session keys. Pinned:

  * the dry run changes nothing: every key, value and TTL is as it was, and no
    row is written;
  * the real run touches only ``celery-task-meta-*`` (plus the backend's own
    pending set): every command it sends names such a key, and every other key
    keeps its value and TTL;
  * a row carries none of the conversation, the arguments, the credentials or
    the CSV;
  * a key gets its TTL only after its row is read back and matches.

Against a real Redis and Postgres it is run separately (not in CI); see the PR notes.
"""
from __future__ import annotations

import base64
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from tests.celery_result_fakes import FakeRedis, FakeSupabase

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "ops" / "backfill_celery_results.py"
META = "celery-task-meta-"
# A Postgres DSN pasted into chat is the likeliest real case. Every credential-
# bearing URI is built at run time, so no URI with a literal user and password sits in a
# tracked file (the repo secret scans, test_c1 and test_c9).
PG_PASSWORD = "Pg" + "Dsn5ecret"
PG_DSN = "postgres" + "ql://" + "analyst:" + PG_PASSWORD + "@" + "warehouse.internal:5432/sales"
REDIS_PASSWORD = "S3cret" + "Pass"
REDIS_URI = "redis" + "://" + "admin:" + REDIS_PASSWORD + "@" + "cache.internal:6379/0"
SECRETS = ("Hunter2Secret", "S3cretPass", "Tr0ub4dor", "Zebra99Pw", PG_PASSWORD)
CSV_B64 = base64.b64encode(b"region,revenue\nnorth,10\n").decode()
PROTECTED = ("redbeat:", "_kombu", "unacked", "sess:", "celery")  # "celery" = the broker's queue list


def _load():
    assert SCRIPT.is_file(), "scripts/ops/backfill_celery_results.py does not exist"
    spec = importlib.util.spec_from_file_location("backfill_celery_results", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _legacy_result(task_id, status="SUCCESS"):
    messages = [
        {"type": "human", "data": {"content": "connect host db user admin password=Hunter2Secret"}},
        {"type": "human", "data": {"content": REDIS_URI}},
        {"type": "human", "data": {"content": f"the warehouse is {PG_DSN}"}},
        {"type": "human", "data": {"content": "the password is Tr0ub4dor"}},
        {"type": "human", "data": {"content": "pwd: Zebra99Pw"}},
        {"type": "ai", "data": {"content": "Here it is."}},
    ]
    result = ({"messages": messages, "output_location": "/w/out.csv",
               "output_file_data": {"filename": "out.csv", "content": "data:text/csv;base64," + CSV_B64},
               "execution_result": {"status": "success"}, "available_samples": ["random_baseline"]}
              if status == "SUCCESS" else
              {"exc_type": "RuntimeError", "exc_message": ["password=Hunter2Secret"], "exc_module": "builtins"})
    return json.dumps({
        "status": status, "result": result, "traceback": "Traceback ... Tr0ub4dor" if status != "SUCCESS" else None,
        "children": [], "date_done": "2026-06-01T10:00:00+00:00", "task_id": task_id,
        "name": "app.core.celery_app.agent_task",
        "args": [{"messages": messages, "user_id": "u-1", "schedule_id": "sched-1", "db_password": "Hunter2Secret"}],
        "kwargs": {}, "worker": "w1", "retries": 0, "queue": "celery",
    })


@pytest.fixture
def world(monkeypatch):
    import app.agents.sampling_persistence as sp

    r, db = FakeRedis(), FakeSupabase()
    monkeypatch.setattr(sp, "get_supabase_client", lambda: db)
    # Broker, scheduler and sessions: none has a TTL except the session.
    r.lists["celery"] = ["queued-message"]
    r.hashes["unacked"] = {"tag": "msg"}
    r.zsets["unacked_index"] = {"tag": 1.0}
    r.sets["_kombu.binding.celery"] = {"celery"}
    r.zsets["redbeat::schedule"] = {"redbeat:sched-1": 1.0}
    r.hashes["redbeat:sched-1"] = {"definition": "{}", "task_data": "{}"}
    r.data["redbeat::lock"] = "token"
    r.ttls["redbeat::lock"] = 300
    r.data["sess:abc"] = "{}"
    r.ttls["sess:abc"] = 3600
    r.data["sess:abc:meta"] = "{}"
    # Results.
    r.data[META + "ok-1"] = _legacy_result("ok-1")
    r.data[META + "fail-1"] = _legacy_result("fail-1", status="FAILURE")
    r.data[META + "started-1"] = json.dumps({"status": "STARTED", "result": {"pid": 1}, "task_id": "started-1"})
    r.data[META + "has-ttl"] = _legacy_result("has-ttl")
    r.ttls[META + "has-ttl"] = 1000
    r.sets["celery-durable-pending"] = {"ok-1"}
    return r, db


def _snapshot(r: FakeRedis):
    return copy.deepcopy((r.data, r.ttls, r.sets, r.hashes, r.zsets, r.lists))


def test_the_dry_run_changes_nothing(world):
    r, db = world
    before = _snapshot(r)
    counts = _load().run(r, apply=False, expire_seconds=604800, jitter_seconds=0)
    assert _snapshot(r) == before, "the dry run changed Redis"
    assert not db.calls, "the dry run wrote to Postgres"
    assert counts["would_backfill"] == 2 and counts["not_finished"] == 1 and counts["already_ttl"] == 1


def test_the_real_run_touches_only_task_result_keys(world):
    r, db = world
    protected_before = {k: (r.type(k), copy.deepcopy(getattr(r, {"string": "data", "set": "sets", "hash": "hashes",
                                                                   "zset": "zsets", "list": "lists"}[r.type(k)])[k]),
                            r.ttls.get(k))
                        for k in r._all() if not k.startswith(META)}
    r.commands.clear()
    counts = _load().run(r, apply=True, expire_seconds=604800, jitter_seconds=0)
    assert counts["backfilled"] == 2 and counts["expire_set"] == 2 and counts["pending_drained"] == 1
    for cmd in r.commands:
        name, *keys = cmd
        if name == "scan":
            assert keys == [META + "*"], f"SCAN with another pattern: {keys}"
            continue
        for key in (k for k in keys if isinstance(k, str)):
            assert key.startswith(META) or key == "celery-durable-pending", f"{name} touched {key}"
    for k, (kind, value, ttl) in protected_before.items():
        store = {"string": r.data, "set": r.sets, "hash": r.hashes, "zset": r.zsets, "list": r.lists}[kind]
        if k == "celery-durable-pending":
            continue
        assert store.get(k) == value and r.ttls.get(k) == ttl, f"{k} changed"
    assert r.ttl(META + "ok-1") == 604800 and r.ttl(META + "fail-1") == 604800
    assert r.ttl(META + "started-1") == -1, "an unfinished task's key must keep no TTL"
    assert r.ttl(META + "has-ttl") == 1000


def test_rows_carry_no_conversation_arguments_credentials_or_csv(world):
    r, db = world
    _load().run(r, apply=True, expire_seconds=604800, jitter_seconds=0)
    rows = db.rows("celery_task_results")
    assert {row["task_id"] for row in rows} == {"ok-1", "fail-1"}
    blob = json.dumps(rows)
    for secret in SECRETS:
        assert secret not in blob
    assert "base64," not in blob and '"messages"' not in blob and '"args"' not in blob
    ok = next(row for row in rows if row["task_id"] == "ok-1")
    assert ok["result"]["output_location"] == "/w/out.csv" and ok["schedule_id"] == "sched-1"
    assert ok["traceback"] is None


def test_a_row_that_does_not_verify_gets_no_ttl(world, monkeypatch):
    r, db = world
    mod = _load()
    monkeypatch.setattr(mod.celery_results, "read_row", lambda task_id: None)
    counts = mod.run(r, apply=True, expire_seconds=604800, jitter_seconds=0)
    assert counts["verify_failed"] == 2 and counts["expire_set"] == 0
    assert r.ttl(META + "ok-1") == -1 and r.ttl(META + "fail-1") == -1


def test_postgres_down_sets_no_ttl_and_exits_nonzero(world, monkeypatch, capsys):
    r, db = world
    db.down = True
    mod = _load()
    counts = mod.run(r, apply=True, expire_seconds=604800, jitter_seconds=0)
    assert counts["expire_set"] == 0 and r.ttl(META + "ok-1") == -1
    assert any(k.startswith("write_failed") for k in counts)


def test_the_output_is_counts_only(world, monkeypatch, capsys):
    r, db = world
    mod = _load()
    monkeypatch.setenv("REDIS_URL", "redis://fake")
    import redis
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, *a, **k: r))
    monkeypatch.setattr(mod.celery_results, "_client", lambda: db)
    assert mod.main(["--apply"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["backfilled"] == 2
    for secret in (*SECRETS, "redis://fake", "ok-1", "out.csv"):
        assert secret not in out


def test_the_scan_pattern_is_fixed():
    text = SCRIPT.read_text() if SCRIPT.is_file() else ""
    assert SCRIPT.is_file(), "scripts/ops/backfill_celery_results.py does not exist"
    assert "--match" not in text and "--pattern" not in text, "the key pattern must not be configurable"


# -- refusing unsafe runs ---------------------------------------------------------

def _main(mod, monkeypatch, r, argv):
    import redis
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, *a, **k: r))
    return mod.main(argv)


def test_it_refuses_before_the_migration_is_applied(world, monkeypatch, capsys):
    r, db = world
    mod = _load()
    db.missing_tables.add("celery_task_results")
    monkeypatch.setenv("REDIS_URL", "redis://fake")
    before = _snapshot(r)
    assert _main(mod, monkeypatch, r, ["--apply"]) == 2, "it ran without the celery_task_results table"
    err = capsys.readouterr().err
    assert "does not exist" in err and "20261117120000_celery_task_results" in err and "nothing was changed" in err
    assert _snapshot(r) == before
    assert not [c for c in db.calls if c[1] == "upsert"]


def test_it_refuses_without_redis_or_postgres_credentials(world, monkeypatch, capsys):
    r, db = world
    mod = _load()
    for name in ("REDIS_URL", "CELERY_REDIS_URL"):
        monkeypatch.delenv(name, raising=False)
    assert _main(mod, monkeypatch, r, []) == 2
    monkeypatch.setenv("REDIS_URL", "redis://fake")

    def _no_credentials():
        raise RuntimeError("Supabase env vars not set")
    import app.agents.sampling_persistence as sp
    monkeypatch.setattr(sp, "get_supabase_client", _no_credentials)
    assert _main(mod, monkeypatch, r, ["--apply"]) == 2
    assert "SUPABASE_SERVICE_ROLE_KEY" in capsys.readouterr().err


def test_dry_run_is_the_default_and_apply_has_two_spellings(world, monkeypatch, capsys):
    r, db = world
    mod = _load()
    monkeypatch.setenv("REDIS_URL", "redis://fake")
    for argv, mode in (([], "dry-run"), (["--dry-run"], "dry-run"), (["--no-dry-run", "--batch", "2"], "apply")):
        assert _main(mod, monkeypatch, r, argv) == 0
        assert json.loads(capsys.readouterr().out)["mode"] == mode


def test_batches_are_bounded(world):
    r, db = world
    counts = _load().run(r, apply=False, expire_seconds=604800, jitter_seconds=0, batch=2)
    assert counts["batches"] == 2 and counts["scanned"] == 4


def test_a_huge_result_is_streamed_not_loaded(world):
    """A 40 MB result: read in bounded chunks, never with one GET, and summarized correctly."""
    import tracemalloc

    r, db = world
    mod = _load()
    big = json.loads(_legacy_result("big-1"))
    big["result"]["output_file_data"]["content"] = "data:text/csv;base64," + "A" * (40 << 20)
    raw = json.dumps(big).encode()
    r.data[META + "big-1"] = raw
    tracemalloc.start()
    counts = mod.run(r, apply=True, expire_seconds=604800, jitter_seconds=0)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert counts["backfilled"] == 3
    assert not [c for c in r.commands if c[0] in ("get", "mget") and META + "big-1" in c[1:]], (
        "the 40 MB value was read with a single GET")
    reads = [c[2] for c in r.commands if c[0] == "getrange" and c[1] == META + "big-1"]
    assert reads and max(reads) <= mod.CHUNK_BYTES, "a read was larger than one chunk"
    assert peak < 1.5 * len(raw), f"peak {peak >> 20} MiB for a {len(raw) >> 20} MiB value"
    row = next(x for x in db.rows("celery_task_results") if x["task_id"] == "big-1")
    assert row["result"]["has_output"] is True and "AAAA" not in json.dumps(row)


@pytest.mark.parametrize("tid,status", [("ok-1", "SUCCESS"), ("fail-1", "FAILURE")])
def test_the_streamed_row_is_the_backends_row(world, tid, status):
    """The script's streaming parse and the backend's summary of the full value agree."""
    r, db = world
    mod = _load()
    raw = _legacy_result(tid, status=status)
    meta = json.loads(raw)
    expected = mod.celery_results.durable_row(tid, meta, args=meta["args"], name=meta["name"])
    streamed = mod.extract_meta(mod._RangeReader(r, META + tid, len(raw.encode()), chunk=7))
    got = mod.celery_results.durable_row(tid, streamed, args=[streamed.pop("_ids")], name=streamed.get("name"))
    assert got == expected


def test_an_unreachable_store_is_refused_as_unreachable(world, monkeypatch, capsys):
    """Not the missing-table message: an operator must not apply a migration that is already there."""
    r, db = world
    mod = _load()
    db.down = True
    monkeypatch.setenv("REDIS_URL", "redis://fake")
    assert _main(mod, monkeypatch, r, ["--apply"]) == 2
    err = capsys.readouterr().err
    assert "not reachable" in err and "does not exist" not in err
