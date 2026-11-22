"""In-memory Redis and PostgREST stand-ins for the Celery result-store tests.

Not a test module. ``FakeRedis`` implements the commands the Celery Redis
result backend, RedBeat-free readers and the backfill script use, with TTLs as
plain integers (no clock). ``FakeSupabase`` implements the PostgREST calls of
app/core/celery_results.py and celery_app's scheduled_runs reads.
"""
from __future__ import annotations

import copy
import fnmatch
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional


class FakeRedisDown(Exception):
    pass


def _k(key):
    return key.decode() if isinstance(key, bytes) else key


class FakeRedis:
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.ttls: Dict[str, int] = {}
        self.sets: Dict[str, set] = {}
        self.hashes: Dict[str, dict] = {}
        self.zsets: Dict[str, dict] = {}
        self.lists: Dict[str, list] = {}
        self.down = False
        self.commands: List[tuple] = []

    def _check(self, *cmd):
        self.commands.append(cmd)
        if self.down:
            import redis
            raise redis.ConnectionError("fake redis is down")

    def _all(self):
        return [*self.data, *self.sets, *self.hashes, *self.zsets, *self.lists]

    # strings
    def get(self, key):
        key = _k(key)
        self._check("get", key)
        return self.data.get(key)

    def mget(self, keys):
        keys = [_k(k) for k in keys]
        self._check("mget", *keys)
        return [self.data.get(k) for k in keys]

    def set(self, key, value, ex=None, **_kw):
        key = _k(key)
        self._check("set", key)
        self.data[key] = value
        if ex:
            self.ttls[key] = int(ex)
        else:
            self.ttls.pop(key, None)
        return True

    def strlen(self, key):
        key = _k(key)
        self._check("strlen", key)
        v = self.data.get(key)
        return len(v.encode() if isinstance(v, str) else v) if v is not None else 0

    def getrange(self, key, start, end):
        key = _k(key)
        self._check("getrange", key, end - start + 1)
        v = self.data.get(key)
        if v is None:
            return b""
        b = v.encode() if isinstance(v, str) else v
        return b[start:end + 1]

    def ping(self):
        self._check("ping")
        return True

    def setex(self, key, ttl, value):
        key = _k(key)
        self._check("setex", key)
        self.data[key] = value
        self.ttls[key] = int(ttl)
        return True

    def delete(self, *keys):
        keys = [_k(k) for k in keys]
        self._check("delete", *keys)
        n = 0
        for k in keys:
            for store in (self.data, self.sets, self.hashes, self.zsets, self.lists):
                if k in store:
                    del store[k]
                    n += 1
            self.ttls.pop(k, None)
        return n

    def publish(self, channel, message):
        self._check("publish", _k(channel))
        return 0

    def ttl(self, key):
        key = _k(key)
        self._check("ttl", key)
        if key not in self._all():
            return -2
        return self.ttls.get(key, -1)

    def expire(self, key, ttl, nx=False, **_kw):
        key = _k(key)
        self._check("expire", key)
        if key not in self._all():
            return False
        if nx and key in self.ttls:
            return False
        self.ttls[key] = int(ttl)
        return True

    def persist(self, key):
        key = _k(key)
        self._check("persist", key)
        return bool(self.ttls.pop(key, None))

    # sets
    def sadd(self, key, *members):
        key = _k(key)
        members = [_k(m) for m in members]
        self._check("sadd", key)
        s = self.sets.setdefault(key, set())
        before = len(s)
        s.update(members)
        return len(s) - before

    def srem(self, key, *members):
        key = _k(key)
        members = [_k(m) for m in members]
        self._check("srem", key)
        s = self.sets.get(key, set())
        n = len(s & set(members))
        s.difference_update(members)
        if not s:
            self.sets.pop(key, None)
        return n

    def smembers(self, key):
        key = _k(key)
        self._check("smembers", key)
        return set(self.sets.get(key, set()))

    # scanning
    def scan_iter(self, match=None, count=None, **_kw):
        self._check("scan", match)
        for key in list(self._all()):
            if match is None or fnmatch.fnmatchcase(key, match):
                yield key

    def type(self, key):
        key = _k(key)
        for name, store in (("string", self.data), ("set", self.sets), ("hash", self.hashes),
                            ("zset", self.zsets), ("list", self.lists)):
            if key in store:
                return name
        return "none"

    def pipeline(self, *_a, **_kw):
        return _FakePipe(self)


class _FakePipe:
    def __init__(self, r: FakeRedis):
        self.r = r
        self.calls: List[Callable[[], Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        target = getattr(self.r, name)

        def queue(*a, **kw):
            self.calls.append(lambda: target(*a, **kw))
            return self
        return queue

    def execute(self):
        out = [c() for c in self.calls]
        self.calls = []
        return out


class _Res(SimpleNamespace):
    pass


class _Query:
    def __init__(self, db: "FakeSupabase", table: str):
        self.db, self.table, self.op, self.payload = db, table, "select", None
        self.filters: List[tuple] = []
        self.n: Optional[int] = None

    def select(self, *_cols, **_kw):
        self.op = "select"
        return self

    def upsert(self, row, on_conflict=None, **_kw):
        self.op, self.payload = "upsert", row
        return self

    def update(self, row):
        self.op, self.payload = "update", row
        return self

    def insert(self, row):
        self.op, self.payload = "insert", row
        return self

    def eq(self, col, val):
        self.filters.append((col, val))
        return self

    def order(self, *_a, **_kw):
        return self

    def limit(self, n):
        self.n = n
        return self

    def execute(self):
        self.db.calls.append((self.table, self.op, copy.deepcopy(self.payload), list(self.filters)))
        if self.table in self.db.missing_tables:
            raise Exception(f'relation "public.{self.table}" does not exist')
        if self.db.down:
            raise ConnectionError("fake postgres is down")
        rows = self.db.tables.setdefault(self.table, {})
        pk = self.db.PK.get(self.table, "id")
        if self.op in ("upsert", "insert"):
            row = copy.deepcopy(self.payload)
            rows[row[pk]] = {**rows.get(row[pk], {}), **row}
            return _Res(data=[row])
        match = [r for r in rows.values() if all(r.get(c) == v for c, v in self.filters)]
        if self.op == "update":
            for r in match:
                r.update(copy.deepcopy(self.payload))
            return _Res(data=match)
        match = [copy.deepcopy(r) for r in match]
        return _Res(data=match[: self.n] if self.n else match)


class _Rpc:
    def __init__(self, db, name, params):
        self.db, self.name, self.params = db, name, params

    def execute(self):
        self.db.calls.append(("rpc", self.name, dict(self.params), []))
        if self.db.down:
            raise ConnectionError("fake postgres is down")
        return _Res(data=self.db.rpcs[self.name](self.db, **self.params))


def _purge(db, cutoff, batch_size):
    """The SQL function's rule, in Python (the real one is tested on Postgres)."""
    cut = datetime.fromisoformat(cutoff)
    rows = db.tables.setdefault("celery_task_results", {})
    doomed = sorted(
        (r for r in rows.values()
         if r.get("status") in ("SUCCESS", "FAILURE", "REVOKED") and r.get("date_done")
         and datetime.fromisoformat(r["date_done"]) < cut),
        key=lambda r: r["date_done"])[:batch_size]
    for r in doomed:
        del rows[r["task_id"]]
    return len(doomed)


class FakeSupabase:
    PK = {"celery_task_results": "task_id", "scheduled_runs": "run_id"}

    def __init__(self):
        self.tables: Dict[str, Dict[str, dict]] = {}
        self.calls: List[tuple] = []
        self.down = False
        self.missing_tables: set = set()
        self.rpcs = {"purge_celery_task_results": _purge}

    def table(self, name):
        return _Query(self, name)

    def rpc(self, name, params):
        return _Rpc(self, name, params)

    def rows(self, table) -> List[dict]:
        return list(self.tables.get(table, {}).values())


def iso_days_ago(days: float, now: Optional[datetime] = None) -> str:
    return ((now or datetime.now(timezone.utc)) - timedelta(days=days)).isoformat()
