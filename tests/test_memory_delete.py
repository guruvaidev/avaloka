"""
Memory can be deleted: by thread, by user and by id, in Milvus (L4 insights)
and Chroma (L2 signatures); deleting a thread, a dataset's threads or an
account removes what they learned (1.6 hotfix, 1.6-CAPABILITY-GAPS item 5a).

Before this, neither client had a delete method and DELETE /threads/{id} left
the thread's memory in place.

Milvus runs on the real MilvusClientImpl over an in-memory stand-in for the
pymilvus Collection that evaluates the boolean expressions Milvus would (and
refuses any other, so the client cannot quietly send something it does not
understand). Chroma runs on the real ChromaClientImpl twice: over an in-memory
chromadb collection, and on its in-process fallback store.

Each test asserts on what is left in the stores, so on the code before the fix
it fails on an assertion (a missing method is reported as one too).
"""

import json
import re
import sys
import types
import uuid

import pytest

try:  # pragma: no cover
    import cachetools  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _ct = types.ModuleType("cachetools")

    class _TTLCache(dict):
        def __init__(self, maxsize=None, ttl=None):
            super().__init__()

    _ct.TTLCache = _TTLCache
    sys.modules["cachetools"] = _ct

import app.services.memory_plane as mp
from app.services.db.chroma_client import ChromaClientImpl
from app.services.db.milvus_client import MilvusClientImpl
from app.services.db.redis_client import RedisClientImpl
from app.services.memory_plane import MemoryOrchestrator

try:  # tests/ as a package, or tests/ on sys.path (pytest rootdir insertion)
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
        SESSIONS, THREAD_TO_SESSION, client, make_auth_headers, _run,
    )
except ImportError:
    from server_harness import (  # noqa: F401
        SESSIONS, THREAD_TO_SESSION, client, make_auth_headers, _run,
    )

U1 = "user-1"
U2 = "user-2"
# A user whose id is a prefix of U1's, and one that a LIKE "user_1%" pattern
# matches when "_" is a wildcard: neither may lose a row when U1 is deleted.
U1_PREFIX_TWIN = "user-10"
U1_LIKE_TWIN = "userX1"


def _k_thread(user, thread):
    return f"{user}#thread:{thread}"


def _k_pref(user):
    return f"{user}#pref"


def _k_dataset(user, ds):
    return f"{user}#dataset:{ds}"


def _method(obj, name):
    fn = getattr(obj, name, None)
    assert callable(fn), f"{type(obj).__name__} has no {name}(): memory cannot be deleted"
    return fn


# ── Milvus: an in-memory Collection ─────────────────────────────────────────


class FakeMilvusCollection:
    """The part of pymilvus.Collection the client uses, over a list of rows.

    Expressions are evaluated as Milvus does for the forms the client sends --
    `field in [...]`, `field == "..."`, `field like "..."` ("%" any run, "_"
    any one character, the broader reading) -- joined by "&&". Any other form
    raises, so a test cannot pass on an expression Milvus would reject."""

    _CLAUSE = re.compile(r'^\s*(id|session_id)\s+(in|==|like)\s+(.+?)\s*$', re.S)

    def __init__(self):
        self.rows = []
        self._next_id = 1

    def insert(self, data, partition_name=None):
        vector, ts, session_id, nb, role, cell, ref, content = (col[0] for col in data)
        self.rows.append({"id": self._next_id, "session_id": session_id, "content": content,
                          "agent_role": role, "timestamp": ts})
        self._next_id += 1

    def flush(self):
        return None

    def _match(self, expr):
        clauses = [c for c in expr.split("&&")]

        def test(row):
            for clause in clauses:
                m = self._CLAUSE.match(clause)
                if not m:
                    raise ValueError(f"unsupported Milvus expression: {expr!r}")
                field, op, literal = m.groups()
                value = row[field]
                if op == "in":
                    if not (literal.startswith("[") and literal.endswith("]")):
                        raise ValueError(f"unsupported Milvus expression: {expr!r}")
                    if value not in json.loads(literal):
                        return False
                elif op == "==":
                    if value != json.loads(literal):
                        return False
                else:
                    pattern = json.loads(literal)
                    rx = "".join(".*" if ch == "%" else "." if ch == "_" else re.escape(ch)
                                 for ch in pattern)
                    if not re.fullmatch(rx, str(value), re.S):
                        return False
            return True

        return test

    def query(self, expr, output_fields=None, limit=None, **_):
        test = self._match(expr)
        hits = [dict(r) for r in self.rows if test(r)]
        return hits[:limit] if limit else hits

    def delete(self, expr, **_):
        test = self._match(expr)
        self.rows = [r for r in self.rows if not test(r)]

    def search(self, data, anns_field, param, limit, expr, partition_names=None, output_fields=None):
        test = self._match(expr.replace("'", '"'))
        hits = [types.SimpleNamespace(id=r["id"], distance=0.0, entity=r)
                for r in self.rows if test(r)][:limit]
        return [hits]


def _milvus():
    client = MilvusClientImpl()
    client._collection = FakeMilvusCollection()
    return client


def _milvus_keys(client):
    return sorted(r["session_id"] for r in client._collection.rows)


def _seed_milvus(client):
    for key in (_k_thread(U1, "a"), _k_thread(U1, "a"), _k_thread(U1, "b"), _k_pref(U1),
                _k_dataset(U1, "patients"), U1, "sess-u1-legacy",
                _k_thread(U2, "a"), _k_pref(U2), U2,
                _k_pref(U1_PREFIX_TWIN), _k_thread(U1_LIKE_TWIN, "a")):
        client.insert_insight(key, f"insight under {key}", "memory_llm")


def test_milvus_delete_by_thread_removes_that_thread_only():
    client = _milvus()
    _seed_milvus(client)
    removed = _method(client, "delete_thread")(U1, "a")
    keys = _milvus_keys(client)
    assert _k_thread(U1, "a") not in keys
    assert removed == 2
    # Another thread of the user, the user's preferences, and another user's
    # thread with the same id are untouched.
    for kept in (_k_thread(U1, "b"), _k_pref(U1), _k_thread(U2, "a")):
        assert kept in keys


def test_milvus_delete_by_user_removes_every_row_of_the_user_and_no_other():
    client = _milvus()
    _seed_milvus(client)
    _method(client, "delete_user")(U1, ["sess-u1-legacy"])
    keys = _milvus_keys(client)
    assert not [k for k in keys if k == U1 or k.startswith(U1 + "#") or k == "sess-u1-legacy"], keys
    assert keys == sorted([_k_thread(U2, "a"), _k_pref(U2), U2,
                           _k_pref(U1_PREFIX_TWIN), _k_thread(U1_LIKE_TWIN, "a")])


def test_milvus_delete_by_id_removes_only_the_users_own_rows():
    client = _milvus()
    _seed_milvus(client)
    own = next(r["id"] for r in client._collection.rows if r["session_id"] == _k_pref(U1))
    other = next(r["id"] for r in client._collection.rows if r["session_id"] == _k_pref(U2))
    removed = _method(client, "delete_by_ids")([own, other], U1)
    keys = _milvus_keys(client)
    assert removed == 1
    assert _k_pref(U1) not in keys
    assert _k_pref(U2) in keys, "a row of another user was deleted by id"


def test_milvus_delete_raises_when_milvus_is_not_connected():
    """A delete that silently did nothing would leave memory behind."""
    client = MilvusClientImpl()
    client._collection = None
    delete_thread = _method(client, "delete_thread")
    with pytest.raises(RuntimeError):
        delete_thread(U1, "a")


# ── Chroma: in-memory chromadb and the in-process fallback ──────────────────


def _chroma_inmemory():
    chromadb = pytest.importorskip("chromadb")
    from chromadb.api.types import EmbeddingFunction

    class _ConstantEmbedding(EmbeddingFunction):
        """No model download: every document gets the same small vector."""

        def __init__(self):
            pass

        def __call__(self, input):
            return [[0.1, 0.2, 0.3] for _ in input]

        @staticmethod
        def name():
            return "avaloka-test-constant"

        def get_config(self):
            return {}

        @staticmethod
        def build_from_config(config):
            return _ConstantEmbedding()

    client = ChromaClientImpl()
    client.client = chromadb.EphemeralClient()
    client.collection = client.client.get_or_create_collection(
        name=f"t{uuid.uuid4().hex[:12]}", embedding_function=_ConstantEmbedding())
    client.use_mock = False
    return client


def _chroma_fallback():
    client = ChromaClientImpl()
    client.use_mock = True
    client.collection = None
    return client


@pytest.fixture(params=["chromadb-in-memory", "fallback"])
def chroma(request):
    return _chroma_inmemory() if request.param == "chromadb-in-memory" else _chroma_fallback()


def _seed_chroma(client):
    for key in (_k_thread(U1, "a"), _k_thread(U1, "a"), _k_thread(U1, "b"), _k_pref(U1), U1,
                "sess-u1-legacy", _k_thread(U2, "a"), _k_pref(U2),
                _k_pref(U1_PREFIX_TWIN), _k_thread(U1_LIKE_TWIN, "a")):
        client.update_user_signature(key, f"signature under {key}")


def _chroma_ids(client, key):
    if client.use_mock or not client.collection:
        return list((client._fallback_namespaces.get(key) or {}).get("ids") or [])
    got = client.collection.get(where={"user_id": {"$eq": key}}, include=[])
    return list(got.get("ids") or [])


def test_chroma_delete_by_thread_removes_that_thread_only(chroma):
    _seed_chroma(chroma)
    removed = _method(chroma, "delete_thread")(U1, "a")
    assert chroma.get_user_signature(_k_thread(U1, "a")) is None
    assert removed == 2
    for kept in (_k_thread(U1, "b"), _k_pref(U1), _k_thread(U2, "a")):
        assert chroma.get_user_signature(kept), kept


def test_chroma_delete_by_user_removes_every_signature_of_the_user_and_no_other(chroma):
    _seed_chroma(chroma)
    _method(chroma, "delete_user")(U1, ["sess-u1-legacy"])
    for gone in (_k_thread(U1, "a"), _k_thread(U1, "b"), _k_pref(U1), U1, "sess-u1-legacy"):
        assert chroma.get_user_signature(gone) is None, gone
    for kept in (_k_thread(U2, "a"), _k_pref(U2), _k_pref(U1_PREFIX_TWIN),
                 _k_thread(U1_LIKE_TWIN, "a")):
        assert chroma.get_user_signature(kept), kept


def test_chroma_delete_by_id_removes_only_the_users_own_signatures(chroma):
    _seed_chroma(chroma)
    own = _chroma_ids(chroma, _k_pref(U1))
    other = _chroma_ids(chroma, _k_pref(U2))
    assert own and other, "precondition: seeded signatures have ids"
    removed = _method(chroma, "delete_by_ids")(own + other, U1)
    assert removed == 1
    assert chroma.get_user_signature(_k_pref(U1)) is None
    assert chroma.get_user_signature(_k_pref(U2)), "a signature of another user was deleted by id"


# ── Through the memory plane and the API routes ─────────────────────────────


@pytest.fixture
def stores(monkeypatch):
    """The process-wide orchestrator the routes use, on in-memory stores."""
    milvus, chroma = _milvus(), _chroma_fallback()
    redis = RedisClientImpl()
    redis.use_mock = True
    redis.client = None
    orch = mp._orchestrator
    monkeypatch.setattr(orch, "milvus", milvus)
    monkeypatch.setattr(orch, "chroma", chroma)
    monkeypatch.setattr(orch, "redis", redis)
    for name in ("milvus", "chroma", "redis"):
        monkeypatch.setitem(orch._store_ready, name, True)
    MemoryOrchestrator.clear_all_sessions()
    yield types.SimpleNamespace(milvus=milvus, chroma=chroma, redis=redis, orch=orch)
    MemoryOrchestrator.clear_all_sessions()


def _seed_thread_memory(stores, user, thread, session=None):
    stores.milvus.insert_insight(_k_thread(user, thread), f"task hint of {thread}", "memory_llm")
    stores.chroma.update_user_signature(_k_thread(user, thread), f"Return what {thread} asked")
    stores.redis.set_json(f"ctxmem:{thread}:hints", [f"session hint of {thread}"])
    stores.redis.set_json(f"ctxmem:{thread}:prefs", [f"explicit preference of {thread}"])
    with MemoryOrchestrator._store_lock:
        MemoryOrchestrator._session_hints_store[thread] = [f"session hint of {thread}"]
        MemoryOrchestrator._session_preferences_store[thread] = [f"explicit preference of {thread}"]


def _thread_memory_left(stores, user, thread):
    left = []
    if _k_thread(user, thread) in _milvus_keys(stores.milvus):
        left.append("milvus")
    if stores.chroma.get_user_signature(_k_thread(user, thread)):
        left.append("chroma")
    if stores.redis.get_json(f"ctxmem:{thread}:hints") or stores.redis.get_json(f"ctxmem:{thread}:prefs"):
        left.append("redis")
    if (MemoryOrchestrator._session_hints_store.get(thread)
            or MemoryOrchestrator._session_preferences_store.get(thread)):
        left.append("in-process")
    return left


def test_deleting_a_thread_deletes_its_memory(client, stores):
    SESSIONS["sid-1"] = {"user_id": U1, "dataset_id": "ds-1"}
    THREAD_TO_SESSION["thread-a"] = "sid-1"
    THREAD_TO_SESSION["thread-b"] = "sid-1"
    _seed_thread_memory(stores, U1, "thread-a")
    _seed_thread_memory(stores, U1, "thread-b")
    stores.milvus.insert_insight(_k_pref(U1), "User prefers metric units", "memory_llm")

    resp = client.delete("/threads/thread-a", headers=make_auth_headers(U1))
    assert resp.status_code == 204, resp.text

    assert _thread_memory_left(stores, U1, "thread-a") == [], "the deleted thread's memory is still stored"
    # The user's other thread and their general preferences stay.
    assert _thread_memory_left(stores, U1, "thread-b") == ["milvus", "chroma", "redis", "in-process"]
    assert _k_pref(U1) in _milvus_keys(stores.milvus)


def test_deleting_a_dataset_deletes_the_memory_of_its_threads_and_its_facts(client, stores):
    SESSIONS["sid-1"] = {"user_id": U1, "dataset_id": "ds-1"}
    import app.services.session_service as session_service
    # The route reads the session blobs from the cache, not the session fakes.
    _run(session_service.cache.set("sess:sid-1", json.dumps(SESSIONS["sid-1"])))
    _run(session_service.cache.sadd(f"user:{U1}:sids", "sid-1"))
    _run(session_service.cache.sadd("sess:sid-1:threads", "thread-a"))
    _seed_thread_memory(stores, U1, "thread-a")
    stores.milvus.insert_insight(_k_dataset(U1, "ds-1"), "Column 'stroke' is 1 when ...", "memory_llm")
    stores.milvus.insert_insight(_k_dataset(U1, "ds-2"), "Column 'ward' is ...", "memory_llm")

    resp = client.delete("/datasets/ds-1", headers=make_auth_headers(U1))
    assert resp.status_code == 204, resp.text

    assert _thread_memory_left(stores, U1, "thread-a") == []
    keys = _milvus_keys(stores.milvus)
    assert _k_dataset(U1, "ds-1") not in keys, "the deleted dataset's learned facts are still stored"
    assert _k_dataset(U1, "ds-2") in keys


def test_deleting_an_account_deletes_all_of_its_memory(client, stores):
    import app.api.server as server
    import app.services.session_service as session_service

    forget = _method(server, "forget_account_memory")
    _run(session_service.cache.sadd(f"user:{U1}:sids", "sid-1"))
    _run(session_service.cache.sadd("sess:sid-1:threads", "thread-a"))
    _seed_thread_memory(stores, U1, "thread-a")
    _seed_thread_memory(stores, U2, "thread-z")
    for key in (_k_pref(U1), _k_dataset(U1, "ds-1"), U1, "sid-1"):
        stores.milvus.insert_insight(key, f"insight under {key}", "memory_llm")
    stores.chroma.update_user_signature(U1, "pre-fix signature of user 1")
    stores.chroma.update_user_signature(_k_pref(U1), "User prefers concise answers")
    stores.milvus.insert_insight(_k_pref(U2), "user 2 preference", "memory_llm")

    result = _run(forget(U1))

    assert _thread_memory_left(stores, U1, "thread-a") == []
    keys = _milvus_keys(stores.milvus)
    assert not [k for k in keys if k in (U1, "sid-1") or k.startswith(U1 + "#")], keys
    assert stores.chroma.get_user_signature(U1) is None
    assert stores.chroma.get_user_signature(_k_pref(U1)) is None
    # Nobody else's memory goes.
    assert _thread_memory_left(stores, U2, "thread-z") == ["milvus", "chroma", "redis", "in-process"]
    assert _k_pref(U2) in keys
    assert not any(str(v).startswith("failed") for v in result.values()), result


def test_a_store_that_is_down_does_not_fail_the_thread_delete(client, stores, monkeypatch):
    SESSIONS["sid-1"] = {"user_id": U1}
    THREAD_TO_SESSION["thread-a"] = "sid-1"
    _seed_thread_memory(stores, U1, "thread-a")
    collection = stores.milvus._collection
    stores.milvus._collection = None  # Milvus unreachable

    resp = client.delete("/threads/thread-a", headers=make_auth_headers(U1))
    assert resp.status_code == 204
    # What could be deleted was; Milvus reports that it was not.
    assert stores.chroma.get_user_signature(_k_thread(U1, "thread-a")) is None
    assert stores.redis.get_json("ctxmem:thread-a:hints") is None
    assert not MemoryOrchestrator._session_hints_store.get("thread-a")
    result = stores.orch.forget_thread(U1, "thread-a", "sid-1")
    assert str(result.get("milvus", "")).startswith("failed"), result
    stores.milvus._collection = collection
    assert _k_thread(U1, "thread-a") in _milvus_keys(stores.milvus)
