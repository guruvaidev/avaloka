import asyncio
import json

import pytest

import app.services.session_service as ss


class FakeAsyncCache:
    """Minimal async cache whose get/set yield control, so concurrent
    read-modify-write cycles genuinely interleave on the event loop."""

    def __init__(self):
        self.data = {}
        self.sets = {}

    async def get(self, k):
        await asyncio.sleep(0)
        return self.data.get(k)

    async def set(self, k, v, ex=None):
        await asyncio.sleep(0)
        self.data[k] = v

    async def sadd(self, k, m):
        self.sets.setdefault(k, set()).add(m)

    async def expire(self, k, ttl):
        pass


@pytest.fixture
def cache(monkeypatch):
    fake = FakeAsyncCache()
    monkeypatch.setattr(ss, "cache", fake, raising=False)
    monkeypatch.setattr(ss, "CACHE_DEGRADED_UNTIL", 0, raising=False)
    # Fresh lock registry per test: each asyncio.run() uses a new event loop and
    # an asyncio.Lock cannot be reused across loops.
    monkeypatch.setattr(ss, "_session_locks", {}, raising=False)
    return fake


async def _seed(sid="s1"):
    await ss.save_session(sid, {"user_id": "u", "base": 1})


# -------------------------
# Unlocked read-modify-write loses one writer's field (the bug)
# -------------------------

async def _unlocked_update(sid, mutator):
    cur = await ss.get_session(sid) or {}
    mutator(cur)
    await ss.save_session(sid, cur)


def test_unlocked_rmw_loses_a_field(cache):
    async def run():
        await _seed()
        await asyncio.gather(
            _unlocked_update("s1", lambda c: c.__setitem__("persist_key", "obj-123")),
            _unlocked_update("s1", lambda c: c.__setitem__("tasks", ["t1"])),
        )
        return await ss.get_session("s1")

    final = asyncio.run(run())
    # The two concurrent full-blob writes clobber each other: at least one
    # field is missing. (This is the MAJ-082 race, reproduced.)
    assert not ("persist_key" in final and "tasks" in final)


# -------------------------
# update_session serializes and merges -> no lost update
# -------------------------

def test_update_session_preserves_both_writers(cache):
    async def run():
        await _seed()
        await asyncio.gather(
            ss.update_session("s1", lambda c: c.__setitem__("persist_key", "obj-123")),
            ss.update_session("s1", lambda c: c.__setitem__("tasks", ["t1"])),
        )
        return await ss.get_session("s1")

    final = asyncio.run(run())
    assert final["persist_key"] == "obj-123"
    assert final["tasks"] == ["t1"]
    assert final["base"] == 1


def test_persist_and_chat_turn_merge(cache):
    # Mirrors the real race: the background persist adds asset keys while the
    # next chat turn overlays its working-copy session.
    async def run():
        await _seed()

        def persist(cur):
            cur["gcs_code_object_key"] = "code-registry/x.py"
            cur.setdefault("code_assets", []).append({"object_key": "code-registry/x.py"})

        chat_turn_working_copy = {"user_id": "u", "base": 1, "tasks": ["task-A"]}

        await asyncio.gather(
            ss.update_session("s1", persist),
            ss.update_session("s1", lambda cur: cur.update(chat_turn_working_copy)),
        )
        return await ss.get_session("s1")

    final = asyncio.run(run())
    # Neither writer's fields are lost.
    assert final["gcs_code_object_key"] == "code-registry/x.py"
    assert final["code_assets"] == [{"object_key": "code-registry/x.py"}]
    assert final["tasks"] == ["task-A"]


def test_update_session_replacement_return(cache):
    async def run():
        await _seed()

        def replace(cur):
            return {"user_id": "u", "replaced": True}

        await ss.update_session("s1", replace)
        return await ss.get_session("s1")

    final = asyncio.run(run())
    assert final == {"user_id": "u", "replaced": True}


def test_update_session_noop_without_cache(monkeypatch):
    monkeypatch.setattr(ss, "cache", None, raising=False)
    assert asyncio.run(ss.update_session("s1", lambda c: c.update({"x": 1}))) is None
