"""The shared Redis cache must never hand one loop's sockets to another loop.

``session_service.cache`` is one object for the whole API process. The API's
own loop uses it, and so does ``avaloka_agent``: LangGraph runs that sync node
on a worker thread, where ``_load_redis_context`` opens a private loop with
``asyncio.run``. With one ``redis.asyncio`` client behind the cache, the worker
loop picked up a pooled connection whose stream belongs to the API loop. Under
the stdlib loop that is "got Future attached to a different loop", which trips
the cache circuit breaker so routes answer "No session found". Under uvloop it
is a worker thread driving a libuv stream the main thread is polling: the API's
exit-139 segfaults.

These tests talk to a real socket through the real redis-py client, so the
connection pool behaves exactly as in production. The server is a few lines of
RESP3 in a thread of its own; no Redis is needed.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

pytest.importorskip("redis")

from app.core.cache import RedisCache  # noqa: E402

_STORE = {
    "thread:t1:sid": "sid-1",
    "sess:sid-1": '{"dataset_id": "ds-1", "input_data_type": "csv"}',
}


def _bulk(value):
    if value is None:
        return b"_\r\n"
    raw = value.encode()
    return b"$%d\r\n%s\r\n" % (len(raw), raw)


# Client connections the server currently holds open (one server thread).
_OPEN = {"n": 0}


async def _serve(reader, writer):
    _OPEN["n"] += 1
    try:
        while True:
            head = await reader.readline()
            if not head:
                return
            args = []
            for _ in range(int(head[1:])):
                await reader.readline()
                args.append((await reader.readline()).rstrip(b"\r\n").decode())
            cmd = args[0].upper()
            if cmd == "HELLO":
                writer.write(b"%1\r\n+proto\r\n:3\r\n")
            elif cmd == "PING":
                writer.write(b"+PONG\r\n")
            elif cmd == "GET":
                writer.write(_bulk(_STORE.get(args[1])))
            else:
                writer.write(b"+OK\r\n")
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        return
    finally:
        _OPEN["n"] -= 1
        writer.close()


@pytest.fixture
def resp_server():
    """A RESP3 server on its own thread and loop; yields its redis:// URL."""
    started = threading.Event()
    box = {}
    _OPEN["n"] = 0

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        box["loop"] = loop
        server = loop.run_until_complete(asyncio.start_server(_serve, "127.0.0.1", 0))
        box["port"] = server.sockets[0].getsockname()[1]
        started.set()
        loop.run_forever()
        server.close()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(server.wait_closed())
        loop.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert started.wait(5)
    yield f"redis://127.0.0.1:{box['port']}"
    box["loop"].call_soon_threadsafe(box["loop"].stop)
    thread.join(5)


@pytest.fixture
def shared_cache(resp_server, monkeypatch):
    from app.services import session_service

    monkeypatch.delenv("REDIS_CLUSTER", raising=False)
    monkeypatch.setenv("REDIS_URL", resp_server)
    cache = RedisCache()
    monkeypatch.setattr(session_service, "cache", cache)
    # The breaker is module state; start closed so one test cannot fail another.
    monkeypatch.setattr(session_service, "CACHE_DEGRADED_UNTIL", 0.0)
    return cache


def test_agent_context_loads_from_a_worker_thread_while_the_api_loop_holds_the_client(shared_cache):
    """The production shape: the API loop has used the client (its pool holds a
    connection bound to that loop), then a sync node on a worker thread loads
    the thread context. On the old single-client cache the worker loop reused
    that connection and the load came back empty, logged as "attached to a
    different loop"."""
    from app.agents.avaloka_agent.agent import _load_redis_context
    from app.services import session_service

    async def api_loop():
        assert await shared_cache.ping() is True        # startup: pool now holds a connection
        loop = asyncio.get_running_loop()
        loaded = []
        for _ in range(5):                               # several turns, as in a conversation
            loaded.append(await loop.run_in_executor(
                None, _load_redis_context, {"thread_id": "t1"}))
            # ... and the API loop keeps serving routes on the same cache.
            assert await session_service.get_session("sid-1") == {
                "dataset_id": "ds-1", "input_data_type": "csv"}
        await shared_cache.close()
        return loaded

    loaded = asyncio.run(api_loop())
    assert [c.get("session_id") for c in loaded] == ["sid-1"] * 5
    assert all(c.get("dataset_id") == "ds-1" for c in loaded)
    assert session_service._cache_unhealthy() is False, \
        "a cross-loop failure tripped the breaker: routes would answer 'No session found'"


def test_each_loop_gets_its_own_client_and_a_loop_keeps_its_client(shared_cache):
    seen = {}

    async def grab(tag):
        first = shared_cache._r
        await shared_cache.get("thread:t1:sid")
        assert shared_cache._r is first                  # stable within a loop
        seen[tag] = first
        await shared_cache.close()

    asyncio.run(grab("a"))
    t = threading.Thread(target=lambda: asyncio.run(grab("b")))
    t.start()
    t.join(10)
    assert seen["a"] is not seen["b"]


def test_worker_turns_load_context_and_leave_only_the_api_loops_connection_open(shared_cache):
    """The agent's asyncio.run path, seen from the Redis side. Each worker turn
    must load the thread's context, and must not leave a connection open behind
    it: after three turns the server holds only the API loop's connection."""
    import time

    from app.agents.avaloka_agent.agent import _load_redis_context

    async def api_loop():
        await shared_cache.ping()
        loop = asyncio.get_running_loop()
        loaded = [await loop.run_in_executor(None, _load_redis_context, {"thread_id": "t1"})
                  for _ in range(3)]
        deadline = time.monotonic() + 3              # closes reach the server asynchronously
        while _OPEN["n"] > 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        still_open = _OPEN["n"]
        await shared_cache.close()
        return loaded, still_open

    loaded, still_open = asyncio.run(api_loop())
    assert [c.get("session_id") for c in loaded] == ["sid-1"] * 3, \
        "a worker-thread turn lost the thread context (cross-loop read)"
    assert still_open == 1, f"{still_open - 1} worker-turn connection(s) left open"
