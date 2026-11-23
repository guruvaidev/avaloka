"""Crash repro for the 1.6 API SIGSEGV under uvloop (dedicated job only).

Not a pass/fail test, and not part of the gate: it needs a live Redis and it
ends in a native crash by design. Measured with 2000 iterations per process:
uvloop crashed 8 of 8 runs on Linux arm64 (the API image) and 6 of 6 on macOS;
the asyncio loop crashed 0 of 8 and 0 of 6. Under asyncio the graph-thread
reads still FAIL (RuntimeError: Future attached to a different loop) -- that
is the underlying defect; the asyncio loop only stops it being a segfault.

Each iteration: the server loop does `await cache.get(...)`, so the pool holds
a server-loop connection; then it awaits run_in_executor() of a sync function
that does asyncio.run(cache.get(...)) -- GRAPH.invoke -> avaloka_agent_node ->
_load_redis_context. `real` drives _load_redis_context itself.

Usage: python scripts/repro/api_uvloop_cross_loop_redis.py <redis_url> <uvloop|asyncio> <iterations> <repo_root> [real]
Exit: 0 = survived (see the outcome counts), 139 / -11 = SIGSEGV.
"""
import asyncio
import faulthandler
import sys
from collections import Counter

faulthandler.enable()
URL, MODE, N, ROOT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
REAL = len(sys.argv) > 5 and sys.argv[5] == "real"
sys.path.insert(0, ROOT)

from app.core.cache import RedisCache  # noqa: E402  production client, unmodified

cache = RedisCache(URL)
outcomes: Counter = Counter()


def graph_thread_read_real(i: int):
    """The production bridge itself: avaloka_agent._load_redis_context."""
    from app.agents.avaloka_agent.agent import _load_redis_context
    ctx = _load_redis_context({"thread_id": "repro-thread"})
    return "ok" if ctx.get("session_id") == "repro-sid" else "empty context (read failed, swallowed)"


def graph_thread_read(i: int):
    """What the LangGraph executor thread does: a fresh loop per call."""
    async def _read():
        await cache.get(f"thread_session:{i}")
        return await cache.get("seed")
    try:
        v = asyncio.run(_read())
        return "ok" if v == "x" else f"wrong value {v!r}"
    except Exception as exc:  # session_service swallows these at debug level
        msg = str(exc)
        if "attached to a different loop" in msg:
            msg = "Task got Future attached to a different loop"
        elif "bound to a different event loop" in msg:
            msg = "asyncio.Lock is bound to a different event loop"
        return f"{type(exc).__name__}: {msg[:60]}"


async def main():
    loop = asyncio.get_running_loop()
    await cache.set("seed", "x")
    fn = graph_thread_read
    if REAL:
        import json
        from app.services import session_service
        session_service.cache = cache
        await cache.set("thread:repro-thread:sid", "repro-sid")
        await cache.set("sess:repro-sid", json.dumps({"dataset_id": "d1"}))
        fn = graph_thread_read_real
    for i in range(N):
        await cache.get("seed")                       # server-loop use of the pool
        r = await loop.run_in_executor(None, fn, i)
        outcomes[r] += 1
    await cache.close()


if MODE == "uvloop":
    import uvloop
    uvloop.run(main())
else:
    asyncio.run(main())

print(f"mode={MODE} survived {N} iterations")
for k, v in outcomes.most_common():
    print(f"  {v:6d}  {k}")
