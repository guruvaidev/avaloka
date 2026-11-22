"""app/core/long_jobs.py: long producers run on their own bounded pool; the
default pool chat uses stays free; a full pool queues visibly.

    pytest tests/test_long_jobs.py -q
"""

from __future__ import annotations

import asyncio
import re
import threading
import time

import pytest

from app.core import long_jobs

TRAINING_RE = re.compile(r"\btrain\b.*\bmodel\b", re.IGNORECASE)


@pytest.fixture
def pool():
    p = long_jobs.LongJobPool(max_workers=2)
    long_jobs.set_pool(p)
    yield p
    p.shutdown()
    long_jobs.set_pool(None)


def _block(evt: threading.Event):
    def fn():
        evt.wait(5)
        return "done"
    return fn


def test_pool_size_comes_from_env(monkeypatch):
    monkeypatch.setenv("AVALOKA_LONG_JOBS_WORKERS", "7")
    assert long_jobs.workers() == 7
    monkeypatch.setenv("AVALOKA_LONG_JOBS_WORKERS", "junk")
    assert long_jobs.workers() == long_jobs.DEFAULT_WORKERS
    monkeypatch.setenv("AVALOKA_LONG_JOBS_WORKERS", "0")
    assert long_jobs.workers() == 1


def test_n_long_jobs_do_not_delay_a_chat_turn_on_the_default_pool(pool):
    """The owner's requirement: with the long pool saturated, a chat turn's
    run_in_executor(None, …) still completes promptly."""
    release = threading.Event()

    async def main():
        loop = asyncio.get_running_loop()
        long = [pool.submit(_block(release)) for _ in range(6)]          # 2 run, 4 queue
        await asyncio.sleep(0.05)
        assert pool.running == 2 and pool.queued == 4 and pool.would_queue()
        t0 = time.monotonic()
        chat = await asyncio.wait_for(loop.run_in_executor(None, lambda: "chat ok"), timeout=2.0)
        elapsed = time.monotonic() - t0
        release.set()
        await asyncio.gather(*long)
        return chat, elapsed

    chat, elapsed = asyncio.run(main())
    assert chat == "chat ok" and elapsed < 1.0
    assert pool.running == 0 and pool.queued == 0


def test_long_jobs_never_run_on_the_default_pool(pool):
    names = []

    async def main():
        await pool.submit(lambda: names.append(threading.current_thread().name))
        await asyncio.get_running_loop().run_in_executor(None, lambda: names.append(threading.current_thread().name))

    asyncio.run(main())
    assert names[0].startswith("avaloka-long-job") and not names[1].startswith("avaloka-long-job")


def test_queued_then_running_is_visible_through_on_start(pool):
    release = threading.Event()
    started = []

    async def main():
        first = [pool.submit(_block(release), on_start=lambda: started.append("busy")) for _ in range(2)]
        await asyncio.sleep(0.05)
        third = pool.submit(lambda: "third", on_start=lambda: started.append("third"))
        await asyncio.sleep(0.05)
        assert started == ["busy", "busy"] and pool.queued == 1          # third is queued, not started
        release.set()
        await asyncio.gather(*first)
        assert await third == "third"
        assert started == ["busy", "busy", "third"]

    asyncio.run(main())


def test_on_start_failure_does_not_fail_the_job(pool):
    async def main():
        return await pool.submit(lambda: 42, on_start=lambda: (_ for _ in ()).throw(RuntimeError("x")))

    assert asyncio.run(main()) == 42 and pool.running == 0


def test_an_exception_in_the_job_propagates_and_frees_the_slot(pool):
    async def main():
        with pytest.raises(ValueError):
            await pool.submit(lambda: (_ for _ in ()).throw(ValueError("boom")))

    asyncio.run(main())
    assert pool.running == 0 and pool.queued == 0


@pytest.mark.parametrize("text,state,kind", [
    ("please train a model on churn", {}, "training"),
    ("hello", {"ready_to_train": True}, "training"),
    ("copy the orders table to the warehouse", {}, "transfer"),
    ("start the transfer", {}, "transfer"),
    ("predict churn for the whole dataset", {}, "inference_batch"),
    ("run inference on all rows", {}, "inference_batch"),
    ("sum sales by region", {"execution_mode": "k8s-ray"}, "ray_job"),
])
def test_long_turns_are_classified(text, state, kind):
    lt = long_jobs.classify_long_turn(text, state, TRAINING_RE)
    assert lt is not None and lt.kind == kind


@pytest.mark.parametrize("text", [
    "sum sales by region", "which rows have a null password", "show me the top 10 customers",
    "what does the transfer column mean", "predict is a verb",
])
def test_ordinary_turns_stay_on_the_default_pool(text):
    assert long_jobs.classify_long_turn(text, {"execution_mode": "local"}, TRAINING_RE) is None
