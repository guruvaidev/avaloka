"""A dedicated, bounded executor for the long producers.

Four graph paths block a thread for the whole job: model training waits on
the Ray job (up to 12 h), a data transfer waits on its container or RayJob,
an analysis executed on Ray waits on the RayJob, and a batch prediction waits
on its Ray job. On develop-1.6 each of them ran inside ``GRAPH.invoke`` on the
event loop's **default** executor -- the same pool every chat turn's
``run_in_executor(None, …)`` uses, ``min(32, cpu + 4)`` threads in all. N long
jobs from any users could starve every other user's turns; the 70 s deferral
hid that from the user who started the job, not from the process.

This module gives those turns their own pool (``AVALOKA_LONG_JOBS_WORKERS``,
default 4) and leaves the default pool to chat. When the pool is full a new
long job waits in the pool's queue; the job ledger shows it ``queued`` and the
pending-turn poll says so, until a worker picks it up, when :func:`submit`'s
``on_start`` fires and the row becomes ``running``.

Classification happens before the graph runs (:func:`classify_long_turn`): the
training regex the server already uses, a transfer or batch-prediction intent,
``execution_mode == "k8s-ray"`` or ``ready_to_train``. A long turn the
classifier misses runs on the default pool as before -- stated here so nobody
reads this as a guarantee. The 1.7 fix (submit-only producers with a poll
chain) removes the long wait itself.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("avaloka.jobs.long")

DEFAULT_WORKERS = 4

#: Kinds a long turn is classified into; the names are job_ledger kinds.
TRANSFER_RE = re.compile(
    r"\b(?:transfer|copy|move|migrate|load|sync|replicate|export|push|ingest)\b[^.?!\n]{0,80}\b(?:to|into|from)\b"
    r"|\b(?:data\s+transfer|initiate\s+transfer|start\s+(?:the\s+)?transfer)\b",
    re.IGNORECASE,
)
BATCH_PREDICT_RE = re.compile(
    r"\b(?:predict|score|run\s+inference|batch\s+(?:inference|prediction)s?)\b[^.?!\n]{0,60}\b(?:on|for|over)\b"
    r"[^.?!\n]{0,40}\b(?:dataset|data|file|table|rows|whole|entire|all)\b",
    re.IGNORECASE,
)


def workers() -> int:
    try:
        return max(1, int(os.getenv("AVALOKA_LONG_JOBS_WORKERS", str(DEFAULT_WORKERS))))
    except ValueError:
        return DEFAULT_WORKERS


class LongJobPool:
    """A ThreadPoolExecutor plus the counters that make "queued" visible."""

    def __init__(self, max_workers: Optional[int] = None) -> None:
        self.max_workers = max_workers or workers()
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="avaloka-long-job")
        self._lock = threading.Lock()
        self._running = 0
        self._queued = 0

    @property
    def running(self) -> int:
        with self._lock:
            return self._running

    @property
    def queued(self) -> int:
        with self._lock:
            return self._queued

    def would_queue(self) -> bool:
        with self._lock:
            return self._running + self._queued >= self.max_workers

    def submit_sync(self, fn: Callable[[], Any], *, on_start: Optional[Callable[[], None]] = None):
        """Run ``fn`` on the long pool from plain sync code (no event loop);
        returns a ``concurrent.futures.Future``. Counters are kept so
        :meth:`would_queue` and :attr:`queued` see this job too."""
        with self._lock:
            self._queued += 1

        def _run():
            with self._lock:
                self._queued -= 1
                self._running += 1
            try:
                if on_start is not None:
                    try:
                        on_start()
                    except Exception:  # noqa: BLE001
                        logger.debug("[long-jobs] on_start failed", exc_info=True)
                return fn()
            finally:
                with self._lock:
                    self._running -= 1

        return self._executor.submit(_run)

    def submit(self, fn: Callable[[], Any], *, on_start: Optional[Callable[[], None]] = None) -> "asyncio.Future":
        """Run ``fn`` on the long pool from a coroutine; the future resolves on
        the calling loop. ``on_start`` runs on the worker thread the moment the
        job leaves the queue (the queued → running transition)."""
        loop = asyncio.get_running_loop()
        with self._lock:
            self._queued += 1

        def _run():
            with self._lock:
                self._queued -= 1
                self._running += 1
            try:
                if on_start is not None:
                    try:
                        on_start()
                    except Exception:  # noqa: BLE001 - a status callback never fails the job
                        logger.debug("[long-jobs] on_start failed", exc_info=True)
                return fn()
            finally:
                with self._lock:
                    self._running -= 1

        return loop.run_in_executor(self._executor, _run)

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)


_pool: Optional[LongJobPool] = None
_pool_lock = threading.Lock()


def get_pool() -> LongJobPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = LongJobPool()
        return _pool


def set_pool(pool: Optional[LongJobPool]) -> None:
    """Tests swap the pool (and may shrink it to one worker)."""
    global _pool
    with _pool_lock:
        _pool = pool


@dataclass
class LongTurn:
    kind: str                       # job_ledger kind
    reason: str                     # which rule fired, for the log line
    extra: Dict[str, Any] = field(default_factory=dict)


def classify_long_turn(text: Optional[str], state: Optional[Dict[str, Any]], training_re: Optional["re.Pattern[str]"] = None) -> Optional[LongTurn]:
    """Which long producer this turn will run, or None for an ordinary turn.

    Order matters only for the label; any match sends the turn to the long
    pool. Deterministic state wins over text: ``execution_mode == "k8s-ray"``
    means the execution node will wait on a RayJob whatever the words say.
    """
    state = state or {}
    text = text or ""
    if state.get("ready_to_train") or (training_re is not None and training_re.search(text)):
        return LongTurn("training", "training_intent")
    if TRANSFER_RE.search(text) or state.get("dta_pending_transfer"):
        return LongTurn("transfer", "transfer_intent")
    if BATCH_PREDICT_RE.search(text) or state.get("inference_scheduled"):
        return LongTurn("inference_batch", "batch_prediction_intent")
    if str(state.get("execution_mode") or "") == "k8s-ray":
        return LongTurn("ray_job", "execution_mode_k8s_ray")
    return None
