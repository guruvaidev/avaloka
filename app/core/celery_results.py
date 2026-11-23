"""Durable Celery task results: Postgres is the record, Redis is the cache.

The Redis result backend was the only copy of every task result, with no expiry
(``result_expires=None``) and each result holding the task's whole final state:
its conversation, its input arguments and the run's output CSV inlined as
base64. Results accumulated without bound until Redis ran out of memory, and
a result that did expire would have read as PENDING, i.e. "still running".

:class:`DurableRedisBackend` keeps the Redis backend's behaviour and adds a
durable copy:

* **write** (worker): a finished result (SUCCESS, FAILURE, REVOKED) is written
  to ``public.celery_task_results`` first, then to Redis with
  ``result_expires`` (7 days by default). If the Postgres write fails, the Redis
  copy is kept with **no** TTL and the task id is added to
  :data:`PENDING_SET`, so the result can neither expire nor be evicted before it
  is durable. ``scripts/ops/backfill_celery_results.py`` drains that set.
* **read** (API, LangGraph, worker): Redis first; on a miss, Postgres, and
  the row is written back to Redis. Every ``AsyncResult(...).state`` /
  ``.result`` / ``.ready()`` reader gets this without a change of its own.

What is stored is a summary, never the state itself (:func:`summarize`): no
messages, no arguments, no inline output. The durable ``scheduled_runs`` row
(written by the worker from the full state) already holds a run's logs, its
output table (capped) and its run truth; readers take those from there
(:func:`load_scheduled_run`).

Retention: rows are deleted once ``date_done`` is older than
``CELERY_RESULT_RETENTION_DAYS`` (default 365; up to 1095 on Enterprise), by
the daily ``purge_celery_task_results`` task (:func:`purge_expired_results`).

Logs from this module carry task ids, counts and exception class names only.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from celery import states
from celery.backends.redis import RedisBackend
from kombu.utils.encoding import bytes_to_str

logger = logging.getLogger(__name__)

TABLE = os.getenv("CELERY_TASK_RESULTS_TABLE", "celery_task_results")
SCHEDULED_RUNS_TABLE = os.getenv("SUPABASE_SCHEDULED_RUNS_TABLE", "scheduled_runs")
PURGE_FUNCTION = "purge_celery_task_results"

#: Task ids whose result is in Redis but not yet in Postgres.
PENDING_SET = "celery-durable-pending"
META_PREFIX = "celery-task-meta-"

RESULT_EXPIRES_DEFAULT = 7 * 24 * 3600

RETENTION_DEFAULT_DAYS = 365
RETENTION_STANDARD_MAX_DAYS = 365
RETENTION_ENTERPRISE_MAX_DAYS = 1095

SUMMARY_VERSION = 1

#: Tasks whose result is an agent state: stored as a summary in Redis too.
AGENT_TASKS = frozenset({
    "app.core.celery_app.agent_task",
    "app.core.celery_app.execution_agent_task",
})

_WRITE_BACKOFF_S = (0.2, 1.0, 3.0)
_MISS_CACHE_S = 15.0
_MISS_CACHE_MAX = 10_000


class ResultStoreConfigError(RuntimeError):
    """A result-store setting is invalid. Raised, never clamped."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def result_expires_from_env() -> Optional[int]:
    """Seconds a result stays in Redis: ``CELERY_RESULT_EXPIRES``, default 7 days.

    ``0`` keeps results in Redis forever (the Postgres copy is written either
    way). Anything that is not a whole number of seconds is a configuration
    error.
    """
    raw = os.getenv("CELERY_RESULT_EXPIRES", "").strip()
    if not raw:
        return RESULT_EXPIRES_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        raise ResultStoreConfigError(
            "CELERY_RESULT_EXPIRES must be a whole number of seconds (0 = never expire)"
        ) from None
    if value < 0:
        raise ResultStoreConfigError("CELERY_RESULT_EXPIRES must not be negative")
    return value or None


def _license_token() -> Optional[str]:
    """The licence token, from the environment or the stored file (as scripts/check_edition.py reads it)."""
    token = os.getenv("AVALOKA_LICENSE_KEY", "").strip()
    if token:
        return token
    path = Path(os.getenv("AVALOKA_HOME", str(Path.home() / ".avaloka"))) / "license"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def deployment_edition() -> str:
    """This deployment's edition, resolved the way scripts/check_edition.py does.

    A licence token decides when there is one (verified offline,
    app/core/license.py ``edition_from_license``, failing closed to ``oss``);
    otherwise ``AVALOKA_EDITION`` (app/core/editions.py ``normalize_edition``).
    A paid edition needs the commercial build (``editions.commercial_build``):
    docs/EDITIONS.md, setting AVALOKA_EDITION=enterprise in an open-source
    checkout grants nothing.
    """
    from app.core import editions
    from app.core.license import edition_from_license

    token = _license_token()
    edition = edition_from_license(token) if token else editions.normalize_edition(os.getenv("AVALOKA_EDITION"))
    if edition != "oss" and not editions.commercial_build():
        return "oss"
    return edition


def retention_days() -> int:
    """``CELERY_RESULT_RETENTION_DAYS``: default 365, 1..365, up to 1095 on Enterprise.

    An invalid value raises :class:`ResultStoreConfigError`; it is never
    clamped, so a deployment never keeps results for a period nobody set.
    """
    raw = os.getenv("CELERY_RESULT_RETENTION_DAYS", "").strip()
    if not raw:
        return RETENTION_DEFAULT_DAYS
    try:
        days = int(raw)
    except ValueError:
        raise ResultStoreConfigError("CELERY_RESULT_RETENTION_DAYS must be a whole number of days") from None
    if days < 1 or days > RETENTION_ENTERPRISE_MAX_DAYS:
        raise ResultStoreConfigError(
            f"CELERY_RESULT_RETENTION_DAYS must be between 1 and {RETENTION_ENTERPRISE_MAX_DAYS}"
        )
    if days > RETENTION_STANDARD_MAX_DAYS and deployment_edition() != "enterprise":
        raise ResultStoreConfigError(
            f"CELERY_RESULT_RETENTION_DAYS above {RETENTION_STANDARD_MAX_DAYS} needs the Enterprise "
            f"edition; this deployment may keep results for at most {RETENTION_STANDARD_MAX_DAYS} days"
        )
    return days


# ---------------------------------------------------------------------------
# Summary: the only shape that is stored
# ---------------------------------------------------------------------------

def _short_str(value: Any, limit: int) -> Optional[str]:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value)
    return text[:limit] if text else None


def summarize(final: Any) -> Dict[str, Any]:
    """The stored form of a task result: what the readers need, and nothing else.

    Allowlisted fields only. Conversation messages, task arguments, the inline
    output (``output_file_data``), generated code, plans and error text never
    appear here; a finished run's logs, output table and run truth are on its
    ``scheduled_runs`` row. Idempotent: a summary summarizes to itself.
    """
    out: Dict[str, Any] = {"_summary": SUMMARY_VERSION}
    if not isinstance(final, dict):
        return out
    er = final.get("execution_result") if isinstance(final.get("execution_result"), dict) else {}
    status = _short_str(er.get("status") if er else final.get("execution_status"), 32)
    if status:
        out["execution_status"] = status
    loc = _short_str(final.get("output_location"), 2048)
    if loc:
        out["output_location"] = loc
    stats = final.get("task_run_statistics")
    if isinstance(stats, dict):
        kept = {k: stats[k] for k in ("start", "end", "time_spent")
                if isinstance(stats.get(k), (int, float)) and not isinstance(stats.get(k), bool)}
        if kept:
            out["task_run_statistics"] = kept
    samples = final.get("available_samples")
    if isinstance(samples, (list, tuple)):
        out["available_samples"] = [s[:128] for s in samples if isinstance(s, str)][:200]
    # analysis_fidelity and selected_sample_name feed the result's execution
    # context (app/api/ui_contract.py build_execution_context_from_state).
    for key in ("sample_status", "profiling_status", "analysis_fidelity", "selected_sample_name"):
        value = _short_str(final.get(key), 64)
        if value:
            out[key] = value
    if final.get("_summary"):
        for key in ("has_output", "has_error"):
            if isinstance(final.get(key), bool):
                out[key] = final[key]
    else:
        out["has_output"] = bool(final.get("output_file_data") or final.get("output_location"))
        out["has_error"] = bool(final.get("error") or (er or {}).get("error"))
    return out


def is_summary(result: Any) -> bool:
    return isinstance(result, dict) and bool(result.get("_summary"))


def _exception_summary(result: Any) -> Dict[str, Any]:
    """A FAILURE/REVOKED result as stored: the exception's class, never its message."""
    if isinstance(result, dict):
        exc_type, exc_module = result.get("exc_type"), result.get("exc_module")
    elif isinstance(result, BaseException):
        exc_type, exc_module = type(result).__name__, type(result).__module__
    else:
        exc_type, exc_module = None, None
    return {
        "exc_type": _short_str(exc_type, 128) or "Exception",
        "exc_module": _short_str(exc_module, 128) or "builtins",
        "exc_message": [],
    }


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return _short_str(value, 64)


def durable_row(task_id: str, meta: Dict[str, Any], *, args: Any = None,
                name: Optional[str] = None) -> Dict[str, Any]:
    """The ``celery_task_results`` row for one finished result's meta.

    ``args`` is read for the two ids only (user and schedule); it is never stored.
    """
    status = meta.get("status")
    result = meta.get("result")
    stored = summarize(result) if status == states.SUCCESS else _exception_summary(result)
    state = args[0] if isinstance(args, (list, tuple)) and args and isinstance(args[0], dict) else {}
    return {
        "task_id": str(task_id),
        "status": status,
        "result": stored,
        # Tracebacks carry source lines and exception text; the run's error is
        # on its scheduled_runs row.
        "traceback": None,
        "date_done": _iso(meta.get("date_done")) or datetime.now(timezone.utc).isoformat(),
        "name": _short_str(name or meta.get("name"), 256),
        "user_id": _short_str(state.get("user_id"), 128),
        "schedule_id": _short_str(state.get("schedule_id"), 256),
    }


def meta_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Celery task meta rebuilt from a durable row (not yet decoded)."""
    return {
        "status": row.get("status") or states.PENDING,
        "result": row.get("result"),
        "traceback": row.get("traceback"),
        "children": [],
        "date_done": row.get("date_done"),
        "task_id": row.get("task_id"),
        "name": row.get("name"),
    }


# ---------------------------------------------------------------------------
# Postgres (PostgREST, service role)
# ---------------------------------------------------------------------------

def _client():
    # Looked up at call time: tests and the ops script replace it.
    from app.agents import sampling_persistence
    return sampling_persistence.get_supabase_client()


def upsert_row(row: Dict[str, Any]) -> None:
    _client().table(TABLE).upsert(row, on_conflict="task_id").execute()


def read_row(task_id: str) -> Optional[Dict[str, Any]]:
    res = (_client().table(TABLE)
           .select("task_id,status,result,traceback,date_done,name")
           .eq("task_id", str(task_id)).limit(1).execute())
    data = getattr(res, "data", None) or []
    return data[0] if data else None


def load_scheduled_run(run_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """The durable ``scheduled_runs`` row for one run (status, result, error), or None."""
    if not run_id:
        return None
    try:
        res = (_client().table(SCHEDULED_RUNS_TABLE)
               .select("run_id,status,started_at,finished_at,duration_ms,result,error")
               .eq("run_id", str(run_id)).limit(1).execute())
    except Exception as exc:  # noqa: BLE001 - a reader degrades, it does not fail the request
        logger.warning("[results] scheduled_runs lookup failed for %s (%s)", run_id, type(exc).__name__)
        return None
    data = getattr(res, "data", None) or []
    return data[0] if data else None


def assistant_messages_from_logs(logs: Any) -> List[Dict[str, str]]:
    """The assistant messages a run log holds (``[assistant] ...`` lines, celery_app._logs_from_final)."""
    out: List[Dict[str, str]] = []
    for line in logs if isinstance(logs, list) else []:
        if isinstance(line, str) and line.startswith("[assistant] "):
            out.append({"role": "assistant", "content": line[len("[assistant] "):]})
    return out


def output_file_data_from_rows(rows: Any, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """A run's output table (``scheduled_runs.result.output_json``) as the data-uri CSV readers return.

    Built for the response only; nothing like it is stored with the result.
    """
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return None
    import base64
    import csv
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(r for r in rows if isinstance(r, dict))
    raw = buf.getvalue().encode("utf-8")
    return {
        "filename": os.path.basename(filename or "") or "output.csv",
        "content": "data:text/csv;base64," + base64.b64encode(raw).decode("ascii"),
        "size": len(raw),
    }


#: scheduled_runs.status words that mean the run is over.
FINISHED_RUN_STATUSES = frozenset({"success", "failure"})


def finished_run_state(row: Optional[Dict[str, Any]]) -> Optional[str]:
    """SUCCESS/FAILURE when a scheduled_runs row says the run is over, else None."""
    status = str((row or {}).get("status") or "").lower()
    if status not in FINISHED_RUN_STATUSES:
        return None
    return states.SUCCESS if status == "success" else states.FAILURE


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------

class DurableRedisBackend(RedisBackend):
    """RedisBackend with a durable Postgres copy of every finished result."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._misses: Dict[str, float] = {}

    # -- what is stored ----------------------------------------------------

    def _get_result_meta(self, *args, **kwargs):
        meta = super()._get_result_meta(*args, **kwargs)
        # result_extended stays on for name and date_done; the arguments are the
        # task's whole input state and no reader uses them.
        meta.pop("args", None)
        meta.pop("kwargs", None)
        return meta

    def _redis_meta_for(self, task_id):
        meta = self.get(self.get_key_for_task(task_id))
        if not meta:
            return {"status": states.PENDING, "result": None}
        return self.decode_result(meta)

    def _store_result(self, task_id, result, state, traceback=None, request=None, **kwargs):
        name = getattr(request, "task", None) if request is not None else None
        if state == states.SUCCESS and name in AGENT_TASKS:
            result = summarize(result)
        meta = self._get_result_meta(result=result, state=state, traceback=traceback, request=request)
        meta["task_id"] = bytes_to_str(task_id)
        # As the parent: a SUCCESS already stored is never overwritten.
        if self._redis_meta_for(task_id)["status"] == states.SUCCESS:
            return result
        key = self.get_key_for_task(task_id)
        encoded = self.encode(meta)
        if state in states.READY_STATES:
            row = durable_row(meta["task_id"], meta, args=getattr(request, "args", None), name=name)
            if not self._write_durable(row):
                # Not durable yet: keep the cached copy until it is (no TTL, so
                # neither expiry nor volatile-lru eviction can take it).
                self.ensure(self._set_persistent, (key, encoded))
                self.client.sadd(PENDING_SET, meta["task_id"])
                return result
            self._misses.pop(meta["task_id"], None)
        self._set_with_state(key, encoded, state)
        return result

    def _set_persistent(self, key, value):
        with self.client.pipeline() as pipe:
            pipe.set(key, value)
            pipe.publish(key, value)
            pipe.execute()

    def _write_durable(self, row: Dict[str, Any]) -> bool:
        last: Optional[BaseException] = None
        for attempt, delay in enumerate((0.0, *_WRITE_BACKOFF_S)):
            if delay:
                time.sleep(delay)
            try:
                upsert_row(row)
                return True
            except RuntimeError as exc:
                # get_supabase_client: the store is not configured. Retrying
                # cannot help.
                last = exc
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
        logger.error(
            "[results] durable write failed for task %s (%s); result kept in Redis without expiry "
            "and queued in %s for the backfill", row["task_id"], type(last).__name__, PENDING_SET,
        )
        return False

    # -- reading -----------------------------------------------------------

    def decode_result(self, payload):
        meta = super().decode_result(payload)
        # Every decoded result passes here: reads, and the result consumer's
        # pub/sub messages (AsyncResult.get). Once a task has a result, an
        # earlier "not in Postgres yet" must not hide its durable row.
        if isinstance(meta, dict) and meta.get("status") != states.PENDING and meta.get("task_id"):
            self._misses.pop(bytes_to_str(meta["task_id"]), None)
        return meta

    def _recent_miss(self, task_id: str) -> bool:
        until = self._misses.get(task_id)
        if until is None:
            return False
        if until < time.monotonic():
            self._misses.pop(task_id, None)
            return False
        return True

    def _note_miss(self, task_id: str) -> None:
        if len(self._misses) >= _MISS_CACHE_MAX:
            self._misses.clear()
        self._misses[task_id] = time.monotonic() + _MISS_CACHE_S

    def _get_task_meta_for(self, task_id):
        task_id = bytes_to_str(task_id)
        try:
            meta = self._redis_meta_for(task_id)
        except self.connection_errors as exc:
            # Redis is down: a finished result can still be read from Postgres.
            row = read_row(task_id)
            if row:
                return self.meta_from_decoded(meta_from_row(row))
            logger.warning("[results] Redis unavailable and no durable row for %s (%s)",
                           task_id, type(exc).__name__)
            raise
        if meta.get("status") != states.PENDING or self._recent_miss(task_id):
            return meta
        try:
            row = read_row(task_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[results] durable read failed for %s (%s)", task_id, type(exc).__name__)
            return meta
        if not row:
            self._note_miss(task_id)
            return meta
        loaded = meta_from_row(row)
        try:
            self._set_with_state(self.get_key_for_task(task_id), self.encode(loaded), loaded["status"])
        except Exception as exc:  # noqa: BLE001 - the cache refill is best effort
            logger.debug("[results] cache refill failed for %s (%s)", task_id, type(exc).__name__)
        return self.meta_from_decoded(loaded)


def backend_url(redis_url: str) -> str:
    """``result_backend`` for :class:`DurableRedisBackend` on ``redis_url``."""
    return f"{__name__}:DurableRedisBackend+{redis_url}"


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

PURGE_BATCH_DEFAULT = 1000
PURGE_MAX_BATCHES_DEFAULT = 100


def purge_expired_results(*, now: Optional[datetime] = None, batch_size: Optional[int] = None,
                          max_batches: Optional[int] = None) -> Dict[str, int]:
    """Delete finished results whose ``date_done`` is past retention, in bounded batches.

    Each batch is one call of the SQL function ``purge_celery_task_results``
    (migration 20261117120000): at most ``batch_size`` rows, chosen on the
    ``date_done`` index, finished states only, ``FOR UPDATE SKIP LOCKED``. So no
    statement holds a long lock, and an unfinished task is never deleted. A run
    stops after ``max_batches``; the next day's run continues.
    """
    days = retention_days()
    batch_size = int(batch_size or os.getenv("CELERY_RESULT_PURGE_BATCH", PURGE_BATCH_DEFAULT))
    max_batches = int(max_batches or os.getenv("CELERY_RESULT_PURGE_MAX_BATCHES", PURGE_MAX_BATCHES_DEFAULT))
    if not 1 <= batch_size <= 10_000:
        raise ResultStoreConfigError("CELERY_RESULT_PURGE_BATCH must be between 1 and 10000")
    if max_batches < 1:
        raise ResultStoreConfigError("CELERY_RESULT_PURGE_MAX_BATCHES must be at least 1")
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    client = _client()
    deleted = batches = 0
    while batches < max_batches:
        res = client.rpc(PURGE_FUNCTION, {"cutoff": cutoff.isoformat(), "batch_size": batch_size}).execute()
        n = int(getattr(res, "data", 0) or 0)
        batches += 1
        deleted += n
        if n < batch_size:
            break
    logger.info("[results] purge: %d row(s) deleted in %d batch(es), retention %d days",
                deleted, batches, days)
    return {"deleted": deleted, "batches": batches, "retention_days": days}


def protected_key(key: str) -> bool:
    """Keys the ops script must never touch (broker, scheduler, sessions)."""
    return key.startswith(("redbeat:", "_kombu", "unacked", "sess:"))


def iter_meta_keys(client, count: int = 500) -> Iterable[str]:
    """Every ``celery-task-meta-*`` key, by SCAN. The only pattern there is."""
    for key in client.scan_iter(match=META_PREFIX + "*", count=count):
        key = bytes_to_str(key)
        if not key.startswith(META_PREFIX) or protected_key(key):
            raise RuntimeError("SCAN returned a key outside celery-task-meta-*; refusing to continue")
        yield key
