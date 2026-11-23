"""The background-job ledger: one row per long-running operation, one
notification per terminal transition, one chat note per job.

Every producer (a deferred chat turn, a Celery scheduled run, a data transfer,
an analysis executed on Ray, full-data sampling, model training, an inference
endpoint, a batch prediction, a swarm mission) records a row in ``ml_jobs``
(migration 20261121110000_background_jobs.sql, which widens train-infer's
20261117130000 table) and calls :func:`finish` exactly where it learns the
outcome. ``finish`` is plain sync so Celery tasks, worker threads and
``asyncio.to_thread`` can all call it; the API never calls it on the event loop.

Exactly one notification per terminal transition, across processes and
retries, rests on one statement::

    update ml_jobs set notified_at = now() where id = :id and notified_at is null returning id

Whoever gets a row back owns the notification. ``notification_service``'s
``ref="job:<id>"`` dedupe is the second net. The chat note is queued in
``job_chat_notes`` (``unique (job_id)``) by :mod:`app.core.job_notes` and
delivered by the API at a turn boundary, never from here.

Nothing here raises into the producer: a ledger failure must never fail the
work it reports on. Text that reaches a user is built from the closed
vocabularies below plus an optional ``label`` that is redacted on the way in;
the raw ``error`` line goes to the row and nowhere else.

Store: the Supabase service-role client, or an in-memory twin for tests and
``AVALOKA_JOB_LEDGER=memory``. The table is shared with ``app.core.ml_store``
(PR #771); that module's dataclass and kind tuple are not touched here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("avaloka.jobs")

TABLE = os.getenv("AVALOKA_ML_JOBS_TABLE", "ml_jobs")

# ---------------------------------------------------------------------------
# Closed vocabularies. The SQL check constraints repeat these exactly.
# ---------------------------------------------------------------------------

KINDS = frozenset({
    # train-infer's (ml_store.JOB_KINDS), kept verbatim
    "training", "register", "deploy_endpoint", "stop_endpoint", "cross_validate", "reap",
    # the ledger's
    "analysis", "scheduled_run", "transfer", "ray_job", "sampling", "inference_batch", "swarm",
})
STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")
TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
FAIL_REASONS = frozenset({"failed", "timed_out", "cancelled", "no_worker", "cluster_error", "refused", "not_answered"})
RESOURCE_TYPES = frozenset({"analysis", "dataset", "model", "endpoint", "transfer", "schedule", "mission"})
ORIGINS = frozenset({"api", "celery", "thread"})

#: What the user reads for a kind. No kind name leaks as an identifier.
KIND_TEXT: Dict[str, str] = {
    "analysis": "Analysis", "scheduled_run": "Scheduled analysis", "training": "Model training",
    "register": "Model registration", "cross_validate": "Cross-validation", "reap": "Cluster cleanup",
    "transfer": "Data transfer", "ray_job": "Analysis on the cluster", "sampling": "Full-data sampling",
    "deploy_endpoint": "Inference endpoint", "stop_endpoint": "Inference endpoint",
    "inference_batch": "Batch prediction", "swarm": "Swarm mission",
}

#: The notification type family per kind (``<family>.completed|failed``). One
#: ``inference`` family for deploy/stop/batch, with the sub-kind in metadata
#: (lead's ruling 4). ``reap`` is housekeeping and rings no bell.
FAMILY: Dict[str, Optional[str]] = {
    "analysis": "analysis", "scheduled_run": "scheduled_run",
    "training": "training", "register": "training", "cross_validate": "training",
    "transfer": "transfer", "ray_job": "ray_job", "sampling": "sampling",
    "deploy_endpoint": "inference", "stop_endpoint": "inference", "inference_batch": "inference",
    "swarm": "swarm", "reap": None,
}
INFERENCE_SUB_KIND = {"deploy_endpoint": "deploy", "stop_endpoint": "stop", "inference_batch": "batch"}

#: Titles. train-infer's wording is kept verbatim for training and endpoints.
TITLES: Dict[str, Dict[str, str]] = {
    "analysis":        {"succeeded": "Analysis completed", "failed": "Analysis failed"},
    "scheduled_run":   {"succeeded": "Scheduled analysis completed", "failed": "Scheduled analysis failed"},
    "training":        {"succeeded": "Model training completed", "failed": "Model training failed"},
    "register":        {"succeeded": "Model registered", "failed": "Model registration failed"},
    "cross_validate":  {"succeeded": "Cross-validation completed", "failed": "Cross-validation failed"},
    "transfer":        {"succeeded": "Data transfer completed", "failed": "Data transfer failed"},
    "ray_job":         {"succeeded": "Analysis completed", "failed": "Analysis failed"},
    "sampling":        {"succeeded": "Full-data sample ready", "failed": "Full-data sampling failed"},
    "deploy_endpoint": {"succeeded": "Inference endpoint ready", "failed": "Inference endpoint failed"},
    "stop_endpoint":   {"succeeded": "Inference endpoint stopped", "failed": "Inference endpoint failed"},
    "inference_batch": {"succeeded": "Batch prediction completed", "failed": "Batch prediction failed"},
    "swarm":           {"succeeded": "Swarm mission completed", "failed": "Swarm mission failed"},
}

#: The only failure text a user ever sees. The producer picks the token.
FAIL_TEXT: Dict[str, str] = {
    "failed": "it failed",
    "timed_out": "it timed out",
    "cancelled": "it was cancelled",
    "no_worker": "no background worker was running",
    "cluster_error": "it failed on the cluster",
    "refused": "the result did not pass the output checks",
    "not_answered": "it ran but did not answer the question",
}

LABEL_MAX = 80


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(v: Any) -> Any:
    return v.isoformat() if isinstance(v, datetime) else v


def _dt(v: Any) -> Optional[datetime]:
    if v is None or isinstance(v, datetime):
        return v
    s = str(v)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


@dataclass
class Job:
    id: str
    user_id: str
    kind: str
    status: str
    thread_id: Optional[str] = None
    session_id: Optional[str] = None
    analysis_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    label: Optional[str] = None
    fail_reason: Optional[str] = None
    error: Optional[str] = None
    origin: str = "api"
    ray_job_name: Optional[str] = None
    celery_task_id: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    attempts: int = 0
    notified_at: Optional[datetime] = None
    notification_id: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def duration_s(self) -> Optional[float]:
        start = self.started_at or self.created_at
        if start and self.finished_at:
            return max(0.0, (self.finished_at - start).total_seconds())
        return None


def _job_from_row(row: Dict[str, Any]) -> Job:
    fields = {f for f in Job.__dataclass_fields__}
    data = {k: v for k, v in row.items() if k in fields}
    for k in ("notified_at", "created_at", "started_at", "finished_at"):
        if k in data:
            data[k] = _dt(data[k])
    data["attempts"] = int(data.get("attempts") or 0)
    return Job(**data)


# ---------------------------------------------------------------------------
# Stores: the same four operations over Supabase or a dict.
# ---------------------------------------------------------------------------

LEDGER_COLUMNS = ("thread_id", "analysis_id", "resource_type", "resource_id", "label",
                  "fail_reason", "notified_at", "notification_id", "origin")


class MemoryJobs:
    """Dict-backed store with the production store's semantics, for tests."""

    def __init__(self) -> None:
        self.rows: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def upsert(self, row: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            cur = self.rows.get(row["id"], {})
            cur.update(row)
            self.rows[row["id"]] = cur
            return dict(cur)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.rows.get(job_id)
            return dict(row) if row else None

    def update(self, job_id: str, changes: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.rows.get(job_id)
            if row is None:
                return None
            row.update(changes)
            return dict(row)

    def claim_notification(self, job_id: str) -> bool:
        with self._lock:
            row = self.rows.get(job_id)
            if row is None or row.get("notified_at") is not None:
                return False
            row["notified_at"] = _now().isoformat()
            return True

    def open_for_thread(self, thread_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self.rows.values()
                    if r.get("thread_id") == thread_id and r.get("status") in ("queued", "running")]


class SupabaseJobs:
    """Service-role writes to ``ml_jobs``. One request per call.

    If the ledger columns are missing (the migration has not run yet) a write
    is retried without them once, the condition is remembered for the process
    and the caller degrades to bell-only. The job itself is never failed.
    """

    def __init__(self, client: Any = None) -> None:
        self._client = client
        self.columns_missing = False

    def _c(self) -> Any:
        if self._client is None:
            from app.agents.sampling_persistence import get_supabase_client
            self._client = get_supabase_client()
        return self._client

    @staticmethod
    def _rows(res: Any) -> List[Dict[str, Any]]:
        data = getattr(res, "data", None)
        if data is None and isinstance(res, dict):
            data = res.get("data")
        return list(data or [])

    @staticmethod
    def _is_missing_column(exc: BaseException) -> bool:
        text = str(exc).lower()
        return "column" in text and ("does not exist" in text or "pgrst204" in text or "schema cache" in text)

    def _strip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in payload.items() if k not in LEDGER_COLUMNS}

    def _write(self, fn, payload: Dict[str, Any]):
        if self.columns_missing:
            payload = self._strip(payload)
        try:
            return fn(payload)
        except Exception as exc:  # noqa: BLE001
            if not self.columns_missing and self._is_missing_column(exc):
                self.columns_missing = True
                logger.warning("[jobs] ml_jobs lacks the ledger columns (migration 20261121110000 not applied); "
                               "degrading to bell-only: %s", str(exc)[:160])
                return fn(self._strip(payload))
            raise

    def upsert(self, row: Dict[str, Any]) -> Dict[str, Any]:
        rows = self._rows(self._write(lambda p: self._c().table(TABLE).upsert(p, on_conflict="id").execute(), row))
        return rows[0] if rows else row

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        rows = self._rows(self._c().table(TABLE).select("*").eq("id", job_id).limit(1).execute())
        return rows[0] if rows else None

    def update(self, job_id: str, changes: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rows = self._rows(self._write(lambda p: self._c().table(TABLE).update(p).eq("id", job_id).execute(), changes))
        return rows[0] if rows else None

    def claim_notification(self, job_id: str) -> bool:
        """One UPDATE with ``notified_at is null`` in its WHERE: atomic on the row."""
        if self.columns_missing:
            return True  # no claim column: let notification_service's ref dedupe decide
        res = (self._c().table(TABLE).update({"notified_at": _now().isoformat()})
               .eq("id", job_id).is_("notified_at", "null").execute())
        return bool(self._rows(res))

    def open_for_thread(self, thread_id: str) -> List[Dict[str, Any]]:
        return self._rows(self._c().table(TABLE).select("*").eq("thread_id", thread_id)
                          .in_("status", ["queued", "running"]).execute())


_store: Any = None
_store_lock = threading.Lock()


def get_store() -> Any:
    global _store
    with _store_lock:
        if _store is None:
            _store = MemoryJobs() if os.getenv("AVALOKA_JOB_LEDGER", "").lower() == "memory" else SupabaseJobs()
        return _store


def set_store(store: Any) -> None:
    """Tests (and ``AVALOKA_JOB_LEDGER=memory`` dev runs) swap the store."""
    global _store
    with _store_lock:
        _store = store


# ---------------------------------------------------------------------------
# Label redaction: closed vocabularies carry the message; the label is the
# one free-text field, and it is scrubbed on the way in, every time.
# ---------------------------------------------------------------------------

_URLISH = re.compile(r"\S*(?:://|[\\/][\w.-]+[\\/]|\.(?:csv|parquet|xlsx|json|tsv|sql|zip|gz)\b)\S*", re.IGNORECASE)
_FALLBACK_SECRET = re.compile(
    r"(?i)\b(?:password|passwd|pwd|pass|secret(?:[_ ]access)?[_ ]key|access[_ ]key(?:[_ ]id)?|token|api[_ ]?key)\b"
    r"\s*(?:[:=]|=>|\bis\b)?\s*[\"'`]?([^\s\"'`,;]+)")


def scrub_label(text: Optional[str]) -> Optional[str]:
    """A user-safe label: sealed/typed credentials, URLs, paths and file names
    removed; whitespace collapsed; clipped to :data:`LABEL_MAX`."""
    if not text:
        return None
    s = str(text)
    try:
        from app.core import typed_secrets
        s = typed_secrets.redact_sealed(s) or ""
        redact_typed = getattr(typed_secrets, "redact_typed_secrets", None)
        if callable(redact_typed):
            out = redact_typed(s)
            s = out[0] if isinstance(out, tuple) else (out or "")
    except Exception:  # noqa: BLE001 - redaction must not depend on that module's health
        pass
    s = _FALLBACK_SECRET.sub(lambda m: m.group(0)[: m.start(1) - m.start(0)] + "[redacted]", s)
    s = _URLISH.sub("", s)
    s = " ".join(s.split())
    if not s:
        return None
    return s if len(s) <= LABEL_MAX else s[: LABEL_MAX - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def _check(value: Optional[str], allowed: frozenset, what: str) -> Optional[str]:
    if value is None:
        return None
    if value not in allowed:
        raise ValueError(f"unknown {what} {value!r}")
    return value


def start(
    user_id: str,
    kind: str,
    *,
    job_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    session_id: Optional[str] = None,
    analysis_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    label: Optional[str] = None,
    origin: str = "api",
    celery_task_id: Optional[str] = None,
    ray_job_name: Optional[str] = None,
    status: str = "queued",
) -> Optional[str]:
    """Record a new job (upsert on ``id``, so a redelivered start is harmless).
    Returns the job id, or None when the ledger is unreachable."""
    _check(kind, KINDS, "kind")
    _check(resource_type, RESOURCE_TYPES, "resource_type")
    _check(origin, ORIGINS, "origin")
    if status not in ("queued", "running"):
        raise ValueError(f"start status must be queued or running, not {status!r}")
    if not user_id:
        return None
    jid = str(job_id or uuid.uuid4())
    now = _now().isoformat()
    row = {
        "id": jid, "user_id": str(user_id), "kind": kind, "status": status,
        "thread_id": thread_id, "session_id": session_id, "analysis_id": analysis_id,
        "resource_type": resource_type, "resource_id": str(resource_id) if resource_id else None,
        "label": scrub_label(label), "origin": origin, "celery_task_id": celery_task_id,
        "ray_job_name": ray_job_name, "attempts": 0, "created_at": now,
        "started_at": now if status == "running" else None, "updated_at": now,
    }
    try:
        get_store().upsert({k: v for k, v in row.items() if v is not None})
        logger.info("[jobs] %s %s %s thread=%s origin=%s", status, kind, jid[:8], (thread_id or "-")[:8], origin)
        return jid
    except Exception as exc:  # noqa: BLE001
        logger.warning("[jobs] could not record %s %s: %s", kind, jid[:8], str(exc)[:160])
        return None


def running(job_id: Optional[str], *, ray_job_name: Optional[str] = None,
            celery_task_id: Optional[str] = None, attempt: Optional[int] = None) -> None:
    """queued → running (a no-op on a row that is already terminal)."""
    if not job_id:
        return
    try:
        store = get_store()
        cur = store.get(job_id)
        if cur is None or cur.get("status") in TERMINAL:
            return
        changes: Dict[str, Any] = {"status": "running", "updated_at": _now().isoformat()}
        if not cur.get("started_at"):
            changes["started_at"] = changes["updated_at"]
        if ray_job_name:
            changes["ray_job_name"] = ray_job_name
        if celery_task_id:
            changes["celery_task_id"] = celery_task_id
        if attempt is not None:
            changes["attempts"] = int(attempt)
        store.update(job_id, changes)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[jobs] could not mark %s running: %s", str(job_id)[:8], str(exc)[:160])


def get(job_id: Optional[str]) -> Optional[Job]:
    """The row, or None (missing or store down)."""
    if not job_id:
        return None
    try:
        row = get_store().get(job_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[jobs] could not read %s: %s", str(job_id)[:8], str(exc)[:160])
        return None
    return _job_from_row(row) if row else None


def claim(job_id: Optional[str], *, kind: Optional[str] = None, resource_type: Optional[str] = None,
          resource_id: Optional[str] = None, label: Optional[str] = None) -> bool:
    """A producer inside the turn says what the turn's job really is: its kind,
    the resource it touched, a user-safe label. A chat turn classified "long"
    whose row nobody claimed is an ordinary turn and is finished quietly. A
    terminal row is left alone. Returns True when something was written."""
    if not job_id:
        return False
    _check(kind, KINDS, "kind")
    _check(resource_type, RESOURCE_TYPES, "resource_type")
    changes: Dict[str, Any] = {}
    if kind:
        changes["kind"] = kind
    if resource_type:
        changes["resource_type"] = resource_type
    if resource_id:
        changes["resource_id"] = str(resource_id)
    if label:
        scrubbed = scrub_label(label)
        if scrubbed:
            changes["label"] = scrubbed
    if not changes:
        return False
    try:
        store = get_store()
        cur = store.get(job_id)
        if cur is None or cur.get("status") in TERMINAL:
            return False
        changes["updated_at"] = _now().isoformat()
        store.update(job_id, changes)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[jobs] could not claim %s: %s", str(job_id)[:8], str(exc)[:160])
        return False


def attach(job_id: Optional[str], *, celery_task_id: Optional[str] = None,
           ray_job_name: Optional[str] = None) -> bool:
    """A worker picks up a job another process recorded: mark it running with
    the worker's ids. Returns False when there is no such row (the caller then
    starts one)."""
    if not job_id:
        return False
    try:
        cur = get_store().get(job_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[jobs] could not read %s: %s", str(job_id)[:8], str(exc)[:160])
        return False
    if cur is None:
        return False
    running(job_id, celery_task_id=celery_task_id, ray_job_name=ray_job_name)
    return True


@dataclass
class FinishResult:
    job: Optional[Job]
    transitioned: bool = False     # this call moved the row to a terminal status
    claimed: bool = False          # this call owns the notification
    notification_id: Optional[str] = None
    note_queued: bool = False
    errors: List[str] = field(default_factory=list)


def finish(
    job_id: Optional[str],
    status: str,
    *,
    fail_reason: Optional[str] = None,
    error_line: Optional[str] = None,
    mlflow_run_id: Optional[str] = None,
    notify: bool = True,
    chat_note: bool = True,
) -> FinishResult:
    """Move a job to a terminal status and emit its one notification.

    Order, each step tolerant of the previous one's failure:
    1. the transition (a second call on a terminal row is a no-op and returns
       ``transitioned=False``);
    2. the ``notified_at`` claim;
    3. the bell, through ``notification_service``;
    4. the chat note, queued for the API to deliver.

    ``notify=False`` records the transition and takes the claim but rings no
    bell (the recurring-schedule rule); ``chat_note=False`` skips the outbox.
    """
    result = FinishResult(job=None)
    if status not in TERMINAL:
        raise ValueError(f"finish needs a terminal status, not {status!r}")
    _check(fail_reason, FAIL_REASONS, "fail_reason")
    if not job_id:
        return result
    try:
        store = get_store()
        cur = store.get(job_id)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"read: {exc}")
        logger.warning("[jobs] could not read %s: %s", str(job_id)[:8], str(exc)[:160])
        return result
    if cur is None:
        result.errors.append("missing")
        return result

    if cur.get("status") in TERMINAL:
        result.job = _job_from_row(cur)
    else:
        now = _now().isoformat()
        changes: Dict[str, Any] = {"status": status, "finished_at": now, "updated_at": now}
        if status == "failed":
            changes["fail_reason"] = fail_reason or "failed"
        elif status == "cancelled":
            changes["fail_reason"] = "cancelled"
        if error_line:
            changes["error"] = " ".join(str(error_line).split())[:300]
        if mlflow_run_id:
            changes["mlflow_run_id"] = mlflow_run_id
        if not cur.get("started_at"):
            changes["started_at"] = cur.get("created_at") or now
        try:
            saved = store.update(job_id, changes)
            result.transitioned = True
            result.job = _job_from_row(saved or {**cur, **changes})
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"update: {exc}")
            logger.warning("[jobs] could not finish %s: %s", str(job_id)[:8], str(exc)[:160])
            result.job = _job_from_row({**cur, **changes})

    job = result.job
    logger.info("[jobs] %s %s %s reason=%s duration_ms=%s", job.status, job.kind, job.id[:8],
                job.fail_reason or "-", int((job.duration_s or 0) * 1000))

    try:
        result.claimed = bool(store.claim_notification(job_id))
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"claim: {exc}")
        logger.warning("[jobs] claim failed for %s: %s", job.id[:8], str(exc)[:160])
        return result
    if not result.claimed:
        return result

    if notify and FAMILY.get(job.kind):
        result.notification_id = _ring_bell(job)
        if result.notification_id:
            try:
                store.update(job_id, {"notification_id": result.notification_id})
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"notification_id: {exc}")
    if chat_note and job.thread_id:
        try:
            from app.core import job_notes
            result.note_queued = bool(job_notes.enqueue(job))
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"note: {exc}")
            logger.warning("[jobs] chat note for %s not queued: %s", job.id[:8], str(exc)[:160])
    return result


def cancel(job_id: Optional[str], user_id: str) -> FinishResult:
    """Owner-scoped cancel: a terminal row is left alone; another user's row is
    treated as missing."""
    result = FinishResult(job=None)
    if not job_id:
        return result
    cur = get_store().get(job_id)
    if cur is None or str(cur.get("user_id")) != str(user_id):
        result.errors.append("missing")
        return result
    return finish(job_id, "cancelled")


# ---------------------------------------------------------------------------
# The bell
# ---------------------------------------------------------------------------

def _duration_text(seconds: Optional[float]) -> Optional[str]:
    if seconds is None or seconds < 1:
        return None
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} m {s:02d} s" if s else f"{m} m"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} m"


def notification_text(job: Job) -> Dict[str, Any]:
    """(type, title, body, extra) for a terminal job, from the closed vocabularies."""
    family = FAMILY.get(job.kind)
    if job.status == "cancelled":
        ntype, title = "job.cancelled", f"{KIND_TEXT.get(job.kind, 'Job')} cancelled"
    else:
        ntype = f"{family}.{'completed' if job.status == 'succeeded' else 'failed'}"
        title = TITLES.get(job.kind, {}).get(job.status) or f"{KIND_TEXT.get(job.kind, 'Job')} {job.status}"
    parts: List[str] = []
    if job.label:
        parts.append(scrub_label(job.label) or "")
    dur = _duration_text(job.duration_s)
    if job.status == "succeeded":
        if dur:
            parts.append(f"finished in {dur}")
    elif job.status == "failed":
        parts.append(FAIL_TEXT.get(job.fail_reason or "failed", FAIL_TEXT["failed"]).capitalize() + ".")
    body = " — ".join(p for p in parts if p) or None
    extra: Dict[str, Any] = {"job_id": job.id, "kind": job.kind, "status": job.status}
    if job.fail_reason:
        extra["fail_reason"] = job.fail_reason
    if job.kind in INFERENCE_SUB_KIND:
        extra["sub_kind"] = INFERENCE_SUB_KIND[job.kind]
    if job.duration_s is not None:
        extra["duration_ms"] = int(job.duration_s * 1000)
    return {"type": ntype, "title": title, "body": body, "extra": extra}


def _release_private_loop_client() -> Any:
    """The #761 pattern: a short-lived loop closes its own Redis client. Soft:
    the attribute exists only once that fix is merged; this code is correct
    before and after."""
    async def _release() -> None:
        try:
            from app.services import session_service
            release = getattr(getattr(session_service, "cache", None), "release_loop_client", None)
            if release is not None:
                await release()
        except Exception:  # noqa: BLE001
            pass
    return _release


def _ring_bell(job: Job) -> Optional[str]:
    """Write the one notification for a claimed job. Sync; runs the async
    writer on a private loop (no loop is running in a Celery task or a
    ``to_thread`` worker). Returns the notification id, or None."""
    text = notification_text(job)
    from app.services import notification_service

    async def _go() -> Optional[Dict[str, Any]]:
        try:
            return await notification_service.create_notification(
                job.user_id, text["type"], text["title"], text["body"],
                thread_id=job.thread_id, ref=f"job:{job.id}", extra=text["extra"],
                resource_type=job.resource_type, resource_id=job.resource_id,
            )
        finally:
            await _release_private_loop_client()()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        on_loop = False
    else:
        on_loop = True
    try:
        if on_loop:
            # A caller on a live loop should have used asyncio.to_thread; do not
            # block its loop with asyncio.run and do not fail its job: ring from
            # a private thread (the app/agents/avaloka_agent pattern) and say so.
            logger.warning("[jobs] finish(%s) called on a running event loop; ringing from a worker thread",
                           job.id[:8])
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                saved = pool.submit(asyncio.run, _go()).result()
        else:
            saved = asyncio.run(_go())
    except Exception as exc:  # noqa: BLE001
        logger.warning("[jobs] bell for %s failed: %s", job.id[:8], str(exc)[:160])
        return None
    return str(saved.get("id")) if isinstance(saved, dict) and saved.get("id") else None


# ---------------------------------------------------------------------------
# Scheduled runs: when does a recurring schedule ring the bell (ruling 3)
# ---------------------------------------------------------------------------

def scheduled_run_should_notify(*, one_off: bool, succeeded: bool, previous_succeeded: Optional[bool]) -> bool:
    """One-off runs always notify. A recurring schedule notifies on every
    failure, and on success only the first time (no previous run) and on a
    recovery (the previous run failed), so a five-minute schedule stays quiet."""
    if one_off or not succeeded:
        return True
    return previous_succeeded is not True
