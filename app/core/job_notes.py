"""The chat-thread outbox for finished background jobs.

A job that was started from a chat thread leaves one short, code-written
assistant note in that thread when it ends. The note is *queued* here by
whoever finishes the job (``job_ledger.finish`` → :func:`enqueue`, from the API,
a Celery worker or a mission thread) and *delivered* only by the API, only at a
turn boundary of that thread (:func:`flush`), so it never lands between a
question and its answer and is never written by two processes.

Rows live in ``public.job_chat_notes`` (migration 20261121110000):
``unique (job_id)`` makes a retried finish queue nothing twice;
``delivered_at`` + ``batch_id`` record the flush that delivered a note.

Delivery writes three things, in this order: the thread's model context (an
``AIMessage`` tagged ``avaloka_job_note``/``avaloka_system_context``, so the
planner reads it as system context and nothing that learns from assistant
turns counts it), one ``analysis_messages`` row for the UI transcript when the
thread has an ``analyses`` row, and the ``delivered_at`` mark. A crash between
the second and the third re-delivers once; ``output.job_note.batch_id`` lets
the UI drop the duplicate. The reverse order would lose notes, which is worse.

Nothing here calls a model. Text comes from ``job_ledger``'s closed
vocabularies plus the job's already-redacted label.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.core import job_ledger as ledger

logger = logging.getLogger("avaloka.jobs.notes")

TABLE = os.getenv("AVALOKA_JOB_CHAT_NOTES_TABLE", "job_chat_notes")
MESSAGES_TABLE = "analysis_messages"

#: More pending notes than this on one thread is a runaway producer: deliver a
#: single summary and mark the rest, the bell has every one of them.
MAX_LISTED = 3
MAX_PENDING = 50

NOTE_FLAG = "avaloka_job_note"
SYSTEM_CONTEXT_FLAG = "avaloka_system_context"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_MARK = {"succeeded": "✓", "failed": "✗", "cancelled": "–"}


def render_note(job: ledger.Job) -> str:
    """One line for one job, from the closed vocabularies only."""
    kind = ledger.KIND_TEXT.get(job.kind, "Background job")
    label = ledger.scrub_label(job.label)
    head = f"{kind} ({label})" if label else kind
    dur = ledger._duration_text(job.duration_s)
    if job.status == "succeeded":
        tail = f"finished{f' in {dur}' if dur else ''}."
    elif job.status == "cancelled":
        tail = "was cancelled."
    else:
        reason = ledger.FAIL_TEXT.get(job.fail_reason or "failed", ledger.FAIL_TEXT["failed"])
        tail = f"did not complete: {reason}."
    return f"{_MARK.get(job.status, '•')} {head} {tail}"


def compose(notes: List[Dict[str, Any]]) -> str:
    """One assistant message for one flush: up to MAX_LISTED lines, then a
    count. Bursts collapse into a single message (back-pressure)."""
    n = len(notes)
    if n == 0:
        return ""
    if n == 1:
        return notes[0]["text"]
    ok = sum(1 for x in notes if (x.get("status") or "") == "succeeded")
    bad = n - ok
    summary = f"{n} background jobs finished: {ok} succeeded, {bad} did not."
    listed = [x["text"] for x in notes[:MAX_LISTED]]
    rest = n - len(listed)
    lines = [summary, *listed]
    if rest > 0:
        lines.append(f"…and {rest} more — every one is in the notifications bell.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

class MemoryNotes:
    def __init__(self) -> None:
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.messages: List[Dict[str, Any]] = []
        self.analyses: Dict[str, str] = {}          # thread_id -> analysis_id
        self._lock = threading.Lock()

    def insert(self, row: Dict[str, Any]) -> bool:
        with self._lock:
            if any(r["job_id"] == row["job_id"] for r in self.rows.values()):
                return False
            rid = row.get("id") or str(uuid.uuid4())
            self.rows[rid] = {**row, "id": rid, "delivered_at": None, "batch_id": None}
            return True

    def pending(self, thread_id: str, limit: int) -> List[Dict[str, Any]]:
        with self._lock:
            rows = [dict(r) for r in self.rows.values() if r["thread_id"] == thread_id and r["delivered_at"] is None]
            rows.sort(key=lambda r: r["created_at"])
            return rows[:limit]

    def mark_delivered(self, ids: List[str], batch_id: str) -> int:
        with self._lock:
            n = 0
            for rid in ids:
                r = self.rows.get(rid)
                if r and r["delivered_at"] is None:
                    r["delivered_at"] = _now()
                    r["batch_id"] = batch_id
                    n += 1
            return n

    def analysis_for_thread(self, thread_id: str) -> Optional[str]:
        return self.analyses.get(thread_id)

    def insert_message(self, row: Dict[str, Any]) -> None:
        with self._lock:
            self.messages.append({**row, "id": str(uuid.uuid4())})


class SupabaseNotes:
    def __init__(self, client: Any = None) -> None:
        self._client = client

    def _c(self) -> Any:
        if self._client is None:
            from app.agents.sampling_persistence import get_supabase_client
            self._client = get_supabase_client()
        return self._client

    @staticmethod
    def _rows(res: Any) -> List[Dict[str, Any]]:
        data = getattr(res, "data", None)
        return list(data or [])

    def insert(self, row: Dict[str, Any]) -> bool:
        try:
            self._c().table(TABLE).insert(row).execute()
            return True
        except Exception as exc:  # noqa: BLE001
            text = str(exc).lower()
            if "duplicate" in text or "unique" in text or "23505" in text:
                return False
            raise

    def pending(self, thread_id: str, limit: int) -> List[Dict[str, Any]]:
        return self._rows(self._c().table(TABLE).select("*").eq("thread_id", thread_id)
                          .is_("delivered_at", "null").order("created_at").limit(limit).execute())

    def mark_delivered(self, ids: List[str], batch_id: str) -> int:
        if not ids:
            return 0
        res = (self._c().table(TABLE).update({"delivered_at": _now(), "batch_id": batch_id})
               .in_("id", ids).is_("delivered_at", "null").execute())
        return len(self._rows(res))

    def analysis_for_thread(self, thread_id: str) -> Optional[str]:
        rows = self._rows(self._c().table("analyses").select("id").eq("thread_id", thread_id).limit(1).execute())
        return str(rows[0]["id"]) if rows and rows[0].get("id") else None

    def insert_message(self, row: Dict[str, Any]) -> None:
        self._c().table(MESSAGES_TABLE).insert(row).execute()


_store: Any = None
_lock = threading.Lock()


def get_store() -> Any:
    global _store
    with _lock:
        if _store is None:
            _store = MemoryNotes() if os.getenv("AVALOKA_JOB_LEDGER", "").lower() == "memory" else SupabaseNotes()
        return _store


def set_store(store: Any) -> None:
    global _store
    with _lock:
        _store = store


# ---------------------------------------------------------------------------
# Queueing (any process)
# ---------------------------------------------------------------------------

def enqueue(job: ledger.Job) -> bool:
    """Queue the note for a finished job started from a chat thread. Sync.
    Returns True when a row was written; False when one already existed."""
    if not job.thread_id or not job.terminal:
        return False
    row = {
        "job_id": job.id,
        "user_id": job.user_id,
        "thread_id": job.thread_id,
        "analysis_id": job.analysis_id,
        "text": render_note(job),
        "created_at": _now(),
    }
    written = get_store().insert({k: v for k, v in row.items() if v is not None})
    logger.info("[jobs] note %s for %s %s", "queued" if written else "already queued", job.kind, job.id[:8])
    return written


# ---------------------------------------------------------------------------
# Delivery (API only, at a turn boundary)
# ---------------------------------------------------------------------------

_TURNS_IN_FLIGHT: Dict[str, int] = {}
_turns_lock = threading.Lock()


def turn_started(thread_id: str) -> None:
    with _turns_lock:
        _TURNS_IN_FLIGHT[thread_id] = _TURNS_IN_FLIGHT.get(thread_id, 0) + 1


def turn_finished(thread_id: str) -> None:
    with _turns_lock:
        n = _TURNS_IN_FLIGHT.get(thread_id, 0) - 1
        if n <= 0:
            _TURNS_IN_FLIGHT.pop(thread_id, None)
        else:
            _TURNS_IN_FLIGHT[thread_id] = n


def turn_in_flight(thread_id: str) -> bool:
    with _turns_lock:
        return _TURNS_IN_FLIGHT.get(thread_id, 0) > 0


MODEL_FRAME = "[Background job update — written by the system, not an assistant reply]"


def make_note_message(text: str, job_ids: List[str], batch_id: str):
    """The history form of a delivered batch: an AIMessage whose content opens
    with :data:`MODEL_FRAME` and whose kwargs carry the tags.

    Why not a SystemMessage: the planner sends ``[system prompt, *history]`` and
    langchain-anthropic rejects a system message anywhere but first, so a
    mid-history SystemMessage would break those providers. The frame in the
    content tells every model what this is; the tags tell our own code
    (planner/coder last-turn scans, the memory plane, feedback) to skip it.
    """
    from langchain_core.messages import AIMessage
    return AIMessage(content=f"{MODEL_FRAME}\n{text}", additional_kwargs={
        NOTE_FLAG: True, SYSTEM_CONTEXT_FLAG: True, "job_ids": list(job_ids), "batch_id": batch_id,
    })


def is_job_note(message: Any) -> bool:
    ak = getattr(message, "additional_kwargs", None) or {}
    return bool(ak.get(NOTE_FLAG))


def flush_sync(thread_id: str, *, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Deliver a thread's pending notes: returns the batch, or None when there
    is nothing to deliver or a turn is in flight (the caller retries at the
    next boundary). Sync; the API calls it through ``asyncio.to_thread`` and
    then appends the returned message to the thread's history itself, on its
    own loop (history is the API's single-writer concern, not this module's).
    """
    if not thread_id or turn_in_flight(thread_id):
        return None
    store = get_store()
    pending = store.pending(thread_id, MAX_PENDING + 1)
    if not pending:
        return None
    if user_id is not None:
        pending = [r for r in pending if str(r.get("user_id")) == str(user_id)]
        if not pending:
            return None
    overflow = len(pending) > MAX_PENDING
    pending = pending[:MAX_PENDING]
    for r in pending:
        r.setdefault("status", _status_from_text(r.get("text") or ""))
    text = compose(pending)
    if overflow:
        text += "\nMore jobs finished than can be listed here; the notifications bell has all of them."
    batch_id = str(uuid.uuid4())
    job_ids = [str(r["job_id"]) for r in pending]
    analysis_id = pending[0].get("analysis_id") or store.analysis_for_thread(thread_id)
    message_row: Optional[Dict[str, Any]] = None
    if analysis_id:
        message_row = {
            "analysis_id": str(analysis_id),
            "role": "assistant",
            "content": text,
            "output": {"job_note": {"batch_id": batch_id, "job_ids": job_ids,
                                    "statuses": [r.get("status") for r in pending]}},
            "author_id": None,
        }
        try:
            store.insert_message(message_row)
        except Exception as exc:  # noqa: BLE001 - the transcript row is best-effort; the history still gets it
            logger.warning("[jobs] analysis_messages row for thread %s not written: %s", thread_id[:8], str(exc)[:160])
            message_row = None
    marked = store.mark_delivered([str(r["id"]) for r in pending], batch_id)
    logger.info("[jobs] delivered %d note(s) to thread %s batch=%s", marked, thread_id[:8], batch_id[:8])
    return {
        "batch_id": batch_id, "thread_id": thread_id, "text": text, "job_ids": job_ids,
        "count": len(pending), "analysis_id": str(analysis_id) if analysis_id else None,
        "transcript_written": message_row is not None,
        "created_at": _now(),
    }


def _status_from_text(text: str) -> str:
    if text.startswith("✓"):
        return "succeeded"
    if text.startswith("–"):
        return "cancelled"
    return "failed"


def pending_count(thread_id: str) -> int:
    try:
        return len(get_store().pending(thread_id, MAX_PENDING + 1))
    except Exception:  # noqa: BLE001
        return 0


def without_job_notes(messages: List[Any]) -> List[Any]:
    """For anything that learns from, or scans for, assistant turns (the
    planner's last-AI-turn pick, the coder's last-instruction pick, the memory
    plane, feedback): the same list with job notes removed."""
    return [m for m in messages if not is_job_note(m)]
