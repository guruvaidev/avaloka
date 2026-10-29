"""Job records for Ray jobs: one row per submission in the Supabase ``ray_jobs`` table.

Why a table and not just Ray: Ray's job list mixes every user's jobs, lives on one
cluster, and is held in the head pod's memory, so a head restart forgets every
job. The Ray jobs API reads ownership and history from here and only asks Ray for
the live status and logs of a job.

Writes come from app/infra/ray_job_runner.py and are best-effort there (a failed
write never fails the job). Reads raise StoreUnavailable so the API can fall back
to Ray metadata when the table is missing or Supabase is down.

Schema: migrations/ray_jobs.sql. Table name override: AVALOKA_RAY_JOBS_TABLE.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

# Columns this module writes; anything else in a payload is dropped so a stray
# key (or a secret someone adds to the tags later) never reaches the table.
COLUMNS = (
    "submission_id", "user_id", "thread_id", "analysis_id", "kind",
    "dashboard_url", "source", "destination", "status", "message",
    "rows_written", "runtime_s", "created_at", "ended_at", "updated_at",
)


class StoreUnavailable(RuntimeError):
    """The ray_jobs table could not be read (missing table, Supabase down, no creds)."""


def _table_name() -> str:
    return os.getenv("AVALOKA_RAY_JOBS_TABLE", "ray_jobs")


def _table():
    from app.agents.sampling_persistence import get_supabase_client
    return get_supabase_client().table(_table_name())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k in COLUMNS}


def record_submission(row: Mapping[str, Any]) -> None:
    """Insert (or overwrite, on a resubmitted id) the record for a new job."""
    payload = _clean(row)
    if not payload.get("submission_id") or not payload.get("user_id"):
        raise ValueError("record_submission needs submission_id and user_id")
    payload.setdefault("created_at", _now())
    payload["updated_at"] = _now()
    _table().upsert(payload, on_conflict="submission_id").execute()


def record_update(submission_id: str, fields: Mapping[str, Any]) -> None:
    payload = _clean(fields)
    payload.pop("submission_id", None)
    payload.pop("user_id", None)  # ownership never changes after submission
    if not payload:
        return
    payload["updated_at"] = _now()
    _table().update(payload).eq("submission_id", submission_id).execute()


def get_for_user(user_id: str, submission_id: str) -> Optional[Dict[str, Any]]:
    """The caller's record for this job, or None (absent, or someone else's)."""
    try:
        res = (
            _table().select("*")
            .eq("submission_id", submission_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise StoreUnavailable(str(exc)) from exc
    rows = getattr(res, "data", None) or []
    return rows[0] if rows else None


def list_for_user(
    user_id: str,
    kind: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """The caller's jobs, newest first."""
    try:
        q = _table().select("*").eq("user_id", user_id)
        if kind:
            q = q.eq("kind", kind)
        if status:
            q = q.eq("status", status.upper())
        res = q.order("created_at", desc=True).limit(limit).execute()
    except Exception as exc:
        raise StoreUnavailable(str(exc)) from exc
    return list(getattr(res, "data", None) or [])
