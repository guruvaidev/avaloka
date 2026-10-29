"""What the Ray jobs API does, without FastAPI: app/api/ray_jobs.py is a thin router
over these functions, so this logic is testable without a web stack.

Rules every function follows:
- A user only ever sees their own jobs. Ownership comes from the ray_jobs table
  (app/infra/ray_jobs_store.py); if the table cannot be read, from the
  ``user_id`` Ray job metadata. Someone else's job id answers 404, the same as a
  missing one, so ids cannot be probed.
- The Ray cluster for a job comes from its record, never from the request.
- The Ray dashboard URL is never returned to the browser.
- All functions block (Ray client, Supabase); the router runs them in a thread.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.infra import ray_job_runner, ray_jobs_store
from app.infra.ray_job_runner import _status_name

logger = logging.getLogger(__name__)

TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED"}
_SUBMISSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_REFRESH_LIMIT = 20  # live status lookups per list call; the rest show their stored status

PUBLIC_FIELDS = (
    "submission_id", "kind", "status", "message", "thread_id", "analysis_id",
    "source", "destination", "rows_written", "runtime_s", "created_at", "ended_at",
)


class RayJobsError(Exception):
    def __init__(self, status_code: int, detail: Any):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def default_dashboard_url() -> str:
    return (os.getenv("RAY_DASHBOARD_URL") or "").strip().rstrip("/")


def public_job(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: rec.get(k) for k in PUBLIC_FIELDS}


def _ms_to_iso(ms: Any) -> Optional[str]:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat() if ms else None
    except (TypeError, ValueError, OSError):
        return None


def _from_job_info(info: Any) -> Dict[str, Any]:
    """Fields Ray knows live about a job (JobDetails / JobInfo)."""
    start, end = getattr(info, "start_time", None), getattr(info, "end_time", None)
    out = {
        "status": _status_name(getattr(info, "status", None)) or None,
        "message": getattr(info, "message", None),
        "ended_at": _ms_to_iso(end),
    }
    if start and end:
        out["runtime_s"] = round((int(end) - int(start)) / 1000, 2)
    return {k: v for k, v in out.items() if v is not None}


def _client(url: str):
    # JobSubmissionClient() calls the cluster with no timeout while holding the
    # runner's client lock; a dead cluster must fail in seconds, not stall every request.
    ray_job_runner._preflight_dashboard(url)
    return ray_job_runner.make_job_client(url)


# ---------------------------------------------------------------- ownership

def _record_from_ray(info: Any, url: str) -> Dict[str, Any]:
    meta = dict(getattr(info, "metadata", None) or {})
    rec = {k: meta.get(k) for k in ("user_id", "thread_id", "analysis_id", "kind", "source", "destination")}
    rec.update(
        submission_id=getattr(info, "submission_id", None),
        dashboard_url=url,
        created_at=_ms_to_iso(getattr(info, "start_time", None)),
        **_from_job_info(info),
    )
    return rec


def _owned_record(user_id: str, submission_id: str) -> Dict[str, Any]:
    if not _SUBMISSION_ID_RE.match(submission_id or ""):
        raise RayJobsError(404, "Job not found")
    try:
        rec = ray_jobs_store.get_for_user(user_id, submission_id)
    except ray_jobs_store.StoreUnavailable as exc:
        logger.warning("[ray-jobs] job records unavailable, using Ray metadata: %s", exc)
        rec = _owned_record_from_ray(user_id, submission_id)
    if not rec:
        raise RayJobsError(404, "Job not found")
    return rec


def _owned_record_from_ray(user_id: str, submission_id: str) -> Optional[Dict[str, Any]]:
    url = default_dashboard_url()
    if not url:
        return None
    try:
        info = _client(url).get_job_info(submission_id)
    except Exception:
        return None
    if (getattr(info, "metadata", None) or {}).get("user_id") != user_id:
        return None
    return _record_from_ray(info, url)


def _refresh(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay Ray's live status on a record that has not finished yet."""
    if (rec.get("status") or "").upper() in TERMINAL or not rec.get("dashboard_url"):
        return rec
    try:
        live = _from_job_info(_client(rec["dashboard_url"]).get_job_info(rec["submission_id"]))
    except Exception as exc:
        logger.info("[ray-jobs] live status unavailable for %s: %s", rec.get("submission_id"), exc)
        return rec
    merged = {**rec, **live}
    if (live.get("status") or "") in TERMINAL:
        # The runner normally records the end itself; this covers a backend that
        # restarted while the job ran.
        try:
            ray_jobs_store.record_update(rec["submission_id"], live)
        except Exception as exc:
            logger.info("[ray-jobs] could not store final status for %s: %s", rec.get("submission_id"), exc)
    return merged


# ---------------------------------------------------------------- endpoints

def cluster_status() -> Dict[str, Any]:
    url = default_dashboard_url()
    if not url:
        return {"configured": False, "reachable": False}
    timeout = float(os.getenv("RAY_DASHBOARD_CONNECT_TIMEOUT_S", "5"))
    try:
        with urllib.request.urlopen(f"{url}/api/version", timeout=timeout) as resp:
            version = json.loads(resp.read().decode() or "{}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.info("[ray-jobs] Ray dashboard unreachable: %s", exc)
        return {"configured": True, "reachable": False}
    out: Dict[str, Any] = {
        "configured": True,
        "reachable": True,
        "ray_version": version.get("ray_version"),
        "nodes": None,
    }
    try:  # best effort: node count from the dashboard's node summary
        with urllib.request.urlopen(f"{url}/nodes?view=summary", timeout=timeout) as resp:
            summary = (json.loads(resp.read().decode() or "{}").get("data") or {}).get("summary")
        if isinstance(summary, list):
            out["nodes"] = len(summary)
    except Exception as exc:
        logger.debug("[ray-jobs] node summary unavailable: %s", exc)
    return out


def list_jobs(user_id: str, kind: Optional[str] = None, status: Optional[str] = None,
              limit: int = 50) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 200))
    try:
        recs = ray_jobs_store.list_for_user(user_id, kind=kind, status=status, limit=limit)
    except ray_jobs_store.StoreUnavailable as exc:
        logger.warning("[ray-jobs] job records unavailable, listing from Ray metadata: %s", exc)
        recs = _list_from_ray(user_id, kind, status, limit)
    out = []
    for i, rec in enumerate(recs):
        out.append(public_job(_refresh(rec) if i < _REFRESH_LIMIT else rec))
    return out


def _list_from_ray(user_id: str, kind: Optional[str], status: Optional[str], limit: int) -> List[Dict[str, Any]]:
    url = default_dashboard_url()
    if not url:
        return []
    try:
        jobs = _client(url).list_jobs()
    except Exception as exc:
        raise RayJobsError(503, f"Ray cluster unavailable: {exc}") from exc
    recs = [
        _record_from_ray(j, url) for j in jobs
        if (getattr(j, "metadata", None) or {}).get("user_id") == user_id
    ]
    if kind:
        recs = [r for r in recs if r.get("kind") == kind]
    if status:
        recs = [r for r in recs if (r.get("status") or "") == status.upper()]
    recs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return recs[:limit]


def get_job(user_id: str, submission_id: str) -> Dict[str, Any]:
    return public_job(_refresh(_owned_record(user_id, submission_id)))


def get_logs(user_id: str, submission_id: str, tail: int = 500) -> Dict[str, Any]:
    rec = _owned_record(user_id, submission_id)
    tail = max(1, min(int(tail), 5000))
    try:
        text = _client(rec["dashboard_url"]).get_job_logs(submission_id) or ""
    except Exception as exc:
        if "does not exist" in str(exc).lower() or "404" in str(exc):
            raise RayJobsError(410, "Logs are no longer on the Ray cluster (it was restarted or the job expired).") from exc
        raise RayJobsError(503, "Ray cluster unavailable") from exc
    lines = text.splitlines()
    return {
        "submission_id": submission_id,
        "lines": lines[-tail:],
        "truncated": len(lines) > tail,
    }


def stop_job(user_id: str, submission_id: str) -> Dict[str, Any]:
    rec = _refresh(_owned_record(user_id, submission_id))
    if (rec.get("status") or "").upper() in TERMINAL:
        raise RayJobsError(409, f"Job already {rec['status']}")
    try:
        client = _client(rec["dashboard_url"])
        stopped = bool(client.stop_job(submission_id))
        new_status = _status_name(client.get_job_status(submission_id))
    except Exception as exc:
        raise RayJobsError(503, "Ray cluster unavailable") from exc
    try:
        ray_jobs_store.record_update(submission_id, {"status": new_status, "message": "Stopped by the user"})
    except Exception as exc:
        logger.info("[ray-jobs] could not store stop for %s: %s", submission_id, exc)
    return {"submission_id": submission_id, "stopped": stopped, "status": new_status}
