"""In-app notifications for background work (design doc D4).

Writes to the EXISTING Supabase ``public.notifications`` table, the one the
frontend bell already reads directly through Supabase (row-level security:
``recipient_id`` = the caller's ``profiles.id``). Its columns:

    id, recipient_id, actor_id, type, title, body, link, resource_type,
    resource_id, metadata, created_at, updated_at, read_at

``recipient_id`` and ``actor_id`` are ``profiles.id``, not the auth uid in the
JWT, so every entry point here translates the auth uid first.

Writes use the service-role client, which bypasses row-level security, so
every query filters on ``recipient_id`` explicitly. Nothing here raises to the
caller: a failed notification must never fail the work it reports on.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("avaloka.notifications")

TABLE = "notifications"

# Same dotted style as the rows the frontend already writes ("analysis.completed").
TYPES = frozenset({
    "analysis.completed", "analysis.failed",
    "training.completed", "training.failed",
    "batch_job.completed", "batch_job.failed",
    "insights.generated", "insights.failed",
})

_MAX_TITLE = 200
_MAX_BODY = 1000
_MAX_LIST = 100

_profile_cache: Dict[str, str] = {}


def _client():
    # Imported lazily so this module loads without Supabase configured.
    from app.agents.sampling_persistence import get_supabase_client
    return get_supabase_client()


def _clip(text: Optional[str], limit: int) -> Optional[str]:
    if text is None:
        return None
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first(res: Any) -> Optional[Dict[str, Any]]:
    data = getattr(res, "data", None) or []
    return data[0] if data else None


async def profile_id_for(user_id: Optional[str]) -> Optional[str]:
    """profiles.id for an auth uid (what the JWT carries)."""
    if not user_id:
        return None
    if user_id in _profile_cache:
        return _profile_cache[user_id]

    def _q():
        return (_client().table("profiles").select("id")
                .eq("user_id", user_id).limit(1).execute())

    try:
        row = _first(await asyncio.to_thread(_q))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[notify] profile lookup failed for %s: %s", str(user_id)[:8], exc)
        return None
    if row and row.get("id"):
        _profile_cache[user_id] = str(row["id"])
        return _profile_cache[user_id]
    logger.warning("[notify] no profiles row for auth uid %s", str(user_id)[:8])
    return None


async def _analysis_id_for_thread(thread_id: Optional[str]) -> Optional[str]:
    """The analyses row for a chat thread, so the notification can link to it."""
    if not thread_id:
        return None

    def _q():
        return (_client().table("analyses").select("id")
                .eq("thread_id", thread_id).limit(1).execute())

    try:
        row = _first(await asyncio.to_thread(_q))
    except Exception as exc:  # noqa: BLE001
        logger.info("[notify] analyses lookup failed for thread %s: %s", thread_id[:8], exc)
        return None
    return str(row["id"]) if row and row.get("id") else None


async def _already_sent(recipient_id: str, ref: str) -> bool:
    def _q():
        return (_client().table(TABLE).select("id")
                .eq("recipient_id", recipient_id)
                .eq("metadata->>ref", ref)
                .limit(1).execute())

    try:
        return _first(await asyncio.to_thread(_q)) is not None
    except Exception:  # noqa: BLE001 - if the check fails, writing once more is fine
        return False


async def create_notification(
    user_id: Optional[str],
    ntype: str,
    title: str,
    body: Optional[str] = None,
    *,
    thread_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    ref: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Write one notification for the user with this auth uid.

    ``ref`` identifies the work (deferred turn id, job id); a second call with
    the same ref for the same user writes nothing. Returns the row, or None.
    """
    if ntype not in TYPES:
        logger.warning("[notify] unknown type %r; not written", ntype)
        return None
    recipient = await profile_id_for(user_id)
    if not recipient:
        return None
    if ref and await _already_sent(recipient, str(ref)):
        logger.info("[notify] %s ref=%s already recorded", ntype, ref)
        return None

    analysis_id = await _analysis_id_for_thread(thread_id)
    metadata: Dict[str, Any] = {
        k: v for k, v in {
            "ref": str(ref) if ref else None,
            "thread_id": thread_id,
            "dataset_id": dataset_id,
            **(extra or {}),
        }.items() if v is not None
    }
    row: Dict[str, Any] = {
        "recipient_id": recipient,
        "actor_id": recipient,          # the user's own work, as in existing rows
        "type": ntype,
        "title": _clip(title, _MAX_TITLE) or ntype,
        "body": _clip(body, _MAX_BODY),
        "resource_type": "analysis",
        "link": f"/analysis?aid={analysis_id}" if analysis_id else "/analysis",
        "metadata": metadata,
    }
    if analysis_id:
        row["resource_id"] = analysis_id

    def _insert():
        return _client().table(TABLE).insert(row).execute()

    try:
        saved = _first(await asyncio.to_thread(_insert))
        logger.info("[notify] %s for profile %s (ref=%s, analysis=%s)",
                    ntype, recipient[:8], ref, analysis_id)
        return saved or row
    except Exception as exc:  # noqa: BLE001 - must never break the caller
        logger.warning("[notify] could not write %s for profile %s: %s", ntype, recipient[:8], exc)
        return None


async def list_notifications(
    user_id: str, unread_only: bool = False, limit: int = 20,
) -> Tuple[List[Dict[str, Any]], int]:
    """(newest notifications, unread count) for the user with this auth uid."""
    recipient = await profile_id_for(user_id)
    if not recipient:
        return [], 0
    limit = max(1, min(int(limit or 20), _MAX_LIST))

    def _items():
        q = (_client().table(TABLE)
             .select("id,type,title,body,link,resource_type,resource_id,metadata,created_at,read_at")
             .eq("recipient_id", recipient))
        if unread_only:
            q = q.is_("read_at", "null")
        return q.order("created_at", desc=True).limit(limit).execute()

    def _count():
        return (_client().table(TABLE)
                .select("id", count="exact")
                .eq("recipient_id", recipient)
                .is_("read_at", "null")
                .limit(1)
                .execute())

    items_res, count_res = await asyncio.gather(
        asyncio.to_thread(_items), asyncio.to_thread(_count)
    )
    items = getattr(items_res, "data", None) or []
    unread = getattr(count_res, "count", None)
    if unread is None:
        unread = sum(1 for i in items if not i.get("read_at"))
    return items, int(unread)


async def mark_read(user_id: str, notification_id: str) -> bool:
    """Mark one of the user's notifications read. False if it is not theirs."""
    recipient = await profile_id_for(user_id)
    if not recipient:
        return False

    def _update():
        return (_client().table(TABLE)
                .update({"read_at": _now_iso(), "updated_at": _now_iso()})
                .eq("id", notification_id)
                .eq("recipient_id", recipient)
                .is_("read_at", "null")
                .execute())

    if getattr(await asyncio.to_thread(_update), "data", None):
        return True

    # Already read counts as success; someone else's row does not.
    def _exists():
        return (_client().table(TABLE).select("id")
                .eq("id", notification_id).eq("recipient_id", recipient)
                .limit(1).execute())

    return _first(await asyncio.to_thread(_exists)) is not None


async def mark_all_read(user_id: str) -> int:
    """Mark every unread notification of the user read. Returns how many."""
    recipient = await profile_id_for(user_id)
    if not recipient:
        return 0

    def _update():
        return (_client().table(TABLE)
                .update({"read_at": _now_iso(), "updated_at": _now_iso()})
                .eq("recipient_id", recipient)
                .is_("read_at", "null")
                .execute())

    return len(getattr(await asyncio.to_thread(_update), "data", None) or [])