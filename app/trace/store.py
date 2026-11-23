"""Where finished turns live: Redis, next to the thread they belong to.

``thread:{id}:trace`` holds the thread's last :data:`MAX_TURNS` turns as one
JSON list, with the session TTL. ``user:{uid}:traces`` indexes the threads a
user has traces for, so the aggregate agent view does not scan Redis.

The trace stays in the cluster. Deleting the thread's history deletes it too
(``app.api.helpers.delete_thread_history``).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.services import session_service

logger = logging.getLogger(__name__)

MAX_TURNS = 25
#: One turn's ceiling. Generated code and result previews are already clipped by
#: the describers; this guards against a pathological turn filling Redis.
MAX_TURN_BYTES = 1_500_000


def _k_trace(thread_id: str) -> str:
    return f"thread:{thread_id}:trace"


def _k_user_traces(user_id: str) -> str:
    return f"user:{user_id}:traces"


def _ttl() -> int:
    return int(getattr(session_service.settings, "session_ttl_seconds", 0) or 7 * 24 * 3600)


async def load_turns(thread_id: str) -> List[Dict[str, Any]]:
    cache = session_service.cache
    if not cache:
        return []
    try:
        raw = await cache.get(_k_trace(thread_id))
    except Exception as e:  # noqa: BLE001
        session_service._note_cache_failure(f"load_trace({thread_id})", e)
        return []
    if not raw:
        return []
    try:
        turns = json.loads(raw)
    except Exception:  # noqa: BLE001
        return []
    return turns if isinstance(turns, list) else []


def _without_credentials(value: Any, key: str = "") -> Any:
    """``value`` with credentials typed into chat redacted: the question in full, a
    sealed token anywhere else (so generated code is shown as it was)."""
    from app.core.typed_secrets import REDACTED, TOKEN_RE, redact_typed_secrets
    if isinstance(value, str):
        return redact_typed_secrets(value)[0] if key in ("question", "prompt") else TOKEN_RE.sub(REDACTED, value)
    if isinstance(value, list):
        return [_without_credentials(v, key) for v in value]
    if isinstance(value, dict):
        return {k: _without_credentials(v, str(k)) for k, v in value.items()}
    return value


async def append_turn(thread_id: str, user_id: Optional[str], turn: Dict[str, Any]) -> bool:
    """Persist one finished turn. Returns whether it was stored. Never raises."""
    cache = session_service.cache
    if not cache or not turn or not turn.get("steps"):
        return False
    try:
        turn = _without_credentials(turn)
        encoded = json.dumps(turn, default=str)
    except Exception:  # noqa: BLE001
        logger.debug("[trace] turn for %s is not serialisable", thread_id, exc_info=True)
        return False
    if len(encoded) > MAX_TURN_BYTES:
        logger.warning("[trace] turn %s for thread %s is %d bytes; storing its summary only",
                       turn.get("turn_id"), thread_id, len(encoded))
        turn = {**turn, "steps": [], "truncated": True}
    turns = [t for t in await load_turns(thread_id) if t.get("turn_id") != turn.get("turn_id")]
    turns = (turns + [turn])[-MAX_TURNS:]
    try:
        await cache.set(_k_trace(thread_id), json.dumps(turns, default=str), ex=_ttl())
    except Exception as e:  # noqa: BLE001
        session_service._note_cache_failure(f"append_trace({thread_id})", e)
        return False
    if user_id:
        # The index only feeds the cross-thread aggregate; losing it must not
        # lose the trace that was just written.
        try:
            await cache.sadd(_k_user_traces(user_id), thread_id)
            await cache.expire(_k_user_traces(user_id), _ttl())
        except Exception:  # noqa: BLE001
            logger.debug("[trace] could not index %s for its user", thread_id, exc_info=True)
    return True


async def user_trace_threads(user_id: str) -> List[str]:
    cache = session_service.cache
    if not cache or not user_id:
        return []
    try:
        return sorted(await cache.smembers(_k_user_traces(user_id)) or [])
    except Exception as e:  # noqa: BLE001
        session_service._note_cache_failure(f"user_traces({user_id})", e)
        return []


async def delete_trace(thread_id: str) -> None:
    cache = session_service.cache
    if not cache:
        return
    try:
        await cache.delete(_k_trace(thread_id))
    except Exception as e:  # noqa: BLE001
        session_service._note_cache_failure(f"delete_trace({thread_id})", e)
