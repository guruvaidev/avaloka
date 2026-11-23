"""Where swarm missions live: the running recorder in memory, the finished
document in Redis when there is one.

A mission runs in a background thread of this process, so its recorder *is*
the live record and the registry here is the only place that can serve
``records?after=`` while it runs. When the mission finishes, its document is
written to Redis as ``swarm:{mission_id}`` with the session TTL, indexed under
``user:{uid}:swarms`` so listing does not scan Redis, following
:mod:`app.trace.store`.

Without Redis (the laptop demo) the registry is all there is: a mission is
served from memory and is lost when the process restarts. The document says
so (``persistence: "memory"``).

The registry is capped at :data:`MAX_MISSIONS`; when full, the oldest
*finished* mission is evicted (a running one is never dropped mid-flight).
Nothing here raises: Redis being down, slow or full degrades to memory only.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services import session_service
from avaloka.swarm_trace import SwarmTraceRecorder, is_finished

logger = logging.getLogger(__name__)

MAX_MISSIONS = 64
#: One mission document's ceiling in Redis. Clone outputs are one-line labels
#: and metadata is small, so a 2-clone demo is a few tens of KB; this guards
#: against a pathological mission filling Redis.
MAX_DOC_BYTES = 1_500_000


def _k_swarm(mission_id: str) -> str:
    return f"swarm:{mission_id}"


def _k_user_swarms(user_id: str) -> str:
    return f"user:{user_id}:swarms"


def _ttl() -> int:
    return int(getattr(session_service.settings, "session_ttl_seconds", 0) or 7 * 24 * 3600)


@dataclass
class Entry:
    """One mission this process started."""

    mission_id: str
    user_id: str
    recorder: SwarmTraceRecorder
    output_dir: str
    created: float = field(default_factory=time.time)
    persisted: bool = False
    thread: Optional[threading.Thread] = None
    error_class: Optional[str] = None

    @property
    def status(self) -> str:
        return self.recorder.mission(include_records=False)["status"]

    @property
    def finished(self) -> bool:
        return is_finished(self.status)


class Registry:
    """In-process, thread-safe map of mission id to its live recorder."""

    def __init__(self, max_missions: int = MAX_MISSIONS) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, Entry] = {}
        self.max_missions = max_missions

    def add(self, entry: Entry) -> None:
        with self._lock:
            if len(self._entries) >= self.max_missions:
                oldest = next((e for e in self._entries.values() if e.finished), None)
                if oldest is not None:
                    del self._entries[oldest.mission_id]
            self._entries[entry.mission_id] = entry

    def get(self, mission_id: str) -> Optional[Entry]:
        with self._lock:
            return self._entries.get(mission_id)

    def for_user(self, user_id: str) -> List[Entry]:
        with self._lock:
            return [e for e in self._entries.values() if e.user_id == user_id]

    def remove(self, mission_id: str) -> Optional[Entry]:
        with self._lock:
            return self._entries.pop(mission_id, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


registry = Registry()


# ---------------------------------------------------------------- persistence
def has_redis() -> bool:
    return bool(session_service.cache)


async def persist(entry: Entry) -> bool:
    """Write a finished mission's document to Redis. Never raises."""
    cache = session_service.cache
    if not cache or not entry.finished:
        return False
    doc = entry.recorder.mission()
    doc["persistence"] = "redis"
    try:
        encoded = json.dumps({"owner": entry.user_id, "mission": doc}, default=str)
    except Exception:  # noqa: BLE001
        logger.debug("[swarm] mission %s is not serialisable", entry.mission_id, exc_info=True)
        return False
    if len(encoded) > MAX_DOC_BYTES:
        logger.warning("[swarm] mission %s is %d bytes; storing it without its records",
                       entry.mission_id, len(encoded))
        doc = {**doc, "records": [], "truncated": True}
        encoded = json.dumps({"owner": entry.user_id, "mission": doc}, default=str)
    try:
        await cache.set(_k_swarm(entry.mission_id), encoded, ex=_ttl())
    except Exception as e:  # noqa: BLE001
        session_service._note_cache_failure(f"persist_swarm({entry.mission_id})", e)
        return False
    try:
        await cache.sadd(_k_user_swarms(entry.user_id), entry.mission_id)
        await cache.expire(_k_user_swarms(entry.user_id), _ttl())
    except Exception:  # noqa: BLE001
        logger.debug("[swarm] could not index %s for its user", entry.mission_id, exc_info=True)
    entry.persisted = True
    return True


async def load(mission_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """The persisted document, only if ``user_id`` owns it. Never raises."""
    cache = session_service.cache
    if not cache:
        return None
    try:
        raw = await cache.get(_k_swarm(mission_id))
    except Exception as e:  # noqa: BLE001
        session_service._note_cache_failure(f"load_swarm({mission_id})", e)
        return None
    if not raw:
        return None
    try:
        wrapped = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(wrapped, dict) or wrapped.get("owner") != user_id:
        return None
    doc = wrapped.get("mission")
    return doc if isinstance(doc, dict) else None


async def user_missions(user_id: str) -> List[str]:
    cache = session_service.cache
    if not cache or not user_id:
        return []
    try:
        members = await cache.smembers(_k_user_swarms(user_id)) or []
    except Exception as e:  # noqa: BLE001
        session_service._note_cache_failure(f"user_swarms({user_id})", e)
        return []
    return sorted(m.decode() if isinstance(m, (bytes, bytearray)) else str(m) for m in members)


async def delete(mission_id: str, user_id: str) -> None:
    cache = session_service.cache
    if not cache:
        return
    try:
        await cache.delete(_k_swarm(mission_id))
        await cache.srem(_k_user_swarms(user_id), mission_id)
    except Exception as e:  # noqa: BLE001
        session_service._note_cache_failure(f"delete_swarm({mission_id})", e)
