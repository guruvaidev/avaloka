from __future__ import annotations
import asyncio
import os
import re
import time
import json
import logging
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from decimal import Decimal
from datetime import datetime, date

from app.core.settings import Settings
from app.core.cache import ICache, RedisCache

logger = logging.getLogger("avaloka.session")

# Load settings once here
settings = Settings()

# This will be set by server.py during startup
cache: Optional[ICache] = None

# ---- Cache circuit breaker state ----
CACHE_DEGRADED_UNTIL = 0.0
CACHE_DEGRADE_SECONDS = float(os.getenv("CACHE_DEGRADE_SECONDS", "15"))
_last_cache_warn = 0.0
_WARN_EVERY = 30.0

# How long a session WRITE may spend before we give up on it.
#
# Reads answer a slow Redis by returning None and recomputing. A write has no
# such luxury: giving up loses the turn's state. So the breaker cannot simply
# skip writes the way it skips reads — instead it shortens their deadline.
#
# Healthy: a generous budget, because a successful slow write still beats a
# lost one. Degraded: ~1s, because redis-py's own connect/retry budget
# (socket_connect_timeout 2.5s, socket_timeout 3.0s, retry_on_timeout) can
# otherwise stall a request for over ten seconds and still lose the data —
# 13s was measured against a brownout at localhost:6380.
SESSION_SAVE_TIMEOUT = float(os.getenv("SESSION_SAVE_TIMEOUT_SECONDS", "5"))
SESSION_SAVE_TIMEOUT_DEGRADED = float(
    os.getenv("SESSION_SAVE_TIMEOUT_DEGRADED_SECONDS", "1")
)

# Infrastructure failures we expect during a brownout. A stack trace for one of
# these is noise -- the interesting fact is "Redis is unwell and a session was
# lost", not the twelve frames inside redis-py that say so. Anything NOT in
# here is a genuine bug and keeps its traceback.
try:  # redis is a hard dependency, but never let logging config break imports
    from redis.exceptions import ConnectionError as _RedisConnectionError
    from redis.exceptions import RedisError as _RedisError
    from redis.exceptions import TimeoutError as _RedisTimeoutError
    _EXPECTED_CACHE_FAILURES = (
        asyncio.TimeoutError, _RedisTimeoutError, _RedisConnectionError, OSError,
    )
except Exception:  # pragma: no cover - redis missing entirely
    _RedisError = ()
    _EXPECTED_CACHE_FAILURES = (asyncio.TimeoutError, OSError)

def _cache_unhealthy() -> bool:
    return time.time() < CACHE_DEGRADED_UNTIL

def _note_cache_failure(where: str, err: Exception):
    global CACHE_DEGRADED_UNTIL, _last_cache_warn
    CACHE_DEGRADED_UNTIL = time.time() + CACHE_DEGRADE_SECONDS
    if time.time() - _last_cache_warn > _WARN_EVERY:
        logger.warning("cache degraded (%s): %s", where, err)
        _last_cache_warn = time.time()

async def _mget_safe(keys: List[str]) -> List[Optional[str]]:
    if not keys or not cache:
        return [None] * len(keys)
    if hasattr(cache, "mget"):
        try:
            return await cache.mget(keys)  # type: ignore[attr-defined]
        except Exception as e:
            _note_cache_failure("mget", e)
    vals: List[Optional[str]] = []
    for k in keys:
        try:
            vals.append(await cache.get(k))
        except Exception as e:
            _note_cache_failure(f"get({k})", e)
            vals.append(None)
    return vals

# ---------- JSON-safe helpers ----------
def _json_default(o):
    """Serialize common non-JSON types safely."""
    try:
        import pandas as pd
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
    except Exception:
        pass
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, Decimal)):
        return float(o)
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", "replace")
    return str(o)

def _jsonify(obj):
    """Deep-convert structures to JSON-safe primitives."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return _json_default(obj)

# ---------- Redis key helpers ----------
def _k_session(sid: str) -> str: return f"sess:{sid}"
def _k_user_sessions(uid: str) -> str: return f"user:{uid}:sids"
def _k_thread_session(tid: str) -> str: return f"thread:{tid}:sid"
def _k_user_dataset(uid: str, dsid: str) -> str: return f"user:{uid}:ds:{dsid}"

# Bulk sample data kept out of the session blob: re-serialising it every turn
# pushed the write past its deadline (24.8MB, 15.6MB of it portfolio_samples).
# Safe to drop -- _resolve_selected_sample_rows falls back to load_portfolio().
_BULK_SESSION_FIELDS = ("portfolio_samples",)


def _shed_bulk_fields(safe: Dict[str, Any]) -> Dict[str, Any]:
    """Drop bulk payloads from the write copy; the caller's dict keeps every field."""
    if not isinstance(safe, dict):
        return safe
    present = [f for f in _BULK_SESSION_FIELDS if safe.get(f)]
    if not present:
        return safe
    trimmed = {k: v for k, v in safe.items() if k not in present}
    trimmed["_shed_fields"] = sorted(present)
    return trimmed


# ---------- Session primitives ----------
async def save_session(sid: str, data: Dict[str, Any]) -> bool:
    """Persist session; return True on success, False otherwise.

    Participates in the cache circuit breaker in BOTH directions, which it
    previously did in neither:

    * It **reports** failures via :func:`_note_cache_failure`. Before this only
      reads did, so a Redis brownout first seen by a write stayed invisible to
      :func:`get_session`, which kept paying full timeouts to discover the same
      thing.
    * It **respects** an open breaker by shortening its deadline rather than by
      skipping. Skipping a write is not graceful degradation, it is silent data
      loss; but blocking a request for 13s to lose the data anyway is worse
      than losing it in 1s.

    Callers should treat ``False`` as "this turn's state is not durable".
    """
    if not cache:
        return False
    uid = data.get("user_id")
    if not uid:
        logger.warning("Refusing to save session without user_id: %s", sid)
        return False

    degraded = _cache_unhealthy()
    budget = SESSION_SAVE_TIMEOUT_DEGRADED if degraded else SESSION_SAVE_TIMEOUT
    try:
        safe = _shed_bulk_fields(_jsonify(data))
        async with asyncio.timeout(budget):
            await cache.set(_k_session(sid), json.dumps(safe, default=_json_default), ex=settings.session_ttl_seconds)
            await cache.sadd(_k_user_sessions(uid), sid)
            # Write-through dataset->session index so chat-turn resolution is a
            # single small GET instead of MGET-ing every session blob the user owns
            # (measured ~1GB / 7s for 62 sessions — always beyond the 3s socket
            # timeout, which tripped the cache circuit and 400'd whole threads).
            dsid = data.get("dataset_id")
            if dsid:
                await cache.set(
                    _k_user_dataset(uid, str(dsid)), sid, ex=settings.session_ttl_seconds
                )
        return True
    except _EXPECTED_CACHE_FAILURES as e:
        # Expected during a brownout: say what was lost, not how redis-py said so.
        _note_cache_failure(f"save_session({sid})", e)
        logger.error(
            "save_session failed for %s after %.1fs (%s: %s)%s — this turn's "
            "session state was NOT persisted.",
            sid, budget, type(e).__name__, e,
            " [cache already degraded]" if degraded else "",
        )
        return False
    except Exception as e:
        # Not an infrastructure failure: keep the traceback, it is a real bug.
        _note_cache_failure(f"save_session({sid})", e)
        logger.exception("save_session failed for %s", sid)
        return False

# Per-session locks serialize read-modify-write cycles within this process so
# the background asset-persistence task and the next chat turn can't clobber
# each other's fields (both run on the same asyncio event loop).
_session_locks: Dict[str, asyncio.Lock] = {}


def _get_session_lock(sid: str) -> asyncio.Lock:
    lock = _session_locks.get(sid)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[sid] = lock
    return lock


async def update_session(
    sid: str,
    mutator: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Atomically read-modify-write a session under a per-session lock.

    ``mutator(current)`` receives the *latest* session dict and either mutates
    it in place or returns a replacement dict. Re-reading inside the lock is
    what prevents lost updates: a concurrent writer that already saved is seen
    here, and this writer only applies its own delta on top.
    """
    if not cache or not sid:
        return None
    async with _get_session_lock(sid):
        current = await get_session(sid) or {}
        result = mutator(current)
        merged = result if isinstance(result, dict) else current
        ok = await save_session(sid, merged)
        return merged if ok else None


async def delete_session(sid: str) -> None:
    if not cache or not sid: return
    try:
        data = await get_session(sid)
        uid = data.get("user_id") if data else None
        await cache.delete(_k_session(sid))
        if uid:
            await cache.srem(_k_user_sessions(uid), sid)
            dsid = data.get("dataset_id") if data else None
            if dsid:
                await cache.delete(_k_user_dataset(uid, str(dsid)))
    except Exception:
        logger.exception("delete_session failed for %s", sid)

async def get_session(sid: Optional[str], *, bypass_circuit: bool = False) -> Optional[Dict[str, Any]]:
    if not cache or not sid:
        return None
    # honor bypass for explicit calls (eg: /preview?session_id=...)
    if _cache_unhealthy() and not bypass_circuit:
        return None
    try:
        s = await cache.get(_k_session(sid))
        return json.loads(s) if s else None
    except Exception as e:
        _note_cache_failure(f"get_session({sid})", e)
        return None

async def refresh_session_ttl(sid: str) -> None:
    if not cache or not sid: return
    try:
        await cache.expire(_k_session(sid), settings.session_ttl_seconds)
    except Exception:
        logger.debug("refresh_session_ttl failed for %s", sid)

async def find_session_by_dataset_for_user(dataset_id: Optional[str], user_id: Optional[str]) -> Optional[str]:
    if not cache or not dataset_id or not user_id:
        return None
    try:
        # Fast path: O(1) index lookup + one small ownership-validating GET.
        sid = await cache.get(_k_user_dataset(user_id, str(dataset_id)))
        if sid:
            sid = sid.decode() if isinstance(sid, (bytes, bytearray)) else str(sid)
            blob = await cache.get(_k_session(sid))
            if blob:
                data = json.loads(blob)
                if data.get("user_id") == user_id and data.get("dataset_id") == dataset_id:
                    return sid
        # Legacy fallback for sessions saved before the index existed: the old
        # full scan. Backfill the index on a hit so each session pays this once.
        sids = list(await cache.smembers(_k_user_sessions(user_id)))
        if not sids:
            return None
        keys = [_k_session(s) for s in sids]
        blobs = await _mget_safe(keys)
        for sid, blob in zip(sids, blobs):
            if not blob: continue
            data = json.loads(blob)
            if data.get("dataset_id") == dataset_id:
                try:
                    await cache.set(
                        _k_user_dataset(user_id, str(dataset_id)), sid,
                        ex=settings.session_ttl_seconds,
                    )
                except Exception:
                    logger.debug("index backfill failed for %s", sid)
                return sid
        return None
    except Exception:
        logger.exception("find_session_by_dataset_for_user failed (uid=%s, ds=%s)", user_id, dataset_id)
        return None

async def find_active_db_customer_for_user(user_id: Optional[str]):
    """Best-effort ``(customer_id, source_table)`` of the user's most-recent active database.

    Scans the user's sessions for a ``type:"database"`` connect session or a
    ``db://`` sample session and returns the newest one's ``customer_id`` and the
    ``source_table`` it analyzed (may be None). Used as a fallback so a DTA transfer
    can find the database the user is analyzing even when the chat thread's active
    dataset session isn't the db:// one (it's orphaned from the thread's dataset group).
    Returns ``None`` when no active database is found.
    """
    if not cache or not user_id:
        return None
    try:
        sids = list(await cache.smembers(_k_user_sessions(user_id)))
        if not sids:
            return None
        blobs = await _mget_safe([_k_session(s) for s in sids])
        best = None   # (created_at, customer_id); ISO timestamps sort lexically
        for blob in blobs:
            if not blob:
                continue
            try:
                data = json.loads(blob)
            except Exception:
                continue
            if data.get("user_id") != user_id:
                continue
            loc = str(data.get("data_source_location") or "")
            is_db = (
                data.get("type") == "database"
                or str(data.get("input_data_type") or "").lower() == "db"
                or loc.startswith("db://")
            )
            if not is_db:
                continue
            cust = (data.get("customer_id") or "").strip()
            if not cust and loc.startswith("db://"):
                m = re.match(r"^db://([^/]+)/", loc)
                cust = m.group(1) if m else ""
            if not cust:
                continue
            created = str(data.get("created_at") or "")   # ISO timestamps sort lexically
            if best is None or created > best[0]:
                best = (created, cust, data.get("source_table") or None)
        return (best[1], best[2]) if best else None
    except Exception:
        logger.exception("find_active_db_customer_for_user failed (uid=%s)", user_id)
        return None

# ---------- Thread helpers ----------
def _k_session_threads(sid: str) -> str: return f"sess:{sid}:threads"

async def bind_thread_session(thread_id: str, sid: str) -> None:
    if not cache: return
    try:
        await cache.set(_k_thread_session(thread_id), sid, ex=settings.session_ttl_seconds)
        # Durable reverse index (session -> threads) so dataset/session
        # cleanup can find and delete persisted thread histories even after
        # a process restart wiped the in-process THREAD_META.
        await cache.sadd(_k_session_threads(sid), thread_id)
        await cache.expire(_k_session_threads(sid), settings.session_ttl_seconds)
    except Exception:
        logger.debug("bind_thread_session failed for %s", thread_id)

async def get_thread_session(thread_id: str) -> Optional[str]:
    if not cache: return None
    try:
        return await cache.get(_k_thread_session(thread_id))
    except Exception:
        logger.debug("get_thread_session failed for %s", thread_id)
        return None

# ---------- Dataset helper ----------
async def _user_datasets(user_id: str) -> List[str]:
    if not user_id or not cache:
        return []
    try:
        sess_set = _k_user_sessions(user_id)
        sids: List[str] = []
        if hasattr(cache, "sscan_iter"):
            async for sid in cache.sscan_iter(sess_set, count=200, cap=2000):  # type: ignore[attr-defined]
                sids.append(sid)
        else:
            sids = list(await cache.smembers(sess_set))[:2000]
        blobs = await _mget_safe([_k_session(s) for s in sids])
        out: List[str] = []
        seen: set[str] = set()
        for blob in blobs:
            if not blob:
                continue
            try:
                ds = json.loads(blob).get("dataset_id")
            except Exception:
                ds = None
            if ds and ds not in seen:
                out.append(ds)
                seen.add(ds)
        return out
    except Exception as e:
        _note_cache_failure("_user_datasets", e)
        return []

# ---------- Cache init ----------
async def _init_cache() -> Optional[ICache]:
    if not settings.redis_url:
        logger.warning("No Redis URL; cache features disabled.")
        return None
    return RedisCache(settings.redis_url)
