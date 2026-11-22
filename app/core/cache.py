from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple
import asyncio
import os
import threading

import redis.asyncio as redis
try:
    # available when redis>=5
    from redis.asyncio.cluster import RedisCluster
except Exception:  # pragma: no cover
    RedisCluster = None


# ---------- Public cache interface ----------
class ICache(ABC):
    @abstractmethod
    async def get(self, key: str) -> Optional[str]: ...
    @abstractmethod
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> None: ...
    @abstractmethod
    async def delete(self, key: str) -> None: ...
    @abstractmethod
    async def smembers(self, key: str) -> List[str]: ...
    @abstractmethod
    async def sadd(self, key: str, *members: str) -> None: ...
    @abstractmethod
    async def srem(self, key: str, *members: str) -> None: ...
    @abstractmethod
    async def expire(self, key: str, ttl: int) -> None: ...
    @abstractmethod
    async def ping(self) -> bool: ...
    @abstractmethod
    async def close(self) -> None: ...

# ---------- Client factory ----------
def _make_redis():
    """
    If REDIS_CLUSTER in {1,true,yes} -> use cluster client (auto-discovery).
    Else -> use single-node client.

    Notes for cluster:
    - Use a discovery endpoint like 'redis://10.0.0.5:6379'
    - Do NOT include '/db' suffix in cluster URLs.
    """
    url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
    use_cluster = os.getenv("REDIS_CLUSTER", "0").lower() in ("1", "true", "yes")

    if use_cluster:
        if RedisCluster is None:
            raise RuntimeError("redis cluster client unavailable; install redis>=5")
        return RedisCluster.from_url(
            url,
            decode_responses=True,
            read_from_replicas=False,   # keep writes on primaries
        )

    # single-node
    return redis.from_url(
        url,
        decode_responses=True,
        socket_keepalive=True,
        retry_on_timeout=True,
        socket_connect_timeout=2.5,
        socket_timeout=3.0,
    )

# ---------- Concrete implementation ----------
class RedisCache(ICache):
    """
    Works in both modes; exposes .mode for /health.
    """
    def __init__(self, url: str | None = None):
        if url:
            os.environ["REDIS_URL"] = url
        self.mode = "cluster" if os.getenv("REDIS_CLUSTER", "0").lower() in ("1", "true", "yes") else "single"
        # One redis.asyncio client per event loop. A client's pool lock and
        # sockets belong to the loop that first used them, and this object is
        # shared process-wide (session_service.cache) while sync LangGraph nodes
        # call it from worker threads through asyncio.run -- a second loop.
        # Sharing one client across those loops raised "attached to a different
        # loop" under asyncio, wedged the pool lock for the main loop, and under
        # uvloop let a worker thread drive a libuv stream the main thread was
        # polling: the API's segfaults. Keyed by id() because uvloop.Loop does
        # not take weak references; the loop is held to tell a reused id apart.
        self._clients: Dict[int, Tuple[asyncio.AbstractEventLoop, Any]] = {}
        self._clients_lock = threading.Lock()

    @property
    def _r(self):
        """The client for the running loop, created on first use there."""
        loop = asyncio.get_running_loop()
        with self._clients_lock:
            entry = self._clients.get(id(loop))
            if entry is not None and entry[0] is loop:
                return entry[1]
            # Forget clients of loops that have finished (asyncio.run closes
            # its loop), so short-lived worker loops do not accumulate.
            for key in [k for k, (lp, _) in self._clients.items() if lp.is_closed()]:
                del self._clients[key]
            client = _make_redis()
            self._clients[id(loop)] = (loop, client)
            return client

    async def release_loop_client(self) -> None:
        """Close and forget the running loop's client.

        For callers that run a short-lived loop (``asyncio.run`` in a worker
        thread): call it before the loop ends so its sockets close on the loop
        that opened them instead of leaking until garbage collection.
        """
        loop = asyncio.get_running_loop()
        with self._clients_lock:
            entry = self._clients.pop(id(loop), None)
        if entry is not None and entry[0] is loop:
            try:
                await entry[1].aclose()
            except Exception:
                pass

    # lifecycle
    async def close(self):
        await self.release_loop_client()

    async def ping(self) -> bool:
        try:
            await self._r.ping()
            return True
        except Exception:
            return False

    # kv
    async def get(self, key: str) -> Optional[str]:
        return await self._r.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        await self._r.set(key, value, ex=ex)

    async def delete(self, key: str) -> None:
        await self._r.delete(key)

    # sets
    async def smembers(self, key: str) -> List[str]:
        return list(await self._r.smembers(key))

    async def sadd(self, key: str, *members: str) -> None:
        await self._r.sadd(key, *members)

    async def srem(self, key: str, *members: str) -> None:
        await self._r.srem(key, *members)

    async def expire(self, key: str, ttl: int) -> None:
        await self._r.expire(key, ttl)

    # optional helpers used by server
    async def mget(self, keys: list[str]) -> list[Optional[str]]:
        if not keys:
            return []
        return await self._r.mget(keys)

    async def sscan_iter(self, key: str, count: int = 200, cap: int = 2000):
        cursor = 0
        scanned = 0
        while True:
            cursor, chunk = await self._r.sscan(key, cursor=cursor, count=count)
            for m in chunk:
                yield m
            scanned += len(chunk)
            if cursor == 0 or scanned >= cap:
                break
