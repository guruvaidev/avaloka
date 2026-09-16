from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List
import os

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
    - Use a discovery endpoint like 'redis://10.128.15.238:6379'
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
        self._r = _make_redis()

    # lifecycle
    async def close(self):
        try:
            await self._r.aclose()
        except Exception:
            pass

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
