"""
Real Redis Client for Domain Context (Layer 1)
Stores dataset schema embeddings and structure with a configurable TTL.

Layer 1 semantic search strategy (best-effort cascade):
  1. Vector similarity  – schema embeddings are stored alongside JSON in Redis;
                          query is embedded and nearest-neighbour found via
                          cosine similarity computed in Python.
                          Requires at least one embedding provider to be active
                          (see embedding_utils.py).
  2. Keyword overlap    – fallback if embedding fails; TF-IDF-style word overlap.
  3. None               – triggers MCP hot-load in memory_plane.

Environment variables:
  REDIS_URL              – Full Redis connection URL         (default: redis://localhost:6379/0)
  REDIS_SOCKET_TIMEOUT   – Socket read/write timeout (s)    (default: 3.0)
  REDIS_SCHEMA_TTL       – Default schema cache TTL (s)     (default: 86400  = 24 h)
  REDIS_VECTOR_SIM_THRESHOLD – Min cosine score to accept   (default: 0.35)
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional

from cachetools import TTLCache

try:
    import redis
except ImportError:
    redis = None
    logger = logging.getLogger(__name__)
    logger.warning(
        "redis package is not installed - Layer 1 will use the in-process fallback store."
    )

logger = logging.getLogger(__name__)

from app.services.memory_runtime import ensure_fallback_allowed


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


class RedisClientImpl:
    def __init__(self):
        self.redis_url        = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._socket_timeout  = _env_float("REDIS_SOCKET_TIMEOUT", 3.0)
        self._default_ttl     = _env_int("REDIS_SCHEMA_TTL", 86400)
        self._sim_threshold   = _env_float("REDIS_VECTOR_SIM_THRESHOLD", 0.35)

        self.client = None
        self._fallback_store: Dict[str, Dict[str, Any]] = {}
        self._fallback_vectors: Dict[str, List[float]] = {}
        # Generic JSON key/value fallback (used by set_json/get_json when
        # Redis is unreachable) — separate from the schema-cache fallback.
        # TTL-bounded and size-capped like the in-process memory stores so a
        # long Redis outage can't leak memory or serve arbitrarily stale data.
        self._fallback_kv: TTLCache = TTLCache(maxsize=10000, ttl=self._default_ttl)
        self.use_mock = False

        logger.info(
            f"Initialized RedisClientImpl — url={self.redis_url}, "
            f"socket_timeout={self._socket_timeout}s, "
            f"default_ttl={self._default_ttl}s, "
            f"sim_threshold={self._sim_threshold}"
        )

    def connect(self) -> bool:
        if redis is None:
            ensure_fallback_allowed("Layer 1 Redis", "redis package is not installed")
            self.use_mock = True
            return False

        try:
            self.client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=self._socket_timeout,
                socket_connect_timeout=self._socket_timeout,
            )
            self.client.ping()
            self.use_mock = False
            logger.info(f"RedisClientImpl connected to {self.redis_url}")
            return True
        except Exception as e:
            ensure_fallback_allowed("Layer 1 Redis", f"Redis unavailable at {self.redis_url}: {e}")
            logger.warning(
                f"Redis unavailable at {self.redis_url}: {e}. "
                "Activating circuit-breaker in-process fallback."
            )
            self.use_mock = True
            return False

    def _ensure_connection(self) -> None:
        """
        Lazily connect on first use so callers outside the retrieval path
        (e.g. store_user_preference) don't need their own connect() call.
        Once use_mock is set, recovery is left to the retrieval path's
        periodic connect() attempts.
        """
        if self.client is None and not self.use_mock:
            self.connect()

    def set_json(self, key: str, value: Any, ttl_seconds: int = None) -> bool:
        """
        Store any JSON-serializable value under *key* with a TTL.
        Returns True when written to real Redis; False on the fallback path.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        payload = json.dumps(value)
        self._ensure_connection()
        if self.use_mock or self.client is None:
            self._fallback_kv[key] = payload
            return False
        try:
            self.client.set(key, payload, ex=ttl)
            # Drop any stale outage-era copy so it can never shadow the
            # authoritative Redis value after recovery.
            self._fallback_kv.pop(key, None)
            return True
        except Exception as e:
            logger.warning(f"[L1] set_json failed for '{key}': {e}")
            self._fallback_kv[key] = payload
            return False

    def get_json(self, key: str, strict: bool = False) -> Optional[Any]:
        """
        Read a value stored via set_json. Returns None on miss/undecodable.

        With strict=True, an unreachable Redis or a failed GET raises
        RuntimeError instead of silently returning the fallback value —
        callers that must distinguish "no data exists" from "could not read"
        (e.g. restart hydration) depend on this.
        """
        self._ensure_connection()
        raw: Optional[str]
        if self.use_mock or self.client is None:
            if strict:
                raise RuntimeError(f"Redis unavailable; cannot read '{key}'")
            raw = self._fallback_kv.get(key)
        else:
            try:
                raw = self.client.get(key)
            except Exception as e:
                logger.warning(f"[L1] get_json failed for '{key}': {e}")
                if strict:
                    raise RuntimeError(f"Redis read failed for '{key}'") from e
                raw = self._fallback_kv.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    def delete_prefix(self, prefix: str) -> None:
        """
        Best-effort delete of every key starting with *prefix*. Never raises.
        Uses SCAN with per-key deletes (cluster-safe; KEYS + multi-key DEL
        blocks production Redis and raises CROSSSLOT on clusters).
        """
        for k in [k for k in self._fallback_kv if k.startswith(prefix)]:
            self._fallback_kv.pop(k, None)
        if self.use_mock or self.client is None:
            return
        try:
            for k in self.client.scan_iter(match=f"{prefix}*", count=100):
                try:
                    self.client.delete(k)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[L1] delete_prefix failed for '{prefix}': {e}")

    def _embed(self, text: str) -> Optional[List[float]]:
        """Return an embedding vector for *text*, or None if all providers fail."""
        try:
            from app.services.embedding_utils import embed_text
            return embed_text(text)
        except Exception as exc:
            logger.debug(f"[L1 Redis] Embedding unavailable for vector search: {exc}")
            return None

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        import math
        dot   = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        return (dot / (mag_a * mag_b)) if mag_a and mag_b else 0.0

    def set_schema(self, user_id: str, dataset_id: str, schema_info: Dict[str, Any],
                   ttl_seconds: int = None) -> None:
        """Store schema JSON (and its embedding vector when available)."""
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl

        vec = self._embed(json.dumps(schema_info))

        if self.use_mock or self.client is None:
            self._fallback_store[f"{user_id}:{dataset_id}"] = {"schema": schema_info, "ttl": ttl}
            if vec:
                self._fallback_vectors[f"{user_id}:{dataset_id}"] = vec
            logger.debug(
                f"[L1] Stored schema for '{user_id}:{dataset_id}' in fallback "
                f"(vec={'yes' if vec else 'no'})."
            )
            return

        payload = json.dumps(schema_info)
        self.client.setex(f"schema:{user_id}:{dataset_id}", ttl, payload)
        if vec:
            self.client.setex(f"schema_vec:{user_id}:{dataset_id}", ttl, json.dumps(vec))
            logger.debug(
                f"[L1] Stored schema + embedding vector for '{user_id}:{dataset_id}' in Redis (TTL={ttl}s)."
            )
        else:
            logger.debug(
                f"[L1] Stored schema for '{user_id}:{dataset_id}' in Redis (TTL={ttl}s, no vector)."
            )

    def get_schema(self, user_id: str, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve schema by exact dataset_id."""
        if self.use_mock or self.client is None:
            record = self._fallback_store.get(f"{user_id}:{dataset_id}")
            return record["schema"] if record else None
        raw = self.client.get(f"schema:{user_id}:{dataset_id}")
        return json.loads(raw) if raw else None

    def search_schema_semantically(self, user_id: str, query: str) -> Optional[Dict[str, Any]]:
        """
        Find the most relevant cached schema for *query*.

        Strategy 1 — vector similarity (preferred):
          Embed *query*, load all stored schema vectors, compute cosine similarity,
          return the best match if score ≥ REDIS_VECTOR_SIM_THRESHOLD.

        Strategy 2 — keyword overlap (fallback when embedding unavailable):
          TF-IDF-style word-overlap scoring across all cached schemas.
        """
        query_vec = self._embed(query)

        if query_vec:
            best_id    = None
            best_score = -1.0

            if self.use_mock or self.client is None:
                for ds_id, vec in self._fallback_vectors.items():
                    if ds_id.startswith(f"{user_id}:"):
                        score = self._cosine(query_vec, vec)
                        if score > best_score:
                            best_score = score
                            best_id    = ds_id.split(":", 1)[1]
            else:
                vec_keys = self.client.keys(f"schema_vec:{user_id}:*")
                for key in vec_keys:
                    raw = self.client.get(key)
                    if not raw:
                        continue
                    stored_vec = json.loads(raw)
                    score      = self._cosine(query_vec, stored_vec)
                    if score > best_score:
                        best_score = score
                        best_id    = key.replace(f"schema_vec:{user_id}:", "")

            if best_id and best_score >= self._sim_threshold:
                logger.debug(
                    f"[L1] Vector similarity match: dataset='{best_id}', "
                    f"score={best_score:.3f} (threshold={self._sim_threshold})"
                )
                return self.get_schema(user_id, best_id)

            if best_id:
                logger.debug(
                    f"[L1] Best vector match '{best_id}' scored {best_score:.3f} "
                    f"— below threshold {self._sim_threshold}; falling back to keyword overlap."
                )

        available_schemas: list = []

        if self.use_mock or self.client is None:
            available_schemas = [
                (ds_id, v.get("schema", {}))
                for ds_id, v in self._fallback_store.items() if ds_id.startswith(f"{user_id}:")
            ]
        else:
            for key in self.client.keys(f"schema:{user_id}:*"):
                raw = self.client.get(key)
                if raw:
                    ds_id = key.replace(f"schema:{user_id}:", "")
                    available_schemas.append((ds_id, json.loads(raw)))

        if not available_schemas:
            return None

        query_words   = set(query.lower().split())
        best_match    = None
        highest_score = 0

        for _, schema in available_schemas:
            schema_str    = json.dumps(schema).lower()
            overlap_score = sum(1 for w in query_words if w in schema_str)
            if overlap_score > highest_score:
                highest_score = overlap_score
                best_match    = schema

        if highest_score > 0:
            logger.debug(
                f"[L1] Keyword overlap match (score={highest_score}). "
                "Install an embedding provider for true vector search."
            )
            return best_match

        return None


redis_client = RedisClientImpl()
