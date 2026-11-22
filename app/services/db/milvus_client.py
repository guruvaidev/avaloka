"""
Real Milvus Client for Episodic Memory (Layer 4)
Stores and retrieves time-series analytical insights via vector similarity.

All tuneable constants are driven by environment variables:

  MILVUS_HOST              – Milvus server hostname          (default: localhost)
  MILVUS_PORT              – Milvus gRPC port                (default: 19530)
  MILVUS_COLLECTION        – Collection name                 (default: ETL_TimeSeries)
  MILVUS_PARTITION         – Default partition name          (default: notebook_partitions)
  MILVUS_INDEX_NLIST       – IVF_FLAT nlist param           (default: 1024)
  MILVUS_SEARCH_NPROBE     – IVF_FLAT nprobe param          (default: 10)
  MILVUS_TOP_K             – Default similarity top-k       (default: 5)
  MILVUS_VECTOR_DIM        – Embedding dimensionality       (default: 1536)
  MILVUS_DEFAULT_NOTEBOOK  – Default notebook_id            (default: default_notebook)
  MILVUS_DEFAULT_CELL      – Default cell_id                (default: default_cell)
"""
import logging
import socket
import time
import os
from typing import List, Dict, Any, Optional
from app.services.memory_runtime import ensure_fallback_allowed

try:
    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
except ImportError:
    connections = FieldSchema = CollectionSchema = DataType = Collection = utility = None
    logging.getLogger(__name__).warning(
        "pymilvus is not installed — Milvus (Layer 4) will be unavailable. "
        "Install with: pip install pymilvus>=2.3.0"
    )

logger = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


class MilvusClientImpl:
    def __init__(self,
                 host: str = None,
                 port: str = None,
                 collection_name: str = None):
        self.host            = host            or os.environ.get("MILVUS_HOST",       "localhost")
        self.port            = port            or os.environ.get("MILVUS_PORT",       "19530")
        self.collection_name = collection_name or os.environ.get("MILVUS_COLLECTION", "ETL_TimeSeries")

        self._nlist      = _env_int("MILVUS_INDEX_NLIST",   1024)
        self._nprobe     = _env_int("MILVUS_SEARCH_NPROBE", 10)
        self._top_k      = _env_int("MILVUS_TOP_K",         5)
        self._vector_dim = _env_int("MILVUS_VECTOR_DIM",    1536)

        self._partition      = os.environ.get("MILVUS_PARTITION",        "notebook_partitions")
        self._default_nb     = os.environ.get("MILVUS_DEFAULT_NOTEBOOK", "default_notebook")
        self._default_cell   = os.environ.get("MILVUS_DEFAULT_CELL",     "default_cell")

        self._collection = None
        # Checked once per process, against the first real vector seen.
        self._space_checked = False
        logger.info(
            f"Initialized MilvusClientImpl — host={self.host}:{self.port}, "
            f"collection={self.collection_name}, dim={self._vector_dim}"
        )

    def connect(self) -> bool:
        """Connect to Milvus vector database and ensure collection exists."""
        if not connections:
            ensure_fallback_allowed("Layer 4 Milvus", "pymilvus package is not installed")
            logger.error("pymilvus not installed. Install pymilvus>=2.3.0 to enable Layer 4.")
            return False

        # Fail fast if the server isn't reachable, instead of letting pymilvus'
        # gRPC retry/backoff block for ~10s. On a healthy server the port is open
        # and this returns in <10ms (no behaviour change); when Milvus is down it
        # returns immediately instead of hanging the caller.
        timeout_s = _env_float("MILVUS_CONNECT_TIMEOUT", 2.0)
        try:
            with socket.create_connection((self.host, int(self.port)), timeout=timeout_s):
                pass
        except (OSError, ValueError) as e:
            ensure_fallback_allowed("Layer 4 Milvus", f"{self.host}:{self.port} unreachable: {e}")
            logger.warning("Milvus not reachable at %s:%s — skipping Layer 4 (%s).", self.host, self.port, e)
            return False

        try:
            connections.connect("default", host=self.host, port=self.port)
            logger.info(f"MilvusClientImpl connected to {self.host}:{self.port}")

            if not utility.has_collection(self.collection_name):
                self._create_collection()
            else:
                self._collection = Collection(self.collection_name)
                self._collection.load()
            return True
        except Exception as e:
            ensure_fallback_allowed("Layer 4 Milvus", f"connect failed at {self.host}:{self.port}: {e}")
            logger.error(f"Failed to connect to Milvus at {self.host}:{self.port}: {e}")
            return False

    def _create_collection(self):
        fields = [
            FieldSchema(name="id",           dtype=DataType.INT64,         is_primary=True, auto_id=True),
            FieldSchema(name="vector",        dtype=DataType.FLOAT_VECTOR,  dim=self._vector_dim),
            FieldSchema(name="timestamp",     dtype=DataType.INT64),
            FieldSchema(name="session_id",    dtype=DataType.VARCHAR,       max_length=255),
            FieldSchema(name="notebook_id",   dtype=DataType.VARCHAR,       max_length=255),
            FieldSchema(name="agent_role",    dtype=DataType.VARCHAR,       max_length=128),
            FieldSchema(name="cell_id",       dtype=DataType.VARCHAR,       max_length=128),
            FieldSchema(name="artifact_ref",  dtype=DataType.VARCHAR,       max_length=512),
            FieldSchema(name="content",       dtype=DataType.VARCHAR,       max_length=65535),
        ]
        schema = CollectionSchema(fields, "Time-Series Analysis Memory for Avaloka")
        self._collection = Collection(self.collection_name, schema)

        index_params = {
            "metric_type": "COSINE",
            "index_type":  "IVF_FLAT",
            "params":      {"nlist": self._nlist},
        }
        self._collection.create_index(field_name="vector", index_params=index_params)
        self._collection.create_partition(self._partition)
        self._collection.load()
        logger.info(
            f"Created Milvus collection '{self.collection_name}' "
            f"(nlist={self._nlist}, partition='{self._partition}')"
        )

    def _check_embedding_space(self, vector: Optional[List[float]]) -> None:
        """Warn, once, when stored vectors come from a different embedding space.

        Before this change embeddings went to OpenAI whenever OPENAI_API_KEY was set;
        they are now local by default. Both are padded to the collection's dimension,
        so nothing fails -- but a local query scored against an OpenAI-embedded
        row is noise. Milvus scores server-side, so rows cannot be skipped per
        query here; this samples the stored rows and says what to do instead.
        Never raises.
        """
        if self._space_checked or not vector or not any(vector) or not self._collection:
            return
        self._space_checked = True
        try:
            from app.services.embedding_utils import same_embedding_space
            rows = self._collection.query(
                expr="timestamp >= 0", output_fields=["vector"], limit=64,
            )
            stored = [r.get("vector") for r in rows if r.get("vector") is not None]
            foreign = sum(1 for v in stored if not same_embedding_space(vector, list(v)))
        except Exception as e:
            logger.debug(f"[L4] embedding-space check skipped: {e}")
            return
        if foreign:
            logger.warning(
                f"[L4] {foreign} of {len(stored)} sampled rows in '{self.collection_name}' were "
                "embedded by a different embedding provider than the one now configured "
                "(EMBEDDING_PROVIDER). Their similarity scores are meaningless: drop the "
                "collection, or re-embed its 'content' field with the current provider."
            )

    def insert_insight(self,
                       session_id:   str,
                       content:      str,
                       agent_role:   str            = "assistant",
                       vector:       List[float]    = None,
                       notebook_id:  str            = None,
                       cell_id:      str            = None,
                       artifact_ref: str            = "none") -> None:
        """Insert an embedded insight into the collection."""
        if not self._collection:
            logger.warning("Milvus collection not initialized. Cannot insert.")
            return

        if vector is None:
            logger.debug(
                "No embedding vector supplied to insert_insight — "
                "using zero-vector fallback. Provide real embeddings for meaningful retrieval."
            )
            vector = [0.0] * self._vector_dim

        self._check_embedding_space(vector)

        nb_id   = notebook_id or self._default_nb
        cell_id = cell_id     or self._default_cell

        data = [
            [vector],
            [int(time.time())],
            [session_id],
            [nb_id],
            [agent_role],
            [cell_id],
            [artifact_ref],
            [content],
        ]

        try:
            self._collection.insert(data, partition_name=self._partition)
            self._collection.flush()
            logger.debug(f"Inserted episodic insight for session '{session_id}' (notebook={nb_id}).")
        except Exception as e:
            logger.error(f"Failed to insert into Milvus: {e}")

    def search_similar_insights(self,
                                session_id:         str,
                                query_vector:       List[float] = None,
                                top_k:              int         = None,
                                notebook_id:        str         = None,
                                time_range_seconds: int         = None) -> List[Dict[str, Any]]:
        """Vector similarity search with optional notebook and temporal scope."""
        if not self._collection:
            logger.warning("Milvus collection not initialized. Returning empty.")
            return []

        effective_top_k = top_k if top_k is not None else self._top_k

        if query_vector is None:
            logger.debug(
                "No query vector supplied to search_similar_insights — "
                "using zero-vector. Results will reflect the nearest zero-point neighbours."
            )
            query_vector = [0.0] * self._vector_dim

        self._check_embedding_space(query_vector)

        search_params = {
            "metric_type": "COSINE",
            "params":      {"nprobe": self._nprobe},
        }

        expr_parts = [f"session_id == '{session_id}'"]
        if notebook_id:
            expr_parts.append(f"notebook_id == '{notebook_id}'")
        if time_range_seconds is not None:
            cutoff_time = int(time.time()) - time_range_seconds
            expr_parts.append(f"timestamp >= {cutoff_time}")

        expr = " && ".join(expr_parts)

        partition_names = None
        if utility and utility.has_partition(self.collection_name, self._partition):
            partition_names = [self._partition]

        try:
            results = self._collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=effective_top_k,
                expr=expr,
                partition_names=partition_names,
                output_fields=["timestamp", "session_id", "notebook_id", "agent_role", "content"],
            )

            parsed = []
            for hits in results:
                for hit in hits:
                    parsed.append({
                        "id":         hit.id,
                        "session_id": hit.entity.get("session_id"),
                        "content":    hit.entity.get("content"),
                        "timestamp":  hit.entity.get("timestamp"),
                        "distance":   hit.distance,
                    })
            return parsed
        except Exception as e:
            logger.error(f"Failed to search Milvus: {e}")
            return []

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------
    # Unlike insert and search, these raise when Milvus cannot be reached: a
    # delete that silently did nothing would leave customer memory behind
    # while the caller believed it gone. Each one selects rows with a query,
    # checks every row belongs to the scope (memory_scope.owns_key), and
    # deletes them by primary key, which every Milvus 2.x accepts.

    def _require_collection(self):
        if not self._collection:
            raise RuntimeError("Milvus collection not initialized; nothing was deleted")
        return self._collection

    def _delete_matching(self, expr: str, keep) -> int:
        """Delete the rows `expr` selects for which keep(session_id) is true,
        in rounds of at most _DELETE_BATCH rows, until a round finds no row it
        has not already deleted (a query may still see a row just deleted)."""
        collection = self._require_collection()
        done: set = set()
        for _ in range(_DELETE_MAX_ROUNDS):
            rows = collection.query(expr=expr, output_fields=["id", "session_id"],
                                    limit=_DELETE_BATCH, consistency_level="Strong")
            ids = sorted({int(r["id"]) for r in rows or []
                          if keep(r.get("session_id"))} - done)
            if not ids:
                break
            collection.delete(f"id in {ids}")
            done.update(ids)
        collection.flush()
        return len(done)

    def delete_keys(self, keys: List[str]) -> int:
        """Delete every row filed under exactly one of `keys` (session_id)."""
        wanted = {str(k) for k in keys or [] if k}
        if not wanted:
            return 0
        expr = "session_id in [" + ", ".join(_expr_str(k) for k in sorted(wanted)) + "]"
        return self._delete_matching(expr, lambda key: key in wanted)

    def delete_thread(self, scope: str, thread_id: str) -> int:
        """Delete what one thread of this user (scope) learned."""
        from app.services.memory_scope import thread_key
        if not scope or not thread_id:
            return 0
        return self.delete_keys([thread_key(scope, thread_id)])

    def delete_user(self, scope: str, legacy_keys: Optional[List[str]] = None) -> int:
        """Delete every row of this user: the bare user key (rows written before
        the thread scope) and every thread, preference and dataset key built
        from it. `legacy_keys` are further exact keys -- the user's session
        ids, under which rows were filed before user scoping and under which
        execution insights are still filed."""
        from app.services.memory_scope import owns_key
        if not scope:
            return 0
        # LIKE only narrows the query (its wildcards can match more than the
        # prefix); owns_key decides what is deleted.
        deleted = self._delete_matching(f"session_id like {_expr_str(scope + '%')}",
                                        lambda key: owns_key(key, scope))
        return deleted + self.delete_keys(legacy_keys or [])

    def delete_by_ids(self, ids: List[int], scope: str) -> int:
        """Delete single rows by primary key -- only those that belong to scope."""
        from app.services.memory_scope import owns_key
        ids = sorted({int(i) for i in ids or []})
        if not ids or not scope:
            return 0
        return self._delete_matching(f"id in {ids}", lambda key: owns_key(key, scope))


# A Milvus query returns at most 16384 rows.
_DELETE_BATCH = 16384
_DELETE_MAX_ROUNDS = 1000


def _expr_str(value: str) -> str:
    """A string literal for a Milvus boolean expression."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


milvus_client = MilvusClientImpl()
