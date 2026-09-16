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


milvus_client = MilvusClientImpl()
