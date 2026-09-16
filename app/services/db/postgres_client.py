"""
Postgres Client for Artifact Context (Layer 3)
Detects redundancy by querying previously generated metadata via persistent relational databases.
"""
import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from app.services.memory_runtime import ensure_fallback_allowed, strict_memory_infra_enabled

logger = logging.getLogger(__name__)

Base = declarative_base()

class Layer3Artifact(Base):
    __tablename__ = 'layer3_artifacts'
    query_hash = Column(String(64), primary_key=True)
    metadata_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class PostgresClientImpl:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        sqlite_path = os.path.join(base_dir, "artifacts", "layer3_artifacts.db")
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        default_url = f"sqlite:///{sqlite_path.replace(os.sep, '/')}"

        self.db_url = os.environ.get("POSTGRES_URL", default_url)
        if strict_memory_infra_enabled() and self.db_url.startswith("sqlite"):
            ensure_fallback_allowed(
                "Layer 3 Postgres",
                "POSTGRES_URL is not set; SQLite fallback is dev-only"
            )

        self.engine = None
        self.SessionLocal = None
        self._mock_table: Dict[str, Dict[str, Any]] = {}
        self.use_mock = False

        logger.info(f"Initialized PostgresClientImpl targeting {self.db_url}")

    def connect(self) -> bool:
        """Connect to Postgres (or SQLite fallback)."""
        try:
            connect_args = {"check_same_thread": False} if "sqlite" in self.db_url else {}
            self.engine = create_engine(self.db_url, connect_args=connect_args)
            Base.metadata.create_all(bind=self.engine)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.use_mock = False
            logger.info("PostgresClientImpl connected and bound tables successfully.")
            return True
        except Exception as e:
            ensure_fallback_allowed("Layer 3 Postgres", f"database bind failed: {e}")
            logger.error(f"Failed to bind Database Engine: {e}. Falling back to volatile dictionary mock.")
            self.use_mock = True
            return False

    def _hash_query(self, query: str, user_id: str = "default") -> str:
        """
        Standardize prompts into deterministic indexes, scoped per tenant.

        The user_id is folded into the hash so two tenants issuing the same
        prompt get distinct keys — otherwise tenant B's identical prompt would
        surface tenant A's artifact paths/metrics (cross-tenant leak).
        """
        return hashlib.sha256(f"{user_id}\x00{query}".encode('utf-8')).hexdigest()

    def record_artifact(self, query: str, artifact_metadata: Dict[str, Any], user_id: str = "default") -> None:
        """Record an artifact's metadata based on a hashed query footprint."""
        query_hash = self._hash_query(query, user_id)

        if self.use_mock or self.SessionLocal is None:
            self._mock_table[query_hash] = artifact_metadata
            logger.debug(f"Recorded artifact metadata for hash '{query_hash}' mapping to mock context.")
            return

        try:
            with self.SessionLocal() as session:
                artifact = session.query(Layer3Artifact).filter_by(query_hash=query_hash).first()
                if artifact:
                    artifact.metadata_json = json.dumps(artifact_metadata)
                else:
                    new_artifact = Layer3Artifact(
                        query_hash=query_hash,
                        metadata_json=json.dumps(artifact_metadata)
                    )
                    session.add(new_artifact)
                session.commit()
                logger.debug(f"Safely recorded artifact mapping context for '{query_hash}' in relational DB.")
        except Exception as e:
            logger.error(f"Postgres artifact insert failure: {e}")

    def check_artifact_exists(self, query: str, user_id: str = "default") -> bool:
        """Check if an artifact already exists for the given query execution path."""
        query_hash = self._hash_query(query, user_id)
        exists = False

        if self.use_mock or self.SessionLocal is None:
            exists = query_hash in self._mock_table
        else:
            try:
                with self.SessionLocal() as session:
                    exists = session.query(Layer3Artifact).filter_by(query_hash=query_hash).first() is not None
            except Exception as e:
                logger.error(f"Postgres lookup failure: {e}")
                exists = False

        if exists:
            logger.debug(f"Redundancy detected for hashed query footpring '{query_hash[:8]}...'.")
        return exists

    def get_artifact(self, query: str, user_id: str = "default") -> Optional[Dict[str, Any]]:
        """Retrieve artifact metadata natively from storage layers."""
        query_hash = self._hash_query(query, user_id)

        if self.use_mock or self.SessionLocal is None:
            return self._mock_table.get(query_hash)

        try:
            with self.SessionLocal() as session:
                artifact = session.query(Layer3Artifact).filter_by(query_hash=query_hash).first()
                if artifact:
                    return json.loads(artifact.metadata_json)
        except Exception as e:
            logger.error(f"Postgres fetch failure: {e}")

        return None

postgres_client = PostgresClientImpl()
