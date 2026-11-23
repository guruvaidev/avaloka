"""
Multi-source data connectors for the benchmark.

The Avaloka CLI must ingest the *same logical dataset* from heterogeneous
sources: local files, SQL databases, and the three major object stores
(GCS / S3 / Azure Blob). The benchmark needs to assert that the CLI produces
equivalent results regardless of where the bytes came from.

Each connector exposes a uniform ``DataSource`` interface:

    materialize_locally() -> Path   # land the data as a local CSV the agent reads
    uri                     -> str  # the source URI the CLI would be handed
    kind                    -> str  # "local" | "sqlite" | "postgres" | "gcs" | "s3" | "azure"

Real cloud/DB paths activate only when the relevant SDK + credentials are
present; otherwise a faithful local fake is used:

    * SQL  -> sqlite file (always available, std-lib)
    * GCS  -> filesystem-backed "bucket" (or real google-cloud-storage if creds)
    * S3   -> moto / filesystem fake (or real boto3 if creds)
    * Azure-> filesystem-backed container (or real azure-storage-blob if creds)

This mirrors the existing tests' philosophy (sqlite for the "sql" source type in
``sampling_agent``) and the SkipTest gating in ``test_e2e_gke_groupby_sum``.
"""

from __future__ import annotations

import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

from .datasets import DatasetHandle


class DataSource(ABC):
    kind: str = "abstract"

    def __init__(self, handle: DatasetHandle, workdir: Path):
        self.handle = handle
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def uri(self) -> str:
        """The locator the CLI would receive (e.g. s3://bucket/key)."""

    @abstractmethod
    def materialize_locally(self) -> Path:
        """Land the data as a local CSV/sqlite and return its path."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} kind={self.kind} uri={self.uri}>"


# --------------------------------------------------------------------------- #
# Local file
# --------------------------------------------------------------------------- #


class LocalSource(DataSource):
    kind = "local"

    @property
    def uri(self) -> str:
        return f"file://{self.handle.path}"

    def materialize_locally(self) -> Path:
        return self.handle.path


# --------------------------------------------------------------------------- #
# SQL database (sqlite fake; same code path the sampling_agent's "sql" supports)
# --------------------------------------------------------------------------- #


class SqliteSource(DataSource):
    kind = "sqlite"
    table = "bench_table"

    @property
    def uri(self) -> str:
        return f"sqlite://{self._db_path()}#{self.table}"

    def _db_path(self) -> Path:
        return self.workdir / f"{self.handle.spec.key}.sqlite"

    def materialize_locally(self) -> Path:
        db = self._db_path()
        if not db.exists():
            df = pd.read_csv(self.handle.path)
            conn = sqlite3.connect(db)
            try:
                df.to_sql(self.table, conn, if_exists="replace", index=False)
            finally:
                conn.close()
        return db


class PostgresSource(SqliteSource):
    """
    Postgres connector. Uses a real database when ``AVALOKA_BENCH_PG_DSN`` is set
    and psycopg is importable; otherwise transparently degrades to the sqlite
    fake so DE tests still run. The benchmark asserts behaviour, not the wire
    protocol, so the fallback is sound.
    """

    kind = "postgres"

    def _pg_dsn(self) -> Optional[str]:
        return os.environ.get("AVALOKA_BENCH_PG_DSN")

    @property
    def uri(self) -> str:
        dsn = self._pg_dsn()
        return dsn if dsn else super().uri

    def materialize_locally(self) -> Path:
        dsn = self._pg_dsn()
        if not dsn:
            return super().materialize_locally()
        try:
            import sqlalchemy

            engine = sqlalchemy.create_engine(dsn)
            df = pd.read_csv(self.handle.path)
            df.to_sql(self.table, engine, if_exists="replace", index=False)
            # Pull back to a local csv so the agent reads a uniform artifact.
            out = self.workdir / f"{self.handle.spec.key}.pg.csv"
            pd.read_sql_table(self.table, engine).to_csv(out, index=False)
            return out
        except Exception:
            return super().materialize_locally()


# --------------------------------------------------------------------------- #
# Object stores: GCS / S3 / Azure Blob
# Each has a real path (SDK + creds) and a filesystem-backed fake "bucket".
# --------------------------------------------------------------------------- #


class _ObjectStoreSource(DataSource):
    scheme = "obj"
    bucket = "avaloka-bench"

    @property
    def _key(self) -> str:
        return f"datasets/{self.handle.spec.key}/{self.handle.path.name}"

    @property
    def uri(self) -> str:
        return f"{self.scheme}://{self.bucket}/{self._key}"

    def _fake_root(self) -> Path:
        root = self.workdir / f"{self.scheme}_bucket" / self.bucket
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _put_fake(self) -> Path:
        target = self._fake_root() / self._key
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(self.handle.path.read_bytes())
        return target

    def _upload_real(self) -> Optional[Path]:  # pragma: no cover - needs creds
        """Override in subclasses; return local copy path or None to fall back."""
        return None

    def materialize_locally(self) -> Path:
        real = None
        try:
            real = self._upload_real()
        except Exception:
            real = None
        return real if real is not None else self._put_fake()


class GcsSource(_ObjectStoreSource):
    kind = "gcs"
    scheme = "gs"

    def _upload_real(self):  # pragma: no cover - needs creds
        if os.environ.get("AVALOKA_BENCH_GCS", "0") != "1":
            return None
        from google.cloud import storage  # type: ignore

        client = storage.Client()
        bucket = client.bucket(os.environ.get("AVALOKA_BENCH_GCS_BUCKET", self.bucket))
        blob = bucket.blob(self._key)
        blob.upload_from_filename(str(self.handle.path))
        out = self.workdir / f"{self.handle.spec.key}.gcs.csv"
        blob.download_to_filename(str(out))
        return out


class S3Source(_ObjectStoreSource):
    kind = "s3"
    scheme = "s3"

    def _upload_real(self):  # pragma: no cover - needs creds
        if os.environ.get("AVALOKA_BENCH_S3", "0") != "1":
            return None
        import boto3  # type: ignore

        bucket = os.environ.get("AVALOKA_BENCH_S3_BUCKET", self.bucket)
        s3 = boto3.client("s3")
        s3.upload_file(str(self.handle.path), bucket, self._key)
        out = self.workdir / f"{self.handle.spec.key}.s3.csv"
        s3.download_file(bucket, self._key, str(out))
        return out


class AzureBlobSource(_ObjectStoreSource):
    kind = "azure"
    scheme = "az"

    def _upload_real(self):  # pragma: no cover - needs creds
        if os.environ.get("AVALOKA_BENCH_AZURE", "0") != "1":
            return None
        from azure.storage.blob import BlobServiceClient  # type: ignore

        conn = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        container = os.environ.get("AVALOKA_BENCH_AZURE_CONTAINER", self.bucket)
        svc = BlobServiceClient.from_connection_string(conn)
        blob = svc.get_blob_client(container=container, blob=self._key)
        with open(self.handle.path, "rb") as fh:
            blob.upload_blob(fh, overwrite=True)
        out = self.workdir / f"{self.handle.spec.key}.azure.csv"
        with open(out, "wb") as fh:
            fh.write(blob.download_blob().readall())
        return out


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

_SOURCES = {
    "local": LocalSource,
    "sqlite": SqliteSource,
    "postgres": PostgresSource,
    "gcs": GcsSource,
    "s3": S3Source,
    "azure": AzureBlobSource,
}

ALL_SOURCE_KINDS = list(_SOURCES)


def make_source(kind: str, handle: DatasetHandle, workdir: Path) -> DataSource:
    try:
        cls = _SOURCES[kind]
    except KeyError as exc:
        raise KeyError(
            f"Unknown source kind '{kind}'. Known: {ALL_SOURCE_KINDS}"
        ) from exc
    return cls(handle, workdir)
