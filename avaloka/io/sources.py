"""Source connectors — resolve a dataset URI from any source to a local file.

The CLI's first release reads local CSV/Parquet. Real deployments pull data from
relational databases and cloud object storage (S3 / GCS / Azure Blob). This
module resolves a *source URI* to a locally materialised file that the rest of
the stack — :func:`avaloka.io.load_dataset`, :func:`avaloka.workload.route`, the
firefly swarm and the multi-Avaloka coordinator — can consume uniformly,
regardless of where the data actually lives.

Supported schemes::

    /path/data.csv            local file (also file:///path/data.csv)
    sqlite:///abs/app.db#tbl  a table, or #query=SELECT ... FROM ...   (SQLAlchemy)
    postgresql://u:p@h/db#tbl any SQLAlchemy-supported database URL
    s3://bucket/key.parquet   object storage via fsspec (needs s3fs)
    gs://bucket/key.csv       object storage via fsspec (needs gcsfs)
    az://container/key.csv    object storage via fsspec (needs adlfs)
    memory://key.csv          fsspec in-memory filesystem (used by tests)

A resolved source is a real local CSV/Parquet path plus provenance, so lineage
stays honest: the mission records where the bytes came from even though the
fireflies only ever touch a local file.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

# Schemes we treat as "already local" (no download needed).
_LOCAL_SCHEMES = {"", "file"}
# Object-storage schemes handled through fsspec.
_FSSPEC_SCHEMES = {"s3", "gs", "gcs", "az", "abfs", "abfss", "memory", "http", "https"}


@dataclass
class ResolvedSource:
    """A dataset URI resolved to a concrete local file, with provenance."""

    uri: str
    scheme: str
    kind: str                      # "file" | "database" | "object_storage"
    local_path: Path
    materialized: bool             # True when we created a local copy we may clean up
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["local_path"] = str(self.local_path)
        return d


class SourceError(RuntimeError):
    """Raised when a source URI cannot be resolved to a local dataset."""


# --------------------------------------------------------------------------
def _cache_dir(cache_dir: str | os.PathLike[str] | None) -> Path:
    if cache_dir is not None:
        p = Path(cache_dir).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p
    return Path(tempfile.mkdtemp(prefix="avaloka-src-"))


def _looks_local(scheme: str, uri: str) -> bool:
    # A bare Windows drive letter ("C:\...") splits as scheme="c"; treat single
    # char schemes as local paths, not URL schemes.
    return scheme in _LOCAL_SCHEMES or len(scheme) == 1


def parse_uri(uri: str) -> tuple[str, str]:
    """Return ``(scheme, remainder)`` for a source URI. Bare paths → scheme ""."""
    split = urlsplit(uri)
    scheme = split.scheme.lower()
    if _looks_local(scheme, uri):
        return "", uri
    return scheme, uri


# --- database -------------------------------------------------------------
def _materialize_database(uri: str, cache_dir: Path) -> ResolvedSource:
    """Read a table or query from a SQLAlchemy database URL into a local Parquet."""
    try:
        import pandas as pd
        import sqlalchemy
    except ImportError as exc:  # pragma: no cover - sqlalchemy is a core dep here
        raise SourceError(f"database sources require SQLAlchemy: {exc}") from exc

    url, _, fragment = uri.partition("#")
    fragment = unquote(fragment)
    if fragment.lower().startswith("query="):
        sql = fragment[len("query="):]
        label = "query"
    elif fragment:
        # A bare fragment is a table name (validated against reflected tables).
        sql = None
        table = fragment
        label = table
    else:
        raise SourceError(
            f"database URI {uri!r} must name a table or query via a fragment, e.g. "
            "'sqlite:///app.db#customers' or 'postgresql://.../db#query=SELECT ...'"
        )

    engine = sqlalchemy.create_engine(url)
    try:
        if sql is None:
            insp = sqlalchemy.inspect(engine)
            available = set(insp.get_table_names())
            if table not in available:
                raise SourceError(
                    f"table {table!r} not found in {url!r} (have: {sorted(available)})"
                )
            frame = pd.read_sql_table(table, engine)
        else:
            frame = pd.read_sql_query(sqlalchemy.text(sql), engine)
    finally:
        engine.dispose()

    out = cache_dir / f"db_{label.replace(' ', '_')[:40]}.parquet"
    frame.to_parquet(out, index=False)
    return ResolvedSource(
        uri=uri, scheme=urlsplit(url).scheme.lower(), kind="database",
        local_path=out, materialized=True,
        detail={"backend": urlsplit(url).scheme.lower(), "selector": label,
                "n_rows": int(frame.shape[0]), "n_cols": int(frame.shape[1])},
    )


# --- object storage -------------------------------------------------------
def _materialize_object_storage(uri: str, scheme: str, cache_dir: Path) -> ResolvedSource:
    """Download a cloud object (S3/GCS/Azure/HTTP/memory) to a local file via fsspec."""
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover
        raise SourceError(f"object-storage sources require fsspec: {exc}") from exc

    # Strip query string (e.g. credentials) before deriving the local suffix.
    clean = uri.split("?", 1)[0]
    suffix = Path(urlsplit(clean).path).suffix.lower() or ".csv"
    protocol = {"gcs": "gs", "abfs": "az", "abfss": "az"}.get(scheme, scheme)

    try:
        with fsspec.open(uri, "rb") as fh:
            data = fh.read()
    except ImportError as exc:
        raise SourceError(
            f"missing filesystem driver for '{scheme}://' — install the matching "
            f"fsspec backend (s3fs/gcsfs/adlfs): {exc}"
        ) from exc
    except FileNotFoundError as exc:
        raise SourceError(f"object not found: {uri}") from exc

    out = cache_dir / f"obj_{abs(hash(uri)) % (10 ** 10)}{suffix}"
    out.write_bytes(data)
    return ResolvedSource(
        uri=uri, scheme=protocol, kind="object_storage",
        local_path=out, materialized=True,
        detail={"protocol": protocol, "bytes": len(data)},
    )


# --- local ----------------------------------------------------------------
def _materialize_local(uri: str) -> ResolvedSource:
    path = Path(uri[len("file://"):] if uri.startswith("file://") else uri).expanduser()
    if not path.exists():
        raise FileNotFoundError(uri)
    return ResolvedSource(uri=uri, scheme="file", kind="file",
                          local_path=path, materialized=False)


# --------------------------------------------------------------------------
def materialize(uri: str, *, cache_dir: str | os.PathLike[str] | None = None) -> ResolvedSource:
    """Resolve any supported source URI to a local CSV/Parquet file.

    The returned :class:`ResolvedSource` carries the local path plus provenance.
    Databases and object stores are copied into ``cache_dir`` (a temp dir by
    default); local files are used in place (``materialized=False``).
    """
    scheme, _ = parse_uri(uri)
    if scheme == "" or scheme == "file":
        return _materialize_local(uri)
    cache = _cache_dir(cache_dir)
    if scheme in _FSSPEC_SCHEMES:
        return _materialize_object_storage(uri, scheme, cache)
    # Anything else is assumed to be a SQLAlchemy database URL (sqlite, postgresql,
    # mysql, mssql, ...). SQLAlchemy validates the driver and raises clearly if not.
    return _materialize_database(uri, cache)


def route(uri: str, *, cache_dir: str | os.PathLike[str] | None = None):
    """Resolve ``uri`` then size it: returns ``(ResolvedSource, WorkloadPlan)``.

    This is the single entry point missions should use so workload routing works
    identically for a local file, a database table or a cloud object.
    """
    from avaloka import workload

    resolved = materialize(uri, cache_dir=cache_dir)
    plan = workload.route(str(resolved.local_path))
    return resolved, plan
