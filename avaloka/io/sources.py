"""Resolve dataset URIs to normalized local files with safe provenance.

Relational tables, MongoDB collections and object-store objects are
materialized behind the same source contract. Connectors are read-only and
credentials/signed query strings are never retained in lineage metadata.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit

from avaloka.io.loader import normalize_records
from avaloka.source_statistics import database_statistics, file_statistics

_LOCAL_SCHEMES = {"", "file"}
_FSSPEC_SCHEMES = {
    "s3", "gs", "gcs", "az", "abfs", "abfss", "memory", "http", "https"
}
_MONGO_SCHEMES = {"mongodb", "mongodb+srv"}
_SAFE_LABEL = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class ResolvedSource:
    """A dataset URI resolved to a concrete local file, with provenance."""

    uri: str
    scheme: str
    kind: str  # file | database | document_database | object_storage
    local_path: Path
    materialized: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["local_path"] = str(self.local_path)
        return data


class SourceError(RuntimeError):
    """Raised when a source URI cannot be resolved to a local dataset."""


def _redact_uri(uri: str) -> str:
    """Return lineage-safe provenance without userinfo or selector values."""
    split = urlsplit(uri)
    if not split.scheme or len(split.scheme) == 1:
        return uri
    host = split.hostname or ""
    if split.port is not None:
        host = f"{host}:{split.port}"
    query = "redacted=1" if split.query else ""
    raw_fragment = split.fragment
    fragment = unquote(raw_fragment)
    if split.scheme.lower() in _MONGO_SCHEMES:
        if "=" in raw_fragment:
            options = parse_qs(raw_fragment, keep_blank_values=True)
            collection = options.get("collection", [""])[0]
        else:
            collection = fragment
        fragment = f"collection={collection}" if collection else ""
    elif fragment.lower().startswith("query="):
        fragment = "query=redacted"
    elif split.scheme.lower() in _FSSPEC_SCHEMES and fragment:
        fragment = "redacted"
    return urlunsplit((split.scheme, host, split.path, query, fragment))


def _cache_dir(cache_dir: str | os.PathLike[str] | None) -> Path:
    if cache_dir is not None:
        path = Path(cache_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix="avaloka-src-"))


def _looks_local(scheme: str, uri: str) -> bool:
    # A bare Windows drive letter (C:\...) must not be treated as a URL scheme.
    return scheme in _LOCAL_SCHEMES or len(scheme) == 1


def parse_uri(uri: str) -> tuple[str, str]:
    """Return ``(scheme, uri)``; bare/file paths normalize to scheme ``""``."""
    split = urlsplit(uri)
    scheme = split.scheme.lower()
    if _looks_local(scheme, uri):
        return "", uri
    return scheme, uri


def _safe_label(value: str) -> str:
    return _SAFE_LABEL.sub("_", value).strip("._")[:60] or "records"


def _materialize_database(uri: str, cache_dir: Path) -> ResolvedSource:
    """Read a table/query from a SQLAlchemy URL into local Parquet."""
    try:
        import pandas as pd
        import sqlalchemy
    except ImportError as exc:  # pragma: no cover
        raise SourceError(f"database sources require SQLAlchemy: {exc}") from exc

    url, _, fragment = uri.partition("#")
    fragment = unquote(fragment)
    if fragment.lower().startswith("query="):
        sql = fragment[len("query="):]
        label = "query"
        table = None
    elif fragment:
        sql = None
        table = fragment
        label = table
    else:
        raise SourceError(
            "database URI must select a table or read-only query using a fragment"
        )

    engine = sqlalchemy.create_engine(url)
    planner_metadata: dict[str, Any] = {}
    try:
        if sql is None:
            inspector = sqlalchemy.inspect(engine)
            available = set(inspector.get_table_names())
            if table not in available:
                raise SourceError(
                    f"table {table!r} not found (available: {sorted(available)})"
                )
            with engine.connect() as connection:
                native_stats = database_statistics(inspector, connection, table)
            # What the planner may see of the whole database (tables, typed
            # columns, indexes, partitions, row counts; no row values), through
            # the connector contract on this same engine.
            from avaloka.connectors.postgres import connector_for
            planner_metadata = connector_for(url, table=table, engine=engine).planner_metadata()
            frame = pd.read_sql_table(table, engine)
        else:
            if not sql.lstrip().lower().startswith(("select", "with")):
                raise SourceError("only read-only SELECT/CTE database queries are allowed")
            frame = pd.read_sql_query(sqlalchemy.text(sql), engine)
            native_stats = {"kind": "database_query", "backend": urlsplit(url).scheme.lower(),
                "indexes": [], "partitioning": {"partitioned": False, "columns": [], "partitions": 1},
                "statistics_partial": True}
    finally:
        engine.dispose()

    out = cache_dir / f"db_{_safe_label(label)}.parquet"
    frame.to_parquet(out, index=False)
    return ResolvedSource(
        uri=_redact_uri(uri),
        scheme=urlsplit(url).scheme.lower(),
        kind="database",
        local_path=out,
        materialized=True,
        detail={
            "backend": urlsplit(url).scheme.lower(),
            "selector": label,
            "n_rows": int(frame.shape[0]),
            "n_cols": int(frame.shape[1]),
            "statistics": native_stats,
            "planner_metadata": planner_metadata,
        },
    )


def _json_option(options: dict[str, list[str]], name: str, default: Any) -> Any:
    raw = options.get(name, [None])[0]
    if raw in (None, ""):
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SourceError(f"MongoDB selector option {name!r} must be valid JSON") from exc


def _mongo_selector(uri: str) -> tuple[str, dict[str, Any]]:
    """Parse a fragment without exposing connection-query credentials.

    Supported forms::

        mongodb://host/db#reviews
        mongodb://host/db#collection=reviews&filter={...}&projection={...}&limit=500
    """
    raw_fragment = urlsplit(uri).fragment
    if not raw_fragment:
        raise SourceError("MongoDB URI must select a collection in its fragment")
    if "=" not in raw_fragment:
        return unquote(raw_fragment), {"filter": {}, "projection": None, "sort": None, "limit": None}

    # parse_qs performs percent-decoding after separating fields. Unquoting the
    # entire fragment first would corrupt JSON strings containing '&' or '='.
    options = parse_qs(raw_fragment, keep_blank_values=True)
    collection = options.get("collection", [""])[0].strip()
    if not collection:
        raise SourceError("MongoDB selector requires collection=<name>")
    limit_raw = options.get("limit", [None])[0]
    try:
        requested_limit = int(limit_raw) if limit_raw not in (None, "") else None
    except ValueError as exc:
        raise SourceError("MongoDB selector limit must be an integer") from exc
    if requested_limit is not None and requested_limit <= 0:
        raise SourceError("MongoDB selector limit must be greater than zero")
    return collection, {
        "filter": _json_option(options, "filter", {}),
        "projection": _json_option(options, "projection", None),
        "sort": _json_option(options, "sort", None),
        "limit": requested_limit,
    }


def _materialize_mongodb(uri: str, cache_dir: Path) -> ResolvedSource:
    """Read a bounded MongoDB collection selection into normalized Parquet."""
    collection_name, selector = _mongo_selector(uri)
    try:
        from pymongo import MongoClient
    except ImportError as exc:  # pragma: no cover - optional connector
        raise SourceError(
            "MongoDB sources require pymongo; install avaloka[sources]"
        ) from exc

    split = urlsplit(uri)
    connection_uri = urlunsplit((split.scheme, split.netloc, split.path, split.query, ""))
    maximum = int(os.getenv("AVALOKA_MONGO_MAX_DOCUMENTS", "100000"))
    limit = min(selector["limit"] or maximum, maximum)
    timeout_ms = int(os.getenv("AVALOKA_MONGO_TIMEOUT_MS", "30000"))

    client = MongoClient(connection_uri, serverSelectionTimeoutMS=timeout_ms)
    try:
        database = client.get_default_database()
        if database is None:
            raise SourceError("MongoDB URI must include a database name")
        collection = database[collection_name]
        cursor = collection.find(selector["filter"], selector["projection"])
        sort = selector["sort"]
        if isinstance(sort, dict):
            cursor = cursor.sort(list(sort.items()))
        elif isinstance(sort, list) and sort:
            cursor = cursor.sort([tuple(item) for item in sort])
        documents = list(cursor.limit(limit))
    except SourceError:
        raise
    except Exception as exc:
        raise SourceError(
            f"MongoDB read failed for collection {collection_name!r}: {type(exc).__name__}"
        ) from exc
    finally:
        client.close()

    frame, structure = normalize_records(documents)
    out = cache_dir / f"mongo_{_safe_label(collection_name)}.parquet"
    frame.to_parquet(out, index=False)
    return ResolvedSource(
        uri=_redact_uri(uri),
        scheme=split.scheme.lower(),
        kind="document_database",
        local_path=out,
        materialized=True,
        detail={
            "backend": "mongodb",
            "selector": collection_name,
            "n_rows": int(frame.shape[0]),
            "n_cols": int(frame.shape[1]),
            "limit": limit,
            "bounded": True,
            "structure": structure,
        },
    )


def _materialize_object_storage(
    uri: str,
    scheme: str,
    cache_dir: Path,
    *,
    max_bytes: int | None = None,
) -> ResolvedSource:
    """Stream a cloud object to a bounded local file through fsspec."""
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover
        raise SourceError(f"object-storage sources require fsspec: {exc}") from exc

    clean = uri.split("?", 1)[0]
    suffix = Path(urlsplit(clean).path).suffix.lower() or ".csv"
    protocol = {"gcs": "gs", "abfs": "az", "abfss": "az"}.get(scheme, scheme)
    out = cache_dir / f"obj_{abs(hash(uri)) % (10 ** 10)}{suffix}"
    total = 0
    try:
        with fsspec.open(uri, "rb") as source, out.open("wb") as destination:
            while True:
                chunk = source.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise SourceError(
                        f"object exceeds materialization limit of {max_bytes} bytes: "
                        f"{_redact_uri(uri)}"
                    )
                destination.write(chunk)
    except SourceError:
        out.unlink(missing_ok=True)
        raise
    except ImportError as exc:
        out.unlink(missing_ok=True)
        raise SourceError(
            f"missing filesystem driver for {scheme!r}; install its fsspec backend"
        ) from exc
    except FileNotFoundError as exc:
        out.unlink(missing_ok=True)
        raise SourceError(f"object not found: {_redact_uri(uri)}") from exc
    except Exception as exc:
        out.unlink(missing_ok=True)
        raise SourceError(
            f"unable to materialize object {_redact_uri(uri)}: {exc}"
        ) from exc

    return ResolvedSource(
        uri=_redact_uri(uri), scheme=protocol, kind="object_storage",
        local_path=out, materialized=True,
        detail={"protocol": protocol, "bytes": total, "bounded": max_bytes is not None,
                "statistics": file_statistics(out)},
    )


def _materialize_local(uri: str) -> ResolvedSource:
    path = Path(uri[len("file://"):] if uri.startswith("file://") else uri).expanduser()
    if not path.exists():
        raise FileNotFoundError(uri)
    return ResolvedSource(
        uri=uri, scheme="file", kind="file", local_path=path, materialized=False,
        detail={"statistics": file_statistics(path)},
    )


def materialize(
    uri: str,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    max_bytes: int | None = None,
) -> ResolvedSource:
    """Resolve any supported source URI to a normalized local data file."""
    scheme, _ = parse_uri(uri)
    if scheme in _LOCAL_SCHEMES:
        return _materialize_local(uri)
    cache = _cache_dir(cache_dir)
    if scheme in _FSSPEC_SCHEMES:
        return _materialize_object_storage(uri, scheme, cache, max_bytes=max_bytes)
    if scheme in _MONGO_SCHEMES:
        return _materialize_mongodb(uri, cache)
    return _materialize_database(uri, cache)


def route(uri: str, *, cache_dir: str | os.PathLike[str] | None = None):
    """Resolve and size a URI, returning ``(ResolvedSource, WorkloadPlan)``."""
    from avaloka import workload

    resolved = materialize(uri, cache_dir=cache_dir)
    plan = workload.route(str(resolved.local_path))
    return resolved, plan
