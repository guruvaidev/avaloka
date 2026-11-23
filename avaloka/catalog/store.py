"""The dataset catalog: every dataset Avaloka touched, with its schema history.

Design: ``docs/design/1.7-DATA-PLATFORM.md`` part C. This is the base, personal
catalog that every edition has (the ``dataset_catalog`` capability). It records
*what Avaloka used*: an upload, a table it sampled, an output an analysis
produced -- its kind, where it lives (a locator, never a credential), the typed
columns it had each time it was seen, how many rows, and which analysis used
which version. So a user can say "the orders table from last week's churn
analysis" and the planner binds to exactly that dataset and version.

What it is not: a second lineage graph (``app/core/lineage.py`` holds the
edges; a catalog dataset id *is* the lineage dataset node id), and not a
governor of the customer's systems. Sharing, stewards, access requests,
certification and the audit trail are the Enterprise shared catalog
(``shared_catalog``) and are not here.

**Versions are written on change only.** A version is a schema fingerprint plus
the source's own version marker (an ETag set, ``last_altered``, a content
hash). Reading an unchanged table twice writes nothing; a retyped column writes
one new version. That is what makes the catalog a history rather than a log.

**Every query carries the owner.** The store is bound to one user at
construction and has no method that reads or writes another user's rows. In
the hosted database, row-level security with the same single predicate
(``owner_user_id``) is the second lock; the migration under
``ui/supabase/migrations/`` creates it, and nothing in this module can widen
visibility past ``personal`` because the trigger there refuses it.

**Locators carry no secrets.** A locator is ``{"connection_id": ..., "schema":
..., "table": ...}``, never a DSN or a key; :func:`check_locator` refuses the
usual secret-bearing keys at the door, and the egress guard test plants names
to prove nothing a user typed reaches an event.

Location: ``AVALOKA_CATALOG_DB_URL``, else whatever ``URL_FALLBACKS`` says: in
the application that is the lineage database, so the catalog and the graph it
indexes live in the same place; in the CLI it is a SQLite file under
``~/.avaloka``. This module imports nothing from ``app``: it is the CLI's
catalog as much as the server's, and the lineage graph it cooperates with is
handed in by the caller (``forget(lineage=...)``, ``add_terms(column_digest=...)``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy import (Column, DateTime, Index, Integer, MetaData, String, Table, Text, func,
                        select)

logger = logging.getLogger(__name__)

ENV_DB_URL = "AVALOKA_CATALOG_DB_URL"

#: Where the catalog goes when AVALOKA_CATALOG_DB_URL is unset: callables tried
#: in order, each returning a URL or None. The application registers the
#: lineage database here (app/catalog/store.py); the CLI registers its local
#: SQLite file (avaloka/catalog/local.py). The store itself has no default.
URL_FALLBACKS: List[Callable[[], Optional[str]]] = []

#: The dataset kinds the design registers (§C.2), and no others.
KINDS: Tuple[str, ...] = ("file", "object", "db_table", "warehouse_table", "lake_table",
                          "kafka_window", "derived")

#: Visibility values the table accepts. Only ``personal`` is reachable from this
#: module; the others exist for the Enterprise shared catalog and the hosted
#: trigger refuses them unless that capability is granted.
VISIBILITIES: Tuple[str, ...] = ("personal", "project", "organization")
STATES: Tuple[str, ...] = ("draft", "certified", "deprecated")
ROW_COUNT_KINDS: Tuple[str, ...] = ("exact", "estimate", "unknown")

#: Locator keys that would mean a credential was put where a name belongs.
_SECRET_KEYS = re.compile(
    r"(password|passwd|pwd|secret|token|api_?key|access_?key|private_?key|credential|dsn|"
    r"connection_?string|sas|signature)", re.IGNORECASE)
_SECRET_IN_VALUE = re.compile(r"://[^/\s]*:[^/\s]*@|[?&](X-Amz-Signature|sig|token)=", re.IGNORECASE)

CONNECT_TIMEOUT_S = 3


class CatalogError(RuntimeError):
    """A caller error: unknown kind, a locator carrying a secret, a bad reference."""


class CatalogUnavailable(RuntimeError):
    """No catalog database is configured or reachable. Registration from the
    analysis path catches this and carries on; the API answers 503."""


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

metadata = MetaData()

dataset_table = Table(
    "catalog_dataset", metadata,
    Column("id", Text, primary_key=True),
    Column("owner_user_id", String(128), nullable=False),
    Column("organization_id", String(128), nullable=True),
    Column("project_id", String(128), nullable=True),
    Column("kind", String(32), nullable=False),
    Column("locator", Text, nullable=False, default="{}"),
    Column("locator_key", String(64), nullable=False),
    Column("display_name", Text, nullable=False, default=""),
    Column("visibility", String(16), nullable=False, default="personal"),
    Column("state", String(16), nullable=False, default="draft"),
    Column("tags", Text, nullable=False, default="[]"),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("deleted_at", DateTime(timezone=True), nullable=True),
    Index("ix_catalog_dataset_owner", "owner_user_id", "kind"),
    Index("ix_catalog_dataset_locator", "owner_user_id", "locator_key"),
)

snapshot_table = Table(
    "source_schema_snapshots", metadata,
    Column("id", Text, primary_key=True),
    Column("owner_user_id", String(128), nullable=False),
    Column("fingerprint", String(64), nullable=False),
    Column("columns", Text, nullable=False, default="[]"),
    Column("column_count", Integer, nullable=False, default=0),
    Column("captured_at", DateTime(timezone=True), nullable=False),
    Index("ix_source_schema_snapshots_owner", "owner_user_id", "fingerprint"),
)

version_table = Table(
    "catalog_dataset_version", metadata,
    Column("id", Text, primary_key=True),
    Column("dataset_id", Text, nullable=False),
    Column("owner_user_id", String(128), nullable=False),
    Column("version_no", Integer, nullable=False),
    Column("source_version", Text, nullable=True),
    Column("snapshot_id", Text, nullable=False),
    Column("row_count", Integer, nullable=True),
    Column("row_count_kind", String(16), nullable=False, default="unknown"),
    Column("bytes", Integer, nullable=True),
    Column("run_id", Text, nullable=True),
    Column("captured_at", DateTime(timezone=True), nullable=False),
    Column("captured_by", String(128), nullable=False),
    Index("ix_catalog_dataset_version_dataset", "dataset_id", "version_no", unique=True),
)

use_table = Table(
    "catalog_dataset_use", metadata,
    Column("analysis_id", Text, primary_key=True),
    Column("dataset_id", Text, primary_key=True),
    Column("owner_user_id", String(128), nullable=False),
    Column("version_id", Text, nullable=False),
    Column("used_at", DateTime(timezone=True), nullable=False),
    Index("ix_catalog_dataset_use_dataset", "owner_user_id", "dataset_id"),
)

term_table = Table(
    "catalog_term", metadata,
    Column("dataset_id", Text, primary_key=True),
    Column("term", Text, primary_key=True),
    Column("owner_user_id", String(128), nullable=False),
    Column("column_digest", Text, nullable=True),
    Column("added_at", DateTime(timezone=True), nullable=False),
)

TABLES: Tuple[str, ...] = tuple(t.name for t in metadata.sorted_tables)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def _digest(*parts: Any, n: int = 24) -> str:
    return hashlib.sha256("\x00".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:n]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def normalise_columns(columns: Any) -> List[Tuple[str, str]]:
    """``[(name, type)]`` from the shapes the agents carry: a dict, a list of
    dicts, a list of pairs, a pandas ``dtypes`` Series, or bare names."""
    if columns is None:
        return []
    if hasattr(columns, "items") and not isinstance(columns, dict):
        columns = dict(columns.items())  # pandas Series of dtypes
    if isinstance(columns, dict):
        columns = [(k, v) for k, v in columns.items()]
    out: Dict[str, str] = {}
    for item in columns:
        if isinstance(item, Mapping):
            name = item.get("name") or item.get("column")
            dtype = item.get("dtype") or item.get("type") or "unknown"
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            name, dtype = item
        else:
            name, dtype = item, "unknown"
        if name is not None and str(name) != "":
            out[str(name)] = str(dtype).strip().lower() or "unknown"
    return list(out.items())


def schema_fingerprint(columns: Any) -> str:
    """One fingerprint for a typed schema, shared with the monitoring
    snapshots (item 10a): sha256 over the sorted ``(name, type)`` pairs.

    Column order is not part of the schema: a CSV with the same columns in a
    different order is the same shape to every recipe. A rename, an added or
    dropped column, or a retype is a new fingerprint.
    """
    pairs = sorted(normalise_columns(columns))
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode("utf-8")).hexdigest()


def canonical_locator(locator: Mapping[str, Any]) -> str:
    return json.dumps({str(k): locator[k] for k in sorted(locator)}, sort_keys=True,
                      separators=(",", ":"), default=str)


def check_locator(locator: Mapping[str, Any]) -> Dict[str, Any]:
    """A locator names where a dataset lives. It is refused when a key or a
    value looks like a credential: those belong in a sealed connection record,
    which the locator points at by ``connection_id``."""
    if not isinstance(locator, Mapping):
        raise CatalogError("locator must be a mapping")
    for key, value in locator.items():
        if _SECRET_KEYS.search(str(key)):
            raise CatalogError(f"locator key {str(key)!r} names a credential; locators carry "
                               "names and ids only, credentials stay in the sealed connection")
        if isinstance(value, str) and _SECRET_IN_VALUE.search(value):
            raise CatalogError(f"locator value for {str(key)!r} carries a credential or a "
                               "signed query string")
        if isinstance(value, Mapping):
            check_locator(value)
    return {str(k): v for k, v in locator.items()}


def slug(name: str) -> str:
    """How a display name is addressed in chat and the CLI: ``my.<slug>``."""
    s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    s = re.sub(r"_(csv|parquet|json|xlsx?|tsv|avro|orc)$", "", s)
    return s or "dataset"


def size_bucket(n: Optional[int]) -> str:
    """The only thing an event says about a dataset's size."""
    if n is None:
        return "unknown"
    for limit, label in ((1_000, "<1k"), (100_000, "<100k"), (10_000_000, "<10m")):
        if n < limit:
            return label
    return ">=10m"


def catalog_event(action: str, *, kind: str, outcome: str, rows: Optional[int] = None,
                  columns: Optional[int] = None) -> Dict[str, str]:
    """A catalog event as telemetry may see it: action, kind, outcome and size
    buckets. No id, no name, no locator, no tag, no term. Every field is from a
    closed vocabulary so a leak is a type error here, not a review finding."""
    if kind not in KINDS:
        raise CatalogError(f"unknown dataset kind {kind!r}")
    return {"event": f"catalog.{action}", "kind": kind, "outcome": outcome,
            "rows": size_bucket(rows), "columns": size_bucket(columns)}


def schema_diff(before: Sequence[Tuple[str, str]], after: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    b, a = dict(before), dict(after)
    return {
        "added": sorted(set(a) - set(b)),
        "removed": sorted(set(b) - set(a)),
        "retyped": sorted((name, b[name], a[name]) for name in set(a) & set(b) if a[name] != b[name]),
    }


# --------------------------------------------------------------------------- #
# Location and engine
# --------------------------------------------------------------------------- #

def database_url() -> Optional[str]:
    """``AVALOKA_CATALOG_DB_URL``, else the first ``URL_FALLBACKS`` answer. ``None`` is off."""
    value = os.environ.get(ENV_DB_URL, "").strip()
    if value:
        if value.startswith("postgres://"):
            value = "postgresql://" + value[len("postgres://"):]
        return value
    for fallback in URL_FALLBACKS:
        found = fallback()
        if found:
            return found
    return None


class _Engines:
    """One engine per URL per process, created with the tables."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engines: Dict[str, Any] = {}

    def get(self, url: str) -> Any:
        with self._lock:
            engine = self._engines.get(url)
            if engine is not None:
                return engine
            from sqlalchemy import create_engine
            from sqlalchemy.engine import make_url
            from sqlalchemy.pool import StaticPool
            try:
                parsed = make_url(url)
            except Exception:  # noqa: BLE001
                raise CatalogUnavailable("the catalog database URL could not be parsed") from None
            backend = parsed.get_backend_name()
            kwargs: Dict[str, Any] = {"hide_parameters": True}
            if backend == "sqlite":
                kwargs["connect_args"] = {"check_same_thread": False}
                if parsed.database in (None, "", ":memory:"):
                    kwargs["poolclass"] = StaticPool
                else:
                    Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
            elif backend == "postgresql":
                kwargs.update(pool_pre_ping=True, pool_size=2, max_overflow=3, pool_recycle=1800,
                              connect_args={"connect_timeout": CONNECT_TIMEOUT_S})
            else:
                raise CatalogUnavailable(f"the catalog supports postgresql and sqlite, not {backend!r}")
            engine = create_engine(url, **kwargs)
            try:
                metadata.create_all(engine)
            except Exception:  # noqa: BLE001
                try:
                    metadata.create_all(engine)
                except Exception as exc:  # noqa: BLE001
                    engine.dispose()
                    raise CatalogUnavailable(f"catalog database unreachable ({type(exc).__name__})") from None
            self._engines[url] = engine
            return engine

    def reset(self) -> None:
        with self._lock:
            engines, self._engines = list(self._engines.values()), {}
        for engine in engines:
            with contextlib.suppress(Exception):
                engine.dispose()


_engines = _Engines()


def reset_for_tests() -> None:
    _engines.reset()


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Registration:
    dataset_id: str
    version_id: str
    version_no: int
    new_dataset: bool
    new_version: bool
    event: Dict[str, str] = field(default_factory=dict)

    @property
    def outcome(self) -> str:
        return "new_dataset" if self.new_dataset else ("new_version" if self.new_version else "unchanged")


@dataclass(frozen=True)
class Binding:
    """What the planner receives for "use that dataset": a dataset and the one
    version it must read, never a name."""
    dataset_id: str
    version_id: str
    version_no: int
    kind: str
    locator: Dict[str, Any]
    display_name: str
    columns: List[Tuple[str, str]]
    pinned_by: Optional[str] = None   # analysis id when the version was pinned by an analysis

    def as_dict(self) -> Dict[str, Any]:
        return {"dataset_id": self.dataset_id, "version_id": self.version_id,
                "version_no": self.version_no, "kind": self.kind, "locator": dict(self.locator),
                "display_name": self.display_name, "columns": [list(c) for c in self.columns],
                "pinned_by": self.pinned_by}


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #

class CatalogStore:
    """One user's catalog. Raises :class:`CatalogUnavailable` when there is no
    database, so callers decide whether that is fatal (the API) or not (the
    analysis path, through ``app.catalog.hooks``)."""

    def __init__(self, owner_user_id: Optional[str], url: Optional[str] = None,
                 *, on_event: Optional[Callable[[Dict[str, str]], None]] = None) -> None:
        self.owner = str(owner_user_id).strip() if owner_user_id else ""
        if not self.owner:
            raise CatalogError("the catalog needs an owner; nothing is recorded for requests "
                               "that carry no user id")
        self.url = url or database_url()
        if not self.url:
            raise CatalogUnavailable(f"no catalog database: set {ENV_DB_URL}")
        self._on_event = on_event or (lambda event: logger.debug("[catalog] %s", event))

    # -- plumbing ------------------------------------------------------------
    @property
    def engine(self) -> Any:
        return _engines.get(self.url)

    @contextlib.contextmanager
    def _tx(self):
        try:
            engine = self.engine
        except CatalogUnavailable:
            raise
        try:
            with engine.begin() as conn:
                yield conn
        except CatalogUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            from sqlalchemy.exc import InterfaceError, OperationalError
            if isinstance(exc, (OperationalError, InterfaceError)):
                raise CatalogUnavailable(f"catalog database unreachable ({type(exc).__name__})") from None
            raise

    def _emit(self, event: Dict[str, str]) -> None:
        try:
            self._on_event(event)
        except Exception:  # noqa: BLE001
            logger.debug("[catalog] event sink failed", exc_info=True)

    def _mine(self, table: Table):
        return table.c.owner_user_id == self.owner

    # -- registration --------------------------------------------------------
    def register(self, kind: str, locator: Mapping[str, Any], columns: Any, *,
                 display_name: str = "", dataset_id: Optional[str] = None,
                 source_version: Any = None, row_count: Optional[int] = None,
                 row_count_kind: str = "unknown", bytes: Optional[int] = None,
                 analysis_id: Optional[str] = None, run_id: Optional[str] = None,
                 tags: Iterable[str] = ()) -> Registration:
        """Record that this user's Avaloka touched a dataset.

        A dataset is identified by ``dataset_id`` when the caller has one (an
        upload id, a lineage output id), else by ``(kind, locator)``. A version
        is written only when the schema fingerprint or ``source_version``
        differs from the latest; otherwise nothing changes except that an
        ``analysis_id`` is recorded as having used the current version.
        """
        if kind not in KINDS:
            raise CatalogError(f"unknown dataset kind {kind!r}; one of {', '.join(KINDS)}")
        if row_count_kind not in ROW_COUNT_KINDS:
            raise CatalogError(f"row_count_kind must be one of {ROW_COUNT_KINDS}")
        locator = check_locator(locator)
        cols = normalise_columns(columns)
        fingerprint = schema_fingerprint(cols)
        source_version_text = (json.dumps(source_version, sort_keys=True, default=str)
                               if source_version is not None else None)
        locator_text = canonical_locator(locator)
        locator_key = _digest(kind, locator_text, n=48)
        now = _now()
        tag_list = sorted({str(t).strip() for t in tags if str(t).strip()})

        with self._tx() as conn:
            row = None
            if dataset_id:
                row = conn.execute(select(dataset_table).where(
                    self._mine(dataset_table) & (dataset_table.c.id == dataset_id)
                    & dataset_table.c.deleted_at.is_(None))).mappings().first()
            if row is None:
                row = conn.execute(select(dataset_table).where(
                    self._mine(dataset_table) & (dataset_table.c.locator_key == locator_key)
                    & dataset_table.c.deleted_at.is_(None))).mappings().first()
                if row is not None and dataset_id and row["id"] != dataset_id:
                    # The same place under a new id: the caller's id wins for
                    # lineage parity, and the old registration is the history.
                    row = None if row["kind"] != kind else row
            new_dataset = row is None
            if new_dataset:
                did = dataset_id or self._mint_id(conn, kind, locator_key)
                conn.execute(dataset_table.insert().values(
                    id=did, owner_user_id=self.owner, kind=kind, locator=locator_text,
                    locator_key=locator_key, display_name=str(display_name or ""),
                    visibility="personal", state="draft", tags=json.dumps(tag_list),
                    created_at=now, updated_at=now))
                latest = None
            else:
                did = row["id"]
                latest = conn.execute(select(version_table, snapshot_table.c.fingerprint)
                                      .join(snapshot_table, snapshot_table.c.id == version_table.c.snapshot_id)
                                      .where(version_table.c.dataset_id == did)
                                      .order_by(version_table.c.version_no.desc()).limit(1)).mappings().first()
                changes: Dict[str, Any] = {"updated_at": now}
                if display_name and not row["display_name"]:
                    changes["display_name"] = str(display_name)
                if tag_list:
                    changes["tags"] = json.dumps(sorted(set(json.loads(row["tags"] or "[]")) | set(tag_list)))
                conn.execute(dataset_table.update().where(dataset_table.c.id == did).values(**changes))

            unchanged = (latest is not None and latest["fingerprint"] == fingerprint
                         and latest["source_version"] == source_version_text)
            if unchanged:
                version_id, version_no = latest["id"], int(latest["version_no"])
                new_version = False
            else:
                snapshot_id = self._snapshot(conn, fingerprint, cols, now)
                version_no = 1 if latest is None else int(latest["version_no"]) + 1
                version_id = "dsv:" + _digest(self.owner, did, version_no)
                conn.execute(version_table.insert().values(
                    id=version_id, dataset_id=did, owner_user_id=self.owner, version_no=version_no,
                    source_version=source_version_text, snapshot_id=snapshot_id,
                    row_count=row_count, row_count_kind=row_count_kind if row_count is not None else "unknown",
                    bytes=bytes, run_id=run_id, captured_at=now, captured_by=self.owner))
                new_version = True
            if analysis_id:
                existing = conn.execute(select(use_table.c.version_id).where(
                    (use_table.c.analysis_id == analysis_id) & (use_table.c.dataset_id == did))).scalar_one_or_none()
                if existing is None:
                    conn.execute(use_table.insert().values(
                        analysis_id=analysis_id, dataset_id=did, owner_user_id=self.owner,
                        version_id=version_id, used_at=now))

        reg = Registration(did, version_id, version_no, new_dataset, new_version)
        event = catalog_event("register", kind=kind, outcome=reg.outcome, rows=row_count, columns=len(cols))
        self._emit(event)
        return Registration(did, version_id, version_no, new_dataset, new_version, event)

    def _mint_id(self, conn: Any, kind: str, locator_key: str) -> str:
        # Per owner: the same table under two users is two catalog entries,
        # and the id says nothing about the locator it was derived from.
        base = f"ds:{kind}:{_digest(self.owner, kind, locator_key)}"
        candidate, n = base, 1
        while conn.execute(select(dataset_table.c.id).where(dataset_table.c.id == candidate)).first():
            n += 1
            candidate = f"{base}-{n}"   # the tombstone keeps the first id
        return candidate

    def _snapshot(self, conn: Any, fingerprint: str, cols: Sequence[Tuple[str, str]], now: datetime) -> str:
        sid = "snap:" + _digest(self.owner, fingerprint)
        found = conn.execute(select(snapshot_table.c.id).where(snapshot_table.c.id == sid)).first()
        if found is None:
            conn.execute(snapshot_table.insert().values(
                id=sid, owner_user_id=self.owner, fingerprint=fingerprint,
                columns=json.dumps([list(c) for c in cols]), column_count=len(cols), captured_at=now))
        return sid

    # -- reads ---------------------------------------------------------------
    def _latest_versions(self, conn: Any, dataset_ids: Sequence[str]) -> Dict[str, Mapping[str, Any]]:
        if not dataset_ids:
            return {}
        rows = conn.execute(select(version_table, snapshot_table.c.columns, snapshot_table.c.column_count)
                            .join(snapshot_table, snapshot_table.c.id == version_table.c.snapshot_id)
                            .where(version_table.c.dataset_id.in_(list(dataset_ids)))
                            .order_by(version_table.c.dataset_id, version_table.c.version_no)).mappings().all()
        latest: Dict[str, Mapping[str, Any]] = {}
        for r in rows:
            latest[r["dataset_id"]] = r
        return latest

    def _summary(self, row: Mapping[str, Any], latest: Optional[Mapping[str, Any]],
                 uses: int = 0) -> Dict[str, Any]:
        cols = json.loads(latest["columns"]) if latest else []
        return {
            "dataset_id": row["id"], "kind": row["kind"], "display_name": row["display_name"],
            "name": f"my.{slug(row['display_name'])}" if row["display_name"] else None,
            "locator": json.loads(row["locator"] or "{}"), "tags": json.loads(row["tags"] or "[]"),
            "state": row["state"], "visibility": row["visibility"],
            "version_no": int(latest["version_no"]) if latest else 0,
            "version_id": latest["id"] if latest else None,
            "columns": len(cols), "column_names": [c[0] for c in cols],
            "bytes": latest["bytes"] if latest else None,
            "row_count": latest["row_count"] if latest else None,
            "row_count_kind": latest["row_count_kind"] if latest else "unknown",
            "captured_at": _iso(latest["captured_at"]) if latest else None,
            "created_at": _iso(row["created_at"]), "used_by": uses,
            "summary": self._sentence(row, latest, cols),
        }

    @staticmethod
    def _sentence(row: Mapping[str, Any], latest: Optional[Mapping[str, Any]], cols: List[Any]) -> str:
        """Code-written: the one line chat shows for a search hit."""
        name = row["display_name"] or row["id"]
        if latest is None:
            return f"{name}: {row['kind']}, no version recorded yet"
        rows = latest["row_count"]
        rows_text = (f"{rows:,} rows" if latest["row_count_kind"] == "exact" else
                     f"about {rows:,} rows" if rows is not None else "row count unknown")
        when = _iso(latest["captured_at"]) or ""
        return (f"{name}: {row['kind']}, {len(cols)} columns, {rows_text}, "
                f"version {int(latest['version_no'])}, captured {when[:10]}")

    def list(self, *, kind: Optional[str] = None, query: Optional[str] = None,
             since: Optional[datetime] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """This user's live datasets, newest first. ``query`` matches the
        display name, a tag, a glossary term or an analysis id that used it."""
        if kind is not None and kind not in KINDS:
            raise CatalogError(f"unknown dataset kind {kind!r}")
        with self._tx() as conn:
            where = self._mine(dataset_table) & dataset_table.c.deleted_at.is_(None)
            if kind:
                where &= dataset_table.c.kind == kind
            if since is not None:
                where &= dataset_table.c.updated_at >= since
            rows = conn.execute(select(dataset_table).where(where)
                                .order_by(dataset_table.c.updated_at.desc())).mappings().all()
            ids = [r["id"] for r in rows]
            latest = self._latest_versions(conn, ids)
            uses: Dict[str, List[str]] = {}
            if ids:
                for a, d in conn.execute(select(use_table.c.analysis_id, use_table.c.dataset_id)
                                         .where(self._mine(use_table) & use_table.c.dataset_id.in_(ids))).all():
                    uses.setdefault(d, []).append(a)
            terms: Dict[str, List[str]] = {}
            if ids:
                for d, t in conn.execute(select(term_table.c.dataset_id, term_table.c.term)
                                         .where(self._mine(term_table) & term_table.c.dataset_id.in_(ids))).all():
                    terms.setdefault(d, []).append(t)
        out = []
        needle = (query or "").strip().casefold()
        if needle.startswith("my."):
            needle = needle[3:]
        for r in rows:
            if needle:
                hay = [r["display_name"], slug(r["display_name"]), r["id"], *json.loads(r["tags"] or "[]"),
                       *terms.get(r["id"], []), *uses.get(r["id"], [])]
                if not any(needle in str(h).casefold() for h in hay if h):
                    continue
            out.append(self._summary(r, latest.get(r["id"]), len(uses.get(r["id"], []))))
            if len(out) >= limit:
                break
        return out

    find = list

    def get(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """One dataset in full: every version with its columns, the schema diff
        between the last two, the analyses that used each version, terms."""
        with self._tx() as conn:
            row = conn.execute(select(dataset_table).where(
                self._mine(dataset_table) & (dataset_table.c.id == dataset_id))).mappings().first()
            if row is None:
                return None
            versions = conn.execute(select(version_table, snapshot_table.c.columns, snapshot_table.c.fingerprint)
                                    .join(snapshot_table, snapshot_table.c.id == version_table.c.snapshot_id)
                                    .where(version_table.c.dataset_id == dataset_id)
                                    .order_by(version_table.c.version_no)).mappings().all()
            uses = conn.execute(select(use_table.c.analysis_id, use_table.c.version_id, use_table.c.used_at)
                                .where(self._mine(use_table) & (use_table.c.dataset_id == dataset_id))).mappings().all()
            terms = [t for (t,) in conn.execute(select(term_table.c.term).where(
                self._mine(term_table) & (term_table.c.dataset_id == dataset_id))).all()]
        if row["deleted_at"] is not None:
            return {"dataset_id": row["id"], "kind": row["kind"], "deleted_at": _iso(row["deleted_at"]),
                    "tombstone": True}
        by_version: Dict[str, List[str]] = {}
        for u in uses:
            by_version.setdefault(u["version_id"], []).append(u["analysis_id"])
        vs = []
        for v in versions:
            cols = [tuple(c) for c in json.loads(v["columns"])]
            vs.append({"version_id": v["id"], "version_no": int(v["version_no"]),
                       "source_version": json.loads(v["source_version"]) if v["source_version"] else None,
                       "columns": [list(c) for c in cols], "fingerprint": v["fingerprint"],
                       "row_count": v["row_count"], "row_count_kind": v["row_count_kind"],
                       "bytes": v["bytes"], "run_id": v["run_id"], "captured_at": _iso(v["captured_at"]),
                       "used_by": sorted(by_version.get(v["id"], []))})
        latest = versions[-1] if versions else None
        detail = self._summary(row, latest, len(uses))
        detail.update({
            "versions": vs,
            "schema_change": (schema_diff([tuple(c) for c in vs[-2]["columns"]],
                                          [tuple(c) for c in vs[-1]["columns"]]) if len(vs) >= 2 else None),
            "terms": sorted(terms),
            "analyses": sorted({u["analysis_id"] for u in uses}),
            "tombstone": False,
        })
        return detail

    # -- reuse ---------------------------------------------------------------
    def resolve(self, reference: str) -> Binding:
        """Turn what a user said into the one ``(dataset, version)`` to read.

        * ``my.<name>`` or ``<name>``: the dataset whose display name slugs to
          ``name``; the latest version.
        * ``my.<name>@<n>``: that dataset at version ``n``.
        * ``analysis:<id>`` or ``from analysis <id>``: the version that analysis
          used (pinned, not the latest), for the dataset it used; with several,
          ``analysis:<id>/<name>`` picks one.
        * a dataset id as stored: its latest version.
        """
        ref = str(reference or "").strip()
        if not ref:
            raise CatalogError("empty dataset reference")
        m = re.match(r"^(?:analysis:|from analysis\s+)(\S+?)(?:/(\S+))?$", ref, re.IGNORECASE)
        if m:
            return self._resolve_analysis(m.group(1), m.group(2))
        at = None
        if "@" in ref:
            ref, _, at_text = ref.rpartition("@")
            if not at_text.isdigit():
                raise CatalogError(f"version must be a number, got {at_text!r}")
            at = int(at_text)
        name = ref[3:] if ref.casefold().startswith("my.") else ref
        with self._tx() as conn:
            rows = conn.execute(select(dataset_table).where(
                self._mine(dataset_table) & dataset_table.c.deleted_at.is_(None))).mappings().all()
            matches = [r for r in rows if r["id"] == name or (r["display_name"] and slug(r["display_name"]) == slug(name))]
            if not matches:
                raise CatalogError(f"no dataset of yours is called {name!r}")
            if len(matches) > 1:
                exact = [r for r in matches if r["id"] == name]
                if len(exact) != 1:
                    ids = ", ".join(sorted(r["id"] for r in matches))
                    raise CatalogError(f"{name!r} names {len(matches)} of your datasets ({ids}); use the id")
                matches = exact
            row = matches[0]
            q = (select(version_table, snapshot_table.c.columns)
                 .join(snapshot_table, snapshot_table.c.id == version_table.c.snapshot_id)
                 .where(version_table.c.dataset_id == row["id"]))
            q = q.where(version_table.c.version_no == at) if at is not None else q.order_by(version_table.c.version_no.desc()).limit(1)
            v = conn.execute(q).mappings().first()
            if v is None:
                raise CatalogError(f"{name!r} has no version {at}" if at is not None else f"{name!r} has no version recorded")
        return self._binding(row, v, pinned_by=None)

    def _resolve_analysis(self, analysis_id: str, name: Optional[str]) -> Binding:
        with self._tx() as conn:
            uses = conn.execute(select(use_table).where(
                self._mine(use_table) & (use_table.c.analysis_id == analysis_id))).mappings().all()
            if not uses:
                raise CatalogError(f"no dataset of yours was used by analysis {analysis_id!r}")
            rows = {r["id"]: r for r in conn.execute(select(dataset_table).where(
                self._mine(dataset_table) & dataset_table.c.id.in_([u["dataset_id"] for u in uses]))).mappings().all()}
            live = [u for u in uses if rows.get(u["dataset_id"]) is not None and rows[u["dataset_id"]]["deleted_at"] is None]
            if name:
                live = [u for u in live if u["dataset_id"] == name or slug(rows[u["dataset_id"]]["display_name"]) == slug(name)]
            if not live:
                raise CatalogError(f"analysis {analysis_id!r} used no dataset that still exists" + (f" called {name!r}" if name else ""))
            if len(live) > 1:
                names = ", ".join(sorted(f"my.{slug(rows[u['dataset_id']]['display_name'])}" for u in live))
                raise CatalogError(f"analysis {analysis_id!r} used {len(live)} datasets ({names}); say which")
            use = live[0]
            v = conn.execute(select(version_table, snapshot_table.c.columns)
                             .join(snapshot_table, snapshot_table.c.id == version_table.c.snapshot_id)
                             .where(version_table.c.id == use["version_id"])).mappings().first()
            if v is None:
                raise CatalogError("the version that analysis used is gone")
            row = rows[use["dataset_id"]]
        return self._binding(row, v, pinned_by=analysis_id)

    @staticmethod
    def _binding(row: Mapping[str, Any], v: Mapping[str, Any], *, pinned_by: Optional[str]) -> Binding:
        return Binding(dataset_id=row["id"], version_id=v["id"], version_no=int(v["version_no"]),
                       kind=row["kind"], locator=json.loads(row["locator"] or "{}"),
                       display_name=row["display_name"], columns=[tuple(c) for c in json.loads(v["columns"])],
                       pinned_by=pinned_by)

    # -- terms and tags --------------------------------------------------------
    def add_terms(self, dataset_id: str, terms: Iterable[str], *, column: Optional[str] = None,
                  column_digest: Optional[str] = None) -> int:
        """User glossary terms for a dataset (or one of its columns, stored as
        the lineage column digest the caller supplies, never the name). Terms
        are the user's text in the user's database and are never sent anywhere."""
        now = _now()
        digest = column_digest if column else None
        added = 0
        with self._tx() as conn:
            if conn.execute(select(dataset_table.c.id).where(
                    self._mine(dataset_table) & (dataset_table.c.id == dataset_id)
                    & dataset_table.c.deleted_at.is_(None))).first() is None:
                raise CatalogError("no such dataset of yours")
            for term in {str(t).strip() for t in terms if str(t).strip()}:
                if conn.execute(select(term_table.c.term).where(
                        (term_table.c.dataset_id == dataset_id) & (term_table.c.term == term))).first():
                    continue
                conn.execute(term_table.insert().values(dataset_id=dataset_id, term=term, owner_user_id=self.owner,
                                                        column_digest=digest, added_at=now))
                added += 1
        return added

    # -- delete ----------------------------------------------------------------
    def forget(self, dataset_id: str, *, lineage: Any = None) -> Dict[str, Any]:
        """Delete a dataset from the catalog and the lineage graph.

        Removes its versions, uses and terms, and the schema snapshots nothing
        else of this user's references; leaves a nameless tombstone (the id,
        kind and time: no name, locator or tags). Removes its lineage nodes,
        column digests and vault rows when a lineage graph is handed in. Lists
        derived datasets rather than deleting them. Does not touch the customer's source system, and says
        which copies it did not reach: those are the job of the owning
        subsystems (uploads, sampling rows, the Redis schema key) and are
        reported, not claimed.
        """
        now = _now()
        with self._tx() as conn:
            row = conn.execute(select(dataset_table).where(
                self._mine(dataset_table) & (dataset_table.c.id == dataset_id))).mappings().first()
            if row is None:
                raise CatalogError("no such dataset of yours")
            if row["deleted_at"] is not None:
                return {"dataset_id": dataset_id, "already_deleted": True}
            snapshot_ids = [s for (s,) in conn.execute(select(version_table.c.snapshot_id).where(
                version_table.c.dataset_id == dataset_id)).all()]
            versions = conn.execute(version_table.delete().where(version_table.c.dataset_id == dataset_id)).rowcount
            uses = conn.execute(use_table.delete().where(use_table.c.dataset_id == dataset_id)).rowcount
            terms = conn.execute(term_table.delete().where(term_table.c.dataset_id == dataset_id)).rowcount
            snapshots = 0
            for sid in set(snapshot_ids):
                still = conn.execute(select(version_table.c.id).where(version_table.c.snapshot_id == sid).limit(1)).first()
                if still is None:
                    snapshots += conn.execute(snapshot_table.delete().where(snapshot_table.c.id == sid)).rowcount
            conn.execute(dataset_table.update().where(dataset_table.c.id == dataset_id).values(
                display_name="", locator="{}", tags="[]", project_id=None, organization_id=None,
                state="deprecated", deleted_at=now, updated_at=now))
        graph: Dict[str, Any] = {"nodes": 0, "edges": 0, "vault": 0}
        derived: List[str] = []
        if lineage is not None:   # the graph is the caller's: anything with descendants() and forget()
            derived = list(lineage.descendants(dataset_id))
            graph = lineage.forget(dataset_id)
        self._emit(catalog_event("forget", kind=row["kind"], outcome="deleted"))
        return {
            "dataset_id": dataset_id, "already_deleted": False, "tombstone": True,
            "removed": {"versions": int(versions or 0), "uses": int(uses or 0), "terms": int(terms or 0),
                        "snapshots": int(snapshots), "lineage": graph},
            "derived_kept": derived,
            "not_reached": ["upload object", "sampling rows", "redis schema cache", "landed copies"],
            "source_system": "untouched",
        }


__all__ = [
    "Binding", "CatalogError", "CatalogStore", "CatalogUnavailable", "ENV_DB_URL", "KINDS",
    "ROW_COUNT_KINDS", "Registration", "STATES", "TABLES", "URL_FALLBACKS", "VISIBILITIES",
    "canonical_locator", "catalog_event", "check_locator", "database_url", "metadata",
    "normalise_columns", "reset_for_tests", "schema_diff", "schema_fingerprint", "size_bucket", "slug",
]
