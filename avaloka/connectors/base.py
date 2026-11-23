"""The connector contract: one shape every data source proves the same way.

A connector answers five questions, and the planner sees one document shape
whatever the source behind it:

1. ``catalogue()`` -- tables (or collections) and their typed columns.
2. ``statistics(table)`` -- row count, size, indexes, partitions, and the
   per-column null and distinct counts the source exposes.
3. ``planner_metadata(max_bytes)`` -- the subset of 1 and 2 a planner prompt
   receives and the trace records: size-capped, never a row value.
4. ``pushdown(predicates, projection)`` -- the query the source's own
   optimizer gets, built from the workload optimizer's plan
   (``workload_optimization.json``: ``pushdown_filters`` and
   ``projection_columns``), parameterised, SELECT-only, identifiers quoted.
5. ``credentials`` -- resolved in this process, never serialised, never in a
   log, and the only way out is a method whose name a grep finds.

Part B of ``docs/design/1.7-DATA-PLATFORM.md`` adds **capabilities** and the
methods the router, the catalog and the tests need, so a CSV upload, a
warehouse table and a Kafka topic are one shape:

6. ``capabilities`` -- the :class:`Capability` set this connector implements.
   Calling a method it did not declare is a :class:`ConnectorError`.
7. ``endpoints()`` -- every host:port it will open, for the egress check.
8. ``source_version(ref)`` -- what "this version" of the source means, stable
   for an unchanged source; the catalog writes a version when it changes.
9. ``read(ref, max_rows=, max_bytes=)`` and ``sample(ref, n=, method=)`` --
   bounded Arrow results that say when they stopped early, each with a
   :class:`ReadReceipt` (rows, bytes, query digest, never query text).
10. ``window(ref, start=, end=)`` -- a bounded window over an unbounded source
    (``STREAM`` connectors only; part D).

Only the standard library lives here; a connector module imports what
``avaloka.io`` already uses (SQLAlchemy, pandas, pyarrow) and nothing more.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence, runtime_checkable

#: Default cap on the serialised planner metadata document, in bytes.
PLANNER_METADATA_MAX_BYTES = 16_384

#: Default bounds on a read. A connector never returns more than this unless
#: the caller raises the bound explicitly; the receipt says when it stopped.
READ_MAX_ROWS = 1_000_000
READ_MAX_BYTES = 256 * 1024 * 1024

#: The sample methods a connector may report. ``head`` is the first rows,
#: ``random`` an order-by-random, ``keyed`` every k-th key, ``tablesample`` the
#: engine's own block sampling. The method actually used is in the receipt.
SAMPLE_METHODS = ("head", "random", "keyed", "tablesample")

#: A method that may come back short, and the one a connector re-draws with
#: when it does: block sampling can miss every row of a small table, and a
#: sample returns min(n, rows). The receipt then names the fallback.
SAMPLE_FALLBACKS = {"tablesample": "random"}


class Capability(str, Enum):
    """What a connector can do. Declared, then proven by the conformance suite."""

    READ = "read"           # bounded read into Arrow batches
    PUSHDOWN = "pushdown"   # the source runs filters/projections itself
    WRITE = "write"         # a sink; off unless declared (D15.4)
    SCHEMA = "schema"       # catalogue(): tables/objects/topics and typed columns
    SAMPLE = "sample"       # a bounded, method-labelled sample
    STREAM = "stream"       # a bounded window over an unbounded source (part D)

#: Operators the workload optimizer emits (``avaloka/fireflies/workload_optimizer.py``)
#: and the SQL each becomes. Anything else is refused, not passed through.
OPERATORS: Mapping[str, str] = {
    "eq": "=", "=": "=", "==": "=",
    "ne": "<>", "!=": "<>",
    "lt": "<", "<": "<", "lte": "<=", "<=": "<=",
    "gt": ">", ">": ">", "gte": ">=", ">=": ">=",
    "in": "IN",
    "is_null": "IS NULL", "null": "IS NULL",
    "not_null": "IS NOT NULL",
}


class ConnectorError(RuntimeError):
    """A contract violation: unknown table or column, an operator outside the plan shape."""


class Credentials:
    """A secret that stays in this process.

    It cannot be pickled, JSON-encoded, or printed: ``repr`` and ``str`` are
    redacted, ``json.dumps`` refuses it, and ``pickle`` raises. The one way to
    the secret is ``reveal()``, so every use is greppable.
    """

    __slots__ = ("_secret",)

    def __init__(self, secret: str) -> None:
        if not isinstance(secret, str) or not secret:
            raise ConnectorError("credentials require a non-empty secret")
        self._secret = secret

    def reveal(self) -> str:
        """The secret, for the driver that opens the connection and nothing else."""
        return self._secret

    def __repr__(self) -> str:
        return "Credentials(<redacted>)"

    __str__ = __repr__

    def __reduce__(self):  # pickle, copy.deepcopy
        raise TypeError("Credentials are never serialised")

    def __getstate__(self):
        raise TypeError("Credentials are never serialised")

    def __eq__(self, other: object) -> bool:
        return self is other

    __hash__ = object.__hash__


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    nullable: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type, "nullable": self.nullable}


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    schema: Optional[str] = None

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "schema": self.schema,
                "columns": [c.as_dict() for c in self.columns]}


@dataclass(frozen=True)
class Predicate:
    """One ``pushdown_filters`` entry: ``{"column", "operator", "value"}``."""

    column: str
    operator: str
    value: Any = None

    @classmethod
    def from_plan(cls, item: Mapping[str, Any]) -> "Predicate":
        return cls(column=str(item.get("column", "")),
                   operator=str(item.get("operator", "eq")).lower(),
                   value=item.get("value"))

    @property
    def sql_operator(self) -> str:
        try:
            return OPERATORS[self.operator]
        except KeyError:
            raise ConnectorError(f"operator {self.operator!r} is outside the plan shape") from None


@dataclass(frozen=True)
class Query:
    """A pushed-down read: SELECT-only, identifiers quoted, values bound by name."""

    sql: str
    parameters: dict[str, Any] = field(default_factory=dict)
    table: str = ""
    columns: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"sql": self.sql, "parameters": dict(self.parameters),
                "table": self.table, "columns": list(self.columns)}


@dataclass(frozen=True)
class Endpoint:
    """One destination a connector will open: what the egress check sees.
    Never a credential, never a path; host and port only."""

    host: str
    port: int
    scheme: str = "tcp"

    def as_dict(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port, "scheme": self.scheme}


@dataclass(frozen=True)
class SourceVersion:
    """What "this version" of a source means, in the source's own terms: an
    ETag set, ``last_altered``, a content hash, a row count and max key. It is
    the same for an unchanged source and differs when the source changed; the
    catalog writes a new dataset version only when it differs."""

    ref: str
    marker: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "marker": dict(self.marker)}

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True, default=str).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class ReadReceipt:
    """What a read, sample or push-down returned, for lineage, the catalog and
    run truth. No row value, no query text (a digest), no credential."""

    connector: str
    ref: str
    rows: int
    bytes: int
    truncated: bool
    stopped_by: Optional[str] = None          # "max_rows" | "max_bytes" | None
    method: Optional[str] = None              # a sample's method, when it is one
    query_digest: Optional[str] = None        # sha256 of the SQL text, never the text
    source_version: Optional[SourceVersion] = None
    columns: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "connector": self.connector, "ref": self.ref, "rows": self.rows, "bytes": self.bytes,
            "truncated": self.truncated, "stopped_by": self.stopped_by, "method": self.method,
            "query_digest": self.query_digest,
            "source_version": self.source_version.as_dict() if self.source_version else None,
            "columns": list(self.columns),
        }


class ReadResult:
    """A bounded result: an Arrow table and the receipt that describes it."""

    __slots__ = ("table", "receipt")

    def __init__(self, table: Any, receipt: ReadReceipt) -> None:
        self.table = table
        self.receipt = receipt

    @property
    def truncated(self) -> bool:
        return self.receipt.truncated

    def __len__(self) -> int:
        return int(self.table.num_rows)

    def batches(self) -> Iterable[Any]:
        return self.table.to_batches()

    def to_pandas(self):
        return self.table.to_pandas()


@dataclass(frozen=True)
class WindowSnapshot:
    """A bounded window over a stream, as offsets (part D). Declared here so
    the contract has one shape for ``STREAM``; no connector in this tree
    implements it yet."""

    ref: str
    start: Any
    end: Any
    offsets: Mapping[str, Any]
    receipt: ReadReceipt


def query_digest(sql: str) -> str:
    """The only form in which a query leaves the connector."""
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()[:24]


def bounded_batches(batches: Iterable[Any], *, max_rows: int, max_bytes: int) -> tuple[list[Any], int, int, Optional[str]]:
    """Take Arrow batches until a bound is reached, trimming the last so that
    neither bound is exceeded. Returns ``(kept, rows, bytes, stopped_by)``;
    ``stopped_by`` is ``None`` when the source ran out first.

    ``batches`` should yield one batch past the bound when it can, so a source
    that fits exactly is not reported as truncated.
    """
    if max_rows <= 0 or max_bytes <= 0:
        raise ConnectorError("max_rows and max_bytes must be positive")
    kept: list[Any] = []
    rows = used = 0
    stopped: Optional[str] = None
    for batch in batches:
        if batch.num_rows == 0:
            continue
        piece = batch
        if rows + piece.num_rows > max_rows:
            piece = piece.slice(0, max_rows - rows)
            stopped = "max_rows"
        if used + piece.nbytes > max_bytes:
            stopped = "max_bytes"
            while piece.num_rows > 0 and used + piece.nbytes > max_bytes:
                piece = piece.slice(0, piece.num_rows // 2)
        if piece.num_rows:
            kept.append(piece)
            rows += piece.num_rows
            used += piece.nbytes
        if stopped:
            break
        if rows >= max_rows:
            # exactly at the bound: only the next non-empty batch decides
            # whether anything was left behind
            for later in batches:
                if later.num_rows:
                    stopped = "max_rows"
                    break
            break
    return kept, rows, used, stopped


def requires(capability: Capability) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Guard a contract method: a connector that did not declare ``capability``
    refuses the call, whatever its class happens to implement."""
    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            declared = getattr(self, "capabilities", frozenset())
            if capability not in declared:
                raise ConnectorError(f"capability not declared: {capability.value} on {getattr(self, 'name', type(self).__name__)}")
            return fn(self, *args, **kwargs)
        return guarded
    return wrap


def predicates_from_plan(plan: Mapping[str, Any]) -> list[Predicate]:
    """The ``pushdown_filters`` of a ``workload_optimization.json`` document."""
    return [Predicate.from_plan(p) for p in plan.get("pushdown_filters") or [] if isinstance(p, Mapping)]


def projection_from_plan(plan: Mapping[str, Any]) -> list[str]:
    """The ``projection_columns`` of a ``workload_optimization.json`` document."""
    return [str(c) for c in plan.get("projection_columns") or []]


def serialised_size(doc: Any) -> int:
    return len(json.dumps(doc, default=str, separators=(",", ":")).encode("utf-8"))


def cap_metadata(doc: dict[str, Any], max_bytes: int = PLANNER_METADATA_MAX_BYTES) -> dict[str, Any]:
    """Shrink a planner-metadata document until it serialises within ``max_bytes``.

    Shedding order: per-table column statistics, then whole tables from the
    end of the list, then column lists. The document says what it lost
    (``truncated``, ``tables_total``) so the planner does not mistake a cap
    for a small database.
    """
    out = json.loads(json.dumps(doc, default=str))
    tables = out.get("tables") or []
    out["tables_total"] = len(tables)
    out["truncated"] = False
    if serialised_size(out) <= max_bytes:
        return out
    out["truncated"] = True
    for table in tables:
        stats = table.get("statistics")
        if isinstance(stats, dict):
            stats.pop("columns", None)
    while tables and serialised_size(out) > max_bytes:
        tables.pop()
    if serialised_size(out) > max_bytes:
        for table in tables:
            table["columns"] = [c.get("name") if isinstance(c, dict) else c for c in table.get("columns") or []]
    return out


@runtime_checkable
class Connector(Protocol):
    """What every source implements. ``avaloka.connectors.postgres`` is the proving one."""

    name: str
    capabilities: frozenset
    dialect: Optional[str]

    @property
    def credentials(self) -> Credentials: ...

    def catalogue(self) -> dict[str, Any]: ...

    def statistics(self, table: str) -> dict[str, Any]: ...

    def planner_metadata(self, max_bytes: int = PLANNER_METADATA_MAX_BYTES) -> dict[str, Any]: ...

    def pushdown(self, predicates: Iterable[Mapping[str, Any] | Predicate],
                 projection: Sequence[str], *, table: Optional[str] = None) -> Query: ...

    def endpoints(self) -> list[Endpoint]: ...

    def source_version(self, ref: Optional[str] = None) -> SourceVersion: ...

    def read(self, ref: Any = None, *, max_rows: int = READ_MAX_ROWS,
             max_bytes: int = READ_MAX_BYTES) -> ReadResult: ...

    def sample(self, ref: Optional[str] = None, *, n: int = 1000, method: str = "head") -> ReadResult: ...

    def window(self, ref: str, *, start: Any, end: Any) -> WindowSnapshot: ...


class ConnectorBase:
    """Default bodies for the capability-gated methods: every one refuses
    until a subclass declares the capability *and* implements the method.
    Subclasses set ``capabilities`` and override what they declare."""

    name = "connector"
    capabilities: frozenset = frozenset()
    dialect: Optional[str] = None

    def endpoints(self) -> list[Endpoint]:
        return []

    def locator_for(self, ref: Optional[str] = None) -> dict[str, Any]:
        """Where ``ref`` lives, as the catalog stores it: names and ids only.
        Never a credential, a DSN, a signed URL or a path with a query string."""
        return {"connector": self.name, "ref": ref}

    @requires(Capability.SCHEMA)
    def catalogue(self) -> dict[str, Any]:
        raise ConnectorError(f"{self.name} declares schema but does not implement catalogue()")

    @requires(Capability.PUSHDOWN)
    def pushdown(self, predicates: Iterable[Mapping[str, Any] | Predicate],
                 projection: Sequence[str], *, table: Optional[str] = None) -> Query:
        raise ConnectorError(f"{self.name} declares pushdown but does not implement it")

    @requires(Capability.READ)
    def read(self, ref: Any = None, *, max_rows: int = READ_MAX_ROWS,
             max_bytes: int = READ_MAX_BYTES) -> ReadResult:
        raise ConnectorError(f"{self.name} declares read but does not implement it")

    @requires(Capability.SAMPLE)
    def sample(self, ref: Optional[str] = None, *, n: int = 1000, method: str = "head") -> ReadResult:
        raise ConnectorError(f"{self.name} declares sample but does not implement it")

    @requires(Capability.STREAM)
    def window(self, ref: str, *, start: Any, end: Any) -> WindowSnapshot:
        raise ConnectorError(f"{self.name} declares stream but does not implement window()")

    @requires(Capability.WRITE)
    def write(self, ref: str, table: Any) -> ReadReceipt:
        raise ConnectorError(f"{self.name} declares write but does not implement it")
