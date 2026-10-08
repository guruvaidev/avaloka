"""Where data came from, what it feeds, and what breaks if it changes.

Asked where a number came from, the honest answer in most organisations is a
Slack thread, a stale wiki page, or the person who built the pipeline two years
ago. Avaloka is in an unusually good position to do better: it is not crawling
someone else's warehouse inferring relationships from table names — **it wrote
the analyses, so it knows the edges**. A derived dataset knows its parent
because Avaloka produced it.

This is a graph in four relational tables, traversed with recursive CTEs, in
the deployment's own Postgres. It was a SQLite file under ``$HOME`` first, and
that was wrong for a cluster in a way that did not crash: the graph is compiled
in two processes (the API and the LangGraph pod), each wrote its own file on
its own ephemeral layer, and "where did this come from?" answered "no recorded
lineage" for data the other pod had recorded. A volume cannot fix that — the
chart's storage is ReadWriteOnce — so the store lives where every pod already
looks. The lineage of an install is thousands of nodes, not millions; a graph
database is still not a prerequisite.

**The store is shared, so every row belongs to a tenant.** ``tenant_id`` is the
leading column of every primary key and a predicate in every query. A
:class:`LineageStore` is bound to one tenant at construction and has no method
that reads or writes across them. File names (node labels) are plaintext, and
one database now holds every customer's.

**Column names are data.** ``patient_hiv_status`` is a diagnosis;
``q3_layoffs_final`` discloses a business event. Names are stored as salted
digests, salted per tenant, with the plaintext in a side table. Be clear about
what that buys now: the vault sits in the same database as the graph, so the
digests protect an export of the graph tables, not the database itself. The
graph never leaves the install and is excluded from telemetry.

Location: ``AVALOKA_LINEAGE_DB_URL``, else ``POSTGRES_URL``. With neither, the
feature is off and says so once. There is deliberately no implicit SQLite
default: a store that quietly appears per pod is the bug this replaced. An
explicit ``sqlite:///`` URL is honoured — that is an operator's stated choice
for a single-process install, and it is what the hermetic tests use.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from sqlalchemy import (Column, Index, MetaData, String, Table, Text, bindparam, func,
                        select, text)

logger = logging.getLogger(__name__)

ENV_DB_URL = "AVALOKA_LINEAGE_DB_URL"
ENV_ENABLED = "AVALOKA_LINEAGE"

#: Traversals are bounded. A cycle from a bad write, or a pathological chain,
#: must not turn "where did this come from?" into an unbounded query.
MAX_DEPTH = 32

#: Lineage is recorded on the request path. An unreachable database may cost a
#: turn this long, once per RETRY_AFTER_S, and never longer.
CONNECT_TIMEOUT_S = 3

#: How long a failed store stays off before one caller tries again. Not
#: forever: on a rollout the API pod is routinely up before Postgres is.
RETRY_AFTER_S = 60.0

#: Rows per INSERT. Keeps a wide dataset under every driver's bind limit.
_CHUNK = 500

_FALSEY = {"0", "off", "false", "no", "disable", "disabled"}


class NodeKind(str, Enum):
    DATASET = "dataset"
    COLUMN = "column"
    ANALYSIS = "analysis"
    MODEL = "model"
    CONNECTION = "connection"


class EdgeKind(str, Enum):
    DERIVED_FROM = "derived_from"     # dataset  -> dataset (child -> parent)
    HAS_COLUMN = "has_column"         # dataset  -> column
    LOADED_FROM = "loaded_from"       # dataset  -> connection
    PRODUCED = "produced"             # analysis -> dataset
    TRAINED_ON = "trained_on"         # model    -> dataset
    USES_FEATURE = "uses_feature"     # model    -> column
    JOINS_ON = "joins_on"             # dataset  -> column
    CLASSIFIED_AS = "classified_as"   # column   -> pii kind (stored as a label)


@dataclass(frozen=True)
class Node:
    id: str
    kind: NodeKind
    label: str = ""
    attrs: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind.value, "label": self.label,
                "attrs": dict(self.attrs)}


@dataclass(frozen=True)
class Edge:
    src: str
    kind: EdgeKind
    dst: str
    attrs: Dict[str, Any] = field(default_factory=dict)


metadata = MetaData()

node_table = Table(
    "lineage_node", metadata,
    Column("tenant_id", String(128), primary_key=True),
    Column("id", Text, primary_key=True),
    Column("kind", String(32), nullable=False),
    Column("label", Text, nullable=False, default=""),
    Column("attrs", Text, nullable=False, default="{}"),
    Index("ix_lineage_node_kind", "tenant_id", "kind"),
)

edge_table = Table(
    "lineage_edge", metadata,
    Column("tenant_id", String(128), primary_key=True),
    Column("src", Text, primary_key=True),
    Column("kind", String(32), primary_key=True),
    Column("dst", Text, primary_key=True),
    Column("attrs", Text, nullable=False, default="{}"),
    Index("ix_lineage_edge_dst", "tenant_id", "dst", "kind"),
)

# Plaintext column names live here and nowhere else, so the graph tables can be
# exported or inspected without disclosing a schema.
vault_table = Table(
    "lineage_name_vault", metadata,
    Column("tenant_id", String(128), primary_key=True),
    Column("id", Text, primary_key=True),
    Column("name", Text, nullable=False),
)

meta_table = Table(
    "lineage_meta", metadata,
    Column("tenant_id", String(128), primary_key=True),
    Column("k", String(64), primary_key=True),
    Column("v", Text, nullable=False),
)


def database_url() -> Optional[str]:
    """Where lineage is stored, or ``None`` when it is switched off or unset.

    Read at call time, not import time, so an operator's change needs no
    rebuild and a test can flip it without reloading the module.
    """
    if os.environ.get(ENV_ENABLED, "").strip().lower() in _FALSEY:
        return None
    for name in (ENV_DB_URL, "POSTGRES_URL"):
        value = os.environ.get(name, "").strip()
        if value:
            # libpq and most hosting dashboards say postgres://; SQLAlchemy
            # only accepts postgresql://.
            if value.startswith("postgres://"):
                value = "postgresql://" + value[len("postgres://"):]
            return value
    return None


class _Backend:
    """One engine per database URL for the whole process.

    A failure to reach the database is remembered for RETRY_AFTER_S and
    reported once per outage, however many callers ask in the meantime.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._lock = threading.Lock()
        self._engine: Any = None
        self._retry_at = 0.0
        self._down = False
        self._seen: set = set()
        self.salts: Dict[str, str] = {}

    def _describe(self) -> str:
        try:
            from sqlalchemy.engine import make_url
            return make_url(self.url).render_as_string(hide_password=True)
        except Exception:  # noqa: BLE001
            return "<unparseable url>"

    def engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            if time.monotonic() < self._retry_at:
                return None
            try:
                engine = self._open()
            except Exception as exc:  # noqa: BLE001
                self._mark_down(exc)
                return None
            if self._down:
                logger.info("[lineage] store at %s is reachable again", self._describe())
            else:
                # Reached once per process: the engine is cached from here on,
                # and the only way back is through an outage, handled above.
                self._announce()
            self._down = False
            self._engine = engine
            return engine

    def _announce(self) -> None:
        """Say where lineage is going, once. Nobody should see SQLite in use
        and have to wonder whether it was chosen or defaulted into."""
        if not self.url.startswith("sqlite"):
            logger.info("[lineage] recording to %s", self._describe())
            return
        named_by = next((name for name in (ENV_DB_URL, "POSTGRES_URL")
                         if os.environ.get(name, "").strip() == self.url), None)
        logger.warning(
            "[lineage] recording to SQLite at %s because %s. This was chosen, not fallen back "
            "to: lineage has no default file. It is only correct when one process records "
            "and reads lineage; the API and LangGraph pods would each keep their own.",
            self._describe(),
            f"{named_by} names it" if named_by else "the caller passed that URL")

    def _open(self) -> Any:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import make_url
        from sqlalchemy.pool import StaticPool

        try:
            parsed = make_url(self.url)
        except Exception:  # noqa: BLE001
            # The parser's own message echoes the string, password included.
            raise ValueError("the lineage database URL could not be parsed") from None
        backend = parsed.get_backend_name()
        # hide_parameters: a failed statement must not put the column and file
        # names it was carrying into the log.
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
            raise ValueError(f"lineage supports postgresql and sqlite, not {backend!r}")
        # create_engine is where the DBAPI import happens, so a missing driver
        # lands in the caller's except with everything else.
        engine = create_engine(self.url, **kwargs)
        try:
            metadata.create_all(engine)
        except Exception:  # noqa: BLE001
            # Two pods creating the tables at the same moment: one loses the
            # race, and the tables exist by the time it looks again.
            metadata.create_all(engine)
        return engine

    def _mark_down(self, exc: BaseException) -> None:
        self._retry_at = time.monotonic() + RETRY_AFTER_S
        if not self._down:
            self._down = True
            logger.warning(
                "[lineage] store unavailable at %s; lineage is off and will be retried every "
                "%ds. Analyses are unaffected. (%s: %s)",
                self._describe(), int(RETRY_AFTER_S), type(exc).__name__,
                str(exc).splitlines()[0] if str(exc) else "")

    def failed(self, what: str, exc: BaseException) -> None:
        """Record a failed operation. Logged once per kind, not once per call."""
        from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

        lost = isinstance(exc, (OperationalError, InterfaceError)) or (
            isinstance(exc, DBAPIError) and exc.connection_invalidated)
        if lost and not self.url.startswith("sqlite"):
            with self._lock:
                engine, self._engine = self._engine, None
                self._mark_down(exc)
            if engine is not None:
                with contextlib.suppress(Exception):
                    engine.dispose()
            return
        key = (what, type(exc).__name__)
        with self._lock:
            first = key not in self._seen
            self._seen.add(key)
        if first:
            logger.warning("[lineage] %s failed; further failures of this kind are not logged",
                           what, exc_info=True)
        else:
            logger.debug("[lineage] %s failed again (%s)", what, type(exc).__name__)


_backends: Dict[str, _Backend] = {}
_backends_lock = threading.Lock()
_notices: set = set()


def _backend_for(url: str) -> _Backend:
    with _backends_lock:
        backend = _backends.get(url)
        if backend is None:
            backend = _backends[url] = _Backend(url)
        return backend


def _notice_once(key: str, message: str) -> None:
    with _backends_lock:
        if key in _notices:
            return
        _notices.add(key)
    logger.warning(message)


def reset_for_tests() -> None:
    """Drop every cached engine, salt and once-only notice."""
    with _backends_lock:
        backends = list(_backends.values())
        _backends.clear()
        _notices.clear()
    for backend in backends:
        if backend._engine is not None:
            backend._engine.dispose()


def _upsert(conn: Any, table: Table, rows: List[Dict[str, Any]], keys: Sequence[str],
            update: Sequence[str]) -> None:
    """One multi-row INSERT .. ON CONFLICT per chunk: a round trip, not one per row."""
    if conn.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    for start in range(0, len(rows), _CHUNK):
        stmt = insert(table).values(rows[start:start + _CHUNK])
        stmt = stmt.on_conflict_do_update(
            index_elements=list(keys), set_={name: stmt.excluded[name] for name in update})
        conn.execute(stmt)


class LineageStore:
    """One tenant's lineage graph. Safe to construct anywhere; never raises.

    Recording lineage must not be able to fail a user's analysis, so every
    operation is best-effort and a failure is logged once. A missing edge
    degrades a later answer; a raised exception loses the work that produced it.

    Constructing one is cheap: the engine is shared per process. With no
    tenant, or no configured database, the store is inert — writes are dropped
    and reads come back empty.
    """

    def __init__(self, tenant_id: Optional[str], url: Optional[str] = None) -> None:
        self.tenant_id = str(tenant_id).strip() if tenant_id else ""
        self.url = url or database_url()
        self._backend: Optional[_Backend] = None
        if not self.url:
            _notice_once(
                "unconfigured",
                f"[lineage] disabled: neither {ENV_DB_URL} nor POSTGRES_URL is set "
                f"(or {ENV_ENABLED} is off). Nothing is recorded and provenance questions "
                "have no answer. There is no local-file fallback.")
        elif not self.tenant_id:
            _notice_once(
                "no-tenant",
                "[lineage] a store was requested without a tenant; nothing is recorded for "
                "requests that carry no user id.")
        else:
            self._backend = _backend_for(self.url)
        self._lock = threading.RLock()
        self._depth = 0
        self._conn: Any = None
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._vault: Dict[str, Dict[str, Any]] = {}
        self._edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    # -- lifecycle ---------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._backend is not None and self._backend.engine() is not None

    def connect(self) -> Any:
        """The shared engine, or ``None`` while the store is off or unreachable."""
        return self._backend.engine() if self._backend is not None else None

    def close(self) -> None:
        """Release this handle. The pooled engine belongs to the process and stays."""
        with self._lock:
            self._release()
            self._discard()

    def _release(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()

    def _discard(self) -> None:
        self._nodes.clear()
        self._vault.clear()
        self._edges.clear()

    def _run(self, what: str, fn: Callable[[Any], Any], default: Any = None) -> Any:
        if self._backend is None:
            return default
        try:
            if self._conn is not None:
                try:
                    return fn(self._conn)
                except Exception:
                    # Nothing buffered has been sent yet, so this loses nothing
                    # and leaves the connection usable for the flush.
                    self._conn.rollback()
                    raise
            engine = self._backend.engine()
            if engine is None:
                return default
            with engine.connect() as conn:
                return fn(conn)
        except Exception as exc:  # noqa: BLE001
            self._backend.failed(what, exc)
            return default

    @contextlib.contextmanager
    def batch(self) -> Iterator["LineageStore"]:
        """Everything recorded inside is one transaction on one connection.

        A lineage event is a dataset, its columns and their edges — dozens of
        rows. Committing each separately is a network round trip per row on the
        request path. Inside a batch, writes are buffered and sent as three
        statements and one commit when the outermost block exits; reads in
        between share the connection and do not see the buffered writes. If the
        block raises, nothing is written: half an event is worse than none.
        """
        with self._lock:
            self._depth += 1
            if self._depth == 1 and self._backend is not None:
                engine = self._backend.engine()
                if engine is not None:
                    try:
                        self._conn = engine.connect()
                    except Exception as exc:  # noqa: BLE001
                        self._backend.failed("connect", exc)
        ok = False
        try:
            yield self
            ok = True
        finally:
            with self._lock:
                self._depth -= 1
                if self._depth == 0:
                    try:
                        if ok:
                            self._flush()
                    finally:
                        self._release()
                        self._discard()

    def _flush(self) -> None:
        nodes, vault, edges = (list(self._nodes.values()), list(self._vault.values()),
                               list(self._edges.values()))
        if not (nodes or vault or edges):
            return

        def write(conn: Any) -> None:
            if nodes:
                _upsert(conn, node_table, nodes, ("tenant_id", "id"), ("label", "attrs"))
            if vault:
                _upsert(conn, vault_table, vault, ("tenant_id", "id"), ("name",))
            if edges:
                _upsert(conn, edge_table, edges, ("tenant_id", "src", "kind", "dst"), ("attrs",))
            conn.commit()

        self._run("recording lineage", write)

    # -- name protection ---------------------------------------------------
    def _column_salt(self) -> Optional[str]:
        """Per-tenant salt, minted once. Without it the digests are a rainbow
        table away from plaintext, since column names are low entropy.

        Minted on its own connection, never inside a batch: a salt that was
        rolled back with its event would leave this process hashing with a
        value no other pod can read.
        """
        backend = self._backend
        if backend is None:
            return None
        cached = backend.salts.get(self.tenant_id)
        if cached is not None:
            return cached
        engine = backend.engine()
        if engine is None:
            return None
        where = (meta_table.c.tenant_id == self.tenant_id) & (meta_table.c.k == "column_salt")
        try:
            with engine.begin() as conn:
                salt = conn.execute(select(meta_table.c.v).where(where)).scalar_one_or_none()
            if salt is None:
                try:
                    with engine.begin() as conn:
                        conn.execute(meta_table.insert().values(
                            tenant_id=self.tenant_id, k="column_salt", v=secrets.token_hex(16)))
                except Exception:  # noqa: BLE001
                    pass  # another pod minted it first; read theirs
                with engine.begin() as conn:
                    salt = conn.execute(select(meta_table.c.v).where(where)).scalar_one_or_none()
        except Exception as exc:  # noqa: BLE001
            backend.failed("reading the column salt", exc)
            return None
        if salt is not None:
            backend.salts[self.tenant_id] = salt
        return salt

    def column_id(self, dataset_id: str, column_name: str) -> str:
        """A stable id for a column that does not disclose its name."""
        digest = hashlib.sha256(
            f"{self._column_salt() or 'unsalted'}\x00{dataset_id}\x00{column_name}".encode()
        ).hexdigest()[:20]
        return f"col:{digest}"

    def resolve_column(self, column_id: str) -> Optional[str]:
        """Plaintext name, for this tenant only."""
        return self._run("column lookup", lambda conn: conn.execute(
            select(vault_table.c.name).where(
                (vault_table.c.tenant_id == self.tenant_id) & (vault_table.c.id == column_id))
        ).scalar_one_or_none())

    def get_nodes(self, node_ids: Iterable[str]) -> List[Node]:
        """Read nodes by id in one query, without any column-vault contents."""
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return []

        def read(conn: Any) -> List[Node]:
            rows = conn.execute(
                select(node_table.c.id, node_table.c.kind, node_table.c.label, node_table.c.attrs)
                .where((node_table.c.tenant_id == self.tenant_id) & node_table.c.id.in_(ids))
            ).fetchall()
            return [Node(r[0], NodeKind(r[1]), r[2], json.loads(r[3])) for r in rows]

        return self._run("node lookup", read, [])

    def get_node(self, node_id: str) -> Optional[Node]:
        found = self.get_nodes([node_id])
        return found[0] if found else None

    def dataset_lineage(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Return a dataset, its direct parents, and its transitive ancestry."""
        dataset = self.get_node(dataset_id)
        if dataset is None or dataset.kind is not NodeKind.DATASET:
            return None

        parent_ids = self._run("parent lookup", lambda conn: [row[0] for row in conn.execute(
            select(edge_table.c.dst).where(
                (edge_table.c.tenant_id == self.tenant_id)
                & (edge_table.c.src == dataset_id)
                & (edge_table.c.kind == EdgeKind.DERIVED_FROM.value))
        ).fetchall()], [])

        def nodes(ids: Iterable[str]) -> List[Node]:
            return sorted(self.get_nodes(ids), key=lambda node: (node.label.casefold(), node.id))

        return {
            "dataset": dataset,
            "parents": nodes(parent_ids),
            "ancestors": nodes(self.ancestors(dataset_id)),
        }

    # -- writes ------------------------------------------------------------
    def add_node(self, node: Node, *, plaintext_name: Optional[str] = None) -> None:
        if self._backend is None:
            return
        with self.batch():
            self._nodes[node.id] = {
                "tenant_id": self.tenant_id, "id": node.id, "kind": node.kind.value,
                "label": node.label, "attrs": json.dumps(node.attrs, default=str)}
            if plaintext_name is not None:
                self._vault[node.id] = {
                    "tenant_id": self.tenant_id, "id": node.id, "name": plaintext_name}

    def add_edge(self, edge: Edge) -> None:
        if self._backend is None:
            return
        with self.batch():
            self._edges[(edge.src, edge.kind.value, edge.dst)] = {
                "tenant_id": self.tenant_id, "src": edge.src, "kind": edge.kind.value,
                "dst": edge.dst, "attrs": json.dumps(edge.attrs, default=str)}

    def record_dataset(self, dataset_id: str, *, label: str = "",
                       columns: Sequence[Tuple[str, str]] = (),
                       derived_from: Optional[str] = None,
                       analysis_id: Optional[str] = None,
                       connection_id: Optional[str] = None,
                       pii: Optional[Dict[str, str]] = None,
                       **attrs: Any) -> None:
        """Record a dataset and everything known about it in one transaction.

        ``columns`` is (name, dtype). ``pii`` maps a column name to its
        classification, so the compliance query in :meth:`models_touching_pii`
        works without a second pass.
        """
        # Without the salt a column id would be one no other pod can reproduce.
        if columns and self._column_salt() is None:
            columns = ()
        with self.batch():
            self.add_node(Node(dataset_id, NodeKind.DATASET, label, attrs))
            for name, dtype in columns:
                cid = self.column_id(dataset_id, name)
                self.add_node(Node(cid, NodeKind.COLUMN, "", {"dtype": dtype}),
                              plaintext_name=name)
                self.add_edge(Edge(dataset_id, EdgeKind.HAS_COLUMN, cid))
                kind = (pii or {}).get(name)
                if kind and kind != "none":
                    self.add_edge(Edge(cid, EdgeKind.CLASSIFIED_AS, f"pii:{kind}"))
            if derived_from:
                self.add_edge(Edge(dataset_id, EdgeKind.DERIVED_FROM, derived_from,
                                   {"analysis_id": analysis_id} if analysis_id else {}))
            if analysis_id:
                self.add_node(Node(analysis_id, NodeKind.ANALYSIS))
                self.add_edge(Edge(analysis_id, EdgeKind.PRODUCED, dataset_id))
            if connection_id:
                self.add_node(Node(connection_id, NodeKind.CONNECTION))
                self.add_edge(Edge(dataset_id, EdgeKind.LOADED_FROM, connection_id))

    def record_model(self, model_id: str, *, trained_on: str,
                     feature_columns: Sequence[str] = (), **attrs: Any) -> None:
        if feature_columns and self._column_salt() is None:
            feature_columns = ()
        with self.batch():
            self.add_node(Node(model_id, NodeKind.MODEL, attrs.pop("label", ""), attrs))
            self.add_edge(Edge(model_id, EdgeKind.TRAINED_ON, trained_on))
            for name in feature_columns:
                self.add_edge(Edge(model_id, EdgeKind.USES_FEATURE,
                                   self.column_id(trained_on, name)))

    # -- reads -------------------------------------------------------------
    def _traverse(self, start: str, kinds: Sequence[str], *, forward: bool,
                  max_depth: int = MAX_DEPTH) -> List[str]:
        """Bounded transitive closure. Depth-capped so a cycle cannot hang.

        The tenant predicate is on the recursive term: a walk only ever steps
        along this tenant's edges, whatever ids another tenant has reused.
        """
        from_col, to_col = ("src", "dst") if forward else ("dst", "src")
        sql = text(
            f"WITH RECURSIVE walk(id, depth) AS ("
            f"  SELECT CAST(:start AS TEXT), 0"
            f"  UNION"
            f"  SELECT e.{to_col}, walk.depth + 1 FROM {edge_table.name} e"
            f"  JOIN walk ON e.{from_col} = walk.id"
            f"  WHERE e.tenant_id = :tenant AND e.kind IN :kinds AND walk.depth < :max_depth"
            f") SELECT DISTINCT id FROM walk WHERE id != CAST(:start AS TEXT)"
        ).bindparams(bindparam("kinds", expanding=True))
        params = {"start": start, "tenant": self.tenant_id, "kinds": list(kinds),
                  "max_depth": max_depth}
        return self._run("traversal", lambda conn: [
            r[0] for r in conn.execute(sql, params).fetchall()], [])

    def ancestors(self, dataset_id: str) -> List[str]:
        """Everything this dataset was derived from. "Where did this come from?" """
        return self._traverse(dataset_id, [EdgeKind.DERIVED_FROM.value], forward=True)

    def descendants(self, dataset_id: str) -> List[str]:
        """Everything derived from this dataset — the blast radius of a change."""
        return self._traverse(dataset_id, [EdgeKind.DERIVED_FROM.value], forward=False)

    def impact_of_column_change(self, dataset_id: str, column_name: str) -> Dict[str, Any]:
        """What breaks if this column is renamed or dropped.

        The question the 2 a.m. pipeline failure asks, and the reason lineage is
        worth capturing at all.
        """
        cid = self.column_id(dataset_id, column_name)
        downstream = self.descendants(dataset_id)
        affected = [dataset_id, *downstream]
        mine = edge_table.c.tenant_id == self.tenant_id
        models = self._run("impact query", lambda conn: [r[0] for r in conn.execute(
            select(edge_table.c.src).where(
                mine & (edge_table.c.kind == EdgeKind.USES_FEATURE.value)
                & (edge_table.c.dst == cid))
            .union(select(edge_table.c.src).where(
                mine & (edge_table.c.kind == EdgeKind.TRAINED_ON.value)
                & edge_table.c.dst.in_(affected)))
        ).fetchall()], [])
        return {"column_id": cid, "column": column_name, "dataset": dataset_id,
                "downstream_datasets": downstream,
                "models_affected": sorted(models),
                "total_affected": len(affected) - 1 + len(models)}

    def models_touching_pii(self) -> List[Dict[str, Any]]:
        """Which models were trained on data that PASSES THROUGH personal data.

        The naive version — check the columns of the dataset the model was
        trained on — gives the wrong answer and gives it reassuringly. Personal
        data is usually dropped during cleaning, so a model trained on the
        cleaned frame has no PII column and looks clean, while its lineage runs
        straight through an email address. For a compliance question the
        ancestry is the answer, which is the whole reason this is a graph rather
        than a table.
        """
        models = self._run("pii query", lambda conn: conn.execute(
            select(edge_table.c.src, edge_table.c.dst).where(
                (edge_table.c.tenant_id == self.tenant_id)
                & (edge_table.c.kind == EdgeKind.TRAINED_ON.value))
        ).fetchall(), [])

        has, classified = edge_table.alias("h"), edge_table.alias("c")
        out: List[Dict[str, Any]] = []
        for model_id, trained_on in models:
            lineage = [trained_on, *self.ancestors(trained_on)]
            rows = self._run("pii lineage query", lambda conn: conn.execute(
                select(classified.c.dst, has.c.src).distinct()
                .select_from(has.join(classified, (classified.c.src == has.c.dst)
                                      & (classified.c.tenant_id == self.tenant_id)
                                      & (classified.c.kind == EdgeKind.CLASSIFIED_AS.value)))
                .where((has.c.tenant_id == self.tenant_id)
                       & (has.c.kind == EdgeKind.HAS_COLUMN.value)
                       & has.c.src.in_(lineage))
            ).fetchall(), [])
            if not rows:
                continue
            kinds = sorted({r[0].split(":", 1)[-1] for r in rows})
            via = sorted({r[1] for r in rows})
            out.append({"model": model_id, "pii_kinds": kinds,
                        **({"via": via} if via != [trained_on] else {})})
        return sorted(out, key=lambda d: d["model"])

    def orphans(self) -> List[str]:
        """Datasets nothing was derived from — candidates for deletion."""
        used = select(edge_table.c.dst).where(
            (edge_table.c.tenant_id == self.tenant_id)
            & edge_table.c.kind.in_([EdgeKind.DERIVED_FROM.value, EdgeKind.TRAINED_ON.value]))
        return sorted(self._run("orphan query", lambda conn: [r[0] for r in conn.execute(
            select(node_table.c.id).where(
                (node_table.c.tenant_id == self.tenant_id)
                & (node_table.c.kind == NodeKind.DATASET.value)
                & node_table.c.id.not_in(used))
        ).fetchall()], []))

    def stats(self) -> Dict[str, int]:
        def read(conn: Any) -> Dict[str, int]:
            nodes = dict(conn.execute(
                select(node_table.c.kind, func.count()).where(
                    node_table.c.tenant_id == self.tenant_id).group_by(node_table.c.kind)
            ).fetchall())
            edges = dict(conn.execute(
                select(edge_table.c.kind, func.count()).where(
                    edge_table.c.tenant_id == self.tenant_id).group_by(edge_table.c.kind)
            ).fetchall())
            return {"nodes": sum(nodes.values()), "edges": sum(edges.values()),
                    **{f"node_{k}": v for k, v in nodes.items()},
                    **{f"edge_{k}": v for k, v in edges.items()}}

        return self._run("stats query", read, {})
