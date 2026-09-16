"""Where data came from, what it feeds, and what breaks if it changes.

Asked where a number came from, the honest answer in most organisations is a
Slack thread, a stale wiki page, or the person who built the pipeline two years
ago. Avaloka is in an unusually good position to do better: it is not crawling
someone else's warehouse inferring relationships from table names — **it wrote
the analyses, so it knows the edges**. A derived dataset knows its parent
because Avaloka produced it.

Nothing recorded that until now.

This is an embedded graph over SQLite, chosen deliberately over a graph
database. The lineage of a single install is thousands of nodes, not millions;
recursive CTEs traverse that in milliseconds; and SQLite is already present
everywhere Python is. A catalog that requires an operator to run a cluster does
not exist for the users who most need it — the same objection that ruled out a
Redis dependency for telemetry, applied to ourselves. :class:`LineageStore` is
an interface so a Neo4j backend can be added for cluster deployments as an
upgrade, never a prerequisite.

**Column names are data.** ``patient_hiv_status`` is a diagnosis;
``q3_layoffs_final`` discloses a business event. Names are stored as salted
digests with the plaintext held only in a local side table, so the graph is
useful for traversal and useless to anyone who obtains it without the salt. The
graph never leaves the install and is excluded from telemetry.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(os.getenv("AVALOKA_LINEAGE_PATH",
                              Path.home() / ".avaloka" / "lineage.db"))

#: Traversals are bounded. A cycle from a bad write, or a pathological chain,
#: must not turn "where did this come from?" into an unbounded query.
MAX_DEPTH = 32


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


_SCHEMA = """
CREATE TABLE IF NOT EXISTS node (
    id    TEXT PRIMARY KEY,
    kind  TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    attrs TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS edge (
    src   TEXT NOT NULL,
    kind  TEXT NOT NULL,
    dst   TEXT NOT NULL,
    attrs TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (src, kind, dst)
);
CREATE INDEX IF NOT EXISTS edge_src ON edge(src, kind);
CREATE INDEX IF NOT EXISTS edge_dst ON edge(dst, kind);
CREATE INDEX IF NOT EXISTS node_kind ON node(kind);
-- Plaintext column names live here and nowhere else, so the graph proper can be
-- copied, inspected or shared without disclosing a schema.
CREATE TABLE IF NOT EXISTS name_vault (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


class LineageStore:
    """Embedded lineage graph. Safe to construct anywhere; never raises on write.

    Recording lineage must not be able to fail a user's analysis, so every
    mutation is best-effort and logged once on failure. A missing edge degrades
    a later answer; a raised exception loses the work that produced it.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_PATH
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._salt: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> Optional[sqlite3.Connection]:
        if self._conn is not None:
            return self._conn
        try:
            if str(self.path) != ":memory:":
                self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), check_same_thread=False)
            conn.executescript(_SCHEMA)
            conn.commit()
            self._conn = conn
            return conn
        except Exception:  # noqa: BLE001
            logger.warning("[lineage] store unavailable at %s; lineage disabled",
                           self.path, exc_info=True)
            return None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- name protection ---------------------------------------------------
    def _column_salt(self) -> str:
        """Per-install salt, generated once. Without it the digests are a
        rainbow table away from plaintext, since column names are low entropy."""
        if self._salt is not None:
            return self._salt
        conn = self.connect()
        if conn is None:
            self._salt = "unsalted"
            return self._salt
        row = conn.execute("SELECT v FROM meta WHERE k='column_salt'").fetchone()
        if row:
            self._salt = row[0]
        else:
            self._salt = secrets.token_hex(16)
            conn.execute("INSERT INTO meta(k, v) VALUES('column_salt', ?)", (self._salt,))
            conn.commit()
        return self._salt

    def column_id(self, dataset_id: str, column_name: str) -> str:
        """A stable id for a column that does not disclose its name."""
        digest = hashlib.sha256(
            f"{self._column_salt()}\x00{dataset_id}\x00{column_name}".encode()
        ).hexdigest()[:20]
        return f"col:{digest}"

    def resolve_column(self, column_id: str) -> Optional[str]:
        """Plaintext name, locally only."""
        conn = self.connect()
        if conn is None:
            return None
        row = conn.execute("SELECT name FROM name_vault WHERE id=?", (column_id,)).fetchone()
        return row[0] if row else None

    # -- writes ------------------------------------------------------------
    def add_node(self, node: Node, *, plaintext_name: Optional[str] = None) -> None:
        conn = self.connect()
        if conn is None:
            return
        try:
            with self._lock:
                conn.execute(
                    "INSERT INTO node(id, kind, label, attrs) VALUES(?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET label=excluded.label, attrs=excluded.attrs",
                    (node.id, node.kind.value, node.label, json.dumps(node.attrs, default=str)))
                if plaintext_name is not None:
                    conn.execute("INSERT OR REPLACE INTO name_vault(id, name) VALUES(?,?)",
                                 (node.id, plaintext_name))
                conn.commit()
        except Exception:  # noqa: BLE001
            logger.warning("[lineage] could not record node %s", node.id, exc_info=True)

    def add_edge(self, edge: Edge) -> None:
        conn = self.connect()
        if conn is None:
            return
        try:
            with self._lock:
                conn.execute(
                    "INSERT INTO edge(src, kind, dst, attrs) VALUES(?,?,?,?) "
                    "ON CONFLICT(src, kind, dst) DO UPDATE SET attrs=excluded.attrs",
                    (edge.src, edge.kind.value, edge.dst,
                     json.dumps(edge.attrs, default=str)))
                conn.commit()
        except Exception:  # noqa: BLE001
            logger.warning("[lineage] could not record edge %s-%s->%s",
                           edge.src, edge.kind.value, edge.dst, exc_info=True)

    def record_dataset(self, dataset_id: str, *, label: str = "",
                       columns: Sequence[Tuple[str, str]] = (),
                       derived_from: Optional[str] = None,
                       analysis_id: Optional[str] = None,
                       connection_id: Optional[str] = None,
                       pii: Optional[Dict[str, str]] = None,
                       **attrs: Any) -> None:
        """Record a dataset and everything known about it in one call.

        ``columns`` is (name, dtype). ``pii`` maps a column name to its
        classification, so the compliance query in :meth:`models_touching_pii`
        works without a second pass.
        """
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
        self.add_node(Node(model_id, NodeKind.MODEL, attrs.pop("label", ""), attrs))
        self.add_edge(Edge(model_id, EdgeKind.TRAINED_ON, trained_on))
        for name in feature_columns:
            self.add_edge(Edge(model_id, EdgeKind.USES_FEATURE,
                               self.column_id(trained_on, name)))

    # -- reads -------------------------------------------------------------
    def _traverse(self, start: str, kinds: Sequence[str], *, forward: bool,
                  max_depth: int = MAX_DEPTH) -> List[str]:
        """Bounded transitive closure. Depth-capped so a cycle cannot hang."""
        conn = self.connect()
        if conn is None:
            return []
        from_col, to_col = ("src", "dst") if forward else ("dst", "src")
        placeholders = ",".join("?" * len(kinds))
        sql = (
            f"WITH RECURSIVE walk(id, depth) AS ("
            f"  SELECT ?, 0"
            f"  UNION"
            f"  SELECT e.{to_col}, walk.depth + 1 FROM edge e"
            f"  JOIN walk ON e.{from_col} = walk.id"
            f"  WHERE e.kind IN ({placeholders}) AND walk.depth < ?"
            f") SELECT DISTINCT id FROM walk WHERE id != ?")
        try:
            rows = conn.execute(sql, (start, *kinds, max_depth, start)).fetchall()
            return [r[0] for r in rows]
        except Exception:  # noqa: BLE001
            logger.warning("[lineage] traversal failed from %s", start, exc_info=True)
            return []

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
        conn = self.connect()
        cid = self.column_id(dataset_id, column_name)
        affected = [dataset_id, *self.descendants(dataset_id)]
        models: List[str] = []
        if conn is not None:
            try:
                marks = ",".join("?" * len(affected))
                models = [r[0] for r in conn.execute(
                    f"SELECT DISTINCT src FROM edge WHERE kind=? AND dst=? "
                    f"UNION SELECT DISTINCT src FROM edge WHERE kind=? AND dst IN ({marks})",
                    (EdgeKind.USES_FEATURE.value, cid,
                     EdgeKind.TRAINED_ON.value, *affected)).fetchall()]
            except Exception:  # noqa: BLE001
                logger.warning("[lineage] impact query failed", exc_info=True)
        return {"column_id": cid, "column": column_name, "dataset": dataset_id,
                "downstream_datasets": self.descendants(dataset_id),
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
        conn = self.connect()
        if conn is None:
            return []
        try:
            models = conn.execute("SELECT src, dst FROM edge WHERE kind=?",
                                  (EdgeKind.TRAINED_ON.value,)).fetchall()
        except Exception:  # noqa: BLE001
            logger.warning("[lineage] pii query failed", exc_info=True)
            return []

        out: List[Dict[str, Any]] = []
        for model_id, trained_on in models:
            lineage = [trained_on, *self.ancestors(trained_on)]
            marks = ",".join("?" * len(lineage))
            try:
                rows = conn.execute(
                    f"SELECT DISTINCT c.dst, h.src FROM edge h "
                    f"JOIN edge c ON c.src = h.dst AND c.kind = ? "
                    f"WHERE h.kind = ? AND h.src IN ({marks})",
                    (EdgeKind.CLASSIFIED_AS.value, EdgeKind.HAS_COLUMN.value,
                     *lineage)).fetchall()
            except Exception:  # noqa: BLE001
                logger.warning("[lineage] pii lineage query failed", exc_info=True)
                continue
            if not rows:
                continue
            kinds = sorted({r[0].split(":", 1)[-1] for r in rows})
            via = sorted({r[1] for r in rows})
            out.append({"model": model_id, "pii_kinds": kinds,
                        **({"via": via} if via != [trained_on] else {})})
        return sorted(out, key=lambda d: d["model"])

    def orphans(self) -> List[str]:
        """Datasets nothing was derived from — candidates for deletion."""
        conn = self.connect()
        if conn is None:
            return []
        rows = conn.execute(
            "SELECT n.id FROM node n WHERE n.kind=? AND NOT EXISTS "
            "(SELECT 1 FROM edge e WHERE e.dst = n.id AND e.kind IN (?,?))",
            (NodeKind.DATASET.value, EdgeKind.DERIVED_FROM.value,
             EdgeKind.TRAINED_ON.value)).fetchall()
        return sorted(r[0] for r in rows)

    def stats(self) -> Dict[str, int]:
        conn = self.connect()
        if conn is None:
            return {}
        nodes = dict(conn.execute("SELECT kind, COUNT(*) FROM node GROUP BY kind").fetchall())
        edges = dict(conn.execute("SELECT kind, COUNT(*) FROM edge GROUP BY kind").fetchall())
        return {"nodes": sum(nodes.values()), "edges": sum(edges.values()),
                **{f"node_{k}": v for k, v in nodes.items()},
                **{f"edge_{k}": v for k, v in edges.items()}}
