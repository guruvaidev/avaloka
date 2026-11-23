"""The app side of the connector contract: what happens after a read.

Design ``docs/design/1.7-DATA-PLATFORM.md`` §B.5 (15b, the catalog half). A
connector's ``read``, ``sample`` or push-down returns a :class:`ReadReceipt`;
this module turns one receipt into the two records the design asks for, in
one place, so every connector feeds the catalog and the graph the same way:

- **catalog**: ``CatalogStore.register`` with the receipt's columns, row count
  and *source version*, so a dataset gets a new version only when the source
  changed, and the analysis that read it pins that version (``app/catalog``);
- **lineage**: ``dataset LOADED_FROM connection`` with the source version's
  digest and the query digest in the edge attributes -- never the query text,
  never a credential (``app/core/lineage.py``).

Run truth (``rows_processed``, ``engine``) is the third record the design
names; it is not on this line yet (P1), so there is a seam for it here and
nothing more.

``avaloka`` must not import ``app``, which is why this lives under ``app/``:
the connector knows how to read; the app knows whose catalog it is.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

from app.catalog.store import CatalogError, CatalogStore, CatalogUnavailable, Registration, check_locator
from app.core.lineage import Edge, EdgeKind, LineageStore
from avaloka.connectors import Connector, ConnectorError, ReadReceipt, ReadResult

logger = logging.getLogger(__name__)

#: Connector ``name`` -> catalog dataset kind (``app.catalog.store.KINDS``).
#: A connector that is not listed cannot register: a new kind is a decision,
#: not a default.
KINDS_BY_CONNECTOR: Mapping[str, str] = {
    "file": "file",
    "objectstore": "object",
    "sqlalchemy": "db_table",
    "postgresql": "db_table",
    "mysql": "db_table",
    "sqlite": "db_table",
    "mongodb": "db_table",
    "redshift": "warehouse_table",
    "snowflake": "warehouse_table",
    "bigquery": "warehouse_table",
    "databricks": "warehouse_table",
    "starrocks": "lake_table",
    "kafka": "kafka_window",
}


def dataset_kind(connector: Any) -> str:
    name = str(getattr(connector, "name", "") or "")
    try:
        return KINDS_BY_CONNECTOR[name]
    except KeyError:
        raise CatalogError(f"connector {name!r} has no catalog dataset kind; add it to "
                           "app.connectors.registry.KINDS_BY_CONNECTOR") from None


def locator_for(connector: Any, ref: Optional[str], *, connection_id: Optional[str] = None) -> Dict[str, Any]:
    """Where the dataset lives, as the catalog stores it: names and ids, never
    a credential. A connector may say it precisely (``locator_for``); otherwise
    its ``describe()`` -- redacted by contract -- and the ref are used."""
    if hasattr(connector, "locator_for"):
        locator = dict(connector.locator_for(ref))
    else:
        locator = {"connector": str(connector.name), "ref": ref}
        if hasattr(connector, "describe"):
            locator["source"] = str(connector.describe())
    if connection_id:
        locator["connection_id"] = str(connection_id)
    locator = {k: v for k, v in locator.items() if v is not None}
    return check_locator(locator)


def _columns(result: ReadResult) -> list[Tuple[str, str]]:
    schema = getattr(result.table, "schema", None)
    if schema is not None:
        return [(str(f.name), str(f.type)) for f in schema]
    return [(str(c), "unknown") for c in result.receipt.columns]


def display_name_for(connector: Any, receipt: ReadReceipt, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if receipt.ref and receipt.ref not in ("", "/"):
        return str(receipt.ref).rsplit("/", 1)[-1]
    return str(getattr(connector, "table", None) or getattr(connector, "name", "dataset"))


def record_read(owner: str, connector: Connector, result: ReadResult, *, connection_id: Optional[str] = None,
                analysis_id: Optional[str] = None, run_id: Optional[str] = None, display_name: Optional[str] = None,
                catalog: Optional[CatalogStore] = None, lineage: Optional[LineageStore] = None,
                tags: Tuple[str, ...] = ()) -> Registration:
    """Register one read in the owner's catalog and graph. Raises
    :class:`CatalogUnavailable` when there is no catalog database and
    :class:`CatalogError` on a contract problem; see :func:`try_record_read`
    for the best-effort shape the agent paths use."""
    receipt = result.receipt
    kind = dataset_kind(connector)
    locator = locator_for(connector, receipt.ref, connection_id=connection_id)
    store = catalog or CatalogStore(owner)
    source_version = receipt.source_version.as_dict() if receipt.source_version is not None else None
    # A truncated read saw only part of the source: its row count is a floor.
    row_count_kind = "estimate" if receipt.truncated else "exact"
    reg = store.register(
        kind, locator, _columns(result), display_name=display_name_for(connector, receipt, display_name),
        source_version=source_version, row_count=int(receipt.rows), row_count_kind=row_count_kind,
        bytes=int(receipt.bytes), analysis_id=analysis_id, run_id=run_id, tags=tags)

    graph = lineage if lineage is not None else LineageStore(owner)
    try:
        if graph.available:
            attrs: Dict[str, Any] = {"kind": kind}
            if receipt.source_version is not None:
                attrs["source_version"] = receipt.source_version.digest
            if receipt.query_digest:
                attrs["query_digest"] = receipt.query_digest
            with graph.batch():
                graph.record_dataset(reg.dataset_id, label=display_name_for(connector, receipt, display_name),
                                     columns=_columns(result), connection_id=connection_id, **attrs)
                if connection_id:
                    graph.add_edge(Edge(reg.dataset_id, EdgeKind.LOADED_FROM, str(connection_id),
                                        {k: v for k, v in attrs.items() if k != "kind"}))
    finally:
        if lineage is None:
            graph.close()
    return reg


def try_record_read(owner: Optional[str], connector: Connector, result: ReadResult, **kwargs: Any) -> Optional[Registration]:
    """``record_read`` that never raises: the agent paths must not fail an
    analysis because the catalog is off. Logged once per kind of failure."""
    if not owner:
        return None
    try:
        return record_read(str(owner), connector, result, **kwargs)
    except CatalogUnavailable as exc:
        logger.debug("[registry] catalog unavailable, read not recorded: %s", exc)
    except (CatalogError, ConnectorError) as exc:
        logger.warning("[registry] read not recorded: %s", exc)
    except Exception:  # noqa: BLE001
        logger.warning("[registry] read not recorded", exc_info=True)
    return None


def read_and_record(owner: str, connector: Connector, ref: Any = None, *, max_rows: Optional[int] = None,
                    max_bytes: Optional[int] = None, **kwargs: Any) -> Tuple[ReadResult, Optional[Registration]]:
    """Read through the contract, then record. The read's bounds are the
    connector's defaults unless given."""
    read_kwargs: Dict[str, Any] = {}
    if max_rows is not None:
        read_kwargs["max_rows"] = max_rows
    if max_bytes is not None:
        read_kwargs["max_bytes"] = max_bytes
    result = connector.read(ref, **read_kwargs)
    return result, try_record_read(owner, connector, result, **kwargs)


__all__ = ["KINDS_BY_CONNECTOR", "dataset_kind", "display_name_for", "locator_for", "read_and_record",
           "record_read", "try_record_read"]
