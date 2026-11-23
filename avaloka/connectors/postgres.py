"""Postgres through the connector contract, on the SQLAlchemy path the CLI already has.

``SQLAlchemyConnector`` is the generic implementation: any SQLAlchemy URL,
tested on sqlite. ``PostgresConnector`` is the proving connector: it adds
what Postgres exposes -- ``pg_class`` row estimates, ``pg_stats`` null and
distinct fractions, relation size and native partitions -- on top of
``avaloka.source_statistics.database_statistics`` (indexes, primary key,
size, partitions), and reads a table into Parquet through
``avaloka.io.sources._materialize_database``.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Optional, Sequence

from avaloka.connectors.base import (
    PLANNER_METADATA_MAX_BYTES, READ_MAX_BYTES, READ_MAX_ROWS, SAMPLE_FALLBACKS, SAMPLE_METHODS, Capability, Column,
    ConnectorBase, ConnectorError, Credentials, Endpoint, Predicate, Query, ReadReceipt, ReadResult,
    SourceVersion, Table, bounded_batches, cap_metadata, query_digest, requires,
)
from avaloka.connectors.sqlguard import verify_pushdown
from avaloka.source_statistics import database_statistics

#: Rows fetched per round trip on a bounded read.
_FETCH_ROWS = int(os.getenv("AVALOKA_CONNECTOR_FETCH_ROWS", "20000"))

_DEFAULT_PORTS = {"postgresql": 5432, "mysql": 3306, "mariadb": 3306, "mssql": 1433, "oracle": 1521,
                  "redshift": 5439, "snowflake": 443, "bigquery": 443, "databricks": 443}

#: Tables whose statistics go into planner metadata; the catalogue lists all.
PLANNER_METADATA_TABLES = int(os.getenv("AVALOKA_PLANNER_METADATA_TABLES", "20"))


def _sqlalchemy():
    try:
        import sqlalchemy
    except ImportError as exc:  # pragma: no cover
        raise ConnectorError(f"database connectors require SQLAlchemy: {exc}") from exc
    return sqlalchemy


def _quote(preparer: Any, name: str, *, schema: bool = False) -> str:
    """Always-quoted identifier: the dialect decides how, never whether."""
    from sqlalchemy.sql import quoted_name
    ident = quoted_name(name, True)
    return preparer.quote_schema(ident) if schema else preparer.quote(ident)


class SQLAlchemyConnector(ConnectorBase):
    """A relational source behind a SQLAlchemy URL.

    The URL is the credential: it is held as :class:`Credentials`, revealed
    only to ``create_engine``, and shown everywhere else with the password
    hidden (``describe()``).
    """

    name = "sqlalchemy"
    capabilities = frozenset({Capability.READ, Capability.PUSHDOWN, Capability.SCHEMA, Capability.SAMPLE})

    def __init__(self, url: str | Credentials | None = None, *, table: Optional[str] = None,
                 schema: Optional[str] = None, engine: Any = None) -> None:
        if engine is None and url is None:
            raise ConnectorError("a URL or an engine is required")
        if isinstance(url, Credentials):
            self._credentials = url
        elif url is not None:
            self._credentials = Credentials(str(url))
        else:
            self._credentials = Credentials(engine.url.render_as_string(hide_password=False))
        self._engine = engine
        self._owns_engine = engine is None
        self.table = table
        self.schema = schema
        self._catalogue: Optional[dict[str, Any]] = None

    # -- credentials -------------------------------------------------------
    @property
    def credentials(self) -> Credentials:
        return self._credentials

    def describe(self) -> str:
        """The URL with its password hidden: what logs and traces may carry."""
        sa = _sqlalchemy()
        return sa.engine.make_url(self._credentials.reveal()).render_as_string(hide_password=True)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.describe()!r}, table={self.table!r})"

    @property
    def engine(self):
        if self._engine is None:
            self._engine = _sqlalchemy().create_engine(self._credentials.reveal())
        return self._engine

    @property
    def dialect(self) -> str:
        return self.engine.dialect.name

    def close(self) -> None:
        if self._engine is not None and self._owns_engine:
            self._engine.dispose()
            self._engine = None

    def __enter__(self) -> "SQLAlchemyConnector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- 1. catalogue ------------------------------------------------------
    @requires(Capability.SCHEMA)
    def catalogue(self) -> dict[str, Any]:
        """Tables and typed columns, as a stable document."""
        if self._catalogue is None:
            sa = _sqlalchemy()
            inspector = sa.inspect(self.engine)
            tables = []
            for name in sorted(inspector.get_table_names(schema=self.schema)):
                columns = tuple(
                    Column(name=str(c["name"]), type=str(c.get("type")), nullable=bool(c.get("nullable", True)))
                    for c in inspector.get_columns(name, schema=self.schema)
                )
                tables.append(Table(name=name, columns=columns, schema=self.schema))
            self._catalogue = {"backend": self.dialect, "schema": self.schema,
                               "tables": [t.as_dict() for t in tables]}
            self._tables = {t.name: t for t in tables}
        return self._catalogue

    def _table(self, name: Optional[str]) -> Table:
        self.catalogue()
        target = name or self.table
        if not target:
            raise ConnectorError("no table selected")
        try:
            return self._tables[target]
        except KeyError:
            raise ConnectorError(f"table {target!r} not found (available: {sorted(self._tables)})") from None

    # -- 2. statistics -----------------------------------------------------
    def statistics(self, table: str) -> dict[str, Any]:
        """Row count, size, indexes, primary key, partitions; column stats where exposed."""
        sa = _sqlalchemy()
        target = self._table(table)
        inspector = sa.inspect(self.engine)
        with self.engine.connect() as connection:
            stats = database_statistics(inspector, connection, target.name, schema=target.schema)
            stats.update(self._row_count(connection, target))
            stats["columns"] = self._column_statistics(connection, target)
        stats["n_columns"] = len(target.columns)
        return stats

    def _qualified(self, table: Table) -> str:
        preparer = self.engine.dialect.identifier_preparer
        quoted = _quote(preparer, table.name)
        return f"{_quote(preparer, table.schema, schema=True)}.{quoted}" if table.schema else quoted

    def _row_count(self, connection: Any, table: Table) -> dict[str, Any]:
        sa = _sqlalchemy()
        count = connection.execute(sa.text(f"SELECT count(*) FROM {self._qualified(table)}")).scalar()
        return {"row_count": int(count or 0), "row_count_estimated": False}

    def _column_statistics(self, connection: Any, table: Table) -> dict[str, Any]:
        return {}

    # -- 3. planner metadata -----------------------------------------------
    def planner_metadata(self, max_bytes: int = PLANNER_METADATA_MAX_BYTES) -> dict[str, Any]:
        """The size-capped document a planner prompt receives and the trace records.

        Every table with its typed columns; statistics for the selected table
        first and then up to ``PLANNER_METADATA_TABLES`` others. No row value
        is ever read into it.
        """
        catalogue = self.catalogue()
        ordered = sorted(catalogue["tables"], key=lambda t: (t["name"] != self.table, t["name"]))
        tables = []
        for i, entry in enumerate(ordered):
            doc = dict(entry)
            if i < PLANNER_METADATA_TABLES:
                try:
                    doc["statistics"] = self.statistics(entry["name"])
                except Exception as exc:  # a table we cannot read is still listed
                    doc["statistics"] = {"statistics_partial": True, "statistics_note": type(exc).__name__}
            tables.append(doc)
        return cap_metadata({
            "kind": "database", "backend": self.dialect, "schema": self.schema,
            "selected_table": self.table, "tables": tables,
        }, max_bytes)

    # -- 4. pushdown -------------------------------------------------------
    @requires(Capability.PUSHDOWN)
    def pushdown(self, predicates: Iterable[Mapping[str, Any] | Predicate],
                 projection: Sequence[str], *, table: Optional[str] = None) -> Query:
        """``SELECT <projection> FROM <table> WHERE <predicates>`` from the optimizer's plan.

        Columns must exist in the catalogue, operators must be in the plan
        shape, identifiers are quoted by the dialect and values are bound by
        name. Nothing from the plan is ever interpolated into the SQL text.
        """
        target = self._table(table)
        known = set(target.column_names)
        preparer = self.engine.dialect.identifier_preparer

        columns = tuple(str(c) for c in projection) or target.column_names
        unknown = [c for c in columns if c not in known]
        if unknown:
            raise ConnectorError(f"projection names unknown column(s) {unknown} on {target.name!r}")

        clauses: list[str] = []
        parameters: dict[str, Any] = {}
        for i, item in enumerate(predicates):
            pred = item if isinstance(item, Predicate) else Predicate.from_plan(item)
            if pred.column not in known:
                raise ConnectorError(f"predicate names unknown column {pred.column!r} on {target.name!r}")
            op = pred.sql_operator
            ident = _quote(preparer, pred.column)
            if op in ("IS NULL", "IS NOT NULL"):
                clauses.append(f"{ident} {op}")
            elif op == "IN":
                values = list(pred.value) if isinstance(pred.value, (list, tuple, set)) else [pred.value]
                if not values:
                    clauses.append("1 = 0")  # IN () is empty, not an error
                    continue
                names = []
                for j, value in enumerate(values):
                    key = f"p{i}_{j}"
                    parameters[key] = value
                    names.append(f":{key}")
                clauses.append(f"{ident} IN ({', '.join(names)})")
            else:
                key = f"p{i}"
                parameters[key] = pred.value
                clauses.append(f"{ident} {op} :{key}")

        sql = f"SELECT {', '.join(_quote(preparer, c) for c in columns)} FROM {self._qualified(target)}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        query = Query(sql=sql, parameters=parameters, table=target.name, columns=columns)
        # Rule 3: the AST allowlist for this dialect, before the query can be sent.
        verify_pushdown(query, self.dialect)
        return query

    # -- 7. endpoints and source version ---------------------------------------
    def endpoints(self) -> list[Endpoint]:
        """The one host:port this connector opens; none for an in-process file.

        Read from the credential URL, which is what the connector was given to
        connect to, so a test can inject an answering engine without changing
        what the egress check sees."""
        url = _sqlalchemy().engine.make_url(self._credentials.reveal())
        if url.get_backend_name() == "sqlite" or not url.host:
            return []
        port = url.port or _DEFAULT_PORTS.get(url.get_backend_name())
        if port is None:
            raise ConnectorError(f"no port known for {url.get_backend_name()}; the URL must name one")
        return [Endpoint(host=str(url.host), port=int(port))]

    def locator_for(self, ref: Optional[str] = None) -> dict[str, Any]:
        """Backend, schema and table: the DSN stays in ``credentials``."""
        target = self._table(ref)
        return {"backend": self.dialect, "schema": target.schema, "table": target.name}

    def source_version(self, ref: Optional[str] = None) -> SourceVersion:
        """Row count plus the max of the first column (the key when there is
        one): the same for an unchanged table, different after a write. A
        source that exposes ``last_altered`` overrides this."""
        sa = _sqlalchemy()
        target = self._table(ref)
        preparer = self.engine.dialect.identifier_preparer
        first = _quote(preparer, target.column_names[0]) if target.columns else None
        with self.engine.connect() as connection:
            if first is not None:
                row = connection.execute(sa.text(
                    f"SELECT count(*), max({first}) FROM {self._qualified(target)}")).one()
                marker = {"row_count": int(row[0] or 0), "max_key": None if row[1] is None else str(row[1]),
                          "key": target.column_names[0]}
            else:
                row = connection.execute(sa.text(f"SELECT count(*) FROM {self._qualified(target)}")).one()
                marker = {"row_count": int(row[0] or 0)}
        return SourceVersion(ref=target.name, marker=marker)

    # -- 8. bounded reads --------------------------------------------------------
    def _run_bounded(self, query: Query, *, max_rows: int, max_bytes: int, method: Optional[str] = None,
                     ref: Optional[str] = None) -> ReadResult:
        import pyarrow as pa
        sa = _sqlalchemy()
        head = query.sql.lstrip().upper()
        if not (head.startswith("SELECT ") or head.startswith("WITH ")):
            raise ConnectorError("connectors read with SELECT only")

        def batches(result: Any, keys: list[str]):
            while True:
                rows = result.fetchmany(_FETCH_ROWS)
                if not rows:
                    return
                yield pa.RecordBatch.from_pylist([dict(zip(keys, r)) for r in rows])

        with self.engine.connect() as connection:
            result = connection.execution_options(stream_results=True).execute(
                sa.text(query.sql), query.parameters)
            keys = list(result.keys())
            kept, rows, used, stopped = bounded_batches(batches(result, keys), max_rows=max_rows, max_bytes=max_bytes)
            result.close()
        if kept:
            # Chunks inferred separately may disagree on a nullable type; unify on the first.
            schema = kept[0].schema
            kept = [b if b.schema.equals(schema) else b.cast(schema) for b in kept]
            table = pa.Table.from_batches(kept)
        else:
            table = pa.table({k: pa.array([], type=pa.null()) for k in keys}) if keys else pa.table({})
        # The receipt carries the table's version as read, so the catalog can
        # tell an unchanged re-read from a changed source (design §B.5).
        version = self.source_version(query.table) if query.table else None
        receipt = ReadReceipt(connector=self.name, ref=ref or query.table or (self.table or ""), rows=rows,
                              bytes=used, truncated=stopped is not None, stopped_by=stopped, method=method,
                              query_digest=query_digest(query.sql), source_version=version, columns=tuple(keys))
        return ReadResult(table, receipt)

    @requires(Capability.READ)
    def read(self, ref: Any = None, *, max_rows: int = READ_MAX_ROWS, max_bytes: int = READ_MAX_BYTES) -> ReadResult:
        """A bounded read of a table (``ref`` is its name, or the selected one)
        or of a pushed-down :class:`Query`. The receipt says when it stopped."""
        if isinstance(ref, Query):
            # A caller-supplied query is sent only once the allowlist passed it.
            verify_pushdown(ref, self.dialect)
            query = ref
        else:
            query = self.pushdown([], [], table=ref if isinstance(ref, str) else None)
        return self._run_bounded(query, max_rows=max_rows, max_bytes=max_bytes, ref=query.table)

    @requires(Capability.SAMPLE)
    def sample(self, ref: Optional[str] = None, *, n: int = 1000, method: str = "head") -> ReadResult:
        """``head`` is the first n rows; ``random`` orders by the engine's
        random(). The method used is in the receipt. ``keyed`` and
        ``tablesample`` are for connectors whose engine has them."""
        if method not in SAMPLE_METHODS:
            raise ConnectorError(f"sample method {method!r} is not one of {SAMPLE_METHODS}")
        if n <= 0:
            raise ConnectorError("a sample needs n > 0")
        base = self.pushdown([], [], table=ref)
        order = self._sample_order(method)
        if order is None:
            raise ConnectorError(f"{self.name} does not support the {method!r} sample method")
        sql = f"{base.sql}{order} LIMIT {int(n)}"
        query = Query(sql=sql, parameters=base.parameters, table=base.table, columns=base.columns)
        result = self._run_bounded(query, max_rows=n, max_bytes=READ_MAX_BYTES, method=method, ref=base.table)
        fallback = SAMPLE_FALLBACKS.get(method)
        if fallback is None or result.receipt.rows >= n or result.receipt.stopped_by is not None:
            return result
        # Block sampling keeps each page at a fixed rate, so it can come back
        # with fewer than n rows -- on a table of a few pages, often none. A
        # sample returns min(n, rows), so re-draw once by the fallback, which
        # does. A 10% draw is expected to come back short only on a table of
        # up to about ten times n rows, where the fallback's scan is small; on
        # a large one it is rare. The receipt names the method that ran, not
        # the one asked for.
        order = self._sample_order(fallback)
        sql = f"{base.sql}{order} LIMIT {int(n)}"
        query = Query(sql=sql, parameters=base.parameters, table=base.table, columns=base.columns)
        return self._run_bounded(query, max_rows=n, max_bytes=READ_MAX_BYTES, method=fallback, ref=base.table)

    def _sample_order(self, method: str) -> Optional[str]:
        if method == "head":
            return ""
        if method == "random":
            return " ORDER BY random()"
        return None

    def materialize(self, cache_dir: Any, *, table: Optional[str] = None):
        """The table as local Parquet with redacted provenance, via ``avaloka.io.sources``."""
        from pathlib import Path
        from avaloka.io.sources import _materialize_database
        target = self._table(table)
        return _materialize_database(f"{self._credentials.reveal()}#{target.name}", Path(cache_dir))


class PostgresConnector(SQLAlchemyConnector):
    """The proving SQL connector: what Postgres exposes beyond the generic path."""

    name = "postgresql"

    def __init__(self, url: str | Credentials | None = None, **kwargs: Any) -> None:
        super().__init__(url, **kwargs)
        if self.dialect != "postgresql":
            raise ConnectorError(f"PostgresConnector needs a postgresql URL, got {self.dialect!r}")

    def _regclass(self, table: Table) -> str:
        return f"{table.schema}.{table.name}" if table.schema else table.name

    def _row_count(self, connection: Any, table: Table) -> dict[str, Any]:
        """The planner's estimate from pg_class; exact only when the planner has none."""
        sa = _sqlalchemy()
        estimate = connection.execute(
            sa.text("SELECT reltuples FROM pg_class WHERE oid = to_regclass(:t)"),
            {"t": self._regclass(table)}).scalar()
        if estimate is not None and float(estimate) >= 0:
            return {"row_count": int(float(estimate)), "row_count_estimated": True}
        return super()._row_count(connection, table)

    def _sample_order(self, method: str) -> Optional[str]:
        if method == "tablesample":
            # Block sampling in the engine; LIMIT still bounds it. Not a uniform row
            # sample, which is why the method is in the receipt.
            return " TABLESAMPLE SYSTEM (10)"
        return super()._sample_order(method)

    def _column_statistics(self, connection: Any, table: Table) -> dict[str, Any]:
        """``pg_stats`` null fraction and distinct count per analysed column."""
        sa = _sqlalchemy()
        rows = connection.execute(
            sa.text("SELECT attname, null_frac, n_distinct FROM pg_stats "
                    "WHERE schemaname = :s AND tablename = :t"),
            {"s": table.schema or "public", "t": table.name}).fetchall()
        return {str(r[0]): {"null_fraction": float(r[1]), "n_distinct": float(r[2])} for r in rows}


def connector_for(url: str | Credentials, **kwargs: Any) -> SQLAlchemyConnector:
    """The connector for a SQLAlchemy URL: Postgres gets the proving one."""
    sa = _sqlalchemy()
    secret = url.reveal() if isinstance(url, Credentials) else str(url)
    backend = sa.engine.make_url(secret).get_backend_name()
    cls = PostgresConnector if backend == "postgresql" else SQLAlchemyConnector
    return cls(url, **kwargs)
