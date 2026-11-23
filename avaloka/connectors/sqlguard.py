"""The push-down allowlist (capability-gaps item 9a, contract rule 3).

A pushed-down query is built from the plan by the connector, so it *should*
be one parameterised SELECT. This module proves it before the query is sent:
it parses the text with sqlglot in the connector's dialect and refuses
anything that is not exactly one SELECT over one table, made only of the node
types a filter-and-projection can produce. A subquery, a function call, a
join, a second statement, DDL or DML, or a literal where a parameter belongs
is refused. The check is on the AST, not on the text, so quoting tricks do
not pass it.

Rule: parameters, not literals. A value from the plan reaches SQL only as a
bound parameter. A string or number literal in a WHERE clause means the
connector interpolated something, and that is refused too.
"""

from __future__ import annotations

from typing import Any, Optional

from avaloka.connectors.base import ConnectorError, Query

#: SQLAlchemy dialect name -> sqlglot dialect name.
DIALECTS = {
    "sqlite": "sqlite", "postgresql": "postgres", "mysql": "mysql", "mariadb": "mysql",
    "snowflake": "snowflake", "bigquery": "bigquery", "redshift": "redshift",
    "databricks": "databricks", "duckdb": "duckdb", "starrocks": "starrocks", "mssql": "tsql",
}

#: Node types a filter-and-projection SELECT is made of, and nothing else.
_ALLOWED = (
    "Select", "From", "Where", "Table", "Column", "Identifier", "Placeholder", "Parameter",
    "And", "Or", "Not", "Paren", "EQ", "NEQ", "LT", "LTE", "GT", "GTE", "In", "Is", "Null",
    "Boolean", "Neg", "Alias",
)


def _sqlglot():
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError as exc:
        raise ConnectorError("push-down verification requires sqlglot (pip install 'avaloka[sources]')") from exc
    return sqlglot, exp


def verify_pushdown(query: Query, dialect: Optional[str]) -> dict[str, Any]:
    """Refuse a query that is not one allowlisted SELECT; return what was checked.

    The return value goes into traces: the dialect, the node types seen and
    the table the query reads. Never the SQL text.
    """
    sqlglot, exp = _sqlglot()
    read = DIALECTS.get((dialect or "").lower())
    if read is None:
        raise ConnectorError(f"no push-down dialect for {dialect!r}")
    try:
        statements = sqlglot.parse(query.sql, read=read)
    except Exception as exc:  # noqa: BLE001  (sqlglot's ParseError carries the SQL text)
        raise ConnectorError(f"push-down SQL does not parse as {read}: {type(exc).__name__}") from None
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise ConnectorError(f"push-down must be exactly one statement, got {len(statements)}")
    root = statements[0]
    if not isinstance(root, exp.Select):
        raise ConnectorError(f"push-down must be a SELECT, got {type(root).__name__}")

    for key in ("distinct", "group", "having", "order", "limit", "offset", "joins", "with", "with_", "windows", "laterals"):
        if root.args.get(key):
            raise ConnectorError(f"push-down must not use {key.rstrip('_').upper()}")
    seen: set[str] = set()
    tables: list[str] = []
    for node in root.walk():
        kind = type(node).__name__
        seen.add(kind)
        if isinstance(node, exp.Literal):
            raise ConnectorError("push-down carries a literal; values are bound parameters")
        if kind not in _ALLOWED:
            raise ConnectorError(f"push-down contains {kind}, which is outside the plan shape")
        if isinstance(node, exp.Table):
            tables.append(node.name)
    if len(tables) != 1:
        raise ConnectorError(f"push-down must read exactly one table, got {len(tables)}")
    if not root.expressions:
        raise ConnectorError("push-down selects nothing")
    for projection in root.expressions:
        if not isinstance(projection, (exp.Column, exp.Alias)):
            raise ConnectorError(f"push-down projects {type(projection).__name__}, not a column")
    placeholders = {n.this if isinstance(n.this, str) else str(n.this)
                    for n in root.find_all(exp.Placeholder)}
    placeholders |= {n.name for n in root.find_all(exp.Parameter)}
    missing = placeholders - set(query.parameters)
    if missing:
        raise ConnectorError(f"push-down names parameters without values: {sorted(missing)}")
    return {"dialect": read, "table": tables[0], "nodes": sorted(seen), "parameters": len(placeholders)}


__all__ = ["DIALECTS", "verify_pushdown"]
