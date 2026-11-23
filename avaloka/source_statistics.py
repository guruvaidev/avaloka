"""Native physical statistics for files and database tables."""
from pathlib import Path

def file_statistics(path):
    source = Path(path)
    result = {"kind": "file", "size_bytes": source.stat().st_size,
              "format": source.suffix.lower().lstrip("."), "indexes": [],
              "partitioning": {"partitioned": False, "columns": [], "partitions": 1}}
    if source.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
            meta = pq.ParquetFile(source).metadata
            result.update(row_count=meta.num_rows, row_groups=meta.num_row_groups)
            result["partitioning"] = {"partitioned": meta.num_row_groups > 1,
                "columns": [], "partitions": meta.num_row_groups, "kind": "row_groups"}
        except Exception:
            result["statistics_partial"] = True
    return result

def database_statistics(inspector, connection, table, schema=None):
    qualified = f"{schema}.{table}" if schema else table
    indexes = [{"name": i.get("name"), "columns": list(i.get("column_names") or []),
                "unique": bool(i.get("unique", False))} for i in inspector.get_indexes(table, schema=schema)]
    pk = inspector.get_pk_constraint(table, schema=schema) or {}
    if pk.get("constrained_columns"):
        indexes.append({"name": pk.get("name") or "primary_key",
            "columns": list(pk["constrained_columns"]), "unique": True, "primary": True})
    stats = {"kind": "database_table", "backend": connection.dialect.name, "table": table,
        "indexes": indexes, "partitioning": {"partitioned": False, "columns": [], "partitions": 1}}
    try:
        if connection.dialect.name == "sqlite":
            pages = int(connection.exec_driver_sql("PRAGMA page_count").scalar() or 0)
            size = int(connection.exec_driver_sql("PRAGMA page_size").scalar() or 0)
            stats["database_size_bytes"] = pages * size
        elif connection.dialect.name == "postgresql":
            from sqlalchemy import text
            stats["size_bytes"] = int(connection.execute(text(
                "SELECT pg_total_relation_size(to_regclass(:table))"), {"table": qualified}).scalar() or 0)
            rows = connection.execute(text("SELECT a.attname FROM pg_partitioned_table p JOIN pg_class c ON c.oid=p.partrelid JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum=ANY(p.partattrs) WHERE c.oid=to_regclass(:table)"), {"table": qualified}).fetchall()
            count = int(connection.execute(text("SELECT count(*) FROM pg_inherits WHERE inhparent=to_regclass(:table)"), {"table": qualified}).scalar() or 0)
            if rows or count:
                stats["partitioning"] = {"partitioned": True,
                    "columns": [str(r[0]) for r in rows], "partitions": count, "kind": "native"}
    except Exception as exc:
        stats.update(statistics_partial=True, statistics_note=type(exc).__name__)
    return stats
