"""A local file (an upload) on the connector contract.

The first row of the capability matrix (design §B.2): read, schema and
sample, no push-down (the local engine does the filtering), no sockets. CSV
and Parquet through pyarrow, so the Arrow batches are the file's own and a
bounded read of a large CSV never loads the whole file.

``source_version`` is the content hash for files up to ``HASH_LIMIT`` bytes,
else size and mtime: either way it is the same for an unchanged file and
differs when the file changed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from avaloka.connectors.base import (
    PLANNER_METADATA_MAX_BYTES, READ_MAX_BYTES, READ_MAX_ROWS, SAMPLE_METHODS, Capability, Column,
    ConnectorBase, ConnectorError, Credentials, Endpoint, ReadReceipt, ReadResult, SourceVersion, Table,
    bounded_batches, cap_metadata, requires,
)

HASH_LIMIT = 64 * 1024 * 1024
_BATCH_ROWS = 65_536
FORMATS = {".csv": "csv", ".tsv": "csv", ".txt": "csv", ".parquet": "parquet", ".pq": "parquet"}


def _pyarrow():
    try:
        import pyarrow
    except ImportError as exc:  # pragma: no cover
        raise ConnectorError(f"file connectors require pyarrow: {exc}") from exc
    return pyarrow


class FileConnector(ConnectorBase):
    """One local CSV or Parquet file, read-only, no network."""

    name = "file"
    capabilities = frozenset({Capability.READ, Capability.SCHEMA, Capability.SAMPLE})
    dialect = None

    def __init__(self, path: str | os.PathLike[str], *, credentials: Optional[Credentials] = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise ConnectorError(f"no such file: {self.path.name}")
        self.format = FORMATS.get(self.path.suffix.lower())
        if self.format is None:
            raise ConnectorError(f"unsupported file format {self.path.suffix!r}; one of {sorted(FORMATS)}")
        # A local file has no secret; the contract still wants a Credentials
        # object, so a placeholder that reveals nothing useful stands in.
        self._credentials = credentials or Credentials("file:local")
        self.table = self.path.stem
        self._schema: Any = None

    # -- contract plumbing --------------------------------------------------
    @property
    def credentials(self) -> Credentials:
        return self._credentials

    def describe(self) -> str:
        return f"file:{self.path.name}"

    def __repr__(self) -> str:
        return f"FileConnector({self.describe()})"

    def endpoints(self) -> list[Endpoint]:
        return []

    def locator_for(self, ref: Optional[str] = None) -> dict[str, Any]:
        self._table(ref)
        return {"filename": self.path.name, "format": self.format}

    def close(self) -> None:
        return None

    def __enter__(self) -> "FileConnector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- schema ---------------------------------------------------------------
    def _arrow_schema(self) -> Any:
        if self._schema is None:
            pa = _pyarrow()
            if self.format == "parquet":
                import pyarrow.parquet as pq
                self._schema = pq.read_schema(self.path)
            else:
                import pyarrow.csv as pcsv
                reader = pcsv.open_csv(self.path, read_options=pcsv.ReadOptions(block_size=1 << 20))
                self._schema = reader.schema
                del reader
            _ = pa
        return self._schema

    def _table(self, name: Optional[str]) -> Table:
        if name not in (None, "", self.table, self.path.name):
            raise ConnectorError(f"table {name!r} not found (available: [{self.table!r}])")
        schema = self._arrow_schema()
        return Table(name=self.table, columns=tuple(
            Column(name=f.name, type=str(f.type), nullable=bool(f.nullable)) for f in schema))

    @requires(Capability.SCHEMA)
    def catalogue(self) -> dict[str, Any]:
        return {"backend": "file", "format": self.format, "schema": None,
                "tables": [self._table(None).as_dict()]}

    def statistics(self, table: Optional[str] = None) -> dict[str, Any]:
        target = self._table(table)
        size = self.path.stat().st_size
        stats: dict[str, Any] = {"kind": "file", "backend": "file", "format": self.format,
                                 "size_bytes": size, "n_columns": len(target.columns)}
        if self.format == "parquet":
            import pyarrow.parquet as pq
            meta = pq.read_metadata(self.path)
            stats.update(row_count=int(meta.num_rows), row_count_estimated=False,
                         row_groups=int(meta.num_row_groups))
        else:
            stats.update(row_count=sum(b.num_rows for b in self._batches()), row_count_estimated=False)
        return stats

    def planner_metadata(self, max_bytes: int = PLANNER_METADATA_MAX_BYTES) -> dict[str, Any]:
        doc = self._table(None).as_dict()
        doc["statistics"] = self.statistics()
        return cap_metadata({"kind": "file", "backend": "file", "schema": None,
                             "selected_table": self.table, "tables": [doc]}, max_bytes)

    # -- version ---------------------------------------------------------------
    def source_version(self, ref: Optional[str] = None) -> SourceVersion:
        self._table(ref)
        st = self.path.stat()
        marker: dict[str, Any] = {"size": int(st.st_size)}
        if st.st_size <= HASH_LIMIT:
            h = hashlib.sha256()
            with self.path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            marker["sha256"] = h.hexdigest()
        else:
            marker["mtime_ns"] = int(st.st_mtime_ns)
        return SourceVersion(ref=self.table, marker=marker)

    # -- read ----------------------------------------------------------------
    def _batches(self) -> Iterable[Any]:
        if self.format == "parquet":
            import pyarrow.parquet as pq
            yield from pq.ParquetFile(self.path).iter_batches(batch_size=_BATCH_ROWS)
        else:
            import pyarrow.csv as pcsv
            reader = pcsv.open_csv(self.path, read_options=pcsv.ReadOptions(block_size=4 << 20))
            while True:
                try:
                    yield reader.read_next_batch()
                except StopIteration:
                    return

    def _result(self, kept: list[Any], rows: int, used: int, stopped: Optional[str], *,
                method: Optional[str] = None) -> ReadResult:
        pa = _pyarrow()
        schema = self._arrow_schema()
        table = pa.Table.from_batches(kept, schema=schema) if kept else schema.empty_table()
        receipt = ReadReceipt(connector=self.name, ref=self.table, rows=rows, bytes=used,
                              truncated=stopped is not None, stopped_by=stopped, method=method,
                              source_version=self.source_version(), columns=tuple(schema.names))
        return ReadResult(table, receipt)

    @requires(Capability.READ)
    def read(self, ref: Any = None, *, max_rows: int = READ_MAX_ROWS, max_bytes: int = READ_MAX_BYTES) -> ReadResult:
        self._table(ref if isinstance(ref, str) else None)
        kept, rows, used, stopped = bounded_batches(self._batches(), max_rows=max_rows, max_bytes=max_bytes)
        return self._result(kept, rows, used, stopped)

    @requires(Capability.SAMPLE)
    def sample(self, ref: Optional[str] = None, *, n: int = 1000, method: str = "head") -> ReadResult:
        if method not in SAMPLE_METHODS:
            raise ConnectorError(f"sample method {method!r} is not one of {SAMPLE_METHODS}")
        if n <= 0:
            raise ConnectorError("a sample needs n > 0")
        if method == "head":
            kept, rows, used, _stopped = bounded_batches(self._batches(), max_rows=n, max_bytes=READ_MAX_BYTES)
            return self._result(kept, rows, used, None, method="head")
        if method == "random":
            import numpy as np
            pa = _pyarrow()
            full = self.read(ref)
            if full.truncated:
                raise ConnectorError("random sample needs the whole file within the read bound; use head")
            total = full.table.num_rows
            if total <= n:
                return self._result(full.table.to_batches(), total, full.table.nbytes, None, method="random")
            idx = np.sort(np.random.default_rng(0).choice(total, size=n, replace=False))
            picked = full.table.take(pa.array(idx))
            return self._result(picked.to_batches(), picked.num_rows, picked.nbytes, None, method="random")
        raise ConnectorError(f"a file supports head and random samples, not {method!r}")


__all__ = ["FORMATS", "HASH_LIMIT", "FileConnector"]
