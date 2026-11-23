"""Cut a dataset into per-clone shard files without holding it whole.

The coordinator used to load the full dataset, split the frame, write a CSV
per shard and keep both the frame and the split copies alive while every
clone loaded its shard again, so the main process held the whole dataset
two or three times over for the whole run. Here it streams the source
instead. A CSV is read in chunks of strings and written straight on into
the shard files; a Parquet file is read batch by batch (row groups stream
through ``iter_batches``) and written with one ``ParquetWriter`` per shard,
so a clone's shard keeps the file's types and the Data Engineer can push its
scan into that shard's reader. Memory is one chunk, whatever the file's size. Formats pandas cannot stream (Excel, JSON,
XML, ORC, Avro, the complex-data directories) are loaded once, split and
released before any clone runs; that is the old behaviour minus the hold.

Shards are contiguous row ranges, sized as ``numpy.array_split`` would size
them from a row count read without parsing: Parquet's footer, or a CSV's
newlines counted block by block. A newline inside a quoted value makes the
CSV count high, so the last shard is open-ended and takes whatever remains,
and a shard that receives no rows is dropped (its directory too), which is
also how a fleet wider than the row count is clamped.

Each clone then loads its own shard through the loader, so typing and
null-like words are decided per shard; the coordinator reconciles the
partition statistics when two shards typed a column differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

#: Rows per streamed chunk. A chunk of strings this size is tens of MB.
CSV_CHUNK_ROWS = 100_000
PARQUET_BATCH_ROWS = 100_000
#: Bytes read at a time when counting a CSV's newlines.
_COUNT_BLOCK = 8 * 1024 * 1024

_CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
_PARQUET_SUFFIXES = {".parquet", ".pq"}


@dataclass
class ShardFile:
    """One clone's share of the dataset, on disk."""

    shard_id: int
    path: Path
    n_rows: int
    #: The source's columns as its header names them (the clone's loader
    #: may still drop a column that is empty throughout its shard).
    columns: list[str] = field(default_factory=list)


@dataclass
class ShardPlan:
    """What was written: the shards that received rows, the row total and the
    source's columns as its header names them."""

    shards: list[ShardFile]
    total_rows: int
    columns: list[str]
    method: str                 # csv_stream | parquet_stream | loaded
    #: The loader's manifest when the source was loaded whole; empty when it
    #: was streamed, because no frame of the whole source ever existed.
    structure: dict[str, Any] = field(default_factory=dict)
    #: The loader's read descriptor (format, sheet, ...) for a loaded source.
    read: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Counts only: this lands in coordination.json and the trace."""
        return {"method": self.method, "total_rows": self.total_rows,
                "rows_per_shard": [s.n_rows for s in self.shards]}


def _sizes(total: int, n: int) -> list[int]:
    """``numpy.array_split`` sizes: the first ``total % n`` shards get one more row."""
    n = max(1, n)
    base, extra = divmod(max(0, int(total)), n)
    return [base + 1] * extra + [base] * (n - extra)


class _Cursor:
    """Hands consecutive rows to consecutive shards; the last shard is open-ended."""

    def __init__(self, sizes: list[int]) -> None:
        self.sizes = sizes
        self.index = 0
        self.filled = 0

    def take(self, available: int) -> tuple[int, int]:
        """``(shard index, rows)`` for the next slice of ``available`` rows."""
        while self.index < len(self.sizes) - 1 and self.filled >= self.sizes[self.index]:
            self.index, self.filled = self.index + 1, 0
        if self.index >= len(self.sizes) - 1:
            return self.index, available
        rows = min(available, self.sizes[self.index] - self.filled)
        self.filled += rows
        return self.index, rows


def peek_columns(path: Path) -> Optional[list[str]]:
    """The source's column names from its header alone, or ``None`` when the
    format has no cheap header (the clone's own load reports a bad column)."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in _PARQUET_SUFFIXES:
            import pyarrow.parquet as pq

            return [str(name) for name in pq.read_schema(path).names]
        if suffix in _CSV_SUFFIXES and path.is_file():
            sep, encoding = _csv_dialect(path)
            head = pd.read_csv(path, sep=sep, encoding=encoding, nrows=0, dtype=str)
            return [str(c) for c in head.columns]
    except Exception:  # noqa: BLE001 - the header is a courtesy; the load decides
        return None
    return None


def write_shards(path: Path, directories: list[Path], *, file_stem: str = "shard",
                 question: Optional[str] = None) -> ShardPlan:
    """Write one shard file per directory in ``directories`` and return the plan.

    Only the directories that received rows appear in the plan; the others
    are removed again. The file is ``<directory>/<file_stem>.csv`` for a
    streamed CSV or a loaded source and ``<directory>/<file_stem>.parquet``
    for a streamed Parquet file. ``question`` reaches the loader for a source
    that is loaded rather than streamed, so a workbook is split off the sheet
    the question is about (loader._choose_sheet).
    """
    path = Path(path)
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if path.is_file() and suffix in _CSV_SUFFIXES:
        plan = _stream_csv(path, [Path(d) / f"{file_stem}.csv" for d in directories])
    elif path.is_file() and suffix in _PARQUET_SUFFIXES:
        plan = _stream_parquet(path, [Path(d) / f"{file_stem}.parquet" for d in directories])
    else:
        plan = _load_and_split(path, [Path(d) / f"{file_stem}.csv" for d in directories],
                               question=question)
    for shard in plan.shards:
        shard.columns = list(plan.columns)
    used = {shard.path.parent for shard in plan.shards}
    for directory in directories:
        directory = Path(directory)
        if directory not in used:
            try:
                directory.rmdir()
            except OSError:  # not empty: something else lives there; leave it
                pass
    return plan


# ---------------------------------------------------------------- CSV
def _csv_dialect(path: Path) -> tuple[str, str]:
    """The delimiter and an encoding that decodes the file's head, as the loader picks them."""
    from avaloka.io.loader import _sniff_delimiter

    sep = "\t" if path.suffix.lower() == ".tsv" else _sniff_delimiter(path)
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with path.open("r", encoding=encoding) as fh:
                for _ in range(200):
                    if not fh.readline():
                        break
        except (UnicodeDecodeError, UnicodeError):
            continue
        return sep, encoding
    return sep, "latin-1"


def _count_rows(path: Path) -> int:
    """Data rows by newline count, header excluded; high when a quoted value
    holds a newline. Reads bytes in blocks, so memory is one block."""
    lines, last = 0, b"\n"
    with path.open("rb") as fh:
        while block := fh.read(_COUNT_BLOCK):
            lines += block.count(b"\n")
            last = block[-1:]
    if last != b"\n":
        lines += 1  # a final row without a trailing newline
    return max(0, lines - 1)


def _stream_csv(path: Path, targets: list[Path]) -> ShardPlan:
    sizes = _sizes(_count_rows(path), len(targets))
    sep, encoding = _csv_dialect(path)
    for attempt in (encoding, "latin-1"):
        try:
            return _stream_csv_as(path, targets, sizes, sep=sep, encoding=attempt)
        except (UnicodeDecodeError, UnicodeError):
            for target in targets:
                target.unlink(missing_ok=True)
            if attempt == "latin-1":
                raise
    raise AssertionError("unreachable")  # pragma: no cover


def _stream_csv_as(path: Path, targets: list[Path], sizes: list[int], *, sep: str,
                   encoding: str) -> ShardPlan:
    # Every value is read as the string it is in the file and written back
    # unchanged, so the clone's own load (null-like words, typing, encoding)
    # sees the rows the way the source holds them.
    reader = pd.read_csv(path, sep=sep, encoding=encoding, dtype=str, keep_default_na=False,
                         na_filter=False, chunksize=CSV_CHUNK_ROWS)
    cursor = _Cursor(sizes)
    counts = [0] * len(targets)
    handles: list[Any] = [None] * len(targets)
    columns: list[str] = []
    try:
        for chunk in reader:
            if not columns:
                columns = [str(c) for c in chunk.columns]
            offset = 0
            while offset < len(chunk):
                index, rows = cursor.take(len(chunk) - offset)
                if handles[index] is None:
                    handles[index] = targets[index].open("w", encoding="utf-8", newline="")
                chunk.iloc[offset:offset + rows].to_csv(
                    handles[index], index=False, header=counts[index] == 0)
                counts[index] += rows
                offset += rows
    finally:
        for handle in handles:
            if handle is not None:
                handle.close()
    if not columns:
        columns = [str(c) for c in pd.read_csv(path, sep=sep, encoding=encoding, nrows=0, dtype=str).columns]
    shards = [ShardFile(i, targets[i], counts[i]) for i in range(len(targets)) if counts[i]]
    return ShardPlan(shards, sum(counts), columns, "csv_stream")


# ------------------------------------------------------------- Parquet
def _stream_parquet(path: Path, targets: list[Path]) -> ShardPlan:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    sizes = _sizes(source.metadata.num_rows, len(targets))
    cursor = _Cursor(sizes)
    counts = [0] * len(targets)
    writers: list[Any] = [None] * len(targets)
    try:
        for batch in source.iter_batches(batch_size=PARQUET_BATCH_ROWS):
            offset = 0
            while offset < batch.num_rows:
                index, rows = cursor.take(batch.num_rows - offset)
                if writers[index] is None:
                    writers[index] = pq.ParquetWriter(targets[index], source.schema_arrow)
                writers[index].write_table(pa.Table.from_batches([batch.slice(offset, rows)]))
                counts[index] += rows
                offset += rows
    finally:
        for writer in writers:
            if writer is not None:
                writer.close()
    columns = [str(name) for name in source.schema_arrow.names]
    shards = [ShardFile(i, targets[i], counts[i]) for i in range(len(targets)) if counts[i]]
    return ShardPlan(shards, sum(counts), columns, "parquet_stream")


# -------------------------------------------------- everything else
def _load_and_split(path: Path, targets: list[Path], *,
                    question: Optional[str] = None) -> ShardPlan:
    """Load once through the loader, split, write, and let the frame go."""
    from avaloka.io.loader import load_dataset

    handle = load_dataset(str(path), question=question)
    frame, structure, read = handle.frame, handle.structure, dict(handle.read or {})
    del handle
    columns = [str(c) for c in frame.columns]
    n = max(1, min(len(targets), len(frame)))
    counts = [0] * len(targets)
    for i, idx in enumerate(np.array_split(np.arange(len(frame)), n)):
        if len(idx):
            frame.iloc[idx].to_csv(targets[i], index=False)
            counts[i] = int(len(idx))
    total = int(len(frame))
    del frame
    shards = [ShardFile(i, targets[i], counts[i]) for i in range(len(targets)) if counts[i]]
    return ShardPlan(shards, total, columns, "loaded", structure=structure, read=read)
