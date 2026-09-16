"""Dataset ingestion for the narrow initial market: tabular files.

Supported data per the spec's first release: CSV and Parquet (PostgreSQL /
object storage are the obvious next connectors and slot in behind the same
:class:`DatasetHandle`). Loading is deliberately defensive — Avaloka must be
able to *report* on a messy file, not crash on it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class UnusableDataset(ValueError):
    """The file was read, but what came back cannot be analysed.

    Distinct from a parse failure. The dangerous case is not the file pandas
    refuses — that surfaces loudly — it is the file pandas *accepts* and turns
    into a plausible-looking frame: a header with no rows, or binary content
    read as one wide column of mojibake. Both previously produced a complete
    mission, a validation verdict and an economic multiplier computed over
    nothing at all.
    """


@dataclass
class DatasetHandle:
    """A loaded dataset plus the provenance needed for lineage."""

    frame: pd.DataFrame
    source: str
    fmt: str
    sha256: str
    n_rows: int
    n_cols: int
    delimiter: str | None = None


def _name_preview(names: list[str], limit: int = 6, width: int = 24) -> str:
    """A short, bounded rendering of column names for an error message.

    A malformed file can parse into one column whose name is the entire first
    line, so this is truncated on both axes: an error that scrolls the terminal
    is not an error anyone reads.
    """
    shown = [n if len(n) <= width else n[: width - 1] + "\u2026" for n in names[:limit]]
    return ", ".join(shown) + ("\u2026" if len(names) > limit else "")


def _printable_ratio(text: str) -> float:
    """Share of characters that could plausibly appear in a column name."""
    if not text:
        return 0.0
    ok = sum(1 for ch in text if ch.isprintable() and (ch.isalnum() or ch in " _-.:/()[]#%&+"))
    return ok / len(text)


def _reject_unusable(frame: pd.DataFrame, path: Path, fmt: str) -> None:
    """Refuse a frame that parsed but holds no analysable data.

    Every message names the file and says what to do, because this fires on
    someone's first run more often than on their hundredth.
    """
    names = [str(c) for c in frame.columns]

    # "Is this even text?" comes first. A binary file usually parses to a
    # single wide mojibake header and zero rows, so checking rows first
    # reported "no rows" and sent the user looking at their export settings
    # instead of at the fact that they passed a .xlsx or a .gz.
    unreadable = [n for n in names if _printable_ratio(n) < 0.7]
    if names and len(unreadable) > len(names) / 2:
        raise UnusableDataset(
            f"{path.name} does not look like {fmt.upper()} text — its column "
            "names are unreadable, which usually means the file is binary, "
            "compressed, or in an encoding I could not detect. Re-export it as "
            "UTF-8 CSV, or pass a Parquet or Excel file."
        )

    if frame.shape[1] == 0:
        raise UnusableDataset(
            f"{path.name} has no usable columns. If this is a spreadsheet, check "
            "the header is not sitting below a title block."
        )

    if frame.shape[0] == 0:
        raise UnusableDataset(
            f"{path.name} has column headers but no rows ({_name_preview(names)}). "
            "There is nothing to analyse. Check the export actually wrote its rows."
        )


def _sha256(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        read = 0
        while read < limit:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def _read_excel(path):
    """Read a spreadsheet, finding the header row rather than assuming row 1.

    Hand-maintained sheets routinely open with a title and a subtitle before the
    real header -- the R1 programme plan this was written against has its header
    on row 4. Reading row 1 as the header yields one named column and a frame of
    mostly-NaN, which then gets profiled as if the data were that shape.

    Picks the sheet with the most data rows, then the first row where a majority
    of cells are non-empty strings and the values below are populated.
    """
    import pandas as pd

    book = pd.read_excel(path, sheet_name=None, header=None)
    if not book:
        raise ValueError("the workbook contains no sheets")
    name, raw = max(book.items(), key=lambda kv: kv[1].notna().sum().sum())

    header_row = 0
    for i in range(min(len(raw), 20)):
        row = raw.iloc[i]
        labelled = sum(isinstance(v, str) and v.strip() != "" for v in row)
        if labelled < 2 or labelled < (row.notna().sum() or 0) * 0.6:
            continue
        below = raw.iloc[i + 1: i + 4]
        if not below.empty and below.notna().sum().sum() >= labelled:
            header_row = i
            break

    frame = pd.read_excel(path, sheet_name=name, header=header_row)
    frame.attrs["sheet_name"] = name
    frame.attrs["header_row"] = header_row + 1        # 1-based, as a human counts
    return frame


def _sniff_delimiter(path: Path) -> str:
    """Pick the most likely delimiter from the header line (',', ';', '\\t', '|')."""
    with path.open("r", errors="replace") as fh:
        header = fh.readline()
    candidates = {d: header.count(d) for d in [",", ";", "\t", "|"]}
    best = max(candidates, key=candidates.get)
    return best if candidates[best] > 0 else ","


def _read_csv(path: Path, delimiter: str):
    """Read a CSV, tolerating non-UTF-8 exports (common in real-world data)."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=delimiter, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Last resort: replace undecodable bytes rather than crash the mission.
    return pd.read_csv(path, sep=delimiter, encoding="utf-8", encoding_errors="replace")


def load_dataset(source: str) -> DatasetHandle:
    """Load a tabular dataset from a local CSV or Parquet file."""
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Data source not found: {source!r}. Supported: local CSV, TSV, "
            "Parquet and Excel files; database and object-storage connectors are "
            "planned."
        )

    suffix = path.suffix.lower()
    delimiter: str | None = None
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        fmt = "parquet"
    elif suffix in {".csv", ".tsv", ".txt"}:
        delimiter = "\t" if suffix == ".tsv" else _sniff_delimiter(path)
        frame = _read_csv(path, delimiter)
        fmt = "csv"
    elif suffix in {".xlsx", ".xlsm", ".xls"}:
        # Spreadsheets are how plans, trackers and hand-maintained data actually
        # arrive. Falling through to the CSV attempt below produced
        # "Unsupported or unreadable data source", which reads like the file is
        # broken rather than like the CLI cannot open it.
        frame, fmt = _read_excel(path), "excel"
    else:
        # Last-ditch attempt: try CSV, then parquet.
        try:
            delimiter = _sniff_delimiter(path)
            frame = _read_csv(path, delimiter)
            fmt = "csv"
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Unsupported or unreadable data source {source!r}: {exc}. "
                "Supported formats: CSV, TSV, Parquet, Excel (.xlsx/.xlsm/.xls)."
            ) from exc

    _reject_unusable(frame, path, fmt)
    # Normalise obvious all-empty / unnamed columns from dirty exports.
    frame = frame.dropna(axis=1, how="all")

    return DatasetHandle(
        frame=frame,
        source=str(path),
        fmt=fmt,
        sha256=_sha256(path),
        n_rows=int(frame.shape[0]),
        n_cols=int(frame.shape[1]),
        delimiter=delimiter,
    )
