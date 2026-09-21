"""Finding the actual table inside a spreadsheet.

A spreadsheet is a canvas, not a table. People put a title on row 0, leave row 1
blank, start the real header on row 2, and indent the whole thing one column to
the right because it looked nicer. ``pd.read_excel`` takes row 0 as the header
regardless, so every column comes back named ``Unnamed: 3`` and the frame is
one row of NaN followed by the data.

Nothing downstream can recover from that. The profiler types every column as
empty, the planner is handed a schema of ``Unnamed: 0 ... Unnamed: 37``, and the
insights it writes are real sentences about columns that do not exist:

    Shows the distribution of Unnamed: 13, highlighting spread and outliers.
    'Unnamed: 13' is a key driver in this dataset.

This module finds where the table actually starts, so the rest of the system is
reading data instead of reading the canvas.
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

#: How far down to look for a header. Past this, whatever the sheet is doing is
#: not a preamble and guessing would do more harm than leaving it alone.
MAX_HEADER_SCAN = 25

_UNNAMED_RE = re.compile(r"^Unnamed:\s*\d+$")

#: Longest a header cell plausibly is. Real headers are labels -- "Role",
#: "Hrs/Week per SME", "US T2 (San Antonio)" -- and top out well under this.
#: A cell carrying a sentence ("assumed. to be confirmed with Krishna") is a
#: value someone typed into a key/value sheet, and treating that row as a
#: header both loses the row and names every column after one record.
MAX_HEADER_CELL_CHARS = 45

#: A header cell is a name, not a sentence. "US T2 (San Antonio)" and
#: "Hrs/Week per SME" are names; "Not specified in the source correspondence."
#: and "assumed. to be confirmed with Krishna" are things someone typed into
#: the value column of a key/value sheet. Naming every column after one such
#: row is worse than admitting the sheet has no header.
MAX_HEADER_CELL_WORDS = 5


def _reads_like_prose(value: Any) -> bool:
    text = str(value).strip()
    if not text:
        return False
    if len(text) > MAX_HEADER_CELL_CHARS:
        return True
    if text.endswith(".") and " " in text:
        return True
    return len(text.split()) > MAX_HEADER_CELL_WORDS


def _is_blank(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def _filled(row) -> int:
    return sum(0 if _is_blank(v) else 1 for v in row)


def _looks_like_a_label(value: Any) -> bool:
    """Headers are labels: text, not measurements.

    A row of numbers is data even when it is the first row with content, so
    requiring text is what stops the first *data* row being eaten as a header.
    """
    if _is_blank(value):
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return False
    text = str(value).strip()
    if not text:
        return False
    # "1200", "12.5", "1,200", "$1,200" are values wearing a label's clothes.
    stripped = re.sub(r"[\s,$£€¥₹%()]", "", text)
    try:
        float(stripped)
        return False
    except ValueError:
        return True


def detect_header_row(raw: pd.DataFrame, max_scan: int = MAX_HEADER_SCAN) -> Optional[int]:
    """Index of the row that is the table's header, or None if there is none.

    Returning None matters as much as returning an index. Plenty of sheets are
    key/value blocks with no header at all, and inventing one from the first
    data row loses a row and names the columns after one arbitrary record.
    """
    if raw.empty:
        return None

    scan = min(len(raw), max_scan)
    counts = [_filled(raw.iloc[i]) for i in range(scan)]
    widest = max(counts) if counts else 0
    if widest < 1:
        return None

    # A one-cell row is a title in a wide sheet and the header in a
    # single-column one, so the floor has to follow the sheet's own width
    # rather than being a flat 2.
    min_filled = max(1, min(2, len(raw.columns)))

    for i in range(scan):
        if counts[i] < max(min_filled, widest * 0.6):
            # Blank rows and one-cell titles are preamble: keep looking.
            continue

        # The first row wide enough to be the table is either its header or its
        # first record. If it is a record, the table has no header and scanning
        # on would find one in the middle of the data -- which is how a
        # key/value sheet ends up named after whichever row looked tidiest.
        cells = [v for v in raw.iloc[i] if not _is_blank(v)]
        labels = [v for v in cells if _looks_like_a_label(v)]
        if len(labels) < len(cells) * 0.7:
            return None         # too many numbers: data starts here
        if any(_reads_like_prose(v) for v in cells):
            return None         # a sentence: data starts here
        names = [str(v).strip().lower() for v in labels]
        if len(set(names)) < len(names):
            return None         # repeated labels: a banded data row
        if i + 1 >= len(raw) or _filled(raw.iloc[i + 1]) == 0:
            return None         # nothing underneath it to be a header *of*
        return i
    return None


#: Symbols that make a number look like text. A column of "₹1,099" is a
#: quantity someone formatted, and leaving it as object means every downstream
#: statistic silently skips it.
_NUMERIC_NOISE = re.compile(r"[₹$€£¥₩₪₫฿¢%,\s\u00a0\u202f']")


def _as_numeric(series: pd.Series) -> Optional[pd.Series]:
    """The column as numbers, or None if it is not one.

    All-or-nothing: a column where nine values parse and one does not is a text
    column with a pattern, and coercing it would turn that tenth value into a
    silent NaN.
    """
    text = series.dropna().astype(str)
    if text.empty:
        return None
    # Accounting negatives: "(1,200)" is -1200.
    cleaned = text.str.strip().str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    cleaned = cleaned.str.replace(_NUMERIC_NOISE, "", regex=True)
    converted = pd.to_numeric(cleaned, errors="coerce")
    if converted.isna().any():
        return None
    out = pd.to_numeric(
        series.astype(str).str.strip()
              .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
              .str.replace(_NUMERIC_NOISE, "", regex=True),
        errors="coerce",
    )
    # Percentages are stored as written; the unit is recorded, not applied,
    # because "64%" meaning 64 or 0.64 depends on the column, not the cell.
    return out


def _as_datetime(series: pd.Series) -> Optional[pd.Series]:
    values = series.dropna()
    if values.empty:
        return None
    try:
        converted = pd.to_datetime(series, errors="coerce")
    except (ValueError, TypeError):
        return None
    if converted.notna().sum() < values.shape[0]:
        return None
    return converted


def reinfer_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Type the columns now the header is no longer one of the rows.

    Reading with ``header=None`` costs pandas' own inference: every column
    contains its header string, so a column of dates arrives as ``object`` and
    stays that way after the header is sliced off. Nothing downstream can treat
    it as a date, so it is re-inferred here -- which is also where currency and
    percent formatting stops hiding a numeric column.
    """
    out = df.copy()
    for column in out.columns:
        series = out[column]
        if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
            continue
        series = series.replace(r"^\s*$", pd.NA, regex=True)
        numeric = _as_numeric(series)
        if numeric is not None:
            out[column] = numeric
            continue
        stamped = _as_datetime(series)
        if stamped is not None:
            out[column] = stamped
            continue
        out[column] = series
    return out


def _dedupe(names: List[str]) -> List[str]:
    seen: dict = {}
    out: List[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def _clean_names(values, width: int) -> List[str]:
    names: List[str] = []
    for position, value in enumerate(values):
        text = "" if _is_blank(value) else re.sub(r"\s+", " ", str(value)).strip()
        # A column the header left blank still needs a name a person can say out
        # loud. "column_4" is not informative, but it does not pretend to be.
        names.append(text or f"column_{position + 1}")
    names += [f"column_{i + 1}" for i in range(len(names), width)]
    return _dedupe(names[:width])


def reshape(raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Return the table inside ``raw``, plus what had to be done to find it.

    ``raw`` must have been read with ``header=None`` so no row has already been
    consumed. The report is returned rather than logged so the caller can tell
    the user what happened -- silently reshaping someone's file is its own kind
    of wrong answer.
    """
    report = {"header_row": None, "dropped_leading_rows": 0,
              "dropped_empty_columns": 0, "dropped_empty_rows": 0,
              "synthesized_names": False}
    if raw.empty:
        return raw, report

    frame = raw.copy()

    # Empty columns first: a table indented one column to the right is the
    # single most common reason column 0 is entirely NaN.
    non_empty_cols = [c for c in frame.columns if _filled(frame[c]) > 0]
    report["dropped_empty_columns"] = len(frame.columns) - len(non_empty_cols)
    if non_empty_cols:
        frame = frame[non_empty_cols]

    header_row = detect_header_row(frame)
    report["header_row"] = header_row

    if header_row is None:
        frame = frame.dropna(how="all")
        frame.columns = _clean_names([None] * len(frame.columns), len(frame.columns))
        report["synthesized_names"] = True
        report["dropped_empty_rows"] = len(raw) - len(frame)
        return frame.reset_index(drop=True), report

    names = _clean_names(frame.iloc[header_row].tolist(), len(frame.columns))
    body = frame.iloc[header_row + 1:].copy()
    body.columns = names
    report["dropped_leading_rows"] = header_row

    before = len(body)
    body = body.dropna(how="all")
    report["dropped_empty_rows"] = before - len(body)

    return body.reset_index(drop=True), report


def needs_reshaping(df: pd.DataFrame) -> bool:
    """Whether an already-loaded frame shows the symptom.

    Used to decide whether to re-read a file rather than to reshape in place:
    once pandas has consumed the blank row as a header, the row is gone and only
    a re-read gets it back.
    """
    if df is None or df.empty:
        return False
    unnamed = sum(1 for c in df.columns if _UNNAMED_RE.match(str(c)))
    return unnamed >= max(1, len(df.columns) * 0.5)


# ---------------------------------------------------------------------------
# Blocks: one sheet is often several tables
# ---------------------------------------------------------------------------
# People put a second table to the right of the first, or stack three down the
# page with a blank row between. Read as one frame it becomes a table with a
# band of empty columns through the middle and a header that only describes the
# left third -- which is how "Management Efforts" ended up as a column name
# beside "Parameter | Value | Unit | Notes".


def _empty_column_indices(frame: pd.DataFrame) -> List[int]:
    return [i for i, c in enumerate(frame.columns) if _filled(frame[c]) == 0]


def _runs(values: List[int], total: int) -> List[Tuple[int, int]]:
    """Contiguous [start, end) spans of indices NOT in ``values``."""
    blocked = set(values)
    spans, start = [], None
    for i in range(total):
        if i in blocked:
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, total))
    return spans


def split_column_blocks(raw: pd.DataFrame, min_width: int = 1) -> List[pd.DataFrame]:
    """Split a sheet on fully-empty columns into side-by-side tables."""
    if raw.empty:
        return []
    spans = _runs(_empty_column_indices(raw), len(raw.columns))
    blocks = [raw.iloc[:, a:b] for a, b in spans if (b - a) >= min_width]
    return [b for b in blocks if _filled(b.values.ravel()) > 0]


def split_row_blocks(frame: pd.DataFrame, min_rows: int = 2) -> List[pd.DataFrame]:
    """Split on fully-empty rows into stacked tables.

    Requires a *pair* of blank rows to split on, because a single blank row is
    usually a spacer inside one table (between a section label and its rows)
    rather than a boundary between two.
    """
    if frame.empty:
        return [frame]
    blank = [i for i in range(len(frame)) if _filled(frame.iloc[i]) == 0]
    boundaries = {b for b in blank if (b + 1) in blank or (b - 1) in blank}
    if not boundaries:
        return [frame]
    spans = _runs(sorted(boundaries), len(frame))
    out = [frame.iloc[a:b] for a, b in spans]
    return [b for b in out if len(b) >= min_rows] or [frame]


def extract_tables(raw: pd.DataFrame, min_rows: int = 2) -> List[Tuple[pd.DataFrame, dict]]:
    """Every table in one sheet, each reshaped and reported on.

    Returns [(frame, report), ...] in reading order: left to right, then top to
    bottom. A sheet with one table returns one entry, which is the common case.
    """
    tables: List[Tuple[pd.DataFrame, dict]] = []
    for column_block in split_column_blocks(raw):
        for row_block in split_row_blocks(column_block):
            frame, report = reshape(row_block)
            if frame.empty or len(frame) < min_rows or not len(frame.columns):
                continue
            report["source_columns"] = [int(c) for c in row_block.columns]
            tables.append((frame, report))
    return tables
