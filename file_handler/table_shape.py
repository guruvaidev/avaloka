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
import numbers
import re
from collections import Counter
from typing import Any, List, Optional, Tuple

import pandas as pd

from .null_policy import keeps_null_words

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


def _looks_like_a_period(value: Any) -> bool:
    """A year standing where a label would: 2021, "2022", 2023.0.

    Period columns are how every financial statement is laid out, and a year
    parses as a number, so on its own it reads as a measurement.
    """
    if _is_blank(value) or isinstance(value, bool):
        return False
    if isinstance(value, numbers.Real):
        return float(value).is_integer() and 1900 <= value <= 2100
    return bool(re.fullmatch(r"(19|20)\d\d", str(value).strip()))


def _is_header_cell(value: Any) -> bool:
    if _looks_like_a_period(value):
        return True
    return _looks_like_a_label(value) and not _reads_like_prose(value)


def _period_headings(row: list, below: Optional[list]) -> list:
    """The cells of ``row`` that are years being used as column names.

    A year is only a heading when the row says so three ways: there is a real
    label beside it, there are at least two of them and no two the same, and
    nothing underneath any of them is a year. ``Acme | 1999 | 2004`` over
    ``Bolt | 2001 | 2010`` is two records; ``Region | 2021 | 2022`` over
    ``Aberdeen | 310000 | 331500`` is a header and a record.
    """
    if below is None:
        return []
    found = [(position, value) for position, value in enumerate(row)
             if _looks_like_a_period(value) and not _looks_like_a_label(value)]
    if len(found) < 2 or not any(_looks_like_a_label(v) for v in row):
        return []
    said = [_name_text(value) for _position, value in found]
    if len(set(said)) < len(said):
        return []
    if any(_looks_like_a_period(below[position]) for position, _value in found):
        return []
    return [value for _position, value in found]


def _differs_in_kind(heading: Any, cell: Any) -> bool:
    """Whether ``cell`` is a measurement sitting under the name ``heading``."""
    if _is_blank(cell) or _looks_like_a_label(cell):
        return False
    return _looks_like_a_label(heading) or not _looks_like_a_period(cell)


def _two_row_header_names(top: list, sub: list, record: list) -> Optional[List[str]]:
    """Column names if ``top`` and ``sub`` are one header written on two rows.

    Two shapes a person makes without thinking of them as unusual:

        Store | Transactions |       | Refunds |           a merged cell: the
              | Count        | Value | Count   | Value     group label is only
                                                           in its first column
        Region | Sales | Sales | Cost | Cost               the group label is
               | 2022  | 2023  | 2022 | 2023               repeated instead

    Neither row is the header on its own, so each column is named from both:
    the group label carried across, then the label under it.

    A wrong yes here costs a record and names the columns after it, so every
    test below has to hold. The top row must be incomplete by itself (a gap or
    a repeat); the row under it must fill exactly what is missing with labels;
    and the row under that must be a record -- it has to hold a number, which
    is what stops two rows of an all-text sheet being read as a header.
    """
    width = len(top)
    top_cells = [v for v in top if not _is_blank(v)]
    sub_cells = [v for v in sub if not _is_blank(v)]
    if not top_cells or not sub_cells:
        return None
    if not all(_looks_like_a_label(v) and not _reads_like_prose(v) for v in top_cells):
        return None
    if not all(_is_header_cell(v) for v in sub_cells):
        return None

    # Which top cell each column belongs to: a blank continues the label on
    # its left, which is what a merged cell looks like once it is unmerged.
    owner: List[Optional[int]] = []
    current: Optional[int] = None
    for position, value in enumerate(top):
        if not _is_blank(value):
            current = position
        owner.append(current)
    spoken = [None if _is_blank(v) else _name_text(v).lower() for v in top]
    times = Counter(name for name in spoken if name)
    merged = {owner[p] for p in range(width) if _is_blank(top[p]) and owner[p] is not None}
    if not merged and not any(n > 1 for n in times.values()):
        return None             # the top row is complete: an ordinary header

    names: List[str] = []
    stands_alone = False
    for position in range(width):
        group = owner[position]
        grouped = group is None or group in merged or times[spoken[group]] > 1
        if _is_blank(sub[position]):
            if grouped:
                return None     # a group with nothing under one of its columns
            stands_alone = stands_alone or not _is_blank(record[position])
        parts = [top[group] if group is not None else None, sub[position]]
        names.append(" ".join(_name_text(v) for v in parts if not _is_blank(v)))
    if len({name.lower() for name in names}) < width:
        return None

    if not any(not _is_blank(v) and not _looks_like_a_label(v) for v in record):
        return None             # no number under it: these are rows of text
    labelled = [p for p in range(width) if not _is_blank(sub[p])]
    differing = [p for p in labelled if _differs_in_kind(sub[p], record[p])]
    if not stands_alone and len(differing) * 2 < len(labelled):
        return None             # the second row looks like the third: a record
    return names


def _repeats_itself(row: list) -> bool:
    said = [_name_text(v).lower() for v in row if not _is_blank(v)]
    return len(set(said)) < len(said)


def detect_header_rows(raw: pd.DataFrame, max_scan: int = MAX_HEADER_SCAN) -> Optional[Tuple[int, int]]:
    """``(first, last)`` rows of the table's header, or None if there is none.

    A header is not always one row, and an index cannot say "rows 0 and 1".
    ``(2, 2)`` is an ordinary header on row 2; ``(0, 1)`` is a group row over
    a label row, and the data starts on the row after ``last``.

    Returning None matters as much as returning a span. Plenty of sheets are
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

        row = raw.iloc[i].tolist()
        below = raw.iloc[i + 1].tolist() if i + 1 < len(raw) else None
        if below is not None and i + 2 < len(raw):
            # Asked first: a group row is rejected by the checks below for the
            # very thing that makes it one, its repeated labels.
            if _two_row_header_names(row, below, raw.iloc[i + 2].tolist()) is not None:
                return i, i + 1
        if below is not None and i > 0 and counts[i - 1] and _repeats_itself(row):
            # A merged group row three columns wide per label is mostly blank,
            # so it was passed over as preamble just now. It is only looked
            # back at when this row cannot be a header without it: a title over
            # an ordinary header must stay a title.
            if _two_row_header_names(raw.iloc[i - 1].tolist(), row, below) is not None:
                return i - 1, i

        # The first row wide enough to be the table is either its header or its
        # first record. If it is a record, the table has no header and scanning
        # on would find one in the middle of the data -- which is how a
        # key/value sheet ends up named after whichever row looked tidiest.
        cells = [v for v in row if not _is_blank(v)]
        labels = [v for v in cells if _looks_like_a_label(v)]
        periods = _period_headings(row, below)
        if len(labels) + len(periods) < len(cells) * 0.7:
            return None         # too many numbers: data starts here
        if any(_reads_like_prose(v) for v in cells):
            return None         # a sentence: data starts here
        names = [_name_text(v).lower() for v in labels + periods]
        if len(set(names)) < len(names):
            return None         # repeated labels: a banded data row
        if below is None or _filled(below) == 0:
            return None         # nothing underneath it to be a header *of*
        return i, i
    return None


def detect_header_row(raw: pd.DataFrame, max_scan: int = MAX_HEADER_SCAN) -> Optional[int]:
    """Index of the row the table's header starts on, or None if there is none.

    For a header written on two rows this is the first of them; a caller that
    needs to know where the data starts wants detect_header_rows.
    """
    span = detect_header_rows(raw, max_scan)
    return None if span is None else span[0]


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
        if not _is_untyped(series):
            continue
        series = series.replace(r"^\s*$", pd.NA, regex=True)
        typed = _typed(series)
        out[column] = series if typed is None else typed
    return out


def _is_untyped(series: pd.Series) -> bool:
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def _typed(series: pd.Series) -> Optional[pd.Series]:
    """The column as numbers or dates, or None if it is neither."""
    numeric = _as_numeric(series)
    if numeric is not None:
        return numeric
    return _as_datetime(series)


def keep_null_word_labels(frame: pd.DataFrame, words: pd.DataFrame) -> pd.DataFrame:
    """Put null-like words back into the text columns of ``frame``.

    ``frame`` is a table found with those words blanked (so header and table
    detection behave as they always have); ``words`` has the same shape and
    holds the word each blanked cell contained (file_handler.null_policy).
    Each column is typed exactly as reinfer_dtypes would type it. A column that
    types keeps its gaps; a text column gets its labels back. The decision
    itself is null_policy.keeps_null_words -- this only asks it.
    """
    if words.empty or not words.notna().to_numpy().any():
        return frame
    out = frame.copy()
    for position in range(len(out.columns)):
        held = words.iloc[:, position]
        if not held.notna().any():
            continue
        series = out.iloc[:, position]
        blanked = series.replace(r"^\s*$", pd.NA, regex=True) if _is_untyped(series) else series
        reader_typed = (not _is_untyped(series)) or _typed(blanked) is not None
        if not keeps_null_words(blanked.dropna().tolist(), reader_typed):
            continue
        restored = series.astype(object).where(held.isna().to_numpy(), held.to_numpy())
        out.isetitem(position, restored)
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


def _name_text(value: Any) -> str:
    """A header cell as a person would say it: 2021, not 2021.0."""
    if _is_blank(value):
        return ""
    if isinstance(value, numbers.Real) and not isinstance(value, bool) and float(value).is_integer():
        value = int(value)
    return re.sub(r"\s+", " ", str(value)).strip()


def _clean_names(values, width: int) -> List[str]:
    names: List[str] = []
    for position, value in enumerate(values):
        text = _name_text(value)
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
    frame, report, _rows, _columns = reshape_with_positions(raw)
    return frame, report


def reshape_with_positions(raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict, list, list]:
    """reshape(), plus the labels of the ``raw`` rows and columns the table's
    cells came from, in the table's own order. A caller holding a second frame
    aligned with ``raw`` can use them to pick out the matching cells."""
    report = {"header_row": None, "header_rows": [], "dropped_leading_rows": 0,
              "dropped_empty_columns": 0, "dropped_empty_rows": 0,
              "leading_text": [], "synthesized_names": False, "placeholder_names": [],
              "numeric_header_cells": [], "rows_below_table": 0,
              "next_table_row": None, "trailing_note_rows": 0,
              "trailing_notes": [], "footnote_markers": [],
              "aggregate_rows": [], "labelled_total_rows_kept": []}
    if raw.empty:
        return raw, report, list(raw.index), list(raw.columns)

    frame = raw.copy()

    # Empty columns first: a table indented one column to the right is the
    # single most common reason column 0 is entirely NaN.
    non_empty_cols = [c for c in frame.columns if _filled(frame[c]) > 0]
    report["dropped_empty_columns"] = len(frame.columns) - len(non_empty_cols)
    if non_empty_cols:
        frame = frame[non_empty_cols]

    span = detect_header_rows(frame)

    if span is None:
        frame = frame.dropna(how="all")
        source_columns = list(frame.columns)
        frame.columns = _clean_names([None] * len(frame.columns), len(frame.columns))
        report["synthesized_names"] = True
        report["placeholder_names"] = list(frame.columns)
        report["dropped_empty_rows"] = len(raw) - len(frame)
        return frame.reset_index(drop=True), report, list(frame.index), source_columns

    first, last = span
    report["header_row"] = first
    report["header_rows"] = list(range(first, last + 1))
    top = frame.iloc[first].tolist()
    if last > first:
        spoken = _two_row_header_names(top, frame.iloc[last].tolist(), frame.iloc[last + 1].tolist())
    else:
        spoken = top
        report["numeric_header_cells"] = [
            _name_text(v) for v in top if not _is_blank(v) and not _looks_like_a_label(v)]
    names = _clean_names(spoken, len(frame.columns))
    report["placeholder_names"] = [
        name for name, said in zip(names, spoken) if not _name_text(said)]
    body = frame.iloc[last + 1:].copy()
    source_columns = list(body.columns)
    report["dropped_leading_rows"] = first
    report["leading_text"] = [
        _name_text(v) for i in range(first) for v in frame.iloc[i] if not _is_blank(v)]

    body = _stop_at_the_next_table(body, report)
    body, notes = _without_trailing_notes(body, report)
    body.columns = names
    if notes:
        body = _without_footnote_markers(body, notes, report)
    body, _kept = without_aggregate_rows(
        body, report, sheet_rows=[_sheet_row(label) for label in body.index])

    before = len(body)
    body = body.dropna(how="all")
    report["dropped_empty_rows"] = before - len(body)

    return body.reset_index(drop=True), report, list(body.index), source_columns


def _blank_rows(frame: pd.DataFrame) -> List[bool]:
    return [_filled(frame.iloc[i]) == 0 for i in range(len(frame))]


def _stop_at_the_next_table(body: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Cut ``body`` where a second table starts underneath the first.

    The boundary is the one split_row_blocks uses -- a pair of blank rows --
    and only when what follows has a header of its own. A gap followed by more
    records is one table with a hole in it and is left whole.

    Cutting leaves rows out, so "has a header" is held to more than looking
    like one: in a sheet of text every record looks like a header. A column
    has to change kind across the gap -- a label where the table above has
    only ever had numbers, or numbers under it where there was only text.
    """
    blank = _blank_rows(body)
    for i in range(len(body) - 1):
        if not (blank[i] and blank[i + 1]):
            continue
        rest = body.iloc[i:]
        span = detect_header_rows(rest)
        if span is None or not _changes_kind(body.iloc[:i], rest, span):
            return body
        left_out = [label for label, empty in zip(rest.index, blank[i:]) if not empty]
        report["rows_below_table"] = len(left_out)
        report["next_table_row"] = _sheet_row(left_out[0]) if left_out else None
        return body.iloc[:i]
    return body


def _changes_kind(above: pd.DataFrame, rest: pd.DataFrame, span: Tuple[int, int]) -> bool:
    header, records = rest.iloc[span[0]], rest.iloc[span[1] + 1:]
    for position in range(len(above.columns)):
        was = [_looks_like_a_label(v) for v in above.iloc[:, position] if not _is_blank(v)]
        if not was:
            continue
        if not any(was) and _looks_like_a_label(header.iloc[position]):
            return True         # a name over what has been a column of numbers
        if all(was) and any(not _is_blank(v) and not _looks_like_a_label(v)
                            for v in records.iloc[:, position]):
            return True         # numbers under what has been a column of text
    return False


def _sheet_row(label: Any) -> Optional[int]:
    """The row number a person sees in the spreadsheet, or None if unknown."""
    return int(label) + 1 if isinstance(label, numbers.Integral) else None


def _without_trailing_notes(body: pd.DataFrame, report: dict) -> Tuple[pd.DataFrame, List[str]]:
    """Drop a block of notes sitting under the table, and return its lines.

    "* excludes contractors", "Source: ONS" -- typed into column A a blank row
    below the last record. Read as data they are extra rows, each one a
    sentence in the first text column and a gap everywhere else. They are only
    notes when all of it holds: a blank row above them, one cell each, that
    cell in the first column, in a table more than one column wide whose last
    row above the gap is a full record -- and every one of them reads like a
    note. A checklist whose last items have nothing in the second column is
    the same picture, and those are records.
    """
    if len(body.columns) < 2 or len(body) < 3:
        return body, []
    blank = _blank_rows(body)
    cut = len(body)
    while cut > 0 and (blank[cut - 1] or _is_a_note_row(body.iloc[cut - 1])):
        cut -= 1
    # ``cut`` is now just past the last row that is neither blank nor a note.
    notes = [str(body.iloc[i, 0]).strip() for i in range(cut, len(body)) if not blank[i]]
    if not notes or cut == 0 or cut >= len(body) or not blank[cut]:
        return body, []
    if not all(_reads_like_a_note(note) for note in notes):
        return body, []
    if _filled(body.iloc[cut - 1]) < 2:
        return body, []
    report["trailing_note_rows"] = len(notes)
    report["trailing_notes"] = notes
    return body.iloc[:cut], notes


def _is_a_note_row(row) -> bool:
    return _filled(row) == 1 and not _is_blank(row.iloc[0])


#: What a footnote reference looks like at the start of the note defining it:
#: "* excludes ...", "(a) as at ...", "[2] restated". Digits are only taken in
#: brackets -- a bare "1" at the end of a label is far more often part of it --
#: and "#" not at all, because "Order #" is a column.
_FOOTNOTE_MARKER = re.compile(r"^(\*+|[†‡§¶]+|\(\w{1,2}\)|\[\w{1,2}\])\s*\S")

#: "Source: ONS", "Note: provisional".
_NOTE_LEAD = re.compile(r"^[A-Za-z][A-Za-z .]{1,20}:\s*\S")


def _reads_like_a_note(text: str) -> bool:
    return bool(_FOOTNOTE_MARKER.match(text) or _NOTE_LEAD.match(text)) or _reads_like_prose(text)


def _without_footnote_markers(body: pd.DataFrame, notes: List[str], report: dict) -> pd.DataFrame:
    """Take footnote references off the column names and the text cells.

    "Headcount (a)" is the column "Headcount", and "Bath*" is the branch
    "Bath" -- but not to anything matching on names. Only a marker that a note
    under the table defines is removed, so a label that simply ends in "*"
    or "(a)" with no footnote to explain it is left exactly as typed.
    """
    markers = []
    for note in notes:
        found = _FOOTNOTE_MARKER.match(note)
        if found and found.group(1) not in markers:
            markers.append(found.group(1))
    if not markers:
        return body
    # Longest first, so "**" is not half-removed as "*".
    ending = re.compile(r"\s*(?:%s)$" % "|".join(
        re.escape(m) for m in sorted(markers, key=len, reverse=True)))

    def clean(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return ending.sub("", value.rstrip()) or value

    out = body.copy()
    for position in range(len(out.columns)):
        series = out.iloc[:, position]
        if _is_untyped(series):
            out.isetitem(position, series.map(clean))
    out.columns = _dedupe([clean(name) for name in out.columns])
    report["footnote_markers"] = markers
    return out


# ---------------------------------------------------------------------------
# Totals: rows that are sums of the other rows
# ---------------------------------------------------------------------------
# A "Total" row under the data is the one shape that leaves no trace. The
# header is right, the names are right, the types are right -- and every sum
# is double, the maximum is the total and the mean is inflated. Nothing above
# this point notices, because nothing had to be repaired.

#: A cell that announces an aggregate: Total, TOTAL:, Grand Total, Sub-total,
#: North subtotal, Sum, Gesamt, Summe, Totaal, Totale, Somme, Suma, Итого, 合计.
#: Only ever a hint. A row is left out because its numbers add up, never
#: because of what it is called -- "Total Quality Management" is a department.
_AGGREGATE_WORD = re.compile(
    r"\b(?:grand[\s-]*|sub[\s-]*)?(?:totals?|totaux|totale|totaal|gesamt\w*|"
    r"summe|sum|somme|suma|aggregate)\b|итого|всего|合计|合計|总计|總計|小计|小計",
    re.IGNORECASE)


def _numbers(series: pd.Series) -> pd.Series:
    """The cells of a column as floats, NaN where a cell is not a number."""
    if pd.api.types.is_bool_dtype(series):
        return pd.Series(float("nan"), index=series.index)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip().str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    out = pd.to_numeric(text.str.replace(_NUMERIC_NOISE, "", regex=True), errors="coerce")
    return out.where(series.notna(), float("nan")).astype(float)


def _adds_up(stated: float, summed: float) -> bool:
    """Whether ``stated`` is ``summed``, allowing for a total typed to the penny.

    A sum of nothing but zeros matches any zero, which is no evidence at all.
    """
    if pd.isna(stated) or pd.isna(summed) or (stated == 0 and summed == 0):
        return False
    return abs(stated - summed) <= max(0.011, 1e-9 * max(abs(stated), abs(summed)))


def _aggregate_labels(frame: pd.DataFrame, numbers: pd.DataFrame) -> dict:
    """``{position: label}`` for the rows a text cell calls a total."""
    labelled: dict = {}
    for column in range(len(frame.columns)):
        series = frame.iloc[:, column]
        if not _is_untyped(series):
            continue
        said = (series.astype(str).str.contains(_AGGREGATE_WORD, na=False)
                & series.notna() & numbers.iloc[:, column].isna()).to_numpy()
        for position in said.nonzero()[0].tolist():
            labelled.setdefault(position, str(series.iloc[position]).strip())
    return labelled


def find_aggregate_rows(frame: pd.DataFrame) -> Tuple[List[Tuple[int, str, str]], List[Tuple[int, str]]]:
    """Rows of ``frame`` that are sums of its other rows.

    Returns ``(found, kept)``. ``found`` is ``[(position, label, kind)]`` with
    kind ``"total"`` or ``"subtotal"``; ``kept`` is ``[(position, label)]`` for
    rows that are *called* a total but could not be shown to be one, and so are
    still data.

    The arithmetic decides. Two shapes are recognised:

    * the last row equals the column sums of every row above it. With a label
      saying so, one agreeing column is enough -- a totals row often carries
      an average or a blank beside its sums. With no label, every number in
      the row must agree and there must be at least two of them.
    * the last row is a grand total over groups: labelled rows above it each
      equal the sum of the rows since the previous one, and the last row
      equals the sum of every row that is not one of those. The grand total
      checks the subtotals, which is what makes it safe to take rows out of
      the middle of a table: one row wrongly read as a subtotal and the
      grand total no longer adds up, so nothing is removed at all.

    Not recognised, deliberately: a subtotal with no grand total to check it
    against. In a financial statement ``Total operating expenses`` is a line
    item people ask for, and nothing in the sheet says otherwise.
    """
    if len(frame) < 3 or len(frame.columns) < 1:
        return [], []
    filled = frame.notna().any(axis=1).to_numpy().nonzero()[0].tolist()
    if len(filled) < 3:
        return [], []
    numbers = pd.concat([_numbers(frame.iloc[:, c]) for c in range(len(frame.columns))],
                        axis=1, ignore_index=True)
    last, above = filled[-1], filled[:-1]
    stated = numbers.iloc[last]
    measured = [c for c in range(len(frame.columns)) if not pd.isna(stated.iloc[c])]
    labelled = _aggregate_labels(frame, numbers)
    called = sorted(labelled.items())
    if not measured:
        return [], called

    sums = numbers.iloc[above].sum(skipna=True)
    agreeing = [c for c in measured if _adds_up(stated.iloc[c], sums.iloc[c])]
    if last in labelled and agreeing:
        return ([(last, labelled[last], "total")],
                [(i, label) for i, label in called if i != last])
    if last not in labelled:
        if len(measured) >= 2 and len(agreeing) == len(measured) and len(above) >= 3:
            return [(last, "", "total")], called
        return [], called

    # A grand total over groups, each closed by its own subtotal.
    found: List[Tuple[int, str, str]] = []
    group: List[int] = []
    records: List[int] = []
    for i in above:
        if i in labelled and group:
            row = numbers.iloc[i]
            group_sums = numbers.iloc[group].sum(skipna=True)
            if any(_adds_up(row.iloc[c], group_sums.iloc[c]) for c in range(len(row))):
                found.append((i, labelled[i], "subtotal"))
                records += group
                group = []
                continue
        group.append(i)
    if not found:
        return [], called
    record_sums = numbers.iloc[records + group].sum(skipna=True)
    if not any(_adds_up(stated.iloc[c], record_sums.iloc[c]) for c in measured):
        return [], called
    found.append((last, labelled[last], "total"))
    taken = {i for i, _label, _kind in found}
    return found, [(i, label) for i, label in called if i not in taken]


def without_aggregate_rows(frame: pd.DataFrame, report: dict,
                           sheet_rows: Optional[List[Optional[int]]] = None
                           ) -> Tuple[pd.DataFrame, List[int]]:
    """``frame`` without its totals rows, and the positions of the rows kept.

    What was taken out is written to ``report["aggregate_rows"]`` so the user
    can be told: a row removed without saying so is its own wrong answer.
    ``sheet_rows`` gives each row's number as the spreadsheet shows it.
    """
    found, kept = find_aggregate_rows(frame)

    def where(position: int) -> Optional[int]:
        return sheet_rows[position] if sheet_rows else None

    report["aggregate_rows"] = [
        {"row": where(position), "label": label, "kind": kind}
        for position, label, kind in found]
    # Rows called a total that stayed. Not listed when the word is simply
    # common in the data ("Total sales" as a metric name on most rows).
    if len(kept) * 2 <= len(frame):
        report["labelled_total_rows_kept"] = [
            {"row": where(position), "label": label} for position, label in kept]
    else:
        report["labelled_total_rows_kept"] = []
    if not found:
        return frame, list(range(len(frame)))
    taken = {position for position, _label, _kind in found}
    keep = [i for i in range(len(frame)) if i not in taken]
    return frame.iloc[keep], keep


def _row_named(entry: dict, first: bool = False) -> str:
    label = f"'{entry['label']}'" if entry.get("label") else "no label"
    if not entry.get("row"):
        return ("A row" if first else "a row") + f" ({label})"
    return ("Row" if first else "row") + f" {entry['row']} ({label})"


def _quoted(values: List[str], limit: int = 3) -> str:
    shown = ", ".join(f"'{v}'" for v in values[:limit])
    return shown + (f" and {len(values) - limit} more" if len(values) > limit else "")


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def describe_shape(report: dict, columns: Optional[List[str]] = None) -> List[str]:
    """What the reader assumed about a sheet, as sentences for the person who
    uploaded it. Empty when nothing was assumed.

    No reader is right about every hand-made spreadsheet, so the useful thing
    is to say what was guessed: which row was the header, what was skipped,
    what was left out. Someone who can see the guess can correct it; someone
    who cannot only sees wrong numbers.

    These sentences name the user's own columns and quote their own notes.
    They are for the user's screen and nowhere else -- never a log line that
    is shipped, never a telemetry or analytics field.
    """
    if not report:
        return []
    said: List[str] = []
    rows = report.get("header_rows") or []
    leading = report.get("dropped_leading_rows") or 0
    if report.get("synthesized_names"):
        names = report.get("placeholder_names") or []
        said.append(
            "No header row was found, so every row was kept as data and the "
            f"{_count(len(names), 'column')} were named "
            f"{names[0]} to {names[-1]}." if len(names) > 1 else
            "No header row was found, so every row was kept as data and the "
            "column was named column_1.")
    elif len(rows) > 1:
        example = f" (for example '{columns[-1]}')" if columns else ""
        said.append(
            f"Rows {rows[0] + 1} and {rows[-1] + 1} were read together as one header, "
            f"so each column is named from both rows{example}.")
    elif leading:
        said.append(f"Row {rows[0] + 1} was taken as the header.")
    if leading and rows:
        titles = report.get("leading_text") or []
        said.append(
            ("The row" if leading == 1 else f"The {leading} rows") + " above the header "
            + (f"({_quoted(titles, 2)}) " if titles else "(blank) ")
            + f"{'was' if leading == 1 else 'were'} not read as data.")
    numeric = report.get("numeric_header_cells") or []
    if numeric:
        said.append(
            f"Row {rows[0] + 1} was read as the header even though {_quoted(numeric)} "
            + ("is a number: it was taken as a column name, not as a value."
               if len(numeric) == 1 else
               "are numbers: they were taken as column names, not as values."))
    placeholders = report.get("placeholder_names") or []
    if placeholders and not report.get("synthesized_names"):
        said.append(
            f"{_count(len(placeholders), 'column')} had no header text and "
            f"{'was' if len(placeholders) == 1 else 'were'} named {_quoted(placeholders)}.")
    if report.get("dropped_empty_columns"):
        said.append(f"{_count(report['dropped_empty_columns'], 'empty column')} "
                    f"{'was' if report['dropped_empty_columns'] == 1 else 'were'} left out.")
    if report.get("dropped_empty_rows"):
        said.append(f"{_count(report['dropped_empty_rows'], 'blank row')} inside the table "
                    f"{'was' if report['dropped_empty_rows'] == 1 else 'were'} skipped.")
    notes = report.get("trailing_notes") or []
    if notes:
        said.append(
            ("The last row" if len(notes) == 1 else f"The last {len(notes)} rows")
            + f" under the table {'was' if len(notes) == 1 else 'were'} read as notes, "
            f"not data: {_quoted(notes, 2)}.")
    markers = report.get("footnote_markers") or []
    if markers:
        said.append(
            (f"Footnote marker {_quoted(markers)} was" if len(markers) == 1
             else f"Footnote markers {_quoted(markers, 5)} were")
            + " removed from column names and labels, because the notes under the "
            "table define them.")
    totals = report.get("aggregate_rows") or []
    if len(totals) == 1:
        said.append(
            f"{_row_named(totals[0], first=True)} equals the sum of the rows above it, "
            "so it was read as a totals row and left out of the data. Sums and "
            "averages are over the remaining rows only.")
    elif totals:
        said.append(
            f"{len(totals)} rows were read as subtotals and a grand total and left out "
            "of the data, because each equals the sum of the rows it closes: "
            + ", ".join(_row_named(t) for t in totals[:4])
            + (f" and {len(totals) - 4} more" if len(totals) > 4 else "") + ".")
    stayed = report.get("labelled_total_rows_kept") or []
    if stayed:
        said.append(
            (f"{_row_named(stayed[0], first=True)} is" if len(stayed) == 1 else
             ", ".join(_row_named(t, first=not n) for n, t in enumerate(stayed[:3]))
             + (f" and {len(stayed) - 3} more" if len(stayed) > 3 else "") + " are")
            + " labelled like a total, but the numbers could not be shown to be the sum "
            "of other rows, so " + ("it was" if len(stayed) == 1 else "they were")
            + " kept as data. If " + ("it is a total" if len(stayed) == 1 else "they are totals")
            + ", sums over the column will count those figures twice.")
    if report.get("rows_below_table"):
        where = report.get("next_table_row")
        said.append(
            "A second table starts "
            + (f"at row {where}" if where else "below the first")
            + f". Its {_count(report['rows_below_table'], 'row')} "
            f"{'is' if report['rows_below_table'] == 1 else 'are'} not part of this dataset.")
    return said


def changes_the_table(report: dict) -> bool:
    """Whether reshape found something a plain header-on-row-0 read gets wrong.

    A sheet that starts at A1 under a one-row header reads the same either
    way, and a caller that already has that read should keep it.
    """
    return bool(len(report.get("header_rows") or []) > 1
                or report.get("trailing_note_rows")
                or report.get("rows_below_table")
                or report.get("aggregate_rows"))


_MANGLED_RE = re.compile(r"^(.*)\.\d+$")


def may_hide_structure(df: pd.DataFrame) -> bool:
    """Whether a default read could be covering for a table reshape would read
    differently. Cheap, and deliberately loose: it only decides whether to
    look. A merged header leaves ``Unnamed: N``, a repeated group label leaves
    pandas' ``Sales.1``, and notes or a second table sit below a blank row.
    """
    if df is None or df.empty:
        return False
    names = [str(c) for c in df.columns]
    if any(_UNNAMED_RE.match(name) for name in names):
        return True
    for name in names:
        found = _MANGLED_RE.match(name)
        if found and found.group(1) in names:
            return True
    return bool(df.isna().all(axis=1).any())


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
    return [(frame, report) for frame, report, _rows, _columns
            in extract_tables_with_positions(raw, min_rows)]


def extract_tables_with_positions(
    raw: pd.DataFrame, min_rows: int = 2
) -> List[Tuple[pd.DataFrame, dict, list, list]]:
    """extract_tables(), each table with its reshape_with_positions() labels."""
    tables: List[Tuple[pd.DataFrame, dict, list, list]] = []
    for column_block in split_column_blocks(raw):
        for row_block in split_row_blocks(column_block):
            frame, report, rows, columns = reshape_with_positions(row_block)
            if frame.empty or len(frame) < min_rows or not len(frame.columns):
                continue
            report["source_columns"] = [int(c) for c in row_block.columns]
            tables.append((frame, report, rows, columns))
    return tables
