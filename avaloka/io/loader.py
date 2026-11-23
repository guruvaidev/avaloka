"""Dataset ingestion: tabular files, plus bounded semi-structured and complex datasets.

Reads the formats the product declares it accepts: CSV, TSV, JSON, XML,
Parquet, ORC, Avro and Excel. Database and object-storage connectors slot in
behind the same :class:`DatasetHandle`.

Four of those had no branch here at all, and the two failure shapes they
produced were not equally visible. JSON and ORC raised; **XML and Avro were
accepted by the delimiter sniffer** and came back as a one-column frame named
after the file's own declaration or its binary schema header. A frame like
that profiles, narrates and prices exactly like a real table, so the gap was
invisible from the outside -- which is why loading is defensive here but never
silent: a file that cannot be read correctly raises
:class:`UnusableDataset` naming what was wrong with it.

Only Avro needs a library outside this package's declared dependencies;
pandas and pyarrow cover the rest, and XML uses the standard library's
``etree`` parser when ``lxml`` is absent.

Semi-structured records (JSON, JSONL/NDJSON and document-database rows) go
through one normalization path: nested mappings become dotted column paths, and
arrays of scalars stay lossless as JSON-encoded strings rather than leaking
Python objects into Parquet or modelling code. What that found is recorded on
``DatasetHandle.structure``. A list of *objects* is still refused (see
``_repeated_groups``): it is a second table, not a column.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pandas as pd

from avaloka.stats import _looks_like_dates
if TYPE_CHECKING:
    from avaloka.io.complex import ComplexDataOptions

_JSON_LINES_SUFFIXES = {".jsonl", ".ndjson"}
#: Every file suffix ``load_dataset`` reads by name (a directory of parts in
#: one of ``_PART_SUFFIXES`` is read too). The CLI guide reads this.
SUPPORTED_SUFFIXES = frozenset({
    ".csv", ".tsv", ".txt", ".parquet", ".pq", ".json", ".jsonl", ".ndjson",
    ".xml", ".orc", ".avro", ".xlsx", ".xlsm", ".xls",
})
_WHITESPACE = re.compile(r"\s+")


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
    # The read that produced ``frame``, as arguments someone else could repeat:
    # the generated analysis.py is written from this, so the script a user is
    # handed reads the file the way the analysis did. ``kind`` is "csv",
    # "parquet" or "excel"; a CSV also carries sep, encoding, encoding_errors,
    # keep_default_na and na_values (None when pandas' defaults were used).
    read: dict[str, Any] = field(default_factory=dict)
    # What normalization found: dotted nested paths, list-valued paths, JSON
    # object columns expanded from a CSV/Parquet, column dtypes. Additive and
    # dict-shaped, so callers that only read ``frame`` are unaffected.
    structure: dict[str, Any] = field(default_factory=dict)


def normalize_text(value: str) -> str:
    """Return stable, searchable text without changing semantic content."""
    value = unicodedata.normalize("NFKC", value).replace("\x00", " ")
    return _WHITESPACE.sub(" ", value).strip()


def _normal_scalar(value: Any) -> Any:
    """Coerce connector-specific scalars into deterministic frame values."""
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, bytes):
        return normalize_text(value.decode("utf-8", errors="replace"))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Decimal, UUID)):
        return str(value)
    # ObjectId and other driver scalars intentionally use their stable string
    # representation. This keeps pyarrow conversion from failing on valid docs.
    return normalize_text(str(value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, set):
        return [_json_safe(v) for v in sorted(value, key=lambda item: repr(item))]
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return _normal_scalar(value)


def _flatten_record(
    record: Mapping[str, Any],
    *,
    separator: str,
    nested_paths: set[str],
    list_paths: set[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            if path:
                nested_paths.add(path)
            if not value and path:
                out[path] = "{}"
                return
            for key, child in value.items():
                child_path = f"{path}{separator}{key}" if path else str(key)
                visit(child, child_path)
            return
        if isinstance(value, (list, tuple, set)):
            list_paths.add(path)
            out[path] = json.dumps(
                _json_safe(value), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            )
            return
        out[path] = _normal_scalar(value)

    for key, value in record.items():
        visit(value, str(key))
    return out


def normalize_records(
    records: Iterable[Mapping[str, Any] | Any], *, separator: str = "."
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Flatten semi-structured records and return their structural manifest.

    Mappings become dotted paths, arrays remain deterministic JSON, and scalar
    records are wrapped in a ``value`` column. The manifest is safe to expose
    to a planner: it contains field names/types, never record values.
    """
    nested_paths: set[str] = set()
    list_paths: set[str] = set()
    flat: list[dict[str, Any]] = []
    for record in records:
        mapping = record if isinstance(record, Mapping) else {"value": record}
        flat.append(_flatten_record(
            mapping, separator=separator, nested_paths=nested_paths,
            list_paths=list_paths,
        ))

    frame = pd.DataFrame.from_records(flat)
    text_columns = [
        str(column)
        for column in frame.columns
        if any(isinstance(v, str) for v in frame[column].dropna().head(100).tolist())
    ]
    manifest = {
        "normalization": "dotted_paths_v1",
        "separator": separator,
        "nested_paths": sorted(nested_paths),
        "list_paths": sorted(list_paths),
        "text_columns": text_columns,
        "columns": {str(c): str(frame[c].dtype) for c in frame.columns},
    }
    return frame, manifest


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


#: Words of a question that say what to compute or how to ask, not what the
#: question is about. Left out before a question is matched against a sheet:
#: "total" would otherwise pick any sheet with a Total row.
_QUESTION_FILLER = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "count", "did", "do",
    "does", "each", "for", "from", "give", "has", "have", "how", "in", "into", "is",
    "it", "its", "many", "me", "mean", "median", "much", "of", "on", "or", "over",
    "per", "show", "sum", "tell", "than", "that", "the", "their", "this", "to",
    "total", "average", "was", "were", "what", "when", "where", "which", "who",
    "why", "with",
})

#: Rows of a sheet whose text is matched against the question: where a title
#: and a header can be (the header search looks no further).
_SHEET_TEXT_ROWS = 25

#: Row labels matched against the question: a statement's line items are in
#: its first column ("Net income"), not in its header.
_SHEET_LABEL_ROWS = 500


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[^\W_]+", str(text).casefold())
            if len(w) >= 2 and w not in _QUESTION_FILLER}


def _sheet_words(name: Any, raw: pd.DataFrame) -> set[str]:
    """The words a sheet says about itself: its name, the text of its first
    rows (title and header) and its first column's labels. Numbers in cells
    are values, not names, and are left out."""
    words = _words(name)
    top = raw.iloc[:_SHEET_TEXT_ROWS]
    for value in top.to_numpy().ravel():
        if isinstance(value, str):
            words |= _words(value)
    filled = [c for c in raw.columns if raw[c].notna().any()]
    if filled:
        for value in raw[filled[0]].iloc[:_SHEET_LABEL_ROWS]:
            if isinstance(value, str):
                words |= _words(value)
    return words


def _choose_sheet(book: dict, question: str | None,
                  sheet: str | None) -> tuple[Any, dict[str, Any]]:
    """Which sheet to read, and why: ``(name, {"by", "matched"})``.

    * ``sheet`` named by the caller wins, and an unknown name is an error that
      lists the sheets -- reading another one instead would be a silent guess.
    * Otherwise, with a question and more than one sheet, the sheet whose own
      words (see _sheet_words) share the most words with the question. Only a
      unique winner counts: a tie, or no shared word at all, decides nothing.
    * Otherwise the sheet with the most non-empty cells, as before.

    Size is a poor proxy for relevance: a reference tab dwarfs the summary
    someone asks about, and the longer of two financial statements wins.
    """
    if sheet is not None:
        if sheet not in book:
            raise ValueError(f"the workbook has no sheet {sheet!r}; its sheets are "
                             f"{', '.join(repr(str(n)) for n in book)}")
        return sheet, {"by": "named", "matched": []}
    cells = {name: int(raw.notna().sum().sum()) for name, raw in book.items()}
    if question and len(book) > 1:
        asked = _words(question)
        shared = {name: sorted(asked & _sheet_words(name, raw)) for name, raw in book.items()}
        best = max(len(words) for words in shared.values())
        winners = [name for name, words in shared.items() if len(words) == best]
        if best and len(winners) == 1:
            return winners[0], {"by": "question", "matched": shared[winners[0]]}
    return max(cells, key=lambda name: cells[name]), {"by": "size", "matched": []}


#: What the table-end stages report, as the profile carries it.
_SHAPE_KEYS = ("rows_below_table", "next_table_row", "trailing_note_rows", "trailing_notes",
               "footnote_markers", "aggregate_rows", "labelled_total_rows_kept")


def _table_ends(frame: pd.DataFrame, first_sheet_row: int) -> tuple[list, list[str], dict[str, Any]]:
    """Where the table under the header ends, from a sheet read under its header.

    ``frame`` is the sheet as ``pd.read_excel(header=...)`` returned it, and
    its row ``i`` is sheet row ``first_sheet_row + i`` (0-based). Returns
    ``(rows to keep, footnote lines, report)``: the rows are labels of a frame
    indexed by sheet row. Four things under a table are not its records, and
    each is left out only on the evidence avaloka.io.table_shape asks for:

    * a second table, two blank rows down, with a header of its own;
    * a block of notes in the first column a blank row under the last record;
    * the footnote markers those notes define, on names and labels (applied
      by the caller with the lines returned here);
    * totals and subtotal rows whose numbers add up to the rows they close.

    The same stages read the upload path's workbooks
    (file_handler/table_shape.py; the copy is held to it by
    tests/test_table_shape_parity.py).
    """
    from avaloka.io import table_shape as ts

    report: dict[str, Any] = {}
    body = frame.copy()
    body.index = range(first_sheet_row, first_sheet_row + len(body))
    body = ts._stop_at_the_next_table(body, report)
    body, notes = ts._without_trailing_notes(body, report)
    if notes:
        body = ts._without_footnote_markers(body, notes, report)
    body, _kept = ts.without_aggregate_rows(
        body, report, sheet_rows=[ts._sheet_row(label) for label in body.index])
    shape = {key: report[key] for key in _SHAPE_KEYS if report.get(key)}
    return list(body.index), notes, shape


def _cut_to(frame: pd.DataFrame, first_sheet_row: int, rows: list, notes: list[str]) -> pd.DataFrame:
    """``frame`` cut to ``rows`` (sheet rows, see _table_ends), its object
    columns typed now the rows that were not records are gone, and the
    footnote markers the notes define taken off names and labels."""
    from avaloka.io import table_shape as ts

    out = frame.copy()
    out.index = range(first_sheet_row, first_sheet_row + len(out))
    out = out.loc[rows].infer_objects()
    if notes:
        out = ts._without_footnote_markers(out, notes, {})
    return out.reset_index(drop=True)


def _two_row_header(raw: pd.DataFrame, header_row: int) -> tuple[int, int, list | None]:
    """``(first header row, last header row, names)`` for the sheet ``raw``.

    ``names`` is None unless table_shape finds a header spanning two rows; it
    then has one name per column of ``raw`` ("Transactions Count"), with the
    empty columns the span ignored left None. Otherwise the one-row header the
    caller found stands: ``(header_row, header_row, None)``.
    """
    from avaloka.io import table_shape as ts

    filled = [position for position in range(len(raw.columns)) if raw.iloc[:, position].notna().any()]
    if not filled:
        return header_row, header_row, None
    table = raw.iloc[:, filled]
    span = ts.detect_header_rows(table)
    if span is None or span[1] == span[0] or span[1] + 1 >= len(table):
        return header_row, header_row, None
    first, last = span
    spoken = ts._two_row_header_names(table.iloc[first].tolist(), table.iloc[last].tolist(),
                                      table.iloc[last + 1].tolist())
    if spoken is None:
        return header_row, header_row, None
    names: list = [None] * len(raw.columns)
    for position, name in zip(filled, ts._clean_names(spoken, len(filled))):
        names[position] = name
    return first, last, names


def _named(frame: pd.DataFrame, names: list | None) -> pd.DataFrame:
    """``frame`` with the two-row header's names, where it has them. A read
    whose width does not match the sheet's keeps pandas' names."""
    if names is None or len(names) != len(frame.columns):
        return frame
    frame = frame.copy()
    frame.columns = [name if name is not None else column
                     for name, column in zip(names, frame.columns)]
    return frame


def _read_excel(path, question: str | None = None, sheet: str | None = None):
    """Read a spreadsheet, finding the header row rather than assuming row 1.

    Hand-maintained sheets routinely open with a title and a subtitle before the
    real header -- the R1 programme plan this was written against has its header
    on row 4. Reading row 1 as the header yields one named column and a frame of
    mostly-NaN, which then gets profiled as if the data were that shape.

    Picks the sheet with the most **non-empty cells** -- not the most rows; the
    docstring said rows for a while and the two are not the same ranking. The
    upload path (``file_handler/excel_connector.py::_resolve_sheet``) does rank
    by rows, so on a workbook where the longer sheet is the sparser one the two
    readers resolve *different sheets* from the same file. Anyone reading the
    old wording would have concluded they agree.

    Then takes the first row where a majority of cells are non-empty strings and
    the values below are populated.

    With a ``question``, the sheet is chosen by what the question is about
    when that settles it, and by size only when it does not (_choose_sheet);
    ``sheet`` names one outright. Which rule decided is recorded in
    ``frame.attrs["sheet_choice"]`` and reported, because a workbook read off
    the wrong sheet reads perfectly and answers confidently. Graded by
    ``GroundTruth.sheet_name`` and ``comprehension.score_sheet``.
    """
    import pandas as pd

    book = pd.read_excel(path, sheet_name=None, header=None)
    if not book:
        raise ValueError("the workbook contains no sheets")
    name, choice = _choose_sheet(book, question, sheet)
    raw = book[name]
    # The sheets not analysed, with the rows they hold, so the choice is
    # reported rather than silent: a workbook of two equal data sheets used to
    # come back as half its rows and nothing said so.
    other_sheets = [
        {"name": str(other), "rows": max(int(sheet.dropna(how="all").shape[0]) - 1, 0)}
        for other, sheet in book.items() if other != name
    ]

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

    # A header written on two rows: a group label merged or repeated across
    # columns over the labels under it ("Transactions | | Refunds |" over
    # "Count | Value | Count | Value"). The scan above can only answer with
    # one row, so the second was read as the first record and its columns
    # came back unnamed or as Sales.1. The upload path's span detection
    # (avaloka/io/table_shape.py) answers with both rows; each column is then
    # named from both, and the data starts under the second. A one-row header
    # is left to the scan above, so its names are exactly as before.
    header_row, last_header_row, two_row_names = _two_row_header(raw, header_row)

    # The null-word rule below applies here as it does to a CSV: read_excel
    # turns "None" and "NA" into missing values in every column by default, so
    # a region literally called None left every group-by. A default read finds
    # the text columns, and only when there are some is the sheet read again.
    frame = _named(pd.read_excel(path, sheet_name=name, header=last_header_row), two_row_names)
    # Where the table ends. A totals row, a footnote block or a second table
    # under the first used to be read as more records: every sum doubled, the
    # row count off, the second table's text in a numeric column. The ends
    # are found once, on this read, and the same rows are kept from the
    # re-read below, so the null-word rule cannot move them. A frame that
    # does not line up with the sheet is left as it was.
    first = last_header_row + 1
    rows, notes, shape = list(range(first, first + len(frame))), [], {}
    if len(frame) == len(raw) - first:
        rows, notes, shape = _table_ends(frame, first)
    original = list(frame.columns)
    frame = _cut_to(frame, first, rows, notes)
    renamed = dict(zip(original, frame.columns))
    # Accounting figures are read as numbers here and only here. A workbook
    # export writes a cost as the text "(1,234)" next to real numbers, so the
    # column was left as text and every statistic skipped it, or a total read
    # its costs as positive. The upload path has read the brackets as a minus
    # sign since table_shape._as_numeric; this brings the CLI to the same read.
    text = _csv_text_columns(frame, accounting=True)
    missing = None
    if text:
        every_word = sorted(NULL_WORDS | {""})
        by_name = {renamed[column]: column for column in original}
        missing = {by_name[column]: ([""] if column in text else every_word)
                   for column in frame.columns}
        frame = _cut_to(_named(pd.read_excel(path, sheet_name=name, header=last_header_row,
                                             keep_default_na=False, na_values=missing),
                               two_row_names),
                        first, rows, notes)
    frame.attrs["na_values"] = missing
    frame.attrs["coerced"] = _coerce_typed_columns(frame, text, accounting=True)
    frame.attrs["sheet_name"] = name
    frame.attrs["header_row"] = header_row + 1        # 1-based, as a human counts
    frame.attrs["other_sheets"] = other_sheets
    frame.attrs["sheet_choice"] = choice
    frame.attrs["shape"] = shape
    # The unit and scale the sheet states about its numbers -- in a title row
    # above the header ("Revenue by region (£ thousands)") or in a header
    # cell ("Cost ($m)"). Read out of the file's own text, never inferred
    # from what a column is called, and never applied: the figures stay as
    # written and the scale is reported beside them (avaloka/units.py).
    from avaloka.units import stated_units
    titles = [(i + 1, value) for i in range(header_row) for value in raw.iloc[i]
              if isinstance(value, str) and value.strip()]
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])
               and not pd.api.types.is_bool_dtype(frame[c])]
    frame.attrs["units"] = stated_units(titles, [str(c) for c in frame.columns],
                                        [str(c) for c in numeric])
    return frame


# --------------------------------------------------------------------------
# Nested documents: one stated rule, not an improvisation
# --------------------------------------------------------------------------
# JSON and XML are trees; a dataframe is a rectangle. Flattening a tree always
# throws information away, and the only dishonest option is to pick an
# interpretation silently -- which is the defect class this module keeps being
# fixed for. XML was being read as a one-column CSV named after its own
# declaration, and nothing downstream could tell.
#
# So the rule is declared here and the benchmark grades it:
#
# 1. An array of flat objects is a table: one column per key.
# 2. **Scalar nesting flattens**, joined with ``.``. ``{"a": {"b": 1}}`` gives a
#    column ``a.b``. Lossless and order-preserving, so the names a user sees are
#    derivable from the document.
# 3. **A repeated group is refused, not guessed.** A record holding an array of
#    objects is one-to-many: exploding it multiplies the row count, keeping the
#    first element drops data, and joining it to a string makes a number
#    unusable. All three are wrong in a way nothing downstream can detect, so
#    the reader refuses and names the offending key.
# 4. **More than one candidate table is refused**, naming the candidates. A
#    document with two top-level arrays has no single answer to "which table is
#    this", and choosing the longer one is a coin toss presented as a fact.
#
# Refusing is the point. A loud refusal naming the key costs the user one
# decision; a silent guess costs them a wrong answer they cannot see.
NESTED_SEPARATOR = "."


def _repeated_groups(records: list[dict]) -> list[str]:
    """Keys whose value is a list of objects -- a one-to-many, in any record."""
    offenders: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            if (isinstance(value, list) and value
                    and any(isinstance(item, (dict, list)) for item in value)
                    and key not in offenders):
                offenders.append(str(key))
    return offenders


def _read_json(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a JSON document under the nesting rule above.

    ``load_dataset`` had no JSON branch at all, so a ``.json`` file fell through
    to the delimiter sniffer, was read as one wide CSV row, and then failed
    ``_reject_unusable`` with "has column headers but no rows" -- an error about
    the wrong thing entirely.
    """
    import json

    try:
        document = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise UnusableDataset(
            f"{path.name} is not valid JSON: {exc}. Expected either an array of "
            "objects, or an object containing exactly one such array."
        ) from exc

    if isinstance(document, dict):
        # One table, or a wrapper around one. Anything else is ambiguous.
        tables = {key: value for key, value in document.items()
                  if isinstance(value, list) and value
                  and all(isinstance(item, dict) for item in value)}
        if len(tables) > 1:
            raise UnusableDataset(
                f"{path.name} holds {len(tables)} candidate tables "
                f"({sorted(tables)}). Avaloka will not choose between them: "
                "extract the one you mean, or say which key holds it.")
        if len(tables) == 1:
            records = next(iter(tables.values()))
        elif any(isinstance(v, (dict, list)) for v in document.values()):
            records = [document]          # a single nested record: one row
        else:
            records = [document]          # a flat object: one row
    elif isinstance(document, list):
        if not document:
            raise UnusableDataset(f"{path.name} contains an empty array.")
        if not all(isinstance(item, dict) for item in document):
            raise UnusableDataset(
                f"{path.name} is an array of values, not of objects, so it has "
                "no columns to name. Expected an array of objects.")
        records = document
    else:
        raise UnusableDataset(
            f"{path.name} holds a single {type(document).__name__}, which is not "
            "a table.")

    offenders = _repeated_groups(records)
    if offenders:
        raise UnusableDataset(
            f"{path.name} has repeated group(s) {offenders}: each record holds a "
            "list of objects, which is a second table rather than a column. "
            "Avaloka refuses rather than guessing -- flattening it would "
            "multiply the row count, and keeping the first element would drop "
            "data, and neither is visible in the result. Extract the nested "
            "records into their own file.")

    return normalize_records(records, separator=NESTED_SEPARATOR)


def _read_json_lines(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read JSONL/NDJSON: one record per line, under the same nesting rule as
    ``_read_json`` so the two formats cannot disagree about a list of objects."""
    records: list[Any] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise UnusableDataset(
                    f"{path.name} line {line_no} is not valid JSON: {exc.msg}."
                ) from exc
    if not records:
        raise UnusableDataset(f"{path.name} contains no records.")
    offenders = _repeated_groups(records)
    if offenders:
        raise UnusableDataset(
            f"{path.name} has repeated group(s) {offenders}: each record holds a "
            "list of objects, which is a second table rather than a column. "
            "Extract the nested records into their own file.")
    return normalize_records(records, separator=NESTED_SEPARATOR)


def _read_xml(path: Path) -> pd.DataFrame:
    """Read an XML document under the same nesting rule.

    This is the branch whose absence was worst. There was no ``.xml`` case, so
    the suffix fell through to the delimiter sniffer, which *accepted* the file
    and returned a frame of 1,682 rows with a single column named
    ``<?xml version='1.0' encoding='utf-8'?>``. It raised nothing. The profile,
    the narration and the economics were all computed over the file's own
    declaration.

    ``lxml`` is preferred when present (it is in the ``benchmark`` extra and the
    app's dependencies) but is **not** required: pandas' ``etree`` parser is in
    the standard library, so this costs the CLI no dependency.
    """
    from xml.etree import ElementTree

    # A data file has no legitimate use for a DTD, and two attacks ride in on
    # one. Uploaded XML is untrusted input:
    #
    #   * an external entity (`<!ENTITY x SYSTEM "file:///etc/passwd">`) turns a
    #     table read into a local-file read, or an outbound request;
    #   * nested internal entities expand quadratically -- a four-level fixture
    #     measured here grew one cell to 10,000 characters, and the shape scales
    #     to memory exhaustion.
    #
    # Refusing the DOCTYPE outright stops both, and does not depend on which
    # parser runs. That matters: the structural scan below uses the standard
    # library, which happens to reject undefined external entities, while
    # `pd.read_xml` prefers `lxml`, which resolves them. So the safety of this
    # function would otherwise rest on the *order* of two unrelated steps --
    # exactly the kind of accident that survives until someone tidies it.
    head = path.read_bytes()[:4096].lstrip()
    if b"<!DOCTYPE" in head.upper():
        raise UnusableDataset(
            f"{path.name} carries a DOCTYPE / DTD declaration, which Avaloka "
            "refuses in a data file. A DTD is how an XML upload reads local "
            "files or expands into an out-of-memory error, and a table needs "
            "none. Strip the DOCTYPE and the <!ENTITY> declarations.")

    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise UnusableDataset(
            f"{path.name} is not well-formed XML: {exc}") from exc

    rows = list(root)
    if not rows:
        raise UnusableDataset(
            f"{path.name} has no child elements under <{root.tag}>, so there "
            "are no rows to read.")

    # A record whose children repeat a tag is a one-to-many, exactly as a JSON
    # array of objects is. pandas' read_xml would silently keep one and drop
    # the rest.
    offenders: list[str] = []
    for row in rows:
        seen: set[str] = set()
        for child in row:
            if child.tag in seen and child.tag not in offenders:
                offenders.append(str(child.tag))
            seen.add(child.tag)
        if any(len(grandchild) for grandchild in row):
            nested = [str(g.tag) for g in row if len(g)]
            for tag in nested:
                if tag not in offenders:
                    offenders.append(tag)
    if offenders:
        raise UnusableDataset(
            f"{path.name} has repeated or nested element(s) {offenders} inside "
            "each record, which is a second table rather than a column. "
            "Avaloka refuses rather than keeping one and dropping the rest.")

    for parser in ("lxml", "etree"):
        try:
            return pd.read_xml(path, parser=parser)
        except ImportError:
            continue                      # lxml absent: fall through to etree
    raise UnusableDataset(f"{path.name} could not be parsed as XML.")


def _read_orc(path: Path) -> pd.DataFrame:
    """Read an ORC file.

    Declared supported and readable by nothing: the app has no ORC connector,
    and here the binary container fell through to the delimiter sniffer. On a
    small file that produced a one-column frame named ``ORC`` -- the file's
    magic bytes -- and on a larger one a ``ValueError``. The failure mode
    depended on the bytes, which is its own problem.

    ``pyarrow`` reads ORC and is already one of the CLI's declared
    dependencies, so this needs nothing new.
    """
    try:
        return pd.read_orc(path)
    except ImportError as exc:            # pragma: no cover - pyarrow is declared
        raise UnusableDataset(
            f"reading {path.name} needs pyarrow, which is a declared dependency "
            f"of this package but is not importable: {exc}") from exc
    except Exception as exc:
        raise UnusableDataset(
            f"{path.name} is not a readable ORC file: {exc}") from exc


def _read_avro(path: Path) -> pd.DataFrame:
    """Read an Avro container.

    The one declared format that genuinely needs a library the CLI does not
    ship: ``pyarrow`` has no Avro reader, so this is behind an optional import
    that fails with the install command rather than a traceback. Previously the
    delimiter sniffer accepted the container and returned columns taken from
    the binary schema header, which then appeared in the generated report.
    """
    try:
        import fastavro
    except ImportError as exc:
        raise UnusableDataset(
            f"reading {path.name} needs an Avro reader, which is not among this "
            "package's runtime dependencies. Install it with: "
            "pip install 'avaloka[formats]'  (or: pip install fastavro)"
        ) from exc
    try:
        with path.open("rb") as handle:
            records = list(fastavro.reader(handle))
    except Exception as exc:
        raise UnusableDataset(
            f"{path.name} is not a readable Avro container: {exc}") from exc
    if not records:
        raise UnusableDataset(f"{path.name} contains no Avro records.")
    offenders = _repeated_groups(records)
    if offenders:
        raise UnusableDataset(
            f"{path.name} has repeated group(s) {offenders} inside each record, "
            "which is a second table rather than a column.")
    return pd.json_normalize(records, sep=NESTED_SEPARATOR)


def _sha256_path(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    """Hash a file or a bounded, deterministic directory manifest/content set."""
    if path.is_file():
        return _sha256(path, limit=limit)
    h = hashlib.sha256()
    consumed = 0
    files = sorted(
        item for item in path.rglob("*") if item.is_file() and not item.is_symlink()
    )
    for item in files[:10_000]:
        relative = str(item.relative_to(path)).encode("utf-8", errors="replace")
        h.update(len(relative).to_bytes(4, "big"))
        h.update(relative)
        h.update(item.stat().st_size.to_bytes(8, "big"))
        if consumed >= limit:
            continue
        with item.open("rb") as fh:
            while consumed < limit:
                chunk = fh.read(min(1 << 20, limit - consumed))
                if not chunk:
                    break
                h.update(chunk)
                consumed += len(chunk)
    h.update(f"files:{len(files)}".encode())
    return h.hexdigest()


def _sniff_delimiter(path: Path) -> str:
    """Pick the most likely delimiter from the header line (',', ';', '\\t', '|')."""
    with path.open("r", errors="replace") as fh:
        header = fh.readline()
    candidates = {d: header.count(d) for d in [",", ";", "\t", "|"]}
    best = max(candidates, key=candidates.get)
    return best if candidates[best] > 0 else ","


# --------------------------------------------------------------------------
# Null-like words: None, NA, N/A, null ...
#
# pandas turns a fixed list of words into missing values the moment it reads a
# CSV. That is right for ``amount = NA`` and wrong for ``payment_method = None``,
# which is a real category: its rows left every group-by and were reported as
# missing from a column that had no gaps.
#
#   An empty cell is always missing.
#   A null-like word is MISSING in a typed column and a LABEL in a text column.
#
# The rule, the word list and the mechanism are file_handler/null_policy.py's;
# read that file for the reasoning. This is a second copy, because the CLI is
# standalone and must not import file_handler or app. What can drift is the
# word list, the text-column decision and the sniff depth. One table of cases,
# tests/shared_cases/null_word_labels.py, is read by both implementations, and
# tests/test_cli_null_policy_parity.py compares what the two return for each
# case: change a rule here or there and it fails.
#
# ONE DELIBERATE DIFFERENCE: dates. The rule says a null-like word in a date
# column is missing, but pandas reads dates from a CSV as strings, so a reader
# that asks only "did pandas type it?" calls a date column text and keeps its
# NA gaps as labels. Here that is not harmless: stats.infer_role recognises a
# date column by how much of it parses, so a tenth of the cells reading "NA"
# turns the column categorical and the planner loses its temporal split. A
# column that stats would read as dates therefore counts as typed.
#
# Do not "simplify" this to keep_default_na=False. That keeps the labels and
# breaks every numeric column that writes its gaps as NA: it loads as text.
# Here a column keeps its null-like words only when pandas, left to its
# defaults, could not type it and it holds at least one real string.
#
# A SECOND CLAUSE (2026-10-18): "could not type it" was asked of pandas alone,
# and pandas gives up on a numeric column the moment one cell says "none" or
# "-" -- words its default list does not carry. A glucose column that was 89%
# numbers and 11% none/N/A/- came back as text, the words became its top
# values, its missing rate read 0.0 and the verdict passed. So a column whose
# real cells (null-like words set aside) are at least MOSTLY_TYPED_SHARE
# numbers is typed, whatever pandas said; the stray words are read as missing
# after the read, and the count is recorded so the profile can say so. The
# same share, as a stated rule in file_handler/null_policy.py, keeps the two
# readers in parity.
#
# WHAT MAY BECOME MISSING (2026-10-19): only a gap token -- a null-like word in
# any case, "-", "?", or a blank cell. The second clause used to read every
# other cell that did not parse as missing too, and on the diabetes data that
# turned 5,058 real ICD-9 codes (V45, E888) in diag_3 into "empty" next to
# 1,423 real "?" gaps. A string that is not a gap token is a REAL value, and
# no share or threshold licenses dropping one: a column that is mostly numbers
# but holds real strings is MIXED. It is kept as text, its gap tokens are
# missing, its real strings are kept, and the coercion record says how many
# real strings stopped it being numeric, with examples, so the profile says so.
# --------------------------------------------------------------------------

#: The strings pandas reads as missing by default, minus the empty string. A
#: literal, so the policy does not silently follow a pandas upgrade.
NULL_WORDS = frozenset({
    "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None",
    "n/a", "nan", "null",
})

#: Where a CSV read leaves its own arguments on the frame it returns.
_READ_ATTR = "avaloka_csv_read"

#: The share of a column's real cells that must parse as numbers for the
#: column to count as numeric despite stray words. Below it ("1.5, unknown,
#: NA, 2.5") the column is untypable and stays text, as before.
MOSTLY_TYPED_SHARE = 0.8

#: Set aside, with the null-like words in any case ("NONE", "none"), before the
#: share is taken: what an export writes for "no value" that pandas' list lacks.
_JUNK_TOKENS = frozenset({"-", "?"})
_NULL_WORDS_FOLDED = frozenset(word.casefold() for word in NULL_WORDS)


def _gap_tokens(cells: pd.Series) -> pd.Series:
    """Where a present cell is a gap token: a null-like word in any case, a
    junk token or blank. The only cells the coercion below may make missing."""
    text = cells.astype(str).str.strip()
    return text.str.casefold().isin(_NULL_WORDS_FOLDED) | text.isin(_JUNK_TOKENS) | (text == "")


def _real_cells(present: pd.Series) -> pd.Series:
    """Non-missing cells that are not null-like words (any case), not a junk
    token and not blank."""
    return present[~_gap_tokens(present)]


#: An accounting negative as a spreadsheet export writes it: the figure in
#: brackets, "(1,234)" or "(12.5)". Digits are required inside, so a label like
#: "(a)" or "(n/a)" is never a number.
_ACCOUNTING_NEGATIVE = r"^\(\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\)$"
#: A figure written with thousands separators in strict groups of three,
#: "1,234,567.5". "1,2" (a decimal comma) and "12,34,567" do not match and stay
#: what they were.
_GROUPED_THOUSANDS = r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$"


def _accounting_text(cells: pd.Series) -> pd.Series:
    """Cells as text, with accounting figures rewritten as plain numbers.

    ``(1,234)`` becomes ``-1234`` and ``1,234`` becomes ``1234``; every other
    cell is returned stripped and otherwise untouched, so ``pd.to_numeric`` on
    the result fails exactly where it failed before for anything that is not
    one of these two forms. Workbooks only (see _read_excel): the CSV readers
    are held to the platform's by tests/test_cli_null_policy_parity.py, and
    neither reads these forms from a CSV today.
    """
    text = cells.astype(str).str.strip()
    text = text.str.replace(_ACCOUNTING_NEGATIVE, r"-\1", regex=True)
    return text.str.replace(_GROUPED_THOUSANDS, lambda m: m.group(0).replace(",", ""),
                            regex=True)


def _is_accounting(cells: pd.Series) -> pd.Series:
    """Where a cell is a text accounting figure: bracketed or grouped."""
    text = cells.astype(str).str.strip()
    return cells.map(lambda v: isinstance(v, str)) & (
        text.str.match(_ACCOUNTING_NEGATIVE) | text.str.match(_GROUPED_THOUSANDS))


def _mostly_numeric(present: pd.Series, accounting: bool = False) -> bool:
    """Whether at least MOSTLY_TYPED_SHARE of the real cells parse as numbers.
    With ``accounting``, "(1,234)" and "1,234" count as numbers."""
    real = _real_cells(present)
    if not len(real):
        return False
    text = _accounting_text(real) if accounting else real.astype(str).str.strip()
    parsed = pd.to_numeric(text, errors="coerce")
    return bool(parsed.notna().mean() >= MOSTLY_TYPED_SHARE)


#: How many of a mixed column's real strings the coercion record names.
_MIXED_EXAMPLES = 5


def _coerce_typed_columns(frame: pd.DataFrame, text_columns: list,
                          accounting: bool = False) -> dict[str, Any]:
    """Read the gap tokens of a typed-but-object column as missing, in place.

    Only gap tokens become missing; every other cell is a real value and is
    kept. Returns ``{column: {"kind", "cells", "words"}}`` for every column
    changed -- ``kind`` "numeric" or "date" when nothing but gap tokens stood
    in the way, so the column is typed; "mixed" when real strings did, so the
    column stays text and the record adds ``kept`` (how many real strings) and
    ``examples``. The handle carries it and the generated script repeats it.
    Text columns are never touched: their words are labels.

    With ``accounting`` (workbooks), a figure written "(1,234)" is the number
    -1234 and "1,234" is 1234, and the record adds ``accounting``: how many
    cells were read that way, so the profile can say the brackets were read as
    a minus sign rather than leaving the user to wonder why a text column
    became numeric.
    """
    coerced: dict[str, Any] = {}
    for column in frame.columns:
        if column in text_columns or not _is_text_dtype(frame[column].dtype):
            continue
        original = frame[column]
        present = original.dropna()
        if not len(present):
            continue
        gaps = original.notna() & _gap_tokens(original)
        if _mostly_numeric(present, accounting):
            kept = original.where(~gaps)
            parsed = pd.to_numeric(_accounting_text(kept) if accounting else kept.astype(str).str.strip(),
                                   errors="coerce")
            real = original.notna() & ~gaps & parsed.isna()
            kind = "mixed" if real.any() else "numeric"
            frame[column] = original.where(~gaps) if real.any() else parsed
        elif _looks_like_dates(_real_cells(present)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(original.where(~gaps), errors="coerce", format="mixed")
            real = original.notna() & ~gaps & parsed.isna()
            if not (gaps.any() or real.any()):
                continue
            kind = "mixed" if real.any() else "date"
            frame[column] = original.where(~gaps)
        else:
            continue
        coerced[str(column)] = {
            "kind": kind, "cells": int(gaps.sum()),
            "words": sorted({str(v) for v in original[gaps]}),
        }
        if accounting and kind == "numeric":
            figures = int((original.notna() & ~gaps & _is_accounting(original)).sum())
            if figures:
                coerced[str(column)]["accounting"] = figures
        if kind == "mixed":
            coerced[str(column)].update(
                kept=int(real.sum()),
                examples=sorted({str(v) for v in original[real]})[:_MIXED_EXAMPLES])
    return coerced

#: How many rows are read to tell text columns from typed ones. A column that
#: looks typed in these rows and holds real text further down is treated as
#: typed, so its null-like words are lost as they were before.
CSV_SNIFF_ROWS = 20000


def _is_text_dtype(dtype) -> bool:
    """Whether pandas left a column untyped. pandas 2 reads text as ``object``;
    pandas 3 reads it as its ``str`` dtype, and asking only ``!= object`` there
    calls every text column typed and loses every label."""
    return dtype == object or isinstance(dtype, pd.StringDtype)


def _csv_text_columns(head: pd.DataFrame, accounting: bool = False) -> list:
    """The columns of a default-read sample whose null-like words are labels.
    ``accounting`` as in _coerce_typed_columns (workbooks only)."""
    text = []
    for position in range(len(head.columns)):
        series = head.iloc[:, position]
        if not _is_text_dtype(series.dtype):
            continue                      # pandas typed it: numbers, booleans
        present = series.dropna()
        if not any(isinstance(cell, str) and cell.strip() for cell in present.tolist()):
            continue                      # no real string: nothing says these are labels
        if _mostly_numeric(present, accounting):
            continue                      # numbers with stray words: typed, see above
        if _looks_like_dates(_real_cells(present)):
            continue                      # dates written as text: typed, see above
        text.append(head.columns[position])
    return text


def _read_csv_keeping_labels(path: Path, **kwargs):
    """``pd.read_csv`` under the rule above: a sniff, then one full read.

    pandas has to be told which strings are missing before it reads, so the
    first CSV_SNIFF_ROWS rows are read with its defaults to find the text
    columns, and the full read then gets a per-column list of missing strings:
    only "" for a text column, the whole default list for every other.
    """
    head = pd.read_csv(path, nrows=CSV_SNIFF_ROWS, **kwargs)
    text = _csv_text_columns(head)
    if not text:
        # Nothing to keep: exactly the read pandas always did.
        frame = pd.read_csv(path, **kwargs)
        frame.attrs[_READ_ATTR] = {"keep_default_na": True, "na_values": None}
    else:
        every_word = sorted(NULL_WORDS | {""})
        missing = {column: ([""] if column in text else every_word) for column in head.columns}
        frame = pd.read_csv(path, keep_default_na=False, na_values=missing, **kwargs)
        frame.attrs[_READ_ATTR] = {"keep_default_na": False, "na_values": missing}
    # Stray words in a typed column ("none", "-") survive both reads as text;
    # they are missing values, and the count is kept for the profile.
    frame.attrs[_READ_ATTR]["coerced"] = _coerce_typed_columns(frame, text)
    return frame


def _read_csv(path: Path, delimiter: str):
    """Read a CSV, tolerating non-UTF-8 exports (common in real-world data).

    The arguments of the read that succeeded are left on the frame, under
    ``frame.attrs[_READ_ATTR]``, for load_dataset to move onto the handle.
    """
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            frame = _read_csv_keeping_labels(path, sep=delimiter, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        frame.attrs[_READ_ATTR].update(sep=delimiter, encoding=encoding, encoding_errors=None)
        return frame
    # Last resort: replace undecodable bytes rather than crash the mission.
    frame = _read_csv_keeping_labels(path, sep=delimiter, encoding="utf-8",
                                     encoding_errors="replace")
    frame.attrs[_READ_ATTR].update(sep=delimiter, encoding="utf-8", encoding_errors="replace")
    return frame


def _decode_json_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str) or not value.lstrip().startswith("{"):
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _expand_json_object_columns(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expand columns that consistently contain JSON objects.

    The original column is retained for lossless lineage. Expansion only
    occurs when at least 80% of sampled non-empty values decode as mappings.
    """
    expanded: list[str] = []
    nested_paths: set[str] = set()
    list_paths: set[str] = set()
    for column in list(frame.columns):
        sample = frame[column].dropna().head(100).tolist()
        if not sample:
            continue
        decoded = [_decode_json_mapping(value) for value in sample]
        successes = sum(value is not None for value in decoded)
        if successes / len(sample) < 0.8:
            continue

        records = [_decode_json_mapping(value) or {} for value in frame[column].tolist()]
        child, manifest = normalize_records(records)
        child.index = frame.index
        for child_column in child.columns:
            path = f"{column}.{child_column}"
            if path not in frame.columns:
                frame[path] = child[child_column]
        expanded.append(str(column))
        nested_paths.add(str(column))
        nested_paths.update(f"{column}.{p}" for p in manifest["nested_paths"])
        list_paths.update(f"{column}.{p}" for p in manifest["list_paths"])

    return frame, {
        "json_object_columns": sorted(expanded),
        "nested_paths": sorted(nested_paths),
        "list_paths": sorted(list_paths),
    }


#: What a directory of parts may hold. Hidden files and job markers
#: (``_SUCCESS``, ``.crc``) are never data.
_PART_SUFFIXES = {
    ".parquet", ".pq", ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xml", ".orc", ".avro",
}


def _part_files(directory: Path) -> list[Path]:
    """The data files under a directory, sorted, one format's worth or mixed."""
    parts = []
    for candidate in sorted(directory.rglob("*")):
        if not candidate.is_file() or candidate.suffix.lower() not in _PART_SUFFIXES:
            continue
        relative = candidate.relative_to(directory).parts
        if any(piece.startswith((".", "_")) for piece in relative):
            continue
        parts.append(candidate)
    return parts


def _read_file(path: Path, *, question: str | None = None,
               sheet: str | None = None) -> tuple[pd.DataFrame, str, str | None, dict[str, Any]]:
    """Read one tabular file by its suffix: ``(frame, fmt, delimiter, structure)``.
    ``question`` and ``sheet`` choose a workbook's sheet; other formats ignore them."""
    suffix = path.suffix.lower()
    delimiter: str | None = None
    structure: dict[str, Any] = {"normalization": "tabular_v1"}
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        fmt = "parquet"
    elif suffix in {".csv", ".tsv", ".txt"}:
        delimiter = "\t" if suffix == ".tsv" else _sniff_delimiter(path)
        frame = _read_csv(path, delimiter)
        fmt = "csv"
    elif suffix == ".json":
        (frame, structure), fmt = _read_json(path), "json"
    elif suffix in _JSON_LINES_SUFFIXES:
        (frame, structure), fmt = _read_json_lines(path), "jsonl"
    elif suffix == ".xml":
        frame, fmt = _read_xml(path), "xml"
    elif suffix in {".orc", ".avro"}:
        frame, fmt = (_read_orc(path), "orc") if suffix == ".orc" else (_read_avro(path), "avro")
        # A schema that types every column as a string ("67.0", "N/A") left
        # age and bmi as labels, so the driver ranking skipped them. CSV's
        # rule applies: a mostly-numeric string column is numeric, and its
        # stray words are missing values, counted.
        frame.attrs["coerced"] = _coerce_typed_columns(frame, _csv_text_columns(frame))
    elif suffix in {".xlsx", ".xlsm", ".xls"}:
        # Spreadsheets are how plans, trackers and hand-maintained data actually
        # arrive. Falling through to the CSV attempt below produced
        # "Unsupported or unreadable data source", which reads like the file is
        # broken rather than like the CLI cannot open it.
        frame, fmt = _read_excel(path, question=question, sheet=sheet), "excel"
    else:
        # Last-ditch attempt: try CSV, then parquet.
        try:
            delimiter = _sniff_delimiter(path)
            frame = _read_csv(path, delimiter)
            fmt = "csv"
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Unsupported or unreadable data source {str(path)!r}: {exc}. "
                "Supported formats: CSV, TSV, JSON, JSONL/NDJSON, XML, Parquet, ORC, Avro, "
                "Excel (.xlsx/.xlsm/.xls), PDF, common images/videos, LeRobot and RLDS."
            ) from exc
    _reject_unusable(frame, path, fmt)
    return frame, fmt, delimiter, structure


def _read_descriptor(frame: pd.DataFrame, fmt: str) -> dict[str, Any]:
    """What was read, and how -- taken off the frame so it does not travel into
    every copy of it (and into the Parquet metadata of the prepared data)."""
    if fmt == "csv":
        return {"kind": "csv", **frame.attrs.pop(_READ_ATTR)}
    if fmt == "excel":
        return {"kind": "excel", "sheet_name": frame.attrs.get("sheet_name"),
                "header_row": frame.attrs.get("header_row"),
                "na_values": frame.attrs.pop("na_values", None),
                "coerced": frame.attrs.pop("coerced", {}),
                "other_sheets": frame.attrs.pop("other_sheets", []),
                "sheet_choice": frame.attrs.pop("sheet_choice", None),
                "shape": frame.attrs.pop("shape", {}),
                "units": frame.attrs.pop("units", {})}
    if fmt in {"orc", "avro"}:
        return {"kind": fmt, "coerced": frame.attrs.pop("coerced", {})}
    return {"kind": fmt}


def _tree_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_dataset(
    source: str,
    *,
    complex_options: "ComplexDataOptions | None" = None,
    question: str | None = None,
    sheet: str | None = None,
) -> DatasetHandle:
    """Load a supported local source into a bounded, normalized dataset.

    ``question`` is what the user asked; for a workbook it chooses the sheet
    when it names something one sheet has and the others do not. ``sheet``
    names the sheet outright. Both are ignored for every other format.

    Existing tabular behavior is unchanged. PDF, image, video, LeRobot and
    RLDS paths are routed through optional complex-data adapters.

    A directory that is not a LeRobot/RLDS dataset is one tabular dataset
    split into parts -- what Spark, Hive and the object-store exporters write.
    Every part must be the same format; the parts are read in sorted order and
    stacked, and ``structure["source"]`` records bytes and rows per part so the
    optimizer sees an uneven split.
    """
    from avaloka.io.complex import (
        ComplexDataOptions,
        detect_complex_format,
        load_complex_data,
    )
    from avaloka.io.schema import infer_complex_schema, validate_complex_schema
    from avaloka.io.sources import materialize, parse_uri

    source_label = source
    scheme, _ = parse_uri(source)
    if scheme:
        options = complex_options or ComplexDataOptions()
        resolved = materialize(source, max_bytes=options.max_input_bytes)
        path = resolved.local_path
        source_label = resolved.uri
    else:
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Data source not found: {source!r}. Supported: local CSV, TSV, "
                "JSON, JSONL, XML, Parquet, ORC, Avro and Excel files, or a directory "
                "of part files in one of those formats, PDF, common images/videos, "
                "LeRobot and RLDS directories, and object-storage URIs."
            )

    delimiter: str | None = None
    complex_fmt = detect_complex_format(path)
    if complex_fmt is not None:
        complex_result = load_complex_data(path, options=complex_options)
        frame = complex_result.frame
        fmt = complex_result.fmt
        structure = complex_result.structure
        _reject_unusable(frame, path, fmt)
        read = {"kind": fmt}
        partitions = [{"bytes": _tree_bytes(path), "rows": int(len(frame))}]
    elif path.is_dir():
        parts = _part_files(path)
        if not parts:
            raise UnusableDataset(
                f"{path.name} is a directory with no data files in it. A directory "
                "is read as one dataset split into parts (.parquet, .csv, .json, "
                ".jsonl, .xml, .orc or .avro); this one has none.")
        pieces = [_read_file(part) for part in parts]
        formats = sorted({fmt for _, fmt, _, _ in pieces})
        if len(formats) > 1:
            raise UnusableDataset(
                f"{path.name} mixes formats ({', '.join(formats)}); a directory of "
                "parts must be one format. Point at the subfolder you mean.")
        fmt = formats[0]
        delimiter = pieces[0][2]
        structure = dict(pieces[0][3])
        reads = [_read_descriptor(frame, fmt) for frame, _, _, _ in pieces]
        frame = pd.concat([piece[0] for piece in pieces], ignore_index=True)
        # pandas alone cannot repeat a stacked read, so the generated script
        # declines it by name (see fireflies/data_engineer._load_src).
        read = {"kind": "directory", "part_kind": fmt, "parts": len(parts),
                "first_part": reads[0]}
        partitions = [{"bytes": part.stat().st_size, "rows": int(len(piece[0]))}
                      for part, piece in zip(parts, pieces)]
    else:
        frame, fmt, delimiter, structure = _read_file(path, question=question, sheet=sheet)
        read = _read_descriptor(frame, fmt)
        partitions = [{"bytes": path.stat().st_size, "rows": int(len(frame))}]

    # Normalise obvious all-empty / unnamed columns from dirty exports.
    frame = frame.dropna(axis=1, how="all")
    frame.columns = [str(column) for column in frame.columns]
    if fmt in {"csv", "parquet"}:
        # A column that consistently holds JSON objects is expanded into dotted
        # columns beside the original, which is kept for lineage.
        frame, expansion = _expand_json_object_columns(frame)
        structure.update(expansion)
    # The same manifest for every format: which columns hold text (the
    # text-versus-category decision downstream starts here) and each dtype.
    # JSON readers set both already; the others used to leave them out, so a
    # Parquet and an ORC file of the same table described themselves
    # differently.
    structure.setdefault("text_columns", [
        str(column) for column in frame.columns
        if any(isinstance(v, str) for v in frame[column].dropna().head(100).tolist())
    ])
    structure.setdefault("columns", {
        str(column): str(frame[column].dtype) for column in frame.columns
    })
    # Bytes and rows per part, for the statistics-aware optimizer. Counts only;
    # no part names, because this manifest is projected toward telemetry.
    structure["source"] = {
        "format": fmt,
        "n_files": len(partitions),
        "total_bytes": sum(part["bytes"] for part in partitions),
        "partitions": partitions,
    }
    if "complex_schema" not in structure:
        contract = infer_complex_schema(frame)
        structure["complex_schema"] = contract
        structure["schema_validation"] = validate_complex_schema(
            frame, contract
        ).to_dict()

    if fmt == "excel":
        structure["sheets"] = {"analysed": read.get("sheet_name"),
                               "others": read.get("other_sheets", []),
                               "chosen_by": read.get("sheet_choice")}
        if read.get("shape"):
            structure["shape"] = read["shape"]
        if read.get("units"):
            structure["units"] = read["units"]
    if read.get("coerced"):
        structure["coerced_columns"] = read["coerced"]

    return DatasetHandle(
        frame=frame,
        source=source_label,
        fmt=fmt,
        sha256=_sha256_path(path),
        n_rows=int(frame.shape[0]),
        n_cols=int(frame.shape[1]),
        delimiter=delimiter,
        read=read,
        structure=structure,
    )
