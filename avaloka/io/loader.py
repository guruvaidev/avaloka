"""Dataset ingestion for the narrow initial market: tabular files.

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
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import pandas as pd

from avaloka.stats import _looks_like_dates


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

    Picks the sheet with the most **non-empty cells** -- not the most rows; the
    docstring said rows for a while and the two are not the same ranking. The
    upload path (``file_handler/excel_connector.py::_resolve_sheet``) does rank
    by rows, so on a workbook where the longer sheet is the sparser one the two
    readers resolve *different sheets* from the same file. Anyone reading the
    old wording would have concluded they agree.

    Then takes the first row where a majority of cells are non-empty strings and
    the values below are populated.

    Neither heuristic asks what the question was about, which is a known
    limitation rather than a subtlety: see ``GroundTruth.sheet_name`` and
    ``comprehension.score_sheet``, which grade the choice, and
    ``docs/design/understanding-agent.md``.
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


def _read_json(path: Path) -> pd.DataFrame:
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

    return pd.json_normalize(records, sep=NESTED_SEPARATOR)


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

#: How many rows are read to tell text columns from typed ones. A column that
#: looks typed in these rows and holds real text further down is treated as
#: typed, so its null-like words are lost as they were before.
CSV_SNIFF_ROWS = 20000


def _csv_text_columns(head: pd.DataFrame) -> list:
    """The columns of a default-read sample whose null-like words are labels."""
    text = []
    for position in range(len(head.columns)):
        series = head.iloc[:, position]
        if series.dtype != object:
            continue                      # pandas typed it: numbers, booleans
        present = series.dropna()
        if not any(isinstance(cell, str) and cell.strip() for cell in present.tolist()):
            continue                      # no real string: nothing says these are labels
        if _looks_like_dates(present):
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
        return frame
    every_word = sorted(NULL_WORDS | {""})
    missing = {column: ([""] if column in text else every_word) for column in head.columns}
    frame = pd.read_csv(path, keep_default_na=False, na_values=missing, **kwargs)
    frame.attrs[_READ_ATTR] = {"keep_default_na": False, "na_values": missing}
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


def load_dataset(source: str) -> DatasetHandle:
    """Load a tabular dataset from a local CSV or Parquet file."""
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Data source not found: {source!r}. Supported: local CSV, TSV, "
            "JSON, XML, Parquet, ORC, Avro and Excel files; database and "
            "object-storage connectors are planned."
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
    elif suffix == ".json":
        frame, fmt = _read_json(path), "json"
    elif suffix == ".xml":
        frame, fmt = _read_xml(path), "xml"
    elif suffix == ".orc":
        frame, fmt = _read_orc(path), "orc"
    elif suffix == ".avro":
        frame, fmt = _read_avro(path), "avro"
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
                "Supported formats: CSV, TSV, JSON, XML, Parquet, ORC, Avro, "
                "Excel (.xlsx/.xlsm/.xls)."
            ) from exc

    _reject_unusable(frame, path, fmt)
    # What was read, and how. Taken off the frame so it does not travel into
    # every copy of it (and into the Parquet metadata of the prepared data).
    if fmt == "csv":
        read = {"kind": "csv", **frame.attrs.pop(_READ_ATTR)}
    elif fmt == "excel":
        read = {"kind": "excel", "sheet_name": frame.attrs.get("sheet_name"),
                "header_row": frame.attrs.get("header_row")}
    else:
        read = {"kind": fmt}
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
        read=read,
    )
