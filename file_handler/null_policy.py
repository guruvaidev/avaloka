"""One policy for cells that look like nulls: ``None``, ``null``, ``NA``, ``N/A``...

pandas turns a fixed list of words into missing values the moment it reads a
file. That is right for ``amount = NA`` and wrong for ``payment_method = None``,
which is a real category: the rows vanished from every group-by, from the Data
table and from the answer, with no trace.

THE RULE
    An empty cell is always missing.
    A null-like word is MISSING in a typed column and a LABEL in a text column.

    cell          text column        numeric / boolean / date column
    ------------  -----------------  -------------------------------
    None  null    label, as written  missing
    nan   NaN     label, as written  missing
    NA    N/A     label, as written  missing
    n/a   NULL    label, as written  missing
    (and the rest of NULL_WORDS)
    empty cell    missing            missing

    A column is TEXT when, with the null-like words set aside, the reader could
    not type it and it holds at least one real string. Otherwise it is typed --
    which includes a column made only of null-like words: nothing in it says
    they are labels, so it stays missing, exactly as before.

WHY LABELS, EVEN FOR ``N/A``
    A user can always filter out a category they can see. They can never
    recover rows that disappeared.

NUMERIC TYPING CANNOT CHANGE, AND MUST NOT BE "SIMPLIFIED" AWAY
    The obvious one-line fix -- stop pandas coercing, e.g.
    ``keep_default_na=False`` -- keeps the labels and breaks every numeric
    column that writes its gaps as ``NA``: it loads as text and arithmetic on it
    fails. Here a column only keeps its null-like words after the reader has
    tried to type it WITHOUT them and failed. A column that types as numbers,
    booleans or dates never reaches that branch, so its typing is untouched by
    construction rather than by care.

ONE DECISION, TWO MECHANISMS
    ``keeps_null_words`` below is the single decision and every reader must call
    it. Readers differ only in how they get to ask:

    * Excel (file_handler/excel_connector.py) has the whole sheet in memory. It
      reads without coercion, lets table detection see the cells blanked as it
      always has, types each finished column, and puts the words back where the
      decision says so. It decides AFTER reading, from the whole column.
    * CSV (read_csv_keeping_labels below) must tell pandas which strings are
      missing BEFORE the read, so it cannot try-and-see without reading the file
      twice (measured: +2 to +6 s on a 170 MB file, against no extra cost for
      this way). It reads the first CSV_SNIFF_ROWS rows with pandas' defaults,
      asks the decision about each column, and then does ONE full read with a
      per-column list of missing strings: only "" for a text column, the whole
      default list for every other column. A column is only called text when
      the sniff found a real string in it, and any column holding a real string
      is already text to pandas -- so a column pandas would type numerically can
      never be given the short list.

    Same word list, same decision, same table. Do not merge the mechanisms, and
    do not give either reader its own copy of the rule.

THIS FILE IS ALSO SHIPPED AS TEXT
    app/agents/execution_agent.py splices this module's source, verbatim, into
    the scripts it runs in the execution environment, where the repository may
    not exist. So this file must stay self-contained: standard library and
    pandas only, no relative imports, and no ``from __future__`` line (it would
    land in the middle of a script). tests/test_null_policy.py enforces all of
    that. There is deliberately no second copy of anything here to keep in step.

KNOWN LIMITS
    * Excel: a ROW holding nothing but null-like words and empty cells is still
      dropped. Table detection has to keep seeing those cells as blank, and to
      it that is a blank row. One real value anywhere in the row keeps it.
    * Only reads that go through read_csv_best_effort get the CSV policy: the
      analysis input, and ``pd.read_csv("literal.csv")`` in generated code, which
      is rewritten to it. Generated code may also call ``pd.read_csv(path)``
      with a VARIABLE path -- the coder prompt allows that for extra datasets --
      and no helper reaches that call. It keeps pandas' defaults whatever is
      done here. The other pandas CSV sites (upload sampler, sandbox service,
      multi-action, training loaders) have not adopted it either.
    * CSV: the sniff sees CSV_SNIFF_ROWS rows. A column that looks typed in
      those rows and holds real text further down is treated as typed, so its
      null-like words are lost exactly as they were before. No worse than
      before; not fixed.
    * CSV: a caller that passes its own ``dtype``, NA options, ``index_col``,
      ``converters`` or asks for chunks is controlling the read itself and gets
      pandas' plain behaviour, unchanged.
    * Daft and the stdlib ``csv`` reader already keep these words as text, but
      they also keep an EMPTY cell as an empty-string label where pandas makes
      it missing. That difference is not addressed here.
"""
from typing import Any, Iterable

import pandas as pd

#: The strings pandas reads as missing by default, minus the empty string. Kept
#: as a literal so the policy does not silently follow a pandas upgrade;
#: tests/test_null_policy.py fails if the two ever differ.
NULL_WORDS = frozenset({
    "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None",
    "n/a", "nan", "null",
})


def is_null_word(value: Any) -> bool:
    """True for a string pandas would have read as missing. Exact match, as
    pandas does it: ``NONE`` and `` NA `` are ordinary text."""
    return isinstance(value, str) and value in NULL_WORDS


def keeps_null_words(present_cells: Iterable[Any], reader_typed: bool) -> bool:
    """THE decision: do this column's null-like words stay as labels?

    ``present_cells`` are the column's cells with empty cells and null-like
    words already set aside. ``reader_typed`` is whether the reader managed to
    type what is left (numbers, booleans, dates).

    Labels are kept only in a text column: not typed, and holding at least one
    real string. Booleans read from a file arrive as ``True``/``False`` objects,
    not strings, so a boolean column counts as typed here even when pandas
    leaves its dtype as ``object``.
    """
    if reader_typed:
        return False
    return any(isinstance(cell, str) and cell.strip() for cell in present_cells)


def blank_null_words(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` as a default pandas read would have returned it: every
    null-like word replaced by a missing value, column types re-inferred."""
    return frame.mask(null_word_cells(frame).notna()).infer_objects()


def null_word_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Same shape as ``frame``: the null-like word where a cell holds one,
    missing everywhere else."""
    # Series.map per column rather than DataFrame.map: the latter needs pandas 2.1
    # and requirements.txt allows 2.0.
    holds_word = frame.apply(lambda column: column.map(is_null_word)).astype(bool)
    return frame.where(holds_word)


# ---------------------------------------------------------------------------
# CSV: decide before the read
# ---------------------------------------------------------------------------

#: How many rows the CSV reader looks at to tell text columns from typed ones.
CSV_SNIFF_ROWS = 20000

#: A caller passing any of these is deciding types or missing values itself.
_CALLER_CONTROLS_THE_READ = (
    "dtype", "keep_default_na", "na_values", "na_filter", "index_col",
    "converters", "chunksize", "iterator",
)


def csv_text_columns(head: pd.DataFrame) -> list:
    """The columns of a default-read sample whose null-like words are labels."""
    text = []
    for position in range(len(head.columns)):
        series = head.iloc[:, position]
        reader_typed = series.dtype != object
        if keeps_null_words(series.dropna().tolist(), reader_typed):
            text.append(head.columns[position])
    return text


def read_csv_keeping_labels(path: Any, **kwargs: Any) -> pd.DataFrame:
    """``pd.read_csv`` under the policy above. One full read; see the module
    docstring for why this is not a read followed by a repair."""
    if any(name in kwargs for name in _CALLER_CONTROLS_THE_READ):
        return pd.read_csv(path, **kwargs)
    sniff = dict(kwargs)
    asked = sniff.get("nrows")
    sniff["nrows"] = CSV_SNIFF_ROWS if asked is None else min(int(asked), CSV_SNIFF_ROWS)
    head = pd.read_csv(path, **sniff)
    text = csv_text_columns(head)
    if not text:
        # Nothing to keep: exactly the read pandas always did.
        return pd.read_csv(path, **kwargs)
    every_word = sorted(NULL_WORDS | {""})
    missing = {column: ([""] if column in text else every_word) for column in head.columns}
    return pd.read_csv(path, keep_default_na=False, na_values=missing, **kwargs)


def read_csv_best_effort(path: str, **kwargs: Any) -> pd.DataFrame:
    """Robust CSV reader for analysis code.

    - tries utf-16 if a BOM is detected, then utf-8-sig, utf-8, cp1252, latin-1
    - keeps null-like words as labels in text columns (read_csv_keeping_labels)
    - passes extra read_csv kwargs through (sep, delimiter, low_memory, ...)
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(4)
    except Exception:
        head = b""

    encodings = []
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        encodings.append("utf-16")
    encodings += ["utf-8-sig", "utf-8", "cp1252", "latin-1"]

    last_err = None
    for enc in encodings:
        try:
            return read_csv_keeping_labels(path, encoding=enc, encoding_errors="replace", **kwargs)
        except TypeError:
            try:
                return read_csv_keeping_labels(path, encoding=enc, **kwargs)
            except Exception as exc:
                last_err = exc
        except Exception as exc:
            last_err = exc

    raise last_err
