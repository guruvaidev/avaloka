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
    * CSV must tell pandas which strings are missing BEFORE the read, so it
      cannot try-and-see without reading the file twice (measured: +2 to +6 s on
      a 170 MB file). It decides from the first rows instead. NOT YET ADOPTED --
      see KNOWN LIMITS.

    Same word list, same decision, same table. Do not merge the mechanisms, and
    do not give either reader its own copy of the rule.

KNOWN LIMITS
    * Excel: a ROW holding nothing but null-like words and empty cells is still
      dropped. Table detection has to keep seeing those cells as blank, and to
      it that is a blank row. One real value anywhere in the row keeps it.
    * The CSV readers do not use this yet. An Excel sheet converted to CSV now
      keeps ``None`` in the file, but ``pd.read_csv`` still drops it when that
      CSV is read back. Until the CSV readers adopt the policy the label
      survives in the converted file and not in the analysis.
    * Generated analysis code may call ``pd.read_csv(path)`` with a variable
      path. No helper reaches that call; it keeps pandas' defaults whatever is
      done here.
    * Daft and the stdlib ``csv`` reader already keep these words as text, but
      they also keep an EMPTY cell as an empty-string label where pandas makes
      it missing. That difference is not addressed here.
"""
from __future__ import annotations

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
