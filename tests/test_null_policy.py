"""The null-word load policy (file_handler/null_policy.py) and its first user,
the Excel connector.

A cell that says ``None`` or ``N/A`` is a label in a text column and a gap in a
typed one. Three things are pinned here, in order of how badly they fail:

  1. Table detection is unchanged. A sheet read for the policy must look, to
     header and table detection, exactly as it did before -- otherwise a filler
     row of ``N/A`` stops being blank and a table boundary moves.
  2. Numeric, boolean and date typing is unchanged.
  3. Text columns keep their null-like labels, through to the converted CSV.

Workbooks are written with pandas/openpyxl into tmp_path; no fixtures on disk.
"""

from __future__ import annotations

import csv

import pandas as pd
import pandas.testing as pdt
import pytest

from file_handler import null_policy
from file_handler.excel_connector import (
    ExcelConnector,
    _read_sheet,
    export_workbook_sheets_to_csv,
)
from file_handler.table_shape import extract_tables, keep_null_word_labels

WORDS = ["None", "null", "nan", "NaN", "NA", "N/A", "n/a", "NULL"]


def _workbook(tmp_path, rows, name="book.xlsx", sheet="data"):
    path = tmp_path / name
    pd.DataFrame(rows).to_excel(path, index=False, header=False, sheet_name=sheet)
    return str(path)


# ---------------------------------------------------------------------------
# The word list and the one decision
# ---------------------------------------------------------------------------


def test_the_word_list_is_exactly_what_pandas_reads_as_missing():
    """NULL_WORDS is a literal so a pandas upgrade cannot change the policy
    unnoticed. If this fails, pandas changed its list: decide, then update."""
    from pandas._libs.parsers import STR_NA_VALUES

    assert null_policy.NULL_WORDS == set(STR_NA_VALUES) - {""}


@pytest.mark.parametrize("value,expected", [
    ("None", True), ("N/A", True), ("nan", True), ("#N/A", True),
    ("NONE", False), (" NA ", False), ("", False), ("Card", False),
    (None, False), (float("nan"), False), (0, False),
])
def test_a_null_word_is_an_exact_string_match(value, expected):
    assert null_policy.is_null_word(value) is expected


@pytest.mark.parametrize("cells,reader_typed,expected", [
    (["Card", "Cash"], False, True),      # text column: labels stay
    (["Card", "Cash"], True, False),      # the reader typed it: gaps stay gaps
    ([1.0, 2.0], False, False),           # no real string anywhere
    ([True, False], False, False),        # booleans are typed, not text
    ([], False, False),                   # only null-like words: nothing says "label"
    (["  "], False, False),               # whitespace is not a real string
    ([1, "abc"], False, True),            # mixed, but it holds real text
])
def test_the_decision(cells, reader_typed, expected):
    assert null_policy.keeps_null_words(cells, reader_typed) is expected


# ---------------------------------------------------------------------------
# 1. Table detection sees what it always saw
# ---------------------------------------------------------------------------

SHEETS = {
    "plain table with labels in a text column": [
        ["region", "method", "amount"], ["East", "Card", 10], ["West", "N/A", 20], ["North", "None", 30]],
    "a column of N/A between two tables": [
        ["region", "amount", "N/A", "team", "units"], ["East", 10, "N/A", "A", 1],
        ["West", 20, "N/A", "B", 2], ["North", 30, "N/A", "C", 3]],
    "a row of N/A inside a table": [
        ["region", "amount"], ["East", 10], ["West", 20], ["N/A", "N/A"], ["team", "units"], ["A", 1], ["B", 2]],
    "title, a filler row of N/A, then the header": [
        ["Quarterly report", None, None], ["N/A", "N/A", "N/A"], ["region", "method", "amount"],
        ["East", "Card", 10], ["West", "Cash", 20]],
    "N/A in the header row": [
        ["region", "N/A", "amount"], ["East", "x", 10], ["West", "y", 20], ["North", "z", 30]],
    "numeric column with NA cells": [
        ["region", "amount"], ["East", 10], ["West", "NA"], ["North", 30], ["South", "n/a"]],
    "currency text with N/A": [
        ["region", "price"], ["East", "$1,200"], ["West", "N/A"], ["North", "$300"]],
    "rows of NA above the table": [
        ["NA", "NA"], ["NA", "NA"], ["region", "amount"], ["East", 10], ["West", 20]],
    "two stacked tables split by blank rows": [
        ["region", "amount"], ["East", 10], ["West", 20], [None, None], [None, None],
        ["team", "note"], ["A", "None"], ["B", "ok"]],
}


@pytest.mark.parametrize("name", sorted(SHEETS))
@pytest.mark.parametrize("header", [None, 0])
def test_the_blanked_read_is_the_read_pandas_gave_before(tmp_path, name, header):
    """Everything about structure rests on this: what table detection is handed
    is cell-for-cell and dtype-for-dtype the old default read."""
    path = _workbook(tmp_path, SHEETS[name])

    blanked, _words = _read_sheet(path, "data", header=header)

    pdt.assert_frame_equal(blanked, pd.read_excel(path, sheet_name="data", header=header))


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_no_table_boundary_or_header_moves(tmp_path, name):
    path = _workbook(tmp_path, SHEETS[name])
    before = extract_tables(pd.read_excel(path, sheet_name="data", header=None))

    after = ExcelConnector(path).load_all_tables(normalize=False)

    assert [(list(t["dataframe"].columns), t["dataframe"].shape, t["shape_report"]) for t in after] == [
        (list(frame.columns), frame.shape, report) for frame, report in before
    ]


def test_a_naive_uncoerced_read_would_have_moved_them(tmp_path):
    """Why _read_sheet returns a blanked frame at all. If this stops holding,
    table detection no longer depends on those cells being blank and the
    blanking can go."""
    path = _workbook(tmp_path, SHEETS["title, a filler row of N/A, then the header"])
    naive = pd.read_excel(path, sheet_name="data", header=None, keep_default_na=False, na_values=[""])

    (frame, report), = extract_tables(naive)

    assert report["header_row"] is None and list(frame.columns) == ["column_1", "column_2", "column_3"]


# ---------------------------------------------------------------------------
# 2. Typed columns keep their typing and their gaps
# ---------------------------------------------------------------------------


def _single(tmp_path, rows):
    return ExcelConnector(_workbook(tmp_path, rows)).load_dataframe()


@pytest.mark.parametrize("word", WORDS)
def test_a_null_word_in_a_numeric_column_is_missing(tmp_path, word):
    df = _single(tmp_path, [["region", "amount"], ["East", 10], ["West", word], ["North", 30]])

    assert pd.api.types.is_numeric_dtype(df["amount"])
    assert df["amount"].isna().tolist() == [False, True, False]
    assert df["amount"].sum() == 40


def test_a_currency_column_written_as_text_is_still_numeric(tmp_path):
    df = _single(tmp_path, [["region", "price"], ["East", "$1,200"], ["West", "N/A"], ["North", "$300"]])

    assert pd.api.types.is_numeric_dtype(df["price"])
    assert df["price"].tolist()[0] == 1200 and pd.isna(df["price"].tolist()[1])


def test_a_null_word_in_a_boolean_column_is_missing(tmp_path):
    """The connector's normalisation turns a boolean column with a gap into
    1.0 / NaN / 0.0, as it did before this policy. What matters here is that
    the gap is still a gap and no 'NA' label appears in a typed column."""
    df = _single(tmp_path, [["region", "active"], ["East", True], ["West", "NA"], ["North", False]])

    values = df["active"].tolist()
    assert values[0] == 1.0 and values[2] == 0.0 and pd.isna(values[1])
    assert pd.api.types.is_numeric_dtype(df["active"])


def test_a_null_word_in_a_date_column_is_missing(tmp_path):
    df = _single(tmp_path, [["region", "day"], ["East", pd.Timestamp("2024-01-05")],
                            ["West", "N/A"], ["North", pd.Timestamp("2024-03-01")]])

    assert pd.api.types.is_datetime64_any_dtype(df["day"])
    assert df["day"].isna().tolist() == [False, True, False]


def test_a_column_of_nothing_but_null_words_stays_missing(tmp_path):
    """Nothing in the column says they are labels, so it is what it was before:
    a named column with no values."""
    df = _single(tmp_path, [["region", "note", "amount"], ["East", "N/A", 10], ["West", "None", 20]])

    assert list(df.columns) == ["region", "note", "amount"]
    assert df["note"].isna().all()


# ---------------------------------------------------------------------------
# 3. Text columns keep their labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("word", WORDS)
def test_a_null_word_in_a_text_column_is_a_label(tmp_path, word):
    df = _single(tmp_path, [["method", "amount"], ["Card", 10], [word, 20], ["Cash", 30]])

    assert df["method"].tolist() == ["Card", word, "Cash"]
    assert df.groupby("method")["amount"].sum().to_dict() == {"Card": 10, word: 20, "Cash": 30}


def test_an_empty_cell_in_a_text_column_is_still_missing(tmp_path):
    df = _single(tmp_path, [["method", "amount"], ["Card", 10], [None, 20], ["None", 30]])

    assert df["method"].tolist()[0] == "Card" and df["method"].tolist()[2] == "None"
    assert pd.isna(df["method"].tolist()[1])


def test_labels_land_on_their_own_cells_in_an_offset_table(tmp_path):
    """The table starts on row 3 and column B, has a blank row inside it, and
    sits next to a second table. A label must come back in the cell it left."""
    rows = [
        [None, "Sales by method", None, None, None, None],
        [None, None, None, None, None, None],
        [None, "method", "amount", None, "team", "lead"],
        [None, "Card", "NA", None, "A", "None"],
        [None, None, None, None, "B", "Ann"],
        [None, "None", 20, None, "C", "N/A"],
        [None, "N/A", 30, None, "D", "Bo"],
    ]
    tables = ExcelConnector(_workbook(tmp_path, rows)).load_all_tables()

    first, second = (t["dataframe"] for t in tables)
    assert first["method"].tolist() == ["Card", "None", "N/A"]
    assert first["amount"].isna().tolist() == [True, False, False]
    assert second["team"].tolist() == ["A", "B", "C", "D"]
    assert second["lead"].tolist() == ["None", "Ann", "N/A", "Bo"]


def test_a_row_of_nothing_but_null_words_is_still_dropped_as_blank(tmp_path):
    """Known limit, pinned so it is a decision and not a surprise. Table
    detection must keep seeing these cells as blank, and to it a row holding
    only null-like words IS a blank row -- so a record whose label is 'N/A' and
    whose every other cell is empty or a null-like word is dropped, as it was
    before. One real value anywhere in the row keeps it."""
    rows = [["method", "amount"], ["Card", 10], ["N/A", "NA"], ["None", None], ["Cash", 30]]

    df = _single(tmp_path, rows)

    assert df["method"].tolist() == ["Card", "Cash"]


def test_labels_reach_every_way_of_reading_the_sheet(tmp_path):
    path = _workbook(tmp_path, [["method", "amount"], ["Card", 10], ["None", 20], ["N/A", 30]])
    connector = ExcelConnector(path)

    assert [row[0] for row in connector.load_data()] == ["Card", "None", "N/A"]
    assert connector.load_dataframe(normalize=False)["method"].tolist() == ["Card", "None", "N/A"]
    (table,) = connector.load_all_tables()
    assert table["dataframe"]["method"].tolist() == ["Card", "None", "N/A"]


def test_labels_survive_into_the_converted_csv(tmp_path):
    """The loss this fixes was permanent: the CSV written here is the file every
    later step reads, and the label was already an empty cell in it.

    Read back with the csv module on purpose. pd.read_csv still drops these
    words until the CSV readers adopt the policy (null_policy KNOWN LIMITS)."""
    path = tmp_path / "book.xlsx"
    pd.DataFrame({
        "method": ["Card", "None", "N/A", None],
        "amount": [10, 20, "NA", 40],
    }).to_excel(path, index=False)

    (export,), _notes = export_workbook_sheets_to_csv(str(path), tmp_path / "out")

    with open(export.csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["method"] for r in rows] == ["Card", "None", "N/A", ""]
    assert [r["amount"] for r in rows] == ["10.0", "20.0", "", "40.0"]


def test_keep_null_word_labels_leaves_a_frame_with_no_words_alone():
    frame = pd.DataFrame({"a": ["x", None], "b": [1, 2]})
    words = pd.DataFrame({"a": [None, None], "b": [None, None]})

    assert keep_null_word_labels(frame, words) is frame


# ===========================================================================
# CSV: the same policy, decided before the read
# ===========================================================================

import ast
import inspect
import subprocess
import sys
import textwrap


def _csv(tmp_path, text, name="data.csv"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


SURVEY_CSV = "payment_method,amount\n" + "".join(
    f"{label},{i + 1}\n" for i, label in enumerate(WORDS + ["", "Card"])
)


def test_the_reported_case_a_category_named_none_survives_a_group_by(tmp_path):
    """The defect as first measured: of 55 only Card's 10 survived."""
    path = _csv(tmp_path, SURVEY_CSV)

    before = pd.read_csv(path).groupby("payment_method")["amount"].sum()
    after = null_policy.read_csv_best_effort(path).groupby("payment_method")["amount"].sum()

    assert before.to_dict() == {"Card": 10}
    assert set(after.index) == set(WORDS) | {"Card"}
    assert int(after.sum()) == 46          # 55 minus the 9 on the genuinely empty cell


@pytest.mark.parametrize("word", WORDS)
def test_csv_a_null_word_is_a_label_in_a_text_column_and_missing_in_a_numeric_one(tmp_path, word):
    path = _csv(tmp_path, f"method,amount\nCard,1\n{word},2\nCash,{word}\n")

    df = null_policy.read_csv_keeping_labels(path)

    assert df["method"].tolist() == ["Card", word, "Cash"]
    assert df["amount"].dtype == "float64"
    assert df["amount"].isna().tolist() == [False, False, True]


CSVS_THE_POLICY_MUST_NOT_CHANGE = {
    "no missing cells": "region,amount,flag\nEast,1,True\nWest,2,False\n",
    "gaps only in numeric columns": "region,amount,qty\nEast,NA,1\nWest,2,n/a\nNorth,,3\n",
    "an empty cell in a text column": "region,amount\nEast,1\n,2\nNorth,3\n",
    "booleans with NA": "region,active\nEast,True\nWest,NA\nNorth,False\n",
    "a column of only null-like words": "region,note\nEast,NA\nWest,None\n",
    "all numeric": "a,b\n1,2\nNA,4\n5,null\n",
}


@pytest.mark.parametrize("name", sorted(CSVS_THE_POLICY_MUST_NOT_CHANGE))
def test_csv_files_with_no_label_to_keep_read_exactly_as_before(tmp_path, name):
    """Values AND dtypes. This is what 'numeric typing cannot change' means."""
    path = _csv(tmp_path, CSVS_THE_POLICY_MUST_NOT_CHANGE[name])

    pdt.assert_frame_equal(null_policy.read_csv_keeping_labels(path), pd.read_csv(path))


def test_csv_typed_columns_next_to_a_labelled_one_are_typed_as_before(tmp_path):
    path = _csv(tmp_path, "method,amount,qty,active\nCard,1.5,1,True\nNone,NA,2,False\nN/A,3.5,n/a,True\n")

    policy, default = null_policy.read_csv_keeping_labels(path), pd.read_csv(path)

    assert policy["method"].tolist() == ["Card", "None", "N/A"]
    pdt.assert_frame_equal(policy.drop(columns="method"), default.drop(columns="method"))


@pytest.mark.parametrize("kwargs", [
    {"dtype": str}, {"keep_default_na": True}, {"na_values": ["Card"]},
    {"na_filter": False}, {"index_col": 0}, {"converters": {"amount": str}},
])
def test_csv_a_caller_controlling_the_read_gets_plain_pandas(tmp_path, kwargs):
    path = _csv(tmp_path, "method,amount\nCard,1\nNone,2\nCash,NA\n")

    pdt.assert_frame_equal(null_policy.read_csv_keeping_labels(path, **kwargs), pd.read_csv(path, **kwargs))


def test_csv_reader_kwargs_still_pass_through(tmp_path):
    path = _csv(tmp_path, "method;amount\nCard;1\nNone;2\n")

    df = null_policy.read_csv_best_effort(path, sep=";", nrows=2)

    assert df["method"].tolist() == ["Card", "None"] and df["amount"].tolist() == [1, 2]


def test_csv_duplicate_headers_are_each_decided_on_their_own(tmp_path):
    """pandas renames the second one ``method.1``; the per-column list of
    missing strings is keyed by those names, so each column gets its own."""
    path = _csv(tmp_path, "method,method,amount\nCard,None,1\nNone,Cash,NA\n")

    df = null_policy.read_csv_keeping_labels(path)

    assert df["method"].tolist() == ["Card", "None"]
    assert df["method.1"].tolist() == ["None", "Cash"]
    assert df["amount"].isna().tolist() == [False, True]


def test_csv_text_that_only_appears_after_the_sniff_keeps_the_old_behaviour(tmp_path, monkeypatch):
    """Known limit, pinned: looks typed in the sniffed rows, real text later."""
    monkeypatch.setattr(null_policy, "CSV_SNIFF_ROWS", 2)
    path = _csv(tmp_path, "code,amount\n1,1\nNA,2\nabc,3\n")

    pdt.assert_frame_equal(null_policy.read_csv_keeping_labels(path), pd.read_csv(path))


def test_csv_utf16_and_cp1252_files_still_open(tmp_path):
    utf16 = tmp_path / "u16.csv"
    utf16.write_bytes("method,amount\nCard,1\nNone,2\n".encode("utf-16"))
    cp1252 = tmp_path / "cp.csv"
    cp1252.write_bytes("method,amount\nCaf\xe9,1\nNone,2\n".encode("cp1252"))

    assert null_policy.read_csv_best_effort(str(utf16))["method"].tolist() == ["Card", "None"]
    assert null_policy.read_csv_best_effort(str(cp1252))["method"].tolist()[1] == "None"


# ---------------------------------------------------------------------------
# One reader, shipped as text: it must stand alone and it must be the only one
# ---------------------------------------------------------------------------


def test_the_policy_module_can_run_with_no_repository():
    """Its source is spliced into scripts that run where this repo may not
    exist. Standard library and pandas only; nothing relative; no __future__
    line, which would be a SyntaxError in the middle of a script."""
    tree = ast.parse(inspect.getsource(null_policy))
    allowed = set(sys.stdlib_module_names) | {"pandas"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            assert node.module != "__future__"
            assert (node.module or "").split(".")[0] in allowed, node.module


def test_execution_scripts_get_the_module_itself_not_a_copy():
    import app.agents.execution_agent as agent

    source = inspect.getsource(agent)

    assert agent.read_csv_best_effort is null_policy.read_csv_best_effort
    assert agent._CSV_READER_SOURCE == inspect.getsource(null_policy)
    assert "def read_csv_best_effort" not in source, "a hand-kept copy of the reader is back"
    # Two script templates, each with the marker on a line of its own.
    assert source.count("\n" + agent._CSV_READER_MARKER + "\n") == 2
    with pytest.raises(ValueError):
        agent._with_csv_reader("import pandas as pd\n")


def test_the_spliced_reader_works_in_a_script_that_cannot_import_this_repo(tmp_path):
    import app.agents.execution_agent as agent

    path = _csv(tmp_path, SURVEY_CSV)
    script = tmp_path / "standalone.py"
    script.write_text(textwrap.dedent('''
        import sys

        class _NoRepo:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("app", "file_handler", "avaloka"):
                    raise ImportError("no repository here: " + name)
        sys.meta_path.insert(0, _NoRepo())
        import pandas as pd
    ''') + agent._with_csv_reader(agent._CSV_READER_MARKER) + textwrap.dedent(f'''
        df = read_csv_best_effort({path!r})
        print("|".join(str(v) for v in df["payment_method"].tolist()))
    '''))

    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "|".join(WORDS + ["nan", "Card"])


def test_an_analysis_run_through_the_real_local_executor_keeps_the_category(tmp_path):
    """End to end through execute_code_on_local: the script it builds, the
    reader spliced into it, the rewrite of the literal path, a real subprocess."""
    import app.agents.execution_agent as agent

    source, output = _csv(tmp_path, SURVEY_CSV), str(tmp_path / "out.csv")
    code = textwrap.dedent('''
        import pandas as pd
        df = read_csv_best_effort("input.csv")
        df.groupby("payment_method", as_index=False)["amount"].sum().to_csv("output.csv", index=False)
    ''')

    agent.execute_code_on_local(code=code, input_data_location=source, output_location=output)

    with open(output, newline="") as handle:
        totals = {row["payment_method"]: row["amount"] for row in csv.DictReader(handle)}
    assert set(totals) == set(WORDS) | {"Card"}
    assert totals["None"] == "1" and totals["Card"] == "10"
