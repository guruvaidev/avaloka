"""Null-like words in a CSV: labels in a text column, missing in a typed one.

``payment_method = None`` is a category and ``amount = NA`` is a gap. The loader
used to read both as missing, so the category left every group-by. The obvious
repair, ``keep_default_na=False``, keeps the category and turns ``amount`` into
text; these tests fail for either mistake.
"""

import pandas as pd
import pandas.testing as pdt
import pytest

from avaloka.io import loader
from avaloka.io.loader import load_dataset
from avaloka.stats import infer_role
from tests.shared_cases.null_word_labels import CASE_IDS, CASES, outcome


def _csv(tmp_path, text, name="data.csv", encoding="utf-8"):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return str(path)



# --- the table shared with the platform's implementation --------------------
# tests/shared_cases/null_word_labels.py is the one list of cases. The platform
# reader is held to the same list, and the two are compared with each other, in
# tests/test_cli_null_policy_parity.py.
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_cli_reader_returns_the_expected_outcome(tmp_path, case):
    path = tmp_path / "case.csv"
    path.write_text(case.csv)
    assert outcome(loader._read_csv(path, ",")) == case.expected


def test_a_text_column_is_detected_as_text_on_the_installed_pandas(tmp_path):
    """The whole policy hangs on one fact about pandas: which dtype a default
    read gives a text column. pandas 2 says ``object``; pandas 3 says its own
    ``str`` dtype, and a check written for 2 found no text column on 3, so the
    loader went back to losing the labels WITHOUT ANY ERROR. This is the test
    that notices: on a pandas with yet another text dtype it fails instead of
    letting the fix turn itself off. (file_handler/null_policy.py depends on
    the same fact.)"""
    path = tmp_path / "case.csv"
    path.write_text("payment_method,amount\ncard,1.5\nNone,NA\ncash,2.5\n")
    head = pd.read_csv(path)
    assert loader._is_text_dtype(head["payment_method"].dtype), (
        f"pandas {pd.__version__} reads a text column as {head['payment_method'].dtype!r}: "
        "avaloka/io/loader.py's _is_text_dtype must learn that dtype")
    assert not loader._is_text_dtype(head["amount"].dtype)
    assert loader._csv_text_columns(head) == ["payment_method"]


def test_reading_the_shared_table_loads_nothing_from_the_platform():
    """The table is imported by the CLI's tests, so it must not drag the
    platform in: importing the CLI and the table loads no app or file_handler
    module. Run in a fresh interpreter, since this process may hold both."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys, avaloka.cli, avaloka.io.loader, tests.shared_cases.null_word_labels\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('app', 'file_handler'))\n"
        "print(bad)\n")
    root = Path(__file__).resolve().parents[2]            # so `tests` resolves from any cwd
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"

PAYMENTS = (
    "payment_method,amount,paid,order_date\n"
    "card,10.5,True,2024-01-01\n"
    "None,NA,False,2024-01-02\n"
    "NA,12.25,NA,NA\n"
    "N/A,13.0,True,2024-01-04\n"
    "cash,NA,False,2024-01-05\n"
    "null,15.5,True,2024-01-06\n"
    ",16.0,False,2024-01-07\n"
)


def test_null_words_in_a_text_column_are_labels(tmp_path):
    frame = load_dataset(_csv(tmp_path, PAYMENTS)).frame
    assert frame["payment_method"].tolist()[:6] == ["card", "None", "NA", "N/A", "cash", "null"]
    assert frame["payment_method"].value_counts().to_dict() == {
        "card": 1, "None": 1, "NA": 1, "N/A": 1, "cash": 1, "null": 1}


def test_an_empty_cell_in_a_text_column_is_still_missing(tmp_path):
    frame = load_dataset(_csv(tmp_path, PAYMENTS)).frame
    assert int(frame["payment_method"].isna().sum()) == 1
    assert pd.isna(frame["payment_method"].iloc[6])


def test_a_numeric_column_keeps_its_type_and_its_gaps(tmp_path):
    """The check keep_default_na=False fails: `amount` would load as text."""
    frame = load_dataset(_csv(tmp_path, PAYMENTS)).frame
    assert frame["amount"].dtype == "float64"
    assert int(frame["amount"].isna().sum()) == 2
    assert frame["amount"].sum() == pytest.approx(10.5 + 12.25 + 13.0 + 15.5 + 16.0)
    assert infer_role(frame["amount"], len(frame)) == "numeric"


def test_a_boolean_column_keeps_its_gap(tmp_path):
    frame = load_dataset(_csv(tmp_path, PAYMENTS)).frame
    assert int(frame["paid"].isna().sum()) == 1
    assert set(frame["paid"].dropna()) == {True, False}


def test_a_date_column_keeps_its_gap_and_is_still_read_as_dates(tmp_path):
    """Dates arrive as strings, so "did pandas type it?" would call this text.

    Keeping its NA cells as labels costs the column its role once enough of
    them are present, and with the role goes the temporal split.
    """
    rows = "".join(
        f"{'NA' if i % 10 == 0 else f'2024-01-{i % 28 + 1:02d}'},{'None' if i % 3 else 'card'}\n"
        for i in range(100))
    frame = load_dataset(_csv(tmp_path, "order_date,method\n" + rows)).frame
    assert int(frame["order_date"].isna().sum()) == 10
    assert infer_role(frame["order_date"], len(frame)) == "datetime"
    assert int((frame["method"] == "None").sum()) == 66      # the text column beside it


def test_a_column_of_only_null_words_stays_missing(tmp_path):
    """Nothing in it says the words are labels, so it is all-missing as before
    (and an all-missing column is dropped, as before)."""
    frame = load_dataset(_csv(tmp_path, "name,note\nann,NA\nbob,None\ncy,N/A\n")).frame
    assert list(frame.columns) == ["name"]


def test_a_file_with_no_text_labels_reads_exactly_as_pandas_does(tmp_path):
    path = _csv(tmp_path, "a,b,c\n1,NA,2024-01-01\n2,3.5,NA\nNA,4.5,2024-01-03\n")
    pdt.assert_frame_equal(load_dataset(path).frame, pd.read_csv(path))


def test_genuine_gaps_written_as_a_word_in_a_text_column_become_a_label(tmp_path):
    """The cost of the rule, pinned so it is a decision and not a surprise.

    A text column cannot say whether its "NA" meant "no answer" or was a
    value, so the rows are kept and shown: a visible category can be filtered
    out, rows that disappeared cannot be recovered. Only an empty cell is a
    gap in a text column.
    """
    frame = load_dataset(_csv(tmp_path, "city,visits\nleeds,3\nNA,4\nyork,5\n,6\n")).frame
    assert frame["city"].tolist()[:3] == ["leeds", "NA", "york"]
    assert int(frame["city"].isna().sum()) == 1


@pytest.mark.parametrize("delimiter", [";", "\t", "|"])
def test_labels_survive_other_delimiters(tmp_path, delimiter):
    name = "data.tsv" if delimiter == "\t" else "data.csv"
    text = PAYMENTS.replace(",", delimiter)
    frame = load_dataset(_csv(tmp_path, text, name=name)).frame
    assert {"None", "NA", "N/A", "null"} <= set(frame["payment_method"].dropna())
    assert frame["amount"].dtype == "float64"


def test_labels_survive_a_latin_1_export(tmp_path):
    text = "méthode,amount\ncarte,1.5\nNone,NA\nespèces,2.5\n"
    frame = load_dataset(_csv(tmp_path, text, encoding="latin-1")).frame
    assert frame["méthode"].tolist() == ["carte", "None", "espèces"]
    assert int(frame["amount"].isna().sum()) == 1


def test_known_limit_text_that_starts_after_the_sniff_is_read_as_before(tmp_path, monkeypatch):
    """A column that looks typed in the sniffed rows is typed: no worse than
    before the policy, and not fixed by it."""
    monkeypatch.setattr(loader, "CSV_SNIFF_ROWS", 3)
    frame = load_dataset(_csv(tmp_path, "code\n1\n2\n3\nNone\nabc\n")).frame
    assert int(frame["code"].isna().sum()) == 1


def test_word_list_is_the_one_pandas_uses():
    """A literal, so a pandas upgrade cannot move the policy unnoticed."""
    from pandas._libs.parsers import STR_NA_VALUES

    assert loader.NULL_WORDS == set(STR_NA_VALUES) - {""}


def test_the_analysis_answers_with_every_category(tmp_path):
    """End to end: the mission's grouped answer has the null-word categories."""
    import json

    from avaloka.benchmark import datasets as ds
    from avaloka.mission.budget import Budget
    from avaloka.mission.context import MissionContext, MissionKind
    from avaloka.missions import run_analyze

    source = ds.build_null_labelled_payments(tmp_path / "payments.csv")
    out = tmp_path / "out"
    run_analyze(MissionContext(
        kind=MissionKind.ANALYZE, goal="Total revenue by payment_method",
        data_source=str(source), output_dir=out, budget=Budget(limit_usd=None)))
    groups = {g["group"]: g["value"]
              for g in json.loads((out / "goal_analysis.json").read_text())["metrics"]["groups"]}
    assert groups == {label: count * 100.0
                      for label, count in ds.NULL_LABELLED_PAYMENT_COUNTS.items()}
    profile = {c["name"]: c for c in json.loads((out / "data_quality.json").read_text())["columns"]}
    assert profile["payment_method"]["missing"] == 0
    assert profile["payment_method"]["n_unique"] == 8
    assert profile["amount"]["role"] == "numeric" and profile["amount"]["missing"] == 3


# --- spreadsheets: the same rule, through the Excel reader --------------------
def _workbook(tmp_path, rows, title="Q3 revenue"):
    """A hand-made sheet: a title row and a blank row above the header, as the
    header guard's sheets have, so the reader also has to find the header."""
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append([title])
    sheet.append([])
    for row in rows:
        sheet.append(row)
    path = tmp_path / "data.xlsx"
    book.save(path)
    return str(path)


def test_a_spreadsheet_keeps_a_null_word_category_and_types_its_gaps(tmp_path):
    path = _workbook(tmp_path, [
        ["region", "revenue"],
        ["east", 100.0], ["None", 100.0], ["NA", "NA"], ["west", None],
    ])
    handle = load_dataset(path)
    assert handle.frame["region"].tolist() == ["east", "None", "NA", "west"]
    assert handle.frame["region"].notna().all()
    assert pd.api.types.is_float_dtype(handle.frame["revenue"])
    assert handle.frame["revenue"].isna().tolist() == [False, False, True, True]
    assert handle.read["na_values"]["region"] == [""]
    assert handle.read["header_row"] == 3


def test_a_spreadsheet_with_no_text_column_is_read_as_pandas_reads_it(tmp_path):
    path = _workbook(tmp_path, [["a", "b"], [1, 2.5], ["NA", 3.5], [4, 4.5]])
    handle = load_dataset(path)
    assert handle.read["na_values"] is None
    assert handle.frame["a"].isna().tolist() == [False, True, False]
