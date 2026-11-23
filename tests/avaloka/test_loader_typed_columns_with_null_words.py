"""A few stray words do not turn a numeric or date column into categories.

release-test's D3 (2026-10-18, integration/2.0): a Glucose column that was 89%
numbers and the rest ``none``/``N/A``/``-`` came back from ``load_dataset`` as
text, the words became its top values, its missing rate read 0.0, mixed-format
dates were categorical with ``N/A`` as their top value, and ``avaloka
coordinate`` still said pass. Cause: the null-word rule (#499) decided "text
column" by asking only whether pandas typed the column, and pandas gives up
on a numeric column the moment one cell says ``none`` -- a word its default
list does not carry.

Second part: the Excel reader analyses one sheet and said nothing about the
others, so a workbook of two equal data sheets reported half its rows.

Every test here was run against the unfixed loader first and failed on the
symptom it names.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from avaloka.fireflies.data_scout import _loader_issues
from avaloka.io.loader import load_dataset
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import run_analyze
from avaloka.stats import _looks_like_dates, profile_column, quality_report

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "null_words"
sys.path.insert(0, str(FIXTURES))
import make_fixtures as fx  # noqa: E402  (the fixture's own numbers are the truth)

_RUN_SCRIPT = """
import runpy, sys
namespace = runpy.run_path(sys.argv[1], run_name="generated_script")
frame = namespace["load"]()
loaded = sorted(m for m in sys.modules if m == "avaloka" or m.startswith("avaloka."))
assert not loaded, f"the generated script imported avaloka: {loaded}"
frame.to_pickle(sys.argv[2])
"""


@pytest.fixture(scope="module")
def handle():
    return load_dataset(str(FIXTURES / "edge_cases.csv"))


def test_a_numeric_column_with_stray_words_is_numeric_and_the_words_are_gaps(handle):
    glucose = handle.frame["Glucose"]
    assert pd.api.types.is_float_dtype(glucose), f"Glucose read as {glucose.dtype}"
    assert int(glucose.isna().sum()) == len(fx.GLUCOSE_WORDS), "each stray word is one missing cell"
    assert glucose.isna()[list(fx.GLUCOSE_WORDS)].all()
    profile = profile_column(glucose, handle.n_rows)
    assert profile["role"] == "numeric" and profile["missing"] == 3
    assert "top_values" not in profile, "a numeric column has no top values"


def test_the_coercion_is_recorded_for_the_profile_and_the_script(handle):
    # pandas already read "N/A" as missing; "none" and "-" were the stray words.
    assert handle.structure["coerced_columns"]["Glucose"] == {
        "kind": "numeric", "cells": 2, "words": ["-", "none"]}
    assert handle.read["coerced"]["Glucose"]["cells"] == 2


def test_mixed_format_dates_parse_and_their_gaps_are_missing(handle):
    dates = handle.frame["visit_date"]
    assert int(dates.isna().sum()) == len(fx.DATE_GAP_ROWS)
    profile = profile_column(dates, handle.n_rows)
    assert profile["role"] == "datetime", f"visit_date read as {profile['role']}"
    # Every surviving cell is a real date, in whichever of the five forms.
    parsed = pd.to_datetime(dates.dropna(), errors="coerce", format="mixed")
    assert parsed.notna().all()


def test_a_text_column_still_keeps_its_words_as_labels(handle):
    """The first clause of the rule (#499) must survive the second."""
    notes = handle.frame["Notes"]
    assert int(notes.isna().sum()) == len(fx.NOTE_EMPTY_ROWS), "only empty cells are gaps"
    assert {"n/a", "none", "NULL", "-", "?"} <= set(notes.dropna())
    assert profile_column(notes, handle.n_rows)["role"] == "categorical"
    assert "Notes" not in handle.structure["coerced_columns"]


def test_the_scout_tells_the_user_what_was_read_as_missing(handle):
    issues = _loader_issues(handle)
    kinds = {issue["kind"]: issue for issue in issues}
    assert kinds["null_words_read_as_missing"]["column"] == "Glucose"
    assert "'none'" in kinds["null_words_read_as_missing"]["detail"]
    columns = [profile_column(handle.frame[c], handle.n_rows) for c in handle.frame.columns]
    quality = quality_report(handle.frame, columns, extra_issues=issues)
    assert any(i["kind"] == "null_words_read_as_missing" for i in quality["issues"])


def test_the_generated_script_repeats_the_coercion(tmp_path):
    out = tmp_path / "bundle"
    ctx = MissionContext(kind=MissionKind.ANALYZE, goal="Which patients are readmitted?",
                         data_source=str(FIXTURES / "edge_cases.csv"), output_dir=out,
                         budget=Budget(limit_usd=None))
    run_analyze(ctx)
    analysed = ctx.blackboard["frame"]
    script = (out / "analysis.py").read_text()
    assert "pd.to_numeric(df['Glucose']" in script, "the script must repeat the coercion"
    pickle = tmp_path / "loaded.pkl"
    done = subprocess.run([sys.executable, "-c", _RUN_SCRIPT, str(out / "analysis.py"), str(pickle)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-600:]
    pdt.assert_frame_equal(pd.read_pickle(pickle), analysed)


def test_the_date_hint_accepts_real_export_forms_but_not_versions_or_ids():
    five_forms = pd.Series([r["visit_date"] for r in fx.rows() if r["visit_date"] != "N/A"])
    assert _looks_like_dates(five_forms)
    assert not _looks_like_dates(pd.Series(["1.2.10", "1.3.4", "2.0.1", "1.2.11"] * 10))
    assert not _looks_like_dates(pd.Series([f"{10480000 + i:08d}" for i in range(40)]))
    assert not _looks_like_dates(pd.Series([f"P{i:04d}" for i in range(40)]))


# ---------------------------------------------------------------------------
# The workbook: one sheet analysed, the others reported
# ---------------------------------------------------------------------------
def test_a_workbook_reports_the_sheets_it_did_not_analyse():
    pytest.importorskip("openpyxl")
    handle = load_dataset(str(FIXTURES / "three_sheets.xlsx"))
    assert handle.n_rows == 60
    assert handle.read["sheet_name"] == "2022"
    assert handle.read["other_sheets"] == [{"name": "2023", "rows": 60}, {"name": "notes", "rows": 5}]
    assert handle.structure["sheets"] == {
        "analysed": "2022",
        "others": [{"name": "2023", "rows": 60}, {"name": "notes", "rows": 5}],
        # No question: the old rule chose, and the profile says which rule.
        "chosen_by": {"by": "size", "matched": []},
    }
    issue = next(i for i in _loader_issues(handle) if i["kind"] == "other_sheets_not_analysed")
    assert issue["severity"] == "medium"
    assert "2 other sheet(s) hold 65 rows" in issue["detail"] and "2023 (60 rows)" in issue["detail"]


def test_coordinate_records_the_analysed_sheet_in_its_provenance(tmp_path):
    pytest.importorskip("openpyxl")
    from avaloka.coordinator import MainAvaloka, load_coordination
    MainAvaloka().coordinate(str(FIXTURES / "three_sheets.xlsx"), goal="profile",
                             output_dir=tmp_path / "fleet", workers=1)
    report = load_coordination(tmp_path / "fleet")
    assert report["n_rows"] == 60
    assert report["source_resolution"]["sheet"] == {
        "analysed": "2022",
        "others": [{"name": "2023", "rows": 60}, {"name": "notes", "rows": 5}],
    }
    assert report["source_resolution"]["structure"]["coerced_columns"]["Glucose"]["kind"] == "numeric"
