"""A time column written in several date forms is read cell by cell.

nullxlsx (2026-10-19 web re-run): training refused with "Time column
'visit_date' contains 48 missing or invalid value(s)". The sheet's 60 rows
hold visit_date in five export forms ("2022-01-01", "02/02/2023",
"03-03-22", "Apr 4, 2023", "20220505") and exactly 2 real gaps ("N/A").
``add_time_order_column`` parsed with one inferred format, so every date
written another way counted as invalid. It now parses each cell on its own
(``format="mixed"``), as the CLI loader does, and only a cell that is
missing or is no date at all counts.

The workbook is tests/fixtures/null_words/three_sheets.xlsx, the file the
re-run uploaded; make_fixtures.py holds its numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from app.agents.mta_v2.failure_diagnostics import build_training_failure
from app.agents.mta_v2.training_docker_image.src.evaluation import (
    TIME_ORDER_COLUMN,
    add_time_order_column,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "null_words"
sys.path.insert(0, str(FIXTURES))
import make_fixtures as fx  # noqa: E402

_SHEET_ROWS = 60
_GAPS = [row for row in fx.DATE_GAP_ROWS if row < _SHEET_ROWS]


@pytest.fixture
def sheet() -> pd.DataFrame:
    """The analysed sheet, as pandas reads it: "N/A" is already missing."""
    return pd.read_excel(FIXTURES / "three_sheets.xlsx", sheet_name="2022")


def test_the_fixture_is_the_shape_the_rerun_saw(sheet):
    assert len(sheet) == _SHEET_ROWS and len(_GAPS) == 2
    assert int(sheet["visit_date"].isna().sum()) == len(_GAPS)
    dates = sheet["visit_date"].dropna().astype(str)
    assert dates.str.contains(",").any() and dates.str.fullmatch(r"\d{8}").any()
    assert dates.str.contains("/").any() and dates.str.fullmatch(r"\d{2}-\d{2}-\d{2}").any()


def test_only_the_true_gaps_are_counted(sheet):
    with pytest.raises(ValueError, match=rf"contains {len(_GAPS)} missing or invalid value"):
        add_time_order_column(sheet, "visit_date")


def test_every_real_date_in_any_of_the_five_forms_is_ordered(sheet):
    dated = sheet.dropna(subset=["visit_date"]).reset_index(drop=True)
    ordered = add_time_order_column(dated, "visit_date")
    # The truth, from the generator: each row's own year, month and day.
    truth = {f"P{i:04d}": (2022 + i % 2, i % 12 + 1, i % 28 + 1) for i in range(_SHEET_ROWS)}
    keys = ordered.sort_values(TIME_ORDER_COLUMN, kind="mergesort")["patient_id"].tolist()
    assert [truth[k][0] for k in keys] == sorted(truth[k][0] for k in keys), \
        "years must come out in order"
    by_key = dict(zip(ordered["patient_id"], pd.to_datetime(ordered[TIME_ORDER_COLUMN], utc=True)))
    assert by_key["P0003"].year == 2023 and by_key["P0003"].month == 4 and by_key["P0003"].day == 4
    assert by_key["P0004"].year == 2022 and by_key["P0004"].month == 5 and by_key["P0004"].day == 5


def test_the_refusal_names_the_true_count_in_plain_words(sheet):
    with pytest.raises(ValueError) as caught:
        add_time_order_column(sheet, "visit_date")
    message = build_training_failure(str(caught.value))["user_message"]
    assert f"{len(_GAPS)} row(s) without a readable date" in message
    assert "`visit_date`" in message and "48" not in message


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    for var in ("MLFLOW_TRACKING_URI", "MLFLOW_BACKEND_STORE_URI",
                "MLFLOW_DEFAULT_ARTIFACT_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "mixed-dates")
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    mlflow.set_tracking_uri(None)


def test_training_on_the_sheet_without_its_gaps_completes(sheet, mlflow_store):
    """With the two gap rows removed -- what the new message tells the user to
    do -- time-ordered training runs on all 58 dated rows."""
    from app.agents.mta_v2.local_trainer import setup_mlflow, training_main

    setup_mlflow()
    data = mlflow_store / "nullxlsx.csv"
    sheet.dropna(subset=["visit_date"]).to_csv(data, index=False)
    result = training_main(
        data_uri=str(data), model_type="classification", target_col="Outcome",
        time_column="visit_date", feature_cols=["Glucose", "Notes"],
        algorithm="gradient_boosting", extra_hyperparameters={"max_iter": 20},
        build_model_card=False,
    )
    assert result.get("mlflow_run_id"), result
