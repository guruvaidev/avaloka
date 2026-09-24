"""Contract tests for chronological validation in local and Ray MTA training."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage

from app.agents.mta_v2.local_trainer import LocalTrainer, split_train_val
from app.agents.mta_v2.ray_trainer import RayTrainer
from app.agents.mta_v2.training_docker_image.src.evaluation import (
    TIME_ORDER_COLUMN,
    add_time_order_column,
    build_holdout_evaluation_report,
    time_series_split_metadata,
)


# The dataset is built here rather than read from
# tests/fixtures/datasets/synthetic_retail_time_series.csv. That directory is a
# download cache for tests/k8s/test_t7_agent_matrix.py and is gitignored, so the
# file was never in the repository and this test could not pass on any clean
# checkout -- it failed with FileNotFoundError on a path nothing creates.
#
# Generating it is also the better fixture: the data is synthetic (the filename
# says so), the assertions only need a strictly increasing date and a varying
# target, and a committed CSV would be one more thing to keep in step with them.
WEEKS = 60


@pytest.fixture
def retail_time_series() -> pd.DataFrame:
    """60 weekly retail observations, deliberately not in chronological order.

    Shuffled so that a test asserting "train is strictly earlier than
    validation" is checking the sort, not the order the rows happened to
    arrive in.
    """
    dates = pd.date_range("2024-01-07", periods=WEEKS, freq="W-SUN")
    frame = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Store": [1 + (i % 3) for i in range(WEEKS)],
            "Dept": [10 + (i % 5) for i in range(WEEKS)],
            "Temperature": [40.0 + (i % 25) for i in range(WEEKS)],
            # A trend plus a repeating seasonal wobble: strictly more than one
            # distinct value, without needing a random seed to stay stable.
            "Weekly_Sales": [
                20000.0 + (i * 125.0) + ((i % 7) * 310.0) for i in range(WEEKS)
            ],
        }
    )
    return frame.sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_time_series_fixture_is_orderable_and_has_a_future_holdout(retail_time_series):
    frame = retail_time_series
    ordered = add_time_order_column(frame, "Date").sort_values(TIME_ORDER_COLUMN)
    metadata = time_series_split_metadata(len(ordered), "Date")

    train_rows = metadata["n_train_rows"]
    train = ordered.iloc[:train_rows]
    validation = ordered.iloc[train_rows:]

    assert len(frame) == WEEKS
    assert frame["Weekly_Sales"].nunique() > 1
    assert train["Date"].max() < validation["Date"].min()


def test_time_values_are_parsed_and_can_be_sorted_chronologically():
    frame = pd.DataFrame(
        {
            "Date": ["2026-03-01", "2026-01-01", "2026-02-01"],
            "sales": [30.0, 10.0, 20.0],
        }
    )

    ordered = add_time_order_column(frame, "Date").sort_values(TIME_ORDER_COLUMN)

    assert ordered["sales"].tolist() == [10.0, 20.0, 30.0]


@pytest.mark.parametrize("bad_value", [None, "not-a-date"])
def test_invalid_time_values_are_rejected(bad_value):
    frame = pd.DataFrame({"Date": ["2026-01-01", bad_value]})

    with pytest.raises(ValueError, match="missing or invalid"):
        add_time_order_column(frame, "Date")


def test_local_time_series_split_uses_earlier_rows_for_training():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=12, freq="D"),
            "sales": range(12),
        }
    )

    train, validation = split_train_val(frame, time_ordered=True, n_splits=5)

    assert len(train) == 10
    assert len(validation) == 2
    assert train["Date"].max() < validation["Date"].min()


def test_time_series_metadata_describes_final_expanding_window_fold():
    metadata = time_series_split_metadata(12, "Date", n_splits=5)

    assert metadata == {
        "splitter": "TimeSeriesSplit",
        "splitter_reason": (
            "rows are ordered by 'Date'; the final expanding-window fold "
            "trains only on earlier rows and validates on later rows, preventing "
            "future-to-past leakage"
        ),
        "time_column": "Date",
        "n_folds": 5,
        "n_train_rows": 10,
        "n_validation_rows": 2,
    }


def test_evaluation_report_exposes_time_series_split_decision():
    report = build_holdout_evaluation_report(
        [10.0, 11.0, 12.0],
        [13.0],
        "regression",
        {"val_rmse": 0.5},
        splitter="TimeSeriesSplit",
        splitter_reason="ordered by Date",
        n_folds=3,
        time_column="Date",
    )

    assert report["splitter"] == "TimeSeriesSplit"
    assert report["splitter_reason"] == "ordered by Date"
    assert report["n_folds"] == 3
    assert report["time_column"] == "Date"


def test_temporal_request_selects_time_column_and_excludes_it_from_features():
    from app.agents.mta_v2.agent import ModelTrainingAgent

    agent = ModelTrainingAgent()
    state = {
        "messages": [
            HumanMessage(
                content="Forecast future Weekly_Sales using this time-series dataset."
            )
        ]
    }

    config = agent._fallback_data_config(
        "gs://example/walmart.csv",
        ["Store", "Dept", "Date", "Temperature", "Weekly_Sales"],
        state,
    )

    assert config["target_column"] == "Weekly_Sales"
    assert config["time_column"] == "Date"
    assert "Date" not in config["feature_columns"]
    assert "Temperature" in config["feature_columns"]


def test_temporal_column_inference_recognizes_descriptive_timestamp_names():
    from app.agents.mta_v2.agent import ModelTrainingAgent

    agent = ModelTrainingAgent()

    assert agent._infer_time_column(
        ["trip_distance", "pickup_datetime", "fare_amount"],
        "Build a time-series model to forecast fare_amount.",
    ) == "pickup_datetime"


def test_non_temporal_request_does_not_enable_time_series_split():
    from app.agents.mta_v2.agent import ModelTrainingAgent

    agent = ModelTrainingAgent()
    state = {
        "messages": [HumanMessage(content="Predict Weekly_Sales from these columns.")]
    }

    config = agent._fallback_data_config(
        "/tmp/walmart.csv",
        ["Store", "Date", "Temperature", "Weekly_Sales"],
        state,
    )

    assert "time_column" not in config


def test_ray_job_environment_forwards_time_column(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://avaloka-mlflow:5000")
    trainer = RayTrainer(user_id="user-1", session_id="session-1")
    plan = {
        "model_type": "regression",
        "model_name": "sales_forecast",
        "model_description": "Chronological sales model",
        "model_version": "v1.0",
        "data_config": {
            "dataset_uri": "gs://example/walmart.csv",
            "feature_columns": ["Store", "Temperature"],
            "target_column": "Weekly_Sales",
            "time_column": "Date",
        },
        "hyperparameter_config": {},
        "ray_config": {},
    }

    job_env = trainer._build_job_env(plan)

    assert job_env["TIME_COLUMN"] == "Date"
    assert "Date" not in job_env["FEATURE_COLUMNS"].split(",")


def test_local_trainer_forwards_time_column_to_training_entrypoint():
    plan = {
        "model_type": "regression",
        "model_name": "sales_forecast",
        "model_description": "Chronological sales model",
        "model_version": "v1.0",
        "data_config": {
            "dataset_uri": "/tmp/walmart.csv",
            "feature_columns": ["Store", "Temperature"],
            "target_column": "Weekly_Sales",
            "time_column": "Date",
        },
        "hyperparameter_config": {},
        "ray_config": None,
    }

    with patch("app.agents.mta_v2.local_trainer.setup_mlflow"), patch(
        "app.agents.mta_v2.local_trainer.training_main",
        return_value={"status": "success"},
    ) as training_entrypoint:
        result = LocalTrainer("user-1", "session-1").train(plan)

    assert result == {"status": "success"}
    assert training_entrypoint.call_args.kwargs["time_column"] == "Date"
