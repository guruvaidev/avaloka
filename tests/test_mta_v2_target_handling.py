import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agents.mta_v2.local_trainer import prepare_training_dataframe

TRAINING_IMAGE_DIR = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "agents"
    / "mta_v2"
    / "training_docker_image"
)
sys.path.insert(0, str(TRAINING_IMAGE_DIR))

from ray_job import (
    cast_regression_target_batch,
    is_valid_target_value,
    resolve_training_worker_count,
)


def test_regression_target_rows_are_dropped_not_imputed():
    df = pd.DataFrame(
        {
            "feature": [None, 2.0, 3.0, 4.0],
            "target": [10.0, 20.0, None, "bad"],
        }
    )

    prepared = prepare_training_dataframe(
        df,
        feature_cols=["feature"],
        target_col="target",
        model_type="regression",
    )

    assert prepared["target"].tolist() == [10.0, 20.0]
    # Feature imputation is deliberately deferred until after the
    # train/validation split so validation values cannot leak into the fitted
    # preprocessing statistics.
    assert pd.isna(prepared["feature"].iloc[0])
    assert prepared["feature"].iloc[1] == 2.0


def test_classification_blank_target_rows_are_dropped_not_imputed():
    df = pd.DataFrame(
        {
            "feature": [1.0, None, 3.0, 4.0],
            "target": ["yes", "", None, "no"],
        }
    )

    prepared = prepare_training_dataframe(
        df,
        feature_cols=["feature"],
        target_col="target",
        model_type="classification",
    )

    assert prepared["target"].tolist() == ["yes", "no"]
    assert prepared["feature"].tolist() == [1.0, 4.0]


def test_all_missing_targets_raise_clear_error():
    df = pd.DataFrame(
        {
            "feature": [1.0, 2.0],
            "target": [None, ""],
        }
    )

    with pytest.raises(ValueError, match="No training rows remain"):
        prepare_training_dataframe(
            df,
            feature_cols=["feature"],
            target_col="target",
            model_type="classification",
        )


def test_ray_target_filter_rejects_missing_or_bad_labels():
    assert is_valid_target_value("yes", "classification")
    assert not is_valid_target_value("", "classification")
    assert not is_valid_target_value("  ", "classification")
    assert not is_valid_target_value(None, "classification")

    assert is_valid_target_value("10.5", "regression")
    assert not is_valid_target_value("bad", "regression")
    assert not is_valid_target_value(np.inf, "regression")
    assert not is_valid_target_value(None, "regression")


def test_ray_regression_target_cast_keeps_only_numeric_label_type():
    batch = {
        "feature": np.array([1.0, 2.0]),
        "target": np.array(["10.0", 20]),
    }

    cleaned = cast_regression_target_batch(batch, "target")

    assert cleaned["target"].dtype == np.float32
    assert cleaned["target"].tolist() == [10.0, 20.0]


def test_ray_training_workers_use_requested_count_not_autoscale_max():
    assert resolve_training_worker_count(
        requested_workers=4,
        max_allowed_workers=25,
        available_blocks=48,
    ) == 4


def test_ray_training_workers_are_still_bounded():
    assert resolve_training_worker_count(
        requested_workers=8,
        max_allowed_workers=4,
        available_blocks=48,
    ) == 4
    assert resolve_training_worker_count(
        requested_workers=8,
        max_allowed_workers=25,
        available_blocks=2,
    ) == 2
    assert resolve_training_worker_count(
        requested_workers=0,
        max_allowed_workers=0,
        available_blocks=0,
    ) == 1
