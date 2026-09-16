"""Automated MTA coverage using real schemas and rows from Kaggle datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterator

import pytest


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "mta_kaggle"
MANIFEST = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))

INSTACART_FEATURES = ["product_id", "add_to_cart_order"]
FRAUD_FEATURES = [
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card4",
    "card6",
    "C1",
    "D1",
]
WALMART_FEATURES = [
    "dept_id",
    "store_id",
    "d_1907",
    "d_1908",
    "d_1909",
    "d_1910",
    "d_1911",
    "d_1912",
]


def _fixture(name: str) -> Path:
    return FIXTURE_DIR / name


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _plan(
    *,
    dataset: Path,
    model_type: str,
    target: str,
    features: list[str],
    name: str,
    epochs: int = 1,
) -> dict[str, object]:
    return {
        "model_type": model_type,
        "model_name": name,
        "model_description": f"Automated {model_type} test using a Kaggle sample",
        "model_version": "test-v1",
        "ray_config": None,
        "hyperparameter_config": {
            "learning_rate": 0.01,
            "epochs": epochs,
            "batch_size": 32,
            "optimizer_name": "adam",
            "random_seed": 42,
            "early_stopping_patience": 1,
            "hidden_layer_sizes": [16, 8],
            "activation": "relu",
            "dropout_rate": 0.0,
            "batch_norm": False,
        },
        "data_config": {
            "dataset_uri": str(dataset),
            "feature_columns": features,
            "target_column": target,
        },
    }


@pytest.fixture()
def isolated_mlflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    import mlflow

    database = tmp_path / "mlflow.db"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{database}")
    monkeypatch.setenv("MLFLOW_BACKEND_STORE_URI", "")
    monkeypatch.setenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", artifacts.as_uri())
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", f"mta-kaggle-{tmp_path.name}")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.end_run()
    yield artifacts
    mlflow.end_run()


@pytest.mark.parametrize(
    ("filename", "target", "expected_columns"),
    [
        (
            "instacart_train_sample.csv",
            "reordered",
            {"order_id", "product_id", "add_to_cart_order", "reordered"},
        ),
        (
            "ieee_fraud_train_sample.csv",
            "isFraud",
            {"TransactionAmt", "ProductCD", "card4", "card6", "isFraud"},
        ),
        (
            "walmart_validation_sample.csv",
            "d_1913",
            {"item_id", "store_id", "d_1907", "d_1912", "d_1913"},
        ),
    ],
)
def test_training_fixtures_have_targets_and_nonconstant_labels(
    filename: str,
    target: str,
    expected_columns: set[str],
) -> None:
    columns, rows = _csv_rows(_fixture(filename))
    labels = [row[target] for row in rows]

    assert expected_columns <= set(columns)
    assert all(label.strip() for label in labels)
    assert len(set(labels)) > 1


def test_fixture_manifest_matches_generated_files() -> None:
    for filename, metadata in MANIFEST["fixtures"].items():
        path = _fixture(filename)
        assert path.exists()
        _, rows = _csv_rows(path)
        assert len(rows) == metadata["rows"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]


def test_instacart_prior_and_train_are_compatible_for_inference() -> None:
    prior_columns, prior = _csv_rows(_fixture("instacart_prior_sample.csv"))
    train_columns, train = _csv_rows(_fixture("instacart_train_sample.csv"))

    assert prior_columns == train_columns
    assert {row["reordered"] for row in train} == {"0", "1"}
    assert {row["reordered"] for row in prior} == {"0", "1"}
    assert "order_id" not in INSTACART_FEATURES


def test_ieee_competition_test_file_is_rejected_for_training() -> None:
    from app.agents.mta_v2.local_trainer import training_main

    dataset = _fixture("ieee_fraud_test_sample.csv")

    with pytest.raises(ValueError, match="Target column 'isFraud' is not present"):
        training_main(
            data_uri=str(dataset),
            model_type="classification",
            target_col="isFraud",
            feature_cols=FRAUD_FEATURES,
            epochs=1,
        )


def test_fraud_preprocessing_handles_categories_missing_values_and_scaling() -> None:
    import numpy as np
    import pandas as pd

    from app.agents.mta_v2.local_trainer import (
        apply_standard_scaling,
        fit_feature_preprocessing,
        prepare_training_dataframe,
        split_train_val,
        transform_feature_dataframe,
    )

    frame = pd.read_csv(_fixture("ieee_fraud_train_sample.csv"))
    prepared = prepare_training_dataframe(
        frame,
        feature_cols=FRAUD_FEATURES,
        target_col="isFraud",
        model_type="classification",
    )
    train, validation = split_train_val(prepared)
    preprocessing = fit_feature_preprocessing(train, FRAUD_FEATURES)
    train = transform_feature_dataframe(train, FRAUD_FEATURES, preprocessing)
    validation = transform_feature_dataframe(validation, FRAUD_FEATURES, preprocessing)
    train["__label__"] = train["isFraud"].astype(np.int64)
    validation["__label__"] = validation["isFraud"].astype(np.int64)
    scaled_train, scaled_validation, preprocessing = apply_standard_scaling(
        train,
        validation,
        [],
        "classification",
        preprocessing,
    )

    assert {"ProductCD", "card4", "card6"} <= set(
        preprocessing["categorical_features"]
    )
    assert set(preprocessing["feature_scaling"]) == {
        feature
        for feature in FRAUD_FEATURES
        if feature not in {"ProductCD", "card4", "card6"}
    }
    assert all(
        metadata["encoding"] == "one_hot"
        for metadata in preprocessing["categorical_features"].values()
    )
    model_features = preprocessing["model_feature_names"]
    assert np.isfinite(scaled_train[model_features].to_numpy()).all()
    assert np.isfinite(scaled_validation[model_features].to_numpy()).all()
    assert set(scaled_train["__label__"].unique()) <= {0, 1}


def test_preprocessing_uses_training_statistics_and_unknown_bucket() -> None:
    """Validation-only values must not affect preprocessing fitted on train rows."""
    import numpy as np
    import pandas as pd

    from app.agents.mta_v2.local_trainer import (
        fit_feature_preprocessing,
        transform_feature_dataframe,
    )

    train = pd.DataFrame(
        {
            "category": ["apple", "banana", None],
            "amount": [0.0, 2.0, np.nan],
        }
    )
    validation = pd.DataFrame({"category": ["orange"], "amount": [100.0]})

    preprocessing = fit_feature_preprocessing(train, ["category", "amount"])
    transformed_train = transform_feature_dataframe(train, ["category", "amount"], preprocessing)
    transformed_validation = transform_feature_dataframe(
        validation,
        ["category", "amount"],
        preprocessing,
    )

    category_metadata = preprocessing["categorical_features"]["category"]
    numeric_metadata = preprocessing["feature_scaling"]["amount"]
    assert category_metadata["categories"] == ["apple", "banana"]
    assert preprocessing["imputation"]["category"]["fill_value"] == "apple"
    assert numeric_metadata["mean"] == pytest.approx(1.0)
    # The missing training value is imputed to the training mean before the
    # scaler is fitted, so the fitted population standard deviation is sqrt(2/3).
    assert numeric_metadata["scale"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert transformed_train[numeric_metadata["output_column"]].tolist() == pytest.approx(
        [-1.0 / numeric_metadata["scale"], 1.0 / numeric_metadata["scale"], 0.0]
    )

    # ``orange`` exists only in validation, so no train category column can
    # represent it. It must use the dedicated unknown bucket instead.
    assert transformed_validation[category_metadata["unknown_column"]].iloc[0] == 1.0
    assert all(
        transformed_validation[column].iloc[0] == 0.0
        for column in category_metadata["output_columns"]
    )
    assert transformed_validation[numeric_metadata["output_column"]].iloc[0] == pytest.approx(
        99.0 / numeric_metadata["scale"]
    )


def test_balanced_class_weights_upweight_minority_class() -> None:
    import pandas as pd

    from app.agents.mta_v2.local_trainer import balanced_class_weights

    weights = balanced_class_weights(pd.Series([0, 0, 0, 0, 1]), num_classes=2)
    assert weights.tolist() == pytest.approx([0.625, 2.5])
    assert weights[1] > weights[0]


def test_instacart_classification_trains_and_batch_infers_from_mlflow(
    isolated_mlflow: Path,
) -> None:
    import pandas as pd

    from app.agents.mta_v2.inference import InferenceInterface
    from app.agents.mta_v2.local_trainer import LocalTrainer

    plan = _plan(
        dataset=_fixture("instacart_train_sample.csv"),
        model_type="classification",
        target="reordered",
        features=INSTACART_FEATURES,
        name="instacart-reorder-test",
    )
    result = LocalTrainer(user_id="pytest", session_id="classification").train(plan)

    assert "error" not in result
    assert result["rows_processed"] == 512
    assert result["num_classes"] == 2
    assert 0.0 <= result["final_accuracy"] <= 1.0
    assert 0.0 <= result["final_f1_score"] <= 1.0
    assert "final_mae" not in result

    inference = InferenceInterface(model_cache_dir=str(isolated_mlflow / "cache"))
    rows = (
        pd.read_csv(_fixture("instacart_prior_sample.csv"))[INSTACART_FEATURES]
        .iloc[[0, 17]]
        .to_dict(orient="records")
    )
    predictions = inference.predict_batch(result["mlflow_run_id"], rows)

    assert len(predictions) == 2
    for prediction in predictions:
        payload = prediction.to_dict()
        assert payload["prediction"] in {"0", "1"}
        assert math.isclose(
            sum(payload["probabilities"].values()),
            1.0,
            abs_tol=1e-5,
        )


def test_fraud_classification_trains_and_handles_unknown_category_at_inference(
    isolated_mlflow: Path,
) -> None:
    """A trained model must score a request containing an unseen category."""
    import pandas as pd

    from app.agents.mta_v2.inference import InferenceInterface
    from app.agents.mta_v2.local_trainer import LocalTrainer

    result = LocalTrainer(user_id="pytest", session_id="fraud-inference").train(
        _plan(
            dataset=_fixture("ieee_fraud_train_sample.csv"),
            model_type="classification",
            target="isFraud",
            features=FRAUD_FEATURES,
            name="ieee-fraud-inference-test",
            epochs=3,
        )
    )
    assert "error" not in result

    row = pd.read_csv(_fixture("ieee_fraud_train_sample.csv"))[FRAUD_FEATURES].iloc[0].to_dict()
    row["ProductCD"] = "unseen-product-code"
    inference = InferenceInterface(model_cache_dir=str(isolated_mlflow / "fraud-cache"))
    prediction = inference.predict_batch(result["mlflow_run_id"], [row])[0].to_dict()

    assert prediction["prediction"] in {"0", "1"}
    assert math.isclose(sum(prediction["probabilities"].values()), 1.0, abs_tol=1e-5)


def test_walmart_regression_trains_scales_and_batch_infers_from_mlflow(
    isolated_mlflow: Path,
) -> None:
    import pandas as pd

    from app.agents.mta_v2.inference import InferenceInterface
    from app.agents.mta_v2.local_trainer import LocalTrainer

    plan = _plan(
        dataset=_fixture("walmart_validation_sample.csv"),
        model_type="regression",
        target="d_1913",
        features=WALMART_FEATURES,
        name="walmart-next-day-sales-test",
    )
    result = LocalTrainer(user_id="pytest", session_id="regression").train(plan)

    assert "error" not in result
    assert result["rows_processed"] == 512
    assert result["num_classes"] == 1
    assert result["final_mae"] >= 0.0
    assert result["final_rmse"] >= result["final_mae"]
    assert math.isfinite(result["final_r2_score"])
    assert "final_accuracy" not in result
    assert "final_f1_score" not in result

    inference = InferenceInterface(model_cache_dir=str(isolated_mlflow / "cache"))
    _, config = inference.load_model(result["mlflow_run_id"])
    assert config.target_column == "d_1913"
    assert config.preprocessing["target_scaling"]["method"] == "standard"
    assert set(config.preprocessing["categorical_features"]) == {"dept_id", "store_id"}

    rows = (
        pd.read_csv(_fixture("walmart_validation_sample.csv"))[WALMART_FEATURES]
        .iloc[[0, 19]]
        .to_dict(orient="records")
    )
    predictions = inference.predict_batch(result["mlflow_run_id"], rows)

    assert len(predictions) == 2
    assert all(math.isfinite(float(item.prediction)) for item in predictions)
    assert all(item.probabilities == {} for item in predictions)


def _held_out_rows(frame, target: str):
    """Create a deterministic, disjoint hold-out set from a committed fixture."""
    test = frame.iloc[::5].reset_index(drop=True)
    train = frame.drop(frame.index[::5]).reset_index(drop=True)
    assert not train.empty and not test.empty
    assert set(test[target].astype(str)) <= set(frame[target].astype(str))
    return train, test


def _binary_metrics(actual: list[str], predicted: list[str]) -> tuple[float, float]:
    """Return accuracy and F1 for the positive ``1`` class without sklearn state."""
    assert len(actual) == len(predicted) and actual
    accuracy = sum(a == p for a, p in zip(actual, predicted)) / len(actual)
    true_positive = sum(a == "1" and p == "1" for a, p in zip(actual, predicted))
    false_positive = sum(a != "1" and p == "1" for a, p in zip(actual, predicted))
    false_negative = sum(a == "1" and p != "1" for a, p in zip(actual, predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = (2 * true_positive / denominator) if denominator else 0.0
    return accuracy, f1


def _binary_roc_auc(actual: list[str], positive_scores: list[float]) -> float:
    """Return ROC AUC as the probability that a positive outranks a negative."""
    assert len(actual) == len(positive_scores) and actual
    positives = [score for label, score in zip(actual, positive_scores) if label == "1"]
    negatives = [score for label, score in zip(actual, positive_scores) if label != "1"]
    assert positives and negatives
    wins = sum(positive > negative for positive in positives for negative in negatives)
    ties = sum(positive == negative for positive in positives for negative in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def _binary_balanced_accuracy(actual: list[str], predicted: list[str]) -> float:
    """Average positive recall and negative recall so both classes matter equally."""
    positives = sum(label == "1" for label in actual)
    negatives = len(actual) - positives
    assert positives and negatives
    true_positive = sum(a == "1" and p == "1" for a, p in zip(actual, predicted))
    true_negative = sum(a != "1" and p != "1" for a, p in zip(actual, predicted))
    return 0.5 * (true_positive / positives + true_negative / negatives)


def test_instacart_held_out_inference_beats_majority_baseline(
    isolated_mlflow: Path,
    tmp_path: Path,
) -> None:
    """Quality gate: score actual inference predictions on rows never trained on."""
    import pandas as pd

    from app.agents.mta_v2.inference import InferenceInterface
    from app.agents.mta_v2.local_trainer import LocalTrainer

    frame = pd.read_csv(_fixture("instacart_train_sample.csv"))
    train, held_out = _held_out_rows(frame, "reordered")
    train_path = tmp_path / "instacart_quality_train.csv"
    train.to_csv(train_path, index=False)

    result = LocalTrainer(user_id="pytest", session_id="instacart-quality").train(
        _plan(
            dataset=train_path,
            model_type="classification",
            target="reordered",
            features=INSTACART_FEATURES,
            name="instacart-held-out-quality",
            epochs=30,
        )
    )
    assert "error" not in result

    inference = InferenceInterface(model_cache_dir=str(isolated_mlflow / "quality-cache"))
    predictions = inference.predict_batch(
        result["mlflow_run_id"],
        held_out[INSTACART_FEATURES].to_dict(orient="records"),
    )
    actual = held_out["reordered"].astype(str).tolist()
    predicted = [str(item.prediction) for item in predictions]
    positive_scores = [float(item.probabilities["1"]) for item in predictions]
    accuracy, f1 = _binary_metrics(actual, predicted)
    roc_auc = _binary_roc_auc(actual, positive_scores)
    balanced_accuracy = _binary_balanced_accuracy(actual, predicted)
    majority_accuracy = max(actual.count("0"), actual.count("1")) / len(actual)
    print(
        "Instacart held-out quality: "
        f"accuracy={accuracy:.3f}, positive_f1={f1:.3f}, "
        f"balanced_accuracy={balanced_accuracy:.3f}, roc_auc={roc_auc:.3f}, "
        f"majority_baseline={majority_accuracy:.3f}"
    )

    # A useful classifier must improve on always predicting the majority class
    # and identify at least some genuine reorder events.
    assert accuracy >= majority_accuracy + 0.01, (
        f"held-out accuracy {accuracy:.3f} did not beat majority baseline "
        f"{majority_accuracy:.3f}"
    )
    assert f1 >= 0.60, f"held-out positive-class F1 is too low: {f1:.3f}"
    assert balanced_accuracy >= 0.52, (
        f"held-out balanced accuracy is too low: {balanced_accuracy:.3f}"
    )
    assert roc_auc >= 0.55, f"held-out ROC AUC is too low: {roc_auc:.3f}"


def test_fraud_held_out_inference_beats_no_skill_baselines(
    isolated_mlflow: Path,
    tmp_path: Path,
) -> None:
    """Quality gate: fraud predictions must generalize beyond no-skill guesses."""
    import pandas as pd

    from app.agents.mta_v2.inference import InferenceInterface
    from app.agents.mta_v2.local_trainer import LocalTrainer

    frame = pd.read_csv(_fixture("ieee_fraud_train_sample.csv"))
    train, held_out = _held_out_rows(frame, "isFraud")
    train_path = tmp_path / "fraud_quality_train.csv"
    train.to_csv(train_path, index=False)

    result = LocalTrainer(user_id="pytest", session_id="fraud-quality").train(
        _plan(
            dataset=train_path,
            model_type="classification",
            target="isFraud",
            features=FRAUD_FEATURES,
            name="ieee-fraud-held-out-quality",
            epochs=30,
        )
    )
    assert "error" not in result

    inference = InferenceInterface(model_cache_dir=str(isolated_mlflow / "quality-cache"))
    predictions = inference.predict_batch(
        result["mlflow_run_id"],
        held_out[FRAUD_FEATURES].to_dict(orient="records"),
    )
    actual = held_out["isFraud"].astype(str).tolist()
    predicted = [str(item.prediction) for item in predictions]
    positive_scores = [float(item.probabilities["1"]) for item in predictions]
    accuracy, f1 = _binary_metrics(actual, predicted)
    balanced_accuracy = _binary_balanced_accuracy(actual, predicted)
    roc_auc = _binary_roc_auc(actual, positive_scores)
    majority_accuracy = max(actual.count("0"), actual.count("1")) / len(actual)
    print(
        "IEEE fraud held-out quality: "
        f"accuracy={accuracy:.3f}, positive_f1={f1:.3f}, "
        f"balanced_accuracy={balanced_accuracy:.3f}, roc_auc={roc_auc:.3f}, "
        f"majority_baseline={majority_accuracy:.3f}"
    )

    assert accuracy >= majority_accuracy + 0.10, (
        f"held-out accuracy {accuracy:.3f} did not beat majority baseline "
        f"{majority_accuracy:.3f} by ten percentage points"
    )
    assert f1 >= 0.60, f"held-out positive-class F1 is too low: {f1:.3f}"
    assert balanced_accuracy >= 0.60, (
        f"held-out balanced accuracy is too low: {balanced_accuracy:.3f}"
    )
    assert roc_auc >= 0.65, f"held-out ROC AUC is too low: {roc_auc:.3f}"


def test_walmart_held_out_inference_beats_mean_baseline(
    isolated_mlflow: Path,
    tmp_path: Path,
) -> None:
    """Quality gate: regression inference must beat predicting the training mean."""
    import numpy as np
    import pandas as pd

    from app.agents.mta_v2.inference import InferenceInterface
    from app.agents.mta_v2.local_trainer import LocalTrainer

    frame = pd.read_csv(_fixture("walmart_validation_sample.csv"))
    train, held_out = _held_out_rows(frame, "d_1913")
    train_path = tmp_path / "walmart_quality_train.csv"
    train.to_csv(train_path, index=False)

    result = LocalTrainer(user_id="pytest", session_id="walmart-quality").train(
        _plan(
            dataset=train_path,
            model_type="regression",
            target="d_1913",
            features=WALMART_FEATURES,
            name="walmart-held-out-quality",
            epochs=40,
        )
    )
    assert "error" not in result

    inference = InferenceInterface(model_cache_dir=str(isolated_mlflow / "quality-cache"))
    predictions = inference.predict_batch(
        result["mlflow_run_id"],
        held_out[WALMART_FEATURES].to_dict(orient="records"),
    )
    actual = held_out["d_1913"].astype(float).to_numpy()
    predicted = np.asarray([float(item.prediction) for item in predictions])
    mae = float(np.mean(np.abs(actual - predicted)))
    mean_baseline_mae = float(np.mean(np.abs(actual - train["d_1913"].mean())))
    rmse = float(np.sqrt(np.mean(np.square(actual - predicted))))
    mean_baseline_rmse = float(
        np.sqrt(np.mean(np.square(actual - train["d_1913"].mean())))
    )
    r2 = 1.0 - float(np.sum(np.square(actual - predicted))) / float(
        np.sum(np.square(actual - actual.mean()))
    )
    print(
        "Walmart held-out quality: "
        f"mae={mae:.3f}, mean_baseline_mae={mean_baseline_mae:.3f}, "
        f"rmse={rmse:.3f}, mean_baseline_rmse={mean_baseline_rmse:.3f}, "
        f"r2={r2:.3f}"
    )

    assert np.isfinite(predicted).all()
    assert mae < mean_baseline_mae, (
        f"held-out MAE {mae:.3f} did not beat mean baseline {mean_baseline_mae:.3f}"
    )
    assert rmse < mean_baseline_rmse, (
        f"held-out RMSE {rmse:.3f} did not beat mean baseline "
        f"{mean_baseline_rmse:.3f}"
    )
    assert r2 > 0.0, f"held-out R2 must be positive, got {r2:.3f}"


def _full_dataset_dir() -> Path:
    configured = os.getenv("MTA_KAGGLE_DATASET_DIR", "").strip()
    if not configured:
        pytest.skip("Set MTA_KAGGLE_DATASET_DIR to validate the full Kaggle downloads")
    return Path(configured).expanduser().resolve()


@pytest.mark.kaggle
@pytest.mark.parametrize(
    ("filename", "required", "forbidden"),
    [
        (
            "mba_order_products__prior.csv",
            {"order_id", "product_id", "add_to_cart_order", "reordered"},
            set(),
        ),
        (
            "mba_order_products__train.csv",
            {"order_id", "product_id", "add_to_cart_order", "reordered"},
            set(),
        ),
        (
            "train_transaction.csv",
            {"TransactionID", "TransactionAmt", "ProductCD", "isFraud", "V339"},
            set(),
        ),
        (
            "test_transaction.csv",
            {"TransactionID", "TransactionAmt", "ProductCD", "V339"},
            {"isFraud"},
        ),
        (
            "walmart_sales_train_validation.csv",
            {"id", "item_id", "store_id", "d_1", "d_1913"},
            {"d_1914"},
        ),
        (
            "walmart_sales_train_evaluation.csv",
            {"id", "item_id", "store_id", "d_1", "d_1941"},
            set(),
        ),
    ],
)
def test_full_kaggle_download_schema(
    filename: str,
    required: set[str],
    forbidden: set[str],
) -> None:
    path = _full_dataset_dir() / filename
    assert path.exists(), f"Missing Kaggle download: {path}"
    assert path.stat().st_size > 0, f"Kaggle download is empty: {path}"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        columns = set(next(csv.reader(handle)))
    assert required <= columns
    assert forbidden.isdisjoint(columns)
