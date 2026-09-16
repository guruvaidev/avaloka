from app.agents.mta_v2.training_docker_image.src.evaluation import (
    build_holdout_evaluation_report,
)


def test_imbalanced_classifier_that_predicts_majority_does_not_beat_baseline():
    report = build_holdout_evaluation_report(
        [0] * 95 + [1] * 5,
        [0] * 19 + [1],
        "classification",
        {"accuracy": 0.95},
    )

    assert report["baseline"]["accuracy"] == 0.95
    assert report["beats_baseline"] is False
    assert "does not beat" in report["warnings"][0]


def test_classifier_above_majority_baseline_is_recognised():
    report = build_holdout_evaluation_report(
        [0] * 95 + [1] * 5,
        [0] * 19 + [1],
        "classification",
        {"accuracy": 1.0},
    )
    assert report["beats_baseline"] is True


def test_regressor_is_compared_with_training_mean_on_validation_rows():
    report = build_holdout_evaluation_report(
        [10.0, 20.0, 30.0],
        [12.0, 28.0],
        "regression",
        {"val_rmse": 4.0, "val_mae": 3.5, "val_r2_score": 0.5},
    )

    assert report["baseline"]["prediction"] == 20.0
    assert report["baseline"]["rmse"] == 8.0
    assert report["beats_baseline"] is True
