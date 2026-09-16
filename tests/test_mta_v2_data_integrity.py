from __future__ import annotations

import pandas as pd

from app.agents.mta_v2.failure_diagnostics import (
    build_training_failure,
    format_training_failure,
)
from app.agents.mta_v2.training_docker_image.src.data_integrity import (
    build_training_integrity_report,
    classify_identifier_features,
    duplicate_rows_error,
    find_duplicate_training_rows,
    find_identifier_features,
    identifier_features_error,
)


def test_portable_integrity_report_is_recordable_on_a_clean_training_run():
    frame = pd.DataFrame({
        "target": [0, 1, 0, 1, 1, 0],
        "feature": [10, 12, 9, 15, 14, 8],
    })

    report = build_training_integrity_report(frame, "target", ["feature"])

    assert report == {
        "safe_to_train": True,
        "rows": 6,
        "checked_columns": 1,
        "n_findings": 0,
        "n_blockers": 0,
        "findings": [],
    }


def test_portable_detector_finds_exact_copied_source_rows():
    frame = pd.DataFrame({
        "PassengerId": [1, 2, 1, 2],
        "Survived": [0, 1, 0, 1],
        "Pclass": [3, 1, 3, 1],
        "Age": [22, 38, 22, 38],
    })

    match = find_duplicate_training_rows(
        frame, "Survived", ["PassengerId", "Pclass", "Age"]
    )

    assert match is not None
    assert match["duplicate_count"] == 2
    assert match["duplicate_ratio"] == 0.5
    assert match["excluded_identifier_columns"] == []


def test_portable_detector_allows_repeated_selected_values_in_distinct_rows():
    frame = pd.DataFrame({
        "order_id": [10, 11, 12, 13],
        "product_id": [7, 7, 7, 7],
        "reordered": [0, 0, 0, 0],
    })

    assert find_duplicate_training_rows(
        frame, "reordered", ["product_id"]
    ) is None


def test_duplicate_row_failure_reports_count():
    match = {
        "duplicate_count": 15,
        "duplicate_ratio": 0.5,
        "excluded_identifier_columns": [],
    }
    failure = build_training_failure(duplicate_rows_error(match))

    assert failure["code"] == "DUPLICATE_ROWS"
    assert failure["duplicate_count"] == 15
    assert failure["duplicate_ratio"] == 0.5
    rendered = format_training_failure(failure)
    assert "15 repeated record(s) (50.0%)" in rendered


def test_portable_detector_flags_unique_passenger_id_feature():
    frame = pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5, 6],
        "Pclass": [3, 1, 3, 1, 2, 3],
        "Survived": [0, 1, 0, 0, 1, 1],
    })

    matches = find_identifier_features(frame, ["PassengerId", "Pclass"])

    assert [match["column"] for match in matches] == ["PassengerId"]
    assert matches[0]["uniqueness_ratio"] == 1.0


def test_repeated_product_id_is_retained_as_an_entity_feature():
    frame = pd.DataFrame({
        "order_id": [10, 10, 11, 11, 12, 12],
        "product_id": [7, 8, 7, 9, 7, 8],
        "reordered": [0, 1, 1, 0, 1, 0],
    })

    decisions = {
        item["column"]: item
        for item in classify_identifier_features(frame, list(frame.columns))
    }

    assert decisions["order_id"]["recommendation"] == "exclude"
    assert decisions["order_id"]["role"] == "row_or_occurrence_identifier"
    assert decisions["product_id"]["recommendation"] == "keep"
    assert decisions["product_id"]["role"] == "repeated_entity_identifier"
    assert decisions["product_id"]["related_identifier_columns"] == ["order_id"]
    assert find_identifier_features(frame, ["order_id", "product_id"]) == [
        decisions["order_id"]
    ]


def test_near_unique_product_id_is_rejected_when_sample_shows_row_key_behavior():
    frame = pd.DataFrame({
        "product_id": list(range(100)),
        "target": [value % 2 for value in range(100)],
    })

    matches = find_identifier_features(frame, ["product_id"])

    assert [match["column"] for match in matches] == ["product_id"]
    assert matches[0]["role"] == "near_unique_identifier"


def test_identifier_failure_names_the_column():
    failure = build_training_failure(identifier_features_error([
        {"column": "PassengerId", "unique_count": 6, "uniqueness_ratio": 1.0}
    ]))

    assert failure["code"] == "IDENTIFIER_FEATURE"
    assert failure["affected_columns"] == ["PassengerId"]
    rendered = format_training_failure(failure)
    assert "`PassengerId` uniquely identifies rows" in rendered
    assert "Remove `PassengerId` from the selected features" in rendered
