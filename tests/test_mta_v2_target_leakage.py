from __future__ import annotations

import pandas as pd

from app.agents.mta_v2.failure_diagnostics import (
    build_training_failure,
    format_training_failure,
)
from app.agents.mta_v2.training_docker_image.src.data_integrity import (
    find_target_leakage,
)


def test_portable_detector_blocks_exact_titanic_target_copy():
    frame = pd.DataFrame({
        "Survived": [0, 1, 1, 0, 1, 0],
        "Pclass": [3, 1, 3, 3, 1, 2],
        "Survived_leak": [0, 1, 1, 0, 1, 0],
    })
    matches = find_target_leakage(
        frame, "Survived", ["Pclass", "Survived_leak"]
    )
    assert [match["column"] for match in matches] == ["Survived_leak"]
    assert matches[0]["method"] == "exact_copy"


def test_portable_detector_allows_normal_titanic_features():
    frame = pd.DataFrame({
        "Survived": [0, 1, 1, 0, 0, 1],
        "Pclass": [3, 1, 3, 1, 3, 2],
        "Age": [22, 38, 26, 35, 35, 28],
    })
    assert find_target_leakage(frame, "Survived", ["Pclass", "Age"]) == []


def test_target_leakage_has_specific_user_safe_guidance():
    failure = build_training_failure(
        "Target leakage detected: feature column 'Survived_leak' duplicates target 'Survived'."
    )
    assert failure["code"] == "TARGET_LEAKAGE"
    assert failure["affected_columns"] == ["Survived_leak"]
    rendered = format_training_failure(failure)
    assert "reveals the target answer" in rendered
    assert "`Survived_leak`" in rendered
    assert "Remove `Survived_leak` from the selected features" in rendered
    assert "`Survived_leak`" in failure["user_message"]
