"""Regression tests: a session whose active dataset drifted to a query output
must be recoverable via the "reset dataset" chat command, restoring the
original-dataset snapshot taken at first activation.
"""
import pytest

from app.api.server import (
    _RESET_DATASET_RE,
    _record_latest_tabular_output,
    _restore_original_dataset_session,
    _snapshot_original_dataset_session,
)

ORIGINAL = {
    "work_local_input": "/tmp/patient_data.csv",
    "active_data_source_location": "/tmp/patient_data.csv",
    "active_data_source_location_local": "/tmp/patient_data.csv",
    "uploaded_csv_columns": ["name", "age", "dob", "price", "country"],
    "uploaded_csv_preview": [{"name": "A", "country": "India"}],
    "schema": {"name": "String", "country": "String"},
    "ddl_schema": "CREATE TABLE t (...)",
    "file_size_bytes": 800,
    "file_size_mb": 0.0,
}


@pytest.mark.parametrize("text", [
    "reset dataset",
    "reset the dataset",
    "restore the original dataset",
    "use the original dataset",
    "go back to the original data",
    "please reset dataset now",
])
def test_reset_phrases_match(text):
    assert _RESET_DATASET_RE.search(text)


@pytest.mark.parametrize("text", [
    "give me rows where Country is India",
    "the original dataset had 20 rows",
    "restore from the backup table",
    "reset my password",
])
def test_ordinary_prompts_do_not_match(text):
    assert not _RESET_DATASET_RE.search(text)


def test_snapshot_then_restore_heals_a_degenerated_session():
    sess = dict(ORIGINAL)
    assert _snapshot_original_dataset_session(sess) is True

    # Simulate the drift: active dataset becomes a tiny query output.
    sess.update({
        "active_data_source_location": "/tmp/output.csv",
        "active_data_source_location_local": "/tmp/output.csv",
        "work_local_input": "/tmp/output.csv",
        "uploaded_csv_columns": ["name"],
        "uploaded_csv_preview": [{"name": "Carla Mendes"}],
        "schema": {"name": "String"},
        "file_size_bytes": 18,
        "data_source_was_modified": True,
    })

    assert _restore_original_dataset_session(sess) is True
    for key, value in ORIGINAL.items():
        assert sess[key] == value, key
    assert sess["data_source_was_modified"] is False
    assert "original_dataset_snapshot" not in sess


def test_restore_without_snapshot_is_a_noop():
    sess = dict(ORIGINAL)
    assert _restore_original_dataset_session(sess) is False


def test_snapshot_is_taken_only_once():
    sess = dict(ORIGINAL)
    assert _snapshot_original_dataset_session(sess) is True
    sess["file_size_bytes"] = 18
    assert _snapshot_original_dataset_session(sess) is False


def test_latest_output_record_survives_original_restore(tmp_path):
    output = tmp_path / "output.csv"
    output.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")
    sess = dict(ORIGINAL)

    assert _record_latest_tabular_output(sess, str(output)) is True
    assert sess["latest_output_location"] == str(output)
    assert sess["latest_output_columns"] == ["feature", "target"]
    assert sess["latest_output_row_count"] == 2
    assert sess["latest_output_is_trainable"] is True

    assert _snapshot_original_dataset_session(sess) is True
    sess.update({
        "active_data_source_location": str(output),
        "active_data_source_location_local": str(output),
        "work_local_input": str(output),
        "data_source_was_modified": True,
    })

    assert _restore_original_dataset_session(sess) is True
    assert sess["active_data_source_location"] == ORIGINAL["active_data_source_location"]
    assert sess["latest_output_location"] == str(output)
