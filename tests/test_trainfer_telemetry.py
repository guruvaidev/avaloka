"""T-22: ML telemetry never carries customer data. (E on the base: module absent.)"""
from __future__ import annotations

import json

import pytest

STATE = {
    "user_id": "user-1",
    "training_plan": {
        "model_name": "Acme Churn Q3", "model_description": "customers likely to leave",
        "data_config": {"dataset_uri": "s3://acme-bucket/customers_2026.csv",
                        "feature_columns": ["salary", "ssn_hash", "zip"], "target_column": "churned"},
    },
    "training_result": {
        "mlflow_run_id": "run-1", "model_name": "Acme Churn Q3", "model_type": "classification",
        "rows_processed": 12345, "num_features": 3, "num_classes": 2,
        "class_names": ["stayed", "churned"], "error": "column salary has nulls",
    },
    "filename": "customers_2026.csv",
    "messages": [{"role": "user", "content": "train on salary and zip"}],
}
FORBIDDEN_VALUES = ["Acme Churn Q3", "customers likely to leave", "s3://acme-bucket", "customers_2026.csv",
                    "salary", "ssn_hash", "zip", "churned", "stayed", "column salary has nulls", "user-1"]


def test_event_from_a_training_state_has_counts_but_no_customer_data():
    from app.core.ml_telemetry import event
    e = event("training.finished", state=STATE, job_id="j1", run_id="run-1", status="succeeded", duration_s=12.5)
    blob = json.dumps(e)
    for value in FORBIDDEN_VALUES:
        assert value not in blob, value
    assert e["n_rows"] == 12345 and e["n_features"] == 3 and e["n_classes"] == 2
    assert e["task_type"] == "classification"
    assert e["user_hash"] != "user-1" and len(e["user_hash"]) == 12
    assert set(e) <= {"event", "job_id", "run_id", "status", "duration_s", "n_rows", "n_features", "n_classes", "task_type", "user_hash"}


def test_explicit_denylisted_fields_raise():
    from app.core.ml_telemetry import TelemetryLeak, event
    with pytest.raises(TelemetryLeak):
        event("x", model_name="Acme")
    with pytest.raises(TelemetryLeak):
        event("x", error="column salary has nulls")


def test_unknown_fields_are_dropped_and_values_containing_customer_text_are_scrubbed():
    from app.core.ml_telemetry import event
    e = event("x", state=STATE, status="failed", error_code="s3://acme-bucket/customers_2026.csv", comment="free text")
    assert "comment" not in e
    assert "error_code" not in e, "a value carrying a dataset URI must be scrubbed"


def test_emit_logs_one_json_line(caplog):
    from app.core.ml_telemetry import emit
    with caplog.at_level("INFO", logger="avaloka.telemetry.ml"):
        emit("endpoint.ready", run_id="run-1", status="ready", server="cpu")
    lines = [r.getMessage() for r in caplog.records if r.name == "avaloka.telemetry.ml"]
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"event": "endpoint.ready", "run_id": "run-1", "server": "cpu", "status": "ready"}
