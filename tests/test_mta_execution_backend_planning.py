"""Regression coverage for deterministic MTA local-versus-Ray planning."""
from __future__ import annotations

from unittest.mock import patch


def _agent():
    from app.agents.mta_v2.agent import ModelTrainingAgent
    return ModelTrainingAgent()


def test_small_gcs_dataset_is_planned_locally():
    agent = _agent()
    with patch.object(agent, "_get_file_size", return_value=1024):
        backend, reason, size = agent._training_backend_decision("gs://bucket/train.csv")
    assert (backend, size) == ("local", 1024)
    assert "same storage URI" in reason


def test_small_minio_s3_dataset_is_planned_locally():
    agent = _agent()
    with patch.object(agent, "_get_file_size", return_value=1024):
        backend, _, _ = agent._training_backend_decision("s3://avaloka/train.csv")
    assert backend == "local"


def test_large_remote_dataset_is_planned_on_ray():
    agent = _agent()
    with patch.object(agent, "_get_file_size", return_value=20 * 1024 * 1024):
        backend, reason, _ = agent._training_backend_decision("s3://avaloka/train.parquet")
    assert backend == "ray"
    assert "Ray threshold" in reason


def test_remote_dataset_with_unknown_size_is_planned_on_ray():
    agent = _agent()
    with patch.object(agent, "_get_file_size", return_value=None):
        backend, reason, _ = agent._training_backend_decision("gs://bucket/train.csv")
    assert backend == "ray"
    assert "size is unknown" in reason


def test_large_local_file_is_never_sent_to_ray():
    agent = _agent()
    with patch.object(agent, "_get_file_size", return_value=20 * 1024 * 1024):
        backend, reason, _ = agent._training_backend_decision("/tmp/train.csv")
    assert backend == "local"
    assert "cannot access" in reason


def test_zero_state_size_falls_back_to_remote_storage_metadata():
    agent = _agent()
    state = {
        "file_size_bytes": 0,
        "dataset_size_bytes": 0,
        "messages": [{"role": "system", "content": "Dataset size bytes: 0"}],
    }

    state_size = agent._state_dataset_size_bytes(state)
    assert state_size is None

    actual_size = 20 * 1024 * 1024
    with patch.object(agent, "_get_file_size", return_value=actual_size) as get_size:
        backend, reason, resolved_size = agent._training_backend_decision(
            "s3://avaloka/housing.csv",
            state_size,
        )

    get_size.assert_called_once_with("s3://avaloka/housing.csv")
    assert backend == "ray"
    assert resolved_size == actual_size
    assert "Ray threshold" in reason


def test_zero_state_size_displays_actual_small_remote_object_size():
    agent = _agent()
    state_size = agent._state_dataset_size_bytes({"file_size_bytes": 0})
    actual_size = 1_423_529

    with patch.object(agent, "_get_file_size", return_value=actual_size):
        backend, reason, resolved_size = agent._training_backend_decision(
            "s3://avaloka/housing.csv",
            state_size,
        )

    assert backend == "local"
    assert resolved_size == actual_size
    assert f"remote dataset is {actual_size} bytes" in reason
