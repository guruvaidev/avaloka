from __future__ import annotations

from typing import Any, Dict


def scheduled_task_metadata(schedule_data: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return user-facing metadata for a scheduled task."""
    data = schedule_data or {}
    task_type = str(data.get("task_type") or "execute")
    task_operation = str(data.get("task_operation") or task_type)

    metadata: Dict[str, Any] = {
        "task_type": task_type,
        "task_operation": task_operation,
        "task_label": "Scheduled task",
        "scheduled_message": "The task is scheduled to run.",
        "running_message": "The task is still running.",
        "success_message": "The scheduled task is complete.",
        "expected_duration_seconds": 300,
        "poll_timeout_seconds": 600,
        "poll_interval_seconds": 5,
    }

    if task_type == "training":
        metadata.update(
            task_label="Model training",
            scheduled_message="Model training is scheduled.",
            running_message="Model training is still running.",
            success_message="Model training is complete.",
            expected_duration_seconds=900,
            poll_timeout_seconds=1800,
        )
    elif task_type == "sample_profile":
        metadata.update(
            task_label="Portfolio sample generation",
            scheduled_message="Portfolio sample generation is scheduled.",
            running_message="Portfolio sample generation is still running.",
            success_message="Portfolio sample generation is complete.",
            expected_duration_seconds=900,
            poll_timeout_seconds=1800,
        )
    elif task_type == "start_inference":
        metadata.update(
            task_label="Inference service setup",
            scheduled_message="Inference service setup is scheduled.",
            running_message="Inference service setup is still running.",
            success_message="Inference service setup is complete.",
            expected_duration_seconds=900,
            poll_timeout_seconds=1800,
        )
    elif task_type == "stop_inference":
        metadata.update(
            task_label="Inference service shutdown",
            scheduled_message="Inference service shutdown is scheduled.",
            running_message="Inference service shutdown is still running.",
            success_message="Inference service shutdown is complete.",
            expected_duration_seconds=300,
            poll_timeout_seconds=900,
        )

    return metadata
