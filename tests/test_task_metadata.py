from app.core.task_metadata import scheduled_task_metadata


def test_inference_service_setup_metadata_has_long_poll_hint():
    metadata = scheduled_task_metadata({
        "task_type": "start_inference",
    })

    assert metadata["task_label"] == "Inference service setup"
    assert metadata["scheduled_message"] == "Inference service setup is scheduled."
    assert metadata["running_message"] == "Inference service setup is still running."
    assert metadata["success_message"] == "Inference service setup is complete."
    assert metadata["poll_timeout_seconds"] >= 1800


def test_portfolio_metadata_is_not_used_for_mta_setup():
    metadata = scheduled_task_metadata({
        "task_type": "start_inference",
    })

    assert "Portfolio" not in metadata["task_label"]
    assert "Portfolio" not in metadata["scheduled_message"]


def test_inference_service_shutdown_metadata():
    metadata = scheduled_task_metadata({
        "task_type": "stop_inference",
    })

    assert metadata["task_label"] == "Inference service shutdown"
    assert metadata["scheduled_message"] == "Inference service shutdown is scheduled."
    assert metadata["running_message"] == "Inference service shutdown is still running."
    assert metadata["success_message"] == "Inference service shutdown is complete."
    assert metadata["poll_timeout_seconds"] >= 900
