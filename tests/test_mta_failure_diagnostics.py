"""Focused tests for structured MTA failure reporting."""

from app.agents.mta_v2.failure_diagnostics import build_training_failure, format_training_failure


def test_missing_pii_key_has_actionable_failure_code():
    failure = build_training_failure(
        "AVALOKA_PII_KEY is not set. Sensitive training column(s) 'customer_email' "
        "require keyed pseudonymisation before model training can start."
    )

    assert failure["code"] == "PII_KEY_MISSING"
    assert "sensitive columns" in failure["summary"].lower()
    assert "Configure AVALOKA_PII_KEY" in format_training_failure(failure)


def test_oom_killed_is_reported_with_pod_and_exit_code():
    failure = build_training_failure(
        "Ray job failed",
        kubernetes={
            "pod_name": "ray-worker-abc", "container_name": "ray-worker",
            "reason": "OOMKilled", "exit_code": 137,
            "message": "Container exceeded its 16Gi memory limit",
        },
        job_id="job-123", namespace="ray-trainer-123",
    )
    assert failure["code"] == "OOM_KILLED"
    assert failure["kubernetes_reason"] == "OOMKilled"
    assert failure["exit_code"] == 137
    assert failure["pod_name"] == "ray-worker-abc"
    assert "out of memory" in failure["title"].lower()


def test_timeout_is_distinct_from_generic_training_failure():
    failure = build_training_failure("Ray job did not complete within 1800 seconds")
    assert failure["code"] == "TRAINING_TIMEOUT"
    assert failure["retryable"] is True


def test_sensitive_values_are_redacted_from_returned_logs():
    failure = build_training_failure(
        "request failed", logs="Authorization: Bearer secret-token API_KEY=do-not-return-this",
    )
    assert "secret-token" not in failure["logs_tail"]
    assert "do-not-return-this" not in failure["logs_tail"]
    assert failure["logs_tail"].count("[REDACTED]") == 2


def test_chat_failure_message_shows_safe_actions_without_diagnostics():
    failure = build_training_failure(
        "job failed", kubernetes={"reason": "ImagePullBackOff", "pod_name": "trainer-pod"},
    )
    message = format_training_failure(failure)
    assert "Model training could not be completed." in message
    assert "What to do" in message
    assert "IMAGE_PULL_FAILED" in message
    assert "configured training image name and tag" in message
    assert "ImagePullBackOff" not in message
    assert "trainer-pod" not in message


def test_ray_failure_uses_last_actionable_log_line_as_technical_cause():
    failure = build_training_failure(
        "Ray job finished with status FAILED.",
        logs="starting worker\nTraceback (most recent call last):\nValueError: target column 'label' is missing",
    )
    assert failure["technical_details"] == "ValueError: target column 'label' is missing"
    assert failure["code"] == "DATA_ERROR"


def test_gcp_cloud_storage_quota_has_provider_specific_recovery_steps():
    failure = build_training_failure(
        "ResourceExhausted: gs://training-artifacts storage quota exceeded",
        provider="gcp",
    )

    assert failure["code"] == "CLOUD_STORAGE_QUOTA_EXCEEDED"
    assert failure["provider"] == "gcp"
    assert any("Google Cloud" in action for action in failure["actions"])
    assert "storage quota" in format_training_failure(failure).lower()


def test_mlflow_disk_capacity_is_distinct_from_cloud_quota():
    failure = build_training_failure(
        "MLflow artifact write failed: No space left on device",
    )

    assert failure["code"] == "ARTIFACT_STORAGE_FULL"
    assert any("MLflow" in action for action in failure["actions"])


def test_ephemeral_storage_pressure_has_kubernetes_recovery_steps():
    failure = build_training_failure(
        "Pod evicted: node was low on resource ephemeral-storage",
    )

    assert failure["code"] == "EPHEMERAL_STORAGE_FULL"
    assert any("ephemeral-storage" in action for action in failure["actions"])


def test_azure_storage_error_uses_azure_recovery_step():
    failure = build_training_failure(
        "StorageAccountIsFull while writing to Azure Blob storage account",
        provider="azure",
    )

    assert failure["code"] == "CLOUD_STORAGE_QUOTA_EXCEEDED"
    assert any("In Azure" in action for action in failure["actions"])


def test_unknown_failure_never_echoes_raw_error_to_user():
    failure = build_training_failure("private internal hostname db-secret.internal failed")
    message = format_training_failure(failure)

    assert failure["code"] == "TRAINING_FAILED"
    assert "TRAINING_FAILED" in message
    assert "db-secret.internal" not in message


def test_formatter_does_not_trust_persisted_summary_or_actions():
    message = format_training_failure({
        "code": "TRAINING_FAILED",
        "summary": "private stack trace from an older run",
        "actions": ["send token=secret-value"],
    })

    assert "private stack trace" not in message
    assert "secret-value" not in message
    assert "Retry the training run once" in message
