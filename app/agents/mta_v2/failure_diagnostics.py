"""Structured, user-safe diagnostics for model-training failures.

Raw provider, Kubernetes, and trainer errors are retained in the structured
failure object for operators.  User-facing formatting is deliberately limited
to a stable error reference, a safe explanation, and concrete recovery steps.
"""
from __future__ import annotations

import ast
import os
import re
from typing import Any, Dict, Optional, Sequence


TRAINING_FAILURE_MESSAGE = "Model training could not be completed."


_RULES = (
    ("PII_KEY_MISSING", (r"avaloka_pii_key is not set", r"sensitive training column.*require keyed"), "PII protection key is missing", "Training includes sensitive columns that require keyed protection, but no PII key is configured.", False),
    ("TARGET_LEAKAGE", (r"target leakage detected", r"duplicate or almost perfectly encode target"), "Target leakage detected", "One or more selected features reveal the target answer, so training was blocked.", False),
    ("DUPLICATE_ROWS", (r"duplicate training rows detected", r"copied records can leak across"), "Duplicate training rows detected", "Repeated records could leak across the training and validation split, so training was blocked.", False),
    ("IDENTIFIER_FEATURE", (r"identifier-like feature detected", r"unique row identifiers.*not be used for prediction"), "Identifier-like feature detected", "A selected feature uniquely identifies rows instead of providing a reusable prediction signal, so training was blocked.", False),
    ("OOM_KILLED", (r"oomkilled", r"out of memory", r"exit(?:ed)?(?: code)?\s*137", r"memory limit"), "Training pod ran out of memory", "The training container exceeded its memory limit.", True),
    ("TRAINING_TIMEOUT", (r"timed? out", r"timeout", r"deadline exceeded", r"did not complete within"), "Training timed out", "The training run did not finish before its configured deadline.", True),
    ("IMAGE_PULL_FAILED", (r"imagepullbackoff", r"errimagepull", r"failed to pull image", r"manifest unknown", r"not found.*(?:image|manifest)"), "Training image could not be pulled", "Kubernetes could not download the container image required for training.", True),
    ("POD_EVICTED", (r"\bevicted\b", r"node had condition"), "Training pod was evicted", "Kubernetes evicted the training pod because the node was under resource pressure.", True),
    ("POD_UNSCHEDULABLE", (r"unschedulable", r"failedscheduling", r"insufficient (?:cpu|memory|nvidia.com/gpu)", r"too many pods"), "Training pod could not be scheduled", "No Kubernetes node currently satisfies the training pod's resource requirements.", True),
    ("MLFLOW_ERROR", (r"mlflow", r"artifact", r"model registry", r"tracking uri"), "MLflow operation failed", "Training could not read or write the required MLflow run or model artifacts.", True),
    ("DATA_ERROR", (r"target column", r"feature column", r"dataset", r"no training rows", r"could not convert", r"unicode(?:decode)?error"), "Training data is invalid", "The dataset or training plan does not satisfy the model's input requirements.", False),
)


_GUIDANCE = {
    "PII_KEY_MISSING": (
        "Configure AVALOKA_PII_KEY in the API deployment secret.",
        "Alternatively, remove the named sensitive columns from the training plan.",
        "Create or confirm the training plan again after applying the change.",
    ),
    "TARGET_LEAKAGE": (
        "Remove any feature identified as a copy or direct encoding of the target column.",
        "Create a new training plan with only information that would exist when making a real prediction.",
        "Review the revised feature list, then start training again.",
    ),
    "DUPLICATE_ROWS": (
        "Remove or consolidate repeated records before training.",
        "Keep identifier columns out of the feature list; they do not make copied observations independent.",
        "Upload the cleaned dataset and create the training plan again.",
    ),
    "IDENTIFIER_FEATURE": (
        "Remove the identified row ID column from the selected features.",
        "Use attributes that will also be available for future records.",
        "Review the revised feature list, then start training again.",
    ),
    "OOM_KILLED": (
        "Increase the training worker memory limit, or reduce the batch size.",
        "Reduce the number of simultaneous workers if the cluster is memory constrained.",
        "Retry training after the resource change.",
    ),
    "TRAINING_TIMEOUT": (
        "Increase the training or scheduled-task timeout.",
        "Alternatively, reduce the epoch count or train on a smaller dataset.",
        "Retry the training run.",
    ),
    "IMAGE_PULL_FAILED": (
        "Verify that the configured training image name and tag exist in the registry.",
        "Confirm that the Kubernetes cluster has permission to pull from that registry.",
        "Correct the image setting, then retry training.",
    ),
    "EPHEMERAL_STORAGE_FULL": (
        "Increase the pod or node ephemeral-storage allocation.",
        "Remove unneeded temporary files, completed pods, or unused container images.",
        "Retry training after storage is available.",
    ),
    "ARTIFACT_STORAGE_FULL": (
        "Remove unused MLflow runs or old model artifacts if they are no longer needed.",
        "Increase the artifact-store bucket quota or persistent-volume capacity.",
        "Retry training after confirming that MLflow can write a test artifact.",
    ),
    "CLOUD_STORAGE_QUOTA_EXCEEDED": (
        "Open the configured cloud account and check the storage service quota and billing status.",
        "Delete unneeded stored data or request/increase the affected storage quota.",
        "Retry training after the cloud provider reports available capacity.",
    ),
    "STORAGE_FULL": (
        "Check the training pod, MLflow artifact store, and configured cloud storage for available capacity.",
        "Delete unneeded data or increase the capacity of the full storage resource.",
        "Retry training after verifying that the resource is writable.",
    ),
    "POD_EVICTED": (
        "Check the Kubernetes pod events to identify memory, disk, or node pressure.",
        "Increase the affected node-pool capacity or reduce the training resource request.",
        "Retry after the cluster has healthy capacity.",
    ),
    "POD_UNSCHEDULABLE": (
        "Reduce the requested worker count, CPU, memory, or GPU resources.",
        "Alternatively, scale the Kubernetes node pool or increase the cloud compute quota.",
        "Retry after Kubernetes has a node that satisfies the request.",
    ),
    "MLFLOW_ERROR": (
        "Verify that the MLflow tracking server is healthy and reachable from the training cluster.",
        "Check the tracking URI, artifact-store credentials, and write permissions.",
        "Retry after a test run and artifact can be written successfully.",
    ),
    "DATA_ERROR": (
        "Confirm that the target and feature columns exist and contain usable values.",
        "Remove invalid rows or convert the file to CSV, JSON, or Parquet with a supported encoding.",
        "Review the training plan and retry with the corrected dataset.",
    ),
    "TRAINING_FAILED": (
        "Retry the training run once.",
        "If it fails again, give support the error reference shown below and the run time.",
    ),
}

_SAFE_SUMMARIES = {
    "PII_KEY_MISSING": "Training includes sensitive columns that require keyed protection, but no PII key is configured.",
    "TARGET_LEAKAGE": "One or more selected features reveal the target answer, so training was blocked.",
    "DUPLICATE_ROWS": "Repeated records could leak across the training and validation split, so training was blocked.",
    "IDENTIFIER_FEATURE": "A selected feature uniquely identifies rows instead of providing a reusable prediction signal, so training was blocked.",
    "OOM_KILLED": "The training container exceeded its memory limit.",
    "TRAINING_TIMEOUT": "The training run did not finish before its configured deadline.",
    "IMAGE_PULL_FAILED": "Kubernetes could not download the container image required for training.",
    "EPHEMERAL_STORAGE_FULL": "The training pod or Kubernetes node ran out of temporary disk space.",
    "ARTIFACT_STORAGE_FULL": "MLflow could not write the run or model artifacts because its storage is full.",
    "CLOUD_STORAGE_QUOTA_EXCEEDED": "The configured cloud storage service rejected the write because its quota or capacity was exhausted.",
    "STORAGE_FULL": "A storage resource required by training has no available capacity.",
    "POD_EVICTED": "Kubernetes evicted the training pod because the node was under resource pressure.",
    "POD_UNSCHEDULABLE": "No Kubernetes node currently satisfies the training pod's resource requirements.",
    "MLFLOW_ERROR": "Training could not read or write the required MLflow run or model artifacts.",
    "DATA_ERROR": "The dataset or training plan does not satisfy the model's input requirements.",
    "TRAINING_FAILED": "Training stopped because of an unclassified error.",
}

_STORAGE_CAPACITY_PATTERNS = (
    r"no space left on device", r"disk quota exceeded", r"storage quota",
    r"quota.?exceeded", r"quota has been exceeded", r"insufficient storage", r"resourceexhausted",
    r"storage(?:account)?(?:limit|capacity).*exceeded", r"storageaccountisfull",
    r"http[^\n]*\b507\b",
)
_EPHEMERAL_STORAGE_PATTERNS = (
    r"ephemeral-storage", r"diskpressure", r"disk pressure", r"local temporary storage",
)
_ARTIFACT_STORAGE_PATTERNS = (
    r"mlflow", r"artifact", r"model registry", r"persistentvolume", r"\bpvc\b",
)
_CLOUD_STORAGE_PATTERNS = (
    r"\bgs://", r"\bs3://", r"\babfs[s]?://", r"azure blob", r"storage account",
    r"cloud storage", r"gcs bucket", r"s3 bucket", r"blob container",
)


def _clean(value: Any, limit: int = 4000) -> str:
    text = str(value or "").strip()
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", text)
    return text[-limit:]


def _safe_column_names(values: Any) -> list[str]:
    """Return bounded column labels that are safe to include in Markdown."""
    if not isinstance(values, (list, tuple)):
        return []
    columns: list[str] = []
    for value in values[:20]:
        if not isinstance(value, str):
            continue
        column = re.sub(r"[\x00-\x1f\x7f]+", " ", value).strip()[:128]
        if column and column not in columns:
            columns.append(column)
    return columns


def _identifier_feature_columns(error_text: str) -> list[str]:
    match = re.search(
        r"identifier-like feature detected:\s*feature column(?:\(s\)|s?)\s+(.+?)\s+"
        r"(?:(?:is|are) unique row identifiers|behave as row, occurrence, or grouping identifiers)",
        error_text,
        re.IGNORECASE,
    )
    if not match:
        return []
    try:
        parsed = ast.literal_eval(f"[{match.group(1)}]")
    except (SyntaxError, ValueError):
        return []
    return _safe_column_names(parsed)


def _target_leakage_columns(error_text: str) -> list[str]:
    """Extract columns from the deterministic target-leakage trainer error."""
    match = re.search(
        r"feature column(?:\(s\)|s?)\s+(.+?)\s+"
        r"(?:duplicates?|duplicate or almost perfectly encodes?)\s+target",
        error_text,
        re.IGNORECASE,
    )
    if not match:
        return []
    try:
        parsed = ast.literal_eval(f"[{match.group(1)}]")
    except (SyntaxError, ValueError):
        return []
    return _safe_column_names(parsed)


def _identifier_feature_message_parts(columns: Sequence[str]) -> tuple[str, list[str]]:
    safe_columns = _safe_column_names(list(columns))
    if not safe_columns:
        return _SAFE_SUMMARIES["IDENTIFIER_FEATURE"], list(_GUIDANCE["IDENTIFIER_FEATURE"])
    labels = ", ".join("`" + column.replace("`", "'") + "`" for column in safe_columns)
    noun = "Feature column" if len(safe_columns) == 1 else "Feature columns"
    summary = f"{noun} {labels} uniquely identifies rows instead of providing a reusable prediction signal, so training was blocked."
    actions = list(_GUIDANCE["IDENTIFIER_FEATURE"])
    actions[0] = f"Remove {labels} from the selected features."
    return summary, actions


def _target_leakage_message_parts(columns: Sequence[str]) -> tuple[str, list[str]]:
    safe_columns = _safe_column_names(list(columns))
    if not safe_columns:
        return _SAFE_SUMMARIES["TARGET_LEAKAGE"], list(_GUIDANCE["TARGET_LEAKAGE"])
    labels = ", ".join("`" + column.replace("`", "'") + "`" for column in safe_columns)
    noun = "Feature column" if len(safe_columns) == 1 else "Feature columns"
    summary = f"{noun} {labels} reveals the target answer, so training was blocked."
    actions = list(_GUIDANCE["TARGET_LEAKAGE"])
    actions[0] = f"Remove {labels} from the selected features."
    return summary, actions


def _duplicate_row_details(error_text: str) -> tuple[Optional[int], Optional[float], list[str]]:
    match = re.search(
        r"duplicate training rows detected:\s*(\d+) repeated row\(s\)\s*"
        r"\(([\d.]+)%\)(?: after excluding identifier column\(s\) (.+?))?\.\s*"
        r"training was blocked",
        error_text,
        re.IGNORECASE,
    )
    if not match:
        return None, None, []
    excluded: list[str] = []
    if match.group(3):
        try:
            excluded = _safe_column_names(ast.literal_eval(f"[{match.group(3)}]"))
        except (SyntaxError, ValueError):
            excluded = []
    return int(match.group(1)), float(match.group(2)) / 100.0, excluded


def _duplicate_row_message_parts(
    duplicate_count: Any,
    duplicate_ratio: Any,
    excluded_identifiers: Any,
) -> tuple[str, list[str]]:
    try:
        count = max(0, int(duplicate_count))
        ratio = min(1.0, max(0.0, float(duplicate_ratio)))
    except (TypeError, ValueError):
        return _SAFE_SUMMARIES["DUPLICATE_ROWS"], list(_GUIDANCE["DUPLICATE_ROWS"])
    excluded = _safe_column_names(excluded_identifiers)
    summary = f"{count} repeated record(s) ({ratio:.1%}) could leak across the training and validation split, so training was blocked."
    if excluded:
        labels = ", ".join("`" + column.replace("`", "'") + "`" for column in excluded)
        summary += f" Identifier column(s) {labels} were ignored when comparing records."
    return summary, list(_GUIDANCE["DUPLICATE_ROWS"])


def _root_cause_from_logs(logs: str) -> str:
    lines = [line.strip() for line in logs.splitlines() if line.strip()]
    actionable = [
        line for line in lines
        if re.search(r"(?i)(oomkilled|out of memory|error|exception|failed|timed? out|traceback|killed)", line)
    ]
    return _clean(actionable[-1], limit=1200) if actionable else ""


def _matches_any(patterns: Sequence[str], text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _provider_from_context(explicit: Optional[str], searchable: str) -> Optional[str]:
    value = (explicit or os.getenv("CLOUD_PROVIDER") or "").strip().lower()
    if value in {"gcp", "aws", "azure", "local"}:
        return value
    if re.search(r"(?i)(?:\bgs://|gcs bucket|google cloud)", searchable):
        return "gcp"
    if re.search(r"(?i)(?:\bs3://|amazon s3|\baws\b)", searchable):
        return "aws"
    if re.search(r"(?i)(?:\babfs[s]?://|azure blob|storage account)", searchable):
        return "azure"
    return None


def _storage_failure(searchable: str) -> Optional[tuple[str, str, str, bool]]:
    capacity_exhausted = _matches_any(_STORAGE_CAPACITY_PATTERNS, searchable)
    ephemeral_pressure = _matches_any(_EPHEMERAL_STORAGE_PATTERNS, searchable)
    if ephemeral_pressure and (
        capacity_exhausted or re.search(r"(?i)(?:evicted|node was low on resource)", searchable)
    ):
        return (
            "EPHEMERAL_STORAGE_FULL", "Temporary training storage is full",
            "The training pod or Kubernetes node ran out of temporary disk space.", True,
        )
    if not capacity_exhausted:
        return None
    if _matches_any(_CLOUD_STORAGE_PATTERNS, searchable):
        return (
            "CLOUD_STORAGE_QUOTA_EXCEEDED", "Cloud storage quota was exceeded",
            "The configured cloud storage service rejected the write because its quota or capacity was exhausted.", True,
        )
    if _matches_any(_ARTIFACT_STORAGE_PATTERNS, searchable):
        return (
            "ARTIFACT_STORAGE_FULL", "Model artifact storage is full",
            "MLflow could not write the run or model artifacts because its storage is full.", True,
        )
    return (
        "STORAGE_FULL", "Training storage is full",
        "A storage resource required by training has no available capacity.", True,
    )


def _provider_guidance(code: str, provider: Optional[str]) -> tuple[str, ...]:
    actions = tuple(_GUIDANCE.get(code, _GUIDANCE["TRAINING_FAILED"]))
    if code != "CLOUD_STORAGE_QUOTA_EXCEEDED" or not provider:
        return actions
    provider_action = {
        "gcp": "In Google Cloud, check the affected Cloud Storage/service quota and project billing.",
        "aws": "In AWS, check the affected S3/service quota and account billing.",
        "azure": "In Azure, check the affected storage account quota and subscription billing.",
        "local": "Increase Docker or Kubernetes disk allocation, or remove unused local artifacts.",
    }.get(provider)
    return (provider_action, *actions[1:]) if provider_action else actions


def build_training_failure(
    error: Any,
    *,
    logs: Any = "",
    kubernetes: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
    namespace: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify raw Ray/Kubernetes failures into a stable API/UI contract."""
    kube = dict(kubernetes or {})
    logs_tail = _clean(logs)
    technical_details = _clean(kube.get("message")) or _clean(error) or "Training failed without an error message."
    log_cause = _root_cause_from_logs(logs_tail)
    if log_cause and re.search(r"(?i)^ray job (?:finished|failed)|^training failed", technical_details):
        technical_details = log_cause
    searchable = "\n".join(str(value or "") for value in (
        kube.get("reason"), kube.get("message"), kube.get("exit_code"), technical_details, logs_tail,
    )).lower()

    safe_provider = _provider_from_context(provider, searchable)
    storage_failure = _storage_failure(searchable)
    code, title, summary, retryable = (
        storage_failure
        or ("TRAINING_FAILED", "Model training failed", "Training stopped because of an unclassified error.", False)
    )
    if not storage_failure:
        for candidate_code, patterns, candidate_title, candidate_summary, candidate_retryable in _RULES:
            if not _matches_any(patterns, searchable):
                continue
            code, title, summary, retryable = candidate_code, candidate_title, candidate_summary, candidate_retryable
            break
    if code == "IDENTIFIER_FEATURE":
        affected_columns = _identifier_feature_columns(technical_details)
    elif code == "TARGET_LEAKAGE":
        affected_columns = _target_leakage_columns(technical_details)
    else:
        affected_columns = []
    duplicate_count: Optional[int] = None
    duplicate_ratio: Optional[float] = None
    excluded_identifier_columns: list[str] = []
    if code == "IDENTIFIER_FEATURE":
        summary, actions = _identifier_feature_message_parts(affected_columns)
    elif code == "TARGET_LEAKAGE":
        summary, actions = _target_leakage_message_parts(affected_columns)
    elif code == "DUPLICATE_ROWS":
        duplicate_count, duplicate_ratio, excluded_identifier_columns = _duplicate_row_details(technical_details)
        summary, actions = _duplicate_row_message_parts(
            duplicate_count, duplicate_ratio, excluded_identifier_columns,
        )
    else:
        actions = list(_provider_guidance(code, safe_provider))
    return {
        "code": code, "title": title, "summary": summary,
        "actions": actions, "user_message": format_training_failure_parts(code, summary, actions),
        "affected_columns": affected_columns,
        "duplicate_count": duplicate_count,
        "duplicate_ratio": duplicate_ratio,
        "excluded_identifier_columns": excluded_identifier_columns,
        "provider": safe_provider,
        "technical_details": technical_details, "retryable": retryable,
        "job_id": job_id, "namespace": namespace, "pod_name": kube.get("pod_name"),
        "container_name": kube.get("container_name"), "kubernetes_reason": kube.get("reason"),
        "exit_code": kube.get("exit_code"), "logs_tail": logs_tail or None,
    }


def format_training_failure_parts(code: str, summary: str, actions: Sequence[str]) -> str:
    """Build a user-safe Markdown message from already-sanitized fields."""
    safe_code = code if re.fullmatch(r"[A-Z0-9_]+", str(code or "")) else "TRAINING_FAILED"
    safe_summary = str(summary or "Training stopped because of an unclassified error.").strip()
    safe_actions = [str(action).strip() for action in actions if str(action).strip()]
    steps = "\n".join(f"{index}. {action}" for index, action in enumerate(safe_actions, start=1))
    return (
        f"**{TRAINING_FAILURE_MESSAGE}**\n\n"
        f"**Cause:** {safe_summary}\n\n"
        f"**What to do:**\n{steps}\n\n"
        f"**Error reference:** `{safe_code}`"
    )


def format_training_failure(failure: Dict[str, Any]) -> str:
    """Return actionable guidance without exposing raw diagnostics or identifiers."""
    code = str(failure.get("code") or "TRAINING_FAILED")
    if code not in _SAFE_SUMMARIES:
        code = "TRAINING_FAILED"
    if code == "IDENTIFIER_FEATURE":
        summary, actions = _identifier_feature_message_parts(failure.get("affected_columns") or [])
    elif code == "TARGET_LEAKAGE":
        summary, actions = _target_leakage_message_parts(failure.get("affected_columns") or [])
    elif code == "DUPLICATE_ROWS":
        summary, actions = _duplicate_row_message_parts(
            failure.get("duplicate_count"),
            failure.get("duplicate_ratio"),
            failure.get("excluded_identifier_columns") or [],
        )
    else:
        summary = _SAFE_SUMMARIES[code]
        actions = _provider_guidance(code, failure.get("provider"))
    return format_training_failure_parts(code, summary, actions)
