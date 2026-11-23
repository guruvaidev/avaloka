"""The MLflow model registry, owner-scoped, idempotent.

Every successful training run becomes a *version* of a registered model named
``avaloka.{user_id}.{slug(model_name)}`` (one registered model per user and
model name; the run id is always in the version tags). Nothing is staged:
MLflow 3 deprecates stages, so the one alias ``serving`` marks the version an
endpoint serves. Stages are never used.

Idempotency: a run that already has a version is reused (``search_model_versions
("run_id = ...")``); ``create_registered_model`` tolerates
``RESOURCE_ALREADY_EXISTS``; the Postgres mirror (``ml_models``) is an upsert on
the run id. A slug collision -- two different model names of the same user
mapping to one slug -- gets ``-{run_id[:8]}`` appended (coordinator ruling O-2).

Credentials: ``MlflowClient()`` reads ``MLFLOW_TRACKING_USERNAME`` /
``MLFLOW_TRACKING_PASSWORD`` from the environment (the basic-auth hotfix puts
them in the chart Secret). Nothing here handles or logs a credential.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional

from app.core.ml_store import MLModel, get_ml_store

logger = logging.getLogger(__name__)

SERVING_ALIAS = "serving"
NAME_PREFIX = os.getenv("AVALOKA_REGISTRY_PREFIX", "avaloka")
MODEL_ARTIFACT_PATH = "model"          # mlflow.pytorch.log_model(..., artifact_path="model")
_SLUG_MAX = 40
_METRIC_KEYS = (
    "final_train_loss", "final_val_loss", "final_accuracy", "final_f1_score",
    "final_mae", "final_rmse", "final_r2_score", "num_epochs_trained", "rows_processed",
)


def slug(text: Any) -> str:
    """Lower-case ``[a-z0-9_-]`` slug, at most 40 chars, never empty."""
    s = re.sub(r"[^a-z0-9_-]+", "-", str(text or "").strip().lower()).strip("-_")
    s = re.sub(r"-{2,}", "-", s)[:_SLUG_MAX].strip("-_")
    return s or "model"


def registered_name(user_id: str, model_name: Any) -> str:
    return f"{NAME_PREFIX}.{user_id}.{slug(model_name)}"


def _client(client: Any = None) -> Any:
    if client is not None:
        return client
    import mlflow
    from mlflow.tracking import MlflowClient

    from app.agents.mta_v2.tracking import resolve_tracking_uri

    mlflow.set_tracking_uri(resolve_tracking_uri(None))
    return MlflowClient()


def _error_code(exc: BaseException) -> str:
    code = getattr(exc, "error_code", None)
    if code:
        return str(code)
    # RestException wraps the code in the message as well.
    return "RESOURCE_ALREADY_EXISTS" if "RESOURCE_ALREADY_EXISTS" in str(exc) else ""


def _task_type(training_result: Dict[str, Any]) -> str:
    return "regression" if "regression" in str(training_result.get("model_type", "")).lower() else "classification"


def _metrics_snapshot(training_result: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in _METRIC_KEYS:
        value = training_result.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = value
    return out


def _ensure_registered_model(client: Any, name: str, user_id: str, model_name: str, run_id: str) -> str:
    """Create the registered model if needed; resolve a slug collision. Returns the name to use."""
    tags = {"avaloka.user_id": str(user_id), "avaloka.model_name": str(model_name)}
    try:
        client.create_registered_model(name, tags=tags, description=f"Avaloka model '{model_name}'")
        return name
    except Exception as exc:  # noqa: BLE001 - only "already exists" is tolerated
        if _error_code(exc) != "RESOURCE_ALREADY_EXISTS":
            raise
    try:
        existing = client.get_registered_model(name)
        existing_tags = dict(getattr(existing, "tags", {}) or {})
    except Exception:  # noqa: BLE001
        existing_tags = {}
    owner = existing_tags.get("avaloka.model_name")
    if owner not in (None, "", str(model_name)):
        # Two different model names collapsed onto one slug: keep them apart.
        collided = f"{name}-{run_id[:8]}"
        logger.info("[registry] slug collision on %s (%r vs %r); using %s", name, owner, model_name, collided)
        try:
            client.create_registered_model(collided, tags=tags, description=f"Avaloka model '{model_name}'")
        except Exception as exc:  # noqa: BLE001
            if _error_code(exc) != "RESOURCE_ALREADY_EXISTS":
                raise
        return collided
    return name


def register_run(user_id: str, run_id: str, training_result: Dict[str, Any], *,
                 client: Any = None, store: Any = None) -> Dict[str, Any]:
    """Make ``run_id`` a registered model version. Safe to call twice."""
    client = _client(client)
    store = store or get_ml_store()
    model_name = training_result.get("model_name") or "model"
    task_type = _task_type(training_result)
    requires_gpu = str(training_result.get("requires_gpu", "")).lower() in {"1", "true", "yes"}

    existing = list(client.search_model_versions(f"run_id = '{run_id}'") or [])
    if existing:
        mv = existing[0]
        record = MLModel(mlflow_run_id=run_id, user_id=str(user_id), registered_name=mv.name,
                         version=int(mv.version), task_type=task_type, requires_gpu=requires_gpu,
                         signature_logged=bool(training_result.get("signature_logged", False)),
                         metrics=_metrics_snapshot(training_result))
        store.upsert_model(record)
        return {"registered_name": mv.name, "version": int(mv.version), "run_id": run_id, "created": False}

    name = _ensure_registered_model(client, registered_name(str(user_id), model_name), str(user_id), model_name, run_id)
    tags = {
        "avaloka.run_id": run_id,
        "avaloka.user_id": str(user_id),
        "avaloka.session_id": str(training_result.get("session_id") or ""),
        "avaloka.task_type": task_type,
        "avaloka.backend": str(training_result.get("execution_backend") or ("ray" if training_result.get("ray_job_id") else "local")),
        "avaloka.requires_gpu": "true" if requires_gpu else "false",
        "avaloka.model_name": str(model_name),
    }
    mv = client.create_model_version(
        name=name,
        source=f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}",
        run_id=run_id,
        tags=tags,
        description=str(training_result.get("model_description") or "")[:500] or None,
    )
    record = MLModel(mlflow_run_id=run_id, user_id=str(user_id), registered_name=name, version=int(mv.version),
                     task_type=task_type, requires_gpu=requires_gpu,
                     signature_logged=bool(training_result.get("signature_logged", False)),
                     metrics=_metrics_snapshot(training_result))
    store.upsert_model(record)
    logger.info("[registry] %s v%s <- run %s", name, mv.version, run_id)
    return {"registered_name": name, "version": int(mv.version), "run_id": run_id, "created": True}


def _record(run_id: str, store: Any) -> Optional[MLModel]:
    return store.get_model(run_id)


def set_serving_alias(run_id: str, *, client: Any = None, store: Any = None) -> Optional[Dict[str, Any]]:
    """Point ``serving`` at this run's version (moves it off a sibling version)."""
    store = store or get_ml_store()
    record = _record(run_id, store)
    if record is None:
        return None
    client = _client(client)
    client.set_registered_model_alias(record.registered_name, SERVING_ALIAS, str(record.version))
    record.alias = SERVING_ALIAS
    store.upsert_model(record)
    for sibling in store.list_models(record.user_id):
        if sibling.registered_name == record.registered_name and sibling.mlflow_run_id != run_id and sibling.alias == SERVING_ALIAS:
            sibling.alias = None
            store.upsert_model(sibling)
    return {"registered_name": record.registered_name, "version": record.version, "alias": SERVING_ALIAS}


def clear_serving_alias(run_id: str, *, client: Any = None, store: Any = None) -> None:
    """Remove ``serving`` only if it points at this run's version."""
    store = store or get_ml_store()
    record = _record(run_id, store)
    if record is None or record.alias != SERVING_ALIAS:
        return
    client = _client(client)
    try:
        current = client.get_model_version_by_alias(record.registered_name, SERVING_ALIAS)
        if str(getattr(current, "version", "")) == str(record.version):
            client.delete_registered_model_alias(record.registered_name, SERVING_ALIAS)
    except Exception as exc:  # noqa: BLE001 - an alias that is already gone is fine
        if "RESOURCE_DOES_NOT_EXIST" not in _error_code(exc) and "does not exist" not in str(exc).lower():
            raise
    record.alias = None
    store.upsert_model(record)


def delete_run_version(run_id: str, *, client: Any = None, store: Any = None) -> None:
    """Delete this run's version; the registered model too when it was the last one."""
    store = store or get_ml_store()
    record = _record(run_id, store)
    if record is None:
        return
    client = _client(client)
    clear_serving_alias(run_id, client=client, store=store)
    client.delete_model_version(record.registered_name, str(record.version))
    remaining = list(client.search_model_versions(f"name = '{record.registered_name}'") or [])
    if not remaining:
        client.delete_registered_model(record.registered_name)
    store.delete_model(run_id)
