"""Durable state for training jobs, registry records and serving endpoints.

Backed by the ``ml_jobs`` / ``ml_models`` / ``ml_endpoints`` /
``ml_cross_validation`` tables (migration 20261117130000), written through
the same Supabase service-role client ``scheduled_runs`` uses
(``app/core/celery_app.py``). Nothing here is cached in Redis: the Redis
outage of 2026-11 showed that state kept only there vanishes with it.

Two implementations share one interface:

* :class:`SupabaseMLStore` -- production; every write is an upsert keyed on the
  row's primary key, so a redelivered Celery task cannot duplicate a row.
* :class:`InMemoryMLStore` -- tests and ``AVALOKA_ML_STORE=memory`` dev runs.

The process-wide store comes from :func:`get_ml_store`; tests swap it with
:func:`set_ml_store`. Rows hold no customer data beyond identifiers the user
already owns in MLflow (run ids, registered names): no column names, file
names or dataset URIs.
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

JOB_KINDS = ("training", "register", "deploy_endpoint", "stop_endpoint", "cross_validate", "reap")
JOB_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")
TERMINAL_JOB_STATUSES = ("succeeded", "failed", "cancelled")

_JOBS = os.getenv("AVALOKA_ML_JOBS_TABLE", "ml_jobs")
_MODELS = os.getenv("AVALOKA_ML_MODELS_TABLE", "ml_models")
_ENDPOINTS = os.getenv("AVALOKA_ML_ENDPOINTS_TABLE", "ml_endpoints")
_CV = os.getenv("AVALOKA_ML_CV_TABLE", "ml_cross_validation")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None or isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #

@dataclass
class MLJob:
    user_id: str
    kind: str
    status: str = "queued"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    ray_job_name: Optional[str] = None
    celery_task_id: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    created_at: datetime = field(default_factory=_now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {self.kind!r}")
        if self.status not in JOB_STATUSES:
            raise ValueError(f"unknown job status {self.status!r}")

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_JOB_STATUSES


@dataclass
class MLModel:
    mlflow_run_id: str
    user_id: str
    registered_name: str
    version: int
    task_type: str
    alias: Optional[str] = None
    requires_gpu: bool = False
    signature_logged: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)


@dataclass
class MLEndpoint:
    mlflow_run_id: str
    user_id: str
    status: str = "pending"
    server: str = "cpu"
    model_uri: Optional[str] = None
    error: Optional[str] = None
    last_request_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


@dataclass
class MLCrossValidation:
    mlflow_run_id: str
    user_id: str
    status: str = "queued"
    k: Optional[int] = None
    splitter: Optional[str] = None
    reason: Optional[str] = None
    aggregate: Optional[Dict[str, Any]] = None
    folds: Optional[List[Dict[str, Any]]] = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


def row_of(obj: Any) -> Dict[str, Any]:
    """A JSON-safe dict of a row dataclass (datetimes as ISO strings)."""
    return {k: _iso(v) for k, v in asdict(obj).items()}


def _from_row(cls: type, row: Dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for name in names:
        if name not in row:
            continue
        value = row[name]
        if name.endswith("_at"):
            value = _parse_dt(value)
        kwargs[name] = value
    return cls(**kwargs)


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #

class MLStore(Protocol):
    # jobs
    def create_job(self, job: MLJob) -> MLJob: ...
    def get_job(self, job_id: str) -> Optional[MLJob]: ...
    def update_job(self, job_id: str, **changes: Any) -> Optional[MLJob]: ...
    def list_jobs(self, user_id: str, *, mlflow_run_id: Optional[str] = None,
                  kind: Optional[str] = None, limit: int = 50) -> List[MLJob]: ...
    def active_jobs(self, *, kind: Optional[str] = None) -> List[MLJob]: ...
    # registry records
    def upsert_model(self, model: MLModel) -> MLModel: ...
    def get_model(self, mlflow_run_id: str) -> Optional[MLModel]: ...
    def list_models(self, user_id: str) -> List[MLModel]: ...
    def delete_model(self, mlflow_run_id: str) -> None: ...
    # endpoints
    def upsert_endpoint(self, endpoint: MLEndpoint) -> MLEndpoint: ...
    def get_endpoint(self, mlflow_run_id: str) -> Optional[MLEndpoint]: ...
    def list_endpoints(self, user_id: str) -> List[MLEndpoint]: ...
    def touch_endpoint(self, mlflow_run_id: str, when: Optional[datetime] = None) -> None: ...
    def delete_endpoint(self, mlflow_run_id: str) -> None: ...
    # cross-validation
    def upsert_cross_validation(self, cv: MLCrossValidation) -> MLCrossValidation: ...
    def get_cross_validation(self, mlflow_run_id: str) -> Optional[MLCrossValidation]: ...


# --------------------------------------------------------------------------- #
# In-memory (tests, dev)
# --------------------------------------------------------------------------- #

class InMemoryMLStore:
    """Dict-backed store with the production store's semantics (upsert, order)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.jobs: Dict[str, MLJob] = {}
        self.models: Dict[str, MLModel] = {}
        self.endpoints: Dict[str, MLEndpoint] = {}
        self.cross_validation: Dict[str, MLCrossValidation] = {}

    # jobs
    def create_job(self, job: MLJob) -> MLJob:
        with self._lock:
            self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> Optional[MLJob]:
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **changes: Any) -> Optional[MLJob]:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            for key, value in changes.items():
                if not hasattr(job, key):
                    raise AttributeError(f"MLJob has no field {key!r}")
                setattr(job, key, value)
            if "status" in changes and changes["status"] not in JOB_STATUSES:
                raise ValueError(f"unknown job status {changes['status']!r}")
            job.updated_at = _now()
            return job

    def list_jobs(self, user_id: str, *, mlflow_run_id: Optional[str] = None,
                  kind: Optional[str] = None, limit: int = 50) -> List[MLJob]:
        rows = [j for j in self.jobs.values() if j.user_id == user_id
                and (mlflow_run_id is None or j.mlflow_run_id == mlflow_run_id)
                and (kind is None or j.kind == kind)]
        rows.sort(key=lambda j: j.created_at, reverse=True)
        return rows[:limit]

    def active_jobs(self, *, kind: Optional[str] = None) -> List[MLJob]:
        return [j for j in self.jobs.values() if not j.terminal and (kind is None or j.kind == kind)]

    # models
    def upsert_model(self, model: MLModel) -> MLModel:
        with self._lock:
            self.models[model.mlflow_run_id] = model
        return model

    def get_model(self, mlflow_run_id: str) -> Optional[MLModel]:
        return self.models.get(mlflow_run_id)

    def list_models(self, user_id: str) -> List[MLModel]:
        return [m for m in self.models.values() if m.user_id == user_id]

    def delete_model(self, mlflow_run_id: str) -> None:
        with self._lock:
            self.models.pop(mlflow_run_id, None)

    # endpoints
    def upsert_endpoint(self, endpoint: MLEndpoint) -> MLEndpoint:
        with self._lock:
            endpoint.updated_at = _now()
            self.endpoints[endpoint.mlflow_run_id] = endpoint
        return endpoint

    def get_endpoint(self, mlflow_run_id: str) -> Optional[MLEndpoint]:
        return self.endpoints.get(mlflow_run_id)

    def list_endpoints(self, user_id: str) -> List[MLEndpoint]:
        return [e for e in self.endpoints.values() if e.user_id == user_id]

    def touch_endpoint(self, mlflow_run_id: str, when: Optional[datetime] = None) -> None:
        with self._lock:
            ep = self.endpoints.get(mlflow_run_id)
            if ep is not None:
                ep.last_request_at = when or _now()

    def delete_endpoint(self, mlflow_run_id: str) -> None:
        with self._lock:
            self.endpoints.pop(mlflow_run_id, None)

    # cv
    def upsert_cross_validation(self, cv: MLCrossValidation) -> MLCrossValidation:
        with self._lock:
            cv.updated_at = _now()
            self.cross_validation[cv.mlflow_run_id] = cv
        return cv

    def get_cross_validation(self, mlflow_run_id: str) -> Optional[MLCrossValidation]:
        return self.cross_validation.get(mlflow_run_id)


# --------------------------------------------------------------------------- #
# Supabase / Postgres
# --------------------------------------------------------------------------- #

class SupabaseMLStore:
    """Production store over the service-role Supabase client.

    Each method issues one request. Writes are upserts on the primary key.
    Read failures return ``None``/``[]`` after logging; write failures raise,
    because a lost state write is a bug the caller must see.
    """

    def __init__(self, client: Any = None) -> None:
        self._client = client

    def _c(self) -> Any:
        if self._client is None:
            from app.agents.sampling_persistence import get_supabase_client
            self._client = get_supabase_client()
        return self._client

    @staticmethod
    def _rows(result: Any) -> List[Dict[str, Any]]:
        data = getattr(result, "data", None)
        if data is None and isinstance(result, dict):
            data = result.get("data")
        return list(data or [])

    # jobs
    def create_job(self, job: MLJob) -> MLJob:
        self._c().table(_JOBS).upsert(row_of(job)).execute()
        return job

    def get_job(self, job_id: str) -> Optional[MLJob]:
        try:
            rows = self._rows(self._c().table(_JOBS).select("*").eq("id", job_id).limit(1).execute())
        except Exception:  # noqa: BLE001 - a read miss is reported as absent
            logger.warning("[ml_store] get_job failed", exc_info=True)
            return None
        return _from_row(MLJob, rows[0]) if rows else None

    def update_job(self, job_id: str, **changes: Any) -> Optional[MLJob]:
        payload = {k: _iso(v) for k, v in changes.items()}
        payload["updated_at"] = _now().isoformat()
        rows = self._rows(self._c().table(_JOBS).update(payload).eq("id", job_id).execute())
        return _from_row(MLJob, rows[0]) if rows else None

    def list_jobs(self, user_id: str, *, mlflow_run_id: Optional[str] = None,
                  kind: Optional[str] = None, limit: int = 50) -> List[MLJob]:
        try:
            q = self._c().table(_JOBS).select("*").eq("user_id", user_id)
            if mlflow_run_id is not None:
                q = q.eq("mlflow_run_id", mlflow_run_id)
            if kind is not None:
                q = q.eq("kind", kind)
            rows = self._rows(q.order("created_at", desc=True).limit(limit).execute())
        except Exception:  # noqa: BLE001
            logger.warning("[ml_store] list_jobs failed", exc_info=True)
            return []
        return [_from_row(MLJob, r) for r in rows]

    def active_jobs(self, *, kind: Optional[str] = None) -> List[MLJob]:
        try:
            q = self._c().table(_JOBS).select("*").in_("status", ["queued", "running"])
            if kind is not None:
                q = q.eq("kind", kind)
            rows = self._rows(q.execute())
        except Exception:  # noqa: BLE001
            logger.warning("[ml_store] active_jobs failed", exc_info=True)
            return []
        return [_from_row(MLJob, r) for r in rows]

    # models
    def upsert_model(self, model: MLModel) -> MLModel:
        self._c().table(_MODELS).upsert(row_of(model)).execute()
        return model

    def get_model(self, mlflow_run_id: str) -> Optional[MLModel]:
        try:
            rows = self._rows(self._c().table(_MODELS).select("*").eq("mlflow_run_id", mlflow_run_id).limit(1).execute())
        except Exception:  # noqa: BLE001
            logger.warning("[ml_store] get_model failed", exc_info=True)
            return None
        return _from_row(MLModel, rows[0]) if rows else None

    def list_models(self, user_id: str) -> List[MLModel]:
        try:
            rows = self._rows(self._c().table(_MODELS).select("*").eq("user_id", user_id).execute())
        except Exception:  # noqa: BLE001
            logger.warning("[ml_store] list_models failed", exc_info=True)
            return []
        return [_from_row(MLModel, r) for r in rows]

    def delete_model(self, mlflow_run_id: str) -> None:
        self._c().table(_MODELS).delete().eq("mlflow_run_id", mlflow_run_id).execute()

    # endpoints
    def upsert_endpoint(self, endpoint: MLEndpoint) -> MLEndpoint:
        endpoint.updated_at = _now()
        self._c().table(_ENDPOINTS).upsert(row_of(endpoint)).execute()
        return endpoint

    def get_endpoint(self, mlflow_run_id: str) -> Optional[MLEndpoint]:
        try:
            rows = self._rows(self._c().table(_ENDPOINTS).select("*").eq("mlflow_run_id", mlflow_run_id).limit(1).execute())
        except Exception:  # noqa: BLE001
            logger.warning("[ml_store] get_endpoint failed", exc_info=True)
            return None
        return _from_row(MLEndpoint, rows[0]) if rows else None

    def list_endpoints(self, user_id: str) -> List[MLEndpoint]:
        try:
            rows = self._rows(self._c().table(_ENDPOINTS).select("*").eq("user_id", user_id).execute())
        except Exception:  # noqa: BLE001
            logger.warning("[ml_store] list_endpoints failed", exc_info=True)
            return []
        return [_from_row(MLEndpoint, r) for r in rows]

    def touch_endpoint(self, mlflow_run_id: str, when: Optional[datetime] = None) -> None:
        try:
            (self._c().table(_ENDPOINTS)
             .update({"last_request_at": (when or _now()).isoformat()})
             .eq("mlflow_run_id", mlflow_run_id).execute())
        except Exception:  # noqa: BLE001 - activity marking never fails a prediction
            logger.warning("[ml_store] touch_endpoint failed", exc_info=True)

    def delete_endpoint(self, mlflow_run_id: str) -> None:
        self._c().table(_ENDPOINTS).delete().eq("mlflow_run_id", mlflow_run_id).execute()

    # cv
    def upsert_cross_validation(self, cv: MLCrossValidation) -> MLCrossValidation:
        cv.updated_at = _now()
        self._c().table(_CV).upsert(row_of(cv)).execute()
        return cv

    def get_cross_validation(self, mlflow_run_id: str) -> Optional[MLCrossValidation]:
        try:
            rows = self._rows(self._c().table(_CV).select("*").eq("mlflow_run_id", mlflow_run_id).limit(1).execute())
        except Exception:  # noqa: BLE001
            logger.warning("[ml_store] get_cross_validation failed", exc_info=True)
            return None
        return _from_row(MLCrossValidation, rows[0]) if rows else None


# --------------------------------------------------------------------------- #
# Process-wide store
# --------------------------------------------------------------------------- #

_STORE: Optional[Any] = None
_STORE_LOCK = threading.Lock()


def get_ml_store() -> Any:
    """The configured store.

    ``AVALOKA_ML_STORE=memory`` selects the in-memory store (tests, a laptop
    without Supabase). Otherwise the Supabase store is used; a missing Supabase
    configuration is reported once and falls back to memory so a dev API still
    starts -- production always has Supabase, and the warning names the loss.
    """
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            return _STORE
        mode = (os.getenv("AVALOKA_ML_STORE") or "").strip().lower()
        if mode == "memory":
            _STORE = InMemoryMLStore()
        elif os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
            _STORE = SupabaseMLStore()
        else:
            logger.warning(
                "[ml_store] SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY unset: training/endpoint "
                "state is held in this process only and is lost on restart. Set them, or "
                "AVALOKA_ML_STORE=memory to silence this on a dev box."
            )
            _STORE = InMemoryMLStore()
        return _STORE


def set_ml_store(store: Optional[Any]) -> None:
    """Replace the process-wide store (tests). ``None`` resets to lazy selection."""
    global _STORE
    with _STORE_LOCK:
        _STORE = store
