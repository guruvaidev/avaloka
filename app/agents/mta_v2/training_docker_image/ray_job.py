"""
ray_job.py — Avaloka Ray Training Job
======================================
Trains a DynamicMLP on tabular data, exports .pth and .onnx artifacts,
and logs everything (params, per-epoch metrics, model files, metadata) to MLflow.
Inference is handled separately via inference.py + mlflow_manager.py.

Streaming / memory notes:
  - MLflow used for experiment tracking, metric logging, and artifact storage.
  - Class detection: distributed ray.remote tasks per block — never loads the full
    dataset into one place; aggregation is O(num_classes) not O(num_rows).
  - Train/val split: single-pass deterministic hash per row → materialize once →
    filter() into train/val. No double pipeline execution.
  - Imputation/scaling/category vocabularies are fitted globally on the training
    split only, then reused unchanged for validation and inference.
  - Categorical features use one-hot encoding with a dedicated unknown bucket.
  - Missing columns auto-removed from feature_cols if not present in dataset schema.
  - ds.count() is called for regression (always) and for classification when
    CLASS_NAMES is pre-supplied (skips the distributed discovery scan).
  - SQL loader uses server-side cursors (yield_per=1000).

Autoscaling notes:
  - Ray is initialised with address="auto" to attach to a running autoscaling cluster.
  - NUM_WORKERS (desired/initial) / MIN_WORKERS (floor) / MAX_WORKERS (ceiling) are
    custom env vars — they are NOT native Ray env vars. The true autoscaler floor/ceiling
    must also be set in your KubeRay RayCluster CR (minReplicas / maxReplicas).
  - concurrency=num_workers (single int) on map_batches controls max parallel workers.
    Tuple concurrency (min, max) is only valid for callable classes, NOT plain functions.
  - filter() and drop_columns() do NOT support concurrency — never pass it to them.
  - ds_tagged is materialised before forking into train/val to avoid executing the full
    preprocessing pipeline twice (once per consumer branch).
  - CPUS_PER_WORKER controls per-worker CPU reservation; keep it < node CPUs to leave
    headroom for Ray system processes.
  - DataContext resource_limits are set to inf so Ray Data can use all autoscaler nodes.

Env vars (MLflow):
  MLFLOW_TRACKING_URI       — tracking server URI (e.g. http://mlflow:5000)
  MLFLOW_BACKEND_STORE_URI  — backend store URI (sqlite, postgresql, etc.)
  MLFLOW_ARTIFACT_ROOT      — artifact root URI (gs://, s3://, or local path)
  MLFLOW_EXPERIMENT_NAME    — experiment name (default: "avaloka-training")

Env vars (Model identity — stored in MLflow tags, params, and model_config.json):
  MODEL_NAME                — human-readable model name (default: "dynamic-mlp")
  MODEL_DESCRIPTION         — free-text description of the model / training run
  MODEL_VERSION             — version string, e.g. "1", "2.0", "v3-prod"
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import time
import uuid

import mlflow
import mlflow.pytorch
import mlflow.exceptions
import pandas as pd
import psutil
import pyarrow
import pyarrow as pa
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import ray.data as rd
from ray import train
from ray.data import DataContext
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from sqlalchemy import create_engine, text

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from sklearn.metrics import f1_score as sk_f1_score

from src.model import DynamicMLP, ModelConfig
from src.data_integrity import (
    build_training_integrity_report,
    duplicate_rows_error,
    find_duplicate_training_rows,
    find_identifier_features,
    find_target_leakage,
    identifier_features_error,
    target_leakage_error,
)
from src.evaluation import (
    TIME_ORDER_COLUMN,
    add_time_order_column,
    build_holdout_evaluation_report,
    time_series_split_metadata,
)
from loader.sql import load_dataset_from_sql
from loader.url import load_dataset_from_url

try:
    from pyarrow.fs import GcsFileSystem
    _GCS_FS_AVAILABLE = True
except ImportError:
    GcsFileSystem = None  # type: ignore[assignment,misc]
    _GCS_FS_AVAILABLE = False


os.environ["RAY_SCHEDULER_SPREAD_THRESHOLD"] = "0.0"

PII_KEY_ENV = "AVALOKA_PII_KEY"


def _pseudonym(value: Any, *, column: str, key: str, length: int = 16) -> str:
    digest = hmac.new(
        key.encode("utf-8"),
        f"{column}\x00{value}".encode("utf-8"),
        hashlib.sha256,
    )
    return digest.hexdigest()[:length]


def transform_pii_value(
    value: Any,
    *,
    column: str,
    kind: str,
    strategy: str,
    key: str,
    reference_year: int = 2026,
) -> Any:
    """Portable counterpart of pii_agent.transform_pii_value for Ray jobs."""
    if value is None or bool(pd.isna(value)):
        return None
    if strategy == "keep":
        return value
    if strategy == "pseudonymise":
        if not key:
            raise RuntimeError(f"{PII_KEY_ENV} is required for PII pseudonymisation.")
        return _pseudonym(value, column=column, key=key)
    if strategy == "tokenise":
        if not key:
            raise RuntimeError(f"{PII_KEY_ENV} is required for PII tokenisation.")
        text = str(value)
        keep = 4
        if len(text) <= keep:
            return _pseudonym(value, column=column, key=key, length=len(text) or 4)
        head = _pseudonym(
            text[:-keep],
            column=column,
            key=key,
            length=max(len(text) - keep, 4),
        )
        return f"{head[:len(text) - keep]}{text[-keep:]}"
    if strategy == "generalise":
        if kind == "date_of_birth":
            stamp = pd.to_datetime(value, errors="coerce")
            if pd.isna(stamp):
                return None
            age = max(reference_year - int(stamp.year), 0)
            low = (age // 10) * 10
            return f"{low}-{low + 9}"
        if kind == "postcode":
            text = str(value).strip()
            return text[:3].upper() if text else None
        if not key:
            raise RuntimeError(f"{PII_KEY_ENV} is required for PII pseudonymisation.")
        return _pseudonym(value, column=column, key=key)
    if strategy == "redact":
        return "__redacted__"
    raise ValueError(f"Unsupported PII transformation strategy: {strategy!r}")


def transform_pii_batch(
    batch: pd.DataFrame,
    *,
    transformations: Dict[str, Dict[str, str]],
    key: str,
) -> pd.DataFrame:
    """Apply the persisted PII contract to one distributed pandas batch."""
    out = batch.copy()
    for column, metadata in transformations.items():
        if column not in out.columns:
            continue
        out[column] = out[column].map(
            lambda value, col=column, meta=metadata: transform_pii_value(
                value,
                column=col,
                kind=str(meta.get("kind", "none")),
                strategy=str(meta.get("strategy", "keep")),
                key=key,
            )
        )
    return out


def _compute_regression_metrics(labels: List[float], predictions: List[float]) -> Dict[str, float]:
    """Return globally aggregated MAE, RMSE, and R-squared values."""
    y_true = np.asarray(labels, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    errors = y_pred - y_true

    # Aggregate sufficient statistics so Ray workers report metrics for the
    # complete distributed validation set, not only rank zero's shard.
    stats = [
        float(y_true.size),
        float(np.sum(np.abs(errors))),
        float(np.sum(np.square(errors))),
        float(np.sum(y_true)),
        float(np.sum(np.square(y_true))),
    ]
    if dist.is_available() and dist.is_initialized():
        backend = dist.get_backend()
        stats_device = (
            torch.device("cuda", torch.cuda.current_device())
            if str(backend) == "nccl"
            else torch.device("cpu")
        )
        stats_tensor = torch.tensor(stats, dtype=torch.float64, device=stats_device)
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        stats = stats_tensor.cpu().tolist()

    count, absolute_error_sum, residual_sum_squares, target_sum, target_square_sum = stats
    if count <= 0:
        return {"mae": 0.0, "rmse": 0.0, "r2_score": 0.0}

    mae = absolute_error_sum / count
    rmse = float(np.sqrt(residual_sum_squares / count))
    total_sum_squares = target_square_sum - ((target_sum * target_sum) / count)
    if total_sum_squares <= np.finfo(np.float64).eps:
        r2 = 1.0 if residual_sum_squares <= np.finfo(np.float64).eps else 0.0
    else:
        r2 = 1.0 - (residual_sum_squares / total_sum_squares)

    return {
        "mae": round(float(mae), 6),
        "rmse": round(rmse, 6),
        "r2_score": round(float(r2), 6),
    }

cpus_per_worker = int(os.getenv("CPUS_PER_WORKER", 4))

# NUM_WORKERS  → desired/initial worker count passed to ScalingConfig and map_batches concurrency.
# MIN_WORKERS  → reserved for autoscaler configuration reference (not used directly in this file).
# MAX_WORKERS  → reserved for autoscaler configuration reference (not used directly in this file).
# NOTE: these are custom env vars, not native Ray variables.
# The actual autoscaler floor/ceiling lives in KubeRay minReplicas/maxReplicas.
# concurrency on map_batches is always set to the single int num_workers — tuple
# concurrency (min, max) is only valid for callable classes, not plain functions.
num_workers = int(os.getenv("NUM_WORKERS", 2))
min_workers = int(os.getenv("MIN_WORKERS", 1))
max_workers = int(os.getenv("MAX_WORKERS", num_workers * 2))


# =========================================================
# GCP Credentials
# =========================================================


def setup_gcp_credentials() -> Optional[str]:
    print("[GCP Credentials] Setting up GCP credentials for Ray worker...")
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "").strip()
    if sa_json:
        sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
        Path(sa_path).write_text(sa_json, encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        return sa_path
    return None


def get_storage_opts(uri: str) -> dict:
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "").strip()
    if sa_json and uri.startswith("gs://"):
        sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/gcp_sa.json")
        Path(sa_path).write_text(sa_json, encoding="utf-8")
        return {"token": sa_path}
    return {}


# =========================================================
# Init Ray  (autoscaling-aware)
# =========================================================


def init_ray() -> None:
    """
    Connect to an autoscaling Ray cluster.

    - address="auto"  → picks up the running cluster (GKE autoscaler, KubeRay, etc.)
    - ignore_reinit_error=True  → safe to call multiple times (e.g. local dev).
    - NOTE: _enable_object_reconstruction must NOT be passed when connecting to an
      existing cluster — it is only valid when Ray is starting a new cluster.
      Enable it at cluster level via rayStartParams instead.
    - DataContext resource_limits set to inf  → removes implicit CPU/GPU caps so
      Ray Data can freely use whatever nodes the autoscaler provisions.
    - locality_with_output=True  → schedules tasks on the node that consumes the
      output, reducing cross-node transfers and implicit buffering.
    """
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        runtime_env={"worker_process_setup_hook": setup_gcp_credentials},
    )

    ctx = DataContext.get_current()
    # Larger blocks reduce scheduling overhead on big clusters.
    ctx.target_max_block_size = 256 * 1024 * 1024
    ctx.execution_options.preserve_order = False

    # ExecutionResources fields are immutable in Ray 2.53. Replace the limits
    # object instead of assigning its cpu/gpu properties (the latter raises
    # ``AttributeError: property 'cpu' ... has no setter``).
    ctx.execution_options.resource_limits = (
        ctx.execution_options.resource_limits.for_limits()
    )

    # Prefer scheduling tasks on the node that will consume the output —
    # reduces cross-node object transfers and implicit memory buffering.
    ctx.execution_options.locality_with_output = True

    # Log cluster state at startup so we can see what the autoscaler gave us.
    nodes = ray.nodes()
    alive = [n for n in nodes if n.get("Alive")]
    print(f"[Ray Init] Cluster nodes alive={len(alive)} / total={len(nodes)}")
    resources = ray.cluster_resources()
    print(f"[Ray Init] Cluster resources: {resources}")


# =========================================================
# MLflow Setup
# =========================================================


def setup_mlflow() -> None:
    """
    Configure MLflow from environment variables.

    MLFLOW_TRACKING_URI      — where the client sends runs (client-side)
    MLFLOW_BACKEND_STORE_URI — where run metadata is persisted (server-side)
    MLFLOW_DEFAULT_ARTIFACT_ROOT     — where large files/models are stored (server-side)

    When MLFLOW_TRACKING_URI is set, it takes full precedence for the client.
    MLFLOW_BACKEND_STORE_URI / MLFLOW_DEFAULT_ARTIFACT_ROOT are used when spinning up
    an embedded tracking server or when pointing directly at a store (no server).
    """
    tracking_uri     = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    backend_store   = os.getenv("MLFLOW_BACKEND_STORE_URI", "").strip()
    artifact_root   = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "").strip()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "avaloka-training").strip()

    # Resolve effective tracking URI:
    # 1. Explicit MLFLOW_TRACKING_URI wins.
    # 2. Fall back to MLFLOW_BACKEND_STORE_URI (direct store access, no HTTP server needed).
    # 3. Default to local ./mlruns.
    effective_uri = tracking_uri or backend_store or "mlruns"
    mlflow.set_tracking_uri(effective_uri)
    print(f"[MLflow] tracking_uri={effective_uri}")

    if artifact_root:
        # Pass artifact_root when creating the experiment so MLflow stores
        # all artifacts for this experiment under that root.
        try:
            mlflow.create_experiment(experiment_name, artifact_location=artifact_root)
            print(f"[MLflow] Created experiment '{experiment_name}' with artifact_root={artifact_root}")
        except mlflow.exceptions.MlflowException:
            # Experiment already exists — that's fine.
            pass

    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Active experiment: {experiment_name}")


# =========================================================
# Env Helpers
# =========================================================


def parse_list_env(name: str) -> List[str]:
    val = os.getenv(name, "").strip()
    if not val:
        return []
    try:
        parsed = json.loads(val)
        return [str(v) for v in parsed]
    except json.JSONDecodeError:
        return [v.strip() for v in val.split(",") if v.strip()]


def parse_int_list_env(name: str) -> List[int]:
    val = os.getenv(name, "").strip()
    if not val:
        return []
    try:
        return [int(v) for v in json.loads(val)]
    except Exception:
        return [int(v.strip()) for v in val.split(",") if v.strip()]


# =========================================================
# Dataset Loading
# =========================================================


def load_dataset(uri: str) -> rd.Dataset:
    if uri.startswith(("postgresql://", "mysql://", "sqlite://", "mssql://")):
        return load_dataset_from_sql(uri)
    return load_dataset_from_url(uri)


# =========================================================
# Feature Column Validation
# — preserves supported categorical columns and removes invalid columns
# =========================================================


def validate_feature_columns(
    ds: rd.Dataset,
    feature_cols: List[str],
    target_col: str,
) -> List[str]:
    """
    Validate requested features without discarding supported categoricals.

    Pass 1 — Remove columns not present in the dataset schema.
      Columns listed in FEATURE_COLUMNS that don't exist in the data are silently
      dropped with a warning rather than crashing at training time.

    Pass 2 — Remove unsupported nested/binary columns. String and dictionary
      columns are intentionally retained and one-hot encoded after the
      train/validation split.

    Returns the cleaned list, which is always a subset of the original.

    Notes
    -----
    The schema object only exposes ``.names`` (List[str]) and ``.types``
    (List[pa.DataType]); iteration over fields and field-level attribute
    access are not supported.
    """
    schema = ds.schema()

    # Build name→type map from parallel .names / .types lists.
    # This is the only safe way to access schema info given that the schema
    # object does not support field iteration or .field() access.
    schema_names: List[str]      = schema.names if hasattr(schema, "names") else []
    schema_types                 = schema.types if hasattr(schema, "types") else []
    field_map: Dict[str, Any]    = dict(zip(schema_names, schema_types))
    schema_field_names: set      = set(schema_names)

    # ---- Pass 1: existence check ----
    existing_cols: List[str] = []
    for col in feature_cols:
        if col == target_col:
            # Target column must never appear in the feature set.
            print(f"[Feature Validation] Removing '{col}' from features because it is the target column.")
            continue
        if schema_field_names and col not in schema_field_names:
            print(f"[Feature Validation] Removing '{col}' from features because it is not found in dataset schema.")
            continue
        existing_cols.append(col)

    if not existing_cols:
        raise ValueError(
            f"No valid feature columns remain after existence check. "
            f"Original: {feature_cols}, Schema: {sorted(schema_field_names)}"
        )

    # ---- Pass 2: reject only feature types that cannot be normalized safely ----
    def _is_unsupported(arrow_type) -> bool:
        """Return True for nested/binary types unsupported by preprocessing."""
        if not isinstance(arrow_type, pa.DataType):
            # Ray may expose pandas/NumPy dtypes instead of Arrow dtypes.
            # Scalar numeric, datetime, bool, object, string, and category
            # dtypes are all handled by version-3 preprocessing.
            return False
        if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
            return True
        if (
            pa.types.is_struct(arrow_type)
            or pa.types.is_list(arrow_type)
            or pa.types.is_large_list(arrow_type)
            or pa.types.is_map(arrow_type)
        ):
            return True
        return False

    usable_cols: List[str] = []
    for col in existing_cols:
        arrow_type = field_map.get(col)
        if arrow_type is not None and _is_unsupported(arrow_type):
            print(
                f"[Feature Validation] Removing '{col}' because its dtype "
                f"({arrow_type}) is not a supported scalar feature."
            )
            continue
        usable_cols.append(col)

    if not usable_cols:
        raise ValueError(
            f"No supported feature columns remain after dtype validation. "
            f"Requested features: {existing_cols}"
        )

    removed = set(feature_cols) - set(usable_cols)
    if removed:
        print(f"[Feature Validation] Removed unsupported feature columns: {sorted(removed)}")
    print(f"[Feature Validation] Using {len(usable_cols)} feature column(s): {usable_cols}")

    return usable_cols


# =========================================================
# Streaming: class names discovery — distributed, no full scan
# =========================================================


def discover_class_names_sql(source_uri: str, target_col: str, table_name: str) -> tuple:
    """
    Fetch distinct label values via a cheap SQL DISTINCT query.
    Never loads row data — only the unique label values are returned.

    Parameters
    ----------
    source_uri  : SQLAlchemy-compatible connection URI.
    target_col  : Column name whose distinct values become class labels.
    table_name  : Table (or view) name to query. Must be supplied separately
                  because source_uri is a connection string, not a table reference.

    Returns
    -------
    (classes, total_count) where classes is a sorted list of strings and
    total_count is the number of distinct classes (not rows).
    """
    engine = create_engine(source_uri)
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT DISTINCT {target_col} FROM {table_name} ORDER BY 1")
        )
        classes = [str(row[0]) for row in result]

    total_count = len(classes)
    return classes, total_count


@ray.remote(num_cpus=cpus_per_worker, scheduling_strategy="SPREAD")
def _collect_unique_labels_from_block(block: Any, target_col: str) -> tuple:
    """
    Ray remote task: runs on one dataset block in isolation.
    Extracts unique label values from that block only — no inter-node data shuffle.
    Returns (unique_labels: List[str], row_count: int).
    Handles both PyArrow Table and pandas DataFrame blocks.
    """
    if isinstance(block, pd.DataFrame):
        return [str(v) for v in block[target_col].unique()], len(block)

    if isinstance(block, pyarrow.Table):
        return [str(v.as_py()) for v in block.column(target_col).unique()], block.num_rows

    return [], 0


def discover_class_names_distributed(ds: rd.Dataset, target_col: str) -> tuple:
    """
    Distributed class label discovery using Ray remote tasks.

    Strategy:
      1. Select only the target column so feature data never crosses the network.
      2. ds.get_internal_block_refs() returns object references to the already-in-memory
         or lazily-staged blocks — it does NOT materialise or scan the dataset.
      3. One lightweight Ray task is dispatched per block to collect unique values
         from that block in parallel across the cluster.
      4. ray.get() collects per-block results on the driver; the final union is
         O(num_classes) not O(num_rows) — only distinct strings are transferred.

    Returns
    -------
    (sorted_class_names: List[str], total_row_count: int)
    total_row_count is the sum of rows across all blocks (approximate if blocks
    are lazily computed and not yet fully staged).
    """
    label_ds = ds.select_columns([target_col])

    # get_internal_block_refs() returns references to the existing block objects.
    # It does not trigger a full dataset scan or materialisation — blocks that have
    # not been computed yet will be computed on demand by each remote task.
    block_refs = label_ds.get_internal_block_refs()

    # Dispatch one lightweight task per block — runs fully in parallel.
    futures = [_collect_unique_labels_from_block.remote(ref, target_col) for ref in block_refs]
    per_block_results: List[tuple] = ray.get(futures)

    # Merge on the driver — a union of small string sets, not of row data.
    all_labels: set = set()
    all_count = 0
    for labels, count in per_block_results:
        all_labels.update(labels)
        all_count += count

    return sorted(all_labels), all_count


def is_valid_target_value(value: Any, model_type: str) -> bool:
    """Return whether a target value is safe to train on without imputation."""
    normalized_model_type = model_type.lower()
    if "classification" in normalized_model_type:
        return not pd.isna(value) and not (
            isinstance(value, str) and value.strip() == ""
        )

    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return False
    try:
        return np.isfinite(float(numeric_value))
    except (TypeError, ValueError, OverflowError):
        return False


def cast_regression_target_batch(
    batch: Dict[str, np.ndarray],
    target_col: str,
) -> Dict[str, np.ndarray]:
    """Normalize regression targets after invalid rows have been filtered out."""
    batch[target_col] = (
        pd.to_numeric(pd.Series(batch[target_col]), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32)
    )
    return batch


def resolve_training_worker_count(
    requested_workers: int,
    max_allowed_workers: int,
    available_blocks: int,
) -> int:
    """Choose the actual TorchTrainer worker count from the requested plan."""
    requested = max(1, int(requested_workers or 1))
    max_allowed = max(1, int(max_allowed_workers or requested))
    blocks = max(1, int(available_blocks or 1))
    return min(requested, max_allowed, blocks)


def inverse_standard_scaling(values: List[float], scaler: Dict[str, Any] | None) -> List[float]:
    """Convert standardized regression values back to their original units."""
    if not scaler or scaler.get("method") != "standard":
        return [float(value) for value in values]
    mean = float(scaler.get("mean", 0.0))
    scale = float(scaler.get("scale", 1.0))
    return [float(value) * scale + mean for value in values]


def _is_categorical_arrow_type(arrow_type: Any) -> bool:
    """Match LocalTrainer's categorical decision for Arrow-backed features."""
    if arrow_type is None:
        return True
    if isinstance(arrow_type, pa.DataType):
        return bool(
            pa.types.is_null(arrow_type)
            or pa.types.is_string(arrow_type)
            or pa.types.is_large_string(arrow_type)
            or pa.types.is_dictionary(arrow_type)
        )
    return bool(
        pd.api.types.is_object_dtype(arrow_type)
        or pd.api.types.is_string_dtype(arrow_type)
        or isinstance(arrow_type, pd.CategoricalDtype)
    )


def _is_datetime_feature_type(feature_type: Any) -> bool:
    """Support datetime detection for both Arrow and pandas/NumPy schemas."""
    if isinstance(feature_type, pa.DataType):
        return bool(
            pa.types.is_timestamp(feature_type)
            or pa.types.is_date(feature_type)
            or pa.types.is_time(feature_type)
        )
    return bool(pd.api.types.is_datetime64_any_dtype(feature_type))


def _normalize_category_value(value: Any) -> Optional[str]:
    """Normalize category spelling while preserving missingness for imputation."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    normalized = str(value).strip()
    return normalized or None


def _numeric_feature_values(series: pd.Series, is_datetime: bool = False) -> pd.Series:
    """Convert a Ray pandas batch feature to finite float64 values."""
    if is_datetime:
        timestamps = pd.to_datetime(series, errors="coerce")
        values = pd.Series(
            timestamps.astype("int64") / 1_000_000_000,
            index=series.index,
        ).mask(timestamps.isna())
    else:
        values = pd.to_numeric(series, errors="coerce")
    return values.astype(np.float64).replace([np.inf, -np.inf], np.nan)


def _group_value_counts(ds: rd.Dataset, column: str) -> Dict[Any, int]:
    """Collect distributed group counts; memory is O(unique values), not O(rows)."""
    rows = ds.select_columns([column]).groupby(column).count().take_all()
    counts: Dict[Any, int] = {}
    for row in rows:
        count_key = next((key for key in row if str(key).startswith("count(")), None)
        if count_key is None:
            continue
        counts[row.get(column)] = int(row[count_key])
    return counts


def balanced_class_weights_from_counts(
    class_counts: Dict[int, int],
    num_classes: int,
) -> List[float]:
    """Return inverse-frequency weights using training-partition counts only."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for label, count in class_counts.items():
        label_index = int(label)
        if 0 <= label_index < num_classes:
            counts[label_index] = max(0, int(count))

    observed = counts > 0
    weights = np.ones(num_classes, dtype=np.float32)
    if observed.any():
        observed_total = float(counts[observed].sum())
        observed_classes = int(observed.sum())
        weights[observed] = observed_total / (
            observed_classes * counts[observed]
        )
    return weights.tolist()


def fit_ray_class_weights(train_ds: rd.Dataset, num_classes: int) -> List[float]:
    """Fit classification loss weights from the Ray training split only."""
    raw_counts = _group_value_counts(train_ds, "__label__")
    class_counts = {int(label): count for label, count in raw_counts.items()}
    return balanced_class_weights_from_counts(class_counts, num_classes)


def fit_ray_feature_preprocessing(
    train_ds: rd.Dataset,
    feature_cols: List[str],
    model_type: str,
) -> Dict[str, Any]:
    """Fit version-3 preprocessing exclusively on the Ray training split.

    The emitted ``model_feature_names`` follow ``feature_cols`` exactly, just
    like :func:`local_trainer.fit_feature_preprocessing`.  This ordering is part
    of the persisted model contract: the same transformed columns in a
    different order would silently permute the tensor presented to the model.
    """
    schema = train_ds.schema()
    schema_names = schema.names if hasattr(schema, "names") else []
    schema_types = schema.types if hasattr(schema, "types") else []
    field_types = dict(zip(schema_names, schema_types))

    categorical_cols = [
        col for col in feature_cols if _is_categorical_arrow_type(field_types.get(col))
    ]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    preprocessing: Dict[str, Any] = {
        "version": 3,
        "numeric_features": numeric_cols,
        "categorical_features": {},
        "feature_scaling": {},
        "imputation": {},
        "model_feature_names": [],
    }

    scaling_cols = list(numeric_cols)
    if "regression" in model_type:
        scaling_cols.append("__label__")

    datetime_cols = {
        col
        for col in numeric_cols
        if field_types.get(col) is not None
        and _is_datetime_feature_type(field_types[col])
    }
    fitted_means: Dict[str, float] = {}
    fitted_scales: Dict[str, float] = {}
    if scaling_cols:
        def numeric_stats_batch(batch: pd.DataFrame) -> pd.DataFrame:
            numeric = pd.DataFrame(index=batch.index)
            for col in scaling_cols:
                numeric[col] = _numeric_feature_values(
                    batch[col],
                    is_datetime=col in datetime_cols,
                )
            return numeric

        numeric_stats_ds = train_ds.map_batches(
            numeric_stats_batch,
            batch_format="pandas",
            num_cpus=cpus_per_worker,
            concurrency=num_workers,
        )
        means = numeric_stats_ds.mean(on=scaling_cols)

        for col in scaling_cols:
            raw_mean = (
                means.get(f"mean({col})", 0.0)
                if isinstance(means, dict)
                else means if len(scaling_cols) == 1 else 0.0
            )
            mean = float(raw_mean or 0.0)
            fitted_means[col] = mean if np.isfinite(mean) else 0.0

        def impute_for_scale(batch: pd.DataFrame) -> pd.DataFrame:
            result = batch.copy()
            for col, mean in fitted_means.items():
                result[col] = result[col].fillna(mean)
            return result

        imputed_stats_ds = numeric_stats_ds.map_batches(
            impute_for_scale,
            batch_format="pandas",
            num_cpus=cpus_per_worker,
            concurrency=num_workers,
        )
        scales = imputed_stats_ds.std(on=scaling_cols, ddof=0)

        def fitted_scale(col: str) -> float:
            raw_scale = (
                scales.get(f"std({col})", 1.0)
                if isinstance(scales, dict)
                else scales if len(scaling_cols) == 1 else 1.0
            )
            scale = float(raw_scale or 1.0)
            if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
                return 1.0
            return scale

        fitted_scales = {col: fitted_scale(col) for col in scaling_cols}

    # Build all persisted feature metadata in the original feature order.  The
    # local trainer uses this same one-pass ordering contract.
    for feature_index, col in enumerate(feature_cols):
        if col in categorical_cols:
            normalized_counts: Dict[str, int] = {}
            for raw_value, count in _group_value_counts(train_ds, col).items():
                value = _normalize_category_value(raw_value)
                if value is not None:
                    normalized_counts[value] = normalized_counts.get(value, 0) + count

            if normalized_counts:
                fill_value = sorted(
                    normalized_counts,
                    key=lambda value: (-normalized_counts[value], value),
                )[0]
                categories = sorted(normalized_counts)
            else:
                fill_value = "__MISSING__"
                categories = [fill_value]

            output_columns = [
                f"__mta_feature_{feature_index}_category_{category_index}"
                for category_index in range(len(categories))
            ]
            unknown_column = f"__mta_feature_{feature_index}_category_unknown"
            preprocessing["categorical_features"][col] = {
                "encoding": "one_hot",
                "categories": categories,
                "output_columns": output_columns,
                "unknown_column": unknown_column,
            }
            preprocessing["imputation"][col] = {
                "strategy": "most_frequent",
                "fill_value": fill_value,
            }
            preprocessing["model_feature_names"].extend(
                output_columns + [unknown_column]
            )
            print(
                f"[Feature Preprocessing] One-hot encoded '{col}' with "
                f"{len(categories)} training categories."
            )
            continue

        if col in numeric_cols:
            mean = fitted_means[col]
            scale = fitted_scales[col]
            output_column = f"__mta_feature_{feature_index}_numeric"
            preprocessing["imputation"][col] = {
                "strategy": "mean",
                "fill_value": mean,
            }
            preprocessing["feature_scaling"][col] = {
                "method": "standard",
                "mean": mean,
                "scale": scale,
                "output_column": output_column,
                "input_type": "datetime" if col in datetime_cols else "numeric",
            }
            preprocessing["model_feature_names"].append(output_column)

    if "regression" in model_type:
        preprocessing["target_scaling"] = {
            "method": "standard",
            "mean": fitted_means["__label__"],
            "scale": fitted_scales["__label__"],
        }

    if not preprocessing["model_feature_names"]:
        raise ValueError("No model features remain after Ray preprocessing.")
    return preprocessing


def transform_ray_feature_batch(
    batch: pd.DataFrame,
    feature_cols: List[str],
    preprocessing: Dict[str, Any],
) -> pd.DataFrame:
    """Apply a fitted version-3 preprocessing contract to one pandas batch."""
    transformed = batch.copy()
    categorical = preprocessing.get("categorical_features") or {}
    imputation = preprocessing.get("imputation") or {}
    feature_scaling = preprocessing.get("feature_scaling") or {}

    for col in feature_cols:
        if col in categorical and categorical[col].get("encoding") == "one_hot":
            meta = categorical[col]
            fill_value = str((imputation.get(col) or {}).get("fill_value", "__MISSING__"))
            values = transformed[col].map(_normalize_category_value).fillna(fill_value)
            categories = [str(value) for value in meta.get("categories", [])]
            for category, output_column in zip(categories, meta.get("output_columns", [])):
                transformed[str(output_column)] = values.eq(category).astype(np.float32)
            transformed[str(meta["unknown_column"])] = (~values.isin(categories)).astype(np.float32)
            continue

        scaler = feature_scaling.get(col) or {}
        fill_value = float((imputation.get(col) or {}).get("fill_value", 0.0))
        values = _numeric_feature_values(
            transformed[col],
            is_datetime=scaler.get("input_type") == "datetime",
        ).fillna(fill_value)
        scale = float(scaler.get("scale", 1.0))
        if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
            scale = 1.0
        transformed[str(scaler.get("output_column", col))] = (
            (values - float(scaler.get("mean", 0.0))) / scale
        ).astype(np.float32)

    target_scaling = preprocessing.get("target_scaling") or {}
    if target_scaling and "__label__" in transformed:
        target_scale = float(target_scaling.get("scale", 1.0))
        if not np.isfinite(target_scale) or target_scale <= np.finfo(np.float32).eps:
            target_scale = 1.0
        transformed["__label__"] = (
            (
                pd.to_numeric(transformed["__label__"], errors="coerce")
                - float(target_scaling.get("mean", 0.0))
            )
            / target_scale
        ).astype(np.float32)
    return transformed


def apply_ray_feature_preprocessing(
    ds: rd.Dataset,
    feature_cols: List[str],
    preprocessing: Dict[str, Any],
) -> rd.Dataset:
    """Apply fitted preprocessing without inspecting validation statistics."""
    return ds.map_batches(
        transform_ray_feature_batch,
        fn_kwargs={
            "feature_cols": feature_cols,
            "preprocessing": preprocessing,
        },
        batch_format="pandas",
        num_cpus=cpus_per_worker,
        concurrency=num_workers,
    )


def validation_loss_improved(loss: float, best_loss: float, min_delta: float) -> bool:
    """Return whether validation loss improved enough to reset patience."""
    return float(loss) < float(best_loss) - max(0.0, float(min_delta))


# =========================================================
# Model Export helpers (thin wrappers — arch lives in model.py)
# =========================================================


def save_model_pth(model: DynamicMLP, path: str) -> None:
    """Save model state dict to a .pth file via DynamicMLP.save_pth()."""
    model.save_pth(path)
    print(f"  [Export] .pth → {path}")


def save_model_onnx(model: DynamicMLP, path: str) -> None:
    """
    Export model to ONNX via DynamicMLP.save_onnx().
    The input size is taken from model.input_size — no need to pass it separately.
    """
    model.save_onnx(path)
    print(f"  [Export] .onnx → {path}")


# =========================================================
# Training Loop (per worker)
# =========================================================


def log_usages(prefix: str = "") -> None:
    """
    Print current CPU and RAM utilisation.
    Not called automatically — invoke manually for ad-hoc debugging,
    e.g. log_usages("before training") inside train_loop_per_worker.
    """
    cpu    = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"[Resource Usage] ({prefix}) CPU: {cpu}%  RAM: {memory.percent}% of {round(memory.total / (1024**3), 2)} GB")


def train_loop_per_worker(config: Dict[str, Any]) -> None:
    """
    Per-worker training loop executed by TorchTrainer on every Ray worker.

    Receives a shard of the train and val datasets via train.get_dataset_shard().
    Streams batches epoch-by-epoch — no full dataset is held in memory.

    Checkpoints the best model (lowest val_loss) to the working directory via
    train.report(..., checkpoint=Checkpoint.from_directory(".")).
    Early stopping triggers when val_loss does not improve for `patience` epochs.

    config keys (passed through train_loop_config in training_main):
      feature_cols, model_type, output_dim, hidden_sizes, activation, dropout,
      batch_norm, learning_rate, epochs, batch_size, optimizer,
      random_seed, early_stopping_patience, use_gpu.
    """
    node_id = ray.get_runtime_context().get_node_id()
    print(f"[Worker] Starting training loop with config: {config}, Node ID: {node_id}")

    torch.manual_seed(config["random_seed"])

    train_ds = train.get_dataset_shard("train")
    val_ds   = train.get_dataset_shard("val")

    feature_cols: List[str] = config["feature_cols"]
    model_type: str         = config["model_type"]
    output_dim: int         = config["output_dim"]
    target_scaling: Dict[str, Any] = config.get("target_scaling") or {}

    # Construct the model directly from raw config values rather than a full
    # ModelConfig — workers receive a plain dict, not the dataclass.
    model = DynamicMLP(
        input_size=len(feature_cols),
        output_size=output_dim,
        hidden_sizes=config["hidden_sizes"],
        activation=config["activation"],
        dropout=config["dropout"],
        batch_norm=config["batch_norm"],
    )
    # prepare_model wraps the model for distributed training (DDP) on this worker.
    model = train.torch.prepare_model(model)
    model_device = next(model.parameters()).device

    print(f"[Worker] Model architecture: {model}")

    opt_map = {
        "adam":    optim.Adam,
        "sgd":     optim.SGD,
        "adamw":   optim.AdamW,
        "rmsprop": optim.RMSprop,
    }
    optimizer = opt_map.get(config["optimizer"], optim.Adam)(
        model.parameters(), lr=config["learning_rate"]
    )
    # CrossEntropyLoss expects integer class indices; fit inverse-frequency
    # weights on the training partition so minority-class errors matter.
    if "classification" in model_type:
        class_weights = torch.tensor(
            config.get("class_weights") or [1.0] * output_dim,
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[Worker] Classification weights: {class_weights.detach().cpu().tolist()}")
    else:
        criterion = nn.MSELoss()

    best_val_loss    = float("inf")
    patience_counter = 0
    patience: int    = max(1, int(config["early_stopping_patience"]))
    min_delta: float = max(0.0, float(config.get("early_stopping_min_delta", 1e-4)))
    history: List[Dict] = []

    for epoch in range(config["epochs"]):
        print(f"[Worker] Epoch {epoch + 1}/{config['epochs']} starting...")

        # -------- TRAIN --------
        model.train()
        train_loss, train_batches = 0.0, 0
        all_train_preds: List[float] = []
        all_train_labels: List[float] = []

        print("[Worker] Iterating over training batches (streaming)...")
        for batch in train_ds.iter_batches(
            batch_size=config["batch_size"],
            batch_format="numpy",
            prefetch_batches=0,
            # Match the local DataLoader(shuffle=True) behavior without
            # materializing the distributed dataset in worker memory.
            local_shuffle_buffer_size=max(1024, config["batch_size"] * 4),
            local_shuffle_seed=config["random_seed"] + epoch,
        ):
            x = torch.stack(
                [torch.tensor(batch[c], dtype=torch.float32) for c in feature_cols],
                dim=1,
            ).to(model_device)
            y = torch.tensor(
                batch["__label__"],
                dtype=torch.long if "classification" in model_type else torch.float32,
            ).to(model_device)
            if y.ndim == 1 and "regression" in model_type:
                y = y.unsqueeze(1)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            if "classification" in model_type:
                all_train_preds.extend(out.argmax(1).cpu().tolist())
                all_train_labels.extend(y.cpu().tolist())
            else:
                all_train_preds.extend(out.detach().cpu().reshape(-1).tolist())
                all_train_labels.extend(y.detach().cpu().reshape(-1).tolist())

            del x, y, batch, out
        avg_train_loss = train_loss / max(train_batches, 1)
        if "classification" in model_type and all_train_labels:
            train_acc = sum(p == l for p, l in zip(all_train_preds, all_train_labels)) / len(all_train_labels)
            train_f1  = float(sk_f1_score(all_train_labels, all_train_preds, average="weighted", zero_division=0))
        else:
            train_acc, train_f1 = None, None
            train_regression_metrics = _compute_regression_metrics(
                inverse_standard_scaling(all_train_labels, target_scaling),
                inverse_standard_scaling(all_train_preds, target_scaling),
            )
        del all_train_preds, all_train_labels

        # -------- VALIDATION --------
        model.eval()
        val_loss, val_batches = 0.0, 0
        all_val_preds: List[float] = []
        all_val_labels: List[float] = []

        with torch.no_grad():
            print("[Worker] Iterating over validation batches...")
            for batch in val_ds.iter_batches(
                batch_size=config["batch_size"],
                batch_format="numpy",
                prefetch_batches=0,
                local_shuffle_buffer_size=None,
            ):
                x = torch.stack(
                    [torch.tensor(batch[c], dtype=torch.float32) for c in feature_cols],
                    dim=1,
                ).to(model_device)
                y = torch.tensor(
                    batch["__label__"],
                    dtype=torch.long if "classification" in model_type else torch.float32,
                ).to(model_device)
                if y.ndim == 1 and "regression" in model_type:
                    y = y.unsqueeze(1)

                out = model(x)
                val_loss += criterion(out, y).item()
                val_batches += 1

                if "classification" in model_type:
                    all_val_preds.extend(out.argmax(1).cpu().tolist())
                    all_val_labels.extend(y.cpu().tolist())
                else:
                    all_val_preds.extend(out.cpu().reshape(-1).tolist())
                    all_val_labels.extend(y.cpu().reshape(-1).tolist())

                del x, y, batch, out

        avg_val_loss = val_loss / max(val_batches, 1)
        if "classification" in model_type and all_val_labels:
            val_acc = sum(p == l for p, l in zip(all_val_preds, all_val_labels)) / len(all_val_labels)
            val_f1  = float(sk_f1_score(all_val_labels, all_val_preds, average="weighted", zero_division=0))
        else:
            val_acc, val_f1 = None, None
            val_regression_metrics = _compute_regression_metrics(
                inverse_standard_scaling(all_val_labels, target_scaling),
                inverse_standard_scaling(all_val_preds, target_scaling),
            )
        del all_val_preds, all_val_labels

        epoch_metrics: Dict[str, Any] = {
            "epoch":        epoch + 1,
            "train_loss":   round(avg_train_loss, 6),
            "val_loss":     round(avg_val_loss, 6),
        }
        if "classification" in model_type:
            epoch_metrics.update({
                "accuracy":     round(train_acc, 6),
                "val_accuracy": round(val_acc, 6),
                "f1_score":     round(train_f1, 6),
                "val_f1_score": round(val_f1, 6),
            })
        else:
            epoch_metrics.update({
                "train_mae":      train_regression_metrics["mae"],
                "val_mae":        val_regression_metrics["mae"],
                "train_rmse":     train_regression_metrics["rmse"],
                "val_rmse":       val_regression_metrics["rmse"],
                "train_r2_score": train_regression_metrics["r2_score"],
                "val_r2_score":   val_regression_metrics["r2_score"],
            })
        history.append(epoch_metrics)

        if "classification" in model_type:
            print(
                f"[Worker] [Epoch {epoch + 1}/{config['epochs']}] "
                f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
                f"acc={train_acc:.4f}  f1={train_f1:.4f}"
            )
        else:
            print(
                f"[Worker] [Epoch {epoch + 1}/{config['epochs']}] "
                f"scaled_train_mse={avg_train_loss:.4f}  scaled_val_mse={avg_val_loss:.4f}  "
                f"val_mae={val_regression_metrics['mae']:.4f}  "
                f"val_rmse={val_regression_metrics['rmse']:.4f}  "
                f"val_r2={val_regression_metrics['r2_score']:.4f}"
            )
        
        val_loss_tensor = torch.tensor([avg_val_loss], dtype=torch.float32)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        global_val_loss = val_loss_tensor.item()

        epoch_metrics["val_loss"] = round(global_val_loss, 6)

        if validation_loss_improved(global_val_loss, best_val_loss, min_delta):
            best_val_loss    = global_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "model.pt")
            with open("history.json", "w") as hf:
                json.dump(history, hf)
            train.report(epoch_metrics, checkpoint=Checkpoint.from_directory("."))
        else:
            patience_counter += 1
            train.report(epoch_metrics)

        should_stop = torch.tensor([1 if patience_counter >= patience else 0], dtype=torch.int32)
        dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)
        if should_stop.item() >= 1:
            print(f"[EarlyStopping] Triggered at epoch {epoch + 1} (global decision)")
            break


# =========================================================
# TRAINING ENTRY
# =========================================================


def training_main() -> None:
    start_time = time.time()

    # ---- Env ----
    task_id       = os.getenv("TASK_ID", str(uuid.uuid4()))
    user_id       = os.getenv("USER_ID", "unknown_user")
    session_id    = os.getenv("SESSION_ID", str(uuid.uuid4()))
    model_type    = os.getenv("MODEL_TYPE", "classification").lower().strip()
    data_uri      = os.getenv("DATA_SOURCE_URI", "")
    target_col    = os.getenv("TARGET_COLUMN", "")
    time_column   = os.getenv("TIME_COLUMN", "").strip()
    class_names   = parse_list_env("CLASS_NAMES")
    feature_cols  = parse_list_env("FEATURE_COLUMNS")
    try:
        pii_transformations = json.loads(os.getenv("PII_TRANSFORMATIONS", "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError("PII_TRANSFORMATIONS must be a JSON object.") from exc
    if not isinstance(pii_transformations, dict):
        raise ValueError("PII_TRANSFORMATIONS must be a JSON object.")
    pii_key = (os.getenv(PII_KEY_ENV) or "").strip()

    # Model identity — stored in MLflow tags, params, and model_config.json
    # so that mlflow_manager / inference.py can surface them without re-reading runs.
    model_name        = os.getenv("MODEL_NAME", "dynamic-mlp").strip()
    model_description = os.getenv("MODEL_DESCRIPTION", "").strip()
    model_version     = os.getenv("MODEL_VERSION", "1").strip()

    learning_rate      = float(os.getenv("LEARNING_RATE", 0.001))
    epochs             = int(os.getenv("EPOCHS", 30))
    batch_size         = int(os.getenv("BATCH_SIZE", 32))
    optimizer_name     = os.getenv("OPTIMIZER", "adam").lower().strip()
    random_seed        = int(os.getenv("RANDOM_SEED", 42))
    early_stopping_pat = int(os.getenv("EARLY_STOPPING_PATIENCE", 5))
    early_stopping_min_delta = float(os.getenv("EARLY_STOPPING_MIN_DELTA", 1e-4))
    use_gpu            = os.getenv("USE_GPU", "false").lower() == "true"
    hidden_sizes       = parse_int_list_env("HIDDEN_SIZES") or [128, 64]
    activation         = os.getenv("ACTIVATION", "relu").lower().strip()
    dropout            = float(os.getenv("DROPOUT", 0.2))
    batch_norm         = os.getenv("BATCH_NORM", "true").lower() == "true"

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print(f"[Config] features={feature_cols}  target={target_col}")
    print(f"[Config] epochs={epochs}  lr={learning_rate}  hidden={hidden_sizes}")
    print(f"[Config] workers num={num_workers}  min={min_workers}  max={max_workers}  cpus_per_worker={cpus_per_worker}")

    # ---- Load dataset (lazy/streaming) ----
    print(f"[Data] Loading from {data_uri}")
    ds = load_dataset(data_uri)

    if time_column:
        available_columns = set(ds.schema().names)
        if time_column not in available_columns:
            raise ValueError(
                f"Time column {time_column!r} is not present in the training dataset."
            )
        if time_column == target_col:
            raise ValueError("The time column must be different from the target column.")
        # Ordering metadata is not a raw MLP feature. Calendar feature
        # engineering can be added separately without one-hot encoding every
        # individual timestamp.
        feature_cols = [column for column in feature_cols if column != time_column]

    # ---- Validate feature columns ----
    # Missing and unsupported nested/binary fields are removed. Supported
    # categorical columns remain and are encoded after the split.
    # Must run BEFORE ds.select_columns() so we only request valid columns.
    feature_cols = validate_feature_columns(ds, feature_cols, target_col)

    # Run the same safety gate as LocalTrainer before preprocessing or splitting.
    # A bounded sample avoids collecting a cloud-scale dataset on the Ray driver;
    # duplicate-record and identifier checks remain bounded on cloud datasets.
    integrity_sample = ds.limit(10_000).to_pandas()
    integrity_report = build_training_integrity_report(
        integrity_sample, target_col, feature_cols
    )
    integrity_report["scope"] = "bounded_training_sample"
    integrity_report["sample_limit"] = 10_000
    integrity_report["pii"] = {
        "applied": {
            column: str(metadata.get("strategy", "keep"))
            for column, metadata in pii_transformations.items()
        },
        "key_persisted": False,
    }
    duplicate_match = find_duplicate_training_rows(integrity_sample, target_col, feature_cols)
    if duplicate_match:
        raise ValueError(duplicate_rows_error(duplicate_match))
    identifier_matches = find_identifier_features(integrity_sample, feature_cols)
    if identifier_matches:
        raise ValueError(identifier_features_error(identifier_matches))
    leakage_matches = find_target_leakage(integrity_sample, target_col, feature_cols)
    if leakage_matches:
        raise ValueError(target_leakage_error(target_col, leakage_matches))

    selected_columns = list(dict.fromkeys(
        feature_cols + [target_col] + ([time_column] if time_column else [])
    ))
    ds = ds.select_columns(selected_columns)

    pii_transformations = {
        str(column): dict(metadata)
        for column, metadata in pii_transformations.items()
        if column in selected_columns and isinstance(metadata, dict)
    }
    keyed_strategies = {"pseudonymise", "tokenise"}
    keyed_columns = [
        column
        for column, metadata in pii_transformations.items()
        if str(metadata.get("strategy")) in keyed_strategies
    ]
    if keyed_columns and not pii_key:
        raise RuntimeError(
            f"{PII_KEY_ENV} is not available in the Ray training pods. Configure "
            "the avaloka-secrets reference on the RayCluster before retrying."
        )
    if pii_transformations:
        ds = ds.map_batches(
            transform_pii_batch,
            fn_kwargs={
                "transformations": pii_transformations,
                "key": pii_key,
            },
            batch_format="pandas",
            num_cpus=cpus_per_worker,
            concurrency=num_workers,
        )
        print(
            "[PII] Applied de-identification to columns: "
            + ", ".join(sorted(pii_transformations))
        )
        target_pii = pii_transformations.get(target_col)
        if target_pii and class_names and "classification" in model_type:
            class_names = [
                str(transform_pii_value(
                    value,
                    column=target_col,
                    kind=str(target_pii.get("kind", "none")),
                    strategy=str(target_pii.get("strategy", "keep")),
                    key=pii_key,
                ))
                for value in class_names
            ]

    num_samples: int = -1

    # ---- Target cleanup — never impute labels ----
    # Use Ray's row filter instead of returning filtered batches from map_batches.
    # Returning a zero-row numpy batch is fragile in the later distributed pipeline.
    def valid_target_row(row: Dict[str, Any]) -> bool:
        value = row[target_col]
        if "classification" in model_type:
            return not pd.isna(value) and not (
                isinstance(value, str) and value.strip() == ""
            )

        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric_value):
            return False
        try:
            return np.isfinite(float(numeric_value))
        except (TypeError, ValueError, OverflowError):
            return False

    ds = ds.filter(valid_target_row)

    if "classification" not in model_type:
        def cast_target_batch(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            batch[target_col] = (
                pd.to_numeric(pd.Series(batch[target_col]), errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .to_numpy(dtype=np.float32)
            )
            return batch

        ds = ds.map_batches(
            cast_target_batch,
            batch_format="numpy",
            num_cpus=cpus_per_worker,
            concurrency=num_workers,
        )

    target_row_count = ds.count()
    if target_row_count <= 0:
        raise ValueError(
            f"No training rows remain after removing missing target values from {target_col!r}."
        )

    # ---- Class detection — distributed, no full materialisation ----
    if "classification" in model_type:
        if not class_names:
            print("[Data] Discovering class names via distributed block scan ...")
            class_names, total_rows = discover_class_names_distributed(ds, target_col)
            print(f"[Data] Found {len(class_names)} classes: {class_names}")
        else:
            total_rows = ds.count()

        le_classes = class_names[:]

        def encode_label(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            batch["__label__"] = np.array(
                [le_classes.index(str(v)) for v in batch[target_col]], dtype=np.int64
            )
            return batch

        ds = ds.map_batches(
            encode_label,
            batch_format="numpy",
            num_cpus=cpus_per_worker,
            concurrency=num_workers,
        )

    else:
        def add_label_regression(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            batch["__label__"] = batch[target_col].astype(np.float32)
            return batch

        ds = ds.map_batches(
            add_label_regression,
            batch_format="numpy",
            num_cpus=cpus_per_worker,
            concurrency=num_workers,
        )

        total_rows = ds.count()

    # ---- Train/val split ----
    if time_column:
        validation_split = time_series_split_metadata(total_rows, time_column)
        ordered_ds = ds.map_batches(
            add_time_order_column,
            fn_kwargs={"time_column": time_column},
            batch_format="pandas",
            num_cpus=cpus_per_worker,
            concurrency=num_workers,
        ).sort(TIME_ORDER_COLUMN)
        train_ds, val_ds = ordered_ds.train_test_split(
            test_size=validation_split["n_validation_rows"],
            shuffle=False,
        )
        drop_after_split = [TIME_ORDER_COLUMN]
        if time_column not in feature_cols:
            drop_after_split.append(time_column)
        train_ds = train_ds.drop_columns(drop_after_split).materialize()
        val_ds = val_ds.drop_columns(drop_after_split).materialize()
        print(
            f"[Data] TimeSeriesSplit column={time_column} "
            f"folds={validation_split['n_folds']} "
            f"train={validation_split['n_train_rows']} "
            f"val={validation_split['n_validation_rows']}"
        )
    else:
        validation_split = {
            "splitter": "DeterministicHashHoldout",
            "splitter_reason": (
                "the trainer uses a deterministic 80/20 row-hash split; "
                "baseline parameters are fitted on training rows only"
            ),
            "time_column": None,
            "n_folds": 1,
        }

        def _tag_train_val(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            n = len(next(iter(batch.values())))
            batch["__is_train__"] = np.array([
                int(hashlib.md5(
                    b"".join(str(batch[c][i]).encode() for c in batch if not c.startswith("__"))
                ).hexdigest(), 16) % 10 < 8
                for i in range(n)
            ], dtype=bool)
            return batch

        ds_tagged = ds.map_batches(
            _tag_train_val,
            batch_format="numpy",
            num_cpus=cpus_per_worker,
            concurrency=num_workers,
        )

        # Materialise BEFORE forking into train/val — avoids executing the full
        # pipeline twice (once per consumer). filter() does NOT support concurrency.
        ds_tagged = ds_tagged.materialize()
        train_ds = ds_tagged.filter(lambda r: bool(r["__is_train__"]))
        val_ds = ds_tagged.filter(lambda r: not bool(r["__is_train__"]))
        train_ds = train_ds.drop_columns(["__is_train__"]).materialize()
        val_ds = val_ds.drop_columns(["__is_train__"]).materialize()

    # Bound the target-only sample collected by the Ray driver. The baseline
    # remains representative without turning a billion-row target column into
    # a driver-memory problem.
    baseline_sample_limit = 100_000
    def _stream_targets(dataset: rd.Dataset):
        for batch in dataset.select_columns([target_col]).limit(
            baseline_sample_limit
        ).iter_batches(
            batch_format="numpy", batch_size=65_536
        ):
            yield from batch[target_col].tolist()

    train_targets_for_evaluation = list(_stream_targets(train_ds))
    val_targets_for_evaluation = list(_stream_targets(val_ds))

    # Fit imputation, category vocabularies, class weights, and scaling only on
    # training rows. Validation is transformed with the frozen contract.
    preprocessing = fit_ray_feature_preprocessing(train_ds, feature_cols, model_type)
    preprocessing["pii_transformations"] = pii_transformations
    preprocessing["validation"] = dict(validation_split)
    if "classification" in model_type:
        preprocessing["class_weights"] = fit_ray_class_weights(
            train_ds,
            len(class_names),
        )
        print(f"[Data] Class weights from training split: {preprocessing['class_weights']}")

    model_feature_cols = list(preprocessing["model_feature_names"])
    train_ds = apply_ray_feature_preprocessing(
        train_ds,
        feature_cols,
        preprocessing,
    ).select_columns(model_feature_cols + ["__label__"]).materialize()
    val_ds = apply_ray_feature_preprocessing(
        val_ds,
        feature_cols,
        preprocessing,
    ).select_columns(model_feature_cols + ["__label__"]).materialize()

    # ---- Ray storage ----
    # Ray Train checkpoints need a storage path separate from MLflow artifacts.
    # We derive a sibling path from MLFLOW_DEFAULT_ARTIFACT_ROOT so checkpoints land
    # in the same GCS bucket as other run outputs — no extra bucket config needed.
    # For a GCS URI like  gs://bucket/path/to/artifact.json  the ray-results dir
    # is placed at  bucket/path/to/ray-results  (the filename component is dropped).
    # Falls back to /tmp/ray-results when MLFLOW_DEFAULT_ARTIFACT_ROOT is not a GCS URI.
    artifact_root    = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "").strip().rstrip("/")
    gcs_ray_storage  = "/tmp/ray-results"
    storage_filesystem = None

    if artifact_root.startswith("gs://"):
        uri_no_scheme = artifact_root[5:]
        parts         = uri_no_scheme.split("/")
        bucket        = parts[0]
        dir_parts     = parts[1:] if len(parts) > 1 else []
        dir_path      = "/".join(dir_parts)
        gcs_ray_storage = f"{bucket}/{dir_path}/ray-results" if dir_path else f"{bucket}/ray-results"

        try:
            if not _GCS_FS_AVAILABLE:
                raise ImportError("pyarrow.fs.GcsFileSystem not available")
            storage_filesystem = GcsFileSystem()
            print("[Ray Storage] GcsFileSystem initialised OK")
        except Exception as e:
            print(f"[Ray Storage] GcsFileSystem init failed: {e} — falling back to /tmp")
            gcs_ray_storage    = "/tmp/ray-results"
            storage_filesystem = None

    print(f"[Ray Storage] storage_path={gcs_ray_storage}")

    # ---- RunConfig with FailureConfig for autoscaling resilience ----
    run_config = RunConfig(
        storage_path=gcs_ray_storage,
        **({  "storage_filesystem": storage_filesystem} if storage_filesystem else {}),
    )

    head_node_id = ray.get_runtime_context().get_node_id()
    print(f"[Ray Cluster] Head node ID: {head_node_id}")

    # ---- ScalingConfig — autoscaling-aware ----
    # NUM_WORKERS is the requested training parallelism from the plan. MAX_WORKERS
    # is only an upper bound for autoscaling, not the desired TorchTrainer size.
    num_training_workers = resolve_training_worker_count(
        num_workers,
        max_workers,
        val_ds.num_blocks(),
    )
    scaling_config = ScalingConfig(
        num_workers=num_training_workers,
        use_gpu=use_gpu,
        resources_per_worker={"CPU": cpus_per_worker - 1},
        placement_strategy="SPREAD",
        trainer_resources={"CPU": 1}
    )
    print(
        f"[ScalingConfig] num_workers={scaling_config.num_workers} "
        f"requested={num_workers} max={max_workers} "
        f"use_gpu={scaling_config.use_gpu}  cpus_per_worker={cpus_per_worker}"
    )

    # ======================================================
    # MLflow Run — wraps the entire training lifecycle
    # ======================================================
    print("[MLflow] Starting run ...")
    with mlflow.start_run(run_name=task_id) as mlflow_run:
        mlflow_run_id = mlflow_run.info.run_id
        print(f"[MLflow] run_id={mlflow_run_id}")

        # ---- Log hyperparameters ----
        mlflow.log_params({
            "task_id":                 task_id,
            "user_id":                 user_id,
            "session_id":              session_id,
            "model_type":              model_type,
            "model_name":              model_name,
            "model_description":       model_description,
            "model_version":           model_version,
            "learning_rate":           learning_rate,
            "epochs":                  epochs,
            "batch_size":              batch_size,
            "optimizer":               optimizer_name,
            "hidden_sizes":            str(hidden_sizes),
            "activation":              activation,
            "dropout":                 dropout,
            "batch_norm":              batch_norm,
            "random_seed":             random_seed,
            "early_stopping_patience": early_stopping_pat,
            "early_stopping_min_delta": early_stopping_min_delta,
            "use_gpu":                 use_gpu,
            "num_workers":             num_workers,
            "num_training_workers":    num_training_workers,
            "cpus_per_worker":         cpus_per_worker,
            "feature_cols":            str(feature_cols),
            "target_col":              target_col,
            "time_column":             time_column,
            "validation_splitter":     validation_split["splitter"],
            "num_features":            len(feature_cols),
            "num_model_features":      len(model_feature_cols),
            "num_classes":             len(class_names) if class_names else 1,
            "class_names":             str(class_names),
            "class_weights":           str(preprocessing.get("class_weights") or []),
        })

        # ---- Train ----
        print(f"[Train] Starting with total_rows={total_rows}")
        trainer = TorchTrainer(
            train_loop_per_worker,
            scaling_config=scaling_config,
            run_config=run_config,
            datasets={"train": train_ds, "val": val_ds},
            train_loop_config={
                "feature_cols":            model_feature_cols,
                "model_type":              model_type,
                "class_names":             class_names,
                "num_classes":             len(class_names) if class_names else 1,
                "output_dim":              len(class_names) if class_names else 1,
                "hidden_sizes":            hidden_sizes,
                "activation":              activation,
                "dropout":                 dropout,
                "batch_norm":              batch_norm,
                "learning_rate":           learning_rate,
                "epochs":                  epochs,
                "batch_size":              batch_size,
                "optimizer":               optimizer_name,
                "random_seed":             random_seed,
                "early_stopping_patience": early_stopping_pat,
                "early_stopping_min_delta": early_stopping_min_delta,
                "use_gpu":                 use_gpu,
                "target_scaling":          preprocessing.get("target_scaling", {}),
                "class_weights":           preprocessing.get("class_weights"),
            },
        )

        result            = trainer.fit()
        final_ray_metrics = result.metrics or {}
        evaluation_report = build_holdout_evaluation_report(
            train_targets_for_evaluation,
            val_targets_for_evaluation,
            model_type,
            final_ray_metrics,
            splitter=validation_split["splitter"],
            splitter_reason=validation_split["splitter_reason"],
            n_folds=validation_split["n_folds"],
            time_column=validation_split["time_column"],
        )
        evaluation_report["baseline_sample_limit"] = baseline_sample_limit
        if (
            len(train_targets_for_evaluation) == baseline_sample_limit
            or len(val_targets_for_evaluation) == baseline_sample_limit
        ):
            evaluation_report["warnings"].append(
                "Ray baseline comparison uses a bounded 100,000-row target sample"
            )

        # ---- Retrieve checkpoint ----
        checkpoint     = result.checkpoint
        checkpoint_dir = checkpoint.to_directory() if checkpoint else None
        history: List[Dict] = []
        local_pth      = "/tmp/model.pt"

        if checkpoint_dir:
            hist_path = os.path.join(checkpoint_dir, "history.json")
            if os.path.exists(hist_path):
                with open(hist_path) as f:
                    history = json.load(f)
            ckpt_model = os.path.join(checkpoint_dir, "model.pt")
            if os.path.exists(ckpt_model):
                shutil.copy(ckpt_model, local_pth)

        num_samples = total_rows

        # ---- Log per-epoch metrics to MLflow ----
        # Each epoch dict contains train_loss, val_loss, accuracy, f1, etc.
        for epoch_row in history:
            epoch_num = epoch_row.get("epoch", 0)
            mlflow.log_metrics(
                {key: value for key, value in epoch_row.items() if key != "epoch"},
                step=epoch_num,
            )

        # ---- Build training history output ----
        training_history_out = [dict(row) for row in history]

        best_epoch = min(training_history_out, key=lambda r: r["val_loss"]) if training_history_out else {}
        runtime_s  = round(time.time() - start_time, 3)

        if "regression" in model_type:
            final_task_metrics = {
                "final_train_mae":      final_ray_metrics.get("train_mae", 0.0),
                "final_mae":            final_ray_metrics.get("val_mae", 0.0),
                "final_train_rmse":     final_ray_metrics.get("train_rmse", 0.0),
                "final_rmse":           final_ray_metrics.get("val_rmse", 0.0),
                "final_train_r2_score": final_ray_metrics.get("train_r2_score", 0.0),
                "final_r2_score":       final_ray_metrics.get("val_r2_score", 0.0),
            }
        else:
            final_task_metrics = {
                "final_accuracy": round(final_ray_metrics.get("accuracy", 0.0), 6),
                "final_f1_score": round(final_ray_metrics.get("f1_score", 0.0), 6),
            }

        # ---- Log final summary metrics ----
        mlflow.log_metrics({
            "final_train_loss": round(final_ray_metrics.get("train_loss", final_ray_metrics.get("loss", 0.0)), 6),
            "final_val_loss":   round(final_ray_metrics.get("val_loss", 0.0), 6),
            "runtime_s":        runtime_s,
            "rows_processed":   float(num_samples),
            "num_epochs_trained": float(len(training_history_out)),
            **final_task_metrics,
        })

        # ---- Export model ----
        # Build ModelConfig — the single source of truth for arch + data metadata.
        # Written to metadata/model_config.json so mlflow_manager can reconstruct
        # the model without re-reading all run params.
        model_cfg = ModelConfig(
            input_size=len(model_feature_cols),
            output_size=len(class_names) if class_names else 1,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            feature_names=feature_cols,
            class_names=class_names,
            target_column=target_col,
            model_type=model_type,
            model_name=model_name,
            model_description=model_description,
            model_version=model_version,
            task_id=task_id,
            mlflow_run_id=mlflow_run_id,
            preprocessing=preprocessing,
        )

        model = DynamicMLP.from_config(model_cfg)
        if os.path.exists(local_pth):
            model.load_weights(local_pth, device="cpu")
        model.eval()

        local_pth_named  = "/tmp/model.pth"
        local_onnx_named = "/tmp/model.onnx"

        save_model_pth(model, local_pth_named)
        save_model_onnx(model, local_onnx_named)

        # ---- Log model artifacts to MLflow ----
        # mlflow.pytorch.log_model logs the full PyTorch model (weights + class)
        # under the "model" artifact path, enabling mlflow.pytorch.load_model().
        # The raw .pth and .onnx files are also logged for direct download/serving.
        print("[MLflow] Logging model artifacts ...")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=None,   # set a name here to auto-register
            serialization_format="pickle"
        )
        mlflow.log_artifact(local_pth_named,  artifact_path="model_files")
        mlflow.log_artifact(local_onnx_named, artifact_path="model_files")

        # Write and log model_config.json — enables mlflow_manager to reconstruct
        # the model arch + metadata without parsing all run params.
        local_model_config = "/tmp/model_config.json"
        with open(local_model_config, "w") as f:
            f.write(model_cfg.to_json())
        mlflow.log_artifact(local_model_config, artifact_path="metadata")
        print(f"[MLflow] Logged model_config.json → metadata/model_config.json")

        # ---- Log metadata JSON as artifact ----
        artifact_meta = {
            "task_id":    task_id,
            "user_id":    user_id,
            "session_id": session_id,
            "model_type":        model_type,
            "model_name":        model_name,
            "model_description": model_description,
            "model_version":     model_version,
            "model_info": {
                "data_info": {
                    "class_info": {
                        "class_indices": {i: c for i, c in enumerate(class_names)} if class_names else {},
                        "class_names":   class_names,
                    },
                    "feature_names": feature_cols,
                    "preprocessing":  preprocessing,
                    "num_classes":   len(class_names) if class_names else 1,
                    "num_features":  len(feature_cols),
                    "num_model_features": len(model_feature_cols),
                    "num_samples":   num_samples,
                    "target_column": target_col,
                    "time_column":   time_column or None,
                }
            },
            "training_history":    training_history_out,
            "best_epoch":          best_epoch,
            "runtime_s":           runtime_s,
            "hyperparameters": {
                "learning_rate":           learning_rate,
                "epochs":                  epochs,
                "batch_size":              batch_size,
                "optimizer":               optimizer_name,
                "hidden_sizes":            hidden_sizes,
                "activation":              activation,
                "dropout":                 dropout,
                "batch_norm":              batch_norm,
                "random_seed":             random_seed,
                "early_stopping_patience": early_stopping_pat,
                "early_stopping_min_delta": early_stopping_min_delta,
                "use_gpu":                 use_gpu,
            },
            "metrics": {
                "runtime_s":        runtime_s,
                "rows_processed":   num_samples,
                "final_train_loss": round(final_ray_metrics.get("train_loss", final_ray_metrics.get("loss", 0.0)), 6),
                "final_val_loss":   round(final_ray_metrics.get("val_loss", 0.0), 6),
                **final_task_metrics,
            },
            "integrity_report": integrity_report,
            "evaluation_report": evaluation_report,
            "output_columns":   feature_cols + [target_col],
            "preprocessing":    preprocessing,
            "output_rows":      [],
            "output_row_count": num_samples,
            "created_at":       datetime.now(timezone.utc).isoformat(),
            # Expose the MLflow run_id so downstream consumers can retrieve
            # artifacts programmatically via mlflow.get_run(mlflow_run_id).
            "mlflow_run_id":    mlflow_run_id,
        }

        local_artifact_json = "/tmp/artifact.json"
        local_metrics_json  = "/tmp/metrics.json"

        output_metrics = {
            "mlflow_run_id":      mlflow_run_id,
            "task_id":            task_id,
            "user_id":            user_id,
            "session_id":         session_id,
            "model_type":         model_type,
            "model_name":         model_name,
            "model_description":  model_description,
            "model_version":      model_version,
            "runtime_s":          runtime_s,
            "rows_processed":     num_samples,
            "num_epochs_trained": len(training_history_out),
            "final_train_loss":   round(final_ray_metrics.get("train_loss", final_ray_metrics.get("loss", 0.0)), 6),
            "final_val_loss":     round(final_ray_metrics.get("val_loss", 0.0), 6),
            "num_features":       len(feature_cols),
            "num_model_features": len(model_feature_cols),
            "num_classes":        len(class_names) if class_names else 1,
            "created_at":         datetime.now(timezone.utc).isoformat(),
            "integrity_report":   integrity_report,
            "evaluation_report":  evaluation_report,
            **final_task_metrics,
        }

        with open(local_artifact_json, "w") as f:
            json.dump(artifact_meta, f, indent=2)
        with open(local_metrics_json, "w") as f:
            json.dump(output_metrics, f, indent=2)

        mlflow.log_artifact(local_artifact_json, artifact_path="metadata")
        mlflow.log_artifact(local_metrics_json,  artifact_path="metadata")

        # ---- Set MLflow tags ----
        # Tags are indexed and searchable in MLflow UI / API — include all fields
        # that list_models() surfaces so no artifact download is needed for listing.
        mlflow.set_tags({
            "task_id":           task_id,
            "user_id":           user_id,
            "session_id":        session_id,
            "model_type":        model_type,
            "model_name":        model_name,
            "model_description": model_description,
            "model_version":     model_version,
            "created_at":        datetime.now(timezone.utc).isoformat(),
        })

    # ---- stdout summary (mlflow_run_id is the canonical output identifier) ----
    print(
        f"\n[Done] task_id={task_id}  mlflow_run_id={mlflow_run_id}  "
        f"model={model_name}  version={model_version}  runtime={runtime_s}s"
    )
    print(f"```json\n{json.dumps(output_metrics, indent=2)}\n```")


# =========================================================
# Entry Point
# =========================================================


def main() -> None:
    setup_gcp_credentials()
    setup_mlflow()
    init_ray()
    training_main()


if __name__ == "__main__":
    main()
