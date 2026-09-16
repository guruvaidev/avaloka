"""
ray_inference_job.py — Avaloka Ray Inference Job
=================================================
Runs distributed inference on a dataset using a trained DynamicMLP model,
writing enriched prediction results to GCS.

Designed to be submitted to a Ray cluster by InferenceInterface.inference_on_ray().
Reads model weights and config from MLflow, streams the input dataset via Ray Data,
and writes a prediction CSV to GCS.

Streaming / memory notes:
  - InferenceInterface handles model loading and caching; this job loads the model
    once on the driver and broadcasts weights to workers via a Ray actor.
  - Input dataset is streamed via Ray Data — never fully materialised on one node.
  - Per-batch inference: workers receive blocks, run forward pass, emit predictions.
  - Output is written partition-by-partition to GCS and merged into a single CSV.
  - String/object columns in the dataset are carried through as metadata; only
    feature_names columns are fed to the model.

Autoscaling notes:
  - Ray is initialised with address="auto" to attach to the running cluster.
  - NUM_WORKERS / CPUS_PER_WORKER are read from env vars (same pattern as ray_job.py).
  - concurrency=num_workers (single int) on map_batches — tuple form only valid for
    callable classes, not plain functions.
  - filter() and drop_columns() do NOT support concurrency — never pass it to them.
  - DataContext resource_limits are set to inf so Ray Data can use all autoscaler nodes.

Env vars (MLflow — model source):
  MLFLOW_TRACKING_URI       — tracking server URI
  MLFLOW_BACKEND_STORE_URI  — backend store URI
  MLFLOW_DEFAULT_ARTIFACT_ROOT — artifact root (gs://, s3://, or local)
  MLFLOW_EXPERIMENT_NAME    — experiment name (default: "avaloka-training")

Env vars (Job identity):
  TASK_ID          — unique job identifier (default: new UUID)
  MLFLOW_RUN_ID    — MLflow run ID of the model to use for inference (required)

Env vars (I/O):
  DATA_SOURCE_URI  — URI of the input dataset (required)
  GCS_BUCKET       — GCS bucket name for output (required for GCS upload)
  OUTPUT_FILENAME  — output CSV filename (default: predictions_<run_id>_<task_id>.csv)
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import ray
import ray.data as rd
import torch
import torch.nn.functional as F
from ray import train
from ray.data import DataContext

from src.mlflow_manager import MLflowManager
from src.model import DynamicMLP, ModelConfig
from loader.sql import load_dataset_from_sql
from loader.url import load_dataset_from_url

try:
    from pyarrow.fs import GcsFileSystem
    _GCS_FS_AVAILABLE = True
except ImportError:
    GcsFileSystem = None  # type: ignore[assignment,misc]
    _GCS_FS_AVAILABLE = False

try:
    from google.cloud import storage as gcs_storage
    _GCS_STORAGE_AVAILABLE = True
except ImportError:
    gcs_storage = None  # type: ignore[assignment]
    _GCS_STORAGE_AVAILABLE = False


os.environ["RAY_SCHEDULER_SPREAD_THRESHOLD"] = "0.0"

cpus_per_worker = int(os.getenv("CPUS_PER_WORKER", 4))
num_workers     = int(os.getenv("NUM_WORKERS", 2))
min_workers     = int(os.getenv("MIN_WORKERS", 1))
max_workers     = int(os.getenv("MAX_WORKERS", num_workers * 2))


# =========================================================
# GCP Credentials  (mirrors ray_job.py)
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
# Init Ray  (mirrors ray_job.py)
# =========================================================


def init_ray() -> None:
    """
    Connect to an autoscaling Ray cluster.

    Same initialisation pattern as ray_job.py:
      - address="auto"  → attaches to a running cluster.
      - ignore_reinit_error=True  → safe to call multiple times.
      - DataContext resource_limits set to inf → no implicit CPU/GPU caps.
      - locality_with_output=True → reduce cross-node transfers.
    """
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        runtime_env={"worker_process_setup_hook": setup_gcp_credentials},
    )

    ctx = DataContext.get_current()
    ctx.target_max_block_size = 256 * 1024 * 1024
    ctx.execution_options.preserve_order = False
    ctx.execution_options.resource_limits.cpu = float("inf")
    ctx.execution_options.resource_limits.gpu = float("inf")
    ctx.execution_options.locality_with_output = True

    nodes = ray.nodes()
    alive = [n for n in nodes if n.get("Alive")]
    print(f"[Ray Init] Cluster nodes alive={len(alive)} / total={len(nodes)}")
    resources = ray.cluster_resources()
    print(f"[Ray Init] Cluster resources: {resources}")


# =========================================================
# MLflow Setup  (mirrors ray_job.py)
# =========================================================


def setup_mlflow() -> None:
    """
    Configure MLflow from environment variables.

    Effective URI priority:
      1. MLFLOW_TRACKING_URI
      2. MLFLOW_BACKEND_STORE_URI
      3. local ./mlruns
    """
    tracking_uri    = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    backend_store   = os.getenv("MLFLOW_BACKEND_STORE_URI", "").strip()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "avaloka-training").strip()

    effective_uri = tracking_uri or backend_store or "mlruns"
    import mlflow
    mlflow.set_tracking_uri(effective_uri)
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] tracking_uri={effective_uri}  experiment={experiment_name}")


# =========================================================
# Dataset Loading  (mirrors ray_job.py)
# =========================================================


def load_dataset(uri: str) -> rd.Dataset:
    if uri.startswith(("postgresql://", "mysql://", "sqlite://", "mssql://")):
        return load_dataset_from_sql(uri)
    return load_dataset_from_url(uri)


# =========================================================
# Ray Actor: Model Server
# Holds the model weights in one place; workers call remote
# methods to avoid serialising the full model per task.
# =========================================================


@ray.remote(num_cpus=1)
class ModelServer:
    """
    Singleton Ray actor that holds the loaded DynamicMLP.

    Workers call predict_batch() via ray.get() to run forward passes.
    This avoids re-serialising model weights for every map_batches call.

    Parameters
    ----------
    mlflow_run_id : str
    device : str
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
    """

    def __init__(self, mlflow_run_id: str, device: str = "cpu") -> None:
        self._device  = device
        self._model, self._cfg = MLflowManager().load_model(
            mlflow_run_id=mlflow_run_id,
            device=device,
            dst_dir=f"/tmp/mlflow_models/{mlflow_run_id}",
        )
        print(f"[ModelServer] Loaded model '{self._cfg.model_name}' v{self._cfg.model_version}")

    def get_config(self) -> dict:
        """Return the ModelConfig serialised as a dict for worker broadcast."""
        return json.loads(self._cfg.to_json())

    def predict_batch(self, feature_matrix: List[List[float]]) -> dict:
        """
        Run a batched forward pass and return decoded predictions.

        Parameters
        ----------
        feature_matrix : list[list[float]]
            Shape (N, num_features).

        Returns
        -------
        dict with keys:
            predictions, predicted_class_indices, raw_outputs,
            probabilities (classification only)
        """
        tensor = torch.tensor(feature_matrix, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            logits = self._model(tensor)  # (N, output_size)

        cfg = self._cfg

        if "classification" in cfg.model_type.lower():
            probs_tensor = F.softmax(logits, dim=1)
            probs_np     = probs_tensor.cpu().numpy()
            pred_indices = probs_np.argmax(axis=1).tolist()

            num_classes = logits.shape[1]
            class_names = (
                cfg.class_names
                if cfg.class_names and len(cfg.class_names) == num_classes
                else [str(k) for k in range(num_classes)]
            )
            predictions = [class_names[idx] for idx in pred_indices]
            raw_outputs = [[round(v, 6) for v in row] for row in logits.cpu().tolist()]

            # Per-class probability dicts: [{class: prob, ...}, ...]
            prob_dicts = [
                {class_names[k]: round(float(probs_np[n, k]), 6) for k in range(num_classes)}
                for n in range(len(feature_matrix))
            ]

            return {
                "predictions":             predictions,
                "predicted_class_indices": pred_indices,
                "raw_outputs":             raw_outputs,
                "probabilities":           prob_dicts,
                "class_names":             class_names,
                "model_type":              cfg.model_type,
            }

        else:
            # Regression
            if logits.shape[-1] == 1:
                scaled_pred_values = logits[:, 0].cpu().tolist()
            else:
                scaled_pred_values = logits.mean(dim=1).cpu().tolist()
            target_scaling = (cfg.preprocessing or {}).get("target_scaling") or {}

            def inverse_target(value: float) -> float:
                if target_scaling.get("method") != "standard":
                    return float(value)
                return (
                    float(value) * float(target_scaling.get("scale", 1.0))
                    + float(target_scaling.get("mean", 0.0))
                )

            pred_values = [round(inverse_target(v), 6) for v in scaled_pred_values]
            raw_outputs = [
                [round(inverse_target(v), 6) for v in row]
                for row in logits.cpu().tolist()
            ]

            return {
                "predictions":             pred_values,
                "predicted_class_indices": [None] * len(feature_matrix),
                "raw_outputs":             raw_outputs,
                "probabilities":           [{}] * len(feature_matrix),
                "class_names":             [],
                "model_type":              cfg.model_type,
            }


# =========================================================
# Batch inference function (used inside map_batches)
# =========================================================

def _encode_feature_value(feat: str, raw: Any, cfg: dict) -> float:
    categorical = (cfg["preprocessing"] or {}).get("categorical_features", {})
    if feat in categorical and not isinstance(raw, (int, float)):
        meta = categorical.get(feat) or {}
        mapping = meta.get("mapping") or {}
        unknown_value = float(meta.get("unknown_value", -1.0))
        if raw is None:
            value = unknown_value
        else:
            value = float(mapping.get(str(raw), unknown_value))

        scaler = ((cfg["preprocessing"] or {}).get("feature_scaling") or {}).get(feat) or {}
        if scaler.get("method") == "standard":
            return (value - float(scaler.get("mean", 0.0))) / float(scaler.get("scale", 1.0))
        return value

    if raw is None:
        scaler = ((cfg["preprocessing"] or {}).get("feature_scaling") or {}).get(feat) or {}
        value = float(scaler.get("mean", 0.0))
        logger.warning("Feature '%s' missing from input row — defaulting to its training mean.", feat)
    else:
        try:
            value = float(raw)
        except (ValueError, TypeError):
            scaler = ((cfg["preprocessing"] or {}).get("feature_scaling") or {}).get(feat) or {}
            value = float(scaler.get("mean", 0.0))
            logger.warning(
                "Feature '%s' value %r cannot be cast to float — defaulting to its training mean.",
                feat,
                raw,
            )

    scaler = ((cfg["preprocessing"] or {}).get("feature_scaling") or {}).get(feat) or {}
    if scaler.get("method") == "standard":
        scale = float(scaler.get("scale", 1.0))
        if not np.isfinite(scale) or abs(scale) <= np.finfo(np.float32).eps:
            scale = 1.0
        return (value - float(scaler.get("mean", 0.0))) / scale
    return value

def make_inference_fn(model_server_handle, cfg_dict: dict):
    """
    Factory that returns a map_batches-compatible function closed over the
    ModelServer actor handle and the serialised ModelConfig dict.

    Keeps the returned function picklable — Ray serialises closures, not
    class instances, so the actor handle (a lightweight ref) is safe.

    Parameters
    ----------
    model_server_handle : ray.actor.ActorHandle
        Reference to the ModelServer actor.
    cfg_dict : dict
        Serialised ModelConfig (from ModelServer.get_config()).

    Returns
    -------
    Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]
    """
    feature_names  = cfg_dict["feature_names"]
    target_column  = cfg_dict["target_column"]
    model_type     = cfg_dict["model_type"]

    def _infer_batch(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Called by Ray Data map_batches for each numpy batch.

        Extracts feature columns, runs inference via the ModelServer actor,
        and appends prediction columns to the batch dict.
        """
        n = len(next(iter(batch.values())))

        # Build feature matrix — missing features default to 0.0
        feature_matrix: List[List[float]] = []
        for i in range(n):
            row = [
                _encode_feature_value(feat, batch[feat][i], cfg_dict)
                for feat in feature_names
            ]
            feature_matrix.append(row)

        # Remote call to ModelServer — blocks until done.
        result = ray.get(model_server_handle.predict_batch.remote(feature_matrix))

        batch["prediction"]             = np.array(result["predictions"])
        batch["predicted_class_index"]  = np.array(
            [idx if idx is not None else -1 for idx in result["predicted_class_indices"]],
            dtype=np.int64,
        )
        batch["raw_output"]             = np.array(
            [str(row) for row in result["raw_outputs"]]
        )

        # Overwrite target column with predictions (mirrors inference.py inference_local)
        if target_column in batch:
            batch[target_column] = batch["prediction"]

        if "classification" in model_type:
            class_names = result.get("class_names", [])
            prob_dicts  = result.get("probabilities", [])

            # One numpy column per class: prob_<class_name>
            for cls in class_names:
                col_name        = f"prob_{cls}"
                batch[col_name] = np.array([d.get(cls, 0.0) for d in prob_dicts], dtype=np.float32)

        return batch

    return _infer_batch


# =========================================================
# GCS Upload helper  (mirrors inference.py)
# =========================================================


def upload_to_gcs(local_path: str, gcs_bucket: str, gcs_blob_name: str) -> str:
    """
    Upload a local file to GCS and return the gs:// URI.

    Parameters
    ----------
    local_path : str
    gcs_bucket : str
        Bucket name without gs://.
    gcs_blob_name : str

    Returns
    -------
    str
        ``gs://<bucket>/<blob>``

    Raises
    ------
    RuntimeError
        On upload failure.
    """
    gcs_uri = f"gs://{gcs_bucket}/{gcs_blob_name}"

    if not _GCS_STORAGE_AVAILABLE:
        raise RuntimeError(
            "google-cloud-storage is not installed. "
            "Cannot upload to GCS. Install it with: pip install google-cloud-storage"
        )

    try:
        client = gcs_storage.Client()
        bucket = client.bucket(gcs_bucket)
        blob   = bucket.blob(gcs_blob_name)
        blob.upload_from_filename(local_path, content_type="text/csv")
        print(f"[GCS] Uploaded {local_path} → {gcs_uri}")
        return gcs_uri
    except Exception as exc:
        raise RuntimeError(
            f"Failed to upload {local_path} to GCS ({gcs_uri}): {exc}"
        ) from exc


# =========================================================
# Inference Entry Point
# =========================================================


def inference_main() -> dict:
    """
    Main distributed inference pipeline.

    Steps
    -----
    1. Read env vars.
    2. Load ModelConfig + weights from MLflow → start ModelServer actor.
    3. Load input dataset as a Ray Dataset.
    4. Validate that required feature columns are present.
    5. Stream dataset through map_batches inference → produce enriched dataset.
    6. Collect results to a pandas DataFrame.
    7. Write CSV to /tmp and upload to GCS.
    8. Print a JSON summary to stdout (consumed by InferenceInterface.inference_on_ray).

    Returns
    -------
    dict
        Output metadata including output_uri, row_count, and runtime_s.
        Also printed as JSON to stdout for the Ray job result parser.
    """
    start_time = time.time()

    # ---- Env ----
    task_id        = os.getenv("TASK_ID", str(uuid.uuid4()))
    mlflow_run_id  = os.getenv("MLFLOW_RUN_ID", "").strip()
    data_uri       = os.getenv("DATA_SOURCE_URI", "").strip()
    gcs_bucket     = os.getenv("GCS_BUCKET", "").strip()
    output_filename = os.getenv(
        "OUTPUT_FILENAME",
        f"predictions_{mlflow_run_id}_{task_id}.csv",
    )
    use_gpu        = os.getenv("USE_GPU", "false").lower() == "true"
    device         = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    if not mlflow_run_id:
        raise ValueError("MLFLOW_RUN_ID env var is required for inference.")
    if not data_uri:
        raise ValueError("DATA_SOURCE_URI env var is required for inference.")

    print(f"[Inference] task_id={task_id}  mlflow_run_id={mlflow_run_id}")
    print(f"[Inference] data_uri={data_uri}  device={device}")
    print(f"[Inference] workers num={num_workers}  cpus_per_worker={cpus_per_worker}")

    # ---- Start ModelServer actor ----
    # Placed on a dedicated node to avoid competing with map_batches workers.
    print("[ModelServer] Starting actor ...")
    model_server = ModelServer.options(
        name="model_server",
        lifetime="detached",
        num_cpus=1,
        scheduling_strategy="DEFAULT",
    ).remote(mlflow_run_id=mlflow_run_id, device=device)

    cfg_dict: dict = ray.get(model_server.get_config.remote())
    feature_names: List[str] = cfg_dict["feature_names"]
    target_column: str       = cfg_dict["target_column"]
    model_type: str          = cfg_dict["model_type"]
    model_name: str          = cfg_dict.get("model_name", "unknown")
    model_version: str       = cfg_dict.get("model_version", "1")

    print(f"[Inference] Model: name={model_name}  version={model_version}  type={model_type}")
    print(f"[Inference] Feature columns ({len(feature_names)}): {feature_names}")
    print(f"[Inference] Target column: {target_column}")

    # ---- Load dataset (lazy/streaming) ----
    print(f"[Data] Loading from {data_uri} ...")
    ds = load_dataset(data_uri)

    # ---- Validate feature presence ----
    # Check schema for the feature columns; warn on missing but do not crash —
    # inference_fn defaults missing features to 0.0 and logs a warning.
    schema_names = set(ds.schema().names) if hasattr(ds.schema(), "names") else set()
    missing      = [f for f in feature_names if f not in schema_names]
    if missing:
        print(
            f"[Warning] The following feature columns are missing from the dataset schema "
            f"and will default to 0.0: {missing}"
        )

    # ---- Build inference batch function ----
    infer_fn = make_inference_fn(model_server, cfg_dict)

    # ---- Run distributed inference ----
    # concurrency=num_workers (single int) — plain functions do NOT support tuple form.
    print("[Inference] Running distributed map_batches inference ...")
    ds_out = ds.map_batches(
        infer_fn,
        batch_format="numpy",
        num_cpus=cpus_per_worker,
        concurrency=num_workers,
    )

    # ---- Collect to pandas ----
    # Materialise into a single DataFrame for CSV serialisation.
    # For very large datasets, consider writing per-partition parquets and
    # merging externally — this is a reasonable default for batch inference jobs.
    print("[Inference] Collecting results ...")
    result_df: pd.DataFrame = ds_out.to_pandas()
    row_count = len(result_df)
    print(f"[Inference] Collected {row_count} rows.")

    # ---- Write local CSV ----
    local_path = f"/tmp/{output_filename}"
    result_df.to_csv(local_path, index=False)
    print(f"[Inference] Written local CSV → {local_path}")

    # ---- Upload to GCS (if configured) ----
    gcs_blob_name = f"user_output/{output_filename}"
    output_uri    = local_path  # fallback if GCS not configured

    if gcs_bucket:
        output_uri = upload_to_gcs(local_path, gcs_bucket, gcs_blob_name)
    else:
        print("[Inference] GCS_BUCKET not set — skipping GCS upload. Output is local only.")

    # ---- Runtime ----
    runtime_s = round(time.time() - start_time, 3)

    # ---- Output summary ----
    output_meta = {
        "task_id":       task_id,
        "mlflow_run_id": mlflow_run_id,
        "model_name":    model_name,
        "model_version": model_version,
        "model_type":    model_type,
        "output_uri":    output_uri,
        "row_count":     row_count,
        "runtime_s":     runtime_s,
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "feature_names": feature_names,
        "target_column": target_column,
    }

    print(f"\n[Done] task_id={task_id}  output_uri={output_uri}  rows={row_count}  runtime={runtime_s}s")
    # Print JSON summary to stdout — consumed by InferenceInterface.get_job_result()
    print(f"```json\n{json.dumps(output_meta, indent=2)}\n```")

    return output_meta


# =========================================================
# Entry Point
# =========================================================


def main() -> None:
    setup_gcp_credentials()
    setup_mlflow()
    init_ray()
    inference_main()


if __name__ == "__main__":
    main()
