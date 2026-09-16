"""
inference.py — Inference Interface
====================================
A pure Python interface class — no API server, no Ray, no GCP dependency.
Designed to be imported and used directly in application code.
"""
from __future__ import annotations

import os
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from app.agents.mta_v2.mlflow_manager import MLflowManager
from app.agents.mta_v2.schema import ModelDetail, ModelSummary, RayConfig
from app.agents.mta_v2.model import DynamicMLP, ModelConfig
from app.agents.mta_v2.utils import load_dataset
from app.agents.pii_agent import transform_pii_value
from app.core import cloud_config

logger = logging.getLogger(__name__)


# =========================================================
# Prediction Result DTOs
# =========================================================


class PredictionResult:
    """
    Holds the output of a single-row inference call.

    Attributes
    ----------
    mlflow_run_id : str
    model_name : str
    model_version : str
    model_type : str
        ``"classification"`` or ``"regression"``.

    For classification:
        prediction            : str   — winning class label
        predicted_class_index : int   — index into class_names
        probabilities         : dict  — {class_label: float} softmax probabilities
        raw_output            : list  — raw logits as list[float]

    For regression:
        prediction            : float — raw model output
        raw_output            : list  — same value wrapped in a list
        predicted_class_index : None
        probabilities         : {}
    """

    def __init__(
        self,
        mlflow_run_id:         str,
        model_name:            str,
        model_version:         str,
        model_type:            str,
        prediction:            Any,
        raw_output:            List[float],
        predicted_class_index: Optional[int] = None,
        probabilities:         Optional[Dict[str, float]] = None,
    ) -> None:
        self.mlflow_run_id         = mlflow_run_id
        self.model_name            = model_name
        self.model_version         = model_version
        self.model_type            = model_type
        self.prediction            = prediction
        self.raw_output            = raw_output
        self.predicted_class_index = predicted_class_index
        self.probabilities         = probabilities or {}

    def to_dict(self) -> dict:
        """Serialise the result to a plain dict (JSON-safe)."""
        return {
            "mlflow_run_id":         self.mlflow_run_id,
            "model_name":            self.model_name,
            "model_version":         self.model_version,
            "model_type":            self.model_type,
            "prediction":            self.prediction,
            "predicted_class_index": self.predicted_class_index,
            "probabilities":         self.probabilities,
            "raw_output":            self.raw_output,
        }

    def __repr__(self) -> str:
        if "classification" in self.model_type:
            return (
                f"PredictionResult(model={self.model_name!r}, v={self.model_version!r}, "
                f"prediction={self.prediction!r}, "
                f"probs={{{', '.join(f'{k}: {v:.3f}' for k, v in self.probabilities.items())}}})"
            )
        return (
            f"PredictionResult(model={self.model_name!r}, v={self.model_version!r}, "
            f"prediction={self.prediction!r})"
        )


class DatasetPredictionResult:
    """
    Holds the output of an inference call on a whole dataset.

    Attributes
    ----------
    mlflow_run_id : str
    model_name : str
    model_version : str
    model_type : str
    output_uri : str
        Where the full prediction results are stored (e.g. GCS path).
    """

    def __init__(
        self,
        mlflow_run_id: str,
        model_name: str,
        model_version: str,
        model_type: str,
        output_uri: str,
    ) -> None:
        self.mlflow_run_id = mlflow_run_id
        self.model_name = model_name
        self.model_version = model_version
        self.model_type = model_type
        self.output_uri = output_uri

    def to_dict(self) -> dict:
        """Serialise the result to a plain dict (JSON-safe)."""
        return {
            "mlflow_run_id": self.mlflow_run_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "output_uri": self.output_uri,
        }

    def __repr__(self) -> str:
        return (
            f"DatasetPredictionResult(model={self.model_name!r}, v={self.model_version!r}, "
            f"output_uri={self.output_uri!r})"
        )


# =========================================================
# InferenceInterface
# =========================================================

# Ray / GKE constants — mirror the pattern in RayTrainer
# Lazy: an unset project must fail where a deployment is attempted, naming the
# variable, rather than preventing this module from importing at all.
def _ray_gcp_project_id() -> str:
    return cloud_config.project_id()


def _ray_inference_docker_uri() -> str:
    return cloud_config.image(
        "RAY_INFERENCE_DOCKER_URI", "inference-docker-image", "test-v0.0.1")
_RAY_GKE_CLUSTER_NAME    = os.getenv("RAY_GKE_CLUSTER_NAME", "ray-gke-trainer")
_RAY_GKE_CLUSTER_LOCATION = os.getenv("RAY_GKE_CLUSTER_LOCATION", "us-central1")
_GCP_SA_JSON_PATH = os.getenv("GCP_SERVICE_ACCOUNT_JSON_PATH", "secrets/ray-bucket-cred.json")


class InferenceInterface:
    """
    High-level interface for model discovery and inference.

    Models are loaded lazily on the first predict() call and cached in memory.
    Call unload_model() to release memory for a specific run, or
    clear_cache() to release all loaded models.

    Parameters
    ----------
    device : str
        PyTorch device for inference, e.g. ``"cpu"``, ``"cuda"``.
    model_cache_dir : str, optional
        Directory to cache downloaded model files.
        Defaults to ``/tmp/mlflow_models``.
    gcs_output_bucket : str, optional
        GCS bucket name (without ``gs://``) where inference_local() writes
        prediction CSVs.  Defaults to the ``GCS_OUTPUT_BUCKET`` env var.
    """

    def __init__(
        self,
        device:             str = "cpu",
        model_cache_dir:    Optional[str] = None,
        gcs_output_bucket:  Optional[str] = None,
    ) -> None:
        self._manager    = MLflowManager()
        self._device     = device
        self._cache_dir  = model_cache_dir or "/tmp/mlflow_models"
        self._gcs_bucket = gcs_output_bucket or os.environ.get("GCS_OUTPUT_BUCKET", "")

        # In-memory model cache: mlflow_run_id → (DynamicMLP, ModelConfig)
        self._model_cache: Dict[str, Tuple[DynamicMLP, ModelConfig]] = {}

    # ----------------------------------------------------------
    # Model Discovery
    # ----------------------------------------------------------

    def list_models(
        self,
        user_id:         Optional[str] = None,
        session_id:      Optional[str] = None,
        experiment_name: Optional[str] = None,
        filter_string:   str = "",
        max_results:     int = 200,
    ) -> List[ModelSummary]:
        """
        Return a lightweight summary of all trained runs in the experiment.

        Each ModelSummary contains:
            mlflow_run_id, model_name, model_type,
            model_description, model_version, status, created_at,
            final_val_loss, plus accuracy/F1 for classification or
            MAE/RMSE/R2 for regression.

        Parameters
        ----------
        user_id : str, optional
            Filter by user ID.
        session_id : str, optional
            Filter by session ID.
        experiment_name : str, optional
            Override the experiment to query.
        filter_string : str
            MLflow search filter, e.g. ``"tags.model_type = 'classification'"``.
        max_results : int
            Maximum number of runs to return (MLflow pagination ceiling).

        Returns
        -------
        List[ModelSummary]
            Sorted newest-first.
        """
        all_models = self._manager.list_models(
            experiment_name=experiment_name,
            filter_string=filter_string,
            max_results=max_results,
        )
        if user_id:
            all_models = [m for m in all_models if m.get("user_id") == user_id]
        if session_id:
            all_models = [m for m in all_models if m.get("session_id") == session_id]
        return all_models

    def get_model_details(self, mlflow_run_id: str) -> Optional[ModelDetail]:
        """
        Return full detail for a single run including architecture, data info,
        hyperparameters, all metrics, and training history.

        Parameters
        ----------
        mlflow_run_id : str

        Returns
        -------
        ModelDetail or None.
        """
        return self._manager.get_model_details(mlflow_run_id)

    # ----------------------------------------------------------
    # Model Loading / Cache
    # ----------------------------------------------------------

    def load_model(self, mlflow_run_id: str) -> Tuple[DynamicMLP, ModelConfig]:
        """
        Load (and cache) a model by mlflow_run_id.

        Subsequent calls with the same ID return the cached instance without
        hitting MLflow or disk again.

        Parameters
        ----------
        mlflow_run_id : str

        Returns
        -------
        tuple[DynamicMLP, ModelConfig]
        """
        if mlflow_run_id not in self._model_cache:
            model, cfg = self._manager.load_model(
                mlflow_run_id=mlflow_run_id,
                device=self._device,
                dst_dir=f"{self._cache_dir}/{mlflow_run_id}",
            )
            self._model_cache[mlflow_run_id] = (model, cfg)
        return self._model_cache[mlflow_run_id]

    def unload_model(self, mlflow_run_id: str) -> None:
        """Remove a model from the in-memory cache."""
        self._model_cache.pop(mlflow_run_id, None)

    def clear_cache(self) -> None:
        """Evict all cached models from memory."""
        self._model_cache.clear()

    @property
    def loaded_run_ids(self) -> List[str]:
        """List of mlflow_run_ids currently held in memory."""
        return list(self._model_cache.keys())

    # ----------------------------------------------------------
    # Row preprocessing
    # ----------------------------------------------------------

    @staticmethod
    def _encoded_numeric_value(raw: Any) -> Optional[float]:
        if isinstance(raw, bool):
            return None
        if isinstance(raw, (int, float, np.integer, np.floating)):
            value = float(raw)
            return value if np.isfinite(value) else None
        if isinstance(raw, str):
            stripped = raw.strip()
            if not stripped:
                return None
            try:
                value = float(stripped)
            except ValueError:
                return None
            return value if np.isfinite(value) else None
        return None

    @staticmethod
    def _scale_feature_value(feat: str, value: float, cfg: ModelConfig) -> float:
        scaler = ((cfg.preprocessing or {}).get("feature_scaling") or {}).get(feat) or {}
        if scaler.get("method") != "standard":
            return float(value)
        mean = float(scaler.get("mean", 0.0))
        scale = float(scaler.get("scale", 1.0))
        if not np.isfinite(scale) or abs(scale) <= np.finfo(np.float32).eps:
            scale = 1.0
        return (float(value) - mean) / scale

    @staticmethod
    def _inverse_target_value(value: float, cfg: ModelConfig) -> float:
        scaler = (cfg.preprocessing or {}).get("target_scaling") or {}
        if scaler.get("method") != "standard":
            return float(value)
        return float(value) * float(scaler.get("scale", 1.0)) + float(scaler.get("mean", 0.0))

    @staticmethod
    def _deidentify_feature_value(feat: str, raw: Any, cfg: ModelConfig) -> Any:
        """Apply the model's persisted PII contract before feature encoding."""
        meta = ((cfg.preprocessing or {}).get("pii_transformations") or {}).get(feat)
        if not meta:
            return raw
        return transform_pii_value(
            raw,
            column=feat,
            kind=meta["kind"],
            strategy=meta["strategy"],
        )

    @staticmethod
    def _encode_feature_value(feat: str, raw: Any, cfg: ModelConfig) -> float:
        raw = InferenceInterface._deidentify_feature_value(feat, raw, cfg)
        categorical = (cfg.preprocessing or {}).get("categorical_features", {})
        if feat in categorical:
            meta = categorical.get(feat) or {}
            mapping = meta.get("mapping") or {}
            unknown_value = float(meta.get("unknown_value", -1.0))
            if raw is None:
                value = unknown_value
            else:
                raw_key = str(raw)
                if raw_key in mapping:
                    value = float(mapping[raw_key])
                else:
                    encoded_value = InferenceInterface._encoded_numeric_value(raw)
                    value = encoded_value if encoded_value is not None else unknown_value
        elif raw is None:
            scaler = ((cfg.preprocessing or {}).get("feature_scaling") or {}).get(feat) or {}
            value = float(scaler.get("mean", 0.0))
            logger.warning("Feature '%s' missing from input row — defaulting to its training mean.", feat)
        else:
            try:
                value = float(raw)
            except (ValueError, TypeError):
                scaler = ((cfg.preprocessing or {}).get("feature_scaling") or {}).get(feat) or {}
                value = float(scaler.get("mean", 0.0))
                logger.warning(
                    "Feature '%s' value %r cannot be cast to float — defaulting to its training mean.",
                    feat,
                    raw,
                )

        return InferenceInterface._scale_feature_value(feat, value, cfg)

    @staticmethod
    def _model_feature_values(row: Dict[str, Any], cfg: ModelConfig) -> List[float]:
        """Transform one raw row into the exact tensor layout used for training."""
        preprocessing = cfg.preprocessing or {}
        model_feature_names = list(preprocessing.get("model_feature_names") or [])
        if int(preprocessing.get("version", 1)) < 3 or not model_feature_names:
            return [
                InferenceInterface._encode_feature_value(feat, row.get(feat), cfg)
                for feat in cfg.feature_names
            ]

        encoded: Dict[str, float] = {}
        categorical = preprocessing.get("categorical_features") or {}
        imputation = preprocessing.get("imputation") or {}
        feature_scaling = preprocessing.get("feature_scaling") or {}

        for feat in cfg.feature_names:
            raw = InferenceInterface._deidentify_feature_value(
                feat, row.get(feat), cfg
            )
            if feat in categorical and categorical[feat].get("encoding") == "one_hot":
                meta = categorical[feat]
                fill_value = str((imputation.get(feat) or {}).get("fill_value", "__MISSING__"))
                if raw is None or (isinstance(raw, str) and not raw.strip()) or pd.isna(raw):
                    value = fill_value
                else:
                    value = str(raw).strip()
                categories = [str(item) for item in meta.get("categories", [])]
                for category, output_column in zip(categories, meta.get("output_columns", [])):
                    encoded[str(output_column)] = float(value == category)
                encoded[str(meta["unknown_column"])] = float(value not in categories)
                continue

            scaler = feature_scaling.get(feat) or {}
            numeric_value: Optional[float]
            if scaler.get("input_type") == "datetime":
                timestamp = pd.to_datetime(raw, errors="coerce")
                numeric_value = None if pd.isna(timestamp) else float(timestamp.value / 1_000_000_000)
            else:
                numeric_value = InferenceInterface._encoded_numeric_value(raw)
            if numeric_value is None:
                numeric_value = float((imputation.get(feat) or {}).get("fill_value", 0.0))
            output_column = str(scaler.get("output_column", feat))
            encoded[output_column] = InferenceInterface._scale_feature_value(feat, numeric_value, cfg)

        return [float(encoded.get(name, 0.0)) for name in model_feature_names]

    @staticmethod
    def _preprocess_input_dataframe(input_df: "pd.DataFrame", cfg: ModelConfig) -> "pd.DataFrame":
        model_feature_names = list((cfg.preprocessing or {}).get("model_feature_names") or [])
        if int((cfg.preprocessing or {}).get("version", 1)) >= 3 and model_feature_names:
            rows = [
                InferenceInterface._model_feature_values(row, cfg)
                for row in input_df.to_dict(orient="records")
            ]
            return pd.DataFrame(rows, columns=model_feature_names, index=input_df.index)

        prepared = input_df.copy()
        categorical = (cfg.preprocessing or {}).get("categorical_features", {})
        feature_scaling = (cfg.preprocessing or {}).get("feature_scaling", {})

        for feat in cfg.feature_names:
            if feat not in prepared.columns:
                continue
            prepared[feat] = prepared[feat].map(
                lambda raw, column=feat: InferenceInterface._deidentify_feature_value(
                    column, raw, cfg
                )
            )
            if feat in categorical:
                meta = categorical.get(feat) or {}
                mapping = meta.get("mapping") or {}
                unknown_value = float(meta.get("unknown_value", -1.0))
                def encode_categorical(raw: Any) -> float:
                    if pd.isna(raw):
                        return unknown_value
                    raw_key = str(raw)
                    if raw_key in mapping:
                        return float(mapping[raw_key])
                    encoded_value = InferenceInterface._encoded_numeric_value(raw)
                    if encoded_value is not None:
                        return encoded_value
                    return unknown_value

                prepared[feat] = prepared[feat].map(encode_categorical).astype(np.float32)
            else:
                scaler = feature_scaling.get(feat) or {}
                fill_value = (
                    float(scaler.get("mean", 0.0))
                    if scaler.get("method") == "standard"
                    else 0.0
                )
                prepared[feat] = (
                    pd.to_numeric(prepared[feat], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(fill_value)
                    .astype(np.float32)
                )
            scaler = feature_scaling.get(feat) or {}
            if scaler.get("method") == "standard":
                mean = float(scaler.get("mean", 0.0))
                scale = float(scaler.get("scale", 1.0))
                if not np.isfinite(scale) or abs(scale) <= np.finfo(np.float32).eps:
                    scale = 1.0
                prepared[feat] = ((prepared[feat].astype(np.float64) - mean) / scale).astype(np.float32)
        return prepared

    def _build_input_tensor(
        self,
        row: Dict[str, Any],
        cfg: ModelConfig,
    ) -> torch.Tensor:
        """
        Convert a raw feature-dict into a ``(1, num_features)`` float32 tensor.

        Feature order follows ``cfg.feature_names`` exactly.  Missing features
        default to ``0.0`` (logged as a warning).  Non-numeric values that
        cannot be cast to float also default to ``0.0``.

        Parameters
        ----------
        row : dict
            Feature key→value pairs.
        cfg : ModelConfig
            Provides the canonical feature ordering.

        Returns
        -------
        torch.Tensor
            Shape ``(1, num_features)``, dtype ``float32``.
        """
        values = self._model_feature_values(row, cfg)

        return torch.tensor([values], dtype=torch.float32, device=self._device)

    # ----------------------------------------------------------
    # Inference
    # ----------------------------------------------------------

    def _run_inference(
        self,
        model:  DynamicMLP,
        cfg:    ModelConfig,
        tensor: torch.Tensor,
    ) -> PredictionResult:
        """
        Core inference: single forward pass + decode output into a PredictionResult.

        Handles both classification (softmax + argmax) and regression (raw scalar).

        Parameters
        ----------
        model : DynamicMLP
            Must already be in eval() mode.
        cfg : ModelConfig
        tensor : torch.Tensor
            Shape ``(1, num_features)``.

        Returns
        -------
        PredictionResult
        """
        model.eval()
        with torch.no_grad():
            logits = model(tensor)  # shape: (1, output_size)

        raw_output = logits[0].cpu().tolist()

        if "classification" in cfg.model_type.lower():
            probs_tensor = F.softmax(logits[0], dim=0)
            probs_list   = probs_tensor.cpu().tolist()

            pred_idx   = int(probs_tensor.argmax().item())
            pred_label = (
                cfg.class_names[pred_idx]
                if cfg.class_names and pred_idx < len(cfg.class_names)
                else str(pred_idx)
            )
            probabilities = {
                (
                    cfg.class_names[i]
                    if cfg.class_names and i < len(cfg.class_names)
                    else str(i)
                ): round(p, 6)
                for i, p in enumerate(probs_list)
            }
            return PredictionResult(
                mlflow_run_id=cfg.mlflow_run_id,
                model_name=cfg.model_name,
                model_version=cfg.model_version,
                model_type=cfg.model_type,
                prediction=pred_label,
                raw_output=[round(v, 6) for v in raw_output],
                predicted_class_index=pred_idx,
                probabilities=probabilities,
            )

        else:
            # Regression: output may be (1,) or (1, 1)
            scaled_val = (
                float(logits[0, 0].item())
                if logits.shape[-1] == 1
                else float(logits[0].mean().item())
            )
            val = self._inverse_target_value(scaled_val, cfg)
            return PredictionResult(
                mlflow_run_id=cfg.mlflow_run_id,
                model_name=cfg.model_name,
                model_version=cfg.model_version,
                model_type=cfg.model_type,
                prediction=round(val, 6),
                raw_output=[round(self._inverse_target_value(v, cfg), 6) for v in raw_output],
                predicted_class_index=None,
                probabilities={},
            )

    def predict(
        self,
        mlflow_run_id: str,
        row:           Dict[str, Any],
    ) -> PredictionResult:
        """
        Run inference on a single input row.

        Parameters
        ----------
        mlflow_run_id : str
            Which model to use.
        row : dict
            Feature key-value pairs. Keys must match the model's
            ``feature_names``. Extra keys are ignored; missing keys default
            to ``0.0``.

        Returns
        -------
        PredictionResult
        """
        model, cfg = self.load_model(mlflow_run_id)
        tensor = self._build_input_tensor(row, cfg)
        return self._run_inference(model, cfg, tensor)

    def predict_batch(
        self,
        mlflow_run_id: str,
        rows:          List[Dict[str, Any]],
    ) -> List[PredictionResult]:
        """
        Run inference on a list of input rows.

        All rows are stacked into a single batched forward pass for efficiency.
        Results are returned in the same order as the input rows.

        Parameters
        ----------
        mlflow_run_id : str
        rows : list[dict]

        Returns
        -------
        List[PredictionResult]
        """
        if not rows:
            return []

        model, cfg = self.load_model(mlflow_run_id)

        # Stack all rows into one batch tensor: (N, num_features)
        tensors = [self._build_input_tensor(row, cfg) for row in rows]
        batch   = torch.cat(tensors, dim=0)

        model.eval()
        with torch.no_grad():
            logits = model(batch)  # (N, output_size)

        results: List[PredictionResult] = []
        for i in range(len(rows)):
            row_logits = logits[i]          # (output_size,)
            raw_output = [round(v, 6) for v in row_logits.cpu().tolist()]

            if "classification" in cfg.model_type.lower():
                probs_tensor = F.softmax(row_logits, dim=0)
                probs_list   = probs_tensor.cpu().tolist()

                pred_idx   = int(probs_tensor.argmax().item())
                pred_label = (
                    cfg.class_names[pred_idx]
                    if cfg.class_names and pred_idx < len(cfg.class_names)
                    else str(pred_idx)
                )
                probabilities = {
                    (
                        cfg.class_names[j]
                        if cfg.class_names and j < len(cfg.class_names)
                        else str(j)
                    ): round(p, 6)
                    for j, p in enumerate(probs_list)
                }
                results.append(
                    PredictionResult(
                        mlflow_run_id=cfg.mlflow_run_id,
                        model_name=cfg.model_name,
                        model_version=cfg.model_version,
                        model_type=cfg.model_type,
                        prediction=pred_label,
                        raw_output=raw_output,
                        predicted_class_index=pred_idx,
                        probabilities=probabilities,
                    )
                )
            else:
                # Regression
                scaled_val = (
                    float(row_logits[0].item())
                    if logits.shape[-1] == 1
                    else float(row_logits.mean().item())
                )
                val = self._inverse_target_value(scaled_val, cfg)
                results.append(
                    PredictionResult(
                        mlflow_run_id=cfg.mlflow_run_id,
                        model_name=cfg.model_name,
                        model_version=cfg.model_version,
                        model_type=cfg.model_type,
                        prediction=round(val, 6),
                        raw_output=[round(self._inverse_target_value(v, cfg), 6) for v in row_logits.cpu().tolist()],
                        predicted_class_index=None,
                        probabilities={},
                    )
                )

        return results

    def predict_numpy(
        self,
        mlflow_run_id: str,
        array:         "np.ndarray",
        feature_names: Optional[List[str]] = None,
    ) -> List[PredictionResult]:
        """
        Run inference on a 2-D numpy array.

        Parameters
        ----------
        mlflow_run_id : str
        array : np.ndarray
            Shape ``(N, num_features)``.
        feature_names : list[str], optional
            Column names matching the array columns. If ``None``, the model's
            stored ``feature_names`` order is assumed.

        Returns
        -------
        List[PredictionResult]

        Raises
        ------
        ValueError
            If *array* is not 2-D, or if the number of columns does not match
            ``len(feature_names)``.
        """
        model, cfg = self.load_model(mlflow_run_id)
        names = feature_names or cfg.feature_names

        if array.ndim != 2:
            raise ValueError(
                f"predict_numpy expects a 2-D array, got shape {array.shape}."
            )
        if len(names) != array.shape[1]:
            raise ValueError(
                f"feature_names length ({len(names)}) != array columns ({array.shape[1]})."
            )

        rows = [
            {names[j]: float(array[i, j]) for j in range(array.shape[1])}
            for i in range(array.shape[0])
        ]
        return self.predict_batch(mlflow_run_id, rows)

    def inference_local(
        self,
        mlflow_run_id: str,
        dataset_uri:   str
    ) -> DatasetPredictionResult:
        """
        Run inference on a whole dataset (e.g. CSV file or SQL table) and
        write the enriched results to a GCS bucket.

        The output CSV contains all original columns plus:
            - ``prediction``      : class label (classification) or float (regression)
            - ``predicted_class_index`` : winning class index (classification only)
            - ``raw_output``      : raw logit list serialised as a string
            - one ``prob_<class>`` column per class (classification only)

        Parameters
        ----------
        mlflow_run_id : str
        dataset_uri : str
            URI pointing to the input dataset, e.g.
            ``"gs://bucket/path/data.csv"`` or
            ``"postgresql://user:pass@host/db?table=tablename"``.

        Returns
        -------
        DatasetPredictionResult
            ``output_uri`` is the full ``gs://`` path to the written CSV.

        Raises
        ------
        ValueError
            If the input dataset is missing features required by the model, or
            if ``gcs_output_bucket`` was not configured.
        RuntimeError
            If the GCS upload fails.
        """
        # ── 1. Load dataset ────────────────────────────────────────────────
        input_df = load_dataset(dataset_uri, limit=None)

        # ── 2. Validate schema ─────────────────────────────────────────────
        model, cfg = self.load_model(mlflow_run_id)
        missing_features = [f for f in cfg.feature_names if f not in input_df.columns]
        if missing_features:
            raise ValueError(
                f"Input dataset is missing required features: {missing_features}"
            )

        # ── 3. Batched forward pass ────────────────────────────────────────
        prepared_df = self._preprocess_input_dataframe(input_df, cfg)
        tensor_feature_names = list((cfg.preprocessing or {}).get("model_feature_names") or cfg.feature_names)
        input_tensor = torch.tensor(
            prepared_df[tensor_feature_names].values,
            dtype=torch.float32,
            device=self._device,
        )

        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)  # (N, output_size)

        # ── 4. Decode predictions into columnar data ───────────────────────
        output_df = input_df.copy()

        if "classification" in cfg.model_type.lower():
            probs_tensor = F.softmax(logits, dim=1)          # (N, num_classes)
            probs_np     = probs_tensor.cpu().numpy()         # (N, num_classes)
            pred_indices = probs_np.argmax(axis=1)            # (N,)

            num_classes  = logits.shape[1]
            class_names  = (
                cfg.class_names
                if cfg.class_names and len(cfg.class_names) == num_classes
                else [str(k) for k in range(num_classes)]
            )

            output_df["prediction"] = [class_names[idx] for idx in pred_indices]
            output_df["predicted_class_index"] = pred_indices.tolist()
            output_df["raw_output"] = [
                str([round(v, 6) for v in row]) for row in logits.cpu().tolist()
            ]
            output_df[cfg.target_column] = output_df["prediction"]  # overwrite target column with predictions

            # One probability column per class
            for k, name in enumerate(class_names):
                output_df[f"prob_{name}"] = [
                    round(float(probs_np[n, k]), 6) for n in range(len(input_df))
                ]

        else:
            # Regression
            if logits.shape[-1] == 1:
                scaled_pred_values = logits[:, 0].cpu().tolist()
            else:
                scaled_pred_values = logits.mean(dim=1).cpu().tolist()

            pred_values = [self._inverse_target_value(value, cfg) for value in scaled_pred_values]

            output_df["prediction"] = [round(v, 6) for v in pred_values]
            output_df["predicted_class_index"] = None
            output_df["raw_output"] = [
                str([round(self._inverse_target_value(v, cfg), 6) for v in row])
                for row in logits.cpu().tolist()
            ]
            output_df[cfg.target_column] = output_df["prediction"]

        # ── 5. Write CSV to a local temp file ─────────────────────────────
        local_filename = f"predictions_{mlflow_run_id}_{uuid.uuid4().hex}.csv"
        local_path     = f"/tmp/{local_filename}"
        output_df.to_csv(local_path, index=False)
        logger.info("Predictions written locally to %s (%d rows)", local_path, len(output_df))

        # ── 6. Upload to GCS ───────────────────────────────────────────────
        gcs_bucket = os.getenv("GCS_BUCKET")
        gcs_blob_name = f"user_output/{local_filename}"
        gcs_uri       = f"gs://{gcs_bucket}/{gcs_blob_name}"

        try:
            from google.cloud import storage as gcs_storage

            gcs_client = gcs_storage.Client()
            bucket     = gcs_client.bucket(gcs_bucket)
            blob       = bucket.blob(gcs_blob_name)
            blob.upload_from_filename(local_path, content_type="text/csv")
            logger.info("Predictions uploaded to %s", gcs_uri)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to upload predictions to GCS ({gcs_uri}): {exc}"
            ) from exc

        # ── 7. Return result ───────────────────────────────────────────────
        return DatasetPredictionResult(
            mlflow_run_id=cfg.mlflow_run_id,
            model_name=cfg.model_name,
            model_version=cfg.model_version,
            model_type=cfg.model_type,
            output_uri=gcs_uri,
        )

    def inference_on_ray(
        self,
        mlflow_run_id: str,
        ray_config:    RayConfig,
        dataset_uri:   str,
    ) -> DatasetPredictionResult:
        """
        Run distributed inference on a large dataset using a Ray cluster on GKE.

        Follows the same GKE lifecycle as RayTrainer.train():
          1. Create a dedicated Kubernetes namespace.
          2. Spin up a RayCluster inside that namespace.
          3. Resolve the Ray dashboard URL via the head service.
          4. Submit the inference job and block until it completes.
          5. Schedule cluster teardown 30 minutes after the job finishes.
          6. Return a DatasetPredictionResult whose ``output_uri`` points to
             the GCS path written by the Ray inference job.

        Parameters
        ----------
        mlflow_run_id : str
            Identifies the model to load inside the Ray job.
        ray_config : RayConfig
            Controls cluster sizing: ``cpu_per_worker``, ``memory_per_worker``,
            ``num_workers``, and ``use_gpu``.
        dataset_uri : str
            URI of the input dataset, e.g. ``"gs://bucket/path/data.csv"``
            or a SQL connection string.

        Returns
        -------
        DatasetPredictionResult
            ``output_uri`` is the ``gs://`` path where the Ray job wrote
            its prediction CSV (resolved from the job result payload).

        Raises
        ------
        FileNotFoundError
            If the GCP service-account JSON file is missing.
        RuntimeError
            If the Ray job fails, times out, or returns no output URI.
        """
        import threading
        import time

        from app.agents.mta_v2.gcp.get_k8s_api import get_k8s_api
        from app.agents.mta_v2.gcp.namespace import create_namespace
        from app.agents.mta_v2.gcp.create_ray_cluster import create_ray_cluster
        from app.agents.mta_v2.gcp.get_ray_head_svc import get_ray_head_svc
        from app.agents.mta_v2.gcp.get_ray_dashboard_url import get_ray_dashboard_url
        from app.agents.mta_v2.gcp.wait_for_ray_job_completion import wait_for_ray_job_completion
        from app.agents.mta_v2.gcp.submit_ray_job import submit_ray_job
        from app.agents.mta_v2.gcp.get_job_result import get_job_result
        from app.agents.mta_v2.gcp.clean_up import delete_ray_cluster

        # ── 0. Read credentials early so we fail fast ──────────────────────
        if not os.path.exists(_GCP_SA_JSON_PATH):
            raise FileNotFoundError(
                f"GCP service-account JSON not found at '{_GCP_SA_JSON_PATH}'. "
                "Set GCP_SERVICE_ACCOUNT_JSON_PATH to the correct path."
            )
        with open(_GCP_SA_JSON_PATH) as f:
            gcp_sa_json = f.read()

        # ── 1. Derive a short task ID for namespace isolation ──────────────
        task_id   = str(uuid.uuid4())[:8]
        namespace = f"ray-inference-{task_id}"

        # Output filename agreed upon between this caller and the Ray job.
        output_filename = f"predictions_{mlflow_run_id}_{task_id}.csv"
        gcs_bucket      = os.getenv("GCS_BUCKET", "")
        gcs_output_uri  = f"gs://{gcs_bucket}/user_output/{output_filename}"

        cpu_per_worker    = ray_config.get("cpu_per_worker", 4)
        memory_per_worker = ray_config.get("memory_per_worker", "16Gi")
        num_workers       = ray_config.get("num_workers", 4)
        use_gpu           = ray_config.get("use_gpu", False)

        api_client = get_k8s_api(
            _RAY_GKE_CLUSTER_NAME,
            _ray_gcp_project_id(),
            _RAY_GKE_CLUSTER_LOCATION,
        )

        def _schedule_deletion(delay_s: int = 60 * 30) -> None:
            """Delete the Ray cluster in a background thread after delay_s seconds."""
            def _delete():
                time.sleep(delay_s)
                try:
                    delete_ray_cluster(api_client, _RAY_GKE_CLUSTER_NAME, namespace=namespace)
                    logger.info(
                        "Deleted Ray inference cluster '%s' in namespace '%s'.",
                        _RAY_GKE_CLUSTER_NAME, namespace,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to delete Ray inference cluster '%s' in namespace '%s': %s",
                        _RAY_GKE_CLUSTER_NAME, namespace, exc,
                    )
            threading.Thread(target=_delete, daemon=True).start()

        try:
            # ── 2. Namespace ───────────────────────────────────────────────
            create_namespace(api_client, namespace)
            logger.info("Created namespace '%s'.", namespace)

            # ── 3. Ray cluster ─────────────────────────────────────────────
            create_ray_cluster(
                api_client,
                _RAY_GKE_CLUSTER_NAME,
                namespace=namespace,
                docker_image=_ray_inference_docker_uri(),
                worker_replicas=num_workers,
                worker_resources={
                    "requests": {"cpu": str(cpu_per_worker), "memory": memory_per_worker},
                    "limits":   {"cpu": str(cpu_per_worker), "memory": memory_per_worker},
                },
            )
            logger.info(
                "Created Ray cluster '%s' in namespace '%s'.",
                _RAY_GKE_CLUSTER_NAME, namespace,
            )

            # ── 4. Head service & dashboard URL ────────────────────────────
            ray_head_svc  = get_ray_head_svc(api_client, _RAY_GKE_CLUSTER_NAME, namespace=namespace)
            dashboard_url = get_ray_dashboard_url(api_client, ray_head_svc, namespace=namespace)
            logger.info("Ray dashboard URL: '%s'.", dashboard_url)

            # ── 5. Submit inference job ────────────────────────────────────
            job_env = {
                # Credentials & GCP
                "GCP_SERVICE_ACCOUNT_JSON":       gcp_sa_json,
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/gcp_sa.json",
                "GOOGLE_CLOUD_PROJECT":           _ray_gcp_project_id(),

                # Ray worker sizing
                "CPUS_PER_WORKER": str(cpu_per_worker),
                "NUM_WORKERS":     str(num_workers),
                "USE_GPU":         str(use_gpu),

                # MLflow — so the Ray job can pull the model
                "MLFLOW_BACKEND_STORE_URI":     os.getenv("MLFLOW_BACKEND_STORE_URI", ""),
                "MLFLOW_DEFAULT_ARTIFACT_ROOT": os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", ""),
                "MLFLOW_EXPERIMENT_NAME":       os.getenv("MLFLOW_EXPERIMENT_NAME", ""),

                # Job identity
                "TASK_ID":        task_id,
                "MLFLOW_RUN_ID":  mlflow_run_id,

                # Data I/O
                "DATA_SOURCE_URI":   dataset_uri,
                "GCS_BUCKET":        gcs_bucket,
                "OUTPUT_FILENAME":   output_filename,
            }

            job_id = submit_ray_job(dashboard_url, job_env)
            logger.info("Submitted Ray inference job with ID: '%s'.", job_id)

            # ── 6. Wait for completion (6-hour timeout) ────────────────────
            wait_for_ray_job_completion(dashboard_url, job_id, 3600 * 6)
            job_result = get_job_result(dashboard_url, job_id)
            logger.info("Ray inference job finished. Result: %s", job_result)

            # ── 7. Schedule deferred cluster cleanup ───────────────────────
            _schedule_deletion(delay_s=60 * 30)

            # ── 8. Resolve the output URI ──────────────────────────────────
            # The Ray job may return a richer output_uri; fall back to the
            # pre-computed path if it does not.
            if job_result and job_result.get("output_uri"):
                gcs_output_uri = job_result["output_uri"]

            if not gcs_output_uri:
                raise RuntimeError(
                    f"Ray inference job '{job_id}' completed but returned no output_uri."
                )

            # ── 9. Fetch model metadata to populate the result DTO ─────────
            _, cfg = self.load_model(mlflow_run_id)

            return DatasetPredictionResult(
                mlflow_run_id=mlflow_run_id,
                model_name=cfg.model_name,
                model_version=cfg.model_version,
                model_type=cfg.model_type,
                output_uri=gcs_output_uri,
            )

        except Exception as exc:
            logger.error("Ray inference failed: %s", exc, exc_info=True)
            # Tear down immediately on failure
            _schedule_deletion(delay_s=0)
            raise

    # ----------------------------------------------------------
    # Convenience
    # ----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"InferenceInterface("
            f"experiment={self._manager._experiment_name!r}, "
            f"device={self._device!r}, "
            f"cached_models={list(self._model_cache.keys())})"
        )
