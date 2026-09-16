"""
local_trainer.py — Local PyTorch Training Job
==============================================
Drop-in local equivalent of ray_job.py.

Trains a DynamicMLP on tabular data using a standard PyTorch training loop
(no Ray, no distributed training). Exports .pth and .onnx artifacts and logs
everything (params, per-epoch metrics, model files, metadata) to MLflow —
identical schema to the Ray job so downstream mlflow_manager / inference.py
work without modification.

Key differences from ray_job.py:
  - No Ray / TorchTrainer / distributed setup — single-process, single-GPU or CPU.
  - Dataset loaded eagerly into a pandas DataFrame (no Ray Dataset streaming).
  - Train/val split via the same deterministic MD5 hash used in the Ray job.
  - Imputation, one-hot encoding, and scaling are fitted on training rows only.
  - Class discovery is a plain pandas unique() call.
  - Local and Ray training use the same preprocessing and artifact schemas so
    downstream inference and model-management code can consume either path.

Env vars: identical to ray_job.py — see README.md / ray_job.py docstring.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pytorch
import mlflow.exceptions
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score as sk_f1_score
from sqlalchemy import create_engine, text
from torch.utils.data import DataLoader, TensorDataset

from app.agents.mta_v2.model import DynamicMLP, ModelConfig
from app.agents.mta_v2.loader.sql import load_dataset_from_sql
from app.agents.mta_v2.loader.url import load_dataset_from_url
from app.agents.mta_v2.training_docker_image.src.data_integrity import (
    build_training_integrity_report,
    duplicate_rows_error,
    find_duplicate_training_rows,
    find_identifier_features,
    find_target_leakage,
    identifier_features_error,
    target_leakage_error,
)
from app.agents.mta_v2.training_docker_image.src.evaluation import (
    TIME_ORDER_COLUMN,
    add_time_order_column,
    build_holdout_evaluation_report,
    time_series_split_metadata,
)
from app.agents.pii_agent import (
    apply_deidentification,
    deidentification_contract,
    require_pii_key,
    scan_dataframe,
    transform_pii_value,
)

logger = logging.getLogger(__name__)


def _compute_regression_metrics(labels: List[float], predictions: List[float]) -> Dict[str, float]:
    """Return stable MAE, RMSE, and R-squared values for regression output."""
    if not labels:
        return {"mae": 0.0, "rmse": 0.0, "r2_score": 0.0}

    y_true = np.asarray(labels, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    errors = y_pred - y_true
    absolute_error = float(np.mean(np.abs(errors)))
    squared_error = float(np.mean(np.square(errors)))
    residual_sum_squares = float(np.sum(np.square(errors)))
    total_sum_squares = float(np.sum(np.square(y_true - np.mean(y_true))))

    # Match sklearn's finite behavior for constant targets: a perfect prediction
    # scores 1.0, otherwise R-squared is 0.0 rather than NaN/-Inf.
    if total_sum_squares <= np.finfo(np.float64).eps:
        r2 = 1.0 if residual_sum_squares <= np.finfo(np.float64).eps else 0.0
    else:
        r2 = 1.0 - (residual_sum_squares / total_sum_squares)

    return {
        "mae": round(absolute_error, 6),
        "rmse": round(float(np.sqrt(squared_error)), 6),
        "r2_score": round(float(r2), 6),
    }


# =========================================================
# MLflow Setup  (identical to ray_job.py)
# =========================================================


def setup_mlflow() -> None:
    tracking_uri    = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    backend_store   = os.getenv("MLFLOW_BACKEND_STORE_URI", "").strip()
    artifact_root   = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "").strip()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "avaloka-training").strip()

    effective_uri = tracking_uri or backend_store or "mlruns"
    mlflow.set_tracking_uri(effective_uri)
    logger.info("[MLflow] tracking_uri=%s", effective_uri)

    if artifact_root:
        try:
            mlflow.create_experiment(experiment_name, artifact_location=artifact_root)
            logger.info("[MLflow] Created experiment '%s' with artifact_root=%s", experiment_name, artifact_root)
        except mlflow.exceptions.MlflowException:
            pass  # already exists

    mlflow.set_experiment(experiment_name)
    logger.info("[MLflow] Active experiment: %s", experiment_name)


# =========================================================
# Env Helpers  (identical to ray_job.py)
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


def load_dataframe(uri: str) -> pd.DataFrame:
    """
    Load the dataset into a pandas DataFrame.
    Delegates to the same SQL / URL loaders as ray_job.py, then calls
    to_pandas() if the loader returns a Ray Dataset.
    """
    if uri.startswith(("postgresql://", "mysql://", "sqlite://", "mssql://")):
        ds = load_dataset_from_sql(uri, use_ray=False)
    else:
        ds = load_dataset_from_url(uri, use_ray=False)

    # Keep compatibility with custom/legacy loaders that may still return a
    # Ray-like object even though the built-in local loaders return pandas.
    if hasattr(ds, "to_pandas"):
        return ds.to_pandas()
    # If loader already returns a DataFrame (in a pure-local setup)
    if isinstance(ds, pd.DataFrame):
        return ds
    raise TypeError(f"Unexpected dataset type: {type(ds)}")


# =========================================================
# Feature Column Validation / Preprocessing
# =========================================================


def validate_feature_columns(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> List[str]:
    schema_field_names = set(df.columns.tolist())

    # Pass 1 — existence + target leak check
    existing_cols: List[str] = []
    for col in feature_cols:
        if col == target_col:
            logger.warning("[Feature Validation] Removing '%s': it is the target column.", col)
            continue
        if col not in schema_field_names:
            logger.warning("[Feature Validation] Removing '%s': not found in dataset schema.", col)
            continue
        existing_cols.append(col)

    if not existing_cols:
        raise ValueError(
            f"No valid feature columns after existence check. "
            f"Original: {feature_cols}, Schema: {sorted(schema_field_names)}"
        )

    removed = set(feature_cols) - set(existing_cols)
    if removed:
        logger.warning("[Feature Validation] Removed invalid columns: %s", sorted(removed))
    logger.info("[Feature Validation] Using %d requested feature column(s): %s", len(existing_cols), existing_cols)
    return existing_cols


def _is_categorical_feature(series: pd.Series) -> bool:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return False
    if pd.api.types.is_numeric_dtype(dtype):
        return False
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return False
    return True


def preprocess_feature_dataframe(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit preprocessing metadata and transform a single dataframe.

    Training uses :func:`fit_feature_preprocessing` on the training split and
    :func:`transform_feature_dataframe` on both splits.  This wrapper remains
    useful for callers that intentionally want to fit and transform one frame.
    """
    preprocessing = fit_feature_preprocessing(df, feature_cols)
    return transform_feature_dataframe(df, feature_cols, preprocessing), preprocessing


def _categorical_values(series: pd.Series) -> pd.Series:
    """Normalize categorical values while preserving missingness for imputation."""
    values = series.astype("string").str.strip()
    return values.mask(values.eq(""), pd.NA)


def _numeric_values(series: pd.Series, is_datetime: bool = False) -> pd.Series:
    """Convert a raw feature to finite float64 values, leaving invalids missing."""
    if is_datetime:
        timestamps = pd.to_datetime(series, errors="coerce")
        values = pd.Series(timestamps.astype("int64") / 1_000_000_000, index=series.index)
        values = values.mask(timestamps.isna())
    else:
        values = pd.to_numeric(series, errors="coerce")
    return values.astype(np.float64).replace([np.inf, -np.inf], np.nan)


def fit_feature_preprocessing(
    train_df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """Fit imputation, one-hot vocabularies, and scalers on training rows only."""
    preprocessing: Dict[str, Any] = {
        "version": 3,
        "numeric_features": [],
        "categorical_features": {},
        "feature_scaling": {},
        "imputation": {},
        "model_feature_names": [],
    }

    for feature_index, col in enumerate(feature_cols):
        if col not in train_df.columns:
            continue

        if _is_categorical_feature(train_df[col]):
            values = _categorical_values(train_df[col])
            modes = values.dropna().mode()
            fill_value = str(modes.iloc[0]) if not modes.empty else "__MISSING__"
            fitted_values = values.fillna(fill_value).astype(str)
            categories = sorted(fitted_values.unique().tolist())
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
            preprocessing["model_feature_names"].extend(output_columns + [unknown_column])
            logger.info(
                "[Feature Preprocessing] One-hot encoded categorical feature '%s' with %d categories.",
                col,
                len(categories),
            )
            continue

        is_datetime = pd.api.types.is_datetime64_any_dtype(train_df[col])
        values = _numeric_values(train_df[col], is_datetime=is_datetime)
        fill_value = float(values.mean()) if values.notna().any() else 0.0
        fitted_values = values.fillna(fill_value)
        output_column = f"__mta_feature_{feature_index}_numeric"
        scaler = _standard_scaler_metadata(fitted_values)
        preprocessing["numeric_features"].append(col)
        preprocessing["imputation"][col] = {
            "strategy": "mean",
            "fill_value": fill_value,
        }
        preprocessing["feature_scaling"][col] = {
            **scaler,
            "output_column": output_column,
            "input_type": "datetime" if is_datetime else "numeric",
        }
        preprocessing["model_feature_names"].append(output_column)

    if not preprocessing["model_feature_names"]:
        raise ValueError("No model features remain after preprocessing.")
    return preprocessing


def transform_feature_dataframe(
    df: pd.DataFrame,
    feature_cols: List[str],
    preprocessing: Dict[str, Any],
) -> pd.DataFrame:
    """Apply a previously fitted preprocessing contract without refitting it."""
    transformed = df.copy()
    categorical = preprocessing.get("categorical_features") or {}
    imputation = preprocessing.get("imputation") or {}
    feature_scaling = preprocessing.get("feature_scaling") or {}

    for col in feature_cols:
        if col not in transformed.columns:
            raise ValueError(f"Required feature {col!r} is missing during preprocessing.")

        if col in categorical and categorical[col].get("encoding") == "one_hot":
            meta = categorical[col]
            fill_value = str((imputation.get(col) or {}).get("fill_value", "__MISSING__"))
            values = _categorical_values(transformed[col]).fillna(fill_value).astype(str)
            categories = [str(value) for value in meta.get("categories", [])]
            output_columns = list(meta.get("output_columns", []))
            for category, output_column in zip(categories, output_columns):
                transformed[output_column] = values.eq(category).astype(np.float32)
            known = values.isin(categories)
            transformed[str(meta["unknown_column"])] = (~known).astype(np.float32)
            continue

        scaler = feature_scaling.get(col) or {}
        fill_value = float((imputation.get(col) or {}).get("fill_value", scaler.get("mean", 0.0)))
        values = _numeric_values(
            transformed[col],
            is_datetime=scaler.get("input_type") == "datetime",
        ).fillna(fill_value)
        mean = float(scaler.get("mean", 0.0))
        scale = float(scaler.get("scale", 1.0))
        if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
            scale = 1.0
        output_column = str(scaler.get("output_column", col))
        transformed[output_column] = ((values - mean) / scale).astype(np.float32)

    return transformed


def _standard_scaler_metadata(series: pd.Series) -> Dict[str, float | str]:
    """Fit a numerically safe standard scaler for one training column."""
    values = pd.to_numeric(series, errors="coerce").astype(np.float64)
    mean = float(values.mean()) if len(values) else 0.0
    scale = float(values.std(ddof=0)) if len(values) else 1.0
    if not np.isfinite(mean):
        mean = 0.0
    if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
        scale = 1.0
    return {"method": "standard", "mean": mean, "scale": scale}


def apply_standard_scaling(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    model_type: str,
    preprocessing: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Fit scalers on the training split and apply them to train/validation data.

    Numeric inputs are standardized for every task. Regression targets are also
    standardized so MSE optimization is not dominated by the target's unit
    scale. The fitted statistics are persisted for inference and target
    inverse-transformation.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    preprocessing = dict(preprocessing or {})
    preprocessing["version"] = max(2, int(preprocessing.get("version", 1)))

    numeric_features = set(preprocessing.get("numeric_features") or [])
    feature_scaling: Dict[str, Dict[str, float | str]] = dict(
        preprocessing.get("feature_scaling") or {}
    )
    for col in feature_cols:
        if col not in numeric_features:
            continue
        scaler = _standard_scaler_metadata(train_df[col])
        mean = float(scaler["mean"])
        scale = float(scaler["scale"])
        train_df[col] = ((train_df[col].astype(np.float64) - mean) / scale).astype(np.float32)
        val_df[col] = ((val_df[col].astype(np.float64) - mean) / scale).astype(np.float32)
        feature_scaling[col] = scaler

    preprocessing["feature_scaling"] = feature_scaling

    if "regression" in model_type.lower():
        target_scaling = _standard_scaler_metadata(train_df["__label__"])
        target_mean = float(target_scaling["mean"])
        target_scale = float(target_scaling["scale"])
        train_df["__label__"] = (
            (train_df["__label__"].astype(np.float64) - target_mean) / target_scale
        ).astype(np.float32)
        val_df["__label__"] = (
            (val_df["__label__"].astype(np.float64) - target_mean) / target_scale
        ).astype(np.float32)
        preprocessing["target_scaling"] = target_scaling

    return train_df, val_df, preprocessing


def inverse_standard_scaling(values: List[float], scaler: Dict[str, Any] | None) -> List[float]:
    """Convert standardized regression values back to their original units."""
    if not scaler or scaler.get("method") != "standard":
        return [float(value) for value in values]
    mean = float(scaler.get("mean", 0.0))
    scale = float(scaler.get("scale", 1.0))
    return [float(value) * scale + mean for value in values]


# =========================================================
# Imputation  (same mean/mode logic as ray_job.py)
# =========================================================


def impute_dataframe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        if df[col].dtype.kind == "f" or pd.api.types.is_float_dtype(df[col]):
            fill = df[col].mean() if df[col].notna().any() else 0.0
            df[col] = df[col].fillna(fill)
        elif df[col].dtype.kind in ("U", "S", "O"):
            mode_vals = df[col].mode()
            fill = mode_vals.iloc[0] if len(mode_vals) > 0 else ""
            df[col] = df[col].fillna(fill).replace("", fill)
    return df


def drop_missing_target_rows(
    df: pd.DataFrame,
    target_col: str,
    model_type: str,
) -> pd.DataFrame:
    df = df.copy()
    initial_rows = len(df)
    normalized_model_type = model_type.lower()

    if "classification" in normalized_model_type:
        valid_mask = df[target_col].notna()
        if (
            pd.api.types.is_object_dtype(df[target_col])
            or pd.api.types.is_string_dtype(df[target_col])
            or isinstance(df[target_col].dtype, pd.CategoricalDtype)
        ):
            non_blank = df[target_col].astype(str).str.strip().ne("")
            valid_mask = valid_mask & non_blank
    else:
        target_values = (
            pd.to_numeric(df[target_col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        valid_mask = target_values.notna()
        df[target_col] = target_values

    cleaned = df.loc[valid_mask].copy()
    dropped_rows = initial_rows - len(cleaned)
    if dropped_rows:
        logger.warning(
            "[Data] Dropped %d row(s) with missing target values in '%s'.",
            dropped_rows,
            target_col,
        )

    if cleaned.empty:
        raise ValueError(
            f"No training rows remain after removing missing target values from {target_col!r}."
        )

    return cleaned


def prepare_training_dataframe(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_type: str,
) -> pd.DataFrame:
    df = df[feature_cols + [target_col]].copy()
    # Feature imputation must be fitted after the split.  Doing it here would
    # let validation values influence the statistics used for training.
    return drop_missing_target_rows(df, target_col, model_type)


# =========================================================
# Class discovery  (same sort order as ray_job.py)
# =========================================================


def discover_class_names(df: pd.DataFrame, target_col: str) -> List[str]:
    return sorted(df[target_col].dropna().astype(str).unique().tolist())


# =========================================================
# Train/val split  (same deterministic MD5 hash as ray_job.py)
# =========================================================


def split_train_val(
    df: pd.DataFrame,
    *,
    time_ordered: bool = False,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split model rows consistently with Ray training.

    Time-ordered frames use the final expanding-window TimeSeriesSplit fold.
    Other frames use the existing deterministic row-hash holdout.
    """
    if time_ordered:
        metadata = time_series_split_metadata(len(df), "time", n_splits=n_splits)
        train_rows = metadata["n_train_rows"]
        return (
            df.iloc[:train_rows].reset_index(drop=True),
            df.iloc[train_rows:].reset_index(drop=True),
        )

    non_meta_cols = [c for c in df.columns if not c.startswith("__")]

    def _is_train(row: pd.Series) -> bool:
        raw = b"".join(str(row[c]).encode() for c in non_meta_cols)
        return int(hashlib.md5(raw).hexdigest(), 16) % 10 < 8

    mask = df.apply(_is_train, axis=1)
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)


# =========================================================
# Model Export helpers  (thin wrappers — same as ray_job.py)
# =========================================================


def save_model_pth(model: DynamicMLP, path: str) -> None:
    model.save_pth(path)
    logger.info("  [Export] .pth -> %s", path)


def save_model_onnx(model: DynamicMLP, path: str) -> None:
    model.save_onnx(path)
    logger.info("  [Export] .onnx -> %s", path)


def balanced_class_weights(labels: pd.Series, num_classes: int) -> torch.Tensor:
    """Return inverse-frequency weights fitted from training labels only."""
    counts = np.bincount(labels.to_numpy(dtype=np.int64), minlength=num_classes)
    observed = counts > 0
    weights = np.ones(num_classes, dtype=np.float32)
    if observed.any():
        observed_total = float(counts[observed].sum())
        observed_classes = int(observed.sum())
        weights[observed] = observed_total / (observed_classes * counts[observed])
    return torch.tensor(weights, dtype=torch.float32)


# =========================================================
# Training Loop  (local single-process PyTorch)
# =========================================================


def train_local(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Dict[str, Any],
) -> tuple[DynamicMLP, List[Dict], Dict]:
    """
    Single-process training loop.

    Mirrors train_loop_per_worker from ray_job.py:
      - Same optimizer choices, loss functions, gradient clipping.
      - Task-appropriate metrics (accuracy/F1 for classification;
        MAE/RMSE/R-squared for regression).
      - Best model checkpoint saved at lowest val_loss.
      - Early stopping via patience counter.

    Returns (best_model, history, final_metrics).
    """
    torch.manual_seed(config["random_seed"])
    device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")
    logger.info("[Train] Device: %s", device)

    feature_cols: List[str] = config["feature_cols"]
    model_type: str         = config["model_type"]
    target_scaling: Dict[str, Any] = config.get("target_scaling") or {}

    # ---- Build tensors ----
    X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
    X_val   = torch.tensor(val_df[feature_cols].values,   dtype=torch.float32)

    if "classification" in model_type:
        y_train = torch.tensor(train_df["__label__"].values, dtype=torch.long)
        y_val   = torch.tensor(val_df["__label__"].values,   dtype=torch.long)
    else:
        y_train = torch.tensor(train_df["__label__"].values, dtype=torch.float32).unsqueeze(1)
        y_val   = torch.tensor(val_df["__label__"].values,   dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    # ---- Model ----
    model = DynamicMLP(
        input_size=len(feature_cols),
        output_size=config["output_dim"],
        hidden_sizes=config["hidden_sizes"],
        activation=config["activation"],
        dropout=config["dropout"],
        batch_norm=config["batch_norm"],
    ).to(device)
    logger.info("[Train] Model architecture:\n%s", model)

    # ---- Optimizer ----
    opt_map = {
        "adam":    optim.Adam,
        "sgd":     optim.SGD,
        "adamw":   optim.AdamW,
        "rmsprop": optim.RMSprop,
    }
    optimizer = opt_map.get(config["optimizer"], optim.Adam)(
        model.parameters(), lr=config["learning_rate"]
    )
    if "classification" in model_type:
        configured_weights = config.get("class_weights")
        class_weights = (
            torch.tensor(configured_weights, dtype=torch.float32)
            if configured_weights is not None
            else balanced_class_weights(train_df["__label__"], config["output_dim"])
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("[Train] Class weights from training split: %s", class_weights.cpu().tolist())
    else:
        criterion = nn.MSELoss()

    best_val_loss    = float("inf")
    patience_counter = 0
    patience: int    = max(1, int(config["early_stopping_patience"]))
    min_delta: float = max(0.0, float(config.get("early_stopping_min_delta", 1e-4)))
    history: List[Dict] = []
    best_state_dict  = None

    for epoch in range(config["epochs"]):
        logger.info("[Train] Epoch %d/%d ...", epoch + 1, config["epochs"])

        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        all_train_preds:  List[float] = []
        all_train_labels: List[float] = []

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            out  = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            if "classification" in model_type:
                all_train_preds.extend(out.argmax(1).cpu().tolist())
                all_train_labels.extend(y_batch.cpu().tolist())
            else:
                all_train_preds.extend(out.detach().cpu().reshape(-1).tolist())
                all_train_labels.extend(y_batch.detach().cpu().reshape(-1).tolist())

        avg_train_loss = train_loss / max(len(train_loader), 1)
        if "classification" in model_type and all_train_labels:
            train_acc = sum(p == l for p, l in zip(all_train_preds, all_train_labels)) / len(all_train_labels)
            train_f1  = float(sk_f1_score(all_train_labels, all_train_preds, average="weighted", zero_division=0))
        else:
            train_acc, train_f1 = None, None
            train_regression_metrics = _compute_regression_metrics(
                inverse_standard_scaling(all_train_labels, target_scaling),
                inverse_standard_scaling(all_train_preds, target_scaling),
            )

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        all_val_preds:  List[float] = []
        all_val_labels: List[float] = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out      = model(x_batch)
                val_loss += criterion(out, y_batch).item()
                if "classification" in model_type:
                    all_val_preds.extend(out.argmax(1).cpu().tolist())
                    all_val_labels.extend(y_batch.cpu().tolist())
                else:
                    all_val_preds.extend(out.cpu().reshape(-1).tolist())
                    all_val_labels.extend(y_batch.cpu().reshape(-1).tolist())

        avg_val_loss = val_loss / max(len(val_loader), 1)
        if "classification" in model_type and all_val_labels:
            val_acc = sum(p == l for p, l in zip(all_val_preds, all_val_labels)) / len(all_val_labels)
            val_f1  = float(sk_f1_score(all_val_labels, all_val_preds, average="weighted", zero_division=0))
        else:
            val_acc, val_f1 = None, None
            val_regression_metrics = _compute_regression_metrics(
                inverse_standard_scaling(all_val_labels, target_scaling),
                inverse_standard_scaling(all_val_preds, target_scaling),
            )

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
            logger.info(
                "[Train] [Epoch %d/%d] train_loss=%.4f  val_loss=%.4f  acc=%.4f  f1=%.4f",
                epoch + 1, config["epochs"], avg_train_loss, avg_val_loss, train_acc, train_f1,
            )
        else:
            logger.info(
                "[Train] [Epoch %d/%d] scaled_train_mse=%.4f  "
                "scaled_val_mse=%.4f  val_mae=%.4f  val_rmse=%.4f  val_r2=%.4f",
                epoch + 1,
                config["epochs"],
                avg_train_loss,
                avg_val_loss,
                val_regression_metrics["mae"],
                val_regression_metrics["rmse"],
                val_regression_metrics["r2_score"],
            )

        # ---- Checkpoint best ----
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            best_state_dict  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info("[EarlyStopping] Triggered at epoch %d", epoch + 1)
            break

    # Restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()

    # The restored weights are from the lowest validation-loss checkpoint, so
    # report the metrics from that same epoch instead of a later degraded epoch.
    final_metrics = min(history, key=lambda row: row["val_loss"]) if history else {}
    return model, history, final_metrics


# =========================================================
# TRAINING ENTRY  (mirrors training_main in ray_job.py)
# =========================================================


def training_main(
    task_id: str = str(uuid.uuid4()),
    user_id: str = "unknown_user",
    session_id: str = str(uuid.uuid4()),
    model_type: str = "classification",
    data_uri: str = "",
    target_col: str = "",
    time_column: str = "",
    class_names: list[str] = None,
    feature_cols: list[str] = [],
    model_name: str = "dynamic-mlp",
    model_description: str = "",
    model_version: str = "1",
    learning_rate: float = 0.001,
    epochs: int = 30,
    batch_size: int = 32,
    optimizer_name: str = "adam",
    random_seed: int = 42,
    early_stopping_pat: int = 5,
    early_stopping_min_delta: float = 1e-4,
    use_gpu: bool = False,
    hidden_sizes: list[int] = None,
    activation: str = "relu",
    dropout: float = 0.2,
    batch_norm: bool = True,
) -> dict[str, Any]:
    start_time = time.time()

    hidden_sizes = hidden_sizes or [128, 64]
    class_names  = class_names  or []

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    logger.info("[Config] features=%s  target=%s", feature_cols, target_col)
    logger.info("[Config] epochs=%d  lr=%s  hidden=%s", epochs, learning_rate, hidden_sizes)

    # ---- Load dataset ----
    logger.info("[Data] Loading from %s", data_uri)
    df = load_dataframe(data_uri)
    logger.info("[Data] Loaded %d rows, %d columns", len(df), len(df.columns))

    time_column = str(time_column or "").strip()
    if time_column:
        if time_column == target_col:
            raise ValueError("The time column must be different from the target column.")
        # The ordering column is metadata. Feeding raw timestamps into the MLP
        # as categorical values would create one category per instant.
        feature_cols = [column for column in feature_cols if column != time_column]
        df = add_time_order_column(df, time_column)
        df = (
            df.sort_values(TIME_ORDER_COLUMN, kind="mergesort")
            .drop(columns=[TIME_ORDER_COLUMN])
            .reset_index(drop=True)
        )
        logger.info("[Data] Rows sorted by time column %s", time_column)

    if target_col not in df.columns:
        preview = ", ".join(map(str, list(df.columns[:30])))
        more = "..." if len(df.columns) > 30 else ""
        raise ValueError(
            f"Target column {target_col!r} is not present in the training dataset. "
            f"Available columns include: {preview}{more}. "
            "Use a labeled training dataset that contains the target column, or update the training plan target."
        )

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(
            "Feature columns are not present in the training dataset: "
            f"{', '.join(missing_features)}. Update the training plan feature list before training."
        )

    # PII protection is a final execution-time guard, not merely a planner
    # suggestion. Only key-backed strategies (pseudonymisation/tokenisation)
    # require AVALOKA_PII_KEY; DOB/postcode generalisation does not.
    pii_columns = list(dict.fromkeys(feature_cols + [target_col]))
    pii_report = scan_dataframe(df, columns=pii_columns)
    pii_key_columns = require_pii_key(pii_report)
    pii_transformations = deidentification_contract(pii_report)

    integrity_report = build_training_integrity_report(df, target_col, feature_cols)
    duplicate_match = find_duplicate_training_rows(df, target_col, feature_cols)
    if duplicate_match:
        raise ValueError(duplicate_rows_error(duplicate_match))

    identifier_matches = find_identifier_features(df, feature_cols)
    if identifier_matches:
        raise ValueError(identifier_features_error(identifier_matches))

    leakage_matches = find_target_leakage(df, target_col, feature_cols)
    if leakage_matches:
        raise ValueError(target_leakage_error(target_col, leakage_matches))

    # ---- Validate + prepare feature columns ----
    feature_cols = validate_feature_columns(df, feature_cols, target_col)

    # Replace sensitive values before they enter preprocessing, class discovery,
    # model artifacts, or MLflow.  The key itself is never persisted; only the
    # safe transformation contract is saved so inference can repeat the exact
    # same deterministic operation on raw input values.
    if pii_transformations:
        df = apply_deidentification(df, pii_report)
        target_pii = pii_transformations.get(target_col)
        if target_pii and class_names and "classification" in model_type:
            class_names = [
                str(transform_pii_value(
                    value,
                    column=target_col,
                    kind=target_pii["kind"],
                    strategy=target_pii["strategy"],
                ))
                for value in class_names
            ]
    integrity_report["pii"] = pii_report.as_dict()
    integrity_report["pii_key_required_columns"] = pii_key_columns

    # Keep only required columns and remove unlabeled rows. Feature statistics
    # are deliberately not fitted until after the train/validation split.
    df = prepare_training_dataframe(df, feature_cols, target_col, model_type)

    # ---- Class detection / label encoding ----
    if "classification" in model_type:
        if not class_names:
            logger.info("[Data] Discovering class names ...")
            class_names = discover_class_names(df, target_col)
            logger.info("[Data] Found %d classes: %s", len(class_names), class_names)

        le_classes = class_names[:]
        df["__label__"] = df[target_col].astype(str).apply(
            lambda v: le_classes.index(v)
        ).astype(np.int64)
    else:
        df["__label__"] = df[target_col].astype(np.float32)

    num_samples = len(df)

    # ---- Train/val split ----
    if time_column:
        validation_split = time_series_split_metadata(num_samples, time_column)
        train_df, val_df = split_train_val(
            df,
            time_ordered=True,
            n_splits=validation_split["n_folds"],
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
        train_df, val_df = split_train_val(df)
    train_targets_for_evaluation = train_df[target_col].tolist()
    val_targets_for_evaluation = val_df[target_col].tolist()

    # Fit every feature transform on training rows only, then reuse the exact
    # same contract for validation and inference.
    preprocessing = fit_feature_preprocessing(train_df, feature_cols)
    preprocessing["pii_transformations"] = pii_transformations
    preprocessing["validation"] = dict(validation_split)
    if "classification" in model_type:
        preprocessing["class_weights"] = balanced_class_weights(
            train_df["__label__"],
            len(class_names),
        ).tolist()
    train_df = transform_feature_dataframe(train_df, feature_cols, preprocessing)
    val_df = transform_feature_dataframe(val_df, feature_cols, preprocessing)
    model_feature_cols = list(preprocessing["model_feature_names"])

    # Feature scaling was already applied by transform_feature_dataframe.
    # This call only fits/applies regression target scaling.
    train_df, val_df, preprocessing = apply_standard_scaling(
        train_df,
        val_df,
        [],
        model_type,
        preprocessing,
    )
    logger.info("[Data] train=%d  val=%d", len(train_df), len(val_df))

    # ---- Build train config ----
    train_config: Dict[str, Any] = {
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
    }

    # ======================================================
    # MLflow Run  (identical structure to ray_job.py)
    # ======================================================
    logger.info("[MLflow] Starting run ...")
    with mlflow.start_run(run_name=task_id) as mlflow_run:
        mlflow_run_id = mlflow_run.info.run_id
        logger.info("[MLflow] run_id=%s", mlflow_run_id)

        # ---- Log hyperparameters (identical to ray_job.py) ----
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
            # num_workers / cpus_per_worker are Ray concepts — set to 1 for local
            "num_workers":             1,
            "cpus_per_worker":         1,
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
        logger.info("[Train] Starting with total_rows=%d", num_samples)
        model, history, final_metrics_raw = train_local(train_df, val_df, train_config)
        evaluation_report = build_holdout_evaluation_report(
            train_targets_for_evaluation,
            val_targets_for_evaluation,
            model_type,
            final_metrics_raw,
            splitter=validation_split["splitter"],
            splitter_reason=validation_split["splitter_reason"],
            n_folds=validation_split["n_folds"],
            time_column=validation_split["time_column"],
        )

        # ---- Log per-epoch metrics (identical to ray_job.py) ----
        for epoch_row in history:
            epoch_num = epoch_row.get("epoch", 0)
            mlflow.log_metrics(
                {key: value for key, value in epoch_row.items() if key != "epoch"},
                step=epoch_num,
            )

        # ---- Build training history output (identical to ray_job.py) ----
        training_history_out = [dict(row) for row in history]

        best_epoch = min(training_history_out, key=lambda r: r["val_loss"]) if training_history_out else {}
        runtime_s  = round(time.time() - start_time, 3)

        if "regression" in model_type:
            final_task_metrics = {
                "final_train_mae":      final_metrics_raw.get("train_mae", 0.0),
                "final_mae":            final_metrics_raw.get("val_mae", 0.0),
                "final_train_rmse":     final_metrics_raw.get("train_rmse", 0.0),
                "final_rmse":           final_metrics_raw.get("val_rmse", 0.0),
                "final_train_r2_score": final_metrics_raw.get("train_r2_score", 0.0),
                "final_r2_score":       final_metrics_raw.get("val_r2_score", 0.0),
            }
        else:
            final_task_metrics = {
                "final_accuracy": round(final_metrics_raw.get("accuracy", 0.0), 6),
                "final_f1_score": round(final_metrics_raw.get("f1_score", 0.0), 6),
            }

        # ---- Log final summary metrics (identical to ray_job.py) ----
        mlflow.log_metrics({
            "final_train_loss":   round(final_metrics_raw.get("train_loss", 0.0), 6),
            "final_val_loss":     round(final_metrics_raw.get("val_loss", 0.0), 6),
            "runtime_s":          runtime_s,
            "rows_processed":     float(num_samples),
            "num_epochs_trained": float(len(training_history_out)),
            **final_task_metrics,
        })

        # ---- Export model (identical to ray_job.py) ----
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
            inference_service_details=None,
            preprocessing=preprocessing,
        )

        temp_dir = f"/tmp/{uuid.uuid4()}"
        os.makedirs(temp_dir, exist_ok=True)

        local_pth_named  = f"{temp_dir}/model.pth"
        local_onnx_named = f"{temp_dir}/model.onnx"

        save_model_pth(model, local_pth_named)
        save_model_onnx(model, local_onnx_named)

        # ---- Log model artifacts to MLflow (identical to ray_job.py) ----
        logger.info("[MLflow] Logging model artifacts ...")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=None,
            serialization_format="pickle"
        )
        mlflow.log_artifact(local_pth_named,  artifact_path="model_files")
        mlflow.log_artifact(local_onnx_named, artifact_path="model_files")

        local_model_config = f"{temp_dir}/model_config.json"
        with open(local_model_config, "w") as f:
            f.write(model_cfg.to_json())
        mlflow.log_artifact(local_model_config, artifact_path="metadata")
        logger.info("[MLflow] Logged model_config.json -> metadata/model_config.json")

        # ---- Artifact metadata JSON (identical schema to ray_job.py) ----
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
                "final_train_loss": round(final_metrics_raw.get("train_loss", 0.0), 6),
                "final_val_loss":   round(final_metrics_raw.get("val_loss", 0.0), 6),
                **final_task_metrics,
            },
            "integrity_report": integrity_report,
            "evaluation_report": evaluation_report,
            "output_columns":   feature_cols + [target_col],
            "preprocessing":    preprocessing,
            "output_rows":      [],
            "output_row_count": num_samples,
            "created_at":       datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id":    mlflow_run_id,
        }

        output_metrics = {
            "mlflow_run_id":      mlflow_run_id,
            "dashboard_url":      None,
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
            "final_train_loss":   round(final_metrics_raw.get("train_loss", 0.0), 6),
            "final_val_loss":     round(final_metrics_raw.get("val_loss", 0.0), 6),
            "num_features":       len(feature_cols),
            "num_classes":        len(class_names) if class_names else 1,
            "created_at":         datetime.now(timezone.utc).isoformat(),
            "integrity_report":   integrity_report,
            "evaluation_report":  evaluation_report,
            **final_task_metrics,
        }

        local_artifact_json = f"{temp_dir}/artifact.json"
        local_metrics_json  = f"{temp_dir}/metrics.json"

        with open(local_artifact_json, "w") as f:
            json.dump(artifact_meta, f, indent=2)
        with open(local_metrics_json, "w") as f:
            json.dump(output_metrics, f, indent=2)

        mlflow.log_artifact(local_artifact_json, artifact_path="metadata")
        mlflow.log_artifact(local_metrics_json,  artifact_path="metadata")

        # ---- Set MLflow tags (identical to ray_job.py) ----
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

    logger.info(
        "[Done] task_id=%s  mlflow_run_id=%s  model=%s  version=%s  runtime=%.3fs",
        task_id, mlflow_run_id, model_name, model_version, runtime_s,
    )
    return output_metrics



class LocalTrainer:
    """
    In-process trainer that mirrors the ``RayTrainer.train()`` interface.

    ``training_plan`` is a ``TrainingPlan`` dataclass (or dict-like) with
    the same fields produced by ``ModelTrainingAgent._extract_training_plan``.

    Returns the same ``output_metrics`` dict that ``training_main`` returns,
    which is also what ``RayTrainer.train()`` returns — so the rest of the
    agent pipeline (``_finish_mta``, inference, MLflow lookup) works unchanged.
    """

    def __init__(self, user_id: str = "unknown_user", session_id: str = "unknown_session") -> None:
        self.user_id    = user_id
        self.session_id = session_id

    def train(self, training_plan: Any) -> Dict[str, Any]:
        """
        Execute local training from a ``TrainingPlan`` and return metrics.

        Accepts both dataclass instances (with attribute access) and plain
        dicts (with key access) so it is robust to different callers.

        Parameters
        ----------
        training_plan : TrainingPlan | dict
            The plan assembled by ``ModelTrainingAgent._extract_training_plan``.

        Returns
        -------
        dict
            ``output_metrics`` from ``training_main`` — same schema as
            ``RayTrainer.train()``:
            {
              "mlflow_run_id", "task_id", "user_id", "session_id",
              "model_type", "model_name", "model_description", "model_version",
              "runtime_s", "rows_processed", "num_epochs_trained",
              "final_train_loss", "final_val_loss", plus task-appropriate
              classification or regression metrics, "num_features",
              "num_classes", "created_at"
            }
        """
        # ---- Resolve TrainingPlan fields (dataclass or dict) ----
        def _get(obj: Any, key: str, default: Any = None) -> Any:
            if obj is None:
                return default
            if hasattr(obj, key):
                return getattr(obj, key)
            if isinstance(obj, dict):
                return obj.get(key, default)
            return default

        def _get_nested(obj: Any, *keys: str, default: Any = None) -> Any:
            """Walk a chain of attribute/key lookups."""
            cur = obj
            for k in keys:
                cur = _get(cur, k)
                if cur is None:
                    return default
            return cur if cur is not None else default

        hp  = _get(training_plan, "hyperparameter_config")
        rc  = _get(training_plan, "ray_config")
        dc  = _get(training_plan, "data_config")

        # Resolve data URI: prefer cloud, fall back to local / generic
        data_uri = (
            _get(dc, "data_source_location_cloud")
            or _get(dc, "data_source_location_local")
            or _get(dc, "data_source_location")
            or _get(dc, "dataset_uri")
            or ""
        )

        task_id = str(uuid.uuid4())

        logger.info(
            "[LocalTrainer] Starting local training — task_id=%s  user_id=%s  data_uri=%s",
            task_id, self.user_id, data_uri,
        )

        # ---- MLflow setup (idempotent) ----
        setup_mlflow()

        try:
            result = training_main(
                task_id            = task_id,
                user_id            = self.user_id,
                session_id         = self.session_id,
                model_type         = _get(training_plan, "model_type", "classification"),
                data_uri           = data_uri,
                target_col         = _get(dc, "target_column") or _get(dc, "target_col") or "",
                time_column       = _get(dc, "time_column") or "",
                class_names        = _get(dc, "class_names") or [],
                feature_cols       = _get(dc, "feature_columns") or _get(dc, "feature_cols") or [],
                model_name         = _get(training_plan, "model_name", "dynamic-mlp"),
                model_description  = _get(training_plan, "model_description", ""),
                model_version      = _get(training_plan, "model_version", "1"),
                learning_rate      = _get(hp, "learning_rate", 0.001),
                epochs             = _get(hp, "epochs", 30),
                batch_size         = _get(hp, "batch_size", 32),
                optimizer_name     = _get(hp, "optimizer_name", "adam"),
                random_seed        = _get(hp, "random_seed", 42),
                early_stopping_pat = _get(hp, "early_stopping_patience", 5),
                early_stopping_min_delta = _get(hp, "early_stopping_min_delta", 1e-4),
                use_gpu            = _get(rc, "use_gpu", False),
                hidden_sizes       = _get(hp, "hidden_layer_sizes") or _get(hp, "hidden_sizes") or [128, 64],
                activation         = _get(hp, "activation", "relu"),
                dropout            = _get(hp, "dropout_rate") or _get(hp, "dropout") or 0.2,
                batch_norm         = _get(hp, "batch_norm", True),
            )
            logger.info("[LocalTrainer] Training finished — mlflow_run_id=%s", result.get("mlflow_run_id"))
            return result

        except Exception as exc:
            logger.error("[LocalTrainer] Training failed: %s", exc, exc_info=True)
            return {
                "status": "error",
                "error": str(exc),
                "model_name": _get(training_plan, "model_name", "dynamic-mlp"),
                "model_type": _get(training_plan, "model_type", "classification"),
                "model_version": _get(training_plan, "model_version", "1"),
            }
