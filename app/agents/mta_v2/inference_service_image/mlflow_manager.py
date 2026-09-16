"""
mlflow_manager.py — MLflow Model Manager
==========================================
Provides a clean interface to:
  - list_models()        → summary list of all trained runs in the experiment
  - get_model_details()  → full detail for a single run by mlflow_run_id
  - download_model()     → download .pth + .onnx + config to a local directory
  - load_model()         → return a ready-to-infer DynamicMLP from a run

All operations require only MLFLOW_TRACKING_URI (or MLFLOW_BACKEND_STORE_URI)
to be set; no Ray or GCP dependency.

Usage:
    from mlflow_manager import MLflowManager

    mgr = MLflowManager()                          # reads env vars
    models = mgr.list_models()                     # list summaries
    detail = mgr.get_model_details(mlflow_run_id)  # full detail
    model, cfg = mgr.load_model(mlflow_run_id)     # load for inference
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient

from model import DynamicMLP, ModelConfig
from schema import ModelDetail, ModelSummary

logger = logging.getLogger(__name__)


# =========================================================
# MLflowManager
# =========================================================


class MLflowManager:
    """
    High-level interface for managing trained DynamicMLP runs stored in MLflow.

    Parameters
    ----------
    tracking_uri : str, optional
        MLflow tracking URI. Falls back to MLFLOW_TRACKING_URI →
        MLFLOW_BACKEND_STORE_URI → "mlruns".
    experiment_name : str, optional
        Experiment to query. Falls back to MLFLOW_EXPERIMENT_NAME →
        "avaloka-training".
    """

    # Artifact sub-paths written by ray_job.py
    PTH_ARTIFACT_PATH    = "model_files/model.pth"
    ONNX_ARTIFACT_PATH   = "model_files/model.onnx"
    CONFIG_ARTIFACT_PATH = "metadata/model_config.json"
    ARTIFACT_META_PATH   = "metadata/artifact.json"

    def __init__(
        self,
        tracking_uri:    Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        uri = (
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI", "").strip()
            or os.getenv("MLFLOW_BACKEND_STORE_URI", "").strip()
            or "mlruns"
        )
        mlflow.set_tracking_uri(uri)
        self._client = MlflowClient()
        self._experiment_name = (
            experiment_name
            or os.getenv("MLFLOW_EXPERIMENT_NAME", "avaloka-training").strip()
        )
        self._experiment = self._resolve_experiment()

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _resolve_experiment(self) -> Optional[Any]:
        """
        Look up the configured experiment by name.

        Returns
        -------
        mlflow.entities.Experiment or None
            None is returned (with a warning) when the experiment does not
            yet exist in the tracking store.
        """
        exp = self._client.get_experiment_by_name(self._experiment_name)
        if exp is None:
            logger.warning(
                "MLflow experiment '%s' not found.", self._experiment_name
            )
        return exp

    def _tag(self, run: Any, key: str, default: str = "") -> str:
        """Return a run tag value, or *default* if the key is absent."""
        return run.data.tags.get(key, default)

    def _param(self, run: Any, key: str, default: str = "") -> str:
        """Return a run parameter value, or *default* if the key is absent."""
        return run.data.params.get(key, default)

    def _metric(self, run: Any, key: str) -> Optional[float]:
        """Return a logged metric value, or None if it was never logged."""
        return run.data.metrics.get(key)

    def _download_artifact(
        self, mlflow_run_id: str, artifact_path: str, dst_dir: str
    ) -> Optional[str]:
        """
        Download a single artifact file from MLflow to *dst_dir*.

        Parameters
        ----------
        mlflow_run_id : str
            Run whose artifact store is queried.
        artifact_path : str
            Relative path inside the run's artifact store (e.g.
            ``"model_files/model.pth"``).
        dst_dir : str
            Local directory to download into.

        Returns
        -------
        str or None
            Absolute local path of the downloaded file, or None if the
            artifact could not be found or downloaded.
        """
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=mlflow_run_id,
                artifact_path=artifact_path,
                dst_path=dst_dir,
            )
            return local_path
        except Exception as exc:
            logger.warning(
                "Could not download artifact '%s' for run '%s': %s",
                artifact_path,
                mlflow_run_id,
                exc,
            )
            return None

    def _load_artifact_json(
        self, mlflow_run_id: str, artifact_path: str
    ) -> Optional[dict]:
        """
        Download a JSON artifact and parse it into a dict.

        Parameters
        ----------
        mlflow_run_id : str
        artifact_path : str

        Returns
        -------
        dict or None
            Parsed JSON contents, or None if the artifact is missing or
            malformed.
        """
        with tempfile.TemporaryDirectory() as tmp:
            local = self._download_artifact(mlflow_run_id, artifact_path, tmp)
            if local and Path(local).exists():
                try:
                    with open(local) as f:
                        return json.load(f)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Failed to parse JSON artifact '%s' for run '%s': %s",
                        artifact_path,
                        mlflow_run_id,
                        exc,
                    )
        return None

    def _build_summary(self, run: Any, experiment_name: str) -> ModelSummary:
        """
        Construct a lightweight :class:`ModelSummary` from a raw MLflow run.

        Parameters
        ----------
        run : mlflow.entities.Run
        experiment_name : str

        Returns
        -------
        ModelSummary
        """
        model_type = self._tag(run, "model_type")
        summary = ModelSummary(
            mlflow_run_id=run.info.run_id,
            user_id=self._tag(run, "user_id"),
            session_id=self._tag(run, "session_id"),
            model_name=self._tag(run, "model_name"),
            model_type=model_type,
            model_description=self._tag(run, "model_description"),
            model_version=self._tag(run, "model_version"),
            experiment_name=experiment_name,
            status=run.info.status,
            created_at=self._tag(run, "created_at"),
            final_val_loss=self._metric(run, "final_val_loss"),
        )
        if "regression" in model_type.lower():
            summary.update(
                final_mae=self._metric(run, "final_mae"),
                final_rmse=self._metric(run, "final_rmse"),
                final_r2_score=self._metric(run, "final_r2_score"),
            )
        else:
            summary.update(
                final_accuracy=self._metric(run, "final_accuracy"),
                final_f1_score=self._metric(run, "final_f1_score"),
            )
        return summary

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def list_models(
        self,
        experiment_name: Optional[str] = None,
        filter_string:   str = "",
        max_results:     int = 200,
    ) -> List[ModelSummary]:
        """
        Return a lightweight summary list of all runs in the experiment.

        Parameters
        ----------
        experiment_name : str, optional
            Override the experiment to query.
        filter_string : str
            MLflow search filter, e.g. ``"tags.model_type = 'classification'"``
        max_results : int
            Maximum number of runs to return (MLflow pagination ceiling).

        Returns
        -------
        List[ModelSummary]
            Sorted newest-first by creation time.
        """
        exp_name = experiment_name or self._experiment_name
        experiment = (
            self._client.get_experiment_by_name(exp_name)
            if experiment_name
            else self._experiment
        )
        if experiment is None:
            logger.warning(
                "list_models: experiment '%s' not found — returning empty list.",
                exp_name,
            )
            return []

        runs = self._client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=["start_time DESC"],
        )

        summaries = [self._build_summary(r, exp_name) for r in runs]
        logger.info(
            "list_models: found %d run(s) in experiment '%s'.",
            len(summaries),
            exp_name,
        )
        return summaries

    def get_model_details(self, mlflow_run_id: str) -> Optional[ModelDetail]:
        """
        Fetch full detail for a single run.

        Combines MLflow run metadata with the richer artifact.json /
        model_config.json written by ray_job.py, so all training history,
        class names, and architecture info are available in one object.

        Parameters
        ----------
        mlflow_run_id : str
            The MLflow run ID (info.run_id), as returned by list_models().

        Returns
        -------
        ModelDetail or None if the run is not found.
        """
        try:
            run = self._client.get_run(mlflow_run_id)
        except Exception as exc:
            logger.warning(
                "get_model_details: run '%s' not found: %s", mlflow_run_id, exc
            )
            return None

        exp = self._client.get_experiment(run.info.experiment_id)
        exp_name = exp.name if exp else ""

        # Try to enrich from stored artifact.json
        artifact_meta: dict = (
            self._load_artifact_json(mlflow_run_id, self.ARTIFACT_META_PATH) or {}
        )
        model_info          = artifact_meta.get("model_info", {})
        data_info           = model_info.get("data_info", {})
        class_info          = data_info.get("class_info", {})
        hyper_from_artifact = artifact_meta.get("hyperparameters", {})

        # Try to enrich from model_config.json
        cfg_dict: dict = (
            self._load_artifact_json(mlflow_run_id, self.CONFIG_ARTIFACT_PATH) or {}
        )

        # Merge: cfg_dict > artifact_meta > mlflow params/tags
        params = run.data.params

        def _int(v, default: int = 0) -> int:
            """Safely cast *v* to int, returning *default* on failure."""
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        def _float(v, default: float = 0.0) -> float:
            """Safely cast *v* to float, returning *default* on failure."""
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        def _list_from_str(s: str) -> list:
            """Parse '[128, 64]' or '128,64' style strings into a list."""
            s = s.strip()
            if s.startswith("["):
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    pass
            return [v.strip() for v in s.split(",") if v.strip()]

        feature_names = (
            cfg_dict.get("feature_names")
            or data_info.get("feature_names")
            or _list_from_str(params.get("feature_cols", ""))
        )
        class_names = (
            cfg_dict.get("class_names")
            or class_info.get("class_names", [])
            or _list_from_str(params.get("class_names", ""))
        )
        hidden_sizes = (
            cfg_dict.get("hidden_sizes")
            or hyper_from_artifact.get("hidden_sizes")
            or _list_from_str(params.get("hidden_sizes", "[128,64]"))
        )
        hidden_sizes = [_int(x) for x in hidden_sizes] if hidden_sizes else [128, 64]

        num_features = _int(
            cfg_dict.get("input_size")
            or data_info.get("num_features")
            or params.get("num_features", 0)
        ) or len(feature_names)

        num_classes = _int(
            cfg_dict.get("output_size")
            or data_info.get("num_classes")
            or params.get("num_classes", 1)
        )

        hyperparameters = {
            "learning_rate": _float(
                params.get("learning_rate", hyper_from_artifact.get("learning_rate", 0.001))
            ),
            "epochs": _int(
                params.get("epochs", hyper_from_artifact.get("epochs", 0))
            ),
            "batch_size": _int(
                params.get("batch_size", hyper_from_artifact.get("batch_size", 32))
            ),
            "optimizer": params.get("optimizer", hyper_from_artifact.get("optimizer", "")),
            "hidden_sizes": hidden_sizes,
            "activation": params.get(
                "activation", hyper_from_artifact.get("activation", "relu")
            ),
            "dropout": _float(
                params.get("dropout", hyper_from_artifact.get("dropout", 0.2))
            ),
            "batch_norm": params.get(
                "batch_norm", str(hyper_from_artifact.get("batch_norm", True))
            ).lower() == "true",
            "random_seed": _int(
                params.get("random_seed", hyper_from_artifact.get("random_seed", 42))
            ),
            "early_stopping_patience": _int(
                params.get(
                    "early_stopping_patience",
                    hyper_from_artifact.get("early_stopping_patience", 5),
                )
            ),
            "use_gpu": params.get(
                "use_gpu", str(hyper_from_artifact.get("use_gpu", False))
            ).lower() == "true",
        }

        metrics_raw      = run.data.metrics
        artifact_metrics = artifact_meta.get("metrics", {})

        metrics = {
            "final_train_loss":   metrics_raw.get("final_train_loss",   artifact_metrics.get("final_train_loss",   0.0)),
            "final_val_loss":     metrics_raw.get("final_val_loss",     artifact_metrics.get("final_val_loss",     0.0)),
            "final_accuracy":     metrics_raw.get("final_accuracy",     artifact_metrics.get("final_accuracy",     0.0)),
            "final_f1_score":     metrics_raw.get("final_f1_score",     artifact_metrics.get("final_f1_score",     0.0)),
            "runtime_s":          metrics_raw.get("runtime_s",          artifact_metrics.get("runtime_s",          0.0)),
            "rows_processed":     metrics_raw.get("rows_processed",     artifact_metrics.get("rows_processed",     0)),
            "num_epochs_trained": metrics_raw.get("num_epochs_trained", 0),
        }

        return ModelDetail(
            mlflow_run_id=mlflow_run_id,
            task_id=self._tag(run, "task_id"),
            user_id=self._tag(run, "user_id"),
            session_id=self._tag(run, "session_id"),
            model_name=self._tag(run, "model_name"),
            model_type=self._tag(run, "model_type"),
            model_description=self._tag(run, "model_description"),
            model_version=self._tag(run, "model_version"),
            experiment_name=exp_name,
            experiment_id=run.info.experiment_id,
            status=run.info.status,
            created_at=self._tag(run, "created_at"),
            input_size=num_features,
            output_size=num_classes,
            hidden_sizes=hidden_sizes,
            activation=hyperparameters["activation"],
            dropout=hyperparameters["dropout"],
            batch_norm=hyperparameters["batch_norm"],
            feature_names=feature_names,
            class_names=class_names,
            target_column=data_info.get("target_column", params.get("target_col", "")),
            num_features=num_features,
            num_classes=num_classes,
            num_samples=_int(
                data_info.get("num_samples", artifact_metrics.get("rows_processed", 0))
            ),
            hyperparameters=hyperparameters,
            metrics=metrics,
            model_info=model_info,
            training_history=artifact_meta.get("training_history", []),
            best_epoch=artifact_meta.get("best_epoch", {}),
            artifact_uri=run.info.artifact_uri,
            pth_artifact_path=self.PTH_ARTIFACT_PATH,
            onnx_artifact_path=self.ONNX_ARTIFACT_PATH,
            runtime_s=metrics["runtime_s"],
            rows_processed=int(metrics["rows_processed"]),
            num_epochs_trained=int(metrics["num_epochs_trained"]),
        )

    def download_model(
        self,
        mlflow_run_id: str,
        dst_dir:       Optional[str] = None,
        include_onnx:  bool = True,
    ) -> Dict[str, str]:
        """
        Download model artifacts for a run to a local directory.

        Downloads:
          - model.pth          (PyTorch weights)
          - model.onnx         (ONNX export, optional)
          - model_config.json  (architecture + metadata config)

        Parameters
        ----------
        mlflow_run_id : str
        dst_dir : str, optional
            Destination directory. Defaults to
            ``/tmp/mlflow_models/<mlflow_run_id>``.
        include_onnx : bool
            Whether to also download the ONNX file.

        Returns
        -------
        dict
            Keys ``"pth"``, ``"onnx"`` (empty string if skipped/missing),
            and ``"config"`` — all are absolute local paths.
        """
        dst = Path(dst_dir or f"/tmp/mlflow_models/{mlflow_run_id}")
        dst.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {"pth": "", "onnx": "", "config": ""}

        pth_local = self._download_artifact(mlflow_run_id, self.PTH_ARTIFACT_PATH, str(dst))
        if pth_local:
            paths["pth"] = str(pth_local)
            logger.info("Downloaded .pth → %s", pth_local)

        if include_onnx:
            onnx_local = self._download_artifact(
                mlflow_run_id, self.ONNX_ARTIFACT_PATH, str(dst)
            )
            if onnx_local:
                paths["onnx"] = str(onnx_local)
                logger.info("Downloaded .onnx → %s", onnx_local)

        cfg_local = self._download_artifact(
            mlflow_run_id, self.CONFIG_ARTIFACT_PATH, str(dst)
        )
        if cfg_local:
            paths["config"] = str(cfg_local)
            logger.info("Downloaded model_config.json → %s", cfg_local)

        return paths

    def load_model(
        self,
        mlflow_run_id: str,
        device:        str = "cpu",
        dst_dir:       Optional[str] = None,
    ) -> Tuple[DynamicMLP, ModelConfig]:
        """
        Download artifacts and return a ready-to-infer DynamicMLP + its ModelConfig.

        The model is set to eval() mode before return.

        Parameters
        ----------
        mlflow_run_id : str
        device : str
            Torch device string, e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``.
        dst_dir : str, optional
            Where to cache downloaded files.

        Returns
        -------
        tuple[DynamicMLP, ModelConfig]

        Raises
        ------
        RuntimeError
            If the run cannot be found or the ``.pth`` weights artifact is
            missing.
        """
        detail = self.get_model_details(mlflow_run_id)
        if detail is None:
            raise RuntimeError(
                f"load_model: run '{mlflow_run_id}' not found in MLflow."
            )

        paths = self.download_model(mlflow_run_id, dst_dir=dst_dir, include_onnx=False)

        if not paths["pth"]:
            raise RuntimeError(
                f"load_model: .pth artifact not found for run '{mlflow_run_id}'."
            )

        # Build ModelConfig — prefer stored config file, fall back to detail dict.
        if paths["config"] and Path(paths["config"]).exists():
            cfg = ModelConfig.from_json_file(paths["config"])
        else:
            logger.warning(
                "model_config.json not found for run '%s'; "
                "reconstructing ModelConfig from MLflow detail.",
                mlflow_run_id,
            )
            cfg = ModelConfig(
                input_size=detail["input_size"],
                output_size=detail["output_size"],
                hidden_sizes=detail["hidden_sizes"],
                activation=detail["activation"],
                dropout=detail["dropout"],
                batch_norm=detail["batch_norm"],
                feature_names=detail["feature_names"],
                class_names=detail["class_names"],
                target_column=detail["target_column"],
                model_type=detail["model_type"],
                model_name=detail["model_name"],
                model_description=detail["model_description"],
                model_version=detail["model_version"],
                task_id=detail["task_id"],
                mlflow_run_id=mlflow_run_id,
            )

        model = DynamicMLP.from_config(cfg)
        model.load_weights(paths["pth"], device=device)
        model.to(device)
        model.eval()

        logger.info(
            "Loaded model '%s' v%s from run '%s' on device '%s'.",
            cfg.model_name,
            cfg.model_version,
            mlflow_run_id,
            device,
        )
        return model, cfg

    def delete_model(self, mlflow_run_id: str) -> None:
        """
        Delete a run and its artifacts from MLflow.

        Parameters
        ----------
        mlflow_run_id : str

        Raises
        ------
        RuntimeError
            If deletion fails (e.g. run not found).
        """
        try:
            self._client.delete_run(mlflow_run_id)
            logger.info("Deleted run '%s' and its artifacts from MLflow.", mlflow_run_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to delete run '{mlflow_run_id}': {exc}") from exc

    def get_model_config(self, mlflow_run_id: str) -> Optional[ModelConfig]:
        """
        Fetch the ModelConfig for a run.

        Parameters
        ----------
        mlflow_run_id : str

        Returns
        -------
        ModelConfig or None if not found.
        """
        cfg_dict = self._load_artifact_json(mlflow_run_id, self.CONFIG_ARTIFACT_PATH)
        if cfg_dict:
            try:
                return ModelConfig.from_dict(cfg_dict)
            except Exception as exc:
                logger.warning(
                    "Failed to parse ModelConfig for run '%s': %s",
                    mlflow_run_id,
                    exc,
                )
        return None

    def update_model_config(self, mlflow_run_id: str, new_config: ModelConfig) -> None:
        """
        Update the model_config.json artifact for a run.

        Parameters
        ----------
        mlflow_run_id : str
        new_config : ModelConfig

        Raises
        ------
        RuntimeError
            If the run is not found or the artifact cannot be updated.
        """
        # First, check that the run exists
        try:
            self._client.get_run(mlflow_run_id)
        except Exception as exc:
            raise RuntimeError(f"Run '{mlflow_run_id}' not found: {exc}") from exc

        # Upload the new config as a temporary file
        tmp_dir = tempfile.mkdtemp()
        tmp_path = f"{tmp_dir}/model_config.json"
        with open(tmp_path, "w") as tmp:
            json.dump(new_config.to_dict(), tmp, indent=2)

        try:
            # MLflow doesn't support updating existing artifacts, so we upload a new one with the same path.
            self._client.log_artifact(
                run_id=mlflow_run_id,
                local_path=tmp_path,
                artifact_path="metadata",
            )
            logger.info("Updated model_config.json for run '%s'.", mlflow_run_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to update model_config.json for run '{mlflow_run_id}': {exc}") from exc
        finally:
            os.remove(tmp_path)
