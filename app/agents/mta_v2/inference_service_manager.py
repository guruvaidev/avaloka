import os
import uuid
import shutil
import logging
import math
import tempfile
from pathlib import Path
import requests
from typing import Dict, Any, List, Optional

from app.agents.mta_v2.inference_autoscale import (RAY_CLUSTER_NAME,
                                                   RayServiceScaler,
                                                   mark_activity,
                                                   policy_from_env)
from app.agents.mta_v2.mlflow_manager import MLflowManager
from app.core import cloud_config

logger = logging.getLogger(__name__)

# Resolved lazily, not at import: an unset project must fail where a deployment
# is actually attempted, naming the variable — not stop the API from importing.
def inference_service_docker_image() -> str:
    return cloud_config.image(
        "INFERENCE_SERVICE_DOCKER_IMAGE", "inference-service-image", "test-v0.0.9")


def inference_service_project_id() -> str:
    return cloud_config.project_id()
INFERENCE_SERVICE_GKE_CLUSTER_LOCATION = os.getenv("INFERENCE_SERVICE_GKE_CLUSTER_LOCATION", "us-central1")
GCP_SA_JSON_PATH = os.getenv("GCP_SERVICE_ACCOUNT_JSON_PATH", "~/Templates/gcp-bucket-cred.json")

# Backend strategy (D3): "rayserve" drives the in-cluster RayService on the one
# shared cluster (RAY_CLUSTER_NAME) and is the only supported path.
#
# "gateway" is the legacy per-run backend: it minted an inference-service-{uuid}
# namespace and a GCP API Gateway for every deployment and reclaimed neither.
# Eighteen such namespaces accumulated between April and
# August 2026, each holding a pod and a LoadBalancer, none serving traffic. It is
# retained only for recovering those deployments and now requires an explicit
# opt-in via AVALOKA_ALLOW_LEGACY_GATEWAY=1.
# "modelserver" (default since 2026-11): one FastAPI Deployment serving many
# models from the MLflow registry, scaled 0<->1 by the infra layer and 1<->N by
# an HPA (app/serve/model_server.py, app/infra/model_server_scaler.py). Idle
# cost is zero pods. "rayserve" stays as the opt-in for GPU-heavy or very
# high-QPS installs; it keeps a Ray head running.
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "modelserver").strip().lower()
ALLOW_LEGACY_GATEWAY = os.getenv("AVALOKA_ALLOW_LEGACY_GATEWAY", "").strip() in {"1", "true", "yes"}
RAY_SERVE_URL = os.getenv("RAY_SERVE_URL", "http://avaloka-inference-serve-svc:8000")
RAYSERVICE_NAME = os.getenv("RAY_SERVICE_NAME", "avaloka-inference")
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://avaloka-model-server:8000")
MODEL_SERVER_TIMEOUT_S = int(os.getenv("MODEL_SERVER_TIMEOUT_S", "30"))


class ServingAsleep(RuntimeError):
    """The serving backend is scaled to zero and is being woken.

    The API answers 503 with ``Retry-After`` instead of holding the request.
    Raised by the model-server backend (PR-1b); defined here so the API's
    contract is in place.
    """

    retry_after = 15


class InferenceServiceManager:
    """Configure/serve/stop model inference.

    The default ``rayserve`` backend targets the in-cluster RayService (Ray Serve)
    at ``RAY_SERVE_URL``; the API's ``/api/models/{run_id}/inference`` route delegates
    straight to :meth:`inference`, so no route changes are needed. The legacy GCP API
    Gateway path is preserved behind ``INFERENCE_BACKEND=gateway``.
    """

    # ---- public API (dispatch on backend) --------------------------------------

    @staticmethod
    def _backend() -> str:
        if INFERENCE_BACKEND not in {"modelserver", "rayserve", "gateway"}:
            raise ValueError(
                f"Unsupported INFERENCE_BACKEND {INFERENCE_BACKEND!r}; expected 'modelserver', 'rayserve' or 'gateway'."
            )
        if INFERENCE_BACKEND == "gateway" and not ALLOW_LEGACY_GATEWAY:
            raise RuntimeError(
                "INFERENCE_BACKEND=gateway creates a new GKE namespace and GCP API "
                "Gateway for every deployment and does not reclaim them; 18 such "
                "namespaces accumulated and billed while idle. Use the default "
                "'rayserve' backend, which serves every model from the shared "
                f"{RAY_CLUSTER_NAME!r} cluster. To run the legacy path anyway (for "
                "recovering an existing deployment), set AVALOKA_ALLOW_LEGACY_GATEWAY=1."
            )
        return INFERENCE_BACKEND

    def _scaler(self) -> RayServiceScaler:
        """Scaler for the shared inference service. Never creates a cluster."""
        return RayServiceScaler(name=RAYSERVICE_NAME, policy=policy_from_env())

    def configure_inference_service(self, mlflow_run_id: str) -> Dict[str, Any]:
        backend = self._backend()
        if backend == "modelserver":
            return self._configure_modelserver(mlflow_run_id)
        if backend == "rayserve":
            return self._configure_rayserve(mlflow_run_id)
        return self._configure_gateway(mlflow_run_id)

    def prepare_inference_service(self, mlflow_run_id: str) -> Dict[str, Any]:
        """Make a run servable without warming anything (ready-on-demand).

        Validates and, if needed, repairs the ONNX artifact and records the
        service details, but does not load the model into the serving backend:
        the first prediction does that. Used by the auto-deploy after training
        so that nothing stays warm (owner ruling: idle cost ~ 0).
        """
        backend = self._backend()
        if backend == "modelserver":
            return self._prepare_modelserver(mlflow_run_id)
        if backend == "rayserve":
            self._ensure_self_contained_onnx(mlflow_run_id)
            model_uri = self._resolve_model_uri(mlflow_run_id)
            details = {
                "backend": "rayserve",
                "endpoint": RAY_SERVE_URL,
                "model_uri": model_uri,
                "run_id": mlflow_run_id,
                "rayservice": RAYSERVICE_NAME,
                "warm": False,
            }
            self._persist_details(mlflow_run_id, details)
            return details
        # The legacy gateway has no "prepared but cold" state: configuring is deploying.
        return self._configure_gateway(mlflow_run_id)

    def inference(self, mlflow_run_id: str, input_data: Dict[str, Any] | List[Dict[str, Any]]) -> Any:
        backend = self._backend()
        if backend == "modelserver":
            return self._inference_modelserver(mlflow_run_id, input_data)
        if backend == "rayserve":
            return self._inference_rayserve(mlflow_run_id, input_data)
        return self._inference_gateway(mlflow_run_id, input_data)

    def stop_inference_service(self, mlflow_run_id: str) -> None:
        backend = self._backend()
        if backend == "modelserver":
            return self._stop_modelserver(mlflow_run_id)
        if backend == "rayserve":
            return self._stop_rayserve(mlflow_run_id)
        return self._stop_gateway(mlflow_run_id)

    # ---- modelserver backend (default) -------------------------------------------

    def _scaler_for(self, mlflow_run_id: str):
        from app.infra.model_server_scaler import GPU_DEPLOYMENT, DEPLOYMENT, ModelServerScaler

        name = DEPLOYMENT
        try:
            from app.core.ml_store import get_ml_store

            ep = get_ml_store().get_endpoint(mlflow_run_id)
            if ep is not None and ep.server == "gpu":
                name = GPU_DEPLOYMENT
        except Exception:  # noqa: BLE001 - default to the CPU server
            pass
        return ModelServerScaler(name)

    def _registry_hint(self, mlflow_run_id: str) -> Dict[str, Any]:
        """The registry record, so the server loads models:/{name}/{version}."""
        try:
            from app.core.ml_store import get_ml_store

            rec = get_ml_store().get_model(mlflow_run_id)
            if rec is not None:
                return {"registered_name": rec.registered_name, "version": rec.version}
        except Exception:  # noqa: BLE001
            pass
        return {}

    def _modelserver_url(self, mlflow_run_id: str, action: str) -> str:
        return f"{MODEL_SERVER_URL.rstrip('/')}/models/{mlflow_run_id}/{action}"

    def _post_modelserver(self, mlflow_run_id: str, action: str, payload: Dict[str, Any],
                          timeout: int = MODEL_SERVER_TIMEOUT_S) -> Dict[str, Any]:
        """POST to the server; translate "nobody is listening" into ServingAsleep
        after asking the infra layer to wake the Deployment."""
        url = self._modelserver_url(mlflow_run_id, action)
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except requests.ConnectionError as exc:
            scaler = self._scaler_for(mlflow_run_id)
            replicas = scaler.replicas()
            if replicas == 0 or replicas is None:
                scaler.wake()
                raise ServingAsleep(f"model server is scaled to zero; waking ({exc.__class__.__name__})") from exc
            raise
        if resp.status_code == 503:
            raise ServingAsleep("model server is starting")
        if resp.status_code == 422:
            detail = resp.json().get("detail") if resp.headers.get("content-type", "").startswith("application/json") else {}
            raise RuntimeError(f"model server could not load the model: {(detail or {}).get('error', resp.text[:200])}")
        resp.raise_for_status()
        result = resp.json()
        if not isinstance(result, dict):
            raise RuntimeError(f"model server returned an invalid response: {result!r}")
        return result

    def _run_algorithm(self, mlflow_run_id: str) -> str:
        """The model family the run trained (``algorithm`` run param on 1.7); ``mlp`` when unknown."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            from app.agents.mta_v2.tracking import resolve_tracking_uri

            mlflow.set_tracking_uri(resolve_tracking_uri(None))
            return str(MlflowClient().get_run(mlflow_run_id).data.params.get("algorithm") or "mlp").lower()
        except Exception:  # noqa: BLE001 - an unreadable run is treated as the MLP it always was
            return "mlp"

    def _prepare_modelserver(self, mlflow_run_id: str) -> Dict[str, Any]:
        """Ready-on-demand: validate the artifact, record the details, load nothing.

        1.7's scikit-learn families (gradient_boosting, random_forest, linear) have
        no ONNX graph: they are logged with mlflow.sklearn under ``model`` and the
        model server loads them through mlflow.pyfunc, so the ONNX self-containment
        check applies to the MLP only and the fallback URI is the model directory.
        """
        algorithm = self._run_algorithm(mlflow_run_id)
        if algorithm == "mlp":
            self._ensure_self_contained_onnx(mlflow_run_id)
        hint = self._registry_hint(mlflow_run_id)
        if hint:
            model_uri = f"models:/{hint['registered_name']}/{hint['version']}"
        elif algorithm == "mlp":
            model_uri = self._resolve_model_uri(mlflow_run_id)
        else:
            model_uri = f"runs:/{mlflow_run_id}/model"
        details = {"backend": "modelserver", "endpoint": MODEL_SERVER_URL, "model_uri": model_uri,
                   "run_id": mlflow_run_id, "warm": False, "algorithm": algorithm, **hint}
        self._persist_details(mlflow_run_id, details)
        return details

    def _configure_modelserver(self, mlflow_run_id: str) -> Dict[str, Any]:
        """Warm: wake the Deployment if asleep, wait for readiness, load the model."""
        details = self._prepare_modelserver(mlflow_run_id)
        scaler = self._scaler_for(mlflow_run_id)
        if scaler.replicas() == 0:
            scaler.wake()
            scaler.wait_ready(timeout_s=int(os.getenv("MODEL_SERVER_WAKE_TIMEOUT_S", "180")))
        payload = {k: details[k] for k in ("registered_name", "version") if k in details}
        if not payload:
            payload = {"model_uri": details["model_uri"]}
        loaded = self._post_modelserver(mlflow_run_id, "load", payload, timeout=120)
        details.update({"warm": True, "bytes": loaded.get("bytes"), "flavor": loaded.get("flavor")})
        self._persist_details(mlflow_run_id, details)
        logger.info("[inference] model server warmed run '%s' (%s)", mlflow_run_id, details["model_uri"])
        return details

    def _inference_modelserver(self, mlflow_run_id: str, input_data: Dict[str, Any] | List[Dict[str, Any]]) -> Any:
        from app.agents.mta_v2.inference import InferenceInterface

        features = input_data.get("features", input_data) if isinstance(input_data, dict) else input_data
        rows = features if isinstance(features, list) else [features]
        if not rows:
            raise ValueError("Inference input must contain at least one row.")
        is_single_vector = isinstance(rows[0], (int, float))
        is_batch = isinstance(features, list) and not is_single_vector
        if is_single_vector:
            rows = [rows]

        config = MLflowManager().get_model_config(mlflow_run_id)
        if config is None:
            raise ValueError(f"Model config not found for MLflow run id '{mlflow_run_id}'")
        hint = self._registry_hint(mlflow_run_id)
        results = []
        for row in rows:
            if isinstance(row, dict):
                vector = InferenceInterface._model_feature_values(row, config)
            elif isinstance(row, (list, tuple)):
                vector = [float(value) for value in row]
            else:
                raise ValueError("Each inference row must be an object or numeric feature vector.")
            service_result = self._post_modelserver(mlflow_run_id, "predict", {"features": vector, **hint})
            if not service_result.get("model_loaded"):
                raise RuntimeError(f"model server did not load run {mlflow_run_id}: {service_result.get('error') or 'unknown'}")
            results.append(self._format_rayserve_prediction(mlflow_run_id, config, service_result))
        return results if is_batch else results[0]

    def _stop_modelserver(self, mlflow_run_id: str) -> None:
        """Unload (best effort: an asleep server has nothing loaded) and clear the details."""
        try:
            requests.post(self._modelserver_url(mlflow_run_id, "unload"), timeout=10)
        except requests.RequestException as exc:
            logger.info("[inference] unload skipped for %s (%s)", mlflow_run_id, exc.__class__.__name__)
        self._persist_details(mlflow_run_id, None)  # type: ignore[arg-type]
        logger.info("[inference] model server endpoint stopped for run '%s'.", mlflow_run_id)

    # ---- rayserve backend ------------------------------------------------------

    def _resolve_model_uri(self, mlflow_run_id: str) -> str:
        """Return the selected run's ONNX artifact as an MLflow artifact URI."""
        uri = os.getenv("MODEL_URI", "").strip()
        if uri:
            return uri
        return f"runs:/{mlflow_run_id}/{MLflowManager.ONNX_ARTIFACT_PATH}"

    def _mlflow_tracking_uri(self) -> str:
        uri = (
            os.getenv("MLFLOW_TRACKING_URI", "").strip()
            or os.getenv("MLFLOW_BACKEND_STORE_URI", "").strip()
        )
        if not uri:
            raise RuntimeError("MLFLOW_TRACKING_URI is required for Ray Serve model loading")
        return uri

    def _ensure_self_contained_onnx(self, mlflow_run_id: str) -> None:
        """Repair older runs whose ONNX external-data sidecar was not uploaded."""
        manager = MLflowManager()
        with tempfile.TemporaryDirectory() as tmp:
            paths = manager.download_model(mlflow_run_id, dst_dir=tmp)
            onnx_path = paths.get("onnx", "")
            if not onnx_path:
                raise RuntimeError(f"Run {mlflow_run_id} has no ONNX model artifact")

            try:
                import onnxruntime as ort

                ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                return
            except Exception as exc:  # noqa: BLE001 - repair legacy exports
                logger.warning(
                    "[inference] ONNX artifact for %s is not self-contained (%s); re-exporting",
                    mlflow_run_id,
                    exc,
                )

            model, _ = manager.load_model(mlflow_run_id, dst_dir=tmp)
            repaired_path = str(Path(tmp) / "model.onnx")
            model.save_onnx(repaired_path)

            # Prove the repaired file is independently loadable before replacing
            # the run artifact.
            import onnxruntime as ort

            ort.InferenceSession(repaired_path, providers=["CPUExecutionProvider"])
            manager.upload_onnx_model(mlflow_run_id, repaired_path)

    def _request_rayserve(self, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        url = RAY_SERVE_URL.rstrip("/") + "/"
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            if not isinstance(result, dict):
                raise RuntimeError(f"Ray Serve returned an invalid response: {result!r}")
            return result
        except requests.RequestException as e:
            logger.error("[inference] RayServe request to '%s' failed: %s", url, e)
            raise

    def _load_rayserve_model(self, model_uri: str) -> Dict[str, Any]:
        result = self._request_rayserve({
            "model_uri": model_uri,
            "mlflow_tracking_uri": self._mlflow_tracking_uri(),
            "load_only": True,
        })
        if not result.get("model_loaded"):
            reason = result.get("error") or result.get("note") or "unknown model-loading error"
            raise RuntimeError(f"Ray Serve could not load {model_uri}: {reason}")
        return result

    def _persist_details(self, mlflow_run_id: str, details: Dict[str, Any]) -> None:
        try:
            mm = MLflowManager()
            cfg = mm.get_model_config(mlflow_run_id)
            if cfg is not None:
                cfg.inference_service_details = details
                mm.update_model_config(mlflow_run_id, cfg)
        except Exception as e:  # noqa: BLE001 - persistence is best-effort for rayserve
            logger.info("[inference] Skipped MLflow persistence for %s (%s)", mlflow_run_id, e)

    def _configure_rayserve(self, mlflow_run_id: str) -> Dict[str, Any]:
        self._ensure_self_contained_onnx(mlflow_run_id)
        model_uri = self._resolve_model_uri(mlflow_run_id)
        self._load_rayserve_model(model_uri)
        details = {
            "backend": "rayserve",
            "endpoint": RAY_SERVE_URL,
            "model_uri": model_uri,
            "run_id": mlflow_run_id,
            "rayservice": RAYSERVICE_NAME,
        }
        self._persist_details(mlflow_run_id, details)
        logger.info("[inference] Configured RayServe inference for run '%s' at %s (model_uri=%s)",
                    mlflow_run_id, RAY_SERVE_URL, model_uri or "<deploy-time default>")
        return details

    def _inference_rayserve(
        self,
        mlflow_run_id: str,
        input_data: Dict[str, Any] | List[Dict[str, Any]],
    ) -> Any:
        from app.agents.mta_v2.inference import InferenceInterface

        # Ramp up first if an idle reap scaled the service to zero, then record
        # the request so the reaper's idle clock restarts. Both are best-effort:
        # neither may fail an inference that could otherwise be served.
        try:
            self._scaler().ensure_capacity()
        except Exception as exc:  # noqa: BLE001 - never block the request path
            logger.warning("[inference] ensure_capacity failed (continuing): %s", exc)
        mark_activity()

        features = input_data.get("features", input_data) if isinstance(input_data, dict) else input_data
        rows = features if isinstance(features, list) else [features]
        if not rows:
            raise ValueError("Inference input must contain at least one row.")

        # A single already-encoded vector is also accepted by the API contract.
        is_single_vector = isinstance(rows[0], (int, float))
        is_batch = isinstance(features, list) and not is_single_vector
        if is_single_vector:
            rows = [rows]

        config = MLflowManager().get_model_config(mlflow_run_id)
        if config is None:
            raise ValueError(f"Model config not found for MLflow run id '{mlflow_run_id}'")

        model_uri = self._resolve_model_uri(mlflow_run_id)
        results = []
        for row in rows:
            if isinstance(row, dict):
                # Build the exact saved tensor contract (including one-hot
                # expansion and PII de-identification), rather than assuming
                # one scalar per original feature.
                vector = InferenceInterface._model_feature_values(row, config)
            elif isinstance(row, (list, tuple)):
                vector = [float(value) for value in row]
            else:
                raise ValueError("Each inference row must be an object or numeric feature vector.")

            service_result = self._request_rayserve(
                {
                    "features": vector,
                    "model_uri": model_uri,
                    "mlflow_tracking_uri": self._mlflow_tracking_uri(),
                },
                timeout=30,
            )
            if not service_result.get("model_loaded"):
                reason = service_result.get("error") or service_result.get("note") or "unknown model-loading error"
                raise RuntimeError(f"Ray Serve did not load {model_uri}: {reason}")
            results.append(self._format_rayserve_prediction(mlflow_run_id, config, service_result))

        return results if is_batch else results[0]

    @staticmethod
    def _format_rayserve_prediction(mlflow_run_id: str, config: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        """Decode raw ONNX gateway output using the same metadata as chat inference."""
        scores = result.get("scores") or []
        raw_output = scores[0] if scores and isinstance(scores[0], list) else scores
        common = {
            "mlflow_run_id": mlflow_run_id,
            "model_name": config.model_name,
            "model_version": config.model_version,
            "model_type": config.model_type,
            "model_uri": result.get("model_uri"),
            "model_loaded": result.get("model_loaded", False),
            "backend": result.get("backend"),
        }

        if "classification" in str(config.model_type).lower():
            logits = [float(value) for value in raw_output]
            if not logits:
                raise RuntimeError("Ray Serve returned no classification scores.")
            max_logit = max(logits)
            exponentials = [math.exp(value - max_logit) for value in logits]
            denominator = sum(exponentials) or 1.0
            probabilities = [value / denominator for value in exponentials]
            index = max(range(len(probabilities)), key=probabilities.__getitem__)
            class_names = config.class_names or [str(i) for i in range(len(probabilities))]
            label = class_names[index] if index < len(class_names) else str(index)
            return {
                **common,
                "prediction": label,
                "predicted_class_index": index,
                "raw_output": [round(value, 6) for value in logits],
                "probabilities": {
                    (class_names[i] if i < len(class_names) else str(i)): round(probability, 6)
                    for i, probability in enumerate(probabilities)
                },
            }

        scaled_value = float(raw_output[0] if raw_output else result.get("prediction"))
        scaler = (config.preprocessing or {}).get("target_scaling") or {}
        prediction = scaled_value
        if scaler.get("method") == "standard":
            prediction = (
                scaled_value * float(scaler.get("scale", 1.0))
                + float(scaler.get("mean", 0.0))
            )
        return {
            **common,
            "prediction": round(prediction, 6),
            "predicted_class_index": None,
            "raw_output": [round(prediction, 6)],
            "probabilities": {},
        }

    def _stop_rayserve(self, mlflow_run_id: str) -> None:
        """Clear this run's configuration without stopping the shared RayService."""
        self._persist_details(mlflow_run_id, None)  # type: ignore[arg-type]
        logger.info("[inference] Stopped RayServe inference for run '%s'.", mlflow_run_id)

    # ---- legacy GCP API Gateway backend (INFERENCE_BACKEND=gateway) -------------

    def _configure_gateway(self, mlflow_run_id: str) -> Dict[str, Any]:
        from app.agents.mta_v2.gcp.run_inference_service import run_inference_service
        from app.agents.mta_v2.gcp.create_api_gateway import create_api_gateway

        mlflow_manager = MLflowManager()
        model_config = mlflow_manager.get_model_config(mlflow_run_id)
        if model_config is None:
            logger.error(f"No model config found for MLflow run id '{mlflow_run_id}'. Cannot configure inference service.")
            raise ValueError(f"No model config found for MLflow run id '{mlflow_run_id}'")

        if model_config.inference_service_details is not None:
            logger.info(
                "Inference service is already configured for MLflow run id '%s'. Reusing existing endpoint.",
                mlflow_run_id,
            )
            return model_config.inference_service_details

        # Generate unique identifiers for the inference service
        unique_id = str(uuid.uuid4())[:8]
        namespace = f"inference-service-{unique_id}"
        api_id = f"inference-api-{unique_id}"
        config_id = f"inference-config-{unique_id}"
        gateway_id = f"inference-gateway-{unique_id}"

        # Deploy the inference service on GKE
        endpoint = run_inference_service(
            project_id=inference_service_project_id(),
            location=INFERENCE_SERVICE_GKE_CLUSTER_LOCATION,
            docker_image=inference_service_docker_image(),
            namespace=namespace,
            env_vars={
                "GCP_SERVICE_ACCOUNT_JSON": open(os.path.expanduser(GCP_SA_JSON_PATH)).read().strip(),
                "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "").strip(),
                "MLFLOW_BACKEND_STORE_URI": os.getenv("MLFLOW_BACKEND_STORE_URI", "").strip(),
                "MLFLOW_EXPERIMENT_NAME": os.getenv("MLFLOW_EXPERIMENT_NAME", "avaloka-training").strip(),
                "MLFLOW_RUN_ID": mlflow_run_id,
                "RAY_INFERENCE_DOCKER_URI": cloud_config.image(
                    "RAY_INFERENCE_DOCKER_URI", "inference-docker-image", "test-v0.0.2"),
                "RAY_GKE_CLUSTER_LOCATION": os.getenv("RAY_GKE_CLUSTER_LOCATION", "us-central1"),
                "RAY_GKE_CLUSTER_NAME": os.getenv("RAY_GKE_CLUSTER_NAME", "ray-gke-trainer"),
                "GOOGLE_CLOUD_PROJECT": inference_service_project_id(),
            },
        )

        # Create API Gateway to expose the inference service
        gateway_url, managed_service = create_api_gateway(
            project_id=inference_service_project_id(),
            location=INFERENCE_SERVICE_GKE_CLUSTER_LOCATION,
            backend_ip=endpoint.split("//")[1].split(":")[0],
            backend_port=endpoint.split(":")[-1],
            api_id=api_id,
            config_id=config_id,
            gateway_id=gateway_id,
        )

        inference_service_details = {
            "namespace": namespace,
            "endpoint": endpoint,
            "api_id": api_id,
            "config_id": config_id,
            "gateway_id": gateway_id,
            "gateway_url": f"https://{gateway_url}",
            "managed_service": managed_service,
        }

        model_config.inference_service_details = inference_service_details
        mlflow_manager.update_model_config(mlflow_run_id, model_config)
        logger.info(f"Inference service configured and deployed for MLflow run id '{mlflow_run_id}'. Endpoint: {endpoint}, Gateway URL: {gateway_url}")

        return inference_service_details

    def _stop_gateway(self, mlflow_run_id: str) -> None:
        from app.agents.mta_v2.gcp.run_inference_service import delete_inference_service
        from app.agents.mta_v2.gcp.create_api_gateway import delete_api_gateway

        mlflow_manager = MLflowManager()
        model_config = mlflow_manager.get_model_config(mlflow_run_id)
        if model_config is None or model_config.inference_service_details is None:
            logger.info(f"No inference service configured for model with run id '{mlflow_run_id}'. Nothing to stop.")
            return

        inference_service_details = model_config.inference_service_details

        delete_api_gateway(
            inference_service_project_id(),
            INFERENCE_SERVICE_GKE_CLUSTER_LOCATION,
            api_id=inference_service_details["api_id"],
            config_id=inference_service_details["config_id"],
            gateway_id=inference_service_details["gateway_id"],
        )
        delete_inference_service(
            inference_service_project_id(),
            INFERENCE_SERVICE_GKE_CLUSTER_LOCATION,
            namespace=inference_service_details["namespace"],
        )

        model_config.inference_service_details = None
        mlflow_manager.update_model_config(mlflow_run_id, model_config)
        logger.info("Inference service stopped and details cleared in MLflow for run id '%s'.", mlflow_run_id)

    def _inference_gateway(self, mlflow_run_id: str, input_data: Dict[str, Any] | List[Dict[str, Any]]) -> Any:
        mlflow_manager = MLflowManager()
        model_config = mlflow_manager.get_model_config(mlflow_run_id)
        if model_config is None or model_config.inference_service_details is None:
            logger.error(f"No inference service configured for model with run id '{mlflow_run_id}'. Cannot perform inference.")
            raise ValueError(f"No inference service configured for model with run id '{mlflow_run_id}'")

        inference_service_details = model_config.inference_service_details
        gateway_url = inference_service_details.get("gateway_url")
        if gateway_url.startswith("http://") or gateway_url.startswith("https://"):
            endpoint = gateway_url + "/inference"
        else:
            endpoint = "https://" + gateway_url + "/inference"

        try:
            response = requests.post(endpoint, json=input_data)
            response.raise_for_status()
            prediction = response.json()
        except requests.RequestException as e:
            logger.error(f"Error during inference request to '{endpoint}': {e}")
            raise

        return prediction
