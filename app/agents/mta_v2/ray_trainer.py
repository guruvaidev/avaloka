import json
import os
import uuid
import time
import logging
import threading
from dotenv import load_dotenv

from app.agents.mta_v2.gcp.get_k8s_api import get_k8s_api
from app.agents.mta_v2.gcp.namespace import create_namespace
from app.agents.mta_v2.gcp.create_ray_cluster import create_ray_cluster
from app.agents.mta_v2.gcp.get_ray_head_svc import get_ray_head_svc
from app.agents.mta_v2.gcp.get_ray_dashboard_url import get_ray_dashboard_url
from app.agents.mta_v2.gcp.wait_for_ray_job_completion import wait_for_ray_job_completion
from app.agents.mta_v2.gcp.submit_ray_job import submit_ray_job
from app.agents.mta_v2.gcp.get_job_result import get_job_logs, get_job_result
from app.agents.mta_v2.gcp.clean_up import delete_ray_cluster
from app.agents.mta_v2.failure_diagnostics import build_training_failure
from app.agents.pii_agent import MissingPIIKey, PII_KEY_ENV
from app.agents.mta_v2.schema import TrainingPlan, TrainingResult
from app.core import cloud_config
from app.core.scheduled_run_context import notify_ray_job_submitted

load_dotenv()

logger = logging.getLogger(__name__)

# Lazy for the same reason as the inference path: fail at deployment with the
# variable name, not at import.
def ray_gcp_project_id() -> str:
    return cloud_config.project_id()


def ray_training_docker_uri() -> str:
    return cloud_config.image(
        "RAY_TRAINING_DOCKER_URI",
        "train-docker-image",
        "test-v0.4.24",
    )


RAY_GKE_CLUSTER_NAME = os.getenv("RAY_GKE_CLUSTER_NAME", "ray-gke-trainer")
RAY_GKE_CLUSTER_LOCATION = os.getenv("RAY_GKE_CLUSTER_LOCATION", "us-central1")
RAY_DOCKER_ENTRYPOINT = "python /app/ray_job.py"

# Path to GCP service account JSON; read once at module level to surface
# missing-file errors early rather than deep inside train().
_GCP_SA_JSON_PATH = os.getenv("GCP_SERVICE_ACCOUNT_JSON_PATH", "~/Templates/gcp-bucket-cred.json")


class RayTrainer:
    """
    Orchestrates a Ray training run for a DynamicMLP model.

    When ``MTA_RAY_DASHBOARD_URL`` is configured, training is submitted to
    that existing Ray cluster through the Ray Jobs API.  This is the normal
    Kubernetes path: Celery only dispatches the scheduled task, and the
    dedicated ``avaloka-raycluster`` workers perform the training.  The
    cluster is shared and is therefore never deleted by this class.

    Without ``MTA_RAY_DASHBOARD_URL`` the legacy behavior is preserved: a
    dedicated RayCluster is created in GKE for the run and cleaned up after
    completion.

    Parameters
    ----------
    user_id : str
        Identifier of the user who initiated training. Forwarded to the
        Ray job as an environment variable and stored in MLflow tags.
    session_id : str
        Conversation/session identifier. Forwarded similarly.
    """

    def __init__(self, user_id: str, session_id: str) -> None:
        self.task_id = str(uuid.uuid4())[:8]
        self.namespace = f"ray-trainer-{self.task_id}"
        self.user_id = user_id
        self.session_id = session_id

    @staticmethod
    def _existing_dashboard_url() -> str:
        """Return the explicitly configured shared Ray Jobs endpoint."""
        return os.getenv("MTA_RAY_DASHBOARD_URL", "").strip().rstrip("/")

    @staticmethod
    def _training_entrypoint() -> str:
        """Return the command installed in the selected Ray cluster image."""
        return (
            os.getenv("MTA_RAY_TRAINING_ENTRYPOINT", RAY_DOCKER_ENTRYPOINT).strip()
            or RAY_DOCKER_ENTRYPOINT
        )

    def _build_job_env(
        self,
        training_plan: TrainingPlan,
        *,
        gcp_sa_json: str = "",
    ) -> dict[str, str]:
        """Build the runtime environment shared by both Ray deployment modes."""
        ray_cfg = training_plan.get("ray_config") or {}
        hyper_cfg = training_plan.get("hyperparameter_config") or {}
        data_cfg = training_plan.get("data_config") or {}
        pii_columns = data_cfg.get("pii_key_required_columns") or []
        if pii_columns and not (os.getenv(PII_KEY_ENV) or "").strip():
            labels = ", ".join(repr(str(column)) for column in pii_columns)
            raise MissingPIIKey(
                f"{PII_KEY_ENV} is not set. Sensitive training column(s) {labels} "
                "require keyed protection before Ray training can start."
            )
        pii_transformations = dict(data_cfg.get("pii_transformations") or {})
        # Preserve protection for pending plans created before the full
        # transformation contract was added.
        for column in pii_columns:
            pii_transformations.setdefault(
                str(column),
                {"kind": "none", "strategy": "pseudonymise"},
            )

        cpu_per_worker = ray_cfg.get("cpu_per_worker", 4)
        num_workers = ray_cfg.get("num_workers", 4)
        job_env = {
            "CPUS_PER_WORKER": str(cpu_per_worker),
            "NUM_WORKERS": str(num_workers),
            "MIN_WORKERS": str(ray_cfg.get("min_workers", 1)),
            "MAX_WORKERS": str(ray_cfg.get("max_workers", 25)),
            "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", ""),
            "MLFLOW_BACKEND_STORE_URI": os.getenv("MLFLOW_BACKEND_STORE_URI", ""),
            "MLFLOW_DEFAULT_ARTIFACT_ROOT": os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", ""),
            "MLFLOW_EXPERIMENT_NAME": os.getenv("MLFLOW_EXPERIMENT_NAME", "avaloka-training"),
            "TASK_ID": self.task_id,
            "USER_ID": self.user_id,
            "SESSION_ID": self.session_id,
            "MODEL_TYPE": training_plan.get("model_type", "classification"),
            "MODEL_NAME": training_plan.get("model_name", "unknown_model"),
            "MODEL_DESCRIPTION": training_plan.get("model_description", ""),
            "MODEL_VERSION": training_plan.get("model_version", "v1.0"),
            "DATA_SOURCE_URI": data_cfg.get("dataset_uri", ""),
            "FEATURE_COLUMNS": ",".join(data_cfg.get("feature_columns", [])),
            "TARGET_COLUMN": data_cfg.get("target_column", ""),
            "TIME_COLUMN": data_cfg.get("time_column", ""),
            "LEARNING_RATE": str(hyper_cfg.get("learning_rate", 0.001)),
            "EPOCHS": str(hyper_cfg.get("epochs", 30)),
            "BATCH_SIZE": str(hyper_cfg.get("batch_size", 32)),
            "OPTIMIZER_NAME": hyper_cfg.get("optimizer_name", "adam"),
            "RANDOM_SEED": str(hyper_cfg.get("random_seed", 42)),
            "EARLY_STOPPING_PATIENCE": str(hyper_cfg.get("early_stopping_patience", 5)),
            "EARLY_STOPPING_MIN_DELTA": str(hyper_cfg.get("early_stopping_min_delta", 1e-4)),
            "USE_GPU": str(ray_cfg.get("use_gpu", False)),
            "HIDDEN_LAYER_SIZES": ",".join(
                map(str, hyper_cfg.get("hidden_layer_sizes", [128, 64]))
            ),
            "ACTIVATION": hyper_cfg.get("activation", "relu"),
            "DROPOUT_RATE": str(hyper_cfg.get("dropout_rate", 0.2)),
            "BATCH_NORM": str(hyper_cfg.get("batch_norm", True)),
            # The secret is supplied to the Ray pods by the Kubernetes Secret
            # already referenced by the RayCluster manifest. Only this safe
            # contract is sent in the Ray Jobs request.
            "PII_TRANSFORMATIONS": json.dumps(
                pii_transformations, sort_keys=True, separators=(",", ":")
            ),
        }
        if gcp_sa_json:
            job_env.update(
                {
                    "GCP_SERVICE_ACCOUNT_JSON": gcp_sa_json,
                    "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/gcp_sa.json",
                }
            )
        return job_env

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_gcp_sa_json(self) -> str:
        """
        Read the GCP service-account JSON from disk.

        Raises
        ------
        FileNotFoundError
            If the file path referenced by GCP_SERVICE_ACCOUNT_JSON_PATH
            does not exist.
        """
        credential_path = os.path.expanduser(_GCP_SA_JSON_PATH)
        if not os.path.exists(credential_path):
            raise FileNotFoundError(
                f"GCP service-account JSON not found at '{credential_path}'. "
                "Set GCP_SERVICE_ACCOUNT_JSON_PATH to the correct path."
            )
        with open(credential_path) as f:
            return f.read()

    def _schedule_cluster_deletion(self, api_client, delay_s: int = 60 * 30) -> None:
        """
        Delete the Ray cluster in a background thread after *delay_s* seconds.

        Parameters
        ----------
        api_client :
            Kubernetes API client used for the deletion call.
        delay_s : int
            Seconds to wait before deleting. Defaults to 30 minutes.
        """
        def _delete():
            time.sleep(delay_s)
            try:
                delete_ray_cluster(api_client, RAY_GKE_CLUSTER_NAME, namespace=self.namespace)
                logger.info(
                    "Deleted Ray cluster '%s' in namespace '%s'.",
                    RAY_GKE_CLUSTER_NAME,
                    self.namespace,
                )
            except Exception as exc:
                logger.error(
                    "Failed to delete Ray cluster '%s' in namespace '%s': %s",
                    RAY_GKE_CLUSTER_NAME,
                    self.namespace,
                    exc,
                )

        threading.Thread(target=_delete, daemon=True).start()

    def _pod_failure_diagnostics(self, api_client) -> dict:
        """Return the most actionable failed pod/container status in this run."""
        if api_client is None:
            return {}
        try:
            from kubernetes import client as k8s_client

            core = k8s_client.CoreV1Api(api_client)
            pods = core.list_namespaced_pod(namespace=self.namespace).items
            candidates = []
            priority = {
                "OOMKilled": 100, "ErrImagePull": 90, "ImagePullBackOff": 90,
                "Evicted": 80, "FailedScheduling": 70, "Error": 60,
            }
            for pod in pods:
                pod_name = pod.metadata.name
                pod_reason = getattr(pod.status, "reason", None)
                pod_message = getattr(pod.status, "message", None)
                if pod_reason or pod_message:
                    candidates.append({
                        "score": priority.get(str(pod_reason), 10),
                        "pod_name": pod_name, "reason": pod_reason, "message": pod_message,
                    })
                statuses = list(getattr(pod.status, "init_container_statuses", None) or [])
                statuses += list(getattr(pod.status, "container_statuses", None) or [])
                for status in statuses:
                    for state in (getattr(status, "state", None), getattr(status, "last_state", None)):
                        if state is None:
                            continue
                        terminated = getattr(state, "terminated", None)
                        waiting = getattr(state, "waiting", None)
                        if terminated is not None:
                            reason = terminated.reason or "Terminated"
                            candidates.append({
                                "score": priority.get(str(reason), 50), "pod_name": pod_name,
                                "container_name": status.name, "reason": reason,
                                "message": terminated.message, "exit_code": terminated.exit_code,
                            })
                        elif waiting is not None and waiting.reason:
                            candidates.append({
                                "score": priority.get(str(waiting.reason), 40), "pod_name": pod_name,
                                "container_name": status.name, "reason": waiting.reason,
                                "message": waiting.message,
                            })
                for condition in list(getattr(pod.status, "conditions", None) or []):
                    if condition.type == "PodScheduled" and condition.status == "False":
                        candidates.append({
                            "score": priority.get(str(condition.reason), 70), "pod_name": pod_name,
                            "reason": condition.reason or "FailedScheduling", "message": condition.message,
                        })
            if not candidates:
                return {}
            diagnostic = max(candidates, key=lambda item: item["score"])
            diagnostic.pop("score", None)
            try:
                diagnostic["logs"] = core.read_namespaced_pod_log(
                    name=diagnostic["pod_name"], namespace=self.namespace,
                    container=diagnostic.get("container_name"), tail_lines=100, timestamps=True,
                )
            except Exception:
                pass
            return diagnostic
        except Exception as exc:
            logger.warning("Could not inspect failed training pods: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, training_plan: TrainingPlan) -> TrainingResult | None:
        """
        Execute a training run on a shared Kubernetes Ray cluster or GKE.

        Parameters
        ----------
        training_plan : TrainingPlan
            Fully populated training plan produced by the planning agent.
            Must contain ``data_config``, ``hyperparameter_config``, and
            ``ray_config`` sub-dicts.

        Returns
        -------
        TrainingResult
            Metrics and identifiers produced by the completed Ray job,
            augmented with the Ray dashboard URL.

        Raises
        ------
        FileNotFoundError
            If the GCP service-account JSON file is missing.
        RuntimeError
            If the Ray job fails or times out (propagated from get_job_result).
        """
        ray_cfg = training_plan.get("ray_config") or {}

        api_client = None
        dashboard_url = None
        job_id = None
        owns_cluster = False
        try:
            cpu_per_worker    = ray_cfg.get("cpu_per_worker", 4)
            memory_per_worker = ray_cfg.get("memory_per_worker", "16Gi")
            num_workers       = ray_cfg.get("num_workers", 4)
            existing_dashboard_url = self._existing_dashboard_url()
            gcp_sa_json = ""

            if existing_dashboard_url:
                dashboard_url = existing_dashboard_url
                self.namespace = os.getenv("MTA_RAY_NAMESPACE", "default").strip() or "default"
                logger.info(
                    "Using existing Ray cluster at '%s' in namespace '%s'.",
                    dashboard_url,
                    self.namespace,
                )
            else:
                # Legacy path: provision a dedicated RayCluster in GKE.
                owns_cluster = True
                gcp_sa_json = self._read_gcp_sa_json()
                api_client = get_k8s_api(
                    RAY_GKE_CLUSTER_NAME,
                    ray_gcp_project_id(),
                    RAY_GKE_CLUSTER_LOCATION,
                )
                create_namespace(api_client, self.namespace)
                logger.info("Created namespace '%s'.", self.namespace)
                create_ray_cluster(
                    api_client,
                    RAY_GKE_CLUSTER_NAME,
                    namespace=self.namespace,
                    docker_image=ray_training_docker_uri(),
                    worker_replicas=num_workers,
                    worker_resources={
                        "requests": {"cpu": str(cpu_per_worker), "memory": memory_per_worker},
                        "limits": {"cpu": str(cpu_per_worker), "memory": memory_per_worker},
                    },
                )
                logger.info(
                    "Created Ray cluster '%s' in namespace '%s'.",
                    RAY_GKE_CLUSTER_NAME,
                    self.namespace,
                )
                ray_head_svc = get_ray_head_svc(
                    api_client, RAY_GKE_CLUSTER_NAME, namespace=self.namespace
                )
                logger.info("Ray head service name: '%s'.", ray_head_svc)
                dashboard_url = get_ray_dashboard_url(
                    api_client, ray_head_svc, namespace=self.namespace
                )
                logger.info("Ray dashboard URL: '%s'.", dashboard_url)

            job_env = self._build_job_env(training_plan, gcp_sa_json=gcp_sa_json)
            job_id = submit_ray_job(
                dashboard_url,
                job_env,
                entry_point=self._training_entrypoint(),
            )
            logger.info("Submitted Ray job with ID: '%s'.", job_id)
            notify_ray_job_submitted({
                "job_id": job_id,
                "status": "SUBMITTED",
                "dashboard_url": dashboard_url,
                "namespace": self.namespace,
            })

            # ---- Wait for completion (12-hour timeout) ----
            job_status = wait_for_ray_job_completion(dashboard_url, job_id, 3600 * 12)
            if job_status != "SUCCEEDED":
                try:
                    logs = get_job_logs(dashboard_url, job_id)
                except Exception as log_exc:
                    logs = f"Unable to retrieve Ray job logs: {log_exc}"
                kube = self._pod_failure_diagnostics(api_client)
                failure = build_training_failure(
                    "Training timed out." if job_status == "UNKNOWN" else f"Ray job finished with status {job_status}.",
                    logs=kube.get("logs") or logs,
                    kubernetes=kube,
                    job_id=job_id,
                    namespace=self.namespace,
                )
                if owns_cluster:
                    self._schedule_cluster_deletion(api_client, delay_s=60 * 30)
                return TrainingResult(
                    status="error", error=failure["technical_details"], failure=failure,
                    task_id=self.task_id, dashboard_url=dashboard_url, user_id=self.user_id,
                    session_id=self.session_id, ray_job_id=job_id,
                    ray_job_status=job_status, ray_namespace=self.namespace,
                )
            job_result = get_job_result(dashboard_url, job_id)
            logger.info("Ray job finished. Result: %s", job_result)

            # ---- Schedule deferred cluster cleanup ----
            if owns_cluster:
                self._schedule_cluster_deletion(api_client, delay_s=60 * 30)

            if not job_result:
                failure = build_training_failure(
                    "Ray training succeeded but returned no parseable result payload.",
                    job_id=job_id, namespace=self.namespace,
                )
                return TrainingResult(
                    status="error", error=failure["technical_details"], failure=failure,
                    task_id=self.task_id, dashboard_url=dashboard_url, user_id=self.user_id,
                    session_id=self.session_id, ray_job_id=job_id,
                    ray_job_status="SUCCEEDED", ray_namespace=self.namespace,
                )
            result_payload = dict(job_result)
            result_payload.update({
                "dashboard_url": dashboard_url,
                "ray_job_id": job_id,
                "ray_job_status": "SUCCEEDED",
                "ray_namespace": self.namespace,
            })
            return TrainingResult(**result_payload)
        except Exception as e:
            logger.error("Training failed: %s", e, exc_info=True)
            kube = self._pod_failure_diagnostics(api_client)
            failure = build_training_failure(
                e, logs=kube.get("logs", ""), kubernetes=kube,
                job_id=job_id, namespace=self.namespace,
            )
            if owns_cluster and api_client is not None:
                self._schedule_cluster_deletion(api_client, delay_s=0)
            ray_metadata = ({
                "ray_job_id": job_id,
                "ray_job_status": "FAILED",
                "ray_namespace": self.namespace,
            } if job_id else {})
            return TrainingResult(
                status="error", error=failure["technical_details"], failure=failure,
                task_id=self.task_id, dashboard_url=dashboard_url, user_id=self.user_id,
                session_id=self.session_id, **ray_metadata,
            )
