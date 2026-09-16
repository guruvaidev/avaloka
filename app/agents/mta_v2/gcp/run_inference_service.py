"""
run_inference_service.py
------------------------
Deploy the Avaloka inference service Docker image onto a brand-new GKE
Autopilot cluster.

High-level flow
---------------
1.  Create a GKE Autopilot cluster  (reuses create_gke_cluster internals,
    but WITHOUT the Ray operator add-on — this cluster runs the inference
    service only).
2.  Obtain a Kubernetes API client for the new cluster.
3.  Create the target namespace (default: "inference").
4.  (Optional) Create an image-pull Secret if a registry credential JSON is
    supplied.
5.  Deploy the inference service as a Kubernetes Deployment:
      - Configures all env-vars the server reads (MLflow, GCP, Ray back-ref).
      - Exposes port 8080.
      - Adds a /health liveness + readiness probe.
6.  Expose the Deployment via a LoadBalancer Service.
7.  Wait until the Deployment is fully available and the Service has an
    external IP, then return the public endpoint.

Cleanup helper
--------------
``delete_inference_service`` tears down the Deployment + Service (and
optionally the whole GKE cluster) so callers can clean up with one call.

Usage example
-------------
>>> from gcp.run_inference_service import run_inference_service
>>> endpoint = run_inference_service(
...     cluster_name="avaloka-inference",
...     project_id="my-gcp-project",
...     location="us-central1",
...     docker_image="gcr.io/my-gcp-project/inference-service:latest",
...     env_vars={
...         "MLFLOW_TRACKING_URI": "https://mlflow.example.com",
...         "MLFLOW_EXPERIMENT_NAME": "avaloka-training",
...         "MLFLOW_RUN_ID": "abc123",
...     },
... )
>>> print(f"Inference endpoint: {endpoint}/inference")
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from google.cloud import container_v1
from kubernetes import client as k8s_client

from .utils import wait_for_operation, wait_for_all_operations_to_complete
from .get_k8s_api import get_k8s_api
from .namespace import create_namespace, delete_namespace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CLUSTER_NAME = "avaloka-inference-service"
DEFAULT_NAMESPACE = "inference"
DEFAULT_REPLICAS = 1
DEFAULT_PORT = 8080

DEFAULT_RESOURCES: dict = {
    "requests": {"cpu": "1", "memory": "2Gi"},
    "limits":   {"cpu": "2", "memory": "4Gi"},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference_service(
    project_id: str,
    location: str,
    docker_image: str,
    *,
    # Kubernetes knobs
    cluster_name: str = DEFAULT_CLUSTER_NAME,
    namespace: str = DEFAULT_NAMESPACE,
    replicas: int = DEFAULT_REPLICAS,
    resources: dict = DEFAULT_RESOURCES,
    port: int = DEFAULT_PORT,
    # Inference service env-vars (all optional; pass only what you need)
    env_vars: Optional[dict[str, str]] = None,
    # Registry pull secret (base64-encoded Docker config JSON string)
    image_pull_secret_json: Optional[str] = None,
    # Timing
    cluster_max_retries: int = 600,
    deploy_timeout_s: int = 600,
) -> str:
    """
    Create a GKE Autopilot cluster and deploy the inference service image.

    Parameters
    ----------
    cluster_name:
        Name for the new GKE cluster.
    project_id:
        GCP project ID.
    location:
        GCP region or zone (e.g. ``"us-central1"``).
    docker_image:
        Full Docker image URI for the inference service
        (e.g. ``"gcr.io/my-project/inference-service:latest"``).
    namespace:
        Kubernetes namespace to deploy into (created if absent).
    replicas:
        Initial replica count for the Deployment.
    resources:
        CPU/memory requests & limits for each pod.
    port:
        Container port (must match the server's ``PORT`` env-var / EXPOSE).
    env_vars:
        Extra environment variables injected into every pod.  Keys map to the
        env-vars consumed by ``server.py``:
          - ``MLFLOW_TRACKING_URI``
          - ``MLFLOW_DEFAULT_ARTIFACT_ROOT``
          - ``MLFLOW_EXPERIMENT_NAME``
          - ``MLFLOW_RUN_ID``
          - ``GOOGLE_CLOUD_PROJECT``
          - ``RAY_GKE_CLUSTER_NAME``
          - ``RAY_GKE_CLUSTER_LOCATION``
          - ``RAY_INFERENCE_DOCKER_URI``
          - ``GCP_SERVICE_ACCOUNT_JSON``
          - ``GOOGLE_APPLICATION_CREDENTIALS``
          - ``DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING``
    image_pull_secret_json:
        If the image lives in a private registry, pass the raw Docker config
        JSON (the contents of ``~/.docker/config.json``).  A Secret named
        ``inference-pull-secret`` will be created in the target namespace.
    cluster_max_retries:
        Polling retries for GKE cluster / operation readiness.
    deploy_timeout_s:
        Seconds to wait for the Deployment to become Available and the
        LoadBalancer to obtain an external IP.

    Returns
    -------
    str
        The public HTTP endpoint, e.g. ``"http://34.56.78.90:8080"``.
    """
    logger.info(
        "Ensuring GKE cluster '%s' exists in %s/%s …",
        cluster_name, project_id, location,
    )
    _create_inference_gke_cluster(cluster_name, project_id, location, cluster_max_retries)
    logger.info("GKE cluster '%s' is ready.", cluster_name)

    api_client = get_k8s_api(cluster_name, project_id, location)

    logger.info("Creating namespace '%s' …", namespace)
    create_namespace(api_client, namespace)

    pull_secret_name: Optional[str] = None
    if image_pull_secret_json:
        pull_secret_name = "inference-pull-secret"
        logger.info("Creating image-pull secret '%s' …", pull_secret_name)
        _create_pull_secret(api_client, pull_secret_name, namespace, image_pull_secret_json)

    deployment_name = f"{cluster_name}-deployment"
    service_name    = f"{cluster_name}-svc"

    logger.info("Deploying inference service as '%s' …", deployment_name)
    _create_deployment(
        api_client=api_client,
        deployment_name=deployment_name,
        namespace=namespace,
        docker_image=docker_image,
        replicas=replicas,
        port=port,
        resources=resources,
        env_vars=env_vars or {},
        pull_secret_name=pull_secret_name,
    )

    logger.info("Creating LoadBalancer Service '%s' …", service_name)
    _create_service(api_client, service_name, deployment_name, namespace, port)

    logger.info("Waiting for Deployment to become available …")
    _wait_for_deployment(api_client, deployment_name, namespace, deploy_timeout_s)

    logger.info("Waiting for external IP on Service '%s' …", service_name)
    external_ip = _wait_for_external_ip(api_client, service_name, namespace, deploy_timeout_s)

    endpoint = f"http://{external_ip}:{port}"
    logger.info("Inference service is live at %s", endpoint)
    return endpoint


def delete_inference_service(
    project_id: str,
    location: str,
    *,
    cluster_name: str = DEFAULT_CLUSTER_NAME,
    namespace: str = DEFAULT_NAMESPACE,
    delete_cluster: bool = False,
    max_retries: int = 600,
) -> None:
    """
    Remove the inference Deployment + Service from the cluster.

    Parameters
    ----------
    cluster_name:
        GKE cluster name. Defaults to ``DEFAULT_CLUSTER_NAME``.
    project_id, location:
        GCP coordinates.
    namespace:
        Namespace where the inference service was deployed.
    delete_cluster:
        When ``True`` the entire GKE cluster is also deleted after removing
        the Kubernetes resources.
    max_retries:
        Polling retries for GKE operations.
    """
    deployment_name = f"{cluster_name}-deployment"
    service_name    = f"{cluster_name}-svc"

    try:
        api_client = get_k8s_api(cluster_name, project_id, location)

        apps_v1  = k8s_client.AppsV1Api(api_client)
        core_v1  = k8s_client.CoreV1Api(api_client)

        # Delete Deployment
        try:
            apps_v1.delete_namespaced_deployment(name=deployment_name, namespace=namespace)
            logger.info("Deleted Deployment '%s'.", deployment_name)
        except k8s_client.ApiException as exc:
            if exc.status != 404:
                raise

        # Delete Service
        try:
            core_v1.delete_namespaced_service(name=service_name, namespace=namespace)
            logger.info("Deleted Service '%s'.", service_name)
        except k8s_client.ApiException as exc:
            if exc.status != 404:
                raise

        # Delete namespace (and wait)
        delete_namespace(api_client, namespace, wait=True)
        logger.info("Deleted namespace '%s'.", namespace)

    except Exception as exc:
        logger.warning(
            "Could not reach the cluster to delete Kubernetes resources: %s. "
            "Proceeding to cluster deletion if requested.",
            exc,
        )

    if delete_cluster:
        logger.info("Deleting GKE cluster '%s' …", cluster_name)
        _delete_gke_cluster(cluster_name, project_id, location, max_retries)
        logger.info("GKE cluster '%s' deleted.", cluster_name)


# ---------------------------------------------------------------------------
# Internal: GKE cluster lifecycle
# ---------------------------------------------------------------------------

def _create_inference_gke_cluster(
    cluster_name: str,
    project_id: str,
    location: str,
    max_retries: int,
) -> None:
    """
    Provision a minimal GKE Autopilot cluster for the inference service.

    Idempotent — if the cluster already exists and is RUNNING, this function
    returns immediately without making any changes.  If it exists but is still
    provisioning, we wait for it to become ready.

    Unlike the Ray cluster variant, this cluster does NOT enable the Ray
    operator add-on — it only needs Workload Identity for GCP API access.
    """
    from google.api_core.exceptions import AlreadyExists, NotFound

    gke_client = container_v1.ClusterManagerClient()
    cluster_path = f"projects/{project_id}/locations/{location}/clusters/{cluster_name}"

    # Check whether the cluster already exists
    try:
        cluster = gke_client.get_cluster(name=cluster_path)
        status = container_v1.Cluster.Status(cluster.status)

        if status == container_v1.Cluster.Status.RUNNING:
            logger.info(
                "GKE cluster '%s' already exists and is RUNNING — skipping creation.",
                cluster_name,
            )
            return

        if status == container_v1.Cluster.Status.PROVISIONING:
            logger.info(
                "GKE cluster '%s' is still PROVISIONING — waiting for it to become ready ...",
                cluster_name,
            )
            # Wait for any in-flight operation on this cluster to complete
            wait_for_all_operations_to_complete(cluster_name, project_id, location, max_retries)
            return

        # Any other status (DEGRADED, ERROR, etc.) — log and attempt recreation
        logger.warning(
            "GKE cluster '%s' exists with unexpected status '%s' — proceeding.",
            cluster_name, status.name,
        )
        return

    except NotFound:
        pass  # Cluster doesn't exist — create it below

    logger.info("GKE cluster '%s' not found — creating ...", cluster_name)

    cluster_body = {
        "name": cluster_name,
        "autopilot": {"enabled": True},
        "workload_identity_config": {
            "workload_pool": f"{project_id}.svc.id.goog"
        },
    }

    try:
        response = gke_client.create_cluster(
            parent=f"projects/{project_id}/locations/{location}",
            cluster=cluster_body,
        )
        operation_id = response.name.split("/")[-1]
        wait_for_operation(gke_client, operation_id, project_id, location, max_retries)
    except AlreadyExists:
        # Race condition: another process created it between our get and create
        logger.info(
            "GKE cluster '%s' was created concurrently — waiting for it to be ready ...",
            cluster_name,
        )
        wait_for_all_operations_to_complete(cluster_name, project_id, location, max_retries)


def _delete_gke_cluster(
    cluster_name: str,
    project_id: str,
    location: str,
    max_retries: int,
) -> None:
    wait_for_all_operations_to_complete(cluster_name, project_id, location, max_retries)
    gke_client = container_v1.ClusterManagerClient()
    op = gke_client.delete_cluster(
        name=f"projects/{project_id}/locations/{location}/clusters/{cluster_name}"
    )
    wait_for_operation(
        gke_client,
        op.name.split("/")[-1],
        project_id,
        location,
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Internal: Kubernetes resource builders
# ---------------------------------------------------------------------------

def _build_env_vars(env_vars: dict[str, str], port: int) -> list[k8s_client.V1EnvVar]:
    """Convert a plain dict to a list of V1EnvVar objects, always including PORT."""
    merged = {"PORT": str(port), **env_vars}
    return [k8s_client.V1EnvVar(name=k, value=v) for k, v in merged.items()]


def _create_pull_secret(
    api_client: k8s_client.ApiClient,
    secret_name: str,
    namespace: str,
    docker_config_json: str,
) -> None:
    """Create a kubernetes.io/dockerconfigjson secret for private registries."""
    import base64
    import json

    core_v1 = k8s_client.CoreV1Api(api_client)

    # docker_config_json may already be base64-encoded or raw JSON
    try:
        raw = json.loads(docker_config_json)
        encoded = base64.b64encode(json.dumps(raw).encode()).decode()
    except (json.JSONDecodeError, ValueError):
        # Assume already base64-encoded
        encoded = docker_config_json

    secret = k8s_client.V1Secret(
        metadata=k8s_client.V1ObjectMeta(name=secret_name, namespace=namespace),
        type="kubernetes.io/dockerconfigjson",
        data={".dockerconfigjson": encoded},
    )
    try:
        core_v1.read_namespaced_secret(name=secret_name, namespace=namespace)
        core_v1.replace_namespaced_secret(name=secret_name, namespace=namespace, body=secret)
    except k8s_client.ApiException as exc:
        if exc.status == 404:
            core_v1.create_namespaced_secret(namespace=namespace, body=secret)
        else:
            raise


def _create_deployment(
    api_client: k8s_client.ApiClient,
    deployment_name: str,
    namespace: str,
    docker_image: str,
    replicas: int,
    port: int,
    resources: dict,
    env_vars: dict[str, str],
    pull_secret_name: Optional[str],
) -> None:
    apps_v1 = k8s_client.AppsV1Api(api_client)

    resource_reqs = k8s_client.V1ResourceRequirements(
        requests=resources.get("requests", {}),
        limits=resources.get("limits", {}),
    )

    liveness_probe = k8s_client.V1Probe(
        http_get=k8s_client.V1HTTPGetAction(path="/health", port=port),
        initial_delay_seconds=20,
        period_seconds=15,
        failure_threshold=4,
    )
    readiness_probe = k8s_client.V1Probe(
        http_get=k8s_client.V1HTTPGetAction(path="/health", port=port),
        initial_delay_seconds=10,
        period_seconds=10,
        failure_threshold=3,
    )

    container = k8s_client.V1Container(
        name="inference-service",
        image=docker_image,
        ports=[k8s_client.V1ContainerPort(container_port=port, name="http")],
        env=_build_env_vars(env_vars, port),
        resources=resource_reqs,
        liveness_probe=liveness_probe,
        readiness_probe=readiness_probe,
        image_pull_policy="Always"
    )

    labels = {"app": deployment_name}

    pod_spec_kwargs: dict = {
        "containers": [container],
    }
    if pull_secret_name:
        pod_spec_kwargs["image_pull_secrets"] = [
            k8s_client.V1LocalObjectReference(name=pull_secret_name)
        ]

    pod_template = k8s_client.V1PodTemplateSpec(
        metadata=k8s_client.V1ObjectMeta(labels=labels),
        spec=k8s_client.V1PodSpec(**pod_spec_kwargs),
    )

    deployment = k8s_client.V1Deployment(
        metadata=k8s_client.V1ObjectMeta(name=deployment_name, namespace=namespace),
        spec=k8s_client.V1DeploymentSpec(
            replicas=replicas,
            selector=k8s_client.V1LabelSelector(match_labels=labels),
            template=pod_template,
            strategy=k8s_client.V1DeploymentStrategy(type="RollingUpdate"),
        ),
    )

    try:
        apps_v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)
        apps_v1.replace_namespaced_deployment(
            name=deployment_name, namespace=namespace, body=deployment
        )
        logger.info("Deployment '%s' already existed — replaced.", deployment_name)
    except k8s_client.ApiException as exc:
        if exc.status == 404:
            apps_v1.create_namespaced_deployment(namespace=namespace, body=deployment)
            logger.info("Deployment '%s' created.", deployment_name)
        else:
            raise


def _create_service(
    api_client: k8s_client.ApiClient,
    service_name: str,
    deployment_name: str,
    namespace: str,
    port: int,
) -> None:
    core_v1 = k8s_client.CoreV1Api(api_client)

    service = k8s_client.V1Service(
        metadata=k8s_client.V1ObjectMeta(name=service_name, namespace=namespace),
        spec=k8s_client.V1ServiceSpec(
            type="LoadBalancer",
            selector={"app": deployment_name},
            ports=[
                k8s_client.V1ServicePort(
                    port=port,
                    target_port=port,
                    protocol="TCP",
                    name="http",
                )
            ],
        ),
    )

    try:
        core_v1.read_namespaced_service(name=service_name, namespace=namespace)
        core_v1.replace_namespaced_service(
            name=service_name, namespace=namespace, body=service
        )
        logger.info("Service '%s' already existed — replaced.", service_name)
    except k8s_client.ApiException as exc:
        if exc.status == 404:
            core_v1.create_namespaced_service(namespace=namespace, body=service)
            logger.info("LoadBalancer Service '%s' created.", service_name)
        else:
            raise


# ---------------------------------------------------------------------------
# Internal: readiness polling
# ---------------------------------------------------------------------------

def _wait_for_deployment(
    api_client: k8s_client.ApiClient,
    deployment_name: str,
    namespace: str,
    timeout_s: int,
) -> None:
    """Block until the Deployment reports all desired replicas available."""
    apps_v1 = k8s_client.AppsV1Api(api_client)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        try:
            dep = apps_v1.read_namespaced_deployment(
                name=deployment_name, namespace=namespace
            )
            desired    = dep.spec.replicas or 1
            available  = dep.status.available_replicas or 0
            ready      = dep.status.ready_replicas or 0

            logger.debug(
                "Deployment '%s': desired=%d, available=%d, ready=%d",
                deployment_name, desired, available, ready,
            )

            if available >= desired and ready >= desired:
                logger.info("Deployment '%s' is fully available.", deployment_name)
                return

        except k8s_client.ApiException as exc:
            logger.warning("Error polling deployment '%s': %s", deployment_name, exc)

        time.sleep(10)

    raise TimeoutError(
        f"Deployment '{deployment_name}' in namespace '{namespace}' did not "
        f"become available within {timeout_s} seconds."
    )


def _wait_for_external_ip(
    api_client: k8s_client.ApiClient,
    service_name: str,
    namespace: str,
    timeout_s: int,
) -> str:
    """Block until the LoadBalancer Service has an external IP and return it."""
    core_v1  = k8s_client.CoreV1Api(api_client)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        try:
            svc = core_v1.read_namespaced_service(name=service_name, namespace=namespace)
            ingresses = (svc.status.load_balancer.ingress or []) if svc.status.load_balancer else []

            for ingress in ingresses:
                ip = ingress.ip or ingress.hostname
                if ip:
                    logger.info("Service '%s' external IP: %s", service_name, ip)
                    return ip

        except k8s_client.ApiException as exc:
            logger.warning("Error polling service '%s': %s", service_name, exc)

        logger.debug("Service '%s' — waiting for external IP …", service_name)
        time.sleep(15)

    raise TimeoutError(
        f"Service '{service_name}' in namespace '{namespace}' did not receive "
        f"an external IP within {timeout_s} seconds."
    )