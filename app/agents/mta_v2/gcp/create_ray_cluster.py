import time
import logging
from kubernetes import client

from app.infra.ray_auth import RAYJOB_AUTH_KEY, require_token

logger = logging.getLogger(__name__)

# Autoscaling defaults
MIN_WORKER_REPLICAS = 1       # Scale down to zero when idle
NUM_WORKER_REPLICAS = 4
MAX_WORKER_REPLICAS = 25     # Hard cap on worker pods

# Per-worker minimum resource requirements
DEFAULT_WORKER_RESOURCES: dict = {
    "requests": {"cpu": "4", "memory": "16Gi"},
    "limits":   {"cpu": "4", "memory": "16Gi"},
}

DEFAULT_HEAD_RESOURCES: dict = {
    "requests": {"cpu": "4", "memory": "8Gi"},
    "limits":   {"cpu": "4", "memory": "8Gi"},
}


def create_ray_cluster(
    api_client: client.ApiClient,
    cluster_name: str,
    # Autoscaling: initial replicas default to 0 (scale-from-zero)
    worker_replicas: int = NUM_WORKER_REPLICAS,
    min_worker_replicas: int = MIN_WORKER_REPLICAS,
    max_worker_replicas: int = MAX_WORKER_REPLICAS,
    head_resources: dict = DEFAULT_HEAD_RESOURCES,
    worker_resources: dict = DEFAULT_WORKER_RESOURCES,
    docker_image: str = "rayproject/ray:2.56.0-py311",
    namespace: str = "default",
    use_sa: bool = False,
    service_account_name: str | None = None,
):
    custom = client.CustomObjectsApi(api_client)

    # Validate worker resource minimums
    _assert_worker_resources(worker_resources)

    # Ray token authentication. This head is published through a LoadBalancer
    # (the app reaches it from another cluster), so the token is what keeps the
    # dashboard and Jobs API closed: the app's own RAY_AUTH_TOKEN goes into a
    # Secret in the run's namespace, every Ray container reads it by
    # secretKeyRef, and authOptions points the KubeRay operator at it.
    auth_secret = _ensure_auth_secret(api_client, cluster_name, namespace)
    auth_env = _auth_env(auth_secret) if auth_secret else []

    head_spec = {
        # ClusterIP: the head is reached only from inside the cluster (API/worker pods).
        "serviceType": "ClusterIP",
        "rayStartParams": {
            "dashboard-host": "0.0.0.0",
            # Head node contributes no task CPUs — workers handle all workloads
            "num-cpus": "0",
        },
        "template": {
            "spec": {
                "containers": [
                    {
                        "name": "ray-head",
                        "image": docker_image,
                        "ports": [
                            {"containerPort": 6379, "name": "redis"},
                            {"containerPort": 8265, "name": "dashboard"},
                            {"containerPort": 10001, "name": "client"},
                        ],
                        "resources": head_resources,
                        "env": list(auth_env),
                    }
                ]
            }
        },
    }

    if use_sa and service_account_name:
        head_spec["template"]["spec"]["serviceAccountName"] = service_account_name

    worker_spec = {
        "groupName": "default-worker-group",
        "replicas": worker_replicas,
        "minReplicas": min_worker_replicas,   # 0 = scale-to-zero when idle
        "maxReplicas": max_worker_replicas,   # hard cap: 25 pods
        "rayStartParams": {},
        "template": {
            "spec": {
                "containers": [
                    {
                        "name": "ray-worker",
                        "image": docker_image,
                        "resources": worker_resources,
                        "env": list(auth_env),
                    }
                ],
            }
        },
    }

    if use_sa and service_account_name:
        worker_spec["template"]["spec"]["serviceAccountName"] = service_account_name

    ray_cluster_yaml = {
        "apiVersion": "ray.io/v1",
        "kind": "RayCluster",
        "metadata": {"name": cluster_name},
        "spec": {
            "rayVersion": "2.56.0",
            # Enable the Ray autoscaler sidecar on the head pod
            "enableInTreeAutoscaling": True,
            "autoscalerOptions": {
                "upscalingMode": "Default",
                # How long a worker must be idle before scaling down
                "idleTimeoutSeconds": 600,
                # The autoscaler talks to GCS, so it needs the token too.
                "env": list(auth_env),
            },
            "headGroupSpec": head_spec,
            "workerGroupSpecs": [worker_spec],
        },
    }
    if auth_secret:
        ray_cluster_yaml["spec"]["authOptions"] = {"mode": "token", "secretName": auth_secret}

    logger.info(
        "Creating RayCluster '%s' in namespace '%s' "
        "(minReplicas=%d, maxReplicas=%d, worker cpu=%s, worker memory=%s)",
        cluster_name,
        namespace,
        min_worker_replicas,
        max_worker_replicas,
        worker_resources["requests"]["cpu"],
        worker_resources["requests"]["memory"],
    )

    try:
        custom.create_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=namespace,
            plural="rayclusters",
            body=ray_cluster_yaml,
        )
    except Exception:
        raise

    logger.info("RayCluster '%s' submitted — waiting for Ready state …", cluster_name)

    while True:
        try:
            resource = custom.get_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace=namespace,
                plural="rayclusters",
                name=cluster_name,
            )

            status = resource.get("status", {})
            state = status.get("state")

            if state and state.lower() == "ready":
                logger.info("RayCluster '%s' is Ready.", cluster_name)
                break
            else:
                logger.debug("RayCluster '%s' state=%s — retrying in 10 s", cluster_name, state)
                time.sleep(10)

        except Exception as e:
            logger.warning("Error polling RayCluster '%s': %s — retrying in 5 s", cluster_name, e)
            time.sleep(5)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _auth_env(secret_name: str) -> list:
    return [
        {"name": "RAY_AUTH_MODE", "value": "token"},
        {"name": "RAY_AUTH_TOKEN", "valueFrom": {"secretKeyRef": {"name": secret_name, "key": RAYJOB_AUTH_KEY}}},
    ]


def _ensure_auth_secret(api_client: client.ApiClient, cluster_name: str, namespace: str) -> str | None:
    """Put the app's Ray token in ``<cluster>-ray-auth`` (key auth_token); its name, or None.

    None when the app runs without RAY_AUTH_MODE=token. Raises when auth is on
    and the app has no token: a cluster published through a LoadBalancer must
    not come up open.
    """
    token = require_token()
    if not token:
        return None
    name = f"{cluster_name}-ray-auth"[:253]
    body = client.V1Secret(
        metadata=client.V1ObjectMeta(name=name, labels={"app.kubernetes.io/part-of": "avaloka"}),
        type="Opaque",
        string_data={RAYJOB_AUTH_KEY: token},
    )
    core = client.CoreV1Api(api_client)
    try:
        core.create_namespaced_secret(namespace=namespace, body=body)
    except client.exceptions.ApiException as exc:
        if exc.status != 409:
            raise RuntimeError(f"could not create the Ray auth Secret {name}: HTTP {exc.status}") from None
        core.replace_namespaced_secret(name=name, namespace=namespace, body=body)
    return name

def _parse_memory_gi(value: str) -> float:
    """Return memory value in GiB from strings like '16Gi', '8192Mi', '17179869184'."""
    value = value.strip()
    if value.endswith("Gi"):
        return float(value[:-2])
    if value.endswith("Mi"):
        return float(value[:-2]) / 1024
    if value.endswith("G"):
        return float(value[:-1]) * (1000 ** 3) / (1024 ** 3)
    # bare bytes
    return float(value) / (1024 ** 3)


def _assert_worker_resources(resources: dict) -> None:
    """Raise ValueError if worker resources fall below the required minimums (4 CPU / 16 GiB)."""
    MIN_CPU = 4
    MIN_MEMORY_GI = 16

    req = resources.get("requests", {})

    cpu_str = str(req.get("cpu", "0"))
    # Support millicore notation (e.g. "4000m")
    if cpu_str.endswith("m"):
        cpu = float(cpu_str[:-1]) / 1000
    else:
        cpu = float(cpu_str)

    memory_gi = _parse_memory_gi(str(req.get("memory", "0")))

    if cpu < MIN_CPU:
        raise ValueError(
            f"Worker requests.cpu={cpu_str} is below the required minimum of {MIN_CPU} CPUs."
        )
    if memory_gi < MIN_MEMORY_GI:
        raise ValueError(
            f"Worker requests.memory={req.get('memory')} is below the required minimum of {MIN_MEMORY_GI}Gi."
        )
