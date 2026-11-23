"""In-cluster address of a Ray head Service (legacy per-job cluster path).

This used to wait for the head Service's LoadBalancer ingress IP and return
``http://<public ip>:8265`` -- a Ray dashboard, Ray Client port and GCS open to
the internet for every training job, and left open when a job's cluster was
not torn down (ray-trainer-5bd923f0 ran like that for 77 days). The only
clients are the API/worker pods in the same cluster, so the address is the
Service's cluster DNS name. Nothing here needs a public IP, and nothing here
is cloud-specific.
"""
from kubernetes import client
import time


def cluster_local_dashboard_url(ray_head_svc: str, namespace: str = "default", port: int = 8265) -> str:
    return f"http://{ray_head_svc}.{namespace}.svc.cluster.local:{port}"


def get_ray_dashboard_url(
    api_client: client.ApiClient,
    ray_head_svc: str,
    namespace: str = "default",
    max_retries: int = 60,
) -> str:
    """Wait until the head Service exists, then return its cluster-local URL."""
    core_api = client.CoreV1Api(api_client)

    retries = 0
    while retries < max_retries:
        try:
            core_api.read_namespaced_service(ray_head_svc, namespace)
            break
        except client.ApiException as exc:
            if exc.status != 404:
                raise
        retries += 1
        time.sleep(1)
    else:
        raise TimeoutError(
            f"Ray head service {ray_head_svc} did not appear within {max_retries} retries"
        )

    return cluster_local_dashboard_url(ray_head_svc, namespace)
