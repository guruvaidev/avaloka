from kubernetes import client
import time


def get_ray_dashboard_url(
    api_client: client.ApiClient,
    ray_head_svc: str,
    namespace: str = "default",
    max_retries: int = 60,
) -> str:
    core_api = client.CoreV1Api(api_client)

    retries = 0
    while retries < max_retries:
        svc = core_api.read_namespaced_service(ray_head_svc, namespace)
        ingress = svc.status.load_balancer.ingress
        if ingress and ingress[0].ip:
            ip = ingress[0].ip
            break
        retries += 1
        time.sleep(1)
    else:
        raise TimeoutError(
            f"Ray head service {ray_head_svc} did not become available within {max_retries} retries"
        )

    return f"http://{ip}:8265"
