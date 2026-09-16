from kubernetes import client
import time


def get_ray_head_svc(
    api_client: client.ApiClient,
    cluster_name: str,
    namespace: str = "default",
    max_retries: int = 60,
) -> str:
    v1 = client.CoreV1Api(api_client)
    retries = 0
    while retries < max_retries:
        svc_list = v1.list_namespaced_service(
            namespace=namespace,
            label_selector=f"ray.io/cluster={cluster_name},ray.io/node-type=head",
        )
        if not svc_list.items:
            retries += 1
            time.sleep(1)
        else:
            return svc_list.items[0].metadata.name
    else:
        raise TimeoutError(
            f"Ray head service for cluster {cluster_name} did not become available within {max_retries} retries"
        )
