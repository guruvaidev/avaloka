import time
from kubernetes import client
from google.cloud import container_v1
from .utils import wait_for_operation, wait_for_all_operations_to_complete


def delete_ray_cluster(
    api_client: client.ApiClient,
    cluster_name: str,
    namespace: str = "default",
    wait: bool = True,
):
    custom = client.CustomObjectsApi(api_client)

    try:
        custom.delete_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=namespace,
            plural="rayclusters",
            name=cluster_name,
            body=client.V1DeleteOptions(),
        )
        if wait:
            while True:
                try:
                    custom.get_namespaced_custom_object(
                        group="ray.io",
                        version="v1",
                        namespace=namespace,
                        plural="rayclusters",
                        name=cluster_name,
                    )
                    time.sleep(5)
                except client.ApiException as e:
                    if e.status == 404:
                        return True
                    raise e

    except client.ApiException as e:
        if e.status == 404:
            return True
        else:
            return False


def delete_gke_cluster(
    cluster_name: str,
    project_id: str,
    location: str,
    max_retries: int = 600,
):
    wait_for_all_operations_to_complete(cluster_name, project_id, location, max_retries)
    client = container_v1.ClusterManagerClient()
    op = client.delete_cluster(
        name=f"projects/{project_id}/locations/{location}/clusters/{cluster_name}"
    )
    wait_for_operation(
        client, op.name.split("/")[-1], project_id, location, max_retries=max_retries
    )
