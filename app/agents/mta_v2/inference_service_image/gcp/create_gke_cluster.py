from google.cloud import container_v1
from .utils import wait_for_operation


def create_gke_cluster(
    cluster_name: str,
    project_id: str,
    location: str,
    max_retries: int = 600,
):
    client = container_v1.ClusterManagerClient()

    cluster = {
        "name": cluster_name,
        "autopilot": {"enabled": True},
        "addons_config": {
            "ray_operator_config": {
                "enabled": True,
                "ray_cluster_monitoring_config": {"enabled": True},
                "ray_cluster_logging_config": {"enabled": True},
            }
        },
        "workload_identity_config": {"workload_pool": f"{project_id}.svc.id.goog"},
    }

    response = client.create_cluster(
        parent=f"projects/{project_id}/locations/{location}",
        cluster=cluster,
    )

    operation_id = response.name.split("/")[-1]

    wait_for_operation(client, operation_id, project_id, location, max_retries)
