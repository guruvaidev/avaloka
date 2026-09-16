import time
import logging
from google.cloud import container_v1

def wait_for_all_operations_to_complete(
    cluster_name,
    project_id: str,
    location: str,
    max_retries: int = 600,
):
    client = container_v1.ClusterManagerClient()
    parent = f"projects/{project_id}/locations/{location}"

    retries = 0
    while retries < max_retries:
        try:
            response = client.list_operations(parent=parent)

            live_ops = []
            for op in response.operations:
                if (
                    cluster_name in op.target_link
                    and op.status != container_v1.Operation.Status.DONE
                ):
                    live_ops.append(
                        {
                            "id": op.name.split("/")[-1],
                            "type": op.operation_type.name,
                            "status": op.status.name,
                            "start_time": op.start_time,
                            "detail": op.detail,
                        }
                    )
            if not live_ops:
                break
            retries += 1
            time.sleep(1)
        except Exception as e:
            raise
    else:
        raise TimeoutError(
            f"Cluster {cluster_name} did not complete within {max_retries} retries"
        )


def wait_for_operation(
    client: container_v1.ClusterManagerClient,
    operation_id: str,
    project_id: str,
    location: str,
    max_retries: int = 600,
):
    retries = 0
    while retries < max_retries:
        try:
            op = client.get_operation(
                name=f"projects/{project_id}/locations/{location}/operations/{operation_id}"
            )

            if op.status == container_v1.Operation.Status.DONE:
                if op.error.message:
                    raise RuntimeError(
                        f"Operation {operation_id} failed: {op.error.message}"
                    )
                return

            retries += 1
            time.sleep(1)
        except Exception as e:
            raise
    else:
        raise TimeoutError(
            f"Operation {operation_id} did not complete within {max_retries} retries"
        )
