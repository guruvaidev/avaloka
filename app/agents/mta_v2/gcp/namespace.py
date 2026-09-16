import time
from kubernetes import client


def create_namespace(api_client: client.ApiClient, namespace: str):
    """Creates a namespace if it doesn't already exist."""
    core_v1 = client.CoreV1Api(api_client)

    try:
        core_v1.read_namespace(name=namespace)
    except client.ApiException as e:
        if e.status == 404:
            metadata = client.V1ObjectMeta(name=namespace)
            body = client.V1Namespace(metadata=metadata)
            core_v1.create_namespace(body=body)
        else:
            raise e


def delete_namespace(api_client: client.ApiClient, namespace: str, wait: bool = True):
    """Deletes a namespace and optionally waits for it to disappear."""
    core_v1 = client.CoreV1Api(api_client)

    try:
        core_v1.delete_namespace(name=namespace)

        if wait:
            while True:
                try:
                    core_v1.read_namespace(name=namespace)
                    time.sleep(5)
                except client.ApiException as e:
                    if e.status == 404:
                        break
                    raise e
    except client.ApiException as e:
        if e.status == 404:
            pass
        else:
            raise e
