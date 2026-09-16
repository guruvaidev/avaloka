from google.cloud import container_v1
from kubernetes import client as k8s_client
import google.auth
import google.auth.transport.requests
import base64
import tempfile

def get_k8s_api(
    cluster_name: str,
    project_id: str,
    location: str,
) -> k8s_client.ApiClient:
    cluster_client = container_v1.ClusterManagerClient()
    name = f"projects/{project_id}/locations/{location}/clusters/{cluster_name}"
    cluster = cluster_client.get_cluster(name=name)

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)

    ca_cert = base64.b64decode(cluster.master_auth.cluster_ca_certificate)
    with tempfile.NamedTemporaryFile(delete=False) as ca_file:
        ca_file.write(ca_cert)
        ca_path = ca_file.name

    configuration = k8s_client.Configuration()
    configuration.host = f"https://{cluster.endpoint}"
    configuration.ssl_ca_cert = ca_path
    configuration.verify_ssl = True
    configuration.api_key = {"authorization": f"Bearer {creds.token}"}

    return k8s_client.ApiClient(configuration)
