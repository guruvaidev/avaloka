import logging
from google.cloud import container_v1
from google.cloud import iam_admin_v1
from google.cloud import resourcemanager_v3
from google.iam.v1 import policy_pb2
from kubernetes import client


def enable_workload_identity(
    cluster_name: str,
    location: str,
    project_id: str,
):
    try:
        c_client = container_v1.ClusterManagerClient()
        cluster_path = (
            f"projects/{project_id}/locations/{location}/clusters/{cluster_name}"
        )
        update = {
            "desired_workload_identity_config": {
                "workload_pool": f"{project_id}.svc.id.goog"
            }
        }
        c_client.update_cluster(name=cluster_path, update=update)
    except Exception as e:
        raise


def create_and_bind_iam(
    gsa_name: str,
    ksa_name: str,
    project_id: str,
    namespace: str = "default",
):
    try:
        iam_client = iam_admin_v1.IAMClient()
        gsa_email = f"{gsa_name}@{project_id}.iam.gserviceaccount.com"
        try:
            iam_client.create_service_account(
                request={
                    "name": f"projects/{project_id}",
                    "account_id": gsa_name,
                    "service_account": {"display_name": "Ray GCS Accessor"},
                }
            )
        except Exception:
            raise

        member = f"serviceAccount:{project_id}.svc.id.goog[{namespace}/{ksa_name}]"
        resource = f"projects/{project_id}/serviceAccounts/{gsa_email}"
        policy = iam_client.get_iam_policy(request={"resource": resource})
        policy.bindings.append(
            policy_pb2.Binding(role="roles/iam.workloadIdentityUser", members=[member])
        )
        iam_client.set_iam_policy(request={"resource": resource, "policy": policy})

        rm_client = resourcemanager_v3.ProjectsClient()
        project_path = f"projects/{project_id}"
        policy = rm_client.get_iam_policy(request={"resource": project_path})
        policy.bindings.append(
            policy_pb2.Binding(
                role="roles/storage.objectAdmin",
                members=[f"serviceAccount:{gsa_email}"],
            )
        )
        rm_client.set_iam_policy(request={"resource": project_path, "policy": policy})
    except Exception as e:
        raise


def annotate_kubernetes_sa(
    api_client: client.ApiClient,
    gsa_name: str,
    ksa_name: str,
    project_id: str,
    namespace: str = "default",
):
    try:
        v1 = client.CoreV1Api(api_client)

        gsa_email = f"{gsa_name}@{project_id}.iam.gserviceaccount.com"
        body = {
            "metadata": {"annotations": {"iam.gke.io/gcp-service-account": gsa_email}}
        }

        v1.patch_namespaced_service_account(ksa_name, namespace, body)
    except Exception as e:
        raise
