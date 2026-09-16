
import unittest
import subprocess
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

@pytest.mark.skip(reason="Skipping this test for now")
class TestK8sDeployment(unittest.TestCase):

    def setUp(self):
        # Set the project ID and cluster name
        self.project_id = os.environ.get("GCP_PROJECT_ID")
        if not self.project_id:
            self.skipTest("GCP_PROJECT_ID environment variable not set")
        self.cluster_name = "test-cluster"
        self.zone = os.environ.get("GCP_ZONE", "us-central1-c") # Default to us-central1-c if not set

        # Check if the cluster already exists
        result = subprocess.run([
            "gcloud", "container", "clusters", "describe", self.cluster_name,
            "--project", self.project_id,
            "--zone", self.zone
        ], capture_output=True, text=True)

        if result.returncode != 0:
            # Create a GKE cluster if it doesn't exist
            subprocess.run([
                "gcloud", "container", "clusters", "create", self.cluster_name,
                "--project", self.project_id,
                "--zone", self.zone,
                "--num-nodes", "1"
            ], check=True)

        # Get credentials for the cluster
        subprocess.run([
            "gcloud", "container", "clusters", "get-credentials", self.cluster_name,
            "--project", self.project_id,
            "--zone", self.zone
        ], check=True)

        # Wait for kubectl to be able to connect to the cluster
        import time
        for _ in range(60):
            try:
                subprocess.run(["kubectl", "get", "nodes"], check=True, capture_output=True)
                break
            except subprocess.CalledProcessError:
                time.sleep(5)
        else:
            raise Exception("kubectl could not connect to the cluster after multiple retries")

    def tearDown(self):
        # Check if the cluster exists before trying to delete it
        result = subprocess.run([
            "gcloud", "container", "clusters", "describe", self.cluster_name,
            "--project", self.project_id,
            "--zone", self.zone
        ], capture_output=True, text=True)

        if result.returncode == 0:
            # Delete the GKE cluster
            subprocess.run([
                "gcloud", "container", "clusters", "delete", self.cluster_name,
                "--project", self.project_id,
                "--zone", self.zone,
                "--quiet"
            ], check=True)

    def test_deploy_to_gke(self):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'app', 'infra', 'config')
        # Apply the kubernetes manifests
        subprocess.run(["kubectl", "apply", "-f", config_path], check=True)

        # Wait for the deployment to be ready
        subprocess.run(["kubectl", "wait", "--for=condition=available", "deployment/infra-agent", "--timeout=300s"], check=True)

        # Check that the deployment is available
        result = subprocess.run(["kubectl", "get", "deployment", "infra-agent"], capture_output=True, text=True)
        self.assertIn("1/1", result.stdout)

if __name__ == '__main__':
    unittest.main()
