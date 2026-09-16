import unittest
import subprocess
import os
import time
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

class TestEksDeployment(unittest.TestCase):

    def setUp(self):
        self.aws_region = os.environ.get("AWS_REGION")
        self.eks_cluster_role_arn = os.environ.get("AWS_EKS_CLUSTER_ROLE_ARN")
        self.aws_subnet_ids = os.environ.get("AWS_SUBNET_IDS")
        self.aws_security_group_ids = os.environ.get("AWS_SECURITY_GROUP_IDS")
        self.cluster_name = "test-eks-cluster"

        if not all([self.aws_region, self.eks_cluster_role_arn, self.aws_subnet_ids, self.aws_security_group_ids]):
            self.skipTest("AWS_REGION, AWS_EKS_CLUSTER_ROLE_ARN, AWS_SUBNET_IDS, or AWS_SECURITY_GROUP_IDS environment variables not set")

        # Assume IAM role and set temporary credentials
        try:
            sts_client = boto3.client('sts', region_name=self.aws_region)
            assumed_role_object = sts_client.assume_role(
                RoleArn=self.eks_cluster_role_arn,
                RoleSessionName="EKSClusterTestSession"
            )
            credentials = assumed_role_object['Credentials']
            os.environ['AWS_ACCESS_KEY_ID'] = credentials['AccessKeyId']
            os.environ['AWS_SECRET_ACCESS_KEY'] = credentials['SecretAccessKey']
            os.environ['AWS_SESSION_TOKEN'] = credentials['SessionToken']
        except ClientError as e:
            self.skipTest(f"Failed to assume IAM role: {e}")

        # Configure AWS CLI default region
        os.environ["AWS_DEFAULT_REGION"] = self.aws_region

        # Check if the EKS cluster already exists
        try:
            subprocess.run([
                "aws", "eks", "describe-cluster",
                "--name", self.cluster_name,
                "--region", self.aws_region
            ], check=True, capture_output=True)
            print(f"EKS cluster {self.cluster_name} already exists.")
        except subprocess.CalledProcessError:
            print(f"Creating EKS cluster {self.cluster_name}...")
            subprocess.run([
                "aws", "eks", "create-cluster",
                "--name", self.cluster_name,
                "--region", self.aws_region,
                "--version", "1.28", # Specify a Kubernetes version
                "--role-arn", self.eks_cluster_role_arn,
                "--resources-vpc-config", f"subnetIds={self.aws_subnet_ids},securityGroupIds={self.aws_security_group_ids}"
            ], check=True)
            print(f"Waiting for EKS cluster {self.cluster_name} to become active...")
            subprocess.run([
                "aws", "eks", "wait", "cluster-active",
                "--name", self.cluster_name,
                "--region", self.aws_region
            ], check=True)

        # Update kubeconfig for the EKS cluster
        subprocess.run([
            "aws", "eks", "update-kubeconfig",
            "--name", self.cluster_name,
            "--region", self.aws_region
        ], check=True)

        # Wait for kubectl to be able to connect to the cluster
        for _ in range(60):
            try:
                subprocess.run(["kubectl", "get", "nodes"], check=True, capture_output=True)
                print("kubectl connected to EKS cluster.")
                break
            except subprocess.CalledProcessError:
                print("Waiting for kubectl to connect to EKS cluster...")
                time.sleep(5)
        else:
            raise Exception("kubectl could not connect to the EKS cluster after multiple retries")

    def tearDown(self):
        # Check if the EKS cluster exists before trying to delete it
        try:
            subprocess.run([
                "aws", "eks", "describe-cluster",
                "--name", self.cluster_name,
                "--region", self.aws_region
            ], check=True, capture_output=True)
            print(f"Deleting EKS cluster {self.cluster_name}...")
            subprocess.run([
                "aws", "eks", "delete-cluster",
                "--name", self.cluster_name,
                "--region", self.aws_region
            ], check=True)
            print(f"Waiting for EKS cluster {self.cluster_name} to be deleted...")
            subprocess.run([
                "aws", "eks", "wait", "cluster-deleted",
                "--name", self.cluster_name,
                "--region", self.aws_region
            ], check=True)
        except subprocess.CalledProcessError:
            print(f"EKS cluster {self.cluster_name} does not exist, skipping deletion.")

    def test_deploy_to_eks(self):
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
