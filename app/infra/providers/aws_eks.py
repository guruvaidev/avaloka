# app/infra/providers/aws_eks.py
"""EKS (Amazon Elastic Kubernetes Service) cluster provider.

Refactored out of ``k8s_invoker`` so the EKS flow is reusable by the new
provider abstraction while keeping identical behavior.
"""
from __future__ import annotations

import os
import subprocess

import boto3
from botocore.exceptions import ClientError

from app.infra.providers.base import ClusterProvider, run_command


class AwsEksProvider(ClusterProvider):
    name = "aws"

    def __init__(self):
        self.region = os.environ.get("AWS_REGION")
        self.role_arn = os.environ.get("AWS_EKS_CLUSTER_ROLE_ARN")
        self.subnet_ids = os.environ.get("AWS_SUBNET_IDS")
        self.security_group_ids = os.environ.get("AWS_SECURITY_GROUP_IDS")
        self.cluster_name = os.environ.get("AWS_CLUSTER_NAME", "test-eks-cluster")

    def _assume_role(self) -> dict | None:
        """Assume the configured IAM role and export temporary creds. Returns an
        error outcome dict on failure, or None on success."""
        try:
            sts_client = boto3.client("sts", region_name=self.region)
            assumed = sts_client.assume_role(
                RoleArn=self.role_arn, RoleSessionName="EKSClusterProvisionSession"
            )
            creds = assumed["Credentials"]
            os.environ["AWS_ACCESS_KEY_ID"] = creds["AccessKeyId"]
            os.environ["AWS_SECRET_ACCESS_KEY"] = creds["SecretAccessKey"]
            os.environ["AWS_SESSION_TOKEN"] = creds["SessionToken"]
            os.environ["AWS_DEFAULT_REGION"] = self.region
            return None
        except ClientError as e:
            return {
                "step_name": "Assume IAM Role for aws",
                "status": "FAILED",
                "message": f"Failed to assume IAM role: {e}",
                "details": str(e),
            }

    def provision_cluster(self) -> dict:
        if not all([self.region, self.role_arn, self.subnet_ids, self.security_group_ids]):
            return {
                "step_name": "Provision aws Cluster",
                "status": "FAILED",
                "message": "Missing AWS_REGION, AWS_EKS_CLUSTER_ROLE_ARN, AWS_SUBNET_IDS, or AWS_SECURITY_GROUP_IDS environment variables.",
                "details": "",
            }
        role_err = self._assume_role()
        if role_err:
            return role_err

        check_cmd = ["aws", "eks", "describe-cluster", "--name", self.cluster_name, "--region", self.region]
        if subprocess.run(check_cmd, capture_output=True, text=True).returncode == 0:
            return {
                "step_name": "Provision aws Cluster",
                "status": "SKIPPED",
                "message": f"EKS cluster {self.cluster_name} already exists.",
                "details": "",
            }
        create_cmd = [
            "aws", "eks", "create-cluster",
            "--name", self.cluster_name,
            "--region", self.region,
            "--version", "1.28",
            "--role-arn", self.role_arn,
            "--resources-vpc-config",
            f"subnetIds={self.subnet_ids},securityGroupIds={self.security_group_ids}",
        ]
        create_outcome = run_command(create_cmd, "Create aws Cluster")
        if create_outcome["status"] == "FAILED":
            return create_outcome
        wait_cmd = ["aws", "eks", "wait", "cluster-active", "--name", self.cluster_name, "--region", self.region]
        return run_command(wait_cmd, "Wait for aws Cluster Active")

    def configure_kubectl(self) -> dict:
        cmd = ["aws", "eks", "update-kubeconfig", "--name", self.cluster_name, "--region", self.region]
        return run_command(cmd, "Configure Kubectl")

    def teardown(self) -> dict:
        if not self.region:
            return {
                "step_name": "Cleanup aws Cluster",
                "status": "SKIPPED",
                "message": "AWS_REGION environment variable not set, skipping cleanup.",
                "details": "",
            }
        check_cmd = ["aws", "eks", "describe-cluster", "--name", self.cluster_name, "--region", self.region]
        if subprocess.run(check_cmd, capture_output=True, text=True).returncode != 0:
            return {
                "step_name": "Cleanup aws Cluster",
                "status": "SKIPPED",
                "message": f"EKS cluster {self.cluster_name} does not exist, skipping deletion.",
                "details": "",
            }
        delete_outcome = run_command(
            ["aws", "eks", "delete-cluster", "--name", self.cluster_name, "--region", self.region],
            "Delete aws Cluster",
        )
        if delete_outcome["status"] == "FAILED":
            return delete_outcome
        wait_cmd = ["aws", "eks", "wait", "cluster-deleted", "--name", self.cluster_name, "--region", self.region]
        return run_command(wait_cmd, "Wait for aws Cluster Deleted")
