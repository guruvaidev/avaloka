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

from app.infra.providers.base import (
    ClusterProvider,
    child_env,
    run_command,
    scoped_subprocess_env,
)


class AwsEksProvider(ClusterProvider):
    name = "aws"

    def __init__(self):
        self.region = os.environ.get("AWS_REGION")
        self.role_arn = os.environ.get("AWS_EKS_CLUSTER_ROLE_ARN")
        self.subnet_ids = os.environ.get("AWS_SUBNET_IDS")
        self.security_group_ids = os.environ.get("AWS_SECURITY_GROUP_IDS")
        self.cluster_name = os.environ.get("AWS_CLUSTER_NAME", "test-eks-cluster")
        # The assumed-role session, once provision_cluster has assumed the role.
        self.session: boto3.Session | None = None

    def _assume_role(self) -> dict | None:
        """Assume the configured IAM role into a provider-scoped boto3 Session.
        Returns an error outcome dict on failure, or None on success.

        The temporary credentials live on ``self.session`` and reach the ``aws``
        CLI only through that child process's own env (``_cli_env``). They are
        never written to ``os.environ``: that made them process-wide, so every
        later request, tenant and boto3 client in the worker acted as this role
        and cloud_config inferred CLOUD_PROVIDER=aws from them (NEW-RS-02).
        """
        try:
            sts_client = boto3.client("sts", region_name=self.region)
            assumed = sts_client.assume_role(
                RoleArn=self.role_arn, RoleSessionName="EKSClusterProvisionSession"
            )
            creds = assumed["Credentials"]
            self.session = boto3.Session(
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
                region_name=self.region,
            )
            return None
        except ClientError as e:
            return {
                "step_name": "Assume IAM Role for aws",
                "status": "FAILED",
                "message": f"Failed to assume IAM role: {e}",
                "details": str(e),
            }

    def _cli_env(self) -> dict | None:
        """Env additions that make an ``aws`` CLI child act as the assumed role,
        or None before the role is assumed (the CLI then uses its ambient chain)."""
        if self.session is None:
            return None
        frozen = self.session.get_credentials().get_frozen_credentials()
        return {
            "AWS_ACCESS_KEY_ID": frozen.access_key,
            "AWS_SECRET_ACCESS_KEY": frozen.secret_key,
            "AWS_SESSION_TOKEN": frozen.token,
            "AWS_DEFAULT_REGION": self.session.region_name or self.region,
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
        with scoped_subprocess_env(self._cli_env()):
            return self._create_cluster()

    def _create_cluster(self) -> dict:
        check_cmd = ["aws", "eks", "describe-cluster", "--name", self.cluster_name, "--region", self.region]
        if subprocess.run(check_cmd, capture_output=True, text=True, env=child_env()).returncode == 0:
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
        if self.role_arn:
            # The cluster's creator is the assumed role, but kubectl and helm run
            # later with only the ambient credentials in their env. --role-arn
            # makes the kubeconfig's `aws eks get-token` assume the role itself,
            # from those ambient credentials, exactly as provision_cluster did.
            cmd += ["--role-arn", self.role_arn]
        with scoped_subprocess_env(self._cli_env()):
            return run_command(cmd, "Configure Kubectl")

    def teardown(self) -> dict:
        if not self.region:
            return {
                "step_name": "Cleanup aws Cluster",
                "status": "SKIPPED",
                "message": "AWS_REGION environment variable not set, skipping cleanup.",
                "details": "",
            }
        with scoped_subprocess_env(self._cli_env()):
            return self._delete_cluster()

    def _delete_cluster(self) -> dict:
        check_cmd = ["aws", "eks", "describe-cluster", "--name", self.cluster_name, "--region", self.region]
        if subprocess.run(check_cmd, capture_output=True, text=True, env=child_env()).returncode != 0:
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
