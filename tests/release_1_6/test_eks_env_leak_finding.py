"""NEW-RS-02: provisioning EKS must not write assumed-role credentials into the process environment.

Found while chasing a test-order failure in the 1.6 release suite (2026-10-31);
not in any source sheet, so it carries a release-suite ID.
``AwsEksProvider._assume_role`` (app/infra/providers/aws_eks.py) exports the
temporary STS credentials by assigning ``os.environ["AWS_ACCESS_KEY_ID"]`` and
friends. That makes them process-wide: every later boto3 client, subprocess and
thread in the API worker inherits them, and app/core/cloud_config.py infers
``CLOUD_PROVIDER='aws'`` from their presence. Fixed by PR #645 (merged into
develop-1.6 @ 96aa2eb1): the credentials now live on a provider-scoped boto3
Session and reach the aws CLI only through the child process's env. The strict
xfail below is now a plain test that keeps it fixed.
"""
from __future__ import annotations

import os

import pytest

_EKS_ENV = {
    "AWS_REGION": "us-east-1",
    "AWS_EKS_CLUSTER_ROLE_ARN": "arn:aws:iam::000000000000:role/test-eks-role",
    "AWS_SUBNET_IDS": "subnet-aaa,subnet-bbb",
    "AWS_SECURITY_GROUP_IDS": "sg-aaa",
    "AWS_CLUSTER_NAME": "release-suite-eks",
}
_LEAKABLE = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION")


class _FakeSts:
    def assume_role(self, **_kw):
        return {"Credentials": {"AccessKeyId": "test-akid",
                                "SecretAccessKey": "test-secret",
                                "SessionToken": "test-token"}}


class _Probe:
    returncode = 1
    stdout = ""
    stderr = "ResourceNotFoundException"


@pytest.fixture()
def provider(monkeypatch):
    import app.infra.providers.aws_eks as aws_eks

    for key, value in _EKS_ENV.items():
        monkeypatch.setenv(key, value)
    # Registered with monkeypatch (set, then delete) so whatever the product
    # writes is removed at teardown and cannot leak into later tests.
    for key in _LEAKABLE:
        monkeypatch.setenv(key, "placeholder")
        monkeypatch.delenv(key)
    calls = []
    monkeypatch.setattr(aws_eks.boto3, "client", lambda *a, **k: (calls.append("sts"), _FakeSts())[1])
    monkeypatch.setattr(aws_eks.subprocess, "run", lambda cmd, **kw: _Probe())
    monkeypatch.setattr(aws_eks, "run_command",
                        lambda cmd, step_name: {"step_name": step_name, "status": "SUCCESS", "message": "", "details": ""})
    return aws_eks.AwsEksProvider(), calls


def test_newrs02_provisioning_assumes_the_role(provider):
    """NEW-RS-02 (control): provision_cluster really goes through the STS assume-role step."""
    eks, calls = provider
    outcome = eks.provision_cluster()
    assert calls == ["sts"]
    assert outcome["status"] != "FAILED", outcome


# NEW-RS-02: converted from a strict xfail after PR #645 merged.
def test_newrs02_provision_cluster_leaves_the_process_environment_unchanged(provider):
    """NEW-RS-02: os.environ is identical before and after provision_cluster()."""
    eks, calls = provider
    before = dict(os.environ)
    eks.provision_cluster()
    assert calls == ["sts"], "the assume-role step under test did not run"
    after = dict(os.environ)
    assert {k: after[k] for k in after.keys() - before.keys()} == {}
    assert {k for k in before if before[k] != after.get(k)} == set()
