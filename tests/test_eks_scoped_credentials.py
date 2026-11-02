# tests/test_eks_scoped_credentials.py
"""NEW-RS-02: EKS provisioning keeps the assumed-role credentials out of os.environ.

Both EKS paths (``AwsEksProvider`` behind cloud_provisioner, and the legacy
``k8s_invoker`` the infra agent calls) used to assign the STS credentials to
``os.environ``, which made them process-wide: every later request and tenant
acted as that role and cloud_config inferred CLOUD_PROVIDER=aws from them.

STS is a real botocore client answered by ``Stubber``; no ``aws`` CLI runs
(subprocess.run is replaced) and no request leaves the process. The
credentials are synthetic and assembled from parts so no literal key shape
appears in the repo.
"""
from __future__ import annotations

import datetime
import os
import threading

import boto3
import pytest
from botocore.awsrequest import AWSResponse
from botocore.stub import Stubber

REGION = "us-east-1"
ROLE_ARN = "arn:aws:iam::000000000000:role/" + "example-eks-role"
ASSUMED = {
    "AccessKeyId": "ASIA" + "EXAMPLE" + "SCOPED0001",
    "SecretAccessKey": "example" + "-assumed-secret-" + "not-real",
    "SessionToken": "example" + "-assumed-session-token",
}
AMBIENT_KEY = "AKIA" + "EXAMPLE" + "AMBIENT001"
_EKS_ENV = {
    "AWS_REGION": REGION,
    "AWS_EKS_CLUSTER_ROLE_ARN": ROLE_ARN,
    "AWS_SUBNET_IDS": "subnet-aaa,subnet-bbb",
    "AWS_SECURITY_GROUP_IDS": "sg-aaa",
    "AWS_CLUSTER_NAME": "scoped-creds-eks",
}
_LEAKABLE = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION")


def _stubbed_sts():
    """A real STS client whose single assume_role call is answered locally."""
    sts = boto3.session.Session(
        aws_access_key_id=AMBIENT_KEY,
        aws_secret_access_key="example-ambient-secret",
        region_name=REGION,
    ).client("sts")
    stubber = Stubber(sts)
    stubber.add_response(
        "assume_role",
        {
            "Credentials": {
                **ASSUMED,
                "Expiration": datetime.datetime(2099, 1, 1, tzinfo=datetime.timezone.utc),
            }
        },
        {"RoleArn": ROLE_ARN, "RoleSessionName": "EKSClusterProvisionSession"},
    )
    stubber.activate()
    return sts, stubber


class _Proc:
    """describe-cluster says "not found"; create/wait succeed."""

    def __init__(self, cmd):
        self.returncode = 1 if "describe-cluster" in cmd else 0
        self.stdout = ""
        self.stderr = "ResourceNotFoundException" if self.returncode else ""


class _Body:
    """The raw-stream shape botocore reads a response body from."""

    def __init__(self, data: bytes):
        self._data = data

    def stream(self, **_kw):
        yield self._data


@pytest.fixture()
def aws_env(monkeypatch):
    for key, value in _EKS_ENV.items():
        monkeypatch.setenv(key, value)
    # Registered with monkeypatch (set, then delete): absent during the test,
    # and whatever a leaking build writes is removed at teardown.
    for key in _LEAKABLE:
        monkeypatch.setenv(key, "placeholder")
        monkeypatch.delenv(key)


@pytest.fixture()
def spawned(monkeypatch):
    """Replace subprocess.run; record each child's argv, the env it was given,
    and whether os.environ itself held the assumed key at that moment."""
    import subprocess

    calls = []

    def fake_run(cmd, **kw):
        calls.append({
            "cmd": list(cmd),
            "env": kw.get("env"),
            "global_key": os.environ.get("AWS_ACCESS_KEY_ID"),
        })
        return _Proc(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


@pytest.fixture()
def sts(monkeypatch):
    client, stubber = _stubbed_sts()
    made = []

    def fake_client(service, **kw):
        made.append(service)
        assert service == "sts", f"unexpected boto3.client({service!r})"
        return client

    monkeypatch.setattr(boto3, "client", fake_client)
    yield made
    stubber.assert_no_pending_responses()


def _assert_child_got_the_role(call):
    env = call["env"]
    assert env is not None, f"{call['cmd'][:3]} inherited the process env instead of its own"
    assert env["AWS_ACCESS_KEY_ID"] == ASSUMED["AccessKeyId"]
    assert env["AWS_SECRET_ACCESS_KEY"] == ASSUMED["SecretAccessKey"]
    assert env["AWS_SESSION_TOKEN"] == ASSUMED["SessionToken"]
    assert env["AWS_DEFAULT_REGION"] == REGION
    # A copy of the process env, not a bare dict: PATH etc. still reach the CLI.
    assert env["AWS_CLUSTER_NAME"] == _EKS_ENV["AWS_CLUSTER_NAME"]
    assert call["global_key"] is None, "the assumed key was in os.environ while the child ran"


# ---------------- AwsEksProvider ---------------- #

def test_provider_provision_leaves_os_environ_byte_identical(aws_env, sts, spawned):
    from app.infra.providers.aws_eks import AwsEksProvider

    before = dict(os.environ)
    outcome = AwsEksProvider().provision_cluster()
    after = dict(os.environ)

    assert sts == ["sts"], "the assume-role step under test did not run"
    assert outcome["status"] == "SUCCESS", outcome
    assert after == before
    assert not any(v in ASSUMED.values() for v in os.environ.values())


def test_provider_session_clients_sign_with_the_assumed_role(aws_env, sts, spawned):
    from app.infra.providers.aws_eks import AwsEksProvider

    provider = AwsEksProvider()
    provider.provision_cluster()
    assert isinstance(getattr(provider, "session", None), boto3.Session), "no assumed-role session"
    frozen = provider.session.get_credentials().get_frozen_credentials()
    assert (frozen.access_key, frozen.secret_key, frozen.token) == (
        ASSUMED["AccessKeyId"], ASSUMED["SecretAccessKey"], ASSUMED["SessionToken"],
    )

    # A client from the session signs with the assumed key. before-send answers
    # locally, so the signed request never leaves the process.
    eks = provider.session.client("eks")
    seen = {}

    def answer(request, **_):
        seen["auth"] = request.headers.get("Authorization", b"")
        seen["token"] = request.headers.get("X-Amz-Security-Token", b"")
        return AWSResponse(request.url, 200, {}, _Body(b'{"clusters": []}'))

    eks.meta.events.register("before-send.eks.ListClusters", answer)
    eks.list_clusters()
    auth = seen["auth"].decode() if isinstance(seen["auth"], bytes) else seen["auth"]
    token = seen["token"].decode() if isinstance(seen["token"], bytes) else seen["token"]
    assert f"Credential={ASSUMED['AccessKeyId']}/" in auth
    assert AMBIENT_KEY not in auth
    assert token == ASSUMED["SessionToken"]


def test_provider_cli_children_get_the_role_only_through_their_own_env(aws_env, sts, spawned):
    from app.infra.providers.aws_eks import AwsEksProvider

    AwsEksProvider().provision_cluster()
    verbs = [" ".join(c["cmd"][:3]) for c in spawned]
    assert verbs == ["aws eks describe-cluster", "aws eks create-cluster", "aws eks wait"], verbs
    for call in spawned:
        _assert_child_got_the_role(call)


def test_provider_configure_and_teardown_reuse_the_scoped_role(aws_env, sts, spawned):
    from app.infra.providers.aws_eks import AwsEksProvider

    provider = AwsEksProvider()
    provider.provision_cluster()
    spawned.clear()
    before = dict(os.environ)
    assert provider.configure_kubectl()["status"] == "SUCCESS"
    provider.teardown()  # describe says "not found" -> SKIPPED, after one child
    assert dict(os.environ) == before

    update = spawned[0]
    assert update["cmd"][:3] == ["aws", "eks", "update-kubeconfig"]
    # kubectl/helm later run with the ambient env, so the kubeconfig itself
    # must assume the creator role.
    assert update["cmd"][-2:] == ["--role-arn", ROLE_ARN]
    for call in spawned:
        _assert_child_got_the_role(call)


def test_provider_without_a_role_lets_children_inherit(aws_env, spawned):
    """Before any role is assumed nothing is injected: the CLI keeps its ambient chain."""
    from app.infra.providers.aws_eks import AwsEksProvider

    AwsEksProvider().teardown()
    assert spawned and all(c["env"] is None for c in spawned)


def test_provider_failed_assume_role_writes_nothing(aws_env, monkeypatch, spawned):
    from app.infra.providers.aws_eks import AwsEksProvider

    client = boto3.session.Session(
        aws_access_key_id=AMBIENT_KEY, aws_secret_access_key="x", region_name=REGION,
    ).client("sts")
    stubber = Stubber(client)
    stubber.add_client_error("assume_role", service_error_code="AccessDenied", http_status_code=403)
    stubber.activate()
    monkeypatch.setattr(boto3, "client", lambda *a, **k: client)

    before = dict(os.environ)
    outcome = AwsEksProvider().provision_cluster()
    assert outcome["status"] == "FAILED" and "assume IAM role" in outcome["message"]
    assert dict(os.environ) == before
    assert spawned == []


# ---------------- the scoped env itself ---------------- #

def test_scoped_subprocess_env_is_per_context_and_resets(monkeypatch, spawned):
    from app.infra.providers.base import child_env, run_command, scoped_subprocess_env

    monkeypatch.delenv("SCOPED_PROBE", raising=False)
    other_thread = {}

    with scoped_subprocess_env({"SCOPED_PROBE": "inside"}):
        run_command(["true"], "probe")
        t = threading.Thread(target=lambda: other_thread.setdefault("env", child_env()))
        t.start()
        t.join()
        assert "SCOPED_PROBE" not in os.environ
    assert child_env() is None
    run_command(["true"], "probe")

    assert spawned[0]["env"]["SCOPED_PROBE"] == "inside"
    assert spawned[1]["env"] is None
    assert other_thread["env"] is None, "the scope reached another thread"


# ---------------- legacy k8s_invoker path (infra agent) ---------------- #

def test_k8s_invoker_provision_leaves_os_environ_byte_identical(aws_env, sts, spawned):
    from app.infra import k8s_invoker

    before = dict(os.environ)
    outcome = k8s_invoker.provision_cluster("aws")
    after = dict(os.environ)

    assert sts == ["sts"], "the assume-role step under test did not run"
    assert outcome["status"] == "SUCCESS", outcome
    assert after == before
    verbs = [" ".join(c["cmd"][:3]) for c in spawned]
    assert verbs == ["aws eks describe-cluster", "aws eks create-cluster", "aws eks wait"], verbs
    for call in spawned:
        _assert_child_got_the_role(call)


def test_k8s_invoker_kubeconfig_assumes_the_creator_role(aws_env, monkeypatch, spawned):
    from app.infra import k8s_invoker

    monkeypatch.setattr(k8s_invoker.time, "sleep", lambda *_: None)
    k8s_invoker.configure_kubectl("aws")
    update = next(c for c in spawned if c["cmd"][:3] == ["aws", "eks", "update-kubeconfig"])
    assert update["cmd"][-2:] == ["--role-arn", ROLE_ARN]
