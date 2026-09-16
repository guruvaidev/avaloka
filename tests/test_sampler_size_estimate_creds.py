"""
Cloud size estimation with per-connection credentials.

Cloud size estimation ignored per-connection credentials: GCS used ambient
default creds and Azure used AZURE_ACCOUNT/AZURE_SAS env vars. On a private
customer bucket the metadata probe failed -> total_bytes 0 -> a hardcoded
500,000-row fallback -> wrong fidelity tier and data_shape.rows.

_estimate_dataset_size now accepts cloud_credentials and threads them into the
per-scheme metadata probes.
"""

import inspect
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.agents.sampling_agent_daft as sad


def test_estimate_size_signature_accepts_credentials():
    # The old signature had no such parameter, so private-bucket creds were dropped.
    assert "cloud_credentials" in inspect.signature(sad._estimate_dataset_size).parameters


def test_estimate_threads_credentials_and_avoids_fallback(monkeypatch):
    creds = {"gcp_service_account_json": '{"project_id": "p"}'}
    seen = {}

    class _Blob:
        def __init__(self, name, size):
            self.name, self.size = name, size

        def download_as_bytes(self, start=0, end=0):
            return b"col_a,col_b\n" + b"value1,value2\n" * 200

    class _Client:
        def bucket(self, name):
            return self

        def get_blob(self, path):
            return _Blob(path, 2_000_000)

        def list_blobs(self, bucket, prefix=""):
            return [_Blob("f.csv", 2_000_000)]

    def fake_gcs(c):
        seen["creds"] = c
        return _Client()

    monkeypatch.setattr(sad, "_gcs_client_from_creds", fake_gcs)

    rows, mb = sad._estimate_dataset_size("gs://private/data.csv", "csv", cloud_credentials=creds)

    assert seen["creds"] == creds                      # per-connection creds used for the probe
    assert rows not in (500_000, 0)                     # real estimate, not the auth-fail fallback
    assert mb == pytest.approx(2_000_000 / (1024 * 1024), rel=1e-6)


def test_s3_client_uses_connection_keys(monkeypatch):
    captured = {}
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda service, **kw: (captured.update(service=service, kw=kw), "S3")[1]
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    sad._s3_client_from_creds({"access_key": "AK", "secret_key": "SK", "region": "us-west-2"})
    assert captured["kw"]["aws_access_key_id"] == "AK"
    assert captured["kw"]["aws_secret_access_key"] == "SK"
    assert captured["kw"]["region_name"] == "us-west-2"

    captured.clear()
    sad._s3_client_from_creds(None)                     # no creds -> ambient client, no overrides
    assert captured["kw"] == {}


def test_azure_client_uses_connection_account_and_sas(monkeypatch):
    captured = {}
    fake_mod = types.ModuleType("azure.storage.blob")
    fake_mod.BlobServiceClient = lambda account_url=None, credential=None: (
        captured.update(url=account_url, credential=credential), "AZ")[1]
    monkeypatch.setitem(sys.modules, "azure", types.ModuleType("azure"))
    monkeypatch.setitem(sys.modules, "azure.storage", types.ModuleType("azure.storage"))
    monkeypatch.setitem(sys.modules, "azure.storage.blob", fake_mod)

    sad._azure_client_from_creds({"storage_account": "acct", "sas_token": "sig=abc"})
    assert captured["url"] == "https://acct.blob.core.windows.net"
    assert captured["credential"] == "sig=abc"
