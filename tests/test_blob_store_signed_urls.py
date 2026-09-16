import asyncio
from datetime import timedelta
from unittest.mock import MagicMock

from app.core.storage import AzureBlobStore, GCSBlobStore, IBlobStore, S3BlobStore
from app.services import persistence_service
from app.services.storage_service import LocalBlobStore


# -------------------------
# Interface contract
# -------------------------

def test_every_blob_store_defines_signed_url():
    assert "signed_url" in IBlobStore.__abstractmethods__
    for cls in (GCSBlobStore, S3BlobStore, AzureBlobStore, LocalBlobStore):
        assert callable(cls.__dict__.get("signed_url")), cls.__name__


# -------------------------
# S3 — real local signing, no network
# -------------------------

def test_s3_signed_url_contains_bucket_key_and_signature():
    store = S3BlobStore(
        bucket="test-bucket",
        prefix="code-registry",
        region="us-east-1",
        access_key="AKIATESTKEY",
        secret_key="testsecret",
    )
    url = store.signed_url("user/sess/file.py", expiry_minutes=15)

    assert url.startswith("https://")
    assert "test-bucket" in url
    assert "code-registry/user/sess/file.py" in url
    assert "Signature" in url


# -------------------------
# GCS
# -------------------------

def _gcs_store_with_blob(blob, prefix=""):
    store = GCSBlobStore.__new__(GCSBlobStore)
    store._prefix = prefix
    bucket = MagicMock()
    bucket.blob.return_value = blob
    store._bucket = bucket
    return store, bucket


def test_gcs_signed_url_uses_v4_get_and_prefixed_key():
    blob = MagicMock()
    blob.generate_signed_url.return_value = "https://signed.example/x"
    store, bucket = _gcs_store_with_blob(blob, prefix="sys")

    url = store.signed_url("code-registry/u/s/file.py", expiry_minutes=20)

    assert url == "https://signed.example/x"
    bucket.blob.assert_called_once_with("sys/code-registry/u/s/file.py")
    kwargs = blob.generate_signed_url.call_args.kwargs
    assert kwargs["version"] == "v4"
    assert kwargs["method"] == "GET"
    assert kwargs["expiration"] == timedelta(minutes=20)


def test_gcs_signed_url_falls_back_to_iam_signing(monkeypatch):
    blob = MagicMock()

    def _sign(**kwargs):
        if "access_token" not in kwargs:
            raise AttributeError("you need a private key to sign credentials")
        return "https://signed.example/iam"

    blob.generate_signed_url.side_effect = _sign
    store, _ = _gcs_store_with_blob(blob)

    fake_creds = MagicMock()
    fake_creds.service_account_email = "svc@proj.iam.gserviceaccount.com"
    fake_creds.token = "ya29.fake-token"
    import google.auth
    monkeypatch.setattr(google.auth, "default", lambda: (fake_creds, "proj"))

    url = store.signed_url("file.py")

    assert url == "https://signed.example/iam"
    fake_creds.refresh.assert_called_once()
    kwargs = blob.generate_signed_url.call_args.kwargs
    assert kwargs["service_account_email"] == "svc@proj.iam.gserviceaccount.com"
    assert kwargs["access_token"] == "ya29.fake-token"
    assert kwargs["version"] == "v4"


# -------------------------
# Azure
# -------------------------

def _azure_store(sas):
    store = AzureBlobStore.__new__(AzureBlobStore)
    store.account = "acct"
    store.container = "cont"
    store.prefix = "pre"
    store.sas = sas
    return store


def test_azure_signed_url_appends_container_sas():
    store = _azure_store("?sv=2024&sig=abc")
    url = store.signed_url("code/file.py")
    assert url == "https://acct.blob.core.windows.net/cont/pre/code/file.py?sv=2024&sig=abc"


def test_azure_signed_url_without_sas_returns_none():
    assert _azure_store("").signed_url("code/file.py") is None


# -------------------------
# Local store
# -------------------------

def test_local_store_signed_url_is_none(tmp_path):
    store = LocalBlobStore(tmp_path)
    assert store.signed_url("any/key.csv") is None


# -------------------------
# generate_signed_url service helper
# -------------------------

def _patch_store(monkeypatch, store):
    async def fake_get_store(connection_id, storage_uri=None):
        return store, ""
    monkeypatch.setattr(persistence_service, "_get_store_for_connection", fake_get_store)


def test_generate_signed_url_delegates_to_store(monkeypatch):
    fake_store = MagicMock()
    fake_store.signed_url.return_value = "https://signed.example/ok"
    _patch_store(monkeypatch, fake_store)

    url = asyncio.run(
        persistence_service.generate_signed_url("k.py", expiry_minutes=5)
    )

    assert url == "https://signed.example/ok"
    fake_store.signed_url.assert_called_once_with("k.py", expiry_minutes=5)


def test_generate_signed_url_none_when_no_store(monkeypatch):
    _patch_store(monkeypatch, None)
    assert asyncio.run(persistence_service.generate_signed_url("k.py")) is None


def test_generate_signed_url_none_on_store_error(monkeypatch):
    fake_store = MagicMock()
    fake_store.signed_url.side_effect = RuntimeError("provider exploded")
    _patch_store(monkeypatch, fake_store)

    assert asyncio.run(persistence_service.generate_signed_url("k.py")) is None
