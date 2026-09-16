import json

import pytest


@pytest.fixture
def server_mod():
    # Imported lazily so this module still collects when an earlier test in the
    # same session has polluted sys.modules for app.agents.visualization_agent.
    from app.api import server as server_mod

    return server_mod


class FakeStore:
    """Records put_file calls; deliberately defines no other upload method."""

    def __init__(self):
        self.calls = []

    def put_file(self, local_path, key):
        self.calls.append((str(local_path), key))
        return key


class ExplodingStore:
    def put_file(self, local_path, key):
        raise RuntimeError("cloud unavailable")


def _sess(tmp_path, **overrides):
    sample = tmp_path / "sample_input.csv"
    sample.write_text("a,b\n1,2\n")
    sess = {
        "data_source_location": "gs://bucket/data/sales.csv",
        "sample_local_input": str(sample),
    }
    sess.update(overrides)
    return sess


def test_uploads_sample_with_put_file_and_returns_uri(server_mod, tmp_path, monkeypatch):
    store = FakeStore()
    seen = {}

    def fake_store_and_key(uri, object_name):
        seen["uri"] = uri
        return store, "sales.csv"

    monkeypatch.setattr(server_mod, "_store_and_key_from_uri", fake_store_and_key)
    sess = _sess(tmp_path)

    uri = server_mod._upload_sample_to_cloud_if_needed(sess)

    assert uri == "gs://bucket/data/_avaloka_samples/sales.csv"
    assert seen["uri"] == uri
    assert store.calls == [(sess["sample_local_input"], "sales.csv")]


def test_sample_key_lands_at_bucket_root_when_object_has_no_dir(server_mod, tmp_path, monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(
        server_mod, "_store_and_key_from_uri", lambda uri, obj: (store, "sales.csv")
    )
    sess = _sess(tmp_path, data_source_location="s3://bucket/sales.csv")

    uri = server_mod._upload_sample_to_cloud_if_needed(sess)

    assert uri == "s3://bucket/_avaloka_samples/sales.csv"


def test_precomputed_sample_uri_short_circuits_upload(server_mod, tmp_path, monkeypatch):
    def must_not_be_called(uri, obj):
        raise AssertionError("store should not be built when a precomputed URI exists")

    monkeypatch.setattr(server_mod, "_store_and_key_from_uri", must_not_be_called)
    sess = _sess(
        tmp_path,
        portfolio_sample_uris=json.dumps({"balanced": "gs://bucket/pre/balanced.csv"}),
        selected_sample_name="balanced",
    )

    assert (
        server_mod._upload_sample_to_cloud_if_needed(sess)
        == "gs://bucket/pre/balanced.csv"
    )


def test_returns_none_when_sample_file_missing(server_mod, tmp_path):
    sess = {
        "data_source_location": "gs://bucket/data/sales.csv",
        "sample_local_input": str(tmp_path / "does_not_exist.csv"),
    }
    assert server_mod._upload_sample_to_cloud_if_needed(sess) is None


def test_returns_none_without_storage_uri(server_mod, tmp_path):
    sample = tmp_path / "sample_input.csv"
    sample.write_text("a\n1\n")
    sess = {"data_source_location": "", "sample_local_input": str(sample)}
    assert server_mod._upload_sample_to_cloud_if_needed(sess) is None


def test_returns_none_when_upload_fails(server_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(
        server_mod, "_store_and_key_from_uri", lambda uri, obj: (ExplodingStore(), "k")
    )
    assert server_mod._upload_sample_to_cloud_if_needed(_sess(tmp_path)) is None


def test_sample_uri_matches_where_store_actually_writes(server_mod, tmp_path, monkeypatch):
    from app.services import storage_service

    captured = {}

    class FakeGCS:
        def __init__(self, bucket, prefix, *args, **kwargs):
            captured["bucket"] = bucket
            captured["prefix"] = prefix

        def put_file(self, local_path, key):
            captured["key"] = key
            return key

    monkeypatch.setattr(storage_service, "GCSBlobStore", FakeGCS)
    monkeypatch.setattr(storage_service, "blob_store", None)
    monkeypatch.setattr(
        server_mod, "_store_and_key_from_uri", storage_service._store_and_key_from_uri
    )
    sess = _sess(tmp_path, data_source_location="gs://mybucket/data/sales.csv")

    uri = server_mod._upload_sample_to_cloud_if_needed(sess)

    assert uri == "gs://mybucket/data/_avaloka_samples/sales.csv"
    written_path = f"{captured['prefix']}/{captured['key']}".strip("/")
    assert f"gs://{captured['bucket']}/{written_path}" == uri
