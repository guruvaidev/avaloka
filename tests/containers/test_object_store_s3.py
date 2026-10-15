"""The S3 blob store against a real S3-compatible server (SeaweedFS).

The default deployment stores every uploaded file in the in-cluster SeaweedFS
through ``S3BlobStore`` and an endpoint override. Until now that path was only
ever tested against ``FakeBlobStore`` -- an in-memory dict -- so nothing checked
that boto3 as pinned, the endpoint override, request signing and multipart
upload actually work against the server that ships.

The store is SeaweedFS 4.48 at the digest the Helm chart pins, started with a
key pair generated for this run. Results are checked with a second boto3
client the code under test does not own: ``S3BlobStore.stat`` returns
``(0, "")`` for every error, so asking the store whether its own write landed
cannot tell a missing object from a refused request.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

import pytest
from botocore.exceptions import ClientError

from app.core.storage import S3BlobStore


def _store(s3_settings, bucket: str, prefix: str = "", **overrides) -> S3BlobStore:
    kwargs = dict(region=s3_settings["region"], access_key=s3_settings["access_key"],
                  secret_key=s3_settings["secret_key"], endpoint_url=s3_settings["endpoint_url"])
    kwargs.update(overrides)
    return S3BlobStore(bucket, prefix, **kwargs)


def _keys(s3_settings, bucket: str) -> dict:
    """Every key in the bucket with its size, read by the independent client."""
    listing = s3_settings["client"].list_objects_v2(Bucket=bucket)
    return {obj["Key"]: obj["Size"] for obj in listing.get("Contents", [])}


def _write(path, data: bytes):
    path.write_bytes(data)
    return path


def test_an_uploaded_file_lands_under_the_prefix_and_reads_back(s3_settings, bucket, tmp_path):
    body = b"id,amount\n1,10.5\n2,20.25\n"
    store = _store(s3_settings, bucket, "tenant-files")

    uri = store.put_file(_write(tmp_path / "sales.csv", body), "uploads/u1/sales.csv")

    assert uri == f"s3://{bucket}/tenant-files/uploads/u1/sales.csv"
    assert _keys(s3_settings, bucket) == {"tenant-files/uploads/u1/sales.csv": len(body)}

    store.get_file("uploads/u1/sales.csv", tmp_path / "back.csv")
    assert (tmp_path / "back.csv").read_bytes() == body
    size, modified = store.stat("uploads/u1/sales.csv")
    assert size == len(body) and modified

    store.delete("uploads/u1/sales.csv")
    assert _keys(s3_settings, bucket) == {}
    assert store.stat("uploads/u1/sales.csv") == (0, "")


def test_a_file_large_enough_for_multipart_upload_survives_intact(s3_settings, bucket, tmp_path):
    # boto3 switches to multipart at 8 MiB, which is a different set of calls
    # (and of checksums) from a single PUT, and the one a real dataset takes.
    body = os.urandom(9 * 1024 * 1024 + 123)
    store = _store(s3_settings, bucket, "big")

    store.put_file(_write(tmp_path / "big.bin", body), "frame.parquet")

    assert _keys(s3_settings, bucket) == {"big/frame.parquet": len(body)}
    store.get_file("frame.parquet", tmp_path / "back.bin")
    assert (hashlib.sha256((tmp_path / "back.bin").read_bytes()).hexdigest()
            == hashlib.sha256(body).hexdigest())


def test_listings_stop_at_the_stores_own_prefix(s3_settings, bucket, tmp_path):
    src = _write(tmp_path / "f.csv", b"x\n1\n")
    team, neighbour = _store(s3_settings, bucket, "team"), _store(s3_settings, bucket, "team-b")
    team.put_file(src, "reports/q1.csv")
    team.put_file(src, "reports/2024/q2.csv")
    team.put_file(src, "top.csv")
    neighbour.put_file(src, "reports/theirs.csv")
    assert len(_keys(s3_settings, bucket)) == 4

    assert sorted(key for key, _, _ in team.list()) == [
        "reports/2024/q2.csv", "reports/q1.csv", "top.csv"]
    assert [key for key, _, _ in neighbour.list()] == ["reports/theirs.csv"]

    folders, files = team.list_hierarchy("reports")
    assert folders == ["reports/2024/"]
    assert [(name, size) for name, size, _ in files] == [("reports/q1.csv", 4)]
    folders, files = team.list_hierarchy("")
    assert folders == ["reports/"]
    assert [name for name, _, _ in files] == ["top.csv"]


def test_a_signed_url_serves_the_object_and_a_tampered_one_does_not(s3_settings, bucket, tmp_path):
    body = b"col\nvalue\n"
    store = _store(s3_settings, bucket, "shared")
    store.put_file(_write(tmp_path / "f.csv", body), "export.csv")

    url = store.signed_url("export.csv", expiry_minutes=5)
    assert url.startswith(s3_settings["endpoint_url"])
    with urllib.request.urlopen(url, timeout=15) as response:
        assert response.read() == body

    # Unsigned, and signed for a different object: the server must refuse both,
    # or the URL is not what is protecting the file.
    unsigned = url.split("?", 1)[0]
    other = url.replace("export.csv", "other.csv")
    for refused in (unsigned, other):
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(refused, timeout=15)
        assert caught.value.code == 403


def test_a_wrong_secret_key_fails_the_upload_loudly(s3_settings, bucket, tmp_path):
    src = _write(tmp_path / "f.csv", b"x\n1\n")
    wrong = _store(s3_settings, bucket, "t", secret_key=s3_settings["secret_key"] + "-wrong")

    with pytest.raises(Exception) as caught:
        wrong.put_file(src, "refused.csv")
    assert "SignatureDoesNotMatch" in str(caught.value) or "AccessDenied" in str(caught.value) \
        or "403" in str(caught.value), f"refused for an unexpected reason: {caught.value}"
    with pytest.raises(ClientError):
        list(wrong.list())
    assert _keys(s3_settings, bucket) == {}

    # The same call with the right key succeeds, so it was the key.
    _store(s3_settings, bucket, "t").put_file(src, "accepted.csv")
    assert _keys(s3_settings, bucket) == {"t/accepted.csv": 4}


# What the API server does at startup and on every later read of an uploaded
# file: build the store from the environment alone, then find a file again
# from the URI it was given back. Run as its own process because
# app.core.settings reads the environment once, at import.
SERVER = r"""
import json, pathlib, sys
from app.services import storage_service

src, dest, foreign_uri, foreign_dest = sys.argv[1:5]
store = storage_service.init_blob_store()
uri = store.put_file(pathlib.Path(src), "uploads/u1/data.csv")

again, key = storage_service._store_and_key_from_uri(uri, None)
again.get_file(key, pathlib.Path(dest))

foreign, foreign_key = storage_service._store_and_key_from_uri(foreign_uri, None)
foreign.get_file(foreign_key, pathlib.Path(foreign_dest))

print(json.dumps({"store": type(store).__name__, "uri": uri, "reused": again is store,
                  "key": key, "foreign_reused": foreign is store, "foreign_key": foreign_key}))
"""


def test_the_server_finds_its_store_from_the_environment_and_a_file_from_its_uri(
        s3_settings, bucket, tmp_path):
    body = b"id,amount\n1,10.5\n"
    theirs = b"shared,by,another,bucket\n"
    client = s3_settings["client"]
    other_bucket = bucket + "-other"
    client.create_bucket(Bucket=other_bucket)
    client.put_object(Bucket=other_bucket, Key="incoming/report.csv", Body=theirs)

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("AWS_", "S3_", "GCS_", "AZURE_", "STORAGE_"))}
    env.update({
        "STORAGE_BACKEND": "s3", "S3_BUCKET": bucket, "S3_PREFIX": "tenant-files",
        "S3_ENDPOINT_URL": s3_settings["endpoint_url"], "AWS_REGION": s3_settings["region"],
        "AWS_DEFAULT_REGION": s3_settings["region"],
        "AWS_ACCESS_KEY_ID": s3_settings["access_key"],
        "AWS_SECRET_ACCESS_KEY": s3_settings["secret_key"],
        # app.core.settings calls load_dotenv(); a developer's .env must not
        # reconfigure this process.
        "PYTHON_DOTENV_DISABLED": "1",
    })
    done = subprocess.run(
        [sys.executable, "-c", SERVER, str(_write(tmp_path / "data.csv", body)),
         str(tmp_path / "back.csv"), f"s3://{other_bucket}/incoming/report.csv",
         str(tmp_path / "foreign.csv")],
        env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    said = json.loads(done.stdout.strip().splitlines()[-1])

    assert said["store"] == "S3BlobStore"
    assert said["uri"] == f"s3://{bucket}/tenant-files/uploads/u1/data.csv"
    assert _keys(s3_settings, bucket) == {"tenant-files/uploads/u1/data.csv": len(body)}
    # The file is found again through the configured store, not a second one.
    assert said["reused"] is True and said["key"] == "uploads/u1/data.csv"
    assert (tmp_path / "back.csv").read_bytes() == body
    # A URI naming another bucket gets its own store, and that store must carry
    # the endpoint override too -- without it the read goes to AWS.
    assert said["foreign_reused"] is False and said["foreign_key"] == "report.csv"
    assert (tmp_path / "foreign.csv").read_bytes() == theirs
