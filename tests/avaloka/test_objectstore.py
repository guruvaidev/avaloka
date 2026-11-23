"""ObjectStoreConnector on its own: what the conformance suite does not say.

The suite (test_connector_conformance.py) runs the seven rules over the
memory, local and moto-S3 cases. This file holds the object-store specifics:
a signed URL is a credential and is refused; options arrive only as a
Credentials object; endpoints per provider; a prefix is a set of objects whose
version moves when any of them does; a ref cannot climb out of the prefix or
name another bucket; anonymous is explicit, never a fallback.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from avaloka.connectors import Capability, ConnectorError, Credentials, Endpoint
from avaloka.connectors.objectstore import ObjectStoreConnector, parse_store_url

FRAME = pd.DataFrame({"id": range(40), "region": ["n", "s"] * 20, "note": [f"row-{i:05d}-xxxx" for i in range(40)]})


@pytest.fixture
def memory_bucket():
    import fsspec
    fs = fsspec.filesystem("memory")
    fs.rm("/bkt", recursive=True) if fs.exists("/bkt") else None
    with fs.open("/bkt/data/orders.csv", "wb") as fh:
        fh.write(FRAME.to_csv(index=False).encode())
    buf = io.BytesIO()
    FRAME.to_parquet(buf, index=False)
    with fs.open("/bkt/data/orders.parquet", "wb") as fh:
        fh.write(buf.getvalue())
    with fs.open("/bkt/other/notes.txt", "wb") as fh:
        fh.write(b"not a table")
    yield fs
    fs.rm("/bkt", recursive=True)


@pytest.mark.parametrize("url", [
    "s3://b/k.csv?X-Amz-Signature=deadbeef&X-Amz-Credential=AKIA",
    "az://acct/c/k.parquet?sv=2024-01-01&sig=SUPERSECRET&se=2026",
    "gs://b/k.csv?GoogleAccessId=svc&Signature=abc",
    "s3://user:pass@b/k.csv",
])
def test_a_signed_or_userinfo_url_is_refused(url):
    """Bite: strip the query string and carry on."""
    with pytest.raises(ConnectorError, match="credential"):
        parse_store_url(url)
    with pytest.raises(ConnectorError):
        ObjectStoreConnector(url, Credentials('{"anonymous": true}'))


def test_options_arrive_only_as_credentials():
    """Rule 4 for a store: a plain dict or string is refused, not wrapped."""
    with pytest.raises(ConnectorError, match="Credentials object"):
        ObjectStoreConnector("s3://acme/raw", {"access_key": "a", "secret_key": "b"})  # type: ignore[arg-type]
    with pytest.raises(ConnectorError, match="Credentials object"):
        ObjectStoreConnector("s3://acme/raw", '{"access_key": "a"}')  # type: ignore[arg-type]
    with pytest.raises(ConnectorError, match="JSON"):
        ObjectStoreConnector("s3://acme/raw", Credentials("not json"))
    c = ObjectStoreConnector("s3://acme/raw", Credentials(json.dumps({"access_key": "AKIA", "secret_key": "S3"})))
    assert isinstance(c.credentials, Credentials) and "S3" not in repr(c) and "AKIA" not in c.describe()
    assert c.describe() == "s3://acme/raw"


def test_anonymous_is_explicit_not_a_fallback():
    c = ObjectStoreConnector("s3://public-bucket/data")
    assert json.loads(c.credentials.reveal()) == {"anonymous": True}
    assert c._fs_kwargs() == {"anon": True}


@pytest.mark.parametrize("url, options, expected", [
    ("s3://acme/raw", {"access_key": "a", "secret_key": "b"}, Endpoint("s3.amazonaws.com", 443, "https")),
    ("s3://acme/raw", {"access_key": "a", "secret_key": "b", "region": "us-east-1"}, Endpoint("s3.amazonaws.com", 443, "https")),
    ("s3://acme/raw", {"access_key": "a", "secret_key": "b", "region": "eu-west-1"}, Endpoint("s3.eu-west-1.amazonaws.com", 443, "https")),
    ("s3://acme/raw", {"access_key": "a", "secret_key": "b", "endpoint_url": "https://seaweed.svc:8333"}, Endpoint("seaweed.svc", 8333, "https")),
    ("s3://acme/raw", {"access_key": "a", "secret_key": "b", "endpoint_url": "http://minio.local"}, Endpoint("minio.local", 80, "http")),
    ("gs://acme/raw", {"service_account_json": {"type": "service_account", "private_key": "k"}}, Endpoint("storage.googleapis.com", 443, "https")),
    ("az://container/raw", {"account_name": "acct", "account_key": "k"}, Endpoint("acct.blob.core.windows.net", 443, "https")),
    ("abfss://container/raw", {"account_name": "acct", "sas_token": "?sv=1&sig=x"}, Endpoint("acct.blob.core.windows.net", 443, "https")),
    ("az://container/raw", {"connection_string": "DefaultEndpointsProtocol=https;AccountName=fromcs;AccountKey=k=="}, Endpoint("fromcs.blob.core.windows.net", 443, "https")),
])
def test_endpoints_name_the_one_host_the_provider_opens(url, options, expected):
    """Rule 5 by URL and options (the socket recorder runs on the moto case).
    Bite: return [] for a provider; use the bucket as the host."""
    c = ObjectStoreConnector(url, Credentials(json.dumps(options)))
    assert c.endpoints() == [expected]
    text = json.dumps([e.as_dict() for e in c.endpoints()])
    for secret in ("private_key", "sig=x", "AccountKey", '"b"'):
        assert secret not in text


def test_azure_without_an_account_cannot_say_its_endpoint():
    with pytest.raises(ConnectorError, match="account_name"):
        ObjectStoreConnector("az://c/p", Credentials(json.dumps({"sas_token": "?sig=x"}))).endpoints()


def test_unknown_scheme_and_missing_bucket_are_refused():
    with pytest.raises(ConnectorError, match="unsupported"):
        parse_store_url("ftp://host/x")
    with pytest.raises(ConnectorError, match="bucket"):
        parse_store_url("s3:///nobucket")
    assert parse_store_url("s3://acme/raw/sub/") == ("s3", "acme", "raw/sub")
    assert parse_store_url("file:///tmp/data") == ("file", "", "/tmp/data")
    assert parse_store_url("memory://bkt") == ("memory", "bkt", "")


def test_catalogue_lists_data_objects_under_the_prefix_only(memory_bucket):
    c = ObjectStoreConnector("memory://bkt/data")
    names = [t["name"] for t in c.catalogue()["tables"]]
    assert names == ["orders.csv", "orders.parquet"]
    assert c.catalogue()["tables"][0]["columns"][1] == {"name": "region", "type": "string", "nullable": True}
    whole = ObjectStoreConnector("memory://bkt")
    assert [t["name"] for t in whole.catalogue()["tables"]] == ["data/orders.csv", "data/orders.parquet"]
    assert c.capabilities == {Capability.READ, Capability.SCHEMA, Capability.SAMPLE}
    with pytest.raises(ConnectorError, match="capability not declared"):
        c.pushdown([], [])


def test_a_prefix_is_a_set_whose_version_moves_with_any_object(memory_bucket):
    """Bite: version the first object only."""
    c = ObjectStoreConnector("memory://bkt/data")
    one = c.source_version("orders.csv")
    assert one.ref == "orders.csv" and one.marker["size"] > 0
    whole = c.source_version()
    assert whole.marker["objects"] == 2 and whole == c.source_version()
    with memory_bucket.open("/bkt/data/orders.parquet", "wb") as fh:
        buf = io.BytesIO()
        pd.concat([FRAME, FRAME.iloc[:1]]).to_parquet(buf, index=False)
        fh.write(buf.getvalue())
    assert c.source_version("orders.csv") == one, "an untouched object keeps its version"
    moved = c.source_version()
    assert moved != whole and moved.marker["objects"] == 2
    with memory_bucket.open("/bkt/data/third.csv", "wb") as fh:
        fh.write(FRAME.to_csv(index=False).encode())
    assert c.source_version().marker["objects"] == 3


def test_a_prefix_read_concatenates_objects_in_name_order_and_is_bounded(memory_bucket):
    c = ObjectStoreConnector("memory://bkt/data")
    whole = c.read()
    assert len(whole) == 80 and whole.receipt.ref == "data" and set(whole.receipt.columns) == {"id", "region", "note"}
    assert whole.to_pandas()["id"].tolist() == list(range(40)) * 2
    part = c.read(max_rows=50, max_bytes=1 << 30)
    assert len(part) == 50 and part.receipt.stopped_by == "max_rows"
    assert c.statistics()["objects"] == 2 and c.statistics()["row_count"] == 80
    one = c.read("orders.parquet")
    assert len(one) == 40 and one.receipt.ref == "orders.parquet" and one.receipt.source_version.ref == "orders.parquet"


def test_a_ref_cannot_leave_the_prefix_or_name_another_store(memory_bucket):
    c = ObjectStoreConnector("memory://bkt/data")
    with pytest.raises(ConnectorError, match="climb"):
        c.read("../other/notes.txt")
    with pytest.raises(ConnectorError, match="another store"):
        c.read("memory://elsewhere/data/orders.csv")
    with pytest.raises(ConnectorError, match="another store"):
        c.read("s3://bkt/data/orders.csv")
    assert len(c.read("memory://bkt/data/orders.csv")) == 40   # the same store, spelled in full
    with pytest.raises(ConnectorError, match="no data object"):
        c.read("missing.csv")
    with pytest.raises(ConnectorError, match="no data object"):
        ObjectStoreConnector("memory://bkt/other").read()


def test_csv_row_count_is_an_estimate_past_the_counting_budget(memory_bucket, monkeypatch):
    from avaloka.connectors import objectstore
    c = ObjectStoreConnector("memory://bkt/data")
    exact = c.statistics("orders.csv")
    assert exact["row_count"] == 40 and exact["row_count_estimated"] is False
    monkeypatch.setattr(objectstore, "_ESTIMATE_BYTES", 64)
    c2 = ObjectStoreConnector("memory://bkt/data")
    est = c2.statistics("orders.csv")
    assert est["row_count_estimated"] is True and 10 <= est["row_count"] <= 200


def test_local_directory_is_a_store_with_no_endpoints(tmp_path):
    FRAME.to_csv(tmp_path / "a.csv", index=False)
    FRAME.to_parquet(tmp_path / "b.parquet", index=False)
    (tmp_path / "readme.md").write_text("x")
    c = ObjectStoreConnector(f"file://{tmp_path}")
    assert c.endpoints() == [] and json.loads(c.credentials.reveal()) == {"local": True}
    assert [t["name"] for t in c.catalogue()["tables"]] == ["a.csv", "b.parquet"]
    assert len(c.read()) == 80
    v = c.source_version("a.csv")
    assert v.marker["size"] > 0 and ("mtime" in v.marker or "etag" in v.marker)
