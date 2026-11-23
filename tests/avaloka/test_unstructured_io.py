"""Offline coverage for nested JSON and read-only Mongo materialization."""

from __future__ import annotations

import json
import sys
import types
from urllib.parse import quote

import pandas as pd
import pytest

from avaloka.io import load_dataset, materialize, normalize_records
from avaloka.io.sources import SourceError


def test_jsonl_flattens_nested_records_and_normalizes_text(tmp_path):
    path = tmp_path / "reviews.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({
                "business": {"id": "b1", "address": {"city": "Montréal"}},
                "text": "Great\u0000   food\nagain",
                "categories": ["café", "lunch"],
            }),
            json.dumps({
                "business": {"id": "b2", "address": {"city": "Toronto"}},
                "text": "Fine",
                "categories": [],
            }),
        ]),
        encoding="utf-8",
    )

    dataset = load_dataset(str(path))

    assert dataset.fmt == "jsonl"
    assert dataset.n_rows == 2
    assert dataset.frame.loc[0, "business.address.city"] == "Montréal"
    assert dataset.frame.loc[0, "text"] == "Great food again"
    assert json.loads(dataset.frame.loc[0, "categories"]) == ["café", "lunch"]
    assert "business.address" in dataset.structure["nested_paths"]
    assert dataset.structure["list_paths"] == ["categories"]


def test_json_wrapper_and_scalar_records_are_supported(tmp_path):
    path = tmp_path / "wrapped.json"
    path.write_text(json.dumps({"records": [{"x": 1}, {"x": 2}]}))
    assert load_dataset(str(path)).frame["x"].tolist() == [1, 2]

    frame, structure = normalize_records([1, " two  ", None])
    assert frame["value"].tolist()[:2] == [1, "two"]
    assert structure["normalization"] == "dotted_paths_v1"


def test_csv_json_object_column_is_expanded_without_losing_original(tmp_path):
    path = tmp_path / "export.csv"
    pd.DataFrame({
        "id": [1, 2],
        "attributes": [
            json.dumps({"hours": {"monday": "9-5"}, "wifi": True}),
            json.dumps({"hours": {"monday": "closed"}, "wifi": False}),
        ],
    }).to_csv(path, index=False)

    dataset = load_dataset(str(path))

    assert "attributes" in dataset.frame.columns
    assert dataset.frame["attributes.hours.monday"].tolist() == ["9-5", "closed"]
    assert dataset.frame["attributes.wifi"].tolist() == [True, False]
    assert dataset.structure["json_object_columns"] == ["attributes"]


class _FakeCursor:
    def __init__(self, documents):
        self.documents = list(documents)
        self.limit_value = None

    def sort(self, _spec):
        return self

    def limit(self, value):
        self.limit_value = value
        return iter(self.documents[:value])


class _FakeCollection:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def find(self, filter_value, projection):
        self.calls.append((filter_value, projection))
        return _FakeCursor(self.documents)


class _FakeDatabase:
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, _name):
        return self.collection


def test_mongodb_materialization_is_bounded_flat_and_redacted(tmp_path, monkeypatch):
    collection = _FakeCollection([
        {"_id": "1", "business": {"city": "Berkeley"}, "stars": 5},
        {"_id": "2", "business": {"city": "Oakland"}, "stars": 4},
        {"_id": "3", "business": {"city": "Albany"}, "stars": 3},
    ])

    class FakeClient:
        instances = []

        def __init__(self, uri, **_kwargs):
            self.uri = uri
            self.closed = False
            self.__class__.instances.append(self)

        def get_default_database(self):
            return _FakeDatabase(collection)

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules, "pymongo", types.SimpleNamespace(MongoClient=FakeClient)
    )
    filter_value = quote(json.dumps({"stars": {"$gte": 4}}))
    uri = (
        "mongodb://analyst:secret@localhost/reviews?authSource=admin"
        f"#collection=businesses&filter={filter_value}&limit=2"
    )

    resolved = materialize(uri, cache_dir=tmp_path)
    frame = pd.read_parquet(resolved.local_path)

    assert resolved.kind == "document_database"
    assert len(frame) == 2
    assert "business.city" in frame.columns
    assert "secret" not in resolved.uri
    assert "authSource=admin" not in resolved.uri
    assert resolved.detail["bounded"] is True
    assert collection.calls[0][0] == {"stars": {"$gte": 4}}
    assert FakeClient.instances[0].closed


def test_mongodb_requires_explicit_collection():
    with pytest.raises(SourceError, match="collection"):
        materialize("mongodb://localhost/reviews")
