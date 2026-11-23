"""T16.11: ``/datasets`` is a view over the catalog that keeps every Redis
item. Bite: drop Redis-only datasets; include a table or a bucket as a
"dataset"; fail the route when the catalog is off."""

from __future__ import annotations

from app.api.schemas import DatasetListItem
from app.catalog import compat
from app.catalog.store import CatalogStore

from tests.catalog.conftest import ORDERS


def _redis_item(dsid: str, created: str) -> DatasetListItem:
    return DatasetListItem(dataset_id=dsid, created_at=created, size_bytes=10, columns=["a"],
                           session_id="s", group_session_id="s", filename=f"{dsid}.csv", alias=None, source_kind="upload")


def test_catalog_file_and_derived_datasets_join_and_redis_items_stay(db_url):
    store = CatalogStore("user-a")
    store.register("file", {"dataset_id": "u1", "filename": "orders_2026.csv", "format": "csv"}, ORDERS,
                   dataset_id="u1", display_name="orders_2026.csv", row_count=1200, row_count_kind="exact", bytes=2048)
    store.register("derived", {"analysis_id": "an:1", "parent_dataset_id": "u1", "output_location": "s3://o/x.csv"},
                   [("region", "object")], dataset_id="ds:output:1", display_name="churn.csv", row_count=2)
    store.register("db_table", {"connection_id": "c1", "table": "orders"}, ORDERS)      # not a "dataset" on that tab
    store.register("file", {"dataset_id": "u2", "filename": "seen.csv"}, [("x", "int64")], dataset_id="u2",
                   display_name="seen.csv")
    CatalogStore("user-b").register("file", {"dataset_id": "u9", "filename": "theirs.csv"}, [("x", "int64")],
                                    dataset_id="u9", display_name="theirs.csv")

    extra = compat.catalog_items("user-a", exclude={"u2"})
    ids = [e["dataset_id"] for e in extra]
    assert ids == ["ds:output:1", "u1"] or sorted(ids) == ["ds:output:1", "u1"]
    by = {e["dataset_id"]: e for e in extra}
    assert by["u1"]["columns"] == ["order_id", "customer_email", "amount"] and by["u1"]["size_bytes"] == 2048
    assert by["u1"]["filename"] == "orders_2026.csv" and by["u1"]["alias"] == "my.orders_2026"
    assert by["u1"]["source_kind"] == "catalog" and by["u1"]["session_id"] is None
    assert by["ds:output:1"]["filename"] == "churn.csv"
    assert "u9" not in by and "u2" not in by

    redis_only = _redis_item("redis-only", "2026-11-14T00:00:00+00:00")
    merged = compat.merge_dataset_items([redis_only, _redis_item("u2", "2026-11-13T00:00:00+00:00")], extra, DatasetListItem)
    assert all(isinstance(it, DatasetListItem) for it in merged)
    assert {it.dataset_id for it in merged} == {"redis-only", "u2", "u1", "ds:output:1"}
    assert merged == sorted(merged, key=lambda it: it.created_at, reverse=True)
    assert [it for it in merged if it.dataset_id == "u2"][0].source_kind == "upload", "the Redis item wins for a seen id"


def test_no_catalog_means_the_route_is_exactly_what_it_was(db_url, monkeypatch):
    monkeypatch.delenv("AVALOKA_LINEAGE_DB_URL")
    assert compat.catalog_items("user-a") == []
    redis_only = _redis_item("r1", "2026-11-14T00:00:00+00:00")
    assert compat.merge_dataset_items([redis_only], [], DatasetListItem) == [redis_only]


def test_the_server_route_uses_the_view():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2] / "app" / "api" / "server.py").read_text(encoding="utf-8")
    route = text[text.index('@app.get("/datasets", response_model=List[DatasetListItem]'):text.index('"/datasets/{dataset_id}",')]
    assert "_catalog_items" in route and "_merge_items" in route
    assert "return []  # genuinely no sessions" not in route, "no sessions must still show the catalog"
