"""The /catalog router on a small app: list, get, resolve, delete, and the
share endpoint's edition sentence (T16.13's HTTP half). Bite: return 200 and
ignore the share; list without the caller's id."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.catalog.api import SHARE_NOT_BUILT, SHARE_SENTENCE, build_router
from app.catalog.store import CatalogStore, CatalogUnavailable
from app.core.editions import resolve_capabilities

from tests.catalog.conftest import ORDERS


def _client(caps=None, store_factory=None):
    app = FastAPI()
    users = {"tok-a": "user-a", "tok-b": "user-b"}
    app.include_router(build_router(
        lambda request: users.get(request.headers.get("authorization", "")),
        capabilities=caps, store_factory=store_factory))
    return TestClient(app)


@pytest.fixture
def seeded(db_url):
    store = CatalogStore("user-a")
    store.register("file", {"dataset_id": "u1", "filename": "orders_2026.csv"}, ORDERS, dataset_id="u1",
                   display_name="orders_2026.csv", analysis_id="an:1", row_count=12, row_count_kind="exact")
    store.register("file", {"dataset_id": "u1", "filename": "orders_2026.csv"}, ORDERS + [("region", "object")],
                   dataset_id="u1", analysis_id="an:2")
    return store


def test_unauthenticated_calls_are_refused(db_url):
    client = _client()
    assert client.get("/catalog/datasets").status_code == 401
    assert client.get("/catalog/datasets/u1").status_code == 401
    assert client.delete("/catalog/datasets/u1").status_code == 401
    assert client.post("/catalog/datasets/u1/share").status_code == 401


def test_list_get_resolve_are_the_callers_own(seeded):
    client = _client()
    a = client.get("/catalog/datasets", headers={"authorization": "tok-a"})
    assert a.status_code == 200 and a.json()["count"] == 1
    item = a.json()["datasets"][0]
    assert item["dataset_id"] == "u1" and item["version_no"] == 2 and item["name"] == "my.orders_2026"
    assert "password" not in str(item)
    b = client.get("/catalog/datasets", headers={"authorization": "tok-b"})
    assert b.status_code == 200 and b.json() == {"datasets": [], "count": 0}
    assert client.get("/catalog/datasets/u1", headers={"authorization": "tok-b"}).status_code == 404
    detail = client.get("/catalog/datasets/u1", headers={"authorization": "tok-a"}).json()
    assert [v["version_no"] for v in detail["versions"]] == [1, 2]
    assert detail["schema_change"] == {"added": ["region"], "removed": [], "retyped": []}
    assert detail["analyses"] == ["an:1", "an:2"]
    r = client.get("/catalog/resolve", params={"ref": "analysis:an:1"}, headers={"authorization": "tok-a"})
    assert r.status_code == 200 and r.json()["version_no"] == 1 and r.json()["pinned_by"] == "an:1"
    r = client.get("/catalog/resolve", params={"ref": "my.payroll"}, headers={"authorization": "tok-a"})
    assert r.status_code == 400 and "no dataset of yours" in r.json()["detail"]
    assert client.get("/catalog/datasets", params={"kind": "spreadsheet"},
                      headers={"authorization": "tok-a"}).status_code == 400
    q = client.get("/catalog/datasets", params={"q": "orders"}, headers={"authorization": "tok-a"})
    assert q.json()["count"] == 1


def test_delete_is_the_owners_and_leaves_a_tombstone(seeded):
    client = _client()
    assert client.delete("/catalog/datasets/u1", headers={"authorization": "tok-b"}).status_code == 404
    r = client.delete("/catalog/datasets/u1", headers={"authorization": "tok-a"})
    assert r.status_code == 200 and r.json()["tombstone"] is True and r.json()["removed"]["versions"] == 2
    assert client.get("/catalog/datasets", headers={"authorization": "tok-a"}).json()["count"] == 0
    assert client.get("/catalog/datasets/u1", headers={"authorization": "tok-a"}).json()["tombstone"] is True


@pytest.mark.parametrize("edition", ["oss", "free", "professional"])
def test_share_answers_403_with_the_edition_sentence_outside_enterprise(seeded, edition):
    client = _client(caps=lambda: resolve_capabilities("avaloka_hosted", edition, has_commercial_build=True))
    r = client.post("/catalog/datasets/u1/share", headers={"authorization": "tok-a"})
    assert r.status_code == 403 and r.json()["detail"] == SHARE_SENTENCE
    assert "Enterprise" in SHARE_SENTENCE
    # and nothing changed
    assert client.get("/catalog/datasets/u1", headers={"authorization": "tok-a"}).json()["visibility"] == "personal"


def test_share_in_enterprise_says_the_shared_catalog_is_not_built_yet(seeded):
    client = _client(caps=lambda: resolve_capabilities("avaloka_hosted", "enterprise", has_commercial_build=True))
    r = client.post("/catalog/datasets/u1/share", headers={"authorization": "tok-a"})
    assert r.status_code == 501 and r.json()["detail"] == SHARE_NOT_BUILT


def test_an_unconfigured_catalog_is_503_not_an_empty_list(db_url, monkeypatch):
    monkeypatch.delenv("AVALOKA_LINEAGE_DB_URL")
    client = _client()
    r = client.get("/catalog/datasets", headers={"authorization": "tok-a"})
    assert r.status_code == 503

    def down(user_id):
        raise CatalogUnavailable("db down")
    client = _client(store_factory=down)
    assert client.get("/catalog/datasets/u1", headers={"authorization": "tok-a"}).status_code == 503


def test_the_server_module_wires_the_router():
    """The include_router line in app/api/server.py (the module itself needs
    the full stack to import, so the wiring is checked as text)."""
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2] / "app" / "api" / "server.py").read_text(encoding="utf-8")
    assert "from app.catalog.api import build_router as _build_catalog_router" in text
    assert "app.include_router(_build_catalog_router(_resolve_user_id))" in text
