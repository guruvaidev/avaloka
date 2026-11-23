"""The catalog on SQLite: the hermetic shape the design names for the CLI and
a one-process install. Postgres (row-level security, the trigger) is the
containers tier and is not exercised here; nothing in this directory claims
it is."""

from __future__ import annotations

import pytest

from app.catalog import store as catalog
from app.core import lineage


@pytest.fixture
def db_url(tmp_path, monkeypatch):
    url = "sqlite:///" + (tmp_path / "catalog.db").as_posix()
    lineage.reset_for_tests()
    catalog.reset_for_tests()
    # The hooks read the location from the environment, as the pods do; the
    # catalog follows lineage when it has no URL of its own.
    monkeypatch.setenv(lineage.ENV_DB_URL, url)
    monkeypatch.delenv(catalog.ENV_DB_URL, raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv(lineage.ENV_ENABLED, raising=False)
    yield url
    lineage.reset_for_tests()
    catalog.reset_for_tests()


@pytest.fixture
def events():
    return []


@pytest.fixture
def store(db_url, events):
    return catalog.CatalogStore("user-a", on_event=events.append)


@pytest.fixture
def other(db_url):
    return catalog.CatalogStore("user-b")


ORDERS = [("order_id", "int64"), ("customer_email", "object"), ("amount", "float64")]
