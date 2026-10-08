"""Where analytics rows go, and what happens when nowhere is configured.

Resolution is ``AVALOKA_ANALYTICS_DB_URL``, else ``POSTGRES_URL``, else analytics
is off. There is no local-file default: these tests fail if one comes back.
"""
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.analytics import config, export, store as store_mod
from app.analytics.collector import build_router
from tests.analytics.conftest import USER, _resolver, batch, ev


@pytest.fixture(autouse=True)
def fresh_default(monkeypatch):
    """No inherited database variable and no store cached by an earlier test."""
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.setattr(store_mod, "_default", None)
    monkeypatch.setattr(store_mod, "_announced", False)


@pytest.fixture
def default_client():
    """The router as app/api/server.py mounts it: no store passed in."""
    app = FastAPI()
    app.include_router(build_router(_resolver))
    return TestClient(app)


@pytest.fixture
def no_engine(monkeypatch):
    """Any attempt to open a database fails the test."""
    def refuse(url, **kwargs):
        raise AssertionError(f"create_engine called with no database configured: {url!r}")
    monkeypatch.setattr(store_mod, "create_engine", refuse)


def _analytics_log(caplog):
    return [r.getMessage() for r in caplog.records if r.name == store_mod.__name__]


# -- nothing configured: off, said once, nothing opened ----------------------

def test_no_database_variable_means_no_url(no_engine):
    assert config.database_source() is None
    assert config.database_url() is None
    assert store_mod.default_store() is None


def test_store_without_a_url_refuses_rather_than_inventing_one(no_engine):
    with pytest.raises(store_mod.NotConfigured):
        store_mod.Store()


def test_unconfigured_collector_is_off_and_opens_nothing(default_client, no_engine):
    cfg = default_client.get("/analytics/config", headers=USER)
    assert cfg.status_code == 200
    assert (cfg.json()["enabled"], cfg.json()["reason"]) == (False, "no_store")

    sent = default_client.post("/analytics/events", json=batch(ev("page.viewed", route="/")), headers=USER)
    assert sent.status_code == 200
    assert sent.json() == {"enabled": False, "reason": "no_store", "accepted": 0}

    # An opt-out that cannot be kept is not reported as kept.
    opt = default_client.put("/analytics/opt-out", json={"opted_out": True}, headers=USER)
    assert opt.status_code == 503 and opt.json()["reason"] == "no_store"


def test_unconfigured_is_logged_once_not_per_request(default_client, no_engine, caplog):
    caplog.set_level(logging.INFO, logger=store_mod.__name__)
    for _ in range(3):
        default_client.get("/analytics/config", headers=USER)
        default_client.post("/analytics/events", json=batch(ev("page.viewed", route="/")), headers=USER)
    lines = _analytics_log(caplog)
    assert len(lines) == 1, lines
    assert "disabled" in lines[0] and config.ENV_DB_URL in lines[0] and "POSTGRES_URL" in lines[0]


def test_unconfigured_export_reports_and_sends_nothing(no_engine, monkeypatch):
    monkeypatch.setenv(config.ENV_EXPORT, "on")
    monkeypatch.setenv(config.ENV_EXPORT_ENDPOINT, "https://ingest.invalid/v1")
    calls = []
    report = export.export_pending(post=lambda *a, **k: calls.append(a) or 200)
    assert report == {"exported": 0, "reason": "no store configured"}
    assert calls == []


def test_configuring_a_database_later_turns_it_on_without_a_restart(default_client, monkeypatch, tmp_path):
    assert default_client.get("/analytics/config", headers=USER).json()["reason"] == "no_store"
    monkeypatch.setenv(config.ENV_DB_URL, f"sqlite:///{tmp_path / 'later.db'}")
    assert default_client.get("/analytics/config", headers=USER).json()["enabled"] is True


# -- an operator names SQLite: honoured, and labelled as their choice --------

def test_explicit_sqlite_url_is_used_and_logged_as_a_choice(default_client, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=store_mod.__name__)
    path = tmp_path / "chosen.db"
    monkeypatch.setenv(config.ENV_DB_URL, f"sqlite:///{path}")

    sent = default_client.post("/analytics/events", json=batch(ev("page.viewed", route="/")), headers=USER)
    assert sent.status_code == 202 and sent.json()["accepted"] == 1
    default_client.get("/analytics/config", headers=USER)

    chosen = store_mod.default_store()
    assert chosen.engine.dialect.name == "sqlite" and chosen.count() == 1
    assert path.exists()

    lines = _analytics_log(caplog)
    assert len(lines) == 1, lines
    assert "SQLite" in lines[0] and config.ENV_DB_URL in lines[0]
    assert "explicit" in lines[0] and "not a fallback" in lines[0]
    assert "disabled" not in lines[0]
    # The variable is named; its value, which can carry credentials, is not.
    assert str(path) not in lines[0]


# -- precedence --------------------------------------------------------------

def test_analytics_variable_wins_over_postgres_url(monkeypatch):
    monkeypatch.setenv("POSTGRES_URL", "postgresql://deploy/db")
    assert config.database_source() == ("POSTGRES_URL", "postgresql://deploy/db")
    monkeypatch.setenv(config.ENV_DB_URL, "postgresql://analytics/db")
    assert config.database_source() == (config.ENV_DB_URL, "postgresql://analytics/db")
    monkeypatch.setenv(config.ENV_DB_URL, "   ")
    assert config.database_url() == "postgresql://deploy/db"
