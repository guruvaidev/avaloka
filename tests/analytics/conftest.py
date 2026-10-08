"""Shared fixtures: an in-memory store, a tiny app, and a clean environment."""
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.analytics import config
from app.analytics.collector import build_router
from app.analytics.schema import SCHEMA_VERSION
from app.analytics.store import Store

_ENV = (config.ENV_ENABLED, config.ENV_UMBRELLA, config.ENV_PROMPT_TEXT, config.ENV_RETENTION_DAYS,
        config.ENV_DB_URL, config.ENV_EXPORT, config.ENV_EXPORT_ENDPOINT)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # tests/conftest.py on some branches sets AVALOKA_TELEMETRY=off for the whole
    # suite; these tests must start from "nothing set" to mean anything.
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def store():
    return Store("sqlite:///:memory:")


def _resolver(request):
    return request.headers.get("X-Test-User") or None


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(build_router(_resolver, store=store))
    return TestClient(app)


def eid():
    return uuid.uuid4().hex


def ev(event_type, seq=0, **fields):
    return {"event_id": eid(), "event_type": event_type, "seq": seq, "t_ms": seq * 10, **fields}


def batch(*events, session_id=None):
    return {"schema_version": SCHEMA_VERSION, "session_id": session_id or eid(), "events": list(events)}


USER = {"X-Test-User": "user-7f3a@acme.example"}
