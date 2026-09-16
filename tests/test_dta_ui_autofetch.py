"""Regression tests: auto-fetching a UI-registered customer for a transfer
must work whether /customers returns a bare list or {"customers": [...]},
and a db_type lookup failure must not discard already-fetched credentials.
"""
import sys
import types

import pytest

import app.agents.planner as planner_mod
from app.agents.planner import InitiateTransferParams, _handle_initiate_transfer

ALIAS = "ui_customer_1"
CREDS = {
    "host": "db.example.com", "port": 3306, "database": "sales",
    "username": "u", "password": "p", "table": "orders",
}
CUSTOMER_ROW = {"customer_id": ALIAS, "database_type": "mysql"}


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    routes = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, **_kwargs):
        for suffix, response in self.routes.items():
            if url.endswith(suffix):
                return response
        return FakeResponse(404, {})


@pytest.fixture
def stubbed_transfer(monkeypatch):
    monkeypatch.setattr(planner_mod.requests, "Session", FakeSession)

    fake_dta = types.ModuleType("app.agents.data_transfer_agent.data_transfer_agent")
    fake_dta.data_transfer_pipeline = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("pipeline stubbed out in tests")
    )
    monkeypatch.setitem(
        sys.modules, "app.agents.data_transfer_agent.data_transfer_agent", fake_dta
    )


def _transfer(source=ALIAS):
    state = {"dta_database_registry": {
        "Warehouse": {"db_type": "mysql", "host": "h", "port": 3306, "database": "d"},
    }}
    msg = _handle_initiate_transfer(
        state,
        InitiateTransferParams(
            source_alias=source,
            destination_alias="Warehouse",
            user_prompt=f"transfer from {source} to Warehouse",
        ),
    )
    return state, msg


def test_bare_list_customers_response_is_accepted(stubbed_transfer):
    FakeSession.routes = {
        f"/customers/{ALIAS}/credentials": FakeResponse(200, CREDS),
        "/customers": FakeResponse(200, [CUSTOMER_ROW]),
    }
    state, msg = _transfer()
    assert "not registered" not in msg
    assert state["dta_database_registry"][ALIAS]["db_type"] == "mysql"
    assert state["dta_database_registry"][ALIAS]["host"] == "db.example.com"


def test_wrapped_dict_customers_response_still_works(stubbed_transfer):
    FakeSession.routes = {
        f"/customers/{ALIAS}/credentials": FakeResponse(200, CREDS),
        "/customers": FakeResponse(200, {"customers": [CUSTOMER_ROW]}),
    }
    state, msg = _transfer()
    assert "not registered" not in msg
    assert state["dta_database_registry"][ALIAS]["db_type"] == "mysql"


def test_db_type_lookup_failure_keeps_fetched_credentials(stubbed_transfer):
    FakeSession.routes = {
        f"/customers/{ALIAS}/credentials": FakeResponse(200, CREDS),
        "/customers": FakeResponse(200, ValueError("invalid json")),
    }
    state, msg = _transfer()
    assert "not registered" not in msg
    assert state["dta_database_registry"][ALIAS]["db_type"] == "unknown"
    assert state["dta_database_registry"][ALIAS]["host"] == "db.example.com"


def test_missing_customer_is_still_reported_unregistered(stubbed_transfer):
    FakeSession.routes = {
        f"/customers/{ALIAS}/credentials": FakeResponse(404, {}),
        "/customers": FakeResponse(200, []),
    }
    _state, msg = _transfer()
    assert "not registered" in msg
