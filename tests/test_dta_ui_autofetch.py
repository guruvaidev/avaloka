"""Regression tests: auto-fetching a UI-registered customer for a transfer
must work whether /customers returns a bare list or {"customers": [...]},
and a db_type lookup failure must not discard already-fetched credentials.

The customer is one the user owns but is not analysing. The planner asks the
credentials route as the signed-in user (an avu1 assertion; merge of
develop-1.7 into integration/2.0, decision 1), never with the database's
api_key, and the route answers with a sealed ``credential_handle`` instead of
a password -- to the owner only.
"""
import hashlib
import secrets
import sys
import types

import pytest

import app.agents.planner as planner_mod
from app.agents.planner import InitiateTransferParams, _handle_initiate_transfer

try:  # tests/ as a package, or tests/ on sys.path
    from tests.onboarding_identity import assertion_user, credentials_status
except ImportError:
    from onboarding_identity import assertion_user, credentials_status

ALIAS = "ui_customer_1"
OWNER = "user-ui"
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

    calls = []

    def get(self, url, **kwargs):
        headers = dict(kwargs.get("headers") or {})
        FakeSession.calls.append((url, headers))
        if url.endswith(f"/customers/{ALIAS}/credentials"):
            # As the real route answers: the owner's assertion only.
            status = credentials_status(headers, OWNER)
            if status != 200:
                return FakeResponse(status, {})
        for suffix, response in self.routes.items():
            if url.endswith(suffix):
                return response
        return FakeResponse(404, {})


@pytest.fixture
def sealed(monkeypatch):
    """A throwaway deployment key, the user's handle for ALIAS, and the route's payload.

    Everything is generated here; no real key, handle or password is a fixture.
    """
    import app.api.cloud_connections as cc
    from app.mcp_server.credentials import credential_handle

    key = secrets.token_hex(16)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(key.encode()).digest(), raising=False)
    api_key, password = "ak_" + secrets.token_hex(8), secrets.token_hex(12)
    creds = {
        "host": "db.example.com", "port": 3306, "database": "sales",
        "username": "u", "table": "orders",
        "credential_handle": credential_handle(f"mysql://u:{password}@db.example.com:3306/sales"),
    }
    return types.SimpleNamespace(
        api_key=api_key, password=password, creds=creds,
        handles={ALIAS: cc.encrypt_secret(api_key)},
    )


@pytest.fixture
def stubbed_transfer(monkeypatch, sealed):
    FakeSession.calls = []
    monkeypatch.setattr(planner_mod.requests, "Session", FakeSession)
    # A name that is not a database falls through to the user's cloud
    # connections (Supabase); none here, and no network.
    monkeypatch.setattr(planner_mod, "_resolve_cloud_conn_by_name", lambda *a, **k: None)

    fake_dta = types.ModuleType("app.agents.data_transfer_agent.data_transfer_agent")
    fake_dta.data_transfer_pipeline = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("pipeline stubbed out in tests")
    )
    monkeypatch.setitem(
        sys.modules, "app.agents.data_transfer_agent.data_transfer_agent", fake_dta
    )


def _transfer(source=ALIAS, handles=None, user_id=OWNER):
    state = {"user_id": user_id, "dta_database_registry": {
        "Warehouse": {"db_type": "mysql", "host": "h", "port": 3306, "database": "d"},
    }, "db_credential_handles": dict(handles or {})}
    msg = _handle_initiate_transfer(
        state,
        InitiateTransferParams(
            source_alias=source,
            destination_alias="Warehouse",
            user_prompt=f"transfer from {source} to Warehouse",
        ),
    )
    return state, msg


def _authenticated_as_the_user(sealed):
    cred_calls = [h for url, h in FakeSession.calls if url.endswith(f"/customers/{ALIAS}/credentials")]
    return (bool(cred_calls) and all(assertion_user(h) == OWNER for h in cred_calls)
            and sealed.api_key not in repr(FakeSession.calls))


def test_bare_list_customers_response_is_accepted(stubbed_transfer, sealed):
    FakeSession.routes = {
        f"/customers/{ALIAS}/credentials": FakeResponse(200, sealed.creds),
        "/customers": FakeResponse(200, [CUSTOMER_ROW]),
    }
    state, msg = _transfer(handles=sealed.handles)
    assert "not registered" not in msg
    assert state["dta_database_registry"][ALIAS]["db_type"] == "mysql"
    assert state["dta_database_registry"][ALIAS]["host"] == "db.example.com"
    # The entry is a handle: the password stays sealed until the transfer runs.
    assert "password" not in state["dta_database_registry"][ALIAS]
    assert state["dta_database_registry"][ALIAS]["conn_id"] == ALIAS
    assert _authenticated_as_the_user(sealed)


def test_wrapped_dict_customers_response_still_works(stubbed_transfer, sealed):
    FakeSession.routes = {
        f"/customers/{ALIAS}/credentials": FakeResponse(200, sealed.creds),
        "/customers": FakeResponse(200, {"customers": [CUSTOMER_ROW]}),
    }
    state, msg = _transfer(handles=sealed.handles)
    assert "not registered" not in msg
    assert state["dta_database_registry"][ALIAS]["db_type"] == "mysql"
    assert _authenticated_as_the_user(sealed)


def test_db_type_lookup_failure_keeps_fetched_credentials(stubbed_transfer, sealed):
    FakeSession.routes = {
        f"/customers/{ALIAS}/credentials": FakeResponse(200, sealed.creds),
        "/customers": FakeResponse(200, ValueError("invalid json")),
    }
    state, msg = _transfer(handles=sealed.handles)
    assert "not registered" not in msg
    assert state["dta_database_registry"][ALIAS]["db_type"] == "unknown"
    assert state["dta_database_registry"][ALIAS]["host"] == "db.example.com"
    # The entry is a handle: the password stays sealed until the transfer runs.
    assert "password" not in state["dta_database_registry"][ALIAS]
    assert state["dta_database_registry"][ALIAS]["conn_id"] == ALIAS


def test_missing_customer_is_still_reported_unregistered(stubbed_transfer, sealed):
    FakeSession.routes = {
        f"/customers/{ALIAS}/credentials": FakeResponse(404, {}),
        "/customers": FakeResponse(200, []),
    }
    _state, msg = _transfer(handles=sealed.handles)
    assert "not registered" in msg


def test_another_users_customer_is_never_resolved(stubbed_transfer, sealed):
    """Even holding ALIAS's api_key handle, another user gets no credentials: ownership, not possession."""
    FakeSession.routes = {
        f"/customers/{ALIAS}/credentials": FakeResponse(200, sealed.creds),
        "/customers": FakeResponse(200, [CUSTOMER_ROW]),
    }
    state, msg = _transfer(handles=sealed.handles, user_id="user-other")
    assert "not registered" in msg
    assert ALIAS not in state["dta_database_registry"]
    assert sealed.api_key not in repr(FakeSession.calls)


def test_without_a_signed_in_user_nothing_is_fetched(stubbed_transfer, sealed):
    FakeSession.routes = {
        f"/customers/{ALIAS}/credentials": FakeResponse(200, sealed.creds),
        "/customers": FakeResponse(200, [CUSTOMER_ROW]),
    }
    _state, msg = _transfer(handles=sealed.handles, user_id=None)
    assert "not registered" in msg
    assert not [url for url, _h in FakeSession.calls if "/credentials" in url]
