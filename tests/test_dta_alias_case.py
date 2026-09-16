"""Regression tests: chat-registered database aliases must resolve
case-insensitively. Registry keys keep their registration casing, but the
user (or the LLM) may type the alias in any case in a transfer request.
"""
import sys
import types

import pytest

import app.agents.planner as planner_mod
from app.agents.planner import (
    InitiateTransferParams,
    RegisterDatabaseParams,
    _handle_initiate_transfer,
    _handle_register_database,
    _resolve_registry_alias,
)

MYSQL_ENTRY = {"db_type": "mysql", "host": "h", "port": 3306, "database": "d"}


# -------------------------
# Resolver unit behavior
# -------------------------

def test_exact_match_wins():
    assert _resolve_registry_alias({"Prod_DB": {}}, "Prod_DB") == "Prod_DB"


def test_case_insensitive_match():
    assert _resolve_registry_alias({"Prod_DB": {}}, "prod_db") == "Prod_DB"
    assert _resolve_registry_alias({"Prod_DB": {}}, "PROD_DB") == "Prod_DB"


def test_missing_alias_returns_none():
    assert _resolve_registry_alias({"Prod_DB": {}}, "warehouse") is None
    assert _resolve_registry_alias({}, "anything") is None


# -------------------------
# Registration: case variants update the existing entry, no duplicates
# -------------------------

def test_reregistering_case_variant_does_not_duplicate():
    state = {"dta_database_registry": {"Prod_DB": dict(MYSQL_ENTRY)}}
    _handle_register_database(
        state,
        RegisterDatabaseParams(alias="prod_db", db_type="mysql", host="h2", port=3306, database="d"),
    )
    registry = state["dta_database_registry"]
    assert list(registry.keys()) == ["Prod_DB"]
    assert registry["Prod_DB"]["host"] == "h2"


# -------------------------
# Transfer lookup accepts any casing of a registered alias
# -------------------------

@pytest.fixture
def offline_transfer(monkeypatch):
    # No network, and no real transfer pipeline import side effects.
    class NoNetworkSession:
        def __enter__(self):
            raise RuntimeError("network disabled in tests")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(planner_mod.requests, "Session", NoNetworkSession)

    fake_dta = types.ModuleType("app.agents.data_transfer_agent.data_transfer_agent")

    def _no_pipeline(*_a, **_k):
        raise RuntimeError("pipeline stubbed out in tests")

    fake_dta.data_transfer_pipeline = _no_pipeline
    monkeypatch.setitem(
        sys.modules, "app.agents.data_transfer_agent.data_transfer_agent", fake_dta
    )


def _transfer(state, source, dest):
    return _handle_initiate_transfer(
        state,
        InitiateTransferParams(
            source_alias=source,
            destination_alias=dest,
            user_prompt=f"transfer from {source} to {dest}",
        ),
    )


def test_mixed_case_aliases_pass_the_registry_gate(offline_transfer):
    state = {"dta_database_registry": {
        "Prod_DB": dict(MYSQL_ENTRY),
        "Warehouse": dict(MYSQL_ENTRY),
    }}
    msg = _transfer(state, "prod_db", "WAREHOUSE")
    assert "not registered" not in msg


def test_unregistered_alias_is_still_reported(offline_transfer):
    state = {"dta_database_registry": {"Prod_DB": dict(MYSQL_ENTRY)}}
    msg = _transfer(state, "prod_db", "Warehouse")
    assert "not registered" in msg
    assert "Warehouse" in msg or "warehouse" in msg
