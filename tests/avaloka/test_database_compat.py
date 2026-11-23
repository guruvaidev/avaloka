"""Import-light checks for the 1.5.2 database workspace compatibility shim."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from app.api import legacy_database_connections as connections
from app.api.database_compat import (
    database_query_text,
    legacy_connect_fields,
    legacy_query_fields,
)


def test_legacy_query_alias_and_modern_content_precedence():
    assert database_query_text(None, " SELECT 1 ") == "SELECT 1"
    assert database_query_text(" SELECT 2 ", "SELECT 1") == "SELECT 2"
    assert database_query_text(None, None) == ""


def test_legacy_connect_response_has_exact_workspace_fields():
    assert legacy_connect_fields(
        session_id="session-1",
        thread_id="thread-1",
        connection_id="connection-1",
        expires_in=3600,
    ) == {
        "session_id": "session-1",
        "thread_id": "thread-1",
        "connection_id": "connection-1",
        "expires_in": 3600,
    }


def test_mcp_rows_are_projected_into_legacy_results_and_schema():
    fields = legacy_query_fields(
        {
            "columns": ["id", "active", "score", "profile"],
            "rows": [
                [7, True, 1.5, {"segment": "enterprise"}],
                [8, False, 2.0, None],
            ],
        },
        execution_time_ms=14,
    )

    assert fields["results"][0] == {
        "id": 7,
        "active": True,
        "score": 1.5,
        "profile": {"segment": "enterprise"},
    }
    assert fields["schema"] == [
        {"name": "id", "type": "integer"},
        {"name": "active", "type": "boolean"},
        {"name": "score", "type": "number"},
        {"name": "profile", "type": "json"},
    ]
    assert fields["row_count"] == 2
    assert fields["execution_time_ms"] == 14


def test_saved_mcp_connection_resolution_is_owner_scoped(monkeypatch):
    calls = []

    def owned_row(table, row_id, user_id):
        calls.append((table, row_id, user_id))
        return {
            "customer_id": "customer-1",
            "api_key": "secret",
            "database_type": "mongodb",
        }

    monkeypatch.setattr(connections, "_owned_row", owned_row)
    resolved = connections._resolve_sync("mcp-1", "user-1")

    assert calls == [("mcp_connections", "mcp-1", "user-1")]
    assert resolved.connection_id == "mcp-1"
    assert resolved.customer_id == "customer-1"
    assert resolved.api_key == "secret"
    assert resolved.database_type == "mongodb"


def test_supabase_service_lookup_always_filters_by_authenticated_owner(monkeypatch):
    filters = []

    class Query:
        def table(self, name):
            filters.append(("table", name))
            return self

        def select(self, fields):
            return self

        def eq(self, field, value):
            filters.append((field, value))
            return self

        def limit(self, value):
            return self

        def execute(self):
            return SimpleNamespace(data=[])

    monkeypatch.setenv("SUPABASE_URL", "https://supabase.invalid")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "configured")
    monkeypatch.setitem(
        sys.modules,
        "supabase",
        SimpleNamespace(create_client=lambda url, key: Query()),
    )

    assert connections._owned_row("mcp_connections", "mcp-1", "user-1") is None
    assert ("id", "mcp-1") in filters
    assert ("user_id", "user-1") in filters


def test_sql_connection_follows_owned_mcp_link(monkeypatch):
    rows = {
        ("mcp_connections", "sql-1"): None,
        ("sql_connections", "sql-1"): {"mcp_connection_id": "mcp-1"},
        ("mcp_connections", "mcp-1"): {
            "customer_id": "customer-1",
            "api_key_ciphertext": "ciphertext",
            "api_key_iv": "iv",
        },
    }
    calls = []

    def owned_row(table, row_id, user_id):
        calls.append((table, row_id, user_id))
        return rows[(table, row_id)]

    monkeypatch.setattr(connections, "_owned_row", owned_row)
    monkeypatch.setattr(connections, "_decrypt_api_key", lambda row: "decrypted")

    resolved = connections._resolve_sync("sql-1", "user-1")

    assert calls == [
        ("mcp_connections", "sql-1", "user-1"),
        ("sql_connections", "sql-1", "user-1"),
        ("mcp_connections", "mcp-1", "user-1"),
    ]
    assert resolved.connection_id == "sql-1"
    assert resolved.api_key == "decrypted"


def test_unlinked_direct_sql_connection_fails_explicitly(monkeypatch):
    def owned_row(table, row_id, user_id):
        if table == "sql_connections":
            return {"mcp_connection_id": None}
        return None

    monkeypatch.setattr(connections, "_owned_row", owned_row)

    with pytest.raises(connections.LegacyConnectionResolutionError) as caught:
        connections._resolve_sync("sql-1", "user-1")

    assert caught.value.status_code == 409
    assert "not linked to an MCP connection" in str(caught.value)
