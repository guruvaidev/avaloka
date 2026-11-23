"""Resolve an owned 1.5.2 connection id to MCP credentials without logging secrets."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LegacyDatabaseCredentials:
    connection_id: str
    customer_id: str
    api_key: str
    database_type: str | None = None


class LegacyConnectionResolutionError(RuntimeError):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _owned_row(table: str, row_id: str, user_id: str) -> dict[str, Any] | None:
    if table == "mcp_connections":
        # Saved connections are not always in Supabase (local auth mode keeps
        # them in the API's own store); app.api.db_connections decides where.
        from app.api import db_connections

        if not db_connections.uses_supabase():
            try:
                return db_connections.owned_row(row_id, user_id)
            except Exception as exc:
                raise LegacyConnectionResolutionError(
                    "Unable to resolve the saved database connection.", status_code=502
                ) from exc
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        raise LegacyConnectionResolutionError(
            "Database connection lookup is unavailable because Supabase is not configured.",
            status_code=503,
        )
    try:
        from supabase import create_client  # type: ignore
    except ImportError as exc:
        raise LegacyConnectionResolutionError(
            "Database connection lookup requires the Supabase client.",
            status_code=503,
        ) from exc

    try:
        response = (
            create_client(url, key)
            .table(table)
            .select("*")
            .eq("id", row_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise LegacyConnectionResolutionError(
            "Unable to resolve the saved database connection.", status_code=502
        ) from exc
    rows = getattr(response, "data", None) or []
    row = rows[0] if isinstance(rows, list) and rows else None
    return row if isinstance(row, dict) else None


def _decrypt_api_key(row: dict[str, Any]) -> str:
    plaintext = str(row.get("api_key") or "").strip()
    if plaintext:
        return plaintext
    # Keep this resolver importable in the lightweight benchmark environment;
    # crypto is needed only when an encrypted legacy record is actually used.
    from app.api.cloud_connections import decrypt_secret

    return (
        decrypt_secret({
            "ciphertext": row.get("api_key_ciphertext"),
            "iv": row.get("api_key_iv"),
        })
        or ""
    )


def _resolve_sync(connection_id: str, user_id: str) -> LegacyDatabaseCredentials:
    requested_id = str(connection_id or "").strip()
    owner = str(user_id or "").strip()
    if not requested_id or not owner:
        raise LegacyConnectionResolutionError(
            "connection_id and authenticated user are required.", status_code=400
        )

    mcp_row = _owned_row("mcp_connections", requested_id, owner)
    mcp_id = requested_id
    if mcp_row is None:
        sql_row = _owned_row("sql_connections", requested_id, owner)
        if sql_row is None:
            raise LegacyConnectionResolutionError(
                "Saved database connection was not found.", status_code=404
            )
        mcp_id = str(sql_row.get("mcp_connection_id") or "").strip()
        if not mcp_id:
            raise LegacyConnectionResolutionError(
                "This legacy SQL connection is not linked to an MCP connection. "
                "Reconnect it from the database workspace before querying.",
                status_code=409,
            )
        mcp_row = _owned_row("mcp_connections", mcp_id, owner)

    if not mcp_row:
        raise LegacyConnectionResolutionError(
            "Linked MCP database connection was not found.", status_code=404
        )
    customer_id = str(mcp_row.get("customer_id") or "").strip()
    api_key = _decrypt_api_key(mcp_row)
    if not customer_id or not api_key:
        raise LegacyConnectionResolutionError(
            "Saved MCP credentials are unavailable or cannot be decrypted.",
            status_code=503,
        )
    return LegacyDatabaseCredentials(
        connection_id=requested_id,
        customer_id=customer_id,
        api_key=api_key,
        database_type=(
            str(mcp_row.get("database_type"))
            if mcp_row.get("database_type")
            else None
        ),
    )


async def resolve_legacy_database_credentials(
    connection_id: str, user_id: str
) -> LegacyDatabaseCredentials:
    return await asyncio.to_thread(_resolve_sync, connection_id, user_id)
