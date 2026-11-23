"""Saved database connections: one API route family for every auth mode.

The web UI used to read and write the ``mcp_connections`` table directly with
supabase-js. In local auth mode (``AVALOKA_AUTH_MODE=local``) there is no
Supabase database, so the connect modal's insert answered 501 *after* the MCP
registration had succeeded: the documented "Connect the sample database" step
ended at "Saving connection failed" and left an orphan registration that a
retry reported as "already registered".

Now the UI talks only to these routes, the same in every mode:

* ``GET    /api/connections``       the caller's connections, no secrets
* ``POST   /api/connections``       register with the onboarding API, then save
* ``PUT    /api/connections/{id}``  rename / mark active or inactive
* ``DELETE /api/connections/{id}``  deactivate the registration, then delete

Register-then-save is one server-side step. If the save fails, the
registration is deactivated through the same signed user assertion that
created it (``app.mcp_server.client.user_headers``), and the onboarding API
accepts a fresh registration over its owner's inactive one, so a retry works.

Where rows live is decided in one place, :func:`uses_supabase`: a hosted
deployment with a Supabase service role keeps the existing ``mcp_connections``
table (its rows, RLS and the legacy resolver stay valid); anything else, and
always local auth mode, uses a JSON file in the API's data directory, next to
the onboarding registry on the same kind of durable volume. Redis is not used
for this: the quickstart runs it without persistence, and a connection list
that forgot its rows on restart while the registry kept them would recreate
the orphan this module exists to prevent.

Ownership is the #544 rule: a connection that is not the caller's is 404, the
same as one that does not exist. Secrets never leave: the api_key and the
database username are stored only as AES-GCM ciphertext, the password is
forwarded to the onboarding API and never stored here, and every response is
built from :data:`PUBLIC_FIELDS`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.api.cloud_connections import encrypt_secret
from app.core import local_auth
from app.core.db_sampling import ANALYSIS_MAX_ROWS

logger = logging.getLogger(__name__)

UserResolver = Callable[[Request], Optional[str]]

SUPABASE_TABLE_ENV = "SUPABASE_MCP_CONNECTIONS_TABLE"
FILE_ENV = "AVALOKA_CONNECTIONS_FILE"

# What a client may see. Everything else on a row (api_key*, username*,
# contact_email, user_id) stays on the server.
PUBLIC_FIELDS = (
    "id", "connection_name", "customer_id", "customer_name", "database_type",
    "host", "port", "database_name", "mcp_endpoint", "status",
    "query_timeout", "max_rows", "created_at",
)

ONBOARDING_TIMEOUT_S = 60.0


# --------------------------------------------------------------------------- #
# Stores
# --------------------------------------------------------------------------- #

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SupabaseConnectionStore:
    """The hosted ``mcp_connections`` table, through the service role."""

    def __init__(self, table: Optional[str] = None):
        self.table = table or os.getenv(SUPABASE_TABLE_ENV, "mcp_connections")

    def _t(self):
        from app.agents.sampling_persistence import get_supabase_client
        return get_supabase_client().table(self.table)

    def list_for(self, user_id: str) -> List[Dict[str, Any]]:
        res = self._t().select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return list(getattr(res, "data", None) or [])

    def get(self, connection_id: str) -> Optional[Dict[str, Any]]:
        res = self._t().select("*").eq("id", connection_id).limit(1).execute()
        data = getattr(res, "data", None) or []
        return data[0] if data else None

    def insert(self, row: Dict[str, Any]) -> Dict[str, Any]:
        res = self._t().insert(row).execute()
        data = getattr(res, "data", None) or []
        if not data:
            raise RuntimeError("insert returned no row")
        return data[0]

    def update(self, connection_id: str, patch: Dict[str, Any]) -> None:
        self._t().update(patch).eq("id", connection_id).execute()

    def delete(self, connection_id: str) -> None:
        self._t().delete().eq("id", connection_id).execute()


class FileConnectionStore:
    """Rows in one JSON file, written atomically, mode 600. One process (local mode)."""

    _lock = threading.Lock()

    def __init__(self, path: Path):
        self.path = Path(path)

    def _read(self) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        rows = data.get("connections") if isinstance(data, dict) else None
        return rows if isinstance(rows, dict) else {}

    def _write(self, rows: Dict[str, Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"connections": rows}, f, indent=2)
        os.replace(tmp, self.path)

    def list_for(self, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = [r for r in self._read().values() if r.get("user_id") == user_id]
        return sorted(rows, key=lambda r: str(r.get("created_at") or ""), reverse=True)

    def get(self, connection_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._read().get(connection_id)
        return dict(row) if row else None

    def insert(self, row: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            rows = self._read()
            saved = {**row, "id": str(uuid.uuid4()), "created_at": _now_iso()}
            rows[saved["id"]] = saved
            self._write(rows)
        return dict(saved)

    def update(self, connection_id: str, patch: Dict[str, Any]) -> None:
        with self._lock:
            rows = self._read()
            if connection_id in rows:
                rows[connection_id] = {**rows[connection_id], **patch, "updated_at": _now_iso()}
                self._write(rows)

    def delete(self, connection_id: str) -> None:
        with self._lock:
            rows = self._read()
            if rows.pop(connection_id, None) is not None:
                self._write(rows)


def uses_supabase() -> bool:
    """The one switch: hosted Supabase table, or the API's own file.

    Local auth mode never has Supabase (local_auth refuses to run beside a
    SUPABASE_URL), so it always gets the file.
    """
    if local_auth.active_config() is not None:
        return False
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


def connections_file() -> Path:
    explicit = os.getenv(FILE_ENV)
    if explicit:
        return Path(explicit)
    return Path(os.getenv("DATA_DIR") or (Path.cwd() / "data")) / "db_connections.json"


def connection_store():
    return SupabaseConnectionStore() if uses_supabase() else FileConnectionStore(connections_file())


def public_view(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: row.get(k) for k in PUBLIC_FIELDS}


def owned_row(connection_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """The caller's own row, or ``None`` (missing and foreign look the same)."""
    row = connection_store().get(connection_id)
    if not row or not user_id or str(row.get("user_id") or "") != str(user_id):
        return None
    return row


# --------------------------------------------------------------------------- #
# Onboarding API (as the authenticated user, #544)
# --------------------------------------------------------------------------- #

async def _onboarding(method: str, path: str, user_id: str,
                      body: Optional[bytes] = None) -> Tuple[int, Dict[str, Any]]:
    """Call the onboarding API as ``user_id``. Never logs the body (it carries the password)."""
    import app.mcp_server.client as onboarding_client

    headers = onboarding_client.user_headers(user_id)
    if not headers:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "DB_ENCRYPTION_KEY is not configured; cannot register a database.",
        )
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method, f"{onboarding_client.ONBOARDING_API_URL}{path}",
                content=body, headers=headers, timeout=ONBOARDING_TIMEOUT_S,
            )
    except httpx.HTTPError as exc:
        logger.warning("[connections] onboarding API unreachable (%s %s): %s",
                       method, path.split("/")[1] if "/" in path else path, type(exc).__name__)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Registration service unavailable")
    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("detail", f"Registration service answered HTTP {resp.status_code}")
    return resp.status_code, payload


async def _deactivate_registration(customer_id: str, user_id: str) -> Optional[str]:
    """Deactivate one registration. ``None`` on success, else why it failed.

    404 counts as done: the registration is gone, or was never this user's
    (a row saved before owners were recorded), and either way nothing the
    caller owns is left active.
    """
    try:
        code, _payload = await _onboarding("DELETE", f"/customers/{quote(customer_id, safe='')}", user_id)
    except HTTPException as exc:
        return str(exc.detail)
    if 200 <= code < 300 or code == 404:
        return None
    return f"HTTP {code}"


class RegistrationRejected(Exception):
    """The onboarding API refused a registration (its status code and detail)."""

    def __init__(self, status_code: int, detail: Any):
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


async def register_and_save(user_id: str, body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Register a database with the onboarding API as ``user_id``, then save the connection.

    The one path a database's credentials take into the sealed store, for the
    connect modal and for a connection typed into chat alike: the onboarding
    API tests the connection and keeps its connection string sealed; the saved
    row keeps the api_key and username sealed; the password is stored nowhere
    else. Returns ``(saved row, sealed api_key)`` -- the sealed api_key is the
    handle a data transfer opens the connection with.

    Raises :class:`RegistrationRejected` for the onboarding API's refusal, and
    ``HTTPException`` (after rolling the registration back) when saving fails.
    """
    connection_name = str(body.get("connection_name") or "").strip()
    registration = {k: v for k, v in body.items() if k != "connection_name"}
    # The registration's max_rows caps every query MCP runs for it, the
    # analysis read included. Left out, the onboarding default (1,000) would
    # turn a 2,000-row table into a sample; ask for the analysis cap instead.
    if _int(registration.get("max_rows"), 0) <= 0:
        registration["max_rows"] = ANALYSIS_MAX_ROWS
    code, reg = await _onboarding(
        "POST", "/customers/register", user_id, json.dumps(registration).encode("utf-8"),
    )
    if not 200 <= code < 300:
        raise RegistrationRejected(code, reg.get("detail"))

    customer_id = str(reg.get("customer_id") or "")
    api_key_enc = encrypt_secret(str(reg.get("api_key") or ""))
    username_enc = encrypt_secret(str(body.get("username") or ""))

    async def _fail(reason: str):
        problem = await _deactivate_registration(customer_id, user_id)
        if problem is None:
            tail = "The database registration was rolled back; you can try again."
        else:
            logger.error("[connections] rollback of %s failed: %s", customer_id, problem)
            tail = (f"Rolling back the database registration also failed ({problem}); "
                    f"delete the registration for '{customer_id}' before retrying.")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Saving connection failed: {reason}. {tail}")

    if not customer_id or not api_key_enc:
        await _fail("the API could not seal the database credentials (DB_ENCRYPTION_KEY)")

    row: Dict[str, Any] = {
        "user_id": user_id,
        "connection_name": connection_name,
        "customer_id": customer_id,
        "customer_name": str(body.get("customer_name") or "").strip(),
        "contact_email": str(body.get("contact_email") or "").strip(),
        "database_type": str(reg.get("database_type") or body.get("database_type") or ""),
        "host": str(body.get("host") or "").strip(),
        "port": _int(body.get("port"), 0),
        "database_name": str(body.get("database_name") or "").strip(),
        # Plaintext columns are NOT NULL in the hosted schema; they stay empty.
        "username": "",
        "api_key": "",
        "mcp_endpoint": str(reg.get("mcp_endpoint") or ""),
        "status": str(reg.get("status") or "active"),
        "query_timeout": _int(body.get("query_timeout"), 30),
        "max_rows": _int(registration.get("max_rows"), ANALYSIS_MAX_ROWS),
        "api_key_ciphertext": api_key_enc["ciphertext"],
        "api_key_iv": api_key_enc["iv"],
    }
    if username_enc:
        row["username_ciphertext"] = username_enc["ciphertext"]
        row["username_iv"] = username_enc["iv"]

    try:
        saved = await asyncio.to_thread(lambda: connection_store().insert(row))
    except Exception as exc:
        logger.error("[connections] save failed for %s: %s", customer_id, type(exc).__name__)
        await _fail("the connection could not be stored")
    logger.info("[connections] saved %s for %s", saved.get("id"), customer_id)
    return saved, api_key_enc


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_router(resolve_user_id: UserResolver) -> APIRouter:
    router = APIRouter(prefix="/api/connections", tags=["connections"])

    def _user(request: Request) -> str:
        user_id = resolve_user_id(request)
        if not user_id:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
        return str(user_id)

    async def _owned_or_404(connection_id: str, user_id: str) -> Dict[str, Any]:
        try:
            row = await asyncio.to_thread(owned_row, connection_id, user_id)
        except Exception as exc:
            logger.error("[connections] lookup failed: %s", type(exc).__name__)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Connection lookup failed")
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Connection not found")
        return row

    @router.get("")
    async def list_connections(request: Request):
        user_id = _user(request)
        try:
            rows = await asyncio.to_thread(lambda: connection_store().list_for(user_id))
        except Exception as exc:
            logger.error("[connections] list failed: %s", type(exc).__name__)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not load connections")
        return {"connections": [public_view(r) for r in rows if r.get("user_id") == user_id]}

    @router.post("", status_code=status.HTTP_201_CREATED)
    async def create_connection(request: Request):
        user_id = _user(request)
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Expected a JSON object.")
        connection_name = str(body.get("connection_name") or "").strip()
        if not connection_name:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "connection_name is required.")

        try:
            saved, _api_key_enc = await register_and_save(user_id, body)
        except RegistrationRejected as exc:
            # 409 "already registered", 422 validation, 400 connection test:
            # the onboarding API's answer, as the modal already renders it.
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return public_view(saved)

    @router.put("/{connection_id}")
    async def update_connection(connection_id: str, request: Request):
        user_id = _user(request)
        await _owned_or_404(connection_id, user_id)
        try:
            body = await request.json()
        except ValueError:
            body = None
        body = body if isinstance(body, dict) else {}
        patch: Dict[str, Any] = {}
        name = str(body.get("connection_name") or "").strip()
        if name:
            patch["connection_name"] = name
        if body.get("status") in ("active", "inactive"):
            patch["status"] = body["status"]
        if not patch:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "Provide connection_name or status (active / inactive).")
        await asyncio.to_thread(lambda: connection_store().update(connection_id, patch))
        return public_view(await _owned_or_404(connection_id, user_id))

    @router.delete("/{connection_id}")
    async def delete_connection(connection_id: str, request: Request):
        user_id = _user(request)
        row = await _owned_or_404(connection_id, user_id)
        customer_id = str(row.get("customer_id") or "")
        if customer_id:
            problem = await _deactivate_registration(customer_id, user_id)
            if problem is not None:
                # Keep the row: deleting it would leave an active registration
                # nobody can see, which is the orphan this module prevents.
                raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                                    f"Could not deactivate the database registration ({problem}); "
                                    "the connection was kept. Try again.")
        await asyncio.to_thread(lambda: connection_store().delete(connection_id))
        return {"deleted": connection_id}

    return router
