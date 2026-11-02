"""The sealed store's one entrance for a database typed into chat (1.6 port of 2.0's).

2.0 keeps this module for the connect modal and chat alike (#599). On 1.6 the
connect modal registers with the onboarding API (``/customers/register``)
straight from the browser, and the onboarding store is where a UI database's
credentials live: the data transfer reads them back through
``/customers/{id}/credentials``, which releases a password only to the API
service, for a customer bound to the user it names. A chat registration takes
the same road: :func:`register_and_save` registers as the signed-in user, so
the customer is bound to that user, and returns the customer id -- the handle.
The password goes in the request body and nowhere else.

Not ported: 2.0's saved-connections table (``connection_store``) and its sealed
``api_key`` handle. 1.6 has neither; its transfer opens a handle with the
service token plus the user, not with the customer's api_key, so the api_key
this registration is issued is dropped here and never reaches the session.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger("avaloka")

ONBOARDING_TIMEOUT_S = 30.0


def _onboarding_url() -> str:
    return os.getenv("ONBOARDING_API_URL", "http://localhost:8011").rstrip("/")


def _user_headers(user_id: Optional[str]) -> Dict[str, str]:
    """The API service acting for ``user_id`` (planner._onboarding_headers); {} without both."""
    token = os.getenv("MCP_SERVICE_TOKEN", "")
    user_id = (user_id or "").strip()
    if not token or not user_id:
        return {}
    return {"X-Avaloka-Service-Token": token, "X-Avaloka-User": user_id}


class RegistrationRejected(Exception):
    """The onboarding API refused a registration (its status code and detail)."""

    def __init__(self, status_code: int, detail: Any):
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


async def _onboarding(method: str, path: str, user_id: str,
                      body: Optional[bytes] = None) -> Tuple[int, Dict[str, Any]]:
    """Call the onboarding API as ``user_id``. Never logs the body (it carries the password)."""
    headers = _user_headers(user_id)
    if not headers:
        raise RegistrationRejected(
            500, "MCP_SERVICE_TOKEN is not configured; cannot register a database for a user.")
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method, f"{_onboarding_url()}{path}",
                content=body, headers=headers, timeout=ONBOARDING_TIMEOUT_S,
            )
    except httpx.HTTPError as exc:
        logger.warning("[connections] onboarding API unreachable (%s): %s", method, type(exc).__name__)
        raise RegistrationRejected(502, "Registration service unavailable")
    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("detail", f"Registration service answered HTTP {resp.status_code}")
    return resp.status_code, payload


async def register_and_save(user_id: str, body: Dict[str, Any]) -> Tuple[Dict[str, Any], None]:
    """Register a database with the onboarding API as ``user_id``. Returns ``(registration, None)``.

    The one path a chat-typed database's credentials take into the sealed
    store: the onboarding API tests the connection, binds the customer to
    ``user_id`` and keeps the connection string; the password is stored
    nowhere else. The returned registration is the customer id plus display
    fields -- never the api_key, never the password. (2.0 also returns its
    sealed api_key handle; 1.6 has none, hence ``None``.)

    Raises :class:`RegistrationRejected` for the onboarding API's refusal.
    """
    connection_name = str(body.get("connection_name") or "").strip()
    registration = {k: v for k, v in body.items() if k != "connection_name"}
    code, reg = await _onboarding(
        "POST", "/customers/register", user_id, json.dumps(registration).encode("utf-8"),
    )
    if not 200 <= code < 300:
        raise RegistrationRejected(code, reg.get("detail"))
    customer_id = str(reg.get("customer_id") or "")
    if not customer_id:
        raise RegistrationRejected(502, "the registration service returned no customer id")
    logger.info("[connections] registered %s for a chat connection", customer_id)
    saved = {
        "customer_id": customer_id,
        "connection_name": connection_name,
        "database_type": str(reg.get("database_type") or body.get("database_type") or ""),
        "status": str(reg.get("status") or "active"),
    }
    return saved, None
