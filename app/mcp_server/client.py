"""Client for the onboarding routes that describe a registered database.

Everything here authenticates as the customer (Bearer api_key), or as the
user the API server authenticated (``user_headers``), and receives no secret
back: ``/customers/{id}/planner-metadata`` is tables, typed columns,
indexes, partitions and row counts, built by the MCP server through the
connector contract with the credential resolved on its side. The api_key
itself is opened in this process from the sealed blob the DB session stores
(``api_key_enc``), so it never sits in a prompt, a graph state or a log.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from app.api.cloud_connections import decrypt_secret

logger = logging.getLogger(__name__)

# Same default as app.agents.planner: the onboarding API is local unless a
# deployment says otherwise (never a live address as a default).
ONBOARDING_API_URL = os.getenv("ONBOARDING_API_URL", "http://localhost:8081").rstrip("/")
DEFAULT_TIMEOUT_S = float(os.getenv("ONBOARDING_API_TIMEOUT_S", "10"))


class OnboardingClientError(RuntimeError):
    """The onboarding API refused or could not answer; the message carries no secret."""


def api_key_from_handle(handle: Any) -> Optional[str]:
    """Open a sealed ``api_key_enc`` blob in process; ``None`` without the key."""
    if not isinstance(handle, dict):
        return None
    return decrypt_secret(handle)


def user_headers(user_id: Optional[str]) -> Dict[str, str]:
    """Headers that name ``user_id`` to the onboarding API, for the routes that act per person.

    Only the API server calls this, for a user it has already authenticated
    (the requester, never a shared analysis's owner). Empty without a user or
    without the deployment key: the route then answers 401, never a list of
    somebody else's databases.
    """
    from app.mcp_server.credentials import user_assertion
    token = user_assertion((user_id or "").strip()) if user_id else None
    return {"Authorization": f"Bearer {token}"} if token else {}


def planner_metadata(
    customer_id: str,
    api_key: str,
    *,
    max_bytes: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """The size-capped ``source_metadata`` document for one registered database."""
    if not customer_id or not api_key:
        raise OnboardingClientError("customer_id and api_key are required")
    url = f"{(base_url or ONBOARDING_API_URL).rstrip('/')}/customers/{customer_id}/planner-metadata"
    params = {"max_bytes": int(max_bytes)} if max_bytes else None
    with requests.Session() as session:
        resp = session.get(
            url, params=params, timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
    if resp.status_code != 200:
        raise OnboardingClientError(f"planner-metadata for {customer_id!r}: HTTP {resp.status_code}")
    payload = resp.json()
    metadata = payload.get("source_metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise OnboardingClientError(f"planner-metadata for {customer_id!r}: malformed response")
    return metadata


def list_tables(customer_id: str, api_key: str, **kwargs: Any) -> List[str]:
    """Table names of a registered database, through the same contract document."""
    metadata = planner_metadata(customer_id, api_key, **kwargs)
    return [str(t.get("name")) for t in metadata.get("tables") or [] if isinstance(t, dict) and t.get("name")]
