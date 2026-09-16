"""Slim HTTP client for the openclaw Gateway.

The openclaw daemon (https://github.com/openclaw/openclaw) is an optional
sidecar. When it is running and ``OPENCLAW_ENABLED=1``, this client invokes
its Gateway RPC for data-source discovery. Otherwise, callers should fall
back to ``discovery_tools`` (native Python).

Design goals:
- "Slim": only the calls Avaloka actually needs (introspect connection,
  list tables/objects, describe schema).
- Fail fast: 2-second connect timeout; raises ``OpenclawUnavailable``
  rather than blocking the chat turn.
- No external dep beyond ``httpx`` (already a transitive of FastAPI/Starlette
  but pinned in requirements.txt to be safe).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OpenclawUnavailable(RuntimeError):
    """Raised when the openclaw Gateway cannot be reached."""


def _is_enabled() -> bool:
    return os.getenv("OPENCLAW_ENABLED", "0").strip() in ("1", "true", "True", "yes")


def _gateway_url() -> str:
    return os.getenv("OPENCLAW_GATEWAY_URL", "http://localhost:18789").rstrip("/")


def _connect_timeout() -> float:
    try:
        return float(os.getenv("OPENCLAW_TIMEOUT_S", "2.0"))
    except ValueError:
        return 2.0


def _api_key() -> Optional[str]:
    return os.getenv("OPENCLAW_API_KEY") or None


class OpenclawDiscoveryClient:
    """Thin wrapper around the openclaw Gateway HTTP RPC.

    Method names mirror the discovery tools shipped in the
    ``deploy/openclaw/skills/avaloka-discovery`` skill template.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_s: Optional[float] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or _gateway_url()).rstrip("/")
        self.timeout_s = timeout_s if timeout_s is not None else _connect_timeout()
        self.api_key = api_key if api_key is not None else _api_key()

    @staticmethod
    def is_enabled() -> bool:
        return _is_enabled()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import httpx  # local import keeps import-time cost low
        except ImportError as exc:
            raise OpenclawUnavailable("httpx is not installed") from exc

        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json() if resp.content else {}
        except Exception as exc:
            raise OpenclawUnavailable(str(exc)) from exc

    def is_available(self) -> bool:
        if not _is_enabled():
            return False
        try:
            import httpx
        except ImportError:
            return False
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.get(f"{self.base_url}/healthz")
                return resp.status_code < 500
        except Exception:
            return False

    # ---- discovery surface (matches deploy/openclaw/skills/avaloka-discovery)

    def discover_database(self, connection_id: str) -> Dict[str, Any]:
        return self._post(
            "/v1/skills/avaloka-discovery/list_tables",
            {"connection_id": connection_id},
        )

    def describe_table(self, connection_id: str, table: str) -> Dict[str, Any]:
        return self._post(
            "/v1/skills/avaloka-discovery/describe_table",
            {"connection_id": connection_id, "table": table},
        )

    def discover_object_store(self, uri: str, connection_id: str) -> Dict[str, Any]:
        return self._post(
            "/v1/skills/avaloka-discovery/list_cloud_objects",
            {"connection_id": connection_id, "uri": uri},
        )

    def sample_rows(self, dataset_id: str, n: int = 50) -> Dict[str, Any]:
        return self._post(
            "/v1/skills/avaloka-discovery/sample_rows",
            {"dataset_id": dataset_id, "n": int(n)},
        )
