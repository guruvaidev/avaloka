"""Customer credentials at rest, and the handle that stands in for them in transit.

1.6 port of 2.0's ``app/mcp_server/credentials.py`` (55a8d5d1, 142a9fac).

``customers.json`` -- the registry both ``multi_tenant_mcp_server`` and
``customer_dbs`` write -- used to hold every customer's connection string
(password included) and API key in plaintext, and the onboarding service
POSTed both to the MCP server's ``/admin/customers`` in plaintext.

The rules now:

* On disk the two secrets are AES-GCM blobs under ``DB_ENCRYPTION_KEY``, the
  key and scheme the API already uses for cloud, MCP and GitHub credentials
  (``app.api.cloud_connections.encrypt_secret``). No second scheme. Beside
  the blob a record keeps display fields parsed from the DSN -- host, port,
  database and user -- never the password.
* In memory the registry is plaintext: a process that holds the key opens a
  credential when it needs it (an engine, the API service's /credentials).
* In transit to the MCP server the connection string and api_key travel as
  the same sealed blobs (``connection_string_enc``, ``api_key_enc``); the MCP
  server opens them with its copy of the key. (2.0 sends them in the clear
  inside a body-signed request; here the body itself carries no secret.)
* A registry written before this change is read as it is and re-sealed on
  first read with the key (``load_registry(..., migrate=True)``).

Not ported from 2.0: the user assertion and the onboarding-to-MCP service MAC.
1.6 authenticates both with MCP_SERVICE_TOKEN, unchanged here.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote, urlparse

from app.api.cloud_connections import decrypt_secret, encrypt_secret

logger = logging.getLogger(__name__)

#: Record fields that never reach disk, or the MCP server, in the clear.
SECRET_FIELDS: Tuple[str, ...] = ("connection_string", "api_key")
ENCRYPTED_SUFFIX = "_enc"
HANDLE_SCHEME = "aes-gcm/DB_ENCRYPTION_KEY"
#: Fields parsed from the DSN and kept beside the sealed blob, for display.
DISPLAY_FIELDS: Tuple[str, ...] = ("host", "port", "database", "username")

_file_lock = threading.Lock()


class CredentialKeyMissing(RuntimeError):
    """DB_ENCRYPTION_KEY is not configured, so a secret cannot be sealed or opened."""


class CredentialKeyWrong(CredentialKeyMissing):
    """A key IS configured, but this record does not open under it.

    Almost always the deployment's DB_ENCRYPTION_KEY changed after the record
    was saved (for example a published key was replaced,
    app/core/provisioning.py); an AES-GCM failure cannot tell that from a
    damaged record, so the message names both. Nothing can open it; the
    connection has to be registered again. A subclass of
    CredentialKeyMissing so every caller that already refuses an unopenable
    record keeps refusing it.
    """


#: The status a record that cannot be opened under the current key is shown with.
NEEDS_REREGISTERING = "needs_reregistering"
WRONG_KEY_MESSAGE = (
    "its saved credentials do not open under the DB_ENCRYPTION_KEY this deployment now uses "
    "(they were saved under a different key, or the stored record is damaged), so they can no "
    "longer be read; delete this connection and register it again"
)


def _wrong_key() -> CredentialKeyWrong:
    return CredentialKeyWrong(WRONG_KEY_MESSAGE)


def _missing_key() -> CredentialKeyMissing:
    return CredentialKeyMissing(
        "DB_ENCRYPTION_KEY is not configured; customer credentials are stored "
        "encrypted and cannot be written or read without it."
    )


def parse_connection_string(conn_str: str) -> Dict[str, Any]:
    """Split a DSN into the parts a data transfer needs. The password is one of them."""
    parsed = urlparse(conn_str or "")
    try:
        port = parsed.port
    except ValueError:
        port = None
    return {
        "host": parsed.hostname or "localhost",
        "port": port,
        "database": (parsed.path or "").lstrip("/").split("?")[0],
        "username": unquote(parsed.username) if parsed.username else None,
        "password": unquote(parsed.password or ""),
    }


def display_fields(conn_str: str) -> Dict[str, Any]:
    """Host, port, database and user of a DSN -- everything but the password."""
    parts = parse_connection_string(conn_str)
    return {f"display_{k}": parts[k] for k in DISPLAY_FIELDS if parts.get(k) not in (None, "")}


def seal_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``record`` fit for disk: secrets sealed, plaintext fields gone."""
    out = dict(record)
    for field in SECRET_FIELDS:
        value = out.pop(field, None)
        if not value:
            continue
        blob = encrypt_secret(str(value))
        if blob is None:
            raise _missing_key()
        out[field + ENCRYPTED_SUFFIX] = blob
        if field == "connection_string":
            out.update(display_fields(str(value)))
    return out


def unseal_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Return ``(in-memory record, was_legacy_plaintext)``.

    A record that still carries a plaintext secret is accepted -- that is what
    every registry written before this change looks like -- and flagged so the
    caller can re-seal the file. A sealed field that will not open (no key, or
    another deployment's key) raises: nothing may run a customer with no
    credential.
    """
    out = dict(record)
    legacy = False
    for field in SECRET_FIELDS:
        blob = out.pop(field + ENCRYPTED_SUFFIX, None)
        if blob is not None:
            value = decrypt_secret(blob)
            if value is None:
                # No key at all, or a key that is not the one this was sealed
                # under: two different faults, two different messages.
                raise _wrong_key() if encrypt_secret("probe") is not None else _missing_key()
            out[field] = value
        elif out.get(field):
            legacy = True
    return out, legacy


def read_raw(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as fh:
        return json.load(fh) or {}


def unreadable_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """A record that cannot be opened, as it may be shown: no secret, no blob,
    status NEEDS_REREGISTERING."""
    out = {k: v for k, v in record.items()
           if k not in SECRET_FIELDS and not k.endswith(ENCRYPTED_SUFFIX)}
    if out.get("status") != "inactive":
        out["status"] = NEEDS_REREGISTERING
    return out


def load_unreadable(path: str) -> Dict[str, Dict[str, Any]]:
    """The records this key cannot open, each as :func:`unreadable_record` shows it."""
    out: Dict[str, Dict[str, Any]] = {}
    for customer_id, record in read_raw(path).items():
        try:
            unseal_record(record)
        except CredentialKeyMissing:
            out[customer_id] = unreadable_record(record)
    return out


def load_registry(path: str, migrate: bool = True) -> Dict[str, Dict[str, Any]]:
    """Read ``customers.json`` into plaintext memory; a missing file is an empty registry.

    Records this key cannot open are left out (and left on disk); see
    :func:`load_unreadable` for them. With ``migrate`` (the default), a file
    that still holds plaintext secrets is re-sealed on disk as it is read, when
    the key is configured. Without the key the file is left as it is.
    """
    raw = read_raw(path)
    registry: Dict[str, Dict[str, Any]] = {}
    legacy = 0
    for customer_id, record in raw.items():
        try:
            registry[customer_id], was_legacy = unseal_record(record)
        except CredentialKeyWrong:
            # Left on disk; shown to its owner as NEEDS_REREGISTERING.
            logger.error("Customer %r cannot be opened: %s.", customer_id, WRONG_KEY_MESSAGE)
            continue
        except CredentialKeyMissing as exc:
            logger.error("Skipping customer %r: %s", customer_id, exc)
            continue
        legacy += int(was_legacy)
    if migrate and legacy and encrypt_secret("probe") is not None:
        try:
            save_registry(path, registry, keep=raw)
            logger.warning("Re-sealed %d plaintext customer credential(s) in %s", legacy, path)
        except (CredentialKeyMissing, OSError) as exc:
            logger.error("Could not re-seal %s: %s", path, type(exc).__name__)
    return registry


def save_registry(path: str, registry: Dict[str, Dict[str, Any]],
                  keep: Optional[Dict[str, Any]] = None) -> None:
    """Write the registry with every secret sealed; refuses to write plaintext.

    ``keep`` is the raw file as read: a record that could not be opened (and
    so is not in ``registry``) is written back exactly as it was.
    """
    sealed = {cid: seal_record(rec) for cid, rec in registry.items()}
    for cid, rec in (keep or {}).items():
        sealed.setdefault(cid, rec)
    with _file_lock:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(sealed, fh, indent=2)
        os.replace(tmp, path)


def api_key_matches(record: Dict[str, Any], presented: Optional[str]) -> bool:
    """Constant-time check of a Bearer api_key against the customer's own."""
    expected = str(record.get("api_key") or "")
    if not expected or not presented:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), str(presented).encode("utf-8"))


def credential_handle(secret: str) -> Dict[str, str]:
    """Seal a secret into the handle that travels in its place."""
    blob = encrypt_secret(secret)
    if blob is None:
        raise _missing_key()
    return {"scheme": HANDLE_SCHEME, **blob}


def open_credential_handle(handle: Any) -> Optional[str]:
    """Open a handle in this process; ``None`` when it is absent or this key cannot."""
    if not isinstance(handle, dict) or handle.get("scheme") != HANDLE_SCHEME:
        return None
    return decrypt_secret({"ciphertext": handle.get("ciphertext"), "iv": handle.get("iv")})
