"""Customer credentials at rest, and the handle that stands in for them.

``customers.json`` -- the registry both ``multi_tenant_mcp_server`` and
``customer_dbs`` write -- used to hold every customer's connection string
(password included) and API key in plaintext on a PVC, and
``GET /customers/{id}/credentials`` handed the password to anyone who asked.

The rules now:

* On disk the two secrets are AES-GCM blobs under ``DB_ENCRYPTION_KEY``, the
  key and scheme the API already uses for cloud and MCP connection rows
  (``app.api.cloud_connections.encrypt_secret``). No second scheme.
* In memory the registry is plaintext: the MCP server resolves a credential
  in process when it opens an engine, and nothing hands it back out.
* An endpoint returns a *handle* -- the connection string sealed under the
  same key -- never the password. A process that holds the deployment key
  (the API, for a data transfer) opens the handle itself; a process that
  does not holds ciphertext.
* A registry written before this change is read as it is and re-sealed on
  first load (``migrate_registry``), so an existing plaintext file is fixed
  the first time a server with the key starts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote, urlparse

import app.api.cloud_connections as _cloud_connections
from app.api.cloud_connections import decrypt_secret, encrypt_secret

logger = logging.getLogger(__name__)

#: Record fields that never reach disk in the clear.
SECRET_FIELDS: Tuple[str, ...] = ("connection_string", "api_key")
ENCRYPTED_SUFFIX = "_enc"
HANDLE_SCHEME = "aes-gcm/DB_ENCRYPTION_KEY"


class CredentialKeyMissing(RuntimeError):
    """DB_ENCRYPTION_KEY is not configured, so a secret cannot be sealed or opened."""


class CredentialKeyWrong(CredentialKeyMissing):
    """A key IS configured, but this record does not open under it.

    Almost always the deployment's DB_ENCRYPTION_KEY changed after the record
    was saved (for example a published key was replaced,
    app/core/provisioning.py); an AES-GCM failure cannot tell that from a
    damaged record, so the message names both. A subclass of
    CredentialKeyMissing so every caller that already refuses an unopenable
    record keeps refusing it. (1.6 port, hotfix/1.6-wrong-key-is-not-missing-key.)
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
    return out


def unseal_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Return ``(in-memory record, was_legacy_plaintext)``.

    A record that still carries a plaintext secret is accepted -- that is what
    every registry written before this change looks like -- and flagged so the
    caller can re-seal the file. A sealed field that will not open (no key, or
    another deployment's key) raises: the server must not silently run a
    customer with no credential.
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


def _read_raw(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as fh:
        return json.load(fh) or {}


def unreadable_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """A record that cannot be opened, as it may be shown: no secret, no blob,
    status NEEDS_REREGISTERING (unless it was deleted)."""
    out = {k: v for k, v in record.items()
           if k not in SECRET_FIELDS and not k.endswith(ENCRYPTED_SUFFIX)}
    if out.get("status") != "inactive":
        out["status"] = NEEDS_REREGISTERING
    return out


def load_unreadable(path: str) -> Dict[str, Dict[str, Any]]:
    """The records this key cannot open, each as :func:`unreadable_record` shows it."""
    out: Dict[str, Dict[str, Any]] = {}
    for customer_id, record in _read_raw(path).items():
        try:
            unseal_record(record)
        except CredentialKeyMissing:
            out[customer_id] = unreadable_record(record)
    return out


def load_registry(path: str) -> Dict[str, Dict[str, Any]]:
    """Read ``customers.json`` into plaintext memory; a missing file is an empty registry.

    Records this key cannot open are left out here (and on disk); see
    :func:`load_unreadable` for them.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r") as fh:
        raw = json.load(fh)
    registry: Dict[str, Dict[str, Any]] = {}
    for customer_id, record in (raw or {}).items():
        try:
            registry[customer_id], _ = unseal_record(record)
        except CredentialKeyWrong:
            # Left on disk; shown to its owner as NEEDS_REREGISTERING.
            logger.error("Customer %r cannot be opened: %s.", customer_id, WRONG_KEY_MESSAGE)
        except CredentialKeyMissing as exc:
            logger.error("Skipping customer %r: %s", customer_id, exc)
    return registry


def save_registry(path: str, registry: Dict[str, Dict[str, Any]]) -> None:
    """Write the registry with every secret sealed; refuses to write plaintext.

    A record on disk that this key cannot open is not in ``registry``; it is
    written back exactly as it was, so a save never deletes it.
    """
    sealed = {cid: seal_record(rec) for cid, rec in registry.items()}
    for cid, rec in load_unreadable(path).items():
        if cid not in sealed:
            sealed[cid] = _read_raw(path)[cid]
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(sealed, fh, indent=2)
    os.replace(tmp, path)


def mark_unreadable_inactive(path: str, customer_id: str, when: str) -> None:
    """Delete (status inactive) a record this key cannot open, leaving its sealed fields as they are."""
    raw = _read_raw(path)
    raw[customer_id] = {**raw[customer_id], "status": "inactive", "deactivated_at": when}
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(raw, fh, indent=2)
    os.replace(tmp, path)


def migrate_registry(path: str) -> int:
    """Re-seal a registry that still holds plaintext secrets. Returns how many records were."""
    if not os.path.exists(path):
        return 0
    with open(path, "r") as fh:
        raw = json.load(fh)
    registry: Dict[str, Dict[str, Any]] = {}
    legacy = 0
    for customer_id, record in (raw or {}).items():
        try:
            registry[customer_id], was_legacy = unseal_record(record)
        except CredentialKeyMissing:
            return 0  # no key: nothing can be re-sealed, leave the file untouched
        legacy += int(was_legacy)
    if legacy:
        save_registry(path, registry)
        logger.warning("Re-sealed %d plaintext customer credential(s) in %s", legacy, path)
    return legacy


def api_key_matches(record: Dict[str, Any], presented: Optional[str]) -> bool:
    """Constant-time check of a Bearer api_key against the customer's own."""
    expected = str(record.get("api_key") or "")
    if not expected or not presented:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), str(presented).encode("utf-8"))


# ---------------------------------------------------------------------------
# The user the API server forwards
# ---------------------------------------------------------------------------
#
# A registration has no api_key yet, and listing is per person, not per
# database, so those routes need the *user*. The API server authenticates the
# browser's Supabase JWT and forwards the user id to the onboarding API as a
# short-lived assertion MACed with a key derived from DB_ENCRYPTION_KEY -- the
# key both processes already hold, and one that already opens every sealed
# credential, so it confers nothing new. No second secret to provision.

USER_ASSERTION_PREFIX = "avu1"
USER_ASSERTION_TTL_S = 300
_USER_ASSERTION_CONTEXT = b"avaloka/onboarding-user-assertion/v1"


def _user_assertion_key() -> Optional[bytes]:
    base = _cloud_connections._get_aesgcm_key()
    if not base:
        return None
    return hmac.new(base, _USER_ASSERTION_CONTEXT, hashlib.sha256).digest()


def _user_assertion_mac(key: bytes, user_id: str, expires: int) -> str:
    msg = f"{USER_ASSERTION_PREFIX}.{user_id}.{expires}".encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def _valid_user_id(user_id: Any) -> bool:
    return isinstance(user_id, str) and 0 < len(user_id) <= 128 and "." not in user_id and user_id.strip() == user_id


def user_assertion(user_id: str, ttl_s: int = USER_ASSERTION_TTL_S) -> Optional[str]:
    """A Bearer value naming ``user_id`` to the onboarding API; ``None`` without the key."""
    if not _valid_user_id(user_id):
        return None
    key = _user_assertion_key()
    if key is None:
        return None
    expires = int(time.time()) + int(ttl_s)
    return f"{USER_ASSERTION_PREFIX}.{user_id}.{expires}.{_user_assertion_mac(key, user_id, expires)}"


def verify_user_assertion(token: Optional[str]) -> Optional[str]:
    """The user id a valid, unexpired assertion names; ``None`` for anything else."""
    if not token or not token.startswith(USER_ASSERTION_PREFIX + "."):
        return None
    parts = token.split(".")
    if len(parts) != 4:
        return None
    _, user_id, expires_s, mac = parts
    if not _valid_user_id(user_id) or not expires_s.isdigit():
        return None
    key = _user_assertion_key()
    if key is None:
        return None
    expected = _user_assertion_mac(key, user_id, int(expires_s))
    if not hmac.compare_digest(expected.encode("utf-8"), mac.encode("utf-8")):
        return None
    if int(expires_s) < int(time.time()):
        return None
    return user_id


# ---------------------------------------------------------------------------
# The onboarding service, to the MCP server's admin routes
# ---------------------------------------------------------------------------
#
# /admin/customers creates or replaces a customer's connection string and
# api_key, and the MCP server connects to the DSN it is given. Only the
# onboarding service may call it. Its request carries a MAC over the method,
# the path, a timestamp and the body's SHA-256, under a key derived from
# DB_ENCRYPTION_KEY with its OWN label: a user assertion (above) is made
# under a different derived key and a different message shape, so neither
# verifies as the other. Bound to the body, so a captured header cannot
# carry a different registration.

SERVICE_MAC_PREFIX = "avs1"
SERVICE_MAC_WINDOW_S = 300
_SERVICE_MAC_CONTEXT = b"avaloka/onboarding-to-mcp-admin/v1"


def _service_mac_key() -> Optional[bytes]:
    base = _cloud_connections._get_aesgcm_key()
    if not base:
        return None
    return hmac.new(base, _SERVICE_MAC_CONTEXT, hashlib.sha256).digest()


def _service_mac(key: bytes, method: str, path: str, body: bytes, ts: int) -> str:
    digest = hashlib.sha256(body or b"").hexdigest()
    msg = f"{SERVICE_MAC_PREFIX}.{ts}.{method.upper()}.{path}.{digest}".encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def service_headers(method: str, path: str, body: bytes) -> Dict[str, str]:
    """Authorization for one onboarding -> MCP admin request; raises without the key."""
    key = _service_mac_key()
    if key is None:
        raise _missing_key()
    ts = int(time.time())
    return {"Authorization": f"Bearer {SERVICE_MAC_PREFIX}.{ts}.{_service_mac(key, method, path, body, ts)}"}


def verify_service_mac(token: Optional[str], method: str, path: str, body: bytes) -> bool:
    """True only for a fresh MAC over exactly this request, under this deployment's key."""
    if not token or not token.startswith(SERVICE_MAC_PREFIX + "."):
        return False
    parts = token.split(".")
    if len(parts) != 3 or not parts[1].isdigit():
        return False
    key = _service_mac_key()
    if key is None:
        return False  # fail closed: no key, no admin
    ts = int(parts[1])
    if abs(int(time.time()) - ts) > SERVICE_MAC_WINDOW_S:
        return False
    expected = _service_mac(key, method, path, body, ts)
    return hmac.compare_digest(expected.encode("utf-8"), parts[2].encode("utf-8"))


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


def credential_handle(conn_str: str) -> Dict[str, str]:
    """Seal a connection string into the handle an endpoint may return."""
    blob = encrypt_secret(conn_str)
    if blob is None:
        raise _missing_key()
    return {"scheme": HANDLE_SCHEME, **blob}


def open_sealed_secret(handle: Any) -> Optional[str]:
    """Open a handle to the secret it seals (a DSN or an api_key), in process.

    ``None`` when it is absent, not a handle, or this deployment's key cannot
    open it. Used by the MCP server for what /admin/customers is sent (1.6 port,
    hotfix/1.6-chat-connection-sealing 88120343).
    """
    if not isinstance(handle, dict) or handle.get("scheme") != HANDLE_SCHEME:
        return None
    return decrypt_secret({"ciphertext": handle.get("ciphertext"), "iv": handle.get("iv")}) or None


def open_credential_handle(handle: Any) -> Optional[Dict[str, Any]]:
    """Open a handle in this process; ``None`` when it is absent or this key cannot."""
    if not isinstance(handle, dict) or handle.get("scheme") != HANDLE_SCHEME:
        return None
    conn_str = decrypt_secret({"ciphertext": handle.get("ciphertext"), "iv": handle.get("iv")})
    if not conn_str:
        return None
    return parse_connection_string(conn_str)
