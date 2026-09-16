"""Offline licence verification for commercial Avaloka deployments.

A Professional or Enterprise deployment may run entirely inside a customer VPC
with no route back to Avaloka, so entitlement cannot be resolved by calling
home. Instead the customer is issued a signed licence token which this module
verifies offline against an embedded public key.

**This module verifies. It does not sign.**

Signing lives in the private ``avaloka_commercial.licensing`` package together
with the Ed25519 private key, and is never published. That asymmetry is the
whole design: publishing the verifier reveals nothing that helps forge a
licence, because minting one requires a private key that is not in this
repository and never will be. Anyone may read this file; nobody can issue
themselves an Enterprise licence with it.

Do not add a ``sign_license`` function here, and do not import the private
signer from application code.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .editions import Edition, normalize_edition


class LicenseError(Exception):
    """Base class for every licence failure. Always fails closed."""


class LicenseFormatError(LicenseError):
    """The token is malformed."""


class LicenseSignatureError(LicenseError):
    """The signature is absent, unverifiable, or forged."""


class LicenseExpiredError(LicenseError):
    """The licence is outside its validity window."""


#: Ed25519 public key (base64) used to verify licence tokens. Safe to publish —
#: it can only *check* a signature, never produce one. Overridable via
#: ``AVALOKA_LICENSE_PUBLIC_KEY`` so staging can use a separate key pair.
DEFAULT_PUBLIC_KEY_B64 = ""


@dataclass(frozen=True)
class License:
    """The verified claims of a licence token."""

    edition: Edition
    organization: str
    seats: int
    issued_at: int
    expires_at: int
    subject: str = ""

    def is_expired(self, *, now: Optional[int] = None) -> bool:
        return (now if now is not None else int(time.time())) >= self.expires_at

    def seats_exceeded(self, active_seats: int) -> bool:
        return self.seats >= 0 and active_seats > self.seats


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    # Translate the URL-safe alphabet by hand so we can pass validate=True,
    # which urlsafe_b64decode does not accept. Validating here means garbage
    # outside the alphabet is reported as a format error rather than silently
    # decoding to something shorter and surfacing as a confusing signature
    # failure further down.
    normalized = segment.replace("-", "+").replace("_", "/") + padding
    try:
        return base64.b64decode(normalized, validate=True)
    except Exception as exc:  # noqa: BLE001 - any decode failure is a format error
        raise LicenseFormatError(f"segment is not valid base64url: {exc}") from exc


def _public_key_b64() -> str:
    return os.getenv("AVALOKA_LICENSE_PUBLIC_KEY", DEFAULT_PUBLIC_KEY_B64).strip()


def _verify_signature(payload: bytes, signature: bytes, public_key_b64: str) -> None:
    """Verify an Ed25519 signature, or raise :class:`LicenseSignatureError`."""
    if not public_key_b64:
        raise LicenseSignatureError(
            "no licence public key configured; set AVALOKA_LICENSE_PUBLIC_KEY"
        )
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise LicenseSignatureError(
            "the 'cryptography' package is required to verify licences"
        ) from exc

    try:
        key = Ed25519PublicKey.from_public_bytes(_b64url_decode(public_key_b64))
    except LicenseFormatError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LicenseSignatureError(f"licence public key is unusable: {exc}") from exc

    try:
        key.verify(signature, payload)
    except InvalidSignature as exc:
        raise LicenseSignatureError("licence signature does not verify") from exc


def _claims_from(payload: Dict[str, Any]) -> License:
    try:
        return License(
            edition=normalize_edition(str(payload["edition"])),
            organization=str(payload["organization"]),
            seats=int(payload.get("seats", -1)),
            issued_at=int(payload["issued_at"]),
            expires_at=int(payload["expires_at"]),
            subject=str(payload.get("subject", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LicenseFormatError(f"licence claims are incomplete: {exc}") from exc


def verify_license(
    token: str,
    *,
    public_key_b64: Optional[str] = None,
    now: Optional[int] = None,
) -> License:
    """Verify a licence token and return its claims.

    The token is ``<base64url(payload)>.<base64url(signature)>``. Every failure
    path raises; there is no partial success and no "unverified but probably
    fine" result, because a licence that cannot be verified must not grant
    anything.
    """
    if not token or not isinstance(token, str):
        raise LicenseFormatError("licence token is empty")
    parts = token.strip().split(".")
    if len(parts) != 2:
        raise LicenseFormatError(
            f"licence token must have 2 dot-separated segments, got {len(parts)}"
        )

    payload_raw = _b64url_decode(parts[0])
    signature = _b64url_decode(parts[1])

    # Signature first: never parse attacker-controlled claims we have not
    # authenticated, and never let a malformed-claims error leak before the
    # signature check has rejected a forgery.
    _verify_signature(payload_raw, signature, public_key_b64 or _public_key_b64())

    try:
        payload = json.loads(payload_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LicenseFormatError(f"licence payload is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LicenseFormatError("licence payload must be a JSON object")

    claims = _claims_from(payload)
    if claims.expires_at <= claims.issued_at:
        raise LicenseFormatError("licence expires before it was issued")
    if claims.is_expired(now=now):
        raise LicenseExpiredError(
            f"licence for {claims.organization!r} expired at {claims.expires_at}"
        )
    return claims


def edition_from_license(
    token: Optional[str],
    *,
    public_key_b64: Optional[str] = None,
    now: Optional[int] = None,
) -> Edition:
    """Resolve an edition from a licence token, failing closed to ``oss``.

    A missing, malformed, forged, or expired licence yields ``oss`` rather than
    an exception, so a self-hosted deployment without a licence still runs the
    open-source capability set instead of refusing to start.
    """
    if not token:
        return "oss"
    try:
        return verify_license(token, public_key_b64=public_key_b64, now=now).edition
    except LicenseError:
        return "oss"
