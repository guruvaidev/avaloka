"""Offline licence verification.

These tests mint tokens with a throwaway key pair generated in-process. The
production signer is not importable from here by design -- see
``app/core/license.py``.
"""

from __future__ import annotations

import base64
import json
import time

import pytest

from app.core.license import (LicenseExpiredError, LicenseFormatError,
                              LicenseSignatureError, edition_from_license,
                              verify_license)

cryptography = pytest.importorskip(
    "cryptography", reason="cryptography is a declared dependency of this project"
)
from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
    Ed25519PrivateKey,
)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


@pytest.fixture
def keypair():
    private = Ed25519PrivateKey.generate()
    from cryptography.hazmat.primitives import serialization
    public_raw = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private, _b64(public_raw)


def mint(private, **claims) -> str:
    """Stand-in for the private signer, for tests only."""
    now = int(time.time())
    payload = {
        "edition": "enterprise",
        "organization": "Acme Corp",
        "seats": 50,
        "issued_at": now - 60,
        "expires_at": now + 86_400,
    }
    payload.update(claims)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"{_b64(raw)}.{_b64(private.sign(raw))}"


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

def test_a_validly_signed_licence_verifies(keypair):
    private, public = keypair
    licence = verify_license(mint(private), public_key_b64=public)
    assert licence.edition == "enterprise"
    assert licence.organization == "Acme Corp"
    assert licence.seats == 50


@pytest.mark.parametrize("edition", ["professional", "enterprise"])
def test_edition_round_trips(keypair, edition):
    private, public = keypair
    token = mint(private, edition=edition)
    assert edition_from_license(token, public_key_b64=public) == edition


# --------------------------------------------------------------------------- #
# Forgery must fail, and must fail closed
# --------------------------------------------------------------------------- #

def test_tampering_with_claims_invalidates_the_signature(keypair):
    """Editing the payload to upgrade yourself must not verify."""
    private, public = keypair
    token = mint(private, edition="professional", seats=5)
    payload_b64, signature_b64 = token.split(".")

    payload = json.loads(base64.urlsafe_b64decode(
        payload_b64 + "=" * (-len(payload_b64) % 4)))
    payload["edition"] = "enterprise"
    payload["seats"] = 10_000
    forged = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    forged_token = f"{_b64(forged)}.{signature_b64}"

    with pytest.raises(LicenseSignatureError):
        verify_license(forged_token, public_key_b64=public)
    # ...and the fail-closed helper degrades to oss rather than raising.
    assert edition_from_license(forged_token, public_key_b64=public) == "oss"


def test_a_licence_signed_by_the_wrong_key_is_rejected(keypair):
    _, public = keypair
    attacker = Ed25519PrivateKey.generate()
    with pytest.raises(LicenseSignatureError):
        verify_license(mint(attacker), public_key_b64=public)


def test_verification_requires_a_configured_public_key(keypair):
    private, _ = keypair
    with pytest.raises(LicenseSignatureError):
        verify_license(mint(private), public_key_b64="")


def test_signature_is_checked_before_claims_are_parsed(keypair):
    """Malformed claims behind a bad signature must surface as a signature
    failure, so an attacker cannot use error messages to probe the parser."""
    _, public = keypair
    attacker = Ed25519PrivateKey.generate()
    raw = b"this is not json at all"
    token = f"{_b64(raw)}.{_b64(attacker.sign(raw))}"
    with pytest.raises(LicenseSignatureError):
        verify_license(token, public_key_b64=public)


# --------------------------------------------------------------------------- #
# Expiry and malformed input
# --------------------------------------------------------------------------- #

def test_an_expired_licence_is_rejected(keypair):
    private, public = keypair
    now = int(time.time())
    token = mint(private, issued_at=now - 200, expires_at=now - 100)
    with pytest.raises(LicenseExpiredError):
        verify_license(token, public_key_b64=public)
    assert edition_from_license(token, public_key_b64=public) == "oss"


def test_a_licence_expiring_before_issue_is_rejected(keypair):
    private, public = keypair
    now = int(time.time())
    with pytest.raises(LicenseFormatError):
        verify_license(mint(private, issued_at=now, expires_at=now - 1),
                       public_key_b64=public)


def test_expiry_is_evaluated_against_an_injectable_clock(keypair):
    private, public = keypair
    now = int(time.time())
    token = mint(private, issued_at=now, expires_at=now + 100)
    assert verify_license(token, public_key_b64=public, now=now + 50)
    with pytest.raises(LicenseExpiredError):
        verify_license(token, public_key_b64=public, now=now + 101)


@pytest.mark.parametrize("token", ["", "   ", "onlyonesegment", "a.b.c", "!!.??"])
def test_malformed_tokens_are_rejected(keypair, token):
    _, public = keypair
    with pytest.raises(LicenseFormatError):
        verify_license(token, public_key_b64=public)


def test_incomplete_claims_are_rejected(keypair):
    private, public = keypair
    raw = json.dumps({"edition": "enterprise"}).encode()
    token = f"{_b64(raw)}.{_b64(private.sign(raw))}"
    with pytest.raises(LicenseFormatError):
        verify_license(token, public_key_b64=public)


def test_absent_licence_degrades_to_oss():
    assert edition_from_license(None) == "oss"
    assert edition_from_license("") == "oss"


def test_seat_limit_is_reported(keypair):
    private, public = keypair
    licence = verify_license(mint(private, seats=3), public_key_b64=public)
    assert licence.seats_exceeded(4) is True
    assert licence.seats_exceeded(3) is False


def test_this_module_does_not_expose_a_signer():
    """The signer is private by design; guard against it drifting in here."""
    import inspect

    import app.core.license as mod
    offenders = [
        name for name, obj in vars(mod).items()
        if not name.startswith("_")
        and callable(obj)
        and not (inspect.isclass(obj) and issubclass(obj, Exception))
        and any(word in name.lower() for word in ("sign", "mint", "issue", "generate"))
    ]
    assert not offenders, f"licence signing must stay private, found: {offenders}"
