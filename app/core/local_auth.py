"""Local authentication for a self-hosted laptop install (``AVALOKA_AUTH_MODE=local``).

The web UI signs in through Supabase. A customer trying Avaloka on a laptop
has no Supabase project, so without this every authenticated page bounced to
``/``. Local mode replaces Supabase's auth server with one local account:

* The API signs its own HS256 JWTs with ``AVALOKA_LOCAL_AUTH_SECRET``, which
  ``scripts/quickstart.sh`` generates on first run into the gitignored
  ``.env.quickstart``. Every local token carries ``iss=avaloka-local`` and
  ``typ=access``; a refresh token (``typ=refresh``) never authenticates a
  request.
* The account is one email and password (``AVALOKA_LOCAL_AUTH_EMAIL``,
  ``AVALOKA_LOCAL_AUTH_PASSWORD``), also generated into ``.env.quickstart``.
* :mod:`app.api.local_auth_routes` serves the few GoTrue endpoints supabase-js
  calls (password sign-in, refresh, ``/user``, logout), so the UI keeps using
  its Supabase client unchanged and only points it at the API.

It is a self-hosted-only mode, and refused otherwise. :func:`refusal` returns
why it may not run, and the server refuses to START when the mode is
requested and refused. The conditions:

1. ``AVALOKA_DEPLOYMENT`` is explicitly ``self_hosted``. The editions module
   treats an *unset* deployment as self-hosted, so that default is not
   enough: someone has to say so.
2. The edition is not one sold only as a hosted plan (``free``).
3. ``SUPABASE_URL`` is unset. A deployment with a real identity provider has
   no business carrying a second, single-password one beside it.
4. The secret and the password are long enough to be worth having.

The quickstart compose also publishes every port on 127.0.0.1 only. The API
cannot see that from inside its container, so it is not one of the checks
above; the deployment flag is the explicit statement.

None of this changes how a Supabase token is verified. When local mode is
off, :func:`active_config` is ``None`` and the server's existing fail-closed
path runs exactly as before.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import jwt
from jwt import InvalidTokenError

from app.core.editions import deployment_state

MODE_ENV = "AVALOKA_AUTH_MODE"
SECRET_ENV = "AVALOKA_LOCAL_AUTH_SECRET"
EMAIL_ENV = "AVALOKA_LOCAL_AUTH_EMAIL"
PASSWORD_ENV = "AVALOKA_LOCAL_AUTH_PASSWORD"

ISSUER = "avaloka-local"
AUDIENCE = "authenticated"
DEFAULT_EMAIL = "local@avaloka.local"
MIN_SECRET_LEN = 32
MIN_PASSWORD_LEN = 12
ACCESS_TTL_S = 3600
REFRESH_TTL_S = 30 * 24 * 3600

# Editions that exist only as a hosted plan. A self-hosted install is ``oss``
# or a commercial edition it bought; it is never ``free``.
_HOSTED_ONLY_EDITIONS = frozenset({"free"})
# One stable namespace, so the same email always maps to the same user id and
# a restart does not orphan the user's threads and datasets.
_USER_NAMESPACE = uuid.UUID("6f1d8a52-3b0e-4c38-9a8e-5f6c1e0d2a71")


class LocalAuthRefused(RuntimeError):
    """Local auth was requested where it must not run."""


@dataclass(frozen=True)
class LocalAuthConfig:
    secret: str
    email: str
    password: str

    @property
    def user_id(self) -> str:
        return str(uuid.uuid5(_USER_NAMESPACE, self.email))


def requested(env: Optional[Dict[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    return (env.get(MODE_ENV) or "").strip().lower() == "local"


def refusal(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Why local auth may not run here, or ``None`` when it may."""
    env = os.environ if env is None else env
    deployment = (env.get("AVALOKA_DEPLOYMENT") or "").strip().lower().replace("-", "_")
    if deployment_state(env) != "self_hosted":
        return (
            f"{MODE_ENV}=local is for a self-hosted install only and needs "
            f"AVALOKA_DEPLOYMENT=self_hosted (got {deployment or 'unset'!r})."
        )
    edition = (env.get("AVALOKA_EDITION") or "").strip().lower()
    if edition in _HOSTED_ONLY_EDITIONS:
        return f"{MODE_ENV}=local is refused for the hosted-only edition {edition!r}."
    if (env.get("SUPABASE_URL") or "").strip():
        return (
            f"{MODE_ENV}=local is refused while SUPABASE_URL is set: a deployment "
            "with a real identity provider must use it."
        )
    if len(env.get(SECRET_ENV) or "") < MIN_SECRET_LEN:
        return f"{MODE_ENV}=local needs {SECRET_ENV} of at least {MIN_SECRET_LEN} characters."
    if len(env.get(PASSWORD_ENV) or "") < MIN_PASSWORD_LEN:
        return f"{MODE_ENV}=local needs {PASSWORD_ENV} of at least {MIN_PASSWORD_LEN} characters."
    return None


def load(env: Optional[Dict[str, str]] = None) -> Optional[LocalAuthConfig]:
    """The config when local auth is requested AND allowed; ``None`` otherwise.

    Raises :class:`LocalAuthRefused` when it is requested but not allowed, so
    a misconfigured hosted deployment stops at boot instead of running with
    an auth mode nobody meant it to have.
    """
    env = os.environ if env is None else env
    if not requested(env):
        return None
    reason = refusal(env)
    if reason:
        raise LocalAuthRefused(reason)
    email = (env.get(EMAIL_ENV) or DEFAULT_EMAIL).strip().lower()
    return LocalAuthConfig(secret=env[SECRET_ENV], email=email, password=env[PASSWORD_ENV])


def active_config() -> Optional[LocalAuthConfig]:
    """The config for this process, or ``None`` when local auth is off or refused.

    Never raises: request paths call this, and a refused config there means
    "no local tokens are valid", which is the fail-closed answer. Boot calls
    :func:`load` and does raise.
    """
    try:
        return load()
    except LocalAuthRefused:
        return None


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def _encode(cfg: LocalAuthConfig, typ: str, ttl_s: int, now: Optional[int] = None) -> tuple[str, int]:
    now = int(time.time()) if now is None else now
    exp = now + ttl_s
    claims = {
        "iss": ISSUER, "aud": AUDIENCE, "typ": typ,
        "sub": cfg.user_id, "email": cfg.email,
        "role": "authenticated", "iat": now, "exp": exp,
    }
    return jwt.encode(claims, cfg.secret, algorithm="HS256"), exp


def issue_session(cfg: LocalAuthConfig) -> Dict[str, Any]:
    """A GoTrue-shaped session, which supabase-js stores and refreshes as is."""
    access, exp = _encode(cfg, "access", ACCESS_TTL_S)
    refresh, _ = _encode(cfg, "refresh", REFRESH_TTL_S)
    return {
        "access_token": access, "token_type": "bearer",
        "expires_in": ACCESS_TTL_S, "expires_at": exp,
        "refresh_token": refresh, "user": user_object(cfg),
    }


def verify(token: str, cfg: LocalAuthConfig, typ: str = "access") -> Optional[Dict[str, Any]]:
    """The claims of a valid local token of type ``typ``; ``None`` for anything else."""
    if not token:
        return None
    try:
        if jwt.get_unverified_header(token).get("alg") != "HS256":
            return None
        claims = jwt.decode(
            token, cfg.secret, algorithms=["HS256"], audience=AUDIENCE, issuer=ISSUER,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )
    except InvalidTokenError:
        return None
    if claims.get("typ") != typ or claims.get("sub") != cfg.user_id:
        return None
    return claims


def user_object(cfg: LocalAuthConfig) -> Dict[str, Any]:
    """The GoTrue user record supabase-js expects from ``/auth/v1/user``."""
    return {
        "id": cfg.user_id, "aud": AUDIENCE, "role": "authenticated",
        "email": cfg.email, "email_confirmed_at": "1970-01-01T00:00:00Z",
        "app_metadata": {"provider": "avaloka-local", "providers": ["avaloka-local"]},
        "user_metadata": {"full_name": "Local user"},
        "identities": [], "created_at": "1970-01-01T00:00:00Z",
        "updated_at": "1970-01-01T00:00:00Z", "is_anonymous": False,
    }


# ---------------------------------------------------------------------------
# Password check, with a small lockout
# ---------------------------------------------------------------------------

_FAIL_WINDOW_S = 300
_FAIL_LIMIT = 10
_failures: list[float] = []
_fail_lock = threading.Lock()


def locked_out(now: Optional[float] = None) -> bool:
    now = time.monotonic() if now is None else now
    with _fail_lock:
        _failures[:] = [t for t in _failures if now - t < _FAIL_WINDOW_S]
        return len(_failures) >= _FAIL_LIMIT


def check_password(cfg: LocalAuthConfig, email: str, password: str) -> bool:
    ok_email = hmac.compare_digest((email or "").strip().lower().encode(), cfg.email.encode())
    ok_pw = hmac.compare_digest((password or "").encode(), cfg.password.encode())
    if ok_email and ok_pw:
        return True
    with _fail_lock:
        _failures.append(time.monotonic())
    return False
