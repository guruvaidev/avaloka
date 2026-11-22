"""DB_ENCRYPTION_KEY: provisioned automatically; a service refuses to start only if it still has none.

Every sealed credential in Avaloka -- cloud connection rows, MCP and GitHub
tokens, customer database connection strings at rest and in transit
(app/mcp_server/credentials.py), chat-typed passwords (app/core/typed_secrets.py)
-- is AES-GCM under a key derived from DB_ENCRYPTION_KEY.

At startup each service (the API, the MCP server, the standalone onboarding
service) calls :func:`require_db_encryption_key`:

1. a usable key in the process (environment or module) is used as is;
2. otherwise -- unset, empty, or a value published in this repository -- the
   generated secrets are provisioned from the .env file
   (app/core/provisioning.py): an existing value there is reused, else, where a
   service may generate (provisioning.startup_may_generate: on a host; in a
   container only with AVALOKA_AUTO_PROVISION=1; never in Kubernetes), one is
   generated and saved there, with a warning; the value is never printed. A
   container that may not generate adopts what its env file holds;
3. only if there is still no usable key does the service refuse to start, with
   how to fix it.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from app.core.provisioning import (
    GENERATED_SECRETS, PUBLISHED_VALUES, adopt_from_file, default_env_file, is_published, is_usable,
    provision, startup_may_generate,
)

logger = logging.getLogger(__name__)

#: Kept for callers of the earlier name; the one list lives in provisioning.
PUBLISHED_DB_ENCRYPTION_KEYS = PUBLISHED_VALUES

HOW_TO_SET = (
    "Avaloka generates DB_ENCRYPTION_KEY on first start and saves it to the .env file "
    "($AVALOKA_ENV_FILE, default ./.env); that failed or was switched off "
    "(AVALOKA_AUTO_PROVISION=0, or a container, which never generates its own). Fix: "
    "make the .env file writable, or run `python -m app.core.provisioning` "
    "(scripts/install.sh does), or set it yourself: `DB_ENCRYPTION_KEY=$(openssl rand -hex 32)` "
    "in .env and pass that file to every container (env_file). On Kubernetes the chart "
    "generates it into the release Secret (leave secrets.dbEncryptionKey empty); with "
    "existingSecret, that Secret must carry DB_ENCRYPTION_KEY. Use the same value for the "
    "API, the MCP/onboarding service and the Supabase edge functions; changing it makes "
    "previously sealed credentials unreadable. See docs/INSTALL.md, Configuration."
)


class InsecureEncryptionKey(RuntimeError):
    """No usable DB_ENCRYPTION_KEY even after provisioning; the service must not start."""


def current_key() -> str:
    """The key the sealing code uses: cloud_connections' module value, else the environment."""
    try:
        import app.api.cloud_connections as cc
        value: Optional[str] = cc.DB_ENCRYPTION_KEY
    except Exception:  # pragma: no cover - the module always imports in the app
        value = None
    return (value if value is not None else os.getenv("DB_ENCRYPTION_KEY") or "").strip()


def _adopt(value: str) -> None:
    """Make ``value`` the key the sealing code uses, in this process."""
    os.environ["DB_ENCRYPTION_KEY"] = value
    try:
        import app.api.cloud_connections as cc
        cc.DB_ENCRYPTION_KEY = value
        cc._AESGCM_KEY = None
    except Exception:  # pragma: no cover
        pass


def key_problem(value: Optional[str]) -> Optional[str]:
    """Why ``value`` is not usable as DB_ENCRYPTION_KEY, or ``None`` when it is."""
    v = (value or "").strip()
    if not v:
        return "DB_ENCRYPTION_KEY is empty"
    if is_published(v):
        return "DB_ENCRYPTION_KEY is a published value (it ships in this repository), so it seals nothing"
    return None


def require_db_encryption_key(service: str) -> None:
    """Provision the generated secrets if needed; raise InsecureEncryptionKey only if the key is still unusable."""
    others_missing = any(not is_usable(os.getenv(n)) for n in GENERATED_SECRETS if n != "DB_ENCRYPTION_KEY")
    if key_problem(current_key()) is None and not others_missing:
        return
    if startup_may_generate():
        os.environ["DB_ENCRYPTION_KEY"] = current_key()
        try:
            provision(default_env_file(), GENERATED_SECRETS)
            _adopt(os.environ.get("DB_ENCRYPTION_KEY", ""))
        except OSError as exc:
            logger.error("%s: could not provision secrets in %s: %s", service, default_env_file(),
                         type(exc).__name__)
    else:
        # A container (or provisioning off): never generate, but adopt what the
        # env file already holds, for every generated secret.
        os.environ["DB_ENCRYPTION_KEY"] = current_key()
        adopt_from_file(default_env_file(), GENERATED_SECRETS)
        if key_problem(os.environ.get("DB_ENCRYPTION_KEY", "")) is None:
            _adopt(os.environ["DB_ENCRYPTION_KEY"].strip())
    problem = key_problem(current_key())
    if problem:
        raise InsecureEncryptionKey(f"{service} refuses to start: {problem}. {HOW_TO_SET}")
