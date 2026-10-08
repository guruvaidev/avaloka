"""Switches for product analytics. Every one is read at call time, not import
time, so an operator's change takes effect without a rebuild and a test can
flip one without reloading modules.

Precedence for "is capture on": ``AVALOKA_ANALYTICS`` if set, otherwise
``AVALOKA_TELEMETRY`` if it is set to an off value (the operator's umbrella
switch from PR #329 -- somebody who turned telemetry off did not mean "except
the analytics"), otherwise on.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

ENV_ENABLED = "AVALOKA_ANALYTICS"
ENV_UMBRELLA = "AVALOKA_TELEMETRY"
ENV_PROMPT_TEXT = "AVALOKA_ANALYTICS_PROMPT_TEXT"
ENV_RETENTION_DAYS = "AVALOKA_ANALYTICS_RETENTION_DAYS"
ENV_DB_URL = "AVALOKA_ANALYTICS_DB_URL"
ENV_EXPORT = "AVALOKA_ANALYTICS_EXPORT"
ENV_EXPORT_ENDPOINT = "AVALOKA_ANALYTICS_EXPORT_ENDPOINT"

DEFAULT_RETENTION_DAYS = 180

_TRUTHY = {"1", "on", "true", "yes", "enable", "enabled"}
_FALSEY = {"0", "off", "false", "no", "disable", "disabled"}


def _flag(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _FALSEY:
        return False
    if value in _TRUTHY:
        return True
    return None


def capture_enabled() -> bool:
    own = _flag(ENV_ENABLED)
    if own is not None:
        return own
    if _flag(ENV_UMBRELLA) is False:
        return False
    return True


def prompt_text_enabled() -> bool:
    """Whether question text is stored. Off leaves behavioural events intact."""
    own = _flag(ENV_PROMPT_TEXT)
    return True if own is None else own


def retention_days() -> int:
    try:
        days = int(os.environ.get(ENV_RETENTION_DAYS, DEFAULT_RETENTION_DAYS))
    except ValueError:
        return DEFAULT_RETENTION_DAYS
    return max(1, min(days, 3650))


def export_requested() -> bool:
    """Whether an operator has affirmatively switched export on.

    The collector prepares the exportable form of a question (names removed)
    only when this is true. With export off, no vocabulary lookup is made and
    no exportable text exists -- so turning export on later cannot ship
    questions captured before the operator chose to.
    """
    return capture_enabled() and _flag(ENV_EXPORT) is True


def export_endpoint() -> Optional[str]:
    """Where events are forwarded, or ``None``.

    Export needs BOTH an explicit on-switch and an endpoint, and there is no
    default host. On develop-1.6 neither is set by anything, so nothing leaves.
    """
    if not export_requested():
        return None
    endpoint = os.environ.get(ENV_EXPORT_ENDPOINT, "").strip()
    return endpoint if endpoint.startswith("https://") else None


def database_source() -> Optional[Tuple[str, str]]:
    """``(variable name, url)`` for the first database variable that is set, or
    ``None``. The name is kept so the store can say where its URL came from."""
    for name in (ENV_DB_URL, "POSTGRES_URL"):
        value = os.environ.get(name, "").strip()
        if value:
            return name, value
    return None


def database_url() -> Optional[str]:
    """The deployment's own database. Never Supabase: that channel is cloud-hosted
    and is inventoried as third-party egress; analytics rows stay in the cluster.

    ``None`` when neither variable is set. There is no local-file default: with
    nowhere configured to keep events, analytics is off (``store.default_store``),
    not writing somewhere nobody looks.
    """
    source = database_source()
    return source[1] if source else None
