"""The swarm console gate: one dependency for every swarm/trace route.

The swarm runs in every edition (``app.core.editions``: OPEN, capped per
edition by ``swarm_limits``). The *console* — the live /swarm page, the
mission routes that feed it (records, clones, episodes, per-clone telemetry)
and the operator's agent metrics — is Professional and Enterprise only, by
owner decision. The thread trace (``/threads/{id}/trace``) and the planner
graph in every format are NOT console: they show a user how their own prompt
was acted upon, and that is open in every edition. This module turns that decision into a FastAPI
dependency so a route states the gate in its signature and cannot forget it::

    @app.get("/api/swarm/missions")
    async def list_swarm_missions(user_id: str = Depends(require_swarm_console)):
        ...

The dependency answers 401 before 403: an anonymous caller learns nothing about
the edition, and an authenticated Free caller gets the one honest line from
:func:`app.core.editions.explain_swarm_console` plus a machine-readable
``code`` the UI keys its gate copy on.

:data:`CONSOLE_PATH_RE` names the route family. The contract test enumerates
``app.routes`` against it and drives every match as a Free caller, so a new
trace route under one of these prefixes that skips the dependency fails the
suite rather than shipping open.
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi import HTTPException, Request, status

from app.core.editions import GateReason, explain_swarm_console, swarm_console_gate

#: Every path that belongs to the console. Keep it a regex over the *route
#: template* (``{thread_id}``), not over concrete URLs.
CONSOLE_PATH_RE = re.compile(
    r"^(?:/api/swarm(?:/|$)"            # missions, records, clones, episodes, telemetry
    r"|/api/agent-metrics"               # per-agent aggregates across a user's traces
    r"|/api/trace(?:/|$)"                # room for the advanced-telemetry routes
    r"|/api/agent-trace(?:/|$))"
)

#: Open in every edition, by the owner's first 1.7 ask: a user sees how their
#: prompt was acted upon. The contract test holds these at 200 on oss.
OPEN_EVERYWHERE = ("/threads/{thread_id}/trace", "/threads/{thread_id}/planner-graph")

GATE_CODE = "swarm_console_unavailable"
CONSOLE_PAGE = "/swarm"


def is_console_path(path: str) -> bool:
    return bool(CONSOLE_PATH_RE.match(path or ""))


def console_capability(edition: Optional[str]) -> Dict[str, Any]:
    """The ``features.swarm_console`` block of /api/capabilities for ``edition``."""
    reason = swarm_console_gate(edition)
    return {
        "enabled": reason is GateReason.OPEN,
        "reason": reason.value,
        "page": CONSOLE_PAGE,
        "message": explain_swarm_console(edition),
    }


def gate_detail(edition: Optional[str]) -> Dict[str, Any]:
    """The 403 body. ``code`` is what the UI switches on."""
    return {
        "code": GATE_CODE,
        "reason": swarm_console_gate(edition).value,
        "edition": edition,
        "message": explain_swarm_console(edition),
    }


def dependency(
    resolve_user_id: Callable[[Request], Optional[str]],
    caller_edition: Callable[[Request], Awaitable[str]],
) -> Callable[[Request], Awaitable[str]]:
    """Build the dependency from the server's own identity helpers.

    Taking them as arguments keeps this module importable without
    ``app.api.server`` (and so testable without the HTTP stack); the server
    binds it once as ``require_swarm_console``.
    """

    async def require_swarm_console(request: Request) -> str:
        user_id = resolve_user_id(request)
        if not user_id:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid auth token")
        edition = await caller_edition(request)
        if swarm_console_gate(edition) is not GateReason.OPEN:
            raise HTTPException(status.HTTP_403_FORBIDDEN, gate_detail(edition))
        return user_id

    return require_swarm_console
