"""Optional capabilities the conversational agent uses when present.

The conversational agent was written against ``avaloka_agent.swarm``, importing
``plan_swarm`` and ``announce`` at module scope. That made a hard dependency out
of something the code itself describes as optional — the comment at the call
site reads *"it narrates the real pipeline, never routes"*, so swarm here is a
narration affordance, not an execution path.

That distinction is what makes this separable. Under the edition boundary swarm
is a **commercial** capability and conversational analysis is **core**, so the
open-source build has to run without it. Rather than fork the agent, the import
becomes a lookup that degrades to a no-op.

The contract is deliberately narrow — two functions with total behaviour when
absent:

* :func:`plan_swarm` returns ``[]`` — no plan to narrate
* :func:`announce` returns ``""`` — nothing to say

A caller that already handles "no plan" handles "no swarm module", so the
agent needed no branching beyond what it had.

Resolution is by import probe, not a flag. A build either ships the module or
it does not; a boolean someone can flip would let an open-source deployment
claim a capability whose code is absent, which is the failure the edition model
exists to prevent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class SwarmProvider(Protocol):
    """What the agent needs from a swarm implementation."""

    def plan_swarm(self, intent: str) -> List[Dict[str, Any]]: ...
    def announce(self, intent: str) -> str: ...


class _AbsentSwarm:
    """Total no-op used when no swarm implementation is installed.

    Returns empty rather than raising: a missing optional capability is a
    normal state for an open-source deployment, not an error the user should
    ever see.
    """

    available = False

    def plan_swarm(self, intent: str) -> List[Dict[str, Any]]:
        return []

    def announce(self, intent: str) -> str:
        return ""


class _ModuleSwarm:
    """Adapts the real swarm module to :class:`SwarmProvider`."""

    available = True

    def __init__(self, module: Any) -> None:
        self._module = module

    def plan_swarm(self, intent: str) -> List[Dict[str, Any]]:
        try:
            return list(self._module.plan_swarm(intent) or [])
        except Exception:  # noqa: BLE001 - narration must never break a reply
            logger.warning("[capabilities] swarm.plan_swarm failed; continuing without it",
                           exc_info=True)
            return []

    def announce(self, intent: str) -> str:
        try:
            return str(self._module.announce(intent) or "")
        except Exception:  # noqa: BLE001
            logger.warning("[capabilities] swarm.announce failed; continuing without it",
                           exc_info=True)
            return ""


_swarm: Optional[Any] = None


def get_swarm() -> Any:
    """The installed swarm provider, or a no-op when none is present."""
    global _swarm
    if _swarm is not None:
        return _swarm
    try:
        from app.agents.avaloka_agent import swarm as swarm_module  # type: ignore
    except ImportError:
        logger.debug("[capabilities] swarm not installed; conversational agent "
                     "runs without swarm narration")
        _swarm = _AbsentSwarm()
    else:
        missing = [n for n in ("plan_swarm", "announce") if not hasattr(swarm_module, n)]
        if missing:
            logger.warning("[capabilities] swarm module is present but missing %s; "
                           "treating it as absent", missing)
            _swarm = _AbsentSwarm()
        else:
            logger.info("[capabilities] swarm narration enabled")
            _swarm = _ModuleSwarm(swarm_module)
    return _swarm


def swarm_available() -> bool:
    return bool(getattr(get_swarm(), "available", False))


def plan_swarm(intent: str) -> List[Dict[str, Any]]:
    """Narration plan for *intent*, or ``[]`` when swarm is not installed."""
    return get_swarm().plan_swarm(intent)


def announce(intent: str) -> str:
    """Narration line for *intent*, or ``""`` when swarm is not installed."""
    return get_swarm().announce(intent)


def reset_cache() -> None:
    """Test helper: forget the resolved provider."""
    global _swarm
    _swarm = None


def describe() -> Dict[str, Any]:
    """Diagnostics for a support request or an /edition endpoint."""
    return {"swarm": {"available": swarm_available(), "reason":
                      "installed" if swarm_available() else "not present in this build"}}
