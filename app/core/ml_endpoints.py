"""Lifecycle of a serving endpoint, as a pure function.

The endpoint row in Postgres (``ml_endpoints``) is the truth the chat and the
inference panel read. Every writer -- the API, the Celery tasks, the reaper --
goes through :func:`transition`, so an illegal step (``pending -> ready``
without a deploy, ``ready -> deploying`` without a stop) is refused in one
place rather than becoming a row nobody can explain.

Idle scale-to-zero is deliberately *not* a state: a ``ready`` endpoint whose
pod is asleep is still ready; the next request wakes it.
"""
from __future__ import annotations

from typing import Dict, Tuple

PENDING = "pending"
DEPLOYING = "deploying"
READY = "ready"
FAILED = "failed"
STOPPED = "stopped"

STATES: Tuple[str, ...] = (PENDING, DEPLOYING, READY, FAILED, STOPPED)

#: (current state, event) -> next state. Anything not listed is illegal.
_TRANSITIONS: Dict[Tuple[str, str], str] = {
    (PENDING, "start_deploy"): DEPLOYING,
    (DEPLOYING, "deployed"): READY,
    (DEPLOYING, "deploy_failed"): FAILED,
    # A user may stop while a deploy is in flight (cancel) or once it is ready.
    (PENDING, "stop"): STOPPED,
    (DEPLOYING, "stop"): STOPPED,
    (READY, "stop"): STOPPED,
    # Redeploying a stopped or failed endpoint starts the cycle again.
    (FAILED, "redeploy"): PENDING,
    (STOPPED, "redeploy"): PENDING,
}

EVENTS: Tuple[str, ...] = tuple(sorted({event for _, event in _TRANSITIONS}))


class IllegalTransition(ValueError):
    """The requested event is not allowed from the current state."""


def transition(current: str, event: str) -> str:
    """Return the state after ``event`` from ``current`` or raise.

    >>> transition("pending", "start_deploy")
    'deploying'
    """
    if current not in STATES:
        raise IllegalTransition(f"unknown endpoint state {current!r}")
    if event not in EVENTS:
        raise IllegalTransition(f"unknown endpoint event {event!r}")
    try:
        return _TRANSITIONS[(current, event)]
    except KeyError:
        raise IllegalTransition(f"cannot {event} an endpoint that is {current}") from None


def can(current: str, event: str) -> bool:
    """True when ``event`` is legal from ``current``."""
    return (current, event) in _TRANSITIONS


def is_servable(status: str) -> bool:
    """Only a ready endpoint may receive predictions."""
    return status == READY
