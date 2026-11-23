"""How wide a swarm may fan out, per edition. The one table.

Swarm itself ships in every edition, including the open-source one: a mission
may clone Avaloka into sub-Avalokas that work the same dataset in parallel
(:mod:`avaloka.coordinator`). What differs by edition is how many clones one
mission may dispatch:

==============  ======================  ======================  ==========
edition         clones per mission      run in parallel         where
==============  ======================  ======================  ==========
oss / free      2                       2                       local
professional    16                      16                      local
enterprise      unlimited               unlimited               local
==============  ======================  ======================  ==========

The open-source and Free limit is a *preview*: enough to see a fleet converge,
not enough to replace a team. The number is two because the swarm goes public
sized for a laptop: each clone is a full sub-Avaloka running its own firefly
swarm on a shard, and two of them is what a laptop runs smoothly. It is a
product decision, and it is written in published code, so a self-hoster can
read it and, under Apache 2.0, change it. It is stated here rather than
dressed up as a gate that cannot be crossed.

Every edition runs clones locally. The coordinator is a thread pool on one
machine; there is no cluster fan-out in this build, for any edition, so none is
advertised.

Resolution follows installed code, as ``app.core.editions`` does: without the
commercial package every edition resolves to the preview limits, whatever
``AVALOKA_EDITION`` claims.

This module imports nothing outside the standard library. The standalone CLI
(which never imports ``app.*``) and the server both read it, so the number a
user is told and the number that is enforced cannot drift apart.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

#: The preview limit for the open-source and Free editions. Two, so that the
#: swarm runs smoothly on a laptop: two full sub-Avalokas, each with its own
#: firefly swarm on a shard, is the headroom a laptop has.
PREVIEW_MAX_CLONES = 2

_EDITION_ALIASES = {"community": "oss", "pro": "professional"}
_EDITIONS = ("oss", "free", "professional", "enterprise")

_LABELS = {
    "oss": "the free open-source edition",
    "free": "the Free plan",
    "professional": "Professional",
    "enterprise": "Enterprise",
}


@dataclass(frozen=True)
class SwarmLimits:
    """The fan-out a mission is allowed. ``None`` means unlimited."""

    edition: str
    max_clones_per_mission: Optional[int]
    max_parallel_clones: Optional[int]
    execution: str = "local"
    cluster_fan_out: bool = False
    preview: bool = False

    @property
    def label(self) -> str:
        return _LABELS.get(self.edition, self.edition)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["label"] = self.label
        out["summary"] = self.summary()
        return out

    def summary(self) -> str:
        """One plain sentence describing these limits, for any UI."""
        if self.max_clones_per_mission is None:
            size = "no limit on clones per mission"
        else:
            size = f"up to {self.max_clones_per_mission} clones per mission"
        return f"Swarm on {self.label}: {size}, run locally."


_TABLE: dict[str, SwarmLimits] = {
    "oss": SwarmLimits("oss", PREVIEW_MAX_CLONES, PREVIEW_MAX_CLONES, preview=True),
    "free": SwarmLimits("free", PREVIEW_MAX_CLONES, PREVIEW_MAX_CLONES, preview=True),
    "professional": SwarmLimits("professional", 16, 16),
    "enterprise": SwarmLimits("enterprise", None, None),
}

#: For tests and embedders that size a fleet explicitly. Never resolved from
#: configuration.
UNLIMITED = SwarmLimits("unlimited", None, None)


def normalize_edition(value: Optional[str]) -> str:
    """Unknown values resolve to ``oss``, the least-privileged edition."""
    candidate = (value or "").strip().lower()
    candidate = _EDITION_ALIASES.get(candidate, candidate)
    return candidate if candidate in _EDITIONS else "oss"


def commercial_build() -> bool:
    """True when the commercial package is installed. An import probe, not a flag."""
    try:
        import avaloka_commercial  # noqa: F401  (import-only probe)
    except ImportError:
        return False
    return True


def limits_for(edition: Optional[str], *, has_commercial_build: Optional[bool] = None) -> SwarmLimits:
    """The swarm limits for an edition in this build.

    ``has_commercial_build`` is injectable for tests; production discovers it.
    """
    commercial = commercial_build() if has_commercial_build is None else has_commercial_build
    plan = normalize_edition(edition)
    if not commercial and plan in ("professional", "enterprise"):
        # The edition is claimed but its build is not installed. Say what this
        # build actually grants rather than what the environment claims.
        return _TABLE["oss"]
    return _TABLE[plan]


def limits_from_env() -> SwarmLimits:
    """The CLI's limits: ``AVALOKA_EDITION`` resolved against the installed build."""
    return limits_for(os.getenv("AVALOKA_EDITION"))


@dataclass(frozen=True)
class SwarmGrant:
    """What a mission asked for, what it gets, and the sentence that says so."""

    requested_clones: int
    clones: int
    requested_parallel: Optional[int]
    parallel: int
    limits: SwarmLimits
    #: How ``requested_clones`` was arrived at when nobody typed it: the
    #: sizing sentence from :mod:`avaloka.swarm_sizing`. ``None`` when the
    #: caller asked for the number explicitly.
    derived_from: Optional[str] = None

    @property
    def limited(self) -> bool:
        return self.clones < self.requested_clones or (
            self.requested_parallel is not None and self.parallel < self.requested_parallel
        )

    @property
    def message(self) -> str:
        """Empty when nothing was capped; otherwise says exactly what was."""
        if not self.limited:
            return ""
        cap = self.limits.max_clones_per_mission
        if self.clones < self.requested_clones:
            asked = (f"the data sized the fleet at {self.requested_clones}"
                     if self.derived_from else f"you asked for {self.requested_clones}")
            return (
                f"Swarm limited to {cap} clones on {self.limits.label}: "
                f"{asked}, so {self.clones} will run."
            )
        return (
            f"Swarm limited to {self.limits.max_parallel_clones} clones in parallel on "
            f"{self.limits.label}: you asked for {self.requested_parallel}, so "
            f"{self.parallel} will run at once."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_clones": self.requested_clones,
            "clones": self.clones,
            "requested_parallel": self.requested_parallel,
            "parallel": self.parallel,
            "limited": self.limited,
            "message": self.message,
            "derived_from": self.derived_from,
            "limits": self.limits.as_dict(),
        }


def grant(requested_clones: int, requested_parallel: Optional[int], limits: SwarmLimits,
          *, derived_from: Optional[str] = None) -> SwarmGrant:
    """Apply ``limits`` to a request. Parallelism never exceeds the clones granted.

    ``derived_from`` is the sizing sentence when the request came from the
    data rather than the caller; it changes the announced message ("the data
    sized the fleet at N" rather than "you asked for N") and is kept on the
    grant.
    """
    if requested_clones < 1:
        raise ValueError("a swarm needs at least one clone")
    if requested_parallel is not None and requested_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    clones = requested_clones
    if limits.max_clones_per_mission is not None:
        clones = min(clones, limits.max_clones_per_mission)
    parallel = min(requested_parallel or clones, clones)
    if limits.max_parallel_clones is not None:
        parallel = min(parallel, limits.max_parallel_clones)
    return SwarmGrant(requested_clones, clones, requested_parallel, max(1, parallel), limits,
                      derived_from=derived_from)
