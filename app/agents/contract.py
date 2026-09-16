"""The agent contract and registry.

Fifteen agents currently wire themselves into the LangGraph each in their own
way, and ``ETLState`` carries 118 fields with no structure per stage. Adding
more agents on that basis makes the graph progressively harder to reason about,
so the contract lands before the agents do.

An agent here is a small, declared thing:

* a **name** and a **stage** in the lifecycle,
* the state keys it **reads** and **writes**,
* a ``run(state) -> dict`` that returns *only* its updates.

Two properties fall out, and both are enforced by tests:

1. **An agent cannot silently clobber another agent's state.** Writes are
   declared, and :func:`run_agent` drops undeclared keys rather than letting
   them through. A typo becomes a caught error, not a corrupted field
   somewhere downstream.
2. **An agent never breaks the graph.** ``run_agent`` catches exceptions and
   converts them into a structured failure recorded on the state. A crashing
   evaluator must not take down an analysis the user is waiting on.

This is deliberately not a framework. It is a dataclass, a registry dict, and
a wrapper that enforces the two properties above.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

State = Dict[str, Any]
AgentFn = Callable[[State], Mapping[str, Any]]


class Stage(str, Enum):
    """Where an agent sits in the data-science lifecycle.

    Ordering is the pipeline order and is used to sanity-check that an agent
    does not read a key produced by a later stage.
    """

    UNDERSTAND = "understand"   # sampling, profiling
    PLAN = "plan"               # planner, summarizer
    PREPARE = "prepare"         # transfer, feature engineering, coder
    INTEGRITY = "integrity"     # leakage and data-integrity checks
    MODEL = "model"             # training
    EVALUATE = "evaluate"       # cross-validation, baselines, metrics
    EXPLAIN = "explain"         # attribution, importance
    VERIFY = "verify"           # claim checking
    DEPLOY = "deploy"           # serving
    MONITOR = "monitor"         # drift
    PRESENT = "present"         # visualization, reporting


_STAGE_ORDER = {stage: i for i, stage in enumerate(Stage)}


@dataclass(frozen=True)
class AgentSpec:
    """What an agent declares about itself."""

    name: str
    stage: Stage
    reads: Tuple[str, ...] = ()
    writes: Tuple[str, ...] = ()
    description: str = ""
    #: When False the graph should skip it rather than fail if a read key is
    #: missing. Most analysis agents are optional; training is not.
    required: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"agent name must be alphanumeric/underscore: {self.name!r}")
        if not self.writes:
            raise ValueError(f"agent {self.name!r} declares no writes; it would be a no-op")
        overlap = set(self.reads) & set(self.writes)
        if overlap:
            # Read-modify-write on the same key makes ordering unanalysable.
            raise ValueError(
                f"agent {self.name!r} both reads and writes {sorted(overlap)}; "
                "split the key or rename the output"
            )


_REGISTRY: Dict[str, AgentSpec] = {}
_IMPLS: Dict[str, AgentFn] = {}


def register(spec: AgentSpec, fn: AgentFn) -> AgentFn:
    """Register an agent. Duplicate names are an error, not a silent overwrite."""
    if spec.name in _REGISTRY:
        raise ValueError(f"agent {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec
    _IMPLS[spec.name] = fn
    logger.debug("[agents] registered %s (%s)", spec.name, spec.stage.value)
    return fn


def agent(spec: AgentSpec) -> Callable[[AgentFn], AgentFn]:
    """Decorator form of :func:`register`."""
    def decorate(fn: AgentFn) -> AgentFn:
        register(spec, fn)
        return fn
    return decorate


def get(name: str) -> Tuple[AgentSpec, AgentFn]:
    if name not in _REGISTRY:
        raise KeyError(f"no agent registered as {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name], _IMPLS[name]


def registry() -> Dict[str, AgentSpec]:
    """A copy of the registry, so callers cannot mutate it."""
    return dict(_REGISTRY)


def by_stage(stage: Stage) -> Tuple[AgentSpec, ...]:
    return tuple(s for s in _REGISTRY.values() if s.stage is stage)


def clear_registry() -> None:
    """Test helper. Never call this from application code."""
    _REGISTRY.clear()
    _IMPLS.clear()


# --------------------------------------------------------------------------- #
# Failure record — a crashing agent must not take the graph down
# --------------------------------------------------------------------------- #

AGENT_ERRORS_KEY = "agent_errors"


def _record_failure(name: str, exc: BaseException) -> Dict[str, Any]:
    return {
        AGENT_ERRORS_KEY: [{
            "agent": name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6),
        }]
    }


def missing_reads(spec: AgentSpec, state: Mapping[str, Any]) -> Tuple[str, ...]:
    """Declared read keys absent from (or None in) the state."""
    return tuple(k for k in spec.reads if state.get(k) is None)


def run_agent(name: str, state: State) -> Dict[str, Any]:
    """Run a registered agent and return only its declared, non-empty updates.

    Undeclared keys are dropped with a warning rather than written. That turns
    a typo into a visible complaint instead of a mystery field that some other
    agent later reads and trusts.
    """
    spec, fn = get(name)

    absent = missing_reads(spec, state)
    if absent:
        message = f"{name}: missing required state {list(absent)}"
        if spec.required:
            logger.error("[agents] %s", message)
            return _record_failure(name, RuntimeError(message))
        logger.info("[agents] skipping %s — %s", name, message)
        return {}

    started = time.perf_counter()
    try:
        raw = fn(state) or {}
    except Exception as exc:  # noqa: BLE001 - an agent must never break the graph
        logger.exception("[agents] %s failed", name)
        return _record_failure(name, exc)
    elapsed = time.perf_counter() - started

    if not isinstance(raw, Mapping):
        return _record_failure(
            name, TypeError(f"agent returned {type(raw).__name__}, expected a mapping")
        )

    declared = set(spec.writes)
    updates = {k: v for k, v in raw.items() if k in declared}
    undeclared = sorted(set(raw) - declared)
    if undeclared:
        logger.warning("[agents] %s returned undeclared keys %s; dropped", name, undeclared)

    logger.info("[agents] %s wrote %s in %.2fs", name, sorted(updates), elapsed)
    return updates


def pipeline_order(names: Iterable[str]) -> Tuple[str, ...]:
    """Sort agent names into lifecycle order. Stable within a stage."""
    resolved = [(_STAGE_ORDER[_REGISTRY[n].stage], i, n) for i, n in enumerate(names)]
    return tuple(n for _, _, n in sorted(resolved))


def check_wiring(names: Iterable[str]) -> Tuple[str, ...]:
    """Return problems found in a proposed agent ordering.

    Catches the mistake that actually happens: an agent reading a key that only
    a later-stage agent produces, which works by accident when a cache is warm
    and fails on a cold run.
    """
    order = list(names)
    problems = []
    produced_at: Dict[str, int] = {}
    for position, name in enumerate(order):
        for key in _REGISTRY[name].writes:
            produced_at.setdefault(key, position)
    for position, name in enumerate(order):
        for key in _REGISTRY[name].reads:
            source = produced_at.get(key)
            if source is not None and source > position:
                problems.append(
                    f"{name} (position {position}) reads {key!r}, "
                    f"produced later by {order[source]} (position {source})"
                )
    return tuple(problems)
