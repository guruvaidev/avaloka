"""Systems under test: Avaloka and the baselines it is measured against.

Adapters are resolved by key via :func:`get_adapter`, so a ``RunConfig.sut`` string
in a suite definition maps to a concrete implementation without the suite importing
it directly (keeps the Avaloka app off the import path until actually needed).
"""

from __future__ import annotations

from typing import Dict, Type

from .base import SystemUnderTest


_REGISTRY: Dict[str, Type[SystemUnderTest]] = {}


def register_adapter(key: str):
    def _wrap(cls: Type[SystemUnderTest]) -> Type[SystemUnderTest]:
        _REGISTRY[key] = cls
        return cls
    return _wrap


def get_adapter(key: str) -> SystemUnderTest:
    if key not in _REGISTRY:
        # Import side-effect modules lazily so registration happens on first use.
        from . import baselines, avaloka_graph, local_swarm, reliability, dab_command  # noqa: F401
    if key not in _REGISTRY:
        raise KeyError(f"No adapter registered for sut={key!r}. "
                       f"Known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


def list_adapters() -> list[str]:
    from . import baselines, avaloka_graph, local_swarm, reliability, dab_command  # noqa: F401
    return sorted(_REGISTRY)


def requires_cloud(key: str) -> bool:
    """Whether an adapter drives real cloud/Groq resources (local-first gate)."""
    return bool(getattr(get_adapter(key), "requires_cloud", False))


def config_requires_cloud(config) -> bool:
    """Same gate, but per-config.

    ``llm_direct`` is one adapter addressing three providers: two hosted and
    billable, one local and free. A flat per-adapter flag can only be wrong for
    one of them — flag it and the free local control is gated behind
    ``--allow-cloud``; leave it unflagged and a full OpenRouter sweep runs by
    default and spends money nobody authorised. Adapters that care implement
    ``config_requires_cloud``; the rest keep the class attribute.
    """
    adapter = get_adapter(config.sut)
    hook = getattr(adapter, "config_requires_cloud", None)
    if callable(hook):
        return bool(hook(config))
    return bool(getattr(adapter, "requires_cloud", False))
