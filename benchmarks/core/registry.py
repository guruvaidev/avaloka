"""Suite registry: name -> (tasks, default configs) factory.

Tracks and external benchmarks register a *suite factory* here. Keeping the mapping
lazy (factories, not eagerly-built task lists) means importing this module is cheap
and never pulls in pandas/numpy or the Avaloka app until a suite is actually built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from .config import RunConfig
from .task import BenchmarkTask

SuiteFactory = Callable[[], Tuple[List[BenchmarkTask], List[RunConfig]]]


@dataclass(frozen=True)
class SuiteInfo:
    name: str
    kind: str        # "track" | "external"
    description: str
    factory: SuiteFactory


_SUITES: Dict[str, SuiteInfo] = {}


def register_suite(name: str, kind: str, description: str):
    def _wrap(factory: SuiteFactory) -> SuiteFactory:
        _SUITES[name] = SuiteInfo(name=name, kind=kind, description=description,
                                  factory=factory)
        return factory
    return _wrap


_LOAD_REPORT = None


def _ensure_loaded() -> None:
    """Load every discoverable suite module, once.

    Previously this hardcoded four imports, so adding a track meant editing
    this file and a track outside the repository could not participate at all.
    Discovery now walks benchmarks/tracks/ and reads the
    ``avaloka.benchmark_suites`` entry point, so adding a suite is creating a
    folder or installing a package.

    A suite that fails to import is recorded and skipped rather than raised:
    one broken benchmark is a broken benchmark, not a broken harness.
    """
    global _LOAD_REPORT
    if _LOAD_REPORT is not None:
        return
    from .plugins import load_all
    _LOAD_REPORT = load_all()


def load_report():
    """What loaded and what did not, for diagnostics and CI."""
    _ensure_loaded()
    return _LOAD_REPORT


def list_suites() -> List[SuiteInfo]:
    _ensure_loaded()
    return sorted(_SUITES.values(), key=lambda s: (s.kind, s.name))


def get_suite(name: str) -> Tuple[List[BenchmarkTask], List[RunConfig]]:
    _ensure_loaded()
    if name not in _SUITES:
        raise KeyError(f"Unknown suite {name!r}. Known: {[s.name for s in list_suites()]}")
    return _SUITES[name].factory()
