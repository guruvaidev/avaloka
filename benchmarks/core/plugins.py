"""Suite discovery — make the harness extensible without editing the harness.

``registry.register_suite`` is already a clean decorator seam. What it lacked
was discovery: ``_ensure_loaded()`` hardcoded four imports, so adding a track
meant editing the registry, and a track living outside this repository could
not participate at all.

That is the gap the "everything is a plugin" harnesses are pointing at. The
seam existed; the loading did not. This module supplies the loading, in three
layers that cost progressively more:

1. **Built-in tracks** — the modules that ship here, listed once.
2. **Local drop-ins** — any module under ``benchmarks/tracks/*/track.py``,
   found by walking the directory. Adding a track becomes creating a folder.
3. **Installed plugins** — third-party packages advertising the
   ``avaloka.benchmark_suites`` entry point. A track can then live in its own
   repository and be pip-installed, which is what makes an internal harness
   usable by anyone else.

Failure to load one plugin must never prevent the others from running: a
benchmark suite that cannot import is a broken suite, not a broken harness.
Load errors are collected and reported rather than raised.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Entry-point group a third-party package advertises to add suites.
ENTRY_POINT_GROUP = "avaloka.benchmark_suites"

#: Modules that ship in this repository. Listed explicitly because they are
#: part of the harness contract, not optional extras.
BUILTIN_MODULES: Tuple[str, ...] = (
    "benchmarks.tracks.adaptive_sampling.track",
    "benchmarks.tracks.swarm_ablation.track",
    "benchmarks.tracks.reliability.track",
    "benchmarks.external.dab.adapter",
)


@dataclass
class LoadReport:
    """What loaded, what did not, and why."""

    loaded: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed

    def as_dict(self) -> Dict[str, object]:
        return {
            "loaded": sorted(self.loaded),
            "failed": dict(self.failed),
            "skipped": dict(self.skipped),
            "n_loaded": len(self.loaded),
            "n_failed": len(self.failed),
        }

    def summary(self) -> str:
        text = f"{len(self.loaded)} suite module(s) loaded"
        if self.failed:
            text += f", {len(self.failed)} failed: " + ", ".join(sorted(self.failed))
        if self.skipped:
            text += f", {len(self.skipped)} skipped"
        return text


def _import(name: str, report: LoadReport) -> None:
    if name in report.loaded:
        return
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - one bad suite must not sink the run
        report.failed[name] = f"{type(exc).__name__}: {exc}"
        logger.warning("[plugins] could not load %s: %s", name, exc)
    else:
        report.loaded.append(name)


def discover_local_tracks(tracks_dir: Optional[Path] = None) -> List[str]:
    """Module paths for every ``benchmarks/tracks/<name>/track.py``.

    Walking the directory rather than listing modules is what turns "add a
    track" into "create a folder" instead of "create a folder and remember to
    edit the registry".
    """
    root = tracks_dir or Path(__file__).resolve().parent.parent / "tracks"
    if not root.is_dir():
        return []
    found: List[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        if (entry / "track.py").is_file():
            found.append(f"benchmarks.tracks.{entry.name}.track")
    return found


def discover_entry_point_suites() -> List[Tuple[str, str]]:
    """``(name, module)`` for installed packages advertising suites."""
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8
        return []
    try:
        eps = entry_points()
        selected = (eps.select(group=ENTRY_POINT_GROUP)
                    if hasattr(eps, "select") else eps.get(ENTRY_POINT_GROUP, []))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[plugins] entry-point scan failed: %s", exc)
        return []
    return [(ep.name, ep.value) for ep in selected]


def load_all(*, include_entry_points: bool = True,
             extra_modules: Sequence[str] = ()) -> LoadReport:
    """Load every discoverable suite module. Never raises on a bad plugin."""
    report = LoadReport()

    for name in BUILTIN_MODULES:
        _import(name, report)

    for name in discover_local_tracks():
        if name not in BUILTIN_MODULES:
            _import(name, report)

    if include_entry_points:
        for ep_name, module in discover_entry_point_suites():
            target = module.split(":", 1)[0]
            _import(target, report)
            if target in report.failed:
                report.failed[f"{ep_name} ({target})"] = report.failed.pop(target)

    for name in extra_modules:
        _import(name, report)

    logger.info("[plugins] %s", report.summary())
    return report
