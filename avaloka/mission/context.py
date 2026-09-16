"""The MissionContext threads through every firefly.

It carries the immutable mission specification (goal, target, metric, budget),
the execution mode and the live accounting objects (ledger + budget). Fireflies
read the spec, do their work, append a :class:`WorkUnit` to the ledger and write
their artifacts under ``output_dir``.
"""

from __future__ import annotations

import enum
import platform
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from avaloka.mission.budget import Budget
from avaloka.mission.ledger import Ledger


class MissionKind(str, enum.Enum):
    ANALYZE = "analyze"
    TRAIN = "train"
    DEPLOY = "deploy"


class ExecutionMode(str, enum.Enum):
    """Software economics are separated from compute economics (spec §5)."""

    LOCAL = "local"            # free, data stays on the user's machine
    BYOC = "byoc"             # bring-your-own-compute: customer pays cloud, Avaloka the control plane
    MANAGED = "managed"        # Avaloka-managed execution with hard cost limits


@dataclass
class MissionContext:
    kind: MissionKind
    goal: str
    data_source: str
    output_dir: Path
    budget: Budget
    ledger: Ledger = field(default_factory=Ledger)

    # task specification (train missions)
    target: str | None = None
    metric: str | None = None
    objective: str | None = None
    deployable: bool = False
    deployment_latency_ms: float | None = None

    execution_mode: ExecutionMode = ExecutionMode.LOCAL
    tier: str = "community"
    loaded_hourly_rate: float = 80.0

    mission_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)
    _start_perf: float = field(default_factory=time.perf_counter)

    # shared blackboard for inter-firefly handoff (kept small & serialisable)
    blackboard: dict[str, Any] = field(default_factory=dict)

    def wall_clock_seconds(self) -> float:
        return time.perf_counter() - self._start_perf

    def path(self, *parts: str) -> Path:
        p = self.output_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def ensure_output(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def environment(self) -> dict[str, str]:
        """A reproducibility fingerprint embedded into lineage / environment.lock."""
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "avaloka_mission_id": self.mission_id,
        }
