"""The single interface every system under test implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.config import RunConfig
from ..core.task import BenchmarkTask, MissionOutcome


@runtime_checkable
class SystemUnderTest(Protocol):
    """Runs one mission and returns a :class:`MissionOutcome`.

    Implementations must be *side-effect honest*: report real cost, latency, and
    data-scanned figures, or leave them at their defaults — never fabricate them.
    The runner handles trial repetition, timing sanity, and aggregation; an adapter
    only executes a single mission.
    """

    #: Stable key used in ``RunConfig.sut`` and on the scorecard.
    key: str

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        ...
