"""The budget one recipe run may spend: rows, bytes, memory, threads, time.

A run cannot exceed its budget. A source over ``max_rows`` or ``max_bytes``
is analysed on a deterministic sample of ``sample_rows`` rows and the report
says so (the ``sampling`` caveat); a source over the sampled lane's ceiling
(``avaloka.workload.SAMPLED_MAX_ROWS`` / ``SAMPLED_MAX_BYTES``) is refused
``too_large``; a run past ``wall_clock_s`` is refused ``engine_timeout``.
A breach is a verdict in plain words, never an exception the user sees
(design §3.2 rule 4, §5 "Execution", §9.3).

The defaults are the ONLINE lane (``avaloka.workload``): what the API pod
runs in-process. A deployment overrides them with ``AVALOKA_RECIPE_*``
variables (:func:`from_env`).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields
from typing import Mapping

from avaloka.workload import ONLINE_MAX_BYTES, ONLINE_MAX_ROWS

#: DuckDB's ``memory_limit`` syntax as this module accepts it: whole MB or GB.
_MEMORY_LIMIT = re.compile(r"^\d+(MB|GB)$")

#: The environment variable behind each field of :class:`Budget`.
ENV_NAMES: Mapping[str, str] = {
    "max_rows": "AVALOKA_RECIPE_MAX_ROWS",            # rows above which the run samples
    "max_bytes": "AVALOKA_RECIPE_MAX_BYTES",          # bytes on disk (or in memory) above which the run samples
    "sample_rows": "AVALOKA_RECIPE_SAMPLE_ROWS",      # rows in the sample
    "memory_limit": "AVALOKA_RECIPE_MEMORY_LIMIT",    # DuckDB SET memory_limit, e.g. 512MB or 2GB
    "threads": "AVALOKA_RECIPE_THREADS",              # DuckDB SET threads
    "wall_clock_s": "AVALOKA_RECIPE_WALL_CLOCK_S",    # seconds a run may take, end to end
    "seed": "AVALOKA_RECIPE_SEED",                    # the sample's seed; the same file samples the same rows
}


@dataclass(frozen=True)
class Budget:
    """What one run may spend. Every field is validated at construction."""
    max_rows: int = ONLINE_MAX_ROWS
    max_bytes: int = ONLINE_MAX_BYTES
    sample_rows: int = ONLINE_MAX_ROWS
    memory_limit: str = "512MB"
    threads: int = 2
    wall_clock_s: float = 60.0
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("max_rows", "max_bytes", "sample_rows", "threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Budget.{name} must be a positive integer, not {value!r}")
        if not isinstance(self.wall_clock_s, (int, float)) or isinstance(self.wall_clock_s, bool) \
                or self.wall_clock_s <= 0:
            raise ValueError(f"Budget.wall_clock_s must be a positive number, not {self.wall_clock_s!r}")
        if not isinstance(self.memory_limit, str) or not _MEMORY_LIMIT.match(self.memory_limit):
            raise ValueError(f"Budget.memory_limit must look like 512MB or 2GB, not {self.memory_limit!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError(f"Budget.seed must be a non-negative integer, not {self.seed!r}")


DEFAULT = Budget()


def from_env(environ: Mapping[str, str] | None = None) -> Budget:
    """The default budget with every ``AVALOKA_RECIPE_*`` override applied.

    An unset variable keeps the default; a set one that does not parse
    raises ``ValueError`` naming the variable, because a deployment that
    meant to cap a run must not silently run uncapped.
    """
    env = os.environ if environ is None else environ
    values: dict[str, object] = {}
    for f in fields(Budget):
        raw = env.get(ENV_NAMES[f.name])
        if raw is None or raw.strip() == "":
            continue
        raw = raw.strip()
        try:
            if f.name == "memory_limit":
                values[f.name] = raw.upper()
            elif f.name == "wall_clock_s":
                values[f.name] = float(raw)
            else:
                values[f.name] = int(raw)
        except ValueError as exc:
            raise ValueError(f"{ENV_NAMES[f.name]}={raw!r} is not a valid {f.name}") from exc
    try:
        return Budget(**values)
    except ValueError as exc:
        raise ValueError(f"{exc} (from the AVALOKA_RECIPE_* environment)") from exc
