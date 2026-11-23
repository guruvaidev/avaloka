"""How many clones a dataset calls for, from what is known about it.

The swarm's width used to be whatever the caller typed (``--swarm N``) or a
flat default of four. This module sizes the fleet from the workload plan
(rows, bytes, lane: :mod:`avaloka.workload`) and the source's native
statistics (row groups or partitions: :mod:`avaloka.source_statistics`), so
that a 12-row-group Parquet file and a 10-million-row table get different
fleets without anyone guessing a number. The edition cap is applied *after*
this, by :func:`avaloka.swarm_limits.grant`, and announced; the open-source
and Free editions therefore run exactly their preview of two whatever the
data says, and are told why.

The formula, stated once here and repeated in every ``derived_from`` string:

    by_rows  = ceil(n_rows  / ROWS_PER_CLONE)        (25,000 rows per clone)
    by_bytes = ceil(n_bytes / BYTES_PER_CLONE)       (64 MiB per clone)
    by_size  = clamp(max(by_rows, by_bytes), MIN_CLONES, MAX_CLONES)
    clones   = max(MIN_CLONES, min(partitions, by_size))  when the source is
               partitioned (Parquet row groups, native table partitions),
               by_size otherwise.

One clone per partition or row group, up to the size-based count: a shard
that follows the file's own boundaries is the one a reader can skip to, and
more clones than partitions would split a partition between two readers.
Indexes are reported but do not change the count; they shape the scan inside
a clone, not the number of clones. ``MIN_CLONES`` is two because a fleet of
one is a mission, not a swarm; ``MAX_CLONES`` is a sanity ceiling for one
machine's thread pool, not an edition limit.

This module imports nothing outside the standard library, like
:mod:`avaloka.swarm_limits`: the CLI, the coordinator and the API all size a
fleet with the same arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

ROWS_PER_CLONE = 25_000
BYTES_PER_CLONE = 64 * 1024 * 1024
MIN_CLONES = 2
MAX_CLONES = 64

#: The fleet when the source cannot be sized (unreadable, or sizing itself
#: raised): the flat default the CLI and API used before 2.0, so a request
#: that fails later still fails in the coordinator with its trace finished.
FALLBACK_CLONES = 4


@dataclass(frozen=True)
class FleetSize:
    """A clone count and the sentence that derives it, numbers only."""

    clones: int
    derived_from: str
    n_rows: int = 0
    n_bytes: int = 0
    partitions: int = 0
    lane: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {"clones": self.clones, "derived_from": self.derived_from,
                "n_rows": self.n_rows, "n_bytes": self.n_bytes,
                "partitions": self.partitions, "lane": self.lane}


def size_fleet(n_rows: int, n_bytes: int, *, lane: Optional[str] = None,
               statistics: Optional[dict[str, Any]] = None, estimated: bool = False) -> FleetSize:
    """Derive the clone count from the workload and the source statistics.

    ``statistics`` is the ``source_statistics`` dictionary a resolved source
    carries (``partitioning: {partitioned, partitions, kind}`` and
    ``indexes``); ``lane`` is the workload lane, reported for context. The
    ``derived_from`` string names only counts and sizes, never a column or a
    file, so it can be recorded anywhere the trace goes.
    """
    n_rows, n_bytes = max(0, int(n_rows)), max(0, int(n_bytes))
    by_rows = math.ceil(n_rows / ROWS_PER_CLONE)
    by_bytes = math.ceil(n_bytes / BYTES_PER_CLONE)
    by_size = min(MAX_CLONES, max(MIN_CLONES, by_rows, by_bytes))
    stats = statistics or {}
    partitioning = stats.get("partitioning") or {}
    partitions = int(partitioning.get("partitions") or 0) if partitioning.get("partitioned") else 0
    kind = str(partitioning.get("kind") or "partitions")
    n_indexes = len(stats.get("indexes") or [])
    approx = "~" if estimated else ""
    mib = n_bytes / (1024 * 1024)
    text = (f"{approx}{n_rows:,} rows, {mib:.1f} MiB"
            + (f", {lane} lane" if lane else "")
            + f": ceil(rows/{ROWS_PER_CLONE:,})={by_rows}, ceil(bytes/64 MiB)={by_bytes}, "
            f"within [{MIN_CLONES}, {MAX_CLONES}] -> {by_size}")
    if partitions > 1:
        clones = max(MIN_CLONES, min(partitions, by_size))
        text += f"; {partitions} {kind}, one clone per partition up to that -> {clones}"
    else:
        clones = by_size
        text += "; source is not partitioned"
    if n_indexes:
        text += f"; {n_indexes} index(es) shape the scan, not the count"
    return FleetSize(clones=clones, derived_from=text, n_rows=n_rows, n_bytes=n_bytes,
                     partitions=partitions, lane=lane)
