"""Byte-aware analysis routing limits and flags.

Single source of truth for the byte-based routing threshold and its feature
flags, shared by the API layer (app/api/server.py) and the samplers
(app/agents/sampling_async.py) — the samplers cannot import from server.py
without a circular import.

ANALYSIS_MAX_INMEMORY_BYTES caps the estimated UNCOMPRESSED in-memory
working-set size (total_rows * measured bytes_per_row) of a frame we will
analyze synchronously. It is deliberately NOT comparable to the 1 GB
LARGE_DATASET_THRESHOLD_BYTES / _RAY_THRESHOLD_BYTES, which measure
on-disk/compressed SOURCE size — for Parquet those differ by ~5-10x. It is
also unrelated to the samplers' 5 MB *sample payload* budget: that bounds what
we store/preview, this bounds what we compute on. Keep them unwired.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Default derivation (UNVALIDATED HYPOTHESIS — tune from shadow-mode data
# before trusting it): pandas typically needs 2-3x the frame size in transient
# headroom during operations, so the frame cap is set to ANALYSIS_MEM_FRACTION
# (default 0.40) of the container/host memory limit. On a ~640 MB-available
# pod that computes to ~256 MB. Override per environment with
# ANALYSIS_MAX_INMEMORY_BYTES.
ANALYSIS_MEM_FRACTION_DEFAULT = 0.40


def container_memory_limit_bytes() -> int:
    """Container (cgroup) memory limit, falling back to host total memory."""
    for cg_path in ("/sys/fs/cgroup/memory.max",                    # cgroup v2
                    "/sys/fs/cgroup/memory/memory.limit_in_bytes"): # cgroup v1
        try:
            raw = Path(cg_path).read_text().strip()
            if raw and raw != "max":
                val = int(raw)
                if 0 < val < (1 << 48):  # v1 reports ~2^63 when unlimited
                    return val
        except Exception:
            pass
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        return 2 * 1024 ** 3  # conservative: assume a 2 GB box


def analysis_max_inmemory_bytes() -> int:
    """The synchronous-analysis cap in estimated in-memory bytes (see module docstring)."""
    env = os.getenv("ANALYSIS_MAX_INMEMORY_BYTES")
    if env:
        try:
            return int(env)
        except ValueError:
            logger.warning("Invalid ANALYSIS_MAX_INMEMORY_BYTES=%r; using derived default", env)
    try:
        frac = float(os.getenv("ANALYSIS_MEM_FRACTION", str(ANALYSIS_MEM_FRACTION_DEFAULT)))
    except ValueError:
        frac = ANALYSIS_MEM_FRACTION_DEFAULT
    return int(container_memory_limit_bytes() * frac)


def byte_routing_enabled() -> bool:
    """Master flag for byte-aware routing behaviour (default OFF; old path intact)."""
    return os.getenv("AVALOKA_BYTE_ROUTING", "0").strip().lower() in {"1", "true", "yes"}


def routing_shadow_enabled() -> bool:
    """Shadow-mode observability flag (log-only; never changes behaviour)."""
    return os.getenv("AVALOKA_ROUTING_SHADOW", "0").strip().lower() in {"1", "true", "yes"}
