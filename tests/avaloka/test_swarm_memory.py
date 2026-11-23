"""The main Avaloka does not hold the dataset while her clones work.

Before avaloka.io.sharding, the coordinator loaded the whole frame, kept it
and a second copy as the list of shard frames alive for the whole run, and
every clone loaded its shard again. On a 211 MB CSV (3,000,000 rows of
numbers, categories, dates and text) the CLI's ``--swarm 2`` peaked at
2,270 MB resident before (2152d59b) and 1,743 MB after, and eight clones
one at a time at 1,534 MB before and 510 MB after, measured with
``/usr/bin/time -v`` in the arm64 CI container; the full table is in the PR.

The gate cannot afford 200 MB, so this runs a scale model: the same shape
at 240,000 rows (about 17 MB), with the stream's chunk scaled by the same
ratio, and bounds the peak growth per MB of file. Eight clones run one at a
time, so what a clone holds is an eighth of the data and what the
coordinator holds is the difference under test: the old coordinator grew by
7.0 MB per MB of file (four runs, 7.00-7.04), streamed shards by 3.47-3.72.
GROWTH_PER_MB_BOUND sits between them.

Each clone's mission is replaced by its Data Scout's load (the mission is
the same code before and after, and the rest of it only slows the gate).
The run happens in a child process, so no earlier test in the gate's
process counts towards it; the child runs once on the file's first rows to
warm its imports, resets its high-water mark (``/proc/self/clear_refs``),
takes its baseline and reads its peak as ``VmHWM``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

#: Rows of the generated file: about 17 MB of mixed numbers and text.
N_ROWS = 240_000

#: The 200 MB file had 3,000,000 rows; the stream's chunk is scaled to match.
CHUNK_ROWS = 100_000 * N_ROWS // 3_000_000

#: The fleet: eight clones, one at a time.
CLONES, PARALLEL = 8, 1

#: Peak RSS growth allowed per MB of CSV. Calibrated in the arm64 CI
#: container on this fixture: 3.47-3.72 streamed, 7.0 with the frame held
#: (see the module docstring). A coordinator that loads the frame whole
#: again, even briefly, fails.
GROWTH_PER_MB_BOUND = 5.0

_CHILD = textwrap.dedent("""
    import json, os, sys
    import avaloka.coordinator as coordinator
    from avaloka.coordinator import MainAvaloka
    from avaloka.mission.context import MissionKind
    from avaloka.swarm_limits import UNLIMITED

    def rss_mb():
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2**20

    def high_water_mb():
        with open("/proc/self/status") as fh:
            kb = next(line for line in fh if line.startswith("VmHWM:")).split()[1]
        return int(kb) / 1024

    try:
        # The fixture is a scale model of the 200 MB run, so the streamed
        # chunk is the same share of it as CSV_CHUNK_ROWS is of that file.
        from avaloka.io import sharding
        sharding.CSV_CHUNK_ROWS = int(sys.argv[6])
    except ImportError:  # a coordinator from before sharding: nothing to scale
        pass

    if sys.argv[5] == "stub":
        # The clone's mission is the same code before and after sharding;
        # past its Data Scout's load it only adds the same work to both sides
        # and slows the gate. The stand-in loads the shard as the Scout does
        # and leaves the handle where the Scout leaves it.
        from avaloka.io import load_dataset

        class _Result:
            summary = {"verdict": "pass", "max_safe_deployment_level": 1, "model": None}

        def _scout_only(ctx, on_progress=None):
            ctx.blackboard["handle"] = load_dataset(ctx.data_source)
            return _Result()

        coordinator.run_analyze = _scout_only

    def run(source, out):
        return MainAvaloka(swarm_limits=UNLIMITED).coordinate(
            source, goal="what drives tickets", output_dir=out,
            workers=int(sys.argv[3]), max_parallel=int(sys.argv[4]), kind=MissionKind.ANALYZE)

    # A first run on the file's first rows imports everything the path
    # imports lazily (pyarrow, the schema inference), so the baseline after
    # it leaves only what the data costs. Its own peak is then forgotten:
    # writing 5 to clear_refs resets the high-water mark (ru_maxrss cannot
    # be reset, and the warm-up alone once peaked above the run under test).
    from pathlib import Path
    warm = Path(sys.argv[2] + "-warm.csv")
    with open(sys.argv[1]) as src, warm.open("w") as dst:
        dst.writelines(line for _, line in zip(range(2001), src))
    run(str(warm), sys.argv[2] + "-warm")
    with open("/proc/self/clear_refs", "w") as fh:
        fh.write("5")
    baseline = rss_mb()
    cr = run(sys.argv[1], sys.argv[2])
    peak = high_water_mb()
    print(json.dumps({"baseline": baseline, "peak": peak, "rows": cr.n_rows,
                      "shards": [s.n_rows for s in cr.shards]}))
""")


def _write_csv(path: Path, n_rows: int = N_ROWS) -> Path:
    """Numbers, categories, dates, free text and a few gaps: the shape the
    200 MB measurement used, at a gate-sized row count."""
    rng = np.random.default_rng(7)
    words = np.array(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"])
    monthly = rng.normal(70, 20, n_rows).round(4)
    monthly[rng.random(n_rows) < 0.02] = np.nan
    pd.DataFrame({
        "id": np.arange(n_rows),
        "tenure": rng.integers(1, 72, n_rows),
        "monthly": monthly,
        "tickets": rng.poisson(1.5, n_rows),
        "balance": rng.normal(1000, 300, n_rows).round(2),
        "contract": rng.choice(["month-to-month", "one-year", "two-year"], n_rows),
        "region": rng.choice(["north", "south", "east", "west"], n_rows),
        "signup": (pd.Timestamp("2020-01-01")
                   + pd.to_timedelta(rng.integers(0, 1500, n_rows), unit="D")).strftime("%Y-%m-%d"),
        "note": [" ".join(t) for t in rng.choice(words, (n_rows, 3))],
    }).to_csv(path, index=False)
    return path


@pytest.mark.skipif(not Path("/proc/self/statm").exists(), reason="RSS is read from /proc (Linux)")
def test_the_fleet_grows_rss_by_a_shard_not_by_the_dataset(tmp_path):
    source = _write_csv(tmp_path / "mixed.csv")
    size_mb = source.stat().st_size / 2**20
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(Path(__file__).resolve().parents[2]), os.environ.get("PYTHONPATH", "")])}
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(source), str(tmp_path / "fleet"), str(CLONES),
         str(PARALLEL), "stub", str(CHUNK_ROWS)],
        capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    run = json.loads(proc.stdout.strip().splitlines()[-1])
    growth = run["peak"] - run["baseline"]
    bound = GROWTH_PER_MB_BOUND * size_mb
    print(f"\nfile {size_mb:.1f} MB; RSS baseline {run['baseline']:.0f} MB, peak {run['peak']:.0f} MB, "
          f"growth {growth:.0f} MB ({growth / size_mb:.1f}/MB); bound {bound:.0f} MB")
    assert run["rows"] == N_ROWS and sum(run["shards"]) == N_ROWS and len(run["shards"]) == CLONES
    assert growth < bound, (f"RSS grew by {growth:.0f} MB on a {size_mb:.1f} MB file "
                            f"(bound {GROWTH_PER_MB_BOUND}/MB = {bound:.0f} MB)")
