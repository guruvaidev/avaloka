"""PR-11's gate (design §8): a 5,000,000-row file ends ``qualified`` with the
``sampling`` caveat within the default budget — never a timeout traceback,
never a process the memory cap could not hold.

Skipped unless ``AVALOKA_HEAVY_TESTS=1``: the file is generated into
pytest's temporary directory (never the repository) in chunks, so the
generator itself stays under the memory rule, and the run takes a minute or
two. Run it alone so the peak RSS it reports is this test's, not the rest of
the suite's::

    AVALOKA_HEAVY_TESTS=1 pytest tests/recipes/test_budget_heavy.py -s
"""

from __future__ import annotations

import os
import resource
import sys
import time

import numpy as np
import pytest

import avaloka.recipes.library  # noqa: F401  (registers the recipes)
from avaloka.recipes import Intent, Verdict, get
from avaloka.recipes.budget import DEFAULT
from avaloka.recipes.run import load_and_run

pytestmark = [pytest.mark.heavy,
              pytest.mark.skipif(os.environ.get("AVALOKA_HEAVY_TESTS") != "1",
                                 reason="set AVALOKA_HEAVY_TESTS=1 to generate a 5M-row file and run on it")]

ROWS = 5_000_000
CHUNK = 250_000
RSS_LIMIT = 1.5 * 1024 ** 3
WALL_LIMIT = 60.0


def peak_rss_bytes() -> int:
    """The process's peak resident set so far (``ru_maxrss`` is bytes on
    macOS, kilobytes on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def write_big_parquet(path, rows: int = ROWS, chunk: int = CHUNK, seed: int = 11) -> None:
    """``rows`` rows of region, date, revenue, units, written ``chunk`` rows at
    a time through one ParquetWriter; each chunk is seeded on its own so the
    file is the same on every machine."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    days = np.array([f"2022-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 29)] +
                    [f"2023-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 29)])
    regions = np.array(["north", "south", "east", "west"])
    schema = pa.schema([("region", pa.string()), ("date", pa.string()), ("revenue", pa.float64()), ("units", pa.int64())])
    with pq.ParquetWriter(str(path), schema) as writer:
        for i in range(rows // chunk):
            rng = np.random.default_rng(seed + i)
            table = pa.table({"region": regions[rng.integers(0, len(regions), chunk)],
                              "date": days[rng.integers(0, len(days), chunk)],
                              "revenue": np.round(rng.lognormal(4, 0.5, chunk), 2),
                              "units": rng.integers(1, 50, chunk)}, schema=schema)
            writer.write_table(table)


@pytest.fixture(scope="module")
def big_parquet(tmp_path_factory):
    path = tmp_path_factory.mktemp("heavy") / "five_million.parquet"
    started = time.monotonic()
    write_big_parquet(path)
    print(f"\n[heavy] wrote {ROWS:,} rows, {path.stat().st_size / 1024 ** 2:.1f} MB in {time.monotonic() - started:.1f}s; "
          f"peak RSS after generation {peak_rss_bytes() / 1024 ** 2:.0f} MB")
    return path


@pytest.mark.parametrize("recipe_id, params", [
    ("describe", {"measure": "revenue"}),
    ("trend", {"measure": "revenue", "time": "date", "grain": "month"}),
])
def test_a_five_million_row_file_ends_qualified_sampling_within_the_budget(big_parquet, recipe_id, params,
                                                                           record_property):
    started = time.monotonic()
    result = load_and_run(get(recipe_id), Intent("q", recipe_id, params, 1.0, "fixture"), big_parquet, budget=DEFAULT)
    elapsed = time.monotonic() - started
    peak = peak_rss_bytes()
    print(f"\n[heavy] {recipe_id}: verdict={result.verdict.value} reason={result.reason} elapsed={elapsed:.1f}s "
          f"peak RSS={peak / 1024 ** 2:.0f} MB budget={result.budget_report}")
    record_property(f"{recipe_id}_elapsed_s", round(elapsed, 2))
    record_property(f"{recipe_id}_peak_rss_mb", round(peak / 1024 ** 2))

    assert result.verdict is Verdict.QUALIFIED, (result.verdict, result.reason, result.refusal)
    assert result.reason is None and result.refusal is None
    assert "sampling" in {c["kind"] for c in result.report["caveats"]}
    assert result.report["method"]["sample"] == {"rows_analyzed": 100_000, "total_rows": ROWS, "is_sampled": True}
    spent = result.budget_report
    assert spent is not None and spent.sampled and not spent.timed_out
    assert (spent.total_rows, spent.rows_analyzed) == (ROWS, 100_000)
    assert spent.total_bytes == big_parquet.stat().st_size
    assert spent.elapsed_s < WALL_LIMIT and elapsed < WALL_LIMIT
    assert peak < RSS_LIMIT, f"peak RSS {peak / 1024 ** 2:.0f} MB"
