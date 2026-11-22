"""The generated Ray script counts every row of the file exactly once, and
reports honestly how many of them the analysis itself used.

The script from ``_build_ray_wrapper`` is executed for real against local CSV
files, with a tiny in-process stand-in for ``ray`` (remote tasks run inline).
Partitions are forced down to 1 MB so a few-MB file is split several times.

Since the merge with the Ray-honesty hotfix (ff7cb585) the workers no longer
parse rows: each one counts the line breaks in its own byte range, the ranges
tile the file exactly, and the driver prints ``ROWS IN FILE = N``. The
analysis (``main(df)``) runs on the first ``SAMPLE_ROWS`` rows and the script
says so (``rows_used`` / ``rows_total`` / ``sample``).

What these tests protect against:
  * partition offsets guessed from a fixed bytes/row -> rows skipped or
    counted twice, so the row total was wrong (30 Oct);
  * a byte range that overlaps or leaves a gap between partitions;
  * a last row with no trailing newline being dropped from the total;
  * each task holding its partition with no ``memory=`` reservation, so
    parallel tasks overran the 3 Gi Ray worker (AVALOKA_RAY_PARTITION_MEMORY_MB);
  * the analysis claiming to cover every row when it saw only a sample (MAJ-030).
"""
from __future__ import annotations

import random
import re
import sys
import types

import pytest

from app.agents.execution_agent import RAY_ANALYSIS_SAMPLE_ROWS, _build_ray_wrapper


# ---------------------------------------------------------------- helpers ---

def _fake_ray_modules() -> dict:
    """Minimal stand-in for the parts of ``ray`` the wrapper script touches."""
    ray = types.ModuleType("ray")
    util = types.ModuleType("ray.util")
    util.get_node_ip_address = lambda: "127.0.0.1"
    ray.util = util
    ray.init = lambda *a, **k: None
    ray.nodes = lambda: [
        {"Alive": True, "Resources": {"CPU": 4.0}, "NodeManagerAddress": "127.0.0.1"}
    ]
    ray.cluster_resources = lambda: {"CPU": 4.0, "memory": 3 * 1024**3}

    class _Remote:
        def __init__(self, fn):
            self._fn = fn

        def remote(self, *args, **kwargs):
            return self._fn(*args, **kwargs)  # run inline; the "ref" is the value

    def remote(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return _Remote(args[0])
        return lambda fn: _Remote(fn)

    ray.remote = remote
    ray.get = lambda refs: list(refs) if isinstance(refs, list) else refs
    return {"ray": ray, "ray.util": util}


def _run_wrapper(csv_path, monkeypatch, capsys, partition_mb: int = 1) -> tuple[str, dict]:
    """Run the wrapper script; return (stdout, the script's globals)."""
    for name, mod in _fake_ray_modules().items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("AVALOKA_RAY_PARTITION_MB", str(partition_mb))
    monkeypatch.delenv("GCP_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    script = _build_ray_wrapper(str(csv_path), "file:///unused/artifact", "file:///unused/metrics")
    scope: dict = {"__name__": "__main__"}
    exec(compile(script, "sample_code.py", "exec"), scope)
    return capsys.readouterr().out, scope


def _total_rows(out: str) -> int:
    m = re.search(r"ROWS IN FILE = (\d+)", out)
    assert m, f"wrapper did not print a row total:\n{out[-2000:]}"
    return int(m.group(1))


def _partitions(out: str) -> int:
    m = re.search(r"Total partitions to submit: (\d+)", out)
    assert m, f"wrapper did not print its partition plan:\n{out[-2000:]}"
    return int(m.group(1))


def _ranges(out: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in re.findall(r"bytes (\d+)-(\d+)", out)]


# ------------------------------------------------------------------ tests ---

def test_script_compiles_and_reserves_memory_per_task():
    script = _build_ray_wrapper("gs://bucket/file.csv", "gs://b/a", "gs://b/m")
    compile(script, "sample_code.py", "exec")
    assert "memory=PARTITION_MEMORY" in script, "tasks must reserve memory so Ray limits concurrency"
    assert "AVALOKA_RAY_PARTITION_MEMORY_MB" in script
    assert "AVALOKA_RAY_PARTITION_MB" in script
    assert "BYTES_PER_ROW" not in script, "row offsets must not be guessed from a fixed row size"
    assert "header_bytes + raw" not in script, "the raw partition must not be copied into a second buffer"


def test_analysis_scope_is_labelled_not_hidden():
    """MAJ-030: the script reports the rows main(df) saw, not just a file total."""
    script = _build_ray_wrapper("gs://bucket/file.csv", "gs://b/a", "gs://b/m")
    assert f"SAMPLE_ROWS = {RAY_ANALYSIS_SAMPLE_ROWS}" in script
    assert "nrows=50000" not in script, "the sample size must come from SAMPLE_ROWS, not a literal"
    for name in ("ROWS_USED", "ROWS_TOTAL", "IS_SAMPLE"):
        assert name in script


def test_byte_ranges_tile_the_file(tmp_path, monkeypatch, capsys):
    """Every byte belongs to exactly one partition: no gaps, no overlaps."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "tile.csv"
    with open(path, "w", newline="") as f:
        f.write("a,b\n")
        for i in range(70_000):
            f.write(f"{i},{'q' * (i % 37)}\n")

    out, _ = _run_wrapper(path, monkeypatch, capsys)

    ranges = sorted(_ranges(out))
    assert len(ranges) == _partitions(out) > 1
    assert ranges[0][0] == 0
    assert ranges[-1][1] == path.stat().st_size
    for (_, end), (start, _) in zip(ranges, ranges[1:]):
        assert start == end, f"gap or overlap at byte {end}/{start}"


def test_rows_on_partition_boundaries_are_counted_once(tmp_path, monkeypatch, capsys):
    """Every 1 MiB boundary falls exactly at the start of a row (64-byte rows)."""
    monkeypatch.chdir(tmp_path)
    header = "id,payload,k\n"
    n_rows = 3 * 16384 + 123  # 3 MiB of 64-byte rows, plus a partial 4th MiB
    with open("aligned.csv", "w", newline="") as f:
        f.write(header)
        for i in range(n_rows):
            row = f"{i:010d},{'x' * 50},{i % 7}\n"
            assert len(row) == 64
            f.write(row)

    out, _ = _run_wrapper(tmp_path / "aligned.csv", monkeypatch, capsys)

    assert _partitions(out) == 4
    assert _total_rows(out) == n_rows


def test_variable_length_rows_are_counted_once(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rnd = random.Random(42)
    n_rows = 60_000
    with open("ragged.csv", "w", newline="") as f:
        f.write("event_time,event_type,brand,price\n")
        for i in range(n_rows):
            brand = "b" * rnd.randint(1, 90)
            f.write(f"2019-11-01 00:00:{i % 60:02d} UTC,view,{brand},{rnd.random() * 1000:.2f}\n")

    out, scope = _run_wrapper(tmp_path / "ragged.csv", monkeypatch, capsys)

    assert _partitions(out) > 1
    assert _total_rows(out) == n_rows
    # More rows than SAMPLE_ROWS: the analysis saw a labelled sample.
    assert scope["ROWS_USED"] == RAY_ANALYSIS_SAMPLE_ROWS
    assert scope["IS_SAMPLE"] is True
    assert "SHAPE (sample):" in out


def test_file_smaller_than_one_partition(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with open("small.csv", "w", newline="") as f:
        f.write("a,b\n")
        for i in range(500):
            f.write(f"{i},{i * 2}\n")

    out, scope = _run_wrapper(tmp_path / "small.csv", monkeypatch, capsys)

    assert _partitions(out) == 1
    assert _total_rows(out) == 500
    # Whole file fits in the sample: the analysis covered every row and says so.
    assert scope["ROWS_USED"] == 500
    assert scope["IS_SAMPLE"] is False
    assert "SHAPE (all rows):" in out


def test_last_row_without_trailing_newline(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rows = [f"{i},{'y' * 40}" for i in range(40_000)]
    (tmp_path / "no_eol.csv").write_text("id,payload\n" + "\n".join(rows))  # no final "\n"

    out, _ = _run_wrapper(tmp_path / "no_eol.csv", monkeypatch, capsys)

    assert _total_rows(out) == 40_000


@pytest.mark.xfail(
    reason="Known limit of counting line breaks: a quoted field containing a newline "
           "counts as an extra row. Needs a quote-aware reader (Daft/pyarrow).",
    strict=False,
)
def test_quoted_newlines_survive_partitioning(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with open("quoted.csv", "w", newline="") as f:
        f.write("id,note\n")
        for i in range(30_000):
            f.write(f'{i},"line one\nline two {"z" * 30}"\n')

    out, _ = _run_wrapper(tmp_path / "quoted.csv", monkeypatch, capsys)

    assert _total_rows(out) == 30_000