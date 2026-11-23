"""Every declared upload format reaches the sampler with CSV's statistics.

The understanding agent (``profiling_agent.profile_full``) and the
feature-engineering context (``extract_feature_engineering_context``) read
``sample_statistics``: ``data_shape``, ``column_statistics`` (type, missing
ratio, cardinality, numeric summary), ``data_quality``, ``bytes_per_row`` and
``estimated_full_bytes``. The optimizer (``should_use_ray``,
``_compute_partitions``, the sizing tier and the adaptive sample sizes) reads
the row and byte estimate that precedes them.

Until 2026-10-17 that estimate was a CSV heuristic -- newlines in the first
64 KB -- applied to every format. A 120-row ORC file estimated as 23 rows, an
Avro file as 1, an XML document as 602, a JSON array as 25, and a two-part
Parquet directory as the first part's rows times two. The loaders read the
files correctly, so nothing raised; the tier and the sample sizes were simply
wrong, and the quick-profile path reported the wrong row count as the shape.

These tests load the checked-in fixtures (``tests/fixtures/formats``, one
table in every format, with nulls and a free-text column so there is
something to measure) through the real sampler and assert the numbers, not
the absence of an exception. The truth file is computed with pandas on the
in-memory frame, never from a file read back.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("daft")

from app.agents import sampling_agent_daft as sad  # noqa: E402
from app.agents import sampling_async as sa  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "formats"
TRUTH = json.loads((FIXTURES / "truth.json").read_text())

#: Declared upload extension -> fixture. ``xls``/``xlsx`` are absent on purpose:
#: /api/upload expands a workbook into one CSV per sheet before the sampler
#: sees it (server.py, Stage A), so the CSV row here is their contract; the
#: CLI test reads the workbook itself.
SINGLE_FILES = {
    "csv": "sales.csv", "tsv": "sales.tsv", "json": "sales.json", "jsonl": "sales.jsonl",
    "ndjson": "sales.ndjson", "xml": "sales.xml", "parquet": "sales.parquet",
    "orc": "sales.orc", "avro": "sales.avro",
}
#: Format -> (directory of parts, truth key holding rows per part).
DIRECTORIES = {
    "parquet": ("sales_parquet_parts", "parquet_part_rows"),
    "csv": ("sales_csv_parts", "csv_part_rows"),
}


def _needs(fmt: str) -> None:
    if fmt == "avro":
        pytest.importorskip("fastavro")
    if fmt == "orc":
        pytest.importorskip("pyarrow.orc")


# ---------------------------------------------------------------------------
# The estimate the optimizer keys on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", sorted(SINGLE_FILES))
def test_size_estimate_is_exact_for_a_small_file_of_every_format(fmt):
    _needs(fmt)
    path = FIXTURES / SINGLE_FILES[fmt]
    rows, size_mb = sad._estimate_dataset_size(str(path), fmt)
    assert rows == TRUTH["n_rows"], f"{fmt}: estimated {rows} rows, the file has {TRUTH['n_rows']}"
    assert size_mb == pytest.approx(path.stat().st_size / (1024 * 1024))

    described = sad.describe_source(str(path), fmt)
    assert described["format"] == fmt
    assert described["rows_exact"] is True, f"{fmt}: a whole small file must count exactly"
    assert described["n_files"] == 1
    assert described["total_bytes"] == path.stat().st_size
    assert described["partitions"] == [{"bytes": path.stat().st_size, "rows": TRUTH["n_rows"]}]


@pytest.mark.parametrize("fmt", ["parquet", "orc", "avro"])
def test_columnar_row_counts_come_from_the_footer_not_from_decoding(fmt, tmp_path, monkeypatch):
    """The exact count must not depend on text heuristics: zero out the head
    reader and the count is unchanged."""
    _needs(fmt)
    monkeypatch.setattr(sad, "_rows_from_head", lambda *a, **k: (0, False))
    described = sad.describe_source(str(FIXTURES / SINGLE_FILES[fmt]), fmt)
    assert described["rows"] == TRUTH["n_rows"] and described["rows_exact"] is True


@pytest.mark.parametrize("fmt", sorted(DIRECTORIES))
def test_a_directory_of_parts_is_sized_part_by_part(fmt):
    """Uneven parts: the old estimate read the first part and multiplied."""
    folder, key = DIRECTORIES[fmt]
    path = FIXTURES / folder
    described = sad.describe_source(str(path), fmt)
    assert [part["rows"] for part in described["partitions"]] == TRUTH[key]
    assert described["rows"] == TRUTH["n_rows"]
    assert described["n_files"] == len(TRUTH[key]), "a _SUCCESS marker is not a part"
    sizes = sorted(f.stat().st_size for f in path.iterdir() if f.suffix == f".{fmt}")
    assert sorted(part["bytes"] for part in described["partitions"]) == sizes
    assert described["total_bytes"] == sum(sizes)

    rows, size_mb = sad._estimate_dataset_size(str(path), fmt)
    assert rows == TRUTH["n_rows"]
    assert size_mb == pytest.approx(sum(sizes) / (1024 * 1024))


def test_text_estimates_scale_when_the_head_is_not_the_whole_file(tmp_path):
    """A JSON array and an XML document larger than the head window are
    estimated per record, not per line (a pretty-printed record spans many)."""
    import pandas as pd
    frame = pd.DataFrame({"k": range(6000), "v": ["x" * 20] * 6000})
    array = tmp_path / "big.json"
    array.write_text(json.dumps(frame.to_dict("records"), indent=2))
    xml = tmp_path / "big.xml"
    frame.to_xml(xml, index=False)
    for path, fmt in ((array, "json"), (xml, "xml")):
        assert path.stat().st_size > sad._HEAD_BYTES, "fixture must exceed the head window"
        described = sad.describe_source(str(path), fmt)
        assert described["rows_exact"] is False
        assert abs(described["rows"] - 6000) / 6000 < 0.05, (
            f"{fmt}: estimated {described['rows']} rows for 6000")


# ---------------------------------------------------------------------------
# The statistics the understanding and feature-engineering agents consume
# ---------------------------------------------------------------------------
def _assert_statistics_contract(stats: dict, *, label: str) -> None:
    shape = stats["data_shape"]
    assert shape["rows"] == TRUTH["n_rows"], f"{label}: data_shape.rows={shape['rows']}"
    assert shape["columns"] == len(TRUTH["columns"])

    columns = stats["column_statistics"]
    assert set(columns) == set(TRUTH["columns"]), f"{label}: columns {sorted(columns)}"
    for name, kind in TRUTH["kinds"].items():
        entry = columns[name]
        nulls = TRUTH["nulls"][name]
        assert entry["missing_ratio"] == pytest.approx(nulls / TRUTH["n_rows"], abs=1e-4), (
            f"{label}: {name} missing_ratio={entry['missing_ratio']}")
        if kind == "numeric":
            assert entry["type"] == "numeric", f"{label}: {name} typed {entry['type']}"
            truth = TRUTH["numeric"][name]
            assert entry["mean"] == pytest.approx(truth["mean"], rel=1e-4)
            assert entry["min"] == pytest.approx(truth["min"]) and entry["max"] == pytest.approx(truth["max"])
            assert entry["n_unique"] == TRUTH["n_unique"][name]
        else:
            # Dates written as text, categories and free text all land as
            # categorical here; the understanding agent decides text-vs-category
            # from cardinality, so the cardinality must be right. The sampler
            # counts "<missing>" as a value, hence the +1 where nulls exist.
            assert entry["type"] == "categorical", f"{label}: {name} typed {entry['type']}"
            assert entry["n_unique"] in {TRUTH["n_unique"][name], TRUTH["n_unique"][name] + (1 if nulls else 0)}, (
                f"{label}: {name} n_unique={entry['n_unique']}")
    quality = stats["data_quality"]
    assert 0.9 < quality["overall_completeness"] <= 1.0
    assert quality["constant_columns"] == []

    assert stats["bytes_per_row"] > 0 and stats["estimated_full_bytes"] > 0
    source = stats["source"]
    assert source["rows"] == TRUTH["n_rows"] and source["total_bytes"] > 0
    assert source["partitions"], f"{label}: no partition sizes for the optimizer"
    for part in source["partitions"]:
        assert set(part) == {"bytes", "rows"}, f"{label}: partition carries more than counts: {part}"


@pytest.mark.parametrize("fmt", sorted(SINGLE_FILES))
def test_full_profile_reports_the_same_statistics_for_every_format(fmt):
    """The path /api/upload takes (sample_with_profiling, Ray off)."""
    _needs(fmt)
    result = sad.sample_with_profiling(str(FIXTURES / SINGLE_FILES[fmt]), fmt, use_ray=False)
    assert not result.get("error"), f"{fmt}: {result.get('error')}"
    assert result["total_rows_exact"] == TRUTH["n_rows"]
    assert set(result["schema"]) == set(TRUTH["columns"])
    _assert_statistics_contract(result["sample_statistics"], label=fmt)


@pytest.mark.parametrize("fmt", sorted(SINGLE_FILES))
def test_quick_profile_reports_the_same_statistics_for_every_format(fmt):
    """The path analysis takes (sample_quick), with the full algorithm for a
    small file; the source block must be present on this path too."""
    _needs(fmt)
    result = sa.sample_quick(str(FIXTURES / SINGLE_FILES[fmt]), fmt, use_ray=False)
    assert not result.get("error"), f"{fmt}: {result.get('error')}"
    assert result["estimated_rows"] == TRUTH["n_rows"]
    _assert_statistics_contract(result["sample_statistics"], label=fmt)


@pytest.mark.parametrize("fmt", sorted(DIRECTORIES))
def test_a_directory_of_parts_profiles_as_one_table(fmt):
    folder, key = DIRECTORIES[fmt]
    result = sad.sample_with_profiling(str(FIXTURES / folder), fmt, use_ray=False)
    assert not result.get("error"), f"{fmt} directory: {result.get('error')}"
    _assert_statistics_contract(result["sample_statistics"], label=f"{fmt} directory")
    assert [p["rows"] for p in result["sample_statistics"]["source"]["partitions"]] == TRUTH[key]


@pytest.mark.parametrize("fmt", ["orc", "avro", "xml", "json"])
def test_a_directory_of_pandas_backed_parts_loads(fmt, tmp_path):
    """Daft has no reader for these; the parts are stacked by hand."""
    _needs(fmt)
    import shutil
    folder = tmp_path / f"{fmt}_parts"
    folder.mkdir()
    shutil.copy(FIXTURES / SINGLE_FILES[fmt], folder / f"part-0.{fmt}")
    shutil.copy(FIXTURES / SINGLE_FILES[fmt], folder / f"part-1.{fmt}")
    (folder / "_SUCCESS").write_text("")
    frame, temp_files = sad._load_file_with_daft(str(folder), fmt)
    assert frame.count_rows() == 2 * TRUTH["n_rows"]
    assert set(frame.column_names) == set(TRUTH["columns"])
    assert sad.describe_source(str(folder), fmt)["partitions"] == [
        {"bytes": (folder / f"part-0.{fmt}").stat().st_size, "rows": TRUTH["n_rows"]}] * 2


def test_cloud_paths_do_not_touch_the_filesystem():
    assert sad.describe_source("gs://bucket/prefix/", "parquet") == {}
    assert sad.describe_source("s3://bucket/key.csv", "csv") == {}
