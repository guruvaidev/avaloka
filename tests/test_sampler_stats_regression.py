"""
Regression tests for the portfolio sampler (sampling_agent_daft).

Covers:
- portfolio stats were computed over pyarrow Scalars, so missing detection and
  numeric parsing never worked (missing_ratio always 0.0, numeric columns
  mis-profiled as categorical).
- stratified sampling crashed whenever the stratify column had a null
  (count() ignores nulls -> 0/0 ZeroDivisionError -> float .limit() ->
  APITypeError).
- quantile sampling did no real stratification, weighted frequencies, honored
  sample_size, and empty datasets.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import daft

from app.agents.sampling_agent_daft import (
    aggregate_statistics_from_portfolio,
    _compute_categorical_stats_hybrid,
    _compute_numeric_stats_hybrid,
    stratified_sample_daft,
    quantile_stratified_sample,
    sample_with_profiling,
)


# ── stats over pyarrow columns ────────────────────────────────────────────────

def test_base_fallback_numeric_column_with_null():
    """Numeric column with a null is profiled as numeric, missing_ratio correct."""
    df = daft.from_pydict({"a": [1.0, 2.0, None, 4.0]})
    portfolio = {"random_baseline": df.with_column("_weight", daft.lit(1.0))}

    stats = aggregate_statistics_from_portfolio(portfolio, df)
    col = stats["column_statistics"]["a"]

    assert col["type"] == "numeric"
    assert col["missing_count"] == 1
    assert col["missing_ratio"] == pytest.approx(0.25)
    assert col["mean"] == pytest.approx((1.0 + 2.0 + 4.0) / 3, abs=1e-3)
    assert col["min"] == 1.0 and col["max"] == 4.0
    assert stats["data_quality"]["overall_completeness"] == pytest.approx(0.75)


def test_numeric_hybrid_detects_nulls_and_parses_floats():
    base = daft.from_pydict({"n": [1.0, 2.0, None, 4.0, 5.0, 100.0]})
    quantile = daft.from_pydict({"n": [1.0, 4.0, 100.0]})

    col = _compute_numeric_stats_hybrid(quantile, base.collect().to_arrow(), "n")

    assert col["type"] == "numeric"
    assert col["missing_count"] == 1
    assert col["missing_ratio"] == pytest.approx(1 / 6, abs=1e-3)
    assert col["mean"] == pytest.approx((1 + 2 + 4 + 5 + 100) / 5, abs=1e-3)
    assert col["min"] == 1.0 and col["max"] == 100.0


def test_categorical_hybrid_detects_missing_tokens():
    base = daft.from_pydict({"c": ["x", "y", None, "x", "z", ""]})
    strat = daft.from_pydict({"c": ["x", "y", "z", "x"]})

    col = _compute_categorical_stats_hybrid(strat, base.collect().to_arrow(), "c")

    assert col["type"] == "categorical"
    assert col["n_unique"] == 3          # x, y, z from the stratified sample
    assert col["missing_count"] == 2     # None and "" both count as missing
    assert col["missing_ratio"] == pytest.approx(2 / 6, abs=1e-3)


# ── stratified sampling on a column with nulls ─────────────────────────────────

def test_stratified_sample_with_null_group_no_crash():
    """Null in the stratify column must not crash and the null row is kept."""
    df = daft.from_pydict({"cat": ["a", "a", "b", None], "val": [1, 2, 3, 4]})

    # k_cap arrives as a float from the caller (K_CAP = max_rows / n_candidates)
    out = stratified_sample_daft(df, "cat", k_cap=2.5, seed=42).collect().to_pydict()

    assert "_weight" in out
    assert None in out["cat"]            # null group represented, not dropped
    assert len(out["cat"]) == 4


def test_stratified_sample_all_null_column_falls_back():
    df = daft.from_pydict({"cat": [None, None, None], "v": [1, 2, 3]})

    out = stratified_sample_daft(df, "cat", k_cap=5.0).collect().to_pydict()

    assert len(out["cat"]) == 3          # fallback returns rows, no APITypeError


def test_quantile_sample_accepts_float_size():
    df = daft.from_pydict({"n": [float(i) for i in range(20)]})

    out = quantile_stratified_sample(df, "n", sample_size=8.0, total_rows=20)

    assert out.count_rows() > 0          # float .limit() no longer raises


def test_sample_with_profiling_null_in_stratify_column(tmp_path):
    """End-to-end: a null in the chosen stratify column no longer errors."""
    csv = tmp_path / "blo167.csv"
    csv.write_text("region,amount\nnorth,10\nnorth,20\nsouth,30\n,40\n")

    res = sample_with_profiling(path=str(csv), source_type="csv", use_ray=False)

    assert not res.get("error")
    assert res.get("schema")
    region = res["profiling_result"]["column_statistics"]["region"]
    assert region["missing_ratio"] == pytest.approx(0.25)


# ── quantile_stratified_sample does real stratification ───────────────────────

def test_quantile_sample_stratifies_across_value_ranges():
    """Buckets rows by the column's quantiles (not N random samples of the whole)."""
    df = daft.from_pydict({"x": [float(i) for i in range(1000)]})

    out = quantile_stratified_sample(df, "x", sample_size=200.0, num_quantiles=4, total_rows=1000).collect().to_pydict()

    assert "_weight" in out
    xs = sorted(out["x"])
    ranges = [(0, 250), (250, 500), (500, 750), (750, 1000)]
    per_bucket = [sum(1 for v in xs if lo <= v < hi) for lo, hi in ranges]
    assert all(c > 0 for c in per_bucket), per_bucket   # every quartile represented
    assert xs[0] < 250 and xs[-1] >= 750                # tails covered


def test_quantile_sample_fallback_carries_weight():
    """The fallback path must still attach _weight or the fused concat breaks."""
    df = daft.from_pydict({"x": [float(i) for i in range(100)]})

    out = quantile_stratified_sample(df, "no_such_col", sample_size=50.0, total_rows=100).collect().to_pydict()

    assert "_weight" in out


def test_quantile_sample_fuses_with_other_samples():
    """All portfolio samples share a schema so .concat() (the fused plan) works."""
    df = daft.from_pydict({"x": [float(i) for i in range(400)], "g": ["a", "b", "c", "d"] * 100})

    q = quantile_stratified_sample(df, "x", sample_size=100.0, total_rows=400)
    s = stratified_sample_daft(df, "g", k_cap=100)
    rb = df.with_column("_weight", daft.lit(1.0))

    combined = q.concat(s).concat(rb)
    assert "_weight" in combined.column_names
    assert combined.count_rows() > 0


# ── categorical frequencies apply _weight (undo K-cap bias) ───────────────────

def test_categorical_frequencies_corrected_by_weight():
    """A K-capped 99/1 column reads ~50/50 raw; _weight must recover 99/1."""
    df = daft.from_pydict({"cat": ["A"] * 9900 + ["B"] * 100})
    base = df.collect().to_arrow()

    strat = stratified_sample_daft(df, "cat", k_cap=200, seed=42)
    res = _compute_categorical_stats_hybrid(strat, base, "cat")

    tv = {t["value"]: t for t in res["top_values"]}
    raw_total = sum(t["sample_count"] for t in res["top_values"])
    raw_b_freq = tv["B"]["sample_count"] / raw_total

    assert raw_b_freq > 0.2                                     # raw sample is biased (true 0.01)
    assert tv["A"]["frequency"] == pytest.approx(0.99, abs=0.02)
    assert tv["B"]["frequency"] == pytest.approx(0.01, abs=0.02)


# ── caller-supplied sample_size is honored (not adaptive) ─────────────────────

def _write_rows_csv(path, n):
    path.write_text("id,val\n" + "\n".join(f"{i},{i * 2}" for i in range(n)) + "\n")


def test_int_sample_size_is_honored(tmp_path):
    """An int sample_size is an explicit count; the adaptive display size (1000
    here) must not override the caller's request."""
    csv = tmp_path / "rows.csv"
    _write_rows_csv(csv, 2000)   # SMALL tier -> adaptive display_sample_size = 1000

    res = sample_with_profiling(path=str(csv), source_type="csv", sample_size=10, use_ray=False)

    assert not res.get("error")
    assert len(res["rows"]) == 10                                   # caller's 10, not 1000
    assert res["profiling_result"]["data_shape"]["rows"] == 2000    # full-scan stats unaffected


def test_float_sample_size_is_a_fraction(tmp_path):
    csv = tmp_path / "rows.csv"
    _write_rows_csv(csv, 2000)

    res = sample_with_profiling(path=str(csv), source_type="csv", sample_size=0.01, use_ray=False)

    assert not res.get("error")
    assert len(res["rows"]) == 20                                   # 1% of 2000


# ── empty dataset returns an empty payload (not division-by-zero) ─────────────

def test_empty_dataset_returns_empty_payload(tmp_path):
    csv = tmp_path / "empty.csv"
    csv.write_text("id,name,value\n")   # header only, 0 data rows

    res = sample_with_profiling(path=str(csv), source_type="csv", use_ray=False)

    assert not res.get("error")                                     # was {'error': 'division by zero'}
    assert res["rows"] == []
    assert list(res["schema"]) == ["id", "name", "value"]           # schema preserved
    assert res["profiling_result"]["data_shape"]["rows"] == 0


def test_all_null_column_is_preserved_and_flagged(tmp_path):
    csv = tmp_path / "all-null.csv"
    csv.write_text("value,empty\n1,\n2,\n3,\n")

    res = sample_with_profiling(path=str(csv), source_type="csv", use_ray=False)

    assert "empty" in res["schema"]
    quality = res["profiling_result"]["data_quality"]
    assert "empty" in quality["columns_with_high_missing"]
    assert res["profiling_result"]["column_statistics"]["empty"]["missing_ratio"] == 1.0
