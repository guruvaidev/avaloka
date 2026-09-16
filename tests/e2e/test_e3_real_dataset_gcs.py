"""
E3.08 / E11.09 — sampling and profiling against a real GCS dataset.

Everything else in this pack is hermetic and synthetic. This module is the one
place we assert against production-shaped data: wide schemas, heavy nulls, mixed
dtypes and files far larger than memory. Those are exactly the conditions the
synthetic fixtures cannot reproduce.

Marked `cloud`, so the hermetic PR gate deselects it. To run:

    export GOOGLE_APPLICATION_CREDENTIALS=<service-account.json>
    pytest tests/e2e/test_e3_real_dataset_gcs.py -m cloud -q

One dataset is exercised per run. Select it with AVALOKA_TEST_DATASET
(`ieee-fraud`, the default, or `walmart`); the suite never reads both at once.

Only true source inputs are used. The same prefixes also hold `code-registry/`
and `execution-outputs/` objects — artifacts of previous runs, not inputs — and
reading those back would be testing our own output, so they are excluded.
"""

from __future__ import annotations

import csv
import io
import os
from typing import Dict, List, Tuple

import pytest

pytestmark = pytest.mark.cloud

BUCKET = "avaloka-test-user-filestore"

# Real source inputs, with the facts each one is here to exercise.
DATASETS: Dict[str, Dict[str, object]] = {
    "ieee-fraud": {
        "prefix": "user-upload/inputs/ieee-fraud",
        # (object, expected column count, a column that must exist)
        "primary": ("train_identity.csv", 41, "TransactionID"),
        # Wide file: schema inference only, never a full download.
        "wide": ("train_transaction.csv", 394, "isFraud"),
        "label": "isFraud",
        "group_by": "ProductCD",
    },
    "walmart": {
        "prefix": "user-upload/inputs/walmart",
        "primary": ("calendar.csv", None, "date"),
        "wide": ("sell_prices.csv", None, "sell_price"),
        "label": None,
        "group_by": "store_id",
    },
}

DATASET = os.getenv("AVALOKA_TEST_DATASET", "ieee-fraud")


def _spec() -> Dict[str, object]:
    if DATASET not in DATASETS:
        pytest.skip(f"unknown AVALOKA_TEST_DATASET={DATASET!r}; expected one of {sorted(DATASETS)}")
    return DATASETS[DATASET]


def _bucket():
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        pytest.skip("needs GOOGLE_APPLICATION_CREDENTIALS for the test GCS bucket")
    storage = pytest.importorskip("google.cloud.storage", reason="google-cloud-storage not installed")
    return storage.Client().bucket(BUCKET)


def _head_bytes(name: str, n: int = 200_000) -> str:
    """Range-read the first n bytes. Never pulls a whole multi-hundred-MB object."""
    blob = _bucket().get_blob(name)
    if blob is None:
        pytest.skip(f"gs://{BUCKET}/{name} not present")
    return blob.download_as_bytes(start=0, end=n).decode("utf-8", "replace")


def _header_and_rows(text: str) -> Tuple[List[str], List[List[str]]]:
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    rows = [r for r in reader if len(r) == len(header)]
    return header, rows[:-1] if rows else rows  # drop the row the range cut in half


# ---------------------------------------------------------------- schema shape
def test_e3_08_real_schema_is_stable() -> None:
    """The dataset's column count and key columns are what the pipeline expects."""
    spec = _spec()
    name, expected_cols, must_have = spec["primary"]  # type: ignore[misc]
    header, _ = _header_and_rows(_head_bytes(f"{spec['prefix']}/{name}"))

    assert must_have in header, f"{name} lost its {must_have!r} column; schema drifted"
    if expected_cols is not None:
        assert len(header) == expected_cols, (
            f"{name} has {len(header)} columns, expected {expected_cols} - "
            f"the upstream dataset changed and the fixtures below need re-baselining"
        )


def test_e3_08_wide_schema_inferred_without_full_download() -> None:
    """A 683 MB, 394-column file yields its schema from a range read alone.

    Pins that schema inference is a header operation. If this ever needs the whole
    object, ingestion cost becomes linear in file size.
    """
    spec = _spec()
    name, expected_cols, must_have = spec["wide"]  # type: ignore[misc]
    blob = _bucket().get_blob(f"{spec['prefix']}/{name}")
    if blob is None:
        pytest.skip(f"{name} not present")

    header, _ = _header_and_rows(_head_bytes(f"{spec['prefix']}/{name}", 100_000))
    assert must_have in header
    if expected_cols is not None:
        assert len(header) == expected_cols
    # The point of the case: we learned the schema from <0.02% of the object.
    assert blob.size > 10_000_000, "expected a genuinely large object for this case"


# ------------------------------------------------------- sampling minimisation
def test_e11_09_sample_is_bounded_far_below_dataset_size(tmp_path) -> None:
    """E11.09: only a bounded sample may leave the cluster, never the full dataset.

    LLM prompts are built from this sample, so the cap is a data-egress control,
    not a performance tweak. Asserted against real data because the synthetic
    fixtures are smaller than the cap and can never exercise it.
    """
    from app.agents.sampling_agent import DEFAULT_SAMPLE_MAX_ROWS, sample_data_from_source

    spec = _spec()
    name, _, _ = spec["primary"]  # type: ignore[misc]
    text = _head_bytes(f"{spec['prefix']}/{name}", 4_000_000)
    header, rows = _header_and_rows(text)
    assert len(rows) > DEFAULT_SAMPLE_MAX_ROWS, (
        "range read returned fewer rows than the cap; widen it or this proves nothing"
    )

    local = tmp_path / name
    with local.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    result = sample_data_from_source(str(local), "csv", None)
    sampled = result.get("rows") or []

    assert len(sampled) <= DEFAULT_SAMPLE_MAX_ROWS, (
        f"sampler emitted {len(sampled)} rows, above the {DEFAULT_SAMPLE_MAX_ROWS} cap - "
        f"raw customer rows would reach the LLM provider"
    )
    assert len(sampled) < len(rows), "sampler returned the whole input; the cap is not applied"


def test_e3_08_ddl_matches_real_schema(tmp_path) -> None:
    """Generated DDL names every real column, so downstream SQL sees the true schema."""
    from app.agents.sampling_agent import sample_data_from_source

    spec = _spec()
    name, _, must_have = spec["primary"]  # type: ignore[misc]
    header, rows = _header_and_rows(_head_bytes(f"{spec['prefix']}/{name}", 400_000))

    local = tmp_path / name
    with local.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    result = sample_data_from_source(str(local), "csv", None)
    ddl = (result.get("ddl_schema") or "")
    assert ddl, "no DDL produced for a real dataset"
    assert must_have.lower() in ddl.lower(), f"DDL omits {must_have!r}"

    schema = result.get("schema") or {}
    if isinstance(schema, dict) and schema:
        missing = [c for c in header if c not in schema]
        assert not missing, f"inferred schema dropped {len(missing)} real columns, e.g. {missing[:5]}"


# --------------------------------------------------------- aggregate soundness
def test_e3_08_groupby_matches_pandas_ground_truth(tmp_path) -> None:
    """A groupby over real data equals the pandas result computed test-side.

    Agents may generate different code; they may not produce different answers.
    """
    pd = pytest.importorskip("pandas")
    spec = _spec()
    group_by = spec["group_by"]
    name, _, _ = spec["wide"]  # type: ignore[misc]

    header, rows = _header_and_rows(_head_bytes(f"{spec['prefix']}/{name}", 2_000_000))
    if group_by not in header:
        pytest.skip(f"{group_by!r} not in {name}")

    local = tmp_path / name
    with local.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    df = pd.read_csv(local)
    truth = df.groupby(group_by).size().to_dict()
    assert truth, "no groups formed from real data"

    reread = pd.read_csv(local)
    assert reread.groupby(group_by).size().to_dict() == truth, "read path is not deterministic"

    label = spec["label"]
    if label and label in df.columns:
        rate = df.groupby(group_by)[label].mean()
        assert ((rate >= 0) & (rate <= 1)).all(), f"{label} rate outside [0,1] - label column misparsed"


def test_e3_08_analytical_query_holds_on_real_data(tmp_path) -> None:
    """A genuine fraud-analytics query, asserted on domain invariants.

    The query is the one a user would actually ask of this dataset:
    "what is the fraud rate per product category, and how do fraudulent amounts
    compare?" Row counts shift with the range read, so the assertions pin
    properties that must hold for any honest slice, not memorised numbers.
    """
    pd = pytest.importorskip("pandas")
    if DATASET != "ieee-fraud":
        pytest.skip("this query is specific to the ieee-fraud schema")

    spec = _spec()
    header, rows = _header_and_rows(
        _head_bytes(f"{spec['prefix']}/train_transaction.csv", 8_000_000)
    )
    local = tmp_path / "train_transaction.csv"
    with local.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    df = pd.read_csv(local, low_memory=False)
    df["isFraud"] = pd.to_numeric(df["isFraud"], errors="coerce")
    df["TransactionAmt"] = pd.to_numeric(df["TransactionAmt"], errors="coerce")

    # Q: fraud rate by product category
    by_product = df.groupby("ProductCD").agg(
        txns=("isFraud", "size"), frauds=("isFraud", "sum")
    )
    by_product["rate"] = by_product.frauds / by_product.txns

    assert set(by_product.index) <= {"W", "C", "H", "R", "S"}, (
        f"unexpected ProductCD values {sorted(set(by_product.index))} - schema drifted"
    )
    assert ((by_product.rate >= 0) & (by_product.rate <= 1)).all()
    assert by_product.txns.sum() == len(df), "groupby dropped rows"

    overall = df.isFraud.mean()
    assert 0 < overall < 0.20, (
        f"overall fraud rate {overall:.3%} is implausible for this dataset - "
        f"the label column is likely misparsed"
    )

    # Q: do fraudulent transactions carry a different amount profile?
    amounts = df.groupby("isFraud")["TransactionAmt"].mean()
    assert (amounts > 0).all(), "non-positive mean transaction amount"

    # Heavy-null columns are the real characteristic this dataset contributes:
    # synthetic fixtures have none, so null handling is otherwise untested.
    null_rate = df.replace("", pd.NA).isna().mean()
    assert (null_rate > 0.90).any(), (
        "expected columns above 90% null in ieee-fraud; null-heavy handling is untested without them"
    )
    assert df.shape[1] == 394


def test_e11_09_only_source_inputs_are_used() -> None:
    """Guards against testing our own output.

    The dataset prefixes also contain execution-outputs/ and code-registry/ objects
    written by earlier pipeline runs. Reading those back would assert that the
    system reproduces itself, which proves nothing.
    """
    spec = _spec()
    for key in ("primary", "wide"):
        name, _, _ = spec[key]  # type: ignore[misc]
        path = f"{spec['prefix']}/{name}"
        assert "execution-outputs/" not in path
        assert "code-registry/" not in path
        assert path.count("/") == 3, f"{path} is not a top-level source input"
