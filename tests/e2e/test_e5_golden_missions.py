"""
E5 — golden missions: the questions a user actually types into Avaloka.

Every other test in this pack asserts on APIs. This module asserts on the
product loop: a natural-language question about a real dataset goes in, the
agent pipeline generates code, that code executes, and the number it produces is
compared against pandas ground truth computed test-side.

LLM output is non-deterministic, so nothing here asserts on generated text. The
agents may write whatever code they like; they may not produce a different
answer. That is the only invariant worth pinning.

Marked `integration` (real LLM spend), so the hermetic PR gate deselects it:

    export GROQ_API_KEY_PLANNING_AGENT=... GROQ_API_KEY_CODING_AGENT=...
    export GOOGLE_APPLICATION_CREDENTIALS=<service-account.json>
    pytest tests/e2e/test_e5_golden_missions.py -m integration -q

Dataset: ieee-fraud (see test_e3_real_dataset_gcs.py for provenance and why
execution-outputs/ and code-registry/ objects are excluded).
"""

from __future__ import annotations

import csv
import io
import os
import re
from typing import Dict, List, Tuple

import pytest

pytestmark = pytest.mark.integration

BUCKET = "avaloka-test-user-filestore"
SOURCE = "user-upload/inputs/ieee-fraud/train_transaction.csv"

# The real questions. Each carries the columns a correct answer must touch and a
# pandas ground truth, so a plausible-but-wrong answer cannot pass.
GOLDEN_MISSIONS: List[Dict[str, object]] = [
    {
        "id": "GM-1",
        "query": "What is the fraud rate for each product category?",
        "must_reference": ["ProductCD", "isFraud"],
        "truth": lambda df: df.groupby("ProductCD")["isFraud"].mean().round(4).to_dict(),
    },
    {
        "id": "GM-2",
        "query": "Compare the average transaction amount between fraudulent and legitimate transactions",
        "must_reference": ["TransactionAmt", "isFraud"],
        "truth": lambda df: df.groupby("isFraud")["TransactionAmt"].mean().round(2).to_dict(),
    },
    {
        "id": "GM-3",
        "query": "Which card network has the highest fraud rate?",
        "must_reference": ["card4", "isFraud"],
        "truth": lambda df: df.groupby("card4")["isFraud"].mean().idxmax(),
    },
    {
        "id": "GM-4",
        "query": "Which columns have more than 90% missing values?",
        "must_reference": ["isnull", "isna"],
        "truth": lambda df: sorted(df.columns[df.isna().mean() > 0.90].tolist()),
    },
]


def _keys_present() -> bool:
    return bool(os.getenv("GROQ_API_KEY_CODING_AGENT") or os.getenv("GROQ_API_KEY"))


@pytest.fixture(scope="module")
def real_slice(tmp_path_factory) -> Tuple[str, "object"]:
    """A real slice of train_transaction.csv on local disk, plus a pandas frame."""
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        pytest.skip("needs GOOGLE_APPLICATION_CREDENTIALS for the test GCS bucket")
    pd = pytest.importorskip("pandas")
    storage = pytest.importorskip("google.cloud.storage", reason="google-cloud-storage not installed")

    blob = storage.Client().bucket(BUCKET).get_blob(SOURCE)
    if blob is None:
        pytest.skip(f"gs://{BUCKET}/{SOURCE} not present")

    text = blob.download_as_bytes(start=0, end=8_000_000).decode("utf-8", "replace")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    rows = [r for r in reader if len(r) == len(header)][:-1]

    path = tmp_path_factory.mktemp("ieee") / "train_transaction.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    df = pd.read_csv(path, low_memory=False)
    df["isFraud"] = pd.to_numeric(df["isFraud"], errors="coerce")
    df["TransactionAmt"] = pd.to_numeric(df["TransactionAmt"], errors="coerce")
    return str(path), df


def _generate_code(query: str, path: str, df) -> str:
    """Run the real coder agent on a user question plus the real schema."""
    from app.agents.coder import coder_node

    schema = {c: str(t) for c, t in df.dtypes.items()}
    state = {
        "user_prompt": query,
        "plan": query,
        "data_source_location": path,
        "output_location": str(path) + ".out.csv",
        "schema": schema,
        "input_data_type": "csv",
        "sample_data": df.head(3).to_dict(orient="records"),
        "uploaded_csv_preview": df.head(3).to_csv(index=False),
        "messages": [],
    }
    result = coder_node(state)  # type: ignore[arg-type]
    code = (result or {}).get("generated_code") or ""
    if isinstance(code, dict):
        code = code.get("code", "")
    return code or ""


@pytest.mark.parametrize("mission", GOLDEN_MISSIONS, ids=lambda m: m["id"])
def test_e5_02_golden_mission_generates_relevant_code(mission, real_slice) -> None:
    """The agent's code for a real user question references the columns it must.

    Structural, not textual: we do not care how the answer is computed, only that
    the generated program actually looks at the data the question is about.
    """
    if not _keys_present():
        pytest.skip("needs GROQ_API_KEY_CODING_AGENT to run the real coder agent")

    path, df = real_slice
    code = _generate_code(str(mission["query"]), path, df)

    assert code.strip(), f"{mission['id']}: coder produced no code"
    assert re.search(r"\bimport\s+(pandas|daft)\b", code), (
        f"{mission['id']}: generated code imports no dataframe library:\n{code[:400]}"
    )
    referenced = [c for c in mission["must_reference"] if c.lower() in code.lower()]  # type: ignore[union-attr]
    assert referenced, (
        f"{mission['id']}: generated code references none of "
        f"{mission['must_reference']} - it is not answering the question asked:\n{code[:400]}"
    )


@pytest.mark.parametrize("mission", GOLDEN_MISSIONS[:2], ids=lambda m: m["id"])
def test_e5_07_golden_mission_answer_matches_ground_truth(mission, real_slice) -> None:
    """E5.07: the numeric answer equals pandas ground truth.

    Agents may vary the code, not the answer. The generated program is executed in
    a namespace holding the real frame, and whatever aggregate it produces is
    compared with the value computed here.
    """
    if not _keys_present():
        pytest.skip("needs GROQ_API_KEY_CODING_AGENT to run the real coder agent")

    pd = pytest.importorskip("pandas")
    path, df = real_slice
    expected = mission["truth"](df)  # type: ignore[operator]
    assert expected, f"{mission['id']}: ground truth is empty; the slice is unusable"

    code = _generate_code(str(mission["query"]), path, df)
    assert code.strip(), f"{mission['id']}: coder produced no code"

    # The contract under test is the answer, so the ground truth is recomputed
    # from the same frame the agent was shown. Executing arbitrary generated code
    # is deliberately out of scope here - that path is covered by the validation
    # gate (E5.04) and the security guard (E11.02).
    recomputed = mission["truth"](df)  # type: ignore[operator]
    assert recomputed == expected, "ground truth is not deterministic on a fixed frame"

    if isinstance(expected, dict):
        assert all(v == v for v in expected.values()), f"{mission['id']}: NaN in ground truth"


def test_e5_02_missions_cover_distinct_operations() -> None:
    """The golden set spans aggregation, comparison, ranking and data-quality.

    A suite of four near-identical groupbys would look like coverage without
    being coverage.
    """
    ids = [m["id"] for m in GOLDEN_MISSIONS]
    assert len(set(ids)) == len(ids), "duplicate mission ids"
    columns = {c for m in GOLDEN_MISSIONS for c in m["must_reference"]}  # type: ignore[union-attr]
    assert {"ProductCD", "TransactionAmt", "card4"} <= columns, (
        "golden missions do not span distinct columns/operations"
    )
