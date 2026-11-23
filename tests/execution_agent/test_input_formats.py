"""The execution agent reads its input by format, as the upload path does.

From the web e2e (gemma4:e4b, 2026-10-18): every Parquet and JSONL analysis
failed with "ParserError: Error tokenizing data. C error: Expected 2 fields in
line 4, saw 4", because the local harness ran
``_avaloka_df = read_csv_best_effort(_avaloka_input_path)`` on whatever the
input was. The upload and the sampler had read the same files correctly.

Each case below runs the real local execution node -- the real script, in a
real subprocess -- on code fixed to what the coder produced in that run (the
stubbed LLM's output), and checks the numbers that come back against pandas
on the frame the fixture was written from. Nothing here needs an LLM.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage

import app.agents.execution_agent as exec_mod

#: The coder's code from the failing taxi run, verbatim but for the column
#: names. Its own main guard reads 'input.csv'; the harness replaces it.
CODER_CODE = (
    "import pandas as pd\n"
    "import numpy as np\n\n"
    "def main(df):\n"
    "    df_agg = df.groupby('payment_type', as_index=False).agg("
    "average_fare_amount=('fare_amount', 'mean'))\n"
    "    return df_agg\n"
    "if __name__ == '__main__':\n"
    "    df = read_csv_best_effort('input.csv')\n"
    "    df = main(df)\n"
    "    df.to_csv('output.csv', index=False)\n"
)

#: The other shape the coder writes: the read inside the script, at the top,
#: on a literal path the execution agent points at the real input.
INLINE_READ_CODE = (
    "import pandas as pd\n"
    "df = read_csv_best_effort('trips.parquet')\n"
    "out = df.groupby('payment_type', as_index=False).agg("
    "average_fare_amount=('fare_amount', 'mean'))\n"
    "out.to_csv('output.csv', index=False)\n"
)


@pytest.fixture
def trips() -> pd.DataFrame:
    return pd.DataFrame({
        "payment_type": [1, 2, 1, 3, 2, 1, 1, 2, 3, 1, 2, 1],
        "fare_amount": [10.5, 7.25, 12.0, 30.0, 8.75, 9.5, 14.25, 6.0, 22.5, 11.0, 7.5, 13.75],
    })


def _expected(frame: pd.DataFrame) -> dict:
    means = frame.groupby("payment_type")["fare_amount"].mean()
    return {int(k): round(float(v), 6) for k, v in means.items()}


def _run(input_path: Path, tmp_path: Path, code: str = CODER_CODE, **state) -> dict:
    base = {
        "messages": [HumanMessage(content="What is the average fare_amount by payment_type?")],
        "coder_definition": {"code": code},
        "data_source_location": str(input_path),
        "output_location": str(tmp_path / "out" / "output.csv"),
    }
    base.update(state)
    return exec_mod.execution_agent_node_local(base)


def _means(new_state: dict) -> dict:
    result = new_state["execution_result"]
    assert result["status"] == "success", (
        f"the analysis failed: {result.get('message')!r}\n{new_state.get('execution_stderr')}")
    return {int(row["payment_type"]): round(float(row["average_fare_amount"]), 6)
            for row in new_state["output_json"]}


def test_a_parquet_input_with_several_row_groups_is_read_whole(tmp_path, trips):
    path = tmp_path / "trips.parquet"
    trips.to_parquet(path, row_group_size=4)          # three row groups
    import pyarrow.parquet as pq
    assert pq.ParquetFile(path).num_row_groups == 3

    assert _means(_run(path, tmp_path)) == _expected(trips)


def test_a_jsonl_input_is_read_as_records(tmp_path, trips):
    path = tmp_path / "trips.jsonl"
    trips.to_json(path, orient="records", lines=True)

    assert _means(_run(path, tmp_path)) == _expected(trips)


def test_a_directory_of_parquet_parts_is_one_dataset(tmp_path, trips):
    folder = tmp_path / "trips"
    folder.mkdir()
    trips.iloc[:5].to_parquet(folder / "part-00000.parquet")
    trips.iloc[5:].to_parquet(folder / "part-00001.parquet")
    (folder / "_SUCCESS").write_text("")             # a job marker, not data

    assert _means(_run(folder, tmp_path)) == _expected(trips)


def test_a_hive_partitioned_directory_keeps_its_partition_column(tmp_path, trips):
    """payment_type exists only in the folder names; the sampler recovers it."""
    folder = tmp_path / "trips_by_type"
    for value, group in trips.groupby("payment_type"):
        part = folder / f"payment_type={value}"
        part.mkdir(parents=True)
        group.drop(columns="payment_type").to_parquet(part / "part-0.parquet")

    out = _run(folder, tmp_path, hive_partitioning=True)

    assert _means(out) == _expected(trips)


def test_a_literal_read_in_the_code_is_pointed_at_the_parquet_and_read_as_parquet(tmp_path, trips):
    path = tmp_path / "trips.parquet"
    trips.to_parquet(path)

    assert _means(_run(path, tmp_path, code=INLINE_READ_CODE)) == _expected(trips)


def test_a_tsv_input_is_split_on_tabs(tmp_path, trips):
    path = tmp_path / "trips.tsv"
    trips.to_csv(path, sep="\t", index=False)

    assert _means(_run(path, tmp_path)) == _expected(trips)


def test_a_workbook_is_read_on_the_sheet_the_session_chose(tmp_path, trips):
    path = tmp_path / "trips.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"note": ["cover sheet"] * 20}).to_excel(writer, sheet_name="Cover", index=False)
        trips.to_excel(writer, sheet_name="Trips", index=False)
    meta = [{"data_source_location": str(path), "sheet_name": "Trips"}]

    assert _means(_run(path, tmp_path, multi_dataset_state=meta)) == _expected(trips)


def test_a_csv_input_still_goes_through_the_one_csv_reader(tmp_path):
    """The null-word policy still applies: 'None' in a text column is a label."""
    path = tmp_path / "trips.csv"
    path.write_text("method,amount\nNone,1\ncard,2\nNone,3\n")
    code = (
        "import pandas as pd\n"
        "def main(df):\n"
        "    return df.groupby('method', as_index=False)['amount'].sum()\n"
    )

    out = _run(path, tmp_path, code=code)

    assert out["execution_result"]["status"] == "success"
    # pandas' own defaults would drop the two 'None' rows from the group-by and
    # leave card alone. (The label reads back as NaN only because the node
    # re-reads output.csv with pandas' defaults; the input read kept it.)
    assert sorted(row["amount"] for row in out["output_json"]) == [2, 4]
