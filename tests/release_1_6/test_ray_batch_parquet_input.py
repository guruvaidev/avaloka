"""The batch Ray path answers a Parquet input from the Parquet, not from read_csv.

``_build_ray_wrapper`` (the ``batch_heavy`` profile) read every input with
``pd.read_csv`` and counted rows as newline bytes, whatever the extension. A
Parquet dataset was decoded as text: the row count was noise and the sample was
garbage or a parse error, while the interactive profile already branched on
``.parquet``. Found while preparing the Ray 2.52 kind run (INVENTORY D2).

Runs the real path as test_maj031_ray_output_contract does: the script is
rendered by ``execution_agent_node_ray``, taken from the RayJob ConfigMap and
run as a subprocess with a stand-in ``ray`` module, against a Parquet file with
several row groups written from the same rows as that module's CSV. The result
and the reported row count are compared with pandas on the same rows.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import app.agents.execution_agent as ea
from tests.release_1_6.test_maj031_ray_output_contract import (
    CORRECT_LLM_CODE, _assert_measure, _frame_from_output_file, _ray_patches, _ray_state,
)


def _run_batch(tmp_path, data_path):
    state, csv_path = _ray_state(tmp_path, "List the lowest and highest price per category",
                                 code=CORRECT_LLM_CODE)
    state["data_source_location_cloud"] = str(data_path)
    state["ray_execution_profile"] = "batch_heavy"
    ps = _ray_patches(tmp_path)
    for p in ps:
        p.start()
    try:
        return ea.execution_agent_node_ray(state), csv_path
    finally:
        for p in reversed(ps):
            p.stop()


def test_batch_ray_wrapper_reads_parquet_input(tmp_path):
    state, csv_path = _ray_state(tmp_path, "x")      # writes the reference CSV
    pq_path = tmp_path / "products.parquet"
    rows = pd.read_csv(csv_path)
    pq.write_table(pa.Table.from_pandas(rows, preserve_index=False), pq_path, row_group_size=16)
    assert pq.ParquetFile(pq_path).metadata.num_row_groups == 4

    out, _ = _run_batch(tmp_path, pq_path)
    er = out["execution_result"]
    logs = er.get("logs", "")
    assert er["status"] in ("success", "succeeded"), (er.get("message"), logs[-2000:])
    assert "Parquet: 61 rows in 4 row groups" in logs, logs[-2000:]
    assert f"ROWS IN FILE = {len(rows)}" in logs, logs[-2000:]
    frame = _frame_from_output_file(out.get("output_file_data"))
    _assert_measure(frame, csv_path, ["min", "max"])


def test_batch_ray_wrapper_still_reads_csv(tmp_path):
    """The text path is unchanged: same answer and the same row count from the CSV."""
    state, csv_path = _ray_state(tmp_path, "x")
    out, _ = _run_batch(tmp_path, csv_path)
    er = out["execution_result"]
    logs = er.get("logs", "")
    assert er["status"] in ("success", "succeeded"), (er.get("message"), logs[-2000:])
    assert "ROWS IN FILE = 61" in logs and "Parquet:" not in logs, logs[-2000:]
    _assert_measure(_frame_from_output_file(out.get("output_file_data")), csv_path, ["min", "max"])
