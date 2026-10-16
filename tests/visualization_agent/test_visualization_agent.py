"""Row accounting and the node's data-selection fallbacks.

Repaired from the quarantined suite that drove `VisualizationAgent.generate_visualizations`.
That class is gone (grep finds no `VisualizationAgent` anywhere in app/ or avaloka/);
the entry points are now `build_visualization_config_from_sample` and the
`visualization_agent_node` wrapper, and the config shape changed from ECharts
(`metadata.total_charts`, `config.xAxis.data`) to a profile whose charts carry
`encodings` + `derived_data.points`.

What is kept here is what the gate does not otherwise assert. The chart-shape
assertions were dropped because tests/test_visualization_grounding.py covers
them more strictly (`test_the_deterministic_path_grounds_every_chart_it_ships`
requires non-empty points on every shipped chart; the scatter tests pin exact
point values). What nothing selected by `-m "not integration"` asserts is the
*row accounting* -- how many rows the config says it profiled, and which table
those rows came from -- and the no-data path. When this was written those
lived only in a file conftest.py auto-marked `integration` on its
`_integration.py` suffix. That file is now tests/test_visualization_unit.py
and runs in the gate, so the two overlap on purpose.

No LLM: viz_llm is forced to None, so chart selection takes its deterministic
path and every number below is computed, not sampled from a model.
"""

from __future__ import annotations

import base64

import pytest

from app.agents.visualization_agent import (
    build_visualization_config_from_sample,
    visualization_agent_node,
)


@pytest.fixture(autouse=True)
def disable_viz_llm(monkeypatch):
    """Ensure tests do not invoke the external LLM."""
    monkeypatch.setattr("app.agents.visualization_agent.viz_llm", None)


#: A categorical that repeats plus two measures: enough for the deterministic
#: chart selector to ship something.
def _rows(n: int = 12) -> list[dict]:
    return [
        {
            "Region": ("North", "South", "East")[i % 3],
            "Sales": 10_000 + (i * 1_300) % 7_000,
            "Units": 50 + (i * 7) % 31,
        }
        for i in range(n)
    ]


def _datauri(csv_text: str) -> str:
    return "data:text/csv;base64," + base64.b64encode(csv_text.encode()).decode()


# ---------------------------------------------------------------------------
# The config reports the rows it actually profiled
#
# Was: assert result["metadata"]["data_rows"] == len(sample_data)
# ---------------------------------------------------------------------------


def test_the_config_reports_the_row_count_it_profiled():
    """`dataset.rows_sampled` is the row count the whole profile is computed on.

    Every share, mean and correlation in the shipped charts is derived from
    these rows, and the UI prints the number next to them, so an off-by-one or
    a stale count mislabels correct charts.
    """
    rows = _rows(12)

    cfg = build_visualization_config_from_sample(dataset_id="ds_rows", sample_rows=rows)

    assert cfg["dataset"]["rows_sampled"] == len(rows)
    # No full_data_path was given, so the charts are grounded on the sample
    # itself and the two counts must agree.
    assert cfg["dataset"]["rows_charted"] == len(rows)
    assert cfg["dataset"]["charts_scope"] == "sample"


# ---------------------------------------------------------------------------
# The node profiles the computed result, not the uploaded file
#
# Was: test_execution_result_rows_used_for_metadata --
#      assert result["metadata"]["data_rows"] == len(execution_output)
# The result now arrives as a data-URI CSV in state["output_file_data"].
# ---------------------------------------------------------------------------


def test_the_node_counts_the_computed_result_rows_not_the_upload_rows():
    """When a transformation produced a table, that table is what gets profiled.

    tests/test_visualization_profiles_the_result_table.py asserts the charted
    *fields* come from the result. It does not assert the row count, so a node
    that profiled the result's columns while reporting the upload's row count
    would pass it.
    """
    result_rows = [
        ("North", 100),
        ("South", 80),
        ("East", 120),
        ("West", 90),
    ]
    csv_text = "Region,Total\n" + "".join(f"{r},{v}\n" for r, v in result_rows)

    out = visualization_agent_node(
        {
            "dataset_id": "ds_exec",
            "schema": {"Region": "object", "Sales": "int64", "Units": "int64"},
            "uploaded_csv_columns": ["Region", "Sales", "Units"],
            # 12 uploaded rows -- deliberately a different count from the result.
            "uploaded_csv_preview": [[r["Region"], r["Sales"], r["Units"]] for r in _rows(12)],
            "output_file_data": {"content": _datauri(csv_text)},
        }
    )

    cfg = out["visualization_config"]
    assert out["visualization_status"] == "ready"
    assert cfg["dataset"]["rows_sampled"] == len(result_rows)
    # And it is the result's columns that were profiled, not the upload's.
    assert {c["name"] for c in cfg["columns"]} == {"Region", "Total"}


# ---------------------------------------------------------------------------
# The node charts a positional preview
#
# Was: test_visualization_agent_node_converts_preview
# ---------------------------------------------------------------------------


def test_the_node_charts_a_positional_preview_with_its_column_names():
    """With no computed result, the uploaded preview is charted.

    The preview reaches the node as list-of-lists plus a separate column list
    (app/api/streamlit_app.py builds it as `[list(row.values()) for row in
    rows]`), so the node has to zip them back into rows before anything can be
    profiled. If that conversion breaks, the node reports "no data" and the
    user sees no charts for a file that has plenty.
    """
    rows = _rows(12)
    columns = ["Region", "Sales", "Units"]
    preview = [[r[c] for c in columns] for r in rows]

    out = visualization_agent_node(
        {
            "dataset_id": "ds_preview",
            "uploaded_csv_columns": columns,
            "uploaded_csv_preview": preview,
        }
    )

    cfg = out["visualization_config"]
    assert out["visualization_status"] == "ready"
    assert cfg["dataset"]["rows_sampled"] == len(preview)
    # The positional values landed under the names they were given, so the
    # profile is of Region/Sales/Units and not of col_0/col_1/col_2.
    assert {c["name"] for c in cfg["columns"]} == set(columns)
    assert cfg["charts"], "a categorical and two measures produced no charts"


# ---------------------------------------------------------------------------
# No data is reported, not raised
#
# Was: test_generate_visualizations_without_data_returns_error --
#      assert result["status"] == "error"; assert "No data" in metadata["error"]
# ---------------------------------------------------------------------------


def test_the_node_reports_no_data_rather_than_failing_the_graph():
    """A state with nothing to chart must leave the graph runnable.

    The node is a LangGraph node: raising here would fail the user's whole run
    over a missing chart. It returns an empty config and a status that names
    the reason instead. app/api/server.py reads that falsy config to fall back
    to the session's charts, so the empty dict is load-bearing, not cosmetic.
    """
    out = visualization_agent_node({"dataset_id": "ds_empty"})

    assert out["visualization_config"] == {}
    assert out["visualization_status"] == "skipped: no data available"
