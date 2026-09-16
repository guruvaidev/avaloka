import copy
from typing import Any, Dict, List

import pytest
import json
# Adjust this to your actual module path
import app.agents.visualization_agent as viz
from app.graph.etl_state import ETLState

# Hard-disable LLM in tests so we don't hit Groq
viz.viz_llm = None

build_visualization_config_from_sample = viz.build_visualization_config_from_sample
visualization_agent_node = viz.visualization_agent_node
_attach_chart_reasons = viz._attach_chart_reasons
MAX_TOTAL_CHARTS = viz.MAX_TOTAL_CHARTS


# -------------------------------------------------------------------
# Test fixtures
# -------------------------------------------------------------------


@pytest.fixture
def small_sample_rows_dict() -> List[Dict[str, Any]]:
    """
    Small synthetic dataset in dict-row form.
    Columns roughly mimic your YouTube dataset structure.
    """
    return [
        {
            "rank": 1,
            "Youtuber": "ChannelA",
            "subscribers": 1_000_000,
            "video views": 50_000_000,
            "category": "Entertainment",
        },
        {
            "rank": 2,
            "Youtuber": "ChannelB",
            "subscribers": 2_000_000,
            "video views": 120_000_000,
            "category": "Music",
        },
        {
            "rank": 3,
            "Youtuber": "ChannelC",
            "subscribers": 500_000,
            "video views": 10_000_000,
            "category": "Gaming",
        },
        {
            "rank": 4,
            "Youtuber": "ChannelD",
            "subscribers": 3_500_000,
            "video views": 200_000_000,
            "category": "Entertainment",
        },
        {
            "rank": 5,
            "Youtuber": "ChannelE",
            "subscribers": 800_000,
            "video views": 30_000_000,
            "category": "Education",
        },
    ]


@pytest.fixture
def small_sample_rows_list() -> Dict[str, Any]:
    """
    Same synthetic dataset, but in list-row form + columns, to exercise
    the _normalize_sample_rows path via visualization_agent_node.
    """
    columns = ["rank", "Youtuber", "subscribers", "video views", "category"]
    rows = [
        [1, "ChannelA", 1_000_000, 50_000_000, "Entertainment"],
        [2, "ChannelB", 2_000_000, 120_000_000, "Music"],
        [3, "ChannelC", 500_000, 10_000_000, "Gaming"],
        [4, "ChannelD", 3_500_000, 200_000_000, "Entertainment"],
        [5, "ChannelE", 800_000, 30_000_000, "Education"],
    ]
    return {"columns": columns, "rows": rows}


# -------------------------------------------------------------------
# Integration tests for build_visualization_config_from_sample
# -------------------------------------------------------------------


def test_build_visualization_config_basic_structure(small_sample_rows_dict):
    config = build_visualization_config_from_sample(
        dataset_id="ds_test",
        sample_rows=small_sample_rows_dict,
        schema=None,
        task_type="unsupervised",
        target_column=None,
    )

    # Top-level structure
    assert config["version"] == "1.0"
    assert config["dataset"]["dataset_id"] == "ds_test"
    assert config["dataset"]["rows_sampled"] == len(small_sample_rows_dict)
    assert config["task"]["type"] == "unsupervised"
    assert config["task"]["target_column"] is None
    assert "columns" in config and len(config["columns"]) > 0

    # Feature ranking
    fr = config["feature_ranking"]
    assert fr["strategy"] == "variance+cardinality"
    assert len(fr["features"]) == len(config["columns"])

    # Charts
    charts = config["charts"]
    assert 1 <= len(charts) <= MAX_TOTAL_CHARTS
    # Every chart should have basic keys and a reason
    for ch in charts:
        assert "id" in ch
        assert "title" in ch
        assert "type" in ch
        assert "intent" in ch
        assert "encodings" in ch
        assert "config" in ch
        assert "derived_data" in ch
        assert "rank" in ch
        assert "reason" in ch
        assert isinstance(ch["reason"], str)
        assert ch["reason"].strip()  # non-empty


def test_build_visualization_config_with_target_column(small_sample_rows_dict):
    """
    Same as above, but with a target column set to ensure nothing breaks
    and that task section is updated accordingly.
    """
    config = build_visualization_config_from_sample(
        dataset_id="ds_target",
        sample_rows=small_sample_rows_dict,
        schema=None,
        task_type="supervised",
        target_column="video views",
    )

    # Task metadata
    assert config["task"]["type"] == "supervised"
    assert config["task"]["target_column"] == "video views"
    assert "Supervised" in config["task"]["reason"]

    # Charts still present and each has a reason
    charts = config["charts"]
    assert len(charts) > 0
    for ch in charts:
        assert "reason" in ch
        assert isinstance(ch["reason"], str)
        assert ch["reason"].strip()


def test_attach_chart_reasons_direct_call():
    """
    Directly exercise _attach_chart_reasons with a simple synthetic chart
    and ranking to verify that it writes a sensible reason.
    """
    charts = [
        {
            "id": "test_scatter",
            "title": "Subscribers vs Views",
            "type": "scatter",
            "intent": "relationship",
            "rank": 1,
            "data_source": "sample",
            "encodings": {
                "x": {"field": "subscribers", "type": "quantitative"},
                "y": {"field": "video views", "type": "quantitative"},
            },
            "config": {},
            "derived_data": {},
        }
    ]

    feature_ranking = {
        "strategy": "variance+cardinality",
        "features": [
            {"name": "video views", "score": 1.0},
            {"name": "subscribers", "score": 0.8},
        ],
    }

    _attach_chart_reasons(charts, feature_ranking, target_column="video views")

    assert "reason" in charts[0]
    reason = charts[0]["reason"]
    assert "**subscribers**" in reason or "subscribers" in reason
    assert "**video views**" in reason or "video views" in reason
    # mention target column note
    assert "target column" in reason.lower()


# -------------------------------------------------------------------
# End-to-end tests for visualization_agent_node
# -------------------------------------------------------------------


def test_visualization_agent_node_no_data():
    """
    When no sample data is present at all, the node should
    skip visualization and set an informative status.
    """
    empty_state: ETLState = {}  # type: ignore[assignment]

    new_state = visualization_agent_node(empty_state)

    assert "visualization_config" in new_state
    assert new_state["visualization_config"] == {}
    assert "visualization_status" in new_state
    assert "skipped: no data available" in new_state["visualization_status"]


def test_visualization_agent_node_with_uploaded_preview(small_sample_rows_list):
    """
    Full E2E test:
      - state carries uploaded_csv_preview as list rows
      - node normalizes rows
      - builds visualization_config
      - attaches charts + reasons
    """
    columns = small_sample_rows_list["columns"]
    rows = small_sample_rows_list["rows"]

    initial_state: ETLState = {  # type: ignore[assignment]
        "dataset_id": "ds_uploaded_preview",
        "uploaded_csv_columns": columns,
        "uploaded_csv_preview": rows,
    }

    # Make sure we don't mutate the original state object internally
    original_state = copy.deepcopy(initial_state)

    new_state = visualization_agent_node(initial_state)

    # state object itself should not be mutated (we create a copy)
    assert initial_state == original_state

    # New state should have visualization fields
    assert "visualization_config" in new_state
    assert "visualization_status" in new_state
    assert new_state["visualization_status"] == "ready"

    config = new_state["visualization_config"]
    assert config["dataset"]["dataset_id"] == "ds_uploaded_preview"
    assert config["dataset"]["rows_sampled"] == len(rows)

    # Columns detected
    assert "columns" in config
    assert len(config["columns"]) == len(columns)

    # Charts present
    charts = config["charts"]
    assert 1 <= len(charts) <= MAX_TOTAL_CHARTS

    # Each chart has a non-empty reason
    for ch in charts:
        assert "reason" in ch
        assert isinstance(ch["reason"], str)
        assert ch["reason"].strip()


def test_visualization_agent_node_prefers_execution_result_over_preview(
    small_sample_rows_dict,
    small_sample_rows_list,
):
    """
    When both execution_result and uploaded preview exist,
    the node should prefer execution_result.output_json.
    We can't easily check internals, but we can ensure it still
    returns a valid config and doesn't crash.
    """
    # execution_result as dict rows
    exec_rows = small_sample_rows_dict
    # preview as list rows
    columns = small_sample_rows_list["columns"]
    rows = small_sample_rows_list["rows"]

    state: ETLState = {  # type: ignore[assignment]
        "dataset_id": "ds_exec_pref",
        "execution_result": {
            "output_json": exec_rows,
            "output_data": [],  # unused because output_json exists
        },
        "uploaded_csv_columns": columns,
        "uploaded_csv_preview": rows,
    }

    new_state = visualization_agent_node(state)

    assert new_state["visualization_status"] == "ready"
    config = new_state["visualization_config"]

    # Because execution_result is used, rows_sampled == len(exec_rows)
    assert config["dataset"]["rows_sampled"] == len(exec_rows)

    # Sanity: charts exist and have reasons
    charts = config["charts"]
    assert len(charts) > 0
    for ch in charts:
        assert "reason" in ch
        assert ch["reason"].strip()


# -------------------------------------------------------------------
# Additional unit-style tests for visualization_agent internals
# -------------------------------------------------------------------

def test_is_missing_and_to_float_behavior():
    # _is_missing cases
    assert viz._is_missing(None) is True
    assert viz._is_missing(float("nan")) is True
    assert viz._is_missing(" NA ") is True
    assert viz._is_missing("nan") is True
    assert viz._is_missing("") is True
    assert viz._is_missing("   ") is True
    assert viz._is_missing("foo") is False
    assert viz._is_missing(0) is False

    # _to_float respects _is_missing
    assert viz._to_float(None) is None
    assert viz._to_float("nan") is None
    assert viz._to_float(" NA ") is None

    # valid conversions
    assert viz._to_float("10") == 10.0
    assert viz._to_float("  10  ") == 10.0
    assert viz._to_float("1,234") == 1234.0
    assert viz._to_float(3) == 3.0
    assert viz._to_float(3.5) == 3.5

    # invalid numeric string
    assert viz._to_float("not-a-number") is None


def test_profile_columns_numeric_and_categorical():
    rows = [
        {"num": "1", "cat": "a"},
        {"num": "2", "cat": "b"},
        {"num": "3", "cat": None},
        {"num": "x", "cat": "a"},
    ]
    profiles = viz.profile_columns(rows)

    by_name = {p["name"]: p for p in profiles}
    assert set(by_name.keys()) == {"num", "cat"}

    num_col = by_name["num"]
    cat_col = by_name["cat"]

    # numeric detection
    assert num_col["dtype"] == "numeric"
    assert num_col["missing_ratio"] == 0.0
    assert num_col["n_unique"] == 3  # 1,2,3
    stats = num_col["stats"]
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
    assert stats["std"] >= 0.0

    # categorical detection + missing handling
    assert cat_col["dtype"] == "categorical"
    assert cat_col["missing_ratio"] == pytest.approx(0.25)
    assert cat_col["n_unique"] >= 2
    assert "top_k" in cat_col["stats"]
    # ensure None turned into "nan" bucket
    top_values = {t["value"] for t in cat_col["stats"]["top_k"]}
    assert "nan" in top_values


def test_compute_unsupervised_feature_ranking_basic():
    rows = [
        {"num": 1, "cat": "a"},
        {"num": 2, "cat": "b"},
        {"num": 3, "cat": "a"},
    ]
    cols = viz.profile_columns(rows)
    ranking = viz.compute_unsupervised_feature_ranking(cols)

    assert ranking["strategy"] == "variance+cardinality"
    features = ranking["features"]
    assert {f["name"] for f in features} == {"num", "cat"}

    # scores are normalized to [0,1]
    max_score = max(f["score"] for f in features)
    min_score = min(f["score"] for f in features)
    assert max_score == pytest.approx(1.0)
    assert 0.0 <= min_score <= 1.0


def test_select_charts_respects_limits_and_types(small_sample_rows_dict):
    # Build viz_profile similar to what build_visualization_config_from_sample uses
    cols = viz.profile_columns(small_sample_rows_dict)
    fr = viz.compute_unsupervised_feature_ranking(cols)
    viz_profile = {
        "dataset": {
            "dataset_id": "ds_select_test",
            "rows_sampled": len(small_sample_rows_dict),
            "columns": [c["name"] for c in cols],
        },
        "columns": cols,
        "feature_ranking": fr,
    }

    charts = viz.select_charts(viz_profile)

    assert 1 <= len(charts) <= viz.MAX_TOTAL_CHARTS

    # Type counts respect caps
    hist_count = sum(1 for c in charts if c["type"] == "histogram")
    scatter_count = sum(1 for c in charts if c["type"] == "scatter")
    bar_count = sum(1 for c in charts if c["type"] == "bar")

    assert hist_count >= 1
    assert hist_count <= viz.MAX_DIST_CHARTS
    assert scatter_count <= viz.MAX_RELATIONSHIP_CHARTS
    assert bar_count <= viz.MAX_CATEGORY_CHARTS

    # ranks must be contiguous 1..N
    ranks = sorted(c["rank"] for c in charts)
    assert ranks == list(range(1, len(charts) + 1))


def test_detect_bias_and_issues_flags_expected_issues():
    columns_meta = [
        {
            "name": "constant_col",
            "dtype": "numeric",
            "missing_ratio": 0.0,
            "n_unique": 1,
            "stats": {},
        },
        {
            "name": "missing_col",
            "dtype": "numeric",
            "missing_ratio": 0.5,
            "n_unique": 10,
            "stats": {},
        },
        {
            "name": "imbalanced_cat",
            "dtype": "categorical",
            "missing_ratio": 0.0,
            "n_unique": 3,
            "stats": {
                "top_k": [
                    {"value": "A", "count": 96},
                    {"value": "B", "count": 2},
                    {"value": "C", "count": 2},
                ]
            },
        },
    ]

    diag = viz.detect_bias_and_issues(columns_meta)

    assert diag["enabled"] is True
    issues = {(i["column"], i["type"]) for i in diag["issues"]}

    assert ("constant_col", "constant") in issues
    assert ("missing_col", "missing") in issues
    assert ("imbalanced_cat", "imbalance") in issues

    # Expect at least one warning per issue
    assert len(diag["warnings"]) >= 3


def test_normalize_sample_rows_variants():
    # 1) Already dict rows: should be returned as-is
    dict_rows = [{"a": 1}, {"a": 2}]
    norm1 = viz._normalize_sample_rows(dict_rows, columns=None)
    assert norm1 is dict_rows  # same object

    # 2) List rows + columns
    columns = ["rank", "name"]
    list_rows = [[1, "A"], [2, "B"]]
    norm2 = viz._normalize_sample_rows(list_rows, columns=columns)
    assert norm2 == [
        {"rank": 1, "name": "A"},
        {"rank": 2, "name": "B"},
    ]

    # 3) List rows without columns -> synthetic col_0, col_1,...
    list_rows2 = [[10, 20]]
    norm3 = viz._normalize_sample_rows(list_rows2, columns=None)
    assert norm3 == [{"col_0": 10, "col_1": 20}]

    # 4) "Weird" scalar rows -> wrapped into {"value": ...}
    scalar_rows = [42, 43]
    norm4 = viz._normalize_sample_rows(scalar_rows, columns=None)
    assert norm4 == [{"value": 42}, {"value": 43}]


def test_merge_charts_deduplicates_by_signature():
    primary = [
        {
            "id": "chart_scatter",
            "title": "X vs Y",
            "type": "scatter",
            "intent": "relationship",
            "rank": 1,
            "data_source": "sample",
            "encodings": {
                "x": {"field": "X", "type": "quantitative"},
                "y": {"field": "Y", "type": "quantitative"},
            },
            "config": {},
            "derived_data": {},
        }
    ]

    extras = [
        # Duplicate scatter (same signature) should be ignored
        {
            "id": "chart_scatter_dup",
            "title": "Duplicate X vs Y",
            "type": "scatter",
            "intent": "relationship",
            "rank": 1,
            "data_source": "sample",
            "encodings": {
                "x": {"field": "X", "type": "quantitative"},
                "y": {"field": "Y", "type": "quantitative"},
            },
            "config": {},
            "derived_data": {},
        },
        # New bar chart should be added
        {
            "id": "chart_bar",
            "title": "Counts by Z",
            "type": "bar",
            "intent": "distribution",
            "rank": 1,
            "data_source": "sample",
            "encodings": {
                "x": {"field": "Z", "type": "nominal"},
                "y": {"aggregate": "count", "type": "quantitative"},
            },
            "config": {},
            "derived_data": {},
        },
    ]

    merged = viz._merge_charts(primary, extras)

    # Only one scatter (the original) + one bar
    types = [c["type"] for c in merged]
    assert types.count("scatter") == 1
    assert "bar" in types

    # ranks normalized
    ranks = sorted(c["rank"] for c in merged)
    assert ranks == list(range(1, len(merged) + 1))


def test_charts_from_llm_suggestions_handles_all_types(monkeypatch):
    # Allow more than 5 charts so we hit every branch
    monkeypatch.setattr(viz, "MAX_TOTAL_CHARTS", 10)

    suggestions = [
        {
            "type": "histogram",
            "title": "Hist Age",
            "intent": "distribution",
            "x_field": "age",
            "y_field": None,
            "config": {"nbins": 40},
        },
        {
            "type": "bar",
            "title": "Sales by Region",
            "intent": "comparison",
            "x_field": "region",
            "y_field": "sales",
            "config": {"aggregate": "sum", "top_k": 5},
        },
        {
            "type": "line",
            "title": "Sales over Time",
            "intent": "trend",
            "x_field": "date",
            "y_field": "sales",
            "config": {"aggregate": "sum"},
        },
        {
            "type": "pie",
            "title": "Share by Category",
            "intent": "composition",
            "x_field": "category",
            "y_field": "sales",
            "config": {"aggregate": "sum", "top_k": 8},
        },
        {
            "type": "pie",
            "title": "Count by Category",
            "intent": "composition",
            "x_field": "category",
            "y_field": None,
            "config": {"aggregate": "count", "top_k": 5},
        },
        {
            "type": "scatter",
            "title": "Height vs Weight",
            "intent": "relationship",
            "x_field": "height",
            "y_field": "weight",
            "config": {},
        },
        {
            "type": "scatter",
            "title": "Broken scatter no y",
            "intent": "relationship",
            "x_field": "height",
            "y_field": None,  # should be skipped
            "config": {},
        },
        {
            "type": "unknown",
            "title": "Should be skipped",
            "intent": "comparison",
            "x_field": "foo",
            "y_field": "bar",
            "config": {},
        },
    ]

    viz_profile = {"dataset": {"rows_sampled": 100}}
    charts = viz._charts_from_llm_suggestions(suggestions, viz_profile)

    # All supported types appear
    chart_types = {c["type"] for c in charts}
    assert chart_types == {"histogram", "bar", "line", "pie", "scatter"}

    titles = {c["title"] for c in charts}
    assert "Broken scatter no y" not in titles
    assert "Should be skipped" not in titles

    # Check the pie handling when y_field is present vs absent
    pie_with_y = next(c for c in charts if c["title"] == "Share by Category")
    pie_no_y = next(c for c in charts if c["title"] == "Count by Category")

    assert pie_with_y["encodings"]["theta"]["field"] == "sales"
    assert pie_with_y["encodings"]["theta"]["aggregate"] == "sum"

    assert pie_no_y["encodings"]["theta"]["field"] == "category"
    assert pie_no_y["encodings"]["theta"]["aggregate"] == "count"

    # ranks normalized
    ranks = sorted(c["rank"] for c in charts)
    assert ranks == list(range(1, len(charts) + 1))


def test_build_visualization_config_uses_llm_when_available(
    monkeypatch, small_sample_rows_dict
):
    """
    Patch viz_llm with a fake model so we exercise the LLM path
    without calling the real Groq API.
    """

    class FakeResp:
        def __init__(self, content: str):
            self.content = content

    class FakeLLM:
        def __init__(self, content: str):
            self._content = content

        def invoke(self, *args, **kwargs):
            # Ignore prompts, always return our pre-baked JSON array
            return FakeResp(self._content)

    llm_suggestions = json.dumps(
        [
            {
                "type": "scatter",
                "title": "LLM Scatter",
                "intent": "relationship",
                "x_field": "subscribers",
                "y_field": "video views",
                "config": {},
            }
        ]
    )

    monkeypatch.setattr(viz, "viz_llm", FakeLLM(llm_suggestions))

    config = viz.build_visualization_config_from_sample(
        dataset_id="ds_llm",
        sample_rows=small_sample_rows_dict,
        schema=None,
        task_type="unsupervised",
        target_column=None,
    )

    # LLM was used (not fallback-only)
    reason = config.get("llm_selection_reason", "")
    assert "Groq LLM" in reason

    titles = {c["title"] for c in config["charts"]}
    assert "LLM Scatter" in titles

    # The LLM chart should still have a reason attached
    llm_chart = next(c for c in config["charts"] if c["title"] == "LLM Scatter")
    assert "reason" in llm_chart
    assert llm_chart["reason"].strip()


def test_visualization_agent_node_handles_exceptions(
    monkeypatch, small_sample_rows_list
):
    """
    Force build_visualization_config_from_sample to raise, to
    ensure the node gracefully catches errors and sets status.
    """

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(viz, "build_visualization_config_from_sample", boom)

    columns = small_sample_rows_list["columns"]
    rows = small_sample_rows_list["rows"]

    state: ETLState = {  # type: ignore[assignment]
        "dataset_id": "ds_error_case",
        "uploaded_csv_columns": columns,
        "uploaded_csv_preview": rows,
    }

    new_state = viz.visualization_agent_node(state)

    assert new_state["visualization_config"] == {}
    assert new_state["visualization_status"].startswith("error: kaboom")

