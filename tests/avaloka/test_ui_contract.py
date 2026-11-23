"""Contract tests against the legacy 1.5.2 UI's execution-context shape."""

import pytest

from app.api.ui_contract import (api_capabilities, build_execution_context,
                                 build_execution_context_from_state)


def _legacy_projection(context):
    """Fields declared by ui/src/services/backendApi.ts on develop-1.5.2."""
    return {
        "mode": context["mode"],
        "mode_label": context["mode_label"],
        "sample": context.get("sample"),
    }


def test_plain_context_is_exactly_the_1_6_shape():
    # develop-1.6's shape, which the shipped UI consumes: the 1.5.2 fields
    # plus the row-count badge fields. No swarm keys on a plain turn.
    context = build_execution_context("portfolio_samples", "stratified")
    assert context == {
        "mode": "portfolio_samples",
        "mode_label": "Portfolio Samples",
        "sample": "stratified",
        "rows_analyzed": None,
        "total_rows": None,
        "is_sampled": False,
    }
    assert _legacy_projection(context) == {
        "mode": "portfolio_samples",
        "mode_label": "Portfolio Samples",
        "sample": "stratified",
    }


def test_optional_arguments_are_keyword_only():
    # 1.6 passed row counts third, 1.7 passed swarm third; positional calls
    # would land in the wrong slot, so the builder refuses them.
    with pytest.raises(TypeError):
        build_execution_context("quick_sample", None, 10, 100)


def test_swarm_context_retains_all_legacy_fields_as_additive_metadata():
    swarm = [{
        "clone": "scout", "name": "Ava-Scout", "role": "Data scout",
        "focus": "inspect schema", "status": "pending",
    }]
    context = build_execution_context(
        "entire_dataset", None, swarm=swarm, swarm_message="Dispatching"
    )

    assert _legacy_projection(context) == {
        "mode": "entire_dataset",
        "mode_label": "Entire Dataset",
        "sample": None,
    }
    assert context["swarm"] == swarm
    assert context["swarm_message"] == "Dispatching"


def test_scheduled_result_uses_state_then_session_fallback():
    context = build_execution_context_from_state(
        {
            "analysis_fidelity": "entire_dataset",
            "avaloka_swarm": [{"clone": "runner"}],
        },
        session={
            "analysis_fidelity": "quick_sample",
            "selected_sample_name": "random_baseline",
        },
    )
    assert context["mode"] == "entire_dataset"
    assert context["sample"] == "random_baseline"
    assert context["swarm"] == [{"clone": "runner"}]


def test_capability_document_marks_swarm_optional_and_nonrouting():
    capabilities = api_capabilities()
    swarm = capabilities["features"]["swarm_execution_context"]
    assert capabilities["api_contract"] == "1.5.2-compatible"
    assert swarm["optional"] is True
    assert swarm["routing_semantics"] is False
    assert capabilities["features"]["async_execution_context"] is True
    complex_data = capabilities["features"]["complex_data"]
    assert complex_data["recursive_schema"] is True
    assert "pdf" in complex_data["upload_formats"]
    assert complex_data["sdk_directory_formats"] == ["lerobot", "rlds"]
    assert "rlds" in complex_data["runtime"]


def test_capability_document_closes_the_swarm_console_unless_told_otherwise():
    """The console block is additive and defaults to closed: a client that
    did not say its edition is not told it has a page it cannot open."""
    closed = api_capabilities()["features"]["swarm_console"]
    assert closed["enabled"] is False and closed["reason"] == "commercial" and closed["page"] == "/swarm"
    opened = api_capabilities(swarm_console={"enabled": True, "reason": "open", "page": "/swarm",
                                             "message": "x"})["features"]["swarm_console"]
    assert opened["enabled"] is True and opened["reason"] == "open"
    # The swarm itself is reported independently of the console.
    assert "swarm" in api_capabilities()["features"]
