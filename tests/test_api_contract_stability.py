"""The evidence-layer fields must not change the existing API contract.

The UI consumes `ChatResponse`. New fields are only safe if:

* every pre-existing field keeps its name, type and default, and
* every new field is Optional with a None default, so a response that omits
  them serialises exactly as it did before.

These tests pin that, so a later change that alters an existing field fails
here rather than in the browser.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from app.api.schemas import ChatResponse  # noqa: E402

#: Fields the UI relied on before the evidence layer landed. Removing or
#: renaming any of these is a breaking change.
PRE_EXISTING = {
    "messages", "planner_definition", "ready_to_summarize", "ready_to_code",
    "coder_definition", "task_info", "output_file_data", "output_json",
    "reasoning", "planner_graph_path", "planner_graph_status",
    "planner_graph_display_url", "visualization_config", "visualization_status",
    "training_scheduled", "inference_scheduled",
    "configure_inference_service_scheduled", "stop_inference_service_scheduled",
    "training_plan", "training_result", "training_task", "training_metrics",
    "model_artifacts", "mlflow_run_id", "training_completed", "ready_to_train",
    "training_status", "dataset_size_bytes", "datasets", "active_dataset_ids",
    "visualization_configs", "visualization_statuses", "analysis_fidelity",
    "selected_sample_name", "analysis_task_id", "execution_context",
    "memory_hints", "prior_artifact_found", "session_logic_signature",
    "memory_context_unavailable",
}

NEW_FIELDS = {
    "integrity_report", "integrity_safe_to_train",
    "evaluation_report", "evaluation_beats_baseline",
    "verification_report", "verification_safe_to_present",
    "agent_errors",
}


def test_no_pre_existing_field_was_removed_or_renamed():
    present = set(ChatResponse.model_fields)
    missing = PRE_EXISTING - present
    assert not missing, f"breaking change — fields disappeared from ChatResponse: {sorted(missing)}"


def test_only_the_declared_new_fields_were_added():
    added = set(ChatResponse.model_fields) - PRE_EXISTING
    assert added == NEW_FIELDS, f"unexpected additions to the API contract: {sorted(added - NEW_FIELDS)}"


def test_every_new_field_is_optional_with_a_none_default():
    """A required new field would break every existing client."""
    for name in NEW_FIELDS:
        info = ChatResponse.model_fields[name]
        assert not info.is_required(), f"{name} is required; that breaks existing clients"
        assert info.get_default() is None, f"{name} must default to None"


def test_a_response_without_evidence_fields_still_constructs():
    """Exactly what an older code path produces."""
    r = ChatResponse(messages=[{"role": "user", "content": "hi"}])
    assert r.integrity_report is None
    assert r.evaluation_beats_baseline is None


def test_new_fields_are_omitted_when_unset():
    """exclude_none keeps the wire payload byte-identical to before."""
    payload = ChatResponse(messages=[]).model_dump(exclude_none=True)
    assert not (NEW_FIELDS & set(payload)), \
        "unset evidence fields must not appear in the serialised response"


def test_evidence_fields_round_trip_when_populated():
    r = ChatResponse(
        messages=[],
        integrity_report={"safe_to_train": False, "n_blockers": 1},
        integrity_safe_to_train=False,
        evaluation_report={"primary_metric": "accuracy"},
        evaluation_beats_baseline=True,
        agent_errors=[{"agent": "evaluation", "error": "boom"}],
    )
    payload = r.model_dump(exclude_none=True)
    assert payload["integrity_safe_to_train"] is False
    assert payload["evaluation_beats_baseline"] is True
    assert payload["agent_errors"][0]["agent"] == "evaluation"


def test_reports_are_json_serialisable():
    """The UI receives these over HTTP; non-JSON types would 500 at response time."""
    import json

    import numpy as np
    import pandas as pd

    from app.agents.integrity_agent import run_integrity_checks

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=120), "y": rng.integers(0, 2, 120)})
    df["leak"] = df["y"]
    report = run_integrity_checks(df, target="y").as_dict()
    json.dumps(report)  # must not raise


def test_evaluation_report_is_json_serialisable():
    import json

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from app.agents.evaluation_agent import evaluate_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)
    payload = evaluate_model(LogisticRegression(max_iter=400), X, y,
                             task="classification").as_dict()
    json.dumps(payload)  # must not raise
