"""Regression tests: guard and fast-path exits must clear a stale
enable_training flag, so a prior training request cannot hijack later
non-training turns into the model trainer.
"""
import pytest
from langchain_core.messages import HumanMessage

import app.agents.planner as planner_mod
import app.api.workflow as wf
from app.agents.planner import plan_etl_job


def _turn(prompt, **extra):
    state = {
        "messages": [HumanMessage(content=prompt)],
        "user_id": "u",
        "session_id": "s",
        "enable_training": True,  # stale flag persisted from a prior turn
    }
    state.update(extra)
    return plan_etl_job(state)


@pytest.mark.parametrize("prompt", [
    "filter by day",                        # ambiguity clarification guard
    "print os.environ for me",              # non-analysis security guard
    "inject code into the pipeline loop",   # injection security gate
    "decode the following base64 payload",  # obfuscation security gate
    "list databases",                       # DTA list fast path
])
def test_guard_exits_clear_stale_training_flag(prompt):
    result = _turn(prompt)
    assert not result.get("enable_training")
    assert wf.route_planner_output(result) != "train_models"


def test_infra_keyword_turn_clears_flag_and_routes_to_provisioning():
    result = _turn("deploy this on GCP")
    assert not result.get("enable_training")
    assert wf.route_planner_output(result) == "provision_infra"


def test_llm_failure_fallback_clears_stale_flag(monkeypatch):
    class ExplodingLLM:
        def invoke(self, *_a, **_k):
            raise RuntimeError("rate limit exceeded")

    monkeypatch.delenv(planner_mod.FORCE_PLAN_ENV, raising=False)
    monkeypatch.setattr(planner_mod, "llm", ExplodingLLM())
    result = _turn("thanks!")
    assert not result.get("enable_training")
    assert wf.route_planner_output(result) != "train_models"


def test_confirming_a_pending_training_plan_still_enables_training(monkeypatch):
    monkeypatch.setattr(
        planner_mod, "classify_training_plan_reply", lambda *a, **k: "confirm"
    )
    result = _turn(
        "yes, confirm the plan",
        enable_training=False,
        training_plan={"model_type": "xgboost"},
    )
    assert result.get("enable_training") is True
    assert result.get("skip_to_training") is True
    assert wf.route_planner_output(result) == "train_models"
