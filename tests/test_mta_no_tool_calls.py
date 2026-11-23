"""A tool-selection reply with no tool call is not an answer (defect D1).

Seen on integration/2.0 with gemma4:e4b via Ollama and "what predicts a
stroke?": the MTA's tool-pick call came back as prose, the agent appended
"Sorry, the model training agent did not receive any tool calls. Please try
again." as the user's answer, and the trace recorded train_models as ok with
no failed model call, so the metrics view could not see it.

What these tests pin down:

* an empty reply is retried once with the provider's forcing tool_choice and
  an explicit instruction, and a tool call on that retry carries on;
* two empty replies fail the step: the trace step is status=error with the
  exception class, every empty model call is outcome=failed, telemetry
  projects the step as failed, and the user gets plain words about what was
  needed -- never the internal string;
* AVALOKA_DISABLE_MODEL_FALLBACK=1 keeps the backup provider out of it and
  keeps the forced retry in; with the backup armed it is asked with the
  tool_choice IT accepts.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agents.mta_v2 import agent as mta
from app.core.model_fallback import _OpenRouterCompat
from app.trace.recorder import instrument, traced_node
from app.trace.telemetry import agent_events

PROSE = "Strokes are predicted by hypertension, age and atrial fibrillation."
INTERNAL = "did not receive any tool calls"


class RecordingFakeChatModel(GenericFakeChatModel):
    """A real chat model (callbacks fire, so the trace records each call) that
    also keeps the kwargs and messages of every invoke."""

    calls: List[Dict[str, Any]] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: D401
        self.calls.append({"messages": list(messages), **kwargs})
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _prose() -> AIMessage:
    return AIMessage(content=PROSE, usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70})


def _tool_call(name: str = "generate_training_plan") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": "call_1"}],
                     usage_metadata={"input_tokens": 60, "output_tokens": 5, "total_tokens": 65})


def _model(*replies: AIMessage, provider: str = "local", model: str = "gemma4:e4b"):
    fake = RecordingFakeChatModel(messages=iter(list(replies)))
    fake.calls = []
    return instrument(fake, agent="mta", provider=provider, model=model)


def _state(**overrides) -> Dict[str, Any]:
    base = {
        "user_id": "u1", "session_id": "s1", "trace_turn_id": "turn-d1",
        "messages": [HumanMessage(content="what predicts a stroke?")],
        "training_plan": None, "training_result": None, "training_scheduled": False,
        "enable_training": True, "skip_to_training": False, "training_task": None,
        "data_source_location": "/tmp/healthcare-dataset-stroke-data.csv",
        "dataset_size_bytes": 1024,
    }
    base.update(overrides)
    return base


def _run(llm, state=None, monkeypatch=None):
    """execute() under the trace recorder, as the graph node runs it."""
    agent = mta.ModelTrainingAgent()
    planned = {"n": 0}

    def plan(st):
        planned["n"] += 1
        out = dict(st)
        out["training_plan"] = {"model_name": "stroke_clf"}
        out["messages"] = list(st["messages"]) + [AIMessage(content="Here is a plan; confirm to train.")]
        return out

    with patch.object(mta, "llm", llm), patch.object(agent, "_generate_training_plan", side_effect=plan):
        result = traced_node("train_models", agent.execute)(state or _state())
    return result, planned["n"]


def _step(result):
    steps = [s for s in result["agent_trace"] if s.get("kind") == "agent" and s.get("node") == "train_models"]
    assert len(steps) == 1
    return steps[0]


def _last_text(result) -> str:
    return next(m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage))


REFUSAL = "I can only help with model training. For that question, please ask the analyst."


def _tool_use_failed_400(text: str = REFUSAL):
    """Groq's HTTP 400 when its parser cannot read the generation as a tool call."""
    import groq
    import httpx

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"message": "Failed to call a function. Please adjust your prompt.",
            "type": "invalid_request_error", "code": "tool_use_failed",
            "failed_generation": text}
    return groq.BadRequestError("Error code: 400", response=httpx.Response(400, request=request),
                                body=body)


class ScriptedFakeChatModel(RecordingFakeChatModel):
    """Replies in order; an exception in the script is raised for that call."""

    script: List[Any] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: D401
        self.calls.append({"messages": list(messages), **kwargs})
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


def _scripted(*items, provider: str = "groq", model: str = "openai/gpt-oss-120b"):
    fake = ScriptedFakeChatModel(messages=iter([]))
    fake.calls = []
    fake.script = list(items)
    return instrument(fake, agent="mta", provider=provider, model=model)


# --------------------------------------------------------------------------- #
# The retry
# --------------------------------------------------------------------------- #

def test_an_empty_reply_is_retried_with_the_forced_choice_and_the_run_proceeds(monkeypatch):
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    llm = _model(_prose(), _tool_call())

    result, planned = _run(llm)

    assert planned == 1, "the tool call on the retry must drive the flow"
    assert len(llm.calls) == 2
    first, retry = llm.calls
    assert first["tool_choice"] == mta.TOOL_CHOICE
    assert retry["tool_choice"] == mta.FORCED_TOOL_CHOICE
    # The retry insists in the prompt too: that is what moved gemma4 on Ollama,
    # where "required" is accepted but not enforced.
    assert isinstance(retry["messages"][-1], SystemMessage)
    assert retry["messages"][-1].content == mta.TOOL_CALL_REQUIRED_NOTE
    assert first["messages"] == retry["messages"][:-1]

    step = _step(result)
    assert step["status"] == "ok" and step["error_class"] is None
    assert [c["outcome"] for c in step["inference"]] == ["failed", "success"]
    assert step["inference"][0]["error_class"] == "NoToolCallsError"
    assert INTERNAL not in _last_text(result)
    assert not result.get("agent_errors")


# --------------------------------------------------------------------------- #
# The failure
# --------------------------------------------------------------------------- #

def test_two_empty_replies_fail_the_step_and_tell_the_user_what_was_needed(monkeypatch):
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    llm = _model(_prose(), _prose())

    result, planned = _run(llm)

    assert planned == 0 and len(llm.calls) == 2
    step = _step(result)
    assert step["status"] == "error"
    assert step["error_class"] == "NoToolCallsError"
    assert step["decision"] == {"type": "error", "value": "failed", "reason": "NoToolCallsError"}
    assert [c["outcome"] for c in step["inference"]] == ["failed", "failed"]
    assert all(c["error_class"] == "NoToolCallsError" for c in step["inference"])

    # The failure travels the way every agent failure does: on agent_errors,
    # which the chat response already returns.
    errors = result["agent_errors"]
    assert len(errors) == 1
    assert errors[0]["agent"] == "train_models"
    assert errors[0]["error_class"] == "NoToolCallsError"
    assert errors[0]["error"].startswith("NoToolCallsError: no tool call after 2 attempt(s)")

    # The user hears what was needed, in Avaloka's words.
    text = _last_text(result)
    assert "tool call" not in text.lower() and INTERNAL not in text
    assert "sorry" not in text.lower()
    assert "predict" in text.lower() and "target column" in text.lower()
    assert "analys" in text.lower()
    assert PROSE not in text  # the model's prose is not passed off as the answer

    # The turn is over: flags cleared so the graph does not loop back in.
    assert result["enable_training"] is False and result["training_task"] is None

    # ...and the metrics view sees it.
    started, completed = agent_events([step])
    assert completed["agent"] == "train_models"
    assert completed["outcome"] == "failed"
    assert completed["error_class"] == "NoToolCallsError"
    assert completed["retry_count"] == 2
    assert completed["llm_provider"] == "local"
    assert PROSE not in json.dumps([started, completed])


def test_a_pending_plan_gets_a_reply_about_the_plan(monkeypatch):
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    llm = _model(_prose(), _prose())
    state = _state(training_plan={"model_name": "stroke_clf"}, training_plan_reply_action="none",
                   messages=[HumanMessage(content="hmm, what do you think?")])

    result, _ = _run(llm, state)

    text = _last_text(result)
    assert "tool call" not in text.lower()
    assert "confirm" in text.lower() and "training plan" in text.lower()
    assert _step(result)["status"] == "error"


# --------------------------------------------------------------------------- #
# The backup provider
# --------------------------------------------------------------------------- #

class _Backup:
    """The OpenRouter backup's inner model: records kwargs, answers with a tool call."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def invoke(self, messages, config=None, **kwargs):
        self.calls.append({"messages": list(messages), **kwargs})
        return _tool_call()


def _primary_with_backup():
    primary = _model(_prose(), _prose())
    inner = _Backup()
    wrapped = primary.with_fallbacks([_OpenRouterCompat(inner, "openai/gpt-oss-120b")],
                                     exceptions_to_handle=(OSError,))
    return wrapped, primary, inner


def test_disabling_the_model_fallback_skips_the_backup_but_keeps_the_forced_retry(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    wrapped, primary, inner = _primary_with_backup()

    result, planned = _run(wrapped)

    assert planned == 0
    assert [c["tool_choice"] for c in primary.calls] == [mta.TOOL_CHOICE, mta.FORCED_TOOL_CHOICE]
    assert inner.calls == [], "the kill switch must keep the backup out of the conversation"
    assert _step(result)["status"] == "error"
    assert _step(result)["error_class"] == "NoToolCallsError"
    assert "tool call" not in _last_text(result).lower()


def test_with_the_backup_armed_it_is_asked_with_the_tool_choice_it_accepts(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("AVALOKA_DISABLE_MODEL_FALLBACK", raising=False)
    wrapped, primary, inner = _primary_with_backup()

    result, planned = _run(wrapped)

    assert planned == 1, "the backup's tool call drives the flow"
    assert [c["tool_choice"] for c in primary.calls] == [mta.TOOL_CHOICE, mta.FORCED_TOOL_CHOICE]
    assert len(inner.calls) == 1
    # Never "required" on OpenRouter: it answers finish_reason="error".
    assert inner.calls[0]["tool_choice"] == "auto"
    assert inner.calls[0]["messages"][-1].content == mta.TOOL_CALL_REQUIRED_NOTE
    step = _step(result)
    assert step["status"] == "ok"
    assert [c["outcome"] for c in step["inference"]][:2] == ["failed", "failed"]
    assert not result.get("agent_errors")


# --------------------------------------------------------------------------- #
# The recorder: a failure written to the state is a failed step
# --------------------------------------------------------------------------- #

def test_a_failure_recorded_on_the_state_is_a_failed_step_with_its_class():
    from app.agents.contract import failure_record

    def node(state):
        return {"agent_errors": list(state.get("agent_errors") or [])
                + [failure_record("train_models", ValueError("acme_payroll column missing"))]}

    before = {"trace_turn_id": "t", "agent_errors": [{"agent": "coder", "error": "KeyError: x",
                                                      "traceback": ""}]}
    out = traced_node("train_models", node)(before)
    step = out["agent_trace"][-1]
    assert step["status"] == "error" and step["error_class"] == "ValueError"
    assert step["decision"]["reason"] == "ValueError"
    assert "acme_payroll" not in json.dumps({k: v for k, v in step.items() if k != "written"})
    (_, completed) = agent_events([step])
    assert completed["outcome"] == "failed" and completed["error_class"] == "ValueError"


def test_an_earlier_nodes_error_does_not_fail_this_step():
    def node(state):
        return {"ready_to_code": True, "agent_errors": state.get("agent_errors")}

    before = {"trace_turn_id": "t", "ready_to_code": False,
              "agent_errors": [{"agent": "coder", "error": "KeyError: x", "traceback": ""}]}
    out = traced_node("plan_etl", node)(before)
    assert out["agent_trace"][-1]["status"] == "ok"


# --------------------------------------------------------------------------- #
# Every turn the agent cannot complete: a failed step, and never an internal string
# --------------------------------------------------------------------------- #
#
# Release testing on gemma4:e4b surfaced a second internal string as the answer
# ("Sorry, the model training result is not found. Please train model at the
# first."): "what predicts the test results?" reached execute_inference with no
# model trained. Every `AIMessage(content="Sorry, ...")` the agent used to
# append is covered here, one case per former site.

from unittest.mock import MagicMock

INTERNAL_STRINGS = [
    "Sorry, the model training plan is not found. Please make plan at the first.",
    "Sorry, there was an error configuring the inference service. Please try again.",
    "Sorry, there was an error stopping the inference service. Please try again.",
    "Sorry, the model training result is not found. Please train model at the first.",
    "Sorry, the model details could not be retrieved. Please try again.",
    "Sorry, there was an error during inference execution. Please try again.",
    "Sorry, the model training agent is currently unavailable. Please try again later.",
    "Sorry, something went wrong while processing your request. Please try again.",
    "Sorry, the model training agent did not receive any tool calls. Please try again.",
    # Not apologies, but the same defect: these ended as ok steps while
    # _execute_inference failed the same condition as TrainedModelMissing.
    "A trained model is required before configuring an inference service.",
    "A trained model is required before stopping an inference service.",
]


def _tool_llm(name: str):
    return _model(_tool_call(name))


def _model_detail(feature_names):
    return {"feature_names": feature_names, "target_column": "y", "model_type": "classification"}


def _case_training_plan_missing(monkeypatch):
    # A scheduled training pass that finds no plan in the state.
    return _model(), _state(training_scheduled=True, training_plan=None), "TrainingPlanMissing"


def _case_trained_model_missing(monkeypatch):
    # The live case: a predictive question routed as an inference request.
    return (_tool_llm("execute_inference"),
            _state(messages=[HumanMessage(content="what predicts the test results?")],
                   training_result=None), "TrainedModelMissing")


def _case_model_details_unavailable(monkeypatch):
    iface = MagicMock()
    iface.return_value.get_model_details.return_value = None
    monkeypatch.setattr(mta, "InferenceInterface", iface)
    return (_tool_llm("execute_inference"),
            _state(messages=[HumanMessage(content='predict {"a": 1, "b": 2}')],
                   training_result={"mlflow_run_id": "run-1"}), "ModelDetailsUnavailable")


def _case_inference_execution_failed(monkeypatch):
    iface = MagicMock()
    iface.return_value.get_model_details.return_value = _model_detail(["a", "b"])
    for method in ("predict", "predict_batch", "inference_local"):
        getattr(iface.return_value, method).side_effect = RuntimeError("model server down")
    monkeypatch.setattr(mta, "InferenceInterface", iface)
    return (_tool_llm("execute_inference"),
            _state(messages=[HumanMessage(content='predict {"a": 1, "b": 2}')],
                   training_result={"mlflow_run_id": "run-1"}), "InferenceExecutionFailed")


def _case_configure_failed(monkeypatch):
    mgr = MagicMock()
    mgr.return_value.configure_inference_service.side_effect = RuntimeError("no gateway")
    monkeypatch.setattr(mta, "InferenceServiceManager", mgr)
    return (_model(), _state(training_result={"mlflow_run_id": "run-1"},
                             configure_inference_service_scheduled=True,
                             task_schedule={"task_type": "start_inference"}),
            "InferenceServiceConfigureFailed")


def _case_stop_failed(monkeypatch):
    mgr = MagicMock()
    mgr.return_value.stop_inference_service.side_effect = RuntimeError("no gateway")
    monkeypatch.setattr(mta, "InferenceServiceManager", mgr)
    return (_model(), _state(training_result={"mlflow_run_id": "run-1"},
                             stop_inference_service_scheduled=True,
                             task_schedule={"task_type": "stop_inference"}),
            "InferenceServiceStopFailed")


def _enqueue_raises(monkeypatch, name, exc):
    from app.infra import training_tasks

    def _raise(*_a, **_k):
        raise exc(f"{name} refused")
    monkeypatch.setattr(training_tasks, name, _raise)


# The turn only enqueues (app/infra/training_tasks); these pin each way the
# enqueue itself can fail, independent of whether a broker is reachable here.
def _case_configure_no_worker(monkeypatch):
    from app.infra.training_tasks import NoWorker
    _enqueue_raises(monkeypatch, "enqueue_deploy_endpoint", NoWorker)
    return (_model(), _state(training_result={"mlflow_run_id": "run-1"},
                             configure_inference_service_scheduled=True,
                             task_schedule={"task_type": "start_inference"}),
            "InferenceServiceConfigureFailed")


def _case_configure_not_queued(monkeypatch):
    _enqueue_raises(monkeypatch, "enqueue_deploy_endpoint", RuntimeError)
    return (_model(), _state(training_result={"mlflow_run_id": "run-1"},
                             configure_inference_service_scheduled=True,
                             task_schedule={"task_type": "start_inference"}),
            "InferenceServiceConfigureFailed")


def _case_stop_no_worker(monkeypatch):
    from app.infra.training_tasks import NoWorker
    _enqueue_raises(monkeypatch, "enqueue_stop_endpoint", NoWorker)
    return (_model(), _state(training_result={"mlflow_run_id": "run-1"},
                             stop_inference_service_scheduled=True,
                             task_schedule={"task_type": "stop_inference"}),
            "InferenceServiceStopFailed")


def _case_stop_not_queued(monkeypatch):
    _enqueue_raises(monkeypatch, "enqueue_stop_endpoint", RuntimeError)
    return (_model(), _state(training_result={"mlflow_run_id": "run-1"},
                             stop_inference_service_scheduled=True,
                             task_schedule={"task_type": "stop_inference"}),
            "InferenceServiceStopFailed")


def _case_agent_unavailable(monkeypatch):
    return None, _state(), "AgentUnavailable"


def _case_generic_exception(monkeypatch):
    broken = MagicMock()
    broken.invoke.side_effect = RuntimeError("provider exploded")
    return broken, _state(), "RuntimeError"


def _case_no_tool_calls(monkeypatch):
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    return _model(_prose(), _prose()), _state(), "NoToolCallsError"


def _case_configure_without_trained_model(monkeypatch):
    # The scheduled pass reaches the service with no mlflow_run_id anywhere.
    return (_model(), _state(training_result={"status": "done"},
                             configure_inference_service_scheduled=True,
                             task_schedule={"task_type": "start_inference"}),
            "TrainedModelMissing")


def _case_stop_without_trained_model(monkeypatch):
    return (_model(), _state(training_result={"status": "done"},
                             stop_inference_service_scheduled=True,
                             task_schedule={"task_type": "stop_inference"}),
            "TrainedModelMissing")


def _case_configure_tool_without_trained_model(monkeypatch):
    # The interactive route: the model picks the tool, no model was trained.
    return (_tool_llm("configure_inference_service"),
            _state(messages=[HumanMessage(content="start an inference service")]),
            "TrainedModelMissing")


def _case_unknown_tools_only(monkeypatch):
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    return (_model(_tool_call("json"), _tool_call("json")), _state(), "NoToolCallsError")


def _case_tool_use_failed(monkeypatch):
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    return (_scripted(_tool_use_failed_400(), _tool_use_failed_400()), _state(), "NoToolCallsError")


CASES = [_case_training_plan_missing, _case_trained_model_missing, _case_model_details_unavailable,
         _case_inference_execution_failed, _case_configure_failed, _case_stop_failed,
         _case_configure_no_worker, _case_configure_not_queued,
         _case_stop_no_worker, _case_stop_not_queued,
         _case_agent_unavailable, _case_generic_exception, _case_no_tool_calls,
         _case_configure_without_trained_model, _case_stop_without_trained_model,
         _case_configure_tool_without_trained_model, _case_unknown_tools_only,
         _case_tool_use_failed]


@pytest.mark.parametrize("case", CASES, ids=[c.__name__[6:] for c in CASES])
def test_every_turn_the_agent_cannot_complete_is_a_failed_step_in_plain_words(case, monkeypatch):
    llm, state, error_class = case(monkeypatch)
    agent = mta.ModelTrainingAgent()

    with patch.object(mta, "llm", llm):
        result = traced_node("train_models", agent.execute)(state)

    text = _last_text(result)
    assert text not in INTERNAL_STRINGS
    assert not text.startswith("Sorry")
    assert "tool call" not in text.lower() and "at the first" not in text
    assert len(text) > 60, "a reply, not a label"
    # What she needed, or what to try: a target column, a trained model, an analysis.
    assert any(k in text.lower() for k in ("target column", "analyse", "try again")), text

    step = _step(result)
    assert step["status"] == "error", step
    assert step["error_class"] == error_class
    assert result["agent_errors"][-1]["agent"] == "train_models"
    assert result["agent_errors"][-1]["error_class"] == error_class
    (_, completed) = agent_events([step])
    assert completed["outcome"] == "failed" and completed["error_class"] == error_class
    # The turn is over: flags cleared so the graph does not route back in.
    assert result["enable_training"] is False and result["training_task"] is None
    assert result["task_schedule"] is None
    assert not result.get("configure_inference_service_scheduled")
    assert not result.get("stop_inference_service_scheduled")


def test_no_internal_sorry_string_survives_in_the_agent_source():
    import inspect

    source = inspect.getsource(mta)
    assert 'AIMessage(content="Sorry' not in source
    for internal in INTERNAL_STRINGS:
        assert internal not in source


# --------------------------------------------------------------------------- #
# Unusable tool-call replies: unknown names, and a provider's tool_use_failed
# --------------------------------------------------------------------------- #

def test_a_reply_naming_only_unknown_tools_is_retried_then_fails_the_step(monkeypatch):
    """It used to be skipped: step "ok", no new message -- a silent turn."""
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    llm = _model(_tool_call("json"), _tool_call("json"))
    before = len(_state()["messages"])

    result, planned = _run(llm)

    assert planned == 0 and len(llm.calls) == 2
    assert llm.calls[1]["tool_choice"] == mta.FORCED_TOOL_CHOICE
    assert llm.calls[1]["messages"][-1].content == mta.TOOL_CALL_REQUIRED_NOTE
    assert len(result["messages"]) == before + 1, "the user gets a reply"
    step = _step(result)
    assert step["status"] == "error" and step["error_class"] == "NoToolCallsError"
    assert [c["error_class"] for c in step["inference"]] == ["UnknownToolCall", "UnknownToolCall"]
    assert "json" not in result["agent_errors"][-1]["error"]


def test_a_known_tool_on_the_retry_after_an_unknown_one_carries_on(monkeypatch):
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    result, planned = _run(_model(_tool_call("json"), _tool_call()))
    assert planned == 1
    assert _step(result)["status"] == "ok"


def test_tool_use_failed_is_re_prompted_and_the_retry_carries_on(monkeypatch):
    """Groq's tool_use_failed is an unusable reply, re-prompted like 0f711844
    does for the data agent -- not an error, and not the answer."""
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    llm = _scripted(_tool_use_failed_400(), _tool_call())

    result, planned = _run(llm)

    assert planned == 1
    assert len(llm.calls) == 2
    assert llm.calls[1]["messages"][-1].content == mta.TOOL_CALL_REQUIRED_NOTE
    step = _step(result)
    assert step["status"] == "ok"
    assert step["inference"][0]["outcome"] == "failed"
    assert step["inference"][0]["error_class"] == "ToolUseFailed"
    assert all(REFUSAL not in str(m.content) for m in result["messages"])


def test_tool_use_failed_text_is_never_shown_as_the_answer(monkeypatch):
    """It used to be: failed_generation was appended as the reply, step "ok"."""
    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    llm = _scripted(_tool_use_failed_400(), _tool_use_failed_400())

    result, planned = _run(llm)

    assert planned == 0 and len(llm.calls) == 2
    text = _last_text(result)
    assert REFUSAL not in text
    step = _step(result)
    assert step["status"] == "error" and step["error_class"] == "NoToolCallsError"
    assert [c["error_class"] for c in step["inference"]] == ["ToolUseFailed", "ToolUseFailed"]
    assert REFUSAL not in json.dumps(result["agent_errors"])


def test_some_other_400_is_still_a_failed_turn_with_its_own_class(monkeypatch):
    import groq
    import httpx

    monkeypatch.setenv("AVALOKA_DISABLE_MODEL_FALLBACK", "1")
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    other = groq.BadRequestError("Error code: 400", response=httpx.Response(400, request=request),
                                 body={"code": "context_length_exceeded"})
    llm = _scripted(other)

    result, _ = _run(llm)

    assert len(llm.calls) == 1, "only tool_use_failed is re-prompted"
    assert _step(result)["error_class"] == "BadRequestError"


# --------------------------------------------------------------------------- #
# The backup loop: each backup gets the tool_choice ITS provider accepts
# --------------------------------------------------------------------------- #

from langchain_core.runnables import Runnable


class _ProviderBackup(Runnable):
    """A backup that is not OpenRouter. None exists in the product today
    (attach_fallback arms only OpenRouter); this pins what one would get."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.calls: List[Dict[str, Any]] = []

    def invoke(self, messages, config=None, **kwargs):
        self.calls.append({"messages": list(messages), **kwargs})
        return _tool_call()


@pytest.mark.parametrize("provider,expected", [("groq", "required"), ("local", "required"),
                                               ("openai", "required")])
def test_a_non_openrouter_backup_is_asked_with_required(monkeypatch, provider, expected):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")  # backup_enabled() gates fallback_models
    monkeypatch.delenv("AVALOKA_DISABLE_MODEL_FALLBACK", raising=False)
    primary = _model(_prose(), _prose())
    backup = _ProviderBackup(provider)
    wrapped = primary.with_fallbacks([backup], exceptions_to_handle=(OSError,))

    result, planned = _run(wrapped)

    assert planned == 1
    assert [c["tool_choice"] for c in backup.calls] == [expected]
    assert _step(result)["status"] == "ok"
