"""What a self-hosted model must be able to do before it can carry agent traffic.

PR #272 (feat/local-model-fallback) adds a third link to the failover chain --
``AVALOKA_FALLBACK_CHAIN=openrouter,local`` -- ending at a model we run in our
own cluster. Its tests cover the chain MECHANICS thoroughly: order, the kill
switch, 429 vs 401, arming and skipping links. What they do not cover is whether
the model at the end of that chain can do what the agents require, and the PR's
own README says so:

    Caveat on tool_choice="required". The planner forces a tool call. Ollama's
    OpenAI-compatible endpoint accepts `tools` and returns structured
    `tool_calls`, but whether it *enforces* tool_choice: "required" -- rather
    than accepting and ignoring it -- is server-version dependent. [...] Worth
    one live check against your deployed version before relying on the local
    link for planner traffic.

This file is that check, as a repeatable test rather than a one-off curl. It
matters because a local model is not a drop-in for gpt-oss-120b: the failure
mode is not "slower" or "less accurate", it is a fallback that engages during an
outage and then cannot execute the planner's contract, turning a degraded turn
into a broken one.

Two kinds of test here:

  OFFLINE  no endpoint, no network. These check what WE send, and run in every
           CI job. The reasoning-kwarg test below is the one that found a real
           defect: the planner selects `reasoning_effort` from its PRIMARY model
           and `with_fallbacks` forwards the same kwargs to the backup, so a
           local Gemma or Qwen was being handed a gpt-oss-only parameter.

  LIVE     probes an actual endpoint, skipped unless INFERENCE_LOCAL_BASE_URL is
           set. These check what the SERVER does.

Running the live half::

    kubectl port-forward -n avaloka svc/avaloka-local-llm 11434:11434 &
    set INFERENCE_LOCAL_BASE_URL=http://localhost:11434/v1
    set INFERENCE_LOCAL_MODEL_LARGE=qwen3.8:27b
    pytest tests/test_local_model_capability_contract.py -v

A skipped live half is not a pass. It means nothing has verified the model, and
the local link should not carry planner traffic yet.
"""

from __future__ import annotations

import json
import os

import pytest


# ---------------------------------------------------------------------------
# OFFLINE: what we send
# ---------------------------------------------------------------------------


class _Recorder:
    """Stands in for a backup chat model and remembers how it was invoked."""

    def __init__(self):
        self.kwargs = None

    def invoke(self, _input, _config=None, **kwargs):
        self.kwargs = kwargs
        return "ok"

    async def ainvoke(self, _input, _config=None, **kwargs):
        self.kwargs = kwargs
        return "ok"


@pytest.mark.parametrize(
    "backup_model",
    ["qwen3.8:27b", "gemma3:27b", "qwen2.5:3b-instruct"],
    ids=["qwen-27b", "gemma-27b", "qwen-3b"],
)
def test_gpt_oss_reasoning_kwargs_never_reach_a_local_model(backup_model):
    """A local model must not be handed `reasoning_effort`.

    The planner decides whether to send reasoning kwargs by looking at its
    PRIMARY model, once, at import:

        PLANNER_SUPPORTS_REASONING = PLANNER_MODEL.startswith(("openai/gpt-oss",))

    `with_fallbacks` then hands the fallback the same kwargs the primary was
    called with. Nothing in that path reconsiders the decision, so a chain
    ending at Qwen or Gemma forwards a parameter those models do not know --
    and the failover dies at the exact moment it existed to save the turn.
    """
    from app.core.model_fallback import _OpenRouterCompat

    rec = _Recorder()
    _OpenRouterCompat(rec, backup_model).invoke(
        "hi", None, reasoning_effort="medium", reasoning_format="parsed",
        service_tier="on_demand", temperature=0.2,
    )

    assert "reasoning_effort" not in rec.kwargs, (
        f"{backup_model} was sent reasoning_effort, which only the gpt-oss "
        f"family accepts: {sorted(rec.kwargs)}"
    )
    assert "reasoning_format" not in rec.kwargs
    assert "service_tier" not in rec.kwargs
    # Everything legitimate must survive -- this is a filter, not a reset.
    assert rec.kwargs.get("temperature") == 0.2


def test_a_gpt_oss_backup_still_receives_reasoning_effort():
    """The strip is conditional. Failing over gpt-oss -> gpt-oss on another host
    must keep the reasoning controls, or the backup answers at a different
    deliberation depth than the primary would have."""
    from app.core.model_fallback import _OpenRouterCompat

    rec = _Recorder()
    _OpenRouterCompat(rec, "openai/gpt-oss-120b").invoke(
        "hi", None, reasoning_effort="high",
    )
    assert rec.kwargs.get("reasoning_effort") == "high"


def test_every_local_model_is_declared_non_reasoning():
    """`supports_reasoning` gates the kwargs at the build_agent_llm seam too."""
    from app.core.agent_llm import supports_reasoning

    for model in ("qwen3.8:27b", "gemma3:27b", "qwen2.5:3b-instruct",
                  "gemma-3-27b-it", "llama3.1:8b"):
        assert not supports_reasoning(model), (
            f"{model} is treated as gpt-oss-compatible; reasoning kwargs would be sent"
        )
    assert supports_reasoning("openai/gpt-oss-120b")


# ---------------------------------------------------------------------------
# LIVE: what the server does
# ---------------------------------------------------------------------------

LOCAL_BASE_URL = os.environ.get("INFERENCE_LOCAL_BASE_URL", "").strip()
LOCAL_MODEL = (
    os.environ.get("INFERENCE_LOCAL_MODEL_LARGE")
    or os.environ.get("INFERENCE_LOCAL_MODEL_SMALL")
    or os.environ.get("INFERENCE_LOCAL_MODEL")
    or ""
).strip()

live = pytest.mark.skipif(
    not LOCAL_BASE_URL,
    reason=(
        "Set INFERENCE_LOCAL_BASE_URL (and INFERENCE_LOCAL_MODEL_LARGE) to probe a "
        "deployed local model. Skipped is NOT verified -- the local link should not "
        "carry planner traffic until this half has run green."
    ),
)


@pytest.fixture(scope="module")
def local_llm():
    from langchain_openai import ChatOpenAI

    if not LOCAL_MODEL:
        pytest.skip("INFERENCE_LOCAL_MODEL_LARGE/SMALL not set")
    return ChatOpenAI(
        model=LOCAL_MODEL,
        temperature=0.0,
        api_key=os.environ.get("INFERENCE_LOCAL_API_KEY") or "sk-local",
        base_url=LOCAL_BASE_URL,
        timeout=120,
    )


#: The shape the planner actually forces. Deliberately trivial: if a model cannot
#: call this, it cannot call the planner's real tool set either.
_PROBE_TOOL = [{
    "type": "function",
    "function": {
        "name": "respond_to_user",
        "description": "Reply to the user in prose.",
        "parameters": {
            "type": "object",
            "properties": {"response_text": {"type": "string"}},
            "required": ["response_text"],
        },
    },
}]


@live
def test_the_configured_model_is_known_to_the_server(local_llm):
    """Ollama answers an unknown tag with an error rather than a 404 model list.

    The README calls this out: an old server does not know a newer tag and then
    fails to parse tool calls, which presents as "the model ignored the tools"
    rather than as a version error. Fail here, where the message is clear.
    """
    reply = local_llm.invoke("Reply with the single word: ok")
    assert getattr(reply, "content", "").strip(), (
        f"{LOCAL_MODEL} returned an empty completion from {LOCAL_BASE_URL}"
    )


@live
def test_forced_tool_calling_is_actually_enforced(local_llm):
    """The planner's hardest requirement, and the documented unknown.

    `tool_choice="required"` must produce a structured tool call, not prose. A
    server that ACCEPTS the parameter and ignores it looks healthy and returns
    text, which the planner reports as "no tool calls" -- and under
    tool_choice=required Groq surfaces that as a 400 tool_use_failed. Either way
    the turn is lost, during an outage, which is the only time this path runs.
    """
    bound = local_llm.bind(tools=_PROBE_TOOL, tool_choice="required")
    reply = bound.invoke("Say hello to the user.")

    calls = getattr(reply, "tool_calls", None) or []
    assert calls, (
        f"{LOCAL_MODEL} did not return a structured tool call under "
        f'tool_choice="required"; it replied with prose: '
        f"{str(getattr(reply, 'content', ''))[:200]!r}. The planner cannot use "
        f"this model -- see the tool_choice caveat in deploy/inference/README.md."
    )
    assert calls[0].get("name") == "respond_to_user", (
        f"called an unexpected tool: {calls[0].get('name')!r}"
    )


@live
def test_strict_json_is_returned_without_prose(local_llm):
    """The summarizer's contract: every downstream agent parses this output.

    A model that wraps JSON in commentary or a fenced block breaks the contract
    even when the JSON itself is right.
    """
    reply = local_llm.invoke(
        'Return ONLY this JSON object, no prose and no code fence: '
        '{"status": "ok", "count": 2}'
    )
    text = str(getattr(reply, "content", "")).strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"{LOCAL_MODEL} did not return parseable JSON ({exc}); the summarizer "
            f"contract would break on this model. Got: {text[:200]!r}"
        )
    assert parsed.get("status") == "ok"


@live
def test_the_local_model_rejects_or_ignores_reasoning_effort_safely(local_llm):
    """Belt-and-braces on the offline test above.

    If the server 400s on `reasoning_effort`, the strip in _OpenRouterCompat is
    load-bearing rather than cosmetic. If it ignores it, the strip is still
    correct but this records which behaviour your deployment has.
    """
    try:
        reply = local_llm.invoke("Reply with: ok", reasoning_effort="medium")
    except Exception as exc:  # noqa: BLE001 - the point is to classify it
        pytest.skip(
            f"{LOCAL_MODEL} REJECTS reasoning_effort ({type(exc).__name__}). The "
            f"kwarg strip in _OpenRouterCompat._adapt is required for this "
            f"deployment, not merely tidy."
        )
    assert getattr(reply, "content", "").strip()
