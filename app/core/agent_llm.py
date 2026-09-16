"""
Central factory for agent LLMs with optional hybrid-reasoning support.

Every subgraph agent (MTA, DTA coder/validator, ...) builds its chat model
through build_agent_llm() so the reasoning wiring lives in exactly one place.
As of the custom-inference-stack work this is a thin wrapper over
``app.core.inference.build_chat_model``: it keeps this module's historical
signature and Groq semantics while gaining provider-agnostic backends (local
OpenAI-spec model in k8s, Groq, OpenRouter, Bedrock, Vertex, Azure) selected
via ``INFERENCE_PROVIDER``. See app/core/inference.py for the full contract.

- Per-agent model override via env (AVALOKA_<AGENT>_MODEL, or a legacy
  env var name like DTA_CODER_MODEL passed via env_model_var). On the Groq
  backend a non-reasoning model (e.g. llama-3.3-70b-versatile) emits a plain
  ChatGroq with no reasoning kwargs — byte-identical to the pre-reasoning
  behavior, no code change needed to revert.
- reasoning_effort defaults per agent (AVALOKA_<AGENT>_REASONING_EFFORT
  overrides). gpt-oss/Groq only.
- reasoning_format defaults to "hidden" for subgraph agents: nothing consumes
  their traces yet. Override with AVALOKA_AGENT_REASONING_FORMAT.

The planner (app/agents/planner.py) builds its model directly through
build_chat_model() (it surfaces its trace with "parsed" and adaptive
per-invoke effort), so it does not use this wrapper.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

# Re-exported for backwards compatibility with any importer of these names.
from app.core.inference import (  # noqa: F401
    REASONING_MODEL_PREFIXES,
    build_chat_model,
    supports_reasoning,
)

from app.core.model_fallback import attach_fallback

logger = logging.getLogger(__name__)


def build_agent_llm(
    *,
    agent: str,
    api_key: Optional[str],
    default_model: str = "openai/gpt-oss-120b",
    temperature: float = 0.2,
    default_effort: str = "low",
    env_model_var: Optional[str] = None,
    role: str = "planning",
    tier: str = "large",
):
    """
    Build an agent's chat model honoring env overrides and reasoning support.

    agent:          short uppercase id used in env-var names and logs
                    (e.g. "MTA", "MTA_CLASSIFIER").
    api_key:        the agent's Groq key; used only when the Groq backend is
                    active. On Groq, None disables the LLM (returns None,
                    matching each agent's existing "disabled" path). On other
                    backends the relevant provider credentials are used instead.
    default_model:  Groq model when no env override is set.
    temperature:    sampling temperature.
    default_effort: reasoning effort when the model supports reasoning and
                    AVALOKA_<AGENT>_REASONING_EFFORT is unset (Groq/gpt-oss).
    env_model_var:  legacy Groq env var name to keep honoring (e.g.
                    DTA_CODER_MODEL); defaults to AVALOKA_<AGENT>_MODEL.
    role:           coarse role for per-role provider selection
                    ("planning" | "coding" | "viz").
    tier:           "large" | "small" — model tier for non-Groq providers.
    """
    return build_chat_model(
        role=role,
        agent=agent,
        tier=tier,
        temperature=temperature,
        groq_model=default_model,
        groq_api_key=api_key,
        env_model_var=env_model_var,
        reasoning_effort=os.getenv(f"AVALOKA_{agent}_REASONING_EFFORT", default_effort),
        reasoning_format=os.getenv("AVALOKA_AGENT_REASONING_FORMAT", "hidden"),
    )
