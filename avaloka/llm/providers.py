"""Which model provider the CLI talks to, and what each one accepts.

The CLI is standalone (S1.08: no ``app.*`` import; S1.09: eight runtime
dependencies), so it cannot route through ``app/core/inference.py``. This is
its own small table, kept deliberately close to that module's decisions:

* ``local``       Ollama or any OpenAI-compatible server you run, e.g.
                  ``gemma4:e4b`` at ``http://127.0.0.1:11434/v1``.
* ``groq``        Groq Cloud, ``openai/gpt-oss-120b``.
* ``openrouter``  OpenRouter.
* ``openai``      OpenAI, or any OpenAI-compatible endpoint you point it at.
* ``anthropic``   Anthropic through its native SDK (the ``llm`` extra). Never
                  through Anthropic's OpenAI-compatible layer, which Anthropic
                  does not support for production use.

The first four share one transport (:mod:`avaloka.llm.openai_compat`, stdlib
``urllib`` only). Selection, highest precedence first:

1. ``--llm-provider`` on the command line,
2. ``AVALOKA_LLM_PROVIDER``,
3. otherwise the first provider that is configured, in :data:`AUTO_ORDER`:
   local (if an Ollama base URL or a local model is set), then Groq,
   OpenRouter, OpenAI and Anthropic.

A provider that is named but cannot be used is an error the caller prints;
it is never a silent fallback to another provider or to canned text.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

LOCAL = "local"
GROQ = "groq"
OPENROUTER = "openrouter"
OPENAI = "openai"
ANTHROPIC = "anthropic"

PROVIDERS = (LOCAL, GROQ, OPENROUTER, OPENAI, ANTHROPIC)

#: Local first (the fast default, and the data never leaves the machine),
#: then Groq, per the test-bench provider policy. Anthropic last: it is a
#: provider a user chooses, not a default anything should land on.
AUTO_ORDER = (LOCAL, GROQ, OPENROUTER, OPENAI, ANTHROPIC)

ALIASES = {
    "ollama": LOCAL, "vllm": LOCAL, "gemma": LOCAL,
    "groqcloud": GROQ, "groq-cloud": GROQ,
    "open-router": OPENROUTER, "router": OPENROUTER,
    "openai-compatible": OPENAI,
    "claude": ANTHROPIC,
}

#: Environment variables that say "a local model server is configured".
LOCAL_SIGNALS = ("OLLAMA_HOST", "INFERENCE_LOCAL_BASE_URL", "AVALOKA_LOCAL_MODEL")


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    key_env: Optional[str]          # None: no key needed (local)
    default_base_url: str
    default_model: str
    transport: str                  # "openai" | "anthropic"


SPECS: dict[str, ProviderSpec] = {
    LOCAL: ProviderSpec(LOCAL, "local model server", None,
                        "http://127.0.0.1:11434/v1", "gemma4:e4b", "openai"),
    GROQ: ProviderSpec(GROQ, "Groq", "GROQ_API_KEY",
                       "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "openai"),
    OPENROUTER: ProviderSpec(OPENROUTER, "OpenRouter", "OPENROUTER_API_KEY",
                             "https://openrouter.ai/api/v1", "openai/gpt-oss-120b", "openai"),
    OPENAI: ProviderSpec(OPENAI, "OpenAI", "OPENAI_API_KEY",
                         "https://api.openai.com/v1", "gpt-4o-mini", "openai"),
    # The current small Claude tier. The legacy Haiku 4.5 id this client used
    # to hard-code is not a default anything should be built on.
    ANTHROPIC: ProviderSpec(ANTHROPIC, "Anthropic", "ANTHROPIC_API_KEY",
                            "https://api.anthropic.com", "claude-haiku-5-5", "anthropic"),
}


@dataclass(frozen=True)
class Capabilities:
    """What a provider AND model accept. Measured asymmetries, not guesses:

    * tool_choice ``"required"`` is rejected by OpenRouter (finish_reason
      "error") and needed by Ollama (``"auto"`` yields prose, no tool call);
      Groq accepts and enforces it. Anthropic spells forcing
      ``{"type": "any"}``, which Haiku accepts and Opus/Sonnet 5.5 reject.
    * ``gemma3:*`` reports no tool support and answers HTTP 400 to any request
      that carries tools; ``gemma4:e4b`` has them.
    * Claude 4.7 and later reject a non-default temperature on both endpoints.
    """

    tools: bool
    forced_tool_choice: Any          # None: no accepted forcing value
    temperature: bool
    reasoning_effort: Optional[str]  # sent only to gpt-oss models


_CLAUDE_VERSION = re.compile(r"claude-(?:opus|sonnet|haiku)-(\d+)-(\d+)")


def capabilities(provider: str, model: str) -> Capabilities:
    m = (model or "").lower()
    tools = not m.startswith("gemma3")
    temperature = True
    forced: Any = "required"
    if provider == OPENROUTER:
        forced = None
    elif provider == ANTHROPIC:
        forced = {"type": "any"} if "haiku" in m else None
        v = _CLAUDE_VERSION.search(m)
        if v and (int(v.group(1)), int(v.group(2))) >= (4, 7):
            temperature = False
    reasoning = "low" if "gpt-oss" in m and provider in (GROQ, OPENROUTER) else None
    return Capabilities(tools=tools, forced_tool_choice=forced,
                        temperature=temperature, reasoning_effort=reasoning)


def tool_choice_for(provider: str, model: str, *, require_tool: bool) -> Any:
    """The tool_choice value to send. Keyed on the provider, never on a client class."""
    caps = capabilities(provider, model)
    auto: Any = {"type": "auto"} if provider == ANTHROPIC else "auto"
    if require_tool and caps.forced_tool_choice is not None:
        return caps.forced_tool_choice
    return auto


@dataclass(frozen=True)
class Selection:
    provider: str
    model: str
    base_url: str
    api_key: Optional[str] = field(default=None, repr=False)  # never in a repr or a log
    how: str = "auto"                # "flag" | "env" | "auto"

    @property
    def spec(self) -> ProviderSpec:
        return SPECS[self.provider]


class LLMConfigError(RuntimeError):
    """A provider was asked for and cannot be used. One line, with the fix."""

    def __init__(self, provider: str, problem: str, fix: str) -> None:
        super().__init__(f"{problem} Fix: {fix}")
        self.provider = provider
        self.problem = problem
        self.fix = fix


def canonical(name: str) -> str:
    n = (name or "").strip().lower()
    return ALIASES.get(n, n)


def _get(env: Mapping[str, str], *names: str) -> str:
    for n in names:
        v = (env.get(n) or "").strip()
        if v:
            return v
    return ""


def _ollama_base(host: str) -> str:
    host = host.rstrip("/")
    if "://" not in host:            # Ollama's own form is host:port
        host = "http://" + host
    return host if host.endswith("/v1") else host + "/v1"


def _local_configured(env: Mapping[str, str]) -> bool:
    return bool(_get(env, *LOCAL_SIGNALS))


def _configured(provider: str, env: Mapping[str, str]) -> bool:
    if provider == LOCAL:
        return _local_configured(env)
    return bool(_get(env, SPECS[provider].key_env or ""))


def _other_keys(env: Mapping[str, str], but: str) -> list[str]:
    return [p for p in AUTO_ORDER if p != but and _configured(p, env)]


def resolve(provider: Optional[str] = None, model: Optional[str] = None,
            env: Optional[Mapping[str, str]] = None) -> Selection:
    """The provider, model, base URL and key to use, or :class:`LLMConfigError`."""
    env = os.environ if env is None else env
    how = "flag" if provider else ("env" if _get(env, "AVALOKA_LLM_PROVIDER") else "auto")
    name = canonical(provider or _get(env, "AVALOKA_LLM_PROVIDER"))

    if not name:
        name = next((p for p in AUTO_ORDER if _configured(p, env)), "")
        if not name:
            raise LLMConfigError(
                "none", "--llm: no model provider is configured.",
                "set GROQ_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY, "
                "or OLLAMA_HOST for a local model, or pass --llm-provider local.")
    if name not in SPECS:
        raise LLMConfigError(
            name, f"unknown LLM provider {name!r}.",
            f"use --llm-provider with one of {', '.join(PROVIDERS)}.")

    spec = SPECS[name]
    key: Optional[str] = None
    if spec.key_env:
        key = _get(env, spec.key_env)
        if not key:
            others = _other_keys(env, name)
            hint = (f" ({', '.join(others)} is configured: --llm-provider {others[0]})"
                    if others else "")
            raise LLMConfigError(
                name, f"{spec.label} was selected but {spec.key_env} is not set.",
                f"export {spec.key_env}, or choose another provider{hint}.")
    else:
        key = _get(env, "INFERENCE_LOCAL_API_KEY") or None

    if name == LOCAL:
        base = (_get(env, "AVALOKA_LLM_BASE_URL", "INFERENCE_LOCAL_BASE_URL")
                or (_ollama_base(_get(env, "OLLAMA_HOST")) if _get(env, "OLLAMA_HOST") else "")
                or spec.default_base_url)
        chosen = model or _get(env, "AVALOKA_LLM_MODEL", "AVALOKA_LOCAL_MODEL") or spec.default_model
    elif name == OPENAI:
        base = _get(env, "AVALOKA_LLM_BASE_URL", "OPENAI_BASE_URL") or spec.default_base_url
        chosen = model or _get(env, "AVALOKA_LLM_MODEL") or spec.default_model
    elif name == ANTHROPIC:
        base = _get(env, "ANTHROPIC_BASE_URL") or spec.default_base_url
        chosen = model or _get(env, "AVALOKA_LLM_MODEL") or spec.default_model
    else:
        base = _get(env, "AVALOKA_LLM_BASE_URL") or spec.default_base_url
        chosen = model or _get(env, "AVALOKA_LLM_MODEL") or spec.default_model
    return Selection(provider=name, model=chosen, base_url=base.rstrip("/"), api_key=key, how=how)
