"""Direct LLM adapter: one prompt, any OpenAI-compatible provider.

Built for the model-matrix track: the same task, the same deterministic
scoring, only the model varies. Providers are addressed by URL, so Groq,
OpenRouter, a local Ollama, and an in-cluster vLLM are all the same code path
— which is the point, since the harness is supposed to separate model effects
from everything else.

Uses urllib rather than langchain so the adapter has zero heavy dependencies
and its latency numbers measure the provider, not a client stack.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict

from benchmarks.adapters import register_adapter
from benchmarks.adapters.base import SystemUnderTest
from benchmarks.core.config import RunConfig
from benchmarks.core.pricing import SOURCE_USAGE_MISSING, price_for
from benchmarks.core.task import BenchmarkTask, MissionOutcome

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions",
             "key_env": ("GROQ_API_KEY", "GROQ_API_KEY_PLANNING_AGENT")},
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions",
                   "key_env": ("OPENROUTER_API_KEY",)},
    "ollama": {"url": os.getenv("OLLAMA_URL", "http://localhost:11434") + "/v1/chat/completions",
               "key_env": ()},   # local; no key
    # Google's OpenAI-compatible endpoint. Model ids there carry no vendor
    # prefix ("gemini-4-argon"); the OpenRouter catalog quotes the same model
    # as "google/gemini-4-argon", so a config sets extra["price_model"] to that
    # id and the quote is labelled openrouter_proxy like Groq's.
    "gemini": {"url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
               "key_env": ("GEMINI_API_KEY",)},
}

SYSTEM_PROMPT = (
    "You are Avaloka, a data-analysis assistant. You are given the computed "
    "evidence for the user's dataset as JSON. Answer the user's message based "
    "ONLY on this evidence. If the evidence contradicts the user, say so "
    "plainly. If the user asks for a number the evidence does not contain, say "
    "it is not in the data rather than estimating. If the request is ambiguous "
    "between materially different readings, ask which one they mean.\n\n"
    "EVIDENCE:\n{evidence}"
)


def require_provider(config: RunConfig) -> str:
    """The provider for a config — REQUIRED, never defaulted.

    ``provider`` decides which ACCOUNT IS BILLED. A default therefore silently
    routes a config to whichever vendor happens to be first in the dict: QA
    proved that a config carrying ``anthropic/claude-opus-5`` with an empty
    ``extra`` posted that model id to ``api.groq.com`` with the real Groq key.
    Defaulting a *model* is a wrong answer; defaulting a *payer* is a wrong
    invoice, so this raises instead.

    Raises :class:`ValueError` so suites can validate at build time, alongside
    ``pricing.validate_models``, rather than discovering it at the first call.
    """
    provider = str((config.extra or {}).get("provider") or "").strip()
    if not provider:
        raise ValueError(
            f"config {config.id!r}: extra['provider'] is required and must be one "
            f"of {sorted(PROVIDERS)}. It selects which account is billed, so it is "
            f"never inferred.")
    if provider not in PROVIDERS:
        raise ValueError(f"config {config.id!r}: unknown provider {provider!r}; "
                         f"known: {sorted(PROVIDERS)}")
    return provider


def _answer_text(content: Any) -> str:
    """Normalise an OpenAI-spec ``message.content`` to text.

    Some gateways return a list of typed parts rather than a string. Passing
    that list through scores 0.00 against a track evaluator that requires
    ``str`` — a correct answer recorded as a wrong one, with no error tag.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content
                 if isinstance(p, dict) and p.get("type") in (None, "text")]
        return "".join(parts)
    return "" if content is None else str(content)


def _key_for(provider: str) -> str:
    for env in PROVIDERS[provider]["key_env"]:
        value = (os.getenv(env) or "").strip()
        if value:
            return value
    return ""


@register_adapter("llm_direct")
class LLMDirect(SystemUnderTest):
    """config.extra: provider (groq|openrouter|ollama), model, max_tokens.

    Reports real spend. The provider returns a ``usage`` block with the
    prompt/completion split; multiplying it by the catalog price
    (:mod:`benchmarks.core.pricing`) is the only way this harness can put a
    number in the Cost column without inventing one. When no price exists the
    cost is 0.0 and ``cost_source`` says ``unpriced`` — a labelled hole, not a
    claim that the call was free.
    """

    key = "llm_direct"

    #: Hosted providers spend real money, so the local-first gate in
    #: ``benchmarks.run`` must hold them back behind ``--allow-cloud``. Ollama is
    #: local and must not be, which is why this is per-config rather than the
    #: flat ``requires_cloud`` class attribute the other adapters use.
    requires_cloud = True

    @staticmethod
    def config_requires_cloud(config: RunConfig) -> bool:
        # An absent provider is treated as cloud, never as the free local one:
        # the safe direction for a gate whose job is holding back paid calls.
        return str(config.extra.get("provider") or "") not in ("ollama",)

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        # ``provider`` decides which ACCOUNT IS BILLED, so it is required rather
        # than defaulted. Defaulting it to groq meant a config authored without
        # one sent its model id to api.groq.com under the real Groq key — e.g.
        # an ``anthropic/*`` model billed to the wrong provider, silently.
        provider = str(config.extra.get("provider") or "")
        if not provider:
            return MissionOutcome(answer=None, cost_usd=0.0, error=(
                "sut_start_error: config.extra['provider'] is required "
                f"(one of {'/'.join(sorted(PROVIDERS))}); refusing to guess "
                "which provider account to bill"))
        model = str(config.extra.get("model") or config.llm)
        if provider not in PROVIDERS:
            return MissionOutcome(answer=None, cost_usd=0.0,
                                  error=f"sut_start_error: unknown provider {provider!r}")
        api_key = _key_for(provider)
        # All three sut_start_error returns here fire before any request is sent,
        # so their cost is a known 0.0, not "unknown" (cost_usd defaults to None).
        if provider != "ollama" and not api_key:
            return MissionOutcome(answer=None, cost_usd=0.0, error=(
                f"sut_start_error: no key for {provider}; set "
                f"{'/'.join(PROVIDERS[provider]['key_env'])}"))

        evidence = json.dumps(task.params.get("evidence", {}), sort_keys=True)
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(evidence=evidence)},
                {"role": "user", "content": task.mission},
            ],
            "max_tokens": int(config.extra.get("max_tokens", 400)),
            "temperature": 0.2,
        }).encode()
        headers = {"Content-Type": "application/json", "User-Agent": "avaloka-bench/1.0"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(PROVIDERS[provider]["url"], data=body,
                                         headers=headers)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=float(
                    config.extra.get("timeout_s", 180))) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            return MissionOutcome(answer=None,
                                  latency_s=time.perf_counter() - started,
                                  error=f"provider_error: HTTP {exc.code}: {detail}")
        except Exception as exc:  # noqa: BLE001
            return MissionOutcome(answer=None,
                                  latency_s=time.perf_counter() - started,
                                  error=f"provider_error: {type(exc).__name__}: {exc}")

        latency = time.perf_counter() - started
        try:
            choice = payload["choices"][0]
            answer = _answer_text(choice["message"].get("content"))
        except (KeyError, IndexError, TypeError):
            return MissionOutcome(answer=None, latency_s=latency,
                                  error="answer_error: no choices in response")

        # Truncation is a HARNESS setting (max_tokens), not a model property.
        # Grading a cut-off answer as the model's answer publishes our own
        # configuration as that model's score — and a reasoning model that
        # spends its whole allowance thinking returns content="" while still
        # billing for every token. Both must be errors, not zeros.
        finish_reason = choice.get("finish_reason")
        truncated = finish_reason == "length"
        empty = not answer.strip()
        # Cost accounting. The usage block is the provider's own count, so the
        # token figures are measured; the dollar figure is derived from it and
        # the catalog quote, and carries that quote's provenance so a reader can
        # tell a real price from a proxy from a hole.
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        price = price_for(provider, str(config.extra.get("price_model") or model))

        # No usage block => the token counts are unknown, so the $0.00 derived
        # from them is meaningless. Tagging it `catalog` would be the exact
        # zero-that-looks-trustworthy this module exists to prevent, and it
        # would make the call invisible to the spend ceiling.
        has_usage = bool(usage) and ("prompt_tokens" in usage or "completion_tokens" in usage)
        cost_source = price.source if has_usage else SOURCE_USAGE_MISSING
        cost_usd = price.cost_usd(prompt_tokens, completion_tokens) if has_usage else 0.0

        error = None
        if truncated:
            error = (f"truncation_error: finish_reason='length' at max_tokens="
                     f"{config.extra.get('max_tokens', 400)}; the answer was cut off "
                     f"by the harness, so grading it would report our setting as the "
                     f"model's score")
        elif empty:
            error = (f"empty_answer_error: finish_reason={finish_reason!r} returned no "
                     f"text (a reasoning model may have spent the whole allowance on "
                     f"reasoning tokens); {completion_tokens} completion tokens were "
                     f"still billed")

        return MissionOutcome(
            answer=answer, latency_s=latency, error=error,
            cost_usd=cost_usd,
            tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_source=cost_source,
            data_scanned_ratio=0.0,
            artifacts={
                "provider": provider,
                "model": model,
                "price": price.as_dict(),
                # OpenRouter returns a generation id that can be reconciled
                # against the account's real billing export after the fact.
                "generation_id": payload.get("id"),
                "usage_raw": usage,
                "finish_reason": finish_reason,
            },
        )
