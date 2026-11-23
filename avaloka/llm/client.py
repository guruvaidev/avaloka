"""Optional LLM narrative enrichment.

Avaloka's deliverables are produced *deterministically* from real statistics so
the free, local mission works fully offline (data never leaves the machine).
An LLM is used only to enrich prose — never to invent numbers.

Providers (see :mod:`avaloka.llm.providers`): a local Ollama / OpenAI-compatible
server, Groq, OpenRouter and OpenAI over one stdlib transport, and Anthropic
through its native SDK (the ``llm`` extra). Choose with ``--llm-provider`` or
``AVALOKA_LLM_PROVIDER`` and ``--llm-model`` or ``AVALOKA_LLM_MODEL``; with
neither, the first configured provider is used, local first.

**Never silent.** A provider that is asked for and cannot be used, or that
fails mid-run, is reported in one line naming the provider and the fix, and
the line says the rest of the run uses Avaloka's deterministic wording. The
client then stops calling the model for the run, so the error is printed once.

Cost follows the platform's pricing hooks: a cost the provider reports in its
usage block (OpenRouter) is used as given; otherwise ``AVALOKA_MODEL_PRICES``
(``{"model": [usd_in_per_mtok, usd_out_per_mtok]}``, as in ``app/trace/prices``)
or ``AVALOKA_LLM_PRICE_IN`` / ``AVALOKA_LLM_PRICE_OUT``; otherwise a local model
costs 0 and a hosted one is recorded as 0 with ``last_cost_known = False``.
There is deliberately no built-in price table.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Optional

from avaloka.llm.openai_compat import LLMCallError, OpenAICompatibleChat
from avaloka.llm.providers import (ANTHROPIC, LOCAL, LLMConfigError, Selection,
                                   capabilities, resolve)

#: Printed after every provider problem: what happens to the run instead.
DETERMINISTIC_NOTE = "Continuing without the model: the rest of this run uses Avaloka's deterministic wording."


def _stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def price_for(model: str, env=None) -> Optional[tuple[float, float]]:
    """USD per million (input, output) tokens, if the operator configured one."""
    env = os.environ if env is None else env
    raw = env.get("AVALOKA_MODEL_PRICES") or ""
    if raw:
        try:
            pair = (json.loads(raw) or {}).get(model)
            if pair is not None:
                return float(pair[0]), float(pair[1])
        except (ValueError, TypeError, IndexError, AttributeError):
            pass
    p_in, p_out = env.get("AVALOKA_LLM_PRICE_IN"), env.get("AVALOKA_LLM_PRICE_OUT")
    if p_in and p_out:
        try:
            return float(p_in), float(p_out)
        except ValueError:
            return None
    return None


class LLMClient:
    """One provider for one run. A no-op (``available`` False) when none is usable.

    ``required`` is True when the user asked for the model (``--llm``): then an
    unusable configuration is reported. Either way, a configured provider that
    fails is reported, never swallowed.
    """

    def __init__(self, model: str | None = None, *, provider: str | None = None,
                 required: bool = False, notify: Optional[Callable[[str], None]] = None) -> None:
        self.notify = notify or _stderr
        self.last_cost_usd = 0.0
        self.last_cost_known = True
        # What the last call used, for the understanding record of the firefly
        # that made it (avaloka.understanding_trace reads these names from the
        # unit's metadata).
        self.last_tokens_in = 0
        self.last_tokens_out = 0
        self.calls = 0
        self.error: Optional[str] = None
        self.selection: Optional[Selection] = None
        self._client: Any = None
        self.model = model or ""
        self.provider = provider or ""
        try:
            sel = resolve(provider, model)
        except LLMConfigError as exc:
            # Nothing configured and nothing asked for is the quiet, local default.
            if required or exc.provider != "none":
                self._fail(exc.provider, str(exc))
            return
        self.selection = sel
        self.model = sel.model
        self.provider = sel.provider
        if sel.provider == ANTHROPIC:
            # The SDK honours ANTHROPIC_BASE_URL; pointed at a local server (Ollama
            # speaks this API) the provider is "local", which is the honest label.
            if os.environ.get("ANTHROPIC_BASE_URL"):
                self.provider = LOCAL
            try:
                import anthropic  # the `llm` extra, never a core dependency
            except ImportError:
                self._fail(ANTHROPIC, "Anthropic was selected but the anthropic package is not "
                           "installed. Fix: pip install 'avaloka[llm]', or choose another "
                           "provider with --llm-provider.")
                return
            self._client = anthropic.Anthropic(api_key=sel.api_key)  # pragma: no cover
        else:
            self._client = OpenAICompatibleChat(sel)
        if required:
            self.notify(f"LLM: {sel.provider} · {sel.model}"
                        + (f" at {sel.base_url}" if sel.provider == LOCAL else ""))

    # ------------------------------------------------------------------ state
    @property
    def available(self) -> bool:
        return self._client is not None

    def _fail(self, provider: str, message: str) -> None:
        self.error = message
        self._client = None
        self.notify(f"LLM error ({provider}): {message} {DETERMINISTIC_NOTE}")

    def _record(self, tokens_in: int, tokens_out: int, reported_cost: Any = None) -> None:
        self.calls += 1
        self.last_tokens_in, self.last_tokens_out = int(tokens_in or 0), int(tokens_out or 0)
        if isinstance(reported_cost, (int, float)):
            self.last_cost_usd, self.last_cost_known = float(reported_cost), True
            return
        price = price_for(self.model)
        if price is not None:
            self.last_cost_usd = (self.last_tokens_in * price[0]
                                  + self.last_tokens_out * price[1]) / 1_000_000
            self.last_cost_known = True
        else:
            self.last_cost_usd = 0.0
            self.last_cost_known = self.provider == LOCAL

    # ------------------------------------------------------------------ calls
    def complete(self, system: str, prompt: str, *, max_tokens: int = 600) -> str | None:
        """Return enriched text, or ``None`` after reporting why there is none."""
        if not self._client:
            return None
        if isinstance(self._client, OpenAICompatibleChat):
            try:
                text, usage = self._client.complete(system, prompt, max_tokens=max_tokens)
            except LLMCallError as exc:
                self._fail(exc.provider, str(exc))
                return None
            self._record(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                         usage.get("cost"))
            return text
        return self._complete_anthropic(system, prompt, max_tokens)

    def _complete_anthropic(self, system: str, prompt: str, max_tokens: int) -> str | None:
        from avaloka.llm.openai_compat import scrub_sealed
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=max_tokens, system=scrub_sealed(system),
                                      messages=[{"role": "user", "content": scrub_sealed(prompt)}])
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - the SDK's error tree is the extra's
            status = getattr(exc, "status_code", None)
            fix = ("check ANTHROPIC_API_KEY" if status in (401, 403)
                   else "check --llm-model / ANTHROPIC_BASE_URL, or choose another provider")
            self._fail(ANTHROPIC, f"Anthropic call failed ({type(exc).__name__}"
                       + (f", HTTP {status}" if status else "") + f"). Fix: {fix}.")
            return None
        usage = getattr(resp, "usage", None)
        self._record(getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        if not text.strip():
            self._fail(ANTHROPIC, "Anthropic returned an empty reply. Fix: raise the token "
                       "budget or choose another model with --llm-model.")
            return None
        return text

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
             max_tokens: int = 1024, require_tool: bool = True) -> dict[str, Any]:
        """One tool-calling turn on an OpenAI-compatible provider.

        tool_choice is mapped per provider and model
        (:func:`avaloka.llm.providers.tool_choice_for`); a rejected or phantom
        tool call is re-prompted (bounded) before
        :class:`~avaloka.llm.openai_compat.LLMToolCallRejected` is raised.
        """
        if not isinstance(self._client, OpenAICompatibleChat):
            raise LLMCallError(self.provider or "none",
                               "tool calls need an OpenAI-compatible provider.",
                               "use --llm-provider local, groq, openrouter or openai.")
        message, usage, _ = self._client.chat(messages, tools, max_tokens=max_tokens,
                                              require_tool=require_tool)
        self._record(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                     usage.get("cost"))
        return message

    @property
    def capabilities(self):
        return capabilities(self.provider, self.model) if self.selection else None


def narrate(client: LLMClient | None, system: str, prompt: str, fallback: str) -> tuple[str, float]:
    """Enrich ``fallback`` prose with the LLM when available.

    Returns ``(text, model_cost_usd)``. The deterministic fallback (and zero
    cost) is used when no LLM is configured or the call fails; a failure has
    already been reported by the client by the time this returns.
    """
    if client is None or not client.available:
        return fallback, 0.0
    out = client.complete(system, prompt)
    if not out:
        return fallback, 0.0
    # The model may reword; it may not give a figure a currency or a scale the
    # deterministic text and its inputs never stated (avaloka/units.py).
    from avaloka.units import introduces_units
    if introduces_units(f"{prompt}\n{fallback}", out):
        return fallback, client.last_cost_usd
    return out.strip(), client.last_cost_usd


def usage_metadata(client: LLMClient | None) -> dict:
    """The model-usage keys a firefly adds to its unit metadata after calling
    ``narrate``: empty when no model answered, so a local run records none."""
    if client is None or not client.calls:
        return {}
    return {"llm_provider": client.provider, "llm_model": client.model, "llm_calls": client.calls,
            "tokens_in": client.last_tokens_in, "tokens_out": client.last_tokens_out}
