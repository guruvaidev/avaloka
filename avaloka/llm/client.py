"""Optional LLM narrative enrichment.

Avaloka's deliverables are produced *deterministically* from real statistics so
the free, local mission works fully offline (data never leaves the machine).
An LLM is used only to enrich prose — never to invent numbers — and the mission
degrades gracefully to templated narrative when no key is configured.

Default provider is Anthropic Claude. Enable by installing the ``llm`` extra and
setting ``ANTHROPIC_API_KEY``. Model id is overridable via ``AVALOKA_LLM_MODEL``
(default: a fast, inexpensive Claude tier suitable for narration).
"""

from __future__ import annotations

import os

# Inexpensive, fast Claude tier — narration is short and cost-sensitive.
DEFAULT_MODEL = os.environ.get("AVALOKA_LLM_MODEL", "claude-haiku-4-5-20251001")

# Rough public per-MTok pricing for the default tier; used only to attribute a
# small, honest model cost to the ledger. Overridable for accuracy.
_PRICE_IN_PER_MTOK = float(os.environ.get("AVALOKA_LLM_PRICE_IN", "1.0"))
_PRICE_OUT_PER_MTOK = float(os.environ.get("AVALOKA_LLM_PRICE_OUT", "5.0"))


class LLMClient:
    """Thin wrapper that is a no-op unless an Anthropic key + SDK are present."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or DEFAULT_MODEL
        self.last_cost_usd = 0.0
        self._client = None
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            try:  # pragma: no cover - exercised only when the extra is installed
                import anthropic

                self._client = anthropic.Anthropic(api_key=key)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(self, system: str, prompt: str, *, max_tokens: int = 600) -> str | None:
        """Return enriched text, or ``None`` to signal the caller to fall back."""
        if not self._client:
            return None
        try:  # pragma: no cover - network path
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = getattr(resp, "usage", None)
            if usage is not None:
                self.last_cost_usd = (
                    usage.input_tokens / 1_000_000 * _PRICE_IN_PER_MTOK
                    + usage.output_tokens / 1_000_000 * _PRICE_OUT_PER_MTOK
                )
            return "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            )
        except Exception:
            return None


def narrate(client: LLMClient | None, system: str, prompt: str, fallback: str) -> tuple[str, float]:
    """Enrich ``fallback`` prose with the LLM when available.

    Returns ``(text, model_cost_usd)``. Always safe: returns the deterministic
    fallback (and zero cost) when no LLM is configured or the call fails.
    """
    if client is None or not client.available:
        return fallback, 0.0
    out = client.complete(system, prompt)
    if not out:
        return fallback, 0.0
    return out.strip(), client.last_cost_usd
