"""Model matrix: the same conversational-integrity tasks across models.

The design principle the harness exists for: model effects and agent effects
must be separable. This track holds the tasks (the six conversational probes)
and the deterministic scoring fixed, and varies ONLY the model — Groq-hosted,
OpenRouter-hosted, and locally served via Ollama — so the resulting table is a
model comparison on Avaloka-relevant behaviour, not a prompt or harness
comparison.

Models are chosen for the deployment question they answer, not for coverage.
The roster spans two and a half orders of magnitude of price, because the
question this track is actually asked is not "which model is best" but "how
much conversational integrity does the cheap tier give up", and you cannot
answer that without both ends of the price range in the same table:

* **Budget tier** (<$0.20/1M out) — ``deepseek-v4-flash``, ``gpt-oss-120b``,
  ``qwen3-coder-30b-a3b-instruct``. What a high-volume production path can
  afford per turn.
* **Mid tier** ($0.28–$3.60/1M out) — ``nemotron-3-super``, ``qwen3-coder-next``,
  ``qwen3.8-27b``, ``nemotron-3-ultra``. The plausible default.
* **Frontier tier** ($6–$25/1M out) — ``grok-4.6``, ``gpt-5.6-sol``,
  ``claude-sonnet-5``, ``claude-opus-5``. The ceiling, and the reference the
  cheaper rows are judged against.

Two rows are **controls, not candidates**: ``m_groq_gptoss120b`` and
``m_or_gptoss120b`` are the *same open-weight model on two providers*. If they
score differently, the difference is serving stack, not model, and every other
comparison in the table needs re-reading. The Ollama row is the third control:
it answers "does this survive with no vendor at all".

Every OpenRouter slug here is validated against the vendored price catalog at
suite-build time (:func:`benchmarks.core.pricing.validate_models`), so a typo
fails at ``list``/``build`` time rather than as an HTTP 400 twenty paid calls
into a sweep. Cost per row comes from that same catalog — see
:mod:`benchmarks.core.pricing` for why prices are never typed by hand.

The local config defaults to gemma4:31b because that is what a 64GB MacBook can
actually hold — override with AVALOKA_MATRIX_LOCAL_MODEL for other machines.
"""

from __future__ import annotations

import os
from typing import List, Tuple

from benchmarks.adapters import llm_direct as _adapter  # noqa: F401 (registration)
from benchmarks.core.config import RunConfig
from benchmarks.core.pricing import validate_models
from benchmarks.core.registry import register_suite
from benchmarks.tracks.conversational.track import _tasks

#: (config_id, provider, model). Ordered cheapest-first so a truncated run
#: (see the spend ceiling in benchmarks.core.spend) still covers the tier the
#: budget question is about, and burns the frontier models last.
MATRIX = [
    # ── controls: same model, two providers; and no provider at all ────────
    ("m_groq_gptoss120b", "groq", "openai/gpt-oss-120b"),
    ("m_groq_gptoss20b", "groq", "openai/gpt-oss-20b"),
    ("m_groq_compound", "groq", "groq/compound"),
    ("m_groq_qwen36_27b", "groq", "qwen/qwen3.6-27b"),
    ("m_local_ollama", "ollama",
     os.getenv("AVALOKA_MATRIX_LOCAL_MODEL", "gemma4:31b")),

    # ── budget tier ────────────────────────────────────────────────────────
    ("m_or_gptoss120b", "openrouter", "openai/gpt-oss-120b"),
    ("m_or_gemma3_4b", "openrouter", "google/gemma-3-4b-it"),
    ("m_or_deepseek_v4_flash", "openrouter", "deepseek/deepseek-v4-flash"),
    ("m_or_qwen3_coder_30b", "openrouter", "qwen/qwen3-coder-30b-a3b-instruct"),

    # ── mid tier ───────────────────────────────────────────────────────────
    ("m_or_nemotron3_super", "openrouter", "nvidia/nemotron-3-super-120b-a12b"),
    ("m_or_gemma4_31b", "openrouter", "google/gemma-4-31b-it"),
    ("m_or_qwen3_coder_next", "openrouter", "qwen/qwen3-coder-next"),
    ("m_or_qwen38_27b", "openrouter", "qwen/qwen3.8-27b"),
    ("m_or_nemotron3_ultra", "openrouter", "nvidia/nemotron-3-ultra-550b-a55b"),

    # ── frontier tier ──────────────────────────────────────────────────────
    ("m_or_grok46", "openrouter", "x-ai/grok-4.6"),
    ("m_or_gpt56_sol", "openrouter", "openai/gpt-5.6-sol"),
    ("m_or_claude_sonnet5", "openrouter", "anthropic/claude-sonnet-5"),
    ("m_or_claude_opus5", "openrouter", "anthropic/claude-opus-5"),

    # ── top of the range ───────────────────────────────────────────────────
    # Both quote $10/$50 per 1M, i.e. twice Opus 5's completion price, and both
    # carry ~1M context. They are last in the list because the roster is
    # ordered cheapest-first: if the spend ceiling truncates a sweep, it should
    # truncate here rather than in the budget tier the cost question is about.
    ("m_or_claude_fable5", "openrouter", "anthropic/claude-fable-5"),
    ("m_or_gpt6_astra", "openrouter", "openai/gpt-6-astra"),
]


@register_suite(
    "track7_model_matrix",
    kind="track",
    description=("Same conversational-integrity tasks, same deterministic "
                 "scoring, only the model varies: Groq production set, "
                 "OpenRouter budget/mid/frontier tiers, local Ollama. Needs "
                 "GROQ_API_KEY / OPENROUTER_API_KEY; the local config needs a "
                 "running Ollama. Costs real money — run "
                 "`--estimate-only` first."),
)
def build() -> Tuple[List, List[RunConfig]]:
    # Fail here, before any credential is read or any call is billed, if a slug
    # in MATRIX is not in the catalog.
    validate_models((provider, model) for _, provider, model in MATRIX)

    tasks = _tasks()
    configs = [
        RunConfig(id=config_id, sut="llm_direct", llm=model,
                  sampling_mode="full", environment="local", trials=3,
                  extra={"provider": provider, "model": model})
        for config_id, provider, model in MATRIX
    ]
    return tasks, configs
