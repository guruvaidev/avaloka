"""2.0-vs-1.0 comparison suites: the conversational probes under three systems.

Companion to ``docs/benchmarks/2.0-vs-1.0.md``. Two suites over the SAME nine
probes and the SAME deterministic evaluators as ``track6_conversational``:

* ``bench20_conv_models`` — model alone. One prompt, the probe's computed
  evidence as JSON, no Avaloka prompt stack, no routing, no verifier. The four
  models are the ones the report compares the harness against; ``price_model``
  maps Google's bare id to the OpenRouter catalog entry so the cost column is
  a labelled proxy rather than a hole.
* ``bench20_conv_avaloka_v1`` — the 1.0 reply surface (``adapters/avaloka_v1``),
  run with the 1.0 tree on ``sys.path``. The 2.0 surface is
  ``track6_conversational --config t6_avaloka_reply``.

Five trials per probe, matching track6. The scripted ``faithful`` and
``sycophant`` references from track6 bracket the metric for all of these.
"""

from __future__ import annotations

import os
from typing import List, Tuple

from benchmarks.adapters import avaloka_v1 as _v1  # noqa: F401 (registration)
from benchmarks.adapters import llm_direct as _adapter  # noqa: F401 (registration)
from benchmarks.core.config import RunConfig
from benchmarks.core.pricing import validate_models
from benchmarks.core.registry import register_suite
from benchmarks.tracks.conversational.track import _tasks

TRIALS = 5

#: (config_id, provider, model, price_model). Cheapest-first, as in track7.
MODELS = [
    ("bm_ollama_gemma4_e4b", "ollama",
     os.getenv("AVALOKA_BENCH_LOCAL_MODEL", "gemma4:e4b"), None),
    ("bm_groq_gptoss120b", "groq", "openai/gpt-oss-120b", None),
    ("bm_or_claude_haiku55", "openrouter", "anthropic/claude-haiku-5.5", None),
    ("bm_gemini_argon", "gemini", "gemini-4-argon", "google/gemini-4-argon"),
]


@register_suite(
    "bench20_conv_models",
    kind="track",
    description=("2.0-vs-1.0 report: the conversational probes answered by a bare "
                 "model (no Avaloka prompt stack): gemma4 local, gpt-oss-120b@Groq, "
                 "claude-haiku-5.5@OpenRouter, gemini-4-argon@Google. Costs money."),
)
def build_models() -> Tuple[List, List[RunConfig]]:
    validate_models((p, price or m) for _, p, m, price in MODELS if p in ("openrouter",))
    configs = []
    for config_id, provider, model, price_model in MODELS:
        extra = {"provider": provider, "model": model, "max_tokens": 600}
        if price_model:
            extra["price_model"] = price_model
        configs.append(RunConfig(id=config_id, sut="llm_direct", llm=model,
                                 sampling_mode="full", environment="local",
                                 trials=TRIALS, extra=extra))
    return _tasks(), configs


@register_suite(
    "bench20_conv_avaloka_v1",
    kind="track",
    description=("2.0-vs-1.0 report: the Avaloka 1.0 reply surface on the "
                 "conversational probes. Run with the 1.0 tree (github/main "
                 "f772b0fb) on PYTHONPATH; needs GROQ_API_KEY."),
)
def build_v1() -> Tuple[List, List[RunConfig]]:
    return _tasks(), [RunConfig(id="t6_avaloka_reply_v1", sut="avaloka_reply_surface_v1",
                                llm="configured", sampling_mode="full",
                                environment="local", trials=TRIALS)]
