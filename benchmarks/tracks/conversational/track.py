"""Conversational integrity track: sycophancy, fidelity, interactiveness, tone.

Registers as ``track6_conversational``. Three configs:

* ``t6_faithful``  — known-good scripted reference (should score ~1.0)
* ``t6_sycophant`` — known-bad scripted reference (should score ~0.0)
* ``t6_avaloka``   — the real conversational agent (needs an LLM key)

The two references bracket the metric: if faithful does not score near 1 and
sycophant near 0, the probes are broken and no number from the real agent
means anything. That bracketing run is free and deterministic, so it belongs
in CI.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from benchmarks.adapters import conversational as _adapters  # noqa: F401 (registration)
from benchmarks.core.config import RunConfig
from benchmarks.core.registry import register_suite
from benchmarks.core.task import BenchmarkTask, DatasetSpec
from benchmarks.tracks.conversational.probes import build_probes


def _tasks() -> List[BenchmarkTask]:
    tasks: List[BenchmarkTask] = []
    for probe in build_probes():
        def make_evaluator(p):
            def evaluate(task: BenchmarkTask, answer) -> Dict[str, float]:
                if not isinstance(answer, str) or not answer.strip():
                    return {"pass": 0.0, "score": 0.0}
                return p.evaluate(answer, p.grading_context)
            return evaluate
        tasks.append(BenchmarkTask(
            id=probe.id,
            track="conversational",
            mission=probe.turns[-1].text,
            dataset=DatasetSpec(id=f"conv-fixture-{probe.id}",
                                fingerprint="conv-v1"),
            evaluator=make_evaluator(probe),
            params={"evidence": probe.evidence, "dimension": probe.dimension},
            tags=(probe.dimension,),
        ))
    return tasks


@register_suite(
    "track6_conversational",
    kind="track",
    description=("Conversational integrity: sycophancy resistance, evidence "
                 "fidelity under social pressure, clarification of ambiguous "
                 "asks, tone floor. Scripted good/bad references bracket the "
                 "metric; the real agent needs an LLM key."),
)
def build() -> Tuple[List[BenchmarkTask], List[RunConfig]]:
    tasks = _tasks()
    configs = [
        RunConfig(id="t6_planner", sut="avaloka_planner_surface", llm="n/a",
                  sampling_mode="full", environment="local", trials=1),
        RunConfig(id="t6_faithful", sut="faithful_reference", llm="n/a",
                  sampling_mode="full", environment="local", trials=1),
        RunConfig(id="t6_sycophant", sut="sycophant_reference", llm="n/a",
                  sampling_mode="full", environment="local", trials=1),
        RunConfig(id="t6_avaloka", sut="avaloka_conversational",
                  llm="configured", sampling_mode="full", environment="local",
                  trials=5),
        RunConfig(id="t6_avaloka_reply", sut="avaloka_reply_surface",
                  llm="configured", sampling_mode="full", environment="local",
                  trials=5),
    ]
    return tasks, configs
