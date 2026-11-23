"""The instant_intent track: the golden questions, scored field by field.

Design §3.3. Each task is one golden question from
``tests/recipes/golden_questions/<recipe>.json`` with its columns and the
expected Intent; the expected object is drawn from closed vocabularies, so
every metric is an exact match and no judge model is involved:

* ``recipe_match`` — the recipe id is the expected one;
* ``params_match`` — every parameter equals the expected one (lists unordered);
* ``flag_precision`` / ``flag_recall`` — over the three INTENT_FLAGS;
* ``parse_failure`` — the reply did not recognize (1.0 when it did not);
* ``pass`` — recipe and params match and no expected flag is missing.

Two configs run without spend: the heuristics and a replay of recorded
replies. The model configs of the design's protocol are PR-12's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ...core.config import Budget, RunConfig
from ...core.registry import register_suite
from ...core.task import BenchmarkTask, DatasetSpec, Split
from . import suts  # noqa: F401  (registers the two adapters)

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "tests" / "recipes" / "golden_questions"
#: Where a sweep leaves its transcripts for the replay config (design §3.3).
REPLAYS_DIR = Path("benchmarks") / "results" / "instant-intent" / "replays"


def load_goldens(directory: Path = GOLDEN_DIR) -> List[Tuple[str, int, Dict[str, Any]]]:
    """``(recipe, index, item)`` for every golden, file order."""
    out = []
    for path in sorted(directory.glob("*.json")):
        for i, item in enumerate(json.loads(path.read_text())):
            out.append((path.stem, i, item))
    return out


def _listed(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (sorted(v) if isinstance(v, (list, tuple)) else v) for k, v in params.items()}


def _ratio(num: int, den: int) -> float:
    return num / den if den else 1.0


def evaluate_intent(task: BenchmarkTask, answer: Any) -> Dict[str, float]:
    expected = task.reference
    if not isinstance(answer, dict) or answer.get("parse_failure"):
        return {"recipe_match": 0.0, "params_match": 0.0, "flag_precision": 0.0,
                "flag_recall": 0.0 if expected["flags"] else 1.0, "parse_failure": 1.0,
                "pass": 0.0, "score": 0.0}
    want, got = set(expected["flags"]), set(answer.get("flags", []))
    recipe = float(answer.get("recipe") == expected["recipe"])
    params = float(_listed(answer.get("params", {})) == _listed(expected["params"]))
    recall = _ratio(len(want & got), len(want))
    q = {
        "recipe_match": recipe,
        "params_match": params,
        "flag_precision": _ratio(len(want & got), len(got)),
        "flag_recall": recall,
        "parse_failure": 0.0,
        "pass": float(recipe and params and recall == 1.0),
    }
    q["score"] = (q["recipe_match"] + q["params_match"] + q["flag_recall"]) / 3
    return q


def _tasks() -> List[BenchmarkTask]:
    datasets: Dict[str, DatasetSpec] = {}
    tasks = []
    for recipe, i, item in load_goldens():
        ds = datasets.setdefault(recipe, DatasetSpec(
            id=f"golden_{recipe}", description=f"golden questions for the {recipe} recipe",
            loader=lambda r=recipe: json.loads((GOLDEN_DIR / f"{r}.json").read_text())))
        tasks.append(BenchmarkTask(
            id=f"ii_{recipe}_{i:02d}", track="instant_intent", mission=item["question"], dataset=ds,
            evaluator=evaluate_intent, reference=item["expected"], split=Split.DEV,
            params={"columns": item["columns"]}, tags=(recipe, *item["expected"]["flags"]),
        ))
    return tasks


def _configs() -> List[RunConfig]:
    # One trial each: both systems are deterministic, so repetition measures nothing.
    return [
        RunConfig(id="intent_heuristic", sut="intent_heuristic", llm="none", sampling_mode="full",
                  trials=1, budget=Budget(max_cost_usd=0.0)),
        RunConfig(id="intent_replay", sut="intent_replay", llm="recorded", sampling_mode="full",
                  trials=1, budget=Budget(max_cost_usd=0.0), extra={"replies_dir": str(REPLAYS_DIR)}),
    ]


@register_suite(
    "instant_intent",
    kind="track",
    description="Instant-analysis intent parsing over the golden questions (no spend): "
                "recipe / params / flag exact match and parse-failure rate for the "
                "heuristics and for replayed model transcripts.",
)
def build_instant_intent() -> Tuple[List[BenchmarkTask], List[RunConfig]]:
    return _tasks(), _configs()
