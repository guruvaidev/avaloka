"""Systems under test for the instant_intent track: the heuristics, and a replay.

Both answer with the same shape, so one evaluator scores them::

    {"recipe": str | None, "params": {name: column | [columns] | grain | window},
     "flags": [..], "parse_failure": bool}

``intent_heuristic`` runs :func:`avaloka.recipes.intent.heuristic_intent`.
``intent_replay`` reads a recorded model reply from
``<replies_dir>/<task_id>.json`` (``{"reply": "<the text the model returned>"}``)
and passes it through the same recognizer a live call would use, so a sweep's
transcripts can be re-scored after the vocabulary or the goldens change
without spending again. A missing or rejected reply is a parse failure, which
is a scored outcome, not an error. No SUT here calls a provider (design §3.3:
the first live sweep is PR-12 and needs spend approval).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ...adapters import register_adapter
from ...core.config import RunConfig
from ...core.task import BenchmarkTask, MissionOutcome


def intent_answer(intent) -> Dict[str, Any]:
    """The answer shape, from an ``avaloka.recipes.Intent``."""
    return {
        "recipe": intent.recipe,
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in intent.params.items()},
        "flags": list(intent.flags),
        "parse_failure": False,
    }


def parse_failure(reason: str) -> Dict[str, Any]:
    return {"recipe": None, "params": {}, "flags": [], "parse_failure": True, "reason": reason}


def _semantics(columns: Dict[str, str]):
    from avaloka.outcomes.semantics import ColumnSemantics
    return {name: ColumnSemantics(name=name, role=role) for name, role in columns.items()}


@register_adapter("intent_heuristic")
class HeuristicIntentAdapter:
    key = "intent_heuristic"
    requires_cloud = False

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        from avaloka.recipes.intent import heuristic_intent
        intent = heuristic_intent(task.mission, _semantics(task.params["columns"]), {})
        return MissionOutcome(answer=intent_answer(intent), cost_usd=0.0, cost_source="free_local",
                              tool_calls=0, data_scanned_ratio=0.0)


@register_adapter("intent_replay")
class ReplayIntentAdapter:
    key = "intent_replay"
    requires_cloud = False

    def run_mission(self, task: BenchmarkTask, config: RunConfig) -> MissionOutcome:
        from avaloka.recipes.intent import IntentRejected, recognize
        from avaloka.recipes.intent_model import parse_reply
        path = Path(config.extra["replies_dir"]) / f"{task.id}.json"
        if not path.is_file():
            answer = parse_failure("no_recording")
        else:
            reply = json.loads(path.read_text())["reply"]
            try:
                intent = recognize(parse_reply(reply), question=task.mission,
                                   columns=task.params["columns"], source="fixture")
                answer = intent_answer(intent)
            except IntentRejected as exc:
                answer = parse_failure(exc.reason)
        # A replay re-scores a transcript; the spend happened when it was recorded.
        return MissionOutcome(answer=answer, cost_usd=None, cost_source="replay",
                              tool_calls=0, data_scanned_ratio=0.0)
