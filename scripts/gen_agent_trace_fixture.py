"""Regenerate ui/tests/fixtures/agent-trace.json from the backend's real shapes.

The UI's agent-trace tests render this file, so they test what the server
actually emits rather than a hand-written guess. Run it after changing
app/trace/build.py or app/trace/describe.py:

    PYTHONPATH=. python scripts/gen_agent_trace_fixture.py

It drives the small real graph in tests/test_agent_trace.py (a fake chat model,
no network) and writes the trace, the planner graph, a training turn's planner
graph (train and evaluate nodes) and a swarm capability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.trace.build import build_turn, planner_graph, trace_response  # noqa: E402
from app.trace.recorder import instrument  # noqa: E402
from avaloka.swarm_limits import limits_for  # noqa: E402
from tests.test_agent_trace import _fake_llm, _run, build_test_graph  # noqa: E402


def main() -> None:
    llm = instrument(_fake_llm(tokens=(1840, 212)), agent="conversational",
                     provider="openrouter", model="openai/gpt-oss-120b")
    final = _run(build_test_graph(llm=llm, fail_first=True), thread="fixture")
    turn = build_turn(
        final["agent_trace"], turn_id=final["trace_turn_id"],
        data_scope={"mode": "entire_dataset", "rows_analyzed": 400, "total_rows": 400, "is_sampled": False},
        swarm_plan=final["avaloka_swarm"], swarm_message="Cloning myself into 2 focused copies.",
    )
    trained = _run(build_test_graph(training=True), text="Predict churned from the other columns",
                   thread="fixture-training")
    training_turn = build_turn(trained["agent_trace"], turn_id=trained["trace_turn_id"],
                               data_scope={"mode": "entire_dataset", "rows_analyzed": 400, "total_rows": 400})
    with open(ROOT / "tests" / "fixtures" / "agent_trace_live_salaries.json") as fh:
        live_turn = json.load(fh)["turn"]
    limits = limits_for("oss", has_commercial_build=False).as_dict()
    doc = {
        "_generated_by": "scripts/gen_agent_trace_fixture.py",
        "trace": trace_response("thread-fixture", [turn], swarm_limits=limits),
        "graph": {**planner_graph(turn), "thread_id": "thread-fixture"},
        "graph_training": {**planner_graph(training_turn), "thread_id": "thread-fixture-training"},
        # A real run's turn (tests/fixtures/agent_trace_live_salaries.json): the
        # agents-heavy picture the owner saw, redrawn around the data lane.
        "graph_live": {**planner_graph(live_turn), "thread_id": "thread-live"},
        "swarm": {
            "enabled": True, "reason": "installed", "limits": limits, "limited": True,
            "message": f"Swarm limited to {limits['max_clones_per_mission']} clones per mission on {limits['label']}, run locally.",
        },
    }
    out = ROOT / "ui" / "tests" / "fixtures" / "agent-trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"wrote {out.relative_to(ROOT)}: {len(doc['graph']['nodes'])} nodes, {len(turn['steps'])} steps")


if __name__ == "__main__":
    main()
