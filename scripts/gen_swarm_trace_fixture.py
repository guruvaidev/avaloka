"""Regenerate ui/tests/fixtures/swarm-trace.json by running a real swarm mission.

The UI's swarm-demo tests feed this file's records through the reducer, so they
test what the server actually emits. Run it after changing avaloka/swarm_trace.py
or the coordinator's narration:

    PYTHONPATH=. python scripts/gen_swarm_trace_fixture.py [--workers N]

It runs ``MainAvaloka.coordinate`` on avaloka/examples/churn.csv under the
open-source limits with a ``SwarmTraceRecorder`` (no model, no network; the
fireflies are the CLI's) and writes the mission document with its records, as
``GET /api/swarm/missions/{id}?records=1`` would serve it, plus the RL episodes.
``--workers`` defaults to one more than the edition cap, so the fixture carries
a limited grant and its message, which the UI must show.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from avaloka.swarm_trace import SwarmTraceRecorder, episodes  # noqa: E402
except ImportError:  # pragma: no cover - the module lands with the swarm API (#511)
    sys.exit("avaloka.swarm_trace is not in this checkout; the fixture cannot be regenerated here.")

from avaloka.coordinator import MainAvaloka  # noqa: E402
from avaloka.mission.context import MissionKind  # noqa: E402
from avaloka.swarm_limits import limits_for  # noqa: E402

SOURCE = ROOT / "avaloka" / "examples" / "churn.csv"
GOAL = "what drives churn"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    limits = limits_for("oss", has_commercial_build=False)
    parser.add_argument("--workers", type=int, default=(limits.max_clones_per_mission or 1) + 1)
    args = parser.parse_args()

    recorder = SwarmTraceRecorder(mission_id="fixture-churn")
    with tempfile.TemporaryDirectory() as tmp:
        MainAvaloka(swarm_limits=limits).coordinate(
            str(SOURCE), goal=GOAL, output_dir=Path(tmp) / "fleet", workers=args.workers,
            kind=MissionKind.ANALYZE, recorder=recorder,
        )
    mission = recorder.mission(include_records=True)
    # What the API adds around the document (app/trace/swarm_service.py).
    mission["persistence"] = "memory"
    mission["persistence_note"] = "no Redis: this mission is lost when the server restarts"
    doc = {
        "_generated_by": "scripts/gen_swarm_trace_fixture.py",
        "_source": f"MainAvaloka.coordinate on {SOURCE.relative_to(ROOT)}, workers={args.workers}, "
                   f"limits={limits.edition}",
        "mission": mission,
        "episodes": {"mission_id": mission["mission_id"], "status": mission["status"],
                     "episodes": episodes(mission)},
        "start_response": {
            "mission_id": mission["mission_id"], "status": "running",
            "grant": {"requested_clones": args.workers, "clones": mission["grant"]["granted_clones"],
                      "requested_parallel": None, "parallel": mission["grant"]["granted_parallel"],
                      "limited": mission["grant"]["limited"], "message": mission["grant"]["message"],
                      "limits": limits.as_dict()},
            "limits": limits.as_dict(), "source": {"kind": "sample", "sample": "churn"},
            "persistence": "memory",
        },
    }
    out = ROOT / "ui" / "tests" / "fixtures" / "swarm-trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    from collections import Counter

    kinds = Counter(r["type"] for r in mission["records"])
    print(f"wrote {out.relative_to(ROOT)}: {len(mission['records'])} records, "
          f"{len(mission['clones'])} clones, status {mission['status']}; " + ", ".join(
              f"{k} {v}" for k, v in sorted(kinds.items())))


if __name__ == "__main__":
    main()
