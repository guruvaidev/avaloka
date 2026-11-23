"""The CLI's team story says a finding, says "you asked for N" only when
someone asked, and never calls the manual-effort heuristic measured.

Records come from a real coordinator run (the fireflies, not a stub).
"""

from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

from avaloka.cli import app
from avaloka.coordinator import MainAvaloka
from avaloka.swarm import SwarmNarrator
from avaloka.mission.ledger import Ledger
from avaloka.swarm_limits import SwarmLimits
from avaloka.swarm_narration import VERBOSE, narrate
from avaloka.swarm_trace import SwarmTraceRecorder

_CAP_TWO = SwarmLimits("oss", 2, 2)
#: Spelled out, not imported, so the test still collects on a tree without it.
MANUAL_EFFORT_CAVEAT = "rough heuristic estimate, not measured"


@pytest.fixture(autouse=True)
def _no_edition(monkeypatch):
    monkeypatch.delenv("AVALOKA_EDITION", raising=False)


@pytest.fixture(scope="module")
def story(tmp_path_factory):
    from tests.avaloka.conftest import write_churn_csv

    base = tmp_path_factory.mktemp("honest")
    data = write_churn_csv(base / "churn.csv")
    recorder = SwarmTraceRecorder()
    MainAvaloka(swarm_limits=_CAP_TWO).coordinate(
        str(data), goal="what drives churn", output_dir=base / "fleet", workers=8,
        recorder=recorder)
    return {"records": recorder.records, "frame": pd.read_csv(data)}


def test_the_story_states_a_finding_from_the_merged_result(story):
    text = "\n".join(narrate(story["records"], VERBOSE))
    frame = story["frame"]
    rate, ones = frame["churned"].mean(), int(frame["churned"].sum())
    assert (f"What the merged data shows: churned is 1 in {rate:.1%} of the {len(frame):,} rows "
            f"with a value ({ones:,} rows)") in text, text


def test_you_asked_for_n_only_when_the_number_was_typed(story):
    records = story["records"]
    text = "\n".join(narrate(records, VERBOSE))
    assert "you asked for 8" in text  # workers=8 is the --swarm 8 a user typed

    sized = [dict(r, derived_from="400 rows: ... -> 8") if r["type"] == "swarm.requested" else r
             for r in records]
    told = "\n".join(narrate(sized, VERBOSE))
    assert "you asked for" not in told, told
    assert "the data sized the fleet at 8" in told, told


def test_the_hours_are_labelled_a_heuristic_everywhere_the_cli_says_them(story, tmp_path):
    text = "\n".join(narrate(story["records"], VERBOSE))
    assert "human team" not in text and "measured work" not in text, text
    assert MANUAL_EFFORT_CAVEAT in text

    said: list[str] = []
    narrator = SwarmNarrator(said.append)
    ledger = Ledger()
    narrator.converge(ledger)
    assert "human team" not in said[-1] and MANUAL_EFFORT_CAVEAT in said[-1], said

    from tests.avaloka.conftest import write_churn_csv
    data = write_churn_csv(tmp_path / "c.csv", n=120)
    runner = CliRunner()
    fleet = runner.invoke(app, ["coordinate", str(data), "--goal", "what drives churned",
                                "-o", str(tmp_path / "fleet"), "--swarm", "2", "-v"])
    assert fleet.exit_code == 0, fleet.output
    out = " ".join(fleet.output.split())
    assert "human team" not in out and "Fleet effort replaced" not in out, out
    assert "Fleet manual effort" in out and MANUAL_EFFORT_CAVEAT in out, out
    single = runner.invoke(app, ["analyze", str(data), "--goal", "what drives churned",
                                 "-o", str(tmp_path / "one")])
    assert single.exit_code == 0, single.output
    out = " ".join(single.output.split())
    assert "Estimated manual effort" in out and MANUAL_EFFORT_CAVEAT in out, out


def test_merged_findings_prefer_the_target_then_the_goal_then_the_worst_gap():
    from avaloka.coordinator import merged_findings

    profile = {"n_rows": 10, "columns": [
        {"name": "id", "role": "numeric", "missing": 0, "count": 10, "min": 1, "max": 10, "mean": 5.5},
        {"name": "fare", "role": "numeric", "missing": 2, "count": 8, "min": 1.5, "max": 30.0, "mean": 12.25},
        {"name": "payment", "role": "categorical", "missing": 0, "count": 10,
         "top_values": [{"value": "card", "count": 7}]},
    ]}
    assert merged_findings(profile, goal="what drives the fare") == [
        "fare averages 12.25 over 8 rows (range 1.5 to 30)",
        "fare is empty in 2 of 10 rows (20.0%)"]
    assert merged_findings(profile, goal="anything", target="payment")[0] == \
        "payment is most often “card” (7 of 10 rows, 70.0%)"
    # No identifier is ever reported as a finding.
    assert not any(f.startswith("id ") for f in merged_findings(profile, goal="", limit=5))
