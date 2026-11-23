"""No success after failure, in the swarm coordinator.

A clone whose mission ended without a verdict was ranked as a pass, cleared
for deployment level 4, and recorded as "done". Each test fails on
base/2.0-stack (ce126765) + #545. The app-side sites are in
tests/test_no_success_after_failure.py.
"""

import json
from types import SimpleNamespace

import avaloka.coordinator as coordinator
from avaloka.coordinator import MainAvaloka, ShardOutcome
from avaloka.swarm_limits import UNLIMITED


def _outcome(verdict, level):
    return ShardOutcome(shard_id=0, n_rows=10, output_dir="out", verdict=verdict,
                        max_safe_deployment_level=level, manual_minutes=0, cost_usd=0)


def test_a_missing_verdict_never_merges_as_pass():
    """coordinator.py:65 and :636 -- ``None: 0`` ranked a missing verdict as pass."""
    assert MainAvaloka._merge_verdicts([_outcome(None, 1)]) != "pass"
    assert MainAvaloka._merge_verdicts([_outcome("pass", 3), _outcome(None, 1)]) != "pass"
    assert MainAvaloka._merge_verdicts([_outcome("unheard-of", 1)]) != "pass"
    assert MainAvaloka._merge_verdicts([]) != "pass"
    assert MainAvaloka._merge_verdicts([_outcome("pass", 3), _outcome("warn", 2)]) == "warn"


def _no_verdict_mission(monkeypatch):
    """Every clone's mission returns, and no Validator ever wrote a verdict."""
    def run_without_verdict(ctx, on_progress=None):
        return SimpleNamespace(summary={"verdict": None, "max_safe_deployment_level": None,
                                        "model": None})

    monkeypatch.setattr(coordinator, "run_analyze", run_without_verdict)


def test_a_missing_level_is_the_floor_not_level_four(churn_csv, tmp_path, monkeypatch):
    """coordinator.py:543 -- ``or 4`` turned a missing level into the top of the ladder."""
    _no_verdict_mission(monkeypatch)

    result = MainAvaloka(swarm_limits=UNLIMITED).coordinate(
        str(churn_csv), goal="no verdict", output_dir=tmp_path / "lvl", workers=2)

    assert result.max_safe_deployment_level == 1
    assert result.verdict != "pass"


def test_a_clone_without_a_verdict_is_not_recorded_done(churn_csv, tmp_path, monkeypatch):
    """coordinator.py:320 -- clone.returned said status "done" with no verdict."""
    _no_verdict_mission(monkeypatch)

    MainAvaloka(swarm_limits=UNLIMITED).coordinate(
        str(churn_csv), goal="no verdict", output_dir=tmp_path / "clone", workers=2)

    trace = json.loads((tmp_path / "clone" / coordinator.SWARM_TRACE_FILE).read_text())
    statuses = [clone["status"] for clone in trace["clones"]]
    assert statuses and "done" not in statuses, statuses
