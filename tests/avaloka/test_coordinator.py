"""Multi-Avaloka coordination tests — the open-claw main agent.

One main Avaloka shards the dataset, dispatches a fleet of sub-Avalokas (each
running its own firefly swarm on its shard) and converges their answers. These
tests pin the two guarantees that make the fleet honest: profiling converges
*exactly*, and decisions converge *conservatively*.
"""

import numpy as np
import pandas as pd
import pytest

from avaloka.coordinator import (CoordinationResult, MainAvaloka,
                                 converged_mean, load_coordination)
from avaloka.mission.context import MissionKind
from avaloka.ray_batch import run_ray_local


def test_coordinate_analyze_converges_exactly(churn_csv, tmp_path):
    df = pd.read_csv(churn_csv)
    events = []
    main = MainAvaloka()
    cr = main.coordinate(str(churn_csv), goal="Understand churn",
                         output_dir=tmp_path / "coord", workers=4,
                         kind=MissionKind.ANALYZE,
                         on_progress=lambda e, m: events.append(e))
    assert isinstance(cr, CoordinationResult)
    assert cr.workers == 4 and cr.n_rows == len(df)
    # exact convergence: fleet mean == single-pass mean (only rounding to 6dp)
    # (the shared churn_csv fixture uses columns: tenure, monthly, tickets)
    assert converged_mean(cr.converged_profile, "tenure") == pytest.approx(df["tenure"].mean(), abs=1e-4)
    assert converged_mean(cr.converged_profile, "monthly") == pytest.approx(
        df["monthly"].mean(), abs=1e-4)
    assert cr.verdict in {"pass", "warn", "fail"}
    assert events[0] == "announce" and events[-1] == "converge"


def test_coordination_matches_single_pass_ray_profile(churn_csv, tmp_path):
    """Fleet convergence and the shipped Ray map-reduce agree to the digit."""
    main = MainAvaloka()
    cr = main.coordinate(str(churn_csv), goal="g", output_dir=tmp_path / "c",
                         workers=5, kind=MissionKind.ANALYZE)
    single = run_ray_local(str(churn_csv), n_partitions=3).converged
    fleet = {c["name"]: c for c in cr.converged_profile["columns"]}
    for col in single["columns"]:
        if col["role"] == "numeric":
            assert fleet[col["name"]]["mean"] == pytest.approx(col["mean"], abs=1e-4)
            assert fleet[col["name"]]["min"] == pytest.approx(col["min"], abs=1e-6)


def test_sub_avalokas_produce_independent_deliverables(churn_csv, tmp_path):
    main = MainAvaloka()
    cr = main.coordinate(str(churn_csv), goal="g", output_dir=tmp_path / "c",
                         workers=3, kind=MissionKind.ANALYZE)
    assert len(cr.shards) == 3
    for shard in cr.shards:
        # each sub-Avaloka wrote a real analysis bundle for its shard
        assert (tmp_path / "c" / f"sub-avaloka-{shard.shard_id:02d}" / "mission"
                / "data_quality.json").exists()
        assert shard.manual_minutes > 0
    # aggregate economics reflect the whole workforce
    assert cr.economics.economic_multiplier > 0
    assert cr.economics.estimated_manual_effort_hours > 0


def test_coordination_report_is_written_and_reloadable(churn_csv, tmp_path):
    main = MainAvaloka()
    main.coordinate(str(churn_csv), goal="g", output_dir=tmp_path / "c",
                    workers=2, kind=MissionKind.ANALYZE)
    report = load_coordination(tmp_path / "c")
    assert report["n_rows"] > 0 and report["workers"] == 2
    assert "converged_profile" in report and "economics" in report


def test_verdict_converges_conservatively_on_leakage(leaky_csv, tmp_path):
    """Any shard that fails must fail the whole fleet."""
    main = MainAvaloka()
    cr = main.coordinate(str(leaky_csv), goal="Predict y", output_dir=tmp_path / "c",
                         workers=3, kind=MissionKind.TRAIN, target="y", metric="roc_auc")
    assert cr.verdict == "fail"
    assert cr.max_safe_deployment_level == 1


def test_train_coordination_selects_a_best_model(churn_csv, tmp_path):
    main = MainAvaloka()
    cr = main.coordinate(str(churn_csv), goal="Predict churn", output_dir=tmp_path / "c",
                         workers=3, kind=MissionKind.TRAIN, target="churned", metric="roc_auc")
    assert cr.selected_model is not None
    assert "from_shard" in cr.selected_model
    assert cr.selected_model["candidates_considered"] >= 1


def test_workers_clamped_to_row_count(tmp_path):
    df = pd.DataFrame({"x": np.arange(3), "y": [0, 1, 0]})
    p = tmp_path / "tiny.csv"
    df.to_csv(p, index=False)
    main = MainAvaloka()
    cr = main.coordinate(str(p), goal="g", output_dir=tmp_path / "c", workers=8,
                         kind=MissionKind.ANALYZE)
    assert cr.workers <= 3 and cr.n_rows == 3
