"""Multi-Avaloka coordination tests — the open-claw main agent.

One main Avaloka shards the dataset, dispatches a fleet of sub-Avalokas (each
running its own firefly swarm on its shard) and converges their answers. These
tests pin the two guarantees that make the fleet honest: profiling converges
*exactly*, and decisions converge *conservatively*.
"""

import threading
import time

import numpy as np
import pandas as pd
import pytest

from avaloka.coordinator import (CoordinationResult, MainAvaloka, ShardOutcome,
                                 _safe_source_uri, converged_mean,
                                 load_coordination)
from avaloka.io import load_dataset
from avaloka.mission.ledger import WorkUnit
from avaloka.ray_batch import _partition_stats
from avaloka.mission.context import MissionKind
from avaloka.ray_batch import run_ray_local
from avaloka.swarm_limits import UNLIMITED


def test_coordinate_analyze_converges_exactly(churn_csv, tmp_path):
    df = pd.read_csv(churn_csv)
    events = []
    main = MainAvaloka(swarm_limits=UNLIMITED)
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
    assert cr.execution["parallelism"] == 4
    assert cr.sampling_summary == {
        "input_rows": len(df), "working_rows": len(df),
        "sampled_shards": 0, "strategies": ["none"],
    }


def test_coordination_matches_single_pass_ray_profile(churn_csv, tmp_path):
    """Fleet convergence and the shipped Ray map-reduce agree to the digit."""
    main = MainAvaloka(swarm_limits=UNLIMITED)
    cr = main.coordinate(str(churn_csv), goal="g", output_dir=tmp_path / "c",
                         workers=5, kind=MissionKind.ANALYZE)
    single = run_ray_local(str(churn_csv), n_partitions=3).converged
    fleet = {c["name"]: c for c in cr.converged_profile["columns"]}
    for col in single["columns"]:
        if col["role"] == "numeric":
            assert fleet[col["name"]]["mean"] == pytest.approx(col["mean"], abs=1e-4)
            assert fleet[col["name"]]["min"] == pytest.approx(col["min"], abs=1e-6)


def test_sub_avalokas_produce_independent_deliverables(churn_csv, tmp_path):
    main = MainAvaloka(swarm_limits=UNLIMITED)
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
    main = MainAvaloka(swarm_limits=UNLIMITED)
    cr = main.coordinate(str(p), goal="g", output_dir=tmp_path / "c", workers=8,
                         kind=MissionKind.ANALYZE)
    assert cr.workers <= 3 and cr.n_rows == 3


def test_training_swarm_rejects_missing_or_unknown_target(churn_csv, tmp_path):
    main = MainAvaloka()
    with pytest.raises(ValueError, match="requires a target"):
        main.coordinate(
            str(churn_csv), goal="train", output_dir=tmp_path / "missing",
            kind=MissionKind.TRAIN,
        )
    with pytest.raises(ValueError, match="available columns"):
        main.coordinate(
            str(churn_csv), goal="train", output_dir=tmp_path / "unknown",
            kind=MissionKind.TRAIN, target="does_not_exist",
        )


def test_sub_avalokas_really_overlap_in_time(churn_csv, tmp_path, monkeypatch):
    """The production coordinator must execute a wave concurrently, not narrate parallelism."""
    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_run(self, *, shard_id, shard, shard_dir, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return (
            ShardOutcome(
                shard_id=shard_id,
                n_rows=shard.n_rows,
                output_dir=str(shard_dir / "mission"),
                verdict="pass",
                max_safe_deployment_level=1,
                manual_minutes=10,
                cost_usd=0,
                sampling={
                    "sampled": False,
                    "n_working_rows": shard.n_rows,
                    "strategy": "none",
                },
                partition_stats=_partition_stats(load_dataset(str(shard.path)).frame),
            ),
            [WorkUnit("data_scout", "Data analyst", "profile", manual_minutes=10)],
        )

    monkeypatch.setattr(MainAvaloka, "_run_shard", fake_run)
    result = MainAvaloka(swarm_limits=UNLIMITED).coordinate(
        str(churn_csv), goal="parallel proof", output_dir=tmp_path / "parallel",
        workers=4, max_parallel=4,
    )

    assert peak >= 2
    assert result.execution["parallelism"] == 4
    assert [shard.shard_id for shard in result.shards] == [0, 1, 2, 3]


def test_database_source_transfer_is_recorded_before_swarm(churn_csv, tmp_path):
    """Connector materialization is a distinct, auditable stage before analysis."""
    sqlalchemy = pytest.importorskip("sqlalchemy")
    frame = pd.read_csv(churn_csv)
    db = tmp_path / "passengers.db"
    engine = sqlalchemy.create_engine(f"sqlite:///{db}")
    frame.to_sql("passengers", engine, index=False)
    engine.dispose()

    result = MainAvaloka().coordinate(
        f"sqlite:///{db}#passengers",
        goal="Profile transferred passenger data",
        output_dir=tmp_path / "sqlite-swarm",
        workers=3,
    )

    assert result.source_resolution["kind"] == "database"
    assert result.source_resolution["scheme"] == "sqlite"
    assert result.source_resolution["materialized"] is True
    assert result.n_rows == len(frame)


def test_source_provenance_redacts_credentials_and_signed_queries():
    safe = _safe_source_uri(
        "postgresql://analyst:secret@db.example:5432/app?sslkey=signed#customers"
    )
    assert safe == "postgresql://analyst:***@db.example:5432/app?redacted=1#customers"
    assert "secret" not in safe and "signed" not in safe
