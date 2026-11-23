"""The clone count comes from the data, and the edition cap comes after.

Until 2.0 the fleet was whatever the caller typed, or a flat four. These
tests drive the real coordinator (with a stand-in sub-Avaloka, so the fleet
is cheap) on one dataset written at three sizes, and pin: three sizes give
three counts under an unlimited edition; a row-grouped file follows its row
groups; the open-source edition runs exactly two and says why; an explicit
``workers`` stays an override; and the reason is in ``swarm.requested``, in
``coordination.json`` and in the CLI's narration, naming no column or file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from avaloka.cli import app
from avaloka.coordinator import MainAvaloka, ShardOutcome, derive_workers, load_coordination
from avaloka.io import load_dataset
from avaloka.mission.ledger import WorkUnit
from avaloka.ray_batch import _partition_stats
from avaloka.swarm_limits import UNLIMITED, grant, limits_for
from avaloka.swarm_sizing import (BYTES_PER_CLONE, MAX_CLONES, MIN_CLONES,
                                  ROWS_PER_CLONE, size_fleet)
from avaloka.swarm_trace import SwarmTraceRecorder


@pytest.fixture(autouse=True)
def _no_edition(monkeypatch):
    monkeypatch.delenv("AVALOKA_EDITION", raising=False)


def _fake_run(self, *, shard_id, shard, shard_dir, **kwargs):
    """A stand-in sub-Avaloka: the sizing is what is under test, not the mission."""
    return (
        ShardOutcome(shard_id=shard_id, n_rows=shard.n_rows, output_dir=str(shard_dir / "mission"),
                     verdict="pass", max_safe_deployment_level=1, manual_minutes=1, cost_usd=0,
                     sampling={"sampled": False, "n_working_rows": shard.n_rows, "strategy": "none"},
                     partition_stats=_partition_stats(load_dataset(str(shard.path)).frame)),
        [WorkUnit("data_scout", "Data analyst", "profile", manual_minutes=1)],
    )


def _sales(n: int, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"region": rng.choice(["west", "east"], n),
                         "amount": rng.normal(100, 30, n).round(2)})


def _write(tmp_path: Path, n: int, *, fmt: str = "parquet", row_group_size: int | None = None) -> Path:
    path = tmp_path / f"sales-{n}.{fmt}"
    frame = _sales(n)
    if fmt == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False, row_group_size=row_group_size or n)
    return path


def _coordinate(source, out, *, limits=UNLIMITED, **kw):
    recorder = SwarmTraceRecorder()
    cr = MainAvaloka(swarm_limits=limits).coordinate(str(source), goal="g", output_dir=out,
                                                     recorder=recorder, **kw)
    return cr, recorder.records[0]


# ------------------------------------------------------------- the formula
def test_size_fleet_states_its_formula_in_numbers_only():
    small = size_fleet(10_000, 200_000)
    assert small.clones == MIN_CLONES
    assert size_fleet(60_000, 1_000_000).clones == 3
    assert size_fleet(120_000, 1_000_000).clones == 5
    # Bytes are a backstop for wide tables.
    assert size_fleet(1_000, 3 * BYTES_PER_CLONE).clones == 3
    # Partitions cap the size-based count; they never raise it past it.
    stats = {"partitioning": {"partitioned": True, "partitions": 3, "kind": "row_groups"}}
    assert size_fleet(120_000, 1_000_000, statistics=stats).clones == 3
    stats = {"partitioning": {"partitioned": True, "partitions": 40, "kind": "native"}}
    assert size_fleet(120_000, 1_000_000, statistics=stats).clones == 5
    # One partition is no partitioning, and the floor still holds.
    stats = {"partitioning": {"partitioned": True, "partitions": 1}}
    assert size_fleet(10, 10, statistics=stats).clones == MIN_CLONES
    # The ceiling is a sanity limit, not an edition limit.
    assert size_fleet(10 ** 9, 10 ** 12).clones == MAX_CLONES
    # The sentence carries the arithmetic and nothing that names data.
    fleet = size_fleet(120_000, 2_800_000, lane="sampled_online", estimated=False,
                       statistics={"partitioning": {"partitioned": True, "partitions": 3,
                                                    "kind": "row_groups", "columns": ["region"]},
                                   "indexes": [{"name": "ix_region", "columns": ["region"]}]})
    assert fleet.clones == 3
    assert f"ceil(rows/{ROWS_PER_CLONE:,})=5" in fleet.derived_from
    assert "3 row_groups, one clone per partition up to that -> 3" in fleet.derived_from
    assert "1 index(es)" in fleet.derived_from and "sampled_online lane" in fleet.derived_from
    assert "region" not in fleet.derived_from and "ix_region" not in fleet.derived_from


def test_grant_tells_the_cap_in_the_data_terms_when_the_count_was_derived():
    oss = limits_for("oss", has_commercial_build=False)
    g = grant(5, None, oss, derived_from="120,000 rows -> 5")
    assert (g.clones, g.limited) == (2, True)
    assert g.message == ("Swarm limited to 2 clones on the free open-source edition: "
                         "the data sized the fleet at 5, so 2 will run.")
    assert g.as_dict()["derived_from"] == "120,000 rows -> 5"
    # An explicit request reads as it always did.
    assert "you asked for 5" in grant(5, None, oss).message
    assert grant(5, None, oss).derived_from is None


# ------------------------------------------------------ the coordinator
def test_the_same_dataset_at_three_sizes_gives_three_counts_under_unlimited(tmp_path, monkeypatch):
    monkeypatch.setattr(MainAvaloka, "_run_shard", _fake_run)
    counts = {}
    for n in (10_000, 60_000, 120_000):
        cr, requested = _coordinate(_write(tmp_path, n), tmp_path / f"fleet-{n}")
        counts[n] = cr.workers
        assert requested["requested_clones"] == cr.workers and requested["limited"] is False
        assert f"{n:,} rows" in requested["derived_from"]
        assert cr.execution["swarm"]["derived_from"] == requested["derived_from"]
        assert load_coordination(tmp_path / f"fleet-{n}")["execution"]["swarm"]["derived_from"]
    assert counts == {10_000: 2, 60_000: 3, 120_000: 5}


def test_a_row_grouped_file_gets_one_clone_per_row_group(tmp_path, monkeypatch):
    """The statistics, not only the size: 120,000 rows in 3 row groups is 3 clones."""
    monkeypatch.setattr(MainAvaloka, "_run_shard", _fake_run)
    source = _write(tmp_path, 120_000, row_group_size=40_000)
    fleet, resolved = derive_workers(str(source))
    assert resolved is not None and resolved.detail["statistics"]["row_groups"] == 3
    assert fleet.partitions == 3 and fleet.clones == 3
    cr, requested = _coordinate(source, tmp_path / "fleet")
    assert cr.workers == 3
    assert "3 row_groups, one clone per partition up to that -> 3" in requested["derived_from"]


def test_open_source_runs_exactly_two_and_says_why(tmp_path, monkeypatch):
    monkeypatch.setattr(MainAvaloka, "_run_shard", _fake_run)
    events = []
    source = _write(tmp_path, 120_000)
    recorder = SwarmTraceRecorder()
    cr = MainAvaloka().coordinate(str(source), goal="g", output_dir=tmp_path / "fleet",
                                  recorder=recorder, on_progress=lambda e, m: events.append((e, m)))
    requested = recorder.records[0]
    assert cr.workers == 2 and requested["granted_clones"] == 2
    assert requested["requested_clones"] == 5 and requested["limited"] is True
    assert requested["message"] == ("Swarm limited to 2 clones on the free open-source edition: "
                                    "the data sized the fleet at 5, so 2 will run.")
    assert "ceil(rows/25,000)=5" in requested["derived_from"]
    assert recorder.mission()["grant"]["derived_from"] == requested["derived_from"]
    kinds = [e for e, _ in events]
    assert kinds.index("sizing") < kinds.index("limit") < kinds.index("announce")
    assert dict(events)["sizing"] == requested["derived_from"]


def test_an_explicit_count_is_an_override(tmp_path, monkeypatch):
    monkeypatch.setattr(MainAvaloka, "_run_shard", _fake_run)
    cr, requested = _coordinate(_write(tmp_path, 120_000), tmp_path / "fleet", workers=3)
    assert cr.workers == 3 and requested["derived_from"] is None
    assert cr.execution["swarm"]["derived_from"] is None


def test_an_unreadable_source_still_fails_in_the_coordinator_with_its_trace(tmp_path):
    recorder = SwarmTraceRecorder()
    with pytest.raises(FileNotFoundError):
        MainAvaloka(swarm_limits=UNLIMITED).coordinate(str(tmp_path / "nope.csv"), goal="g",
                                                       output_dir=tmp_path / "fleet", recorder=recorder)
    assert [r["type"] for r in recorder.records] == ["swarm.requested", "swarm.finished"]
    assert recorder.records[0]["requested_clones"] == 4
    assert recorder.records[0]["derived_from"].startswith("source could not be sized (FileNotFoundError)")
    assert recorder.records[-1]["status"] == "failed"


def test_the_reason_never_names_a_column_or_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(MainAvaloka, "_run_shard", _fake_run)
    source = tmp_path / "acme_layoffs_q3.parquet"
    pd.DataFrame({"patient_hiv_status": np.arange(120_000), "ward": "a"}).to_parquet(
        source, index=False, row_group_size=40_000)
    _, requested = _coordinate(source, tmp_path / "fleet")
    for planted in ("acme", "layoffs", "patient", "ward", "parquet", str(tmp_path)):
        assert planted not in requested["derived_from"], planted


# ---------------------------------------------------------------- the CLI
def test_cli_narrates_the_sizing_and_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(MainAvaloka, "_run_shard", _fake_run)
    source = _write(tmp_path, 120_000, fmt="csv")
    out = tmp_path / "fleet"
    result = CliRunner().invoke(app, ["coordinate", str(source), "--goal", "x", "-o", str(out)])
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "sized the fleet from the data" in text and "ceil(rows/25,000)=5" in text
    assert "the data sized the fleet at 5, so 2 will run" in text
    report = load_coordination(out)
    assert report["workers"] == 2 and report["execution"]["swarm"]["derived_from"]
    trace = json.loads((out / "swarm_trace.json").read_text())
    assert trace["grant"]["derived_from"] == report["execution"]["swarm"]["derived_from"]


def test_cli_explicit_workers_is_not_narrated_as_derived(tmp_path, monkeypatch):
    monkeypatch.setattr(MainAvaloka, "_run_shard", _fake_run)
    out = tmp_path / "fleet"
    result = CliRunner().invoke(app, ["coordinate", str(_write(tmp_path, 120_000, fmt="csv")),
                                      "--goal", "x", "--workers", "2", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert "sized the fleet from the data" not in result.output
    assert load_coordination(out)["execution"]["swarm"]["derived_from"] is None


# ------------------------------------------------------- the document
def test_editions_doc_states_the_sizing_formula():
    doc = (Path(__file__).resolve().parents[2] / "docs" / "EDITIONS.md").read_text()
    section = doc[doc.index("### Adaptive clone count"):]
    section = section[: section.index("\n### ", 1)]
    assert f"ceil(rows  / {ROWS_PER_CLONE:,})" in section
    assert f"ceil(bytes / {BYTES_PER_CLONE // (1024 * 1024)} MiB)" in section
    assert f"clamp(max(by_rows, by_bytes), {MIN_CLONES}, {MAX_CLONES})" in section
    assert "avaloka/swarm_sizing.py" in section
    message = grant(5, None, limits_for("oss", has_commercial_build=False),
                    derived_from="x").message
    assert re.sub(r"\s+", " ", message) in re.sub(r"\s+", " ", section)
