import json
import sqlite3

import pandas as pd

from avaloka.fireflies.workload_optimizer import WorkloadOptimizer
from avaloka.io.sources import materialize
from avaloka.mission.budget import Budget
from avaloka.mission.context import MissionContext, MissionKind
from avaloka.missions import analyze_fireflies


def test_optimizer_is_in_swarm_before_code_generation():
    names = [item.name for item in analyze_fireflies()]
    assert names.index("workload_optimizer") < names.index("data_engineer")


def test_filter_factor_prefers_indexed_selective_predicate(tmp_path):
    frame = pd.DataFrame({"region": ["west"] * 2 + ["east"] * 8, "amount": range(10)})
    source = tmp_path / "sales.csv"
    frame.to_csv(source, index=False)
    ctx = MissionContext(MissionKind.ANALYZE, "west sales", str(source), tmp_path / "out", Budget())
    from avaloka.fireflies.data_scout import DataScout
    DataScout(ctx).activate()
    ctx.blackboard["analysis_filters"] = [{"column": "region", "operator": "eq", "value": "west"}]
    ctx.blackboard["source_statistics"] = {"indexes": [{"name": "ix_region", "columns": ["region"]}],
        "partitioning": {"partitioned": False, "columns": [], "partitions": 1}}
    WorkloadOptimizer(ctx).activate()
    plan = json.loads((tmp_path / "out" / "workload_optimization.json").read_text())
    assert plan["strategy"] == "index_filtered_scan"
    assert plan["filter_factors"][0]["estimated_filter_factor"] == 0.2


def test_sqlite_materialization_collects_native_index_and_size(tmp_path):
    database = tmp_path / "sales.db"
    con = sqlite3.connect(database)
    con.executescript("CREATE TABLE sales(id INTEGER PRIMARY KEY, region TEXT); CREATE INDEX ix_region ON sales(region); INSERT INTO sales VALUES(1, 'west');")
    con.close()
    resolved = materialize(f"sqlite:///{database}#sales", cache_dir=tmp_path / "cache")
    stats = resolved.detail["statistics"]
    assert stats["database_size_bytes"] > 0
    assert any(index["columns"] == ["region"] for index in stats["indexes"])
