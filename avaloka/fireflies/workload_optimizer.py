"""Statistics-driven physical planning for generated analysis code.

The plan is executed, not only written: the Data Engineer reads it back,
applies the projection and the predicates (pushed into the Parquet reader
where the source allows it) and records what it measured under ``applied``,
beside the ``estimated_rows_after_filters`` ceiling this firefly recorded.
"""
from avaloka.fireflies.base import Firefly
from avaloka.goal_analysis import goal_filters
from avaloka.source_statistics import file_statistics
from avaloka.util import write_json

class WorkloadOptimizer(Firefly):
    name = "workload_optimizer"
    human_role = "Query and workload optimization engineer"

    def perform(self):
        profile = self.ctx.blackboard["profile"]
        stats = dict(self.ctx.blackboard.get("source_statistics") or {}) or file_statistics(
            self.ctx.blackboard["handle"].source)
        # The predicates come from the caller when it has them, otherwise from
        # the goal itself: "west sales" is a scan of the rows where region is
        # west, and that is the scan this plan is for.
        filters = self.ctx.blackboard.get("analysis_filters")
        filters_derived_from = "caller"
        if filters is None:
            filters = goal_filters(self.ctx.goal, profile)
            filters_derived_from = "goal"
            self.ctx.blackboard["analysis_filters"] = filters
        factors = [self._factor(p, profile, stats) for p in filters]
        factors.sort(key=lambda p: (p["estimated_filter_factor"], not p["indexed"]))
        constants = set(profile["quality"].get("constant_columns", []))
        projected = [c["name"] for c in profile["columns"] if c["name"] not in constants]
        indexed = sorted({c for i in stats.get("indexes", []) for c in i.get("columns", [])})
        partitions = list(stats.get("partitioning", {}).get("columns", []))
        file_pushdown = stats.get("format") in {"parquet", "pq"}
        pushdown = [p for p in factors if file_pushdown or p["column"] in set(indexed + partitions)]
        strategy = ("partition_pruned_scan" if any(p["partition_prunable"] for p in pushdown)
                    else "index_filtered_scan" if any(p["indexed"] for p in pushdown)
                    else "projected_scan")
        selectivity = 1.0
        for item in factors:
            selectivity *= item["estimated_filter_factor"]
        plan = {"source_statistics": stats, "analysis_filters": list(filters),
            "filters_derived_from": filters_derived_from, "filter_factors": factors,
            "predicate_order": [p["column"] for p in factors], "projection_columns": projected,
            "indexed_columns": indexed, "partition_columns": partitions,
            "pushdown_filters": pushdown, "strategy": strategy,
            "estimated_rows_after_filters": round(profile["dataset"]["n_rows"] * selectivity),
            # The connector contract's size-capped catalogue for a database
            # source (empty for a file): recorded with the plan it informed.
            "planner_metadata": dict(self.ctx.blackboard.get("planner_metadata") or {})}
        self.ctx.blackboard["workload_optimization"] = plan
        write_json(self.ctx.path("workload_optimization.json"), plan)
        return (f"Optimized scan via {strategy}; evaluated {len(factors)} filter factor(s) "
                f"from the {filters_derived_from}; estimated "
                f"{plan['estimated_rows_after_filters']:,} rows after filters.",
                45.0, {"strategy": strategy, "n_filter_factors": len(factors),
                       "filters_derived_from": filters_derived_from,
                       "estimated_rows_after_filters": plan["estimated_rows_after_filters"]})

    @staticmethod
    def _factor(predicate, profile, stats):
        column = str(predicate.get("column", ""))
        operator = str(predicate.get("operator", "eq")).lower()
        col = next((c for c in profile["columns"] if c["name"] == column), {})
        factor = 1.0
        if operator in {"eq", "=", "=="}:
            top = {str(v["value"]): int(v["count"]) for v in col.get("top_values", [])}
            value = str(predicate.get("value"))
            factor = top.get(value, 0) / max(1, profile["dataset"]["n_rows"])
            if value not in top:
                factor = 1.0 / max(1, int(col.get("n_unique", 1)))
        elif operator == "in":
            factor = min(1.0, len(predicate.get("value") or []) / max(1, int(col.get("n_unique", 1))))
        elif operator in {"is_null", "null"}:
            factor = float(col.get("missing_pct", 0.0))
        elif operator in {"lt", "lte", "gt", "gte", "<", "<=", ">", ">="}:
            factor = 0.3333
        return {**predicate, "column": column, "operator": operator,
            "estimated_filter_factor": round(max(0.0, min(1.0, factor)), 6),
            "indexed": any(column in (i.get("columns") or []) for i in stats.get("indexes", [])),
            "partition_prunable": column in stats.get("partitioning", {}).get("columns", [])}
