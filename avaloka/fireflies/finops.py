"""FinOps — replaces the infrastructure / cost engineer.

Economic output: a cost-quality optimisation. FinOps reframes the candidate
panel as a business decision — is more computation worth it? — and produces the
Economical / Balanced / Maximum-quality options table with serving-cost
estimates and a recommendation. This is the "per-dollar data scientist" story
made concrete.
"""

from __future__ import annotations

from typing import Any

from avaloka.fireflies.base import Firefly
from avaloka.util import write_json

CPU_USD_PER_HOUR = 0.12          # commodity vCPU price for training-cost attribution
# Monthly serving cost model: a base endpoint fee plus a latency-driven term
# (slower models need more/larger replicas to hold throughput).
SERVING_BASE_USD = 18.0
SERVING_LATENCY_USD_PER_MS = 1.4


class FinOps(Firefly):
    name = "finops"
    human_role = "Infrastructure engineer"

    def perform(self) -> tuple[str, float, dict[str, Any]]:
        training = self.ctx.blackboard["training"]
        higher_is_better = training.metric not in {"rmse", "mae"}

        rows = []
        for c in training.candidates:
            train_cost = c.train_seconds / 3600.0 * CPU_USD_PER_HOUR
            serving = SERVING_BASE_USD + c.inference_ms_per_1k * SERVING_LATENCY_USD_PER_MS
            rows.append({
                "candidate": c.name,
                "validation_score": round(c.primary_score, 4),
                "metric": c.primary_metric,
                "training_cost_usd": round(train_cost, 4),
                "api_latency_ms": round(c.inference_ms_per_1k, 2),
                "est_monthly_serving_usd": round(serving, 2),
            })

        ranked = sorted(rows, key=lambda r: r["validation_score"], reverse=higher_is_better)
        cheapest = min(rows, key=lambda r: r["est_monthly_serving_usd"])
        best = ranked[0]
        # Balanced = best score among the cheaper half.
        cheaper_half = sorted(rows, key=lambda r: r["est_monthly_serving_usd"])[: max(1, len(rows) // 2 + 1)]
        balanced = sorted(cheaper_half, key=lambda r: r["validation_score"],
                          reverse=higher_is_better)[0]

        options = {
            "Economical": cheapest,
            "Balanced": balanced,
            "Maximum quality": best,
        }
        recommendation = self._recommend(options, higher_is_better)

        finops = {"options": options, "all_candidates": rows, "recommendation": recommendation,
                  "assumptions": {"cpu_usd_per_hour": CPU_USD_PER_HOUR,
                                  "serving_base_usd": SERVING_BASE_USD,
                                  "serving_latency_usd_per_ms": SERVING_LATENCY_USD_PER_MS}}
        self.ctx.blackboard["finops"] = finops
        write_json(self.ctx.path("cost_quality.json"), finops)

        return (
            f"Cost-quality analysis across {len(rows)} candidate(s); recommended "
            f"'{recommendation['choice']}'.",
            45.0,
            {"recommended": recommendation["choice"]},
        )

    def _recommend(self, options: dict[str, dict[str, Any]], higher: bool) -> dict[str, Any]:
        eco, bal, mx = options["Economical"], options["Balanced"], options["Maximum quality"]
        gain_bal = bal["validation_score"] - eco["validation_score"]
        gain_max = mx["validation_score"] - bal["validation_score"]
        if not higher:  # rmse/mae: lower is better, invert sense
            gain_bal, gain_max = -gain_bal, -gain_max
        cost_ratio = mx["est_monthly_serving_usd"] / max(0.01, eco["est_monthly_serving_usd"])

        if gain_bal <= 0.005:
            choice, text = "Economical", (
                "Higher-cost models add negligible quality; the Economical option is the "
                "rational default.")
        elif gain_max < gain_bal * 0.5 and cost_ratio > 2:
            choice, text = "Balanced", (
                f"Balanced gains {gain_bal:+.3f} over Economical. Maximum quality adds only "
                f"{gain_max:+.3f} more but is ~{cost_ratio:.1f}x the serving cost.")
        else:
            choice, text = "Maximum quality", (
                "The quality gain justifies the additional serving cost for this objective.")
        return {"choice": choice, "rationale": text}
