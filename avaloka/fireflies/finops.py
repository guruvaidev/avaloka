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
from avaloka.modeling import selection_key
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
        # Rank on the number the selection ranked on (the CV mean over the
        # shared folds, unless CV failed), so "Maximum quality" and the
        # selected model cannot disagree about which candidate is better.
        # This ranked on the holdout while the selection used CV.
        basis = getattr(training, "selection_basis", "holdout")
        key = {c.name: selection_key(c, basis, not higher_is_better) for c in training.candidates}

        rows = []
        for c in training.candidates:
            train_cost = c.train_seconds / 3600.0 * CPU_USD_PER_HOUR
            serving = SERVING_BASE_USD + c.inference_ms_per_1k * SERVING_LATENCY_USD_PER_MS
            rows.append({
                "candidate": c.name,
                "validation_score": round(c.cv_mean if basis == "cv" else c.primary_score, 4),
                "score_basis": basis,
                "holdout_score": round(c.primary_score, 4),
                "metric": c.primary_metric,
                "training_cost_usd": round(train_cost, 4),
                "api_latency_ms": round(c.inference_ms_per_1k, 2),
                "est_monthly_serving_usd": round(serving, 2),
            })

        def by_quality(r):
            return key[r["candidate"]]

        ranked = sorted(rows, key=by_quality, reverse=True)
        cheapest = min(rows, key=lambda r: r["est_monthly_serving_usd"])
        best = ranked[0]
        # Balanced = best score among the cheaper half.
        cheaper_half = sorted(rows, key=lambda r: r["est_monthly_serving_usd"])[: max(1, len(rows) // 2 + 1)]
        balanced = sorted(cheaper_half, key=by_quality, reverse=True)[0]

        options = {
            "Economical": cheapest,
            "Balanced": balanced,
            "Maximum quality": best,
        }
        recommendation = self._recommend(options, higher_is_better)
        recommendation.update(self._against_selection(
            options[recommendation["choice"]], {r["candidate"]: r for r in rows},
            training.best.name, training.metric, basis, higher_is_better))

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

    @staticmethod
    def _against_selection(option: dict[str, Any], rows: dict[str, dict[str, Any]],
                           selected: str, metric: str, basis: str,
                           higher: bool) -> dict[str, Any]:
        """Say how the recommendation relates to the selected model.

        The two answer different questions -- the selection is the best model
        on quality alone, the recommendation weighs quality against serving
        cost -- so they may differ, but never silently: the e2e run selected
        the ensemble while cost_quality.json recommended logistic regression
        and neither said why.
        """
        candidate = option["candidate"]
        if candidate == selected or selected not in rows:
            return {"candidate": candidate, "selected": selected, "agrees_with_selection": True,
                    "selection_note": f"The recommended option is the selected model '{selected}'."}
        sel = rows[selected]
        gap = option["validation_score"] - sel["validation_score"]
        if not higher:
            gap = -gap
        saving = sel["est_monthly_serving_usd"] - option["est_monthly_serving_usd"]
        label = "CV" if basis == "cv" else "holdout"
        note = (f"Selection picks '{selected}' on quality alone ({label} {metric} "
                f"{sel['validation_score']:.4f}). On cost this recommends '{candidate}' "
                f"({option['validation_score']:.4f}, {gap:+.4f}), which serves for "
                f"${abs(saving):.2f}/month {'less' if saving >= 0 else 'more'}. "
                f"The bundle's model.pkl is '{selected}'; choose '{candidate}' only if "
                f"that trade is acceptable.")
        return {"candidate": candidate, "selected": selected, "agrees_with_selection": False,
                "selection_note": note}
