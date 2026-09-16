"""Sampling Specialist — replaces the data scientist's sampling judgement.

Economic output: a statistically defensible working sample. The specialist
decides *whether* sampling is even appropriate (small data should never be
sampled) and, when it is, draws a stratified sample so downstream modelling is
both cheaper and representative. Every decision is recorded as an assumption.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from avaloka.fireflies.base import Firefly

# Above this row count, sampling starts to pay for itself on commodity CPU.
SAMPLE_THRESHOLD_ROWS = 100_000
DEFAULT_SAMPLE_ROWS = 50_000


class SamplingSpecialist(Firefly):
    name = "sampling_specialist"
    human_role = "Data scientist"

    def perform(self) -> tuple[str, float, dict[str, Any]]:
        frame: pd.DataFrame = self.ctx.blackboard["frame"]
        n_rows = len(frame)
        target = self.ctx.target

        decision: dict[str, Any] = {
            "sampled": False,
            "reason": "",
            "n_input_rows": n_rows,
            "n_working_rows": n_rows,
            "strategy": "none",
            "stratify_by": None,
        }

        # Respect the workload lane chosen up front: only the live (ONLINE) lane
        # analyses the full dataset; larger lanes get a defensible sample now and
        # the full pass is deferred to `avaloka batch` on Ray.
        workload = self.ctx.blackboard.get("workload")
        lane = workload.get("lane") if workload else None
        full_ok = (lane == "online") if lane else (n_rows <= SAMPLE_THRESHOLD_ROWS)

        if full_ok:
            decision["reason"] = (
                f"{n_rows:,} rows fits the live full-analysis lane; the full dataset is used "
                "so no information is discarded."
            )
            working = frame
        else:
            frac = min(1.0, DEFAULT_SAMPLE_ROWS / n_rows)
            stratify = target if (target and target in frame.columns
                                  and frame[target].nunique(dropna=True) <= 50) else None
            if stratify is not None:
                working = (
                    frame.groupby(stratify, group_keys=False)
                    .apply(lambda g: g.sample(frac=frac, random_state=42))
                )
                decision["strategy"] = "stratified"
                decision["stratify_by"] = stratify
            else:
                working = frame.sample(frac=frac, random_state=42)
                decision["strategy"] = "random"
            decision["sampled"] = True
            decision["n_working_rows"] = len(working)
            decision["reason"] = (
                f"{n_rows:,} rows exceeds {SAMPLE_THRESHOLD_ROWS:,}; a {decision['strategy']} "
                f"sample of {len(working):,} rows (seed=42) gives a representative, "
                "reproducible working set."
            )

        self.ctx.blackboard["working_frame"] = working.reset_index(drop=True)
        self.ctx.blackboard["sampling"] = decision

        return (
            f"Working sample: {decision['n_working_rows']:,} rows "
            f"({decision['strategy']}). {decision['reason']}",
            30.0,
            {"sampled": decision["sampled"], "strategy": decision["strategy"]},
        )
