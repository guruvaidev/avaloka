"""Deterministic A/B experiment design and analysis with causal guardrails."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentPlan:
    treatment_column: str
    outcome: str
    control: Any
    treatment: Any
    assignment_unit: str | None = None
    metric: str = "mean"  # mean | proportion
    randomized: bool = False
    alpha: float = 0.05
    expected_treatment_share: float | None = None


@dataclass(frozen=True)
class ExperimentResult:
    status: str  # pass | warn | cannot_verify
    effect: float | None
    relative_effect: float | None
    confidence_interval: tuple[float, float] | None
    control_value: float | None
    treatment_value: float | None
    control_n: int
    treatment_n: int
    sample_ratio_mismatch: bool | None
    diagnostics: tuple[dict[str, Any], ...]
    #: Power of the test that was run (see :func:`power_analysis`); None when
    #: the analysis stopped before an effect was estimated.
    power: dict[str, Any] | None = None


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_quantile(probability: float) -> float:
    try:
        from scipy.stats import norm
        return float(norm.ppf(probability))
    except Exception:  # pragma: no cover - optional dependency boundary
        table = {0.975: 1.959963984540054, 0.8: 0.8416212335729143, 0.9: 1.2815515655446004}
        if probability in table:
            return table[probability]
        raise RuntimeError("a quantile other than the defaults requires scipy")


def power_analysis(
    standard_error: float,
    effect: float,
    control_n: int,
    treatment_n: int,
    *,
    alpha: float = 0.05,
    target_power: float = 0.8,
) -> dict[str, Any]:
    """What the test could detect, stated three ways.

    ``achieved_power`` is the two-sided normal approximation at the observed
    effect -- read as "a true effect this size would be found this often",
    not as evidence about this effect. ``minimum_detectable_effect`` is what
    these arm sizes resolve at the target power, and
    ``required_n_per_arm`` is what the observed effect would have needed; the
    last two are the numbers to plan the next experiment with.
    """
    z_alpha = _normal_quantile(1 - alpha / 2)
    z_power = _normal_quantile(target_power)
    if standard_error <= 0:
        achieved = 1.0 if effect else None
        mde = 0.0
    else:
        achieved = _normal_cdf(abs(effect) / standard_error - z_alpha)
        mde = (z_alpha + z_power) * standard_error
    per_arm = (control_n + treatment_n) / 2.0
    required = None
    if effect and standard_error > 0:
        # se^2 scales as 1/n per arm, so the n for the observed effect follows
        # from the n that produced this se.
        required = int(math.ceil(per_arm * ((z_alpha + z_power) * standard_error / abs(effect)) ** 2))
    return {
        "achieved_power": None if achieved is None else round(float(achieved), 4),
        "minimum_detectable_effect": round(float(mde), 6),
        "required_n_per_arm_for_observed_effect": required,
        "alpha": alpha,
        "target_power": target_power,
    }


def _normal_interval(effect: float, standard_error: float, alpha: float) -> tuple[float, float]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    # The initial engine supports the conventional 95% interval exactly. Other
    # alpha values require scipy rather than silently using the wrong critical value.
    if not math.isclose(alpha, 0.05, rel_tol=0, abs_tol=1e-12):
        try:
            from scipy.stats import norm
            critical = float(norm.ppf(1 - alpha / 2))
        except Exception as exc:  # pragma: no cover - optional dependency boundary
            raise RuntimeError("non-0.05 alpha requires scipy") from exc
    else:
        critical = 1.959963984540054
    return effect - critical * standard_error, effect + critical * standard_error


def analyze_ab_test(frame: pd.DataFrame, plan: ExperimentPlan) -> ExperimentResult:
    """Analyze a two-arm randomized experiment; refuse unsupported causal claims."""
    required = [plan.treatment_column, plan.outcome]
    if plan.assignment_unit:
        required.append(plan.assignment_unit)
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"experiment inputs not found: {missing}")
    if not plan.randomized:
        return ExperimentResult(
            status="cannot_verify", effect=None, relative_effect=None,
            confidence_interval=None, control_value=None, treatment_value=None,
            control_n=0, treatment_n=0, sample_ratio_mismatch=None,
            diagnostics=({"kind": "causal_guard", "detail": "Random assignment was not established."},),
        )

    data = frame[required].dropna().copy()
    if plan.assignment_unit and data[plan.assignment_unit].duplicated().any():
        raise ValueError("assignment unit occurs more than once; aggregate to assignment grain first")
    control = pd.to_numeric(
        data.loc[data[plan.treatment_column] == plan.control, plan.outcome], errors="coerce"
    ).dropna()
    treatment = pd.to_numeric(
        data.loc[data[plan.treatment_column] == plan.treatment, plan.outcome], errors="coerce"
    ).dropna()
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("each experiment arm requires at least two valid outcomes")

    if plan.metric == "proportion":
        if not set(pd.concat([control, treatment]).unique()).issubset({0, 1}):
            raise ValueError("proportion outcome must be binary 0/1")
        control_value, treatment_value = float(control.mean()), float(treatment.mean())
        variance = control_value * (1 - control_value) / len(control)
        variance += treatment_value * (1 - treatment_value) / len(treatment)
    elif plan.metric == "mean":
        control_value, treatment_value = float(control.mean()), float(treatment.mean())
        variance = float(control.var(ddof=1) / len(control) + treatment.var(ddof=1) / len(treatment))
    else:
        raise ValueError(f"unsupported experiment metric: {plan.metric}")

    effect = treatment_value - control_value
    relative = effect / abs(control_value) if control_value else None
    standard_error = math.sqrt(max(0.0, variance))
    interval = _normal_interval(effect, standard_error, plan.alpha)
    power = power_analysis(standard_error, effect, len(control), len(treatment), alpha=plan.alpha)
    srm: bool | None = None
    diagnostics: list[dict[str, Any]] = []
    if plan.expected_treatment_share is not None:
        expected = float(plan.expected_treatment_share)
        if not 0 < expected < 1:
            raise ValueError("expected_treatment_share must be in (0, 1)")
        total = len(control) + len(treatment)
        expected_t, expected_c = total * expected, total * (1 - expected)
        chi_square = (len(treatment) - expected_t) ** 2 / expected_t
        chi_square += (len(control) - expected_c) ** 2 / expected_c
        # df=1, p<0.01 corresponds to 6.635; use a conservative integrity gate.
        srm = chi_square > 6.635
        diagnostics.append({"kind": "sample_ratio", "chi_square": chi_square, "threshold": 6.635})
    status = "warn" if srm else "pass"
    return ExperimentResult(
        status=status, effect=effect, relative_effect=relative,
        confidence_interval=interval, control_value=control_value,
        treatment_value=treatment_value, control_n=len(control),
        treatment_n=len(treatment), sample_ratio_mismatch=srm,
        diagnostics=tuple(diagnostics), power=power,
    )


def minimum_sample_size_per_arm(
    baseline_std: float,
    minimum_detectable_effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Normal-approximation sample size for a balanced two-arm mean test."""
    if baseline_std <= 0 or minimum_detectable_effect <= 0:
        raise ValueError("baseline_std and minimum_detectable_effect must be positive")
    try:
        from scipy.stats import norm
        z_alpha = float(norm.ppf(1 - alpha / 2))
        z_power = float(norm.ppf(power))
    except Exception:  # pragma: no cover - deterministic common defaults
        if not (math.isclose(alpha, 0.05) and math.isclose(power, 0.8)):
            raise RuntimeError("non-default alpha/power requires scipy")
        z_alpha, z_power = 1.959963984540054, 0.8416212335729143
    return int(math.ceil(2 * ((z_alpha + z_power) * baseline_std / minimum_detectable_effect) ** 2))
