"""Cross-validated evaluation with a mandatory baseline.

The MTA agents already compute accuracy, F1, RMSE, R² and MAE — that part is
real. What was missing is everything that makes those numbers *mean* something:

* **No cross-validation anywhere.** Searching the codebase for ``cross_val``,
  ``KFold`` or ``StratifiedKFold`` returns nothing. The only mention is an
  instruction in ``coder.py:897`` telling the model not to use ``KFold`` naively
  on grouped data — good advice with nothing enforcing it. A model could report
  0.94 accuracy from one lucky split and nothing could contest it.
* **No baseline.** A number with nothing to compare against is not evidence.
  0.94 accuracy is excellent on balanced data and *worse than useless* when 95%
  of rows are one class.
* **No uncertainty on the metric itself.** A single point estimate hides how
  much of the score is noise.

This agent addresses all three, and the baseline is not optional: every result
carries `beats_baseline`, because that is the claim a reader actually cares
about.

**Splitter choice is a correctness issue, not a preference.** Grouped data
needs ``GroupKFold`` or entity boundaries leak across folds; time series needs
``TimeSeriesSplit`` or the model trains on the future. :func:`choose_splitter`
picks from the declared shape of the data and records *why*, so the choice is
auditable rather than implicit.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.agents.contract import AgentSpec, Stage, agent

logger = logging.getLogger(__name__)

DEFAULT_FOLDS = 5
#: Below this many rows, k-fold estimates are too noisy to be worth reporting
#: as if they were stable.
MIN_ROWS_FOR_CV = 30


@dataclass
class MetricResult:
    """One metric, across folds."""

    name: str
    values: List[float]

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else float("nan")

    @property
    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        m = self.mean
        return math.sqrt(sum((v - m) ** 2 for v in self.values) / (len(self.values) - 1))

    def confidence_interval(self, z: float = 1.96) -> Tuple[float, float]:
        """Normal-approximation CI on the mean across folds.

        Folds are not fully independent, so this understates uncertainty
        slightly. It is reported as an interval rather than a p-value for
        exactly that reason — it is an honest indication of spread, not a
        significance test.
        """
        if len(self.values) < 2:
            return (self.mean, self.mean)
        half = z * self.std / math.sqrt(len(self.values))
        return (self.mean - half, self.mean + half)

    def as_dict(self) -> Dict[str, Any]:
        low, high = self.confidence_interval()
        return {
            "name": self.name,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "ci_low": round(low, 6),
            "ci_high": round(high, 6),
            "folds": [round(v, 6) for v in self.values],
        }


@dataclass
class EvaluationReport:
    task: str
    splitter: str
    splitter_reason: str
    n_folds: int
    n_rows: int
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    baseline: Dict[str, float] = field(default_factory=dict)
    primary_metric: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def beats_baseline(self) -> Optional[bool]:
        """Whether the model beats the trivial baseline by more than fold noise.

        The bar is the **confidence interval**, not the mean: the baseline must
        fall outside the interval, so an improvement smaller than fold-to-fold
        variation does not count as a win.

        A bare ``mean > baseline`` comparison is not good enough. A classifier
        that only ever predicts the majority class scores exactly the baseline,
        and floating-point noise then records it as "beats baseline by +0.0000"
        — which is the precise failure this agent exists to prevent.

        ``None`` when it cannot be determined — never optimistically True.
        """
        if not self.primary_metric or self.primary_metric not in self.metrics:
            return None
        if self.primary_metric not in self.baseline:
            return None
        metric = self.metrics[self.primary_metric]
        base = self.baseline[self.primary_metric]
        if metric.mean != metric.mean or base != base:  # NaN
            return None
        low, high = metric.confidence_interval()
        if _higher_is_better(self.primary_metric):
            return bool(low > base)
        return bool(high < base)

    @property
    def lift_over_baseline(self) -> Optional[float]:
        if self.beats_baseline is None:
            return None
        return round(self.metrics[self.primary_metric].mean - self.baseline[self.primary_metric], 6)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "splitter": self.splitter,
            "splitter_reason": self.splitter_reason,
            "n_folds": self.n_folds,
            "n_rows": self.n_rows,
            "primary_metric": self.primary_metric,
            "metrics": {k: v.as_dict() for k, v in self.metrics.items()},
            "baseline": {k: round(v, 6) for k, v in self.baseline.items()},
            "beats_baseline": self.beats_baseline,
            "lift_over_baseline": self.lift_over_baseline,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        if not self.primary_metric or self.primary_metric not in self.metrics:
            return "Evaluation produced no primary metric."
        m = self.metrics[self.primary_metric]
        low, high = m.confidence_interval()
        text = (f"{self.primary_metric} {m.mean:.4f} "
                f"(95% CI {low:.4f}–{high:.4f}) over {self.n_folds}-fold {self.splitter}")
        verdict = self.beats_baseline
        if verdict is None:
            text += "; no baseline comparison available"
        elif verdict:
            text += f"; beats baseline by {self.lift_over_baseline:+.4f}"
        else:
            text += (f"; DOES NOT BEAT the trivial baseline "
                     f"({self.baseline[self.primary_metric]:.4f})")
        return text


def _higher_is_better(metric: str) -> bool:
    return not any(k in metric.lower() for k in ("error", "loss", "rmse", "mae", "mse"))


# --------------------------------------------------------------------------- #
# Splitter selection — a correctness decision, recorded
# --------------------------------------------------------------------------- #

def choose_splitter(*, task: str, n_rows: int, groups: Optional[Sequence] = None,
                    time_ordered: bool = False, n_folds: int = DEFAULT_FOLDS) -> Tuple[str, str]:
    """Return ``(splitter_name, reason)``.

    Order matters: temporal structure outranks grouping, which outranks
    stratification. Getting this wrong silently invalidates every number
    downstream, which is why the reason travels with the choice.
    """
    if time_ordered:
        return ("TimeSeriesSplit",
                "data is time-ordered; random folds would train on the future")
    if groups is not None and len(set(groups)) > 1:
        return ("GroupKFold",
                "rows share entity groups; random folds would leak an entity across folds")
    if task == "classification":
        return ("StratifiedKFold",
                "classification; stratification keeps class balance stable across folds")
    return ("KFold", "regression with independent rows")


def baseline_scores(y: Sequence, task: str) -> Dict[str, float]:
    """Scores for the trivial model: majority class, or the mean.

    This is the bar every model must clear. Reported always, not on request.
    """
    values = [v for v in y if v is not None]
    if not values:
        return {}
    if task == "classification":
        counts: Dict[Any, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        majority = max(counts.values())
        acc = majority / len(values)
        return {"accuracy": acc, "f1_macro": _majority_f1_macro(counts, len(values))}
    mean = sum(values) / len(values)
    mse = sum((v - mean) ** 2 for v in values) / len(values)
    mae = sum(abs(v - mean) for v in values) / len(values)
    return {"rmse": math.sqrt(mse), "mae": mae, "r2": 0.0}


def _majority_f1_macro(counts: Dict[Any, int], total: int) -> float:
    """Macro-F1 of always predicting the majority class.

    Deliberately included: it collapses toward zero as classes multiply, which
    is the honest signal that accuracy alone hides on imbalanced data.
    """
    if not counts:
        return 0.0
    majority_class = max(counts, key=lambda k: counts[k])
    per_class = []
    for cls, n in counts.items():
        if cls == majority_class:
            precision = n / total
            recall = 1.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        else:
            f1 = 0.0
        per_class.append(f1)
    return sum(per_class) / len(per_class)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_model(estimator, X, y, *, task: str = "classification",
                   groups: Optional[Sequence] = None, time_ordered: bool = False,
                   n_folds: int = DEFAULT_FOLDS) -> EvaluationReport:
    """Cross-validate ``estimator`` and compare against the trivial baseline."""
    n_rows = len(y)
    splitter_name, reason = choose_splitter(
        task=task, n_rows=n_rows, groups=groups, time_ordered=time_ordered, n_folds=n_folds
    )
    report = EvaluationReport(
        task=task, splitter=splitter_name, splitter_reason=reason,
        n_folds=n_folds, n_rows=n_rows,
        primary_metric="accuracy" if task == "classification" else "rmse",
    )
    report.baseline = baseline_scores(list(y), task)

    if n_rows < MIN_ROWS_FOR_CV:
        report.warnings.append(
            f"only {n_rows} rows; cross-validated estimates are too noisy to be reliable"
        )
        report.n_folds = 0
        return report

    try:
        from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold,
                                             TimeSeriesSplit, cross_validate)
    except ImportError:
        report.warnings.append("scikit-learn not installed; cross-validation skipped")
        return report

    effective_folds = max(2, min(n_folds, n_rows // 2))
    if effective_folds != n_folds:
        report.warnings.append(f"reduced folds {n_folds} -> {effective_folds} for {n_rows} rows")
        report.n_folds = effective_folds

    splitters = {
        "TimeSeriesSplit": lambda: TimeSeriesSplit(n_splits=effective_folds),
        "GroupKFold": lambda: GroupKFold(n_splits=effective_folds),
        "StratifiedKFold": lambda: StratifiedKFold(n_splits=effective_folds, shuffle=True,
                                                   random_state=0),
        "KFold": lambda: KFold(n_splits=effective_folds, shuffle=True, random_state=0),
    }
    scoring = (["accuracy", "f1_macro"] if task == "classification"
               else ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"])

    try:
        cv = splitters[splitter_name]()
        scores = cross_validate(
            estimator, X, y,
            cv=cv, scoring=scoring,
            groups=groups if splitter_name == "GroupKFold" else None,
            error_score="raise",
        )
    except Exception as exc:  # noqa: BLE001 - report, never crash the analysis
        report.warnings.append(f"cross-validation failed: {type(exc).__name__}: {exc}")
        return report

    rename = {
        "neg_root_mean_squared_error": "rmse",
        "neg_mean_absolute_error": "mae",
    }
    for key, raw in scores.items():
        if not key.startswith("test_"):
            continue
        metric = key[len("test_"):]
        values = [float(v) for v in raw]
        if metric in rename:                      # sklearn returns these negated
            values = [-v for v in values]
            metric = rename[metric]
        report.metrics[metric] = MetricResult(name=metric, values=values)

    if report.beats_baseline is False:
        report.warnings.append(
            f"model does not beat the trivial baseline on {report.primary_metric}"
        )
    return report


EVALUATION_SPEC = AgentSpec(
    name="evaluation",
    stage=Stage.EVALUATE,
    reads=("trained_estimator", "eval_X", "eval_y"),
    writes=("evaluation_report", "evaluation_beats_baseline"),
    description="Cross-validates the model and compares it against a trivial baseline.",
)


@agent(EVALUATION_SPEC)
def evaluation_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graph node."""
    report = evaluate_model(
        state["trained_estimator"], state["eval_X"], state["eval_y"],
        task=state.get("task_kind", "classification"),
        groups=state.get("group_column_values"),
        time_ordered=bool(state.get("time_column")),
        n_folds=int(state.get("cv_folds", DEFAULT_FOLDS)),
    )
    logger.info("[evaluation] %s", report.summary())
    return {
        "evaluation_report": report.as_dict(),
        "evaluation_beats_baseline": report.beats_baseline,
    }
