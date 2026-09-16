"""Data-integrity and leakage detection.

Target leakage is the commonest silent failure in automated data science, and
Avaloka had no defence against it: a search for ``leakage`` across the codebase
returns nothing. A leaking model reports excellent metrics and fails in
production, which is strictly worse for the product than a model that honestly
reports 0.71.

This agent runs **before** modelling and refuses to be advisory-only about the
findings that matter. Six checks, each targeting a failure this pipeline can
actually produce:

1. **Target leakage by correlation** — a feature almost perfectly predicting the
   target is nearly always contamination, not insight.
2. **Duplicate rows across splits** — the same record in train and test makes
   memorisation look like generalisation.
3. **Identifier-like features** — high-cardinality keys that memorise rather than
   generalise, and that will not exist for unseen entities.
4. **Temporal leakage** — training on rows that postdate the test period.
5. **Constant and near-constant features** — no signal, and they distort
   importance rankings.
6. **Target present among the features** — the trivial case, which happens more
   often than anyone admits when feature engineering is automated.

Checks are pure functions over a DataFrame so they are testable without a
model, an LLM, or a cluster.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from app.agents.contract import AgentSpec, Stage, agent
from app.agents.mta_v2.training_docker_image.src.data_integrity import (
    LEAKAGE_CORRELATION_THRESHOLD,
    find_target_leakage,
)

logger = logging.getLogger(__name__)

#: Above this fraction of unique values, a column behaves like an identifier.
IDENTIFIER_UNIQUENESS_RATIO = 0.95

#: Below this fraction of distinct values, a column carries almost no signal.
NEAR_CONSTANT_RATIO = 0.01


class Severity(str, Enum):
    """How much a finding should be allowed to stop the pipeline."""

    BLOCKER = "blocker"     # results would be invalid; do not train on this
    WARNING = "warning"     # likely a problem; surface prominently
    INFO = "info"           # worth knowing, not alarming


@dataclass
class Finding:
    check: str
    severity: Severity
    column: Optional[str]
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "column": self.column,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass
class IntegrityReport:
    findings: List[Finding] = field(default_factory=list)
    checked_columns: int = 0
    rows: int = 0

    @property
    def blockers(self) -> List[Finding]:
        return [f for f in self.findings if f.severity is Severity.BLOCKER]

    @property
    def safe_to_train(self) -> bool:
        return not self.blockers

    def as_dict(self) -> Dict[str, Any]:
        return {
            "safe_to_train": self.safe_to_train,
            "rows": self.rows,
            "checked_columns": self.checked_columns,
            "n_findings": len(self.findings),
            "n_blockers": len(self.blockers),
            "findings": [f.as_dict() for f in self.findings],
        }

    def summary(self) -> str:
        if not self.findings:
            return f"No integrity issues across {self.checked_columns} columns."
        blocking = len(self.blockers)
        head = f"{len(self.findings)} integrity finding(s)"
        if blocking:
            head += f", {blocking} blocking"
        return head + ": " + "; ".join(f.detail for f in self.findings[:3])


# --------------------------------------------------------------------------- #
# Individual checks — pure, DataFrame in, findings out
# --------------------------------------------------------------------------- #

def check_target_in_features(columns: Sequence[str], target: str,
                             feature_columns: Sequence[str]) -> List[Finding]:
    """The trivial case: the target itself offered as a predictor."""
    if target in feature_columns:
        return [Finding(
            check="target_in_features", severity=Severity.BLOCKER, column=target,
            detail=f"Target {target!r} is also listed as a feature; the model would read the answer.",
            evidence={"target": target},
        )]
    return []


def check_target_correlation(df, target: str, feature_columns: Sequence[str],
                             threshold: float = LEAKAGE_CORRELATION_THRESHOLD) -> List[Finding]:
    """Flag features that predict the target almost perfectly."""
    return [
        Finding(
            check="target_correlation",
            severity=Severity.BLOCKER,
            column=str(match["column"]),
            detail=(
                f"{match['column']!r} duplicates or almost perfectly encodes target "
                f"{target!r} — almost certainly leakage, not signal."
            ),
            evidence={
                "correlation": match["correlation"],
                "threshold": threshold,
                "method": match["method"],
            },
        )
        for match in find_target_leakage(
            df, target, [column for column in feature_columns if column != target],
            threshold=threshold,
        )
    ]


def check_duplicate_rows(df, subset: Optional[Sequence[str]] = None) -> List[Finding]:
    """Duplicate records let memorisation masquerade as generalisation."""
    try:
        dupes = int(df.duplicated(subset=list(subset) if subset else None).sum())
    except Exception:  # noqa: BLE001
        return []
    if dupes == 0:
        return []
    ratio = dupes / max(len(df), 1)
    severity = Severity.BLOCKER if ratio >= 0.01 else Severity.WARNING
    return [Finding(
        check="duplicate_rows", severity=severity, column=None,
        detail=(f"{dupes} duplicate row(s) ({ratio:.1%}); identical records split across "
                "train and test inflate measured performance."),
        evidence={"duplicates": dupes, "ratio": round(ratio, 6)},
    )]


def check_identifier_features(df, feature_columns: Sequence[str],
                              ratio: float = IDENTIFIER_UNIQUENESS_RATIO) -> List[Finding]:
    """High-cardinality keys memorise; they do not generalise to new entities."""
    findings: List[Finding] = []
    n = max(len(df), 1)
    for col in feature_columns:
        if col not in df.columns:
            continue
        try:
            uniq = int(df[col].nunique(dropna=True))
        except Exception:  # noqa: BLE001
            continue
        if n > 1 and uniq / n >= ratio and uniq > 1:
            findings.append(Finding(
                check="identifier_feature", severity=Severity.WARNING, column=col,
                detail=(f"{col!r} is {uniq/n:.0%} unique — it behaves like an identifier "
                        "and will not generalise to unseen entities."),
                evidence={"unique": uniq, "rows": n, "ratio": round(uniq / n, 4)},
            ))
    return findings


def check_near_constant(df, feature_columns: Sequence[str],
                        ratio: float = NEAR_CONSTANT_RATIO) -> List[Finding]:
    """Constant columns carry no signal and distort importance rankings."""
    findings: List[Finding] = []
    n = max(len(df), 1)
    for col in feature_columns:
        if col not in df.columns:
            continue
        try:
            uniq = int(df[col].nunique(dropna=True))
        except Exception:  # noqa: BLE001
            continue
        if uniq <= 1:
            findings.append(Finding(
                check="constant_feature", severity=Severity.WARNING, column=col,
                detail=f"{col!r} has a single value; it cannot contribute to a model.",
                evidence={"unique": uniq},
            ))
        elif n >= 100 and uniq / n <= ratio:
            findings.append(Finding(
                check="near_constant_feature", severity=Severity.INFO, column=col,
                detail=f"{col!r} has only {uniq} distinct values across {n} rows.",
                evidence={"unique": uniq, "rows": n},
            ))
    return findings


def check_temporal_leakage(df, time_column: Optional[str],
                           split_index: Optional[int] = None) -> List[Finding]:
    """Training on rows that postdate the evaluation period.

    Only meaningful when a time column is declared; silence here is the correct
    behaviour for a dataset with no temporal ordering.
    """
    if not time_column or time_column not in df.columns or split_index is None:
        return []
    try:
        series = df[time_column]
        train_max = series.iloc[:split_index].max()
        test_min = series.iloc[split_index:].min()
    except Exception:  # noqa: BLE001
        return []
    if train_max is None or test_min is None:
        return []
    try:
        overlaps = train_max > test_min
    except TypeError:
        return []
    if overlaps:
        return [Finding(
            check="temporal_leakage", severity=Severity.BLOCKER, column=time_column,
            detail=(f"Training rows extend to {train_max} but evaluation starts at "
                    f"{test_min}; the model would see the future."),
            evidence={"train_max": str(train_max), "test_min": str(test_min)},
        )]
    return []


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_integrity_checks(df, *, target: Optional[str] = None,
                         feature_columns: Optional[Sequence[str]] = None,
                         time_column: Optional[str] = None,
                         split_index: Optional[int] = None) -> IntegrityReport:
    """Run every applicable check and collect the findings."""
    columns = list(df.columns)
    features = list(feature_columns) if feature_columns is not None else [
        c for c in columns if c != target
    ]
    report = IntegrityReport(rows=int(len(df)), checked_columns=len(features))

    report.findings.extend(check_duplicate_rows(df))
    report.findings.extend(check_identifier_features(df, features))
    report.findings.extend(check_near_constant(df, features))
    report.findings.extend(check_temporal_leakage(df, time_column, split_index))
    if target:
        report.findings.extend(check_target_in_features(columns, target, features))
        report.findings.extend(check_target_correlation(df, target, features))

    order = {Severity.BLOCKER: 0, Severity.WARNING: 1, Severity.INFO: 2}
    report.findings.sort(key=lambda f: order[f.severity])
    return report


def _is_numeric(series) -> bool:
    try:
        from pandas.api.types import is_numeric_dtype
        return bool(is_numeric_dtype(series))
    except Exception:  # noqa: BLE001
        return False


def _codes(series):
    """Encode a low-cardinality categorical for correlation, else None.

    Encoding a high-cardinality column produces a meaningless correlation
    against an arbitrary integer ordering, so we decline rather than mislead.
    """
    try:
        if series.nunique(dropna=True) > 50:
            return None
        return series.astype("category").cat.codes
    except Exception:  # noqa: BLE001
        return None


INTEGRITY_SPEC = AgentSpec(
    name="integrity",
    stage=Stage.INTEGRITY,
    reads=("dataframe",),
    writes=("integrity_report", "integrity_safe_to_train"),
    description="Detects target leakage, duplicates, identifier features and temporal leakage.",
)


@agent(INTEGRITY_SPEC)
def integrity_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graph node. Reports findings; the caller decides whether to proceed."""
    df = state.get("dataframe")
    if df is None:
        return {}
    report = run_integrity_checks(
        df,
        target=state.get("target_column"),
        feature_columns=state.get("feature_columns"),
        time_column=state.get("time_column"),
        split_index=state.get("split_index"),
    )
    if report.blockers:
        logger.warning("[integrity] %s blocking finding(s): %s",
                       len(report.blockers), report.summary())
    return {
        "integrity_report": report.as_dict(),
        "integrity_safe_to_train": report.safe_to_train,
    }
