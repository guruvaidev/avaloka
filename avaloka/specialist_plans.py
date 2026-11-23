"""Which analytical specialist a goal calls for, read from the goal and the data.

The Statistician, Experimenter and ConformalCalibrator existed and were never
woken: ``missions.py`` inserted them only when ``statistical_plan``,
``experiment_plan`` or ``conformal_config`` was already on the blackboard,
and no production code wrote those keys. The AnalysisPlanner writes them
from here. The rules are deterministic and column-agnostic in the same way
``goal_analysis`` is: a goal selects a supported method and the columns it
names; when the columns cannot be resolved the answer is "no plan", written
with its reason, never a guess at what the user meant.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from avaloka.goal_analysis import is_categorical_code

#: Words that make a goal read as a two-arm comparison.
_AB_GOAL = re.compile(
    r"\b(a/b|ab[- ]test|variant|arm|arms|treatment|control|experiment|better than|"
    r"outperform\w*|lift|uplift|versus|vs\.?)\b", re.IGNORECASE)
#: Column names that describe an assignment to an arm.
_ARM_COLUMN = re.compile(
    r"(variant|arm|treatment|group|cohort|bucket|experiment|condition|assignment)",
    re.IGNORECASE)
#: "B better than A", "treatment vs control", "B compared with A".
_PAIR = re.compile(
    r"(?P<first>[\w.-]+)\s+(?:is\s+)?(?:better than|outperforms?|beats?|vs\.?|versus|against|"
    r"compared (?:to|with)|over)\s+(?:the\s+)?(?P<second>[\w.-]+)", re.IGNORECASE)
_CONTROL_LIKE = re.compile(r"^(control|baseline|a|0|false|old|current|default|holdout)$", re.IGNORECASE)

DEFAULT_COVERAGE = 0.9


def _mentioned(goal: str, columns: list[str]) -> list[str]:
    return [c for c in columns
            if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", goal, re.IGNORECASE)]


def _two_valued(series: pd.Series) -> list[Any]:
    values = series.dropna().unique().tolist()
    return values if len(values) == 2 else []


def _is_binary(series: pd.Series) -> bool:
    values = set(series.dropna().unique().tolist())
    return bool(values) and values <= {0, 1, True, False}


def experiment_plan_from_goal(goal: str, frame: pd.DataFrame) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """An ``ExperimentPlan`` (as a dict) for a goal that reads like an A/B question.

    Returns ``(plan, rationale)``; the plan is None when the goal is not a
    two-arm question or the arm and outcome columns cannot be resolved, and
    the rationale says which. The arm column is one the goal names with two
    values, or failing that one whose name says it holds an assignment. The
    arms are read from the goal ("B better than A": B is the treatment), or
    the control is the one called control/baseline/A.

    Randomisation cannot be read from a table. It is assumed when the data
    describes itself as an experiment -- an assignment-like column name, or
    the goal saying variant/arm/A/B/experiment -- and the assumption is
    written down so the report can carry it as a limitation.
    """
    columns = [str(c) for c in frame.columns]
    if not _AB_GOAL.search(goal):
        return None, {"planned": False, "reason": "The goal does not read as a two-arm comparison."}
    mentioned = _mentioned(goal, columns)

    # The arm column: named in the goal and called an assignment, else named
    # and categorical, else any column called an assignment. A lone 0/1
    # column the goal names is the outcome, not the arms, unless the goal
    # names a second two-valued column.
    two_valued = [c for c in mentioned if _two_valued(frame[c])]
    arm_column = (next((c for c in two_valued if _ARM_COLUMN.search(c)), None)
                  or next((c for c in two_valued if not _is_binary(frame[c])), None)
                  or next((c for c in columns if _ARM_COLUMN.search(c) and _two_valued(frame[c])), None)
                  or (two_valued[0] if len(two_valued) >= 2 else None))
    if arm_column is None:
        return None, {"planned": False,
                      "reason": "No two-valued column names the arms; name the assignment column."}
    arms = _two_valued(frame[arm_column])
    by_text = {str(v).lower(): v for v in arms}

    treatment = control = None
    for match in _PAIR.finditer(goal):
        first, second = match.group("first").lower(), match.group("second").lower()
        if first in by_text and second in by_text and first != second:
            treatment, control = by_text[first], by_text[second]
            break
    if treatment is None:
        ordered = sorted(arms, key=lambda v: str(v))
        control = next((v for v in ordered if _CONTROL_LIKE.match(str(v))), ordered[0])
        treatment = next(v for v in ordered if v != control)

    outcome_candidates = [
        c for c in mentioned
        if c != arm_column and (pd.api.types.is_numeric_dtype(frame[c])
                                or pd.api.types.is_bool_dtype(frame[c]) or _is_binary(frame[c]))
    ]
    if not outcome_candidates:
        return None, {"planned": False, "arm_column": arm_column,
                      "reason": "No numeric or binary outcome column is named in the goal."}
    after_on = re.search(r"\bon\s+([\w.-]+)", goal, re.IGNORECASE)
    outcome = next((c for c in outcome_candidates
                    if after_on and c.lower() == after_on.group(1).lower()), outcome_candidates[0])
    metric = "proportion" if _is_binary(frame[outcome]) else "mean"

    designed = bool(_ARM_COLUMN.search(arm_column)) or bool(re.search(
        r"\b(a/b|ab[- ]test|variant|arm|arms|experiment|randomi[sz]ed)\b", goal, re.IGNORECASE))
    plan = {
        "treatment_column": arm_column, "outcome": outcome,
        "control": control, "treatment": treatment,
        "metric": metric, "randomized": designed, "alpha": 0.05,
        "expected_treatment_share": 0.5,
    }
    rationale = {
        "planned": True,
        "reason": (f"The goal compares '{treatment}' against '{control}' in '{arm_column}' "
                   f"on '{outcome}' ({metric})."),
        "assumptions": [
            ("Assignment to the arms is taken as randomised because the data describes an "
             "experiment; confirm it was." if designed else
             "Random assignment was not established; the analysis will refuse a causal claim."),
            "A balanced design (50/50) is assumed for the sample-ratio check.",
        ],
    }
    return plan, rationale


_STAT_WORDS = (
    ("correlation", re.compile(r"\bcorrelat\w*\b", re.IGNORECASE)),
    ("chi_square", re.compile(r"\b(chi[- ]?squared?|chi2|independen(?:ce|t)|association between)\b",
                              re.IGNORECASE)),
    ("proportion", re.compile(r"\b(proportion|share|percentage|percent|rate)\b", re.IGNORECASE)),
    ("median", re.compile(r"\bmedian\b", re.IGNORECASE)),
    ("mean", re.compile(r"\b(average|mean)\b", re.IGNORECASE)),
)


def statistical_plan_from_goal(goal: str, frame: pd.DataFrame) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """A ``StatisticalPlan`` (as a dict) for a goal that names a statistic.

    Correlation and chi-square need two named columns (numeric, and
    categorical, respectively); proportion needs one named two-valued column
    and reads "of <value>" from the goal when the goal names a value; mean
    and median need one named numeric column and the mean carries a
    bootstrap interval. The first statistic the goal names wins.
    """
    columns = [str(c) for c in frame.columns]
    mentioned = _mentioned(goal, columns)
    numeric = [c for c in mentioned if pd.api.types.is_numeric_dtype(frame[c])
               and not pd.api.types.is_bool_dtype(frame[c])]
    categorical = [c for c in mentioned if not pd.api.types.is_numeric_dtype(frame[c])
                   or _is_binary(frame[c])]
    method = next((name for name, pattern in _STAT_WORDS if pattern.search(goal)), None)
    if method is None:
        return None, {"planned": False, "reason": "The goal names no statistic."}

    if method == "correlation":
        if len(numeric) < 2:
            return None, {"planned": False, "method": method,
                          "reason": "A correlation needs two numeric columns named in the goal."}
        kind = "spearman" if re.search(r"\bspearman\b", goal, re.IGNORECASE) else "pearson"
        plan = {"method": "correlation", "variables": [numeric[0], numeric[1]],
                "estimand": f"{kind} correlation of {numeric[0]} and {numeric[1]}",
                "parameters": {"kind": kind}}
    elif method == "chi_square":
        if len(categorical) < 2:
            return None, {"planned": False, "method": method,
                          "reason": "A chi-square test needs two categorical columns named in the goal."}
        plan = {"method": "chi_square", "variables": [categorical[0], categorical[1]],
                "estimand": f"independence of {categorical[0]} and {categorical[1]}",
                "parameters": {}}
    elif method == "proportion":
        column = next((c for c in mentioned if _two_valued(frame[c]) or _is_binary(frame[c])), None)
        if column is None:
            return None, {"planned": False, "method": method,
                          "reason": "A proportion needs a two-valued column named in the goal."}
        values = _two_valued(frame[column]) or sorted(frame[column].dropna().unique().tolist())
        named = next((v for v in values if re.search(rf"(?<!\w){re.escape(str(v))}(?!\w)", goal,
                                                       re.IGNORECASE)), None)
        if named is None:
            named = next((v for v in values if v in (1, True)), values[-1])
        plan = {"method": "proportion", "variables": [column],
                "estimand": f"share of rows with {column} = {named}",
                "parameters": {"equals": named}}
    else:
        # "Average fare_amount by payment_type": the measure is fare_amount;
        # payment_type is the grouping, and a column of category codes is
        # never the thing averaged ("mean of payment_type: 1.16").
        by = re.search(r"\bby\s+([\w.-]+)", goal, re.IGNORECASE)
        grouping = {c for c in mentioned if by and c.lower() == by.group(1).lower()}
        codes = [c for c in numeric if is_categorical_code(c, frame[c])]
        measures = [c for c in numeric if c not in codes and c not in grouping]
        if not measures:
            reason = (f"{', '.join(codes)} holds category codes, not quantities; a {method} of codes "
                      "is meaningless." if codes and not set(codes) <= grouping else
                      f"A {method} needs a numeric column named in the goal.")
            return None, {"planned": False, "method": method, "reason": reason}
        numeric = measures
        plan = {"method": method, "variables": [numeric[0]], "estimand": f"{method} of {numeric[0]}",
                "parameters": {"bootstrap": True, "confidence": 0.95} if method == "mean" else {}}
    return plan, {"planned": True, "method": plan["method"],
                  "reason": f"The goal asks for the {plan['estimand']}."}


def conformal_config_for(task: str, target_coverage: float = DEFAULT_COVERAGE) -> dict[str, Any] | None:
    """The calibration request for a train mission: a 90% interval or label set.

    Only the shape of the request; the Model Scientist fills in the
    calibration scores from the winner's holdout once there is a winner.
    """
    if task == "regression":
        kind = "regression"
    elif task.endswith("classification"):
        kind = "classification"
    else:
        return None
    return {"task": kind, "target_coverage": target_coverage, "method": "split"}
