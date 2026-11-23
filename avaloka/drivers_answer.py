""""What predicts X" answered from the shared driver ranking, never from code.

rc12 asked the clinic table "what are the strongest predictors of stroke?"
three times. Each run sent it to the coder, and each reply carried figures
the ranking does not hold: "ever married ... the last barely 0.003", raw
logistic coefficients ("work_type_Self-employed (0.32)"), "hypertension
follows at 0.093". The verifier withheld some sentences and let others
through. The figures were wrong because a model wrote them.

This module is the one rule and the one template every entry point uses: the
web chat and the planner graph (app/agents/drivers_route.py, which reads the
rows), the swarm's merged answer (avaloka/coordinator.py) and the CLI, whose
own drivers answer is :func:`avaloka.goal_analysis.analyze_goal`. It lives in
the CLI package so the swarm and the CLI can use it without the app.

THE ROUTING RULE (:func:`drivers_route`). A question goes this way when
(a) it asks what predicts, drives, affects, influences or determines an
    outcome, or for the predictors, drivers, (risk) factors or determinants
    of one
    (:data:`DRIVERS_QUESTION`);
(b) it does not explicitly ask for a model (:data:`MODEL_REQUEST`): "train a
    random forest and show feature importance", "fit a logistic regression"
    still go to the coder, and the claim verifier checks what comes back;
(c) its outcome resolves to a column of the rows in hand
    (:func:`avaloka.goal_analysis.driver_target`); and
(d) the shared ranking answers it (status ``answered``). Anything else falls
    through to the normal pipeline unchanged.
Wording that does not name a model ("strongest predictors", "what matters
most for") is not a model request, so it takes the deterministic route.
No LLM writes a word of the answer: every number is formatted from the ranking.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

#: (a) A drivers question: what predicts / drives / affects X, the
#: predictors / drivers / (risk) factors of X, which factors affect X.
DRIVERS_QUESTION = re.compile(
    r"\b(?:what|which)\b[^?.!;]{0,40}?\b(?:predicts?|drives?|affects?|influences?|determines?)\b"
    r"|\b(?:predictors?|drivers?|factors?|determinants?)\s+(?:of|for|behind|affecting)\b",
    re.IGNORECASE,
)

#: (b) The escape hatch: the question names a model, a model's output, or
#: the act of training one.
MODEL_REQUEST = re.compile(
    r"\b(?:train|fit|build|tune)\w*\b[^?.!;]{0,60}?\b(?:models?|regressions?|classifiers?|regressors?|"
    r"forests?|trees?|xgboost|lightgbm|catboost|gbm|networks?|svm|knn)\b"
    r"|\b(?:logistic|linear|lasso|ridge|poisson)\s+regressions?\b"
    r"|\brandom[- ]forests?\b|\bdecision[- ]trees?\b|\bgradient[- ]boost\w*\b"
    r"|\b(?:xgboost|lightgbm|catboost|shap)\b"
    r"|\b(?:feature|permutation)[- ]importances?\b|\bcoefficients?\b|\bmachine[- ]learning\b",
    re.IGNORECASE,
)

#: How many ranked columns the answer names; the CLI shows the same five.
SHOWN = 5

_SYMBOL = {"pearson_r": "r", "eta": "eta", "cramers_v": "Cramer's V"}


def wants_a_model(question: str) -> bool:
    """(b): the question explicitly asks for a model or its importances."""
    return bool(MODEL_REQUEST.search(question or ""))


def is_drivers_question(question: str) -> bool:
    """(a) and not (b): the question takes the deterministic route if its
    outcome resolves."""
    text = " ".join((question or "").split())
    return bool(DRIVERS_QUESTION.search(text)) and not wants_a_model(text)


def drivers_route(frame: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
    """The CLI's drivers answer for ``question`` over ``frame``, or None when
    the routing rule sends the question elsewhere."""
    if frame is None or frame.empty or not is_drivers_question(question):
        return None
    from avaloka.goal_analysis import analyze_goal, driver_target

    target = driver_target(question, {}, frame)
    if not target or target.get("column") not in frame.columns:
        return None
    result = analyze_goal(frame, question, {})
    if result.get("kind") != "drivers" or result.get("status") != "answered":
        return None
    return result


def _figure(d: Dict[str, Any]) -> str:
    return f"{_SYMBOL[d['measure']]} = {abs(d['strength']):.3f}"


def shown_drivers(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The ranked columns the answer names, strongest first: the ones the
    CLI's findings list (leakage set aside unless the question named it)."""
    metrics = result.get("metrics") or {}
    by_name = {d["column"]: d for d in metrics.get("drivers") or []}
    return [by_name[c] for c in metrics.get("shown") or [] if c in by_name][:SHOWN]


def drivers_table(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The ranking as table rows: the figures every sentence is made from."""
    return [{"rank": i, "column": d["column"], "measure": _SYMBOL[d["measure"]],
             "association": round(float(d["strength"]), 3), "in_the_data": d.get("detail") or ""}
            for i, d in enumerate(shown_drivers(result), 1)]


def table_markdown(rows: List[Dict[str, Any]]) -> List[str]:
    """The table every entry point shows, as markdown lines, from
    :func:`drivers_table` rows: r signed, eta and Cramer's V unsigned."""
    def figure(r: Dict[str, Any]) -> str:
        return f"{r['association']:+.3f}" if r["measure"] == "r" else f"{r['association']:.3f}"

    return (["| # | Column | Measure | Association | In the data |", "|---|---|---|---|---|"]
            + [f"| {r['rank']} | {r['column']} | {r['measure']} | {figure(r)} | {r['in_the_data']} |"
               for r in rows])


def narrate_drivers(result: Dict[str, Any], basis: str) -> str:
    """THE NARRATION TEMPLATE. Every figure is formatted from the ranking:

    "<top> has the strongest association with <target> (<m> = <v>), followed
    by <c2> (<m> = <v>), ... and <c5> (<m> = <v>). Each figure is one
    column's association with the outcome on its own and does not control
    for the other columns: it shows what moves with <target>, not what
    causes it." then the table, the CLI's leakage / near-outcome notes, and
    the basis line (N of M rows).
    """
    from avaloka.goal_analysis import association_label

    metrics = result.get("metrics") or {}
    shown = shown_drivers(result)
    target = (f"{metrics['target']} = {metrics['target_value']}" if metrics.get("target_value") is not None
              else str(metrics.get("target")))
    head = association_label(shown[0], target).split(";")[0]
    rest = [f"{d['column']} ({_figure(d)})" for d in shown[1:]]
    if rest:
        head += ", followed by " + (", ".join(rest[:-1]) + " and " + rest[-1] if len(rest) > 1 else rest[0])
    lines = [head + ". Each figure is one column's association with the outcome on its own and does not "
             f"control for the other columns: it shows what moves with {target}, not what causes it.", ""]
    lines += table_markdown(drivers_table(result))
    notes = [f for f in result.get("findings") or []
             if f.startswith("Set aside as possible leakage") or "almost the outcome itself" in f]
    lines += [""] + notes + ["Row identifiers are never ranked.", basis]
    return "\n".join(lines).strip()
