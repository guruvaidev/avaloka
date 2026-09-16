"""Compose scanning, planning and explanation into one step the planner can call.

Three agents each answer part of "is this data usable?" — the PII scanner finds
personal data, the preparation agent decides what cleaning it needs, the
narrator says both in plain language. Calling them separately means every caller
re-implements the order and the policy, and the two policy decisions that matter
get made by accident.

They are made here instead, once:

**Personal data is scanned on every analysis.** Classification is cheap — a
sample of at most 1000 rows per column, no model, no network — and the cost of
not knowing is that personal data reaches an LLM prompt before anyone has
noticed it is personal. The scan is not optional; acting on it is.

**Nothing is applied without being asked.** This step produces a *proposal*: a
plan, a report, and an explanation of both. It does not transform the frame.
Cleaning changes what every later number means, and a user who did not ask for a
column to be filled should not discover it in a result. :func:`apply_proposal`
exists for when they say yes.

The separation matters more for PII than for cleaning. De-identifying without
being asked destroys the join keys someone may have been relying on; leaving it
undone while saying nothing hides a disclosure risk. Reporting always and
applying on request is the only combination that is honest in both directions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.agents.contract import AgentSpec, Stage, agent
from app.agents.pii_agent import PIIReport, Sensitivity, apply_deidentification, scan_dataframe
from app.agents.preparation_agent import (Objective, apply_preparation, fit_preparation,
                                          suggest_ratio_features)
from app.agents.preparation_narrator import narrate_headline, narrate_preparation

logger = logging.getLogger(__name__)


@dataclass
class PreparationProposal:
    """What Avaloka would do, and why — with nothing done yet."""

    plan: Dict[str, Any]
    pii: Dict[str, Any]
    explanation: str
    headline: str
    _plan_obj: Any = None
    _pii_obj: Optional[PIIReport] = None

    @property
    def needs_attention(self) -> bool:
        """Whether a person should look before anything runs."""
        return bool(self.plan.get("unimputable")) or any(
            f["sensitivity"] == Sensitivity.DIRECT.value
            for f in self.pii.get("findings", []))

    def as_dict(self) -> Dict[str, Any]:
        return {"plan": self.plan, "pii": self.pii, "explanation": self.explanation,
                "headline": self.headline, "needs_attention": self.needs_attention}


def propose_preparation(df, *, objective: Objective = Objective.BOTH,
                        target: Optional[str] = None,
                        columns: Optional[Sequence[str]] = None,
                        train_index: Optional[Sequence[Any]] = None,
                        ratio_features: Sequence[Any] = (),
                        dataset_name: Optional[str] = None) -> PreparationProposal:
    """Scan, plan and explain. Changes nothing."""
    # Fit on training rows when the caller knows which they are; see the
    # preparation agent for why fitting on the full frame is leakage.
    fit_frame = df.loc[list(train_index)] if train_index is not None else df

    pii_report = scan_dataframe(df, columns=columns)

    plan = fit_preparation(fit_frame, objective=objective, target=target, columns=columns)
    if ratio_features:
        plan = suggest_ratio_features(fit_frame, plan, ratio_features)

    plan_dict = plan.as_dict()
    pii_dict = pii_report.as_dict()
    return PreparationProposal(
        plan=plan_dict, pii=pii_dict,
        explanation=narrate_preparation(plan_dict, pii_dict, dataset_name=dataset_name),
        headline=narrate_headline(plan_dict, pii_dict),
        _plan_obj=plan, _pii_obj=pii_report)


def apply_proposal(df, proposal: PreparationProposal, *, deidentify: bool = False,
                   pii_overrides: Optional[Dict[str, Any]] = None,
                   pii_key: Optional[bytes] = None):
    """Apply an accepted proposal. De-identification stays opt-in even here.

    Returns a new frame; the input is not modified.
    """
    if proposal._plan_obj is None:
        raise ValueError("proposal carries no fitted plan; it cannot be applied")
    out = apply_preparation(df, proposal._plan_obj)
    if deidentify:
        if proposal._pii_obj is None:
            raise ValueError("proposal carries no PII report; cannot de-identify")
        out = apply_deidentification(out, proposal._pii_obj,
                                     overrides=pii_overrides, key=pii_key)
    return out


PREPARATION_PIPELINE_SPEC = AgentSpec(
    name="preparation_pipeline",
    stage=Stage.PREPARE,
    reads=("dataframe",),
    writes=("preparation_proposal", "preparation_explanation", "preparation_headline",
            "pii_report", "preparation_needs_attention"),
    description=("Scans for personal data, plans cleansing and feature engineering, and "
                 "explains both in plain language. Proposes only — applies nothing."),
)


@agent(PREPARATION_PIPELINE_SPEC)
def preparation_pipeline_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graph node. Runs after profiling; writes a proposal the planner can present.

    The explanation is written for the user, not for a log: the conversational
    agent can emit ``preparation_explanation`` verbatim.
    """
    df = state.get("dataframe")
    if df is None:
        return {}

    objective = state.get("objective") or Objective.BOTH
    try:
        objective = Objective(objective)
    except ValueError:
        objective = Objective.BOTH

    proposal = propose_preparation(
        df, objective=objective, target=state.get("target_column"),
        columns=state.get("feature_columns"), train_index=state.get("train_index"),
        ratio_features=state.get("ratio_features") or (),
        dataset_name=state.get("dataset_name"))

    logger.info("[preparation] %s", proposal.headline)
    return {
        "preparation_proposal": proposal.as_dict(),
        "preparation_explanation": proposal.explanation,
        "preparation_headline": proposal.headline,
        "pii_report": proposal.pii,
        "preparation_needs_attention": proposal.needs_attention,
    }
