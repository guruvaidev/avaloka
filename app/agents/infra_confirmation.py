"""Provisioning needs an explicit request and the user's confirmation.

The planner used to route any message containing "deploy", "deployment",
"gcp", "aws" or "kubernetes" into ``provision_infra`` and reply "Got it —
provisioning GCP infrastructure for your deployment now", so "analyze our AWS
costs" or "deployment frequency by team" created a cluster and, through the
python-docker path, an unauthenticated exec service. Now:

* :func:`explicit_request` recognises an imperative provisioning request
  ("deploy this to GCP", "provision a GKE cluster"), not a question that
  merely names a cloud;
* the planner then asks (:func:`confirmation_prompt`) and records the request
  on the durable ``pending_clarification`` channel with type
  :data:`PENDING_TYPE`, so it survives to the next turn;
* only a reply :func:`is_affirmative` sets ``infrastructure_request``; the
  reply says provisioning is starting (it is: ``provision_infra`` runs next in
  the same turn), and :func:`outcome_report` says afterwards what happened.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

PENDING_TYPE = "infrastructure_confirmation"
#: Set on confirmation; the planner reports the outcome when provision_infra loops back.
REPORT_TYPE = "infrastructure_report"

#: Start of the message or of a clause ("Looks good, deploy to GCP").
_LEAD = r"(?:^|[.,;:!]\s*|\band\s+(?:then\s+)?)\s*(?:(?:please|kindly|now|so|ok|okay|then)[\s,]+|(?:can|could|would|will)\s+you\s+|(?:let'?s|i\s+want\s+to|i'?d\s+like\s+to|go\s+ahead\s+and)\s+)*"
_VERB_RE = re.compile(_LEAD + r"(?P<verb>deploy(?:ment)?|provision|spin\s+up|stand\s+up|set\s+up|launch)\b(?P<rest>.*)$",
                      re.IGNORECASE | re.DOTALL)
#: Verbs that name provisioning by themselves; the others ("set up a connection
#: to AWS") only provision when infrastructure is their object.
_DEPLOY_VERB_RE = re.compile(r"^(?:deploy(?:ment)?|provision)$", re.IGNORECASE)
#: A cloud named as the destination: "to GCP", "on AWS", "into kubernetes".
_DESTINATION_RE = re.compile(
    r"\b(?:to|on|in|onto|into)\s+(?:a\s+|an\s+|the\s+|my\s+|our\s+)?"
    r"(?P<target>gcp|google\s+cloud|gke|aws|amazon(?:\s+web\s+services)?|eks|kubernetes|k8s|cloud|cluster)\b",
    re.IGNORECASE,
)
#: Infrastructure named as the object: "provision a GKE cluster", "spin up infrastructure".
_OBJECT_RE = re.compile(r"\b(?P<target>gke|eks|kubernetes|k8s|cluster|infra(?:structure)?)\b", re.IGNORECASE)
_GCP_RE = re.compile(r"\b(?:gcp|google\s+cloud|gke)\b", re.IGNORECASE)
_AWS_RE = re.compile(r"\b(?:aws|amazon|eks)\b", re.IGNORECASE)

_YES_RE = re.compile(r"^\s*(?:yes|y|yep|yeah|sure|confirm(?:ed)?|go\s+ahead|proceed|do\s+it|ok(?:ay)?)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(?:no|not|don'?t|do\s+not|cancel|stop|never|abort)\b", re.IGNORECASE)


def _platform(text: str) -> str:
    if _GCP_RE.search(text):
        return "gcp"
    if _AWS_RE.search(text):
        return "aws"
    return (os.getenv("DEFAULT_INFRA_PLATFORM") or "gcp").strip().lower() or "gcp"


def explicit_request(text: str) -> Optional[Dict[str, Any]]:
    """The infrastructure request ``text`` explicitly makes, or None.

    Explicit means an imperative provisioning verb at the start of the
    message or of a clause: "deploy"/"provision" with a cloud as destination
    or infrastructure as object; "set up"/"spin up"/"stand up"/"launch" only
    with infrastructure as object.
    A question about data that names a cloud is not one.
    """
    for m in _VERB_RE.finditer(text or ""):
        rest = m.group("rest")
        destination = _DESTINATION_RE.search(rest) and _DEPLOY_VERB_RE.match(m.group("verb"))
        if destination or _OBJECT_RE.search(rest):
            return {"type": _platform(text), "app_type": "python-docker"}
    return None


#: What joins the provisioning clause to the rest of a compound request:
#: "deploy this on GCP and then analyze average sales by region".
_THEN_RE = re.compile(
    r"^(?:\s+(?!and\b|then\b)[\w-]+){0,4}?\s*[,;.]?\s*(?:and\s+then|and|then|after\s+that|afterwards)[\s,]+(?P<rest>\S.*)$",
    re.IGNORECASE | re.DOTALL,
)


def remainder(text: str) -> Optional[str]:
    """The request that follows the provisioning clause, or None.

    "deploy this on GCP and then analyze average sales by region" ->
    "analyze average sales by region". It is carried on the pending
    confirmation and planned once the provisioning outcome is reported, so a
    compound request is not cut in half by the confirmation step.
    """
    for m in _VERB_RE.finditer(text or ""):
        rest = m.group("rest")
        target = (_DESTINATION_RE.search(rest) if _DEPLOY_VERB_RE.match(m.group("verb")) else None) \
            or _OBJECT_RE.search(rest)
        if not target:
            continue
        tail = _THEN_RE.match(rest[target.end():])
        if tail:
            follow = tail.group("rest").strip()
            if follow and not explicit_request(follow):
                return follow
        return None
    return None


def is_affirmative(text: str) -> bool:
    return bool(_YES_RE.match(text or "")) and not _NO_RE.search(text or "")


def is_negative(text: str) -> bool:
    return bool(_NO_RE.match((text or "").strip())) or (text or "").strip().lower() in {"n", "nope"}


def pending(request: Dict[str, Any], resume: Optional[str] = None) -> Dict[str, Any]:
    out = {"type": PENDING_TYPE, "infrastructure_request": dict(request)}
    if resume:
        out["resume"] = resume
    return out


def report_pending(resume: Optional[str] = None) -> Dict[str, Any]:
    """Set on "yes": the loop-back reports the outcome, then plans ``resume``."""
    out = {"type": REPORT_TYPE}
    if resume:
        out["resume"] = resume
    return out


def confirmation_prompt(request: Dict[str, Any], resume: Optional[str] = None) -> str:
    platform = str(request.get("type") or "gcp").upper()
    then = (f"\n\nEither way I will then continue with the rest of your request: \"{resume}\"."
            if resume else "")
    return (
        f"You asked me to provision {platform} infrastructure. Nothing has been provisioned yet.\n\n"
        f"Provisioning can create resources in your {platform} account (for example a Kubernetes "
        f"cluster), which may cost money. Reply **yes** to go ahead, or anything else to cancel."
        + then
    )


def started_message(request: Dict[str, Any], resume: Optional[str] = None) -> str:
    platform = str(request.get("type") or "gcp").upper()
    then = f" Then I will continue with: \"{resume}\"." if resume else ""
    return (f"Confirmed: starting {platform} provisioning now. "
            f"I will report what was set up, or what failed, when it finishes.{then}")


CANCELLED_MESSAGE = "Cancelled. Nothing was provisioned."


def outcome_report(provisioned: Optional[Dict[str, Any]]) -> str:
    """What the provisioning run did, from infra_agent_node's own record of it."""
    if not isinstance(provisioned, dict):
        return "Provisioning did not run, so nothing was provisioned."
    status = str(provisioned.get("status") or "unknown")
    platform = str(provisioned.get("type") or "").upper() or "the requested platform"
    lines = [f"Provisioning on {platform} finished with status **{status}**."]
    for step in provisioned.get("details") or []:
        if isinstance(step, dict):
            lines.append(f"- {step.get('step_name') or step.get('step') or 'step'}: "
                         f"{step.get('status')}. {step.get('message') or ''}".rstrip())
    return "\n".join(lines)
