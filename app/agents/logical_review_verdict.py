# logical_review_verdict.py
"""Structured record of what the LLM logical reviewer decided.

``validator.logical_semantic_validator_node`` asks a model whether generated
code is logically correct. In default mode a "not correct" verdict is reported
to the pipeline as ``code_validation_feedback: "APPROVED"``, and several other
outcomes (no reviewer configured, an unparseable reply) are let through the
same way. Before this module nothing recorded which of those happened, so
nobody could count how often the judge said "wrong" and was overruled, or how
often it returned no verdict at all.

This module is instrumentation only. It decides nothing: the node computes its
result exactly as before and attaches the fields built here.

Everything recorded is a boolean, ``None``, an integer counter, or a member of
the closed ``LogicalReviewDecision`` vocabulary. No model output, rationale,
code or column name can enter a field: ``record_logical_verdict`` coerces every
flag through ``bool`` and every decision through the enum. The values are plain
``bool`` / ``None`` / ``str`` (not Enum instances) so the LangGraph checkpointer
can serialize them.
"""
import logging
import threading
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class LogicalReviewDecision(str, Enum):
    """Which rule decided the outcome. One member per exit of the node."""

    # The reviewer said the code is logically correct.
    APPROVED = "approved"
    # First pass said fabricated; the recheck did not confirm it and judged
    # the code logically correct.
    APPROVED_ON_RECHECK = "approved_on_recheck"
    # The reviewer said NOT logically correct; default mode reported APPROVED.
    SOFT_FLAG_DEFAULT_MODE = "soft_flag_default_mode"
    # As above, but the fabrication flag (first reply or recheck) was not a
    # boolean, so fabrication was never resolved. Not a clean overrule.
    SOFT_FLAG_FABRICATION_UNREADABLE = "soft_flag_fabrication_unreadable"
    # AVALOKA_STRICT_LOGIC_VALIDATION=1 and the reviewer said not correct.
    BLOCKED_STRICT_MODE = "blocked_strict_mode"
    # Default mode; fabrication verdict confirmed by the recheck.
    BLOCKED_FABRICATION_CONFIRMED = "blocked_fabrication_confirmed"
    # Default mode; the recheck itself raised, so the first verdict was kept.
    BLOCKED_FABRICATION_RECHECK_FAILED = "blocked_fabrication_recheck_failed"
    # The reply was not a JSON object; the node lets the code through.
    NO_VERDICT_UNPARSEABLE = "no_verdict_unparseable"
    # The reply was a JSON object but its verdict was not a boolean (nor
    # "true"/"false", nor 1/0); the node lets the code through.
    NO_VERDICT_MALFORMED_VALUE = "no_verdict_malformed_value"
    # The reviewer call raised; the exception propagates out of the node.
    NO_VERDICT_LLM_ERROR = "no_verdict_llm_error"
    # No reviewer model is configured; the node reports APPROVED.
    SKIPPED_NO_REVIEWER = "skipped_no_reviewer"
    # A syntax/static error is already set; the node returns without reviewing.
    SKIPPED_UPSTREAM_ERROR = "skipped_upstream_error"


# State keys written by the node, in the order they are logged.
LOGICAL_REVIEW_VERDICT_KEYS = (
    "logical_review_decided_by",
    "logical_review_judged_correct",
    "logical_review_judged_fabricated",
    "logical_review_recheck_ran",
    "logical_review_recheck_confirmed",
    "logical_review_overruled",
)

# Every decision falls in exactly one outcome bucket. The four review buckets
# partition ``reviews``; ``not_reviewed`` (an upstream error meant no review
# was due) is counted in ``calls`` only, so a coder syntax error can never be
# read as the judge failing.
_D = LogicalReviewDecision
_OUTCOME = {
    _D.APPROVED: "approved_by_reviewer",
    _D.APPROVED_ON_RECHECK: "approved_by_reviewer",
    _D.SOFT_FLAG_DEFAULT_MODE: "overruled",
    _D.SOFT_FLAG_FABRICATION_UNREADABLE: "overruled",
    _D.BLOCKED_STRICT_MODE: "blocked",
    _D.BLOCKED_FABRICATION_CONFIRMED: "blocked",
    _D.BLOCKED_FABRICATION_RECHECK_FAILED: "blocked",
    _D.NO_VERDICT_UNPARSEABLE: "no_verdict",
    _D.NO_VERDICT_MALFORMED_VALUE: "no_verdict",
    _D.NO_VERDICT_LLM_ERROR: "no_verdict",
    _D.SKIPPED_NO_REVIEWER: "no_verdict",
    _D.SKIPPED_UPSTREAM_ERROR: "not_reviewed",
}

_TOTAL_KEYS = (
    "calls",
    "reviews",
    "approved_by_reviewer",
    "overruled",
    "no_verdict",
    "blocked",
    "not_reviewed",
    "judged_correct",
    "judged_not_correct",
    "fabrication_flagged",
    "fabrication_rechecked",
    "fabrication_confirmed",
)


class _LogicalReviewCounters:
    """Running totals for this process. Integers only; reset on restart."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: Dict[str, int] = {}
        self._by_decision: Dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._totals = {key: 0 for key in _TOTAL_KEYS}
            self._by_decision = {d.value: 0 for d in LogicalReviewDecision}

    def record(self, decision: LogicalReviewDecision, fields: dict) -> Dict[str, int]:
        outcome = _OUTCOME[decision]
        with self._lock:
            t = self._totals
            t["calls"] += 1
            self._by_decision[decision.value] += 1
            t[outcome] += 1
            if outcome != "not_reviewed":
                t["reviews"] += 1
            if fields["logical_review_judged_correct"] is True:
                t["judged_correct"] += 1
            elif fields["logical_review_judged_correct"] is False:
                t["judged_not_correct"] += 1
            if fields["logical_review_judged_fabricated"] is True:
                t["fabrication_flagged"] += 1
            if fields["logical_review_recheck_ran"] is True:
                t["fabrication_rechecked"] += 1
            if fields["logical_review_recheck_confirmed"] is True:
                t["fabrication_confirmed"] += 1
            return dict(t)

    def snapshot(self) -> dict:
        with self._lock:
            snap: dict = dict(self._totals)
            snap["by_decision"] = dict(self._by_decision)
            return snap


_COUNTERS = _LogicalReviewCounters()


def logical_review_counts() -> dict:
    """Running totals for this process, plus a per-decision breakdown.

    ``reviews == approved_by_reviewer + overruled + no_verdict + blocked``, so
    the overrule rate is ``overruled / reviews``. Per process, not per
    deployment: each API/worker process keeps its own totals.
    """
    return _COUNTERS.snapshot()


def reset_logical_review_counts() -> None:
    _COUNTERS.reset()


def _flag(value) -> Optional[bool]:
    return None if value is None else bool(value)


def record_logical_verdict(
    decided_by: LogicalReviewDecision,
    *,
    judged_correct: Optional[bool] = None,
    judged_fabricated: Optional[bool] = None,
    recheck_ran: bool = False,
    recheck_confirmed: Optional[bool] = None,
) -> dict:
    """Build the verdict fields for one exit of the node, count it, log it.

    ``judged_correct`` / ``judged_fabricated`` are the first reply as the node
    read it (by type; see ``validator._read_verdict_flag``), or ``None`` when
    no usable verdict was obtained. ``recheck_confirmed`` is ``None`` unless a recheck produced a
    reading. Raises ``ValueError`` for a decision outside the closed set.
    """
    decision = LogicalReviewDecision(decided_by)
    fields = {
        "logical_review_decided_by": decision.value,
        "logical_review_judged_correct": _flag(judged_correct),
        "logical_review_judged_fabricated": _flag(judged_fabricated),
        "logical_review_recheck_ran": bool(recheck_ran),
        "logical_review_recheck_confirmed": _flag(recheck_confirmed),
        # The reviewer's last word was "not correct" and the pipeline reported
        # APPROVED anyway: the default-mode soft flags.
        "logical_review_overruled": _OUTCOME[decision] == "overruled",
    }
    totals = _COUNTERS.record(decision, fields)
    logger.info(
        "logical_review_verdict decided_by=%s judged_correct=%s judged_fabricated=%s "
        "recheck_ran=%s recheck_confirmed=%s overruled=%s "
        "total_calls=%d total_reviews=%d total_approved_by_reviewer=%d "
        "total_overruled=%d total_no_verdict=%d total_blocked=%d total_not_reviewed=%d",
        *(fields[key] for key in LOGICAL_REVIEW_VERDICT_KEYS),
        totals["calls"], totals["reviews"], totals["approved_by_reviewer"],
        totals["overruled"], totals["no_verdict"], totals["blocked"], totals["not_reviewed"],
    )
    return fields
