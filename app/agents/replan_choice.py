"""Missing values in the measured column: ask, then resume on the answer.

The owner's ruling: when the column a calculation measures (the one it sums
or averages) has missing values, the replan loop (app/agents/replan.py)
neither fills them nor leaves them out on its own. It asks:

    "12 of 1,240 rows have no value in revenue, the column this calculation
     measures. Should I fill them with the median, or leave them out?"

and the next turn resumes on the answer. The same question, with the same
resume, ends a turn where a repair would pass one of the loop's limits.

This is the question-and-resume pattern of the infrastructure confirmation
(``app/agents/infra_confirmation.py`` on fix/2.x-ray-seed-needs-confirmation
and fix/2.x-compound-deploy-resumes), which is the planner's own pattern for
``filter_value`` / ``metric_choice`` on this base:

* **pause** -- the question is a plain, code-written reply, and the pending
  question is recorded on the durable ``pending_clarification`` channel with
  its own ``type`` (:data:`PENDING_TYPE`);
* **persist** -- ``server.send_message`` already saves ``pending_clarification``
  to the session and loads it into the next turn; nothing new is stored;
* **resume** -- ``avaloka_agent_node`` hands a reply to a pending planner
  question to the planner, and ``plan_etl_job`` resolves it first thing
  (:func:`resolve`): a choice resumes the run, "cancel" ends it, anything
  else is a new request and the question lapses.

What resumes is exactly what was asked about: the pending record carries the
code that failed, a fingerprint of the dataset it read and the counts in the
question. A resume whose dataset changed, whose code is gone, or which is too
old is refused, and nothing is filled or left out (:func:`check`).

No answer is possible on a scheduled or background run. There the loop does
not guess: it ends with the plain explanation and the question, and a run can
carry the answer up front in ``AVALOKA_REPLAN_MEASURED_CHOICE``
(``median`` or ``leave_out``).
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

PENDING_TYPE = "replan_measured_choice"

MEDIAN = "median"
LEAVE_OUT = "leave_out"
CHOICES = (MEDIAN, LEAVE_OUT)
NO_ANSWER = "no_answer"
CANCEL = "cancel"

#: How long a question may wait for its answer.
PENDING_TTL_S = int(os.getenv("AVALOKA_REPLAN_PENDING_TTL_S", str(24 * 3600)))
#: The run's answer given up front, for runs no one can answer.
CHOICE_ENV = "AVALOKA_REPLAN_MEASURED_CHOICE"

_MEDIAN_RE = re.compile(r"\b(?:median|fill|impute|fill\s+(?:them|it)\s+in)\b", re.IGNORECASE)
_LEAVE_RE = re.compile(r"\b(?:leave(?:\s+(?:them|it|those(?:\s+rows)?))?\s+out|exclude|drop|skip|remove|omit)\b",
                       re.IGNORECASE)
_CANCEL_RE = re.compile(r"^\s*(?:no|cancel|stop|never\s*mind|forget\s+it|neither|don'?t)\b", re.IGNORECASE)


def parse_choice(text: str) -> Optional[str]:
    """``median``, ``leave_out``, ``cancel`` or None (not an answer to the question)."""
    text = str(text or "").strip()
    if not text:
        return None
    median, leave = bool(_MEDIAN_RE.search(text)), bool(_LEAVE_RE.search(text))
    if median and not leave:
        return MEDIAN
    if leave and not median:
        return LEAVE_OUT
    if _CANCEL_RE.search(text) and not (median or leave):
        return CANCEL
    return None


def env_choice() -> Optional[str]:
    value = (os.getenv(CHOICE_ENV) or "").strip().lower().replace("-", "_").replace(" ", "_")
    return value if value in CHOICES else None


def interactive(state: Mapping[str, Any]) -> bool:
    """Whether someone can answer a question on this run.

    Not on a scheduled run (Celery's ``agent_task`` sets the scheduled-run
    context and carries ``schedule_id``), and not on a deferred background
    turn.
    """
    try:
        from app.core.scheduled_run_context import _CURRENT_RUN

        if _CURRENT_RUN.get() is not None:
            return False
    except Exception:  # noqa: BLE001
        pass
    return not (state.get("schedule_id") or state.get("analysis_defer_required")
                or state.get("replan_interactive") is False)


def _n(value: Any) -> str:
    return f"{int(value):,}"


def question_text(ask: Mapping[str, Any]) -> str:
    """The question, with code-written counts."""
    col = ask.get("column")
    n, total = int(ask.get("rows_affected") or 0), int(ask.get("rows_total") or 0)
    rows = "row has" if n == 1 else "rows have"
    why = ""
    if ask.get("reason") == "measured_choice":
        why = f", the column this calculation measures"
    elif ask.get("reason") == "impute_limit":
        why = f", more than the {float(ask.get('limit') or 0):.0%} I fill in without asking"
    elif ask.get("reason") == "exclude_limit":
        why = f"; leaving them out would drop more than the {float(ask.get('limit') or 0):.0%} of rows I leave out without asking"
    return (f"{_n(n)} of {_n(total)} {rows} no value in {col}{why}. "
            "Should I fill them with the median, or leave them out?")


def fingerprint(path: Optional[str]) -> Optional[str]:
    """The dataset as it was when the question was asked: size and content hash."""
    if not path:
        return None
    try:
        p = Path(str(path))
        digest = hashlib.sha256()
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return f"{p.stat().st_size}:{digest.hexdigest()[:16]}"
    except OSError:
        return None


def pending(ask: Mapping[str, Any], state: Mapping[str, Any], *, data_key: Optional[str],
            code_sha: str) -> Dict[str, Any]:
    """The record the next turn resumes from. Plain JSON: the session stores it."""
    return {
        "type": PENDING_TYPE,
        "question": question_text(ask),
        "column": ask.get("column"),
        "rows_affected": int(ask.get("rows_affected") or 0),
        "rows_total": int(ask.get("rows_total") or 0),
        "reason": ask.get("reason"),
        "code": (state.get("coder_definition") or {}).get("code") or "",
        "code_sha": code_sha,
        "data_source_location": state.get("data_source_location"),
        # The file the question was asked about, as the next turn will find it
        # (check() compares the same path's fingerprint).
        "data_fingerprint": fingerprint(state.get("data_source_location") or data_key),
        "asked_at": time.time(),
    }


def check(pending_record: Mapping[str, Any], state: Mapping[str, Any], *, now: Optional[float] = None) -> Optional[str]:
    """Why this pending question may not be resumed, or None when it may.

    A resume applies the answer to the run that was asked about, or not at
    all: a different dataset, a changed file, missing code or an old question
    is refused before anything runs.
    """
    if not isinstance(pending_record, Mapping) or pending_record.get("type") != PENDING_TYPE:
        return "not_pending"
    if not pending_record.get("code") or not pending_record.get("column"):
        return "incomplete"
    asked = float(pending_record.get("asked_at") or 0)
    if (now or time.time()) - asked > PENDING_TTL_S:
        return "expired"
    if str(pending_record.get("data_source_location") or "") != str(state.get("data_source_location") or ""):
        return "dataset_changed"
    current = fingerprint(state.get("data_source_location"))
    if not current or current != pending_record.get("data_fingerprint"):
        return "data_changed"
    return None


REFUSALS = {
    "expired": "That question has expired, so I haven't changed anything. Ask again and I'll rerun it.",
    "dataset_changed": ("You've switched datasets since I asked, so I haven't applied that choice. "
                        "Ask again on this dataset and I'll check it afresh."),
    "data_changed": ("The data has changed since I asked, so I haven't applied that choice: the counts "
                     "may no longer hold. Ask again and I'll check it afresh."),
    "incomplete": "I no longer have what that question was about, so I haven't changed anything. Ask again.",
    "not_pending": "There is no question waiting for that answer, so I haven't changed anything.",
}
CANCELLED = "Okay. I haven't filled or left out anything; nothing was changed."


def resolve(pending_record: Mapping[str, Any], reply: str, state: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The planner's resume step. Returns the state update, or None when the
    reply is not an answer (it is a new request, and the question lapses).

    A choice the record allows resumes the run: the failed code goes straight
    back to execution (``replan_resume``), and the replan loop applies the
    choice to the column it asked about.
    """
    from app.api import reply_segments as rs

    choice = parse_choice(reply)
    if choice is None:
        return None
    base = {"pending_clarification": None, "ready_to_code": False, "ready_to_summarize": False,
            "enable_training": False}
    messages = list(state.get("messages") or [])
    if choice == CANCEL:
        return {**base, "messages": messages + [rs.plain_message(content=CANCELLED)],
                "replan_measured_choice": {"column": pending_record.get("column"), "choice": NO_ANSWER,
                                           "question": pending_record.get("question")}}
    refusal = check(pending_record, state)
    if refusal:
        return {**base, "messages": messages + [rs.plain_message(content=REFUSALS[refusal])],
                "replan_resume": False}
    return {
        **base,
        "coder_definition": {"code": pending_record["code"]},
        "replan_resume": True,
        "replan_measured_choice": {
            "column": pending_record["column"], "choice": choice, "chosen_by": "user",
            "rows_affected": pending_record.get("rows_affected"),
            "rows_total": pending_record.get("rows_total"),
            "question": pending_record.get("question"),
        },
    }
