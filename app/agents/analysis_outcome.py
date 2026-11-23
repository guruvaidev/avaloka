"""Did this turn's analysis produce a result? One answer for every reader.

The web e2e found an analysis that failed to read its Parquet input followed
by "I checked 4 claims in my answer against the numbers; all of them hold" and
"That's my answer to ...". The router knew the run had failed (it skipped the
chart and the narrator). The claim verifier did not: it checked the four
sentences of the failure message against the validator's dry run on the
sample, found nothing wrong with them, and the chat voice read "4 checked,
0 problems" as a verified answer. Each reader decided "did it work" for itself
and two of them never asked.

So the decision lives here, once:

    outcome     execution_result                       what may follow
    ----------  -------------------------------------  ---------------------------
    not_run     absent or empty (chat, training turn)  the turn's own evidence
    failed      status error/failed, or an error set   the failure, in plain words
    no_result   status success-like, but no rows       nothing is verified
    refused     status refused (a request declined)    nothing ran, nothing is verified
    partial     status partial, with rows              claims checked, never "answered"
    answered    status success-like, with rows         claims checked against rows

It is the CLI's verdict gate in the app's terms. avaloka/fireflies/validator.py
(#542) only gives PASS when ``goal_answered`` and ``goal_reconciliation`` both
hold: the goal produced figures, and the figures recompute from the data. The
app has no per-figure recomputation recipe -- its result is whatever table the
generated code wrote -- so "answered" here is "a successful step with rows",
and reconciliation is the claim verifier finding every cited number in those
rows. The notion is mirrored, not imported: the CLI's check runs on its
blackboard (frames and figure specs) and the app has neither. If the two ever
share a result shape, this is the place to replace with the import.
"""

from __future__ import annotations

import base64
import csv
import io
from typing import Any, Dict, List, Mapping, Optional

#: The statuses app/api/workflow.py::route_after_execution treats as a failed
#: run (it sends those turns straight to verification, skipping the chart).
FAILED_STATUSES = frozenset({"error", "failed"})
#: The statuses the API counts as a completed run (server.py, execution_succeeded).
SUCCESS_STATUSES = frozenset({"success", "succeeded", "completed", "done"})
#: A request the planner declined before anything ran (a security refusal).
REFUSED_STATUSES = frozenset({"refused"})
#: A run that computed some of what was asked and not the rest
#: (app/api/workflow.py::execute_structured_actions_node).
PARTIAL_STATUSES = frozenset({"partial"})

NOT_RUN = "not_run"
FAILED = "failed"
NO_RESULT = "no_result"
REFUSED = "refused"
PARTIAL = "partial"
ANSWERED = "answered"
#: Outcomes after which nothing in the reply may be called verified.
NOTHING_TO_VERIFY = frozenset({FAILED, NO_RESULT, REFUSED})


def _csv_data_uri_records(data_uri: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(data_uri, str) or not data_uri.startswith("data:text/csv;base64,"):
        return None
    try:
        text = base64.b64decode(data_uri.split("base64,", 1)[1]).decode()
        return list(csv.DictReader(io.StringIO(text)))
    except Exception:  # noqa: BLE001 - an undecodable file is no result
        return None


def result_records(state: Mapping[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """The rows the executed analysis produced, or None.

    Only the executed result counts. ``execution_output_data`` is the
    validator's dry run on the sample before execution; a number that appears
    there and not in the result is not a number the analysis produced.
    """
    output = state.get("output_json")
    if isinstance(output, list) and output:
        return output
    ofd = state.get("output_file_data")
    if isinstance(ofd, dict):
        decoded = _csv_data_uri_records(ofd.get("content"))
        if decoded:
            return decoded
    result = state.get("execution_result")
    if isinstance(result, dict):
        for key in ("output_json", "output_data"):
            value = result.get(key)
            if isinstance(value, list) and value:
                return value
    tables = state.get("output_tables")
    if isinstance(tables, list):
        rows = [row for table in tables if isinstance(table, dict)
                for row in (table.get("rows") or []) if isinstance(row, dict)]
        if rows:
            return rows
    return None


def execution_status(state: Mapping[str, Any]) -> str:
    result = state.get("execution_result")
    if not isinstance(result, dict):
        return ""
    return str(result.get("status") or "").strip().lower()


def analysis_outcome(state: Mapping[str, Any]) -> str:
    """``not_run``, ``failed``, ``refused``, ``no_result``, ``partial`` or
    ``answered`` for this turn."""
    status = execution_status(state)
    if status in FAILED_STATUSES or state.get("execution_error"):
        return FAILED
    if not status:
        return NOT_RUN
    if status in REFUSED_STATUSES:
        return REFUSED
    if status in PARTIAL_STATUSES and result_records(state):
        return PARTIAL
    if status in SUCCESS_STATUSES and result_records(state):
        return ANSWERED
    # A success with no rows, a dry run, a job still running: in each case
    # there is no computed result for a claim to be checked against.
    return NO_RESULT


# --------------------------------------------------------------------------- #
# Plain words for a failed step
# --------------------------------------------------------------------------- #

#: (markers in the error text, what the user is told). First match wins. The
#: user hears what went wrong in her terms -- the class and the traceback are
#: in the logs and on the trace step -- the way #522 words a failed training turn.
_PLAIN_REASONS = (
    (("ParserError", "UnicodeDecodeError", "EmptyDataError", "ArrowInvalid",
      "UnusableDataset", "Error tokenizing data", "Unsupported or unreadable"),
     "I couldn't read the data file"),
    (("timed out", "TimeoutExpired", "TimeoutError"),
     "the calculation ran past its time limit"),
    (("KeyError",), "the code asked for a column the data doesn't have"),
    (("ReadMapError",), "the code read a file that isn't one of this chat's datasets, or read one twice"),
    (("MemoryError",), "the data was too large to work on in memory"),
    (("returned None and did not create an output file", "without creating a fresh output",
      "zero-byte output"),
     "the code finished without producing a result"),
)
_DEFAULT_REASON = "the analysis code stopped with an error"


def plain_failure_reason(detail: Any) -> str:
    """What went wrong with a failed analysis step, in plain words."""
    text = str(detail or "")
    for markers, reason in _PLAIN_REASONS:
        if any(marker in text for marker in markers):
            return reason
    return _DEFAULT_REASON


def failure_detail(state: Mapping[str, Any]) -> str:
    """The error text a failed step left on the state, for classification."""
    result = state.get("execution_result") if isinstance(state.get("execution_result"), dict) else {}
    return "\n".join(str(part) for part in (
        result.get("message"), result.get("error"), state.get("execution_error"),
        state.get("execution_stderr"),
    ) if part)


class AnalysisStepFailed(RuntimeError):
    """An analysis step that ran and did not produce a result.

    The class name is the trace step's ``error_class`` (telemetry sends the
    class only); the message is the plain reason, never the stderr.
    """


def record_failed_step(name: str, node):
    """Wrap execution node *node* so a failed run is a failed trace step.

    The execution nodes answer the user on failure instead of raising, so the
    trace recorder marked them ``ok`` -- it only sees an exception or a fresh
    ``agent_errors`` entry (app.trace.recorder.recorded_failure). Every backend
    (local, Ray, Kubernetes) writes ``execution_result`` the same way, so one
    wrapper covers them instead of a record at each of their failure branches.
    """
    from functools import wraps

    from app.agents.contract import failure_record

    # Test doubles stay unwrapped, as app.trace.recorder.traced_node leaves
    # them: tests find a mocked node by its mock attributes.
    if node is None or hasattr(node, "_mock_return_value"):
        return node

    @wraps(node, updated=())
    def wrapped(state, *args, **kwargs):
        before = state.get("agent_errors") if isinstance(state, Mapping) else None
        before = list(before) if isinstance(before, list) else []
        result = node(state, *args, **kwargs)
        if not isinstance(result, dict) or analysis_outcome(result) != FAILED:
            return result
        # A full-state node returns the list; a partial-update node leaves it
        # out, and what earlier steps recorded must survive the append.
        errors = result.get("agent_errors")
        errors = list(errors) if isinstance(errors, list) else before
        if any(entry not in before for entry in errors):
            return result               # the node recorded its own failure
        exc = AnalysisStepFailed(plain_failure_reason(failure_detail(result)))
        out = dict(result)
        out["agent_errors"] = errors + [failure_record(name, exc)]
        return out

    return wrapped
