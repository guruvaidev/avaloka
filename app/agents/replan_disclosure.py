"""What the replan loop did, in plain words, written by code from its records.

c-strict (app/api/reply_segments.py): every sentence here is a template over
counts the loop measured (app/agents/replan.py), so each is *plain*. No model
writes or edits these lines; the narrator may mention a repair in its own
words, and those words stay commentary.

These sentences go to the user's own chat reply, so they name the column and,
for a conversion, one example value. They never reach telemetry.

No line here carries an exception class, an error message, a traceback, a
path or code: the error is in the logs and on the trace step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


def _n(value: Any) -> str:
    return f"{int(value):,}"


def _rows(n: Any) -> str:
    return "1 row" if int(n) == 1 else f"{_n(n)} rows"


def _values(n: Any) -> str:
    return "1 value" if int(n) == 1 else f"{_n(n)} values"


def _pct(part: int, whole: int) -> str:
    share = part / max(whole, 1)
    return f"{share:.1%}" if share < 0.1 else f"{share:.0%}"


def fill_value(value: Any) -> str:
    """A fill statistic as the reply shows it: two decimals for a number."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return f"“{value}”"


def transform_line(t: Mapping[str, Any]) -> List[str]:
    """The sentence(s) for one repair record."""
    op, col = t.get("op"), t.get("column")
    total = int(t.get("rows_total") or 0)
    out: List[str] = []
    if op == "coerce_numeric":
        n = int(t.get("values_converted") or 0)
        if n:
            ex = f" (for example “{t['example']}”)" if t.get("example") else ""
            out.append(f"{_values(n)} in {col} were numbers stored as text{ex}; I read them as numbers.")
        k = int(t.get("null_tokens") or 0)
        if k:
            words = ", ".join(f"“{w}”" for w in t.get("null_token_examples") or [])
            words = f" ({words})" if words else ""
            out.append(f"{_values(k)} in {col}{words} meant “no value”; I treated them as missing.")
        u = int(t.get("unreadable") or 0)
        if u:
            out.append(f"{_values(u)} in {col} could not be read as numbers; I treated them as missing.")
    elif op == "parse_datetime":
        n, u = int(t.get("values_converted") or 0), int(t.get("unreadable") or 0)
        out.append(f"{_values(n)} in {col} were dates stored as text; I read them as dates."
                   + (f" {_values(u)} could not be read as dates and were treated as missing." if u else ""))
    elif op == "impute":
        n = int(t.get("rows_affected") or 0)
        value = fill_value(t.get("value"))
        if t.get("chosen_by"):
            # The answer to the question: say what it does to the figure.
            out.append(f"{_n(n)} of {_n(total)} rows had no value in {col}; I filled them with the "
                       f"{t.get('method')} ({value}){_as_chosen(t)}. This pulls the average of {col} "
                       f"toward {value}.")
        else:
            out.append(f"{_n(n)} of {_n(total)} rows had no value in {col}; I filled them with the "
                       f"{t.get('method')} ({value}) so the calculation could run.")
    elif op == "exclude_rows":
        n = int(t.get("rows_affected") or 0)
        if t.get("chosen_by"):
            out.append(f"I left out {_n(n)} of {_n(total)} rows ({_pct(n, total)}) that had no value in "
                       f"{col}{_as_chosen(t)}; the result is computed over the other {_n(total - n)}.")
        elif int(t.get("non_finite") or 0):
            out.append(f"I left out {_n(n)} of {_n(total)} rows ({_pct(n, total)}) where {col} was "
                       "missing or infinite.")
        else:
            out.append(f"I left out {_n(n)} of {_n(total)} rows ({_pct(n, total)}) that had no value in "
                       f"{col}, rather than fill them in, because the calculation is of {col} itself.")
    return out


def _as_chosen(t: Mapping[str, Any]) -> str:
    if t.get("chosen_by") == "env":
        return ", as this run was set to do (AVALOKA_REPLAN_MEASURED_CHOICE)"
    return ", as you chose"


def disclosure_lines(transforms: Sequence[Mapping[str, Any]]) -> List[str]:
    """One plain line per repair, in the order applied. Empty when nothing was changed."""
    lines: List[str] = []
    for t in transforms or []:
        for line in transform_line(t):
            if line not in lines:
                lines.append(line)
    if lines:
        lines.insert(0, "The first run stopped on a problem in the data, so I repaired it and ran the same "
                        "calculation again. Your source data was not changed:")
    return lines


_DIAGNOSIS_WORDS = {
    "dtype_mismatch": "a column held text where numbers were expected",
    "missing_values": "a column had missing values the calculation could not use",
    "non_finite": "a column held infinite values",
    "bad_dates": "a column held dates stored as text",
    "column_reference": "my code asked for a column the data doesn't have",
    "code_bug": "there was an error in my code",
    "not_recoverable": "the run hit a limit I can't work around here",
}


def _action_words(attempt: Mapping[str, Any]) -> str:
    if attempt.get("action") == "data_step":
        done = []
        for t in attempt.get("transforms") or []:
            op, col = t.get("op"), t.get("column")
            if op == "coerce_numeric":
                done.append(f"read {_values(t.get('values_converted') or 0)} in {col} as numbers")
            elif op == "parse_datetime":
                done.append(f"read {_values(t.get('values_converted') or 0)} in {col} as dates")
            elif op == "impute":
                done.append(f"filled {_n(t.get('rows_affected') or 0)} missing values in {col} with the "
                            f"{t.get('method')}")
            elif op == "exclude_rows":
                done.append(f"left out {_rows(t.get('rows_affected') or 0)} with no value in {col}")
        return ("I " + "; ".join(done) + " and ran it again.") if done else "I repaired the data and ran it again."
    return "I rewrote the code and ran it again."


_STOP_WORDS = {
    "attempts": "That was my last attempt, so I stopped here.",
    "oscillating": "It went back to an error I had already tried to fix, so I stopped rather than go round in circles.",
    "code_unchanged": "My rewrite came back unchanged, so another run would fail the same way; I stopped here.",
    "budget": "I stopped here because the time or effort set aside for retries ran out.",
    "not_recoverable": "This is not something another attempt can fix, so I stopped here.",
    "no_repair": "I found nothing more I could safely change, so I stopped here.",
}


def exhausted_reply(attempts: Sequence[Mapping[str, Any]], stopped: Optional[Mapping[str, Any]]) -> str:
    """The reply when every attempt failed: what was tried, and why it stopped."""
    n = len(attempts)
    lines = [f"I couldn't finish this analysis. I tried {n} time{'s' if n != 1 else ''}:"]
    for i, a in enumerate(attempts, 1):
        why = _DIAGNOSIS_WORDS.get(str(a.get("diagnosis")), "the calculation stopped with an error")
        lead = "The calculation stopped because" if i == 1 else "It then stopped because"
        lines.append(f"{i}. {lead} {why}. {_action_words(a)}")
    final = stopped or {}
    last = _DIAGNOSIS_WORDS.get(str(final.get("diagnosis")), "the calculation stopped with an error")
    lines.append(f"The last run stopped because {last}. "
                 + _STOP_WORDS.get(str(final.get("reason")), _STOP_WORDS["attempts"]))
    changed = any(a.get("action") == "data_step" for a in attempts)
    lines.append("")
    lines.append("Your source data was not changed." if changed else "Nothing I tried changed your data.")
    lines.append("Naming the exact columns and the figure you want usually helps.")
    return "\n".join(lines)


def needs_user_reply(question: Mapping[str, Any], transforms: Sequence[Mapping[str, Any]] = (),
                     *, interactive: bool = True) -> str:
    """The reply when the loop needs the user's decision: the facts, and the choice.

    Interactive: the question, and the next turn resumes on the answer
    (app/agents/replan_choice.py). No one to answer (a scheduled or background
    run): the same question, nothing applied, and how to answer up front.
    """
    reason = question.get("reason")
    if reason == "column_limit":
        return (f"Fixing this would mean changing {int(question.get('columns') or 0)} columns, more than the "
                f"{int(question.get('limit') or 0)} I change without asking, so I haven't changed anything. "
                "Which columns should I repair?")
    from app.agents.replan_choice import CHOICE_ENV, question_text

    lead = "I stopped before changing your data. " + question_text(question)
    if interactive:
        return lead + " Nothing has been filled or left out yet."
    return (lead + " This run could not wait for an answer, so I haven't filled or left out anything. "
            f"To answer ahead of time, run it again with {CHOICE_ENV}=median or {CHOICE_ENV}=leave_out, "
            "or ask me in the chat.")


def apply_repair_disclosure(report: Dict[str, Any], state: Mapping[str, Any]) -> Dict[str, Any]:
    """The verifier's hook: the repair lines go on the report as caveats.

    Caveats are the verifier's own text, rendered as plain segments after the
    answer (claim_verifier.presentable_segments), the same path as the
    row-filter scope line (app/agents/filter_disclosure.py).
    """
    lines = disclosure_lines(state.get("replan_transforms") or [])
    if not lines:
        return report
    caveats = list(report.get("caveats") or [])
    report["caveats"] = caveats + [line for line in lines if line not in caveats]
    report["data_repaired"] = {"lines": lines, "transforms": len(state.get("replan_transforms") or [])}
    return report
