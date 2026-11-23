"""Avaloka talks the user through every step, starting from the auto-insights.

After an upload, and after each step of a chat turn (understanding, sampling,
plan, execution, verification, result), Avaloka posts one short message in
her own voice -- what she found or did -- and asks one concrete question about
what to do next with the data, with quick replies the UI renders as buttons.

The voice is the one in :mod:`avaloka.persona`: first person, warm, brief,
meta-cognitive, and *grounded*. Every observation here is read from a measured
statistic (``sample_statistics``), the profile, or the finished graph state;
nothing is invented and no number is guessed. A question names the actual
column it is about -- the likely outcome column, the dominant segment, the
time column, the columns the user's question touched -- so "is ``Test Results``
the outcome you care about?" replaces "what would you like to analyse?".

Two entry points, both pure:

* :func:`prompts_after_upload` -- the ``auto_insights`` step.
* :func:`prompts_after_turn` -- the steps a chat turn actually took, in order,
  each only when the state carries evidence it happened.

Everything stays in the HTTP response (``assistant_prompts``); none of it is
telemetry, and column names never leave the response.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from app.agents.analysis_outcome import (ANSWERED, NOT_RUN, NOTHING_TO_VERIFY, PARTIAL, REFUSED,
                                         analysis_outcome, failure_detail, plain_failure_reason,
                                         result_records)
from app.agents.claim_verifier import withheld_claims
from app.api import reply_segments
from app.api.ui_contract import assistant_prompt
# The CLI's reading of a predicts/drives goal (the app may import avaloka).
from avaloka.goal_analysis import column_mentions, driver_target

#: How many quick replies a step offers at most: enough to choose, few enough to read.
MAX_REPLIES = 4

_TIME_RE = re.compile(r"(date|time|_at$|^at_|year|month|week|day|timestamp|period|quarter)", re.I)
_ID_RE = re.compile(r"(^id$|_id$|^id_|uuid|identifier|^name$|_name$|email|phone|address|zip|postcode)", re.I)
_OUTCOME_RE = re.compile(
    r"(result|outcome|status|churn|label|target|class|default|fraud|survived|condition|"
    r"converted|approved|readmit|response|verdict|diagnosis|risk|segment|tier|grade)", re.I)
#: Names of an EVENT the rows record: a two-valued column called this is what
#: the data is about ("stroke", "died"). A describing word ("smoking_status")
#: matches _OUTCOME_RE too, so a binary column outranks it, not the word alone.
_EVENT_RE = re.compile(
    r"(stroke|death|died|deceased|mortality|survival|attrition|cancel|clicked|purchased|"
    r"bought|defaulted|failed|failure|left|^y$|^is_|^has_)", re.I)

#: Identifier-like by name though few-valued: ``VendorID`` holds two vendor
#: codes, so it looks two-valued, but nobody asks what drives a vendor code.
#: Camel-case ``...ID``/``...Id`` is matched case-sensitively ("paid" is not one).
_ID_LIKE_RE = re.compile(r"(?:[a-z0-9]ID|[a-z0-9]Id|[a-z0-9]IDs)$|(?i:vendor|provider|(?:^|_)code(?:$|_)|"
                         r"(?:^|_)(?:cd|nbr|num|no|key)$)")
#: Bookkeeping fields: how a row was listed, flagged or stored, not what
#: happened. ``initial_list_status`` is "w" or "f" on every loan; it matches
#: "status" but is not the outcome when ``loan_status`` is there.
_BOOKKEEPING_RE = re.compile(
    r"(initial_list|list_status|pymnt_plan|policy_code|application_type|verification_status|"
    r"disbursement|store_and_fwd|_flag$|^flag_|record_type|row_status|is_deleted|source_system|"
    r"batch|load_)", re.I)
#: Names that are the outcome itself: these may hold up to a dozen values
#: (``loan_status``: Fully Paid, Charged Off, Current, Late, Default, ...).
_STRONG_OUTCOME_RE = re.compile(
    r"(^loan_status$|default|readmit|churn|^outcome$|^target$|^label$|fraud|survived|^y$)", re.I)

#: How she names what she understood. Anything else reads as its own words.
_INTENT_LABELS = {
    "clarification": "something to pin down before I run anything, so I offered options",
    "statistical_analysis": "a statistical analysis",
    "visualization": "a chart to draw",
    "ml_training": "a model to train",
    "ml_inference": "a prediction to run",
    "data_transfer": "data to move and shape",
    "schedule": "a job to put on a schedule",
    "chitchat": "conversation, not a job",
    "discussion": "a question about results already in hand",
}

#: The swarm quick reply. The text matches app.api.chat_swarm.TRIGGER_RE, so
#: pressing it starts a coordinated mission on the next turn in every edition.
SWARM_REPLY = {"label": "Send my team at it", "send": "Swarm this: {goal}"}


def _n(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _pct(ratio: Any) -> str:
    """A share as a whole percent, unless rounding would hide the rest.

    "100%" and "0%" are said only of an exact 1 and 0: `store_and_fwd_flag`
    is "N" on 99.6% of the taxi rows, and "100% of rows" hid the 754 "Y"
    rows. A share that rounds to either end gets one decimal ("99.6%"), and
    one that still would says which side it is on ("over 99.9%").
    """
    try:
        r = float(ratio)
    except (TypeError, ValueError):
        return "?"
    if r in (0.0, 1.0) or f"{r:.0%}" not in ("0%", "100%"):
        return f"{r:.0%}"
    if f"{r:.1%}" not in ("0.0%", "100.0%"):
        return f"{r:.1%}"
    return "over 99.9%" if r > 0.5 else "under 0.1%"


def _col(name: str) -> str:
    return f"`{name}`"


# ---------------------------------------------------------------- the read
@dataclass
class DataRead:
    """What Avaloka measured about the dataset, from the statistics alone."""

    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    domain: Optional[str] = None
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    time_columns: list[str] = field(default_factory=list)
    id_columns: list[str] = field(default_factory=list)
    #: (column, missing ratio), worst first, only ratios >= 5%.
    missing: list[tuple[str, float]] = field(default_factory=list)
    #: (column, top value, share of rows), most dominant first.
    segments: list[tuple[str, str, float]] = field(default_factory=list)
    #: column -> every listed value tied with its top value, when two or more
    #: are: such a column has no single leader to name.
    ties: dict[str, list[str]] = field(default_factory=dict)
    #: Likely outcome columns: low-cardinality, not an id, not dominated.
    targets: list[str] = field(default_factory=list)
    #: (column, skewness) for |skew| > 1.5.
    skewed: list[tuple[str, float]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    #: Columns whose share of empty cells was actually measured.
    measured: list[str] = field(default_factory=list)
    #: The profiler asked a model for ``domain`` (``source: "model"``): its
    #: choice from the vocabulary, not a measurement, so it is commentary.
    domain_by_model: bool = False

    @property
    def target(self) -> Optional[str]:
        return self.targets[0] if self.targets else None

    @property
    def segment(self) -> Optional[tuple[str, str, float]]:
        return self.segments[0] if self.segments else None

    @property
    def time_column(self) -> Optional[str]:
        return self.time_columns[0] if self.time_columns else None


def read_statistics(
    sample_statistics: Optional[Mapping[str, Any]],
    profiling_result: Optional[Mapping[str, Any]] = None,
    schema: Optional[Iterable[str]] = None,
) -> DataRead:
    """Fold ``sample_statistics`` (sampling_async: type, missing_ratio,
    n_unique, top_values, skewness) and the profile into a :class:`DataRead`.
    Every field is optional on the way in; nothing here raises on a gap."""
    stats = dict(sample_statistics or {})
    read = DataRead()
    shape = stats.get("data_shape") or {}
    read.n_rows = shape.get("rows") if isinstance(shape.get("rows"), int) else None
    read.n_cols = shape.get("columns") if isinstance(shape.get("columns"), int) else None
    columns = stats.get("column_statistics") or {}
    if not isinstance(columns, dict):
        columns = {}
    read.columns = list(columns) or [str(c) for c in (schema or [])]
    if read.n_cols is None and read.columns:
        read.n_cols = len(read.columns)

    prof = dict(profiling_result or {})
    domain = (prof.get("domain") or {}).get("category") if isinstance(prof.get("domain"), dict) else None
    if isinstance(domain, str) and domain.strip() and domain.strip().lower() != "unknown":
        read.domain = domain.strip()
        read.domain_by_model = str(prof["domain"].get("source") or "") == "model"
    predictive = prof.get("predictive_values") if isinstance(prof.get("predictive_values"), dict) else {}

    candidates: list[tuple[int, int, str]] = []
    names = list(columns)
    for name, stat in columns.items():
        if not isinstance(stat, dict) or "error" in stat:
            continue
        total = stat.get("total_count") or read.n_rows or 0
        n_unique = stat.get("n_unique")
        is_numeric = stat.get("type") == "numeric"
        # One distinct value per row marks a text column as an identifier; a
        # numeric column with no repeats is just a measure (revenue, billing).
        looks_id = bool(_ID_RE.search(name)) or (
            not is_numeric and isinstance(n_unique, int) and total
            and n_unique >= 0.95 * total and total > 20)
        looks_time = bool(_TIME_RE.search(name)) and not is_numeric
        if looks_time:
            read.time_columns.append(name)
        elif looks_id:
            read.id_columns.append(name)
        elif is_numeric:
            read.numeric.append(name)
        else:
            read.categorical.append(name)

        ratio = stat.get("missing_ratio")
        if isinstance(ratio, (int, float)):
            read.measured.append(name)
        if isinstance(ratio, (int, float)) and ratio >= 0.05:
            read.missing.append((name, float(ratio)))

        skew = stat.get("skewness")
        if is_numeric and isinstance(skew, (int, float)) and abs(skew) > 1.5:
            read.skewed.append((name, float(skew)))

        # An identifier-like name is never offered as the outcome, however
        # few values it holds.
        offerable = not looks_id and not looks_time and not _ID_LIKE_RE.search(name)
        if not is_numeric and not looks_id and not looks_time and isinstance(n_unique, int) and 2 <= n_unique <= 12:
            top = (stat.get("top_values") or [{}])[0]
            share = _top_share(top, stat, read.n_rows)
            if top.get("value") is not None and share is not None and share > 0:
                read.segments.append((name, str(top["value"]), share))
                tied = _tied_with_top(stat.get("top_values") or [])
                if len(tied) > 1:
                    read.ties[name] = tied
            few = n_unique <= 6 or _STRONG_OUTCOME_RE.search(name)
            if offerable and few and (share is None or share <= 0.9):
                candidates.append((-_outcome_score(name, n_unique, names, predictive), n_unique, name))
        elif is_numeric and offerable and n_unique == 2:
            # A 0/1 column is the commonest outcome there is ("stroke",
            # "Outcome", "churned"), and it profiles as numeric.
            candidates.append((-_outcome_score(name, n_unique, names, predictive), n_unique, name))

    read.missing.sort(key=lambda t: -t[1])
    read.segments.sort(key=lambda t: -t[2])
    read.skewed.sort(key=lambda t: -abs(t[1]))
    ranked = [name for _, _, name in sorted(candidates)]
    # A bookkeeping field is offered only when nothing stronger is there,
    # and then not as an alternative to it either.
    real = [n for n in ranked if not _BOOKKEEPING_RE.search(n)]
    read.targets = real or ranked
    return read


def _tied_with_top(top_values: Sequence[Any]) -> list[str]:
    """The listed values that hold exactly as many rows as the first.

    Compared by ``count`` when every value has one (a weighted estimate is
    still the same estimate for each), else by ``frequency``. 20/20/20 in
    the null-words workbook made "`Outcome` leans on “Recovered”" a claim of
    a leader that is not there.
    """
    values = [t for t in top_values if isinstance(t, Mapping) and t.get("value") is not None]
    if not values:
        return []
    key = "count" if all(isinstance(t.get("count"), (int, float)) and not isinstance(t.get("count"), bool)
                         for t in values) else "frequency"
    first = values[0].get(key)
    if not isinstance(first, (int, float)):
        return []
    return [str(t["value"]) for t in values if t.get(key) == first]


def _top_share(top: Mapping[str, Any], stat: Mapping[str, Any], n_rows: Optional[int]) -> Optional[float]:
    """The share of rows the top value holds, over its OWN denominator.

    The sampler reports ``frequency`` (count over the sample it came from);
    a stratified sample's ``count`` is a weighted estimate whose total is not
    the file's row count, so dividing it by ``n_rows`` gave "105% of rows".
    ``count / total_count`` is used only where both came from one pass. A
    share that still comes out above 1 is not a share: ``None``.
    """
    freq = top.get("frequency")
    if isinstance(freq, (int, float)) and not isinstance(freq, bool):
        share = float(freq)
    else:
        total = stat.get("total_count") or n_rows or 0
        share = (top.get("count") or 0) / total if total else 0.0
    return share if 0.0 <= share <= 1.0 else None


def _outcome_score(name: str, n_unique: int, names: Sequence[str],
                   predictive: Mapping[str, Any]) -> int:
    """How much a column looks like the outcome to understand.

    A column named for an event outranks a descriptive column whose name
    merely contains "status": in the stroke data that is ``stroke`` over
    ``smoking_status``. Two values and being the last column are weak hints
    (most exported tables put the label last); neither alone beats a name.
    """
    score = 0
    if _STRONG_OUTCOME_RE.search(name):
        score += 3
    if _OUTCOME_RE.search(name):
        score += 2
    if _EVENT_RE.search(name):
        score += 2
    if n_unique == 2:
        score += 1
    if names and name == names[-1]:
        score += 1
    if predictive.get(name) == "high":
        score += 1
    return score


# ------------------------------------------------------------- the upload
def prompts_after_upload(
    *,
    sample_statistics: Optional[Mapping[str, Any]],
    profiling_result: Optional[Mapping[str, Any]] = None,
    schema: Optional[Iterable[str]] = None,
    filename: Optional[str] = None,
    total_rows: Optional[int] = None,
) -> list[dict[str, Any]]:
    """The ``auto_insights`` step: what she saw first, and one question."""
    read = read_statistics(sample_statistics, profiling_result, schema)
    rows = total_rows if isinstance(total_rows, int) else read.n_rows
    if rows is None and read.n_cols is None:
        return []  # nothing measured yet: say nothing rather than something generic

    name = f" at {filename}" if filename else ""
    size = (f"{_n(rows)} rows across {read.n_cols} columns" if rows is not None and read.n_cols
            else f"{read.n_cols} columns" if read.n_cols else f"{_n(rows)} rows")
    domain = f" — this reads like {read.domain.lower()} data" if read.domain and not read.domain_by_model else ""
    first = f"I've had a first look{name}: {size}{domain}."
    # A domain a model picked is its reading, said as its own sentence.
    guessed = f"It reads like {read.domain.lower()} data." if read.domain and read.domain_by_model else ""

    observation = ""
    if read.segment:
        col, value, share = read.segment
        tied = read.ties.get(col)
        if tied:
            names = ", ".join(f"“{v}”" for v in tied[:-1]) + f" and “{tied[-1]}”"
            observation = (f"{_col(col)} has no single leader: {names} hold {_pct(share)} of rows each "
                           "— a split worth naming.")
        else:
            observation = f"{_col(col)} leans on “{value}” ({_pct(share)} of rows) — a segment worth naming."
        if read.missing:
            mcol, ratio = read.missing[0]
            observation += f" {_col(mcol)} is {_pct(ratio)} empty, so I'll be careful there."
    elif read.missing:
        mcol, ratio = read.missing[0]
        observation = f"{_col(mcol)} is {_pct(ratio)} empty — I won't paper over that."
    elif read.skewed:
        scol, skew = read.skewed[0]
        observation = f"{_col(scol)} is skewed (skew {skew:.1f}); a few extreme values could run the show."
    elif read.columns and len(read.measured) == len(read.columns):
        observation = "Clean and quiet so far — no column is more than 5% empty."
    elif read.columns:
        # "Clean" is a measurement. With no statistics (a failed sampler, only
        # a schema) nothing was measured, and saying so is the observation.
        observation = ("I haven't measured how complete the columns are yet, so I can't say "
                       "whether any are empty." if not read.measured else
                       f"I measured {len(read.measured)} of {len(read.columns)} columns; "
                       "none of those is more than 5% empty.")

    replies: list[Any] = []
    if read.target:
        question = f"Is {_col(read.target)} the outcome you want to understand, or shall we explore first?"
        replies.append(f"What drives {read.target}?")
        replies.append(f"Predict {read.target}")
    else:
        question = "What matters most here — a number to explain, a segment to compare, or a trend over time?"
        if read.numeric:
            replies.append(f"Explain {read.numeric[0]}")
    if read.segment:
        replies.append(f"Compare by {read.segment[0]}")
    if read.time_column:
        replies.append(f"Trend over {read.time_column}")
    replies.append("Just explore")
    message: Any = f"{first} {observation}".strip()
    if guessed:
        message = [*reply_segments.plain(f"{first} "), *reply_segments.commentary(guessed),
                   *reply_segments.plain(f" {observation}".rstrip())]
    return [assistant_prompt(
        "auto_insights", message,
        question=question, quick_replies=replies[:MAX_REPLIES],
    )]


# --------------------------------------------------------------- the turn
def _turn_nodes(final: Mapping[str, Any]) -> Optional[frozenset[str]]:
    """The graph nodes that ran in THIS turn, from the turn's own trace.

    The checkpointed graph state carries every key a node once wrote into
    later turns (``generated_code``, ``plan``, ``avaloka_intent``, and the
    training state the server reloads from the session), so a value on
    ``final`` is not evidence that this turn did anything. ``agent_trace``
    holds this turn's steps only (app.trace.recorder.merge_agent_trace),
    each with the turn's id. None when the state carries no trace for the
    turn: a caller then falls back to the fields alone.
    """
    turn, trace = final.get("trace_turn_id"), final.get("agent_trace")
    if not turn or not isinstance(trace, list):
        return None
    nodes = frozenset(str(s["node"]) for s in trace
                      if isinstance(s, Mapping) and s.get("turn_id") == turn and s.get("node"))
    return nodes or None


def _model_swaps(final: Mapping[str, Any]) -> list[dict[str, Any]]:
    """This turn's model calls that a fallback tier answered, from the agent trace.

    Each is the record the tier wrote (app/core/model_tiers.answered_record via
    app/trace/recorder.py): configured_model, answered_model, provider, reason. The
    reason is built from the recorded HTTP status and message, never from a model.
    """
    turn, trace = final.get("trace_turn_id"), final.get("agent_trace")
    if not turn or not isinstance(trace, list):
        return []
    swaps, seen = [], set()
    for step in trace:
        if not isinstance(step, Mapping) or step.get("turn_id") != turn:
            continue
        for call in step.get("inference") or []:
            rec = call.get("answered_by") if isinstance(call, Mapping) else None
            if not isinstance(rec, Mapping) or call.get("outcome") != "success":
                continue
            key = (rec.get("configured_model"), rec.get("answered_model"), rec.get("provider"),
                   rec.get("reason"))
            if all(key[:2]) and key not in seen:
                seen.add(key)
                swaps.append(dict(rec))
    return swaps


def model_swap_sentence(rec: Mapping[str, Any]) -> str:
    """The plain line: <configured> did not answer (<reason>); <answered> via <provider> answered instead."""
    why = f" ({rec['reason']})" if rec.get("reason") else ""
    via = f" via {rec['provider']}" if rec.get("provider") else ""
    return f"{rec['configured_model']} did not answer{why}; {rec['answered_model']}{via} answered instead."


def _ran(final: Mapping[str, Any], *nodes: str) -> bool:
    """Whether any of *nodes* ran this turn (True when the turn has no trace)."""
    ran = _turn_nodes(final)
    return ran is None or any(n in ran for n in nodes)


#: Where code records the options a reply offers, when it offers them:
#: ``pending_suggestions`` (the planner's suggest_analysis, or
#: app.agents.suggestions.remember_offered_options) and
#: ``pending_clarification`` (a clarifying question with its choices).
_OPTION_KEYS = ("pending_suggestions", "pending_clarification")


def _offered_options(final: Mapping[str, Any]) -> bool:
    """Whether THIS turn put options in front of the user and ran nothing.

    ``avaloka_intent == "clarification"`` is not that: it is the keyword
    classifier's "no keyword matched", and the planner takes the turn and
    usually answers or runs code (rc18b: "What is the average Glucose ..."
    ran code under it). Both option keys outlive their turn, so with a trace
    they count only when a step of this turn wrote them; without one, their
    presence is the evidence.
    """
    result = final.get("execution_result")
    if _turn_code(final) or (isinstance(result, Mapping) and result) or isinstance(final.get("output_json"), list):
        return False
    turn, trace = final.get("trace_turn_id"), final.get("agent_trace")
    if turn and isinstance(trace, list) and _turn_nodes(final) is not None:
        written = {str(k) for s in trace if isinstance(s, Mapping) and s.get("turn_id") == turn
                   for k in s.get("written") or []}
        return any(k in written and final.get(k) for k in _OPTION_KEYS)
    return any(final.get(k) for k in _OPTION_KEYS)


def _turn_code(final: Mapping[str, Any]) -> Optional[str]:
    """The code the coder wrote THIS turn, or None.

    ``coder_definition`` is reset at the start of every turn (server.py's
    per-run fields) and set only when the coding sub-graph's validators
    passed it (workflow.coding_subgraph_node). ``generated_code`` is not
    reset: a chit-chat turn still carries the previous turn's code, and the
    plan step once said "I wrote the code for it" on a turn that wrote none.
    """
    cd = final.get("coder_definition")
    code = cd.get("code") if isinstance(cd, Mapping) else None
    return code if isinstance(code, str) and code.strip() else None


def _code_text(final: Mapping[str, Any]) -> str:
    code = (final.get("coder_definition") or {}).get("code") if isinstance(final.get("coder_definition"), dict) else None
    return "\n".join(str(c) for c in (code, final.get("generated_code")) if isinstance(c, str))


def datasets_read(final: Mapping[str, Any], datasets: Optional[Sequence[Mapping[str, Any]]],
                  ) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """(the datasets the turn's code read, in the turn's order; the other
    sheets of the workbook it read that it did not).

    When the sandbox recorded its reads (``execution_result["reads"]``), they
    alone say what was read, the primary included: a main(df) that never
    used its frame did not read the primary. A read is matched by the alias
    or dataset id it was annotated with, or by the ``input_<id>`` file it
    names. Without recorded reads the primary counts as read, and another
    dataset when the code names its alias, file name or dataset id; every
    dataset of the turn is mapped for the sandbox, so being mapped is not
    being read. On rc5 both sheets were read through path variables and the
    basis said the second was "not analysed".
    """
    items = [d for d in (datasets or []) if isinstance(d, Mapping)]
    if not items:
        return [], []
    primary, others = items[0], items[1:]
    code = _code_text(final)
    result = final.get("execution_result") if isinstance(final.get("execution_result"), Mapping) else {}
    reads = [r for r in result.get("reads") or [] if isinstance(r, Mapping) and r.get("used") is not False]

    def _recorded(d: Mapping[str, Any]) -> bool:
        did = d.get("dataset_id")
        return any((d.get("alias") and r.get("alias") == d.get("alias"))
                   or (did and r.get("dataset_id") == did)
                   or (did and os.path.basename(str(r.get("path") or "")).startswith(f"input_{did}."))
                   for r in reads)

    recorded = [d for d in items if _recorded(d)]
    if recorded:
        used = recorded
    else:
        used = [primary] + [d for d in others if code and any(
            re.search(rf"(?<![\w]){re.escape(str(d[k]))}(?![\w])", code)
            for k in ("alias", "filename", "dataset_id") if d.get(k))]
    workbooks = {d.get("source_workbook") for d in used if d.get("source_workbook")}
    unread = [d for d in items if d not in used and d.get("source_workbook") in workbooks]
    return used, unread


def _kind(items: Sequence[Mapping[str, Any]]) -> str:
    sheets = all(d.get("sheet_name") for d in items)
    word = "sheet" if sheets else "dataset"
    return word if len(items) == 1 else word + "s"


def _source_list(items: Sequence[Mapping[str, Any]], first_rows: Optional[int] = None) -> str:
    """"sheet '2022' (60 rows), sheet '2023' (60 rows)"; a file by its name."""
    out = []
    for i, d in enumerate(items):
        name = f"sheet '{d['sheet_name']}'" if d.get("sheet_name") else f"`{d.get('filename') or d.get('alias')}`"
        rows = first_rows if i == 0 and first_rows is not None else d.get("rows")
        out.append(name + (f" ({_n(rows)} rows)" if isinstance(rows, int) else ""))
    return ", ".join(out[:-1]) + (" and " if len(out) > 1 else "") + out[-1] if out else ""


def _columns_in(text: str, columns: Sequence[str]) -> list[str]:
    """The dataset columns the user's words touched, longest names first.

    Whole words only, with a plural or past tense folded ("returns" touches
    ``returned``, "channels" touches ``channel``); a substring is not a
    mention, so "average" does not touch ``age`` nor "turn" ``returned``.
    """
    hits = [c for _s, _e, c in column_mentions(text or "", [c for c in columns if c])]
    return sorted(set(hits), key=lambda c: (-len(c), c))


_PICK_RE = re.compile(r"^\s*(?:run|do|option|number|#)?\s*#?\s*(\d{1,2})\s*[.!]?\s*$", re.I)


def _goal(question: str) -> str:
    q = (question or "").strip().rstrip("?.! ")
    pick = _PICK_RE.match(question or "")
    if pick:
        return f"option {pick.group(1)}"  # the user chose from the list she offered
    return q[:80] if q else "this question"


def _plan_steps(planner_definition: Any) -> list[str]:
    if not isinstance(planner_definition, dict):
        return []
    steps = planner_definition.get("transformations") or planner_definition.get("steps") or []
    names: list[str] = []
    for step in steps if isinstance(steps, list) else []:
        if isinstance(step, dict):
            label = step.get("operation") or step.get("name") or step.get("type") or step.get("description")
        else:
            label = step
        if label:
            names.append(str(label).replace("_", " ")[:40])
    return names


def _training_failure(final: Mapping[str, Any]) -> Optional[str]:
    """The user-safe reason a training turn failed, or None if it did not.

    ``training_result`` carries ``status: "error"`` / ``error`` /
    ``execution_error`` on failure, and ``failure`` -- the classified, already
    sanitised summary from app.agents.mta_v2.failure_diagnostics -- when the
    agent built one. Raw ``error`` text is never shown: it holds paths.
    """
    tr = final.get("training_result")
    tr = tr if isinstance(tr, Mapping) else {}
    failure = tr.get("failure") if isinstance(tr.get("failure"), Mapping) else {}
    failed = (
        str(tr.get("status") or "").lower() in {"error", "failed", "failure"}
        or bool(tr.get("error") or tr.get("execution_error") or failure)
        or (final.get("training_completed") is False and bool(tr))
    )
    if not failed:
        return None
    summary = str(failure.get("summary") or "").strip()
    return summary or "the run stopped with an error before a model was saved."


def _cv_beats_baseline(final: Mapping[str, Any]) -> Optional[bool]:
    """Whether the cross-validated model beat its baseline, when the turn has CV."""
    for source in (final.get("evaluation_report"), (final.get("training_result") or {}).get("evaluation_report")
                   if isinstance(final.get("training_result"), dict) else None):
        cv = source.get("cross_validation") if isinstance(source, dict) else None
        if isinstance(cv, dict) and isinstance(cv.get("beats_baseline"), bool):
            return cv["beats_baseline"]
    return None


def prompts_after_turn(
    final: Mapping[str, Any],
    *,
    question: str,
    execution_context: Optional[Mapping[str, Any]] = None,
    sample_statistics: Optional[Mapping[str, Any]] = None,
    profiling_result: Optional[Mapping[str, Any]] = None,
    schema: Optional[Iterable[str]] = None,
    datasets: Optional[Sequence[Mapping[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """One prompt per step the turn took, in the order it took them.

    A step appears only when ``final`` (the graph's finished state) or the
    execution context carries evidence of it, so a chit-chat turn says
    nothing and a failed run says it failed. ``datasets`` are the turn's
    datasets, the primary first (``alias``, ``filename``, ``sheet_name``,
    ``source_workbook``, ``rows``): the basis names every one the code read.
    """
    read = read_statistics(sample_statistics, profiling_result, schema)
    goal = _goal(question)
    touched = _columns_in(question, read.columns)
    # The outcome a "what drives / predicts X" goal names comes first,
    # whatever the heuristic ranked first. Only that phrase counts: "average
    # Billing by Admission Type" groups on a column, it does not name an outcome.
    driver = driver_target(question or "", {})
    phrase = driver["phrase"] if driver and driver.get("explicit_verb") else ""
    named = [c for c in read.targets if c in _columns_in(phrase, read.columns)] if phrase else []
    read.targets = named + [c for c in read.targets if c not in named]
    seg = read.segment
    out: list[dict[str, Any]] = []

    # understanding -------------------------------------------------------
    # The intent and the plan text outlive their turn in the graph state;
    # they count only when the node that writes them ran this turn.
    intent = final.get("avaloka_intent") if _ran(final, "avaloka_agent") else None
    plan_text = (final.get("plan") if isinstance(final.get("plan"), str) and _ran(final, "plan_etl", "summarize_etl")
                 else None)
    plan_steps = _plan_steps(final.get("planner_definition"))
    if intent or plan_text or plan_steps or touched:
        if plan_steps:
            # What the planner did outranks a stale or vague intent label.
            # The steps' names are the planner model's words: the plan step
            # says them, marked as commentary, not this template sentence.
            label = "a job for my planner"
        elif intent == "clarification" and not _offered_options(final):
            # The classifier matched no keyword and handed the question to
            # the planner; "I offered options" is said only of a turn that did.
            label = "a question for my planner"
        else:
            label = _INTENT_LABELS.get(str(intent), str(intent).replace("_", " ")) if intent else "an analysis"
        msg = f"I read “{goal}” as {label}"
        msg += f", about {', '.join(_col(c) for c in touched[:3])}." if touched else "."
        if touched:
            alt = next((c for c in (read.targets + read.numeric) if c not in touched), None)
            q = f"Is {_col(touched[0])} the right column for this, or did you mean another?"
            replies = [f"Yes, {touched[0]} is right"] + ([f"Use {alt} instead"] if alt else []) + ["Explain your reading"]
        else:
            focus = read.target or (read.numeric[0] if read.numeric else None)
            q = (f"Should I keep {_col(focus)} in focus for the next steps?" if focus
                 else "Should I keep this framing for the next steps?")
            replies = ["Keep it", "Change the focus", "Explain your reading"]
        out.append(assistant_prompt("understanding", msg, question=q, quick_replies=replies[:MAX_REPLIES]))

    # sampling ------------------------------------------------------------
    ctx = dict(execution_context or {})
    rows_analyzed, total = ctx.get("rows_analyzed"), ctx.get("total_rows")
    if isinstance(rows_analyzed, int) and rows_analyzed > 0:
        mode = ctx.get("mode_label") or ctx.get("mode") or "this run"
        read_from, unread = datasets_read(final, datasets)
        # rows_analyzed is the primary's count; another dataset has its own.
        primary = next((d for d in datasets or [] if isinstance(d, Mapping)), None)
        rows = [rows_analyzed if d is primary else d.get("rows") for d in read_from]
        known = bool(rows) and all(isinstance(r, int) for r in rows)
        if ctx.get("is_sampled") and isinstance(total, int):
            msg = f"I worked on {_n(rows_analyzed)} of {_n(total)} rows ({mode}) so you'd have an answer quickly."
            if len(read_from) > 1:
                msg += f" The code also read {_source_list(read_from[1:])}, combined with it."
            q = "Want me to rerun on the entire dataset for the final number?"
            replies = ["Use the entire dataset", "Keep the sample"]
        elif len(read_from) > 1:
            # #551's basis rule: "all N rows" names what N is made of. The
            # code combined datasets (a workbook's sheets), so the count is
            # theirs together, not the primary's.
            if len(read_from) == 2 and _kind(read_from) == "sheets":
                names = " and ".join(f"'{d['sheet_name']}'" for d in read_from)
                counts = f" ({' + '.join(_n(r) for r in rows)} rows)" if known else ""
                msg = f"I ran this on both sheets {names}{counts} combined ({mode}), so these numbers are theirs together, not one sheet's"
            else:
                msg = (f"I ran this on {_n(sum(rows)) + ' rows of ' if known else ''}{len(read_from)} "
                       f"{_kind(read_from)} combined: {_source_list(read_from, first_rows=rows[0] if known else None)} "
                       f"({mode}), so these numbers are theirs together, not one {_kind(read_from[:1])}'s")
            msg += (f"; {_source_list(unread)} not used" if unread else "") + "."
            # Its own question: these two branches once had none and spoke
            # the understanding step's, or raised UnboundLocalError without one.
            q = "Keep these together, or look at one at a time?"
            replies = ["Keep them together", "Look at one at a time"]
        elif read_from and read_from[0].get("sheet_name"):
            sheet = read_from[0]["sheet_name"]
            n_rows = _n(rows[0]) if known else _n(rows_analyzed)
            named = re.search(rf"(?<!\w){re.escape(str(sheet))}(?!\w)", question or "", re.I)
            if unread and not named:
                # The question names no sheet, so it is about the workbook.
                msg = (f"The code read only sheet '{sheet}', not the whole workbook your question is about: "
                       f"I ran this on all {n_rows} rows of it ({mode}); {_source_list(unread)} not analysed, "
                       "so these numbers are that sheet's, not the whole workbook's.")
            else:
                msg = (f"I ran this on all {n_rows} rows of {_source_list(read_from[:1])} ({mode})"
                       + (f"; {_source_list(unread)} not analysed, so these numbers are that sheet's, "
                          "not the whole workbook's." if unread else ", so these numbers are the whole sheet's."))
            q = (f"Include {_source_list(unread)} as well?" if unread
                 else "Keep using the whole sheet, or switch to a quick sample while we iterate?")
            replies = (["Include them", "Keep this sheet"] if unread
                       else ["Keep the whole sheet", "Use a quick sample"])
        else:
            msg = f"I ran this on all {_n(rows_analyzed)} rows ({mode}), so these numbers are the whole file's."
            q = "Keep using the whole file, or switch to a quick sample while we iterate?"
            replies = ["Keep the whole file", "Use a quick sample"]
        out.append(assistant_prompt("sampling", msg, question=q, quick_replies=replies))

    # plan ----------------------------------------------------------------
    # Only this turn's plan and code: planner_definition and coder_definition
    # are reset every turn, generated_code is not (_turn_code).
    result = final.get("execution_result") if isinstance(final.get("execution_result"), dict) else None
    output = final.get("output_json") if isinstance(final.get("output_json"), list) else None
    if plan_steps or _turn_code(final):
        if plan_steps:
            n = len(plan_steps)
            shown = ", ".join(plan_steps[:4]) + (f" and {n - 4} more" if n > 4 else "")
            # The count is the plan's; the names are what the planner model
            # called its steps ("Group by Cuisine" on a file with no cuisine
            # column), so they carry the commentary mark.
            msg = [*reply_segments.plain(f"My plan had {n} step{'s' if n != 1 else ''}. "),
                   *reply_segments.commentary(f"My planner named {'them' if n != 1 else 'it'}: {shown}.")]
        else:
            # "before it ran" only when it ran this turn (execution_result and
            # output_json are reset every turn); a deferred job has not.
            msg = ("I wrote the code for it and had my validators read it"
                   + (" before it ran." if result or output is not None else "."))
        replies = []
        if seg and seg[0] not in touched:
            replies.append(f"Break it down by {seg[0]}")
        if read.time_column and read.time_column not in touched:
            replies.append(f"Add a trend over {read.time_column}")
        replies += ["Show the code", "Keep it simple"]
        q = (f"Want a breakdown by {_col(seg[0])} next, or keep it simple?" if seg and seg[0] not in touched
             else "Want to see the code, or keep going?")
        out.append(assistant_prompt("plan", msg, question=q, quick_replies=replies[:MAX_REPLIES]))

    # execution -----------------------------------------------------------
    # Set by any step that failed. Nothing after a failure may read as success.
    failed = False
    # Neither is reset between turns: they are this turn's only if the coding
    # step ran in it.
    coded = _ran(final, "code_etl")
    violations = (final.get("contract_violations")
                  if coded and isinstance(final.get("contract_violations"), list) else [])
    blocked = (final.get("coder_blocked_reason")
               if coded and isinstance(final.get("coder_blocked_reason"), str) else None)
    if not result and output is None and (violations or blocked):
        # The code never ran: her own checks stopped it first.
        first = str(violations[0] if violations else blocked).strip().splitlines()[0][:160]
        msg = f"I wrote the code, but it didn't pass my own checks, so I didn't run it: {first}"
        q = "Shall I try again, or would you name the grouping and the figure you want per group?"
        replies = ["Try again"]
        if seg and read.numeric:
            replies.append(f"Average {read.numeric[0]} by {seg[0]}")
        replies.append("Show the code")
        out.append(assistant_prompt("execution", msg, question=q, quick_replies=replies[:MAX_REPLIES]))
        failed = True
    elif result or output is not None:
        status = str((result or {}).get("status") or ("success" if output else "")).lower()
        n_out = (result or {}).get("output_row_count")
        if not isinstance(n_out, int) and output is not None:
            n_out = len(output)
        if status == "refused":
            msg = "I didn't run anything for this request; it was declined before any code ran."
            q = "Would you put the question another way?"
            replies = ["Rephrase", "Just explore"]
        elif status == "partial":
            msg = "The calculation came back with only part of what you asked; some figures could not be calculated."
            q = "Shall I look at why the missing figures failed?"
            replies = ["Explain what's missing", "Show the code"]
        elif status == "success":
            msg = "The code ran" + (f" and came back with {_n(n_out)} row{'s' if n_out != 1 else ''}." if isinstance(n_out, int) else ".")
            replies = ["Chart it"]
            if seg:
                replies.append(f"Filter to {seg[0]} = {seg[1]}")
            replies.append("Show the code")
            q = (f"Want it as a chart, or filtered to {_col(seg[0])} = “{seg[1]}”?" if seg
                 else "Want it as a chart, or the code behind it?")
        else:
            # Plain words, the reply's own (execution_agent._friendly_failure_message):
            # "Script failed with exit status 1" is the mechanism, not what went wrong.
            msg = f"I couldn't finish the analysis: {plain_failure_reason(failure_detail(final))}."
            q = "Shall I retry, or show you what I tried?"
            replies = ["Retry", "Show the code"]
            # The replan loop already retried (app/agents/replan.py); its own
            # reply says what it tried. Say how it ended, not "Shall I retry".
            _replan_end = final.get("replan_final")
            _tries = len(final.get("replan_attempts") or [])
            if _replan_end == "needs_user":
                msg = "I stopped before changing your data: the repair needs your decision."
                q = "Should I leave those rows out, or fill them with the median?"
                replies = ["Leave them out", "Fill with the median", "Show the code"]
            elif _tries:
                msg = (f"I couldn't finish the analysis after {_tries} attempt"
                       f"{'s' if _tries != 1 else ''} to repair it.")
                q = "Would you name the exact columns and the figure you want?"
                replies = ["Show what I tried", "Show the code"]
            failed = True
        out.append(assistant_prompt("execution", msg, question=q, quick_replies=replies[:MAX_REPLIES]))

    # training (a turn that trained instead of querying) ------------------
    # The server reloads the training state from the session every turn, so
    # it says a model was trained only on the turn the training node ran.
    plan = (final.get("training_plan") if isinstance(final.get("training_plan"), dict)
            and _ran(final, "train_models") else None)
    if plan and (final.get("training_completed") or final.get("training_result")):
        target = (plan.get("data_config") or {}).get("target_column")
        # The model's kind is checked in code (mta_v2: classification or
        # regression); its name is the training planner model's words.
        kind = str(plan.get("model_type") or "")
        model = f"a {kind} model" if kind in ("classification", "regression") else "a model"
        name = str(plan.get("model_name") or "").strip()
        named = reply_segments.commentary(f"My planner named it {name}.") if name else []
        predicts = f" to predict {_col(target)}" if target else ""
        trained = _training_failure(final)
        if trained is not None:
            # A training_result is also what a failed run leaves behind. This
            # once said "I trained ..." right under "Training stopped because
            # of an unclassified error". Say the failure, in its safe summary.
            msg = f"I couldn't train {model}{predicts}: {trained}"
            q = "Shall I retry the training, or change the plan first?"
            replies = ["Retry training", "Change the plan"]
            failed = True
        else:
            beats = final.get("evaluation_beats_baseline")
            msg = f"I trained {model}{predicts}."
            cv_beats = _cv_beats_baseline(final)
            if beats is True and cv_beats is False:
                # The holdout can edge the baseline while the folds do not
                # (stroke: 0.963 vs 0.957 held out, 0.9477 vs 0.9497 over 5 folds).
                msg += (" I have to be straight with you: it beats the trivial baseline on the holdout but not "
                        "in cross-validation, so it does not reliably beat the trivial baseline.")
            elif beats is True:
                msg += " It beats the trivial baseline by more than fold noise."
            elif beats is False:
                msg += " I have to be straight with you: it does not beat the trivial baseline, so it has found nothing yet."
            q = "Want the feature importances, or a different target?"
            replies = ["Show feature importance", "Try another target", "Explain the baseline"]
        out.append(assistant_prompt("execution", [*reply_segments.plain(msg + (" " if named else "")), *named],
                                    question=q, quick_replies=replies))

    # verification --------------------------------------------------------
    report = final.get("verification_report") if isinstance(final.get("verification_report"), dict) else None
    outcome = analysis_outcome(final)
    # A turn with no numeric claim (a clarification, a plan offered) has
    # nothing to verify, and saying so is noise. A failed analysis, or one
    # with no result, had nothing to verify either: its reply is the failure,
    # and "all of them hold" about it is the claim this step exists to stop.
    if (report and outcome not in NOTHING_TO_VERIFY and report.get("analysis") not in NOTHING_TO_VERIFY
            and isinstance(report.get("claims_checked"), int) and report["claims_checked"] > 0):
        checked, problems = report["claims_checked"], report.get("n_problems") or 0
        seen = report.get("claims_seen")
        # A superlative over a cut of the result that nothing could confirm
        # (app/agents/claim_extremes): not checked, and said so.
        unchecked = len(report.get("unconfirmed_claims") or [])
        safe = report.get("safe_to_present", final.get("verification_safe_to_present"))
        if isinstance(seen, int) and seen > checked:
            # D13.2: the count of what a check resolved, out of what the answer
            # says; never more than were checked (rc18d said "5" of 2 checks).
            msg = f"I checked {checked} of the {seen} statements in my answer against the numbers"
        else:
            msg = f"I checked {checked} claim{'s' if checked != 1 else ''} in my answer against the numbers"
        # Owner's ruling (2.0): no verdict when none failed. "all of them hold"
        # said more than the check does; the count is what is known.
        msg += "." if not problems else f"; {problems} did not hold and I've flagged {'it' if problems == 1 else 'them'}."
        if unchecked:
            msg += (f" {'One statement' if unchecked == 1 else f'{unchecked} statements'} I couldn't check against "
                    "all the rows, only the ones the result kept; the answer says so.")
        if report.get("scoped"):
            # app/agents/filter_disclosure: true only of the rows a filter kept.
            msg += " They are figures of a filtered subset, not of every row; the answer now says what was left out."
        # app/api/server.py withholds those claims from the reply (presentable_reply):
        # the unsafe ones, and any figure that did not reconcile with the result.
        withheld = safe is False or bool(withheld_claims(report))
        if withheld and problems:
            msg += " I've held back the statements the numbers don't support."
        if (report.get("derived_categories") or {}).get("withheld"):
            msg += " I've held back the category breakdown: the keyword matching that made its categories doesn't fit the data."
        if withheld:
            q = "Shall I redo the statement with only what the numbers support?"
            replies = ["Redo it", "Show the findings"]
        else:
            q = "Want to see how I checked, or move on?"
            replies = ["Explain the checks", "Move on"]
        out.append(assistant_prompt("verification", msg, question=q, quick_replies=replies))

    # result ----------------------------------------------------------------
    # Only after something ran: a clarification turn ends on its question.
    # And "That's my answer" only when the turn produced one: after a failed or
    # blocked run, or one with no rows, she says there is no answer yet.
    trained = bool(plan and (final.get("training_completed") or final.get("training_result")))
    produced = outcome == ANSWERED or (outcome == NOT_RUN and (trained or bool(result_records(final))))
    attempted = any(p["step"] in ("execution", "verification") for p in out)
    if outcome == REFUSED:
        pass  # nothing ran: the refusal is the reply, and there is no answer to speak of
    elif attempted and outcome == PARTIAL:
        out.append(assistant_prompt(
            "result", f"I have only part of an answer to “{goal}”; the figures I couldn't calculate are missing from it.",
            question="Shall I try the missing figures another way?",
            quick_replies=["Try again", "Explain what's missing", "Show the code"]))
    elif attempted and not produced:
        out.append(assistant_prompt(
            "result", f"I don't have an answer to “{goal}” yet; nothing above is a result.",
            question="Shall I try again, or would you put the question another way?",
            quick_replies=["Try again", "Show the code",
                           {"label": SWARM_REPLY["label"], "send": SWARM_REPLY["send"].format(goal=goal)}]))
    elif attempted:
        options = []
        replies: list[Any] = []
        if seg and seg[0] not in touched:
            options.append(f"compare by {_col(seg[0])}")
            replies.append(f"Compare by {seg[0]}")
        if read.time_column and read.time_column not in touched:
            options.append(f"trend it over {_col(read.time_column)}")
            replies.append(f"Trend over {read.time_column}")
        # The outcome the goal named comes first in read.targets, so "what
        # drives stroke?" is offered a model of stroke, not of the
        # heuristic's favourite column; any other column the question only
        # touched is not offered again.
        if read.target and (read.target in named or read.target not in touched):
            options.append(f"train a model on {_col(read.target)}")
            replies.append(f"Train a model on {read.target}")
        replies.append({"label": SWARM_REPLY["label"], "send": SWARM_REPLY["send"].format(goal=goal)})
        # After a failure there is no answer to point at: "That's my answer"
        # under "the run didn't finish" claims one that does not exist.
        msg = (f"I don't have an answer to “{goal}” yet." if failed
               else f"That's my answer to “{goal}”.")
        if options:
            msg += " From here I could " + ", ".join(options[:-1]) + (f" or {options[-1]}" if len(options) > 1 else options[0]) + "."
        msg += " Or I can send my team of clones at it in parallel."
        out.append(assistant_prompt("result", msg, question="What would help you most next?",
                                    quick_replies=replies[:MAX_REPLIES]))

    # A model swap is never silent (owner ruling 2026-11-02): when a fallback
    # tier answered this turn, a plain, code-written line says which model did
    # and why, before the turn's result.
    swaps = [assistant_prompt("verification", model_swap_sentence(rec)) for rec in _model_swaps(final)]
    if swaps:
        at = next((i for i, prompt in enumerate(out) if prompt.get("step") == "result"), len(out))
        out[at:at] = swaps
    return out
