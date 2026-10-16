"""Comprehension scorers — did the system understand the file it was given?

Every scorer in :mod:`avaloka.benchmark.scoring` grades what Avaloka *did* with
a frame: the lane the router picked, the roles the profiler inferred, whether
the model beat a baseline. None of them grades whether the frame is the table
that was in the file. A workbook read two rows above its header yields a frame
whose columns are ``Unnamed: 0 … Unnamed: 13`` and whose first data rows are
the real header — and every check in that module can pass on it, because each
one grades an operation on whatever frame it is handed. That is how a run whose
insight text named ``'Unnamed: 13'`` as a key driver scored 1.00.

These scorers compare the reader's answer against ground truth declared by the
dataset builder (:class:`avaloka.benchmark.spec.GroundTruth`), and they read the
*on-disk artifacts* a user would open rather than in-memory objects, matching
the convention in ``scoring.py``.

Two deliberate differences from the rest of the suite, both about refusing to
pass vacuously:

* A missing artifact is a failing check with a detail naming the file, not an
  exception and not a skip. "The mission wrote no profile" is itself an answer
  about comprehension.
* A check that had nothing to look at says so in its detail, with counts, so a
  green line in the scorecard can be distinguished from a line that graded
  nothing.
"""

from __future__ import annotations

import html as _html
import json
import re
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

from avaloka.benchmark.spec import CheckResult, Expectations

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
#: The two shapes a reader invents when it has no name for a column: pandas'
#: own filler for an unlabelled spreadsheet column, and Avaloka's filler for a
#: headerless CSV. Anchored, for testing a resolved column name.
PLACEHOLDER_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^Unnamed:\s*\d+$"),
    re.compile(r"^column_\d+$"),
)

#: The same two shapes, unanchored, for finding them inside prose. A report
#: sentence that contains ``Unnamed: 13`` anywhere is the user-visible defect
#: whether or not the name happens to be quoted.
_PLACEHOLDER_IN_TEXT = re.compile(r"Unnamed:\s*\d+|column_\d+")

#: Artifacts that carry generated prose about the data. The ones a mission
#: actually writes today are ``goal_analysis.json`` (the measured answer and
#: its findings list, written by AnalysisPlanner for analyze missions),
#: ``mission.json``, the three HTML reports and ``model_card.md``.
#: ``findings.json`` is listed because the brief for these scorers named it and
#: because `avaloka/arena/submission.py` reads one from a submission bundle —
#: no mission firefly writes it, so it is scanned when present and never
#: required. If this list ever reduces to files a mission does not write, the
#: narration check grades nothing; that is why it fails when none are found.
NARRATION_ARTIFACTS: tuple[str, ...] = (
    "goal_analysis.json",
    "findings.json",
    "mission.json",
    "model_card.md",
    "executive_report.html",
    "technical_report.html",
    "evaluation_report.html",
)

#: Artifacts carrying *measured* figures, for grading a planted answer.
_MEASURED_ARTIFACTS: tuple[str, ...] = ("goal_analysis.json", "data_quality.json")

#: Quoted tokens that are never column references. Every entry is a value one
#: of the real emitters puts in quotes, traced to the line that quotes it, so
#: this list is auditable and cannot quietly grow to swallow a real violation.
#: Nothing generic belongs here — no "total", "count", "value" — because those
#: are plausible column names and excluding them would hide the defect.
#:
#:   verdicts     reporter._narrative: "Validation returned 'warn'"
#:   tasks        reporter._narrative: "planned a 'regression' approach"
#:   splits       analysis_planner's ledger line: "split='time_based_holdout'"
#:   algorithms   reporter._findings: "Best model 'random_forest' scored …"
#:   metrics      the metric name in that same sentence
#:   roles        stats.infer_role, quoted in quality-issue details
#:   lanes        workload.route, quoted in routing prose
#:   fireflies    Firefly.name, printed in the ledger table of every report
#:   'Unknown'    goal_analysis.py: the imputation category it recommends
#:   'n/a'        reporter._narrative's fallback when there is no verdict
_NON_COLUMN_TOKENS: frozenset[str] = frozenset({
    # verdicts
    "pass", "warn", "fail", "n/a",
    # planner tasks. These are the literal values AnalysisPlanner emits into
    # mission.json's "task" and quotes in the executive report, enumerated from
    # the source rather than guessed: the first version of this list carried
    # "exploratory" where the emitter writes "exploratory_analysis", so the
    # narration check reported a violation on every single analyze task. A
    # check that cries wolf on every run is a check that gets ignored, which is
    # the same outcome as not having it.
    "regression", "binary_classification", "multiclass_classification",
    "classification", "exploratory_analysis", "goal_analysis", "clustering",
    # split strategies
    "time_based_holdout", "stratified_train_test", "random_train_test", "none",
    # candidate algorithms
    "logistic_regression", "random_forest", "gradient_boosting",
    "linear_regression",
    # metrics
    "roc_auc", "f1", "f1_macro", "accuracy", "r2", "rmse", "mae",
    # inferred column roles
    "numeric", "categorical", "datetime", "boolean", "constant", "id", "text",
    # workload lanes
    "online", "sampled_online", "batch",
    # firefly names
    "data_scout", "sampling_specialist", "data_engineer", "analysis_planner",
    "model_scientist", "ml_engineer", "validator", "finops", "reporter",
    # the one literal goal_analysis quotes that is not a column
    "unknown",
})

#: Opening quote not preceded by a word character, closing quote not followed by
#: one. That is what keeps possessives and contractions ("Avaloka's own",
#: "isn't") from being read as a quoted token and flooding the check with noise.
_QUOTED_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<!\w)'([^'\n]{1,60})'(?!\w)"),
    re.compile(r'(?<!\w)"([^"\n]{1,60})"(?!\w)'),
)

_NUMBER_IN_TEXT = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
_HTML_TAG = re.compile(r"<[^>]+>")
_PURE_NUMBER = re.compile(r"^[-+]?[\d,]+(?:\.\d+)?%?$")
#: Words too common to carry meaning when matching a planted answer's label.
_LABEL_STOPWORDS = frozenset({"the", "of", "for", "and", "per", "in", "by", "a", "an"})


# --------------------------------------------------------------------------
# Artifact helpers
# --------------------------------------------------------------------------
def _read_json_or_none(path: Path) -> Optional[Any]:
    """Parse an artifact, or None when it is absent or unreadable.

    Unreadable is folded into absent on purpose: both mean the check cannot be
    graded from this file, and the caller turns that into a failing check that
    names the file.
    """
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _resolved_columns(out: Path) -> Optional[list[str]]:
    """The column names the mission resolved, as its own profile reports them.

    Read from ``data_quality.json`` rather than from the frame, because the
    profile is what every downstream deliverable was built from and what a user
    inspecting the run would read.
    """
    profile = _read_json_or_none(out / "data_quality.json")
    if not isinstance(profile, dict):
        return None
    columns = profile.get("columns")
    if not isinstance(columns, list):
        return None
    return [str(c.get("name")) for c in columns if isinstance(c, dict) and "name" in c]


def _is_placeholder(name: str) -> bool:
    return any(p.match(name) for p in PLACEHOLDER_NAME_PATTERNS)


def _strip_html(markup: str) -> str:
    """Tags to spaces, then entities back to characters.

    The order matters and the second half is not optional. The reports are
    rendered by Jinja2 with ``autoescape=True``, so a finding that reads
    ``'Unnamed: 13' and 'revenue' are strongly associated`` is on disk as
    ``&#39;Unnamed: 13&#39; and &#39;revenue&#39; …``. A quoted-token regex run
    over the raw markup finds nothing at all in the HTML reports.
    """
    return _html.unescape(_HTML_TAG.sub(" ", markup))


def _json_strings(node: Any) -> Iterator[str]:
    """Every string anywhere in a parsed JSON document, keys included."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str):
                yield key
            yield from _json_strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _json_strings(item)


def _narration_sources(out: Path) -> list[tuple[str, str]]:
    """``(artifact name, text)`` for each narration-bearing artifact present."""
    sources: list[tuple[str, str]] = []
    for name in NARRATION_ARTIFACTS:
        path = out / name
        if not path.is_file():
            continue
        if name.endswith(".json"):
            document = _read_json_or_none(path)
            if document is None:
                continue
            text = "\n".join(_json_strings(document))
        else:
            try:
                raw = path.read_text()
            except OSError:
                continue
            text = _strip_html(raw) if name.endswith(".html") else raw
        sources.append((name, text))
    return sources


def _goal_echo(out: Path) -> str:
    """The user's own goal, as the mission recorded it.

    The goal is echoed verbatim into ``goal_analysis.json``, into
    ``mission.json`` and into the subtitle of every HTML report. It is the
    user's wording, not a claim the system made about the data, so it is
    removed before prose is read for column references: a goal that asks about
    ``'Gross margin'`` must not be scored as Avaloka naming a column that does
    not exist. Placeholder names are still looked for in the full text, because
    no user writes ``Unnamed: 13`` in a question.
    """
    for name in ("goal_analysis.json", "mission.json"):
        document = _read_json_or_none(out / name)
        if isinstance(document, dict) and isinstance(document.get("goal"), str):
            return document["goal"]
    return ""


def _quoted_candidates(text: str) -> list[str]:
    """Tokens the text presents as a quoted name, deduplicated in order."""
    seen: dict[str, None] = {}
    for pattern in _QUOTED_TOKEN_PATTERNS:
        for match in pattern.finditer(text):
            token = match.group(1).strip()
            if token:
                seen.setdefault(token, None)
    return list(seen)


#: Fields in ``goal_analysis.json`` that hold column names as *data* rather than
#: as prose. ``goal_analysis`` interpolates names into its findings unquoted
#: ("Gross margin: 12 missing of 122 rows"), so the quoted-token source cannot
#: see them on the analyze path -- which is the path this defect was reported
#: on. These fields carry the same names structurally, with no regex involved.
_GOAL_ANALYSIS_COLUMN_FIELDS: tuple[str, ...] = ("time_column", "value_column",
                                                 "group_column")

#: The ``kind`` values for which ``metrics`` is keyed by column name (
#: goal_analysis builds one entry per selected column). For the other kinds it
#: is keyed by "pairs", "groups", "monthly_values" and the like, which are not
#: column names and must not be tested as such.
_COLUMN_KEYED_METRICS_KINDS: frozenset[str] = frozenset({"missingness", "column_summary"})


def _structured_column_references(out: Path) -> list[tuple[str, str]]:
    """``(field, column name)`` for every column named as data, not as prose."""
    document = _read_json_or_none(out / "goal_analysis.json")
    if not isinstance(document, dict):
        return []
    found: list[tuple[str, str]] = []
    for name in document.get("columns") or []:
        if isinstance(name, str):
            found.append(("goal_analysis.json:columns", name))
    metrics = document.get("metrics")
    if isinstance(metrics, dict):
        for field in _GOAL_ANALYSIS_COLUMN_FIELDS:
            value = metrics.get(field)
            if isinstance(value, str):
                found.append((f"goal_analysis.json:metrics.{field}", value))
        if document.get("kind") in _COLUMN_KEYED_METRICS_KINDS:
            for key in metrics:
                if isinstance(key, str):
                    found.append(("goal_analysis.json:metrics key", key))
    return found


def _is_column_reference(token: str) -> bool:
    """Whether a quoted token is worth testing against the frame's columns."""
    if len(token) < 2:
        return False
    if _PURE_NUMBER.match(token):
        return False
    return token.lower() not in _NON_COLUMN_TOKENS


# --------------------------------------------------------------------------
# Comprehension scorers
# --------------------------------------------------------------------------
def score_placeholder_column_names(out: Path, exp: Expectations) -> list[CheckResult]:
    """No resolved column may be a name the reader invented.

    Catches the workbook read off its header row: pandas labels the unnamed
    columns ``Unnamed: N`` and the real header becomes a data row, so the
    mission profiles, transforms and reports a table that is not in the file.
    Nothing else in the suite looks at the names themselves.
    """
    if not exp.no_placeholder_column_names:
        return []
    resolved = _resolved_columns(out)
    if resolved is None:
        return [CheckResult(
            "no_placeholder_column_names", False,
            f"cannot grade: no readable 'columns' list in {out / 'data_quality.json'}",
            weight=2.0)]
    if not resolved:
        return [CheckResult(
            "no_placeholder_column_names", False,
            "data_quality.json resolved zero columns, so no name was graded",
            weight=2.0)]
    offenders = [c for c in resolved if _is_placeholder(c)]
    detail = (f"{len(offenders)} placeholder name(s) resolved: {offenders} "
              f"(of {len(resolved)} columns: {resolved})") if offenders else (
        f"all {len(resolved)} resolved names are real: {resolved}")
    return [CheckResult("no_placeholder_column_names", not offenders, detail, weight=2.0)]


def score_header_row(out: Path, exp: Expectations,
                     frame: pd.DataFrame) -> list[CheckResult]:
    """The reader must identify the row the header is actually on.

    ``avaloka.io.loader._read_excel`` searches for the header row and records
    its choice in ``frame.attrs["header_row"]`` (1-based). Reporting nothing is
    a failure, not an absence of evidence: a reader that cannot say which row it
    used has not identified one, and the off-by-two read is invisible until it
    does.
    """
    if exp.header_row_identified is None:
        return []
    got = frame.attrs.get("header_row")
    if got is None:
        return [CheckResult(
            "header_row_identified", False,
            "the reader reported no header row at all (frame.attrs['header_row'] "
            f"is unset); expected {exp.header_row_identified}. A reader that "
            "cannot name the row it used has not identified it")]
    return [CheckResult(
        "header_row_identified", got == exp.header_row_identified,
        f"reader reports header on row {got} (1-based), expected "
        f"{exp.header_row_identified}")]


def score_sheet(out: Path, exp: Expectations, frame: pd.DataFrame) -> list[CheckResult]:
    """The reader must land on the sheet the answer is on.

    Both readers choose a sheet by size -- ``avaloka/io/loader.py::_read_excel``
    takes the sheet with the most non-empty cells, and
    ``file_handler/excel_connector.py::_resolve_sheet`` the one with the most
    rows. Neither asks what the question was about. So a reference tab, an
    appendix, or simply the longer of two financial statements wins, and the
    mission then profiles, narrates and answers off a table nobody asked for.

    This is the worst failure shape available: no exception, no placeholder
    column, no missing value. The wrong sheet reads perfectly, so every other
    check in this module can pass on it and the number that comes out is
    confident and wrong.

    Graded only when the expectation names a sheet; a single-sheet file has no
    choice to get wrong and would only add noise.
    """
    if not exp.sheet_identified:
        return []
    chosen = frame.attrs.get("sheet_name")
    if chosen is None:
        return [CheckResult(
            "sheet_identified", False,
            f"the reader did not report which sheet it used, so its choice "
            f"cannot be checked against '{exp.sheet_identified}'. A reader that "
            "cannot name the sheet it read has not identified it.", weight=2.0)]
    passed = str(chosen) == str(exp.sheet_identified)
    detail = (f"reader resolved sheet '{chosen}', expected "
              f"'{exp.sheet_identified}'")
    if not passed:
        detail += (" -- the answer is not on the sheet that was read, so every "
                   "figure downstream is off the wrong table")
    return [CheckResult("sheet_identified", passed, detail, weight=2.0)]


def score_expected_columns(out: Path, exp: Expectations) -> list[CheckResult]:
    """Every ground-truth column name must be resolved exactly.

    Exactly: no case-folding and no whitespace-stripping. ``' Revenue '`` is a
    different name from ``Revenue`` to every consumer downstream — a transform
    keyed on the name, a feature contract, a user's own query — so a reader that
    keeps the spreadsheet's padding has not resolved the column.
    """
    if not exp.expected_columns_resolved:
        return []
    resolved = _resolved_columns(out)
    if resolved is None:
        return [CheckResult(
            "expected_columns_resolved", False,
            f"cannot grade: no readable 'columns' list in {out / 'data_quality.json'}")]
    missing = [name for name in exp.expected_columns_resolved if name not in resolved]
    detail = (f"missing {len(missing)} of {len(exp.expected_columns_resolved)} "
              f"ground-truth column(s): {missing}; resolved instead: {resolved}") if missing else (
        f"all {len(exp.expected_columns_resolved)} ground-truth columns resolved; "
        f"resolved: {resolved}")
    return [CheckResult("expected_columns_resolved", not missing, detail)]


def score_expected_dtypes(out: Path, exp: Expectations,
                          frame: pd.DataFrame) -> list[CheckResult]:
    """Columns must arrive as the kind of thing they are.

    A header read as data makes every column ``object``: the revenue column
    becomes text, so it is profiled as categorical, dropped from correlations
    and never modelled. The column is present and correctly named, and the
    frame is still wrong.
    """
    if not exp.expected_dtypes:
        return []
    checks: list[CheckResult] = []
    for col, kind in exp.expected_dtypes.items():
        if col not in frame.columns:
            checks.append(CheckResult(
                f"dtype[{col}]", False,
                f"column is ABSENT from the frame, so its dtype could not be "
                f"checked against '{kind}'; frame has {list(frame.columns)}"))
            continue
        series = frame[col]
        actual = str(series.dtype)
        numeric = bool(pd.api.types.is_numeric_dtype(series))
        if kind == "numeric":
            ok = numeric
        elif kind == "datetime":
            ok = bool(pd.api.types.is_datetime64_any_dtype(series))
        elif kind == "text":
            ok = bool(pd.api.types.is_object_dtype(series)
                      or isinstance(series.dtype, pd.StringDtype)) and not numeric
        else:
            checks.append(CheckResult(
                f"dtype[{col}]", False,
                f"expectation names an unknown dtype kind '{kind}'; "
                "use 'numeric', 'datetime' or 'text'"))
            continue
        checks.append(CheckResult(
            f"dtype[{col}]", ok, f"dtype is {actual}, expected kind '{kind}'"))
    return checks


def score_row_count(out: Path, exp: Expectations,
                    frame: pd.DataFrame) -> list[CheckResult]:
    """The table's row count, exactly.

    Off-by-a-few is the signature of a misread boundary: a header counted as
    data, a totals row counted as data, a trailing blank block counted as rows.
    Each one changes every mean, every missingness percentage and every quality
    score computed from the frame, and none of them is visible in a check that
    only grades those numbers against each other.
    """
    if exp.expected_row_count is None:
        return []
    got = len(frame)
    diff = got - exp.expected_row_count
    profile = _read_json_or_none(out / "data_quality.json")
    reported: Any = None
    if isinstance(profile, dict):
        reported = (profile.get("dataset") or {}).get("n_rows")
    suffix = f"; data_quality.json reports n_rows={reported}" if reported is not None else ""
    return [CheckResult(
        "row_count", diff == 0,
        f"frame has {got} rows, expected {exp.expected_row_count} "
        f"(difference {diff:+d}){suffix}")]


def score_narration_columns(out: Path, exp: Expectations,
                            frame: pd.DataFrame) -> list[CheckResult]:
    """Generated prose must not name a column that does not exist.

    This is the user-visible defect: an executive report whose key finding was
    ``'Unnamed: 13' and 'Unnamed: 7' are strongly associated``. It is measured
    nowhere else, because every other scorer reads structured fields and the
    defect lives in the sentences.

    Three sources of candidates, because the real emitters differ:

    * Quoted tokens. ``reporter._findings`` quotes column names
      (``f"'{p['a']}' and '{p['b']}' are strongly associated"``), so a quoted
      token that is not a column of the frame is a violation. The quote must
      not be flanked by word characters, which is what keeps possessives and
      contractions out of the candidate set, and the echoed goal is removed
      first (see :func:`_goal_echo`) so the user's own wording is not graded as
      the system's claim.
    * Placeholder names anywhere in the text, quoted or not, and *regardless*
      of whether the frame contains them. A frame that genuinely has an
      ``Unnamed: 13`` column is the bug, not an excuse for the prose naming it,
      and treating membership as a defence would make this check pass on
      exactly the run it was written for.
    * Structured references in ``goal_analysis.json`` (``columns``,
      ``metrics.{time,value,group}_column``, and the ``metrics`` keys for the
      column-keyed kinds). ``goal_analysis`` interpolates names into prose
      *unquoted*, and on an analyze mission ``reporter._findings`` returns its
      list verbatim -- so without this source the quoted branch sees nothing on
      the common path and only the placeholder branch bites.

    The exclusion list (``_NON_COLUMN_TOKENS``) is a module constant with each
    entry traced to the line that quotes it, so that the one way this check can
    be defused — growing that list until nothing is left to flag — is visible
    in review.
    """
    if not exp.narration_names_only_real_columns:
        return []
    sources = _narration_sources(out)
    if not sources:
        return [CheckResult(
            "narration_names_only_real_columns", False,
            "cannot grade: none of the narration-bearing artifacts "
            f"{list(NARRATION_ARTIFACTS)} exist in {out}, so no sentence was read",
            weight=2.0)]

    columns = {str(c) for c in frame.columns}
    goal = _goal_echo(out)
    violations: list[str] = []
    n_quoted = 0
    for name, text in sources:
        for token in _quoted_candidates(text.replace(goal, " ") if goal else text):
            if not _is_column_reference(token):
                continue
            n_quoted += 1
            if token not in columns:
                violations.append(f"{name}: quoted '{token}'")
        for match in dict.fromkeys(_PLACEHOLDER_IN_TEXT.findall(text)):
            entry = f"{name}: placeholder '{match}'"
            if entry not in violations:
                violations.append(entry)

    structured = _structured_column_references(out)
    for where, name in structured:
        if name not in columns:
            entry = f"{where}: '{name}'"
            if entry not in violations:
                violations.append(entry)

    if violations:
        detail = (f"{len(violations)} name(s) in generated text are not columns of "
                  f"the frame: {violations}; frame columns: {sorted(columns)}")
    else:
        detail = (f"{n_quoted} quoted and {len(structured)} structured column "
                  f"reference(s) across {len(sources)} artifact(s) "
                  f"({[n for n, _ in sources]}) all name real columns, and no "
                  "placeholder name appears in any of them")
    return [CheckResult("narration_names_only_real_columns", not violations,
                        detail, weight=2.0)]


def score_narration_against_ground_truth(out: Path, exp: Expectations) -> list[CheckResult]:
    """Generated prose must name only columns the **file** has.

    The strong form of :func:`score_narration_columns`, and a different
    question. That one grades the prose against the frame the mission built, so
    on a misread file it mostly asks whether the prose is self-consistent --
    and prose generated *from* that frame nearly always is. Only the
    placeholder branch bites, which makes the frame-relative check narrower
    than its name suggests.

    Here the reference set is ``expected_columns_resolved`` -- what the builder
    declared is in the file -- plus ``narration_allowed_extra_columns`` for
    anything the mission legitimately engineers. That catches the case the weak
    form cannot see: a two-row header collapsing to ``Sales`` and ``Sales.1``
    gives a report naming ``'Sales'``, which is true of the frame and false of
    the file.

    Skipped when no ground truth is declared. A task with no declared columns
    (a real EDGAR filing, say) would otherwise have every name in its report
    counted as a violation, which is noise rather than a finding.
    """
    if not exp.narration_columns_from_ground_truth:
        return []
    truth = {str(c) for c in exp.expected_columns_resolved}
    if not truth:
        return [CheckResult(
            "narration_columns_from_ground_truth", True,
            "not graded: the task declares no ground-truth columns, so there is "
            "no file-relative reference set to check the prose against",
            weight=0.0)]
    allowed = truth | {str(c) for c in exp.narration_allowed_extra_columns}

    sources = _narration_sources(out)
    if not sources:
        return [CheckResult(
            "narration_columns_from_ground_truth", False,
            "cannot grade: none of the narration-bearing artifacts "
            f"{list(NARRATION_ARTIFACTS)} exist in {out}, so no sentence was read",
            weight=2.0)]

    goal = _goal_echo(out)
    violations: list[str] = []
    checked = 0
    for name, text in sources:
        body = text.replace(goal, " ") if goal else text
        for token in _quoted_candidates(body):
            if not _is_column_reference(token):
                continue
            checked += 1
            if token not in allowed:
                entry = f"{name}: quoted '{token}'"
                if entry not in violations:
                    violations.append(entry)
        for match in dict.fromkeys(_PLACEHOLDER_IN_TEXT.findall(text)):
            entry = f"{name}: placeholder '{match}'"
            if entry not in violations:
                violations.append(entry)

    for where, candidate in _structured_column_references(out):
        checked += 1
        if candidate not in allowed:
            entry = f"{where}: '{candidate}'"
            if entry not in violations:
                violations.append(entry)

    if violations:
        detail = (f"{len(violations)} name(s) in generated text are not columns of "
                  f"the FILE: {violations}; ground truth: {sorted(truth)}"
                  + (f"; allowed extras: {sorted(set(allowed) - truth)}"
                     if allowed - truth else ""))
    else:
        detail = (f"{checked} column reference(s) across {len(sources)} artifact(s) "
                  f"({[n for n, _ in sources]}) all name columns the file has "
                  f"({sorted(truth)}), and no placeholder name appears in any of them")
    return [CheckResult("narration_columns_from_ground_truth", not violations,
                        detail, weight=2.0)]


# --------------------------------------------------------------------------
# The planted answer
# --------------------------------------------------------------------------
#: Keys whose string value is the *question*, echoed into the deliverables.
#: Excluded from answer-side matching: ``goal_analysis.json`` records the goal
#: verbatim and every HTML report prints it in the subtitle, so a label lifted
#: from the goal would be "named by the artifacts" on the strength of the
#: question having been asked. That is the vacuous pass this check exists to
#: avoid, and it is not hypothetical -- it was observed while building these.
_QUESTION_ECHO_KEYS: frozenset[str] = frozenset({"goal"})


def _words(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _numeric_leaves(node: Any, path: str = "",
                    context: tuple[str, ...] = ()) -> Iterator[tuple[str, float, set[str]]]:
    """``(key path, value, context words)`` for every number in a JSON document.

    ``context`` is the strings of the enclosing object and its parent -- which
    is where a metrics block says which column a figure belongs to
    (``{"value_column": "Revenue", "groups": [{"group": "EMEA", "value": …}]}``).
    Two levels only: carrying the whole document down would put the echoed goal
    text in every leaf's context and bind any label to any number.
    """
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        yield path, float(node), _words(" ".join(context) + " " + path)
    elif isinstance(node, dict):
        local = " ".join(
            [str(k) for k in node]
            + [v for k, v in node.items()
               if isinstance(v, str) and str(k).lower() not in _QUESTION_ECHO_KEYS])
        inherited = context[-1:] if context else ()
        for key, value in node.items():
            yield from _numeric_leaves(value, f"{path}.{key}" if path else str(key),
                                       inherited + (local,))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _numeric_leaves(item, f"{path}[{index}]", context)


def _label_words(label: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z0-9_]+", label.lower())
            if len(w) >= 3 and w not in _LABEL_STOPWORDS]


def _label_anchor(label: str) -> Optional[str]:
    """The label's most distinctive word: the longest, ties broken lexically.

    One word rather than all of them, because the benchmark author's wording and
    the emitter's wording differ on the statistic ("total Revenue" against
    ``aggregation: "sum"``) but agree on the subject. The anchor is reported in
    the detail so how strict this is can be read off the scorecard.
    """
    words = _label_words(label)
    return max(words, key=lambda w: (len(w), w)) if words else None


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _answer_texts(out: Path) -> list[tuple[str, str]]:
    """Narration with the echoed question removed, for answer-side matching."""
    goal = _goal_echo(out)
    if not goal:
        return _narration_sources(out)
    return [(name, text.replace(goal, " ")) for name, text in _narration_sources(out)]


def score_expected_answer(out: Path, exp: Expectations) -> list[CheckResult]:
    """A planted figure must be reproduced, and attached to what it measures.

    "Avaloka answered the question" has been reported on the strength of a
    report existing. This grades the answer: the number has to be in the
    measured artifacts within tolerance, and it has to appear *together* with
    the thing it is a figure for -- under one key path in
    ``goal_analysis.json``'s metrics, or in one sentence of the findings. A
    number loose somewhere in a profile is not an answer to anything, and the
    label's presence on its own is no evidence either, because the goal is
    echoed into every deliverable.

    Only ``goal_analysis.json`` carries measured, full-precision, goal-specific
    figures, and AnalysisPlanner writes it for analyze missions only. When it is
    absent the check fails and names the artifact a train mission would need,
    rather than falling back to the quality profile and passing on a coincidence.
    """
    if not exp.expected_answer:
        return []
    label = str(exp.expected_answer.get("label") or "")
    if "value" not in exp.expected_answer:
        return [CheckResult(
            "answer_value", False,
            f"expected_answer {exp.expected_answer!r} carries no 'value', so "
            "there is no figure to look for")]
    want = float(exp.expected_answer["value"])
    # Absolute tolerance when declared. The default is relative and tiny rather
    # than zero, because the emitters round measured figures to six decimals
    # (goal_analysis._number) and an exact float compare would fail on that
    # alone, which would make every planted answer unachievable.
    tolerance = exp.expected_answer.get("tolerance")
    tol = float(tolerance) if tolerance is not None else max(abs(want) * 1e-6, 1e-9)

    measured = {name: _read_json_or_none(out / name) for name in _MEASURED_ARTIFACTS}
    if measured.get("goal_analysis.json") is None:
        return [CheckResult(
            "answer_gradable", False,
            f"cannot grade the planted answer {exp.expected_answer!r}: "
            f"{out / 'goal_analysis.json'} does not exist. It is the only "
            "artifact carrying measured, goal-specific figures, and "
            "AnalysisPlanner writes it for analyze missions only -- a train "
            "mission would need an equivalent measured-answer artifact before "
            "this can be checked. Not passing on the quality profile's numbers, "
            "which answer no question")]

    leaves: list[tuple[str, str, float, set[str]]] = []
    for name, document in measured.items():
        if document is None:
            continue
        for path, value, context in _numeric_leaves(document):
            leaves.append((name, path, value, context))

    checks: list[CheckResult] = []
    hits = [row for row in leaves if abs(row[2] - want) <= tol]
    if hits:
        detail = (f"value {want!r} (+/-{tol!r}) measured at "
                  f"{[f'{n}:{p}={v!r}' for n, p, v, _ in hits[:5]]}")
    else:
        nearest = min(leaves, key=lambda row: abs(row[2] - want), default=None)
        detail = (f"no measured figure within +/-{tol!r} of {want!r}; nearest of "
                  f"{len(leaves)} figures was "
                  + (f"{nearest[0]}:{nearest[1]}={nearest[2]!r}" if nearest
                     else "nothing -- the artifacts hold no numbers"))
    checks.append(CheckResult("answer_value", bool(hits), detail))

    anchor = _label_anchor(label)
    if anchor is None:
        checks.append(CheckResult(
            "answer_bound", False,
            f"expected_answer label {label!r} has no word of three or more "
            "characters, so there is nothing to tie the figure to"))
        return checks + _answer_extras(out, exp)

    bound: list[str] = []
    for name, path, value, context in hits:
        if anchor in context:
            bound.append(f"{name}:{path} (context names {anchor!r})")
    if not bound:
        # Prose tolerance is looser than the figure tolerance on purpose: the
        # findings print measured values through '%.3g', so the sentence that
        # states the answer carries a rounded form of it.
        prose_tol = max(tol, abs(want) * 5e-3)
        for name, text in _answer_texts(out):
            for sentence in _sentences(text):
                if anchor not in _words(sentence):
                    continue
                numbers = [float(m.replace(",", ""))
                           for m in _NUMBER_IN_TEXT.findall(sentence)]
                if any(abs(n - want) <= prose_tol for n in numbers):
                    bound.append(f"{name}: {sentence.strip()[:120]}")
                    break
            if bound:
                break
    checks.append(CheckResult(
        "answer_bound", bool(bound),
        f"figure {want!r} appears together with the label's anchor word "
        f"{anchor!r} (from label {label!r}) at {bound[:3]}" if bound
        else (f"the label's anchor word {anchor!r} (from label {label!r}) never "
              f"appears together with {want!r}: no metrics key path or sibling "
              "value names it, and no findings sentence states both -- so "
              "nothing in the artifacts asserts this figure IS that answer")))
    return checks + _answer_extras(out, exp)


def _answer_extras(out: Path, exp: Expectations) -> list[CheckResult]:
    """The optional ``period`` and ``unit`` qualifiers, when declared.

    ``unit`` is here because the contract carries it, but no Avaloka artifact
    emits a unit string today: goal_analysis records aggregations and values,
    never currencies. A declared unit therefore fails until something writes
    one, and the detail says so rather than letting the reader conclude the
    mission dropped it.
    """
    checks: list[CheckResult] = []
    combined = "\n".join(text for _, text in _answer_texts(out)).lower()
    names = [n for n, _ in _answer_texts(out)]
    for key in ("period", "unit"):
        wanted = exp.expected_answer.get(key)
        if wanted in (None, ""):
            continue
        present = str(wanted).lower() in combined
        note = ("" if key != "unit" or present
                else "; note that no Avaloka artifact emits a unit string today")
        checks.append(CheckResult(
            f"answer_{key}", present,
            f"{key} {str(wanted)!r} {'appears' if present else 'does NOT appear'} "
            f"in {names} (with the echoed goal removed){note}"))
    return checks



# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def score_comprehension(out: Path, exp: Expectations,
                        frame: pd.DataFrame) -> list[CheckResult]:
    """Every comprehension check for one run, in reading order.

    The narration and placeholder checks carry weight 2.0: they are the two
    defects a user sees without opening an artifact, and a run that fails either
    one has not understood its input, whatever the rest of the scorecard says.
    """
    checks: list[CheckResult] = []
    checks += score_placeholder_column_names(out, exp)
    checks += score_header_row(out, exp, frame)
    checks += score_sheet(out, exp, frame)
    checks += score_expected_columns(out, exp)
    checks += score_expected_dtypes(out, exp, frame)
    checks += score_row_count(out, exp, frame)
    checks += score_narration_columns(out, exp, frame)
    checks += score_narration_against_ground_truth(out, exp)
    checks += score_expected_answer(out, exp)
    return checks


__all__ = [
    "PLACEHOLDER_NAME_PATTERNS",
    "NARRATION_ARTIFACTS",
    "score_comprehension",
    "score_expected_answer",
    "score_expected_columns",
    "score_expected_dtypes",
    "score_header_row",
    "score_narration_against_ground_truth",
    "score_narration_columns",
    "score_placeholder_column_names",
    "score_row_count",
    "score_sheet",
]
