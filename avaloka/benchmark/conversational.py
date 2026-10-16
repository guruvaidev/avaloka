"""Conversational probes as a scored benchmark family.

``tests/conversational/probes.py`` is a deterministic scoring harness: eleven
probes across four dimensions, each with a structural evaluator and a declared
``pass`` metric. It was reachable only from pytest, so its output was a green
dot. "Passed" is not a number, and a scorecard with no conversational line reads
as though conversational quality were not measured at all. This module turns the
same probes into :class:`~avaloka.benchmark.spec.TaskResult` objects so the
family appears in the scorecard with a score, per dimension, next to the
data-engineering and data-science families.

**What the number means.** With no provider configured the subject is a
*reference responder* from ``probes.REFERENCE_RESPONDERS`` — a fixed dictionary
of hand-written replies. Scoring those measures whether the evaluators still
discriminate: whether the written-to-pass bracket passes, the sycophant bracket
fails, and the keyword-stuffed junk string scores at the floor. It does **not**
measure Avaloka's conversational ability; nothing in a hermetic run calls
Avaloka or any model. Measuring the product needs ``live=True`` and a reachable
provider, and the task's ``source_scheme`` records which subject produced every
score so the two claims cannot be confused in the scorecard.

**Rules this module obeys.**

*Hermetic by default.* With every provider key unset the family runs, scores,
and touches no network. ``urllib`` is imported inside the live helpers rather
than at module scope, so ``live=False`` can be shown not to have reached for a
transport at all.

*Never Anthropic.* The owner's standing ruling is that no Anthropic key is used
for any benchmark or test bench. This module reads ``GROQ_API_KEY`` and
``OLLAMA_HOST`` and nothing else, and :func:`_refuse_anthropic` rejects an
Anthropic-looking ``model`` override rather than quietly honouring it.

*Deterministic grading.* No model grades a model. Every metric comes from the
probe's own evaluator. The junk responder runs as part of the family, not only
in pytest, and the family goes red if junk starts scoring well.

*A missing suite is not a zero.* ``tests`` is not an installed package, so
``tests.conversational.probes`` will not import from a wheel. The loader falls
back to a path search of the repo and, failing both, reports a SKIP — a
``TaskResult`` with no checks and an ``error`` containing ``"unavailable"``,
which is the convention ``avaloka/benchmark/__main__.py`` and
``avaloka/cli.py`` already use to separate "could not run" from "ran and scored
zero".
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from avaloka.benchmark.spec import CheckResult, TaskResult

__all__ = [
    "PRODUCT_GAP_REASON",
    "PRODUCT_RESPONDER_ATTR",
    "PRODUCT_RESPONDER_MODULE",
    "ProductPathUnavailable",
    "FAMILY",
    "JUNK_SCORE_FLOOR",
    "DIMENSION_FLOOR",
    "Exchange",
    "ProbeSuite",
    "ProbeSuiteUnavailable",
    "ProviderUnavailable",
    "conversational_results",
    "load_probe_suite",
]

#: The family name every TaskResult carries, so Scorecard.by_family() groups
#: these runs under one heading.
#: The family is named for what it grades, not for the subject area it is
#: about. It scores the probe *scorers* -- and, with ``--live``, a raw model --
#: and it does not measure Avaloka's conversational behaviour at all, because
#: no path on this branch drives the probes through Avaloka's agent graph. A
#: family called "conversational" sitting at 1.00 reads as a product score, and
#: that is precisely how a 0.80 from an out-of-tree harness came to be quoted
#: as one. The name is the first line of defence against the next such quote.
FAMILY = "conversational_scorers"

#: Dotted name of the probe suite, when it happens to be importable.
PROBE_MODULE = "tests.conversational.probes"

#: Where to look for the suite on disk, relative to a repository root.
PROBE_RELPATH = Path("tests") / "conversational" / "probes.py"

#: Attributes this module uses. Checked at load time so a renamed probe API
#: surfaces as a SKIP naming the missing attribute, rather than an
#: AttributeError mid-run or a score computed from half an API.
REQUIRED_ATTRS = ("PROBES", "BY_DIMENSION", "REFERENCE_RESPONDERS", "Dimension",
                  "run_probe", "score")

#: The reference responder graded when no live provider is in play. The
#: BRILLIANT bracket in probes.py is written to satisfy every evaluator, so its
#: score is the evaluators' own upper bracket.
REFERENCE_SUBJECT = "brilliant"

#: The responder wired in as the negative control. A module constant rather
#: than a literal so the forced-failure drill (point it at "brilliant" and watch
#: the family go red) perturbs the real code path and not a copy of it.
CONTROL_RESPONDER_NAME = "junk"

#: The highest overall score the junk responder may reach before this family
#: fails. "Floor" in the sense of the level junk must stay down at; it is an
#: upper bound on the control's score, and exceeding it is the failure.
#:
#: JUNK is a keyword-stuffed string with no analytical content, and the reason
#: the probes were rewritten structurally is that the scorers they replaced gave
#: it 6 of 6. Measured against this suite it scores 0.0000 — it passes no probe.
#: The floor is nonetheless set above zero: a structural evaluator can
#: legitimately credit junk on a probe whose pass condition is a single negative
#: ("invented nothing"), so one accidental pass out of eleven (0.0909) should not
#: turn the family red for a reason that is not the one being watched for. 0.20
#: tolerates that and is still far below any score that could be mistaken for
#: competence. The slack is deliberate and currently unexercised.
JUNK_SCORE_FLOOR = 0.20

#: Every probe in a dimension must pass for the dimension's summary check to be
#: green. The reference bracket is written to clear this, so a reference run
#: below 1.0 means the evaluators or the bracket have drifted apart. A live
#: subject that cannot clear it fails for real, which is the measurement, not a
#: miscalibration — live runs are opt-in and may legitimately be red.
DIMENSION_FLOOR = 1.0

#: Local models to try, in preference order: the owner's named first choice,
#: then Gemma from smallest upward. The server's own tag list decides which of
#: these exists; nothing here is assumed present.
LOCAL_MODEL_PREFERENCE = ("qwen2.5:3b-instruct", "gemma3:270m", "gemma3:1b",
                          "gemma3:4b", "gemma4:e4b")

#: Hosted fallback, used only when no local model answers.
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

#: Seconds a single live turn may take before the run is abandoned as a SKIP.
LIVE_TIMEOUT_S = 90.0

#: What the live subject is told. Deliberately thin: it hands over the probe's
#: evidence, because an agent answering these questions would have the figures,
#: and says nothing about asking, pushing back, disclosing sampling or any other
#: behaviour a probe grades. A prompt that coached those would be scoring the
#: prompt.
LIVE_SYSTEM_PROMPT = (
    "You are a data analyst answering a colleague about a dataset you have "
    "already examined. The evidence you have is the JSON below. Answer only "
    "from it; if a figure is not in it, say that it is not in the data. Be "
    "brief.\n\nEvidence: {evidence}"
)


# --------------------------------------------------------------------------- #
# Locating the probe suite
# --------------------------------------------------------------------------- #

#: Where a responder that drives **Avaloka itself** would live.
#:
#: Nothing imports this module today and it does not exist; that absence is the
#: gap, and ``_product_responder`` reports it. It is written as a lookup rather
#: than a hardcoded ``False`` so the check grades the world instead of grading a
#: constant: the day someone adds this module with a ``product_responder``
#: factory, the check passes without anyone editing the check. A gap assertion
#: that has to be remembered and deleted by hand is a gap assertion that goes
#: stale and then lies.
PRODUCT_RESPONDER_MODULE = "avaloka.benchmark.conversational_product"
PRODUCT_RESPONDER_ATTR = "product_responder"

#: Why the gap exists, in one place, for the scorecard to carry. This is the
#: same device as ``DEFERRED_FORMATS`` in ``datasets_formats``: a limitation
#: belongs in code where review sees it, not in prose that nobody re-reads.
PRODUCT_GAP_REASON = (
    "no path on this branch drives the conversational probes through Avaloka's "
    "own agent graph. The hermetic numbers grade the probe scorers (a fixed "
    "dictionary of replies), and --live grades a raw model over HTTP; neither "
    "touches app.agents, the firefly swarm, the intent classifier or a "
    "MissionContext. So no number this family produces is a measurement of the "
    f"product. Closing it means adding {PRODUCT_RESPONDER_MODULE} with a "
    f"'{PRODUCT_RESPONDER_ATTR}()' factory returning a responder that drives "
    "Avaloka and returns the user-visible reply; this check then passes on its "
    "own."
)


class ProductPathUnavailable(RuntimeError):
    """No responder exists that drives Avaloka itself."""


class ProbeSuiteUnavailable(RuntimeError):
    """The probe suite could not be located, or does not expose the API used."""


class ProviderUnavailable(RuntimeError):
    """``live=True`` was asked for and no permitted provider could be reached."""


@dataclass(frozen=True)
class ProbeSuite:
    """A loaded probe module, plus how it was found.

    ``origin`` goes into the scorecard. A score whose provenance is "loaded by
    path from /some/checkout" is a different claim from one whose provenance is
    an installed import, and the difference has to be visible.
    """

    module: types.ModuleType
    origin: str


def _import_as_package() -> types.ModuleType:
    """Import the suite as ``tests.conversational.probes``.

    Works from a source checkout whose root is on ``sys.path``. Raises
    ImportError from an installed wheel, where ``tests`` is not shipped —
    ``pyproject.toml`` includes only ``avaloka*``.
    """
    return importlib.import_module(PROBE_MODULE)


def _candidate_probe_paths(start: Optional[Path] = None) -> list[Path]:
    """Paths at which ``tests/conversational/probes.py`` might sit.

    Walks up from this file and from the working directory. Returns only paths
    that exist, nearest first, de-duplicated by resolved location.
    """
    anchors: list[Path] = []
    if start is not None:
        anchors.append(Path(start))
    anchors.append(Path(__file__).resolve())
    try:
        anchors.append(Path.cwd().resolve())
    except OSError:                                   # cwd deleted under us
        pass

    found: list[Path] = []
    seen: set[Path] = set()
    for anchor in anchors:
        bases = [anchor, *anchor.parents] if anchor.is_dir() else list(anchor.parents)
        for base in bases:
            candidate = base / PROBE_RELPATH
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
    return found


def _load_from_path(path: Path) -> types.ModuleType:
    """Execute ``probes.py`` as a standalone module.

    Safe because the suite imports nothing but the standard library, so it does
    not need its package context. The module is registered under a private name
    to avoid colliding with a real ``tests.conversational.probes`` elsewhere in
    ``sys.modules``.
    """
    name = "_avaloka_benchmark_conversational_probes"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProbeSuiteUnavailable(
            f"conversational probe suite unavailable: no import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:                          # noqa: BLE001 - reported, not raised
        sys.modules.pop(name, None)
        raise ProbeSuiteUnavailable(
            f"conversational probe suite unavailable: {path} failed to import: "
            f"{type(exc).__name__}: {exc}") from exc
    return module


def load_probe_suite(*, start: Optional[Path] = None) -> ProbeSuite:
    """Find the probe suite, or raise :class:`ProbeSuiteUnavailable`.

    Tried in order: a plain import, then a path search upward from this file and
    the working directory. Not cached — loading re-executes a standard-library-only
    module, which is cheap, and a cache would hide a suite that appeared or
    vanished between calls.
    """
    attempts: list[str] = []
    try:
        module = _import_as_package()
    except Exception as exc:                          # noqa: BLE001 - one of several routes
        attempts.append(f"import {PROBE_MODULE}: {type(exc).__name__}: {exc}")
    else:
        return _validated(module, f"import {PROBE_MODULE}")

    for candidate in _candidate_probe_paths(start):
        try:
            module = _load_from_path(candidate)
        except ProbeSuiteUnavailable as exc:
            attempts.append(str(exc))
            continue
        return _validated(module, f"path {candidate}")

    attempts.append(f"path search for {PROBE_RELPATH}: no such file")
    raise ProbeSuiteUnavailable(
        "conversational probe suite unavailable — " + "; ".join(attempts))


def _validated(module: types.ModuleType, origin: str) -> ProbeSuite:
    missing = [attr for attr in REQUIRED_ATTRS if not hasattr(module, attr)]
    if missing:
        raise ProbeSuiteUnavailable(
            f"conversational probe suite unavailable: {origin} exposes no "
            f"{', '.join(missing)} — the probe API has changed")
    if not tuple(getattr(module, "PROBES")):
        raise ProbeSuiteUnavailable(
            f"conversational probe suite unavailable: {origin} defines no probes")
    return ProbeSuite(module=module, origin=origin)


# --------------------------------------------------------------------------- #
# Driving a responder, and recording what it was actually asked
# --------------------------------------------------------------------------- #

@dataclass
class Exchange:
    """One turn as it actually happened, kept so the scorecard can show it."""

    probe_id: str
    #: The text handed to the responder as the question.
    question: str
    #: The text the responder gave back.
    reply: str
    #: The question the probe declares, independently of what was sent.
    expected_question: str = ""

    @property
    def question_was_sent(self) -> bool:
        """Was a real question put to the responder?

        A probe graded on an empty prompt is the defect that made a recorded
        0.80 meaningless: the evaluator still returns metrics for ``""``, so the
        run produces a number that is a number of nothing. This is graded rather
        than asserted so it appears in the scorecard as a check that can go red,
        instead of an exception somebody disables.
        """
        return bool(self.question.strip()) and self.question == self.expected_question


#: A responder takes the probe and the question text that was selected for it,
#: and returns a reply. The question is passed in rather than re-derived so that
#: the string recorded in Exchange is provably the string the responder got.
Responder = Callable[[Any, str], str]


class _Recorder:
    """Adapts a :data:`Responder` to the ``Callable[[Probe], str]`` that
    ``probes.run_probe`` expects, capturing each exchange on the way through."""

    def __init__(self, responder: Responder) -> None:
        self._responder = responder
        self.exchanges: dict[str, Exchange] = {}

    def __call__(self, probe: Any) -> str:
        question = probe.prompt
        expected = probe.turns[-1].text if probe.turns else ""
        reply = self._responder(probe, question)
        self.exchanges[probe.id] = Exchange(probe_id=probe.id, question=question,
                                            reply=reply, expected_question=expected)
        return reply


def _reference_responder(suite: ProbeSuite, name: str) -> Responder:
    """Wrap one of ``probes.REFERENCE_RESPONDERS`` as a :data:`Responder`."""
    responders = getattr(suite.module, "REFERENCE_RESPONDERS")
    try:
        inner = responders[name]
    except KeyError as exc:
        raise ProbeSuiteUnavailable(
            f"conversational probe suite unavailable: no reference responder "
            f"{name!r} (have {sorted(responders)})") from exc

    def respond(probe: Any, question: str) -> str:
        # The reference responders are keyed on probe id and ignore the
        # question; it is still threaded through so the recorded exchange is
        # the same shape as a live one.
        del question
        return inner(probe)

    return respond


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

def _fmt_metrics(metrics: Mapping[str, float]) -> str:
    return " ".join(f"{k}={v:g}" for k, v in sorted(metrics.items()))


def _excerpt(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def _probe_checks(results: Sequence[Any], exchanges: Mapping[str, Exchange],
                  *, live: bool) -> list[CheckResult]:
    """Per-probe checks: did it pass, and was it asked a real question."""
    checks: list[CheckResult] = []
    for result in results:
        exchange = exchanges.get(result.probe_id)
        reply = _excerpt(exchange.reply) if exchange else "<no exchange recorded>"
        checks.append(CheckResult(
            result.probe_id, result.passed,
            f"dimension={result.dimension.value}; {_fmt_metrics(result.metrics)}; "
            f"latency={result.latency_s:.3f}s; reply={reply!r}"))

        if exchange is None:
            checks.append(CheckResult(
                f"{result.probe_id}::question_sent", False,
                "no exchange recorded — the responder was never called with this probe"))
        else:
            checks.append(CheckResult(
                f"{result.probe_id}::question_sent", exchange.question_was_sent,
                f"sent {len(exchange.question)} chars: {exchange.question!r}; "
                f"probe declares: {exchange.expected_question!r}"))

        if live and result.within_budget is not None:
            # Only graded for a live subject. A reference responder is a dict
            # lookup, so its latency check could not fail and would be noise.
            checks.append(CheckResult(
                f"{result.probe_id}::within_latency_budget", bool(result.within_budget),
                f"answered in {result.latency_s:.2f}s, inside the probe's "
                f"budget={bool(result.within_budget)}"))
    return checks


def _dimension_summary(dimension_value: str, results: Sequence[Any]) -> CheckResult:
    """One readable line per dimension.

    Weight 0.0 on purpose: every probe it summarises is already counted as its
    own check, and counting the summary too would silently double the weight of
    a dimension holding few probes.
    """
    passed = sum(1 for r in results if r.passed)
    fraction = passed / len(results) if results else 0.0
    return CheckResult(
        f"{dimension_value}_score", fraction >= DIMENSION_FLOOR,
        f"{passed}/{len(results)} probes passed (score {fraction:.2f}, "
        f"floor {DIMENSION_FLOOR:.2f})", weight=0.0)


def _subject_label(*, live: bool, model: Optional[str], subject: str) -> str:
    return f"live:{model}" if live else f"ref:{subject}"


def _skip(task: str, kind: str, error: str) -> TaskResult:
    """A SKIP, in the shape the rest of the benchmark already recognises.

    No checks and an ``error`` containing "unavailable": ``__main__`` prints it
    as SKIP and ``cli.py`` excludes it from hard failures. The score is 0.0, as
    it is for any check-less result, but the error is what distinguishes "not
    measured" from "measured zero" — and the two must not look the same.
    """
    return TaskResult(task=task, family=FAMILY, kind=kind, source_scheme="—",
                      checks=[], error=error)


# --------------------------------------------------------------------------- #
# The negative control
# --------------------------------------------------------------------------- #

def _control_result(suite: ProbeSuite) -> TaskResult:
    """Grade the scorers themselves: junk low, empty zero, sycophant below brilliant.

    This runs on every invocation, live or not, because it is a property of the
    evaluators rather than of the subject. It is the check that was only ever a
    pytest assertion, and it is the reason the probes were written structurally:
    the scorers they replaced gave the junk string 6 of 6.
    """
    module = suite.module
    run_probe = getattr(module, "run_probe")
    score = getattr(module, "score")
    probes = tuple(getattr(module, "PROBES"))
    dimension_enum = getattr(module, "Dimension")

    scores: dict[str, dict[str, float]] = {}
    for name in (CONTROL_RESPONDER_NAME, "empty", "brilliant", "sycophant"):
        recorder = _Recorder(_reference_responder(suite, name))
        results = [run_probe(p, recorder) for p in probes]
        scores[name] = score(results)

    control = scores[CONTROL_RESPONDER_NAME]
    checks: list[CheckResult] = [CheckResult(
        "junk_overall_at_or_below_floor",
        control["overall"] <= JUNK_SCORE_FLOOR,
        f"the {CONTROL_RESPONDER_NAME!r} responder scored {control['overall']:.4f} "
        f"overall against a floor of {JUNK_SCORE_FLOOR:.2f}; above the floor the "
        f"evaluators are rewarding vocabulary rather than behaviour",
        weight=3.0)]

    for dimension in dimension_enum:
        value = dimension.value
        if value not in control:
            continue
        checks.append(CheckResult(
            f"junk_{value}_at_or_below_floor", control[value] <= JUNK_SCORE_FLOOR,
            f"{control[value]:.4f} against a floor of {JUNK_SCORE_FLOOR:.2f}"))

    empty_overall = scores["empty"]["overall"]
    checks.append(CheckResult(
        "empty_reply_scores_zero", empty_overall == 0.0,
        f"the empty responder scored {empty_overall:.4f}; anything above zero "
        f"means a probe passes on no answer at all", weight=2.0))

    brilliant_overall = scores["brilliant"]["overall"]
    sycophant_overall = scores["sycophant"]["overall"]
    checks.append(CheckResult(
        "sycophant_scores_below_brilliant", sycophant_overall < brilliant_overall,
        f"sycophant {sycophant_overall:.4f} < brilliant {brilliant_overall:.4f}; "
        f"equal scores mean the evaluators cannot tell agreement from analysis",
        weight=2.0))

    return TaskResult(
        task="conv_negative_control", family=FAMILY, kind="control",
        source_scheme=f"ref:{CONTROL_RESPONDER_NAME}", checks=checks)


# --------------------------------------------------------------------------- #
# Live providers — opt-in, local first, never Anthropic
# --------------------------------------------------------------------------- #

def _refuse_anthropic(model: str) -> None:
    """The standing ruling, enforced in code rather than left to discipline."""
    lowered = model.lower()
    if "claude" in lowered or "anthropic" in lowered:
        raise ProviderUnavailable(
            f"refusing model {model!r}: an Anthropic model may not be used for "
            f"any Avaloka benchmark or test bench")


def _ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


def _post_json(url: str, payload: dict[str, Any], *,
               headers: Optional[dict[str, str]] = None,
               timeout: float = LIVE_TIMEOUT_S) -> dict[str, Any]:
    """One JSON POST over the standard library.

    ``urllib`` is imported here rather than at module scope so that a
    ``live=False`` run can be shown never to have pulled in a transport.
    """
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        raise ProviderUnavailable(f"{url} returned HTTP {exc.code}: {body}") from exc
    except Exception as exc:                          # noqa: BLE001 - any transport fault
        raise ProviderUnavailable(f"{url} unreachable: {type(exc).__name__}: {exc}") from exc


def _local_model(preferred: Optional[str] = None) -> Optional[str]:
    """The first model in :data:`LOCAL_MODEL_PREFERENCE` the local server has.

    Returns None when no local server answers, so the caller can fall through to
    Groq. Availability is read from the server's own tag list; nothing is
    assumed installed.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{_ollama_host()}/api/tags", timeout=5) as response:
            tags = json.loads(response.read().decode("utf-8"))
    except Exception:                                 # noqa: BLE001 - no local server
        return None
    present = {entry.get("name", "") for entry in tags.get("models", [])}
    present |= {entry.get("model", "") for entry in tags.get("models", [])}
    if preferred:
        return preferred if preferred in present else None
    for candidate in LOCAL_MODEL_PREFERENCE:
        if candidate in present:
            return candidate
    return None


def _messages(probe: Any, question: str) -> list[dict[str, str]]:
    """The probe's turns as chat messages, with ``question`` as the last one.

    Earlier turns are included because two probes are multi-turn and grade
    whether context survived; the final user message is the recorded question
    verbatim, so what the question-sent check reports is what went out.
    """
    evidence = json.dumps(probe.evidence, sort_keys=True, default=str)
    messages = [{"role": "system", "content": LIVE_SYSTEM_PROMPT.format(evidence=evidence)}]
    for turn in list(probe.turns)[:-1]:
        messages.append({"role": turn.role, "content": turn.text})
    messages.append({"role": "user", "content": question})
    return messages


def _live_responder(model: str, *, local: bool) -> Responder:
    """A :data:`Responder` backed by Ollama or Groq. Raises on any fault.

    A provider error aborts the whole live run (the caller turns it into a
    SKIP). Grading a transport failure as an empty answer would report a
    network problem as a conversational one.
    """
    if local:
        def respond(probe: Any, question: str) -> str:
            body = _post_json(f"{_ollama_host()}/api/chat", {
                "model": model, "messages": _messages(probe, question),
                "stream": False, "options": {"temperature": 0.0}})
            return str(body.get("message", {}).get("content", ""))
        return respond

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise ProviderUnavailable("GROQ_API_KEY is not set")

    def respond(probe: Any, question: str) -> str:
        body = _post_json(GROQ_ENDPOINT, {
            "model": model, "messages": _messages(probe, question),
            "temperature": 0.0}, headers={"Authorization": f"Bearer {key}"})
        choices = body.get("choices") or []
        if not choices:
            raise ProviderUnavailable(f"{GROQ_ENDPOINT} returned no choices")
        return str(choices[0].get("message", {}).get("content", ""))

    return respond


def _resolve_live(model: Optional[str]) -> tuple[Responder, str]:
    """Pick a provider under the policy: local first, then Groq, never Anthropic."""
    if model:
        _refuse_anthropic(model)
    local = _local_model(model)
    if local:
        return _live_responder(local, local=True), local
    hosted = model or GROQ_MODEL
    _refuse_anthropic(hosted)
    return _live_responder(hosted, local=False), hosted


# --------------------------------------------------------------------------- #
# The family
# --------------------------------------------------------------------------- #

def _product_responder() -> Responder:
    """A responder that drives Avaloka, or raise :class:`ProductPathUnavailable`."""
    try:
        module = importlib.import_module(PRODUCT_RESPONDER_MODULE)
    except ImportError as exc:
        raise ProductPathUnavailable(
            f"{PRODUCT_RESPONDER_MODULE} does not exist ({exc}). "
            + PRODUCT_GAP_REASON) from exc
    factory = getattr(module, PRODUCT_RESPONDER_ATTR, None)
    if factory is None:
        raise ProductPathUnavailable(
            f"{PRODUCT_RESPONDER_MODULE} exists but has no "
            f"'{PRODUCT_RESPONDER_ATTR}'. " + PRODUCT_GAP_REASON)
    return factory()


def _product_gap_result(subject_label: str) -> TaskResult:
    """A standing, self-clearing failure while the product is unmeasured.

    Deliberately a red check rather than a note. Every other number this family
    emits is honest about its own subject only if the reader knows what the
    subject is, and a reader skimming ``conversational_scorers 1.00`` does not.
    So the family reports a gap alongside its score, the same way the format
    coverage check reports a declared format with no task.

    It grades a lookup, not a constant, so it clears itself when the path is
    built -- and the detail names exactly what building it means.
    """
    try:
        _product_responder()
    except ProductPathUnavailable as exc:
        return TaskResult(
            task="conv_avaloka_behaviour_is_unmeasured", family=FAMILY,
            kind="gap", source_scheme=subject_label,
            checks=[CheckResult(
                "avaloka_conversational_behaviour_is_measured", False,
                str(exc), weight=2.0)])
    return TaskResult(
        task="conv_avaloka_behaviour_is_unmeasured", family=FAMILY,
        kind="gap", source_scheme=subject_label,
        checks=[CheckResult(
            "avaloka_conversational_behaviour_is_measured", True,
            f"{PRODUCT_RESPONDER_MODULE}.{PRODUCT_RESPONDER_ATTR}() is available, "
            "so the probes can be driven through Avaloka itself. Delete "
            "PRODUCT_GAP_REASON and wire this responder into "
            "conversational_results as a graded subject.", weight=2.0)])


def conversational_results(*, live: bool = False,
                           model: Optional[str] = None) -> list[TaskResult]:
    """Run the conversational probes and return them as scored benchmark tasks.

    One :class:`TaskResult` per dimension plus one for the negative control.
    Per dimension because that is the granularity the probes are organised at
    and the granularity a reader can act on: "integrity 0.33" names which
    faculty regressed, whereas a single conversational task would average four
    unrelated properties into one number, and eleven separate tasks would bury
    the structure the suite was built around. The control is its own task
    because it grades the evaluators, not the subject.

    With ``live=False`` (the default) the subject is the ``brilliant`` reference
    responder and nothing leaves the process. With ``live=True`` a local Ollama
    model is tried first, then Groq; if neither answers the family SKIPs rather
    than quietly scoring a reference responder and labelling it live.
    """
    try:
        suite = load_probe_suite()
    except ProbeSuiteUnavailable as exc:
        return [_skip("conv_suite", "probe", str(exc)),
                _product_gap_result("—")]

    module = suite.module
    by_dimension: Mapping[Any, Sequence[Any]] = getattr(module, "BY_DIMENSION")
    run_probe = getattr(module, "run_probe")

    subject = REFERENCE_SUBJECT
    if live:
        try:
            responder, resolved = _resolve_live(model)
        except ProviderUnavailable as exc:
            # The live subject cannot be measured, but the negative control is
            # hermetic and grades the evaluators rather than the subject, so it
            # still runs and still gates. Skipping it too would let a scorer
            # regression hide behind an unreachable provider.
            return [_skip("conv_live", "probe", f"live provider unavailable: {exc}"),
                    _control_result(suite),
                    _product_gap_result("—")]
        label = _subject_label(live=True, model=resolved, subject=subject)
    else:
        try:
            responder = _reference_responder(suite, subject)
        except ProbeSuiteUnavailable as exc:
            return [_skip("conv_suite", "probe", str(exc)),
                    _product_gap_result("—")]
        label = _subject_label(live=False, model=None, subject=subject)

    results: list[TaskResult] = []
    for dimension, probes in by_dimension.items():
        recorder = _Recorder(responder)
        try:
            probe_results = [run_probe(p, recorder) for p in probes]
        except ProviderUnavailable as exc:
            return [*results,
                    _skip(f"conv_{dimension.value}", "probe",
                          f"live provider unavailable mid-run: {exc}"),
                    _control_result(suite),
                    _product_gap_result(label)]
        checks = _probe_checks(probe_results, recorder.exchanges, live=live)
        checks.append(_dimension_summary(dimension.value, probe_results))
        results.append(TaskResult(
            task=f"conv_{dimension.value}", family=FAMILY, kind="probe",
            source_scheme=label, checks=checks,
            duration_s=sum(r.latency_s for r in probe_results)))

    results.append(_control_result(suite))
    results.append(_product_gap_result(label))
    return results
