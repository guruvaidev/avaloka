"""Do the prompts still say what downstream code depends on them saying?

Prompts are the least-tested surface in an agentic system and the easiest to
break: a reworded instruction is a one-line diff that passes review, ships, and
silently changes behaviour for every user. There is no type system for a prompt.

These checks pin the parts of the prompt that other code *depends on* — not the
wording, which should be free to improve, but the contracts:

* the Coder is told to emit a numbered pseudocode blueprint, because the
  pipeline authors pseudocode-first
* the Summarizer is told to emit JSON only, because its output is parsed
* prompts that promise a schema actually name the fields the parser reads
* no prompt tells the model to invent data when it lacks it

The last is the one that matters for a regulated pilot. An instruction like
"if unsure, make a reasonable assumption" is how a confident fabrication gets
into a report.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from benchmarks.smoke.result import Severity, SmokeResult

SURFACE = "prompts"

REPO = Path(__file__).resolve().parents[2]

#: (file, needle, why it matters). Substring rather than exact text, so wording
#: can improve without breaking the check.
REQUIRED: Tuple[Tuple[str, str, str], ...] = (
    ("app/agents/coder.py", "pseudocode",
     "the pipeline is pseudocode-first; without the instruction the Coder emits raw code"),
    ("app/agents/summarizer.py", "JSON",
     "the summarizer's output is parsed as JSON by every downstream agent"),
)

PROMPT_FILES: Tuple[str, ...] = (
    "app/agents/coder.py",
    "app/agents/summarizer.py",
    "app/agents/validator.py",
    "app/agents/profiling_agent.py",
    "app/agents/planner.py",
)


def _read(rel: str) -> str:
    path = REPO / rel
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def run(*, include_slow: bool = False) -> List[SmokeResult]:
    return [_check_required_instructions(),
            _check_anti_fabrication_guardrails(),
            _check_no_explicit_fabrication_instruction(),
            _check_json_prompts_name_their_fields()]


def _check_required_instructions() -> SmokeResult:
    missing: List[str] = []
    for rel, needle, why in REQUIRED:
        body = _read(rel)
        if not body:
            missing.append(f"{rel}: file not found")
        elif needle.lower() not in body.lower():
            missing.append(f"{rel}: no mention of {needle!r} — {why}")
    return SmokeResult(
        check="required_prompt_instructions", surface=SURFACE, passed=not missing,
        severity=Severity.CRITICAL,
        detail="prompt contracts intact" if not missing else f"{len(missing)} missing",
        evidence={"missing": missing},
    )


def _check_anti_fabrication_guardrails() -> SmokeResult:
    """Assert the guardrails EXIST, rather than hunting for their absence.

    This started as a keyword hunt for instructions like "invent" and was wrong
    twice. Avaloka's prompts live inside Python files next to the code that
    *implements* anti-fabrication: `is_fabricated`, `_detect_fabricated_reference`,
    log lines reading "Blocked fabricated-reference request", and prompt text
    explaining that fabrication is a logical error. A substring match flagged all
    of it, then a negation-aware match still flagged the identifiers.

    A precise positive check beats an imprecise negative one. The question worth
    answering is not "does any file contain the word invent" but "does the
    pipeline still refuse to fabricate" -- and that has concrete, greppable
    machinery behind it.
    """
    signals = {
        "coder forbids inventing columns":
            ("app/agents/coder.py", ("do not invent", "does not exist in the schema")),
        "validator treats fabrication as an error":
            ("app/agents/validator.py", ("is_fabricated", "FABRICATION IS A LOGICAL ERROR")),
        "planner blocks fabricated references":
            ("app/agents/planner.py", ("_detect_fabricated_reference", "won't invent")),
    }
    missing: List[str] = []
    present: List[str] = []
    for label, (rel, needles) in signals.items():
        body = _read(rel).lower()
        if body and any(n.lower() in body for n in needles):
            present.append(label)
        else:
            missing.append(f"{label} ({rel})")
    return SmokeResult(
        check="anti_fabrication_guardrails", surface=SURFACE, passed=not missing,
        severity=Severity.CRITICAL,
        detail=(f"{len(present)} anti-fabrication guardrail(s) intact" if not missing
                else f"{len(missing)} guardrail(s) missing — the pipeline may fabricate"),
        evidence={"present": present, "missing": missing},
    )


def _check_no_explicit_fabrication_instruction() -> SmokeResult:
    """Only unambiguous imperatives. Reported as a WARNING, not a gate.

    Kept deliberately narrow: these phrasings have no legitimate use in a
    prompt, whereas the word "invent" plainly does.
    """
    blatant = ("if unsure, assume", "if you don't know, assume", "make up a value",
               "make up the value", "fabricate a value", "invent a value")
    hits: List[str] = []
    for rel in PROMPT_FILES:
        body = _read(rel).lower()
        for phrase in blatant:
            if phrase in body:
                hits.append(f"{rel}: {phrase!r}")
    return SmokeResult(
        check="no_explicit_fabrication_instruction", surface=SURFACE, passed=not hits,
        severity=Severity.WARNING,
        detail=("no prompt explicitly instructs the model to invent a value"
                if not hits else "a prompt explicitly instructs fabrication"),
        evidence={"hits": hits},
    )


def _check_json_prompts_name_their_fields() -> SmokeResult:
    """A prompt promising JSON should name the fields the parser reads.

    Warning rather than critical: a schema can legitimately be injected at
    runtime (the Summarizer builds its schema from the Pydantic model), so
    absence is a smell rather than proof of breakage.
    """
    soft: List[str] = []
    for rel in PROMPT_FILES:
        body = _read(rel)
        if not body or "json" not in body.lower():
            continue
        names_fields = bool(re.search(r'"[a-z_]{3,}"\s*:', body)) or "schema" in body.lower()
        if not names_fields:
            soft.append(rel)
    return SmokeResult(
        check="json_prompts_name_fields", surface=SURFACE, passed=not soft,
        severity=Severity.WARNING,
        detail=("JSON-emitting prompts reference a schema or field names"
                if not soft else "a JSON prompt names no fields and no schema"),
        evidence={"files": soft},
    )
