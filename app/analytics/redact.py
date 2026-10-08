"""Redaction of the one free-text field product analytics stores: the question.

Runs in the collector, before anything is written. The browser sends the
question to this same API to have it answered, so capturing it here exposes it
to no component that did not already hold it; redacting here means the stored
copy is the redacted one and there is no raw copy to leak.

What this removes, each by shape:

* credentials -- PEM blocks, labelled secrets (``password is x``, ``token: x``),
  provider-prefixed keys, AWS access keys, JWTs, long high-entropy runs, hex
  digests, UUIDs;
* URLs and connection strings, whole (they carry hosts, paths and passwords);
* email addresses;
* file names and paths -- the standing rule is that file names never leave;
* quoted literals -- ``where status = 'churned'`` quotes a cell value;
* dates in ``d/m/y`` form, IBANs, and any run holding six or more digits:
  phone numbers, card and account numbers, national ids, postcodes.

What this does NOT remove, stated so nobody mistakes it for more than it is:

* **Column and table names.** Analysts type them, and they are collected by
  decision. ``patient_hiv_status`` typed into a question is stored.
* **Names of people and organisations, and street addresses**, typed as prose.
  Nothing distinguishes "Ravi Kumar" from any other two words.
* **Unquoted values.** ``customers in Bengaluru`` keeps the city.
* Numbers of five digits or fewer, by design: years, limits and thresholds are
  what makes a question analysable.

It errs toward masking. An ISO date or a seven-figure revenue threshold is
masked too; losing those is the price of catching phone numbers without
understanding the text.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from app.analytics.schema import MAX_PROMPT_CHARS

#: Longest input the patterns are run over. A bound on work, not on storage:
#: anything past it is discarded before matching, so no pattern meets a
#: megabyte of pasted text.
MAX_INPUT_CHARS = 4000

_SECRET_LABELS = (r"pass(?:word|wd|phrase)?|pwd|secret|token|api[ _-]?key|access[ _-]?key|"
                  r"private[ _-]?key|client[ _-]?secret|credentials?|auth(?:orization)?")
_FILE_EXT = (r"csv|tsv|xlsx?|xlsm|xlsb|parquet|json|jsonl|ndjson|txt|pdf|docx?|pptx?|sql|db|"
             r"sqlite|zip|gz|tar|pkl|pickle|feather|orc|avro|xml|ya?ml|log|ipynb|sav|dta")


def _digit_run(match: "re.Match[str]") -> str:
    text = match.group(0)
    return "[number]" if sum(c.isdigit() for c in text) >= 6 else text


#: (category, pattern, replacement). Order matters: wide containers first, so a
#: URL is removed whole before the e-mail or digit rules see its inside.
_RULES: List[Tuple[str, "re.Pattern[str]", object]] = [
    ("credential", re.compile(r"-----BEGIN [A-Z ]{0,40}-----.*?(?:-----END [A-Z ]{0,40}-----|$)", re.S),
     "[secret]"),
    ("url", re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.\-]{1,20}://\S+"), "[url]"),
    ("url", re.compile(r"\bwww\.[^\s]+", re.I), "[url]"),
    ("credential", re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=\-]{8,}", re.I), "[secret]"),
    ("credential", re.compile(rf"\b({_SECRET_LABELS})(\s*[:=]\s*|\s+is\s+)\S+", re.I), r"\1 [secret]"),
    ("credential", re.compile(r"\b(?:sk|pk|rk|gsk|ghp|gho|ghs|ghu|glpat|xox[baprs]|AIza|ya29)[-_.]?[A-Za-z0-9_\-]{12,}"),
     "[secret]"),
    ("credential", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{12,}"), "[secret]"),
    ("credential", re.compile(r"\bey[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]*"), "[secret]"),
    ("email", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+"), "[email]"),
    # Quoted literals. The single-quote form requires a non-word character
    # before the opening quote so "Ravi's and Meena's" is not read as a quote.
    ("literal", re.compile(r"\"[^\"\n]{1,200}\"|[“][^”\n]{1,200}[”]"), "[value]"),
    ("literal", re.compile(r"(?<![A-Za-z0-9])'[^'\n]{1,200}'(?![A-Za-z0-9])|‘[^’\n]{1,200}’"), "[value]"),
    ("file", re.compile(r"\b[A-Za-z]:\\[^\s]+"), "[file]"),
    ("file", re.compile(r"\\\\[^\s\\]+\\[^\s]+"), "[file]"),
    ("file", re.compile(r"(?<![\w/])(?:~|\.{1,2})?/(?:[\w.\-]+/)+[\w.\-]*"), "[file]"),
    ("file", re.compile(rf"[\w\-.()]+\.(?:{_FILE_EXT})\b", re.I), "[file]"),
    ("id", re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
     "[id]"),
    ("number", re.compile(r"\b[A-Z]{2}\d{2}[ ]?(?:[A-Z0-9]{4}[ ]?){2,7}[A-Z0-9]{1,4}\b"), "[number]"),
    ("date", re.compile(r"\b\d{1,4}[/.]\d{1,2}[/.]\d{2,4}\b"), "[date]"),
    ("number", re.compile(r"(?<![\w.])\+?\(?\d[\d\s\-.()]{4,}\d(?!\w)"), _digit_run),
    ("credential", re.compile(r"\b[0-9a-fA-F]{32,}\b"), "[secret]"),
    ("credential", re.compile(r"(?<![A-Za-z0-9_\-])(?=[A-Za-z0-9_\-]{24,}(?![A-Za-z0-9_\-]))"
                              r"(?=[A-Za-z0-9_\-]*[a-z])(?=[A-Za-z0-9_\-]*[A-Z])(?=[A-Za-z0-9_\-]*\d)"
                              r"[A-Za-z0-9_\-]{24,}"), "[secret]"),
]

_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿­]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def scrub(text: object, limit: int = MAX_PROMPT_CHARS) -> Tuple[Optional[str], Dict[str, int]]:
    """Redact ``text``. Returns ``(cleaned, counts_by_category)``.

    ``cleaned`` is ``None`` when the input is not usable text or when any rule
    fails: a prompt that could not be redacted is not stored at all. There is
    no path through this function that returns the input unprocessed.
    """
    if not isinstance(text, str):
        return None, {}
    try:
        # NFKC folds full-width digits and letters onto the forms the patterns
        # match; invisible characters are removed so they cannot split a token.
        cleaned = unicodedata.normalize("NFKC", text[:MAX_INPUT_CHARS])
        cleaned = _INVISIBLE.sub("", cleaned)
        cleaned = _CONTROL.sub(" ", cleaned)
        counts: Dict[str, int] = {}
        for category, pattern, replacement in _RULES:
            before = cleaned
            cleaned, n = pattern.subn(replacement, cleaned)  # type: ignore[arg-type]
            if n and cleaned != before:
                counts[category] = counts.get(category, 0) + n
    except Exception:
        return None, {}

    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None, counts
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip()
        # A cut can land inside a mask token; never leave half of one behind.
        cleaned = re.sub(r"\[[a-z]*$", "", cleaned).rstrip() + "…"
    return cleaned, counts
