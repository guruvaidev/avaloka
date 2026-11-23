"""Classify an Avaloka insight as sensible / vacuous / nonsense, with a reason.

An insight is *nonsense* when it states something the data contradicts (a
"dominant segment" that is 0% of rows, a correlation that does not exist, an
identifier column read as a segment), *vacuous* when it could have been said
about any dataset (no column named, no number, a "does not map to a supported
calculation" shrug), and *sensible* when it names real columns and the numbers
it quotes recompute from the data within tolerance.

The deterministic CLI narrative is checked by recomputation. LLM prose is only
checked structurally (names a real column, quotes a number, is not an apology),
so a "sensible" there is "sensible (unverified)" and the report says so.
"""
from __future__ import annotations

import difflib
import re
from typing import Any

import pandas as pd

SEGMENT = re.compile(r"(?P<col>\S.*?) leans hard on [“\"](?P<val>.+?)[”\"] \((?P<pct>\d+)% of rows\)")
PULL = re.compile(r"pull between (?P<a>\S+) and (?P<b>\S+) .*?\|corr\| (?P<r>[0-9.]+)")
SKEW = re.compile(r"(?P<col>\S+) is skewed \(skew (?P<skew>-?[0-9.]+)\)")
APOLOGY = re.compile(r"i can help|please (upload|provide)|as an ai|i (do not|don't) have access|unable to (access|see) (the|your) data", re.I)
AGENT_FAILED = re.compile(r"did not receive any tool calls|please try again|something went wrong|failed to (generate|produce)|could not complete|result is not found|i couldn't complete", re.I)
HONEST_REFUSAL = re.compile(r"i could write code for this, but|haven't used it|didn't match what the analysis was supposed to", re.I)
BROKEN = re.compile(r"traceback|\bnan\b|\bnone\b|exception|<class|internal server error", re.I)
UNANSWERED = "does not map to a supported goal-specific calculation"

ID_UNIQUENESS = 0.5   # a column with >50% distinct values is an identifier, not a segment
MIN_SEGMENT_PCT = 5   # a "dominant" segment under 5% of rows is not dominant


def _strip_markup(text: str) -> str:
    return re.sub(r"\[/?[a-z ]+\]", "", text)


def _col(frame: pd.DataFrame, name: str) -> pd.Series | None:
    name = name.strip()
    if name in frame.columns:
        return frame[name]
    for c in frame.columns:
        if str(c).strip().lower() == name.lower():
            return frame[c]
    return None


def classify_cli(lines: list[str], frame: pd.DataFrame) -> tuple[str, str]:
    """Classify the CLI's deterministic observations against the data they describe."""
    text = _strip_markup("\n".join(lines))
    if not text.strip():
        return "vacuous", "no observation was produced"
    verdicts: list[tuple[str, str]] = []
    n = max(1, len(frame))
    for m in SEGMENT.finditer(text):
        col, val, pct = m.group("col").strip(), m.group("val"), int(m.group("pct"))
        s = _col(frame, col)
        if s is None:
            verdicts.append(("nonsense", f"segment claim names column {col!r} that is not in the data"))
            continue
        uniq = s.nunique(dropna=True) / n
        if uniq > ID_UNIQUENESS:
            verdicts.append(("nonsense", f"{col!r} is {uniq:.0%} unique (an identifier), not a segment"))
            continue
        if pct < MIN_SEGMENT_PCT:
            verdicts.append(("nonsense", f"'dominant' segment {val!r} is only {pct}% of rows"))
            continue
        actual = (s.astype(str) == str(val)).mean() * 100
        if abs(actual - pct) > 2.5:
            verdicts.append(("nonsense", f"{col!r}={val!r} is {actual:.0f}% of rows, not {pct}%"))
        else:
            verdicts.append(("sensible", f"{col!r}={val!r} share {pct}% recomputes to {actual:.0f}%"))
    for m in PULL.finditer(text):
        a, b, r = m.group("a"), m.group("b"), float(m.group("r"))
        sa, sb = _col(frame, a), _col(frame, b)
        if sa is None or sb is None:
            verdicts.append(("nonsense", f"correlation names a column not in the data ({a}, {b})"))
            continue
        # an identifier is a mostly-unique text or integer column; a continuous float is a measurement
        ids = [c for c, s in ((a, sa), (b, sb))
               if s.nunique(dropna=True) / n > ID_UNIQUENESS and not pd.api.types.is_float_dtype(s)]
        if ids:
            verdicts.append(("nonsense", f"correlation between identifier column(s) {ids} is not a relationship"))
            continue
        try:
            actual = abs(pd.to_numeric(sa, errors="coerce").corr(pd.to_numeric(sb, errors="coerce")))
        except Exception:
            actual = float("nan")
        if actual != actual or abs(actual - r) > 0.1:
            verdicts.append(("nonsense", f"|corr({a},{b})| recomputes to {actual:.2f}, claimed {r}"))
        else:
            verdicts.append(("sensible", f"|corr({a},{b})|={r} recomputes to {actual:.2f}"))
    for m in SKEW.finditer(text):
        s = _col(frame, m.group("col"))
        if s is None:
            verdicts.append(("nonsense", f"skew claim names unknown column {m.group('col')!r}"))
        else:
            verdicts.append(("sensible", f"skew claim on real numeric column {m.group('col')!r}"))
    if "identifiers, not signal" in text:
        verdicts.append(("sensible", "identifier columns named and excluded"))
    if "Clean and quiet" in text and not verdicts:
        return "vacuous", "only the default 'clean and quiet' line; nothing measured was reported"
    if UNANSWERED in text and not verdicts:
        return "vacuous", "goal not answered and no observation made"
    if not verdicts:
        return "vacuous", "no checkable claim in the narrative"
    if any(v == "nonsense" for v, _ in verdicts):
        return "nonsense", "; ".join(r for v, r in verdicts if v == "nonsense")
    return "sensible", "; ".join(r for _, r in verdicts)


MAX_VALUE_CARDINALITY = 2000
EXACT_VALUE_CARDINALITY = 20000
PLAN_NOT_FINDINGS = re.compile(r"^\s*(here is what i would|i would (start|look|begin)|first,? (i|we) (would|should)|steps?:)", re.I)
WORD = re.compile(r"[^\W\d_][\w'’\-]*")


def altered_values(text: str, frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Proper-noun-like tokens in the answer that are a near miss of a data value
    but not equal to any value: the model changed a name it was quoting."""
    values: set[str] = set()
    for col in frame.columns:
        s = frame[col]
        if s.dtype == object or str(s.dtype) in ("string", "str"):
            u = s.dropna().astype(str).unique()
            if len(u) <= MAX_VALUE_CARDINALITY:
                values.update(v.strip() for v in u if v.strip())
    if not values:
        return []
    lowered = {v.lower() for v in values}
    out: list[tuple[str, str]] = []
    words_all = [m.group(0) for m in WORD.finditer(text)]
    spans = {" ".join(words_all[i:i + k]).strip(" .,;:")
             for i in range(len(words_all)) for k in (1, 2, 3) if i + k <= len(words_all)}
    for tok in spans:
        tok = re.sub(r"[’']s$", "", tok)  # "EIRE's" quotes EIRE
        non_ascii = any(ord(ch) > 127 for ch in tok)
        if tok.lower() in lowered or not (tok[0].isupper() or non_ascii):
            continue
        if len(tok) < 4 and not non_ascii:
            continue
        words = tok.split()
        # "José Álvarez is" / "and Zoë Müller": a span containing an exact value is a quote plus prose
        if any(" ".join(words[i:j]).lower() in lowered
               for i in range(len(words)) for j in range(i + 1, len(words) + 1) if j - i < len(words)):
            continue
        cutoff = 0.5 if (non_ascii and len(tok) <= 3) else 0.8
        close = difflib.get_close_matches(tok, values, n=1, cutoff=cutoff)
        if close and close[0].lower() != tok.lower():
            out.append((tok, close[0]))
    return out


def classify_llm(text: str, columns: list[str], frame: pd.DataFrame | None = None) -> tuple[str, str]:
    """Structural check of model prose: names a real column, quotes a number, is not a
    shrug — and, when the data is at hand, did not alter the values it quotes."""
    if not text or not text.strip():
        return "vacuous", "empty answer"
    if BROKEN.search(text) and len(text) < 400:
        return "nonsense", "answer is an error or a null leaked as text"
    if AGENT_FAILED.search(text):
        return "vacuous", f"agent failure message returned as the answer: {text.strip()[:120]!r}"
    if HONEST_REFUSAL.search(text):
        return "vacuous", f"honest refusal, no finding delivered: {text.strip()[:120]!r}"
    if PLAN_NOT_FINDINGS.search(text):
        return "vacuous", "a plan of what to look at, not findings"
    if APOLOGY.search(text):
        return "vacuous", "answer is an apology/request, not an analysis"
    named = [c for c in columns if c and len(str(c)) > 2
             and re.search(r"(?<![A-Za-z0-9_])" + re.escape(str(c)) + r"(?![A-Za-z0-9_])", text, re.I)]
    # numbers that are not list markers ("1. Perform...")
    has_number = bool(re.search(r"(?<![\n^])(?<!\d)\d[\d,.]*(?!\.\s)", re.sub(r"(?m)^\s*\d+\.\s", "", text)))
    quoted = quoted_values(text, frame) if frame is not None else []
    if not named and not quoted and not has_number:
        return "vacuous", "names no column from the schema and quotes no number"
    if not named and not quoted:
        return "vacuous", "quotes numbers but names no column or value from the data"
    if frame is not None:
        altered = altered_values(text, frame)
        if altered:
            return "nonsense", "quotes values that are not in the data: " + "; ".join(f"{a!r} (data has {b!r})" for a, b in altered[:4])
    what = (f"names {len(named)} real column(s): {', '.join(map(str, named[:4]))}" if named else "") + \
           (f"; quotes {len(quoted)} data value(s): {', '.join(quoted[:3])}" if quoted else "")
    return "sensible (unverified)", what.strip("; ")


def quoted_values(text: str, frame: pd.DataFrame) -> list[str]:
    """Exact data values (from string columns up to EXACT_VALUE_CARDINALITY distinct) the answer quotes."""
    low = text.lower(); out: list[str] = []
    for col in frame.columns:
        s = frame[col]
        if s.dtype == object or str(s.dtype) in ("string", "str"):
            u = s.dropna().astype(str).unique()
            if len(u) <= EXACT_VALUE_CARDINALITY:
                out.extend(v for v in u if len(v.strip()) >= 4 and v.strip().lower() in low)
    return sorted(set(out))
