"""How a figure in a reply is read: as a bound, and as one of a paired list.

Two ways a true sentence was withheld on rc14 (groq, "train a random forest
to predict stroke and show feature importance"):

- "avg_glucose_level, age, bmi -- together they account for **over 70 %** of
  the model's importance". 70% is a bound, not a value; the three importances
  sum to 72.4%. :func:`bound_supported` checks a bound against what the claim
  names: the values of the result rows it names, and their sum and share.
- "hypertension and heart_disease have similar importance (~0.028 and 0.025)"
  was withheld as "0.028 is hypertension's importance, not heart_disease's":
  each figure was given to the entity named last before it, and the pairing
  was respective. :func:`respective` pairs "X and Y (a and b)" and "X, Y and
  Z (a, b, c)" in order, and leaves a list it cannot pair unattributed rather
  than give a reason that contradicts the right pairing.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.agents import claim_verifier as cv

#: Words that make the figure after them a bound: "over 70%", "at most 0.05".
_BOUND_WORDS = (
    ("no more than", "upper"), ("not more than", "upper"), ("no less than", "lower"), ("not less than", "lower"),
    ("more than", "lower"), ("greater than", "lower"), ("at least", "lower"), ("upwards of", "lower"),
    ("in excess of", "lower"), ("exceeding", "lower"), ("over", "lower"), ("above", "lower"),
    ("less than", "upper"), ("fewer than", "upper"), ("at most", "upper"), ("up to", "upper"),
    ("under", "upper"), ("below", "upper"),
)
_BOUND_SIDE = dict(_BOUND_WORDS)
_BOUND = re.compile(r"(?<![\w-])(?P<word>" + "|".join(re.escape(w) for w, _ in _BOUND_WORDS) + r")\s*[$€£₹]?\s*$",
                    re.IGNORECASE)


def bound_of(text: str, start: int) -> Optional[str]:
    """The bound word right before the figure at *start*, lower case, or None."""
    m = _BOUND.search(text[max(0, start - 20):start])
    return m.group("word").lower() if m else None


def _place(value: float) -> float:
    """The place of *value*'s last significant digit: 70 -> 10, 75 -> 1, 0.05 -> 0.01."""
    if value == 0:
        return 1.0
    for exponent in range(int(math.floor(math.log10(abs(value)))), -12, -1):
        unit = 10.0 ** exponent
        if abs(value / unit - round(value / unit)) < 1e-6:
            return unit
    return 10.0 ** -12


def result_rows(flat: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The result's rows, from the flattened evidence."""
    rows: Dict[int, Dict[str, Any]] = {}
    for key, value in flat.items():
        m = cv._CELL.match(key)
        if m:
            rows.setdefault(int(m.group(1)), {})[m.group(2)] = value
    return [rows[i] for i in sorted(rows)]


#: A code: letters and digits, no space and no decimal point ("84077", "85123A").
_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_/\-]*$")
#: The last word of a column name that says it holds identifiers or codes.
_KEY_WORDS = frozenset({"id", "code", "sku", "key", "no", "num", "number", "zip", "postcode", "ref", "isbn"})


def key_columns(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Columns that key the result: one row per value, every value a code,
    and either some value is not a number (StockCode: "84077" beside
    "85123A") or the name says it is an identifier or code (customer_id).

    rweb16's messy reply said "Lowest revenue: 84077 - $1,160.21". 84077 is a
    StockCode, but as a number it was never a label, so no check saw an
    entity in the claim. A count, a total or a year is a number, not a key.
    """
    if len(rows) < 2:
        return []
    out = []
    for col in rows[0]:
        values = [row.get(col) for row in rows]
        if any(v is None or isinstance(v, bool) or isinstance(v, float) for v in values):
            continue
        text = [str(v).strip() for v in values]
        if not all(_CODE.match(t) for t in text) or len(set(text)) != len(text):
            continue
        words = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(col)).lower())
        if any(cv._as_number(t) is None for t in text) or (words and words[-1] in _KEY_WORDS):
            out.append(col)
    return out


def row_labels(rows: Sequence[Mapping[str, Any]], keys: Sequence[str] = ()) -> Dict[str, List[tuple]]:
    """label -> [(row index, column)] for each text cell of the result, and
    each code of a key column (:func:`key_columns`), digits only or not."""
    out: Dict[str, List[tuple]] = {}
    for i, row in enumerate(rows):
        for col, value in row.items():
            if col in keys and value is not None and str(value).strip():
                out.setdefault(str(value).strip(), []).append((i, col))
            elif isinstance(value, str) and value.strip() and cv._as_number(value) is None:
                out.setdefault(value.strip(), []).append((i, col))
    return out


#: Rows of a result column whose pairwise gaps and ratios are figures a reply may bound.
PAIRWISE_MAX_ROWS = 50


def _gaps(values: Sequence[float], ratios: bool = True) -> List[float]:
    """Every difference (and ratio) of two of *values*: "the gap between
    enterprise and SMB is over $360" is 545.08 - 182.27 (rc15 related). A
    ratio is never a share: "over 80% of the importance" is not 0.218/0.263."""
    out: List[float] = []
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            out.append(abs(a - b))
            if ratios and a and b:
                out += [a / b, b / a]
    return out


def _named_values(text: str, flat: Mapping[str, Any], ratios: bool = True) -> Optional[List[float]]:
    """The values of the result rows *text* names, and per numeric column
    their sum, their share of the column's total, and their gaps and ratios;
    None when it names none."""
    rows = result_rows(flat)
    labels = row_labels(rows)
    spans = cv._label_spans(text, labels)
    named = sorted({i for _s, _e, label in spans for i, _c in labels.get(label, ())})
    if not named:
        return None
    out: List[float] = []
    columns = {c for i in named for c in rows[i]}
    for col in columns:
        mine = [cv._as_number(rows[i].get(col)) for i in named]
        mine = [v for v in mine if v is not None]
        out += mine
        if len(mine) >= 2:
            total = sum(mine)
            out.append(total)
            out += _gaps(mine[:PAIRWISE_MAX_ROWS], ratios)
            every = [cv._as_number(r.get(col)) for r in rows]
            if all(v is not None and v >= 0 for v in every) and sum(every) > 0:
                out.append(total / sum(every))
    return out


def column_gaps(flat: Mapping[str, Any], ratios: bool = True) -> List[float]:
    """The gaps and ratios of two values of the same result column."""
    rows = result_rows(flat)
    if not rows or len(rows) > PAIRWISE_MAX_ROWS:
        return []
    out: List[float] = []
    for col in {c for r in rows for c in r}:
        values = [cv._as_number(r.get(col)) for r in rows]
        if all(v is not None for v in values):
            out += _gaps(values, ratios)
    return out


def bound_supported(text: str, flat: Mapping[str, Any], value: float, raw: str, bound: str) -> bool:
    """A bound holds when a value it is about lies on its side of it, and
    near enough to be what it rounds: "over 70%" of 72.4%, not of 99%.

    What it is about is the result rows the claim names (their values, and
    their sum and share), or any figure of the evidence when it names none."""
    ratios = not cv._is_percent(raw)
    candidates = _named_values(text, flat, ratios)
    if candidates is None:
        candidates = cv._evidence_numbers(cv._flatten(dict(flat))) + column_gaps(flat, ratios)
    if cv._is_percent(raw):
        candidates = [c * 100 for c in candidates if 0 <= c <= 1] + [c for c in candidates if 0 <= c <= 100]
    tol = cv.figure_tolerance(value, raw)
    band = max(0.25 * abs(value), _place(value))
    if _BOUND_SIDE[bound] == "lower":
        return any(value - tol <= c <= value + band for c in candidates)
    return any(value - band <= c <= value + tol for c in candidates)


# --------------------------------------------------------------------------- #
# Paired lists
# --------------------------------------------------------------------------- #
#: Between two entities of a list: ", ", " and ", ", and ", " or ", " & ", "/".
#: Markdown emphasis around the entities ("**a** and **b**") is not a word.
_LIST_SEP = re.compile(r"^[\s*_`]*(?:,[\s*_`]*(?:and\s+|or\s+|&\s+)?|and|or|&|/)[\s*_`]*$", re.IGNORECASE)
#: Between two entities compared: "Enterprise leads by a wide margin over SMB".
_COMPARED = re.compile(r"^[^.;:!?()]{0,60}\b(?:over|than|vs\.?|versus|against|compared (?:to|with))[\s*_`]*$",
                       re.IGNORECASE)
#: Between the two figures of a comparison: "$545.08 vs. $182.27".
_VERSUS = re.compile(r"^[\s*_`]*(?:vs\.?|versus|against|compared (?:to|with)|to)[\s*_`]*(?:[≈~∼]\s*)?[$€£₹]?$",
                     re.IGNORECASE)
#: Between two figures of a list, each perhaps said to be approximate: " and ~".
_FIGURE_SEP = re.compile(r"^[\s*_`]*(?:,\s*(?:and\s+)?|and|&|/|;)[\s*_`]*(?:[≈~∼]\s*)?[$€£₹]?$", re.IGNORECASE)


def respective(text: str, spans: Sequence[tuple], figures: Sequence[Mapping[str, Any]]) -> Dict[int, Optional[tuple]]:
    """``{figure index: span}`` for each figure that follows a list of
    entities: "X and Y (a and b)" gives a to X and b to Y. A list followed
    by as many figures as it has entities is paired in order; a list of
    figures of another length cannot be paired, and its figures map to None
    (no owner). One figure after a list ("..., the last barely 0.003", "each
    around 0.13") is left to the caller, as are figures not after a list.
    *spans* are ``(start, end, ...)`` in text order."""
    out: Dict[int, Optional[tuple]] = {}
    kept: List[tuple] = []
    for span in sorted(spans, key=lambda s: (s[0], -s[1])):     # one span per place in the text
        if not kept or span[0] >= kept[-1][1]:
            kept.append(span)
    spans = kept
    i = 0
    while i < len(spans):
        j = i
        while j + 1 < len(spans) and _LIST_SEP.match(text[spans[j][1]:spans[j + 1][0]]):
            j += 1
        run = list(spans[i:j + 1])
        i = j + 1
        if len(run) < 2:
            # "A ... over B (a vs. b)": a comparison of two, its figures in the same order.
            if j + 1 < len(spans) and _COMPARED.match(text[spans[j][1]:spans[j + 1][0]]):
                nxt = spans[j + 2][0] if j + 2 < len(spans) else len(text)
                after = [k for k, f in enumerate(figures) if spans[j + 1][1] <= f["start"] < nxt]
                if len(after) >= 2 and _VERSUS.match(text[figures[after[0]]["end"]:figures[after[1]]["start"]]):
                    out.update({after[0]: spans[j], after[1]: spans[j + 1]})
            continue
        nxt = spans[j + 1][0] if j + 1 < len(spans) else len(text)
        after = [k for k, f in enumerate(figures) if run[-1][1] <= f["start"] < nxt]
        if not after:
            continue
        group = [after[0]]
        for k in after[1:]:
            if not _FIGURE_SEP.match(text[figures[group[-1]]["end"]:figures[k]["start"]]):
                break
            group.append(k)
        if len(group) == len(run):
            out.update(zip(group, run))
        elif len(group) >= 2:
            out.update((k, None) for k in group)
    return out
