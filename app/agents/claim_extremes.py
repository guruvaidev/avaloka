"""Superlatives and trends a reply states, checked on all of what they are about.

Every figure of rc14's messy_xlsx reply was in the result, and three of its
claims were false. Asked "what sells most?", the code kept the top 10 items by
QUANTITY, and the reply called the best of those by revenue the best overall:
"The item with the highest sales is WHITE HANGING HEART T-LIGHT HOLDER with
$9,570.72". Over all rows REGENCY CAKESTAND 3 TIER sold 24,615.97 and DOTCOM
POSTAGE 20,910.16. The result was a subset; the claim was about the whole.

On nullxlsx, a table of glucose bins was narrated as a trend it does not show
("Higher glucose levels increase the likelihood of readmission": 31%, 15%,
47%, 29% across the bins) and a bin as the largest share it is not ("(130.0,
174.0] ... the largest recovery share (≈71%)": (88.0, 107.5] has 84.6%).

Three checks, deterministic and recomputed, never guessed:

- :meth:`Extremes.superlative`: "highest / largest / top / most / best /
  lowest / smallest / least / fewest", or a rank ("second highest"), said of
  an entity and a measure. The entity must hold that rank among the rows of
  the result that are comparable to its own (same subgroup); and when the
  result keeps only some of the groups the data has, among ALL the groups of
  the data. The measure is recomputed from the frames the run read, by the
  expression that reproduces the result's own values on the rows it shows
  (a count, a sum, mean, min or max of a column, or a sum of a product of
  two). A measure that cannot be reproduced checks nothing beyond the result.
- :meth:`Extremes.trend`: "increases / decreases / rises with / the higher
  X the more Y", said of a result grouped by ordered bins of X, must be
  monotonic across them.
- :meth:`Extremes.range_over_set`: "all smoking_status and work_type
  categories contribute modestly (≈0.02–0.017)" puts every member of those
  groups in the range; each is checked.

rc18d said "The white hanging heart t-light holder sold the most units. / - It
moved 3,648 pieces, more than double the next highest item." over a revenue
top 10, and both claims passed: "units" named no column, the figure was in the
next sentence, and nothing read a ratio. Over all rows WORLD WAR 2 GLIDERS sold
5,147, and 3,648 is 1.52 times the next item in the result. So (D13.1):

- a closed lexicon of measure words (:data:`MEASURE_WORDS`) names a column
  when the reply's words do not: "units" is TotalQuantity, if it is the one
  quantity-like column; two candidates are no match, never a guess;
- a sentence that starts "It / This / That / They" and names nothing takes
  the entity, and the measure, of the claim just before it;
- :meth:`Extremes.ratio`: "more than double / twice / N times the next" is
  checked against the ranking (``ratio_contradicted``);
- a cut the code made (:func:`result_truncation`, ``sort_values(A).head(N)``)
  fails a superlative closed: unless the recompute over all rows confirms
  it, or the cut is by its own measure, it is unconfirmed
  (:attr:`Extremes.unconfirmed`): never counted as checked, never verified.

:attr:`Extremes.resolved` holds the claims a check here judged true, so the
verifier counts only what it checked (D13.2).
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.agents import claim_verifier as cv
from app.agents.claim_figures import key_columns, result_rows, row_labels

logger = logging.getLogger(__name__)

_RANKS = {"second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}
_LOW_TERMS = frozenset({"lowest", "smallest", "least", "fewest", "bottom"})
_COUNT_AFTER = r"(?:\d|one|two|three|four|five|six|seven|eight|nine|ten)\b"
#: A superlative, perhaps ranked: "highest", "second highest", "top-selling",
#: "the most", "most sold". Not "at least", not a list ("top 10", "top
#: three"), and not "next largest", which has no rank.
SUPERLATIVE = re.compile(
    r"(?<![\w-])(?<!next )(?<!next-)(?:(?P<rank>second|2nd|third|3rd|fourth|4th|fifth|5th)[\s\-‐‑]+(?:most\s+)?)?"
    r"(?P<term>highest|largest|biggest|greatest|best|top|lowest|smallest|fewest|(?<!at )least|(?<=the )most"
    r"|(?<=the )bottom"
    r"|most(?=[\s-]+(?:sold|popular|purchased|ordered|frequent|common|revenue|sales|units|orders|profitable)\b))"
    rf"(?![\w‐‑-]*\s+{_COUNT_AFTER})(?:[\s\-‐‑]+(?:selling|seller|grossing|earning))?\b",
    re.IGNORECASE)
#: Says the superlative is of a selection: "the highest among these", "of the top 10".
_SCOPED = re.compile(
    r"\b(?:among|of these|of those|of the (?:top|selected|listed|shown|ones?)|in (?:this|the) (?:list|table|"
    r"selection|sample|subset)|in the top|within (?:the|this|these) (?:top|list|selection|sample)|listed|shown|"
    r"selected)\b", re.IGNORECASE)
#: Not the extreme itself (1.6's rule): "one of the lowest", "not the lowest",
#: "nearly the highest", "never the top".
_QUALIFIED = re.compile(r"\b(?:one of|among|not|n't|never|nearly|almost|close to|near)\s+(?:the\s+)?$",
                        re.IGNORECASE)
#: Where what a superlative says about a measure stops.
_CLAUSE_END = re.compile(r"[,;]|\s[-–—]\s|\b(?:while|whereas|but|although|though)\b", re.IGNORECASE)
#: A figure right after a comma: "the most units, 3,648" is one clause.
_FIGURE_NEXT = re.compile(r"\s*(?:[$£€]\s?)?\d[\d,]*(?:\.\d+)?\s?%?")
#: Measure words a reply says for a column it does not name: "sold the most
#: units" is TotalQuantity. ``(words, stems)``: a word maps to the stems a
#: column name must hold (a whole token of it). Closed, in code, no model.
MEASURE_WORDS = (
    (frozenset({"units", "pieces", "items", "volume", "quantity", "quantities", "qty", "sold"}),
     frozenset({"quantity", "qty", "units", "volume", "pieces", "count"})),
    (frozenset({"revenue", "revenues", "sales", "turnover", "grossing", "grossed", "earning", "earnings",
                "earned", "takings", "income"}),
     frozenset({"revenue", "sales", "amount", "turnover", "income", "takings"})),
    (frozenset({"orders", "invoices", "transactions", "purchases"}),
     frozenset({"orders", "order", "invoices", "invoice", "transactions", "transaction", "purchases"})),
)
#: Words a measure phrase may start with: "the highest number of units".
_MEASURE_FILLER = frozenset({"number", "of", "total", "overall", "the", "in", "amount", "terms", "by"})
#: Grammar around a bare superlative: "has the least subscribers on average".
_BARE_FILLER = frozenset({"on", "at", "a", "an", "is", "are", "was", "were", "its", "their"})
#: A stem too generic to name a measure on its own when another group's stem is
#: in the same column name: OrderCount counts orders, not units.
_WEAK_STEMS = frozenset({"count", "amount"})
#: A claim that names nothing and is about the one the claim before it named.
_PRONOUN = re.compile(r"^\s*(?:[-+•*]\s+|\d+[.)]\s+)?(?:\*\*)?(?:(?:this|that)\s+(?:item|product|one|code)|"
                      r"it|this|that|they)\b", re.IGNORECASE)
#: ... and gives it a figure: "It moved 3,648 pieces".
_CARRIED_VERB = re.compile(
    r"^\s*(?:[-+•*]\s+|\d+[.)]\s+)?(?:\*\*)?(?:(?:this|that)\s+(?:item|product|one|code)|it|they)\s+"
    r"(?:also\s+|alone\s+)?(?:moved|sold|shipped|had|has|generated|brought\s+in|earned|made|grossed|recorded|"
    r"reached|totall?ed|accounted\s+for|contributed|delivered)\b", re.IGNORECASE)
#: A stated multiple of the next in a ranking: "more than double the next
#: highest item", "about 1.4 times the next", "twice as many units as the runner-up".
_MULTIPLES = {"double": 2.0, "twice": 2.0, "triple": 3.0, "treble": 3.0, "two": 2.0, "three": 3.0, "four": 4.0,
              "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0}
RATIO = re.compile(
    r"(?P<bound>more\s+than|over|above|at\s+least|nearly|almost|about|around|roughly|approximately|close\s+to|"
    r"[≈~])?\s*(?P<mult>double|twice|triple|treble|(?P<n>\d+(?:\.\d+)?|two|three|four|five|six|seven|eight|nine|"
    r"ten)(?:\s*[-‐‑]?\s*(?:times|fold)\b|\s?[×x](?![\w])))"
    r"(?:\s+(?:as\s+(?:[A-Za-z]+\s+){1,2}as|that\s+of|those\s+of|what))?"
    r"\s+(?:the\s+|its\s+)?(?:next|second|runner[\s‐‑-]?up)(?![\s‐‑-]+(?:lowest|smallest|fewest|least)\b)",
    re.IGNORECASE)
#: How far a plain or approximate multiple may be from the ratio: "double" is 1.8 to 2.2.
RATIO_TOLERANCE = 0.1
#: The calls that cut a result to its first or last rows.
_CUTS = frozenset({"head", "tail", "nlargest", "nsmallest"})
#: Chain steps between a sort and its cut that keep the order.
_ORDER_KEEPING = frozenset({"reset_index", "copy", "round", "rename", "astype", "fillna"})
#: Labels that are no group: the missing-value bin is not a bin of the measure.
_NULL_LABELS = frozenset({"", "nan", "none", "null", "missing", "n/a", "na", "<na>", "nat"})
#: Words of a binned column's name that are not what it bins: "Glucose_bin" is glucose.
_BIN_WORDS = frozenset({"bin", "bins", "binned", "group", "groups", "band", "bands", "range", "ranges", "bucket",
                        "buckets", "cat", "category", "level", "levels", "class", "grp", "quartile", "quintile",
                        "decile", "quantile", "tier", "std", "num", "label"})
#: A result column holding a rate, whose values read as percentages.
_RATE = re.compile(r"rate|share|ratio|proportion|pct|percent|prob|likelihood|mean|avg", re.IGNORECASE)
_RATE_WORDS = re.compile(r"\b(?:likelihood|rate|share|chances?|probability|odds|risk|percentage|proportion)\b",
                         re.IGNORECASE)
#: How many numeric columns a measure is looked for among, and pairs of them.
MAX_MEASURE_COLUMNS = 12

_NUM = r"-?\d+(?:\.\d+)?"
#: Labels of ordered groups: "(88.0, 107.5]", "18-25", "65+", "<18", ">= 80".
_INTERVAL = re.compile(rf"^\s*[(\[]\s*(?P<lo>{_NUM}|-inf)\s*,\s*(?P<hi>{_NUM}|inf)\s*[)\]]\s*$")
_SPAN = re.compile(rf"^\s*(?P<lo>{_NUM})\s*(?:-|–|—|to)\s*(?P<hi>{_NUM})\s*$")
_OPEN_TOP = re.compile(rf"^\s*(?:(?P<lo>{_NUM})\s*\+|(?:>=?|over|above)\s*(?P<lo2>{_NUM}))\s*$", re.IGNORECASE)
_OPEN_BOTTOM = re.compile(rf"^\s*(?:<=?|under|below)\s*(?P<hi>{_NUM})\s*$", re.IGNORECASE)

#: Trend words: up, and down.
_UP = r"higher|greater|larger|bigger|more|increasing|rising|growing|elevated|older|increases?|increased|rises?|" \
      r"raises?|grows?|climbs?|boosts?|goes up"
_DOWN = r"lower|smaller|less|fewer|decreasing|falling|declining|younger|decreases?|decreased|declines?|falls?|" \
        r"drops?|reduces?|goes down|lowers"
_DIRECTION = re.compile(rf"\b(?:(?P<up>{_UP})|(?P<down>{_DOWN}))\b", re.IGNORECASE)
#: Said of X itself, right before it: "higher glucose", "lower fasting glucose".
_X_DIRECTION = re.compile(r"\b(?:(?P<up>higher|greater|larger|bigger|more|increasing|rising|growing|elevated|older)"
                          r"|(?P<down>lower|smaller|less|fewer|decreasing|falling|declining|younger))\b",
                          re.IGNORECASE)
_PREPOSITION = re.compile(r"^\s*(?:with|as|in|of|by|across|on|for|to|the|and)?\s*$", re.IGNORECASE)
#: A trend said not to hold, or said with a shape: nothing monotonic is claimed.
_NO_TREND = re.compile(
    r"\b(?:not|no|n't|never|neither|nor|without|inconsistent|non-?monotonic|u-?shaped|mixed|varies|fluctuat\w*|"
    r"peaks?|dips?|except|unclear|weak)\b", re.IGNORECASE)


#: The code's filters that :func:`_filtered` applies.
_APPLICABLE_OPS = frozenset({"==", "!=", "<", "<=", ">", ">="})


#: Thresholds over every group (1.6 demo, PR #699): "No category exceeds 30 million".
_TH_HINT = re.compile(r"\b(?:no|none|all|every|each|not a single|not one)\b", re.IGNORECASE)
_TH_CHARS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "*": " "})
_TH_CLAUSE_END = re.compile(r",\s|;|:\s|\s-\s|\b(?:while|whereas|but|although|though)\b")
_TH_LEAD = re.compile(r"^\s*(?:(?:and|so|thus|hence|also|yet|indeed|notably|overall|in fact|in short)\s+)*")
_TH_UP = (r"(?:exceeds?|exceeded|surpass(?:es|ed)?|tops|topped"
          r"|(?:goes|go|went|rises?|rose|climbs?|climbed)\s+(?:above|over|beyond)"
          r"|(?:(?:is|are|was|were|sits?|sat|stays?|stayed|remains?|remained)\s+)?"
          r"(?:above|over|more than|greater than|higher than|larger than|bigger than))")
_TH_DOWN = (r"(?:(?:falls?|fell|dips?|dipped|drops?|dropped|goes|go|went|sinks?|sank)\s+(?:below|under|beneath)"
            r"|(?:(?:is|are|was|were|sits?|sat|stays?|stayed|remains?|remained)\s+)?"
            r"(?:below|under|beneath|less than|lower than|smaller than|fewer than))")
#: A word that starts the comparison, so it is never a measure word.
_TH_NOT_VERB = (r"(?!(?:is|are|was|were|has|have|had|goes|go|went|falls?|fell|sits?|sat|stays?|stayed|remains?|"
                r"remained|dips?|dipped|drops?|dropped|rises?|rose|climbs?|climbed|sinks?|sank|exceeds?|surpass\w*|"
                r"tops?|above|over|below|under|beneath|more|less|fewer|greater|higher|lower|larger|smaller|"
                r"bigger)\b)")
_THRESHOLD = re.compile(
    r"^(?:(?P<neg>not a single|not one|none of the|none of them|none|no)|(?P<all>all of the|all the|all|every|each))"
    r"(?:\s+(?:\d+\s+)?(?P<noun>[a-z_]+)(?:'s?(?=\s))?)?"
    r"(?:\s+(?P<m1>(?:" + _TH_NOT_VERB + r"[a-z_]+\s+){0,2}?" + _TH_NOT_VERB + r"[a-z_]+)(?=\s+(?:exceed|surpass|"
    r"top|go|went|rise|rose|climb|is|are|was|were|sit|sat|stay|remain|above|over|more|greater|higher|larger|bigger|"
    r"fall|fell|dip|drop|sink|sank|below|under|beneath|less|lower|smaller|fewer|has|have|had)))?"
    r"(?:\s+(?:has|have|had)\s+(?:an?\s+|any\s+|the\s+)?(?P<m2>(?:[a-z_]+\s+){0,2}?[a-z_]+)?(?:\s+of)?)?"
    r"\s+(?:(?P<up>" + _TH_UP + r")|(?P<down>" + _TH_DOWN + r"))"
    r"\s+[$€£]?(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s?(?P<scale>thousand|million|billion|trillion|mn|bn|tn|k|m|b))?(?P<pct>\s?%|\s?percent)?(?![\w])"
    r"(?P<after>.*)$")
_TH_AFTER_STOP = frozenset({"and", "so", "which", "because", "while", "but", "than", "as", "in", "per", "each",
                            "or", "for", "across", "on", "at", "with", "to", "the", "a", "an", "is", "are",
                            "was", "were", "this", "that", "it"})
_TH_SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6,
              "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12, "tn": 1e12}
_TH_SYNONYMS = {"average": "mean", "avg": "mean"}
#: A row that sums the others is not one of them.
_TOTAL_LABELS = frozenset({"total", "grand total", "subtotal", "all", "overall", "sum"})
#: A column whose name gives its unit: "median_subscribers_million" holds 39.0 for 39,000,000.
_COLUMN_UNITS = {"thousand": 1e3, "thousands": 1e3, "million": 1e6, "millions": 1e6, "mn": 1e6,
                 "billion": 1e9, "billions": 1e9, "bn": 1e9}


#: Aggregates a measure word or a column name may give.
_AGGREGATES = {"average": "mean", "mean": "mean", "avg": "mean", "median": "median", "total": "sum",
               "sum": "sum", "maximum": "max", "max": "max", "minimum": "min", "min": "min"}
#: A label followed by another capitalised word is part of a longer name.
_LONGER_NAME = re.compile(r"[ \t\u00a0\u202f]+[A-Z][a-z]")


def _aggregates(text: str) -> set:
    from avaloka.goal_analysis import _ordered_tokens

    return {_AGGREGATES[t] for t in _ordered_tokens(str(text)) if t in _AGGREGATES}


def _column_unit(column: str) -> float:
    from avaloka.goal_analysis import _ordered_tokens

    return next((_COLUMN_UNITS[t] for t in _ordered_tokens(str(column)) if t in _COLUMN_UNITS), 1.0)


def _filtered(frame: Any, filters: Sequence[tuple]) -> Any:
    """The rows of *frame* every numeric ``(column, op, value)`` keeps, or None
    when one names a column *frame* does not have."""
    import operator

    import pandas as pd

    ops = {"==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt,
           ">=": operator.ge}
    keep = pd.Series(True, index=frame.index)
    for column, op, value in filters:
        if column not in map(str, frame.columns):
            return None
        values = pd.to_numeric(frame[column], errors="coerce")
        keep &= ops[op](values, value).fillna(False).astype(bool)
    return frame[keep]


def _null(label: Any) -> bool:
    return str(label).strip().lower() in _NULL_LABELS


def _order_key(label: Any) -> Optional[float]:
    """Where an ordered group's label sorts: its lower edge, or None."""
    text = str(label)
    for pattern in (_INTERVAL, _SPAN):
        m = pattern.match(text)
        if m:
            lo = m.group("lo")
            return float("-inf") if lo == "-inf" else float(lo)
    m = _OPEN_TOP.match(text)
    if m:
        return float(m.group("lo") or m.group("lo2"))
    m = _OPEN_BOTTOM.match(text)
    if m:
        return float("-inf")
    return None


def _alike(word: str, want: str) -> bool:
    """*word* is a form of *want*: "readmission" of "Readmitted", "recovery" of "Recovered"."""
    from avaloka.goal_analysis import word_stems

    word, want = word.lower(), want.lower()
    if word_stems(word) & word_stems(want):
        return True
    n = 0
    for a, b in zip(word, want):
        if a != b:
            break
        n += 1
    return n >= 5 and n >= 0.6 * min(len(word), len(want))


def _words(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z]{3,}", text)]


def _names(text: str, name: Any, generic: frozenset = frozenset()) -> bool:
    """*text* names the column or value *name* by a distinctive word of it."""
    from avaloka.goal_analysis import _ordered_tokens

    tokens = [t for t in _ordered_tokens(str(name)) if len(t) >= 4 and t not in generic]
    words = _words(text)
    return bool(tokens) and any(_alike(w, t) for t in tokens for w in words)


def _lexicon(words: Sequence[str], columns: Sequence[str]) -> List[str]:
    """The one column of *columns* the first measure word of *words* maps to
    (:data:`MEASURE_WORDS`), or none: two candidates are no match."""
    from avaloka.goal_analysis import _ordered_tokens

    word = next((w.lower() for w in words if w.lower() not in _MEASURE_FILLER), None)
    group = next((i for i, (said, _stems) in enumerate(MEASURE_WORDS) if word in said), None)
    if group is None:
        return []
    stems = MEASURE_WORDS[group][1]
    others = frozenset().union(*(st for i, (_w, st) in enumerate(MEASURE_WORDS) if i != group))
    out = []
    for col in columns:
        tokens = set(_ordered_tokens(str(col)))
        hit = tokens & stems
        if not hit or (hit <= _WEAK_STEMS and tokens & (others - _WEAK_STEMS)):
            continue
        out.append(col)
    return out if len(out) == 1 else []


def _int_of(node: Any, ints: Mapping[str, int]) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        return ints.get(node.id)
    return None


def _str_of(node: Any) -> Optional[str]:
    """``'TotalRevenue'`` of ``'TotalRevenue'`` or ``['TotalRevenue', ...]``."""
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        node = node.elts[0]
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _sorted_by(node: Any, assigns: Mapping[str, List[Tuple[int, Any]]], line: int,
               depth: int = 0) -> Optional[Tuple[Optional[str], bool]]:
    """``(column, ascending)`` of the sort the chain *node* ends in, followed
    through an assignment of its variable (``agg = agg.sort_values(...)``)."""
    for _ in range(20):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "sort_values":
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                by = _str_of(kw.get("by", node.args[0] if node.args else None))
                asc = kw.get("ascending")
                if isinstance(asc, (ast.List, ast.Tuple)) and asc.elts:
                    asc = asc.elts[0]
                ascending = asc.value if isinstance(asc, ast.Constant) and isinstance(asc.value, bool) else True
                return by, ascending
            if node.func.attr not in _ORDER_KEEPING:
                return None
            node = node.func.value
        elif isinstance(node, ast.Name) and depth < 3:
            earlier = [(n, v) for n, v in assigns.get(node.id, []) if n < line]
            if not earlier:
                return None
            n, value = earlier[-1]
            return _sorted_by(value, assigns, n, depth + 1)
        else:
            return None
    return None


def result_truncation(code: Any, n_rows: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """The cut that made the result a part of what the code computed:
    ``agg.sort_values(by='TotalRevenue', ascending=False).head(TOP_N)`` keeps
    the top 10 by TotalRevenue. ``{"n", "by", "top", "call", "line"}``, read
    from the code's syntax and never run; ``by`` and ``top`` are None for a
    cut with no sort before it. Only a cut that keeps as many rows as the
    result has (*n_rows*) is the result's; the last such one counts."""
    if not isinstance(code, str) or not code.strip():
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    ints: Dict[str, int] = {}
    assigns: Dict[str, List[Tuple[int, Any]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value = _int_of(node.value, {})
            if value is not None:
                ints[name] = value
            assigns.setdefault(name, []).append((node.lineno, node.value))
    for name in assigns:
        assigns[name].sort(key=lambda nv: nv[0])
    found: Optional[Dict[str, Any]] = None
    for node in ast.walk(tree):
        cut = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _CUTS:
            attr = node.func.attr
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            n_node = node.args[0] if node.args else kw.get("n")
            n = _int_of(n_node, ints) if n_node is not None else (5 if attr in ("head", "tail") else None)
            if n is None or n <= 0:
                continue
            if attr in ("nlargest", "nsmallest"):
                by = _str_of(node.args[1] if len(node.args) > 1 else kw.get("columns"))
                cut = {"n": n, "by": by, "top": attr == "nlargest" if by else None}
            else:
                order = _sorted_by(node.func.value, assigns, node.lineno)
                by, ascending = order if order else (None, True)
                cut = {"n": n, "by": by, "top": (attr == "head") != ascending if by else None}
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice) \
                and node.slice.lower is None and node.slice.step is None and node.slice.upper is not None:
            n = _int_of(node.slice.upper, ints)
            base = node.value.value if isinstance(node.value, ast.Attribute) and node.value.attr == "iloc" \
                else node.value
            if n is None or n <= 0 or isinstance(base, ast.Constant):
                continue
            order = _sorted_by(base, assigns, node.lineno)
            if order is None and not (isinstance(node.value, ast.Attribute) and node.value.attr == "iloc"):
                continue                       # a slice of a list or a string, not of rows
            by, ascending = order if order else (None, True)
            cut = {"n": n, "by": by, "top": not ascending if by else None}
        if cut is None or (n_rows is not None and cut["n"] != n_rows):
            continue
        cut.update(call=ast.unparse(node)[-80:], line=getattr(node, "lineno", 0))
        if found is None or cut["line"] >= found["line"]:
            found = cut
    return found


def _fmt(value: float, percent: bool) -> str:
    if percent:
        return f"{value:.1%}"
    if float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}" if abs(value) >= 1 else f"{value:.4g}"


_UNSET = object()
#: :meth:`Extremes._over_all_rows` found the claim true of all the groups.
_CONFIRMED = object()


class Extremes:
    """The result's rows and the frames the run read, for one reply."""

    def __init__(self, flat: Mapping[str, Any], used: Optional[cv.DataUsed] = None, code: Any = None):
        self.flat = flat
        self.rows = result_rows(flat)
        self.used = used
        self.code = code
        columns: Dict[str, List[Any]] = {}
        for row in self.rows:
            for col, value in row.items():
                columns.setdefault(col, []).append(value)
        #: Columns of codes that key the result (StockCode), digits only or not: names, not measures.
        self.keys = key_columns(self.rows)
        self.numeric = [c for c, vals in columns.items() if c not in self.keys
                        and all(cv._as_number(v) is not None for v in vals if v is not None and str(v).strip())
                        and any(cv._as_number(v) is not None for v in vals)]
        self.text = [c for c in columns if c not in self.numeric
                     and (c in self.keys or all(isinstance(v, str) for v in columns[c] if v is not None))]
        self.labels = row_labels(self.rows, self.keys)
        #: Claims a check here judged true: the verifier counts them as checked.
        self.resolved: set = set()
        #: Spans of a claim a check here judged (a ratio's multiple): no figure of the result.
        self.judged_spans: Dict[str, List[Tuple[int, int]]] = {}
        #: Superlatives over a cut that nothing confirmed: claim -> the line the reply owes.
        self.unconfirmed: Dict[str, str] = {}
        #: The measure each superlative was about, for the claim after it ("It moved ...").
        self._measures: Dict[str, str] = {}
        #: Superlatives whose measure their own claim does not give: the next claim may.
        self._open: Dict[str, Tuple[Any, Tuple[str, str, List[int]]]] = {}
        self._cut: Any = _UNSET

    @property
    def cut(self) -> Optional[Dict[str, Any]]:
        """The code's cut of the result (:func:`result_truncation`), or None."""
        if self._cut is _UNSET:
            try:
                self._cut = result_truncation(self.code, len(self.rows)) if self.rows else None
            except Exception:  # noqa: BLE001 - reading the code is extra evidence, never a crash
                logger.debug("[claim-verifier] could not read the code's cut", exc_info=True)
                self._cut = None
        return self._cut

    # -- shared ----------------------------------------------------------------
    def _entity(self, claim: str) -> Optional[Tuple[str, str, List[int]]]:
        """``(label, key column, row indices)`` of the one result entity *claim*
        names. Labels of the same rows are one entity: "EDWARDIAN PARASOL BLACK
        (StockCode 15056BL)" names one row by two columns (the description is
        that of 15056bl too), and is keyed by the label that names only it."""
        spans = cv._label_spans(claim, [label for label, cells in self.labels.items()
                                        if any(c in self.text for _i, c in cells)])
        if not spans and re.search(r"\s+and\s+", claim):
            # "Autos and Vehicles" is the result's "Autos & Vehicles" (1.6).
            return self._entity(re.sub(r"\s+and\s+", " & ", claim))
        named = {label: {(i, c) for i, c in self.labels[label] if c in self.text} for _s, _e, label in spans}
        if not named:
            return None
        rows = set.intersection(*({i for i, _c in cells} for cells in named.values()))
        named = {label: cells for label, cells in named.items() if {i for i, _c in cells} == rows}
        if not rows or not named:
            return None
        for label, cells in sorted(named.items(), key=lambda kv: self._repeats(next(iter(kv[1]))[1])):
            keys = {c for _i, c in cells}
            if len(keys) == 1:
                key = keys.pop()
                return label, key, sorted(i for i, _c in cells)
        return None

    def _repeats(self, column: str) -> int:
        """How many rows share a value of *column* with another row: 0 for a key."""
        values = [str(r.get(column)).strip() for r in self.rows]
        return len(values) - len(set(values))

    def _subgroups(self, key: str) -> List[str]:
        """The text columns that split the result into subgroups (Outcome_std),
        not the key or another name of each row (StockCode beside Description)."""
        return [c for c in self.text if c != key and self._repeats(c) >= len(self.rows) / 2]

    def _subgroup(self, claim: str, key: str, rows: Sequence[int]) -> Optional[List[int]]:
        """*rows* narrowed by the values of the other text columns *claim* names."""
        out = list(rows)
        for col in self._subgroups(key):
            values = {str(self.rows[i].get(col)).strip() for i in out}
            if len(values) < 2:
                continue
            named = [v for v in values if not _null(v) and _names(claim, v)]
            if len(named) != 1:
                return None
            out = [i for i in out if str(self.rows[i].get(col)).strip() == named[0]]
        return out

    def _peers(self, i: int, key: str, measure: str) -> List[Tuple[str, float, str]]:
        """``(label, value, shown)`` of the rows comparable to row *i*: the same value in
        every subgroup column. A label is given with the row's other names:
        "22423 (REGENCY CAKESTAND 3 TIER)"."""
        groups = self._subgroups(key)
        context = {c: str(self.rows[i].get(c)).strip() for c in groups}
        names = [c for c in self.text if c != key and c not in groups]
        out = []
        for row in self.rows:
            if any(str(row.get(c)).strip() != v for c, v in context.items()):
                continue
            value = cv._as_number(row.get(measure))
            label = str(row.get(key)).strip()
            if value is None or _null(label) or label.lower() in _TOTAL_LABELS:
                continue                            # a total row sums the groups; it is none of them
            also = [str(row.get(c)).strip() for c in names if not _null(row.get(c))]
            out.append((label, value, f"{label} ({', '.join(also)})" if also else label))
        return out

    def _percent(self, measure: str, values: Sequence[float]) -> bool:
        return bool(_RATE.search(measure)) and all(0 <= v <= 1 for v in values)

    def _context(self, i: int, key: str) -> str:
        parts = [f"{c} = {str(self.rows[i].get(c)).strip()}" for c in self._subgroups(key)]
        return f" among rows with {', '.join(parts)}" if parts else ""

    # -- superlatives ----------------------------------------------------------
    def superlative(self, claim: str) -> Optional[cv.ClaimFinding]:
        if not self.rows or not self.numeric or _SCOPED.search(claim):
            return None
        hits = list(SUPERLATIVE.finditer(claim))
        if len(hits) != 1:
            return None
        m = hits[0]
        if _QUALIFIED.search(claim[:m.start()]):
            return None                     # "one of the lowest" is no claim to the extreme
        entity = self._entity(claim)
        if entity is None:
            return None
        if any(_LONGER_NAME.match(claim, e) for _s, e, _l in cv._label_spans(claim, [entity[0]])):
            return None                     # "Gaming Shorts" is not the result's "Gaming"
        picked = self._measure(claim, m, entity[1], entity[2])
        if picked is None:
            self._open[claim] = (m, entity)
            return None
        return self._judge(claim, m, entity, picked)

    def carried_superlative(self, claim: str, prior: Optional[str]) -> Optional[cv.ClaimFinding]:
        """"X sold the most. / It moved 3,648 pieces": a superlative whose
        measure *prior* does not give, given by the figure of *claim*, which
        names nothing else. The finding is *prior*'s: that is what it says."""
        if prior is None or prior not in self._open or not _PRONOUN.match(claim) or SUPERLATIVE.search(claim) \
                or self._entity(claim) is not None:
            return None
        m, entity = self._open.pop(prior)
        picked = self._figure_column(claim, entity[2])
        if picked is None:
            return None
        return self._judge(prior, m, entity, picked)

    def _judge(self, claim: str, m: Any, entity: Tuple[str, str, List[int]],
               picked: Tuple[int, str]) -> Optional[cv.ClaimFinding]:
        label, key, rows = entity
        term = m.group("term").lower().replace("the ", "")
        rank = _RANKS.get((m.group("rank") or "").lower(), 1)
        phrase = re.sub(r"\s+", " ", m.group(0).lower()).replace("the ", "")
        i, measure = picked
        self._measures[claim] = measure
        peers = self._peers(i, key, measure)
        if len(peers) < 2 or [p for p, _v, _s in peers].count(label) != 1:
            return None
        high = term not in _LOW_TERMS
        values = sorted({v for _p, v, _s in peers}, reverse=high)
        if rank > len(values):
            return None
        mine = cv._as_number(self.rows[i].get(measure))
        percent = self._percent(measure, [v for _p, v, _s in peers])
        if abs(mine - values[rank - 1]) > 1e-9 * max(1.0, abs(mine)):
            if cv.check_ordering_contradicted(claim, dict(self.flat)) is not None:
                return None                     # the result's own ordering check says it already
            winner = next(shown for _p, v, shown in peers if v == values[rank - 1])
            return cv.ClaimFinding(
                claim=claim, verdict=cv.Verdict.CONTRADICTED, check="superlative_contradicted",
                detail=(f"{label}'s {measure} ({_fmt(mine, percent)}) is not the {phrase}{self._context(i, key)}: "
                        f"{winner} has {_fmt(values[rank - 1], percent)}"),
                evidence_ref=f"execution_output.{measure}")
        # The result may keep only some of the groups: the claim is about all of them.
        # Other text columns may only be other names of each row ("85123A" beside
        # its description, rc16); a column that splits the rows into subgroups
        # makes "all of them" ambiguous.
        found = None if self._subgroups(key) else self._over_all_rows(claim, label, key, measure, rank, high,
                                                                       phrase)
        if isinstance(found, cv.ClaimFinding):
            return found
        cut = self.cut
        if found is _CONFIRMED or cut is None or (cut["by"] == measure and cut["top"] == high and rank <= cut["n"]):
            # True of all the groups, or nothing says the result is a part of
            # them, or the code cut it by this very measure from this end.
            self.resolved.add(claim)
            return None
        # The code cut the result by another measure, or from the other end,
        # and the recompute over all rows could not run: the claim holds of
        # the kept rows only as far as anyone knows. Fail closed (rc18d): it
        # is not checked, not counted as checked, and the reply says so in
        # the code's words (rc14's unreproducible measure stays unflagged).
        if cut["by"]:
            kept = f"the {'top' if cut['top'] else 'bottom'} {cut['n']} by {cut['by']}"
        else:
            kept = f"the first {cut['n']} the code kept"
        self.unconfirmed[claim] = (f"Not checked over all rows: {label} has the {phrase} {measure} among the "
                                   f"{len(self.rows)} rows the result keeps ({kept}); whether it does over all rows "
                                   "could not be checked.")
        return None

    def _measure(self, claim: str, m: Any, key: str, rows: Sequence[int]) -> Optional[Tuple[int, str]]:
        """``(row, column)`` the superlative is about: the column the words after
        it name ("highest sales"), else the column holding the first figure
        after it in its clause ("Highest revenue: $9,570.72")."""
        end = m.end()
        stop = _CLAUSE_END.search(claim, end)
        if stop is not None and stop.group(0) == "," and len(re.findall(r"[A-Za-z]+", claim[end:stop.start()])) <= 3:
            figure = _FIGURE_NEXT.match(claim, stop.end())
            if figure is not None:             # "the most units, 3,648": the figure is the measure's
                stop = _CLAUSE_END.search(claim, figure.end())
        clause = claim[end:stop.start() if stop else len(claim)]
        words = re.findall(r"[A-Za-z]+", clause)
        head = " ".join(words[:3])
        named = [c for c in self.numeric if c != key and _names(head, c, _BIN_WORDS | {"total", "sum"})]
        if not named:
            # "sold the most units": no column is named units; the lexicon maps the word.
            said = re.findall(r"[A-Za-z]+", m.group(0))[1:] if re.search(r"grossing|earning", m.group(0), re.I) \
                else words[:4]
            named = _lexicon(said, [c for c in self.numeric if c != key])
        if not named and len(self.numeric) == 1 and self.numeric[0] != key:
            # Over a result of one measure, a bare superlative is about it
            # (1.6): "X is the lowest", "sits at the bottom", "ranks highest",
            # "has the least subscribers on average". Not when its words name
            # something else: "the largest category", "highest-earning channel".
            only = self.numeric[0]
            extra = [w for w in words[:4] + re.findall(r"[A-Za-z]+", m.group(0))[1:]
                     if w.lower() not in _MEASURE_FILLER | _BARE_FILLER and w.lower() not in _AGGREGATES
                     and w.lower() != str(only).lower() and not _names(w, only)]
            if not extra:
                named = [only]
        said = _aggregates(head)
        if said:
            # "the lowest average" is no claim about a median column (1.6 demo,
            # PR #699): a column named for another aggregate is not its measure.
            pool = [c for c in (named or self.numeric) if c != key
                    and (not _aggregates(c) or _aggregates(c) & said)]
            if not pool:
                return None
            named = [c for c in named if c in pool]
        labels = cv._label_spans(claim, self.labels)
        figures = [f for f in cv.attached_figures(claim)
                   if not any(s <= f["start"] < e for s, e, _l in labels)]
        after = [f for f in figures if end <= f["start"] < end + len(clause)]
        if not after and stop is not None and stop.group(0).strip() in ("-", "–", "—") \
                and any(end <= s < stop.start() for s, _e, _l in labels):
            # "Highest revenue: 85123A – $10,132.48": the dash puts the entity's
            # figure after it, still the superlative's (rc16).
            nxt = _CLAUSE_END.search(claim, stop.end())
            after = [f for f in figures if stop.end() <= f["start"] < (nxt.start() if nxt else len(claim))]
        candidates = []
        narrowed = self._subgroup(claim, key, rows) or list(rows)
        for i in narrowed:
            for col in named or self.numeric:
                if col == key or (said and _aggregates(col) and not _aggregates(col) & said):
                    continue
                value = cv._as_number(self.rows[i].get(col))
                if value is None:
                    continue
                if after and not any(cv.figure_matches(f["value"], f["raw"], value, f["approx"], f["high"])
                                     for f in after[:1]):
                    continue
                if not after and not named:
                    continue
                candidates.append((i, col))
        if len(candidates) == 1:
            return candidates[0]
        if len({i for i, _c in candidates}) == 1 and len(candidates) > 1 \
                and self._same_order([c for _i, c in candidates]):
            # "The lowest median is X" over median_subscribers and
            # median_subscribers_million: two names of one measure, one answer.
            return candidates[0]
        if len(named) == 1 and len(narrowed) == 1 and not after:
            return narrowed[0], named[0]
        return None

    def _same_order(self, columns: Sequence[str]) -> bool:
        """*columns* rank the result's rows the same way: one measure in two units."""
        orders = []
        for col in columns:
            values = [cv._as_number(r.get(col)) for r in self.rows]
            if any(v is None for v in values):
                return False
            orders.append(sorted(range(len(values)), key=lambda k, vs=values: (vs[k], k)))
        return all(o == orders[0] for o in orders)

    def _figure_column(self, claim: str, rows: Sequence[int]) -> Optional[Tuple[int, str]]:
        """``(row, column)`` of the one entity row whose value in exactly one
        numeric column is the first figure of *claim* (not in a label)."""
        if len(rows) != 1:
            return None
        labels = cv._label_spans(claim, self.labels)
        figures = [f for f in cv.attached_figures(claim) if not any(s <= f["start"] < e for s, e, _l in labels)]
        if not figures:
            return None
        f, i = figures[0], rows[0]
        cols = [c for c in self.numeric if c not in self.keys and cv._as_number(self.rows[i].get(c)) is not None
                and cv.figure_matches(f["value"], f["raw"], cv._as_number(self.rows[i].get(c)), f["approx"], f["high"])]
        return (i, cols[0]) if len(cols) == 1 else None

    def _carried(self, claim: str, prior: Optional[str]) -> Optional[Tuple[str, str, List[int]]]:
        """The entity of *prior* when *claim* starts "It / This / That / They"
        and names none of its own: only the claim just before it counts."""
        if prior is None or not _PRONOUN.match(claim) or self._entity(claim) is not None:
            return None
        return self._entity(prior)

    # -- ratios --------------------------------------------------------------------
    def ratio(self, claim: str, prior: Optional[str] = None) -> Optional[cv.ClaimFinding]:
        """"It moved 3,648 pieces, more than double the next highest item": the
        entity's value over the next one below it in the ranking of the result."""
        if not self.rows or not self.numeric or _SCOPED.search(claim):
            return None
        hits = list(RATIO.finditer(claim))
        if len(hits) != 1:
            return None
        m = hits[0]
        carried = None
        entity = self._entity(claim)
        if entity is None:
            entity = carried = self._carried(claim, prior)
        if entity is None:
            return None
        label, key, rows = entity
        picked = self._figure_column(claim, rows)
        if picked is None:
            named = _lexicon(re.findall(r"[A-Za-z]+", claim[:m.start()])[-3:], [c for c in self.numeric if c != key]) \
                or [c for c in self.numeric if c != key and _names(claim[:m.start()], c, _BIN_WORDS | {"total", "sum"})]
            measure = named[0] if len(named) == 1 else (self._measures.get(prior) if carried and prior else None)
            narrowed = self._subgroup(claim, key, rows) or list(rows)
            if measure is None or len(narrowed) != 1:
                return None
            picked = (narrowed[0], measure)
        i, measure = picked
        mine = cv._as_number(self.rows[i].get(measure))
        peers = self._peers(i, key, measure)
        if mine is None or mine <= 0 or [p for p, _v, _s in peers].count(label) != 1:
            return None
        below = sorted({v for p, v, _s in peers if p != label and v < mine}, reverse=True)
        if not below or below[0] <= 0:
            return None
        nxt = below[0]
        r = mine / nxt
        word = re.match(r"[a-z]+|\d+(?:\.\d+)?", m.group("mult").lower()).group(0)
        n = _MULTIPLES.get(word) or (float(word) if word[0].isdigit() else None)
        if not n or n <= 1:
            return None
        bound = re.sub(r"\s+", " ", (m.group("bound") or "").lower())
        written = m.group("n") or ""
        slack = max(RATIO_TOLERANCE * n, 0.5 * 10 ** -len(written.split(".")[1]) if "." in written else 0.0)
        if bound in ("more than", "over", "above", "at least"):
            low, high = n * 0.99, float("inf")
        elif bound in ("nearly", "almost"):
            low, high = n * (1 - 2 * RATIO_TOLERANCE), n * 1.01
        else:
            low, high = n - slack, n + slack
        cut = self.cut
        whole = cut is None or (cut["by"] == measure and cut["top"] is True)
        if r < low or (whole and r > high):
            winner = next(shown for p, v, shown in peers if v == nxt)
            percent = self._percent(measure, [v for _p, v, _s in peers])
            stated = re.sub(r"\s+", " ", m.group(0).strip())
            return cv.ClaimFinding(
                claim=claim, verdict=cv.Verdict.CONTRADICTED, check="ratio_contradicted",
                detail=(f"{label}'s {measure} ({_fmt(mine, percent)}) is {r:.2f} times the next highest"
                        f"{self._context(i, key)}, {winner} with {_fmt(nxt, percent)}, not {stated}"),
                evidence_ref=f"execution_output.{measure}")
        if whole:
            # A multiple of the ranking, judged here: not a figure the result must hold.
            self.resolved.add(claim)
            self.judged_spans.setdefault(claim, []).append(m.span("mult"))
        return None

    def carried_figure(self, claim: str, prior: Optional[str]) -> Optional[cv.ClaimFinding]:
        """"X ... / It moved 2,393 pieces": the figure is said of X, and is another row's."""
        entity = self._carried(claim, prior) if _CARRIED_VERB.match(claim) else None
        if entity is None or len(entity[2]) != 1:
            return None
        label, key, (i,) = entity
        labels = cv._label_spans(claim, self.labels)
        figures = [f for f in cv.attached_figures(claim) if not any(s <= f["start"] < e for s, e, _l in labels)]
        if not figures:
            return None
        f = figures[0]

        def has(row: Mapping[str, Any], col: str) -> bool:
            value = cv._as_number(row.get(col))
            return value is not None and cv.figure_matches(f["value"], f["raw"], value, f["approx"], f["high"])

        cols = [c for c in self.numeric if c not in self.keys]
        if any(has(self.rows[i], c) for c in cols):
            return None
        others = [(j, c) for j, row in enumerate(self.rows) if j != i for c in cols if has(row, c)]
        if len(others) != 1:
            return None
        j, col = others[0]
        mine = cv._as_number(self.rows[i].get(col))
        percent = self._percent(col, [mine] if mine is not None else [])
        return cv.ClaimFinding(
            claim=claim, verdict=cv.Verdict.CONTRADICTED, check="figure_misattributed",
            detail=(f"{f['raw']} is {str(self.rows[j].get(key)).strip()}'s {col}, not {label}'s"
                    + (f" ({_fmt(mine, percent)})" if mine is not None else "")),
            evidence_ref=f"execution_output.{col}")

    # -- the whole data ----------------------------------------------------------
    def _over_all_rows(self, claim: str, label: str, key: str, measure: str, rank: int, high: bool,
                       phrase: str) -> Any:
        """A finding, :data:`_CONFIRMED`, or None when it cannot be told."""
        if self.used is None or not self.used.frames:
            return None
        # A filter of the code is applied before the groups are compared: the
        # claim is about all the groups of the rows the analysis kept. A filter
        # that cannot be applied here (on a string, a derived column) leaves
        # "all of them" unknown, and nothing is checked.
        filters = cv.code_filters(self.code)
        if any(op not in _APPLICABLE_OPS or isinstance(value, (str, tuple)) for _c, op, value in filters):
            return None
        # The row may be named by more than one column ("WHITE HANGING HEART
        # T-LIGHT HOLDER (StockCode 85123A)"); the data groups by whichever of
        # them reproduces the result (a description is spelt several ways in a
        # messy file, a stock code once).
        found = None
        i = next((i for i, row in enumerate(self.rows) if str(row.get(key)).strip() == label), None)
        for name in [key, *[c for c in self.text if c != key]] if i is not None else [key]:
            try:
                found = self._recompute(name, measure, filters)
            except Exception:  # noqa: BLE001 - a measure that cannot be recomputed checks nothing
                logger.debug("[claim-verifier] could not recompute a superlative's measure", exc_info=True)
                found = None
            if found is not None:
                label, key = str(self.rows[i].get(name)).strip(), name
                break
        if found is None:
            return None
        totals, how, kept = found
        shown = {str(r.get(key)).strip() for r in self.rows if not _null(r.get(key))}
        totals = totals[[not _null(k) for k in totals.index]]
        if label not in totals.index:
            return None
        if len(totals) <= len(shown):
            return _CONFIRMED                   # the result holds every group
        ordered = totals.sort_values(ascending=not high, kind="mergesort")
        values = sorted(set(ordered.tolist()), reverse=high)
        if rank > len(values):
            return None
        mine = float(totals[label])
        expected = values[rank - 1]
        if abs(mine - expected) <= 1e-6 * max(1.0, abs(expected)):
            return _CONFIRMED
        winner = str(ordered[ordered == expected].index[0])
        place = values.index(mine) + 1 if mine in values else None
        where = f"; {label} ranks {place} ({_fmt(mine, False)})" if place else ""
        return cv.ClaimFinding(
            claim=claim, verdict=cv.Verdict.CONTRADICTED, check="superlative_of_subset",
            detail=(f"{label} is the {phrase} only among the {len(shown)} {key} values the analysis kept; over all "
                    f"rows{kept} the {phrase} {measure} ({how}) is {winner} ({_fmt(expected, False)}){where}"),
            evidence_ref=f"data.{key}")

    def _recompute(self, key: str, measure: str, filters: Sequence[tuple] = ()) -> Optional[Tuple[Any, str, str]]:
        """``(the measure per group over all rows, how it is computed, which rows)``
        by the expression that reproduces the result's own values on the rows it
        shows, over the rows *filters* (the code's ``(column, op, value)``) keep."""
        import pandas as pd

        target: Dict[str, float] = {}
        for row in self.rows:
            k, v = str(row.get(key)).strip(), cv._as_number(row.get(measure))
            if _null(k) or v is None:
                continue
            if k in target:
                return None                         # one row per group, or it is no group total
            target[k] = v
        if not target:
            return None
        frames = [f for _l, f in self.used.frames if key in map(str, f.columns)]
        if not frames:
            return None
        bases = [pd.concat(frames, ignore_index=True)] if len(frames) > 1 else []
        bases += frames
        kept = ""
        if filters:
            kept = " with " + " and ".join(f"{c} {op} {v:g}" if isinstance(v, (int, float)) else f"{c} {op} {v}"
                                           for c, op, v in dict.fromkeys(filters))
        for base in bases:
            if filters:
                base = _filtered(base, filters)
                if base is None:
                    return None
            keys = base[key].astype(str).str.strip()
            if not set(target) <= set(keys.unique()):
                continue
            numeric: Dict[str, Any] = {}
            for col in base.columns:
                if str(col) == key or len(numeric) >= MAX_MEASURE_COLUMNS:
                    continue
                series = base[col]
                if pd.api.types.is_bool_dtype(series):
                    series = series.astype(float)
                values = pd.to_numeric(series, errors="coerce") if not pd.api.types.is_numeric_dtype(series) \
                    else series.astype(float)
                if values.notna().mean() >= 0.5:
                    numeric[str(col)] = values
            shown = keys.isin(target)
            for how, compute in self._expressions(numeric):
                sub = compute(shown)
                if sub is None:
                    continue
                per = sub.groupby(keys[shown]).agg(how[0])
                if all(k in per.index and abs(float(per[k]) - v) <= max(0.01, 1e-6 * abs(v))
                       for k, v in target.items()):
                    full = compute(None).groupby(keys).agg(how[0])
                    return full.astype(float), f"{how[1]} per {key}", kept
        return None

    @staticmethod
    def _expressions(numeric: Mapping[str, Any]):
        """``((aggregation, description), compute)``: compute(mask) gives the series to aggregate."""
        def pick(series):
            return lambda mask: series if mask is None else series[mask]

        names = list(numeric)
        ones = next(iter(numeric.values()), None)
        if ones is not None:
            unit = ones.notna() | ones.isna()
            yield ("sum", "row count"), pick(unit.astype(float))
        for name in names:
            for agg, word in (("sum", "sum"), ("mean", "mean"), ("max", "max"), ("min", "min")):
                yield (agg, f"{word} of {name}"), pick(numeric[name])
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                yield ("sum", f"sum of {a} × {b}"), pick(numeric[a] * numeric[b])

    # -- trends ----------------------------------------------------------------
    def trend(self, claim: str, causal: Optional[cv.ClaimFinding] = None) -> Optional[cv.ClaimFinding]:
        """"Higher X increases Y" over a result grouped by ordered bins of X."""
        if len(self.rows) < 3 or _NO_TREND.search(claim):
            return None
        for key in self.text:
            if not _names(claim, key, _BIN_WORDS):
                continue
            order = {str(lab).strip(): _order_key(lab) for lab in (r.get(key) for r in self.rows) if not _null(lab)}
            if len(order) < 3 or any(v is None for v in order.values()):
                continue
            finding = self._trend_over(claim, key, order, causal)
            if finding is not None:
                return finding
        return None

    def _trend_over(self, claim: str, key: str, order: Mapping[str, float],
                    causal: Optional[cv.ClaimFinding]) -> Optional[cv.ClaimFinding]:
        from avaloka.goal_analysis import _ordered_tokens

        tokens = [t for t in _ordered_tokens(key) if len(t) >= 4 and t not in _BIN_WORDS]
        x_at = [w for w in re.finditer(r"[A-Za-z]{3,}", claim) if any(_alike(w.group(0), t) for t in tokens)]
        if not x_at:
            return None
        x = x_at[0]
        moves = list(_DIRECTION.finditer(claim))
        before = [d for d in _X_DIRECTION.finditer(claim[:x.start()])
                  if len(claim[d.end():x.start()].split()) <= 1
                  and (not claim[d.end():x.start()].strip() or not _PREPOSITION.match(claim[d.end():x.start()]))]
        after = [d for d in moves if d.start() >= x.end() and re.match(r"^\s*(?:\w+\s+){0,1}$",
                                                                      claim[x.end():d.start()])
                 and re.search(r"\bas\s+(?:\w+\s+)?$", claim[:x.start()])]
        x_dir = before[-1] if before else (after[0] if after else None)
        if x_dir is not None:
            sign_x = 1 if x_dir.group("up") else -1
            rest = [d for d in moves if d.start() != x_dir.start()]
        elif re.search(r"\b(?:with|as)\s+(?:\w+\s+)?$", claim[:x.start()]):
            sign_x, rest = 1, moves                              # "Y rises with X"
        else:
            return None
        if not rest:
            return None
        sign_y = 1 if rest[0].group("up") else -1
        sign = sign_x * sign_y

        # What rises: a subgroup the claim names, in a column it names or a rate.
        others = [c for c in self.text if c != key]
        rows = list(range(len(self.rows)))
        what = []
        for col in others:
            values = {str(self.rows[i].get(col)).strip() for i in rows} - {""}
            named = [v for v in values if not _null(v) and _names(claim, v)]
            if len(named) != 1:
                return None
            rows = [i for i in rows if str(self.rows[i].get(col)).strip() == named[0]]
            what.append(named[0])
        measures = [c for c in self.numeric if c != key]
        named = [c for c in measures if _names(claim, c, _BIN_WORDS)]
        if len(named) == 1:
            measure = named[0]
        elif _RATE_WORDS.search(claim) and len([c for c in measures if _RATE.search(c)]) == 1:
            measure = next(c for c in measures if _RATE.search(c))
        elif len(measures) == 1:
            measure = measures[0]
        else:
            return None
        series: Dict[str, float] = {}
        for i in rows:
            label = str(self.rows[i].get(key)).strip()
            value = cv._as_number(self.rows[i].get(measure))
            if label not in order or value is None:
                continue
            if label in series:
                return None
            series[label] = value
        if len(series) < 3:
            return None
        points = sorted(series.items(), key=lambda kv: order[kv[0]])
        values = [v for _l, v in points]
        steps = [b - a for a, b in zip(values, values[1:])]
        if all(s * sign >= -1e-9 for s in steps):
            return None
        percent = self._percent(measure, values)
        of = f"{measure} of {', '.join(what)}" if what else measure
        detail = (f"it says {of} {'rises' if sign > 0 else 'falls'} with {key}, but across the {key} groups it is "
                  f"not monotonic: " + ", ".join(f"{label} {_fmt(v, percent)}" for label, v in points))
        if causal is not None:
            detail += f"; and it {causal.detail}"
        return cv.ClaimFinding(claim=claim, verdict=cv.Verdict.CONTRADICTED, check="trend_contradicted",
                               detail=detail, evidence_ref=f"execution_output.{measure}")

    # -- a range said of every member of a group ---------------------------------
    def range_over_set(self, claim: str) -> Optional[cv.ClaimFinding]:
        """"all smoking_status and work_type categories ... (≈0.02–0.017)":
        each member of those groups (the result rows named ``group_value``)
        must lie in the range."""
        if not re.search(r"\b(?:all|every|each)\b", claim, re.IGNORECASE) or len(self.numeric) != 1:
            return None
        measure = self.numeric[0]
        text = cv._minus_normalized(claim)
        figures = [(m.start(), m.end(), m.group(0), cv._parse_number(m.group(0))) for m in cv._NUMBER.finditer(text)]
        ranges = [(a, b) for a, b in zip(figures, figures[1:])
                  if re.fullmatch(r"\s*(?:-|to)\s*", text[a[1]:b[0]]) and a[3] is not None and b[3] is not None]
        if len(ranges) != 1:
            return None
        a, b = ranges[0]
        approx = bool(cv._APPROX.search(text[max(0, a[0] - 18):a[0]]))
        (lo, lo_raw), (hi, hi_raw) = sorted(((a[3], a[2]), (b[3], b[2])))
        lo -= cv.figure_tolerance(lo, lo_raw, approx)
        hi += cv.figure_tolerance(hi, hi_raw, approx)
        quantifier = re.search(r"\b(?:all|every|each)\b", text[:a[0]], re.IGNORECASE)
        if quantifier is None:
            return None
        scope = text[quantifier.end():a[0]]
        members: Dict[str, float] = {}
        groups = []
        for prefix in self._prefixes():
            pattern = r"[_\s]+".join(re.escape(p) for p in prefix.split("_") if p)
            if not re.search(rf"(?<![\w]){pattern}(?![\w])", scope, re.IGNORECASE):
                continue
            groups.append(prefix)
            for label, cells in self.labels.items():
                if label.startswith(prefix + "_"):
                    for i, _c in cells:
                        value = cv._as_number(self.rows[i].get(measure))
                        if value is not None:
                            members[label] = value
        if not groups or len(members) < 2:
            return None
        outside = [(label, v) for label, v in members.items() if not lo <= v <= hi]
        if not outside:
            return None
        said = claim[a[0]:b[1]] if len(text) == len(claim) else text[a[0]:b[1]]
        return cv.ClaimFinding(
            claim=claim, verdict=cv.Verdict.CONTRADICTED, check="range_contradicted",
            detail=(f"it puts every {' and '.join(groups)} category at {said}, but "
                    + ", ".join(f"{label} is {_fmt(v, False)}" for label, v in outside[:4])
                    + f" ({measure})"),
            evidence_ref=f"execution_output.{measure}")

    def _prefixes(self) -> List[str]:
        """Group names the result's one-hot labels share: ``work_type`` of ``work_type_Private``, ...
        A prefix of one label only is no group."""
        counts: Dict[str, int] = {}
        for label in self.labels:
            for m in re.finditer("_", label):
                counts[label[:m.start()]] = counts.get(label[:m.start()], 0) + 1
        return sorted((p for p, n in counts.items() if n >= 2 and len(p) >= 3), key=len, reverse=True)

    # -- thresholds ----------------------------------------------------------------
    def threshold(self, claim: str) -> Optional[cv.ClaimFinding]:
        """"No category exceeds 30 million" when a group does (1.6 demo, PR #699).

        "No / none / not one X is above|below N" and "all / every / each X is
        above|below N" must hold for every group: X is the key column ("category"),
        or the result's one key for a bare "none". The bound is read at the
        precision it is stated ("30 million" is broken only beyond 30.5 million).
        Held on the result's rows, it is checked on all the groups the data has
        when the frames the run read can recompute the measure; over a cut that
        cannot be recomputed it is unconfirmed, as a superlative is.
        """
        if not self.rows or not self.numeric or _SCOPED.search(claim) or not _TH_HINT.search(claim):
            return None
        text = re.sub(r"\s+", " ", cv._fold_chars(claim).translate(_TH_CHARS)).strip().lower()
        text = re.sub(r"^(?:[-+*•]|\d+[.)])\s+", "", text)
        for clause in _TH_CLAUSE_END.split(text):
            m = _THRESHOLD.match(_TH_LEAD.sub("", clause).strip())
            if m is None or m.group("pct"):
                continue
            quantifier, noun = m.group("neg") or m.group("all"), m.group("noun")
            key = self._threshold_key(noun, quantifier in ("none", "none of them"))
            if key is None:
                continue
            after: List[str] = []
            for word in re.findall(r"[a-z_]+", m.group("after").split(".")[0])[:3]:
                if word in _TH_AFTER_STOP:
                    break
                after.append(word)
            words = re.findall(r"[a-z_]+", f"{m.group('m1') or ''} {m.group('m2') or ''} {' '.join(after)}")
            columns = self._threshold_columns(words, key)
            if not columns:
                continue
            digits = m.group("num").replace(",", "")
            scale = _TH_SCALES.get(m.group("scale") or "", 1.0)
            bound = float(digits) * scale
            tol = 0.5 * 10 ** -(len(digits.split(".", 1)[1]) if "." in digits else 0) * scale
            # "no X exceeds N" says every group is at most N; "all X exceed N" says at least.
            at_most = bool(m.group("up")) == bool(m.group("neg"))
            raw = m.group("num") + (f" {m.group('scale')}" if m.group("scale") else "")
            said = (f"{quantifier if noun is None else ('no' if m.group('neg') else 'every') + ' ' + noun} "
                    f"is {'above' if m.group('up') else 'below'} {raw}")
            rows = [i for i, r in enumerate(self.rows) if not _null(r.get(key))
                    and str(r.get(key)).strip().lower() not in _TOTAL_LABELS]
            verdicts = []
            for column in columns:
                unit = _column_unit(column)
                values = [(i, cv._as_number(self.rows[i].get(column))) for i in rows]
                broken = [(i, v) for i, v in values if v is not None
                          and (v * unit > bound + tol if at_most else v * unit < bound - tol)]
                verdicts.append((broken, column))
            if any(b for b, _c in verdicts) != all(b for b, _c in verdicts):
                continue                                   # the candidate columns disagree
            if verdicts[0][0]:
                broken, column = verdicts[0]
                i, worst = (max if at_most else min)(broken, key=lambda p: p[1])
                more = f"; {len(broken)} groups break it" if len(broken) > 1 else ""
                return cv.ClaimFinding(
                    claim=claim, verdict=cv.Verdict.CONTRADICTED, check="threshold_contradicted",
                    detail=(f"says {said}, but {str(self.rows[i].get(key)).strip()} is at "
                            f"{_fmt(worst, False)} by {column} in the computed result{more}"),
                    evidence_ref=f"execution_output.{column}")
            # The bound is a limit the claim sets, not a figure of the result:
            # judged here, it is no "invented number" (as a ratio's multiple is not).
            for f in cv._NUMBER.finditer(claim):
                value = cv._parse_number(f.group(0))
                if value is not None and abs(value - bound) <= 1e-9 * max(1.0, abs(bound)):
                    self.judged_spans.setdefault(claim, []).append(f.span())
                    break
            return self._threshold_over_all_rows(claim, key, columns, bound, tol, at_most, said)
        return None

    def _threshold_key(self, noun: Optional[str], bare: bool) -> Optional[str]:
        """The key column the claim's groups are: one row per value, no subgroups."""
        keys = [c for c in self.text if self._repeats(c) == 0 and not self._subgroups(c)]
        if not bare:
            if noun is None:
                return None
            keys = [c for c in keys if _names(noun, c) or noun.rstrip("s") in ("group", "categorie", "category")
                    and _names("category", c)]
        return keys[0] if len(keys) == 1 else None

    def _threshold_columns(self, words: Sequence[str], key: str) -> List[str]:
        """The numeric columns every measure word names; all of them for none."""
        from avaloka.goal_analysis import _ordered_tokens

        said = [w for w in words if w not in _MEASURE_FILLER and len(w) >= 3]
        numeric = [c for c in self.numeric if c != key]
        if not said:
            return numeric
        if any(_names(w, key) for w in said):
            return []                                       # "the largest category": the groups, not a measure
        named = [c for c in numeric
                 if all(any(_alike(_TH_SYNONYMS.get(w, w), t) for t in _ordered_tokens(str(c))) for w in said)]
        return named or _lexicon(said, numeric)

    def _threshold_over_all_rows(self, claim: str, key: str, columns: Sequence[str], bound: float, tol: float,
                                 at_most: bool, said: str) -> Optional[cv.ClaimFinding]:
        """Held on the result's rows: on all the groups of the data, when they can be recomputed."""
        filters = cv.code_filters(self.code) if self.used is not None and self.used.frames else []
        recomputed = None
        if self.used is not None and self.used.frames \
                and not any(op not in _APPLICABLE_OPS or isinstance(v, (str, tuple)) for _c, op, v in filters):
            for column in columns:
                try:
                    found = self._recompute(key, column, filters)
                except Exception:  # noqa: BLE001 - a measure that cannot be recomputed checks nothing
                    logger.debug("[claim-verifier] could not recompute a threshold's measure", exc_info=True)
                    found = None
                if found is not None:
                    recomputed = (column, found)
                    break
        if recomputed is not None:
            column, (totals, how, kept) = recomputed
            totals = totals[[not _null(k) and str(k).strip().lower() not in _TOTAL_LABELS for k in totals.index]]
            unit = _column_unit(column)
            broken = totals[(totals * unit > bound + tol) if at_most else (totals * unit < bound - tol)]
            if len(broken):
                label = str((broken.idxmax() if at_most else broken.idxmin()))
                worst = float(broken[label])
                shown = sum(1 for r in self.rows if not _null(r.get(key)))
                return cv.ClaimFinding(
                    claim=claim, verdict=cv.Verdict.CONTRADICTED, check="threshold_of_subset",
                    detail=(f"says {said}, which holds only for the {shown} {key} values the analysis kept; "
                            f"over all rows{kept} {label} has {_fmt(worst, False)} ({how})"),
                    evidence_ref=f"data.{key}")
            self.resolved.add(claim)
            return None
        cut = self.cut
        if cut is None:
            self.resolved.add(claim)                        # nothing says the result is a part of the groups
            return None
        kept = (f"the {'top' if cut['top'] else 'bottom'} {cut['n']} by {cut['by']}" if cut["by"]
                else f"the first {cut['n']} the code kept")
        self.unconfirmed[claim] = (f"Not checked over all rows: {said} among the {len(self.rows)} rows the result "
                                   f"keeps ({kept}); whether it does over all rows could not be checked.")
        return None

    # -- one claim ----------------------------------------------------------------
    def check(self, claim: str, prior: Optional[str] = None) -> Optional[cv.ClaimFinding]:
        """*claim*'s finding; *prior* is the claim just before it, whose entity
        a claim starting "It" is about."""
        causal = cv.check_causal_overreach(claim, dict(self.flat))
        return (self.trend(claim, causal) or self.superlative(claim) or self.threshold(claim)
                or self.ratio(claim, prior)
                or self.carried_figure(claim, prior) or self.range_over_set(claim))
