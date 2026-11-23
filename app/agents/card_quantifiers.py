"""Quantifier words on upload cards, checked against the computed profile.

The insight and profiling models write the words on an upload card ("Airport
fees are almost always zero", "smokers have a markedly higher stroke rate").
The numbers that decide whether those words are true were already computed at
upload: distinct counts, zero and missing shares, value shares, per-group
rates. This module reads a sentence's quantifier, works out which column and
which share it is about, and compares the word with the share. No model is
asked; the same sentence and the same data always give the same answer.

A sentence is withheld (the caller shows the data-written text instead) when
its quantifier disagrees with the computed share, and also when it carries a
quantifier that cannot be checked: the share it is about is not one the profile
computes, or its condition ("when the payment method is credit card") names no
value that is in the data. The model writes words; it never gets to state a
fact the profile cannot back.

Thresholds (a share is of the rows -- or, on a sum chart, of the total -- the
claim is about, missing cells left out):

=====================================  ==========================================
word                                    the computed share must be
=====================================  ==========================================
always, invariably                      100%
constant, a single value                exactly 1 distinct value
almost/nearly constant, the same for    top value >= 95% of rows
  virtually all
almost/nearly/virtually always / all    >= 95%
vast / overwhelming / great majority    >= 80%
the / a majority, majority of           >  50%
frequently, often, commonly             >= 25%
rarely, seldom, almost never            <= 10%
unique (with a share word)              share of distinct values seen once
markedly / substantially / ... higher   larger value >= 1.25x the smaller, and
  or lower                              for rates >= 1 percentage point apart
a handful / a few / a small number /     the top k items (k = 5 for handful/few,
  top N ... account for the bulk /        N for "top N") hold >= 50% of rows (or of
  most / majority, dominate               the chart's total)
  ... a disproportionate share            top-k share >= max(5x the even share k/n, 20%)
most ... seen only once / unique /      singleton share > 50%: of distinct values
  their own                               when "most" refers to the values, of rows
                                          when it refers to the records
steady / consistent increase / decline  >= 4 periods and Spearman rho >= 0.8 (<= -0.8
  over time                               for a decline) between period and value
no other X / only X appears             X has exactly 1 distinct value
A associated with / correlates with B   |Pearson r| >= 0.1, and the sign the sentence
  (two numeric columns)                   states (higher..lower = negative)
X makes up a large portion / large      the named group(s) hold >= 20% of the rows (or
  share / significant share / ...         of the chart's total)
A is consistently higher / lower        A beats every compared group in >= 80% of the
  than B                                  periods of the data's date column: days, or
                                          months when there are more than 31 days. A
                                          period where a compared group has no rows
                                          does not count as holding
A ... higher / lower ... than B, or     the groups named on each side are values of the
  A ..., while B ... higher               card's group column, and the side called higher
                                          is higher (every named one against every other)
the standard / typical / usual X of V   V is X's most common value (the mode)
mixed / inconsistent case(-ing) in X    X has >= 2 values that differ only in case
=====================================  ==========================================

"Most ... seen only once" is not checked on a sample of the file: a value
seen once in a sample may repeat in the rows that were not read (rweb16:
65.1% of the distinct Instructions in a 4,018-row sample, 29.7% in the
12,190-row file), so on a sample it is withheld as uncheckable.

A comparison that names none of the card's group values ("credit cards" on a
payment_type card whose values are the codes 1-4) is withheld as unmappable,
like a condition that names no value in the data. "One vendor ... than the
other" names no group and is not a claim about a particular one.

"most often", "more frequently" are comparisons, not shares, and are left to
the comparison checks.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ALWAYS_SHARE = 1.0
ALMOST_ALWAYS_MIN = 0.95
NEAR_CONSTANT_MIN = 0.95
VAST_MAJORITY_MIN = 0.80
MAJORITY_MIN = 0.50          # strictly more than half
FREQUENT_MIN = 0.25
RARE_MAX = 0.10
MARKED_RATIO_MIN = 1.25
MARKED_RATE_POINTS_MIN = 1.0
FEW_GROUPS = 5               # "a handful / a few X" = the top 5 groups
CONCENTRATION_MIN = 0.50     # bulk / most / majority / dominate
DISPROPORTION_MULTIPLE = 5.0
DISPROPORTION_MIN = 0.20
SINGLETON_MOST_MIN = 0.50    # "most are seen only once": strictly more than half
TREND_MIN_PERIODS = 4
TREND_RHO_MIN = 0.8
ASSOCIATION_R_MIN = 0.1
PORTION_MIN = 0.20           # "a large portion / large share / significant share"
CONSISTENT_MIN = 0.80        # "consistently higher": share of periods where it holds
CONSISTENT_DAYS_MAX = 31     # periods are days up to this many distinct days, else months
#: In the reason a claim is withheld for being checked on a sample only.
SAMPLE_ONLY_REASON = "only a sample of"

# Longest phrases first: "almost always" must not be read as "always".
_SHARE_RE = re.compile(
    r"(?<!\bmost )(?<!\bmore )(?<!\bless )(?<!\bleast )"
    r"\b(?:(?P<almost_always>(?:almost|nearly|virtually|practically)\s+(?:always|all|entirely))"
    r"|(?P<vast_majority>(?:vast|overwhelming|great)\s+majority)"
    r"|(?P<rare>rarely|seldom|almost\s+never|hardly\s+ever)"
    r"|(?P<majority>(?:the|a)\s+majority|majority\s+of)"
    r"|(?P<always>always|invariably)"
    r"|(?P<frequent>frequently|often|commonly))\b",
    re.IGNORECASE,
)
_SHARE_RULES = {
    "always": ("100%", lambda s: s >= ALWAYS_SHARE - 1e-12),
    "almost_always": (">= 95%", lambda s: s >= ALMOST_ALWAYS_MIN),
    "vast_majority": (">= 80%", lambda s: s >= VAST_MAJORITY_MIN),
    "majority": ("> 50%", lambda s: s > MAJORITY_MIN),
    "frequent": (">= 25%", lambda s: s >= FREQUENT_MIN),
    "rare": ("<= 10%", lambda s: s <= RARE_MAX),
}
_CONSTANT_RE = re.compile(
    r"\b(?P<near>(?:almost|nearly|virtually|practically|mostly|largely)\s+)?"
    r"(?:constant|(?:a|one)\s+single\s+value)\b(?P<generic>\s+(?:columns?|fields?|features?))?",
    re.IGNORECASE,
)
# "stays constant across age groups" is a flat trend, not a constant column.
_CONSTANT_TREND_RE = re.compile(
    r"^\W*(?:over|across|through)\s+(?:time|the\s+years?|years|months|periods?|groups|"
    r"categories|bins|bands|ages?|age\s+groups|segments)\b",
    re.IGNORECASE,
)
# "No constant columns", "without constant columns", "constant columns: none".
_NONE_BEFORE_RE = re.compile(r"\b(?:no|without|zero|not\s+any)\s+(?:\w+\s+)?$", re.IGNORECASE)
_NONE_AFTER_RE = re.compile(r"^\W*(?:\w+\W+)?(?:none|not\s+(?:present|found|detected))\b", re.IGNORECASE)
_MARKED_RE = re.compile(
    r"\b(?P<adv>markedly|substantially|significantly|considerably|dramatically|sharply|much|far|"
    r"noticeably|notably|strikingly|vastly|meaningfully)\s+(?:a\s+)?"
    r"(?P<dir>higher|greater|more|larger|bigger|elevated|increased|lower|less|fewer|smaller|reduced)\b",
    re.IGNORECASE,
)
_HIGH_DIRS = {"higher", "greater", "more", "larger", "bigger", "elevated", "increased"}
_COMPARATOR_RE = re.compile(r"\b(?:than|compared\s+(?:with|to)|versus|vs\.?|relative\s+to)\b",
                            re.IGNORECASE)
_OTHERS_RE = re.compile(r"\b(?:others?|the\s+rest|remaining|next\s+highest|next\s+largest|"
                        r"runner-?up|any\s+other)\b", re.IGNORECASE)
_SOME_RE = re.compile(r"\b(?:certain|some|a\s+few|several|specific|top|leading|handful)\b",
                      re.IGNORECASE)
_FEW_RE = re.compile(r"\b(?:a\s+(?:(?:small\s+)?handful|few|small\s+number|small\s+set)|few\s+specific)\b",
                     re.IGNORECASE)
_ONE_RE = re.compile(r"\b(?:one|a\s+single)\s+(?:specific\s+|single\s+)?[a-z]+\b", re.IGNORECASE)
_MEAN_RE = re.compile(r"\b(?:mean|average)\b", re.IGNORECASE)
_MEDIAN_RE = re.compile(r"\bmedian\b", re.IGNORECASE)
_CONDITION_RE = re.compile(r"\b(?:when|whenever|where|if|among)\b", re.IGNORECASE)
_LEADING_CONDITION_RE = re.compile(r"^\W*(?:when|whenever|where|if|among)\b(?P<cond>[^,;]+)[,;]",
                                   re.IGNORECASE)
_CLAUSE_SPLIT_RE = re.compile(
    r"[;,:]|\b(?:while|whereas|but|although|though|even\s+though|despite|suggesting|indicating|"
    r"which|so)\b", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_PREDICATES = [
    ("missing", re.compile(r"\b(?:missing|null|nulls|empty|blank|n/a|nan)\b", re.IGNORECASE)),
    ("unique", re.compile(r"\b(?:unique|distinct|one-?offs?)\b", re.IGNORECASE)),
    ("zero", re.compile(r"(?<![\w.])(?:zero|zeros|zeroes|0|0\.0+|\$0)(?!\w|\.\d)", re.IGNORECASE)),
    ("same", re.compile(r"\bthe\s+same\b", re.IGNORECASE)),
]
_QUOTED_RE = re.compile(r"['\"‘’“”`]([^'\"‘’“”`]{1,60})"
                        r"['\"‘’“”`]")
_NEGATION_WORDS = {"non", "not", "no", "without", "never"}
_MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}
_YES = {"1", "yes", "true", "y"}
_NO = {"0", "no", "false", "n"}
_GENERIC_NAME_WORDS = {"is", "has", "flag", "ind", "yn", "bool", "id", "the", "of"}


# ---------------------------------------------------------------- text helpers

def _words(text: Any) -> List[str]:
    # camelCase and snake_case names read as words ("VendorID" -> vendor id).
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    return re.findall(r"[a-z0-9]+", t.lower())


def _stem(word: str) -> str:
    """A crude stem, the same on both sides: smokes/smoked/smokers/smoking ->
    smok, former/formerly -> form."""
    if word.endswith("ly") and len(word) >= 6:
        word = word[:-2]
    for suf in ("ers", "ing", "ed", "es", "er", "s", "e"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def _stems(text: Any) -> List[str]:
    return [_stem(w) for w in _words(text)]


def _same_word(a: str, b: str) -> bool:
    """Stems that name the same thing: equal, or a long shared start
    (hypertension / hypertensive)."""
    if a == b:
        return True
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= 5 and n >= 0.75 * min(len(a), len(b))


def _has_word(hay: Sequence[str], w: str) -> bool:
    return any(_same_word(h, w) for h in hay)


def _contains_run(hay: Sequence[str], needle: Sequence[str]) -> bool:
    n = len(needle)
    return n > 0 and any(all(_same_word(h, w) for h, w in zip(hay[i:i + n], needle))
                         for i in range(len(hay) - n + 1))


def _label(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and v.strip().lower() in _MISSING_TOKENS


def _to_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _group_label(v: Any) -> str:
    """A group's label as the card shows it: 2.0 and "2" are the same vendor."""
    f = _to_float(v)
    return _label(f) if f is not None and not isinstance(v, bool) else _label(v)


# ---------------------------------------------------------------- the data side

class ColumnFacts:
    """Shares a claim can be checked against: from the rows when the column is
    in them, else from the column statistics table (column_stats rows)."""

    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None,
                 column_stats: Optional[Iterable[Dict[str, Any]]] = None,
                 sample_of: Optional[int] = None):
        self.rows = rows or []
        #: Rows in the whole file when *rows* are a sample of it, else None.
        self.sample_of = sample_of if (sample_of and self.rows and sample_of > len(self.rows)) else None
        self.stats = {str(r.get("name")): r for r in (column_stats or []) if r.get("name") is not None}
        row_cols = list(self.rows[0].keys()) if self.rows else []
        self.columns = list(dict.fromkeys([str(c) for c in row_cols] + list(self.stats)))
        self._row_cols = {str(c): c for c in row_cols}
        self._cache: Dict[Tuple[str, Any], Any] = {}

    def _present(self, col: str, subset: Optional[Tuple[str, str]] = None) -> Optional[List[Any]]:
        key = (col, subset)
        if key in self._cache:
            return self._cache[key]
        if col not in self._row_cols or (subset and subset[0] not in self._row_cols):
            return None
        raw = self._row_cols[col]
        rows = self.rows
        if subset:
            scol, sval = self._row_cols[subset[0]], subset[1]
            rows = [r for r in rows if not _is_missing(r.get(scol)) and _label(r.get(scol)).lower() == sval.lower()]
        out = [r.get(raw) for r in rows if not _is_missing(r.get(raw))]
        self._cache[key] = out
        return out

    def _binary01(self, col: str) -> bool:
        st = self.stats.get(col) or {}
        return st.get("n_unique") == 2 and st.get("min") == 0 and st.get("max") == 1

    def values(self, col: str) -> List[str]:
        """Distinct value labels (from the rows, else the stats' top values)."""
        vals = self._present(col)
        if vals is not None:
            return list(dict.fromkeys(_label(v) for v in vals))
        if self._binary01(col):
            return ["0", "1"]
        st = self.stats.get(col) or {}
        return [_label(t.get("value")) for t in st.get("top_values") or [] if t.get("value") is not None]

    def n_unique(self, col: str) -> Optional[int]:
        vals = self._present(col)
        if vals is not None:
            return len({_label(v) for v in vals})
        n = (self.stats.get(col) or {}).get("n_unique")
        return int(n) if n is not None else None

    def top_share(self, col: str) -> Optional[float]:
        vals = self._present(col)
        if vals is not None:
            return (Counter(_label(v) for v in vals).most_common(1)[0][1] / len(vals)) if vals else None
        if self.n_unique(col) == 1:
            return 1.0
        top = (self.stats.get(col) or {}).get("top_values") or []
        return (float(top[0]["pct"]) / 100.0) if top and top[0].get("pct") is not None else None

    def zero_share(self, col: str, subset: Optional[Tuple[str, str]] = None) -> Optional[float]:
        vals = self._present(col, subset)
        if vals is not None:
            nums = [_to_float(v) for v in vals]
            nums = [x for x in nums if x is not None]
            return (sum(1 for x in nums if x == 0) / len(nums)) if nums else None
        if subset:
            return None
        st = self.stats.get(col) or {}
        lo, hi = st.get("min"), st.get("max")
        ranged = isinstance(lo, (int, float)) and isinstance(hi, (int, float))
        if ranged:
            if lo == 0 and hi == 0:
                return 1.0
            if lo > 0 or hi < 0:
                return 0.0
        if st.get("zero_pct") is not None:
            share = float(st["zero_pct"]) / 100.0
            # Rounded to 100.0 while min/max show a nonzero value: not "always".
            return min(share, 1.0 - 1e-9) if ranged else share
        return self.value_share(col, "0")

    def missing_share(self, col: str) -> Optional[float]:
        if col in self._row_cols:
            raw = self._row_cols[col]
            return sum(1 for r in self.rows if _is_missing(r.get(raw))) / len(self.rows)
        mp = (self.stats.get(col) or {}).get("missing_pct")
        return float(mp) / 100.0 if mp is not None else None

    def value_share(self, col: str, value: str, subset: Optional[Tuple[str, str]] = None) -> Optional[float]:
        vals = self._present(col, subset)
        if vals is not None:
            return (sum(1 for v in vals if _label(v).lower() == value.lower()) / len(vals)) if vals else None
        if subset:
            return None
        st = self.stats.get(col) or {}
        if self._binary01(col) and isinstance(st.get("mean"), (int, float)) and value in ("0", "1"):
            return float(st["mean"]) if value == "1" else 1.0 - float(st["mean"])
        for t in st.get("top_values") or []:
            if _label(t.get("value")).lower() == value.lower() and t.get("pct") is not None:
                return float(t["pct"]) / 100.0
        return None

    def topk_share(self, col: str, k: int) -> Tuple[Optional[float], Optional[int]]:
        """(share of present rows held by the k most common values, distinct count)."""
        vals = self._present(col)
        if vals is not None:
            if not vals:
                return None, None
            counts = Counter(_label(v) for v in vals)
            return sum(c for _, c in counts.most_common(k)) / len(vals), len(counts)
        st = self.stats.get(col) or {}
        top = st.get("top_values") or []
        if len(top) >= min(k, st.get("n_unique") or k) and all(t.get("pct") is not None for t in top[:k]):
            return sum(float(t["pct"]) for t in top[:k]) / 100.0, st.get("n_unique")
        return None, None

    def singleton_row_share(self, col: str) -> Optional[float]:
        """Share of rows whose value occurs only once in the column."""
        vals = self._present(col)
        if vals is not None:
            counts = Counter(_label(v) for v in vals)
            return (sum(1 for c in counts.values() if c == 1) / len(vals)) if vals else None
        st = self.stats.get(col) or {}
        n_rows = st.get("n_present")
        if st.get("n_singletons") is not None and n_rows:
            return float(st["n_singletons"]) / float(n_rows)
        return None

    def is_measure(self, col: str) -> bool:
        """A numeric column with more than 10 distinct values (not a code or a flag)."""
        role = (self.stats.get(col) or {}).get("role")
        if role and role != "measure":
            return False
        return self.is_numeric(col) and (self.n_unique(col) or 0) > 10

    def is_numeric(self, col: str) -> bool:
        st = self.stats.get(col) or {}
        if st.get("dtype"):
            return st.get("dtype") == "numeric"
        vals = self._present(col)
        return bool(vals) and all(_to_float(v) is not None for v in vals[:200])

    def pearson(self, a: str, b: str) -> Optional[float]:
        if a not in self._row_cols or b not in self._row_cols:
            return None
        ra, rb = self._row_cols[a], self._row_cols[b]
        pairs = [(_to_float(r.get(ra)), _to_float(r.get(rb))) for r in self.rows]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if len(pairs) < 3:
            return None
        mx = sum(x for x, _ in pairs) / len(pairs)
        my = sum(y for _, y in pairs) / len(pairs)
        sxy = sum((x - mx) * (y - my) for x, y in pairs)
        sx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
        sy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
        return sxy / (sx * sy) if sx and sy else None

    def time_series(self, time_col: Optional[str], measure: Optional[str], aggregate: str,
                    daily: bool = False) -> Optional[List[Tuple[str, float]]]:
        """(period, value) in time order from the rows: months, or days when the
        sentence says daily or the data spans fewer than TREND_MIN_PERIODS months."""
        if not time_col or time_col not in self._row_cols:
            return None
        tcol = self._row_cols[time_col]
        mcol = self._row_cols.get(measure) if measure else None
        if measure and mcol is None:
            return None
        def build(width: int) -> Dict[str, List[float]]:
            out: Dict[str, List[float]] = {}
            for r in self.rows:
                d = _DATE_KEY_RE.search(str(r.get(tcol)))
                if not d:
                    continue
                key = d.group(0)[:width]
                v = 1.0 if mcol is None else _to_float(r.get(mcol))
                if v is None:
                    continue
                out.setdefault(key, []).append(v)
            return out
        agg = (aggregate or "count").lower()
        if mcol is not None and agg not in _SERIES_AGGREGATES:
            # A row share or a name nothing computes: no series to check a
            # trend against, rather than a sum the card never showed.
            return None
        groups = build(10 if daily else 7)
        if not daily and len(groups) < TREND_MIN_PERIODS:
            groups = build(10)
        def reduce(vs: List[float]) -> float:
            if mcol is None or agg == "count":
                return float(len(vs))
            if agg in ("mean", "average", "avg"):
                return sum(vs) / len(vs)
            if agg == "median":
                return statistics.median(vs)
            if agg == "min":
                return min(vs)
            if agg == "max":
                return max(vs)
            return sum(vs)
        return [(k, reduce(groups[k])) for k in sorted(groups)]

    def mode(self, col: str) -> Optional[Tuple[str, float]]:
        """(the most common value's label, its share of the present rows)."""
        vals = self._present(col)
        if vals is not None:
            if not vals:
                return None
            label, n = Counter(_label(v) for v in vals).most_common(1)[0]
            return label, n / len(vals)
        top = (self.stats.get(col) or {}).get("top_values") or []
        if top and top[0].get("value") is not None and top[0].get("pct") is not None:
            return _label(top[0]["value"]), float(top[0]["pct"]) / 100.0
        return None

    def group_series(self, time_col: Optional[str], group_col: Optional[str], measure: Optional[str],
                     aggregate: str) -> Optional[List[Tuple[str, Dict[str, float]]]]:
        """[(period, {group: value})] in time order: days, or months when the
        rows span more than CONSISTENT_DAYS_MAX distinct days."""
        if not time_col or not group_col or time_col not in self._row_cols or group_col not in self._row_cols:
            return None
        tcol, gcol = self._row_cols[time_col], self._row_cols[group_col]
        mcol = self._row_cols.get(measure) if measure else None
        if measure and mcol is None:
            return None
        cells: List[Tuple[str, str, float]] = []
        for r in self.rows:
            d = _DATE_KEY_RE.search(str(r.get(tcol)))
            g = r.get(gcol)
            v = 1.0 if mcol is None else _to_float(r.get(mcol))
            if d and v is not None and not _is_missing(g):
                cells.append((d.group(0), _group_label(g), v))
        if not cells:
            return None
        width = 10 if len({c[0] for c in cells}) <= CONSISTENT_DAYS_MAX else 7
        acc: Dict[str, Dict[str, List[float]]] = {}
        for day, g, v in cells:
            acc.setdefault(day[:width], {}).setdefault(g, []).append(v)
        agg = (aggregate or "count").lower()

        def reduce(vs: List[float]) -> float:
            if mcol is None or agg == "count":
                return float(len(vs))
            if agg in ("mean", "average", "avg"):
                return sum(vs) / len(vs)
            return sum(vs)
        return [(k, {g: reduce(vs) for g, vs in acc[k].items()}) for k in sorted(acc)]

    def singleton_share(self, col: str) -> Optional[float]:
        """Share of the distinct values that occur exactly once."""
        vals = self._present(col)
        if vals is not None:
            counts = Counter(_label(v) for v in vals)
            return (sum(1 for c in counts.values() if c == 1) / len(counts)) if counts else None
        st = self.stats.get(col) or {}
        if st.get("n_singletons") is not None and st.get("n_unique"):
            return float(st["n_singletons"]) / float(st["n_unique"])
        return None


#: The aggregates time_series computes for a measure (a card's y encoding).
_SERIES_AGGREGATES = frozenset({"count", "sum", "total", "mean", "average", "avg", "median", "min", "max"})


def columns_named(text: str, columns: Iterable[str]) -> List[str]:
    """Columns a phrase names: every word of the name in order ("Airport fees"
    names Airport_fee, "tip amounts" names tip_amount), or the first word of
    a name no other column starts with ("payment method" -> payment_type)."""
    hay = _stems(text)
    cols = [c for c in columns if _words(c)]
    found: List[str] = []
    for c in sorted(cols, key=lambda c: -len(_words(c))):
        st = [_stem(w) for w in _words(c) if w not in _GENERIC_NAME_WORDS] or [_stem(w) for w in _words(c)]
        if _contains_run(hay, st) and not any(_contains_run([_stem(w) for w in _words(f)], st) for f in found):
            found.append(c)
    first: Dict[str, List[str]] = {}
    for c in cols:
        first.setdefault(_stem(_words(c)[0]), []).append(c)
    for w, names in first.items():
        if len(names) == 1 and len(w) >= 3 and w in hay and names[0] not in found:
            found.append(names[0])
    return found


def _binary(values: List[str]) -> bool:
    low = {v.lower() for v in values}
    return len(low) == 2 and low <= (_YES | _NO)


def _yes_no_value(text: str, col: str, values: List[str]) -> Optional[str]:
    """'with hypertension' -> the yes value, 'without hypertension' or
    'non-hypertensive' -> the no value."""
    words = [_stem(w) for w in _words(col) if w not in _GENERIC_NAME_WORDS]
    hay = _words(text)
    stems = [_stem(w) for w in hay]
    if not words:
        return None
    for i, st in enumerate(stems):
        if not _same_word(st, words[0]) and not any(_same_word(st, w) for w in words):
            continue
        # Every word of the name near here, in any order ("settled their
        # debt" names debt_settlement_flag).
        window = stems[max(0, i - 3):i + len(words) + 3]
        if not all(_has_word(window, w) for w in words):
            continue
        negated = bool(_NEGATION_WORDS & set(hay[max(0, i - 4):i]))
        pool = _NO if negated else _YES
        for v in values:
            if v.lower() in pool:
                return v
    return None


def resolve_values(text: str, col: str, values: List[str]) -> List[str]:
    """Every value of `col` a phrase names, most specific first."""
    if _binary(values):
        yn = _yes_no_value(text, col, values)
        if yn is not None:
            return [yn]
    hay = _stems(text)
    quoted = {q.strip().lower() for q in _QUOTED_RE.findall(text)}
    found: List[Tuple[int, str]] = []
    for v in values:
        if v.strip().lower() in quoted:
            found.append((99, v))
            continue
        st = _stems(v)
        if not st or (len(st) == 1 and len(st[0]) < 2 and not st[0].isdigit()):
            continue
        if all(_has_word(hay, s) for s in st):
            found.append((len(st), v))
    found.sort(key=lambda t: -t[0])
    # A value whose words are all inside a longer named value is that value.
    out: List[str] = []
    for n, v in found:
        if not any(set(_stems(v)) < set(_stems(o)) for o in out):
            out.append(v)
    return out


def resolve_value(text: str, col: str, values: List[str]) -> Optional[str]:
    """The one value of `col` a phrase names, or None (none, or ambiguous)."""
    vals = resolve_values(text, col, values)
    return vals[0] if len(vals) == 1 else None


# ---------------------------------------------------------------- the checks

def _clause_around(sentence: str, start: int, end: int) -> str:
    before = sentence[:start]
    cut = 0
    for m in _CLAUSE_SPLIT_RE.finditer(before):
        cut = m.end()
    after = sentence[end:]
    m = _CLAUSE_SPLIT_RE.search(after)
    return sentence[cut:start] + sentence[start:end] + (after[: m.start()] if m else after)


def _predicate(clause: str, q_at: int, cols: List[str], facts: ColumnFacts,
               group_shares: Optional[Dict[str, float]], x_field: Optional[str]):
    """(kind, column, value): what the share is OF -- the zero / missing /
    unique / same-value word closest to the quantifier, or a value of a column."""
    best = None
    for kind, rx in _PREDICATES:
        for m in rx.finditer(clause):
            d = abs(m.start() - q_at)
            if best is None or d < best[0]:
                best = (d, kind)
    if best is not None:
        return best[1], None, None
    if group_shares:
        named = resolve_values(clause, x_field or "", list(group_shares))
        if named:
            return "group", None, named
        if _FEW_RE.search(clause):
            return "few", None, None
        if _ONE_RE.search(clause):
            return "one", None, None
    for c in cols:
        vals = facts.values(c)
        v = resolve_value(clause, c, [x for x in vals if len(x) >= 2 or x.isdigit()] if not _binary(vals)
                          else vals) if vals else None
        if v is None and vals:
            quoted = resolve_values(clause, c, vals)
            v = quoted[0] if len(quoted) == 1 else None
        if v is not None:
            return "value", c, v
    return None, None, None


def _share_conflict(sentence: str, m: "re.Match[str]", facts: ColumnFacts, default_cols: List[str],
                    group_shares: Optional[Dict[str, float]], x_field: Optional[str]) -> Optional[str]:
    kind = m.lastgroup or ""
    word = m.group(0)
    need, holds = _SHARE_RULES[kind]
    clause = _clause_around(sentence, m.start(), m.end())
    q_at = clause.lower().find(word.lower())
    cond_m = _CONDITION_RE.search(clause)
    main, cond = (clause[: cond_m.start()], clause[cond_m.end():]) if cond_m else (clause, "")
    if not cond:
        # "When paid by card, tips are frequently zero": the condition leads.
        lead = _LEADING_CONDITION_RE.match(sentence[: m.start()])
        if lead:
            cond = lead.group("cond")

    named = columns_named(main, facts.columns) or columns_named(sentence, facts.columns)
    cols = list(dict.fromkeys([c for c in default_cols if c in facts.columns] + named))
    subset: Optional[Tuple[str, str]] = None
    if cond:
        ccols = columns_named(cond, facts.columns)
        if not ccols:
            return f"it says {word!r} under a condition ({cond.strip()!r}) that names no column"
        val = ccol = None
        for c in ccols:
            val = resolve_value(cond, c, facts.values(c))
            if val is not None:
                ccol = c
                break
        if val is None:
            return (f"it says {word!r} under a condition ({cond.strip()!r}) that names no value "
                    f"of {ccols[0]} in the data")
        subset = (ccol, val)
        named = [c for c in named if c not in ccols] or named
        cols = [c for c in cols if c not in ccols]

    # The card's own columns first: "the vast majority of orders are not
    # returned" on a chart of `returned` is about returned, not order_id.
    pred, pcol, value = _predicate(main, q_at, cols, facts, None if subset else group_shares, x_field)
    if pred is None:
        return f"it says {word!r} but not of what (zero, missing, unique or a value), so it cannot be checked"
    scope = f" when {subset[0]} is {subset[1]}" if subset else ""
    if pred == "group":
        share = sum(group_shares[g] for g in value)
        what = " and ".join(value)
        return None if holds(share) else (
            f"it says {word!r} ({need}), but {what} make{'s' if len(value) == 1 else ''} up {_pct(share)}")
    if pred in ("few", "one"):
        top = sorted(group_shares.values(), reverse=True)[:FEW_GROUPS if pred == "few" else 1]
        share = sum(top)
        return None if holds(share) else (
            f"it says a few groups are {word!r} ({need}), but the top {len(top)} make up {_pct(share)}")
    if pred == "value":
        share = facts.value_share(pcol, str(value), subset)
        if share is None:
            return f"it says {word!r} but the share of rows where {pcol} is {value}{scope} is not computed"
        return None if holds(share) else (
            f"it says {word!r} ({need}), but {pcol} is {value} in {_pct(share)} of rows{scope}")
    targets = named or cols
    if not targets:
        return f"it says {word!r} but names no column the share can be computed for"
    for c in targets:
        if pred == "zero":
            share, what = facts.zero_share(c, subset), f"{c} is zero"
        elif pred == "missing":
            share, what = (None if subset else facts.missing_share(c)), f"{c} is missing"
        elif pred == "unique":
            share, what = (None if subset else facts.singleton_share(c)), f"the distinct {c} values occur once"
        else:
            share, what = (None if subset else facts.top_share(c)), f"{c} holds its most common value"
        if share is None and pred == "zero" and kind == "always" and not subset:
            st = facts.stats.get(c) or {}
            if st.get("min") is not None and st.get("min") != st.get("max"):
                return f"it says {word!r} zero, but {c} ranges from {st['min']:g} to {st['max']:g}"
        if share is None:
            return f"it says {word!r} but the share of rows where {what}{scope} is not computed"
        if not holds(share):
            return f"it says {word!r} ({need}), but {what} in {_pct(share)} of rows{scope}"
    return None


def _constant_conflict(sentence: str, m: "re.Match[str]", facts: ColumnFacts) -> Optional[str]:
    if _CONSTANT_TREND_RE.search(sentence[m.end():]):
        return None
    clause = _clause_around(sentence, m.start(), m.end())
    negated = bool(_NONE_BEFORE_RE.search(sentence[: m.start()])
                   or _NONE_AFTER_RE.search(sentence[m.end():]))
    constant_cols = [c for c in facts.columns if facts.n_unique(c) is not None and facts.n_unique(c) <= 1]
    cols = [] if m.group("generic") else columns_named(clause, facts.columns)
    if negated or not cols:
        # "No constant columns", "needs preprocessing of a constant column":
        # true when none / some column holds a single value.
        if negated and constant_cols:
            return f"it says no column is constant, but {constant_cols[0]} holds a single value"
        if not negated and not constant_cols:
            return "it calls a column constant, but no column holds a single value"
        return None
    for c in cols:
        if m.group("near"):
            share = facts.top_share(c)
            if share is None:
                return f"it calls {c} nearly constant, but its top-value share is not computed"
            if share < NEAR_CONSTANT_MIN:
                return f"it calls {c} nearly constant (>= 95% one value), but its top value is {_pct(share)}"
            continue
        n = facts.n_unique(c)
        if n is None:
            return f"it calls {c} constant, but its distinct count is not computed"
        if n > 1:
            return f"it calls {c} constant, but it has {n:,} distinct values"
    return None


def _groups(points: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    return [(str(p.get("x")), float(p["y"])) for p in points or []
            if isinstance(p.get("y"), (int, float)) and str(p.get("x")) != "Other"]


def _marked_conflict(sentence: str, m: "re.Match[str]", points: List[Dict[str, Any]],
                     x_field: Optional[str], is_rate: bool, overall: Optional[float],
                     summary: Optional[Dict[str, float]]) -> Optional[str]:
    word = f"{m.group('adv')} {m.group('dir')}"
    says_high = m.group("dir").lower() in _HIGH_DIRS
    comp = _COMPARATOR_RE.search(sentence, m.end())
    left = sentence[: comp.start()] if comp else sentence[: m.start()]
    if comp:
        rest = sentence[comp.end():]
        stop = re.search(r"[,;.](?!\d)", rest)
        right = rest[: stop.start()] if stop else rest
    else:
        right = ""

    # "the mean is significantly higher than the median" on a distribution.
    s = summary or {}
    if comp and isinstance(s.get("mean"), (int, float)) and isinstance(s.get("median"), (int, float)):
        pairs = {(True, False): ("mean", "median"), (False, True): ("median", "mean")}
        key = (bool(_MEAN_RE.search(left)) and not _MEDIAN_RE.search(left),
               bool(_MEDIAN_RE.search(left)) and not _MEAN_RE.search(left))
        if key in pairs and (_MEAN_RE.search(right) or _MEDIAN_RE.search(right)):
            a, b = pairs[key]
            return _magnitude(word, says_high, a, float(s[a]), b, float(s[b]), False)

    groups = _groups(points)
    if len(groups) < 2:
        return f"it says {word!r} but there are no computed groups to compare"
    values = {g: v for g, v in groups}
    labels = list(values)

    def pick(text: str) -> List[str]:
        if not text.strip():
            return []
        if x_field and _binary(labels):
            yn = _yes_no_value(text, x_field, labels)
            if yn is not None:
                return [yn]
        return resolve_values(text, x_field or "", labels)

    a_named = pick(left)
    clause = _clause_around(sentence, m.start(), m.end())
    some = re.search(r"\b(?:certain|some|several)\b", left, re.IGNORECASE)
    if a_named and some and not comp:
        a_named = []          # "some dishes, such as X, ...": the examples are not the claim
    if a_named and some and comp and _OTHERS_RE.search(right):
        a_named = []
    if not a_named and not says_high and _OTHERS_RE.search(clause):
        # "other countries represent much smaller opportunities": every group
        # but the largest, against the largest.
        ordered = sorted(groups, key=lambda g: -g[1])
        return _magnitude(word, says_high, ordered[1][0], ordered[1][1], ordered[0][0], ordered[0][1], is_rate)
    if not a_named:
        if _SOME_RE.search(left) or _SOME_RE.search(_clause_around(sentence, m.start(), m.end())):
            # "certain months have significantly higher activity": the most
            # extreme group against the other extreme.
            hi_g = max(groups, key=lambda g: g[1])
            lo_g = min(groups, key=lambda g: g[1])
            a, b = (hi_g, lo_g) if says_high else (lo_g, hi_g)
            return _magnitude(word, says_high, a[0], a[1], b[0], b[1], is_rate)
        return f"it says {word!r} but names none of the compared groups ({', '.join(labels[:6])})"
    if not comp and _OTHERS_RE.search(clause) and not pick(clause):
        # "dominated by Private; other categories are much smaller": the named
        # groups against every group the sentence does not name.
        others = [g for g in labels if g not in a_named]
        if not others:
            return None
        a_name = min(a_named, key=lambda g: values[g]) if not says_high else max(a_named, key=lambda g: values[g])
        b_name = max(others, key=lambda g: values[g]) if not says_high else min(others, key=lambda g: values[g])
        # The others are the smaller ones: compare named (big) vs others (small).
        return _magnitude(word, says_high, b_name, values[b_name], a_name, values[a_name], is_rate)
    # Several named ("A and B are much more frequent than others"): the claim
    # must hold for the weakest of them.
    a = (min(a_named, key=lambda g: values[g]) if says_high
         else max(a_named, key=lambda g: values[g]))
    if comp:
        b_named = [g for g in pick(right) if g not in a_named]
        if b_named:
            b = b_named[0]
        else:
            # Than "those without", "the next highest", "other countries": the
            # group closest to it, so the claim must hold against every other.
            others = [g for g in labels if g not in a_named]
            if not others:
                return f"it says {a} is {word} but every compared group is named on the same side"
            b = (max(others, key=lambda g: values[g]) if says_high
                 else min(others, key=lambda g: values[g]))
        return _magnitude(word, says_high, a, values[a], b, values[b], is_rate)
    if overall is not None:
        return _magnitude(word, says_high, a, values[a], "overall", float(overall), is_rate)
    others = sorted(values[g] for g in labels if g != a)
    mid = others[len(others) // 2]
    return _magnitude(word, says_high, a, values[a], "the typical other group", mid, is_rate)


def _magnitude(word: str, says_high: bool, a: str, va: float, b: str, vb: float,
               is_rate: bool) -> Optional[str]:
    if (says_high and va <= vb) or (not says_high and va >= vb):
        return f"it says {a} is {word} than {b}, but {a} is {va:g} and {b} is {vb:g}"
    hi, lo = max(va, vb), min(va, vb)
    ratio = (hi / lo) if lo > 0 else math.inf
    if ratio < MARKED_RATIO_MIN:
        return (f"it says {a} is {word} than {b} ({va:g} vs {vb:g}), but the larger is only "
                f"{ratio:.2f}x the smaller (needs >= {MARKED_RATIO_MIN:g}x)")
    if is_rate and hi - lo < MARKED_RATE_POINTS_MIN:
        return (f"it says {a} is {word} than {b} ({va:g}% vs {vb:g}%), but they are only "
                f"{hi - lo:.1f} points apart (needs >= {MARKED_RATE_POINTS_MIN:g})")
    return None


# ---------------------------------------------------------------- rc15 shapes

_CONCENTRATION_RE = re.compile(
    r"\b(?P<few>(?:a\s+(?:(?:small\s+)?handful|few|small\s+number|small\s+set)|few\s+specific|"
    r"(?:the\s+)?top\s+(?P<n>\d{1,2}|two|three|four|five|ten)))\b(?P<mid>[^.;]{0,120}?)"
    r"\b(?P<verb>dominat\w*|(?:the\s+)?bulk|(?:a\s+)?disproportionate(?:\s+share)?|"
    r"(?:account\w*|make\w*\s+up|represent\w*)\s+for\s+most|(?:the|a)\s+majority)\b",
    re.IGNORECASE,
)
_NUM_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}
_MOST_ONCE_RE = re.compile(
    r"\bmost(?:\s+of\s+them)?(?P<noun>(?:\s+[\w\-]+){0,3}?)\s+(?:are|is|have|has|appear\w*|occur\w*|get)?\s*"
    r"(?:(?:seen|used|listed)\s+)?(?:only\s+once|just\s+once|once|unique|(?:their|its)\s+own(?:\s+unique)?)\b",
    re.IGNORECASE,
)
_TREND_RE = re.compile(
    r"\b(?:(?:steady|steadily|consistent|consistently|continuous|continuously|monotonic|monotonically|"
    r"uninterrupted|constant|constantly)\s+(?P<a>increase|increasing|rise|rising|growth|growing|upward|climb\w*|"
    r"decrease|decreasing|decline|declining|fall|falling|drop|dropping|downward)"
    r"|(?P<b>increas\w*|ris\w*|grow\w*|climb\w*|decreas\w*|declin\w*|fall\w*|drop\w*)\s+"
    r"(?:steadily|consistently|continuously))\b",
    re.IGNORECASE,
)
_UP_RE = re.compile(r"^(?:increas|ris|grow|growth|upward|climb)", re.IGNORECASE)
_NO_OTHER_RE = re.compile(
    r"\b(?:no\s+other\s+(?P<a>[\w\- ]{2,30}?)\s+(?:is|are|appears?|exists?|is\s+listed|are\s+listed|"
    r"is\s+present|are\s+present|shows?\s+up)"
    r"|(?:the\s+)?only\s+(?P<b>[\w\-]{2,30})\s+(?:that\s+)?(?:appears?|present|listed|in\s+the\s+data))\b",
    re.IGNORECASE,
)
_ASSOC_RE = re.compile(r"\b(?:associated\s+with|correlat\w*\s+with|linked\s+(?:to|with)|tied\s+to|"
                       r"go(?:es)?\s+hand\s+in\s+hand\s+with)\b", re.IGNORECASE)
_NEG_DIR_RE = re.compile(r"\b(?:higher|more|larger|greater|increas\w*)\b[^.;]{0,60}\b(?:lower|less|fewer|smaller|"
                         r"decreas\w*)\b|\b(?:lower|less|fewer|smaller)\b[^.;]{0,60}\b(?:higher|more|larger|"
                         r"greater)\b|\b(?:inverse\w*|negative\w*)\b", re.IGNORECASE)
_POS_DIR_RE = re.compile(r"\b(?:higher|more|larger|greater)\b[^.;]{0,60}\b(?:higher|more|larger|greater)\b|"
                         r"\b(?:lower|less|fewer|smaller)\b[^.;]{0,60}\b(?:lower|less|fewer|smaller)\b|"
                         r"\bpositive\w*\b", re.IGNORECASE)
_DATE_KEY_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _concentration_conflict(sentence: str, m: "re.Match[str]", facts: ColumnFacts, default_cols: List[str],
                            group_shares: Optional[Dict[str, float]], x_field: Optional[str]) -> Optional[str]:
    """'A handful of URLs account for the bulk of the records' (top-k share)."""
    n = m.group("n")
    k = (int(n) if n.isdigit() else _NUM_WORDS[n.lower()]) if n else FEW_GROUPS
    verb = m.group("verb").lower()
    phrase = m.group("few") + m.group("mid")
    # The items are the card's groups (its x / colour field) unless the phrase names a column.
    cols = columns_named(phrase, facts.columns) or \
        [c for c in ([x_field] if x_field else []) + default_cols if c in facts.columns]
    # A distribution ("a few very large values dominate the mean") is not a
    # claim about items of a column: not this shape.
    cols = [c for c in cols if not facts.is_measure(c)]
    if not cols and not group_shares:
        return None
    share = n_groups = None
    col = None
    named = columns_named(phrase, facts.columns)
    if group_shares and not (named and named[0] != x_field):
        # What the card totals (rows, units, importance), group by group.
        top = sorted(group_shares.values(), reverse=True)
        share, n_groups, col = sum(top[:k]), len(top), x_field or "the chart's groups"
    for c in ([] if share is not None else cols):
        share, n_groups = facts.topk_share(c, k)
        top1 = facts.top_share(c)
        if share is not None and n_groups and top1 is not None and top1 * n_groups <= 1.15:
            # (1.15: top-value shares in the stats table are rounded to 0.1%.)
            # One row per value (a feature-importance table): row shares say
            # nothing about what dominates; the card's measure shares are needed.
            return (f"it says a few {c} values {verb}, but each holds one row and the card's "
                    f"shares of its measure are not computed")
        if share is not None:
            col = c
            break
    if share is None:
        return f"it says a few items {verb}, but no top-{k} share is computed for what it names"
    if "disproportionate" in verb:
        bar = min(max(DISPROPORTION_MULTIPLE * k / max(n_groups or k, k), DISPROPORTION_MIN), 1.0)
        if share < bar:
            return (f"it says {k} or fewer {col} values hold a disproportionate share, but the top {k} "
                    f"hold {_pct(share)} (needs >= {_pct(bar)})")
        return None
    if share < CONCENTRATION_MIN:
        return (f"it says a few {col} values {verb}, but the top {k} of {n_groups:,} hold "
                f"{_pct(share)} (needs >= 50%)")
    return None


def _most_once_conflict(sentence: str, m: "re.Match[str]", facts: ColumnFacts,
                        default_cols: List[str]) -> Optional[str]:
    """'most are seen only once' (distinct values) / 'most recipes have their
    own unique steps' (rows)."""
    noun = (m.group("noun") or "").strip()
    named_in_noun = columns_named(noun, facts.columns) if noun else []
    cols = named_in_noun or columns_named(sentence, facts.columns) or \
        [c for c in default_cols if c in facts.columns]
    if not cols:
        return "it says most values occur once, but names no column"
    # "most are ..." / "most texts" (the column's values) vs "most recipes" (the records).
    by_rows = bool(noun) and not named_in_noun and not re.search(
        r"\b(?:of\s+them|values?|texts?|entries|ones|items?|names?|titles?)\b", noun, re.IGNORECASE)
    if facts.sample_of and any(c in facts._row_cols for c in cols):
        return (f"it says most occur once, but {SAMPLE_ONLY_REASON} {len(facts.rows):,} of the file's "
                f"{facts.sample_of:,} rows was read, and a value seen once in a sample may repeat "
                f"in the rows that were not")
    for c in cols:
        share = facts.singleton_row_share(c) if by_rows else facts.singleton_share(c)
        unit = "rows" if by_rows else "distinct values"
        if share is None:
            return f"it says most {c} values occur once, but the share is not computed"
        if share <= SINGLETON_MOST_MIN:
            return f"it says most occur once, but only {_pct(share)} of {c} {unit} do (needs > 50%)"
    return None


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if not sx or not sy:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def _ordered_points(points: List[Dict[str, Any]]) -> Optional[List[Tuple[str, float]]]:
    """A card's groups in their natural order when every label is a number, a
    year or a band ("71.76-82"): age bands, years. None for categories."""
    keyed = []
    for p in points or []:
        y = p.get("y")
        mm = re.match(r"^\s*(?:age\s+)?(-?\d[\d,]*(?:\.\d+)?)", str(p.get("x")))
        if not isinstance(y, (int, float)) or not mm:
            return None
        keyed.append((float(mm.group(1).replace(",", "")), str(p.get("x")), float(y)))
    return [(lab, y) for _, lab, y in sorted(keyed)] if keyed else None


def _trend_conflict(sentence: str, m: "re.Match[str]", facts: ColumnFacts,
                    time_field: Optional[str], measure: Optional[str], aggregate: str,
                    points: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    word = m.group(0)
    up = bool(_UP_RE.match((m.group("a") or m.group("b") or "")))
    series = facts.time_series(time_field, measure, aggregate,
                               daily=bool(re.search(r"\b(?:daily|day|days)\b", sentence, re.IGNORECASE)))
    if not series:
        # "Stroke risk rises steadily with age" on age bands, "over the years" on years.
        series = _ordered_points(points or [])
    if not series or len(series) < TREND_MIN_PERIODS:
        return (f"it says {word!r}, but there are {len(series or [])} computed periods "
                f"(needs >= {TREND_MIN_PERIODS})")
    rho = _spearman(list(range(len(series))), [v for _, v in series])
    eps = 1e-9
    if rho is None or (up and rho < TREND_RHO_MIN - eps) or (not up and rho > -TREND_RHO_MIN + eps):
        return (f"it says {word!r}, but over {len(series)} periods ({series[0][0]}..{series[-1][0]}) "
                f"Spearman rho = {0.0 if rho is None else rho:.2f} (needs {'>=' if up else '<='} "
                f"{TREND_RHO_MIN if up else -TREND_RHO_MIN:g})")
    return None


def _no_other_conflict(sentence: str, m: "re.Match[str]", facts: ColumnFacts) -> Optional[str]:
    noun = (m.group("a") or m.group("b") or "").strip()
    cols = columns_named(noun, facts.columns)
    if not cols:
        return None          # "no other variable": not a column, not this shape
    c = cols[0]
    n = facts.n_unique(c)
    if n is None:
        return f"it says no other {c} appears, but its distinct count is not computed"
    if n != 1:
        return f"it says no other {c} appears, but {c} has {n:,} distinct values"
    return None


def _association_conflict(sentence: str, m: "re.Match[str]", facts: ColumnFacts,
                          default_cols: List[str], correlation: Optional[float]) -> Optional[str]:
    if correlation is not None and len(default_cols) == 2:
        numeric = list(default_cols)     # a scatter card: the claim is about its two axes
    else:
        numeric = [c for c in columns_named(sentence, facts.columns) if facts.is_measure(c)]
    if len(numeric) < 2:
        return None          # not two numeric measures (codes, 0/1 flags): not this shape
    a, b = numeric[0], numeric[1]
    r = correlation if (correlation is not None and set(default_cols) == {a, b}) else facts.pearson(a, b)
    if r is None:
        return f"it says {a} is associated with {b}, but no correlation is computed"
    if abs(r) < ASSOCIATION_R_MIN:
        return f"it says {a} is associated with {b}, but r = {r:.2f} (needs |r| >= {ASSOCIATION_R_MIN:g})"
    neg, pos = bool(_NEG_DIR_RE.search(sentence)), bool(_POS_DIR_RE.search(sentence))
    if neg and not pos and r > 0:
        return f"it says {a} and {b} move in opposite directions, but r = {r:.2f}"
    if pos and not neg and r < 0:
        return f"it says {a} and {b} move together, but r = {r:.2f}"
    return None


# ---------------------------------------------------------------- rc16 shapes

_PORTION_RE = re.compile(
    r"\b(?P<adj>large|significant|substantial|sizeable|sizable|considerable|major)\s+"
    r"(?:portion|share|part|proportion|fraction|chunk)\b", re.IGNORECASE)
CONSISTENT_RE = re.compile(
    r"\bconsistently\s+(?:[\w\-]+\s+){0,2}?(?P<dir>higher|greater|more|larger|bigger|lower|less|fewer|smaller)\b",
    re.IGNORECASE)
_BARE_CMP_RE = re.compile(r"\b(?P<dir>higher|greater|larger|bigger|more|lower|less|fewer|smaller)\b",
                          re.IGNORECASE)
_CONTRAST_RE = re.compile(r"\b(?:than|while|whereas|compared\s+(?:with|to)|versus|vs\.?)\b", re.IGNORECASE)
_GENERIC_SIDE_RE = re.compile(r"\b(?:one|the\s+other|another|some|certain|several|a\s+few)\b", re.IGNORECASE)
_STANDARD_RE = re.compile(
    r"\b(?P<adj>standard|typical|usual|normal|default)\s+(?P<noun>(?:[A-Za-z][\w\-]*\s+){0,3}?)"
    r"(?:of|is|was|at|=)\s+(?P<val>\$?\d[\d,]*(?:\.\d+)?|(?:a|an|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|zero)(?:\s+and\s+a\s+half)?\s+(?:dollars?|cents?|euros?|pounds?)|zero)\b", re.IGNORECASE)
_NUMBER_WORDS = {"zero": 0, "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_CASE_RE = re.compile(
    r"\b(?:mixed|inconsistent|varying|different)\s+(?:[\w\-]+\s+){0,2}?(?:case|casing|capitali[sz]ation)\b"
    r"|\bcase\s+(?:variants?|inconsistenc\w+)\b", re.IGNORECASE)


def _portion_conflict(sentence: str, m: "re.Match[str]", facts: ColumnFacts, default_cols: List[str],
                      group_shares: Optional[Dict[str, float]], x_field: Optional[str]) -> Optional[str]:
    """'Herzhafte Pfannkuchen ... accounting for a large portion of the collection' (3.1%)."""
    word = m.group(0)
    subject = sentence[: m.start()]
    if group_shares:
        named = resolve_values(subject, x_field or "", list(group_shares))
        if named:
            share = sum(group_shares[g] for g in named)
            what = " and ".join(named)
            return None if share >= PORTION_MIN else (
                f"it says {what} make{'s' if len(named) == 1 else ''} up a {word}, but "
                f"{'it is' if len(named) == 1 else 'they are'} {_pct(share)} (needs >= {_pct(PORTION_MIN)})")
    for c in [c for c in ([x_field] if x_field else []) + default_cols if c in facts.columns]:
        vals = [v for v in facts.values(c) if len(v) >= 2 or v.isdigit()]
        named = resolve_values(subject, c, vals)
        if not named:
            continue
        shares = [facts.value_share(c, v) for v in named]
        if any(x is None for x in shares):
            return f"it says {' and '.join(named)} make up a {word}, but that share is not computed"
        share = sum(shares)
        return None if share >= PORTION_MIN else (
            f"it says {' and '.join(named)} make{'s' if len(named) == 1 else ''} up a {word}, but "
            f"{c} is {' or '.join(named)} in {_pct(share)} of rows (needs >= {_pct(PORTION_MIN)})")
    return None               # no group or value named: not this shape


def _sides(sentence: str, m: "re.Match[str]") -> Tuple[str, str]:
    """(the side the comparative is said of, the other side): 'A ... higher
    than B' -> (A, B); 'A ..., while B show higher' -> (B, A)."""
    clause = _clause_around(sentence, m.start(), m.end())
    at = sentence.find(clause)
    than = re.search(r"\b(?:than|compared\s+(?:with|to)|versus|vs\.?)\b", sentence[m.end():], re.IGNORECASE)
    if than:
        rest = sentence[m.end() + than.end():]
        stop = re.search(r"[,;.](?!\d)|\b(?:while|whereas|but|suggesting|indicating|which)\b", rest)
        return sentence[: m.end()], rest[: stop.start()] if stop else rest
    return clause, sentence[:at] + sentence[at + len(clause):]


def _consistent_conflict(sentence: str, m: "re.Match[str]", facts: ColumnFacts, points: List[Dict[str, Any]],
                         x_field: Optional[str], period_field: Optional[str], measure: Optional[str],
                         aggregate: str) -> Optional[str]:
    """'One vendor consistently generates higher total fares than the other'."""
    word = f"consistently {m.group('dir').lower()}"
    says_high = m.group("dir").lower() in _HIGH_DIRS
    groups = [(_group_label(g), v) for g, v in _groups(points)]
    if len(groups) < 2:
        return f"it says {word!r} but there are no computed groups to compare"
    labels = [g for g, _ in groups]
    side_a, side_b = _sides(sentence, m)
    a_named = resolve_values(side_a, x_field or "", labels)
    b_named = [g for g in resolve_values(side_b, x_field or "", labels) if g not in a_named]
    if not a_named:
        ranked = sorted(groups, key=lambda g: -g[1] if says_high else g[1])
        a_named = [ranked[0][0]]
    b_named = b_named or [g for g in labels if g not in a_named]
    series = facts.group_series(period_field, x_field, measure, aggregate)
    if not series or len(series) < 2:
        return f"it says {word!r}, but there are no computed periods to compare the groups in"
    held = 0
    for _, vals in series:
        if all(g in vals for g in a_named + b_named):
            a_v = min(vals[g] for g in a_named) if says_high else max(vals[g] for g in a_named)
            b_v = max(vals[g] for g in b_named) if says_high else min(vals[g] for g in b_named)
            held += (a_v > b_v) if says_high else (a_v < b_v)
    share = held / len(series)
    if share < CONSISTENT_MIN:
        return (f"it says {' and '.join(a_named)} is {word} than {' and '.join(b_named)}, but that holds in "
                f"{held} of {len(series)} periods ({series[0][0]}..{series[-1][0]}; needs >= "
                f"{_pct(CONSISTENT_MIN)})")
    return None


def _comparison_conflict(sentence: str, m: "re.Match[str]", facts: ColumnFacts, points: List[Dict[str, Any]],
                         x_field: Optional[str]) -> Optional[str]:
    """'Trips paid with credit cards tend to have little or no tip, while cash
    payments show higher average tips' on a card of tip by payment_type."""
    if not x_field or not _CONTRAST_RE.search(sentence):
        return None
    if x_field not in columns_named(sentence, list(dict.fromkeys(facts.columns + [x_field]))):
        return None              # not a comparison of this card's groups
    groups = {_group_label(g): v for g, v in _groups(points)}
    if len(groups) < 2:
        return None
    labels = list(groups)
    says_high = m.group("dir").lower() in _HIGH_DIRS
    side_a, side_b = _sides(sentence, m)
    a_named = resolve_values(side_a, x_field, labels)
    b_named = [g for g in resolve_values(side_b, x_field, labels) if g not in a_named]
    if not a_named and not b_named:
        if _GENERIC_SIDE_RE.search(sentence):
            return None          # "one vendor ... than the other": no particular group
        return (f"it compares {x_field} groups, but names none of its values "
                f"({', '.join(labels[:6])}), so the comparison cannot be checked")
    if not a_named:
        return (f"it says some {x_field} group is {m.group('dir')}, but names none of its values on that "
                f"side ({', '.join(labels[:6])})")
    if not b_named:
        return None              # "A is higher" alone: the ranking checks own it
    a_v = min(groups[g] for g in a_named) if says_high else max(groups[g] for g in a_named)
    b_v = max(groups[g] for g in b_named) if says_high else min(groups[g] for g in b_named)
    if (says_high and a_v <= b_v) or (not says_high and a_v >= b_v):
        return (f"it says {' and '.join(a_named)} {m.group('dir')} than {' and '.join(b_named)}, but "
                f"{' and '.join(a_named)} is {a_v:g} and {' and '.join(b_named)} is {b_v:g}")
    return None


def _standard_value(text: str) -> Optional[float]:
    t = text.strip().lower().lstrip("$").replace(",", "")
    try:
        return float(t)
    except ValueError:
        pass
    words = t.split()
    if not words or words[0] not in _NUMBER_WORDS:
        return None
    v = float(_NUMBER_WORDS[words[0]]) + (0.5 if "half" in words else 0.0)
    return v / 100.0 if any(w.startswith("cent") for w in words) else v


def _standard_conflict(sentence: str, m: "re.Match[str]", facts: ColumnFacts, default_cols: List[str],
                       x_field: Optional[str]) -> Optional[str]:
    """'Most trips carry the standard extra fee of one dollar' when 0 is the mode."""
    adj, val = m.group("adj").lower(), m.group("val")
    target = _standard_value(val)
    if target is None:
        return None
    cols = columns_named(m.group("noun") or "", facts.columns) or \
        [c for c in ([x_field] if x_field else []) + default_cols if c in facts.columns]
    if not cols:
        return f"it calls {val} the {adj} value, but names no column it can be checked in"
    c = cols[0]
    mode = facts.mode(c)
    if mode is None:
        return f"it calls {val} the {adj} {c}, but {c}'s most common value is not computed"
    label, share = mode
    if _to_float(label) is not None and abs(_to_float(label) - target) < 1e-9:
        return None
    at = facts.value_share(c, _label(target))
    return (f"it calls {val} the {adj} {c}, but the most common {c} is {label} ({_pct(share)} of rows)"
            + (f"; {_label(target)} is {_pct(at)}" if at is not None else ""))


def _case_conflict(sentence: str, m: "re.Match[str]", facts: ColumnFacts,
                   default_cols: List[str]) -> Optional[str]:
    """'Mixed case in Residence_type' when it holds only Urban and Rural."""
    cols = columns_named(sentence, facts.columns) or [c for c in default_cols if c in facts.columns]
    for c in cols:
        values = facts.values(c)
        folded: Dict[str, set] = {}
        for v in values:
            folded.setdefault(v.lower(), set()).add(v)
        if any(len(spellings) >= 2 for spellings in folded.values()):
            continue
        n = facts.n_unique(c)
        if n is not None and n <= len(values):
            return (f"it says {c} has mixed case, but none of its {n:,} values ({', '.join(values[:6])}) "
                    f"differ only in case")
        return (f"it says {c} has mixed case, but only {len(values)} of its {n if n is not None else 'many'} "
                f"values are computed and none of them differ only in case")
    return None


def quantifier_conflict(
    text: str,
    facts: ColumnFacts,
    *,
    default_columns: Optional[List[str]] = None,
    points: Optional[List[Dict[str, Any]]] = None,
    x_field: Optional[str] = None,
    is_rate: bool = False,
    overall: Optional[float] = None,
    group_shares: Optional[Dict[str, float]] = None,
    summary: Optional[Dict[str, float]] = None,
    time_field: Optional[str] = None,
    measure: Optional[str] = None,
    aggregate: str = "count",
    correlation: Optional[float] = None,
    period_field: Optional[str] = None,
    group_card: bool = False,
) -> Optional[str]:
    """Why the text's quantifier words disagree with the computed shares, or
    cannot be checked against them; None when every quantifier holds (or the
    text has none).

    default_columns: the card's own columns, read first.
    points / x_field / is_rate / overall: the card's computed groups (and the
    overall rate) for "markedly higher than" comparisons.
    group_shares: each group's share of the chart's total (count or sum), for
    "the UK accounts for the vast majority of units sold".
    summary: a distribution's mean and median.
    time_field / measure / aggregate: the card's time axis and value, for
    "steady increase"; correlation: a scatter card's r.
    period_field: the data's date column, for "consistently higher".
    group_card: the card compares groups of x_field (a bar or pie of
    categories), for "A ... while B show higher"."""
    if not text:
        return None
    default_cols = list(default_columns or [])
    for sentence in _SENTENCE_RE.split(str(text)):
        for m in _SHARE_RE.finditer(sentence):
            why = _share_conflict(sentence, m, facts, default_cols, group_shares, x_field)
            if why:
                return why
        for m in _CONSTANT_RE.finditer(sentence):
            why = _constant_conflict(sentence, m, facts)
            if why:
                return why
        for m in _MARKED_RE.finditer(sentence):
            why = _marked_conflict(sentence, m, points or [], x_field, is_rate, overall, summary)
            if why:
                return why
        for m in _CONCENTRATION_RE.finditer(sentence):
            why = _concentration_conflict(sentence, m, facts, default_cols, group_shares, x_field)
            if why:
                return why
        for m in _MOST_ONCE_RE.finditer(sentence):
            why = _most_once_conflict(sentence, m, facts, default_cols)
            if why:
                return why
        for m in _TREND_RE.finditer(sentence):
            why = _trend_conflict(sentence, m, facts, time_field, measure, aggregate, points)
            if why:
                return why
        for m in _NO_OTHER_RE.finditer(sentence):
            why = _no_other_conflict(sentence, m, facts)
            if why:
                return why
        for m in _ASSOC_RE.finditer(sentence):
            why = _association_conflict(sentence, m, facts, default_cols, correlation)
            if why:
                return why
        for m in _PORTION_RE.finditer(sentence):
            why = _portion_conflict(sentence, m, facts, default_cols, group_shares, x_field)
            if why:
                return why
        for m in _STANDARD_RE.finditer(sentence):
            why = _standard_conflict(sentence, m, facts, default_cols, x_field)
            if why:
                return why
        for m in _CASE_RE.finditer(sentence):
            why = _case_conflict(sentence, m, facts, default_cols)
            if why:
                return why
        consistent = list(CONSISTENT_RE.finditer(sentence))
        for m in consistent:
            why = _consistent_conflict(sentence, m, facts, points or [], x_field, period_field or time_field,
                                       measure, aggregate)
            if why:
                return why
        if group_card and not consistent and not _MARKED_RE.search(sentence):
            m = _BARE_CMP_RE.search(sentence)
            why = _comparison_conflict(sentence, m, facts, points or [], x_field) if m else None
            if why:
                return why
    return None
