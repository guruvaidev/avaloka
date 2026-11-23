"""Row filters the answer did not say it applied.

Asked "How does the average order amount differ by customer segment?", the
code dropped every returned order first (``orders[orders['returned'] == False]``)
and the reply said "Enterprise customers spend the most on average, with
$545.08 per order ..." -- the mean over 18,478 of 20,000 orders. Over every
order it is $545.33. The filter was a plan step nobody asked for, it showed only
on the plan card, and the verifier called every figure verified, because the
figures are what the filtered code computed (rc7 to rc14).

This module reads the code that RAN, deterministically, and finds each row
filter whose result reaches the returned table:

* boolean-mask indexing, ``df[mask]`` and ``df.loc[mask, ...]``, a mask held in
  a variable included;
* ``df.query("...")`` and ``df.drop(df[mask].index)``;
* ``dropna`` on rows (null handling, see below); not on one column
  (``df['Country'].dropna()``), which drops that series' blanks, not rows;
* ``WHERE`` in SQL the code sent to a database (``read_sql``, ``execute``);
* ``head`` / ``tail`` / ``nlargest`` / ``nsmallest`` / ``sample`` / ``iloc[a:b]``
  when the rows they keep are aggregated afterwards. A ``head(10)`` of a result
  that was already aggregated is presentation, not a population.

A local helper the code passes a frame to (``def clean(d): return
d[d['Quantity'] > 0]``) is read at each call, on the frame it is given, and
``pd.concat`` of frames the replay holds is replayed too.

A filter the question asks for is not reported: one of its values, its flag
column, or (for a comparison) its column with a comparison word or its number
is in the user's words ("for consumers", "excluding returns", "amount over
100"). Only the user's own messages count as asking. The plan is consulted for
the step's name, never as a request: the plan is where the unrequested filter
came from.

Null handling (``dropna``, ``notna`` masks, ``IS NOT NULL``) is reported only
when it drops more than :data:`NULL_DISCLOSE_SHARE` of the rows it is applied
to. Below that it is housekeeping, and a scope line on every answer that
skipped a few blanks would stop being read.

How many rows a filter dropped is counted on the frames the run read
(:func:`app.agents.claim_verifier.data_used`), replaying only the filter and
the column conversions before it through a small evaluator of comparisons,
``isin``, ``between``, ``str`` matches and ``to_numeric``; the generated code is
never executed here. A filter that cannot be replayed (on a join, on a column
built by a function) is still disclosed, as "computed on a filtered subset".

Deterministic and LLM-free, like the rest of the evidence layer.
"""

from __future__ import annotations

import ast
import contextvars
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

#: X: a null-only filter is disclosed when it drops more than this share of
#: the rows it is applied to (5%).
NULL_DISCLOSE_SHARE = 0.05

#: The replay never evaluates more rows than this. A larger frame is replayed on
#: a fixed-seed sample of this many rows and its counts scaled, worded "about".
REPLAY_MAX_ROWS = 200_000
REPLAY_SAMPLE_SEED = 0

_SUBSET_METHODS = frozenset({"head", "tail", "nlargest", "nsmallest", "sample"})
#: Calls whose result is a statistic of the rows, not the rows.
_AGG_METHODS = frozenset({"groupby", "agg", "aggregate", "mean", "sum", "count", "size", "value_counts",
                          "pivot_table", "describe", "median", "nunique", "max", "min", "std", "var",
                          "crosstab", "quantile", "resample", "cumsum", "prod"})
#: Chain steps that keep every row.
_ROW_KEEPING = frozenset({"copy", "reset_index", "sort_values", "sort_index", "infer_objects"})
_NULL_CALLS = {"notna": True, "notnull": True, "isna": False, "isnull": False}
_SQL_WHERE = re.compile(r"\bselect\b.*?\bfrom\b.*?\bwhere\b(?P<cond>.*?)(?:\bgroup\s+by\b|\border\s+by\b|"
                        r"\bhaving\b|\blimit\b|\bunion\b|;|$)", re.IGNORECASE | re.DOTALL)
_FRAME_WORDS = frozenset({"df", "data", "frame", "dataframe", "table", "tbl", "raw", "input", "src",
                          "merged", "result", "results", "filtered", "subset", "out", "output", "final"})

#: Words in the question that ask for a null filter, a comparison, a subset.
_NULL_ASK = re.compile(r"\b(?:missing|null|nulls|blank|blanks|empty|non[- ]?null|non[- ]?missing|valid|complete|"
                       r"known|available|nan)\b", re.IGNORECASE)
_COMPARE_ASK = re.compile(r"\b(?:over|above|under|below|more than|less than|greater|fewer|at least|at most|"
                          r"after|before|since|until|between|exceed\w*|higher than|lower than|older|younger|"
                          r"positive|negative|nonzero|non-zero|zero)\b|[<>]", re.IGNORECASE)
_SUBSET_ASK = re.compile(r"\b(?:top|bottom|first|last|largest|smallest|highest|lowest|biggest|most|least|"
                         r"sample|recent|latest|oldest)\b", re.IGNORECASE)
#: Words with which an answer says it left rows out.
#: Not "without" or "among": "patients with hypertension versus without" and
#: "the fewest units among the items shown" say nothing about dropped rows.
_DISCLOSES = re.compile(r"\b(?:exclud\w*|except\w*|omit\w*|remov\w*|drop\w*|filter\w*|only|non[- ]?\w+|"
                        r"leav\w* out|left out|ignor\w*|restricted|limited)\b", re.IGNORECASE)

#: Values that make the compared column a yes/no flag.
_FLAG_STRINGS = frozenset({"0", "1", "true", "false", "yes", "no", "y", "n", "t", "f"})
#: Column-name words that say how a flag is stored, not what it is.
_STORAGE_WORDS = frozenset({"num", "numeric", "flag", "int", "bool", "binary", "is", "has", "clean", "str",
                            "encoded", "bin", "code", "cat", "col", "val", "value"})

_SHOWN_OPS = {ast.Eq: "=", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
              ast.In: "in", ast.NotIn: "not in"}
_NEGATED_OPS = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE, ast.GtE: ast.Lt, ast.Gt: ast.LtE,
                ast.LtE: ast.Gt, ast.In: ast.NotIn, ast.NotIn: ast.In}


class _Unsupported(Exception):
    """The replay evaluator does not handle this expression."""


@dataclass
class RowFilter:
    """One row filter of the executed code."""
    target: str                     # the variable it filtered
    kind: str                       # mask | query | dropna | drop | sql_where | subset
    condition: str                  # the kept condition as the code wrote it
    readable: Optional[str] = None  # the kept condition in words: "returned = False"
    lineno: int = 0
    null_only: bool = False
    columns: List[str] = field(default_factory=list)
    values: List[Any] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)      # boolean columns used as a flag
    excluded: Optional[str] = None  # what the dropped rows are: "returned = True"
    adjective: Optional[str] = None  # "returned" in "1,522 returned orders"
    noun: str = "rows"
    rows_before: Optional[int] = None
    rows_dropped: Optional[int] = None
    asked: bool = False
    disclosed: bool = False
    aggregated: bool = True         # a subset counts only once its rows are aggregated
    mask: Any = None                # the kept-rows expression (AST), for the replay
    negate: bool = False            # drop(df[mask].index): the mask is what goes
    approximate: bool = False       # counted on a sample of a frame over REPLAY_MAX_ROWS
    unnamed: bool = False           # a null test whose column could not be named

    @property
    def changed(self) -> Optional[bool]:
        return None if self.rows_dropped is None else self.rows_dropped > 0

    def reportable(self) -> bool:
        """Whether the answer owes the reader a line about this filter."""
        if self.asked or self.disclosed or not self.aggregated:
            return False
        if self.null_only:
            if self.rows_dropped is None or self.rows_before is None:
                # Uncounted is not "under the threshold": a dropna on a derived
                # column that dropped 83% of the rows was never shown (rc15).
                return True
            if not self.rows_before:
                return False
            return self.rows_dropped / self.rows_before > NULL_DISCLOSE_SHARE
        return self.changed is not False

    def line(self) -> str:
        """The scope line: what was left out, counted when it could be."""
        noun = self.noun
        if self.rows_dropped is None or self.rows_before is None:
            where = ("only where a value the code tests is present; its column could not be named"
                     if self.unnamed else f"only where {self.readable or self.condition}")
            return (f"Scope: the figures above are computed on a filtered subset of the {noun} "
                    f"({where}), not on every row; how many were left "
                    "out could not be counted, and the question did not ask for that filter.")
        kept = self.rows_before - self.rows_dropped
        n = _about if self.approximate else (lambda v: f"{v:,}")
        rest = f"the figures above are computed on the remaining {n(kept)} of {self.rows_before:,} {noun}"
        if self.kind == "subset":
            return (f"Scope: uses only {n(kept)} of {self.rows_before:,} {noun} ({self.condition}), "
                    "a subset the question did not ask for; the figures above are computed on those rows only.")
        if self.unnamed:
            return (f"Scope: excludes {n(self.rows_dropped)} {noun} where a value the code tests is missing "
                    f"(its column could not be named), a filter the question did not ask for; {rest}.")
        if self.null_only:
            share = self.rows_dropped / self.rows_before
            return f"Scope: excludes {n(self.rows_dropped)} {noun} {self.excluded} ({share:.1%}); {rest}."
        what = f"{self.adjective} {noun}" if self.adjective else noun
        return (f"Scope: excludes {n(self.rows_dropped)} {what} ({self.excluded}), a filter the question "
                f"did not ask for; {rest}.")

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "condition": self.condition,
                "readable": self.readable, "line_no": self.lineno,
                "null_only": self.null_only, "excluded": self.excluded, "rows_before": self.rows_before,
                "rows_dropped": self.rows_dropped, "approximate": self.approximate, "asked": self.asked,
                "disclosed": self.disclosed}


def _about(value: int) -> str:
    """An estimate to two significant figures: ``about 1,500`` for 1,517."""
    if value < 100:
        return f"about {value:,}"
    digits = len(str(int(value))) - 2
    return f"about {int(round(value, -digits)):,}"


# --------------------------------------------------------------------------- #
# Regular expressions from the generated code
# --------------------------------------------------------------------------- #
# A pattern in model-written code is evaluated here, in the API process, and a
# running ``re`` match cannot be interrupted. So only patterns that cannot
# backtrack catastrophically are ever run: a literal (any metacharacter
# escaped) or an alternation of literals ("returned|refunded"). Character
# classes, quantifiers, anchors and groups make the filter uncounted rather
# than counted with other semantics; nested quantifiers, backreferences and
# lookarounds are named in the rejection because they are the dangerous ones.

try:
    from re import _parser as _sre_parse      # Python 3.11+
except ImportError:  # pragma: no cover - older layout
    import sre_parse as _sre_parse            # type: ignore[no-redef]

_REPEATS = {"MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"}
_LOOKAROUND = {"ASSERT", "ASSERT_NOT"}
_BACKREF = {"GROUPREF", "GROUPREF_EXISTS", "GROUPREF_IGNORE", "GROUPREF_LOC_IGNORE", "GROUPREF_UNI_IGNORE"}


def _sre_items(tree: Any):
    """(opcode name, argument) for every node of a parsed pattern, depth first."""
    for op, arg in tree:
        yield str(op), arg
        for sub in _sre_children(str(op), arg):
            yield from _sre_items(sub)


def _sre_children(op: str, arg: Any) -> List[Any]:
    if op in _REPEATS:
        return [arg[2]]
    if op == "SUBPATTERN":
        return [arg[-1]]
    if op == "ATOMIC_GROUP":
        return [arg]
    if op == "BRANCH":
        return list(arg[1])
    if op in _LOOKAROUND:
        return [arg[1]]
    if op == "GROUPREF_EXISTS":
        return [a for a in arg[1:] if a is not None]
    return []


def unsafe_pattern(pattern: Any) -> Optional[str]:
    """Why *pattern* may backtrack catastrophically, or None: a quantifier
    over something that itself repeats or alternates (``(a+)+``, ``(a|aa)+``),
    a backreference, or a lookaround."""
    try:
        tree = _sre_parse.parse(str(pattern))
    except Exception:  # noqa: BLE001 - not a pattern Python can run
        return "does not parse"
    for op, arg in _sre_items(tree):
        if op in _BACKREF:
            return "a backreference"
        if op in _LOOKAROUND:
            return "a lookaround"
        if op in _REPEATS and any(inner in _REPEATS or inner == "BRANCH" for inner, _a in _sre_items(arg[2])):
            return "a nested quantifier"
    return None


def plain_pattern(pattern: Any) -> bool:
    """Whether *pattern* is a literal or an alternation of literals: no
    class, quantifier, anchor, group or dot; ``\\.`` and other escaped
    punctuation are literals."""
    text = str(pattern)
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            if i + 1 >= len(text) or text[i + 1].isalnum():
                return False                    # \d, \w, \b, \1: classes, anchors, references
            i += 2
            continue
        if ch in ".^$*+?{}[]()":
            return False
        i += 1
    return unsafe_pattern(text) is None


def _safe_regex_call(method: str, args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The kwargs a ``str.<method>`` call is replayed with, or _Unsupported."""
    if method not in ("contains", "match", "fullmatch") or not args:
        return kwargs
    pattern = args[0]
    if not isinstance(pattern, str):
        raise _Unsupported
    if method == "contains" and kwargs.get("regex") is False:
        return kwargs                           # a literal search: no regex runs
    if unsafe_pattern(pattern) or not plain_pattern(pattern):
        raise _Unsupported
    if method == "contains" and "|" not in pattern.replace("\\|", ""):
        kwargs = dict(kwargs, regex=False)      # a plain literal: the same rows without the regex engine
        args[0] = re.sub(r"\\(.)", r"\1", pattern)
    return kwargs


# --------------------------------------------------------------------------- #
# Reading the code
# --------------------------------------------------------------------------- #

#: Names the code binds, as ``(modules and functions, variables)``: ``pd``,
#: ``np``, ``pearsonr``, a local ``def``; an assigned name or a parameter. None
#: of them is a column. Set by :func:`detect_row_filters` for the code it reads.
#: ``pd.notna(x)`` once read ``pd`` as the column ("only where pd has a value",
#: rc18b nullxlsx).
_CODE_NAMES: contextvars.ContextVar = contextvars.ContextVar("filter_disclosure_code_names",
                                                            default=(frozenset(), frozenset()))


def _code_names(tree: Any) -> tuple:
    """``(modules and functions, variables)`` *tree* binds."""
    callables: Set[str] = set()
    variables: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                callables.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            callables.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            variables.add(node.id)
        elif isinstance(node, ast.arg):
            variables.add(node.arg)
    return frozenset(callables), frozenset(variables)


def _null_subject(call: ast.Call) -> Any:
    """What a null test tests: ``s`` in ``s.notna()`` and in ``pd.notna(s)``."""
    return call.args[0] if call.args else call.func.value


def _root(node: Any) -> Optional[str]:
    """The variable at the base of a chain: ``orders`` in ``orders.loc[m].copy()``."""
    for _ in range(40):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
        else:
            return None
    return None


def _column(node: Any, frames: Optional[Set[str]] = None) -> Optional[str]:
    """``'returned'`` for ``orders['returned']`` / ``orders.returned`` (and the
    same under ``.astype``/``.str``/``.fillna``), else None."""
    for _ in range(8):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("astype", "fillna", "lower", "upper", "strip", "abs", "round"):
            node = node.func.value
        elif isinstance(node, ast.Attribute) and node.attr in ("str", "dt"):
            node = node.value
        else:
            break
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
            and isinstance(node.slice.value, str) and isinstance(_root(node.value), str):
        return node.slice.value
    callables, variables = _CODE_NAMES.get()
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id not in callables \
            and (frames is None or node.value.id in frames) and node.attr not in ("loc", "iloc", "index", "columns"):
        return node.attr
    if isinstance(node, ast.Name) and frames is not None and node.id not in frames \
            and node.id not in callables and node.id not in variables:
        return node.id                    # a column inside a query string
    return None


def _literal(node: Any) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) \
            and isinstance(node.operand.value, (int, float)):
        return -node.operand.value
    if isinstance(node, ast.Name) and node.id in ("True", "False", "None"):
        return {"True": True, "False": False, "None": None}[node.id]
    raise _Unsupported


def _literals(node: Any) -> List[Any]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out = []
        for e in node.elts:
            try:
                out.append(_literal(e))
            except _Unsupported:
                pass
        return out
    try:
        return [_literal(node)]
    except _Unsupported:
        return []


def _is_mask(node: Any, masks: Mapping[str, Any]) -> bool:
    """Whether an indexer selects rows by a condition (not columns or labels)."""
    if isinstance(node, ast.Name):
        return node.id in masks
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Invert, ast.Not)):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in ("isin", "notna", "notnull", "isna", "isnull", "between", "contains",
                                  "startswith", "endswith", "match", "fullmatch", "duplicated",
                                  "eq", "ne", "lt", "le", "gt", "ge")
    return False


def _parts(cond: Any, masks: Mapping[str, Any]) -> List[Any]:
    """The conjuncts of a mask, a mask variable resolved."""
    if isinstance(cond, ast.Name) and cond.id in masks:
        return _parts(masks[cond.id], masks)
    if isinstance(cond, ast.BinOp) and isinstance(cond.op, ast.BitAnd):
        return _parts(cond.left, masks) + _parts(cond.right, masks)
    if isinstance(cond, ast.BoolOp) and isinstance(cond.op, ast.And):
        return [p for v in cond.values for p in _parts(v, masks)]
    return [cond]


def _resolve(cond: Any, masks: Mapping[str, Any]) -> Any:
    class _Sub(ast.NodeTransformer):
        def visit_Name(self, node):           # noqa: N802 - ast API
            return _resolve(masks[node.id], {k: v for k, v in masks.items() if k != node.id}) \
                if node.id in masks else node
    return _Sub().visit(ast.parse(ast.unparse(cond), mode="eval").body)


def _null_part(part: Any, frames: Set[str]) -> Optional[str]:
    """The column a conjunct keeps non-null, when that is all it does."""
    if isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute) and part.func.attr in _NULL_CALLS \
            and _NULL_CALLS[part.func.attr]:
        return _column(_null_subject(part), frames)
    if isinstance(part, ast.UnaryOp) and isinstance(part.op, (ast.Invert, ast.Not)):
        inner = part.operand
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) \
                and inner.func.attr in _NULL_CALLS and not _NULL_CALLS[inner.func.attr]:
            return _column(_null_subject(inner), frames)
    if isinstance(part, ast.Compare) and len(part.ops) == 1 and isinstance(part.ops[0], ast.Eq) \
            and ast.dump(part.left) == ast.dump(part.comparators[0]):
        return _column(part.left, frames)     # x == x: not NaN
    return None


def _unnamed_null(cond: Any, frames: Set[str]) -> bool:
    """Whether a conjunct tests for nulls in something no column names:
    ``pd.notna(x)`` on a helper's parameter, a module function's result."""
    for part in _parts(cond, {}):
        if isinstance(part, ast.UnaryOp) and isinstance(part.op, (ast.Invert, ast.Not)):
            part = part.operand
        if isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute) and part.func.attr in _NULL_CALLS \
                and _column(_null_subject(part), frames) is None:
            return True
    return False


def _describe(flt: RowFilter, cond: Any, frames: Set[str]) -> None:
    """Fill columns/values/flags and what the dropped rows are, from the kept condition."""
    parts = _parts(cond, {})
    nulls = [_null_part(p, frames) for p in parts]
    if all(nulls):
        flt.null_only = True
        flt.columns = [c for c in nulls if c]
        flt.excluded = "with no value in " + ", ".join(dict.fromkeys(flt.columns))
        return
    for node in ast.walk(cond):
        if isinstance(node, ast.Compare):
            col = _column(node.left, frames)
            if col:
                flt.columns.append(col)
            for comp in node.comparators:
                literals = _literals(comp)
                if col and literals and all(_is_flag_value(v) for v in literals) \
                        and all(isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)) for op in node.ops):
                    flt.flags.append(col)         # stroke == 1, returned == False: a flag column
                    continue
                flt.values.extend(v for v in literals if not isinstance(v, bool) and v is not None)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            col = _column(_null_subject(node) if node.func.attr in _NULL_CALLS else node.func.value, frames)
            if col:
                flt.columns.append(col)
            for arg in node.args:
                flt.values.extend(v for v in _literals(arg) if not isinstance(v, bool) and v is not None)
    flt.excluded = _excluded(parts[0] if len(parts) == 1 else cond, frames, flt)


def _single_column(value: Any, data: Set[str]) -> Optional[str]:
    """``'stroke'`` for ``stroke_num = pd.to_numeric(df['stroke'], errors='coerce')``."""
    if _has_agg(value) or isinstance(value, (ast.Compare, ast.BoolOp)):
        return None
    if any(isinstance(n, ast.Subscript) and (_is_mask(n.slice, {}) or (
            isinstance(n.value, ast.Attribute) and n.value.attr in ("loc", "iloc"))) for n in ast.walk(value)):
        return None                       # orders[orders['returned'] == False]: rows, not a column
    cols = {n.slice.value for n in ast.walk(value) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
            and n.value.id in data and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)}
    return cols.pop() if len(cols) == 1 else None


def _as_columns(cond: Any, series: Mapping[str, tuple]) -> Any:
    """*cond* with each one-column variable written as its column."""
    class _Sub(ast.NodeTransformer):
        def visit_Name(self, node):           # noqa: N802 - ast API
            if node.id in series:
                return ast.Subscript(value=ast.Name(id="_", ctx=ast.Load()),
                                     slice=ast.Constant(series[node.id][0]), ctx=ast.Load())
            return node
    return _Sub().visit(ast.parse(ast.unparse(cond), mode="eval").body)


def _plain(cond: Any) -> str:
    """A condition as a reader reads it: ``returned == False`` for ``orders['returned'] == False``."""
    class _Sub(ast.NodeTransformer):
        def visit_Subscript(self, node):      # noqa: N802 - ast API
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str) \
                    and isinstance(node.value, ast.Name):
                name = node.slice.value
                return ast.Name(id=name if name.isidentifier() else f"`{name}`", ctx=ast.Load())
            return self.generic_visit(node)
    return ast.unparse(_Sub().visit(ast.parse(ast.unparse(cond), mode="eval").body))


def _is_flag_value(value: Any) -> bool:
    if isinstance(value, bool) or (isinstance(value, (int, float)) and value in (0, 1)):
        return True
    return isinstance(value, str) and value.strip().lower() in _FLAG_STRINGS


def _shown(value: Any) -> str:
    return str(value) if not isinstance(value, float) else f"{value:g}"


def _excluded(cond: Any, frames: Set[str], flt: RowFilter) -> str:
    """The dropped rows in words: the negation of a simple kept condition."""
    negate = flt.negate
    if isinstance(cond, ast.UnaryOp) and isinstance(cond.op, (ast.Invert, ast.Not)):
        cond, negate = cond.operand, not negate
    col = _column(cond, frames)
    if col is not None and not isinstance(cond, ast.Compare):        # a bare flag column
        flt.flags.append(col)
        kept_true = not negate
        if kept_true:
            return f"{col} = False"
        flt.adjective = _adjective(col)
        return f"{col} = True"
    if isinstance(cond, ast.Compare) and len(cond.ops) == 1:
        col = _column(cond.left, frames)
        values = _literals(cond.comparators[0])
        if col and len(values) == 1:
            op = type(cond.ops[0])
            value = values[0]
            if isinstance(value, bool) and op in (ast.Eq, ast.NotEq):
                keeps_true = (op is ast.Eq) == value
                if negate:
                    keeps_true = not keeps_true
                if keeps_true:
                    return f"{col} = False"
                flt.adjective = _adjective(col)
                return f"{col} = True"
            shown_op = op if negate else _NEGATED_OPS.get(op, op)
            return f"{col} {_SHOWN_OPS.get(shown_op, '?')} {_shown(value)}"
    if isinstance(cond, ast.Call) and isinstance(cond.func, ast.Attribute) and cond.func.attr == "isin" and cond.args:
        col = _column(cond.func.value, frames)
        values = _literals(cond.args[0])
        if col and values:
            return f"{col} {'in' if negate else 'not in'} " + ", ".join(_shown(v) for v in values)
    text = _plain(cond)
    return f"rows matching {text}" if negate else f"rows not matching {text}"


def _has_value(columns: Sequence[str]) -> str:
    """``month has a value`` / ``Glucose and month have a value``."""
    cols = list(dict.fromkeys(map(str, columns)))
    if not cols:
        return "every column has a value"
    if len(cols) == 1:
        return f"{cols[0]} has a value"
    return f"{', '.join(cols[:-1])} and {cols[-1]} have a value"


def _kept_words(cond: Any, frames: Set[str], negate: bool = False) -> str:
    """The rows a mask keeps, in words: ``returned = False`` for
    ``orders['returned'] == False``, ``returned is one of false, no`` for an
    ``isin``. What cannot be put in words is shown as :func:`_plain` writes it."""
    parts = _parts(cond, {}) if not negate else [cond]
    out: List[str] = []
    for part in parts:
        neg = negate
        if isinstance(part, ast.UnaryOp) and isinstance(part.op, (ast.Invert, ast.Not)):
            part, neg = part.operand, not neg
        out.append(_part_words(part, frames, neg) or (f"not ({_plain(part)})" if neg else _plain(part)))
    return " and ".join(out)


def _part_words(part: Any, frames: Set[str], neg: bool) -> Optional[str]:
    null = _null_part(part, frames)
    if null and not neg:
        return f"{null} has a value"
    if isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute) and part.func.attr in _NULL_CALLS:
        col = _column(_null_subject(part), frames)
        if col:
            return f"{col} {'has a value' if _NULL_CALLS[part.func.attr] != neg else 'has no value'}"
    col = _column(part, frames)
    if col is not None and not isinstance(part, (ast.Compare, ast.Call)):
        return f"{col} = {not neg}"                                  # a bare flag column
    if isinstance(part, ast.Compare) and len(part.ops) == 1:
        col = _column(part.left, frames)
        values = _literals(part.comparators[0])
        op = type(part.ops[0])
        if col and values and op in (ast.In, ast.NotIn):
            inside = (op is ast.In) != neg
            return f"{col} is {'one' if inside else 'none'} of " + ", ".join(_shown(v) for v in values)
        if col and len(values) == 1:
            value = values[0]
            if isinstance(value, bool) and op in (ast.Eq, ast.NotEq):
                return f"{col} = {((op is ast.Eq) == value) != neg}"
            shown_op = _NEGATED_OPS.get(op, op) if neg else op
            return f"{col} {_SHOWN_OPS.get(shown_op, '?')} {_shown(value)}"
    if isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute) and part.args:
        col = _column(part.func.value, frames)
        values = _literals(part.args[0])
        attr = part.func.attr
        if col and values and attr == "isin":
            return f"{col} is {'none' if neg else 'one'} of " + ", ".join(_shown(v) for v in values)
        if col and attr == "between" and len(part.args) >= 2:
            hi = _literals(part.args[1])
            if values and hi:
                return f"{col} is {'not ' if neg else ''}between {_shown(values[0])} and {_shown(hi[0])}"
        if col and len(values) == 1 and attr in ("contains", "startswith", "endswith"):
            verb = {"contains": ("contains", "contain"), "startswith": ("starts with", "start with"),
                    "endswith": ("ends with", "end with")}[attr]
            return f"{col} {'does not ' + verb[1] if neg else verb[0]} '{values[0]}'"
    return None


def _adjective(col: str) -> Optional[str]:
    """``returned`` names the rows it is true of ("returned orders"); ``is_vip`` does not."""
    return col if re.fullmatch(r"[a-z]{4,}ed", col) else None


def _subset_condition(call: ast.Call) -> str:
    return ast.unparse(call)[ast.unparse(call).find(call.func.attr):] if isinstance(call.func, ast.Attribute) \
        else ast.unparse(call)


def _sql_condition(text: str) -> Optional[str]:
    match = _SQL_WHERE.search(text or "")
    return re.sub(r"\s+", " ", match.group("cond")).strip() if match and match.group("cond").strip() else None


def _sql_filter(target: str, cond: str, lineno: int) -> RowFilter:
    flt = RowFilter(target=target, kind="sql_where", condition=cond, lineno=lineno)
    conjuncts = re.split(r"\band\b", cond, flags=re.IGNORECASE)
    flt.null_only = all(re.fullmatch(r"\s*[\w.\"`\[\]]+\s+is\s+not\s+null\s*", c, re.IGNORECASE) for c in conjuncts)
    flt.values = re.findall(r"'([^']*)'", cond) + [float(n) for n in re.findall(r"(?<![\w'])\d+(?:\.\d+)?", cond)]
    flt.columns = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", re.sub(r"'[^']*'", " ", cond))
                   if w.lower() not in ("and", "or", "not", "is", "null", "in", "like", "between", "true", "false")]
    flt.excluded = f"rows not matching WHERE {cond}"
    return flt


class _Walker:
    """Walks the executed code in source order, tracking which filters reach
    which variables and replaying them on the frames the run read."""

    def __init__(self, used: Any, code: str):
        self.used = used
        self.filters: List[RowFilter] = []
        self.lineage: Dict[str, Set[int]] = {}
        self.aggregated: Set[str] = set()
        self.masks: Dict[str, Any] = {}
        self.data: Set[str] = set()               # variables holding rows of the data
        self.scale: Dict[str, float] = {}         # variable -> rows it stands for per replayed row
        self.strings: Dict[str, str] = {}         # variable -> the path (or text) it holds
        self.series: Dict[str, tuple] = {}        # variable -> (column, expression): one column of the data
        self.sql: Dict[str, str] = {}             # variable -> the SQL text it holds
        self.frames: Dict[str, Any] = {}          # variable -> its replayed DataFrame
        self.nouns: Dict[str, str] = {}
        self.returned: Set[str] = set()
        self.saved: Set[str] = set()
        self.helpers: Dict[str, ast.FunctionDef] = {}   # local functions a frame is passed to
        self.inlining: List[str] = []                     # the helpers being followed, innermost last
        self.code = code
        try:
            from app.agents.claim_verifier import _read_paths
            self.paths = _read_paths(code)
        except Exception:  # noqa: BLE001 - names only help the replay
            self.paths = {}

    # -- frames ------------------------------------------------------------- #
    def _bind_frame(self, name: str, path: Optional[str] = None) -> None:
        """A variable read from a file (or a ``main`` parameter) gets its frame."""
        used = self.used
        if used is None or not getattr(used, "frames", None):
            return
        index = None
        sources = list(getattr(used, "sources", None) or [])
        if path and path in sources:
            index = sources.index(path)
        try:
            from app.agents.claim_verifier import _frame_of
            index = _frame_of(name, used, self.paths) if index is None else index
        except Exception:  # noqa: BLE001
            index = None
        if index is None and len(used.frames) == 1:
            index = 0
        if index is None:
            return
        label, frame = used.frames[index]
        self.scale[name] = 1.0
        if len(frame) > REPLAY_MAX_ROWS:
            self.scale[name] = len(frame) / REPLAY_MAX_ROWS
            frame = frame.sample(n=REPLAY_MAX_ROWS, random_state=REPLAY_SAMPLE_SEED).sort_index()
        self.frames[name] = frame                 # copied on the first column it rebuilds
        self.nouns[name] = _noun(str(label), name)

    # -- statements ---------------------------------------------------------- #
    def walk(self, body: Sequence[ast.stmt], in_main: bool = False) -> None:
        for stmt in body:
            if isinstance(stmt, ast.FunctionDef):
                if stmt.name == "main":
                    for arg in stmt.args.args:
                        self.lineage.setdefault(arg.arg, set())
                        self.data.add(arg.arg)
                        self._bind_frame(arg.arg)
                elif stmt.args.args:
                    # A helper is read where it is called, on the frame it is
                    # given (:meth:`_follow`): ``def clean(d): return d[d['Quantity'] > 0]``
                    # read here filtered nothing, and the filter was never shown (rc16).
                    self.helpers[stmt.name] = stmt
                    continue
                self.walk(stmt.body, in_main=stmt.name == "main")
            elif isinstance(stmt, ast.If) and _is_main_guard(stmt):
                self.walk([s for s in stmt.body if not _calls_main(s)])
            elif isinstance(stmt, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                for block in ("body", "orelse", "finalbody"):
                    self.walk(getattr(stmt, block, []) or [], in_main)
                for handler in getattr(stmt, "handlers", []) or []:
                    self.walk(handler.body, in_main)
            elif isinstance(stmt, ast.Assign):
                self._assign(stmt.targets, stmt.value, stmt.lineno)
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                self._assign([stmt.target], stmt.value, stmt.lineno)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                self._call_statement(stmt.value, stmt.lineno)
            elif isinstance(stmt, ast.Return) and stmt.value is not None and in_main:
                ids, _frame, _noun_ = self._scan(stmt.value, stmt.lineno, "<returned>")
                self.lineage["<returned>"] = self.lineage.get("<returned>", set()) | ids
                self.returned.add("<returned>")

    def _call_statement(self, call: ast.Call, lineno: int) -> None:
        func = call.func
        if not isinstance(func, ast.Attribute):
            return
        if func.attr in ("to_csv", "to_parquet", "to_json", "to_excel"):
            self.saved |= {n for n in [_root(func.value)] if n}
            return
        root = _root(func.value)
        inplace = any(k.arg == "inplace" and isinstance(k.value, ast.Constant) and k.value.value is True
                      for k in call.keywords)
        if root and inplace:
            self._assign([ast.Name(id=root, ctx=ast.Store())], call, lineno)

    def _assign(self, targets: Sequence[Any], value: Any, lineno: int) -> None:
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        for t in targets:
            if isinstance(t, ast.Subscript) and isinstance(_root(t.value), str):
                self._column_assign(t, value, lineno)
            elif isinstance(t, ast.Tuple):
                names += [e.id for e in t.elts if isinstance(e, ast.Name)]
        if not names:
            return
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in self.helpers \
                and value.func.id not in self.inlining and len(self.inlining) < 4:
            self._follow(self.helpers[value.func.id], value, targets, lineno)
            return
        if _is_mask(value, self.masks) and not isinstance(value, ast.Name) \
                and _loaded(value) & self.data and not _reads_file(value):
            for name in names:
                self.masks[name] = value
            return
        # *value* is read with the bindings it was written against: in
        # ``orders = orders[orders['returned'] == False]`` the ``orders`` on the
        # right is the frame, not the new name. Bound first, the new name was
        # taken for a one-column series and the filter was shown as
        # "returned['returned'] == False" (rc15).
        column = _single_column(value, self.data)
        if column is None and isinstance(value, ast.Name) and value.id in self.series:
            column = self.series[value.id][0]     # s = counts_of: still one column
        text = _lit_or_none(value)
        if not (isinstance(text, str) and _sql_condition(text)):
            ids, frame, noun = self._scan(value, lineno, names[0])
        for name in names:
            self.series.pop(name, None)
            if column is not None:
                self.series[name] = (column, value) if not isinstance(value, ast.Name) else self.series[value.id]
        for name in names:
            self.strings.pop(name, None)
            if isinstance(text, str):
                self.strings[name] = text
        if isinstance(text, str) and _sql_condition(text):
            for name in names:
                self.sql[name] = text
            return
        is_agg = _has_agg(value) or any(n in self.aggregated for n in _loaded(value))
        is_data = _reads_file(value) or bool(_carried(value) & self.data) or _sql_read(value, self.sql)
        source = None
        if frame is None and isinstance(value, ast.Name) and value.id in self.frames:
            source = value.id                     # d = df_2010: the same rows
            frame, noun = self.frames[source], self.nouns.get(source) or noun
        elif frame is None:
            frame, noun, source = self._concat(value, noun)
        for name in names:
            if is_data:
                self.data.add(name)
            else:
                self.data.discard(name)
            self.lineage[name] = ids
            self.aggregated.discard(name)
            if is_agg:
                self.aggregated.add(name)
            self.frames.pop(name, None)
            if frame is not None:
                self.frames[name] = frame
                self.nouns[name] = noun
                self.scale[name] = self.scale.get(source or _root(value) or "", 1.0)
            elif _reads_file(value):
                first = value.args[0] if value.args else None
                path = self.strings.get(first.id) if isinstance(first, ast.Name) else _lit_or_none(first)
                self._bind_frame(name, path if isinstance(path, str) else None)

    def _follow(self, func: ast.FunctionDef, call: ast.Call, targets: Sequence[Any], lineno: int) -> None:
        """``out = helper(df)``: the helper's body read with its parameters bound
        to the call's arguments, and what it returns assigned to *out*."""
        params = [a.arg for a in func.args.args]
        local = set(params) | {n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        local -= {t.id for t in targets if isinstance(t, ast.Name)}
        state = [self.lineage, self.data, self.frames, self.scale, self.nouns, self.series, self.masks,
                 self.aggregated, self.strings, self.sql]
        saved = [{k: (k in d, d[k] if isinstance(d, dict) and k in d else None) for k in local} for d in state]
        self.inlining.append(func.name)
        try:
            given = list(call.args) + [k.value for k in call.keywords if k.arg in params]
            for param, arg in zip(params, given):
                self._assign([ast.Name(id=param, ctx=ast.Store())], arg, lineno)
            for stmt in func.body:
                if isinstance(stmt, ast.Return):
                    if stmt.value is not None:
                        self._assign(targets, stmt.value, stmt.lineno)
                    break
                self.walk([stmt])
        finally:
            self.inlining.pop()
            # The helper's names are its own: a parameter ``df`` is not main's ``df``.
            for d, before in zip(state, saved):
                for k, (had, v) in before.items():
                    if isinstance(d, dict):
                        if had:
                            d[k] = v
                        else:
                            d.pop(k, None)
                    elif had:
                        d.add(k)
                    else:
                        d.discard(k)

    def _concat(self, value: Any, noun: str):
        """``pd.concat([a, b])`` of replayed frames: (frame, noun, source), or Nones."""
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "concat"
                and isinstance(value.func.value, ast.Name) and value.func.value.id in ("pd", "pandas")
                and value.args and isinstance(value.args[0], (ast.List, ast.Tuple))):
            return None, noun, None
        parts = value.args[0].elts
        if not parts or not all(isinstance(p, ast.Name) and p.id in self.frames and self.scale.get(p.id, 1.0) == 1.0
                                for p in parts):
            return None, noun, None
        if any(k.arg not in ("ignore_index", "sort") for k in value.keywords):
            return None, noun, None                 # axis=1, keys=...: not rows stacked
        try:
            import pandas as pd

            frame = pd.concat([self.frames[p.id] for p in parts], ignore_index=True)
        except Exception:  # noqa: BLE001 - frames the replay cannot stack are unknown
            return None, noun, None
        name = "<concat>"
        self.scale[name] = 1.0
        if len(frame) > REPLAY_MAX_ROWS:
            self.scale[name] = len(frame) / REPLAY_MAX_ROWS
            frame = frame.sample(n=REPLAY_MAX_ROWS, random_state=REPLAY_SAMPLE_SEED).sort_index()
        nouns = {self.nouns.get(p.id) for p in parts}
        return frame, (nouns.pop() if len(nouns) == 1 and None not in nouns else noun), name

    def _column_assign(self, target: ast.Subscript, value: Any, lineno: int) -> None:
        """``orders['returned'] = orders['returned'].astype('bool')``: replayed when it can be."""
        name = _root(target.value)
        if not isinstance(target.value, ast.Name):
            return                                  # df.loc[mask, 'col'] = v: an update, not a filter
        self.lineage.setdefault(name, set())
        self.lineage[name] |= set().union(*[self.lineage.get(n, set()) for n in _carried(value)])
        frame = self.frames.get(name)
        column = _literals(target.slice)
        if frame is None or len(column) != 1 or not isinstance(column[0], str):
            return
        try:
            series = _Eval(self, frame, name).value(value)
            frame = frame.copy()
            frame[column[0]] = series
        except Exception:  # noqa: BLE001 - a column the replay cannot build is unknown
            frame = frame.drop(columns=[column[0]], errors="ignore")
        self.frames[name] = frame

    # -- expressions --------------------------------------------------------- #
    def _scan(self, value: Any, lineno: int, target: str):
        """The filters in *value*, the lineage it carries, and (when it is a
        linear chain from a replayed frame) the frame it evaluates to."""
        ids: Set[int] = set()
        for name in _carried(value):
            ids |= self.lineage.get(name, set())
        new: List[int] = []
        frame, noun = self._chain(value, lineno, target, new)
        if frame is _NOT_CHAIN:
            frame = None
            for node in ast.walk(value):
                self._node_filters(node, lineno, target, new)
        ids |= set(new)
        for agg in [n for n in ast.walk(value) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in _AGG_METHODS]:
            inner = set(_loaded(agg.func.value))
            for i in new:
                if self.filters[i].kind == "subset" and _contains(agg.func.value, self.filters[i].mask):
                    self.filters[i].aggregated = True
            for name in inner:
                for i in self.lineage.get(name, set()):
                    if self.filters[i].kind == "subset":
                        self.filters[i].aggregated = True
        return ids, frame, noun

    def _chain(self, value: Any, lineno: int, target: str, new: List[int]):
        """Replay a linear chain ``root[...].loc[...].query(...).dropna()``.
        Returns (_NOT_CHAIN, None) when *value* is not one."""
        steps: List[Any] = []
        node = value
        while True:
            if isinstance(node, ast.Subscript):
                steps.append(node)
                node = node.value
            elif isinstance(node, ast.Attribute) and node.attr in ("loc", "iloc"):
                steps.append(node)
                node = node.value
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and not _reads_file(node):
                steps.append(node)
                node = node.func.value
            else:
                break
        if not isinstance(node, ast.Name) or node.id not in self.data or not steps:
            return _NOT_CHAIN, None
        root = node.id
        frame = self.frames.get(root)
        noun = self.nouns.get(root) or _noun("", root)
        aggregated = root in self.aggregated
        # One column taken out of the frame (``df['Country']``): a dropna on it
        # drops values of that series, not rows of the frame (rc16: a Series
        # dropna was shown as "only where every column has a value").
        one_column = root in self.series
        selected: List[str] = []                  # df[['a', 'b']]: the columns a bare dropna reads
        steps.reverse()
        i = 0
        while i < len(steps):
            step = steps[i]
            nxt = steps[i + 1] if i + 1 < len(steps) else None
            flt: Optional[RowFilter] = None
            if isinstance(step, ast.Attribute):          # .loc / .iloc, then its indexer
                if not isinstance(nxt, ast.Subscript):
                    return _NOT_CHAIN, None
                indexer = nxt.slice.elts[0] if isinstance(nxt.slice, ast.Tuple) and nxt.slice.elts else nxt.slice
                if step.attr == "loc" and _is_mask(indexer, self.masks) and not aggregated:
                    flt = self._mask_filter(root, "mask", indexer, lineno, frame)
                elif step.attr == "iloc" and isinstance(indexer, ast.Slice) and not aggregated:
                    flt = self._iloc_filter(root, indexer, lineno, frame, nxt)
                elif step.attr == "loc":
                    frame = None                          # labels: rows unknown
                i += 2
            elif isinstance(step, ast.Subscript):
                if _is_mask(step.slice, self.masks):
                    if not aggregated:
                        flt = self._mask_filter(root, "mask", step.slice, lineno, frame)
                    frame = None if aggregated else frame
                else:
                    cols = _literals(step.slice)
                    if isinstance(step.slice, ast.Name) and isinstance(self.strings.get(step.slice.id), str):
                        cols = [self.strings[step.slice.id]]      # df_all[required_col]
                    if cols and all(isinstance(c, str) for c in cols):
                        one_column = not isinstance(step.slice, (ast.List, ast.Tuple))
                        selected = [] if one_column else [str(c) for c in cols]
                    if frame is not None and cols and all(isinstance(c, str) for c in cols) \
                            and set(cols) <= set(map(str, frame.columns)):
                        frame = frame[cols] if isinstance(step.slice, (ast.List, ast.Tuple)) else frame[[cols[0]]]
                    else:
                        frame = None
                i += 1
            else:                                         # a method call
                attr = step.func.attr
                if attr in _AGG_METHODS:
                    aggregated = True
                    frame = None
                elif attr == "query" and step.args and isinstance(_lit_or_none(step.args[0]), str) \
                        and not _sql_condition(step.args[0].value):
                    flt = self._query_filter(root, step, lineno, frame)
                elif attr == "dropna" and not _drops_columns(step):
                    # On an aggregated table it drops groups of the result,
                    # like a mask there: presentation, not the population.
                    # On one column it drops that column's blanks only.
                    if not aggregated and not one_column:
                        flt = self._dropna_filter(root, step, lineno, frame, selected)
                elif attr == "drop" and _drop_mask(step) is not None:
                    flt = self._mask_filter(root, "drop", _drop_mask(step), lineno, frame, negate=True)
                elif attr in _SUBSET_METHODS:
                    if not aggregated:
                        flt = self._subset_filter(root, step, lineno, frame)
                elif attr in _ROW_KEEPING:
                    pass
                else:                                     # merge, assign, ...: rows unknown
                    for arg in [*step.args, *[k.value for k in step.keywords]]:
                        for sub in ast.walk(arg):
                            self._node_filters(sub, lineno, target, new)
                    self._node_filters(step, lineno, target, new)
                    frame = None
                i += 1
            if flt is not None:
                flt.noun = noun
                self.filters.append(flt)
                new.append(len(self.filters) - 1)
                frame = _apply(flt, frame)
        return frame, noun

    def _node_filters(self, node: Any, lineno: int, target: str, new: List[int]) -> None:
        """Filters found anywhere in a non-linear expression (no replay)."""
        flt: Optional[RowFilter] = None
        if isinstance(node, ast.Subscript):
            base = node.value
            indexer = node.slice
            if isinstance(base, ast.Attribute) and base.attr == "loc":
                indexer = indexer.elts[0] if isinstance(indexer, ast.Tuple) and indexer.elts else indexer
                base = base.value
            elif isinstance(base, ast.Attribute) and base.attr == "iloc":
                return
            if _is_mask(indexer, self.masks) and _root(base) in self.data and _root(base) not in self.aggregated \
                    and not _has_agg(base):
                flt = self._mask_filter(_root(base), "mask", indexer, lineno, None)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            root = _root(node.func.value) or target
            text = _lit_or_none(node.args[0]) if node.args else None
            if isinstance(node.args[0] if node.args else None, ast.Name) and node.args[0].id in self.sql:
                text = self.sql[node.args[0].id]
            if isinstance(text, str) and _sql_condition(text):
                flt = _sql_filter(target, _sql_condition(text), lineno)
            elif root not in self.data:
                pass
            elif attr == "query" and isinstance(text, str):
                flt = self._query_filter(root, node, lineno, None)
            elif attr == "dropna" and not _drops_columns(node) and root not in self.aggregated \
                    and not _has_agg(node.func.value) and not _one_column(node.func.value, self.series, self.strings):
                flt = self._dropna_filter(root, node, lineno, None)
            elif attr in _SUBSET_METHODS and not _has_agg(node.func.value) and root not in self.aggregated:
                flt = self._subset_filter(root, node, lineno, None)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
            text = _lit_or_none(node.args[0])
            if isinstance(node.args[0], ast.Name) and node.args[0].id in self.sql:
                text = self.sql[node.args[0].id]
            if isinstance(text, str) and _sql_condition(text):
                flt = _sql_filter(target, _sql_condition(text), lineno)
        if flt is not None:
            flt.noun = self.nouns.get(flt.target) or _noun("", flt.target)
            self.filters.append(flt)
            new.append(len(self.filters) - 1)

    # -- one filter each ------------------------------------------------------ #
    def _mask_filter(self, root: str, kind: str, indexer: Any, lineno: int, frame: Any,
                     negate: bool = False) -> RowFilter:
        cond = _resolve(indexer, self.masks)
        shown = _as_columns(cond, self.series)
        flt = RowFilter(target=root, kind=kind, condition=_plain(shown), lineno=lineno, mask=cond, negate=negate)
        _describe(flt, shown, set(self.lineage) | self.data | {"_"})
        flt.readable = _kept_words(shown, set(self.lineage) | self.data | {"_"}, negate)
        flt.unnamed = _unnamed_null(shown, set(self.lineage) | self.data | {"_"})
        if negate:
            flt.condition = f"not ({flt.condition})"
        self._count(flt, frame, root)
        return flt

    def _query_filter(self, root: str, call: ast.Call, lineno: int, frame: Any) -> RowFilter:
        text = str(call.args[0].value)
        flt = RowFilter(target=root, kind="query", condition=text, lineno=lineno)
        try:
            cond = ast.parse(re.sub(r"`([^`]*)`", lambda m: re.sub(r"\W", "_", m.group(1)), text), mode="eval").body
        except SyntaxError:
            flt.excluded = f"rows not matching {text}"
            return flt
        flt.mask = cond
        _describe(flt, cond, set())
        flt.readable = _kept_words(cond, set())
        flt.unnamed = _unnamed_null(cond, set())
        self._count(flt, frame, None)
        return flt

    def _dropna_filter(self, root: str, call: ast.Call, lineno: int, frame: Any,
                       selected: Sequence[str] = ()) -> RowFilter:
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        subset = _literals(kw["subset"]) if "subset" in kw else list(selected)
        flt = RowFilter(target=root, kind="dropna", condition=_subset_condition(call), lineno=lineno, null_only=True,
                        columns=[str(c) for c in subset])
        flt.excluded = ("with no value in " + ", ".join(map(str, subset))) if subset else "with a missing value"
        flt.readable = _has_value(flt.columns)
        if frame is not None:
            try:
                args = {k: _literal(v) if k != "subset" else subset for k, v in kw.items()
                        if k in ("subset", "how", "thresh")}
                if subset and "subset" not in args:
                    args["subset"] = subset
                self._counted(flt, len(frame), len(frame) - len(frame.dropna(**args)))
            except Exception:  # noqa: BLE001 - uncounted
                flt.rows_before = flt.rows_dropped = None
        return flt

    def _subset_filter(self, root: str, call: ast.Call, lineno: int, frame: Any) -> RowFilter:
        flt = RowFilter(target=root, kind="subset", condition=_subset_condition(call), lineno=lineno,
                        aggregated=False, mask=call)
        flt.values = [v for a in call.args for v in _literals(a) if isinstance(v, (int, float))]
        flt.columns = [v for a in call.args for v in _literals(a) if isinstance(v, str)]
        if frame is not None:
            try:
                total = round(len(frame) * self.scale.get(root, 1.0))
                n = _literal(call.args[0]) if call.args else 5
                if call.func.attr == "sample":
                    kw = {k.arg: _literal(k.value) for k in call.keywords if k.arg in ("n", "frac")}
                    n = int(kw["frac"] * total) if "frac" in kw else int(kw.get("n", n if call.args else 1))
                kept = min(int(n), total) if n >= 0 else max(total + int(n), 0)
                flt.rows_before, flt.rows_dropped = total, total - kept   # exact: n is a count of rows
            except Exception:  # noqa: BLE001
                pass
        return flt

    def _iloc_filter(self, root: str, sl: ast.Slice, lineno: int, frame: Any, node: ast.Subscript) -> RowFilter:
        flt = RowFilter(target=root, kind="subset", condition=ast.unparse(node)[len(root) + 1:] if root else
                        ast.unparse(node), lineno=lineno, aggregated=False, mask=node)
        if frame is not None:
            try:
                lo = _literal(sl.lower) if sl.lower else None
                hi = _literal(sl.upper) if sl.upper else None
                total = round(len(frame) * self.scale.get(root, 1.0))
                kept = len(range(total)[lo:hi])
                flt.rows_before, flt.rows_dropped = total, total - kept
            except Exception:  # noqa: BLE001
                pass
        return flt

    def _counted(self, flt: RowFilter, before: int, dropped: int) -> None:
        """Record a count made on the replayed rows, scaled up when they are a sample."""
        scale = self.scale.get(flt.target, 1.0)
        flt.approximate = scale != 1.0
        flt.rows_before = round(before * scale)
        flt.rows_dropped = round(dropped * scale)

    def _count(self, flt: RowFilter, frame: Any, root: Optional[str]) -> None:
        if frame is None or flt.mask is None:
            return
        try:
            mask = _Eval(self, frame, root, query=flt.kind == "query").mask(flt.mask)
            if flt.negate:
                mask = ~mask
            self._counted(flt, len(frame), int(len(frame) - int(mask.sum())))
            flt._kept_mask = mask                  # noqa: SLF001 - used by _apply
        except Exception:  # noqa: BLE001 - a mask the replay cannot evaluate is uncounted
            flt.rows_before = flt.rows_dropped = None


_NOT_CHAIN = object()


def _apply(flt: RowFilter, frame: Any) -> Any:
    """The frame after *flt*, when it was counted on it."""
    if frame is None or flt.rows_dropped is None:
        return None
    if flt.kind == "dropna":
        return None if flt.rows_dropped else frame
    if flt.kind == "subset":
        return None
    mask = getattr(flt, "_kept_mask", None)
    return frame[mask.values] if mask is not None else None


class _Eval:
    """Evaluates a mask or a column conversion on a frame. Only comparisons,
    boolean logic, a fixed set of Series methods and ``pd.to_numeric`` /
    ``pd.to_datetime``; anything else raises, and the filter stays uncounted."""

    _SERIES_METHODS = frozenset({"isin", "notna", "notnull", "isna", "isnull", "between", "astype", "fillna",
                                 "abs", "round", "eq", "ne", "lt", "le", "gt", "ge", "lower", "upper", "strip",
                                 "contains", "startswith", "endswith", "len", "duplicated", "match",
                                 "fullmatch", "title", "capitalize", "casefold"})

    def __init__(self, walker: _Walker, frame: Any, root: Optional[str], query: bool = False):
        self.walker, self.frame, self.root, self.query = walker, frame, root, query

    def mask(self, node: Any) -> Any:
        import pandas as pd

        out = self.value(node)
        if not isinstance(out, pd.Series):
            raise _Unsupported
        out = out.reindex(self.frame.index, fill_value=False) if not out.index.equals(self.frame.index) else out
        return out.fillna(False).astype(bool)

    def value(self, node: Any) -> Any:
        import numpy as np
        import pandas as pd

        if isinstance(node, (ast.Constant, ast.UnaryOp)) and not (
                isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Invert, ast.Not))):
            try:
                return _literal(node)
            except _Unsupported:
                pass
        if isinstance(node, ast.UnaryOp):
            inner = self.value(node.operand)
            if isinstance(node.op, (ast.Invert, ast.Not)):
                return ~inner.astype(bool) if isinstance(inner, pd.Series) else (not inner)
            if isinstance(node.op, ast.USub):
                return -inner
            raise _Unsupported
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return [self.value(e) for e in node.elts]
        if isinstance(node, ast.Name):
            if node.id in ("True", "False", "None"):
                return _literal(node)
            if node.id in self.walker.series:
                return self.value(self.walker.series[node.id][1])
            if node.id in self.walker.masks:
                return self.value(self.walker.masks[node.id])
            if self.query and node.id in self.frame.columns:
                return self.frame[node.id]
            raise _Unsupported
        if isinstance(node, ast.Subscript):
            key = _literals(node.slice)
            if isinstance(node.value, ast.Name) and node.value.id in (self.root, *self.walker.frames) \
                    and len(key) == 1 and isinstance(key[0], str):
                frame = self.frame if node.value.id == self.root else self.walker.frames[node.value.id]
                if key[0] not in frame.columns:
                    raise _Unsupported
                return frame[key[0]]
            raise _Unsupported
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == self.root and node.attr in self.frame.columns:
                return self.frame[node.attr]
            if node.attr in ("str", "dt"):
                return _Accessor(self.value(node.value), node.attr)
            if isinstance(node.value, (ast.Attribute,)) and node.value.attr == "dt":
                return getattr(self.value(node.value.value).dt, node.attr)
            raise _Unsupported
        if isinstance(node, ast.Compare):
            left = self.value(node.left)
            out = None
            for op, comp in zip(node.ops, node.comparators):
                right = self.value(comp)
                if isinstance(op, (ast.In, ast.NotIn)) or (self.query and isinstance(right, list)
                                                         and isinstance(op, (ast.Eq, ast.NotEq))):
                    res = left.isin(right)
                    res = ~res if isinstance(op, (ast.NotIn, ast.NotEq)) else res
                else:
                    res = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
                           ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
                           ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b}[type(op)](left, right)
                out = res if out is None else (out & res)
                left = right
            return out
        if isinstance(node, ast.BoolOp):
            values = [self.value(v) for v in node.values]
            out = values[0]
            for v in values[1:]:
                out = (out & v) if isinstance(node.op, ast.And) else (out | v)
            return out
        if isinstance(node, ast.BinOp):
            a, b = self.value(node.left), self.value(node.right)
            ops = {ast.BitAnd: lambda x, y: x & y, ast.BitOr: lambda x, y: x | y, ast.BitXor: lambda x, y: x ^ y,
                   ast.Add: lambda x, y: x + y, ast.Sub: lambda x, y: x - y, ast.Mult: lambda x, y: x * y,
                   ast.Div: lambda x, y: x / y}
            if type(node.op) not in ops:
                raise _Unsupported
            return ops[type(node.op)](a, b)
        if isinstance(node, ast.Call):
            kwargs = {k.arg: _literal(k.value) for k in node.keywords if k.arg}
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                    and func.value.id in ("pd", "pandas") and func.attr in ("to_numeric", "to_datetime") and node.args:
                kwargs = {k: v for k, v in kwargs.items() if k in ("errors", "format", "utc", "dayfirst")}
                return getattr(pd, func.attr)(self.value(node.args[0]), **kwargs)
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                    and func.value.id in ("np", "numpy") and func.attr in ("isnan",) and node.args:
                return np.isnan(self.value(node.args[0]))
            if isinstance(func, ast.Attribute) and func.attr in self._SERIES_METHODS:
                owner = self.value(func.value)
                if func.attr == "astype":
                    # astype(str) names a builtin, which value() does not
                    # evaluate: read before the arguments are, or every column
                    # built with .astype(str) was dropped from the replay and
                    # each filter on it left uncounted (rc15).
                    first = node.args[0] if node.args else None
                    dtype = first.id if isinstance(first, ast.Name) and first.id in ("str", "int", "float", "bool") \
                        else (self.value(first) if first is not None else None)
                    if dtype not in ("bool", "str", "string", "int", "int64", "Int64", "float",
                                     "float64", "category", "boolean"):
                        raise _Unsupported
                    return owner.astype(dtype)
                args = [self.value(a) for a in node.args]
                kwargs = {k: v for k, v in kwargs.items() if k in ("case", "na", "regex", "inclusive", "keep")}
                if isinstance(owner, _Accessor):
                    kwargs = _safe_regex_call(func.attr, args, kwargs)
                elif func.attr in ("match", "fullmatch"):
                    raise _Unsupported
                return getattr(owner, func.attr)(*args, **kwargs)
            raise _Unsupported
        raise _Unsupported


class _Accessor:
    """``series.str`` / ``series.dt`` for :class:`_Eval`."""

    def __init__(self, series: Any, kind: str):
        self.series, self.kind = series, kind

    def __getattr__(self, name: str) -> Any:
        if name not in _Eval._SERIES_METHODS and name not in ("year", "month", "day"):
            raise _Unsupported
        return getattr(getattr(self.series, self.kind), name)


# --------------------------------------------------------------------------- #
# Small AST helpers
# --------------------------------------------------------------------------- #

def _loaded(node: Any) -> Set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _carried(node: Any) -> Set[str]:
    """The variables whose ROWS *node* carries. ``len(hit) / len(df)`` carries
    none: a filter whose rows are only counted defines what is counted (the
    numerator of a share), not the population a figure is of. Nor does a
    model fitted on them, or its ``coef_``."""
    out: Set[str] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "len":
            continue
        if isinstance(n, ast.Attribute) and (n.attr in ("shape", "empty", "columns", "dtypes")
                                             or (n.attr.endswith("_") and not n.attr.startswith("_"))):
            continue                       # model.coef_: a fitted model's attribute, not rows
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "fit":
            continue                       # a fitted model, not rows
        if isinstance(n, ast.Name):
            out.add(n.id)
        stack.extend(ast.iter_child_nodes(n))
    return out


def _one_column(node: Any, series: Mapping[str, tuple], strings: Mapping[str, str]) -> bool:
    """Whether *node* is one column of a frame: ``df['Country']``, ``df[col]``,
    ``df['x'].astype(str)``, or a variable holding one."""
    for _ in range(40):
        if isinstance(node, ast.Name):
            return node.id in series
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Name):
                return isinstance(strings.get(node.slice.id), str)
            return isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
        elif isinstance(node, ast.Attribute):
            node = node.value
        else:
            return False
    return False


def _contains(tree: Any, node: Any) -> bool:
    return node is not None and any(n is node for n in ast.walk(tree))


def _has_agg(node: Any) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in _AGG_METHODS
               for n in ast.walk(node))


def _reads_file(node: Any) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    return name.startswith("read_") or name in ("read_csv_best_effort", "read_input_frame")


def _sql_read(node: Any, sql: Mapping[str, str]) -> bool:
    """``pd.read_sql(q, conn)`` / ``con.execute(q).df()`` with SQL text: rows of the data."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and n.args:
            first = n.args[0]
            text = sql.get(first.id) if isinstance(first, ast.Name) else _lit_or_none(first)
            if isinstance(text, str) and re.search(r"\bselect\b.*\bfrom\b", text, re.IGNORECASE | re.DOTALL):
                return True
    return False


def _lit_or_none(node: Any) -> Any:
    try:
        return _literal(node)
    except _Unsupported:
        return None


def _drops_columns(call: ast.Call) -> bool:
    for k in call.keywords:
        if k.arg == "axis" and _lit_or_none(k.value) in (1, "columns"):
            return True
    return bool(call.args) and _lit_or_none(call.args[0]) in (1, "columns")


def _drop_mask(call: ast.Call) -> Any:
    """The mask of ``df.drop(df[mask].index)`` / ``df.drop(index=df[mask].index)``."""
    candidates = list(call.args[:1]) + [k.value for k in call.keywords if k.arg in ("index", "labels")]
    for node in candidates:
        if isinstance(node, ast.Attribute) and node.attr == "index" and isinstance(node.value, ast.Subscript):
            sub = node.value
            indexer = sub.slice
            if isinstance(sub.value, ast.Attribute) and sub.value.attr == "loc":
                indexer = indexer.elts[0] if isinstance(indexer, ast.Tuple) else indexer
            if _is_mask(indexer, {}):
                return indexer
    return None


def _is_main_guard(stmt: ast.If) -> bool:
    return "__name__" in ast.unparse(stmt.test)


def _calls_main(stmt: ast.stmt) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "main"
               for n in ast.walk(stmt))


def _noun(label: str, name: str) -> str:
    """What the rows are: ``orders`` for orders.csv or a variable ``orders``."""
    for text in (re.sub(r"\.\w+$", "", label.lower()), name.lower()):
        words = [w for w in re.findall(r"[a-z]+", text) if w not in _FRAME_WORDS]
        if len(words) == 1 and len(words[0]) > 3 and words[0].endswith("s") and not words[0].endswith("ss"):
            return words[0]
    return "rows"


# --------------------------------------------------------------------------- #
# Asked for? Disclosed?
# --------------------------------------------------------------------------- #

def _stems(text: str) -> Set[str]:
    from avaloka.goal_analysis import word_stems

    out: Set[str] = set()
    for word in re.findall(r"[a-z][a-z0-9]*", str(text or "").lower()):
        out |= set(word_stems(word))
    return out


def _words_of(value: Any) -> List[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9]*", str(value).lower()) if len(w) >= 3]


def _numbers(text: str) -> Set[float]:
    out = set()
    for raw in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", str(text or "")):
        try:
            out.add(float(raw.replace(",", "")))
        except ValueError:
            pass
    return out


def asked_for(flt: RowFilter, question: str) -> bool:
    """Whether the user's words ask for this filter. Deterministic matching
    of the filter's values, flag column, column and numbers to the question."""
    q_stems = _stems(question)
    q_numbers = _numbers(question)

    def named(value: Any) -> bool:
        words = [w for w in _words_of(value) if w not in _STORAGE_WORDS]
        if not words:
            return False
        return all(_stems(w) & q_stems for w in words)

    if flt.kind == "subset":
        return bool(_SUBSET_ASK.search(question or "")) or any(float(v) in q_numbers for v in flt.values)
    if flt.null_only:
        return bool(_NULL_ASK.search(question or "")) and (
            not flt.columns or any(named(c) for c in flt.columns) or "missing" in question.lower())
    for value in flt.values:
        if isinstance(value, str) and named(value):
            return True
        if isinstance(value, (int, float)) and float(value) in q_numbers:
            return True
        if isinstance(value, str) and _numbers(value) & q_numbers:     # '2024-01-01' and "in 2024"
            return True
    for flag in flt.flags:
        if named(flag.replace("is_", "").replace("has_", "")):
            return True
    if _COMPARE_ASK.search(question or "") and any(named(c) for c in flt.columns):
        return True
    return False


def disclosed_in(flt: RowFilter, narrative: str) -> bool:
    """Whether the answer already says which rows it is about: it names the
    filter's value or flag ("returned orders average ...", "for consumers"),
    or a sentence has an exclusion word and the filter's column."""
    named = [t for t in [*flt.flags, *[v for v in flt.values if isinstance(v, str)]] if _words_of(t)]
    stems = _stems(narrative)
    if any(all(_stems(w) & stems for w in _words_of(t)) for t in named):
        return True
    tokens = [t for t in flt.columns if _words_of(t)]
    if not tokens:
        return False
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", narrative or ""):
        if not _DISCLOSES.search(sentence) or (flt.null_only and not _NULL_ASK.search(sentence)):
            continue
        stems = _stems(sentence)
        if any(all(_stems(w) & stems for w in _words_of(t)) for t in tokens):
            return True
    return False


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #

def detect_row_filters(code: str, used: Any = None) -> List[RowFilter]:
    """Every row filter of *code* whose rows reach what ``main`` returns (or
    what the script saves), counted on *used* where the replay can."""
    try:
        tree = ast.parse(str(code or ""))
    except (SyntaxError, ValueError):
        return []
    token = _CODE_NAMES.set(_code_names(tree))
    try:
        walker = _Walker(used, str(code or ""))
        walker.walk(tree.body)
    finally:
        _CODE_NAMES.reset(token)
    sinks = walker.returned | walker.saved
    if sinks:
        reach: Set[int] = set()
        for name in sinks:
            reach |= walker.lineage.get(name, set())
    else:
        reach = set(range(len(walker.filters)))
    return _merged([f for i, f in enumerate(walker.filters) if i in reach])


def _merged(filters: Sequence[RowFilter]) -> List[RowFilter]:
    """One filter per line of code: a helper called on each sheet filters each,
    and the rows it left out are the sum (uncounted if any one is)."""
    out: List[RowFilter] = []
    by_line: Dict[tuple, RowFilter] = {}
    for flt in filters:
        key = (flt.kind, flt.condition, flt.lineno, flt.readable)
        first = by_line.get(key)
        if first is None:
            by_line[key] = flt
            out.append(flt)
            continue
        if None in (first.rows_before, first.rows_dropped, flt.rows_before, flt.rows_dropped):
            first.rows_before = first.rows_dropped = None
        else:
            first.rows_before += flt.rows_before
            first.rows_dropped += flt.rows_dropped
        first.approximate = first.approximate or flt.approximate
        first.aggregated = first.aggregated or flt.aggregated
    return out


def user_question(state: Mapping[str, Any]) -> str:
    """Every message the user wrote in the thread: a clarification answered
    with "yes, only consumers" asks as much as the first message did.

    Only what the user typed (app/agents/user_text). The web chat appends an
    "[Analysis context]" block to every message it hands the graph: each
    dataset's columns and preview rows (``"returned": false``). Read as the
    user's words, that block "asked" for every filter on a column of the data,
    and on rc15 no scope line was ever shown, though the tests (bare
    questions) all passed."""
    from app.agents.user_text import user_texts

    return "\n".join(user_texts(state))


def scope_filters(code: str, question: str, narrative: str, used: Any = None) -> List[RowFilter]:
    """The filters of the executed code that the answer owes a scope line."""
    out = []
    for flt in detect_row_filters(code, used):
        flt.asked = asked_for(flt, question)
        flt.disclosed = disclosed_in(flt, narrative)
        if flt.reportable():
            out.append(flt)
    return out


def apply_filter_scope(report: Dict[str, Any], state: Mapping[str, Any], narrative: str, code: str,
                       used: Any = None) -> Dict[str, Any]:
    """The verifier's hook: a report whose figures were computed on rows the
    question did not ask to filter, and that the answer does not say so,
    carries a scope line (as a caveat :func:`presentable_reply` appends) and
    is ``scoped``, never plainly ``verified``."""
    try:
        filters = scope_filters(code, user_question(state), narrative, used)
    except Exception:  # noqa: BLE001 - scope is extra evidence, never a crash
        import logging

        logging.getLogger(__name__).debug("[filter-disclosure] could not read the code's filters", exc_info=True)
        return report
    if not filters:
        return report
    lines = list(dict.fromkeys(f.line() for f in filters))
    plan = state.get("planner_definition") if isinstance(state.get("planner_definition"), Mapping) else {}
    steps = [str(t.get("name")) for t in plan.get("transformations") or []
             if isinstance(t, Mapping) and str(t.get("type") or "").lower() == "filter" and t.get("name")]
    report["caveats"] = list(report.get("caveats") or []) + [line for line in lines
                                                             if line not in (report.get("caveats") or [])]
    report["scope"] = {"filtered": True, "disclosed_in_answer": False, "filters": [f.as_dict() for f in filters],
                       "lines": lines, "plan_steps": steps}
    from app.agents.claim_verifier import _NUMBER, split_claims

    report["scoped_claims"] = [c[:400] for c in split_claims(narrative) if _NUMBER.search(c)]
    report["scoped"] = True
    if "verified" in report:
        report["verified"] = False
    return report


#: Number words a question or an answer may give a cut in: "the top ten".
_NUMBER_WORDS = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
                 12: "twelve", 15: "fifteen", 20: "twenty", 25: "twenty-five", 50: "fifty", 100: "hundred"}
_CUT_WORDS = r"(?:top|first|bottom|last|best|worst|largest|smallest|highest|lowest|biggest|leading)"


def truncation_line(cut: Mapping[str, Any]) -> str:
    """The scope line of a cut the code made, from the code alone."""
    n = cut["n"]
    rows = "row" if n == 1 else f"{n:,} rows"
    if cut.get("by"):
        which = "top" if cut.get("top") else "bottom"
        return (f"Scope: the result keeps only the {which} {rows} by {cut['by']}, a cut the code made after "
                "sorting; every other row is left out of it, and the question did not ask for that cut.")
    return (f"Scope: the result keeps only the first {rows}, a cut the code made; every other row is left "
            "out of it, and the question did not ask for that cut.")


def _says_cut(text: str, n: int) -> bool:
    """*text* gives the cut: "top 10", "the 10 best", "top ten"."""
    said = rf"(?:{n}|{re.escape(_NUMBER_WORDS[n])})" if n in _NUMBER_WORDS else str(n)
    return bool(re.search(rf"\b{_CUT_WORDS}[\s-]+{said}\b|\b{said}\s+(?:\w+\s+)?{_CUT_WORDS}\b", text or "",
                          re.IGNORECASE))


def apply_truncation_scope(report: Dict[str, Any], state: Mapping[str, Any], narrative: str, code: str,
                           rows: Any = None) -> Dict[str, Any]:
    """The verifier's hook for a cut the code made to its result
    (``sort_values('TotalRevenue').head(10)``): the result is a part of what
    the code computed, and a "most" said of it is said of that part. Listed
    as a scope line unless the question asked for the cut or the answer
    gives it. The figures of the kept rows are still the code's, so this
    neither scopes them nor unverifies them; a superlative the cut leaves
    unconfirmed does (app/agents/claim_extremes)."""
    try:
        from app.agents.claim_extremes import result_truncation

        n_rows = len(rows) if isinstance(rows, list) and rows else None
        cut = result_truncation(code, n_rows) if n_rows else None
    except Exception:  # noqa: BLE001 - scope is extra evidence, never a crash
        import logging

        logging.getLogger(__name__).debug("[filter-disclosure] could not read the code's cut", exc_info=True)
        return report
    if not cut:
        return report
    question = user_question(state)
    # "what sells most?" asks for the one at the top: a cut to one row is the answer, not a scope.
    asked = _says_cut(question, cut["n"]) or (cut["n"] == 1 and bool(_SUBSET_ASK.search(question)))
    disclosed = _says_cut(narrative, cut["n"])
    report["truncation"] = {"n": cut["n"], "by": cut.get("by"), "top": cut.get("top"), "call": cut.get("call"),
                            "asked": asked, "disclosed_in_answer": disclosed}
    if asked or disclosed:
        return report
    line = truncation_line(cut)
    report["truncation"]["line"] = line
    if line not in (report.get("caveats") or []):
        report["caveats"] = list(report.get("caveats") or []) + [line]
    return report
