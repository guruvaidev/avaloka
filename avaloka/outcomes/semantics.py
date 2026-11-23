"""Column semantics for the outcome layer (§1.2 of docs/design/2.0-OUTCOMES.md).

Every column gets exactly one **role** and, for measures, a **unit**. Roles are
computed from names, dtypes and values; values are read to decide and are
never copied into the result, so a :class:`ColumnSemantics` is safe to log.

Precedence, highest first: ``target`` (named by the caller), ``geo`` (the §1.3
detector, which outranks the identifier name rule so ``zip`` and
``state_fips`` are places, not ids), ``time``, ``id``, ``measure``,
``dimension`` / ``high_card_dimension``, ``text``. ``stage`` is a property of a
*set* of count measures and is reported by :func:`stage_set`.

The two name rules are copies of ``app/agents/visualization_agent.py:306-317``
(``_DATE_RE``, ``_ID_NAME_RE``): ``avaloka/`` must not import ``app`` (S1.08).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from avaloka.domains import normalise
from avaloka.outcomes import geo as _geo

ROLES = ("time", "id", "geo", "measure", "dimension", "high_card_dimension", "stage", "target", "text")
UNITS = ("currency", "fraction", "percent", "count", "unitless")
GRAINS = ("day", "week", "month", "quarter", "year")

_DATE_RE = re.compile(
    r"^\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?\s*$"
)
_ID_NAME_RE = re.compile(
    r"(?:^|[\s_\-])(id|no|num|number|code|uuid|key|phone|zip|zipcode|postcode|pin|pincode"
    r"|tract|census[\s_\-]?tract|fips|geoid|geo[\s_\-]?id|msamd|msa|cbsa)$",
    re.IGNORECASE,
)
_YEAR_NAME_RE = re.compile(r"(?:^|[\s_\-])(year|yr|fy|fiscal[\s_\-]?year)(?:$|[\s_\-])", re.IGNORECASE)
_CURRENCY_NAME_RE = re.compile(
    r"\b(amount|amt|price|revenue|sales|spend|spent|cost|balance|income|profit|margin|"
    r"charge|charges|fee|fees|payment|exposure|budget|gmv|aov|usd|eur|gbp|value)\b")
_RATE_NAME_RE = re.compile(r"\b(rate|share|pct|percent|percentage|ratio|ctr|cvr|roas|apr)\b")
_CURRENCY_CODE_NAME_RE = re.compile(r"\b(currency|ccy|currency code)\b")
_FUNNEL_WORDS = ("impression", "view", "visit", "session", "click", "lead", "signup", "sign up",
                 "add to cart", "cart", "checkout", "purchase", "order", "conversion", "install")

TIME_MATCH = 0.95       # §1.2: _DATE_RE on >= 95 % of non-null values
ID_DISTINCT = 0.98      # §1.2: distinct / non-null >= 0.98 for a non-float column
DIM_MIN, DIM_MAX = 2, 50
STAGE_MIN = 3


@dataclass(frozen=True)
class ColumnSemantics:
    name: str
    role: str
    unit: str | None = None
    grain: str | None = None
    geo_kind: str | None = None
    distinct: int = 0
    non_null: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _grain(ts: pd.Series) -> str | None:
    vals = pd.Series(ts.dropna().unique()).sort_values()
    if len(vals) < 2:
        return None
    days = vals.diff().dropna().dt.total_seconds().median() / 86400.0
    for limit, grain in ((1.5, "day"), (8, "week"), (32, "month"), (95, "quarter")):
        if days <= limit:
            return grain
    return "year"


def _as_time(s: pd.Series, name: str) -> tuple[bool, str | None]:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True, _grain(s)
    nn = s.dropna()
    if nn.empty:
        return False, None
    if pd.api.types.is_integer_dtype(s) and nn.between(1900, 2100).all() and (
            _YEAR_NAME_RE.search(name) or nn.nunique() <= 60):
        return True, "year"
    if s.dtype == object:
        txt = nn.astype(str)
        if txt.str.match(_DATE_RE).mean() >= TIME_MATCH:
            return True, _grain(pd.to_datetime(txt, errors="coerce"))
    return False, None


def _measure_unit(s: pd.Series, norm: str, currency_hint: bool) -> str:
    nn = s.dropna()
    rate_name = bool(_RATE_NAME_RE.search(norm))
    if rate_name and len(nn) and nn.between(0, 1).all():
        return "fraction"
    if rate_name and len(nn) and nn.between(0, 100).all():
        return "percent"
    if _CURRENCY_NAME_RE.search(norm) or (currency_hint and re.search(r"\b(amount|balance|price)\b", norm)):
        return "currency"
    if len(nn) and (nn >= 0).all() and (pd.api.types.is_integer_dtype(s) or (nn % 1 == 0).all()):
        return "count"
    return "unitless"


def classify_column(name: str, s: pd.Series, *, target: str | None = None,
                    currency_hint: bool = False, small_table: bool = False) -> ColumnSemantics:
    norm = normalise(name)
    nn = s.dropna()
    distinct, non_null = int(nn.nunique()), int(len(nn))
    base = {"name": name, "distinct": distinct, "non_null": non_null}
    if target is not None and name == target:
        unit = _measure_unit(s, norm, currency_hint) if _is_numeric(s) else None
        return ColumnSemantics(role="target", unit=unit, reason="named as the analysis target", **base)

    found = _geo.detect_column(name, s)
    if found is not None:
        return ColumnSemantics(role="geo", geo_kind=found.kind,
                               reason=f"{found.kind} codes on {found.match_rate:.0%} of distinct values", **base)

    is_time, grain = _as_time(s, name)
    if is_time:
        return ColumnSemantics(role="time", grain=grain, reason="dates or calendar years", **base)

    if _ID_NAME_RE.search(name.strip()) or _ID_NAME_RE.search(norm):
        return ColumnSemantics(role="id", reason="the name labels an identifier", **base)
    is_float = pd.api.types.is_float_dtype(s)
    if (not is_float and non_null and distinct / non_null >= ID_DISTINCT and not small_table
            and not (_is_numeric(s) and _CURRENCY_NAME_RE.search(norm))):
        return ColumnSemantics(role="id", reason="nearly every value is distinct", **base)

    if _is_numeric(s):
        return ColumnSemantics(role="measure", unit=_measure_unit(s, norm, currency_hint),
                               reason="numeric", **base)
    if distinct < DIM_MIN:
        return ColumnSemantics(role="text", reason="a single value", **base)
    if distinct <= DIM_MAX:
        return ColumnSemantics(role="dimension", reason=f"{distinct} categories", **base)
    if s.dtype == object and nn.astype(str).str.len().median() > 60:
        return ColumnSemantics(role="text", reason="free text", **base)
    return ColumnSemantics(role="high_card_dimension", reason=f"{distinct} categories; top-N only", **base)


def infer(df: pd.DataFrame, *, target: str | None = None,
          domain: Mapping[str, Any] | None = None) -> dict[str, ColumnSemantics]:
    """Roles for every column of ``df``, in column order.

    ``domain`` is #532's ``classify`` result; its unit hints let ``amount`` and
    ``balance`` read as currency in a finance table.
    """
    units = " ".join((domain or {}).get("hints", {}).get("units", [])).lower()
    currency_hint = "currency" in units
    small = len(df) <= 50
    out: dict[str, ColumnSemantics] = {}
    latlon = _geo.detect_latlon(df)
    for col in df.columns:
        name = str(col)
        if latlon and name in latlon:
            s = df[col]
            out[name] = ColumnSemantics(name=name, role="geo", geo_kind="latlon",
                                        distinct=int(s.nunique()), non_null=int(s.notna().sum()),
                                        reason="latitude/longitude pair in range")
            continue
        out[name] = classify_column(name, df[col], target=target, currency_hint=currency_hint, small_table=small)
    return out


def currency_column(df: pd.DataFrame, sem: Mapping[str, ColumnSemantics]) -> str | None:
    """The column naming the currency code, if any (§1.2: never a guessed symbol)."""
    for name, c in sem.items():
        if c.role in ("dimension", "text") and _CURRENCY_CODE_NAME_RE.search(normalise(name)):
            vals = df[name].dropna().astype(str)
            if len(vals) and vals.str.fullmatch(r"[A-Z]{3}").all():
                return name
    return None


def currency_code(df: pd.DataFrame, sem: Mapping[str, ColumnSemantics]) -> str | None:
    """One ISO 4217 code when the whole table is in one currency, else None."""
    col = currency_column(df, sem)
    if col is None:
        return None
    vals = df[col].dropna().unique()
    return str(vals[0]) if len(vals) == 1 else None


def funnel_candidates(sem: Mapping[str, ColumnSemantics]) -> list[str]:
    """Count measures named like funnel stages, in funnel order, monotone or not."""
    ranked: list[tuple[int, str]] = []
    for name, c in sem.items():
        if c.role != "measure" or c.unit != "count":
            continue
        norm = normalise(name)
        for i, w in enumerate(_FUNNEL_WORDS):
            if w in norm:
                ranked.append((i, name))
                break
    ranked.sort()
    return [n for _, n in ranked]


def stage_means_monotone(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    means = [float(df[c].mean()) for c in cols]
    return len(cols) >= STAGE_MIN and all(a >= b for a, b in zip(means, means[1:]))


def stage_set(df: pd.DataFrame, sem: Mapping[str, ColumnSemantics]) -> list[str]:
    """Count measures that form a funnel: >= 3 whose column means fall
    monotonically in the order a funnel names them (impressions >= clicks >=
    conversions). Empty when no such set exists (§1.2 ``stage``)."""
    cols = funnel_candidates(sem)
    return cols if stage_means_monotone(df, cols) else []


def by_role(sem: Mapping[str, ColumnSemantics], *roles: str) -> list[str]:
    return [n for n, c in sem.items() if c.role in roles]
