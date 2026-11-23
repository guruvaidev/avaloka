"""Geo detection, key resolution and the map privacy floor (§1.3).

2.0 maps three kinds: ``country`` (ISO 3166-1 alpha-2, alpha-3 or English
name), ``us_state`` (USPS code, name, or a two-digit state FIPS under a
``fips`` name) and ``latlon`` (a named, in-range pair). ``us_zip`` is
*detected*, so a ZIP column is a place rather than an identifier, but it is not
mappable until the second cut's ZIP3 rollup; :data:`MAPPABLE` says which kinds
draw.

A detector fires only above its threshold over **distinct** non-null values;
values decide and the column name breaks a tie. All tables are bundled
(``assets/``, built by ``scripts/build_outcome_assets.py``); nothing here
touches the network.

**Privacy floor.** Any region or cell with fewer than ``k`` rows (default
:data:`K_FLOOR` = 10, owner decision O9) is suppressed: its value is left out
of the data the export embeds and it is counted in ``suppressed``. Raw points
are drawn only when the caller reports no PII finding on the coordinate
columns; otherwise points are binned to geohash precision 5 under the same
floor.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
import re
from typing import Any, Mapping

import pandas as pd

K_FLOOR = 10
COUNTRY_MATCH = 0.90
STATE_MATCH = 0.90
LATLON_MATCH = 0.95
ZIP_MATCH = 0.95
MIN_DISTINCT = 3
MAX_GEO_DISTINCT = 2000
GEOHASH_PRECISION = 5

KINDS = ("country", "us_state", "latlon", "us_zip")
MAPPABLE = ("country", "us_state", "latlon")
ASSET_FOR = {"country": "world_110m", "us_state": "us_states_20m", "latlon": "world_110m"}

_ZIP_RE = re.compile(r"^\d{5}(-\d{4})?$")
_LAT_NAMES = {"lat", "latitude"}
_LON_NAMES = {"lon", "lng", "long", "longitude"}
_GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"


def _asset_path(name: str):
    return resources.files("avaloka.outcomes").joinpath("assets", name)


def asset_text(name: str) -> str:
    """The raw text of a bundled asset (``world_110m.json``, ``echarts.min.js``...)."""
    return _asset_path(name).read_text(encoding="utf-8")


def _fold(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().casefold()


@lru_cache(maxsize=1)
def country_table() -> dict[str, tuple[str, str]]:
    """Folded alias -> (alpha3, display name)."""
    out: dict[str, tuple[str, str]] = {}
    with _asset_path("iso3166.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["alpha3"], row["name"])
            for alias in [row["alpha2"], row["alpha3"], row["name"], *filter(None, row["alt_names"].split("|"))]:
                out.setdefault(_fold(alias), key)
    return out


@lru_cache(maxsize=1)
def state_table() -> dict[str, tuple[str, str]]:
    """Folded alias -> (usps, display name)."""
    out: dict[str, tuple[str, str]] = {}
    with _asset_path("us_states.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["usps"], row["name"])
            for alias in (row["usps"], row["name"]):
                out[_fold(alias)] = key
    return out


@lru_cache(maxsize=1)
def state_fips_table() -> dict[str, tuple[str, str]]:
    with _asset_path("us_states.csv").open(encoding="utf-8") as fh:
        return {row["fips"]: (row["usps"], row["name"]) for row in csv.DictReader(fh)}


def _fips_key(v: Any) -> str:
    s = str(v).strip()
    if re.fullmatch(r"\d+(\.0)?", s):
        s = s.split(".")[0].zfill(2)
    return s


@dataclass(frozen=True)
class Detection:
    kind: str
    match_rate: float
    distinct: int


def _name_tokens(name: str) -> set[str]:
    from avaloka.domains import normalise
    return set(normalise(name).split())


def _rate(distinct: list[Any], table: Mapping[str, Any], keyf=_fold) -> float:
    return sum(1 for v in distinct if keyf(v) in table) / len(distinct) if distinct else 0.0


def detect_column(name: str, s: pd.Series) -> Detection | None:
    """``country``, ``us_state`` or ``us_zip`` for a single column, else None."""
    nn = s.dropna()
    if nn.empty or pd.api.types.is_float_dtype(s) or pd.api.types.is_bool_dtype(s):
        return None
    distinct = list(pd.unique(nn))
    tokens = _name_tokens(name)
    # More distinct values than any code table has aliases cannot match 90 %:
    # skip the per-value lookups on id-like columns.
    if len(distinct) > MAX_GEO_DISTINCT and not tokens & {"zip", "zipcode", "postal", "postcode"}:
        return None

    if tokens & {"zip", "zipcode", "postal", "postcode"}:
        rate = sum(1 for v in distinct if _ZIP_RE.match(str(v).strip())) / len(distinct)
        if rate >= ZIP_MATCH:
            return Detection("us_zip", rate, len(distinct))
    if "fips" in tokens:
        rate = _rate(distinct, state_fips_table(), _fips_key)
        if rate >= STATE_MATCH and len(distinct) >= MIN_DISTINCT:
            return Detection("us_state", rate, len(distinct))
        return None
    if pd.api.types.is_numeric_dtype(s) or len(distinct) < MIN_DISTINCT:
        return None

    c_rate = _rate(distinct, country_table())
    s_rate = _rate(distinct, state_table())
    c_ok, s_ok = c_rate >= COUNTRY_MATCH, s_rate >= STATE_MATCH
    if c_ok and s_ok:
        # "CA", "DE", "IN" are both. The name decides; without a cue the
        # higher rate does, and an exact tie maps nothing rather than guess.
        if tokens & {"state", "states", "province"}:
            return Detection("us_state", s_rate, len(distinct))
        if tokens & {"country", "countries", "nation", "iso", "iso2", "iso3"}:
            return Detection("country", c_rate, len(distinct))
        if c_rate == s_rate:
            return None
        c_ok, s_ok = c_rate > s_rate, s_rate > c_rate
    if s_ok:
        return Detection("us_state", s_rate, len(distinct))
    if c_ok:
        return Detection("country", c_rate, len(distinct))
    return None


def detect_latlon(df: pd.DataFrame) -> tuple[str, str] | None:
    """A (lat, lon) column pair: names end in a lat / lon token with the same
    prefix, both numeric, and >= 95 % of rows inside [-90, 90] x [-180, 180]."""
    from avaloka.domains import normalise
    lats: dict[str, str] = {}
    lons: dict[str, str] = {}
    for col in df.columns:
        toks = normalise(str(col)).split()
        if not toks:
            continue
        prefix = " ".join(toks[:-1])
        if toks[-1] in _LAT_NAMES:
            lats.setdefault(prefix, str(col))
        elif toks[-1] in _LON_NAMES:
            lons.setdefault(prefix, str(col))
    for prefix, lat in lats.items():
        lon = lons.get(prefix)
        if lon is None:
            continue
        a, b = df[lat], df[lon]
        if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
            continue
        both = a.notna() & b.notna()
        if not both.any():
            continue
        ok = (a[both].between(-90, 90) & b[both].between(-180, 180)).mean()
        if ok >= LATLON_MATCH:
            return lat, lon
    return None


def resolve(values, kind: str) -> tuple[dict[Any, tuple[str, str]], list[Any]]:
    """Map each distinct value to its region (key, display name).

    Keys are ISO alpha-3 for countries and USPS codes for states, which are
    the ``nameProperty`` values the bundled GeoJSON carries. Returns the map
    and the values that resolved to nothing.
    """
    if kind == "country":
        table, keyf = country_table(), _fold
    elif kind == "us_state":
        table = state_table()
        fips = state_fips_table()

        def keyf(v):
            k = _fold(v)
            return k if k in table else _fips_key(v)
        table = {**table, **fips}
    else:
        raise ValueError(f"{kind!r} has no region table")
    resolved: dict[Any, tuple[str, str]] = {}
    unresolved: list[Any] = []
    for v in pd.unique(pd.Series(list(values)).dropna()):
        hit = table.get(keyf(v))
        if hit is None:
            unresolved.append(v)
        else:
            resolved[v] = hit
    return resolved, unresolved


@dataclass
class RegionTable:
    kind: str
    asset: str
    k_floor: int
    rows: list[dict[str, Any]] = field(default_factory=list)   # key, name, value, n (shown regions only)
    suppressed: int = 0
    unresolved: int = 0

    @property
    def drawable(self) -> bool:
        return bool(self.rows)


def region_table(df: pd.DataFrame, geo_col: str, kind: str, measure: str | None = None,
                 agg: str = "sum", k: int = K_FLOOR) -> RegionTable:
    """Aggregate ``measure`` per region and apply the k floor.

    A region with fewer than ``k`` rows keeps neither its value nor its row
    count in the output: only the number of such regions survives.
    """
    if kind not in ("country", "us_state"):
        raise ValueError(f"{kind!r} is not a region kind")
    resolved, unresolved = resolve(df[geo_col], kind)
    keys = df[geo_col].map({v: key for v, (key, _name) in resolved.items()})
    frame = pd.DataFrame({"key": keys})
    frame["v"] = df[measure] if measure else 1
    frame = frame.dropna(subset=["key"])
    grouped = frame.groupby("key")["v"]
    n = grouped.size()
    val = grouped.agg(agg if measure else "size")
    names = {key: name for key, name in resolved.values()}
    out = RegionTable(kind=kind, asset=ASSET_FOR[kind], k_floor=k, unresolved=len(unresolved))
    for key in sorted(n.index):
        if int(n[key]) < k:
            out.suppressed += 1
            continue
        out.rows.append({"key": key, "name": names[key], "value": float(val[key]), "n": int(n[key])})
    return out


def geohash_encode(lat: float, lon: float, precision: int = GEOHASH_PRECISION) -> str:
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    bits, bit, ch, even, out = (16, 8, 4, 2, 1), 0, 0, True, []
    while len(out) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            if lon >= mid:
                ch |= bits[bit]
                lon_lo = mid
            else:
                lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat >= mid:
                ch |= bits[bit]
                lat_lo = mid
            else:
                lat_hi = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            out.append(_GEOHASH_ALPHABET[ch])
            bit, ch = 0, 0
    return "".join(out)


def geohash_centre(gh: str) -> tuple[float, float]:
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    even = True
    for c in gh:
        cd = _GEOHASH_ALPHABET.index(c)
        for mask in (16, 8, 4, 2, 1):
            if even:
                mid = (lon_lo + lon_hi) / 2
                lon_lo, lon_hi = (mid, lon_hi) if cd & mask else (lon_lo, mid)
            else:
                mid = (lat_lo + lat_hi) / 2
                lat_lo, lat_hi = (mid, lat_hi) if cd & mask else (lat_lo, mid)
            even = not even
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2


@dataclass
class PointTable:
    binned: bool
    k_floor: int
    points: list[dict[str, Any]] = field(default_factory=list)   # lat, lon, value[, n]
    suppressed: int = 0


def point_table(df: pd.DataFrame, lat: str, lon: str, measure: str | None = None, *,
                pii: bool = False, k: int = K_FLOOR, max_points: int = 2000) -> PointTable:
    """Raw points when ``pii`` is False; geohash-5 cells under the k floor when True."""
    sub = df[[lat, lon] + ([measure] if measure else [])].dropna()
    if not pii:
        sub = sub.head(max_points)
        pts = [{"lat": round(float(r[lat]), 5), "lon": round(float(r[lon]), 5),
                "value": float(r[measure]) if measure else 1.0} for _, r in sub.iterrows()]
        return PointTable(binned=False, k_floor=k, points=pts)
    cells = sub.apply(lambda r: geohash_encode(float(r[lat]), float(r[lon])), axis=1)
    frame = pd.DataFrame({"cell": cells, "v": sub[measure] if measure else 1.0})
    g = frame.groupby("cell")["v"]
    n, total = g.size(), g.sum()
    out = PointTable(binned=True, k_floor=k)
    for cell in sorted(n.index):
        if int(n[cell]) < k:
            out.suppressed += 1
            continue
        clat, clon = geohash_centre(cell)
        out.points.append({"lat": round(clat, 4), "lon": round(clon, 4), "value": float(total[cell]),
                           "n": int(n[cell])})
    return out
