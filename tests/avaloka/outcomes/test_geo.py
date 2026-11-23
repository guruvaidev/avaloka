"""B2 and the k-floor: geo detection at its thresholds, key resolution, and
suppression of regions with fewer than k = 10 rows (§1.3, O9).

Bites (run by hand, in the PR):
- set ``COUNTRY_MATCH = 0.75``: the 8-of-10 column becomes a country map,
  ``test_two_unknowns_in_ten_countries_degrade_the_map`` goes red.
- change ``if int(n[key]) < k`` to ``< 1`` in ``region_table``:
  the k-floor tests go red.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from avaloka.outcomes import geo as G
from avaloka.outcomes import semantics as S
from avaloka.outcomes.recommend import recommend


@pytest.mark.parametrize("values,key", [
    (["US", "FR", "DE", "JP", "BR"], ["USA", "FRA", "DEU", "JPN", "BRA"]),
    (["usa", "fra", "deu", "jpn", "bra"], ["USA", "FRA", "DEU", "JPN", "BRA"]),
    (["United States of America", "france", "Germany", "Japan", "BRAZIL"], ["USA", "FRA", "DEU", "JPN", "BRA"]),
    (["United Kingdom", "Norway", "Singapore", "Malta"], ["GBR", "NOR", "SGP", "MLT"]),
])
def test_country_codes_and_names_resolve_to_alpha3(values, key):
    det = G.detect_column("country", pd.Series(values))
    assert det is not None and det.kind == "country"
    resolved, unresolved = G.resolve(values, "country")
    assert [resolved[v][0] for v in values] == key and unresolved == []


@pytest.mark.parametrize("name,values,key", [
    ("state", ["CA", "NY", "TX", "PR", "DC"], ["CA", "NY", "TX", "PR", "DC"]),
    ("state", ["California", "new york", "Texas"], ["CA", "NY", "TX"]),
    ("state_fips", [6, 36, 48], ["CA", "NY", "TX"]),
    ("state_fips", ["06", "36", "48"], ["CA", "NY", "TX"]),
])
def test_us_states_resolve_to_usps(name, values, key):
    det = G.detect_column(name, pd.Series(values))
    assert det is not None and det.kind == "us_state"
    resolved, _ = G.resolve(values, "us_state")
    assert [resolved[v][0] for v in values] == key


def test_ambiguous_codes_follow_the_name_and_a_tie_maps_nothing():
    both = pd.Series(["CA", "DE", "IN", "GA"])     # all four are states and countries
    assert G.detect_column("state", both).kind == "us_state"
    assert G.detect_column("country", both).kind == "country"
    assert G.detect_column("region", both) is None


def test_two_unknowns_in_ten_countries_degrade_the_map():
    vals = ["US", "FR", "DE", "JP", "BR", "IN", "CN", "MX", "Atlantis", "Narnia"]
    assert G.detect_column("country", pd.Series(vals)) is None      # 80 % < 90 %
    df = pd.DataFrame({"country": np.repeat(vals, 12), "sales": np.arange(120.0)})
    sem = S.infer(df)
    assert sem["country"].role == "dimension"
    rec = recommend("sales by country on a map", df, sem)
    assert "choropleth" not in [c.chart_kind for c in rec.candidates]
    nine = vals[:9] + ["GB"]
    assert G.detect_column("country", pd.Series(nine)).kind == "country"   # 9/10 = 90 %


def test_fewer_than_three_distinct_values_is_not_a_country_column():
    assert G.detect_column("country", pd.Series(["US", "FR"] * 5)) is None


@pytest.mark.parametrize("lat,lon,ok", [("lat", "lon", True), ("Latitude", "Longitude", True),
                                        ("pickup_lat", "pickup_lng", True), ("lat", "dropoff_lon", False)])
def test_latlon_pair_by_name(lat, lon, ok):
    df = pd.DataFrame({lat: [40.7, 34.0, 51.5], lon: [-74.0, -118.2, -0.1]})
    assert (G.detect_latlon(df) == (lat, lon)) is ok


def test_latlon_needs_95_percent_in_range():
    df = pd.DataFrame({"lat": [40.0] * 18 + [200.0, 300.0], "lon": [-74.0] * 20})
    assert G.detect_latlon(df) is None                 # 90 % in range
    df.loc[19, "lat"] = 41.0
    assert G.detect_latlon(df) == ("lat", "lon")       # 95 %


def _states(counts: dict[str, int]) -> pd.DataFrame:
    rows = [(s, 1.0) for s, n in counts.items() for _ in range(n)]
    return pd.DataFrame(rows, columns=["state", "sales"])


def test_k_floor_suppresses_regions_below_ten_rows():
    df = _states({"CA": 25, "NY": 10, "TX": 9, "WY": 1})
    t = G.region_table(df, "state", "us_state", "sales")
    assert [r["key"] for r in t.rows] == ["CA", "NY"]
    assert t.suppressed == 2 and t.k_floor == 10
    dump = json.dumps(t.rows)
    assert "TX" not in dump and "WY" not in dump and "Texas" not in dump


def test_k_floor_all_suppressed_refuses_the_map():
    df = _states({"CA": 9, "NY": 9, "TX": 9})
    sem = S.infer(df)
    rec = recommend("sales by state", df, sem)
    assert rec.question_type == "geo" and rec.candidates == []
    assert "fewer than 10 rows" in rec.refused[0]["reason"]


def test_points_with_a_pii_finding_are_binned_under_the_floor():
    rng = np.random.default_rng(1)
    dense = pd.DataFrame({"lat": 40.7128 + rng.uniform(0, 0.001, 30), "lon": -74.006 + rng.uniform(0, 0.001, 30)})
    lone = pd.DataFrame({"lat": [34.05], "lon": [-118.24]})
    df = pd.concat([dense, lone], ignore_index=True)
    raw = G.point_table(df, "lat", "lon", pii=False)
    assert len(raw.points) == 31 and not raw.binned
    binned = G.point_table(df, "lat", "lon", pii=True)
    assert binned.binned and binned.suppressed == 1 and len(binned.points) == 1
    assert binned.points[0]["n"] == 30
    assert all(abs(p["lat"] - 34.05) > 1 for p in binned.points)


def test_geohash_round_trip():
    gh = G.geohash_encode(57.64911, 10.40744, 11)
    assert gh == "u4pruydqqvj"
    lat, lon = G.geohash_centre(gh)
    assert abs(lat - 57.64911) < 1e-5 and abs(lon - 10.40744) < 1e-5


def test_bundled_geometry_carries_the_keys_resolution_produces():
    world = json.loads(G.asset_text("world_110m.json"))
    us = json.loads(G.asset_text("us_states_20m.json"))
    w_keys = {f["properties"]["iso_a3"] for f in world["features"]}
    u_keys = {f["properties"]["usps"] for f in us["features"]}
    assert {"USA", "FRA", "NOR", "DEU", "BRA"} <= w_keys
    assert len(u_keys) == 52 and {"AK", "HI", "PR", "DC"} <= u_keys
    for _, (usps, _n) in G.state_table().items():
        assert usps in u_keys
