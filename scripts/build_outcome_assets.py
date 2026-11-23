#!/usr/bin/env python3
"""Rebuild every bundled outcome asset from pinned public sources.

    python scripts/build_outcome_assets.py            # download, verify, build
    python scripts/build_outcome_assets.py --cache D  # reuse downloads kept in D
    python scripts/build_outcome_assets.py --check    # rebuild in memory, diff SHA256SUMS

Writes ``avaloka/outcomes/assets/``: ``echarts.min.js`` (copied byte for byte
from the npm tarball), ``world_110m.json``, ``us_states_20m.json``,
``iso3166.csv``, ``us_states.csv``, the upstream licence files, and
``SHA256SUMS``. Every input is pinned by URL and sha256; a mismatch stops the
build rather than shipping something nobody reviewed. Stdlib only: the
shapefile and DBF readers below cover exactly the two files we read
(polygon shapes, character fields) and nothing more.

Licences: ECharts is Apache-2.0 (its LICENSE and NOTICE are copied next to it);
Natural Earth is public domain; US Census cartographic boundaries are a US
Government work, public domain. ``NOTICE`` in the asset directory records all
three with versions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import struct
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "avaloka" / "outcomes" / "assets"

ECHARTS_VERSION = "5.6.0"
NATURAL_EARTH_VERSION = "v5.1.2"
CENSUS_VINTAGE = "2023"
_NE = f"https://raw.githubusercontent.com/nvkelso/natural-earth-vector/{NATURAL_EARTH_VERSION}"

# name -> (url, sha256 of the downloaded bytes)
SOURCES: dict[str, tuple[str, str]] = {
    "echarts.tgz": (
        f"https://registry.npmjs.org/echarts/-/echarts-{ECHARTS_VERSION}.tgz",
        "e1b82487dd7f8d9f0438eec1f1433051148e0095ac7a15d4db890afda45cf610",
    ),
    "ne_110m.shp": (
        f"{_NE}/110m_cultural/ne_110m_admin_0_countries.shp",
        "08e341606e8391e458c3f08deb312de664b56bfae376064c5aa0aee6681a5f55",
    ),
    "ne_110m.dbf": (
        f"{_NE}/110m_cultural/ne_110m_admin_0_countries.dbf",
        "1fee677cd4e03b367876e03861eb10197e4022a846bf92060e0313432863785b",
    ),
    # The 10m table lists the small countries 110m geometry leaves out
    # (Singapore, Malta, ...), so detection knows their codes.
    "ne_10m.dbf": (
        f"{_NE}/10m_cultural/ne_10m_admin_0_countries.dbf",
        "c5dbd3dd5fd7e2ef49051fc88562c03819e8ea63a382642df6eadd1243bf4b49",
    ),
    "census_states.zip": (
        f"https://www2.census.gov/geo/tiger/GENZ{CENSUS_VINTAGE}/shp/cb_{CENSUS_VINTAGE}_us_state_20m.zip",
        "0fd2d6562708ff8182c00d5d25b5556d049ecf2794d97b89ed2dac4d5e9e2c8d",
    ),
}

# Douglas-Peucker tolerance in degrees, then coordinates rounded to 2 dp
# (about 1 km), as §1.3 specifies.
WORLD_TOLERANCE = 0.08
US_TOLERANCE = 0.03
DECIMALS = 2

# 50 states + DC + PR (§1.3). The Census file has exactly these 52.
_US_EXPECTED = 52

OUTPUTS = ("echarts.min.js", "world_110m.json", "us_states_20m.json", "iso3166.csv",
           "us_states.csv", "licenses/echarts-LICENSE.txt", "licenses/echarts-NOTICE.txt",
           "licenses/echarts-LICENSE-d3.txt", "NOTICE")


def fetch(name: str, cache: Path | None) -> bytes:
    url, want = SOURCES[name]
    path = cache / name if cache else None
    if path and path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (pinned https)
            data = resp.read()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise SystemExit(f"{name}: sha256 {got} does not match the pinned {want} ({url})")
    return data


# ── shapefile / DBF (the subset these two sources use) ────────────────────


def read_dbf(data: bytes) -> list[dict[str, str]]:
    n_rec = struct.unpack("<I", data[4:8])[0]
    header_len, rec_len = struct.unpack("<HH", data[8:12])
    fields: list[tuple[str, int]] = []
    pos = 32
    while data[pos] != 0x0D:
        fname = data[pos:pos + 11].split(b"\0")[0].decode("ascii")
        fields.append((fname, data[pos + 16]))
        pos += 32
    rows = []
    for i in range(n_rec):
        rec = data[header_len + i * rec_len: header_len + (i + 1) * rec_len]
        off, row = 1, {}
        for fname, flen in fields:
            row[fname] = rec[off:off + flen].decode("utf-8", "replace").replace("\0", "").strip()
            off += flen
        rows.append(row)
    return rows


def read_shp_polygons(data: bytes) -> list[list[list[tuple[float, float]]]]:
    """Each record as a list of rings (shape type 5, Polygon)."""
    out = []
    pos = 100
    while pos < len(data):
        _, content_len = struct.unpack(">ii", data[pos:pos + 8])
        rec = data[pos + 8: pos + 8 + content_len * 2]
        pos += 8 + content_len * 2
        shape_type = struct.unpack("<i", rec[:4])[0]
        if shape_type == 0:
            out.append([])
            continue
        if shape_type != 5:
            raise SystemExit(f"unexpected shape type {shape_type}")
        n_parts, n_points = struct.unpack("<ii", rec[36:44])
        parts = list(struct.unpack(f"<{n_parts}i", rec[44:44 + 4 * n_parts]))
        pts_off = 44 + 4 * n_parts
        pts = [struct.unpack("<dd", rec[pts_off + 16 * k: pts_off + 16 * (k + 1)]) for k in range(n_points)]
        parts.append(n_points)
        out.append([pts[parts[j]:parts[j + 1]] for j in range(n_parts)])
    return out


# ── geometry ──────────────────────────────────────────────────────────────


def _perp(p, a, b) -> float:
    (x, y), (x1, y1), (x2, y2) = p, a, b
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
    return abs(dy * x - dx * y + x2 * y1 - y2 * x1) / (dx * dx + dy * dy) ** 0.5


def simplify(ring: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Iterative Douglas-Peucker; the ring stays closed."""
    if len(ring) < 5:
        return ring
    keep = [False] * len(ring)
    keep[0] = keep[-1] = True
    stack = [(0, len(ring) - 1)]
    while stack:
        i, j = stack.pop()
        best, idx = 0.0, -1
        for k in range(i + 1, j):
            d = _perp(ring[k], ring[i], ring[j])
            if d > best:
                best, idx = d, k
        if idx >= 0 and best > tol:
            keep[idx] = True
            stack += [(i, idx), (idx, j)]
    return [p for p, k in zip(ring, keep) if k]


def _round_ring(ring) -> list[list[float]]:
    out: list[list[float]] = []
    for x, y in ring:
        pt = [round(x, DECIMALS), round(y, DECIMALS)]
        if not out or out[-1] != pt:
            out.append(pt)
    if out and out[0] != out[-1]:
        out.append(out[0])
    return out


def _shoelace(ring) -> float:
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:])) / 2


def to_geometry(rings, tol: float, transform=None) -> dict | None:
    """Shapefile rings -> GeoJSON (Multi)Polygon. Outer rings are clockwise
    in a shapefile; each starts a polygon and following holes attach to it."""
    polys: list[list[list[list[float]]]] = []
    for ring in rings:
        if transform:
            ring = [transform(x, y) for x, y in ring]
        r = _round_ring(simplify(ring, tol))
        if len(r) < 4 or abs(_shoelace(r)) == 0:
            continue
        if _shoelace(ring) < 0 or not polys:   # clockwise in source = outer
            polys.append([r[::-1]])           # GeoJSON outer rings are counter-clockwise
        else:
            polys[-1].append(r[::-1])
    if not polys:
        return None
    if len(polys) == 1:
        return {"type": "Polygon", "coordinates": polys[0]}
    return {"type": "MultiPolygon", "coordinates": polys}


def _dump(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")


def _code(row: dict, key: str) -> str:
    v = row.get(f"{key}_EH") or row.get(key) or ""
    return "" if v.startswith("-99") else v


# ── builders ──────────────────────────────────────────────────────────────


def build_world(shp: bytes, dbf: bytes) -> bytes:
    feats = []
    for rings, row in zip(read_shp_polygons(shp), read_dbf(dbf)):
        a2, a3 = _code(row, "ISO_A2"), _code(row, "ISO_A3")
        if not a3:
            continue   # Northern Cyprus, Somaliland, Kosovo-as-disputed: no ISO code, nothing to join on
        geom = to_geometry(rings, WORLD_TOLERANCE)
        if geom:
            feats.append({"type": "Feature", "properties": {"iso_a2": a2, "iso_a3": a3, "name": row["NAME"]},
                          "geometry": geom})
    feats.sort(key=lambda f: f["properties"]["iso_a3"])
    return _dump({"type": "FeatureCollection", "features": feats})


def build_iso(dbf: bytes) -> bytes:
    seen: dict[str, dict] = {}
    for row in read_dbf(dbf):
        a2, a3 = _code(row, "ISO_A2"), _code(row, "ISO_A3")
        if len(a2) != 2 or len(a3) != 3 or a3 in seen:
            continue
        names = [row.get(k, "") for k in ("NAME", "NAME_LONG", "ADMIN", "FORMAL_EN", "NAME_SORT", "NAME_CIAWF")]
        alt = sorted({n for n in names[1:] if n and n != names[0]})
        seen[a3] = {"alpha2": a2, "alpha3": a3, "name": names[0], "alt_names": "|".join(alt)}
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["alpha2", "alpha3", "name", "alt_names"], lineterminator="\n")
    w.writeheader()
    for a3 in sorted(seen):
        w.writerow(seen[a3])
    return buf.getvalue().encode("utf-8")


def _inset(kind: str):
    """Move Alaska and Hawaii below the lower 48 (§1.3: AK/HI inset)."""
    if kind == "AK":
        cx, cy, s, tx, ty = -152.0, 62.0, 0.35, -113.0, 26.5
    else:  # HI
        cx, cy, s, tx, ty = -157.5, 20.5, 1.0, -104.0, 25.0

    def f(x, y):
        if x > 0:          # Aleutians across the antimeridian
            x -= 360
        return ((x - cx) * s + tx, (y - cy) * s + ty)
    return f


def build_us(zip_bytes: bytes) -> tuple[bytes, bytes]:
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    stem = f"cb_{CENSUS_VINTAGE}_us_state_20m"
    rows = read_dbf(z.read(f"{stem}.dbf"))
    shapes = read_shp_polygons(z.read(f"{stem}.shp"))
    if len(rows) != _US_EXPECTED:
        raise SystemExit(f"expected {_US_EXPECTED} state rows, got {len(rows)}")
    feats, table = [], []
    for rings, row in zip(shapes, rows):
        usps = row["STUSPS"]
        tf = _inset(usps) if usps in ("AK", "HI") else None
        geom = to_geometry(rings, US_TOLERANCE, tf)
        feats.append({"type": "Feature", "properties": {"usps": usps, "fips": row["STATEFP"], "name": row["NAME"]},
                      "geometry": geom})
        table.append({"usps": usps, "fips": row["STATEFP"], "name": row["NAME"]})
    feats.sort(key=lambda f: f["properties"]["usps"])
    table.sort(key=lambda r: r["usps"])
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["usps", "fips", "name"], lineterminator="\n")
    w.writeheader()
    w.writerows(table)
    return _dump({"type": "FeatureCollection", "features": feats}), buf.getvalue().encode("utf-8")


def build_echarts(tgz: bytes) -> dict[str, bytes]:
    out = {}
    with tarfile.open(fileobj=io.BytesIO(tgz), mode="r:gz") as tar:
        for member, dest in (("package/dist/echarts.min.js", "echarts.min.js"),
                             ("package/LICENSE", "licenses/echarts-LICENSE.txt"),
                             ("package/NOTICE", "licenses/echarts-NOTICE.txt"),
                             ("package/licenses/LICENSE-d3", "licenses/echarts-LICENSE-d3.txt")):
            fh = tar.extractfile(member)
            if fh is None:
                raise SystemExit(f"{member} missing from the ECharts tarball")
            out[dest] = fh.read()
    js = out["echarts.min.js"].decode("utf-8")
    # The export inlines this file in a <script>; a terminator would end it early.
    if "</script" in js.lower() or "<!--" in js:
        raise SystemExit("echarts.min.js contains a script terminator; it cannot be inlined")
    return out


NOTICE = f"""Bundled outcome assets: sources, versions and licences
=========================================================

Rebuilt by scripts/build_outcome_assets.py from the pinned sources below;
SHA256SUMS records the result. Nothing here is fetched at run time.

echarts.min.js
  Apache ECharts {ECHARTS_VERSION}, npm tarball {SOURCES["echarts.tgz"][0]}
  tarball sha256 {SOURCES["echarts.tgz"][1]}
  Licence: Apache License 2.0 (licenses/echarts-LICENSE.txt, NOTICE in
  licenses/echarts-NOTICE.txt; the d3 portions are BSD-3-Clause,
  licenses/echarts-LICENSE-d3.txt). Copied unmodified.

world_110m.json, iso3166.csv
  Natural Earth {NATURAL_EARTH_VERSION}, 1:110m admin-0 countries (geometry) and
  1:10m admin-0 countries (code table), github.com/nvkelso/natural-earth-vector.
  Licence: public domain (naturalearthdata.com/about/terms-of-use).
  Processing: properties cut to iso_a2/iso_a3/name (the *_EH codes, so
  France and Norway keep theirs); features without an ISO code dropped;
  Douglas-Peucker at {WORLD_TOLERANCE} degrees; coordinates rounded to {DECIMALS} dp.

us_states_20m.json, us_states.csv
  US Census Bureau cartographic boundary file cb_{CENSUS_VINTAGE}_us_state_20m
  (50 states, DC and Puerto Rico). Licence: US Government work, public domain.
  Processing: properties cut to usps/fips/name; Alaska scaled 0.35 and
  Alaska and Hawaii moved below the lower 48 as insets; Douglas-Peucker at
  {US_TOLERANCE} degrees; coordinates rounded to {DECIMALS} dp.
"""


def build(cache: Path | None) -> dict[str, bytes]:
    files = build_echarts(fetch("echarts.tgz", cache))
    files["world_110m.json"] = build_world(fetch("ne_110m.shp", cache), fetch("ne_110m.dbf", cache))
    files["iso3166.csv"] = build_iso(fetch("ne_10m.dbf", cache))
    files["us_states_20m.json"], files["us_states.csv"] = build_us(fetch("census_states.zip", cache))
    files["NOTICE"] = NOTICE.encode("utf-8")
    return files


def sums(files: dict[str, bytes]) -> bytes:
    return "".join(f"{hashlib.sha256(files[k]).hexdigest()}  {k}\n" for k in sorted(files)).encode()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", type=Path, help="directory to keep and reuse downloads")
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the committed SHA256SUMS")
    args = ap.parse_args(argv)
    files = build(args.cache)
    manifest = sums(files)
    if args.check:
        committed = (ASSETS / "SHA256SUMS").read_bytes()
        if committed != manifest:
            sys.stdout.write("rebuilt assets differ from SHA256SUMS\n")
            return 1
        sys.stdout.write("assets reproduce SHA256SUMS\n")
        return 0
    for name, data in files.items():
        dest = ASSETS / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    (ASSETS / "SHA256SUMS").write_bytes(manifest)
    for name in sorted(files):
        sys.stdout.write(f"{len(files[name]):>9}  {name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
