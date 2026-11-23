"""Bundled outcome assets: every file is the one SHA256SUMS records, every
licence the NOTICE names is shipped, and the package declares them as data.

``scripts/build_outcome_assets.py --check`` rebuilds from the pinned sources
(network) and compares with SHA256SUMS; this offline test holds the
committed bytes to the same manifest.

Bite (run by hand, in the PR): append one byte to ``us_states.csv`` and
``test_assets_match_sha256sums`` goes red. Remove the ``.gitignore`` exception
for ``avaloka/outcomes/assets/*.csv`` and ``test_every_asset_is_tracked_by_git``
goes red.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import tomllib

import pytest

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "avaloka" / "outcomes" / "assets"


def _manifest() -> dict[str, str]:
    out = {}
    for line in (ASSETS / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        out[name] = digest
    return out


def test_assets_match_sha256sums():
    m = _manifest()
    assert {"echarts.min.js", "world_110m.json", "us_states_20m.json", "iso3166.csv", "us_states.csv",
            "NOTICE"} <= set(m)
    for name, digest in m.items():
        assert hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() == digest, name


# Written by hand on feat/2.0-outcome-spec and held to its golden vectors there,
# not rebuilt from an upstream source.
NOT_BUILT = {"SHA256SUMS", "outcome-runtime.js"}


def test_every_shipped_asset_is_in_the_manifest():
    shipped = {str(p.relative_to(ASSETS)) for p in ASSETS.rglob("*") if p.is_file()} - NOT_BUILT
    assert shipped == set(_manifest())


def test_notice_names_source_version_and_licence():
    notice = (ASSETS / "NOTICE").read_text()
    for needle in ("Apache ECharts 5.6.0", "Apache License 2.0", "Natural Earth v5.1.2", "public domain",
                   "cb_2023_us_state_20m"):
        assert needle in notice
    assert "Apache License" in (ASSETS / "licenses" / "echarts-LICENSE.txt").read_text()


def test_every_asset_is_tracked_by_git():
    # A file on disk passes the hash test in a working copy even when an ignore
    # rule keeps it out of the commit: `*.csv` did exactly that to iso3166.csv
    # and us_states.csv, and every clone then raised FileNotFoundError.
    if not (REPO / ".git").exists() or shutil.which("git") is None:
        pytest.skip("not a git checkout")
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "--", str(ASSETS)],
                         capture_output=True, check=True).stdout.decode()
    tracked = {str((REPO / f).relative_to(ASSETS)) for f in out.split("\0") if f}
    missing = (set(_manifest()) | NOT_BUILT) - tracked
    assert not missing, f"assets not tracked by git (check .gitignore): {sorted(missing)}"


@pytest.mark.parametrize("pattern", ["outcomes/assets/*.js", "outcomes/assets/*.json", "outcomes/assets/*.csv",
                                     "outcomes/assets/licenses/*.txt", "outcomes/assets/NOTICE"])
def test_package_data_ships_the_assets(pattern):
    data = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["avaloka"]
    assert pattern in data
