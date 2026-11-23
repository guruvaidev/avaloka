"""The repository carries exactly the lockfiles that a build reads.

ui/bun.lock sat beside ui/package-lock.json for months. No Dockerfile, CI job or
script installed from it -- ui/Dockerfile and the GitHub workflow run `npm ci` --
so nobody updated it, and it went on resolving package versions with critical
advisories that package-lock.json had long since moved past. Scanners that read
every lockfile reported them against the public repository all the same.

A lockfile nothing installs from only goes stale, so any lockfile not on this
list fails, whatever package manager wrote it. Adding one is a deliberate edit
here, made together with the build step that reads it.
"""
from __future__ import annotations

import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Lockfiles that a build actually installs from, and what reads each one.
ALLOWED_LOCKFILES = {
    "ui/package-lock.json",       # ui/Dockerfile `npm ci`; .github/workflows/ci.yml
    "deploy/avaloka/Chart.lock",  # helm dependency lock for the deploy/avaloka chart
    # Python: none. Every Python manifest is a requirements file or
    # pyproject.toml; a poetry/uv/pdm/Pipfile lock must be listed here first.
}

LOCKFILE_NAMES = {
    # JavaScript
    "package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml",
    "bun.lock", "bun.lockb",
    # Python
    "poetry.lock", "uv.lock", "Pipfile.lock", "pdm.lock", "pylock.toml",
    # Helm
    "Chart.lock",
}

# Installed or generated trees, never part of the source.
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache",
             ".pytest_cache", "dist", "build", ".output", ".nitro", ".tanstack"}


def _lockfiles() -> tuple[set[str], int]:
    found, seen = set(), 0
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith("venv")]
        seen += len(filenames)
        for name in filenames:
            if name in LOCKFILE_NAMES or name.startswith("pylock.") and name.endswith(".toml"):
                found.add(pathlib.Path(dirpath, name).relative_to(REPO_ROOT).as_posix())
    return found, seen


def test_only_listed_lockfiles_exist():
    found, seen = _lockfiles()
    # Guard against walking the wrong directory and passing on an empty tree.
    assert seen > 500, f"walked only {seen} files under {REPO_ROOT}"
    unexpected = sorted(found - ALLOWED_LOCKFILES)
    assert not unexpected, (
        f"lockfiles that no build installs from: {unexpected}. Delete them, or wire them "
        "into a build and list them in ALLOWED_LOCKFILES with what reads them."
    )


def test_listed_lockfiles_are_present():
    found, _ = _lockfiles()
    missing = sorted(ALLOWED_LOCKFILES - found)
    assert not missing, f"listed lockfiles missing from the tree: {missing}"
