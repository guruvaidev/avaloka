"""Supabase migration versions are unique (hermetic).

Two trackers apply ui/supabase/migrations (and the chart copy in
deploy/helm/avaloka/files/supabase-migrations):

* the chart's migrate hook, deploy/helm/avaloka/files/db-migrate.sh, records
  each file in avaloka_meta.schema_migrations under its FULL file name without
  ``.sql`` (``v="$(basename "$f" .sql)"``), so two files that share a numeric
  prefix are two different versions to it;
* the Supabase CLI records supabase_migrations.schema_migrations under the
  NUMERIC prefix only, so two files that share a prefix are one version to it.
  ``supabase db push`` then cannot record both.

So a shared prefix is harmless to the chart and breaks CLI-managed projects. One
such pair already shipped (below). It stays as it is: both files are merged and
may be applied, and because the hook keys by full name, renaming either file
would make the hook run it a second time. Any NEW shared prefix fails here.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DIRS = (
    REPO_ROOT / "ui" / "supabase" / "migrations",
    REPO_ROOT / "deploy" / "helm" / "avaloka" / "files" / "supabase-migrations",
)
HOOK = REPO_ROOT / "deploy" / "helm" / "avaloka" / "files" / "db-migrate.sh"

#: Shipped before this guard existed (#649 and #650, both merged to develop-1.6).
#: Keep, never rename: see the module docstring, and the operator note in
#: docs/deployment.md ("Duplicate migration version 20261101120000").
KNOWN_DUPLICATES = {
    "20261101120000": {
        "20261101120000_plan_enforced_org_features.sql",
        "20261101120000_support_admin_from_roles.sql",
    },
}

_VERSIONED = re.compile(r"^(\d{14})_.+\.sql$")


def _by_version(directory: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for path in directory.glob("*.sql"):
        m = _VERSIONED.match(path.name)
        if m:
            out[m.group(1)].add(path.name)
    return out


def test_no_new_duplicate_migration_versions():
    for directory in DIRS:
        assert directory.is_dir(), directory
        dupes = {v: names for v, names in _by_version(directory).items() if len(names) > 1}
        unexpected = {v: sorted(n) for v, n in dupes.items() if KNOWN_DUPLICATES.get(v) != n}
        assert not unexpected, (
            f"{directory.relative_to(REPO_ROOT)}: migration version(s) shared by more than one file "
            f"{unexpected}. The Supabase CLI keys migrations by this numeric prefix alone, so "
            "`supabase db push` cannot record both. Give the new file a later, unused version."
        )


def test_the_known_pair_is_still_exactly_the_shipped_pair():
    """The allowlist does not silently grow: the pair is present, unchanged, in both copies."""
    for directory in DIRS:
        assert _by_version(directory).get("20261101120000") == KNOWN_DUPLICATES["20261101120000"], directory


def test_the_chart_hook_keys_by_full_file_name():
    """Why the pair is not renamed: the hook records the whole name, so a rename re-runs the file."""
    text = HOOK.read_text()
    assert 'v="$(basename "$f" .sql)"' in text
    assert "INSERT INTO avaloka_meta.schema_migrations (version)" in text
