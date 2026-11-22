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

import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest

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
    # Merged on develop-1.6; a migration that has run is never renamed (the
    # tracker keys on the file name, so a rename would apply it again).
    "20261101120000": {
        "20261101120000_plan_enforced_org_features.sql",
        "20261101120000_support_admin_from_roles.sql",
    },
    "20261103120000": {
        "20261103120000_admin_no_cross_tenant.sql",
        "20261103120000_bg01_no_org_for_self_serve.sql",
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


# ---------------------------------------------------------------------------
# A renamed migration ships once: never beside the copy it replaced
# ---------------------------------------------------------------------------
#
# A migration is renumbered by giving the same file a new version prefix (BG-01,
# BG-03 and BG-05/06 went 20261101* -> 20261102* -> 20261103*). Because the hook
# keys on the full file name, an old copy that is put back runs as well, in its
# old position: the old BG-05/06 created its unique indexes before the dedupe that
# the renumbered copy waits for, so an install with duplicate department names
# failed the upgrade. Two rules, per directory:
#
# * same purpose: two files with the same name after the version prefix. That
#   name says what the migration does and a renumbering keeps it. A migration
#   that supersedes another under a new name (20261101140000_plan_rules_pro_collab
#   over 20261101120000_plan_enforced_org_features) is new work meant to run after
#   it, and is not matched;
# * same content: two files whose SQL is identical once comments and whitespace
#   are removed, i.e. a copy under another name.
#
# Removing a copy the hook has recorded re-runs nothing: the hook only applies
# files it has not recorded, and an install that never ran the copy gets the
# successor alone.

_SQL_COMMENT = re.compile(r"/\*.*?\*/|--[^\n]*", re.S)

#: Same statement, shipped twice before this guard. Each is one idempotent
#: ADD COLUMN IF NOT EXISTS, so the second run changes nothing; kept as shipped.
KNOWN_SAME_SQL = (
    {"20260617090108_610b2197-44f9-44ce-8b81-41785062db9d.sql", "20261106140000_team_description.sql"},
    {"20260726160620_31b8d83f-9eb3-419f-ae96-138d4393cacc.sql", "20260727101213_3688193f-ebd6-4f67-ad23-3f9992c767cc.sql"},
)


def _purpose(name: str) -> str:
    m = re.match(r"^\d{14}_(.+)\.sql$", name)
    return m.group(1) if m else name


def _normalised_sql(path: Path) -> str:
    return " ".join(_SQL_COMMENT.sub(" ", path.read_text()).split()).lower()


def test_no_migration_ships_beside_its_renamed_successor():
    for directory in DIRS:
        by_purpose: dict[str, set[str]] = defaultdict(set)
        for path in directory.glob("*.sql"):
            by_purpose[_purpose(path.name)].add(path.name)
        twice = {p: sorted(n) for p, n in by_purpose.items() if len(n) > 1}
        assert not twice, (
            f"{directory.relative_to(REPO_ROOT)}: one migration ships under more than one version {twice}. "
            "The hook runs every name it has not recorded, so the old copy runs too, in its old "
            "position. Ship only the renumbered file."
        )


def test_no_two_migrations_carry_the_same_sql():
    for directory in DIRS:
        by_sql: dict[str, set[str]] = defaultdict(set)
        for path in directory.glob("*.sql"):
            by_sql[_normalised_sql(path)].add(path.name)
        same = sorted(sorted(n) for n in by_sql.values() if len(n) > 1)
        unexpected = [n for n in same if set(n) not in KNOWN_SAME_SQL]
        assert not unexpected, (
            f"{directory.relative_to(REPO_ROOT)}: migrations with identical SQL {unexpected}: a copy "
            "shipped beside the file it was copied from. Ship one of them."
        )
        assert sorted(sorted(k) for k in KNOWN_SAME_SQL) == same, (
            f"{directory.relative_to(REPO_ROOT)}: KNOWN_SAME_SQL no longer matches what ships; "
            "drop the entries that are gone"
        )


# ---------------------------------------------------------------------------
# Order: no new migration may sort before one already merged
# ---------------------------------------------------------------------------
#
# The Supabase CLI applies migrations in version order and `supabase db push`
# refuses a local migration older than the newest one already applied to the
# project (it needs --include-all). That happened when an unmerged PR's
# migrations ended up older than a repair merged after them. Two checks:
#
# * against the target branch, when the pipeline provides it: a pull-request
#   pipeline fetches its destination branch and exports
#   AVALOKA_MIGRATION_ORDER_BASE=origin/<branch> (the CI pipeline's test step).
#   Every migration the change adds must sort after the newest version
#   already on that branch. Locally, set the same variable to the target
#   (e.g. AVALOKA_MIGRATION_ORDER_BASE=origin/develop-1.6) to run it; unset,
#   it skips;
# * always, against MERGED_MIGRATIONS below: every tracked (post-baseline)
#   migration, listed in version order. A PR that adds a migration adds it
#   here. The list proves each file is accounted for; it cannot show merge
#   order, so the target-branch check above is what catches a late, older file.
#
# Merged out of version order on 1.6 (kept, never renamed: the chart hook keys
# on the full file name, so a rename would run the file again):
# 20261103145000_dedupe_seed_departments_roles.sql merged after
# 20261103160000_plan_descriptions_true.sql, and the 20261106* team series
# merged after 20261108100000_bg01_cleanup_selfserve_orgs.sql. A CLI-managed
# project that recorded the newer files first needs `supabase db push
# --include-all` once to apply them (docs/deployment.md). The chart hook is
# unaffected.
#
# Baseline files (db-migrations.baseline) predate tracking and are exempt, and so
# is the known 20261101120000 pair (KNOWN_DUPLICATES).

BASELINE = REPO_ROOT / "deploy" / "helm" / "avaloka" / "files" / "db-migrations.baseline"

#: Tracked migrations on this branch, in version order. Add yours in place.
MERGED_MIGRATIONS = (
    "20261101120000_plan_enforced_org_features.sql",
    "20261101120000_support_admin_from_roles.sql",
    "20261101140000_plan_rules_pro_collab.sql",
    "20261102100000_owner_only_project_visibility.sql",
    "20261103090000_restore_support_ticket_policies.sql",
    "20261103100000_plan_enforce_comments_notifications.sql",
    "20261103110000_plan_enforce_analysis_share_links.sql",
    "20261103120000_admin_no_cross_tenant.sql",
    "20261103120000_bg01_no_org_for_self_serve.sql",
    "20261103130000_fix_bg03_analysis_visibility.sql",
    "20261103140000_fix_bg01_org_membership.sql",
    "20261103145000_dedupe_seed_departments_roles.sql",
    "20261103150000_bg05_bg06_roles_departments.sql",
    "20261103160000_plan_descriptions_true.sql",
    "20261106120000_standalone_team_ownership.sql",
    "20261106130000_remove_team_membership_role.sql",
    "20261106140000_team_description.sql",
    "20261106150000_team_invitation_delivery_and_plan_gate.sql",
    "20261106160000_team_lead_profile.sql",
    "20261106170000_standalone_analysis_team_sharing.sql",
    "20261106180000_reconcile_team_invitation_signup.sql",
    "20261106190000_remove_legacy_team_account_trigger.sql",
    "20261108090000_get_my_report_access_readonly.sql",
    "20261108100000_bg01_cleanup_selfserve_orgs.sql",
    "20261111200000_project_team_sharing.sql",
    "20261111210000_professional_project_plan_access.sql",
    "20261111220000_analysis_move_requires_edit.sql",
    "20261111230000_shared_analysis_display_profiles.sql",
    "20261111233000_team_member_share_revocation.sql",
    "20261116130000_cloud_datasets_sealed_keys.sql",
    "20261116140000_team_members_any_plan.sql",
    "20261117110000_org_name_keys_per_org.sql",
    "20261117120000_celery_task_results.sql",
    "20261117130000_ml_jobs_endpoints.sql",
    "20261121090000_api_roles_no_truncate.sql",
    "20261121100000_chat_credentials_redacted.sql",
    "20261121110000_background_jobs.sql",
)

_KNOWN_PAIR = {name for names in KNOWN_DUPLICATES.values() for name in names}


def _baseline_names() -> set[str]:
    assert BASELINE.is_file(), BASELINE
    return {f"{l}.sql" for l in BASELINE.read_text().splitlines() if l and not l.startswith("#")}


def _version(name: str) -> str | None:
    m = _VERSIONED.match(name)
    return m.group(1) if m else None


def _tracked(directory: Path) -> set[str]:
    return {p.name for p in directory.glob("*.sql")} - _baseline_names()


def test_every_tracked_migration_is_listed_in_merge_order():
    for directory in DIRS:
        unlisted = sorted(_tracked(directory) - set(MERGED_MIGRATIONS))
        assert not unlisted, (
            f"{directory.relative_to(REPO_ROOT)}: {unlisted} not in MERGED_MIGRATIONS. "
            "Append a new migration to the end of that list in the same change."
        )
    missing = [n for n in MERGED_MIGRATIONS if not all((d / n).is_file() for d in DIRS)]
    assert not missing, f"MERGED_MIGRATIONS names file(s) missing from a migrations directory: {missing}"


def test_merged_migrations_are_in_version_order():
    out_of_order = []
    for earlier, later in zip(MERGED_MIGRATIONS, MERGED_MIGRATIONS[1:]):
        ve, vl = _version(earlier), _version(later)
        assert ve and vl, f"untimestamped tracked migration: {earlier if not ve else later}"
        if vl < ve or (vl == ve and not {earlier, later} <= KNOWN_DUPLICATES.get(vl, set())):
            out_of_order.append(f"{later} after {earlier}")
    assert not out_of_order, (
        f"migration(s) sort before one merged earlier: {out_of_order}. `supabase db push` refuses "
        "these without --include-all (docs/deployment.md). Give the new file a version later than "
        f"{max(_version(n) for n in MERGED_MIGRATIONS)}."
    )


def _target_versions(ref: str, directory: Path) -> dict[str, str]:
    rel = directory.relative_to(REPO_ROOT).as_posix()
    out = subprocess.run(
        ["git", "ls-tree", "--name-only", f"{ref}:{rel}"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert out.returncode == 0, f"cannot read {rel} at {ref}: {out.stderr.strip()}"
    return {n: v for n in out.stdout.split() if (v := _version(n))}


def test_new_migrations_sort_after_the_target_branch():
    ref = os.environ.get("AVALOKA_MIGRATION_ORDER_BASE", "").strip()
    if not ref:
        pytest.skip("AVALOKA_MIGRATION_ORDER_BASE not set (pull-request pipelines set it; set it to the "
                    "target branch to run this locally). Nothing else checks migration order: "
                    "MERGED_MIGRATIONS is in version order and only proves every file is listed")
    for directory in DIRS:
        on_target = _target_versions(ref, directory)
        newest = max(on_target.values())
        added = sorted({p.name for p in directory.glob("*.sql")} - set(on_target) - _KNOWN_PAIR)
        older = [n for n in added if (v := _version(n)) and v <= newest]
        assert not older, (
            f"{directory.relative_to(REPO_ROOT)}: {older} sort at or before {newest}, the newest "
            f"migration on {ref}. `supabase db push` refuses them without --include-all. "
            f"Renumber them after {newest}."
        )
