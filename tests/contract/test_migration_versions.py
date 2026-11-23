"""Migration versions stay unique and forward-only under the migrate hook's tracker.

The chart's migrate hook (deploy/helm/avaloka/files/db-migrate.sh) records each
file by version (its name without .sql) in avaloka_meta.schema_migrations and
never runs a recorded version again. That makes a reused or out-of-order version
dangerous in ways the old run-everything hook hid:

* two files with one Supabase CLI version (the 14-digit prefix) collide in
  supabase_migrations.schema_migrations on CLI-managed databases;
* a new file that reuses a baseline version is never run on an existing install;
* a new file that sorts before the newest baseline file runs in a different order
  on a fresh install than on an upgraded one;
* a new file present in only one of the two copies, or different between them,
  means the chart and the UI source disagree about the schema.

Baseline files (pre-tracking) are exempt from the copy-equality rule: two of
them already differ between the copies and are frozen.

One shared prefix is allowed, and only that one: KNOWN_DUPLICATES, the pair 1.6
shipped under 20261101120000 (#649's support_admin_from_roles and #650's
plan_enforced_org_features), matching 1.6's tests/test_migration_versions_unique.py.
Neither may be renamed: the hook records the full file name, so a renamed copy
would run a second time on every 1.6 database that already recorded it. Both
are on 2.x now; tests/test_migration_versions_unique.py (ported from 1.6) pins
that the pair is present, unchanged, in both copies.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
UI_DIR = REPO_ROOT / "ui" / "supabase" / "migrations"
CHART_DIR = REPO_ROOT / "deploy" / "helm" / "avaloka" / "files" / "supabase-migrations"
BASELINE = REPO_ROOT / "deploy" / "helm" / "avaloka" / "files" / "db-migrations.baseline"
_CLI_VERSION = re.compile(r"^(\d{14})_")

#: The one shared prefix that shipped on 1.6 before a guard existed (#649, #650).
#: Never rename either file, and never add to this: give a new file an unused version.
KNOWN_DUPLICATES = {
    "20261101120000": {
        "20261101120000_plan_enforced_org_features",
        "20261101120000_support_admin_from_roles",
    },
    # Also shipped on 1.6 before the guard (tests/test_migration_versions_unique.py
    # carries the same pair); merged files are never renamed.
    "20261103120000": {
        "20261103120000_admin_no_cross_tenant",
        "20261103120000_bg01_no_org_for_self_serve",
    },
}


def _versions(d: pathlib.Path) -> list[str]:
    return sorted(p.stem for p in d.glob("*.sql"))


def _baseline() -> list[str]:
    assert BASELINE.is_file(), "no frozen baseline (db-migrations.baseline): versions are not tracked"
    return [l for l in BASELINE.read_text().splitlines() if l and not l.startswith("#")]


def test_no_two_migrations_share_a_version() -> None:
    for d in (UI_DIR, CHART_DIR):
        stems = _versions(d)
        lowered = [s.lower() for s in stems]
        assert len(lowered) == len(set(lowered)), f"{d.name}: versions differ only by case"
        by_prefix: dict[str, set[str]] = {}
        for stem in stems:
            if (m := _CLI_VERSION.match(stem)):
                by_prefix.setdefault(m.group(1), set()).add(stem)
        unexpected = {
            p: sorted(names) for p, names in by_prefix.items()
            if len(names) > 1 and not names <= KNOWN_DUPLICATES.get(p, set())
        }
        assert not unexpected, (
            f"{d.relative_to(REPO_ROOT)}: CLI version(s) used by more than one file: {unexpected}. "
            "The Supabase CLI keys migrations by this prefix alone. Give the new file a later, unused version."
        )
        # The allowance covers the known pair only: any file under a known prefix
        # must be one of the pair (a lone third file under it fails too).
        for p, allowed in KNOWN_DUPLICATES.items():
            stray = sorted(by_prefix.get(p, set()) - allowed)
            assert not stray, f"{d.relative_to(REPO_ROOT)}: {stray} reuse the known duplicate prefix {p}"


def test_the_known_pair_is_identical_in_both_copies() -> None:
    """Whichever members of the known pair are present are present in both copies, byte-identical."""
    for allowed in KNOWN_DUPLICATES.values():
        for stem in sorted(allowed):
            ui, chart = UI_DIR / f"{stem}.sql", CHART_DIR / f"{stem}.sql"
            assert ui.is_file() == chart.is_file(), f"{stem}: present in only one of the two copies"
            if ui.is_file():
                assert ui.read_bytes() == chart.read_bytes(), f"{stem}: the UI and chart copies differ"


def test_baseline_is_unique_and_names_shipped_files() -> None:
    base = _baseline()
    assert len(base) == len(set(base)), "the baseline lists a version twice"
    missing = sorted(set(base) - set(_versions(CHART_DIR)))
    assert not missing, f"baseline names migrations the chart does not ship: {missing[:3]}"


def test_new_migrations_are_forward_only_and_identical_in_both_copies() -> None:
    base = set(_baseline())
    newest_base = max(m.group(1) for v in base if (m := _CLI_VERSION.match(v)))
    new = sorted((set(_versions(CHART_DIR)) | set(_versions(UI_DIR))) - base)
    for v in new:
        m = _CLI_VERSION.match(v)
        assert m, f"{v}: a tracked migration needs a 14-digit timestamp version"
        assert m.group(1) > newest_base, (
            f"{v}: sorts before the newest baseline migration ({newest_base}), so a fresh "
            "install and an upgraded one would apply it in different orders"
        )
        ui, chart = UI_DIR / f"{v}.sql", CHART_DIR / f"{v}.sql"
        assert ui.is_file() and chart.is_file(), f"{v}: present in only one of the two copies"
        assert ui.read_bytes() == chart.read_bytes(), f"{v}: the UI and chart copies differ"


def test_rules_catch_an_out_of_order_version() -> None:
    assert not ("20260101000000" > "20260202000000")


# ---------------------------------------------------------------------------
# Order: no new migration may sort before one already merged
# ---------------------------------------------------------------------------
#
# `supabase db push` refuses a local migration older than the newest one already
# applied (it needs --include-all, docs/deployment.md). Two checks, as on 1.6
# (tests/test_migration_versions_unique.py there):
#
# * against the target branch, when the pipeline provides it: a pull-request
#   pipeline fetches its destination branch and exports
#   AVALOKA_MIGRATION_ORDER_BASE=origin/<branch> (the CI pipeline's test step).
#   Every migration the change adds must sort after the newest version on it;
# * always, against MERGED_MIGRATIONS: the tracked (post-baseline) migrations in
#   merge order. A change that adds a migration appends it; the list must stay in
#   version order.
#
# Baseline files and the known 20261101120000 pair are exempt.

#: Tracked (post-baseline) migrations on this line, in version order. A change
#: that adds a migration adds it here in place. After the develop-1.6 merge-forward
#: the 1.6 names are the names (the 1.7 copies of ml_jobs_endpoints,
#: chat_credentials_redacted and restore_support_ticket_policies were duplicates
#: of 1.6 files under other versions and are gone); 20261116120000_dataset_catalog
#: is 1.7's own.
MERGED_MIGRATIONS = (
    "20261101120000_plan_enforced_org_features",
    "20261101120000_support_admin_from_roles",
    "20261101140000_plan_rules_pro_collab",
    "20261102100000_owner_only_project_visibility",
    "20261103090000_restore_support_ticket_policies",
    "20261103100000_plan_enforce_comments_notifications",
    "20261103110000_plan_enforce_analysis_share_links",
    "20261103120000_admin_no_cross_tenant",
    "20261103120000_bg01_no_org_for_self_serve",
    "20261103130000_fix_bg03_analysis_visibility",
    "20261103140000_fix_bg01_org_membership",
    "20261103145000_dedupe_seed_departments_roles",
    "20261103150000_bg05_bg06_roles_departments",
    "20261103160000_plan_descriptions_true",
    "20261106120000_standalone_team_ownership",
    "20261106130000_remove_team_membership_role",
    "20261106140000_team_description",
    "20261106150000_team_invitation_delivery_and_plan_gate",
    "20261106160000_team_lead_profile",
    "20261106170000_standalone_analysis_team_sharing",
    "20261106180000_reconcile_team_invitation_signup",
    "20261106190000_remove_legacy_team_account_trigger",
    "20261108090000_get_my_report_access_readonly",
    "20261108100000_bg01_cleanup_selfserve_orgs",
    "20261111200000_project_team_sharing",
    "20261111210000_professional_project_plan_access",
    "20261111220000_analysis_move_requires_edit",
    "20261111230000_shared_analysis_display_profiles",
    "20261111233000_team_member_share_revocation",
    "20261116120000_dataset_catalog",
    "20261116130000_cloud_datasets_sealed_keys",
    "20261116140000_team_members_any_plan",
    "20261117110000_org_name_keys_per_org",
    "20261117120000_celery_task_results",
    "20261117130000_ml_jobs_endpoints",
    "20261121090000_api_roles_no_truncate",
    "20261121100000_chat_credentials_redacted",
    "20261121110000_background_jobs",
)

#: Migrations carried from develop-1.6 byte for byte, under their 1.6 names: every
#: tracked file except 2.x's own. A 1.6 database that upgrades has already recorded
#: them by name, so they are never renumbered, and they are exempt from the
#: target-branch order check (a pull request onto a 2.x branch adds them older than
#: that branch's newest file). MERGED_MIGRATIONS above is in version order anyway;
#: tests/test_migration_versions_unique.py checks that over the whole list.
NATIVE_2X = {"20261116120000_dataset_catalog"}
# Everything carried from develop-1.6: the tracked list's 1.6 entries, plus the 1.6
# migrations the list does not track (tests/test_migration_versions_unique.py names them).
from tests.test_migration_versions_unique import CARRIED_FROM_1_6  # noqa: E402

FROM_1_6: set[str] = (set(MERGED_MIGRATIONS) - NATIVE_2X) | set(CARRIED_FROM_1_6)

_KNOWN_PAIR = {n for names in KNOWN_DUPLICATES.values() for n in names}


def _version_of(stem: str) -> str | None:
    m = _CLI_VERSION.match(stem)
    return m.group(1) if m else None


def test_every_tracked_migration_is_listed_in_merge_order() -> None:
    base = set(_baseline())
    for d in (UI_DIR, CHART_DIR):
        unlisted = sorted(set(_versions(d)) - base - set(MERGED_MIGRATIONS))
        assert not unlisted, (
            f"{d.relative_to(REPO_ROOT)}: {unlisted} not in MERGED_MIGRATIONS. "
            "Append a new migration to the end of that list in the same change."
        )
    missing = [n for n in MERGED_MIGRATIONS if not all((d / f"{n}.sql").is_file() for d in (UI_DIR, CHART_DIR))]
    assert not missing, f"MERGED_MIGRATIONS names file(s) missing from a migrations directory: {missing}"


def test_merged_migrations_are_in_version_order() -> None:
    out_of_order = []
    native = [n for n in MERGED_MIGRATIONS if n not in FROM_1_6]
    for earlier, later in zip(native, native[1:]):
        ve, vl = _version_of(earlier), _version_of(later)
        assert ve and vl, f"untimestamped tracked migration: {earlier if not ve else later}"
        if vl < ve or (vl == ve and not {earlier, later} <= KNOWN_DUPLICATES.get(vl, set())):
            out_of_order.append(f"{later} after {earlier}")
    assert not out_of_order, (
        f"migration(s) sort before one merged earlier: {out_of_order}. `supabase db push` refuses "
        "these without --include-all (docs/deployment.md). Give the new file a later version."
    )


def test_new_migrations_sort_after_the_target_branch() -> None:
    ref = os.environ.get("AVALOKA_MIGRATION_ORDER_BASE", "").strip()
    if not ref:
        pytest.skip("AVALOKA_MIGRATION_ORDER_BASE not set (only pull-request pipelines set it); "
                    "MERGED_MIGRATIONS still checks the order")
    for d in (UI_DIR, CHART_DIR):
        rel = d.relative_to(REPO_ROOT).as_posix()
        out = subprocess.run(["git", "ls-tree", "--name-only", f"{ref}:{rel}"],
                             cwd=REPO_ROOT, capture_output=True, text=True)
        assert out.returncode == 0, f"cannot read {rel} at {ref}: {out.stderr.strip()}"
        on_target = {pathlib.Path(n).stem for n in out.stdout.split() if n.endswith(".sql")}
        newest = max(v for n in on_target if (v := _version_of(n)))
        added = sorted(set(_versions(d)) - on_target - _KNOWN_PAIR - FROM_1_6)
        older = [n for n in added if (v := _version_of(n)) and v <= newest]
        assert not older, (
            f"{rel}: {older} sort at or before {newest}, the newest migration on {ref}. "
            f"`supabase db push` refuses them without --include-all. Renumber them after {newest}."
        )
