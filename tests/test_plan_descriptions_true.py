"""Plan descriptions claim only what each plan includes (owner ruling 2026-11-03).

Migration 20261103160000_plan_descriptions_true.sql rewrites the taglines that
20260921000000_seed_plans.sql seeded, and only those: a row is matched by its
seeded id AND its exact seeded text, so a description an operator changed is
left alone.

Against real Postgres, with the chart's own migration set applied by the
chart's own migrate hook, reusing the harness of
tests/test_plan_enforced_org_features.py. The upgrade case is the realistic
one: the set without this migration is applied, an operator customises one
description, then the hook runs again with this migration present.

Needs AVALOKA_TEST_PG_ADMIN_URL (a role that may CREATE DATABASE), psql and sh.
Skips on a laptop without them and FAILS when CI is set. The two file checks
at the top need nothing.
"""

from __future__ import annotations

import pathlib

import pytest

from tests.test_plan_enforced_org_features import (BASELINE, MIGRATIONS, REPO_ROOT,
                                                   _chart_set, _drop, _new_db, _ok)

VERSION = "20261103160000_plan_descriptions_true"
UI_COPY = REPO_ROOT / "ui" / "supabase" / "migrations"

#: The seeded rows: id -> (seeded text, rewritten text).
SEEDED = {
    "b6697d97-cee2-4e5c-b89b-3179acad41fd": (
        "Perfect for individuals exploring AI-powered data analysis.",
        "Conversational data analysis, Auto Insights and file connectors.",
    ),
    "2abfb8c8-e811-4f7c-aa1a-8f70b773d7db": (
        "For growing teams who need advanced ecommerce and reporting capabilities.",
        "Scheduling, Reports, database and cloud connectors, and teams that share and comment on analyses.",
    ),
    "9b66fe6b-ef14-4de3-8a4f-b2f27befaeb5": (
        "Chosen by 300K+ users.",
        "Scheduling, Reports, database and cloud connectors, and teams that share and comment on analyses.",
    ),
    "1905809c-e56b-49dd-9655-1f303cd1c452": (
        "Built for organizations with advanced AI and security needs.",
        "Everything in Professional, plus departments, roles, branches and app users.",
    ),
    "e79a2977-4939-47f5-b0c7-874b45fe3917": (
        "Chosen by 300+ users.",
        "Everything in Professional, plus departments, roles, branches and app users.",
    ),
}
CUSTOMISED_ID = "2abfb8c8-e811-4f7c-aa1a-8f70b773d7db"
CUSTOM_TEXT = "Our own Professional tagline, set by the operator."


# ---- the files (no database) ----------------------------------------------------

def test_both_copies_exist_and_are_byte_identical():
    chart = MIGRATIONS / f"{VERSION}.sql"
    ui = UI_COPY / f"{VERSION}.sql"
    assert chart.is_file() and ui.is_file()
    assert chart.read_bytes() == ui.read_bytes()


def test_version_sorts_after_the_last_reconciled_migration_and_is_not_in_the_baseline():
    assert VERSION[:14] > "20261103150000"
    baseline = {ln.strip() for ln in BASELINE.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    assert VERSION not in baseline


# ---- real Postgres ----------------------------------------------------------------

def _descriptions(d) -> dict[str, str]:
    ids = ",".join(f"'{i}'" for i in SEEDED)
    rows = d.scalar(f"SELECT id || '|' || coalesce(description, '') FROM public.plans WHERE id IN ({ids}) ORDER BY id;")
    return dict(line.split("|", 1) for line in rows.splitlines())


@pytest.fixture(scope="module")
def upgraded(tmp_path_factory):
    """An install migrated without this migration, one description customised,
    then the hook run again with it present, then a third time."""
    d = _new_db()
    try:
        mig = _chart_set(tmp_path_factory.mktemp("plandesc"))
        new = mig / f"{VERSION}.sql"
        assert new.is_file(), f"migration {VERSION} is missing from the chart's migration set"
        held = new.read_bytes()
        new.unlink()
        _ok(d.run_hook(mig))
        before = _descriptions(d)
        d.psql(f"UPDATE public.plans SET description = '{CUSTOM_TEXT}' WHERE id = '{CUSTOMISED_ID}';")
        new.write_bytes(held)
        upgrade_log = _ok(d.run_hook(mig))
        after = _descriptions(d)
        rerun_log = _ok(d.run_hook(mig))
        yield d, before, after, upgrade_log, rerun_log
    finally:
        _drop(d)


@pytest.mark.integration
def test_the_seed_wrote_the_old_text_before_the_upgrade(upgraded):
    _, before, _, _, _ = upgraded
    assert before == {i: old for i, (old, _) in SEEDED.items()}


@pytest.mark.integration
def test_seeded_descriptions_are_rewritten(upgraded):
    _, _, after, _, _ = upgraded
    for plan_id, (_, new) in SEEDED.items():
        if plan_id != CUSTOMISED_ID:
            assert after[plan_id] == new, plan_id
    assert not any("ecommerce" in text for text in after.values())


@pytest.mark.integration
def test_an_operator_customised_description_is_left_alone(upgraded):
    _, _, after, _, _ = upgraded
    assert after[CUSTOMISED_ID] == CUSTOM_TEXT


@pytest.mark.integration
def test_the_hook_applies_it_once(upgraded):
    _, _, _, upgrade_log, rerun_log = upgraded
    assert VERSION in upgrade_log
    assert VERSION not in rerun_log


@pytest.mark.integration
def test_running_the_sql_again_changes_nothing(upgraded):
    d = upgraded[0]
    sql = (MIGRATIONS / f"{VERSION}.sql").read_text()
    stamp = "SELECT string_agg(id || '|' || coalesce(description, '') || '|' || updated_at, ',' ORDER BY id) FROM public.plans;"
    before = d.scalar(stamp)
    result = d.psql(sql, check=True)
    # psql -q hides the command tag, so compare state: no row, and no updated_at, moved.
    assert d.scalar(stamp) == before, result.stdout + result.stderr


@pytest.mark.integration
def test_on_a_fresh_install_every_seeded_row_is_rewritten(tmp_path_factory):
    d = _new_db()
    try:
        _ok(d.run_hook(_chart_set(tmp_path_factory.mktemp("plandesc_fresh"))))
        assert _descriptions(d) == {i: new for i, (_, new) in SEEDED.items()}
    finally:
        _drop(d)
