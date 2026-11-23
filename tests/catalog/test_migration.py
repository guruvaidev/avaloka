"""The hosted migration and the SQLAlchemy tables describe the same catalog,
and the migration's row-level security has exactly one predicate per table.

Postgres is not run here (containers tier, T16.2/T16.13); this holds the two
descriptions together so they cannot drift apart silently, and holds the
policy text to the owner-only rule. Bite: add an organisation predicate to a
personal policy; create a table in one place and not the other."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.catalog.store import TABLES, VISIBILITIES, metadata

NAME = "20261116120000_dataset_catalog.sql"
ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "ui" / "supabase" / "migrations" / NAME
CHART_COPY = ROOT / "deploy" / "helm" / "avaloka" / "files" / "supabase-migrations" / NAME


@pytest.fixture(scope="module")
def sql():
    return MIGRATION.read_text(encoding="utf-8")


def test_the_chart_ships_the_same_migration_byte_for_byte(sql):
    """db-migrate.sh applies the chart's copy; the UI's Supabase project applies
    the other. Bite: fix one copy and forget the other."""
    assert CHART_COPY.is_file(), CHART_COPY
    assert CHART_COPY.read_bytes() == MIGRATION.read_bytes()


def test_every_store_table_is_in_the_migration_with_the_same_columns(sql):
    assert set(TABLES) == {"catalog_dataset", "source_schema_snapshots", "catalog_dataset_version",
                           "catalog_dataset_use", "catalog_term"}
    for table in metadata.sorted_tables:
        m = re.search(rf"CREATE TABLE IF NOT EXISTS public\.{table.name} \((.*?)\n\);", sql, re.DOTALL)
        assert m, f"{table.name} is not created by the migration"
        body = m.group(1)
        declared = {line.strip().split()[0] for line in body.splitlines()
                    if line.strip() and not line.strip().startswith(("CHECK", "PRIMARY", "--"))}
        assert declared == set(table.c.keys()), (table.name, declared ^ set(table.c.keys()))


def test_row_level_security_is_owner_only_on_every_table(sql):
    for table in TABLES:
        assert re.search(rf"ALTER TABLE public\.{table}\s+ENABLE ROW LEVEL SECURITY", sql), table
        policies = re.findall(rf"CREATE POLICY \w+ ON public\.{table}\s+(.*?);", sql, re.DOTALL)
        assert len(policies) == 1, (table, len(policies))
        (policy,) = policies
        predicates = re.findall(r"(?:USING|WITH CHECK) \((.*?)\)\s*(?:WITH CHECK|$)", policy, re.DOTALL)
        assert predicates, policy
        for predicate in predicates:
            assert predicate.strip() == "owner_user_id = (select auth.uid())::text", (table, predicate)
        for widening in ("organization", "project", "is_resource_collaborator", "current_org"):
            assert widening not in policy, (table, widening)


def test_the_visibility_trigger_refuses_anything_but_personal(sql):
    assert "CREATE TRIGGER catalog_dataset_personal_only" in sql
    assert "BEFORE INSERT OR UPDATE OF visibility ON public.catalog_dataset" in sql
    fn = re.search(r"FUNCTION public\.catalog_dataset_personal_only\(\)(.*?)\$\$;", sql, re.DOTALL).group(1)
    assert "NEW.visibility IS DISTINCT FROM 'personal'" in fn and "catalog_shared_enabled()" in fn
    assert "RAISE EXCEPTION" in fn
    assert re.search(r"catalog_shared_enabled\(\)\s*RETURNS boolean.*?SELECT false", sql, re.DOTALL)
    check = re.search(r"visibility\s+varchar\(16\)\s+NOT NULL DEFAULT 'personal'\s+CHECK \(visibility IN \((.*?)\)\)", sql)
    assert check and {v.strip("'") for v in check.group(1).split(",")} == set(VISIBILITIES)
