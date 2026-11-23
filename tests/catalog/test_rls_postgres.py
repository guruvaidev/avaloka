"""T16.2 and T16.13 on a real Postgres: the catalog's row-level security has the
owner as its only predicate, and the trigger refuses any visibility but
``personal`` while the shared catalog is off.

Applies the REAL migration (20261116120000_dataset_catalog) to a scratch
database on the server named by ``AVALOKA_TEST_PG_ADMIN_URL``, seeds rows for
two users as the service, then queries as Supabase's ``authenticated`` role
with request.jwt.claims set the way PostgREST sets them. The shims are the
same ones tests/test_support_admin_rls.py uses.

Each test fails without the thing it checks: with RLS off Bob reads Alice's
rows; with the trigger gone a ``project`` insert succeeds. Without the URL or
psql the module skips on a laptop and FAILS when CI is set.
"""

from __future__ import annotations

import json
import os
import pathlib
import secrets
import shutil
import subprocess
from urllib.parse import unquote, urlsplit

import pytest

from app.catalog.store import TABLES

pytestmark = pytest.mark.integration

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MIGRATIONS = REPO_ROOT / "ui" / "supabase" / "migrations"
CATALOG = "20261116120000_dataset_catalog.sql"

ADMIN_URL = os.environ.get("AVALOKA_TEST_PG_ADMIN_URL", "").strip()

SHIMS = """
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN CREATE ROLE anon NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN CREATE ROLE authenticated NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN CREATE ROLE service_role NOLOGIN BYPASSRLS; END IF;
END $$;
CREATE SCHEMA auth;
CREATE FUNCTION auth.jwt() RETURNS jsonb LANGUAGE sql STABLE AS
  $$ SELECT coalesce(nullif(current_setting('request.jwt.claims', true), ''), '{}')::jsonb $$;
CREATE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS
  $$ SELECT nullif(auth.jwt() ->> 'sub', '')::uuid $$;
GRANT USAGE ON SCHEMA auth TO anon, authenticated, service_role;
GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;
"""

ALICE = "11111111-1111-1111-1111-111111111111"
BOB = "22222222-2222-2222-2222-222222222222"


def _require(condition: bool, message: str) -> None:
    if condition:
        return
    if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
        pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
    pytest.skip(message)


class Db:
    def __init__(self, admin_url: str, name: str):
        parts = urlsplit(admin_url)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("PG")}
        self.env.update({
            "PGHOST": parts.hostname or "localhost",
            "PGPORT": str(parts.port or 5432),
            "PGUSER": unquote(parts.username or "postgres"),
            "PGPASSWORD": unquote(parts.password or ""),
        })
        self.admin_db = (parts.path or "/postgres").lstrip("/") or "postgres"
        self.name = name

    def psql(self, sql: str, db: str | None = None) -> str:
        env = dict(self.env, PGDATABASE=db or self.name)
        out = subprocess.run(["psql", "-X", "-q", "-tA", "-v", "ON_ERROR_STOP=1"], input=sql,
                             env=env, check=True, capture_output=True, text=True)
        return out.stdout.strip()

    def fails(self, sql: str) -> str:
        """The error text of a statement that must be refused; '' when it ran."""
        env = dict(self.env, PGDATABASE=self.name)
        out = subprocess.run(["psql", "-X", "-q", "-tA", "-v", "ON_ERROR_STOP=1"], input=sql,
                             env=env, capture_output=True, text=True)
        return out.stderr.strip() if out.returncode != 0 else ""

    def refusal(self, sql: str) -> dict:
        """How the server refused ``sql``: its SQLSTATE, the constraint it names and the
        PL/pgSQL context, read with GET STACKED DIAGNOSTICS. {} when the statement ran."""
        body = sql.strip().rstrip(";").replace("'", "''")
        return json.loads(self.psql(f"""
CREATE TEMP TABLE _refusal (state text, constraint_name text, context text);
DO $refusal$
DECLARE s text; c text; x text;
BEGIN
  EXECUTE '{body}';
EXCEPTION WHEN OTHERS THEN
  GET STACKED DIAGNOSTICS s = RETURNED_SQLSTATE, c = CONSTRAINT_NAME, x = PG_EXCEPTION_CONTEXT;
  INSERT INTO _refusal VALUES (s, c, x);
END $refusal$;
SELECT coalesce((SELECT json_build_object('sqlstate', state, 'constraint', constraint_name, 'context', context)
                 FROM _refusal), '{{}}'::json);
""") or "{}")

    def apply(self, filename: str) -> None:
        env = dict(self.env, PGDATABASE=self.name)
        subprocess.run(["psql", "-X", "-q", "-v", "ON_ERROR_STOP=1", "--single-transaction",
                        "-f", str(MIGRATIONS / filename)], env=env, check=True, capture_output=True, text=True)

    @staticmethod
    def _as(uid: str, sql: str, *, end: str) -> str:
        claims = json.dumps({"sub": uid, "email": f"{uid[:8]}@example.com", "role": "authenticated"}).replace("'", "''")
        return ("BEGIN;\nSET LOCAL ROLE authenticated;\n"
                f"SELECT set_config('request.jwt.claims', '{claims}', true) \\g /dev/null\n"
                f"{sql}\n{end};\n")

    def as_user(self, uid: str, sql: str) -> str:
        return self.psql(self._as(uid, sql, end="ROLLBACK"))

    def as_user_fails(self, uid: str, sql: str) -> str:
        return self.fails(self._as(uid, sql, end="ROLLBACK"))


def _seed(db: Db) -> None:
    """One dataset per user, each with a snapshot, a version, a use and a term,
    written as the service (the backend's path, which scopes by owner itself)."""
    rows = []
    for who, ds in ((ALICE, "ds:a"), (BOB, "ds:b")):
        rows.append(f"""
INSERT INTO public.catalog_dataset (id, owner_user_id, kind, locator, locator_key, display_name)
  VALUES ('{ds}', '{who}', 'file', '{{"dataset_id": "{ds}"}}', 'lk-{ds}', 'orders_{ds[-1]}.csv');
INSERT INTO public.source_schema_snapshots (id, owner_user_id, fingerprint, columns, column_count)
  VALUES ('snap:{ds}', '{who}', 'fp-{ds}', '[["amount","float64"]]', 1);
INSERT INTO public.catalog_dataset_version (id, dataset_id, owner_user_id, version_no, snapshot_id, captured_by)
  VALUES ('dsv:{ds}', '{ds}', '{who}', 1, 'snap:{ds}', '{who}');
INSERT INTO public.catalog_dataset_use (analysis_id, dataset_id, owner_user_id, version_id)
  VALUES ('an:{ds}', '{ds}', '{who}', 'dsv:{ds}');
INSERT INTO public.catalog_term (dataset_id, term, owner_user_id) VALUES ('{ds}', 'revenue', '{who}');
""")
    db.psql("".join(rows))


@pytest.fixture
def db():
    _require(bool(ADMIN_URL), "AVALOKA_TEST_PG_ADMIN_URL is not set; the catalog RLS was not exercised")
    _require(shutil.which("psql") is not None, "psql is not installed; the catalog RLS cannot be exercised")
    d = Db(ADMIN_URL, f"catalog_rls_{secrets.token_hex(6)}")
    d.psql(f'CREATE DATABASE "{d.name}"', db=d.admin_db)
    try:
        d.psql(SHIMS)
        d.apply(CATALOG)
        _seed(d)
        yield d
    finally:
        d.psql(f'DROP DATABASE IF EXISTS "{d.name}"', db=d.admin_db)


def test_the_service_sees_both_users_rows(db):
    """The seed is real: without this the owner checks below could pass on an
    empty database."""
    for table in TABLES:
        assert db.psql(f"SELECT count(*) FROM public.{table};") == "2", table


@pytest.mark.parametrize("table", TABLES)
def test_each_user_reads_only_their_own_rows_on_every_table(db, table):
    """Bite: disable RLS on one table, or add an organisation predicate."""
    assert db.as_user(ALICE, f"SELECT count(*) FROM public.{table};") == "1", table
    assert db.as_user(BOB, f"SELECT count(*) FROM public.{table};") == "1", table
    assert db.as_user(ALICE, f"SELECT count(*) FROM public.{table} WHERE owner_user_id = '{BOB}';") == "0", table
    assert db.psql(f"SELECT relrowsecurity FROM pg_class WHERE oid = 'public.{table}'::regclass;") == "t", table


def test_the_policies_have_exactly_one_predicate_the_owner(db):
    rows = db.psql(
        "SELECT tablename || '|' || policyname || '|' || coalesce(qual, '') || '|' || coalesce(with_check, '') "
        "FROM pg_policies WHERE schemaname = 'public' AND tablename LIKE 'catalog_%' "
        "OR (schemaname = 'public' AND tablename = 'source_schema_snapshots') ORDER BY tablename;")
    lines = [line for line in rows.splitlines() if line]
    assert len(lines) == len(TABLES), lines
    for line in lines:
        table, _policy, qual, check = line.split("|")
        for expr in (qual, check):
            assert "owner_user_id" in expr and "auth.uid()" in expr, (table, expr)
            for widening in ("organization", "project", "collaborator", " OR "):
                assert widening not in expr, (table, widening, expr)


def test_a_user_cannot_write_or_delete_another_users_rows(db):
    """Bite: a policy FOR SELECT only, or a USING without WITH CHECK."""
    touched = ("WITH u AS (UPDATE public.catalog_dataset SET display_name = 'mine now' "
               f"WHERE id = 'ds:b' RETURNING 1) SELECT count(*) FROM u;")
    assert db.as_user(ALICE, touched) == "0"
    deleted = "WITH d AS (DELETE FROM public.catalog_dataset_version WHERE dataset_id = 'ds:b' RETURNING 1) SELECT count(*) FROM d;"
    assert db.as_user(ALICE, deleted) == "0"
    forged = ("INSERT INTO public.catalog_dataset (id, owner_user_id, kind, locator, locator_key, display_name) "
              f"VALUES ('ds:forged', '{BOB}', 'file', '{{}}', 'lk-forged', 'x');")
    err = db.as_user_fails(ALICE, forged)
    assert "row-level security" in err, err
    own = ("INSERT INTO public.catalog_dataset (id, owner_user_id, kind, locator, locator_key, display_name) "
           f"VALUES ('ds:a2', '{ALICE}', 'file', '{{}}', 'lk-a2', 'y'); SELECT count(*) FROM public.catalog_dataset;")
    assert db.as_user(ALICE, own) == "2"
    assert db.psql("SELECT count(*) FROM public.catalog_dataset;") == "2"   # rolled back, as the harness does


@pytest.mark.parametrize("visibility", ["project", "organization"])
def test_the_trigger_refuses_any_visibility_but_personal_for_everyone(db, visibility):
    """Bite: drop the trigger; return 200 and ignore. Even the service cannot
    widen a row: the trigger, not the policy, is the lock."""
    widen = f"UPDATE public.catalog_dataset SET visibility = '{visibility}' WHERE id = 'ds:a';"
    err = db.fails(widen)
    assert "Enterprise shared catalog" in err and visibility in err, err
    err = db.as_user_fails(ALICE, widen)
    assert "Enterprise shared catalog" in err, err
    insert = ("INSERT INTO public.catalog_dataset (id, owner_user_id, kind, locator, locator_key, display_name, visibility) "
              f"VALUES ('ds:a3', '{ALICE}', 'file', '{{}}', 'lk-a3', 'z', '{visibility}');")
    assert "Enterprise shared catalog" in db.fails(insert)
    assert db.psql("SELECT count(*) FROM public.catalog_dataset WHERE visibility <> 'personal';") == "0"


def test_the_trigger_follows_catalog_shared_enabled(db):
    """When the Enterprise migration turns the function on, the same statement
    passes: the gate is that function, not a hard-coded refusal."""
    db.psql("CREATE OR REPLACE FUNCTION public.catalog_shared_enabled() RETURNS boolean "
            "LANGUAGE sql STABLE SET search_path = '' AS $$ SELECT true $$;")
    assert db.fails("UPDATE public.catalog_dataset SET visibility = 'project' WHERE id = 'ds:a';") == ""
    assert db.psql("SELECT visibility FROM public.catalog_dataset WHERE id = 'ds:a';") == "project"
    # and the single-predicate policy still hides it from Bob: visibility is not reach
    assert db.as_user(BOB, "SELECT count(*) FROM public.catalog_dataset;") == "1"


#: Both refusals report SQLSTATE 23514 (check_violation): the CHECK constraints, and the
#: personal-only trigger, which raises check_violation itself. A BEFORE ROW trigger fires
#: before CHECK constraints are evaluated, so 'public' always meets the trigger first and
#: its message never names a constraint. Each mechanism is pinned on its own below.
CHECK_VIOLATION = "23514"
PERSONAL_ONLY = "function public.catalog_dataset_personal_only()"


def test_a_shared_visibility_is_refused_by_the_personal_only_trigger(db):
    refusal = db.refusal("UPDATE public.catalog_dataset SET visibility = 'public' WHERE id = 'ds:a';")
    assert refusal.get("sqlstate") == CHECK_VIOLATION and PERSONAL_ONLY in (refusal.get("context") or ""), (
        f"visibility 'public' was not refused by catalog_dataset_personal_only(): {refusal}")


def test_a_bad_kind_is_refused_by_its_check_constraint(db):
    """The trigger only looks at visibility, so an unknown kind reaches the CHECK."""
    refusal = db.refusal(
        "INSERT INTO public.catalog_dataset (id, owner_user_id, kind, locator, locator_key) "
        f"VALUES ('ds:x', '{ALICE}', 'spreadsheet', '{{}}', 'lk-x');")
    assert refusal.get("sqlstate") == CHECK_VIOLATION and refusal.get("constraint") == "catalog_dataset_kind_check", (
        f"kind 'spreadsheet' was not refused by catalog_dataset_kind_check: {refusal}")
