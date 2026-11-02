"""NEW-BG-01, server side: team and collaborator writes need an Enterprise plan.

Against real Postgres, with the chart's own migration set applied by the chart's
own migrate hook (deploy/helm/avaloka/files/db-migrate.sh, version-tracked).
Supabase pieces plain Postgres lacks (auth schema and auth.uid(), the API roles)
come from a minimal stub; a query "as a user" runs under SET ROLE authenticated
with the PostgREST JWT claim set, so RLS is evaluated as for a signed-in browser.

What is pinned (migration 20261101140000_plan_rules_pro_collab.sql, which
supersedes 20261101120000_plan_enforced_org_features.sql):
  * Free, Professional, and a missing or inactive plan are refused team INSERT /
    UPDATE and collaborator INSERT / UPDATE; Enterprise (and an Enterprise trial,
    and a support-desk admin) is allowed.
  * Rows a Free organisation already has stay readable and deletable.
  * Under the version-tracked hook the migration runs once, in its own
    transaction with ON_ERROR_STOP, on a fresh database and after the backfill of
    an install the old hook managed; a second run applies nothing.

Every refusal test also proves the caller passes the pre-existing permissive
checks (org admin of that organisation, owner of the resource), so a refusal can
only come from the plan check.

Needs AVALOKA_TEST_PG_ADMIN_URL (a role that may CREATE DATABASE), psql and sh.
Skips on a laptop without them and FAILS when CI is set.
"""

from __future__ import annotations

import os
import pathlib
import secrets
import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = REPO_ROOT / "deploy" / "helm" / "avaloka" / "files"
SCRIPT = FILES / "db-migrate.sh"
MIGRATIONS = FILES / "supabase-migrations"
BASELINE = FILES / "db-migrations.baseline"
VERSION = "20261101140000_plan_rules_pro_collab"
# The pre-revision migration it supersedes (merged with #650; left untouched).
SUPERSEDED = "20261101120000_plan_enforced_org_features"
# Every org-structure table the migration gates (owner, 2026-11-01).
GATED = ("organization_teams", "resource_collaborators", "departments", "config_roles", "branches", "app_users")
ADMIN_URL = os.environ.get("AVALOKA_TEST_PG_ADMIN_URL", "").strip()

SUPABASE_STUB = r"""
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN CREATE ROLE anon NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN CREATE ROLE authenticated NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN CREATE ROLE service_role NOLOGIN BYPASSRLS; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'supabase_admin') THEN CREATE ROLE supabase_admin NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'supabase_auth_admin') THEN CREATE ROLE supabase_auth_admin NOLOGIN; END IF;
END $$;
CREATE SCHEMA IF NOT EXISTS auth;
CREATE SCHEMA IF NOT EXISTS storage;
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;
CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA extensions;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA extensions;
CREATE TABLE IF NOT EXISTS auth.users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  instance_id uuid, aud text, role text,
  email text, encrypted_password text, phone text,
  email_confirmed_at timestamptz, last_sign_in_at timestamptz, confirmed_at timestamptz,
  raw_user_meta_data jsonb DEFAULT '{}'::jsonb,
  raw_app_meta_data jsonb DEFAULT '{}'::jsonb,
  is_super_admin boolean, is_anonymous boolean DEFAULT false,
  created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now(),
  deleted_at timestamptz
);
CREATE OR REPLACE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS
$f$ SELECT nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $f$;
CREATE OR REPLACE FUNCTION auth.role() RETURNS text LANGUAGE sql STABLE AS
$f$ SELECT nullif(current_setting('request.jwt.claim.role', true), '') $f$;
CREATE OR REPLACE FUNCTION auth.email() RETURNS text LANGUAGE sql STABLE AS
$f$ SELECT nullif(current_setting('request.jwt.claim.email', true), '') $f$;
CREATE OR REPLACE FUNCTION auth.jwt() RETURNS jsonb LANGUAGE sql STABLE AS
$f$ SELECT coalesce(nullif(current_setting('request.jwt.claims', true), ''), '{}')::jsonb $f$;
CREATE TABLE IF NOT EXISTS storage.buckets (
  id text PRIMARY KEY, name text, owner uuid, public boolean DEFAULT false,
  file_size_limit bigint, allowed_mime_types text[],
  created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now()
);
CREATE TABLE IF NOT EXISTS storage.objects (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), bucket_id text, name text, owner uuid,
  metadata jsonb, created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now()
);
ALTER TABLE storage.objects ENABLE ROW LEVEL SECURITY;
CREATE OR REPLACE FUNCTION storage.foldername(name text) RETURNS text[] LANGUAGE sql IMMUTABLE AS
$f$ SELECT (string_to_array(name, '/'))[1:array_length(string_to_array(name, '/'), 1) - 1] $f$;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_publication WHERE pubname = 'supabase_realtime') THEN
    CREATE PUBLICATION supabase_realtime;
  END IF;
END $$;
GRANT USAGE ON SCHEMA auth, storage, extensions, public TO anon, authenticated, service_role;
GRANT SELECT ON auth.users TO service_role;
GRANT ALL ON storage.objects, storage.buckets TO authenticated, service_role;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA auth TO anon, authenticated, service_role;
"""

RLS_DENIED = "row-level security"


def _require(condition: bool, message: str) -> None:
    if condition:
        return
    if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
        pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
    pytest.skip(message)


def _uuid() -> str:
    return str(uuid.uuid4())


def _migration_sql(copy: pathlib.Path = MIGRATIONS) -> str:
    path = copy / f"{VERSION}.sql"
    assert path.is_file(), f"migration {VERSION} is missing from {copy.relative_to(REPO_ROOT)}"
    return path.read_text()


class Db:
    def __init__(self, admin_url: str, name: str):
        parts = urlsplit(admin_url)
        self.base_env = {
            "PGHOST": parts.hostname or "localhost",
            "PGPORT": str(parts.port or 5432),
            "PGUSER": unquote(parts.username or "postgres"),
            "PGPASSWORD": unquote(parts.password or ""),
        }
        self.admin_db = (parts.path or "/postgres").lstrip("/") or "postgres"
        self.name = name

    def env(self, **extra: str) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("PG")}
        env.update(self.base_env)
        env["PGDATABASE"] = self.name
        env.update(extra)
        return env

    def psql(self, sql: str, *, db: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
        env = self.env(PGDATABASE=db) if db else self.env()
        return subprocess.run(["psql", "-X", "-q", "-tA", "-v", "ON_ERROR_STOP=1"], input=sql, env=env,
                              check=check, capture_output=True, text=True, timeout=300)

    def scalar(self, sql: str) -> str:
        return self.psql(sql).stdout.strip()

    def admin(self, sql: str) -> None:
        """As the superuser, with triggers off so the seed is exactly what is written."""
        self.psql("SET session_replication_role = replica;\n" + sql)

    def as_user(self, auth_uid: str, sql: str) -> subprocess.CompletedProcess:
        """One transaction as a signed-in browser: role authenticated, JWT claims set."""
        script = (
            "BEGIN;\nSET LOCAL ROLE authenticated;\n"
            f"SET LOCAL \"request.jwt.claim.sub\" = '{auth_uid}';\n"
            "SET LOCAL \"request.jwt.claim.role\" = 'authenticated';\n"
            f"SET LOCAL \"request.jwt.claims\" = '{{\"sub\": \"{auth_uid}\", \"role\": \"authenticated\"}}';\n"
            f"{sql}\nCOMMIT;\n"
        )
        return self.psql(script, check=False)

    def run_hook(self, mig: pathlib.Path) -> subprocess.CompletedProcess:
        return subprocess.run(["sh", str(SCRIPT)], env=self.env(MIGRATIONS_DIR=str(mig), ADMIN_EMAILS=""),
                              capture_output=True, text=True, timeout=900)

    def run_old_hook_on_baseline(self, mig: pathlib.Path) -> None:
        """What the chart did before tracking, over the files that existed then."""
        names = set(_baseline_versions())
        for f in sorted(mig.glob("*.sql")):
            if f.stem in names:
                subprocess.run(["psql", "-X", "-q", "-v", "ON_ERROR_STOP=0", "-f", str(f)], env=self.env(),
                               capture_output=True, text=True, timeout=300)


def _baseline_versions() -> list[str]:
    return [ln.strip() for ln in BASELINE.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def _chart_set(tmp: pathlib.Path) -> pathlib.Path:
    mig = tmp / "migrations"
    shutil.copytree(MIGRATIONS, mig)
    shutil.copyfile(BASELINE, mig / "baseline.txt")
    return mig


def _new_db() -> Db:
    _require(bool(ADMIN_URL), "AVALOKA_TEST_PG_ADMIN_URL is not set; the plan policies were not exercised")
    _require(shutil.which("psql") is not None, "psql is not installed")
    _require(shutil.which("sh") is not None, "no sh")
    d = Db(ADMIN_URL, f"planrls_{secrets.token_hex(6)}")
    d.psql(f'CREATE DATABASE "{d.name}"', db=d.admin_db)
    d.psql(SUPABASE_STUB)
    d.psql(f'ALTER DATABASE "{d.name}" SET search_path TO "$user", public, extensions')
    return d


def _drop(d: Db) -> None:
    d.psql(f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{d.name}' AND pid <> pg_backend_pid();"
           f'\nDROP DATABASE IF EXISTS "{d.name}";', db=d.admin_db, check=False)


def _ok(result: subprocess.CompletedProcess) -> str:
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"migrate hook failed rc={result.returncode}\n{out[-4000:]}"
    return out


# ---- the world: one organisation per plan state ----------------------------------

TIERS = ("free", "professional", "enterprise", "enterprise_trial", "missing", "canceled", "past_due",
         "no_plan_row", "admin_on_free", "admin_on_pro")


@dataclass
class Org:
    key: str
    org: str = field(default_factory=_uuid)
    auth: str = field(default_factory=_uuid)
    profile: str = field(default_factory=_uuid)
    owner_app_user: str = field(default_factory=_uuid)
    member_app_user: str = field(default_factory=_uuid)
    project: str = field(default_factory=_uuid)
    team: str = field(default_factory=_uuid)            # pre-existing team
    collaborator: str = field(default_factory=_uuid)    # pre-existing collaborator row
    department: str = field(default_factory=_uuid)      # pre-existing department
    role: str = field(default_factory=_uuid)            # pre-existing config role
    branch: str = field(default_factory=_uuid)          # pre-existing branch
    # Two more people in the organisation, each with their OWN plan (personal
    # profile): one Professional, one Free. For the team-member rule.
    pro_member_auth: str = field(default_factory=_uuid)
    pro_member_app: str = field(default_factory=_uuid)
    free_member_auth: str = field(default_factory=_uuid)
    free_member_app: str = field(default_factory=_uuid)


@dataclass
class World:
    plans: dict[str, str]
    orgs: dict[str, Org]


def _seed(d: Db) -> World:
    plans = {"free": _uuid(), "professional": _uuid(), "enterprise": _uuid()}
    names = {"free": "Free", "professional": "Professional", "enterprise": "Enterprise"}
    sql = [f"INSERT INTO public.plans (id, plan_type, name) VALUES ('{pid}', '{t}', '{names[t]}');"
           for t, pid in plans.items()]
    orgs = {k: Org(k) for k in TIERS}
    for o in orgs.values():
        sql += [
            f"INSERT INTO auth.users (id, email, email_confirmed_at) VALUES ('{o.auth}', 'owner-{o.key}@example.com', now());",
            f"INSERT INTO public.profiles (id, user_id, full_name, organization_id) VALUES ('{o.profile}', '{o.auth}', 'Owner {o.key}', '{o.org}');",
            f"INSERT INTO public.organizations (id, name, owner_profile_id) VALUES ('{o.org}', 'Org {o.key}', '{o.profile}');",
            f"INSERT INTO public.app_users (id, organization_id, email, auth_user_id, profile_id, role) VALUES "
            f"('{o.owner_app_user}', '{o.org}', 'owner-{o.key}@example.com', '{o.auth}', '{o.profile}', 'owner'),"
            f"('{o.member_app_user}', '{o.org}', 'member-{o.key}@example.com', NULL, NULL, 'member');",
            f"INSERT INTO public.projects (id, owner_id, name, organization_id) VALUES ('{o.project}', '{o.profile}', 'Project {o.key}', '{o.org}');",
            f"INSERT INTO public.organization_teams (id, organization_id, name, branch) VALUES ('{o.team}', '{o.org}', 'Existing team', 'HQ');",
            f"INSERT INTO public.resource_collaborators (id, resource_type, resource_id, app_user_id, access_level, invited_by) "
            f"VALUES ('{o.collaborator}', 'project', '{o.project}', '{o.member_app_user}', 'view', '{o.auth}');",
            f"INSERT INTO public.departments (id, organization_id, department, code) VALUES ('{o.department}', '{o.org}', 'Existing dept', 'D-{o.key}');",
            f"INSERT INTO public.config_roles (id, organization_id, role, department) VALUES ('{o.role}', '{o.org}', 'Existing role', 'Existing dept');",
            f"INSERT INTO public.branches (id, organization_id, name, location) VALUES ('{o.branch}', '{o.org}', 'Existing branch', 'HQ');",
        ]
        for who, a, app in (("pro", o.pro_member_auth, o.pro_member_app), ("free", o.free_member_auth, o.free_member_app)):
            prof = _uuid()
            sql += [
                f"INSERT INTO auth.users (id, email, email_confirmed_at) VALUES ('{a}', '{who}-{o.key}@example.com', now());",
                f"INSERT INTO public.profiles (id, user_id, full_name, organization_id) VALUES ('{prof}', '{a}', 'Member', NULL);",
                f"INSERT INTO public.app_users (id, organization_id, email, auth_user_id, role) VALUES ('{app}', '{o.org}', '{who}-{o.key}@example.com', '{a}', 'member');",
            ]
            if who == "pro":
                sql.append(f"INSERT INTO public.subscriptions (owner_profile_id, organization_id, plan_id, payment_provider, status) "
                           f"VALUES ('{prof}', NULL, '{plans['professional']}', 'stripe', 'active');")

    def sub(o: Org, plan: str, status: str, *, org_scoped: bool = True, trial_ends: str = "NULL") -> str:
        org_col = f"'{o.org}'" if org_scoped else "NULL"
        return (f"INSERT INTO public.subscriptions (owner_profile_id, organization_id, plan_id, payment_provider, status, trial_ends_at) "
                f"VALUES ('{o.profile}', {org_col}, '{plan}', 'stripe', '{status}', {trial_ends});")

    sql += [
        sub(orgs["free"], plans["free"], "active"),
        # Professional is a personal subscription: no organization_id, owned by the profile.
        sub(orgs["professional"], plans["professional"], "active", org_scoped=False),
        sub(orgs["enterprise"], plans["enterprise"], "active"),
        sub(orgs["enterprise_trial"], plans["enterprise"], "trialing", trial_ends="now() + interval '10 days'"),
        sub(orgs["canceled"], plans["enterprise"], "canceled"),
        sub(orgs["past_due"], plans["enterprise"], "past_due"),
        # A subscription whose plan row does not exist (an unrecognised plan).
        sub(orgs["no_plan_row"], _uuid(), "active"),
        sub(orgs["admin_on_free"], plans["free"], "active"),
        f"INSERT INTO public.user_roles (user_id, role) VALUES ('{orgs['admin_on_free'].auth}', 'admin');",
        sub(orgs["admin_on_pro"], plans["professional"], "active", org_scoped=False),
        f"INSERT INTO public.user_roles (user_id, role) VALUES ('{orgs['admin_on_pro'].auth}', 'admin');",
    ]
    d.admin("\n".join(sql))
    return World(plans, orgs)


@pytest.fixture(scope="module")
def applied(tmp_path_factory):
    """A fresh database migrated by the chart's tracked hook, run twice."""
    d = _new_db()
    try:
        mig = _chart_set(tmp_path_factory.mktemp("planrls"))
        first = _ok(d.run_hook(mig))
        second = _ok(d.run_hook(mig))
        yield d, first, second
    finally:
        _drop(d)


@pytest.fixture(scope="module")
def world(applied) -> World:
    return _seed(applied[0])


# ---- migration under tracking ----------------------------------------------------


def test_migration_is_new_not_baseline_and_ships_in_both_copies():
    assert VERSION not in _baseline_versions(), "a new migration must not be in the frozen baseline"
    # The Supabase CLI keys schema_migrations by the numeric prefix alone, so no
    # other migration may share it (tests/test_migration_versions_unique.py
    # guards every file; this pins the new one).
    prefix = VERSION.split("_", 1)[0]
    for copy in (MIGRATIONS, REPO_ROOT / "ui" / "supabase" / "migrations"):
        same = sorted(p.name for p in copy.glob(f"{prefix}_*.sql"))
        assert same == [f"{VERSION}.sql"], f"version {prefix} is not unique in {copy.relative_to(REPO_ROOT)}: {same}"
    # The superseded migration stays exactly where it is: it may be applied.
    assert (MIGRATIONS / f"{SUPERSEDED}.sql").is_file()
    ui = _migration_sql(REPO_ROOT / "ui" / "supabase" / "migrations")
    assert ui == _migration_sql(), "the UI and chart copies differ"


def test_fresh_install_applies_it_once_inside_the_tracked_transaction(applied):
    d, first, second = applied
    assert f"applying {VERSION}\n" in first, "not applied as a tracked (transactional, ON_ERROR_STOP) migration"
    assert d.scalar(f"SELECT how FROM avaloka_meta.schema_migrations WHERE version = '{VERSION}'") == "applied"
    assert f"applying {VERSION}" not in second
    assert "migration state: tracked" in second
    assert "migrations complete: 0 applied," in second


def test_applying_the_sql_again_is_a_no_op(applied):
    """Idempotent even outside the tracker (e.g. a hand re-run, or supabase db push repair)."""
    d, _, _ = applied
    policies = ("SELECT string_agg(tablename || '.' || policyname || ':' || permissive || ':' || cmd, ',' ORDER BY tablename, policyname) "
                "FROM pg_policies WHERE schemaname = 'public'")
    triggers = ("SELECT string_agg(tgrelid::regclass::text || '.' || tgname, ',' ORDER BY tgrelid::regclass::text) FROM pg_trigger "
                "WHERE tgname = 'audit_org_feature_bypass'")
    before = (d.scalar(policies), d.scalar(triggers))
    d.psql(_migration_sql())
    assert (d.scalar(policies), d.scalar(triggers)) == before


def test_existing_install_gets_it_after_the_backfill(tmp_path):
    d = _new_db()
    try:
        mig = _chart_set(tmp_path)
        d.run_old_hook_on_baseline(mig)
        assert d.scalar("SELECT to_regclass('avaloka_meta.schema_migrations') IS NULL") == "t"
        out = _ok(d.run_hook(mig))
        assert "existing install, backfilled" in out
        assert f"applying {VERSION}\n" in out
        rows = d.scalar(f"SELECT how FROM avaloka_meta.schema_migrations WHERE version = '{VERSION}'")
        assert rows == "applied"
        assert d.scalar("SELECT count(*) FROM pg_policies WHERE policyname LIKE 'enterprise plan to %' OR policyname LIKE 'paid plan to %'") == str(len(GATED) * 2)
        out = _ok(d.run_hook(mig))
        assert f"applying {VERSION}" not in out
    finally:
        _drop(d)


# ---- the helper ------------------------------------------------------------------

EXPECTED_TIER = {
    "free": "free", "professional": "professional", "enterprise": "enterprise",
    "enterprise_trial": "enterprise", "missing": "free", "canceled": "free", "past_due": "free",
    "no_plan_row": "free", "admin_on_free": "free", "admin_on_pro": "professional",
}
# Owner rulings 2026-11-01 (revised): teams and collaborator invites need
# Professional or higher; departments, roles, branches and app users need
# Enterprise. The admin_on_* callers get in only through the audited app-admin
# bypass where their own plan falls short.
ALLOWED_COLLAB = {"professional", "enterprise", "enterprise_trial", "admin_on_free", "admin_on_pro"}
ALLOWED_ORG = {"enterprise", "enterprise_trial", "admin_on_free", "admin_on_pro"}


def test_plan_tier_mirrors_the_ui_resolver_and_fails_closed(applied, world):
    d = applied[0]
    got = {}
    for key, o in world.orgs.items():
        r = d.as_user(o.auth, "SELECT public.current_plan_tier() || '|' || public.collab_features_enabled()::text"
                              " || '|' || public.org_features_enabled()::text;")
        assert r.returncode == 0, r.stderr
        got[key] = r.stdout.strip()
    assert got == {k: f"{EXPECTED_TIER[k]}|{str(k in ALLOWED_COLLAB).lower()}|{str(k in ALLOWED_ORG).lower()}" for k in TIERS}


def test_anonymous_and_unknown_callers_have_no_org_features(applied, world):
    d = applied[0]
    r = d.as_user(_uuid(), "SELECT public.current_plan_tier() || '|' || public.collab_features_enabled()::text"
                           " || '|' || public.org_features_enabled()::text;")
    assert r.stdout.strip() == "free|false|false", r.stderr
    r = d.psql("BEGIN; SET LOCAL ROLE anon; SELECT public.org_features_enabled(); COMMIT;", check=False)
    assert r.returncode != 0 and "permission denied" in r.stderr


# ---- writes per plan --------------------------------------------------------------


def _preconditions(d: Db, o: Org) -> None:
    """The caller passes every pre-existing permissive check: org admin of its own
    organisation and owner of the project."""
    r = d.as_user(o.auth, "SELECT public.current_org_id()::text || '|' || public.is_org_admin()::text"
                          f" || '|' || public.can_manage_resource_share('project', '{o.project}')::text;")
    assert r.stdout.strip() == f"{o.org}|true|true", f"{o.key}: precondition failed: {r.stdout} {r.stderr}"


@pytest.mark.parametrize("key", TIERS)
def test_team_insert(applied, world, key):
    d, o = applied[0], world.orgs[key]
    _preconditions(d, o)
    r = d.as_user(o.auth, f"INSERT INTO public.organization_teams (organization_id, name, branch) "
                          f"VALUES ('{o.org}', 'New team {key}', 'HQ') RETURNING name;")
    if key in ALLOWED_COLLAB:
        assert r.returncode == 0 and r.stdout.strip() == f"New team {key}", r.stderr
    else:
        assert r.returncode != 0 and RLS_DENIED in r.stderr, f"{key} created a team: {r.stdout}{r.stderr}"


@pytest.mark.parametrize("key", TIERS)
def test_team_update(applied, world, key):
    d, o = applied[0], world.orgs[key]
    _preconditions(d, o)
    r = d.as_user(o.auth, f"WITH u AS (UPDATE public.organization_teams SET description = 'changed by {key}' "
                          f"WHERE id = '{o.team}' RETURNING 1) SELECT count(*) FROM u;")
    assert r.returncode == 0, r.stderr
    desc = d.scalar(f"SELECT coalesce(description, '') FROM public.organization_teams WHERE id = '{o.team}'")
    if key in ALLOWED_COLLAB:
        assert (r.stdout.strip(), desc) == ("1", f"changed by {key}")
    else:
        assert (r.stdout.strip(), desc) == ("0", ""), f"{key} changed a team"


@pytest.mark.parametrize("key", TIERS)
def test_collaborator_insert(applied, world, key):
    d, o = applied[0], world.orgs[key]
    _preconditions(d, o)
    # A second resource, so the insert is not a duplicate of the seeded row.
    pid = _uuid()
    d.admin(f"INSERT INTO public.projects (id, owner_id, name, organization_id) VALUES ('{pid}', '{o.profile}', 'P2 {key}', '{o.org}');")
    r = d.as_user(o.auth, "INSERT INTO public.resource_collaborators (resource_type, resource_id, app_user_id, access_level, invited_by) "
                          f"VALUES ('project', '{pid}', '{o.member_app_user}', 'view', '{o.auth}') RETURNING access_level;")
    if key in ALLOWED_COLLAB:
        assert r.returncode == 0 and r.stdout.strip() == "view", r.stderr
    else:
        assert r.returncode != 0 and RLS_DENIED in r.stderr, f"{key} invited a collaborator: {r.stdout}{r.stderr}"


@pytest.mark.parametrize("key", TIERS)
def test_collaborator_update(applied, world, key):
    d, o = applied[0], world.orgs[key]
    _preconditions(d, o)
    r = d.as_user(o.auth, f"WITH u AS (UPDATE public.resource_collaborators SET access_level = 'edit' "
                          f"WHERE id = '{o.collaborator}' RETURNING 1) SELECT count(*) FROM u;")
    assert r.returncode == 0, r.stderr
    level = d.scalar(f"SELECT access_level FROM public.resource_collaborators WHERE id = '{o.collaborator}'")
    if key in ALLOWED_COLLAB:
        assert (r.stdout.strip(), level) == ("1", "edit")
    else:
        assert (r.stdout.strip(), level) == ("0", "view"), f"{key} changed a collaborator"


# ---- org structure: departments, roles, branches, app users ----------------------

def _structure_insert(o: Org, table: str, tag: str) -> str:
    u = _uuid()
    return {
        "departments": f"INSERT INTO public.departments (organization_id, department, code) VALUES ('{o.org}', 'Dept {tag}', 'C-{u[:8]}') RETURNING 'ok';",
        "config_roles": f"INSERT INTO public.config_roles (organization_id, role, department) VALUES ('{o.org}', 'Role {tag}', 'Dept') RETURNING 'ok';",
        "branches": f"INSERT INTO public.branches (organization_id, name, location) VALUES ('{o.org}', 'Branch {tag}', 'HQ') RETURNING 'ok';",
        "app_users": f"INSERT INTO public.app_users (organization_id, email, role) VALUES ('{o.org}', 'new-{u[:8]}@example.com', 'member') RETURNING 'ok';",
    }[table]


def _structure_update(o: Org, table: str, tag: str) -> tuple[str, str, str]:
    """(update statement returning the row count, read-back query, value written)."""
    target = {"departments": ("department", o.department), "config_roles": ("role", o.role),
              "branches": ("name", o.branch), "app_users": ("role", o.member_app_user)}[table]
    col, rid = target
    value = f"changed {tag}"
    return (f"WITH u AS (UPDATE public.{table} SET {col} = '{value}' WHERE id = '{rid}' RETURNING 1) SELECT count(*) FROM u;",
            f"SELECT {col} FROM public.{table} WHERE id = '{rid}'", value)


STRUCTURE = ("departments", "config_roles", "branches", "app_users")


@pytest.mark.parametrize("table", STRUCTURE)
@pytest.mark.parametrize("key", TIERS)
def test_structure_insert(applied, world, key, table):
    d, o = applied[0], world.orgs[key]
    _preconditions(d, o)
    r = d.as_user(o.auth, _structure_insert(o, table, key))
    if key in ALLOWED_ORG:
        assert r.returncode == 0 and r.stdout.strip() == "ok", f"{key}/{table}: {r.stderr}"
    else:
        assert r.returncode != 0 and RLS_DENIED in r.stderr, f"{key} inserted into {table}: {r.stdout}{r.stderr}"


@pytest.mark.parametrize("table", STRUCTURE)
@pytest.mark.parametrize("key", TIERS)
def test_structure_update(applied, world, key, table):
    d, o = applied[0], world.orgs[key]
    _preconditions(d, o)
    stmt, readback, value = _structure_update(o, table, key)
    before = d.scalar(readback)
    r = d.as_user(o.auth, stmt)
    if table == "app_users" and key in ALLOWED_COLLAB - ALLOWED_ORG:
        # Professional passes the app_users UPDATE policy (team membership), and
        # the trigger refuses any other change with the Enterprise message.
        assert r.returncode != 0 and "need an Enterprise plan" in r.stderr, f"{key} changed app_users: {r.stdout}{r.stderr}"
        assert d.scalar(readback) == before
        return
    assert r.returncode == 0, r.stderr
    after = d.scalar(readback)
    if key in ALLOWED_ORG:
        assert (r.stdout.strip(), after) == ("1", value), f"{key}/{table}"
    else:
        assert (r.stdout.strip(), after) == ("0", before), f"{key} changed {table}"


# ---- team members and referrals (owner, 2026-11-01 revision) ----------------------

TEAM_MSG = "Team members need a Professional plan or higher."


def _add_to_team(d: Db, o: Org, app_user: str) -> subprocess.CompletedProcess:
    return d.as_user(o.auth, f"WITH u AS (UPDATE public.app_users SET team_id = '{o.team}' WHERE id = '{app_user}' "
                             "RETURNING 1) SELECT count(*) FROM u;")


def _team_of(d: Db, app_user: str) -> str:
    return d.scalar(f"SELECT coalesce(team_id::text, '') FROM public.app_users WHERE id = '{app_user}'")


def test_member_plan_helper_returns_only_a_boolean(applied, world):
    d, o = applied[0], world.orgs["professional"]
    r = d.as_user(o.auth, f"SELECT public.user_is_pro_or_higher('{o.pro_member_auth}')::text || '|' || "
                          f"public.user_is_pro_or_higher('{o.free_member_auth}')::text || '|' || "
                          f"public.user_is_pro_or_higher('{_uuid()}')::text || '|' || "
                          f"pg_typeof(public.user_is_pro_or_higher('{o.pro_member_auth}'))::text;")
    assert r.stdout.strip() == "true|false|false|boolean", r.stderr
    # The resolver behind it is not callable by an API role.
    r = d.as_user(o.auth, f"SELECT avaloka_meta.plan_tier_of('{o.pro_member_auth}');")
    assert r.returncode != 0 and "permission denied" in r.stderr
    r = d.psql(f"BEGIN; SET LOCAL ROLE anon; SELECT public.user_is_pro_or_higher('{o.pro_member_auth}'); COMMIT;", check=False)
    assert r.returncode != 0 and "permission denied" in r.stderr


def test_team_member_allowed_rpc_matches_the_rule(applied, world):
    """The RPC the server functions use for the member check agrees with the trigger."""
    d = applied[0]
    got = {}
    for key in ("free", "professional", "enterprise"):
        o = world.orgs[key]
        r = d.as_user(o.auth, f"SELECT public.team_member_allowed('{o.pro_member_app}')::text || '|' || "
                              f"public.team_member_allowed('{o.free_member_app}')::text;")
        got[key] = r.stdout.strip()
    assert got == {"free": "false|false", "professional": "true|false", "enterprise": "true|true"}


def test_pro_team_accepts_a_pro_member(applied, world):
    d, o = applied[0], world.orgs["professional"]
    r = _add_to_team(d, o, o.pro_member_app)
    assert (r.returncode, r.stdout.strip()) == (0, "1"), r.stderr
    assert _team_of(d, o.pro_member_app) == o.team


def test_pro_team_refuses_a_free_member_with_the_upgrade_message(applied, world):
    d, o = applied[0], world.orgs["professional"]
    r = _add_to_team(d, o, o.free_member_app)
    assert r.returncode != 0 and TEAM_MSG in r.stderr, f"Free member added to a Pro team: {r.stdout}{r.stderr}"
    assert _team_of(d, o.free_member_app) == ""


def test_pro_team_refuses_a_member_who_never_signed_up(applied, world):
    d, o = applied[0], world.orgs["professional"]
    r = _add_to_team(d, o, o.member_app_user)   # invited, no auth account: no plan
    assert r.returncode != 0 and TEAM_MSG in r.stderr
    assert _team_of(d, o.member_app_user) == ""


def test_pro_can_remove_a_team_member(applied, world):
    d, o = applied[0], world.orgs["professional"]
    d.admin(f"UPDATE public.app_users SET team_id = '{o.team}' WHERE id = '{o.free_member_app}';")
    r = d.as_user(o.auth, f"WITH u AS (UPDATE public.app_users SET team_id = NULL WHERE id = '{o.free_member_app}' "
                          "RETURNING 1) SELECT count(*) FROM u;")
    assert (r.returncode, r.stdout.strip()) == (0, "1"), r.stderr


@pytest.mark.parametrize("key", ["enterprise", "enterprise_trial"])
def test_enterprise_team_accepts_any_org_member(applied, world, key):
    d, o = applied[0], world.orgs[key]
    for member in (o.free_member_app, o.member_app_user):
        r = _add_to_team(d, o, member)
        assert (r.returncode, r.stdout.strip()) == (0, "1"), f"{key}: {r.stderr}"
        assert _team_of(d, member) == o.team


@pytest.mark.parametrize("key", ["free", "missing", "canceled"])
def test_free_cannot_add_team_members(applied, world, key):
    d, o = applied[0], world.orgs[key]
    r = _add_to_team(d, o, o.pro_member_app)
    assert (r.returncode, r.stdout.strip()) == (0, "0"), f"{key} added a team member: {r.stderr}"
    assert _team_of(d, o.pro_member_app) == ""


def test_pro_invites_a_free_user_as_collaborator_the_referral_path(applied, world):
    """Professional may invite a Free user onto its own resource -- from its own
    organisation, and from someone else's."""
    d, o = applied[0], world.orgs["professional"]
    for invitee in (o.free_member_app, world.orgs["free"].free_member_app):
        pid = _uuid()
        d.admin(f"INSERT INTO public.projects (id, owner_id, name, organization_id) VALUES ('{pid}', '{o.profile}', 'Ref', '{o.org}');")
        r = d.as_user(o.auth, "INSERT INTO public.resource_collaborators (resource_type, resource_id, app_user_id, access_level, invited_by) "
                              f"VALUES ('project', '{pid}', '{invitee}', 'view', '{o.auth}') RETURNING access_level;")
        assert (r.returncode, r.stdout.strip()) == (0, "view"), r.stderr


def test_free_cannot_invite_even_a_pro_user(applied, world):
    d, o = applied[0], world.orgs["free"]
    pid = _uuid()
    d.admin(f"INSERT INTO public.projects (id, owner_id, name, organization_id) VALUES ('{pid}', '{o.profile}', 'X', '{o.org}');")
    r = d.as_user(o.auth, "INSERT INTO public.resource_collaborators (resource_type, resource_id, app_user_id, access_level, invited_by) "
                          f"VALUES ('project', '{pid}', '{o.pro_member_app}', 'view', '{o.auth}') RETURNING 1;")
    assert r.returncode != 0 and RLS_DENIED in r.stderr


# ---- existing rows stay readable and removable ----------------------------------


@pytest.mark.parametrize("key", ["free", "professional", "missing"])
def test_existing_rows_stay_readable_and_deletable(applied, world, key):
    d, o = applied[0], world.orgs[key]
    team, collab = _uuid(), _uuid()
    pid = _uuid()
    d.admin(
        f"INSERT INTO public.projects (id, owner_id, name, organization_id) VALUES ('{pid}', '{o.profile}', 'P3 {key}', '{o.org}');"
        f"INSERT INTO public.organization_teams (id, organization_id, name, branch) VALUES ('{team}', '{o.org}', 'Old team', 'HQ');"
        "INSERT INTO public.resource_collaborators (id, resource_type, resource_id, app_user_id, access_level, invited_by) "
        f"VALUES ('{collab}', 'project', '{pid}', '{o.member_app_user}', 'comment', '{o.auth}');"
    )
    r = d.as_user(o.auth, f"SELECT (SELECT name FROM public.organization_teams WHERE id = '{team}') || '|' || "
                          f"(SELECT access_level FROM public.resource_collaborators WHERE id = '{collab}');")
    assert r.stdout.strip() == "Old team|comment", f"{key}: existing rows not readable: {r.stdout}{r.stderr}"
    r = d.as_user(o.auth, f"WITH a AS (DELETE FROM public.organization_teams WHERE id = '{team}' RETURNING 1), "
                          f"b AS (DELETE FROM public.resource_collaborators WHERE id = '{collab}' RETURNING 1) "
                          "SELECT (SELECT count(*) FROM a) || '|' || (SELECT count(*) FROM b);")
    assert r.stdout.strip() == "1|1", f"{key}: existing rows not deletable: {r.stdout}{r.stderr}"


@pytest.mark.parametrize("table", STRUCTURE)
@pytest.mark.parametrize("key", ["free", "professional", "missing"])
def test_existing_structure_rows_stay_readable_and_deletable(applied, world, key, table):
    d, o = applied[0], world.orgs[key]
    rid = _uuid()
    seed = {
        "departments": f"INSERT INTO public.departments (id, organization_id, department, code) VALUES ('{rid}', '{o.org}', 'Old dept', 'O-{rid[:8]}');",
        "config_roles": f"INSERT INTO public.config_roles (id, organization_id, role, department) VALUES ('{rid}', '{o.org}', 'Old role', 'Dept');",
        "branches": f"INSERT INTO public.branches (id, organization_id, name, location) VALUES ('{rid}', '{o.org}', 'Old branch', 'HQ');",
        "app_users": f"INSERT INTO public.app_users (id, organization_id, email, role) VALUES ('{rid}', '{o.org}', 'old-{rid[:8]}@example.com', 'member');",
    }[table]
    d.admin(seed)
    r = d.as_user(o.auth, f"SELECT count(*) FROM public.{table} WHERE id = '{rid}';")
    assert r.stdout.strip() == "1", f"{key}: existing {table} row not readable: {r.stdout}{r.stderr}"
    r = d.as_user(o.auth, f"WITH x AS (DELETE FROM public.{table} WHERE id = '{rid}' RETURNING 1) SELECT count(*) FROM x;")
    assert r.stdout.strip() == "1", f"{key}: existing {table} row not deletable: {r.stdout}{r.stderr}"


# ---- audit of the app-admin bypass ---------------------------------------------

LOG = "avaloka_meta.org_feature_bypass_log"
LOG_COLUMNS = "id,logged_at,actor_user_id,organization_id,source,action,operation"


def _require_log(d: Db) -> None:
    assert d.scalar(f"SELECT to_regclass('{LOG}') IS NOT NULL") == "t", f"{LOG} does not exist"


def _log_rows(d: Db, actor: str) -> list[str]:
    _require_log(d)
    out = d.scalar(f"SELECT coalesce(string_agg(row_to_json(l)::text, E'\\n' ORDER BY id), '') FROM {LOG} l "
                   f"WHERE actor_user_id = '{actor}'")
    return [ln for ln in out.splitlines() if ln]


def _log_count(d: Db, actor: str) -> int:
    _require_log(d)
    return int(d.scalar(f"SELECT count(*) FROM {LOG} WHERE actor_user_id = '{actor}'"))


def test_audit_log_holds_who_what_where_when_and_nothing_else(applied):
    d = applied[0]
    cols = d.scalar("SELECT string_agg(column_name, ',' ORDER BY ordinal_position) FROM information_schema.columns "
                    "WHERE table_schema = 'avaloka_meta' AND table_name = 'org_feature_bypass_log'")
    assert cols == LOG_COLUMNS


@pytest.mark.parametrize("table", GATED)
def test_a_admin_bypass_write_is_allowed_and_audited_once(applied, world, table):
    """(a) An admin on a Free organisation writes through the bypass: the write
    succeeds and exactly one audit row records who, what, which org and when,
    with no customer data."""
    d, o = applied[0], world.orgs["admin_on_free"]
    marker = f"Secret-{secrets.token_hex(4)}"
    if table == "organization_teams":
        stmt = (f"INSERT INTO public.organization_teams (organization_id, name, branch) "
                f"VALUES ('{o.org}', '{marker}', 'HQ') RETURNING 'ok';")
    elif table == "resource_collaborators":
        pid = _uuid()
        d.admin(f"INSERT INTO public.projects (id, owner_id, name, organization_id) VALUES ('{pid}', '{o.profile}', '{marker}', '{o.org}');")
        stmt = ("INSERT INTO public.resource_collaborators (resource_type, resource_id, app_user_id, access_level, invited_by) "
                f"VALUES ('project', '{pid}', '{o.member_app_user}', 'view', '{o.auth}') RETURNING 'ok';")
    else:
        stmt = _structure_insert(o, table, marker)
    before = _log_count(d, o.auth)
    r = d.as_user(o.auth, stmt)
    assert r.returncode == 0 and r.stdout.strip() == "ok", f"bypass write refused: {r.stderr}"
    rows = _log_rows(d, o.auth)
    assert len(rows) == before + 1, f"expected exactly one new audit row, got {len(rows) - before}"
    row = json.loads(rows[-1])
    assert set(row) == set(LOG_COLUMNS.split(","))
    assert (row["actor_user_id"], row["organization_id"], row["source"], row["action"], row["operation"]) == \
        (o.auth, o.org, "rls", table, "INSERT")
    assert row["logged_at"]
    assert marker not in rows[-1] and "@example.com" not in rows[-1], "customer data in the audit row"


def test_a2_admin_bypass_update_is_audited_once(applied, world):
    d, o = applied[0], world.orgs["admin_on_free"]
    before = _log_count(d, o.auth)
    r = d.as_user(o.auth, f"WITH u AS (UPDATE public.branches SET location = 'Elsewhere' WHERE id = '{o.branch}' RETURNING 1) SELECT count(*) FROM u;")
    assert r.stdout.strip() == "1", r.stderr
    rows = _log_rows(d, o.auth)
    assert len(rows) == before + 1
    row = json.loads(rows[-1])
    assert (row["action"], row["operation"], row["organization_id"]) == ("branches", "UPDATE", o.org)
    assert "Elsewhere" not in rows[-1]


@pytest.mark.parametrize("key", ["free", "professional", "missing"])
def test_b_non_admin_refused_and_not_audited(applied, world, key):
    """(b) A non-admin below Enterprise is refused, and nothing is logged."""
    d, o = applied[0], world.orgs[key]
    before = _log_count(d, o.auth)
    r = d.as_user(o.auth, _structure_insert(o, "branches", key))
    assert r.returncode != 0 and RLS_DENIED in r.stderr
    r = d.as_user(o.auth, "SELECT public.log_org_feature_bypass('upsertBranch');")
    assert r.returncode == 0, r.stderr
    assert _log_count(d, o.auth) == before == 0


@pytest.mark.parametrize("key", ["enterprise", "enterprise_trial"])
def test_c_enterprise_writes_are_not_bypass_rows(applied, world, key):
    """(c) Enterprise writes are not bypass uses: no audit row, from RLS or the RPC."""
    d, o = applied[0], world.orgs[key]
    r = d.as_user(o.auth, _structure_insert(o, "departments", key))
    assert r.returncode == 0, r.stderr
    r = d.as_user(o.auth, "SELECT public.log_org_feature_bypass('upsertDepartment');")
    assert r.returncode == 0, r.stderr
    assert _log_count(d, o.auth) == 0


def test_server_action_bypass_is_logged_with_its_name_only(applied, world):
    d, o = applied[0], world.orgs["admin_on_free"]
    before = _log_count(d, o.auth)
    r = d.as_user(o.auth, "SELECT public.log_org_feature_bypass('inviteAppUser');")
    assert r.returncode == 0, r.stderr
    rows = _log_rows(d, o.auth)
    assert len(rows) == before + 1
    row = json.loads(rows[-1])
    assert (row["source"], row["action"], row["operation"], row["organization_id"]) == ("server", "inviteAppUser", "CALL", o.org)
    # Free text (an email, a name) is refused, so it cannot end up in the log.
    r = d.as_user(o.auth, "SELECT public.log_org_feature_bypass('invite someone@example.com');")
    assert r.returncode != 0 and "invalid action name" in r.stderr
    assert _log_count(d, o.auth) == before + 1


def test_a3_admin_on_pro_adding_a_free_member_is_allowed_and_audited_once(applied, world):
    d, o = applied[0], world.orgs["admin_on_pro"]
    before = _log_count(d, o.auth)
    r = _add_to_team(d, o, o.free_member_app)
    assert (r.returncode, r.stdout.strip()) == (0, "1"), r.stderr
    rows = _log_rows(d, o.auth)
    assert len(rows) == before + 1, f"expected exactly one new audit row, got {len(rows) - before}"
    row = json.loads(rows[-1])
    assert (row["actor_user_id"], row["organization_id"], row["source"], row["action"], row["operation"]) == \
        (o.auth, o.org, "rls", "app_users.team_member", "UPDATE")
    assert "@example.com" not in rows[-1]


def test_a4_admin_on_pro_logs_enterprise_bypass_but_not_pro_allowed_writes(applied, world):
    d, o = applied[0], world.orgs["admin_on_pro"]
    before = _log_count(d, o.auth)
    r = d.as_user(o.auth, f"INSERT INTO public.organization_teams (organization_id, name, branch) VALUES ('{o.org}', 'Pro team', 'HQ') RETURNING 'ok';")
    assert r.stdout.strip() == "ok", r.stderr
    assert _log_count(d, o.auth) == before, "a team write Professional is allowed to make is not a bypass"
    r = d.as_user(o.auth, _structure_insert(o, "departments", "a4"))
    assert r.stdout.strip() == "ok", r.stderr
    rows = _log_rows(d, o.auth)
    assert len(rows) == before + 1
    assert json.loads(rows[-1])["action"] == "departments"


def test_b2_pro_non_admin_free_member_refused_and_not_audited(applied, world):
    d, o = applied[0], world.orgs["professional"]
    r = _add_to_team(d, o, o.free_member_app)
    assert r.returncode != 0 and TEAM_MSG in r.stderr
    assert _log_count(d, o.auth) == 0


def test_c2_professional_team_and_invite_writes_are_not_bypass_rows(applied, world):
    d, o = applied[0], world.orgs["professional"]
    r = d.as_user(o.auth, f"INSERT INTO public.organization_teams (organization_id, name, branch) VALUES ('{o.org}', 'T', 'HQ') RETURNING 'ok';")
    assert r.stdout.strip() == "ok", r.stderr
    r = d.as_user(o.auth, "SELECT public.log_org_feature_bypass('upsertTeam', 'professional');")
    assert r.returncode == 0, r.stderr
    assert _log_count(d, o.auth) == 0


def test_server_rpc_logs_only_at_the_level_the_bypass_was_needed(applied, world):
    d, o = applied[0], world.orgs["admin_on_pro"]
    before = _log_count(d, o.auth)
    r = d.as_user(o.auth, "SELECT public.log_org_feature_bypass('upsertTeam', 'professional');")
    assert r.returncode == 0 and _log_count(d, o.auth) == before, "Professional needs no bypass for teams"
    r = d.as_user(o.auth, "SELECT public.log_org_feature_bypass('upsertBranch', 'enterprise');")
    assert r.returncode == 0 and _log_count(d, o.auth) == before + 1
    r = d.as_user(o.auth, "SELECT public.log_org_feature_bypass('upsertBranch', 'platinum');")
    assert r.returncode != 0 and "invalid plan level" in r.stderr


@pytest.mark.parametrize("role", ["anon", "authenticated", "service_role"])
def test_d_api_roles_cannot_read_or_tamper_with_the_audit_log(applied, world, role):
    """(d) No API role can read, add to, change or erase the audit log."""
    d = applied[0]
    _require_log(d)
    assert d.scalar(f"SELECT has_schema_privilege('{role}', 'avaloka_meta', 'USAGE')") == "f"
    for priv in ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE"):
        assert d.scalar(f"SELECT has_table_privilege('{role}', '{LOG}', '{priv}')") == "f", f"{role} has {priv}"
    for stmt in (f"SELECT count(*) FROM {LOG};",
                 f"INSERT INTO {LOG} (actor_user_id, source, action, operation) VALUES (gen_random_uuid(), 'rls', 'x', 'INSERT');",
                 f"UPDATE {LOG} SET action = 'x';",
                 f"DELETE FROM {LOG};"):
        r = d.psql(f"BEGIN; SET LOCAL ROLE {role}; {stmt} COMMIT;", check=False)
        assert r.returncode != 0 and "permission denied" in r.stderr, f"{role}: {stmt} -> {r.stdout}{r.stderr}"
    assert d.scalar("SELECT count(*) FROM information_schema.role_table_grants WHERE table_schema = 'avaloka_meta' "
                    "AND table_name = 'org_feature_bypass_log' AND grantee IN ('anon','authenticated','service_role','PUBLIC')") == "0"


def test_the_migration_writes_no_data():
    """Policies and functions only: no existing row is inserted, changed or removed."""
    sql = _migration_sql().upper()
    for verb in ("INSERT INTO PUBLIC.", "UPDATE PUBLIC.", "DELETE FROM PUBLIC.", "TRUNCATE"):
        assert verb not in sql, f"the migration writes data ({verb.strip()})"


def test_share_links_are_not_plan_gated(applied, world):
    """Email results and public links are not org features: a Free owner can still
    create a share link for their own project."""
    d, o = applied[0], world.orgs["free"]
    r = d.as_user(o.auth, "INSERT INTO public.resource_share_links (resource_type, resource_id, is_public) "
                          f"VALUES ('project', '{o.project}', true) RETURNING is_public;")
    assert r.returncode == 0 and r.stdout.strip() == "t", r.stderr
