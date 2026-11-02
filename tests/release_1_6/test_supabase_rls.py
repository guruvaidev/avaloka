"""1.6 release suite: Supabase row-level security, against the migrations a
self-hosted install actually applies.

What is applied, and why this way
---------------------------------
deploy/helm/avaloka/templates/supabase.yaml runs a post-install Job that does

    for f in $(ls /migrations/*.sql | sort); do
      psql ... -v ON_ERROR_STOP=0 -f "$f" || echo "(non-fatal errors ignored)"
    done

over deploy/helm/avaloka/files/supabase-migrations/. The harness below
replays exactly that: the same directory, the same sort order, statement by
statement on an autocommit connection, so a failing statement is skipped and
an explicit ``begin; ... commit;`` block aborts the way psql's does.

Supabase provides objects that plain Postgres does not (the ``auth`` schema
with ``auth.users`` / ``auth.uid()`` / ``auth.jwt()``, the ``storage``
schema, the ``supabase_realtime`` publication, and the ``anon`` /
``authenticated`` / ``service_role`` roles). ``_SUPABASE_STUB`` creates the
minimum of each. ``auth.uid()`` reads ``request.jwt.claim.sub`` the way
PostgREST sets it, and a query "as a user" runs under ``SET ROLE
authenticated`` with that claim, so the policies are evaluated exactly as
they are for a signed-in browser.

Every test asserts its preconditions (the policy or function under test is
present after the replay) so a harness that silently failed to apply a
migration fails loudly instead of passing for the wrong reason.

Marker ``integration``: needs a Postgres admin URL in
``AVALOKA_TEST_POSTGRES_URL``. A throwaway database is created per session
and dropped afterwards.
"""
from __future__ import annotations

import json
import os
import secrets
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pytest

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
MIGRATIONS = REPO / "deploy" / "helm" / "avaloka" / "files" / "supabase-migrations"

_SUPABASE_STUB = r"""
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


def _split(sql: str) -> List[str]:
    import sqlparse

    return [s for s in (x.strip() for x in sqlparse.split(sql)) if s and not _only_comments(s)]


def _only_comments(stmt: str) -> bool:
    return all((not ln.strip()) or ln.strip().startswith("--") for ln in stmt.splitlines())


def replay_migrations(dsn: str, directory: Path = MIGRATIONS) -> List[Tuple[str, str]]:
    """Apply every *.sql in ``directory`` in ``ls | sort`` order, one
    statement at a time, ignoring failures like ``psql -v ON_ERROR_STOP=0``.
    Each file gets its own session, as each psql -f does (0-schema.sql sets
    search_path for its session only). Returns (file, first error line) for
    every failed statement."""
    import psycopg2

    errors: List[Tuple[str, str]] = []
    for path in sorted(directory.glob("*.sql"), key=lambda p: p.name):
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        cur = conn.cursor()
        try:
            for stmt in _split(path.read_text(encoding="utf-8")):
                try:
                    cur.execute(stmt)
                except Exception as exc:  # noqa: BLE001 - mirrors ON_ERROR_STOP=0
                    errors.append((path.name, str(exc).splitlines()[0][:200]))
        finally:
            conn.close()  # an unfinished begin; is rolled back, as at psql exit
    return errors


class RlsDb:
    def __init__(self, url: str, errors: List[Tuple[str, str]]):
        self.url = url
        self.errors = errors

    @contextmanager
    def admin(self) -> Iterator:
        import psycopg2

        conn = psycopg2.connect(self.url)
        conn.autocommit = True
        try:
            yield conn.cursor()
        finally:
            conn.close()

    @contextmanager
    def as_user(self, auth_uid: str, email: Optional[str] = None) -> Iterator:
        """A cursor that runs as a signed-in browser: role ``authenticated``
        with the PostgREST JWT claims set, inside one transaction."""
        import psycopg2

        conn = psycopg2.connect(self.url)
        try:
            cur = conn.cursor()
            cur.execute("SET LOCAL ROLE authenticated")
            claims = {"sub": auth_uid, "role": "authenticated", "email": email or ""}
            cur.execute("SELECT set_config('request.jwt.claim.sub', %s, true)", (auth_uid,))
            cur.execute("SELECT set_config('request.jwt.claim.role', 'authenticated', true)")
            cur.execute("SELECT set_config('request.jwt.claim.email', %s, true)", (email or "",))
            cur.execute("SELECT set_config('request.jwt.claims', %s, true)", (json.dumps(claims),))
            yield cur
            conn.commit()
        finally:
            conn.close()

    def has_policy(self, table: str, name: str) -> bool:
        with self.admin() as cur:
            cur.execute(
                "SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename=%s AND policyname=%s",
                (table, name),
            )
            return cur.fetchone() is not None

    def has_function(self, name: str) -> bool:
        with self.admin() as cur:
            cur.execute(
                "SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                "WHERE n.nspname='public' AND p.proname=%s",
                (name,),
            )
            return cur.fetchone() is not None


@pytest.fixture(scope="module")
def rls_db() -> Iterator[RlsDb]:
    admin_url = os.environ.get("AVALOKA_TEST_POSTGRES_URL")
    if not admin_url:
        pytest.skip("AVALOKA_TEST_POSTGRES_URL is not set (an admin Postgres URL; "
                    "see bitbucket-pipelines.yml release-1-6-integration)")
    import psycopg2
    from sqlalchemy.engine import make_url

    name = "rls16_" + secrets.token_hex(4)
    root = psycopg2.connect(admin_url)
    root.autocommit = True
    root.cursor().execute(f'CREATE DATABASE "{name}"')
    url = make_url(admin_url).set(database=name, drivername="postgresql")
    dsn = url.render_as_string(hide_password=False)
    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        conn.cursor().execute(_SUPABASE_STUB)
        # Supabase's default search_path; the migrations rely on it.
        conn.cursor().execute(f'ALTER DATABASE "{name}" SET search_path TO "$user", public, extensions')
        conn.close()
        errors = replay_migrations(dsn)
        yield RlsDb(dsn, errors)
    finally:
        root.cursor().execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s", (name,))
        root.cursor().execute(f'DROP DATABASE IF EXISTS "{name}"')
        root.close()


# ---------------------------------------------------------------------------
# Seed: one organization, users A..F. Rows are written as the admin with
# session_replication_role = replica, so signup triggers do not run and the
# policies see exactly this data. All identities are synthetic.
# ---------------------------------------------------------------------------

def _uuid() -> str:
    return str(uuid.uuid4())


class World:
    def __init__(self):
        self.org = _uuid()
        self.users: Dict[str, Dict[str, str]] = {}
        for key, role in [("A", "owner"), ("B", "member"), ("C", "member"),
                          ("D", "member"), ("E", "member"), ("F", "member")]:
            self.users[key] = {"auth": _uuid(), "profile": _uuid(), "app_user": _uuid(),
                               "email": f"user-{key.lower()}@example.com", "role": role}

    def auth(self, key: str) -> str:
        return self.users[key]["auth"]

    def profile(self, key: str) -> str:
        return self.users[key]["profile"]

    def app_user(self, key: str) -> str:
        return self.users[key]["app_user"]

    def email(self, key: str) -> str:
        return self.users[key]["email"]


@pytest.fixture(scope="module")
def world(rls_db) -> World:
    w = World()
    with rls_db.admin() as cur:
        cur.execute("SET session_replication_role = replica")
        for key, u in w.users.items():
            cur.execute("INSERT INTO auth.users (id, email, raw_user_meta_data) VALUES (%s, %s, %s)",
                        (u["auth"], u["email"], json.dumps({"full_name": f"Jane Doe {key}"})))
            cur.execute("INSERT INTO public.profiles (id, user_id, full_name, organization_id) "
                        "VALUES (%s, %s, %s, %s)", (u["profile"], u["auth"], f"Jane Doe {key}", w.org))
        cur.execute("INSERT INTO public.organizations (id, name, owner_profile_id) VALUES (%s, %s, %s)",
                    (w.org, "Example Org", w.profile("A")))
        for key, u in w.users.items():
            cur.execute("INSERT INTO public.app_users (id, organization_id, email, auth_user_id, profile_id, role) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (u["app_user"], w.org, u["email"], u["auth"], u["profile"], u["role"]))
        cur.execute("SET session_replication_role = origin")
    return w


def _admin_insert(rls_db, sql: str, params: tuple) -> None:
    with rls_db.admin() as cur:
        cur.execute("SET session_replication_role = replica")
        cur.execute(sql, params)
        cur.execute("SET session_replication_role = origin")


# ---------------------------------------------------------------------------
# NEW-BG-03: an analysis outside a project is private to its owner
# ---------------------------------------------------------------------------

@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "NEW-BG-03: policy 'org members read org analyses' "
    "(ui/supabase/migrations/20260821143606_...sql) lets every member of the "
    "organization read every analysis, with no project check"))
def test_newbg03_analysis_outside_project_is_private(rls_db, world):
    """NEW-BG-03: an analysis that is not in a project is visible to its owner
    and not to another member of the same organization."""
    assert rls_db.has_policy("analyses", "own analyses select")
    aid = _uuid()
    _admin_insert(rls_db,
                  "INSERT INTO public.analyses (id, owner_id, project_id, name, organization_id) "
                  "VALUES (%s, %s, NULL, %s, %s)",
                  (aid, world.profile("A"), "Private analysis", world.org))

    with rls_db.as_user(world.auth("A"), world.email("A")) as cur:
        cur.execute("SELECT count(*) FROM public.analyses WHERE id = %s", (aid,))
        assert cur.fetchone()[0] == 1, "precondition: the owner sees their own analysis"

    with rls_db.as_user(world.auth("B"), world.email("B")) as cur:
        cur.execute("SELECT public.current_org_id()")
        assert str(cur.fetchone()[0]) == world.org, "precondition: B is in the same organization"
        cur.execute("SELECT count(*) FROM public.analyses WHERE id = %s", (aid,))
        assert cur.fetchone()[0] == 0


# ---------------------------------------------------------------------------
# UI-B18: adding a collaborator to a project works, and the collaborator sees it
# ---------------------------------------------------------------------------

def test_uib18_adding_project_collaborator_does_not_recurse(rls_db, world):
    """UI-B18: the project owner adds an org member as a collaborator without a
    policy error, and the collaborator can then read the project (fix:
    20261013120000_fix_resource_collaborators_rls.sql)."""
    pid = _uuid()
    _admin_insert(rls_db,
                  "INSERT INTO public.projects (id, owner_id, name, organization_id) VALUES (%s, %s, %s, %s)",
                  (pid, world.profile("A"), "Shared project", world.org))

    with rls_db.as_user(world.auth("A"), world.email("A")) as cur:
        cur.execute(
            "INSERT INTO public.resource_collaborators "
            "(resource_type, resource_id, app_user_id, access_level, invited_by) "
            "VALUES ('project', %s, %s, 'view', %s) RETURNING id",
            (pid, world.app_user("C"), world.auth("A")))
        assert cur.fetchone()[0]
        cur.execute("SELECT id FROM public.projects WHERE id = %s", (pid,))
        assert [str(r[0]) for r in cur.fetchall()] == [pid]

    with rls_db.as_user(world.auth("C"), world.email("C")) as cur:
        cur.execute("SELECT id, name FROM public.projects WHERE id = %s", (pid,))
        assert [(str(r[0]), r[1]) for r in cur.fetchall()] == [(pid, "Shared project")]


# ---------------------------------------------------------------------------
# UI-B17: report access levels for collaborators
# ---------------------------------------------------------------------------

def test_uib17_shared_report_access_levels(rls_db, world):
    """UI-B17: get_my_report_access returns edit for the creator, the shared
    level for each collaborator, and none for an org member it was not shared
    with (fix: 20261013130000_fix_project_delete_and_report_access.sql)."""
    aid, rid = _uuid(), _uuid()
    _admin_insert(rls_db,
                  "INSERT INTO public.analyses (id, owner_id, project_id, name, organization_id) "
                  "VALUES (%s, %s, NULL, %s, %s)",
                  (aid, world.profile("A"), "Report source", world.org))
    _admin_insert(rls_db,
                  "INSERT INTO public.reports (id, organization_id, analysis_id, title, created_by) "
                  "VALUES (%s, %s, %s, %s, %s)",
                  (rid, world.org, aid, "Quarterly report", world.auth("A")))
    for key, level in [("C", "view"), ("D", "comment"), ("E", "edit")]:
        _admin_insert(rls_db,
                      "INSERT INTO public.resource_collaborators "
                      "(resource_type, resource_id, app_user_id, access_level, invited_by) "
                      "VALUES ('report', %s, %s, %s, %s)",
                      (rid, world.app_user(key), level, world.auth("A")))

    got = {}
    for key in ["A", "C", "D", "E", "F"]:
        with rls_db.as_user(world.auth(key), world.email(key)) as cur:
            cur.execute("SELECT public.get_my_report_access(%s)", (rid,))
            got[key] = cur.fetchone()[0]
    assert got == {"A": "edit", "C": "view", "D": "comment", "E": "edit", "F": "none"}
