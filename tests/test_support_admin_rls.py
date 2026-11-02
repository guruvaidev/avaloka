"""Support-ticket RLS: "support admin" is the user_roles 'admin' role, not an email.

Applies the REAL migrations that define support_tickets, user_roles,
is_support_admin() and the support policies, in the hook's order, to a scratch
database on the server named by ``AVALOKA_TEST_PG_ADMIN_URL``, then queries as
Supabase's ``authenticated`` role with request.jwt.claims set the way PostgREST
sets them. auth.uid()/auth.jwt() are the minimal Supabase shims.

The address the old is_support_admin() hard-coded is read from that old
migration at run time, so it is not repeated here.

Without the URL or psql the test skips on a laptop and FAILS when CI is set.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import secrets
import shutil
import subprocess
from urllib.parse import unquote, urlsplit

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MIGRATIONS = REPO_ROOT / "ui" / "supabase" / "migrations"
ROLES = "20260614152614_16551223-ff94-4d2a-8084-7867c54c5eb6.sql"
TICKETS = "20260801111759_bd149db9-b166-4eca-80bf-4f51f6b200b2.sql"
OLD_HELPER = "20260818080148_785fcca9-6d9d-4c06-9a61-c1894879e82c.sql"
FORWARD = "20261101120000_support_admin_from_roles.sql"

ADMIN_URL = os.environ.get("AVALOKA_TEST_PG_ADMIN_URL", "").strip()

SHIMS = """
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN CREATE ROLE anon NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN CREATE ROLE authenticated NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN CREATE ROLE service_role NOLOGIN BYPASSRLS; END IF;
END $$;
CREATE SCHEMA auth;
CREATE TABLE auth.users (id uuid PRIMARY KEY, email text, email_confirmed_at timestamptz);
CREATE FUNCTION auth.jwt() RETURNS jsonb LANGUAGE sql STABLE AS
  $$ SELECT coalesce(nullif(current_setting('request.jwt.claims', true), ''), '{}')::jsonb $$;
CREATE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS
  $$ SELECT nullif(auth.jwt() ->> 'sub', '')::uuid $$;
GRANT USAGE ON SCHEMA auth TO anon, authenticated, service_role;
GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;
"""


def _require(condition: bool, message: str) -> None:
    if condition:
        return
    if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
        pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
    pytest.skip(message)


def _old_hardcoded_address() -> str:
    text = (MIGRATIONS / OLD_HELPER).read_text(encoding="utf-8")
    m = re.search(r"auth\.jwt\(\)\s*->>\s*'email'\)\s*=\s*'([^']+)'", text)
    assert m, "could not read the address the old is_support_admin() compared with"
    return m.group(1)


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

    def apply(self, filename: str) -> None:
        env = dict(self.env, PGDATABASE=self.name)
        subprocess.run(["psql", "-X", "-q", "-v", "ON_ERROR_STOP=1", "--single-transaction",
                        "-f", str(MIGRATIONS / filename)], env=env, check=True, capture_output=True, text=True)

    def as_user(self, uid: str, email: str, sql: str) -> str:
        claims = json.dumps({"sub": uid, "email": email, "role": "authenticated"}).replace("'", "''")
        return self.psql(
            "BEGIN;\nSET LOCAL ROLE authenticated;\n"
            f"SELECT set_config('request.jwt.claims', '{claims}', true) \\g /dev/null\n"
            f"{sql}\nROLLBACK;\n"
        )


ALICE = "11111111-1111-1111-1111-111111111111"   # ordinary user, two tickets
BOB = "22222222-2222-2222-2222-222222222222"     # ordinary user, one ticket
DESK = "33333333-3333-3333-3333-333333333333"    # holds the old hard-coded address, no role
ADMIN = "44444444-4444-4444-4444-444444444444"   # 'admin' in user_roles, ordinary address


def _setup(db: Db) -> str:
    desk_email = _old_hardcoded_address()
    db.psql(SHIMS)
    # Every migration in the hook's order that these objects depend on. A tree
    # without the forward migration runs against the old helper, so a revert
    # fails on the assertions below rather than on a missing file.
    files = [ROLES, TICKETS, OLD_HELPER] + ([FORWARD] if (MIGRATIONS / FORWARD).is_file() else [])
    for f in files:
        db.apply(f)
    db.psql(
        "INSERT INTO auth.users (id, email, email_confirmed_at) VALUES "
        f"('{ALICE}', 'alice@example.com', now()), ('{BOB}', 'bob@example.com', now()), "
        f"('{DESK}', '{desk_email}', now()), ('{ADMIN}', 'ops@example.com', now());\n"
        f"INSERT INTO public.user_roles (user_id, role) VALUES ('{ADMIN}', 'admin');\n"
        "INSERT INTO public.support_tickets (user_id, subject, message) VALUES "
        f"('{ALICE}', 'a1', 'm'), ('{ALICE}', 'a2', 'm'), ('{BOB}', 'b1', 'm');\n"
    )
    return desk_email


@pytest.fixture
def db():
    _require(bool(ADMIN_URL), "AVALOKA_TEST_PG_ADMIN_URL is not set; the support RLS was not exercised")
    _require(shutil.which("psql") is not None, "psql is not installed; the support RLS cannot be exercised")
    d = Db(ADMIN_URL, f"rls_{secrets.token_hex(6)}")
    d.psql(f'CREATE DATABASE "{d.name}"', db=d.admin_db)
    try:
        yield d
    finally:
        d.psql(f'DROP DATABASE IF EXISTS "{d.name}"', db=d.admin_db)


COUNT = "SELECT count(*) FROM public.support_tickets;"
TOUCH_BOBS = "WITH u AS (UPDATE public.support_tickets SET status = 'answered' WHERE subject = 'b1' RETURNING 1) SELECT count(*) FROM u;"


def test_the_old_hardcoded_address_without_the_role_sees_no_other_tickets(db):
    desk_email = _setup(db)
    assert db.as_user(DESK, desk_email, COUNT) == "0", "the old literal address still reads every ticket"
    assert db.as_user(DESK, desk_email, TOUCH_BOBS) == "0", "the old literal address can still update others' tickets"
    assert db.as_user(DESK, desk_email, "SELECT public.is_support_admin();") == "f"


def test_an_admin_role_account_reads_and_updates_every_ticket(db):
    _setup(db)
    assert db.as_user(ADMIN, "ops@example.com", COUNT) == "3"
    assert db.as_user(ADMIN, "ops@example.com", TOUCH_BOBS) == "1"
    assert db.as_user(ADMIN, "ops@example.com", "SELECT public.is_support_admin();") == "t"


def test_an_ordinary_user_sees_only_their_own(db):
    _setup(db)
    assert db.as_user(ALICE, "alice@example.com", COUNT) == "2"
    assert db.as_user(BOB, "bob@example.com", COUNT) == "1"
    assert db.as_user(ALICE, "alice@example.com", TOUCH_BOBS) == "0"


def test_helper_keeps_its_grants_and_search_path_hardening(db):
    _setup(db)
    row = db.psql(
        "SELECT p.prosecdef, coalesce(array_to_string(p.proconfig, ','), ''), "
        "has_function_privilege('anon', p.oid, 'EXECUTE'), "
        "has_function_privilege('authenticated', p.oid, 'EXECUTE'), "
        "has_function_privilege('service_role', p.oid, 'EXECUTE') "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname = 'public' AND p.proname = 'is_support_admin';"
    )
    secdef, config, anon, authed, service = row.split("|")
    assert secdef == "t"
    assert config.startswith("search_path="), config
    assert (anon, authed, service) == ("f", "t", "t")
    public_exec = db.psql(
        "SELECT count(*) FROM pg_proc p, aclexplode(p.proacl) a "
        "WHERE p.proname = 'is_support_admin' AND a.grantee = 0 AND a.privilege_type = 'EXECUTE';"
    )
    assert public_exec == "0", "PUBLIC can execute is_support_admin()"
