"""The migrate hook waits for gotrue, and a forward migration repairs installs it did not.

The chart's migrate hook runs post-install, which does not wait for the gotrue
Deployment to start. The supabase/postgres image's own auth schema has auth.uid()
but no auth.jwt(); gotrue's migrations add it. 20260818080148 (a baseline file)
defines is_support_admin() with auth.jwt() and creates the four support-ticket
policies on top of it. Run before gotrue, those statements fail, the hook records
the baseline file anyway, and the policies are never created.

These tests run the hook's own script (deploy/helm/avaloka/files/db-migrate.sh)
over the REAL migrations that define the support objects, from the chart's copy
and with the chart's frozen baseline, against a scratch database on the server
named by ``AVALOKA_TEST_PG_ADMIN_URL``. The database is first made to look like
the DB image before gotrue: API roles, an auth schema with auth.users and
auth.uid(), and no auth.jwt(). "gotrue arrives" is creating auth.jwt().

Without the URL or psql the test skips on a laptop and FAILS when CI is set.
"""

from __future__ import annotations

import os
import pathlib
import secrets
import shutil
import subprocess
import time
from urllib.parse import unquote, urlsplit

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = REPO_ROOT / "deploy" / "helm" / "avaloka" / "files"
SCRIPT = FILES / "db-migrate.sh"
BASELINE = FILES / "db-migrations.baseline"
MIGRATIONS = FILES / "supabase-migrations"
ROLES = "20260614152614_16551223-ff94-4d2a-8084-7867c54c5eb6"
TICKETS = "20260801111759_bd149db9-b166-4eca-80bf-4f51f6b200b2"
OLD_HELPER = "20260818080148_785fcca9-6d9d-4c06-9a61-c1894879e82c"
FROM_ROLES = "20261101120000_support_admin_from_roles"
REPAIR = "20261103090000_restore_support_ticket_policies"

ADMIN_URL = os.environ.get("AVALOKA_TEST_PG_ADMIN_URL", "").strip()

# The supabase/postgres image before gotrue has run (init-scripts/00000000000001-auth-schema.sql).
IMAGE_BEFORE_GOTRUE = """
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN CREATE ROLE anon NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN CREATE ROLE authenticated NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN CREATE ROLE service_role NOLOGIN BYPASSRLS; END IF;
END $$;
CREATE SCHEMA auth;
CREATE TABLE auth.users (id uuid PRIMARY KEY, email text);
CREATE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS
  $$ SELECT nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $$;
GRANT USAGE ON SCHEMA auth TO anon, authenticated, service_role;
GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;
"""

# What gotrue's migrations add that the support migrations need.
GOTRUE_ARRIVES = """
CREATE FUNCTION auth.jwt() RETURNS jsonb LANGUAGE sql STABLE AS
  $$ SELECT coalesce(nullif(current_setting('request.jwt.claims', true), ''), '{}')::jsonb $$;
"""

SUPPORT_POLICIES = (
    ("support_ticket_messages", "Read own ticket messages"),
    ("support_ticket_messages", "Write own ticket messages"),
    ("support_tickets", "Support admin can view all tickets"),
    ("support_tickets", "Support admin can update tickets"),
)

POLICY_COUNT = "SELECT count(*) FROM pg_policies WHERE schemaname = 'public' AND (tablename, policyname) IN (" + ", ".join(
    f"('{t}', '{p}')" for t, p in SUPPORT_POLICIES
) + ")"

# Everything the repair migration could touch, including object identities: a
# policy dropped and recreated, or a function recreated, would change an oid.
FINGERPRINT = """
SELECT string_agg(x, E'\\n' ORDER BY x) FROM (
  SELECT 'fn ' || p.oid || ' owner=' || p.proowner::regrole || ' acl=' || coalesce(p.proacl::text, '-')
         || E'\\n' || pg_get_functiondef(p.oid)
    FROM pg_proc p WHERE p.oid = to_regprocedure('public.is_support_admin()')
  UNION ALL
  SELECT 'policy ' || pol.oid || ' ' || c.relname || ' / ' || pol.polname || ' cmd=' || pol.polcmd::text
         || ' permissive=' || pol.polpermissive || ' roles=' || pol.polroles::text
         || ' using=' || coalesce(pg_get_expr(pol.polqual, pol.polrelid), '-')
         || ' check=' || coalesce(pg_get_expr(pol.polwithcheck, pol.polrelid), '-')
    FROM pg_policy pol JOIN pg_class c ON c.oid = pol.polrelid
   WHERE c.relnamespace = 'public'::regnamespace AND c.relname IN ('support_tickets', 'support_ticket_messages')
  UNION ALL
  SELECT 'table ' || c.relname || ' rls=' || c.relrowsecurity || ' acl=' || coalesce(c.relacl::text, '-')
    FROM pg_class c
   WHERE c.relnamespace = 'public'::regnamespace AND c.relname IN ('support_tickets', 'support_ticket_messages')
) s(x)
"""


def _require(condition: bool, message: str) -> None:
    if condition:
        return
    if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
        pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
    pytest.skip(message)


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

    def psql(self, sql: str, db: str | None = None) -> str:
        out = subprocess.run(["psql", "-X", "-q", "-tA", "-v", "ON_ERROR_STOP=1"], input=sql,
                             env=self.env(PGDATABASE=db or self.name), capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"psql failed rc={out.returncode}: {out.stderr.strip()}")
        return out.stdout.strip()

    def apply(self, version: str, stop_on_error: bool = True) -> None:
        args = ["psql", "-X", "-q", "-v", f"ON_ERROR_STOP={1 if stop_on_error else 0}", "-f", str(MIGRATIONS / f"{version}.sql")]
        if stop_on_error:
            args.insert(-2, "--single-transaction")
        subprocess.run(args, env=self.env(), check=stop_on_error, capture_output=True, text=True)

    def hook(self, mig: pathlib.Path, wait_seconds: int = 60) -> subprocess.Popen:
        env = self.env(MIGRATIONS_DIR=str(mig), ADMIN_EMAILS="", GOTRUE_MAILER_AUTOCONFIRM="true",
                       AUTH_WAIT_SECONDS=str(wait_seconds))
        return subprocess.Popen(["sh", str(SCRIPT)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def policies(self) -> int:
        return int(self.psql(POLICY_COUNT))


@pytest.fixture
def db():
    _require(bool(ADMIN_URL), "AVALOKA_TEST_PG_ADMIN_URL is not set; the migrate hook's wait for gotrue was not exercised")
    _require(shutil.which("psql") is not None, "psql is not installed; the migrate hook cannot be exercised")
    _require(shutil.which("sh") is not None, "no sh")
    d = Db(ADMIN_URL, f"auth_{secrets.token_hex(6)}")
    d.psql(f'CREATE DATABASE "{d.name}"', db=d.admin_db)
    try:
        d.psql(IMAGE_BEFORE_GOTRUE)
        yield d
    finally:
        d.psql(f'DROP DATABASE IF EXISTS "{d.name}"', db=d.admin_db)


def _migration_set(tmp_path: pathlib.Path) -> pathlib.Path:
    """The real support migrations as the hook sees them; the repair only if this tree ships it."""
    mig = tmp_path / "migrations"
    mig.mkdir()
    for v in (ROLES, TICKETS, OLD_HELPER, FROM_ROLES, REPAIR):
        src = MIGRATIONS / f"{v}.sql"
        if src.is_file():
            shutil.copyfile(src, mig / src.name)
    shutil.copyfile(BASELINE, mig / "baseline.txt")
    return mig


def _finish(proc: subprocess.Popen, timeout: int = 180) -> tuple[int | None, str]:
    """(returncode, output); returncode None when the hook was still running at the timeout."""
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return None, out + err
    return proc.returncode, out + err


def test_the_support_files_are_baseline_and_the_repair_is_not():
    baseline = {l.strip() for l in BASELINE.read_text(encoding="utf-8").splitlines()}
    assert {ROLES, TICKETS, OLD_HELPER} <= baseline
    assert REPAIR not in baseline, "the repair must run as a tracked migration on every install"
    assert "auth.jwt()" in (MIGRATIONS / f"{OLD_HELPER}.sql").read_text(encoding="utf-8")


def _fresh_install_racing_gotrue(db: Db, tmp_path: pathlib.Path) -> str:
    """The race: the hook starts while gotrue is still coming up; gotrue arrives 6s later."""
    assert db.psql("SELECT to_regprocedure('auth.jwt()') IS NULL") == "t"
    proc = db.hook(_migration_set(tmp_path))
    time.sleep(6)
    db.psql(GOTRUE_ARRIVES)
    rc, out = _finish(proc)
    assert rc == 0, out
    return out


def test_fresh_install_with_the_hook_before_gotrue_ends_with_the_support_policies(db, tmp_path):
    out = _fresh_install_racing_gotrue(db, tmp_path)
    assert db.policies() == 4, f"support-ticket policies missing after a fresh install:\n{out}"
    assert f"error(s) in baseline migration {OLD_HELPER}" not in out, f"the hook did not wait for gotrue:\n{out}"
    body = db.psql("SELECT prosrc FROM pg_proc WHERE oid = to_regprocedure('public.is_support_admin()')")
    assert "user_roles" in body and "auth.jwt" not in body, f"is_support_admin() is not the user_roles form: {body}"


ALICE = "11111111-1111-1111-1111-111111111111"   # owns the ticket
BOB = "22222222-2222-2222-2222-222222222222"     # another ordinary user
TICKET = "aaaaaaaa-0000-0000-0000-000000000001"


def _as_user(db: Db, uid: str, sql: str) -> subprocess.CompletedProcess:
    """sql as Supabase's authenticated role with this user's claims, rolled back; never raises."""
    claims = f'{{"sub": "{uid}", "role": "authenticated"}}'
    script = (
        "BEGIN;\nSET LOCAL ROLE authenticated;\n"
        f"SELECT set_config('request.jwt.claim.sub', '{uid}', true) \\g /dev/null\n"
        f"SELECT set_config('request.jwt.claims', '{claims}', true) \\g /dev/null\n"
        f"{sql}\nROLLBACK;\n"
    )
    return subprocess.run(["psql", "-X", "-q", "-tA", "-v", "ON_ERROR_STOP=1"], input=script,
                          env=db.env(), capture_output=True, text=True)


def test_after_the_race_a_user_reads_and_posts_on_their_own_ticket(db, tmp_path):
    """Without the support policies support_ticket_messages has RLS on and no policy:
    an ordinary user could neither read nor post on their own ticket."""
    _fresh_install_racing_gotrue(db, tmp_path)
    db.psql(
        f"INSERT INTO auth.users (id, email) VALUES ('{ALICE}', 'alice@example.com'), ('{BOB}', 'bob@example.com');\n"
        f"INSERT INTO public.support_tickets (id, user_id, subject, message) VALUES ('{TICKET}', '{ALICE}', 's', 'm');\n"
        f"INSERT INTO public.support_ticket_messages (ticket_id, author_id, body) VALUES ('{TICKET}', '{ALICE}', 'first');"
    )
    read = f"SELECT count(*) FROM public.support_ticket_messages WHERE ticket_id = '{TICKET}';"
    mine = _as_user(db, ALICE, read)
    assert mine.returncode == 0 and mine.stdout.strip() == "1", (
        f"a user cannot read the messages on their own ticket (got {mine.stdout.strip()!r}) {mine.stderr.strip()}"
    )
    post = _as_user(db, ALICE, f"INSERT INTO public.support_ticket_messages (ticket_id, author_id, body) "
                               f"VALUES ('{TICKET}', '{ALICE}', 'reply') RETURNING 'posted';")
    assert post.returncode == 0 and "posted" in post.stdout, f"a user cannot post on their own ticket: {post.stderr.strip()}"
    # Still closed to everyone else.
    theirs = _as_user(db, BOB, read)
    assert theirs.returncode == 0 and theirs.stdout.strip() == "0", "another user reads someone else's ticket messages"
    intrude = _as_user(db, BOB, f"INSERT INTO public.support_ticket_messages (ticket_id, author_id, body) "
                                f"VALUES ('{TICKET}', '{BOB}', 'x') RETURNING 'posted';")
    assert intrude.returncode != 0 and "row-level security" in intrude.stderr, "another user posts on someone else's ticket"


def test_hook_gives_up_visibly_and_applies_nothing_when_gotrue_never_comes(db, tmp_path):
    rc, out = _finish(db.hook(_migration_set(tmp_path), wait_seconds=4), timeout=60)
    assert rc is not None, f"the hook was still waiting after 60s with AUTH_WAIT_SECONDS=4:\n{out}"
    assert rc != 0, f"the hook proceeded without auth.jwt():\n{out}"
    assert "auth.jwt() does not exist after 4s" in out
    assert db.psql("SELECT to_regclass('avaloka_meta.schema_migrations') IS NULL") == "t", "something was recorded"
    assert db.psql("SELECT to_regclass('public.support_tickets') IS NULL") == "t", "something was applied"


def test_an_install_damaged_by_the_race_is_repaired_on_the_next_deploy(db, tmp_path):
    """State left by the hook before it waited: baseline applied with errors and
    recorded, #649 applied, policies absent. gotrue has since started."""
    for v in (ROLES, TICKETS, OLD_HELPER):
        db.apply(v, stop_on_error=False)       # the hook's baseline behaviour, errors not fatal
    db.apply(FROM_ROLES)
    db.psql(
        "CREATE SCHEMA avaloka_meta;\n"
        "CREATE TABLE avaloka_meta.schema_migrations (version text PRIMARY KEY,"
        " applied_at timestamptz NOT NULL DEFAULT now(), how text NOT NULL DEFAULT 'applied');\n"
        "INSERT INTO avaloka_meta.schema_migrations (version) VALUES "
        + ", ".join(f"('{v}')" for v in (ROLES, TICKETS, OLD_HELPER, FROM_ROLES)) + ";"
    )
    db.psql(GOTRUE_ARRIVES)
    assert db.policies() == 0, "the damaged state was not reproduced"
    assert db.psql("SELECT relrowsecurity FROM pg_class WHERE oid = 'public.support_ticket_messages'::regclass") == "t"

    rc, out = _finish(db.hook(_migration_set(tmp_path)))
    assert rc == 0, out
    assert db.policies() == 4, f"a later deploy did not restore the support-ticket policies:\n{out}"
    body = db.psql("SELECT prosrc FROM pg_proc WHERE oid = to_regprocedure('public.is_support_admin()')")
    assert "user_roles" in body and "auth.jwt" not in body, f"is_support_admin() is not the user_roles form: {body}"


def test_the_repair_changes_nothing_on_a_correct_install(db):
    assert (MIGRATIONS / f"{REPAIR}.sql").is_file(), "this tree ships no repair migration"
    db.psql(GOTRUE_ARRIVES)
    for v in (ROLES, TICKETS, OLD_HELPER, FROM_ROLES):
        db.apply(v)
    assert db.policies() == 4
    before = db.psql(FINGERPRINT)
    db.apply(REPAIR)
    db.apply(REPAIR)
    assert db.psql(FINGERPRINT) == before
