"""The chart's Supabase migrate hook applies each migration once, against real Postgres.

Runs the hook's own script (deploy/helm/avaloka/files/db-migrate.sh) with
``sh`` and ``psql`` against scratch databases on the server named by
``AVALOKA_TEST_PG_ADMIN_URL`` (a role that may CREATE DATABASE). The migration
set is synthetic so the test needs no Supabase image; it mimics the pieces the
script relies on: the marker tables of the newest baseline migrations, auth.users
and public.user_roles.

The canary migration is deliberately non-idempotent: it CREATEs a table (fails
on a re-run) and appends a row to a log (duplicates on a re-run). A re-applied
canary is therefore visible either as an error or as a second log row.

Without the URL or psql the test skips on a laptop and FAILS when CI is set:
a skipped migration test is a green line that tested nothing.
"""

from __future__ import annotations

import os
import pathlib
import secrets
import shutil
import subprocess
from urllib.parse import unquote, urlsplit

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy" / "helm" / "avaloka" / "files" / "db-migrate.sh"
OLD_HOOK_LOOP = 'for f in $(ls "$MIGRATIONS_DIR"/*.sql | sort); do psql -X -q -v ON_ERROR_STOP=0 -f "$f" >/dev/null 2>&1 || true; done'

ADMIN_URL = os.environ.get("AVALOKA_TEST_PG_ADMIN_URL", "").strip()


def _require(condition: bool, message: str) -> None:
    if condition:
        return
    if os.environ.get("CI", "").strip().lower() not in ("", "0", "false"):
        pytest.fail(message + ". CI is set, so this is a failure rather than a skip.")
    pytest.skip(message)


# ---- the synthetic migration set ------------------------------------------------

BASELINE = {
    # Supabase provides these in the real image; the test database does not.
    "0-schema": """
        CREATE SCHEMA IF NOT EXISTS auth;
        CREATE TABLE IF NOT EXISTS auth.users (
            id uuid PRIMARY KEY, email text, email_confirmed_at timestamptz);
    """,
    "20260614152614_roles": """
        DO $$ BEGIN CREATE TYPE public.app_role AS ENUM ('admin','moderator','user');
        EXCEPTION WHEN duplicate_object THEN NULL; END $$;
        CREATE TABLE IF NOT EXISTS public.user_roles (
            id bigserial PRIMARY KEY,
            user_id uuid NOT NULL,
            role public.app_role NOT NULL,
            UNIQUE (user_id, role));
    """,
    "20260619141649_canary": """
        CREATE TABLE public.canary_once (id int);
        CREATE TABLE IF NOT EXISTS public.canary_log (ran_at timestamptz DEFAULT now());
        INSERT INTO public.canary_log DEFAULT VALUES;
    """,
    "20261009094500_notifications": "CREATE TABLE IF NOT EXISTS public.notifications (id int);",
    "20261025120000_ray_jobs": "CREATE TABLE IF NOT EXISTS public.ray_jobs (submission_id text PRIMARY KEY);",
}


def _write_set(root: pathlib.Path, files: dict[str, str], baseline: list[str]) -> pathlib.Path:
    mig = root / "migrations"
    mig.mkdir(parents=True, exist_ok=True)
    for version, body in files.items():
        (mig / f"{version}.sql").write_text(body, encoding="utf-8")
    (mig / "baseline.txt").write_text(
        "# frozen\n" + "\n".join(baseline) + "\n", encoding="utf-8"
    )
    return mig


# ---- database plumbing ----------------------------------------------------------


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

    def admin_sql(self, sql: str) -> None:
        env = self.env(PGDATABASE=self.admin_db)
        subprocess.run(["psql", "-X", "-q", "-v", "ON_ERROR_STOP=1", "-c", sql], env=env, check=True,
                       capture_output=True, text=True)

    def scalar(self, sql: str) -> str:
        out = subprocess.run(["psql", "-X", "-q", "-tA", "-v", "ON_ERROR_STOP=1", "-c", sql],
                             env=self.env(), check=True, capture_output=True, text=True)
        return out.stdout.strip()

    def run_script(self, mig: pathlib.Path, admin_emails: str = "", autoconfirm: str | None = "false") -> subprocess.CompletedProcess:
        extra = {"MIGRATIONS_DIR": str(mig), "ADMIN_EMAILS": admin_emails}
        if autoconfirm is not None:
            extra["GOTRUE_MAILER_AUTOCONFIRM"] = autoconfirm
        return subprocess.run(
            ["sh", str(SCRIPT)], env=self.env(**extra), capture_output=True, text=True, timeout=300,
        )

    def run_old_hook(self, mig: pathlib.Path) -> None:
        """What the chart did before tracking: every file, errors ignored."""
        subprocess.run(["sh", "-c", OLD_HOOK_LOOP], env=self.env(MIGRATIONS_DIR=str(mig)),
                       check=True, capture_output=True, text=True, timeout=300)

    def recorded(self) -> dict[str, str]:
        rows = self.scalar("SELECT version || '=' || how FROM avaloka_meta.schema_migrations ORDER BY version")
        return dict(line.split("=", 1) for line in rows.splitlines() if line)


@pytest.fixture
def db():
    _require(bool(ADMIN_URL), "AVALOKA_TEST_PG_ADMIN_URL is not set; the migrate hook was not exercised against Postgres")
    _require(shutil.which("psql") is not None, "psql is not installed; the migrate hook cannot be exercised")
    _require(shutil.which("sh") is not None, "no sh")
    d = Db(ADMIN_URL, f"mig_{secrets.token_hex(6)}")
    d.admin_sql(f'CREATE DATABASE "{d.name}"')
    try:
        yield d
    finally:
        d.admin_sql(f'DROP DATABASE IF EXISTS "{d.name}"')


def _ok(result: subprocess.CompletedProcess) -> str:
    assert result.returncode == 0, f"migrate script failed rc={result.returncode}\n{result.stdout}\n{result.stderr}"
    return result.stdout + result.stderr


# ---- (a) fresh database ---------------------------------------------------------


def test_a_fresh_database_applies_every_migration_once_and_records_it(db, tmp_path):
    mig = _write_set(tmp_path, BASELINE, list(BASELINE))
    out = _ok(db.run_script(mig))
    assert "untracked" in out
    assert db.recorded() == {v: "applied" for v in BASELINE}
    assert db.scalar("SELECT count(*) FROM public.canary_log") == "1"
    assert "5 applied, 0 already recorded" in out


# ---- (b) second run -------------------------------------------------------------


def test_b_a_second_run_applies_nothing(db, tmp_path):
    mig = _write_set(tmp_path, BASELINE, list(BASELINE))
    _ok(db.run_script(mig))
    before = db.scalar("SELECT string_agg(version || applied_at::text, ',' ORDER BY version) FROM avaloka_meta.schema_migrations")
    out = _ok(db.run_script(mig))
    assert "migration state: tracked" in out
    assert "0 applied, 5 already recorded" in out
    assert db.scalar("SELECT count(*) FROM public.canary_log") == "1"
    after = db.scalar("SELECT string_agg(version || applied_at::text, ',' ORDER BY version) FROM avaloka_meta.schema_migrations")
    assert before == after


# ---- (c) existing install, applied the old way ----------------------------------


def test_c_an_existing_install_is_backfilled_without_reapplying(db, tmp_path):
    mig = _write_set(tmp_path, BASELINE, list(BASELINE))
    db.run_old_hook(mig)
    db.run_old_hook(mig)  # the old hook re-ran everything on every deploy
    # The canary shows the old behaviour: run twice, logged twice.
    assert db.scalar("SELECT count(*) FROM public.canary_log") == "2"
    assert db.scalar("SELECT to_regclass('avaloka_meta.schema_migrations') IS NULL") == "t"

    out = _ok(db.run_script(mig))
    assert "existing install, backfilled 5 baseline migration(s)" in out
    assert db.recorded() == {v: "backfilled" for v in BASELINE}
    # Not re-applied: no third canary row, and no "already exists" error.
    assert db.scalar("SELECT count(*) FROM public.canary_log") == "2"
    assert "ERROR" not in out
    assert "0 applied, 5 already recorded" in out


def test_c2_an_install_older_than_the_newest_baseline_is_not_backfilled(db, tmp_path):
    """Markers missing: the newest baseline file never ran there, so recording it
    as applied would skip it forever. The script runs the baseline the old way."""
    older = {k: v for k, v in BASELINE.items() if k != "20261025120000_ray_jobs"}
    db.run_old_hook(_write_set(tmp_path / "old", older, list(older)))
    mig = _write_set(tmp_path / "new", BASELINE, list(BASELINE))
    out = _ok(db.run_script(mig))
    assert "untracked" in out
    assert db.scalar("SELECT to_regclass('public.ray_jobs') IS NOT NULL") == "t"
    assert db.recorded() == {v: "applied" for v in BASELINE}
    # The canary errors on this one last old-style run (exactly what the old hook
    # would have done on this deploy), is recorded, and never runs again.
    assert "statement error(s) in baseline migration 20260619141649_canary" in out
    out = _ok(db.run_script(mig))
    assert "0 applied, 5 already recorded" in out
    assert db.scalar("SELECT count(*) FROM public.canary_log") == "2"


# ---- (d) a migration added later ------------------------------------------------


@pytest.mark.parametrize("history", ["fresh", "existing"])
def test_d_a_new_migration_runs_once(db, tmp_path, history):
    mig = _write_set(tmp_path, BASELINE, list(BASELINE))
    if history == "existing":
        db.run_old_hook(mig)
    _ok(db.run_script(mig))
    canary_rows = db.scalar("SELECT count(*) FROM public.canary_log")

    (mig / "20261101000000_new.sql").write_text(
        "CREATE TABLE public.added_later (id int);\nINSERT INTO public.canary_log DEFAULT VALUES;\n",
        encoding="utf-8",
    )
    out = _ok(db.run_script(mig))
    assert "1 applied, 5 already recorded" in out
    assert db.recorded()["20261101000000_new"] == "applied"
    assert int(db.scalar("SELECT count(*) FROM public.canary_log")) == int(canary_rows) + 1

    out = _ok(db.run_script(mig))
    assert "0 applied, 6 already recorded" in out
    assert int(db.scalar("SELECT count(*) FROM public.canary_log")) == int(canary_rows) + 1


def test_d2_a_failing_new_migration_fails_the_job_and_leaves_nothing_behind(db, tmp_path):
    mig = _write_set(tmp_path, BASELINE, list(BASELINE))
    _ok(db.run_script(mig))
    (mig / "20261101000000_broken.sql").write_text(
        "CREATE TABLE public.half_done (id int);\nSELECT * FROM public.no_such_table;\n", encoding="utf-8"
    )
    result = db.run_script(mig)
    assert result.returncode != 0
    assert "20261101000000_broken" not in db.recorded()
    assert db.scalar("SELECT to_regclass('public.half_done') IS NULL") == "t", "not rolled back"


def test_version_table_is_not_reachable_by_api_roles(db, tmp_path):
    mig = _write_set(tmp_path, BASELINE, list(BASELINE))
    _ok(db.run_script(mig))
    role = f"anon_{secrets.token_hex(4)}"
    db.admin_sql(f'CREATE ROLE "{role}" NOLOGIN')
    try:
        assert db.scalar(f"SELECT has_schema_privilege('{role}', 'avaloka_meta', 'USAGE')") == "f"
        assert db.scalar(f"SELECT has_table_privilege('{role}', 'avaloka_meta.schema_migrations', 'INSERT')") == "f"
    finally:
        db.admin_sql(f'DROP ROLE IF EXISTS "{role}"')


# ---- admin bootstrap from configuration -----------------------------------------

_ALICE = "11111111-1111-1111-1111-111111111111"
_BOB = "22222222-2222-2222-2222-222222222222"
_CAROL = "33333333-3333-3333-3333-333333333333"
_DAVE = "44444444-4444-4444-4444-444444444444"


def _seed_users(db: Db) -> None:
    db.scalar(
        "INSERT INTO auth.users (id, email, email_confirmed_at) VALUES "
        f"('{_ALICE}', 'alice@example.com', now()),"
        f"('{_BOB}', 'Bob@Example.com', now()),"
        f"('{_CAROL}', 'carol@example.com', NULL),"       # unconfirmed
        f"('{_DAVE}', 'dave@example.com', now())"
    )
    # An admin granted some other way, not in the list: must survive.
    db.scalar(f"INSERT INTO public.user_roles (user_id, role) VALUES ('{_DAVE}', 'admin')")


def _admins(db: Db) -> set[str]:
    rows = db.scalar("SELECT user_id FROM public.user_roles WHERE role = 'admin' ORDER BY 1")
    return set(rows.splitlines())


def test_admin_bootstrap_grants_only_listed_confirmed_accounts_and_never_revokes(db, tmp_path):
    mig = _write_set(tmp_path, BASELINE, list(BASELINE))
    _ok(db.run_script(mig))
    _seed_users(db)

    out = _ok(db.run_script(mig, admin_emails=" ALICE@example.com ,bob@example.com,carol@example.com,nobody@example.com,"))
    assert _admins(db) == {_ALICE, _BOB, _DAVE}
    assert "2 new grant(s)" in out
    for addr in ("alice@", "bob@", "carol@", "nobody@"):
        assert addr not in out.lower(), "the script printed an admin address"

    # Idempotent, and dropping an address from the list revokes nothing.
    out = _ok(db.run_script(mig, admin_emails="alice@example.com"))
    assert "0 new grant(s)" in out
    assert _admins(db) == {_ALICE, _BOB, _DAVE}


def test_admin_bootstrap_waits_for_gotrue_and_never_fails_the_job(db, tmp_path):
    """The DB image's own auth.users has no email_confirmed_at until gotrue migrates."""
    mig = _write_set(tmp_path, BASELINE, list(BASELINE))
    _ok(db.run_script(mig))
    _seed_users(db)
    db.scalar("ALTER TABLE auth.users DROP COLUMN email_confirmed_at")
    out = _ok(db.run_script(mig, admin_emails="alice@example.com"))
    assert "not migrated by gotrue yet, nothing granted" in out
    assert _admins(db) == {_DAVE}
    # Any other failure is a warning, not a failed deploy.
    db.scalar("DROP TABLE public.user_roles")
    db.scalar("CREATE TABLE public.user_roles (user_id uuid, role public.app_role)")  # no unique key
    db.scalar("ALTER TABLE auth.users ADD COLUMN email_confirmed_at timestamptz DEFAULT now()")
    out = _ok(db.run_script(mig, admin_emails="alice@example.com"))
    assert "admin bootstrap failed (SQLSTATE" in out
    assert "alice@" not in out.lower()


@pytest.mark.parametrize("autoconfirm", ["true", "TRUE", "", None])
def test_admin_bootstrap_skips_with_a_warning_unless_autoconfirm_is_off(db, tmp_path, autoconfirm):
    """Autoconfirm on (or not stated): a listed address could be claimed without
    proving ownership, so nothing is granted and the log says why."""
    mig = _write_set(tmp_path, BASELINE, list(BASELINE))
    _ok(db.run_script(mig))
    _seed_users(db)
    result = db.run_script(mig, admin_emails="alice@example.com,bob@example.com", autoconfirm=autoconfirm)
    out = _ok(result)
    assert "WARNING: admin bootstrap skipped: GOTRUE_MAILER_AUTOCONFIRM is not false" in result.stderr
    assert "new grant(s)" not in out
    assert _admins(db) == {_DAVE}
    assert "alice@" not in out.lower() and "bob@" not in out.lower()


@pytest.mark.parametrize("value", ["", " ", " , ,"])
def test_empty_admin_emails_grants_nothing(db, tmp_path, value):
    mig = _write_set(tmp_path, BASELINE, list(BASELINE))
    _ok(db.run_script(mig))
    _seed_users(db)
    out = _ok(db.run_script(mig, admin_emails=value))
    assert "ADMIN_EMAILS is empty, nothing granted" in out
    assert _admins(db) == {_DAVE}


def test_script_is_the_one_the_chart_mounts():
    template = (REPO_ROOT / "deploy" / "helm" / "avaloka" / "templates" / "supabase.yaml").read_text()
    assert '.Files.Get "files/db-migrate.sh"' in template
    assert '"/migrations/migrate.sh"' in template
