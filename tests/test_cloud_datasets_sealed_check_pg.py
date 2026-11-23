"""Migration 20261116130000 and the sealer on a real Postgres: every shape the
old code wrote is sealed, the deferred VALIDATE then succeeds, plaintext is
refused from then on, and the old RPC is revoked from anon -- against the
grant the shipped schema really gives it.

Builds the original table, the original ``encrypt_cloud_credentials`` and its
``GRANT ... TO anon/authenticated`` from the CREATE/GRANT statements in
0-schema.sql (so the privilege test is held to the shipped grant, not to a
database that never had one), seeds the six legacy shapes the upgrade harness
uses (s1 AWS pair, s2 Azure pair, s3 GCP with project, s4 GCP with
access_key='', s5 already sealed, s6 rewritten by the old RPC: raw text and no
IV), applies the migration, runs the real ``seal_plaintext_rows`` through a
psql-backed client, and checks the rows, the VALIDATE, and that every sealed
column opens to the seeded value. Needs ``AVALOKA_TEST_PG_ADMIN_URL`` and psql;
skips on a laptop, FAILS when CI is set. Each check fails without its
mechanism: without the sealer fix s4 and s6 keep plaintext and VALIDATE fails;
without the REVOKE anon can still execute the RPC.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import secrets
import shutil
import subprocess
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlsplit

import pytest

from app.api import cloud_connections as cc

pytestmark = pytest.mark.integration

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MIGRATIONS = REPO_ROOT / "ui" / "supabase" / "migrations"
SCHEMA = "0-schema.sql"
SEALED = "20261116130000_cloud_datasets_sealed_keys.sql"
ADMIN_URL = os.environ.get("AVALOKA_TEST_PG_ADMIN_URL", "").strip()
U1 = "11111111-1111-1111-1111-111111111111"
RPC_SIG = "public.encrypt_cloud_credentials(uuid, text, text)"

SHIMS = """
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN CREATE ROLE anon NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN CREATE ROLE authenticated NOLOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN CREATE ROLE service_role NOLOGIN BYPASSRLS; END IF;
END $$;
CREATE SCHEMA IF NOT EXISTS auth;
CREATE OR REPLACE FUNCTION auth.jwt() RETURNS jsonb LANGUAGE sql STABLE AS
  $$ SELECT coalesce(nullif(current_setting('request.jwt.claims', true), ''), '{}')::jsonb $$;
CREATE OR REPLACE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS
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


def _schema_text() -> str:
    return (MIGRATIONS / SCHEMA).read_text(encoding="utf-8")


def _original_table_ddl() -> str:
    m = re.search(r'CREATE TABLE IF NOT EXISTS "public"\."cloud_datasets" \(.*?\n\);', _schema_text(), re.DOTALL)
    assert m, "0-schema.sql no longer defines cloud_datasets where this test looks"
    assert "check_credentials_format" in m.group(0)
    return m.group(0)


def _original_rpc_and_grants() -> str:
    """The shipped function and the grants the shipped schema gives it, verbatim."""
    text = _schema_text()
    fn = re.search(r'CREATE OR REPLACE FUNCTION "public"\."encrypt_cloud_credentials".*?\n\$\$;', text, re.DOTALL)
    assert fn, "0-schema.sql no longer defines encrypt_cloud_credentials"
    grants = re.findall(r'^GRANT ALL ON FUNCTION "public"\."encrypt_cloud_credentials"[^\n]*TO "(?:anon|authenticated|service_role)";$',
                        text, re.MULTILINE)
    assert any('"anon"' in g for g in grants), "the shipped schema grants the RPC to anon; this test exists because of it"
    return fn.group(0) + "\n" + "\n".join(grants) + "\n"


class Db:
    def __init__(self, admin_url: str, name: str):
        parts = urlsplit(admin_url)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("PG")}
        self.env.update({"PGHOST": parts.hostname or "localhost", "PGPORT": str(parts.port or 5432),
                         "PGUSER": unquote(parts.username or "postgres"), "PGPASSWORD": unquote(parts.password or "")})
        self.admin_db = (parts.path or "/postgres").lstrip("/") or "postgres"
        self.name = name

    def psql(self, sql: str, db: Optional[str] = None) -> str:
        env = dict(self.env, PGDATABASE=db or self.name)
        out = subprocess.run(["psql", "-X", "-q", "-tA", "-v", "ON_ERROR_STOP=1"], input=sql, env=env,
                             check=True, capture_output=True, text=True)
        return out.stdout.strip()

    def fails(self, sql: str) -> str:
        env = dict(self.env, PGDATABASE=self.name)
        out = subprocess.run(["psql", "-X", "-q", "-tA", "-v", "ON_ERROR_STOP=1"], input=sql, env=env,
                             capture_output=True, text=True)
        return out.stderr.strip() if out.returncode != 0 else ""

    def apply(self, filename: str) -> None:
        env = dict(self.env, PGDATABASE=self.name)
        subprocess.run(["psql", "-X", "-q", "-v", "ON_ERROR_STOP=1", "--single-transaction", "-f",
                        str(MIGRATIONS / filename)], env=env, check=True, capture_output=True, text=True)

    def rows(self) -> Dict[str, Dict[str, Any]]:
        out = self.psql("SELECT coalesce(json_agg(row_to_json(t)), '[]') FROM (SELECT id::text, name, provider, access_key, "
                        "secret_key, access_key_ciphertext, access_key_iv, secret_key_ciphertext, secret_key_iv "
                        "FROM public.cloud_datasets) t;")
        return {r["name"]: r for r in json.loads(out or "[]")}


def _q(v: Any) -> str:
    return "NULL" if v is None else "'" + str(v).replace("'", "''") + "'"


class PsqlTable:
    """Just enough of supabase-py's builder, over psql, for the real sealer."""

    def __init__(self, db: Db, name: str):
        self.db, self.name = db, name
        self._filters: List[str] = []
        self._payload: Optional[Dict[str, Any]] = None
        self._negate = False

    def select(self, *_a, **_k):
        return self

    @property
    def not_(self):
        self._negate = True
        return self

    def is_(self, col, val):
        assert val == "null"
        self._filters.append(f"{col} IS {'NOT ' if self._negate else ''}NULL")
        self._negate = False
        return self

    def eq(self, col, val):
        self._filters.append(f"{col} = {_q(val)}")
        return self

    def update(self, payload):
        self._payload = dict(payload)
        return self

    def execute(self):
        where = " AND ".join(self._filters) or "true"
        if self._payload is None:
            out = self.db.psql(f"SELECT coalesce(json_agg(row_to_json(t)), '[]') FROM (SELECT id::text, provider, access_key, "
                               f"secret_key, access_key_ciphertext, access_key_iv, secret_key_ciphertext, secret_key_iv "
                               f"FROM public.{self.name} WHERE {where}) t;")
            return type("R", (), {"data": json.loads(out or "[]")})()
        sets = ", ".join(f"{k} = {_q(v)}" for k, v in self._payload.items())
        self.db.psql(f"UPDATE public.{self.name} SET {sets} WHERE {where};")
        return type("R", (), {"data": []})()


class PsqlClient:
    def __init__(self, db: Db):
        self.db = db

    def table(self, name):
        return PsqlTable(self.db, name)


SA3 = json.dumps({"type": "service_account", "project_id": "seed-gcp-project", "private_key": "seed-pk-0003-PRIVATE"})
SA4 = json.dumps({"type": "service_account", "private_key": "seed-pk-0004-PRIVATE", "client_email": "s4@example.invalid"})
#: What each legacy row must open to once sealed: (access, secret).
EXPECTED = {
    "conn-s1": ("SEEDAWSACCESS0001", "seed-aws-secret-0001"),
    "conn-s2": ("seedazureacct0002", "seed-azure-key-0002=="),
    "conn-s3": ("seed-gcp-project", SA3),
    "conn-s4": ("service_account", SA4),      # access_key='' -> the new route's marker (the JSON has no project)
    "conn-s5": ("seed-sealed-ak-0005", "seed-sealed-sk-0005"),
    "conn-s6": ("seed-rpc-ak-0006", "seed-rpc-sk-0006"),
}
PLAINTEXT_MARKERS = [v for pair in EXPECTED.values() for v in pair if v != "service_account"] + ["seed-pk-0004-PRIVATE"]


def _seed(db: Db) -> None:
    def ins(name, provider, access, secret):
        db.psql("INSERT INTO public.cloud_datasets (user_id, name, provider, bucket_name, access_key, secret_key) "
                f"VALUES ('{U1}', '{name}', '{provider}', 'b', {_q(access)}, {_q(secret)});")
    ins("conn-s1", "aws", *EXPECTED["conn-s1"])
    ins("conn-s2", "azure", *EXPECTED["conn-s2"])
    ins("conn-s3", "gcp", "seed-gcp-project", SA3)
    ins("conn-s4", "gcp", "", SA4)                                   # shape 1: the wizard with the project id blank
    a, s = cc.encrypt_secret("seed-sealed-ak-0005"), cc.encrypt_secret("seed-sealed-sk-0005")
    db.psql("INSERT INTO public.cloud_datasets (user_id, name, provider, bucket_name, access_key_ciphertext, access_key_iv, "
            f"secret_key_ciphertext, secret_key_iv) VALUES ('{U1}', 'conn-s5', 'aws', 'b', {_q(a['ciphertext'])}, "
            f"{_q(a['iv'])}, {_q(s['ciphertext'])}, {_q(s['iv'])});")
    ins("conn-s6", "aws", "plain-before-rpc", "plain-before-rpc-secret")
    # shape 2: the old RPC, called as the signed-in owner, stores its inputs as "ciphertext" with no IV
    db.psql(f"BEGIN; SET LOCAL ROLE authenticated; SELECT set_config('request.jwt.claims', "
            f"'{{\"sub\": \"{U1}\", \"role\": \"authenticated\"}}', true) \\g /dev/null\n"
            f"SELECT public.encrypt_cloud_credentials((SELECT id FROM public.cloud_datasets WHERE name = 'conn-s6'), "
            f"'seed-rpc-ak-0006', 'seed-rpc-sk-0006') \\g /dev/null\nCOMMIT;")


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", "pg-test-key-" + secrets.token_hex(8))
    monkeypatch.setattr(cc, "_AESGCM_KEY", None)
    yield
    monkeypatch.setattr(cc, "_AESGCM_KEY", None)


@pytest.fixture
def db(key):
    _require(bool(ADMIN_URL), "AVALOKA_TEST_PG_ADMIN_URL is not set; the sealed-keys migration was not exercised")
    _require(shutil.which("psql") is not None, "psql is not installed; the sealed-keys migration cannot be exercised")
    d = Db(ADMIN_URL, f"sealed_{secrets.token_hex(6)}")
    d.psql(f'CREATE DATABASE "{d.name}"', db=d.admin_db)
    try:
        d.psql(SHIMS)
        d.psql(_original_table_ddl())
        d.psql("GRANT SELECT, INSERT, UPDATE ON public.cloud_datasets TO authenticated;")
        d.psql(_original_rpc_and_grants())
        _seed(d)
        before = d.rows()
        assert before["conn-s6"]["access_key_ciphertext"] == "seed-rpc-ak-0006" and before["conn-s6"]["access_key_iv"] is None, \
            "the old RPC must have produced the raw-ciphertext/no-IV shape"
        assert before["conn-s4"]["access_key"] == ""
        assert d.psql(f"SELECT has_function_privilege('anon', '{RPC_SIG}', 'EXECUTE');") == "t", \
            "the shipped grant: anon can execute the RPC before the migration (else the revoke test is vacuous)"
        d.apply(SEALED)
        yield d
    finally:
        d.psql(f'DROP DATABASE IF EXISTS "{d.name}"', db=d.admin_db)


def test_legacy_rows_survive_the_migration_until_sealed_and_validate_fails_while_they_do(db):
    rows = db.rows()
    assert len(rows) == 6
    assert db.psql("SELECT convalidated FROM pg_constraint WHERE conname = 'check_credentials_sealed';") == "f"
    assert db.psql("SELECT count(*) FROM pg_constraint WHERE conname = 'check_credentials_format';") == "0"
    err = db.fails("ALTER TABLE public.cloud_datasets VALIDATE CONSTRAINT check_credentials_sealed;")
    assert "check_credentials_sealed" in err, "unsealed rows remain: VALIDATE must still refuse"


def test_the_sealer_seals_every_legacy_shape_and_then_validate_succeeds(db):
    """Bite: skip access_key='' as half a pair; select only access_key IS NOT NULL."""
    client = PsqlClient(db)
    counts = cc.count_plaintext_rows(client)
    assert counts == {"plaintext": 3, "gcp_without_project": 1, "raw_ciphertext_no_iv": 1, "needs_reentering": 0}, counts
    assert cc.seal_plaintext_rows(client, apply=False) == 5
    assert db.rows()["conn-s4"]["access_key"] == "", "the dry run changed nothing"

    assert cc.seal_plaintext_rows(client, apply=True) == 5
    rows = db.rows()
    assert len(rows) == 6
    unsealed = db.psql("SELECT coalesce(string_agg(name, ','), '') FROM public.cloud_datasets WHERE NOT (access_key IS NULL "
                       "AND secret_key IS NULL AND access_key_ciphertext IS NOT NULL AND access_key_iv IS NOT NULL "
                       "AND secret_key_ciphertext IS NOT NULL AND secret_key_iv IS NOT NULL);")
    assert unsealed == "", unsealed
    for name, (access, secret) in EXPECTED.items():
        r = rows[name]
        got = (cc.decrypt_secret({"ciphertext": r["access_key_ciphertext"], "iv": r["access_key_iv"]}),
               cc.decrypt_secret({"ciphertext": r["secret_key_ciphertext"], "iv": r["secret_key_iv"]}))
        assert got == (access, secret), name
    dump = json.dumps(rows)
    assert not [m for m in PLAINTEXT_MARKERS if m in dump], "a seeded value survives in some column"
    # the deferred VALIDATE is now possible
    assert db.fails("ALTER TABLE public.cloud_datasets VALIDATE CONSTRAINT check_credentials_sealed;") == ""
    assert db.psql("SELECT convalidated FROM pg_constraint WHERE conname = 'check_credentials_sealed';") == "t"
    # re-applying the migration file (the hook is re-runnable) must not undo the VALIDATE
    db.apply(SEALED)
    assert db.psql("SELECT convalidated FROM pg_constraint WHERE conname = 'check_credentials_sealed';") == "t", \
        "a re-run dropped and re-added the constraint NOT VALID"
    # and a second run has nothing to do
    assert cc.seal_plaintext_rows(client, apply=True) == 0
    assert cc.count_plaintext_rows(client) == {"plaintext": 0, "gcp_without_project": 0, "raw_ciphertext_no_iv": 0,
                                               "needs_reentering": 0}


def test_plaintext_keys_are_refused_and_sealed_ones_accepted(db):
    """Bite: keep the plaintext branch of the CHECK."""
    plain = ("INSERT INTO public.cloud_datasets (user_id, name, provider, bucket_name, access_key, secret_key) "
             f"VALUES ('{U1}', 'again', 'aws', 'b', 'AKIAPLAIN', 'plainsecret');")
    assert "check_credentials_sealed" in db.fails(plain)
    assert "check_credentials_sealed" in db.fails("UPDATE public.cloud_datasets SET access_key = 'AKIANEW' WHERE name = 'conn-s1';")
    no_iv = ("INSERT INTO public.cloud_datasets (user_id, name, provider, bucket_name, access_key_ciphertext, "
             f"secret_key_ciphertext) VALUES ('{U1}', 'noiv', 'aws', 'b', 'Y3Q=', 'Y3Qy');")
    assert "check_credentials_sealed" in db.fails(no_iv), "ciphertext without an IV is not sealed"
    sealed = ("INSERT INTO public.cloud_datasets (user_id, name, provider, bucket_name, access_key_ciphertext, "
              f"access_key_iv, secret_key_ciphertext, secret_key_iv) VALUES ('{U1}', 'new', 'aws', 'b', 'Y3Q=', 'aXY=', 'Y3Qy', 'aXYy');")
    assert db.fails(sealed) == ""


def test_the_old_rpc_refuses_and_anon_lost_the_execute_it_was_shipped_with(db):
    """Bite: drop anon from the migration's REVOKE -- the shipped GRANT ALL ... TO anon
    then survives and this fails; the DB here carries that grant (see the fixture)."""
    assert db.psql(f"SELECT has_function_privilege('anon', '{RPC_SIG}', 'EXECUTE');") == "f"
    err = db.fails(f"SET ROLE anon; SELECT public.encrypt_cloud_credentials('{U1}', 'a', 'b');")
    assert "permission denied" in err, err
    err = db.fails(f"BEGIN; SET LOCAL ROLE authenticated; SELECT set_config('request.jwt.claims', "
                   f"'{{\"sub\": \"{U1}\", \"role\": \"authenticated\"}}', true) \\g /dev/null\n"
                   f"SELECT public.encrypt_cloud_credentials((SELECT id FROM public.cloud_datasets WHERE name = 'conn-s1'), 'a', 'b'); ROLLBACK;")
    assert "sealed by the backend" in err, err
    assert "sealed by the backend" in db.fails(f"SELECT public.encrypt_cloud_credentials('{U1}', 'a', 'b');")
