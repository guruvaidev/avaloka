"""Cloud storage keys are sealed, never plaintext (hotfix/1.6-cloud-keys-sealed).

- seal_row turns a plaintext pair into the ciphertext/iv columns and refuses
  to write unsealed when no key is configured;
- seal_plaintext_rows seals every row the old code could leave unsealed --
  a plaintext pair, a GCP row written with access_key='' (the wizard's optional
  project id left blank), the old RPC's raw text in *_ciphertext with no IV --
  leaves a row with no secret anywhere to its owner, and has a dry run; the
  CLI defaults to that dry run and prints each shape's count, never a value;
- POST /api/cloud-connections seals in process and writes no plaintext column;
  without a key it is 503, not a plaintext insert;
- every reader works on a sealed-only row once get_cloud_connection has opened
  it in memory, and refuses the row unopened: get_cloud_connection itself, the
  Ray job submitter, the Kubernetes secret builder, the planner;
- the migration (both trees, identical to the 1.7 line's copy on
  feat/2.x-object-stores) drops the plaintext
  branch, is NOT VALID, retires the anon-callable RPC, and is in
  MERGED_MIGRATIONS; the server wires the route and the start-up sealer; the
  UI no longer inserts keys itself.

Bites: let seal_row pass plaintext through; keep the old CHECK; insert the
body as it arrived; print a value from the CLI; skip decryption in
get_cloud_connection.
"""

from __future__ import annotations

import io
import json
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import cloud_connections as cc

ROOT = Path(__file__).resolve().parents[1]
NAME = "20261116130000_cloud_datasets_sealed_keys.sql"
SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCY" + "EXAMPLEKEY"
ACCESS = "AKIAEXAMPLE" + "FAKE"
SA_JSON = json.dumps({"type": "service_account", "project_id": "proj", "private_key": "PK-" + "material"})


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", "unit-test-key-" + "x" * 16)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None)
    yield
    monkeypatch.setattr(cc, "_AESGCM_KEY", None)


@pytest.fixture
def unkeyed(monkeypatch):
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", None)
    monkeypatch.setattr(cc, "_AESGCM_KEY", None)
    yield
    monkeypatch.setattr(cc, "_AESGCM_KEY", None)


class FakeTable:
    """Just enough of supabase-py's query builder for the sealer and the route."""

    def __init__(self, store: "FakeClient", name: str):
        self.store, self.name = store, name
        self._op: Any = None
        self._filters: List[tuple] = []
        self._payload: Any = None
        self._negate = False
        self._single = False

    def select(self, *_a, **_k):
        self._op = "select"
        return self

    @property
    def not_(self):
        self._negate = True
        return self

    def is_(self, col, val):
        self._filters.append(("is_not" if self._negate else "is", col, val))
        self._negate = False
        return self

    def eq(self, col, val):
        self._filters.append(("eq", col, val))
        return self

    def single(self):
        self._single = True
        return self

    def insert(self, payload):
        self._op, self._payload = "insert", dict(payload)
        return self

    def update(self, payload):
        self._op, self._payload = "update", dict(payload)
        return self

    def _match(self, row):
        for kind, col, val in self._filters:
            if kind == "eq" and row.get(col) != val:
                return False
            if kind == "is" and val == "null" and row.get(col) is not None:
                return False
            if kind == "is_not" and val == "null" and row.get(col) is None:
                return False
        return True

    def execute(self):
        rows = self.store.rows.setdefault(self.name, [])
        if self._op == "select":
            found = [dict(r) for r in rows if self._match(r)]
            return types.SimpleNamespace(data=(found[0] if found else None) if self._single else found)
        if self._op == "insert":
            row = {"id": f"row-{len(rows) + 1}", **self._payload}
            self.store.check(row)
            rows.append(row)
            return types.SimpleNamespace(data=[dict(row)])
        if self._op == "update":
            out = []
            for r in rows:
                if self._match(r):
                    r.update(self._payload)
                    self.store.check(r)
                    out.append(dict(r))
            return types.SimpleNamespace(data=out)
        raise AssertionError(self._op)


class FakeClient:
    """Rows plus the migration's CHECK, so an unsealed write fails here as it
    would in Postgres."""

    def __init__(self, rows=None, enforce=True):
        self.rows: Dict[str, List[Dict[str, Any]]] = {"cloud_datasets": list(rows or [])}
        self.enforce = enforce

    def table(self, name):
        return FakeTable(self, name)

    def check(self, row):
        if not self.enforce:
            return
        ok = (row.get("access_key") is None and row.get("secret_key") is None
              and row.get("access_key_ciphertext") and row.get("access_key_iv")
              and row.get("secret_key_ciphertext") and row.get("secret_key_iv"))
        if not ok:
            raise RuntimeError('new row for relation "cloud_datasets" violates check constraint "check_credentials_sealed"')


def _sealed_row(provider="aws", access=ACCESS, secret=SECRET, **extra) -> Dict[str, Any]:
    a, s = cc.encrypt_secret(access), cc.encrypt_secret(secret)
    row = {"id": "conn-1", "user_id": "u", "name": "prod", "provider": provider, "region": "eu-west-1",
           "bucket_name": "acme-raw", "access_key": None, "secret_key": None,
           "access_key_ciphertext": a["ciphertext"], "access_key_iv": a["iv"],
           "secret_key_ciphertext": s["ciphertext"], "secret_key_iv": s["iv"]}
    row.update(extra)
    return row


# --------------------------------------------------------------------------- #
# seal_row, the sealer, the CLI
# --------------------------------------------------------------------------- #

def test_seal_row_writes_the_sealed_columns_and_nothing_plaintext(keyed):
    row = cc.seal_row({"user_id": "u", "provider": "aws", "access_key": ACCESS, "secret_key": SECRET})
    assert row["access_key"] is None and row["secret_key"] is None
    assert row["access_key_ciphertext"] and row["access_key_iv"] and row["secret_key_ciphertext"] and row["secret_key_iv"]
    text = json.dumps(row)
    assert ACCESS not in text and SECRET not in text
    assert cc.decrypt_secret({"ciphertext": row["secret_key_ciphertext"], "iv": row["secret_key_iv"]}) == SECRET
    already = {"access_key_ciphertext": "c", "access_key_iv": "i", "secret_key_ciphertext": "c2", "secret_key_iv": "i2"}
    assert cc.seal_row(dict(already)) == already
    with pytest.raises(ValueError, match="both halves"):
        cc.seal_row({"access_key": ACCESS, "secret_key": ""})


def test_seal_row_refuses_to_write_unsealed_without_a_key(unkeyed):
    """Bite: fall back to plaintext when the key is missing."""
    with pytest.raises(RuntimeError, match="DB_ENCRYPTION_KEY"):
        cc.seal_row({"access_key": ACCESS, "secret_key": SECRET})


def _opens(row, half):
    return cc.decrypt_secret({"ciphertext": row[f"{half}_ciphertext"], "iv": row[f"{half}_iv"]})


def _is_sealed(row):
    return (row.get("access_key") is None and row.get("secret_key") is None and row.get("access_key_ciphertext")
            and row.get("access_key_iv") and row.get("secret_key_ciphertext") and row.get("secret_key_iv"))


SA_WITH_PROJECT = json.dumps({"type": "service_account", "project_id": "proj-3", "private_key": "PK-3"})
SA_NO_PROJECT = json.dumps({"type": "service_account", "private_key": "PK-4", "client_email": "s4@example.invalid"})

#: The shapes the old code wrote, as the lead's upgrade harness seeds them (s1-s6),
#: plus a row with no secret anywhere.
LEGACY_ROWS = [
    {"id": "s1", "provider": "aws", "access_key": "AKIA1", "secret_key": SECRET},
    {"id": "s2", "provider": "azure", "access_key": "acct2", "secret_key": "sas-2=="},
    {"id": "s3", "provider": "gcp", "access_key": "proj-3", "secret_key": SA_WITH_PROJECT},
    {"id": "s4", "provider": "gcp", "access_key": "", "secret_key": SA_NO_PROJECT},            # shape 1
    {"id": "s6", "provider": "aws", "access_key": None, "secret_key": None,                     # shape 2: the old RPC
     "access_key_ciphertext": "rpc-ak-6", "access_key_iv": None, "secret_key_ciphertext": "rpc-sk-6", "secret_key_iv": None},
    {"id": "s7", "provider": "aws", "access_key": "AKIA7", "secret_key": None},               # no secret anywhere
]


def test_every_legacy_shape_is_classified_and_the_sealable_ones_are_sealed(keyed):
    """Bite: treat access_key='' as half a pair; select only access_key IS NOT NULL."""
    s5 = _sealed_row() | {"id": "s5"}
    client = FakeClient(rows=[dict(r) for r in LEGACY_ROWS] + [dict(s5)])
    assert cc.count_plaintext_rows(client) == {"plaintext": 3, "gcp_without_project": 1, "raw_ciphertext_no_iv": 1,
                                               "needs_reentering": 1}
    assert cc.seal_plaintext_rows(client, apply=False) == 5, "a dry run counts the sealable shapes"
    assert client.rows["cloud_datasets"][0]["access_key"] == "AKIA1", "and changes nothing"
    assert cc.seal_plaintext_rows(client) == 5
    rows = {r["id"]: r for r in client.rows["cloud_datasets"]}
    for rid in ("s1", "s2", "s3", "s4", "s5", "s6"):
        assert _is_sealed(rows[rid]), rid
    assert (_opens(rows["s1"], "access_key"), _opens(rows["s1"], "secret_key")) == ("AKIA1", SECRET)
    assert (_opens(rows["s2"], "access_key"), _opens(rows["s2"], "secret_key")) == ("acct2", "sas-2==")
    assert (_opens(rows["s3"], "access_key"), _opens(rows["s3"], "secret_key")) == ("proj-3", SA_WITH_PROJECT)
    # shape 1: the service account is the secret; the access half is the project id when the
    # JSON has one, else the marker the new route writes -- never '' and never left plaintext
    assert _opens(rows["s4"], "secret_key") == SA_NO_PROJECT and _opens(rows["s4"], "access_key") == "service_account"
    # shape 2: the raw text the old RPC stored is what gets sealed
    assert (_opens(rows["s6"], "access_key"), _opens(rows["s6"], "secret_key")) == ("rpc-ak-6", "rpc-sk-6")
    for col in ("access_key_ciphertext", "access_key_iv", "secret_key_ciphertext", "secret_key_iv"):
        assert rows["s5"][col] == s5[col], f"an already-sealed row is not touched ({col})"
    assert rows["s7"]["access_key"] == "AKIA7" and rows["s7"]["secret_key"] is None, "no secret anywhere: left for the owner"
    assert cc.seal_plaintext_rows(client) == 0, "idempotent"
    assert cc.count_plaintext_rows(client) == {"plaintext": 0, "gcp_without_project": 0, "raw_ciphertext_no_iv": 0,
                                               "needs_reentering": 1}
    assert "PK-4" not in json.dumps(client.rows) and SECRET not in json.dumps(client.rows)


def test_the_gcp_access_half_is_the_project_when_the_json_has_one():
    assert cc.gcp_access_for(SA_WITH_PROJECT) == "proj-3"
    assert cc.gcp_access_for(SA_NO_PROJECT) == "service_account"
    assert cc.gcp_access_for("not json") == "service_account"
    assert cc.classify_row({"provider": "gcp", "access_key": "", "secret_key": SA_NO_PROJECT}) == "gcp_without_project"
    assert cc.classify_row({"provider": "aws", "access_key": "", "secret_key": SECRET}) == "needs_reentering", \
        "an S3 row without an access key really is missing half"
    assert cc.classify_row({"provider": "gcp", "access_key": None, "secret_key": None}) == "needs_reentering"


def test_a_half_sealed_row_keeps_its_sealed_half_and_seals_the_other(keyed):
    sealed_secret = cc.encrypt_secret(SECRET)
    client = FakeClient(rows=[{"id": "m", "provider": "aws", "access_key": "AKIA-m", "secret_key": None,
                               "secret_key_ciphertext": sealed_secret["ciphertext"], "secret_key_iv": sealed_secret["iv"]}])
    assert cc.classify_row(client.rows["cloud_datasets"][0]) == "plaintext"
    assert cc.seal_plaintext_rows(client) == 1
    (row,) = client.rows["cloud_datasets"]
    assert _is_sealed(row) and row["secret_key_ciphertext"] == sealed_secret["ciphertext"], "the sealed half is untouched"
    assert _opens(row, "access_key") == "AKIA-m"


def test_the_sealer_refuses_without_a_key(unkeyed):
    with pytest.raises(RuntimeError, match="DB_ENCRYPTION_KEY"):
        cc.seal_plaintext_rows(FakeClient(rows=[{"id": "a", "access_key": "x", "secret_key": "y"}]))
    assert cc.seal_at_startup() is None


def test_the_cli_defaults_to_a_dry_run_and_prints_counts_only(keyed):
    """Bite: print a row or a key; seal without --apply."""
    client = FakeClient(rows=[{"id": "a", "provider": "aws", "access_key": ACCESS, "secret_key": SECRET},
                              {"id": "g", "provider": "gcp", "access_key": "", "secret_key": SA_NO_PROJECT},
                              {"id": "r", "provider": "aws", "access_key_ciphertext": "raw-ak", "secret_key_ciphertext": "raw-sk"},
                              {"id": "c", "provider": "aws", "access_key": "AKIA3", "secret_key": None}])
    out = io.StringIO()
    assert cc.seal_cli(["seal"], client_factory=lambda: client, out=out) == 0
    text = out.getvalue()
    assert "dry run: 3 row(s) would be sealed" in text
    assert "1 with a plaintext key pair" in text and "1 GCP without a project id" in text \
        and "1 with raw text in the ciphertext columns" in text, "each shape is counted on its own"
    assert "1 row(s) have no secret in any column and need re-entering" in text
    for value in (ACCESS, SECRET, "AKIA3", "PK-4", "raw-ak", "raw-sk", "row-", "conn-"):
        assert value not in text
    assert client.rows["cloud_datasets"][0]["access_key"] == ACCESS, "the dry run sealed nothing"
    out = io.StringIO()
    assert cc.seal_cli(["seal", "--apply"], client_factory=lambda: client, out=out) == 0
    assert "sealed 3 row(s)" in out.getvalue() and ACCESS not in out.getvalue() and SECRET not in out.getvalue()
    assert client.rows["cloud_datasets"][0]["access_key"] is None
    assert _is_sealed(client.rows["cloud_datasets"][1]) and _is_sealed(client.rows["cloud_datasets"][2])
    assert client.rows["cloud_datasets"][3]["access_key"] == "AKIA3", "no secret anywhere: untouched, and said so"
    assert "would fail" in out.getvalue() and "VALIDATE CONSTRAINT" not in out.getvalue(), \
        "a row still unsealed: the operator is told not to validate yet"
    client.rows["cloud_datasets"][3]["secret_key"] = SECRET          # the owner re-entered it
    out = io.StringIO()
    assert cc.seal_cli(["seal", "--apply"], client_factory=lambda: client, out=out) == 0
    assert "ALTER TABLE public.cloud_datasets VALIDATE CONSTRAINT check_credentials_sealed;" in out.getvalue(), \
        "every row sealed: the deferred VALIDATE is the printed next step"
    assert "SELECT convalidated FROM pg_constraint" in out.getvalue() and "left check_credentials_sealed NOT VALID" not in out.getvalue(), \
        "the sentence must still be true once the operator has already validated: check first"
    out = io.StringIO()
    assert cc.seal_cli(["bogus"], client_factory=lambda: client, out=out) == 2 and "usage" in out.getvalue()


def test_the_cli_without_a_key_says_so_and_touches_nothing(unkeyed):
    client = FakeClient(rows=[{"id": "a", "access_key": ACCESS, "secret_key": SECRET}], enforce=False)
    out = io.StringIO()
    assert cc.seal_cli(["seal", "--apply"], client_factory=lambda: client, out=out) == 1
    assert "DB_ENCRYPTION_KEY" in out.getvalue() and SECRET not in out.getvalue()
    assert client.rows["cloud_datasets"][0]["access_key"] == ACCESS


# --------------------------------------------------------------------------- #
# The readers, on a sealed-only row
# --------------------------------------------------------------------------- #

def _install_fake_supabase(monkeypatch, row):
    client = FakeClient(rows=[row])
    fake_mod = types.ModuleType("supabase")
    fake_mod.create_client = lambda url, key: client
    monkeypatch.setitem(sys.modules, "supabase", fake_mod)
    monkeypatch.setattr(cc, "SUPABASE_URL", "https://example.supabase.co", raising=False)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "role-key")
    return client


@pytest.mark.asyncio
async def test_get_cloud_connection_opens_a_sealed_only_row_in_memory(keyed, monkeypatch, caplog):
    """Bite: skip _maybe_decrypt_into; or write the opened keys back."""
    client = _install_fake_supabase(monkeypatch, _sealed_row())
    import logging
    with caplog.at_level(logging.INFO, logger="avaloka"):
        conn = await cc.get_cloud_connection("conn-1")
    assert conn["access_key"] == ACCESS and conn["secret_key"] == SECRET and conn["provider"] in ("aws", "s3")
    stored = client.rows["cloud_datasets"][0]
    assert stored["access_key"] is None and stored["secret_key"] is None, "opened in memory, not written back"
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert SECRET not in logged and ACCESS not in logged


@pytest.mark.asyncio
async def test_get_cloud_connection_refuses_a_sealed_row_it_cannot_open(unkeyed, monkeypatch):
    from fastapi import HTTPException
    _install_fake_supabase(monkeypatch, _sealed_row() if False else {
        "id": "conn-1", "provider": "aws", "bucket_name": "b", "access_key": None, "secret_key": None,
        "access_key_ciphertext": "Y3Q=", "access_key_iv": "aXY=", "secret_key_ciphertext": "Y3Qy", "secret_key_iv": "aXYy"})
    with pytest.raises(HTTPException) as exc:
        await cc.get_cloud_connection("conn-1")
    assert exc.value.status_code == 500 and "access_key/secret_key" in str(exc.value.detail)


def _opened(**kw) -> Dict[str, Any]:
    """What get_cloud_connection hands a reader: the sealed row with the keys
    opened in memory (and the provider normalised)."""
    row = _sealed_row(**kw)
    for field in ("access_key", "secret_key"):
        cc._maybe_decrypt_into(row, field)
    row["provider"] = "s3" if row["provider"] in ("aws", "s3") else row["provider"]
    return row


def test_the_ray_job_submitter_builds_s3_creds_from_an_opened_row_and_refuses_an_unopened_one(keyed):
    from app.infra import rayjob_submitter as rj
    creds = rj._build_creds("s3", _opened())
    assert creds["access_key"] == ACCESS and creds["secret_key"] == SECRET and creds["region"] == "eu-west-1"
    assert creds["bucket_name"] == "acme-raw"
    with pytest.raises(ValueError, match="access_key/secret_key"):
        rj._build_creds("s3", _sealed_row())   # sealed, not opened: the submitter cannot use it


def test_the_kubernetes_secret_builder_reads_an_opened_row_and_refuses_an_unopened_one(keyed):
    from app.infra import k8s_secrets as ks
    env = ks.cloud_secret_env("s3", _opened())
    assert env["AWS_ACCESS_KEY_ID"] == ACCESS and env["AWS_SECRET_ACCESS_KEY"] == SECRET
    assert env["AWS_DEFAULT_REGION"] == "eu-west-1"
    with pytest.raises(ValueError, match="Missing AWS credentials"):
        ks.cloud_secret_env("s3", _sealed_row())


def test_the_planner_keeps_the_handle_and_opens_keys_only_for_the_transfer(keyed):
    planner = pytest.importorskip("app.agents.planner")
    entry = planner._cloud_conn_to_entry(_sealed_row())
    assert entry["db_type"] == "aws_s3" and entry["conn_id"] == "conn-1" and entry["bucket_name"] == "acme-raw"
    assert "access_key" not in entry and "secret_key" not in entry, "the registry keeps the handle, not the keys"
    assert ACCESS not in json.dumps(entry) and SECRET not in json.dumps(entry)
    secrets = planner._cloud_conn_secrets(_opened(), "aws_s3")
    assert secrets == {"access_key": ACCESS, "secret_key": SECRET}
    assert planner._cloud_conn_secrets(_sealed_row(), "aws_s3") == {}, "unopened: no keys, not plaintext columns"
    gcs = planner._cloud_conn_secrets(_opened(provider="gcp", access="proj", secret=SA_JSON) | {"provider": "gcp"}, "gcs")
    assert json.loads(gcs["gcp_service_account_json"])["project_id"] == "proj"


# --------------------------------------------------------------------------- #
# POST /api/cloud-connections
# --------------------------------------------------------------------------- #

def _client(monkeypatch, fake: FakeClient):
    import app.agents.sampling_persistence as sp
    monkeypatch.setattr(sp, "get_supabase_client", lambda: fake)
    app = FastAPI()
    users = {"tok-a": "user-a"}
    app.include_router(cc.build_router(lambda request: users.get(request.headers.get("authorization", ""))))
    return TestClient(app)


BODY = {"name": "Prod bucket", "provider": "aws", "bucket_name": "acme-raw", "region": "us-east-1",
        "access_key": ACCESS, "secret_key": SECRET}


def test_the_route_seals_before_the_insert_and_stores_no_plaintext(keyed, monkeypatch):
    fake = FakeClient()
    client = _client(monkeypatch, fake)
    assert client.post("/api/cloud-connections", json=BODY).status_code == 401
    r = client.post("/api/cloud-connections", json=BODY, headers={"authorization": "tok-a"})
    assert r.status_code == 201, r.text
    assert r.json()["id"] == "row-1" and r.json()["provider"] == "aws" and r.json()["bucket_name"] == "acme-raw"
    (row,) = fake.rows["cloud_datasets"]
    assert row["user_id"] == "user-a" and "access_key" not in row and "secret_key" not in row
    assert cc.decrypt_secret({"ciphertext": row["secret_key_ciphertext"], "iv": row["secret_key_iv"]}) == SECRET
    assert SECRET not in json.dumps(row) and ACCESS not in json.dumps(row)


def test_the_route_is_503_not_a_plaintext_insert_without_a_key(unkeyed, monkeypatch):
    """Bite: insert the body as it arrived when encryption is unavailable."""
    fake = FakeClient(enforce=False)
    client = _client(monkeypatch, fake)
    r = client.post("/api/cloud-connections", json=BODY, headers={"authorization": "tok-a"})
    assert r.status_code == 503 and "DB_ENCRYPTION_KEY" in r.json()["detail"]
    assert fake.rows["cloud_datasets"] == []


def test_the_route_validates_providers_and_gcp_keys(keyed, monkeypatch):
    fake = FakeClient()
    client = _client(monkeypatch, fake)
    h = {"authorization": "tok-a"}
    assert client.post("/api/cloud-connections", json={**BODY, "provider": "ftp"}, headers=h).status_code == 400
    assert client.post("/api/cloud-connections", json={**BODY, "access_key": ""}, headers=h).status_code == 400
    gcp = {"name": "GCP", "provider": "gcp", "bucket_name": "lake", "project_id": "proj", "secret_key": "not json"}
    assert client.post("/api/cloud-connections", json=gcp, headers=h).status_code == 400
    gcp["secret_key"] = SA_JSON
    r = client.post("/api/cloud-connections", json=gcp, headers=h)
    assert r.status_code == 201 and r.json()["provider"] == "gcp"
    row = fake.rows["cloud_datasets"][-1]
    assert cc.decrypt_secret({"ciphertext": row["access_key_ciphertext"], "iv": row["access_key_iv"]}) == "proj"
    sa = json.loads(cc.decrypt_secret({"ciphertext": row["secret_key_ciphertext"], "iv": row["secret_key_iv"]}))
    assert sa["project_id"] == "proj" and "PK-" in sa["private_key"]
    assert "PK-" not in json.dumps(row)


# --------------------------------------------------------------------------- #
# Wiring, UI and the migration text (the Postgres run is test_cloud_datasets_sealed_check_pg.py)
# --------------------------------------------------------------------------- #

def test_the_server_wires_the_route_and_seals_at_startup():
    text = (ROOT / "app" / "api" / "server.py").read_text(encoding="utf-8")
    assert "app.include_router(_cloud_connections.build_router(_resolve_user_id))" in text
    lifespan = text[text.index("async def server_lifespan"):text.index("app = FastAPI(")]
    assert "seal_at_startup" in lifespan, "existing plaintext rows are sealed when the API starts"
    assert lifespan.index("require_db_encryption_key") < lifespan.index("seal_at_startup")


def test_the_ui_no_longer_writes_keys_to_the_table_itself():
    """Both UI writers post to the backend. Bite: restore the direct insert."""
    for rel in ("ui/src/lib/api/backendApi.ts", "ui/src/lib/gcp-connection.ts"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert not re.search(r'\.from\("cloud_datasets"\)\s*\.insert\(', text), f"{rel} still inserts into cloud_datasets"
        for block in re.findall(r'\.from\("cloud_datasets"\)(.*?)\.(?:select|eq)\(', text, re.DOTALL):
            assert "secret_key" not in block and "access_key" not in block, rel
    api = (ROOT / "ui/src/lib/api/backendApi.ts").read_text(encoding="utf-8")
    assert 'proxyFetch(`/api/cloud-connections`' in api


def test_the_migration_is_byte_identical_in_both_trees_and_drops_the_plaintext_branch():
    ui = ROOT / "ui" / "supabase" / "migrations" / NAME
    chart = ROOT / "deploy" / "helm" / "avaloka" / "files" / "supabase-migrations" / NAME
    assert ui.is_file() and chart.is_file() and ui.read_bytes() == chart.read_bytes()
    sql = ui.read_text(encoding="utf-8")
    assert "DROP CONSTRAINT IF EXISTS check_credentials_format" in sql
    m = re.search(r"ADD CONSTRAINT check_credentials_sealed CHECK \((.*?)\) NOT VALID;", sql, re.DOTALL)
    assert m, "the sealed CHECK must be NOT VALID so rows written before it survive until sealed"
    assert "DROP CONSTRAINT IF EXISTS check_credentials_sealed" not in sql, \
        "a re-run must not drop and re-add the constraint: that resets a validated one to NOT VALID"
    assert re.search(r"IF NOT EXISTS \(\s*SELECT 1 FROM pg_constraint\s+WHERE conname = 'check_credentials_sealed'", sql), \
        "the ADD is guarded so a re-run after VALIDATE leaves convalidated = t"
    check = m.group(1)
    assert "access_key IS NULL AND secret_key IS NULL" in check
    for col in ("access_key_ciphertext", "access_key_iv", "secret_key_ciphertext", "secret_key_iv"):
        assert f"{col} IS NOT NULL" in check
    assert "access_key IS NOT NULL" not in check and " OR " not in check, "no plaintext branch"
    assert "RAISE EXCEPTION" in sql.split("encrypt_cloud_credentials", 1)[1]
    assert re.search(r"REVOKE ALL ON FUNCTION public\.encrypt_cloud_credentials.*FROM PUBLIC, anon", sql)
    from tests.test_migration_versions_unique import MERGED_MIGRATIONS
    # Listed in merge order, and nothing merged after it sorts before it. (It was
    # pinned as the newest entry, which every later 1.6 migration made untrue.)
    assert NAME in MERGED_MIGRATIONS
    later = MERGED_MIGRATIONS[MERGED_MIGRATIONS.index(NAME) + 1:]
    assert all(n > NAME for n in later), f"merged after {NAME} but sorts before it: {[n for n in later if n <= NAME]}"
