"""Graph state never holds a plaintext database or cloud credential.

A synthetic credential, ZZ-SYNTH-PW-1, is planted where each planner path
picks one up -- the sealed-handle fetch, the auto-fetch of a connection the
user named, the implicit (analysed) source, a UI cloud connection by id and by
name, and a connection typed into chat -- and every test asserts the same two
things: the registry holds only handles and display fields, and the value is
nowhere in state. The transfers still run: the credentials are opened by the
data transfer agent after the user confirms, and reach the copy / the runner.

Before this change each of these paths copied the secret into
``state["dta_database_registry"]`` (planner.py at d199d120: :3799 via :3835,
:3715-3718, :3253), where every later checkpoint, trace and debug dump of the
thread carried it.
"""
from __future__ import annotations

import ast
import hashlib
import re
import secrets
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.planner as planner
from app.agents.planner import InitiateTransferParams, RegisterDatabaseParams

try:  # tests/ as a package, or tests/ on sys.path
    from tests.dta_confirm import run_confirmed
    from tests.onboarding_identity import assertion_user
except ImportError:  # pragma: no cover
    from dta_confirm import run_confirmed
    from onboarding_identity import assertion_user

SYN = "ZZ-SYNTH-PW-1"
REPO = Path(__file__).resolve().parents[1]
# Field names that may never appear in a registry entry.
SECRET_HINTS = ("password", "passwd", "pwd", "secret", "key", "token", "conn_str",
                "connection_string", "credential", "sas", "private")
CLOUD_ID = "6f1c2b9e-0d1a-4e5b-9c3d-2a7e8f901234"


@pytest.fixture
def sealing_key(monkeypatch):
    """A throwaway deployment key made here."""
    import app.api.cloud_connections as cc
    key = secrets.token_hex(16)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(key.encode()).digest(), raising=False)


def assert_no_credentials(state: dict) -> None:
    registry = state.get("dta_database_registry") or {}
    for alias, entry in registry.items():
        bad = [k for k in entry if any(h in k.lower() for h in SECRET_HINTS)]
        assert not bad, f"registry entry {alias!r} carries {bad}"
    assert SYN not in repr(registry), "the synthetic credential is in the registry"
    assert SYN not in repr(state), "the synthetic credential is in graph state"


# --------------------------------------------------------------------------- #
# Databases: handle fetch, named source auto-fetched, implicit source
# --------------------------------------------------------------------------- #

def _sqlite(path: Path, rows: int = 0) -> Path:
    con = sqlite3.connect(path)
    if rows:
        con.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, age REAL)")
        con.executemany("INSERT INTO patients VALUES (?, ?)", [(i, 20 + i) for i in range(rows)])
        con.commit()
    con.close()
    return path


def _count(path: Path, table: str):
    con = sqlite3.connect(path)
    try:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone():
            return None
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


@pytest.fixture
def databases(tmp_path, monkeypatch, sealing_key):
    """Two connected databases whose passwords exist only in their sealed handles."""
    from app.api.cloud_connections import encrypt_secret
    from app.mcp_server.credentials import credential_handle

    files = {"clinic_db": _sqlite(tmp_path / "clinic.db", rows=37),
             "warehouse_db": _sqlite(tmp_path / "warehouse.db")}
    opened = []  # every password the copy was handed

    import app.agents.data_transfer_agent.data_transfer_agent as dta

    def build(creds, db_type):
        opened.append(creds.get("password"))
        return f"sqlite:///{files[creds['database']]}", {}

    def tables(creds, *_a, **_k):
        return sorted(r[0] for r in sqlite3.connect(files[creds["database"]]).execute(
            "SELECT name FROM sqlite_master WHERE type='table'"))

    monkeypatch.setattr(dta, "build_connection_string", build)
    monkeypatch.setattr(planner, "_list_dest_tables", tables)
    monkeypatch.setattr(planner, "_list_source_tables", tables, raising=False)
    monkeypatch.delenv("RAY_DASHBOARD_URL", raising=False)
    monkeypatch.delenv("RAY_DASHBOARD_URL_INCLUSTER", raising=False)

    keys = {"clinic": "ak_clinic", "warehouse": "ak_warehouse"}
    route = {
        cid: {"host": "db", "port": 5432, "database": f"{cid}_db", "username": "u", "table": "patients",
              "credential_handle": credential_handle(f"postgresql://u:{SYN}@db:5432/{cid}_db")}
        for cid in keys
    }

    def get(url, headers=None, **_kw):
        # As the real onboarding API answers since the 1.7 merge (decision 1):
        # /credentials to the owner's signed assertion only, never an api_key.
        resp = MagicMock()
        cid = url.split("/customers/")[-1].split("/")[0] if "/customers/" in url else None
        user = assertion_user(headers)
        if cid in route and url.endswith("/credentials") and user == "user-1":
            resp.status_code, payload = 200, route[cid]
        elif url.endswith("/customers") and user == "user-1":
            resp.status_code, payload = 200, [{"customer_id": c, "database_type": "postgresql"} for c in route]
        else:
            resp.status_code, payload = 404, {}
        resp.json.return_value = payload
        return resp

    session = patch("app.agents.planner.requests.Session")
    m = session.start()
    m.return_value.__enter__.return_value.get.side_effect = get
    state = {
        "user_id": "user-1", "messages": [], "dta_database_registry": {},
        "db_credential_handles": {cid: encrypt_secret(k) for cid, k in keys.items()},
    }
    yield state, files, opened
    session.stop()


def _copy(state, **kw):
    params = InitiateTransferParams(
        destination_alias="warehouse", user_prompt="Transfer patients to warehouse as patients_copy",
        dest_table="patients_copy", create_if_missing=True, **kw)
    return run_confirmed(state, params)


def test_implicit_source_and_handle_fetch_keep_the_password_out_of_state(databases):
    state, files, opened = databases
    state["active_db_customer_id"] = "clinic"          # the analysed database (implicit source)
    state["active_db_source_table"] = "patients"
    result = _copy(state)
    assert _count(files["warehouse_db"], "patients_copy") == 37, result
    assert opened and set(opened) == {SYN}, "the copy must get the password the handle seals"
    assert_no_credentials(state)
    assert state["dta_database_registry"]["clinic"]["conn_id"] == "clinic"
    assert SYN not in result


def test_named_source_auto_fetched_from_the_ui_keeps_the_password_out_of_state(databases):
    state, files, opened = databases
    result = _copy(state, source_alias="clinic", source_table="patients")
    assert _count(files["warehouse_db"], "patients_copy") == 37, result
    assert set(opened) == {SYN}
    assert_no_credentials(state)


def test_a_legacy_registry_entry_with_a_password_is_scrubbed_and_never_used(databases):
    """A checkpoint written before this rule: the password is dropped on read, not run on."""
    state, files, opened = databases
    state["dta_database_registry"] = {"legacy": {
        "db_type": "postgresql", "host": "db", "port": 5432, "database": "clinic_db",
        "user": "u", "password": SYN, "table": "patients"}}
    result = _copy(state, source_alias="legacy", source_table="patients")
    assert "can't open the credentials" in result, result
    assert _count(files["warehouse_db"], "patients_copy") is None
    assert opened == []
    assert_no_credentials(state)


# --------------------------------------------------------------------------- #
# Cloud connections: by id and by name
# --------------------------------------------------------------------------- #

@pytest.fixture
def cloud(monkeypatch, sealing_key):
    import app.api.cloud_connections as cc
    rows = {
        CLOUD_ID: {"id": CLOUD_ID, "name": "SrcS3", "provider": "aws", "bucket_name": "src-bucket",
                   "region": "us-east-1", "access_key": "AKIA-SYNTH", "secret_key": SYN},
        "9a1b2c3d-0000-4000-8000-000000000001": {
            "id": "9a1b2c3d-0000-4000-8000-000000000001", "name": "DstS3", "provider": "aws",
            "bucket_name": "dst-bucket", "region": "us-east-1",
            "access_key": "AKIA-SYNTH-2", "secret_key": SYN + "-dest"},
    }

    async def get_cloud_connection(cid):
        return dict(rows[cid])

    async def list_cloud_connections(user_id=None):
        return [{k: v for k, v in r.items() if k not in ("access_key", "secret_key")} for r in rows.values()]

    monkeypatch.setattr(cc, "get_cloud_connection", get_cloud_connection)
    monkeypatch.setattr(cc, "list_cloud_connections", list_cloud_connections)
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://127.0.0.1:8265")
    monkeypatch.setenv("EXECUTION_ENV", "gke")
    launched = {}

    def gke(injection_script, job_id=None, cloud_env=None, gcp_sa_json=None,
            dest_gcp_sa_json=None, dashboard_url=None):
        launched.update(cloud_env=dict(cloud_env or {}))
        from app.data_transfer_docker.run_result import TransferResult
        return TransferResult(success=True)

    pipeline = patch("app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline",
                     return_value=({"error_message": None, "warnings": None}, "print('x')"))
    runner = patch("app.data_transfer_docker.gke_run.launch_gke_pipeline", side_effect=gke)
    pipeline.start(), runner.start()
    yield {"user_id": "user-1", "messages": [], "dta_database_registry": {}}, launched
    pipeline.stop(), runner.stop()


def test_cloud_connections_by_id_and_by_name_keep_the_keys_out_of_state(cloud):
    state, launched = cloud
    params = InitiateTransferParams(
        source_alias=CLOUD_ID, destination_alias="DstS3", user_prompt="copy it",
        source_object="in/data.csv", dest_object="out/data.csv")
    result = run_confirmed(state, params)
    assert "SUCCESS" in result or "complete" in result.lower(), result
    # Opened after the confirmation, for the runner only.
    assert launched["cloud_env"].get("AWS_SECRET_ACCESS_KEY") == SYN
    assert launched["cloud_env"].get("DEST_AWS_SECRET_ACCESS_KEY") == SYN + "-dest"
    assert_no_credentials(state)
    assert SYN + "-dest" not in repr(state)


def test_cloud_entry_from_a_ui_row_is_a_handle():
    entry = planner._cloud_conn_to_entry({"id": CLOUD_ID, "provider": "gcp", "bucket_name": "b",
                                          "secret_key": '{"private_key": "%s"}' % SYN})
    assert SYN not in repr(entry) and "gcp_service_account_json" not in entry
    assert entry.get("conn_id") == CLOUD_ID


# --------------------------------------------------------------------------- #
# Chat registration
# --------------------------------------------------------------------------- #

@pytest.fixture
def sealed_store(monkeypatch, sealing_key):
    """The onboarding API's register-and-save, recorded: the one place the password goes."""
    import app.api.db_connections as dbc
    from app.api.cloud_connections import encrypt_secret
    seen = []

    async def register_and_save(user_id, body):
        seen.append(dict(body))
        return {"id": "row-1", "customer_id": body["customer_id"]}, encrypt_secret("ak_chat")

    monkeypatch.setattr(dbc, "register_and_save", register_and_save, raising=False)
    return seen


def test_chat_registration_sends_the_password_to_the_sealed_store_only(sealed_store):
    state = {"user_id": "user-1", "messages": [], "dta_database_registry": {}}
    reply = planner._handle_register_database(state, RegisterDatabaseParams(
        alias="wh_chat", db_type="postgresql", host="db", port=5432, database="w",
        user="u", password=SYN))
    assert_no_credentials(state)
    assert SYN not in reply
    assert sealed_store and sealed_store[0]["password"] == SYN
    assert state["dta_database_registry"]["wh_chat"]["conn_id"] == sealed_store[0]["customer_id"]


def test_chat_typed_password_is_sealed_before_the_model_the_logs_and_state(sealed_store, monkeypatch, caplog):
    """Through the planner: the model sees a token, the store gets the value, state neither."""
    sent = []

    def model(msgs, tool_choice, invoke_kwargs):
        sent.append(msgs)
        text = next(m for m in reversed(msgs) if isinstance(m, HumanMessage)).content
        password = re.search(r"password\s+(\S+)", text).group(1)   # copies what it was shown
        return AIMessage(content="", tool_calls=[{"name": "register_database", "id": "r", "args": {
            "alias": "wh_chat", "db_type": "postgresql", "host": "db", "port": 5432,
            "database": "w", "user": "u", "password": password}}])

    monkeypatch.setattr(planner, "_invoke_planner_with_recovery", model)
    monkeypatch.setattr(planner, "llm", MagicMock(), raising=False)
    for name in ("_seed_ray_fields", "_seed_infra_request_if_needed"):
        monkeypatch.setattr(planner, name, lambda *a, **k: None)
    state = {"user_id": "user-1", "dta_database_registry": {}, "messages": [HumanMessage(
        content=f"register a postgres connection wh_chat: host db port 5432 database w user u password {SYN}")]}
    caplog.set_level("DEBUG")
    out = planner.plan_etl_job(state)
    state.update(out)
    assert sent, "the planner model was not reached"
    assert SYN not in repr(sent), "the typed password was sent to the model"
    assert sealed_store and sealed_store[0]["password"] == SYN
    assert_no_credentials(state)                     # messages included
    assert SYN not in caplog.text, "the typed password reached the planner's logs"
    replies = [m.content for m in state["messages"] if isinstance(m, AIMessage)]
    assert replies and all(SYN not in r for r in replies)


# --------------------------------------------------------------------------- #
# GKE: the RayJob carries no credential
# --------------------------------------------------------------------------- #

def test_gke_rayjob_carries_no_credential_and_reads_a_secret(monkeypatch):
    import app.data_transfer_docker.gke_run as gk
    captured = {}

    class _Outcome:
        status, logs, details = "SUCCEEDED", "", None

    monkeypatch.setattr(gk, "run_rayjob_from_yaml", lambda **kw: (captured.update(kw), _Outcome())[1])
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://127.0.0.1:8265")
    env = {"DTA_SOURCE_CONN_STR": f"postgresql://u:{SYN}@db:5432/clinic",
           "AWS_SECRET_ACCESS_KEY": SYN}
    gk.launch_gke_pipeline("print('x')", job_id="t1", cloud_env=env)
    yaml_text = captured["yaml_text"]
    assert SYN not in yaml_text, "a credential was rendered into the RayJob manifest"
    assert "secretKeyRef" in yaml_text and "dta-t1-env" in yaml_text
    for key in env:
        assert f"key: {key}" in yaml_text


def test_kuberay_mode_puts_the_credentials_in_an_owned_job_secret(monkeypatch):
    import json as _json
    from app.infra import ray_job_runner as rjr
    calls = []

    def kubectl(*args, timeout_s=120, input_text=None):
        calls.append((args, input_text))
        if args[:2] == ("get", "rayjob") and "jsonpath={.metadata.uid}" in args:
            return 0, "uid-123", ""
        if args[:2] == ("get", "rayjob"):
            return 0, _json.dumps({"status": {"jobStatus": "SUCCEEDED", "jobDeploymentStatus": "Complete"}}), ""
        return 0, "", ""

    monkeypatch.delenv("RAY_DASHBOARD_URL", raising=False)
    monkeypatch.setattr(rjr, "_kubectl", kubectl)
    monkeypatch.setattr(rjr, "_get_current_context", lambda: "test")
    monkeypatch.setattr(rjr, "fetch_rayjob_logs", lambda *a, **k: "")
    yaml_text = "apiVersion: ray.io/v1\nkind: RayJob\nmetadata:\n  name: dta-t1\n"
    rjr.run_rayjob_from_yaml(yaml_text, namespace="jobs", rayjob_name="dta-t1",
                             secret_env={"DTA_SOURCE_CONN_STR": SYN}, secret_name="dta-t1-env")
    created = [(a, i) for a, i in calls if a[0] == "create"]
    assert created, "no job Secret was created"
    secret = _json.loads(created[0][1])
    assert secret["stringData"] == {"DTA_SOURCE_CONN_STR": SYN}
    # Owned by the RayJob in the create itself: no patch, so no read access is needed.
    assert secret["metadata"]["ownerReferences"] == [
        {"apiVersion": "ray.io/v1", "kind": "RayJob", "name": "dta-t1", "uid": "uid-123"}]
    order = [a[0] for a, _i in calls]
    assert order.index("apply") < order.index("create"), "the RayJob must exist before its Secret"
    assert not [a for a, _i in calls if a[0] == "patch" or a[:2] == ("get", "secret")]
    assert all(SYN not in " ".join(a) for a, _i in calls), "a credential reached a kubectl argument"
    assert [a for a, _i in calls if a[:3] == ("delete", "secret", "dta-t1-env")]


# --------------------------------------------------------------------------- #
# Static: nothing writes a credential field into the registry
# --------------------------------------------------------------------------- #

_SECRET_FIELDS = {"password", "secret_key", "access_key", "gcp_service_account_json", "az_access_key",
                  "az_sas_token", "az_storage_account", "azure_connection_string", "conn_str",
                  "connection_string", "token", "api_key", "service_account_json"}


def _str_keys(node: ast.Dict):
    return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def _returns_secret_dict(fn: ast.FunctionDef) -> bool:
    return any(isinstance(n, ast.Return) and isinstance(n.value, ast.Dict) and _str_keys(n.value) & _SECRET_FIELDS
               for n in ast.walk(fn))


def _registry_writes_with_secrets(tree: ast.Module, where: str):
    """Writes that put a credential into a registry-shaped entry (one with a db_type)."""
    found = []
    secret_fns = {fn.name for fn in ast.walk(tree)
                  if isinstance(fn, ast.FunctionDef) and _returns_secret_dict(fn)}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        entry_names, secret_names, dump_names, registry_names = set(), set(), set(), set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name, value = node.targets[0].id, node.value
                if isinstance(value, ast.Dict) and "db_type" in _str_keys(value):
                    entry_names.add(name)
                if isinstance(value, ast.Call):
                    callee = ast.unparse(value.func)
                    if callee.split(".")[-1] in secret_fns:
                        secret_names.add(name)
                    if callee.endswith("model_dump"):
                        dump_names.add(name)
                    if callee.endswith("_get_db_registry"):
                        registry_names.add(name)
        for node in ast.walk(fn):
            # entry["password"] = ... on a dict that is a registry entry
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                            and t.value.id in entry_names and isinstance(t.slice, ast.Constant)
                            and t.slice.value in _SECRET_FIELDS):
                        found.append(f"{where}:{node.lineno} {fn.name}: {ast.unparse(t)} = ...")
                    # registry[alias] = <a model_dump of the typed params>
                    if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                            and t.value.id in registry_names and isinstance(node.value, ast.Name)
                            and node.value.id in dump_names):
                        found.append(f"{where}:{node.lineno} {fn.name}: registry entry is a raw model_dump")
            # {"db_type": ..., **creds} where creds came from a function returning a password
            if isinstance(node, ast.Dict) and "db_type" in _str_keys(node):
                for k, v in zip(node.keys, node.values):
                    if k is None and isinstance(v, ast.Name) and v.id in secret_names:
                        found.append(f"{where}:{node.lineno} {fn.name}: registry entry splats {v.id}")
                bad = _str_keys(node) & _SECRET_FIELDS
                if bad:
                    found.append(f"{where}:{node.lineno} {fn.name}: registry entry literal with {sorted(bad)}")
    return found


def test_no_code_writes_credential_fields_into_the_registry():
    files = sorted((REPO / "app").rglob("*.py")) + sorted((REPO / "avaloka").rglob("*.py"))
    found = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        found += _registry_writes_with_secrets(tree, str(path.relative_to(REPO)))
    assert not found, "credential fields written into registry entries:\n" + "\n".join(found)


def test_the_static_check_bites():
    """The checker finds each shape the pre-handle planner used."""
    src = '''
def _creds(data):
    return {"host": data["h"], "password": data["p"]}
def fetch(data):
    creds = _creds(data)
    return {"db_type": "postgresql", **creds}
def cloud(conn):
    entry = {"db_type": "aws_s3", "bucket_name": conn["b"]}
    entry["secret_key"] = conn["s"]
    return entry
def register(state, params):
    registry = _get_db_registry(state)
    entry = params.model_dump(exclude_none=True)
    registry[params.alias] = entry
'''
    found = _registry_writes_with_secrets(ast.parse(src), "x.py")
    assert len(found) == 3, found


# --------------------------------------------------------------------------- #
# The chat transcript: a typed credential is sealed at the API, before anything reads it
# --------------------------------------------------------------------------- #

try:  # tests/ as a package, or tests/ on sys.path
    from tests.server_harness import client, make_auth_headers, _do_upload  # noqa: F401
except ImportError:  # pragma: no cover
    from server_harness import client, make_auth_headers, _do_upload  # noqa: F401


def test_a_typed_password_is_sealed_before_the_transcript_and_the_graph(client, sealing_key):
    import app.api.server as server
    upload = _do_upload(client, user_id="u-seal")
    seen = []
    real = server.GRAPH.invoke

    def invoke(state_in, config=None):
        seen.append(state_in["messages"][-1].content)
        return real(state_in, config)

    server.GRAPH.invoke = invoke
    text = f"register a postgres connection wh: host db port 5432 database w user u password {SYN}"
    resp = client.post(f"/threads/{upload['thread_id']}/messages",
                       json={"content": text, "metadata": {"dataset_id": upload["dataset_id"]}},
                       headers=make_auth_headers("u-seal"))
    assert resp.status_code == 200, resp.text
    assert seen and SYN not in seen[-1], "the graph received the typed password"
    assert SYN not in resp.text, "the turn's response repeats the typed password"
    history = client.get(f"/threads/{upload['thread_id']}/messages", headers=make_auth_headers("u-seal"))
    assert SYN not in history.text, "the stored transcript holds the typed password"
    from app.core.typed_secrets import open_sealed
    assert text in open_sealed(seen[-1])            # sealed, not lost: the DTA can open it


@pytest.mark.parametrize("text", [
    "show the rows where password is null",
    "what share of users reset their password in March?",
    "group by secret_santa and count",
])
def test_ordinary_questions_are_not_sealed(text, sealing_key):
    from app.core.typed_secrets import seal_typed_secrets
    assert seal_typed_secrets(text) == (text, 0)


@pytest.mark.parametrize("text", [
    f"connect postgresql://etl:{SYN}@db:5432/w",
    f"register mysql host h port 3306 user u password: '{SYN}' database d",
    f'register gcs bucket b service account {{"type":"service_account","private_key":"{SYN}"}}',
    f"register s3 bucket b access_key AKIA1 secret_key={SYN} region us-east-1",
])
def test_typed_credentials_are_sealed_and_open_back(text, sealing_key):
    from app.core.typed_secrets import open_sealed, redact_sealed, seal_typed_secrets
    sealed, n = seal_typed_secrets(text)
    assert n >= 1 and SYN not in sealed
    assert seal_typed_secrets(sealed) == (sealed, 0)          # sealing twice changes nothing
    assert open_sealed(sealed) == text
    assert SYN not in redact_sealed(sealed) and "<redacted>" in redact_sealed(sealed)
