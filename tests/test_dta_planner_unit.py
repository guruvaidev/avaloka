"""
Integration tests for the Planner ↔ Data-Transfer-Agent (DTA) handoff.

These tests exercise `app.agents.planner._handle_initiate_transfer`, which is the
exact seam where the planner hands a user request off to the DTA. That function:

  1. Resolves the source/destination aliases from the chat registry, falling back
     to the customer_dbs UI API (``http://localhost:8081``) when not present.
  2. Fetches full credentials and calls the DTA ``data_transfer_pipeline``
     (code generation + schema validation).
  3. Routes execution to the local Docker runner or the GKE Ray runner based on
     the ``EXECUTION_ENV`` env var (default ``gke``).
  4. Translates every outcome into a user-facing chat string.

Unlike the live scripts (``test_dta_d2d.py`` / ``test_dta_c2c.py``) which spin up
real databases, generate code with an LLM, and launch real containers, these
tests **mock the DTA pipeline and the runners**. They verify the *wiring and
routing* of the integration — fast, deterministic, no network, no Docker, no LLM.

A live end-to-end smoke runner (mirroring test_dta_d2d) is provided at the bottom
behind ``RUN_LIVE=1`` for when you want a real transfer.

Run:  pytest tests/test_dta_planner_unit.py -v
"""

import os
from contextlib import contextmanager
from unittest import mock
from unittest.mock import MagicMock

import pytest

from app.agents.planner import (
    _handle_initiate_transfer,
    InitiateTransferParams,
    plan_etl_job,
)

# Patch targets — note the DTA pipeline and runners are imported *lazily* inside
# _handle_initiate_transfer, so we patch them at their source modules.
_PIPELINE = "app.agents.data_transfer_agent.data_transfer_agent.data_transfer_pipeline"
_DOCKER_RUNNER = "app.data_transfer_docker.docker_run.launch_docker_pipeline"
_GKE_RUNNER = "app.data_transfer_docker.gke_run.launch_gke_pipeline"


# ==============================================================================
# FIXTURES
# ==============================================================================

try:  # tests/ as a package, or tests/ on sys.path
    from tests.dta_confirm import run_confirmed
except ImportError:  # pragma: no cover
    from dta_confirm import run_confirmed


@pytest.fixture
def mock_state():
    """A chat state with both DBs already in the registry (no UI auto-fetch needed)."""
    return {
        "dta_database_registry": {
            "source_db": {
                "db_type": "mysql", "host": "host.docker.internal", "port": 3309,
                "database": "cali_db", "user": "root", "password": "rootpassword",
                "table": "housing",
            },
            "dest_db": {
                "db_type": "postgresql", "host": "host.docker.internal", "port": 5436,
                "database": "avaloka_dest", "user": "avaloka", "password": "avalokapassword",
                "table": "housing_high_income",
            },
        },
        "messages": [],
    }


@pytest.fixture
def transfer_params():
    # dest_table is set so a DB destination reaches the pipeline. Without it the
    # planner short-circuits at Case 1 ("Which table should I transfer into?"),
    # since a database can hold many tables and it won't guess.
    return InitiateTransferParams(
        source_alias="source_db",
        destination_alias="dest_db",
        user_prompt="Output the 'longitude', 'latitude', and 'median_house_value' columns.",
        write_mode="append",
        dest_table="housing_high_income",
    )


@pytest.fixture(autouse=True)
def _ray_executor_configured(monkeypatch):
    """These tests drive the (mocked) Ray runner, so a Ray dashboard is configured.

    The planner now checks for one before the pipeline can create a destination
    table; with none it takes the no-Ray path (tests/test_dta_no_ray_transfer.py).
    """
    monkeypatch.setenv("RAY_DASHBOARD_URL", "http://127.0.0.1:8265")


@pytest.fixture(autouse=True)
def no_network():
    """
    Replace ``planner.requests`` so credential auto-fetch never hits the network.

    By default every GET returns 404, which makes:
      - alias-not-in-registry  → auto-fetch returns None → "not registered"
      - alias-in-registry      → _fetch_full_creds falls back to the registry entry
    Individual tests can override by patching planner.requests themselves.
    """
    with mock.patch("app.agents.planner.requests", _build_requests_mock({})):
        yield


# ==============================================================================
# HELPERS
# ==============================================================================

def _build_requests_mock(responses):
    """
    Build a fake ``requests`` module.

    ``responses`` maps a URL substring -> (status_code, json_payload). Any GET whose
    URL contains a key returns that response; everything else returns 404.
    """
    session = MagicMock()

    def _get(url, *args, **kwargs):
        resp = MagicMock()
        for needle, (code, payload) in responses.items():
            if needle in url:
                resp.status_code = code
                resp.json.return_value = payload
                return resp
        resp.status_code = 404
        resp.json.return_value = {}
        return resp

    session.get.side_effect = _get

    req = MagicMock()
    # `with requests.Session() as session:` → __enter__ yields our session
    req.Session.return_value.__enter__.return_value = session
    return req


@contextmanager
def _patched_dta(
    *,
    pipeline_return=None,
    pipeline_exc=None,
    docker_success=True,
    gke_success=True,
    docker_exc=None,
    gke_exc=None,
):
    """Patch the DTA pipeline + both runners with controllable outcomes."""
    with mock.patch(_PIPELINE) as m_pipe, \
         mock.patch(_DOCKER_RUNNER) as m_docker, \
         mock.patch(_GKE_RUNNER) as m_gke:

        if pipeline_exc is not None:
            m_pipe.side_effect = pipeline_exc
        else:
            m_pipe.return_value = pipeline_return

        m_docker.side_effect = docker_exc
        m_docker.return_value = docker_success
        m_gke.side_effect = gke_exc
        m_gke.return_value = gke_success

        yield m_pipe, m_docker, m_gke


@contextmanager
def _execution_env(value):
    """Set (or clear, if value is None) EXECUTION_ENV for the duration.

    Also opts out of the VPC-private-host auto-forcing (DTA_GKE_REACHES_PRIVATE)
    so routing follows EXECUTION_ENV alone. Without this the fixture DBs on
    ``host.docker.internal`` are treated as GKE-unreachable and every run is
    forced onto Local Docker, masking the EXECUTION_ENV switch under test.
    """
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ["DTA_GKE_REACHES_PRIVATE"] = "1"
        if value is None:
            os.environ.pop("EXECUTION_ENV", None)
        else:
            os.environ["EXECUTION_ENV"] = value
        yield


# A pipeline result representing a fully successful code-gen with an executable script.
_GOOD_PIPELINE = ({"error_message": None, "warnings": None}, "def transform_data(df):\n    return df")


# ==============================================================================
# SECTION 1 — ALIAS RESOLUTION (registry + UI auto-fetch)
# ==============================================================================

def test_source_not_registered_and_autofetch_fails(mock_state, transfer_params):
    transfer_params.source_alias = "ghost_source"
    with _patched_dta(pipeline_return=_GOOD_PIPELINE) as (m_pipe, _, _):
        result = run_confirmed(mock_state, transfer_params)
    assert "Source **`ghost_source`** not registered" in result
    m_pipe.assert_not_called()  # never reaches the DTA pipeline


def test_destination_not_registered_and_autofetch_fails(mock_state, transfer_params):
    transfer_params.destination_alias = "ghost_dest"
    with _patched_dta(pipeline_return=_GOOD_PIPELINE) as (m_pipe, _, _):
        result = run_confirmed(mock_state, transfer_params)
    assert "Destination **`ghost_dest`** not registered" in result
    m_pipe.assert_not_called()


@pytest.fixture
def encryption_key(monkeypatch):
    """The deployment key both the MCP server and this process hold."""
    import hashlib
    import app.api.cloud_connections as cc
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", "unit-test-key", raising=False)
    monkeypatch.setattr(cc, "_AESGCM_KEY", hashlib.sha256(b"unit-test-key").digest(), raising=False)


def test_source_autofetched_from_ui_then_proceeds(mock_state, transfer_params, encryption_key):
    """A source absent from the chat registry is pulled from the customer_dbs API.

    The endpoint carries no password: it carries a handle (the connection string
    sealed under DB_ENCRYPTION_KEY) that the planner opens in process, and it
    answers only a service-class caller: the planner asks as the signed-in user
    (an avu1 assertion; 1.7 merge, decision 1), never with the api_key.
    """
    from app.api.cloud_connections import encrypt_secret
    from app.mcp_server.credentials import credential_handle

    transfer_params.source_alias = "ui_source"
    # Named by the user: the endpoint's "table" is its auto-detected first table,
    # never read in place of a named one (J1).
    transfer_params.source_table = "housing"
    mock_state["active_db_customer_id"] = "ui_source"
    mock_state["active_db_credential_handle"] = encrypt_secret("ak_ui_source_key")
    mock_state["user_id"] = "user-1"
    ui_responses = {
        "/customers/ui_source/credentials": (200, {
            "host": "host.docker.internal", "port": 3309, "database": "cali_db",
            "username": "root", "table": "housing",
            "credential_handle": credential_handle("mysql+pymysql://root:" + "rootpassword" + "@host.docker.internal:3309/cali_db"),
        }),
        "/customers": (200, {"customers": [
            {"customer_id": "ui_source", "database_type": "mysql"},
        ]}),
    }
    req = _build_requests_mock(ui_responses)
    with mock.patch("app.agents.planner.requests", req):
        with _execution_env("docker"), _patched_dta(pipeline_return=_GOOD_PIPELINE) as (m_pipe, m_docker, _):
            result = run_confirmed(mock_state, transfer_params)

    assert "ui_source" in mock_state["dta_database_registry"]  # cached into registry
    assert mock_state["dta_database_registry"]["ui_source"]["_source"] == "ui"
    m_pipe.assert_called_once()
    m_docker.assert_called_once()
    assert "✅ **Transfer complete" in result
    # The password reached the pipeline from the handle, not from the payload ...
    _, kwargs = m_pipe.call_args
    assert kwargs["source_credentials"]["password"] == "rootpassword"
    assert kwargs["source_credentials"]["user"] == "root"
    # ... and every call to the credential routes authenticated as the user, never the api_key.
    from app.mcp_server.credentials import verify_user_assertion
    session = req.Session.return_value.__enter__.return_value
    cred_calls = [c for c in session.get.call_args_list if "/credentials" in c.args[0]]
    assert cred_calls and all(
        verify_user_assertion(c.kwargs.get("headers", {}).get("Authorization", "")[len("Bearer "):]) == "user-1"
        for c in cred_calls)
    assert "ak_ui_source_key" not in repr(session.get.call_args_list)


def test_source_not_autofetched_without_a_credential_handle(mock_state, transfer_params, encryption_key):
    """No signed-in user (so no identity the credentials route answers): no call, no secret."""
    transfer_params.source_alias = "ui_source"
    req = _build_requests_mock({"/customers/ui_source/credentials": (200, {"host": "h", "password": "leak"})})
    with mock.patch("app.agents.planner.requests", req):
        with _patched_dta(pipeline_return=_GOOD_PIPELINE) as (m_pipe, _, _):
            result = run_confirmed(mock_state, transfer_params)
    assert "not registered" in result
    m_pipe.assert_not_called()
    session = req.Session.return_value.__enter__.return_value
    assert not [c for c in session.get.call_args_list if "/credentials" in c.args[0]]


def test_planner_ignores_a_password_the_endpoint_should_not_send(mock_state, transfer_params, encryption_key):
    """A regressed endpoint that sends `password` again is not read: the handle is the only source."""
    from app.api.cloud_connections import encrypt_secret

    transfer_params.source_alias = "ui_source"
    mock_state["active_db_customer_id"] = "ui_source"
    mock_state["active_db_credential_handle"] = encrypt_secret("ak_ui_source_key")
    mock_state["user_id"] = "user-1"
    ui_responses = {
        "/customers/ui_source/credentials": (200, {
            "host": "h", "port": 1, "database": "d", "username": "u", "password": "plaintext-again",
        }),
        "/customers": (200, {"customers": [{"customer_id": "ui_source", "database_type": "mysql"}]}),
    }
    req = _build_requests_mock(ui_responses)
    # With a signed-in user a name that is not a database is next tried as one
    # of their cloud connections (Supabase); none here, and no network.
    with mock.patch("app.agents.planner.requests", req), \
            mock.patch("app.agents.planner._resolve_cloud_conn_by_name", return_value=None):
        with _patched_dta(pipeline_return=_GOOD_PIPELINE) as (m_pipe, _, _):
            result = run_confirmed(mock_state, transfer_params)
    assert "not registered" in result
    m_pipe.assert_not_called()
    session = req.Session.return_value.__enter__.return_value
    assert [c for c in session.get.call_args_list if "/credentials" in c.args[0]], \
        "the endpoint was never asked, so this tested nothing"


# ==============================================================================
# SECTION 2 — DTA PIPELINE OUTCOME HANDLING (before execution)
# ==============================================================================

def test_pipeline_error_message_halts(mock_state, transfer_params):
    bad = ({"error_message": "Connection Error: Could not fetch sample data", "warnings": None}, None)
    with _patched_dta(pipeline_return=bad) as (_, m_docker, m_gke):
        result = run_confirmed(mock_state, transfer_params)
    assert "❌ **Transfer Setup Failed:**" in result
    assert "Could not fetch sample data" in result
    m_docker.assert_not_called()
    m_gke.assert_not_called()  # never executes on a setup failure


def test_pipeline_no_script_halts(mock_state, transfer_params):
    """Code-gen 'succeeded' (no error) but produced no script → schema mismatch case."""
    no_script = ({"error_message": None, "warnings": None}, None)
    with _patched_dta(pipeline_return=no_script) as (_, m_docker, m_gke):
        result = run_confirmed(mock_state, transfer_params)
    assert "no executable script was produced" in result
    m_docker.assert_not_called()
    m_gke.assert_not_called()


def test_pipeline_raises_is_caught(mock_state, transfer_params):
    with _patched_dta(pipeline_exc=RuntimeError("LLM key missing")) as (_, m_docker, m_gke):
        result = run_confirmed(mock_state, transfer_params)
    assert "❌ **Transfer failed (System Error):**" in result
    assert "LLM key missing" in result
    m_docker.assert_not_called()
    m_gke.assert_not_called()


def test_pipeline_receives_correct_args(mock_state, transfer_params):
    """The planner must forward normalized types, creds, and write_mode to the DTA."""
    transfer_params.write_mode = "overwrite"
    with _execution_env("docker"), _patched_dta(pipeline_return=_GOOD_PIPELINE) as (m_pipe, _, _):
        run_confirmed(mock_state, transfer_params)

    _, kwargs = m_pipe.call_args
    assert kwargs["source_type"] == "mysql"
    assert kwargs["destination_type"] == "postgresql"
    assert kwargs["write_mode"] == "overwrite"
    assert kwargs["source_credentials"]["host"] == "host.docker.internal"
    assert kwargs["destination_credentials"]["port"] == 5436


# ==============================================================================
# SECTION 3 — EXECUTION ROUTING (EXECUTION_ENV switch)
# ==============================================================================

def test_routes_to_docker_when_env_docker(mock_state, transfer_params):
    with _execution_env("docker"), _patched_dta(pipeline_return=_GOOD_PIPELINE) as (_, m_docker, m_gke):
        result = run_confirmed(mock_state, transfer_params)
    m_docker.assert_called_once()
    m_gke.assert_not_called()
    assert "✅ **Transfer complete" in result
    assert "Local Docker" in result


def test_routes_to_gke_when_env_gke(mock_state, transfer_params):
    with _execution_env("gke"), _patched_dta(pipeline_return=_GOOD_PIPELINE) as (_, m_docker, m_gke):
        result = run_confirmed(mock_state, transfer_params)
    m_gke.assert_called_once()
    m_docker.assert_not_called()
    assert "GKE Ray cluster" in result


def test_defaults_to_gke_when_env_unset(mock_state, transfer_params):
    with _execution_env(None), _patched_dta(pipeline_return=_GOOD_PIPELINE) as (_, m_docker, m_gke):
        run_confirmed(mock_state, transfer_params)
    m_gke.assert_called_once()
    m_docker.assert_not_called()


def test_runner_returns_false_reports_execution_failure(mock_state, transfer_params):
    with _execution_env("docker"), _patched_dta(pipeline_return=_GOOD_PIPELINE, docker_success=False):
        result = run_confirmed(mock_state, transfer_params)
    assert "❌ **Transfer failed during execution**" in result
    assert "Local Docker" in result
    assert mock_state["dta_last_transfer"]["success"] is False


def test_runner_raises_reports_execution_crash(mock_state, transfer_params):
    with _execution_env("gke"), _patched_dta(pipeline_return=_GOOD_PIPELINE, gke_exc=Exception("ray dashboard unreachable")):
        result = run_confirmed(mock_state, transfer_params)
    # Customer-facing: short reason + job reference, never a raw traceback dump.
    assert "❌ **The transfer could not be completed**" in result
    assert "Reason: ray dashboard unreachable" in result
    assert "```" not in result


def test_success_records_transfer_metadata(mock_state, transfer_params):
    with _execution_env("docker"), _patched_dta(pipeline_return=_GOOD_PIPELINE):
        result = run_confirmed(mock_state, transfer_params)

    last = mock_state["dta_last_transfer"]
    assert last["source"] == "source_db"
    assert last["destination"] == "dest_db"
    assert last["execution_env"] == "docker"
    assert last["success"] is True
    assert last["job_id"].startswith("dta-")
    assert mock_state["dta_injection_script"]  # script stashed on state
    assert "- Write mode: `append`" in result
    assert "- Job ID:" in result


def test_warnings_are_surfaced_on_success(mock_state, transfer_params):
    warned = ({"error_message": None, "warnings": ["Destination table was empty"]},
              "def transform_data(df):\n    return df")
    with _execution_env("docker"), _patched_dta(pipeline_return=warned):
        result = run_confirmed(mock_state, transfer_params)
    assert "✅ **Transfer complete" in result
    assert "⚠️ **Warnings:**" in result
    assert "Destination table was empty" in result


# ==============================================================================
# SECTION 4 — PLANNER ENTRYPOINT ROUTING (keyword fast-path → DTA)
# ==============================================================================

def test_plan_etl_job_keyword_fastpath_routes_to_dta():
    """
    "transfer from X to Y" in chat should bypass the LLM and call the DTA handler,
    flagging the turn as a DTA request. We mock the handler and the seed helpers to
    isolate the routing decision.
    """
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="transfer from source_db to dest_db")],
        "user_prompt": "",  # keep the non-analysis guard from tripping
        "dta_database_registry": {},
    }

    with mock.patch("app.agents.planner._handle_initiate_transfer", return_value="DTA_HANDLED") as m_handle, \
         mock.patch("app.agents.planner._seed_ray_fields"), \
         mock.patch("app.agents.planner._seed_infra_request_if_needed"):
        out = plan_etl_job(state)

    m_handle.assert_called_once()
    # A named source is threaded through (not silently dropped): the handler's ladder
    # resolves it as a connection, falls back to the analyzed table when the token is
    # actually that, and otherwise reports it unresolvable. Dropping it here meant
    # "transfer from X to Y" silently moved the ACTIVE dataset instead of X.
    _, called_params = m_handle.call_args[0]
    assert called_params.source_alias == "source_db"
    assert called_params.destination_alias == "dest_db"
    assert out.get("is_dta_request") is True
    assert out["messages"][-1].content == "DTA_HANDLED"


# ==============================================================================
# LIVE END-TO-END SMOKE RUNNER (opt-in)
# ==============================================================================
# Set RUN_LIVE=1 to run a *real* transfer through the planner against the
# registered local databases. Requires: DBs reachable, EXECUTION_ENV configured,
# Docker (or GKE) available, and a valid coding-agent LLM key. Mirrors test_dta_d2d.

def _run_live_smoke():
    print("\n" + "=" * 80)
    print("🚀 LIVE planner → DTA integration smoke test")
    print("=" * 80)

    state = {
        "dta_database_registry": {
            "source_db": {
                "db_type": "mysql", "host": "host.docker.internal", "port": 3309,
                "database": "cali_db", "user": "root", "password": "rootpassword",
                "table": "housing",
            },
            "dest_db": {
                "db_type": "postgresql", "host": "host.docker.internal", "port": 5436,
                "database": "avaloka_dest", "user": "avaloka", "password": "avalokapassword",
                "table": "housing_high_income",
            },
        },
        "messages": [],
    }
    params = InitiateTransferParams(
        source_alias="source_db",
        destination_alias="dest_db",
        user_prompt="Output the 'longitude', 'latitude', and 'median_house_value' columns.",
        write_mode="append",
    )
    print(f"EXECUTION_ENV = {os.environ.get('EXECUTION_ENV', 'gke (default)')}")
    result = run_confirmed(state, params)
    print("\n--- Planner reply ---")
    print(result)
    print("\n--- dta_last_transfer ---")
    print(state.get("dta_last_transfer"))


if __name__ == "__main__":
    if os.environ.get("RUN_LIVE") == "1":
        _run_live_smoke()
    else:
        import sys
        sys.exit(pytest.main([__file__, "-v"]))


# Registry credentials these tests seed live in a stand-in sealed store, not in
# state: the planner opens them by handle after the confirm gate (tests/dta_sealed.py).
try:  # tests/ as a package, or tests/ on sys.path
    from tests import dta_sealed as _dta_sealed
except ImportError:  # pragma: no cover
    import dta_sealed as _dta_sealed


@pytest.fixture(autouse=True)
def _sealed_store(monkeypatch):
    return _dta_sealed.install(monkeypatch)
