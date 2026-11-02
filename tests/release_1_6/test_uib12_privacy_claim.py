"""UI-B12: the shipped privacy copy must match what the product does.

The dashboard footer (ui/src/components/dashboard/SecurityFooter.tsx:7) says
"Your data is encrypted, securely processed, and never shared outside your
workspace." The database-connect dialog
(ui/src/components/database/DatabaseConnectModal.tsx:285) says "Credentials are
encrypted server-side." The UI BUGS sheet marked the claim "Not an Issue"; the
owner asked for it to be verified (PLAN item 7). Two clauses do not hold at
develop-1.6 @ 49e9ab4d:

1. "never shared outside your workspace": the coder's pseudocode step sends
   the schema AND a real data row (the first uploaded-preview row) to the coding
   LLM (app/agents/coder.py:246-249, 345-346, 362-366). With nothing configured,
   that LLM is a hosted third party (Groq or OpenRouter; app/core/inference.py,
   resolve_provider), never the in-cluster LOCAL provider.
2. "Credentials are encrypted server-side": registering a database stores the
   full connection string, password included, in plaintext in customers.json
   (app/mcp_server/customer_dbs.py:488-494 builds it, :443-446 json.dumps it).
   Only the MCP api_key and username are AES-GCM encrypted afterwards
   (app/api/server.py /api/mcp-connections/{id}/encrypt).

These were strict xfails until PR #641 (fix/1.6-privacy-copy, merged into
develop-1.6 @ b342007e) changed the copy to say what it can prove: the footer now
says cloud and service credentials are encrypted at rest and that schema and
sample rows go to a hosted model provider; the dialog says the database password
is not encrypted yet. The two tests are now plain tests that pin copy and
behaviour TOGETHER: the withdrawn claim is absent, the disclosure is present,
and the disclosure is required for as long as the behaviour it discloses holds.
Hermetic: fake LLM, fake engine, fake MCP POST, tmp customers file.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parents[2]
FOOTER = ROOT / "ui/src/components/dashboard/SecurityFooter.tsx"
DB_MODAL = ROOT / "ui/src/components/database/DatabaseConnectModal.tsx"

NEVER_SHARED = "never shared outside your workspace"
CREDS_ENCRYPTED = "Credentials are encrypted server-side"
#: Disclosures shipped by PR #641 (whitespace-normalised before matching).
ROWS_DISCLOSED = "sample rows are sent to the model provider you configure"
PASSWORD_DISCLOSED = "Your database password is not encrypted yet"
#: The dialog line once database passwords are sealed (hotfix/1.6-chat-connection-sealing).
PASSWORD_SEALED_CLAIM = "Your database password is encrypted at rest"


def _text(path: Path) -> str:
    return " ".join(path.read_text().split())

#: A cell value that cannot appear in a prompt by accident.
SENTINEL = "ZQ-SENTINEL-4417"

_PROVIDER_ENV = (
    "INFERENCE_PROVIDER", "INFERENCE_PROVIDER_CODING", "AVALOKA_CODER_PROVIDER",
    "OPENROUTER_API_KEY", "INFERENCE_OPENROUTER_API_KEY", "GROQ_API_KEY",
    "GROQ_API_KEY_PLANNING_AGENT", "GROQ_API_KEY_CODING_AGENT", "OPENAI_API_KEY",
)


class _RecordingLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, messages, *a, **k):
        self.prompts.append("\n".join(str(getattr(m, "content", m)) for m in messages))
        return AIMessage(content="1. Group df by city and average fare.")


def _coder_prompt(monkeypatch) -> str:
    """Run the real pseudocode step with a recording LLM; return what it was sent."""
    import app.agents.coder as coder

    llm = _RecordingLLM()
    monkeypatch.setattr(coder, "coder_llm", llm)
    state = {
        "plan": "average fare by city",
        "user_prompt": "average fare by city",
        "schema": {"city": "object", "fare": "float64"},
        # What send_message seeds from the upload: list[dict] of real rows.
        "uploaded_csv_preview": [{"city": SENTINEL, "fare": 12.5}, {"city": "Other", "fare": 3.0}],
        "messages": [],
    }
    try:
        coder._generate_pseudocode(state)
    except Exception:  # parsing the fake reply is not under test
        pass
    assert llm.prompts, "the pseudocode step never called the coding LLM"
    return llm.prompts[0]


def _default_coding_provider_is_external(monkeypatch) -> bool:
    import app.core.inference as inference

    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    return inference.resolve_provider(role="coding", agent="coder") != inference.LOCAL


def test_uib12_harness_sees_the_coder_prompt(monkeypatch):
    """UI-B12 (control): the recorder captures the real coder prompt, schema included.

    Without this, the xfail below could pass simply because nothing was recorded.
    """
    prompt = _coder_prompt(monkeypatch)
    assert "fare" in prompt and "city" in prompt


# UI-B12: converted from a strict xfail after PR #641 (privacy copy) merged.
def test_uib12_never_shared_claim_matches_default_llm_egress(monkeypatch):
    """UI-B12: the footer no longer claims data is never shared, and it discloses that sample rows go to the configured hosted model whenever the default path sends one."""
    footer = _text(FOOTER)
    external = _default_coding_provider_is_external(monkeypatch)
    row_value_sent = SENTINEL in _coder_prompt(monkeypatch)

    assert NEVER_SHARED not in footer, "the withdrawn 'never shared' claim is back in the footer"
    assert ROWS_DISCLOSED in footer, "the footer no longer discloses that sample rows go to the model provider"
    # The disclosure is not decorative: today the default path really sends a row.
    assert external and row_value_sent, (
        "the default coding path no longer sends a data row to a hosted model; "
        "update the footer copy and this test together"
    )


# ---- credentials at rest -------------------------------------------------------

class _FakeAsyncClient:
    #: Every request body the registration route sent to the MCP server.
    sent: list = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, *a, **k):
        _FakeAsyncClient.sent.append(json.dumps({"args": a, "kwargs": k}, default=str))
        return SimpleNamespace(status_code=200, text="ok")


@pytest.fixture
def registered_password(tmp_path, monkeypatch):
    """Register one database through the real /customers/register route; return
    (password, customers.json text, MCP request bodies)."""
    import app.api.cloud_connections as cc
    import app.mcp_server.customer_dbs as cdb

    monkeypatch.setenv("MCP_SERVICE_TOKEN", secrets.token_urlsafe(32))
    # A private key, so a build that seals credentials can register (it refuses
    # without one); a build that does not seal ignores it.
    key = "uib12-" + secrets.token_hex(32)
    monkeypatch.setenv("DB_ENCRYPTION_KEY", key)
    monkeypatch.setattr(cc, "DB_ENCRYPTION_KEY", key, raising=False)
    _FakeAsyncClient.sent = []
    path = tmp_path / "customers.json"
    path.write_text("{}")
    mgr = cdb.customer_manager
    monkeypatch.setattr(mgr, "customers_file", str(path))
    monkeypatch.setattr(mgr, "customers", {})
    monkeypatch.setattr(cdb.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(
        cdb.DatabaseConnectionManager, "test_database_connection",
        staticmethod(lambda conn: {"success": True, "database": "sales", "table_count": 1, "version": "16"}),
    )
    app = FastAPI()
    app.include_router(cdb.router)
    password = "pw-" + secrets.token_hex(12)
    resp = TestClient(app).post("/customers/register", json={
        "customer_id": "uib12_cust", "customer_name": "uib12", "database_type": "postgresql",
        "host": "db.test.invalid", "port": 5432, "database_name": "sales",
        "username": "user-1", "password": password, "contact_email": "user-1@example.com",
    })
    assert resp.status_code == 200, resp.text
    return password, path.read_text(), "\n".join(_FakeAsyncClient.sent)


def test_uib12_registration_persists_the_customer(registered_password):
    """UI-B12 (control): registration really writes the record the xfail below inspects."""
    _, on_disk, mcp_sent = registered_password
    assert "uib12_cust" in json.loads(on_disk)
    assert "uib12_cust" in mcp_sent, "the registration never reached the (fake) MCP server"


# UI-B12: converted from a strict xfail after PR #641 (privacy copy) merged; made
# state-correct so it holds before and after database passwords are sealed.
def test_uib12_encrypted_credentials_claim_matches_storage(registered_password):
    """UI-B12: the connect dialog's statement about the database password matches how it is stored and sent, and the withdrawn blanket claims never return."""
    password, on_disk, mcp_sent = registered_password
    modal, footer = _text(DB_MODAL), _text(FOOTER)

    assert CREDS_ENCRYPTED not in modal, "the withdrawn 'Credentials are encrypted server-side' claim is back"
    assert "Your data is encrypted" not in footer, "the withdrawn blanket encryption claim is back"

    discloses_plaintext = PASSWORD_DISCLOSED in modal
    claims_sealed = PASSWORD_SEALED_CLAIM in modal
    assert discloses_plaintext != claims_sealed, (
        "the connect dialog must state exactly one of the two known positions on the "
        f"database password ({PASSWORD_DISCLOSED!r} or {PASSWORD_SEALED_CLAIM!r})"
    )
    if claims_sealed:
        assert password not in on_disk, "the dialog says the password is encrypted, but customers.json holds it in plaintext"
        assert password not in mcp_sent, "the dialog says the password is sent sealed, but the MCP request carries it in plaintext"
    # discloses_plaintext: the dialog says it is not encrypted yet; storage is unconstrained.
