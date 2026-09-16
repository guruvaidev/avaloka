"""A broken analysis lookup must not be reported as a missing analysis.

Reported symptom: `GET /analysis/<aid>/code` answered `404 Not Found` while the
server log carried a Supabase `401 Invalid API key`. The 404 is an answer about
the DATA, and the truth was that the query never ran -- which is why an auth
misconfiguration got debugged as a missing record.
"""

import asyncio

import pytest
from fastapi import HTTPException

import app.api.server as server
from app.agents import sampling_persistence as sp


class _APIError(Exception):
    """Stands in for postgrest.exceptions.APIError, which carries .code."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- #
# Telling a credential failure apart from anything else
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("exc", [
    _APIError("JSON could not be generated", code=401),
    _APIError("forbidden", code=403),
    _APIError("JSON could not be generated", code="401"),   # postgrest sends strings too
    Exception('{"message":"Invalid API key","hint":"Double check your Supabase key."}'),
    RuntimeError("Supabase env vars not set (SUPABASE_SERVICE_ROLE_KEY)."),
])
def test_credential_failures_are_recognised(exc):
    assert server._is_supabase_auth_failure(exc) is True


@pytest.mark.parametrize("exc", [
    _APIError("relation does not exist", code=404),
    _APIError("statement timeout", code=500),
    TimeoutError("connection timed out"),
    ValueError("some programming error"),
])
def test_other_failures_are_not_mistaken_for_credential_failures(exc):
    assert server._is_supabase_auth_failure(exc) is False


# --------------------------------------------------------------------------- #
# What the endpoint tells the client
# --------------------------------------------------------------------------- #

def _resolve(monkeypatch, *, raises=None, returns=None):
    def _fake_client():
        if raises:
            raise raises
        class _R:
            data = returns or []
        class _T:
            def select(self, *a, **k): return self
            def eq(self, *a, **k): return self
            def limit(self, *a, **k): return self
            def execute(self): return _R()
        class _C:
            def table(self, *a, **k): return _T()
        return _C()

    monkeypatch.setattr(sp, "get_supabase_client", _fake_client)
    monkeypatch.setattr(server, "get_supabase_client", _fake_client, raising=False)
    return asyncio.run(server._resolve_session_from_aid("aid-1", "user-1"))


def test_a_rejected_key_is_a_503_not_a_404(monkeypatch):
    """The exact reported failure. 404 would claim the analysis does not exist;
    we never found out either way."""
    with pytest.raises(HTTPException) as ei:
        _resolve(monkeypatch, raises=_APIError("Invalid API key", code=401))
    assert ei.value.status_code == 503
    assert "credentials" in ei.value.detail.lower()


def test_any_lookup_failure_is_a_503(monkeypatch):
    """Not just auth: if the query could not run, 'not found' is unknowable."""
    with pytest.raises(HTTPException) as ei:
        _resolve(monkeypatch, raises=TimeoutError("supabase unreachable"))
    assert ei.value.status_code == 503


def test_a_genuinely_absent_analysis_still_returns_none(monkeypatch):
    """The 404 path has to survive: an empty result set is a real answer."""
    assert _resolve(monkeypatch, returns=[]) is None


# --------------------------------------------------------------------------- #
# Credentials are read when used, not frozen at import
# --------------------------------------------------------------------------- #

def test_client_uses_the_current_environment(monkeypatch):
    """The module-level constants bind at import, so whether they hold anything
    depended on who called load_dotenv() first."""
    monkeypatch.setattr(sp, "SUPABASE_URL", None, raising=False)
    monkeypatch.setattr(sp, "SUPABASE_SERVICE_ROLE_KEY", None, raising=False)
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")

    seen = {}
    fake = type("m", (), {"create_client": staticmethod(
        lambda u, k: seen.update(url=u, key=k) or "client")})
    monkeypatch.setitem(__import__("sys").modules, "supabase", fake)

    assert sp.get_supabase_client() == "client"
    assert seen == {"url": "https://proj.supabase.co", "key": "service-key"}


def test_missing_credentials_name_what_is_missing(monkeypatch):
    monkeypatch.setattr(sp, "SUPABASE_URL", None, raising=False)
    monkeypatch.setattr(sp, "SUPABASE_SERVICE_ROLE_KEY", None, raising=False)
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    with pytest.raises(RuntimeError) as ei:
        sp.get_supabase_client()
    assert "SUPABASE_SERVICE_ROLE_KEY" in str(ei.value)
    assert "SUPABASE_URL" not in str(ei.value), "named a variable that was set"
