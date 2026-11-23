"""AVALOKA_AUTH_MODE=local: allowed only self-hosted, and never a way around auth.

What has to hold (app/core/local_auth.py):

* the mode is refused, and the server will not start, unless the deployment
  says self_hosted explicitly -- a hosted deployment, a hosted-only edition,
  or a configured Supabase each refuse it;
* when it is on, only this server's own access tokens authenticate: no
  Supabase-signed token, no refresh token, no forged or expired token;
* when it is off or refused, nothing changes: the Supabase path still fails
  closed without SUPABASE_JWT_SECRET;
* the Helm chart never turns it on.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from types import SimpleNamespace

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core import local_auth

ROOT = Path(__file__).resolve().parents[1]
SECRET = "s" * 48
PASSWORD = "correct-horse-battery"

LOCAL_ENV = {
    "AVALOKA_AUTH_MODE": "local",
    "AVALOKA_DEPLOYMENT": "self_hosted",
    "AVALOKA_EDITION": "oss",
    "AVALOKA_LOCAL_AUTH_SECRET": SECRET,
    "AVALOKA_LOCAL_AUTH_PASSWORD": PASSWORD,
}
_MANAGED = list(LOCAL_ENV) + ["AVALOKA_LOCAL_AUTH_EMAIL", "SUPABASE_URL"]


@pytest.fixture
def local_env(monkeypatch):
    for k in _MANAGED:
        monkeypatch.delenv(k, raising=False)
    for k, v in LOCAL_ENV.items():
        monkeypatch.setenv(k, v)
    local_auth._failures.clear()
    yield monkeypatch
    local_auth._failures.clear()


# --------------------------------------------------------------------------
# Refusal
# --------------------------------------------------------------------------

def test_allowed_when_self_hosted():
    cfg = local_auth.load(dict(LOCAL_ENV))
    assert cfg is not None and cfg.email == local_auth.DEFAULT_EMAIL


@pytest.mark.parametrize("deployment", ["avaloka_hosted", "avaloka-hosted", "hosted", ""])
def test_refused_unless_deployment_is_explicitly_self_hosted(deployment):
    env = dict(LOCAL_ENV, AVALOKA_DEPLOYMENT=deployment)
    with pytest.raises(local_auth.LocalAuthRefused, match="self_hosted"):
        local_auth.load(env)


def test_refused_when_deployment_unset():
    # editions.py reads an unset deployment as self-hosted; this mode must not.
    env = {k: v for k, v in LOCAL_ENV.items() if k != "AVALOKA_DEPLOYMENT"}
    with pytest.raises(local_auth.LocalAuthRefused):
        local_auth.load(env)


def test_refused_for_hosted_only_edition():
    with pytest.raises(local_auth.LocalAuthRefused, match="free"):
        local_auth.load(dict(LOCAL_ENV, AVALOKA_EDITION="free"))


def test_refused_beside_a_real_supabase():
    with pytest.raises(local_auth.LocalAuthRefused, match="SUPABASE_URL"):
        local_auth.load(dict(LOCAL_ENV, SUPABASE_URL="https://example.supabase.co"))


@pytest.mark.parametrize("key,value", [
    ("AVALOKA_LOCAL_AUTH_SECRET", "short"),
    ("AVALOKA_LOCAL_AUTH_SECRET", ""),
    ("AVALOKA_LOCAL_AUTH_PASSWORD", "short"),
])
def test_refused_with_weak_secrets(key, value):
    with pytest.raises(local_auth.LocalAuthRefused):
        local_auth.load(dict(LOCAL_ENV, **{key: value}))


def test_off_unless_requested():
    assert local_auth.load({k: v for k, v in LOCAL_ENV.items() if k != "AVALOKA_AUTH_MODE"}) is None


# --------------------------------------------------------------------------
# The server: boot refusal and token resolution
# --------------------------------------------------------------------------

@pytest.fixture
def server():
    import app.api.server as server
    return server


def _req(token: str):
    return SimpleNamespace(headers={"Authorization": f"Bearer {token}"})


def test_server_refuses_to_start_when_hosted(server, local_env):
    local_env.setenv("AVALOKA_DEPLOYMENT", "avaloka_hosted")
    # The insecure-start escape hatch must not rescue it either.
    local_env.setattr(server, "ALLOW_INSECURE_AUTH", True, raising=False)
    local_env.setattr(server, "JWT_SECRET", "", raising=False)
    with pytest.raises(local_auth.LocalAuthRefused):
        server._validate_auth_config()


def test_server_starts_in_local_mode_without_supabase_secret(server, local_env):
    local_env.setattr(server, "JWT_SECRET", "", raising=False)
    server._validate_auth_config()  # no raise


def test_local_token_authenticates(server, local_env):
    cfg = local_auth.load()
    token = local_auth.issue_session(cfg)["access_token"]
    assert server._resolve_user_id(_req(token)) == cfg.user_id


def test_refresh_token_does_not_authenticate(server, local_env):
    cfg = local_auth.load()
    refresh = local_auth.issue_session(cfg)["refresh_token"]
    assert server._resolve_user_id(_req(refresh)) is None


def test_supabase_token_does_not_fall_through_in_local_mode(server, local_env):
    # A token the Supabase path WOULD accept is refused while local mode is on.
    local_env.setattr(server, "JWT_SECRET", "supabase-secret-" + "x" * 32, raising=False)
    sb = jwt.encode({"sub": "u-1", "exp": int(time.time()) + 600}, server.JWT_SECRET, algorithm="HS256")
    assert server._resolve_user_id(_req(sb)) is None


@pytest.mark.parametrize("mutate", [
    lambda c: {**c, "iss": "someone-else"},
    lambda c: {**c, "exp": int(time.time()) - 10},
    lambda c: {**c, "sub": "another-user"},
    lambda c: {**c, "typ": "refresh"},
])
def test_forged_or_stale_local_tokens_are_refused(server, local_env, mutate):
    cfg = local_auth.load()
    good = jwt.decode(local_auth.issue_session(cfg)["access_token"], SECRET, algorithms=["HS256"],
                      audience=local_auth.AUDIENCE)
    bad = jwt.encode(mutate(good), SECRET, algorithm="HS256")
    assert server._resolve_user_id(_req(bad)) is None


def test_wrong_secret_is_refused(server, local_env):
    cfg = local_auth.load()
    claims = jwt.decode(local_auth.issue_session(cfg)["access_token"], SECRET, algorithms=["HS256"],
                        audience=local_auth.AUDIENCE)
    assert server._resolve_user_id(_req(jwt.encode(claims, "t" * 48, algorithm="HS256"))) is None


def test_refused_mode_accepts_no_local_token_and_stays_fail_closed(server, local_env):
    cfg = local_auth.load()
    token = local_auth.issue_session(cfg)["access_token"]
    local_env.setenv("AVALOKA_DEPLOYMENT", "avaloka_hosted")
    local_env.setattr(server, "JWT_SECRET", "", raising=False)
    # Refused => the Supabase path runs, and with no secret it accepts nothing.
    assert server._resolve_user_id(_req(token)) is None


def test_other_modes_unchanged(server, local_env):
    local_env.delenv("AVALOKA_AUTH_MODE")
    local_env.setattr(server, "JWT_SECRET", "supabase-secret-" + "x" * 32, raising=False)
    sb = jwt.encode({"sub": "u-1", "exp": int(time.time()) + 600}, server.JWT_SECRET, algorithm="HS256")
    assert server._resolve_user_id(_req(sb)) == "u-1"
    local_env.setattr(server, "JWT_SECRET", "", raising=False)
    assert server._resolve_user_id(_req(sb)) is None


# --------------------------------------------------------------------------
# The GoTrue subset the UI's supabase-js calls
# --------------------------------------------------------------------------

@pytest.fixture
def api():
    from app.api.local_auth_routes import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_password_sign_in_user_and_refresh(api, local_env):
    r = api.post("/local-auth/auth/v1/token?grant_type=password",
                 json={"email": "LOCAL@avaloka.local", "password": PASSWORD})
    assert r.status_code == 200, r.text
    s = r.json()
    assert {"access_token", "refresh_token", "expires_at", "user"} <= set(s)
    u = api.get("/local-auth/auth/v1/user", headers={"Authorization": f"Bearer {s['access_token']}"})
    assert u.status_code == 200 and u.json()["id"] == s["user"]["id"]
    rr = api.post("/local-auth/auth/v1/token?grant_type=refresh_token", json={"refresh_token": s["refresh_token"]})
    assert rr.status_code == 200 and rr.json()["access_token"]
    # An access token is not a refresh token.
    bad = api.post("/local-auth/auth/v1/token?grant_type=refresh_token", json={"refresh_token": s["access_token"]})
    assert bad.status_code == 400


def test_wrong_password_and_lockout(api, local_env):
    for _ in range(local_auth._FAIL_LIMIT):
        r = api.post("/local-auth/auth/v1/token?grant_type=password",
                     json={"email": local_auth.DEFAULT_EMAIL, "password": "nope"})
        assert r.status_code == 400 and "access_token" not in r.json()
    r = api.post("/local-auth/auth/v1/token?grant_type=password",
                 json={"email": local_auth.DEFAULT_EMAIL, "password": PASSWORD})
    assert r.status_code == 429


def test_rest_reads_are_empty_and_writes_say_so(api, local_env):
    cfg = local_auth.load()
    h = {"Authorization": f"Bearer {local_auth.issue_session(cfg)['access_token']}"}
    assert api.get("/local-auth/rest/v1/profiles?select=*", headers=h).json() == []
    assert api.post("/local-auth/rest/v1/profiles", headers=h, json={}).status_code == 501
    assert api.get("/local-auth/rest/v1/profiles").status_code == 401


@pytest.mark.parametrize("path,method", [
    ("/local-auth/auth/v1/token?grant_type=password", "post"),
    ("/local-auth/auth/v1/user", "get"),
    ("/local-auth/rest/v1/profiles", "get"),
])
def test_every_route_is_404_when_hosted(api, local_env, path, method):
    local_env.setenv("AVALOKA_DEPLOYMENT", "avaloka_hosted")
    r = getattr(api, method)(path, json={"email": local_auth.DEFAULT_EMAIL, "password": PASSWORD}) \
        if method == "post" else api.get(path)
    assert r.status_code == 404


# --------------------------------------------------------------------------
# Helm never turns it on
# --------------------------------------------------------------------------

def test_helm_chart_never_enables_local_auth():
    chart = ROOT / "deploy" / "helm"
    hits = []
    for f in chart.rglob("*"):
        if f.is_file() and f.suffix in (".yaml", ".yml", ".tpl", ".txt", ".json"):
            text = f.read_text(errors="replace")
            for m in re.finditer(r"AVALOKA_AUTH_MODE[^\n]*", text):
                hits.append(f"{f.relative_to(ROOT)}: {m.group(0)}")
            if re.search(r"AUTH_MODE\s*[:=]\s*[\"']?local\b", text):
                hits.append(f"{f.relative_to(ROOT)}: AUTH_MODE local")
    assert not hits, f"the Helm chart must never set the laptop-only local auth mode: {hits}"
