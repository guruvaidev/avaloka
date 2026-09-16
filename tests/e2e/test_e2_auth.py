"""
L5 API pack -- plan suite E2.05-E2.08 + E11.01: authentication, authorization,
tenant isolation.

The Avaloka auth model has no route-level ``Depends``: every handler calls
``_resolve_user_id`` by hand (app/api/server.py:655-678). A route added without
that call therefore fails open and no static review catches it. So the 401 sweep
here is GENERATED from ``route_inventory()`` -- never hand-listed -- and a new
route is swept the day it is merged.

Runs hermetically (default) and unchanged against a deployment (AVALOKA_API_URL):

  E2.05   anonymous sweep, every non-public route x 5 unauthenticated probes
  E2.05a  DEFECT xfail: POST /threads never calls _resolve_user_id
  E2.06   JWT attack matrix + the pinned no-issuer-validation gap
  E2.06a  identity fallback: sub OR email
  E2.07   AVALOKA_ALLOW_INSECURE_AUTH -- start-up refusal, not request refusal
  E2.08   cross-tenant "never 200" sweep over every {id} route
  E2.08a  the dual-credential model: bearer token AND X-Avaloka-Session
  E11.01  tenant-scoped dataset listing

E2.08 is deliberately a *shallow* invariant -- the "never 200" rule: the sweep
makes every {id} placeholder a real resource owned by user A, then probes it with
user B's token and asserts the answer is always one of 400/401/403/404/415/422/503
and never carries an owner-only value. Deeper per-route ownership assertions
(threads/datasets/models/assets) already live in tests/test_server_integration.py;
this sweep exists so a newly added {id} route cannot silently skip the ownership
check. The single documented 200 a non-owner can obtain -- an orphan thread's
empty history -- is pinned separately.

Legs that need a backend the in-memory fake stack does not provide (MLflow) skip
with a reason rather than failing.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest

from tests.e2e.conftest import (
    MODE,
    UNAUTH_MESSAGE,
    ApiClient,
    mint_token,
    route_id,
    route_inventory,
)

# Routes that are known to fail open TODAY. Every member must be covered by a
# dedicated @pytest.mark.defect xfail test, and is excluded from the generated
# sweep so the pack stays green enough to gate PRs.
#   POST /threads -- app/api/server.py:2028-2049, see test_e2_05a_* below.
KNOWN_FAIL_OPEN: set[Tuple[str, str]] = {("POST", "/threads")}

# Cross-tenant sweep: statuses that provably leak nothing.
NON_LEAKING_STATUSES = {400, 401, 403, 404, 415, 422, 503}

# The sweep probes resources that really belong to user A. These ids are the
# placeholders conftest.PATH_PARAM_SAMPLE substitutes into {…} path parameters.
OWNER = "user-a"
INTRUDER = "user-b"
OWNED_SESSION = "session-does-not-exist"
OWNED_DATASET = "dataset-does-not-exist"
OWNED_THREAD = "thread-does-not-exist"
OWNED_TASK = "task-does-not-exist"
# A value that only ever lives inside the owner's session, never in a URL.
OWNER_SECRET = "code-registry/owner-private-artifact.py"

# Minimal schema-valid payloads so an anonymous probe reaches the handler's auth
# check instead of being turned away by FastAPI body/query validation (422).
# A new route that needs one and lacks an entry fails the sweep loudly -- that is
# intended: the pack cannot prove such a route is protected.
_PROBE_PAYLOAD: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("GET", "/buckets/list"): {"params": {"backend": "gcs", "bucket": "any-bucket"}},
    ("POST", "/api/database/connect"): {"json": {"customer_id": "c-1", "api_key": "k-1"}},
    ("POST", "/api/register-existing-storage"): {
        "json": {"storage_uri": "gs://external/p", "key": "foo.csv", "connection_id": "conn-1"}
    },
    ("POST", "/api/v1/database/query"): {"json": {"content": "select 1"}},
    ("POST", "/threads/{thread_id}/messages"): {"json": {"content": "hello"}},
}

_ANON_PROBES: List[Tuple[str, Optional[Dict[str, str]]]] = [
    ("no_header", None),
    ("bearer_no_token", {"Authorization": "Bearer"}),
    ("basic_scheme", {"Authorization": "Basic xyz"}),
    ("unparseable_scheme", {"Authorization": "garbage"}),
    ("wellformed_garbage_token", {"Authorization": "Bearer aaa.bbb.ccc"}),
]


def _payload(method: str, template: str) -> Dict[str, Any]:
    override = _PROBE_PAYLOAD.get((method, template))
    if override is not None:
        return dict(override)
    return {"json": {}} if method in {"POST", "PUT", "PATCH"} else {}


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _alg_none_token(user_id: str = "user-attacker") -> str:
    """Unsigned token: header {"alg":"none"} + payload + empty signature."""
    header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = _b64url(
        json.dumps({"sub": user_id, "iat": int(time.time()), "exp": int(time.time()) + 3600}).encode()
    )
    return f"{header}.{payload}."


def _tampered_payload_token(user_id: str = "user-1", victim: str = "user-victim") -> str:
    """Valid signature, payload bytes mutated after signing."""
    header, payload, signature = mint_token(user_id).split(".")
    padded = payload + "=" * (-len(payload) % 4)
    claims = json.loads(base64.urlsafe_b64decode(padded))
    claims["sub"] = victim
    return f"{header}.{_b64url(json.dumps(claims).encode())}.{signature}"


def _bearer(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _param_routes() -> List[Tuple[str, str, str]]:
    return [r for r in route_inventory() if "{" in r[1] and (r[0], r[1]) not in KNOWN_FAIL_OPEN]


def _sweep_routes() -> List[Tuple[str, str, str]]:
    return [r for r in route_inventory() if (r[0], r[1]) not in KNOWN_FAIL_OPEN]


def _sweep_ids(routes: Iterable[Tuple[str, str, str]]) -> List[str]:
    return [route_id(m, p) for m, p, _ in routes]


# Third-party packages the in-memory fake stack does not stand in for. An
# exception from one of these means "this route needs real infrastructure", so
# the leg skips; anything else (a TypeError in a handler, say) still fails loudly.
_MISSING_BACKEND_MODULES = {"mlflow", "redis", "celery", "kombu", "redbeat", "socket"}
_MISSING_BACKEND_BUILTINS = (ConnectionError, OSError)


def _is_missing_backend(exc: BaseException) -> bool:
    root = type(exc).__module__.split(".", 1)[0]
    return root in _MISSING_BACKEND_MODULES or isinstance(exc, _MISSING_BACKEND_BUILTINS)


def _probe_authenticated(api: ApiClient, method: str, concrete: str, template: str, **kw: Any):
    """Issue an authenticated probe; skip hermetically when a real backend is required."""
    try:
        return api.request(method, concrete, **kw)
    except Exception as exc:  # noqa: BLE001 -- fake stack has no MLflow / live Redis
        if api.mode == "hermetic" and _is_missing_backend(exc):
            pytest.skip(
                f"{route_id(method, template)} reaches a backend the hermetic fakes do not "
                f"provide ({type(exc).__name__}); covered in deployed mode"
            )
        raise


def _requires_hermetic_state() -> None:
    if MODE != "hermetic":
        pytest.skip(
            "seeds session state straight into the in-process fake store; "
            "a deployment needs the same state provisioned through its own API first"
        )


def _seed_session(session_id: str, **fields: Any) -> None:
    from tests import test_server_integration as si

    si.SESSIONS[session_id] = dict(fields)


def _seed_thread(thread_id: str, session_id: str) -> None:
    from tests import test_server_integration as si

    si.THREAD_TO_SESSION[thread_id] = session_id


def _seed_owner_resources() -> None:
    """Make every swept {id} placeholder a real resource owned by OWNER."""
    _seed_session(
        OWNED_SESSION,
        user_id=OWNER,
        dataset_id=OWNED_DATASET,
        tasks=[OWNED_TASK],
        gcs_code_object_key=OWNER_SECRET,
    )
    _seed_thread(OWNED_THREAD, OWNED_SESSION)


# ---------------------------------------------------------------------------
# E2.05 -- generated anonymous sweep
# ---------------------------------------------------------------------------


def test_e2_05_route_inventory_is_populated() -> None:
    """The generated sweep must actually enumerate routes, never silently empty."""
    inventory = route_inventory()
    assert len(inventory) >= 20, f"route inventory collapsed to {len(inventory)} routes"
    assert ("GET", "/datasets") in {(m, p) for m, p, _ in inventory}


@pytest.mark.parametrize("method,template,concrete", _sweep_routes(), ids=_sweep_ids(_sweep_routes()))
@pytest.mark.parametrize("probe,headers", _ANON_PROBES, ids=[p for p, _ in _ANON_PROBES])
def test_e2_05_anonymous_probe_is_401(
    api: ApiClient,
    method: str,
    template: str,
    concrete: str,
    probe: str,
    headers: Optional[Dict[str, str]],
) -> None:
    """Every non-public route rejects an unauthenticated caller with 401 + the standard detail; never 200, never 500."""
    resp = api.request(method, concrete, headers=headers, **_payload(method, template))
    assert resp.status_code == 401, (
        f"{route_id(method, template)} probe={probe} -> {resp.status_code} {resp.text[:300]}"
    )
    assert resp.json().get("detail") == UNAUTH_MESSAGE


@pytest.mark.parametrize("method,template", sorted(KNOWN_FAIL_OPEN))
def test_e2_05_known_fail_open_entries_still_exist(method: str, template: str) -> None:
    """Each excluded fail-open route is still a live route, so the exclusion cannot rot silently."""
    assert (method, template) in {(m, p) for m, p, _ in route_inventory()}


# ---------------------------------------------------------------------------
# E2.05a -- confirmed defect
# ---------------------------------------------------------------------------


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason="DEFECT E2.05a (P0): POST /threads never calls _resolve_user_id "
    "(app/api/server.py:2028-2049) - an anonymous caller creates LangGraph threads "
    "bound to a caller-chosen session. Remove this xfail when fixed.",
)
def test_e2_05a_post_threads_rejects_anonymous_caller(api: ApiClient) -> None:
    """POST /threads must reject an unauthenticated caller with 401 instead of creating a thread."""
    resp = api.post("/threads", json={})
    assert resp.status_code == 401, f"anonymous thread created: {resp.status_code} {resp.text[:300]}"
    assert resp.json().get("detail") == UNAUTH_MESSAGE


@pytest.mark.defect
@pytest.mark.xfail(
    strict=True,
    reason="DEFECT E2.05a (P0): POST /threads binds the new thread to any caller-supplied "
    "X-Avaloka-Session without checking the token (app/api/server.py:2042-2044). "
    "Remove this xfail when fixed.",
)
def test_e2_05a_post_threads_rejects_anonymous_session_binding(api: ApiClient) -> None:
    """An anonymous POST /threads carrying another user's session id must be rejected, not bound."""
    _seed_session("sess-victim", user_id="user-a", dataset_id="ds-a")
    resp = api.post("/threads", json={}, session="sess-victim")
    assert resp.status_code == 401, f"anonymous binding accepted: {resp.status_code} {resp.text[:300]}"


# ---------------------------------------------------------------------------
# E2.06 -- JWT attack matrix
# ---------------------------------------------------------------------------

_JWT_ATTACKS: List[Tuple[str, Any]] = [
    ("expired", lambda: mint_token("user-1", expires_in=-60)),
    ("alg_none", _alg_none_token),
    ("wrong_secret", lambda: mint_token("user-1", secret="attacker-secret-not-the-real-one")),
    ("tampered_payload", _tampered_payload_token),
    ("missing_sub", lambda: mint_token("user-1", claims={"sub": None})),
]

_PROTECTED_PROBE_ROUTES = ["/datasets", "/threads"]


@pytest.mark.parametrize("path", _PROTECTED_PROBE_ROUTES)
@pytest.mark.parametrize("attack,make_token", _JWT_ATTACKS, ids=[a for a, _ in _JWT_ATTACKS])
def test_e2_06_forged_token_is_401(api: ApiClient, attack: str, make_token: Any, path: str) -> None:
    """Expired, alg=none, wrong-secret, tampered and sub-less tokens are all rejected with 401."""
    resp = api.get(path, headers=_bearer(make_token()))
    assert resp.status_code == 401, f"{attack} accepted on {path}: {resp.status_code} {resp.text[:300]}"
    assert resp.json().get("detail") == UNAUTH_MESSAGE


def test_e2_06_pins_no_issuer_validation(api: ApiClient) -> None:
    """Pins the OPEN DECISION: jwt.decode runs with verify_aud=False and no issuer check
    (server.py:667-671), so a token from a foreign issuer signed with the same secret is
    accepted. Product must decide whether to enforce iss/aud before this becomes a defect."""
    token = mint_token(
        "user-1",
        claims={"iss": "https://evil.example/auth/v1", "aud": "some-other-project"},
    )
    whoami = api.get("/debug/whoami", headers=_bearer(token))
    assert whoami.status_code == 200
    assert whoami.json()["user_id"] == "user-1"

    protected = api.get("/datasets", headers=_bearer(token))
    assert protected.status_code == 200, protected.text[:300]


def test_e2_06a_pins_email_identity_fallback(api: ApiClient) -> None:
    """Pins the OPEN DECISION: _resolve_user_id returns sub OR email (server.py:678), so a
    sub-less token authenticates as the raw email string. Risk: two tokens for one human
    (one with sub, one with only email) map to two different tenant ids and two data sets."""
    token = mint_token("ignored", claims={"sub": None, "email": "person@example.com"})
    whoami = api.get("/debug/whoami", headers=_bearer(token))
    assert whoami.status_code == 200
    assert whoami.json()["user_id"] == "person@example.com"

    protected = api.get("/datasets", headers=_bearer(token))
    assert protected.status_code == 200, protected.text[:300]


# ---------------------------------------------------------------------------
# E2.07 -- AVALOKA_ALLOW_INSECURE_AUTH
# ---------------------------------------------------------------------------


def test_e2_07_missing_secret_without_override_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no SUPABASE_JWT_SECRET and no override the lifespan raises: the server refuses to
    start (crash-loop) rather than serving rejections."""
    import app.api.server as server

    monkeypatch.setattr(server, "JWT_SECRET", "", raising=False)
    monkeypatch.setattr(server, "ALLOW_INSECURE_AUTH", False, raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        server._validate_auth_config()

    message = str(excinfo.value)
    assert "SUPABASE_JWT_SECRET" in message
    assert "AVALOKA_ALLOW_INSECURE_AUTH" in message


def test_e2_07_missing_secret_with_override_starts_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With the override set start-up succeeds and logs CRITICAL instead of raising."""
    import logging

    import app.api.server as server

    monkeypatch.setattr(server, "JWT_SECRET", "", raising=False)
    monkeypatch.setattr(server, "ALLOW_INSECURE_AUTH", True, raising=False)

    with caplog.at_level(logging.CRITICAL, logger=server.logger.name):
        assert server._validate_auth_config() is None

    assert any(r.levelno == logging.CRITICAL for r in caplog.records)


def test_e2_07_no_secret_rejects_token_signed_with_empty_secret(
    api: ApiClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no secret configured every request is unauthenticated -- a token signed with the
    empty string is never verified against it (server.py:660-663)."""
    if MODE != "hermetic":
        pytest.skip(
            "patches the in-process server module's JWT_SECRET; a deployment's secret "
            "cannot be unset from a black-box test"
        )

    import app.api.server as server

    monkeypatch.setattr(server, "JWT_SECRET", "", raising=False)

    resp = api.get("/datasets", headers=_bearer(mint_token("user-1", secret="")))
    assert resp.status_code == 401, resp.text[:300]
    assert resp.json().get("detail") == UNAUTH_MESSAGE


# ---------------------------------------------------------------------------
# E2.08 -- cross-tenant "never 200" sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method,template,concrete", _param_routes(), ids=_sweep_ids(_param_routes()))
def test_e2_08_foreign_resource_never_returns_data(
    api: ApiClient, method: str, template: str, concrete: str
) -> None:
    """An authenticated user-B token against a user-A-owned {id} resource never yields 200 --
    only 400/401/403/404/415/422/503."""
    if MODE == "hermetic":
        _seed_owner_resources()

    resp = _probe_authenticated(
        api, method, concrete, template, user=INTRUDER, **_payload(method, template)
    )

    assert resp.status_code in NON_LEAKING_STATUSES, (
        f"{route_id(method, template)} -> {resp.status_code} {resp.text[:300]}"
    )
    assert OWNER_SECRET not in resp.text


def test_e2_08_pins_orphan_thread_history_is_an_empty_200(api: ApiClient) -> None:
    """Pins the one documented 200 a non-owner can get: a thread with no session binding has
    no owner to authorize against, so GET /threads/{id}/messages returns {"messages": []}
    rather than 404 (server.py:4786-4800). Empty, so it leaks nothing -- but it is why the
    E2.08 sweep binds every swept thread to an owner first."""
    _requires_hermetic_state()

    resp = api.get("/threads/orphan-thread-with-no-session/messages", user=INTRUDER)
    assert resp.status_code == 200, resp.text[:300]
    assert resp.json() == {"messages": []}


# ---------------------------------------------------------------------------
# E2.08a -- the dual-credential model (token AND session)
# ---------------------------------------------------------------------------

_TASK_ROUTES = [
    ("GET", "/tasks"),
    ("GET", "/tasks/task-a-1/info"),
    ("GET", "/tasks/task-a-1/status"),
    ("GET", "/tasks/task-a-1/result/0"),
    ("DELETE", "/tasks/task-a-1"),
]


@pytest.mark.parametrize("method,path", _TASK_ROUTES, ids=[f"{m} {p}" for m, p in _TASK_ROUTES])
def test_e2_08a_pins_tasks_need_session_as_well_as_token(
    api: ApiClient, method: str, path: str
) -> None:
    """Pins the DUAL-CREDENTIAL contract: /tasks needs a bearer token AND X-Avaloka-Session;
    a valid token with no session is 400 "No session found", not 401 -- so the sweep's 400s
    must not be misread as auth failures."""
    resp = _probe_authenticated(api, method, path, path, user="user-a")
    assert resp.status_code == 400, f"{method} {path} -> {resp.status_code} {resp.text[:300]}"
    assert "No session found" in resp.json().get("detail", "")


def test_e2_08a_valid_token_plus_own_session_reaches_the_handler(api: ApiClient) -> None:
    """Token + the caller's own session id gets past both credentials and lists that session's tasks."""
    _requires_hermetic_state()

    _seed_session("sess-a", user_id="user-a", dataset_id="ds-a", tasks=["task-a-1"])
    resp = api.get("/tasks", user="user-a", session="sess-a")
    assert resp.status_code == 200, resp.text[:300]
    assert isinstance(resp.json(), list)


@pytest.mark.parametrize(
    "method,path",
    _TASK_ROUTES,
    ids=[f"{m} {p}" for m, p in _TASK_ROUTES],
)
def test_e2_08a_foreign_session_id_never_returns_the_owners_tasks(
    api: ApiClient, method: str, path: str
) -> None:
    """User A's token plus user B's session id is treated as no session at all (server.py:4432)
    -- 400/403/404 and never B's task data."""
    _requires_hermetic_state()

    _seed_session("sess-b", user_id="user-b", dataset_id="ds-b", tasks=["task-b-secret"])

    resp = _probe_authenticated(api, method, path, path, user="user-a", session="sess-b")
    assert resp.status_code in {400, 403, 404}, f"{resp.status_code} {resp.text[:300]}"
    assert "task-b-secret" not in resp.text


# ---------------------------------------------------------------------------
# E11.01 -- tenant isolation on the collection endpoints
# ---------------------------------------------------------------------------


def test_e11_01_dataset_listing_is_tenant_scoped(api: ApiClient) -> None:
    """GET /datasets returns only the caller's own datasets; another tenant's ids never appear."""
    _requires_hermetic_state()

    _seed_session("sess-a", user_id="user-a", dataset_id="ds-owned-by-a")
    _seed_session("sess-b", user_id="user-b", dataset_id="ds-owned-by-b")

    resp = api.get("/datasets", user="user-a")
    assert resp.status_code == 200, resp.text[:300]
    ids = {item["dataset_id"] for item in resp.json()}
    assert "ds-owned-by-b" not in ids
    assert "ds-owned-by-b" not in resp.text


@pytest.mark.parametrize("path", ["/datasets/ds-owned-by-a", "/datasets/ds-owned-by-a/preview"])
def test_e11_01_foreign_dataset_read_is_404(api: ApiClient, path: str) -> None:
    """Reading another tenant's dataset by id is 404 "not found for this user", never its rows."""
    _requires_hermetic_state()

    _seed_session("sess-a", user_id="user-a", dataset_id="ds-owned-by-a")

    resp = api.get(path, user="user-b")
    assert resp.status_code == 404, f"{resp.status_code} {resp.text[:300]}"


def test_e11_01_foreign_dataset_delete_is_404(api: ApiClient) -> None:
    """Deleting another tenant's dataset is refused with 404 and leaves the owner's session intact."""
    _requires_hermetic_state()

    from tests import test_server_integration as si

    _seed_session("sess-a", user_id="user-a", dataset_id="ds-owned-by-a")

    resp = api.delete("/datasets/ds-owned-by-a", user="user-b")
    assert resp.status_code == 404, f"{resp.status_code} {resp.text[:300]}"
    assert si.SESSIONS["sess-a"]["dataset_id"] == "ds-owned-by-a"
