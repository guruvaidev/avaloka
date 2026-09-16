# L5 API pack — executable form of the Avaloka 1.5.2 master test plan

Two packs implement the plan's clusterless layers:

| Pack | Path | Needs | Runs in |
|---|---|---|---|
| L5 API | `tests/e2e/` | nothing (hermetic) or a deployment | seconds |
| Static contract | `tests/contract/` | nothing — filesystem + git only | < 5 s |

Everything here is designed to gate a pull request. No Docker, no cluster, no
API keys, no network.

## Running

```bash
# Both packs, hermetic. This is the PR gate.
pytest tests/e2e tests/contract -q

# Show which confirmed defects are still open (xfail reasons)
pytest tests/e2e tests/contract -q -rxX

# Same case bodies against a live deployment (adds the cluster-only legs)
export AVALOKA_API_URL=http://localhost:9000
export SUPABASE_JWT_SECRET=<the cluster's secret>
pytest tests/e2e -q
```

In hermetic mode the API runs in-process (FastAPI `TestClient`) over the
in-memory fakes defined in `tests/test_server_integration.py` — one definition
of the fake blob store, cache and LangGraph, shared rather than forked. Setting
`AVALOKA_API_URL` switches the same test bodies to black-box HTTP and mints
tokens with the cluster's `SUPABASE_JWT_SECRET`. Legs that can only be observed
against a real deployment call `requires_deployed()` and skip with a reason.

## How to read a result

- **passed** — the contract holds.
- **xfailed** — a *confirmed defect*. The test asserts the behaviour we want;
  the `reason` names the case id, priority, and `file:line`. Every xfail is
  `strict=True`, so the day the bug is fixed the test **fails** and tells you to
  delete the marker. Nothing silently rots.
- **skipped** — the leg needs something this mode does not have; the reason says
  what (a deployment, a branch, a binary).

Never make a red test green by weakening the assertion. If reality differs from
the assertion, either the code is wrong (convert to a strict xfail with a
reason) or the expectation was wrong (fix it and say why in the docstring).

## Why the auth sweep is generated

The API has no route-level `Depends`; each route calls `_resolve_user_id`
by hand (`app/api/server.py:655`). A route added without that call fails open —
which is not hypothetical: `POST /threads` does exactly that today. So
`route_inventory()` in `conftest.py` derives the sweep from the running app
(or from `/openapi.json` when deployed) instead of a hand-written list. A route
added tomorrow is swept tomorrow, with no plan edit.

Known fail-open routes are held in a single `KNOWN_FAIL_OPEN` set in
`test_e2_auth.py`, each covered by a dedicated strict-xfail test, so the sweep
itself stays green and can gate merges while the defect is open.

## Layout

| File | Plan suite |
|---|---|
| `tests/e2e/conftest.py` | harness: dual mode, token minting, route inventory |
| `tests/e2e/test_e2_contract.py` | E2.01–E2.04, E2.09, E2.13, E2.14 |
| `tests/e2e/test_e2_auth.py` | E2.05–E2.08, E11.01 |
| `tests/e2e/test_e2_hygiene.py` | E2.10, E2.11, E2.12 |
| `tests/e2e/test_e3_ingestion.py` | E3.01–E3.03, E3.11, E3.13 |
| `tests/e2e/test_e7_inference.py` | E7.02–E7.07, E12.07 |
| `tests/contract/test_c1_repo_hygiene.py` | E11.05, E11.09 |
| `tests/contract/test_c2_chart_invariants.py` | E1, E2.07, E6.08, E6.09, E7.09, E11.10, E11.12 |
| `tests/contract/test_c3_release_coherence.py` | E14.01, E14.02, E14.07, marker audit |
| `tests/contract/test_c4_ui_wiring.py` | E9 (skips unless `ui/` is present) |

`tests/contract/test_c4_ui_wiring.py` targets files that exist only on
`ui-k8s-deploy-1.5.2`. It collects and skips cleanly on other branches, and does
real work once the UI merge lands.

## What this pack deliberately does not cover

Layers that need real infrastructure or real LLM spend stay in the existing
tiers: `tests/k8s/` T1–T6 (deployment, KubeRay, cloud, UI), and the L6 golden
missions. This pack is the layer beneath them — the one that must be cheap
enough to run on every push.
