# Running the swarm demo on a laptop

**Status:** Built on the 1.7 line (`/swarm` in the UI, `/api/swarm/missions` on the API).
**Audience:** Engineering, anyone giving the demo.

The page at `/swarm` starts a coordinated mission through the real API and
draws it live: Avaloka announces the fleet, the clones appear as they are
dispatched, each one's firefly steps fill in as they run, the cards settle on
their verdicts, and a convergence panel shows the merge. Finished missions can
be reopened, and three tabs show what each clone produced (its document), the
RL episodes, and the telemetry projection: exactly what would leave the
machine, and nothing is sent. The API is `docs/design/AGENT_TRACE.md` §6.

Nothing here needs Redis, Postgres, Supabase or a model key. A mission runs
the CLI fireflies on a thread of the API process and is held in memory (the
page says so: it is gone when the API restarts).

## 1. The API

From the repository root, Python 3.11 or 3.12 with the project installed
(`pip install -r requirements.txt`, or the `avaloka-api` image):

```bash
export AVALOKA_EDITION=oss
export AVALOKA_ALLOW_INSECURE_AUTH=1
export SUPABASE_JWT_SECRET='demo-only-change-me-0123456789abcdef'   # any string you choose
uvicorn app.api.server:app --host 0.0.0.0 --port 9000 --loop asyncio
```

**Why the secret, when authentication is "insecure".** Every swarm route is
owner-only and reads a Bearer token, like `/threads/{id}/trace`.
`AVALOKA_ALLOW_INSECURE_AUTH=1` only lets the server *start* without
`SUPABASE_JWT_SECRET`; with the secret missing, `_resolve_user_id` fails
closed and answers 401 to every request (`app/api/server.py`, "never verify
against an empty secret"). So the laptop recipe sets a secret of its own and
mints a token signed with it. Keep `AVALOKA_ALLOW_INSECURE_AUTH=1` anyway: it
is harmless once the secret is set, and it is the switch the server names in
its own error message.

Mint a token (PyJWT is already a dependency; the `sub` must be a UUID, which
is what the server stores as the owner):

```bash
export AVALOKA_DEMO_TOKEN=$(python - <<'EOF'
import os, time, uuid, jwt
print(jwt.encode({"sub": str(uuid.uuid4()), "iat": int(time.time())},
                 os.environ["SUPABASE_JWT_SECRET"], algorithm="HS256"))
EOF
)
```

Check the API before touching the UI:

```bash
curl -s http://127.0.0.1:9000/health
curl -s http://127.0.0.1:9000/api/capabilities \
  | python -c 'import json,sys; print(json.load(sys.stdin)["features"]["swarm"])'
# → the swarm limits: max_clones_per_mission 2 on the open-source edition (no token needed)
curl -s -i -H "Authorization: Bearer $AVALOKA_DEMO_TOKEN" http://127.0.0.1:9000/api/swarm/missions | head -1
# → HTTP/1.1 200 OK  (the token verifies; the list is empty so far)
curl -s -H "Authorization: Bearer $AVALOKA_DEMO_TOKEN" -H 'Content-Type: application/json' \
  -d '{"source": {"sample": "churn"}, "goal": "what drives churn", "kind": "analyze", "workers": 2}' \
  http://127.0.0.1:9000/api/swarm/missions
# → {"mission_id": "...", "status": "running", "grant": {...}, "limits": {...}, "persistence": "memory"}
curl -s -H "Authorization: Bearer $AVALOKA_DEMO_TOKEN" \
  "http://127.0.0.1:9000/api/swarm/missions/<mission_id>/records?after=0"
# → 32 records for a two-clone churn mission, done: true
```

A 401 on `/api/swarm/missions` means the token was not signed with the secret
the server has, or the secret is empty.

## 2. The UI

The UI is a TanStack Start app. The browser never calls the API directly:
every call goes to the UI's own same-origin route `/api/public/backend-proxy`
(`ui/src/routes/api/public/backend-proxy.ts`), which forwards it to the backend
named by the server-side environment variable **`API_BASE`** (read by
`ui/src/lib/runtime-env.ts` at request time and sent to the browser as
`window.__ENV__`; there is no `VITE_*` variable for it). The default is
`http://127.0.0.1:8010`, so set it:

```bash
cd ui
npm ci                      # Node 22 (ui/tests/README.md); node_modules is not committed
API_BASE=http://127.0.0.1:9000 npm run dev     # UI on http://localhost:5173
```

**The token.** The UI normally takes its Bearer token from the Supabase
session, which needs a Supabase project to sign in to. For this demo the
client reads a pasted token first: `localStorage["avaloka-dev-token"]`
(`DEV_TOKEN_KEY` in `ui/src/lib/api/backendApi.ts`). It is read only when set
and the server still verifies it. The signed-in-user guard on every
`/_authenticated` page is skipped by the existing `admin-bypass` flag.

Open <http://localhost:5173/> in a browser, open the developer console, and
run (paste the token printed above in place of `<token>`):

```js
localStorage.setItem("admin-bypass", "1");
localStorage.setItem("avaloka-dev-token", "<token>");
```

Then open <http://localhost:5173/swarm>.

## 3. The demo

1. Dataset: **churn (bundled sample)**. Goal: anything, e.g. *What drives
   churn, and how confident can I be?* Kind: **analyze**. Clones: **2**
   (the control is bounded by the edition's cap, read from
   `/api/capabilities`). **Start.**
2. Watch: *Cloning myself into 2 copies…*, then *Sharded 1,200 rows into 2
   shards…*, then **Ava-00** and **Ava-01** appear with 600 rows and 7
   columns each; their six fireflies (data scout → sampling specialist →
   data engineer → analysis planner → validator → reporter) fill in with the
   human role, the output, the duration and the cost, the running one pulsing;
   each card settles on its verdict; the **Convergence** panel shows the
   merged verdict, the rule used and a contributions table; the last line is
   *Done in … ms* with the economics.
3. Ask for more clones than the cap (type 3: the control clamps to 2, so
   use the API or a plan above oss) to see the limit said in amber: *Swarm
   limited to 2 clones on the free open-source edition: you asked for 3, so 2
   will run.*
4. Kind **train** with target `churned` runs a training mission: each card
   then shows its model score, and the merge highlights the clone whose model
   it selected.
5. Tabs: **Clone documents** (one clone's full record), **RL episodes**
   (`reward` is null on purpose), **Telemetry (what would leave)**.
6. **Missions** at the bottom lists finished missions; **Open** replays one.

On the agent-trace page of an analysis, the **Swarm** tab links to
`/swarm?thread=<id>`, which offers *This analysis's dataset* as the source;
that path needs the chat stack (an uploaded dataset in a Redis-backed
session) and is not part of the laptop recipe.

## 4. In containers

The same, with two things to get right:

- The UI's proxy runs on the UI's *server* side, so `API_BASE` must be
  reachable from the UI container, not from the browser: with the API in
  another container, `API_BASE=http://host.docker.internal:9000` (Docker
  Desktop) or the API container's name on a shared network. Publish the UI's
  port (`-p 5173:5173`) and run `npm run dev -- --host 0.0.0.0`.
- Mint the token inside the API container (it has PyJWT and the secret).

```bash
# API
docker run --rm -p 9000:9000 -e AVALOKA_EDITION=oss -e AVALOKA_ALLOW_INSECURE_AUTH=1 \
  -e SUPABASE_JWT_SECRET='demo-only-change-me-0123456789abcdef' \
  <avaloka-api image> uvicorn app.api.server:app --host 0.0.0.0 --port 9000 --loop asyncio
# UI (from the repository root)
docker run --rm -p 5173:5173 -v "$PWD/ui:/app" -w /app -e NODE_ENV=development \
  -e API_BASE=http://host.docker.internal:9000 node:22 \
  sh -c 'npm ci && npm run dev -- --host 0.0.0.0'
```

## 5. What is assumed

- The API starts with no Redis and no Postgres. `/health` then reports
  `redis_connected: false`; the swarm store falls back to memory and says so
  in every mission document (`persistence: "memory"`). Routes that need a
  session (uploads, chat, `{"thread_id": …}` sources) are outside this demo.
- `AVALOKA_EDITION=oss` resolves the swarm cap to 2 clones per mission
  (`avaloka/swarm_limits.py`, after the 1.7 change from 3). A different value
  here only changes the cap and the limit message.
- The UI dev server's own Supabase client points at the local default
  (`http://localhost:30091`) when `SUPABASE_URL` is unset. Nothing on `/swarm`
  signs in, and the pasted token is used before the session is consulted;
  console errors from the sidebar's plan lookup are expected and harmless.
- These steps were written against the code; the UI was built and its tests
  run in a Node 22 container, but a browser was not driven on this machine.
