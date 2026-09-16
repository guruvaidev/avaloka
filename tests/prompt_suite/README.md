# Avaloka prompt-suite runner

Drives Avaloka end-to-end through its real API (upload/register → chat → training
→ inference → transfer → schedule), printing each prompt and the live response,
and logging structured results to `results/*.jsonl`. Three suites:

| Suite | Items | Origin |
|---|---|---|
| `v4` | 86 | Functionality coverage — every capability triggered (F1…F13) |
| `v2` | 96 | Evidence-based, per cloud connection |
| `v1` | 44 | Legacy Kaggle arcs (dead-file items pruned) |
| `heavy` | 19 | Entire-dataset / wide-file items — **opt-in, k8s only** (see below) |

The main suites (`v4`/`v2`/`v1`) contain **no resource-bomb items** — they run
safely on any deployment with no special flags. The `heavy` suite holds the
entire-dataset and ~1,900-column items that materialize 200 MB–1.9 GB in-process;
on a bare (unsupervised) API those OOM the whole service, so run `heavy` **only**
against a memory-limited, supervised deploy (k8s with resource limits + liveness
probes, which contain the OOM to a pod restart):

```bash
python3 run_suite.py --suite heavy --list
python3 run_suite.py --suite heavy --batch v4:F11   # against a k8s deploy only
```

## Setup

The runner is fully environment-driven — nothing about the target is hardcoded.
Create `env.sh` (gitignored) next to `run_suite.py`:

```bash
export AVALOKA_API_URL=http://localhost:8080          # where the Avaloka API listens
export AVALOKA_AUTH_URL=https://<project>.supabase.co  # Supabase this deploy trusts
export AVALOKA_ANON_KEY=<anon-or-service-role key>     # apikey for the auth call
export AVALOKA_TEST_EMAIL=<a confirmed account there>
export AVALOKA_TEST_PASSWORD=<its password>
export AVALOKA_CONNS=/path/to/conns.json               # {"instacart-mba":"<conn-id>", ...}
```

`conns.json` maps a connection **folder name** to its cloud-connection id (the
suite consumes existing connections; it does not create them). Local upload
files (`housing.csv`, `train.csv`) are read from `$HOME/v3assets` — adjust
`DATA_DIR` in `run_suite.py` if they live elsewhere.

## Run

```bash
source ./env.sh
python3 run_suite.py --suite v4 --list             # show batches
python3 run_suite.py --suite v4 --batch F2          # one batch, live output
python3 run_suite.py --suite v4 --batch F3 --from TR-5   # resume mid-batch
python3 run_suite.py --suite v4 --batch F6 --only MX-1   # single item
python3 run_suite.py --suite v4 --batch F8 --skip-heavy  # exclude resource-bomb items
./run_all.sh                                         # everything, safe order
```

Flags: `--reset` (forget cached datasets/threads), `--from <id>`, `--only <id>`,
`--skip-heavy`.

## Verdict taxonomy

| Verdict | Meaning |
|---|---|
| `OK`   | HTTP 200, artifact returned |
| `FAIL` | HTTP ≥400 with a body — a genuine product/response failure |
| `INFRA` | Connection-level failure (`http=0`): the API was unreachable/wedged, **not** a product result. The runner health-gates before each item and waits for recovery; 3 in a row aborts the batch instead of emitting fake fails |
| `SKIP` / `SKIP-HEAVY` | Not run: manual/k8s item, no prerequisite (e.g. no trained model), or a `--skip-heavy` resource-bomb item |

## Design notes / caveats

- **MTA rule** — a trained model is never *used* via chat. `IN-*` items call the
  inference API directly (`configure-inference-service` then `POST /inference`
  with a bare row array); `MX-5/6` read the registry via `/api/models`. Chat
  genuinely cannot see the registry, so those must be API calls.
- **Inference serving needs Ray Serve** — `configure-inference-service` spins up
  `avaloka-inference-serve-svc:8000`, which exists only in a k8s deployment with
  Ray Serve. On a bare (uvicorn) deploy the runner reports
  `inference_service_unavailable` honestly rather than faking a prediction.
- **Heavy items** — entire-dataset asks on 200 MB–1.9 GB files can OOM an
  unsupervised API (unbounded in-process materialization). `--skip-heavy`
  excludes them by design; run them only against a memory-limited/supervised
  deploy (k8s probes, or a keeper loop).
- **GROQ keys required for training** — the MTA agent reads
  `GROQ_API_KEY_PLANNING_AGENT` at import time; it must be in the API process's
  environment (export before launching uvicorn / set in the pod), not just in a
  late-loaded `.env`, or training returns "model training agent is unavailable".
- **`v1` is legacy** — written against public Kaggle schemas; dead-file items
  (different datasets, alias connections) are pruned. Prefer `v4`/`v2`.
- **Environment specifics** — `ING-8` needs a `talkingdata` connection (the
  0-byte negative probe); if absent it fails at registration. `/buckets/list`
  (`ING-7`) enforces connection ownership; other endpoints don't — a noted
  authorization inconsistency, not a runner bug.
