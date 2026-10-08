# Installing Avaloka

Open source and commercial editions install through the **same path**. The only
difference is one extra package. There is no separate download, no build
variant, and no flag to flip.

```bash
git clone https://github.com/guruvaidev/avaloka.git
cd avaloka
./scripts/install.sh                    # open source
source .venv/bin/activate
```

Verify at any time:

```bash
./scripts/install.sh --check
```

---

## Requirements

| | Minimum | Notes |
| --- | --- | --- |
| Python | **3.11 – 3.12** | `pyproject.toml` declares `requires-python = ">=3.11,<3.13"`; CI runs 3.11 and 3.12 |
| Graphviz | any recent | **required** — a system package, not a pip one; see below |
| OS | Linux, macOS | Windows via WSL2 |
| Memory | 8 GB | 16 GB+ for local Ray |
| Disk | 5 GB | plus your data |

> **Graphviz is a system package and the installer does not install it.** The
> `graphviz` entry in `requirements.txt` is only a Python binding to the `dot`
> binary. Without `dot` on your PATH the planner-graph path returns
> `error: Graphviz system executables not found`
> (`app/agents/planner_graph_agent.py:186`) instead of a rendered plan, and the
> planner-graph tests skip themselves rather than failing — so a green test run
> does not prove it is present. CI installs it explicitly
> (`.github/workflows/ci.yml`, "System packages"). Install it yourself:
>
> ```bash
> brew install graphviz           # macOS
> sudo apt-get install -y graphviz  # Debian/Ubuntu
> dot -V                          # confirm
> ```
>
> `scripts/doctor.py` does **not** check for it. Run `dot -V` yourself.

> **macOS: check which Python you have.** `python` still resolves to the system
> Python 2.7 on many Macs, and the resulting failures are confusing. The
> installer detects this and stops with a clear message. Run `python3 -V` to
> confirm, or point the installer at a specific interpreter:
>
> ```bash
> PYTHON=/opt/homebrew/bin/python3.11 ./scripts/install.sh
> ```

> **3.13 and newer are not supported.** `pyproject.toml` caps the interpreter
> at `<3.13`, and `scripts/install.sh` refuses anything below 3.11
> (`MIN_PY_MINOR=11`, `scripts/install.sh:52`). The floor is a real
> requirement, not caution: `app/services/session_service.py:170` uses
> `asyncio.timeout`, which is 3.11+. The installer searches for `python3.12`,
> `python3.11`, then `python3` in that order; let it choose unless you have a
> reason not to.
>
> Note that the installer checks only the lower bound — it will proceed on a
> 3.13 interpreter if that is what `python3` resolves to, and the failure then
> arrives later as a dependency with no matching wheel. Pass `PYTHON=` to pin
> the interpreter if your `python3` is newer than 3.12.

Optional, depending on what you run:

- **Docker** — Postgres, Redis and Milvus, each in its own compose file
  (see [Running it](#running-it); the default `docker-compose.yml` is Milvus,
  with SeaweedFS as its object store).
- **kind** or **minikube** — a local Kubernetes cluster
- **An LLM API key** — you bring your own; see [Configuration](#configuration)

---

## Open source

```bash
./scripts/install.sh
```

This creates `.venv`, installs dependencies, and prints an edition report.

Options:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--venv PATH` | `.venv` | virtualenv location |
| `--no-venv` | off | install into the active environment |
| `--check` | — | verify an existing install and exit |
| `--edition` | `oss` | `oss`, `professional` or `enterprise` |
| `--license-key` | `$AVALOKA_LICENSE_KEY` | licence token for a commercial edition |
| `--index-url` | `https://pypi.avaloka.ai/simple` | private index for the commercial package |
| `PYTHON=...` | autodetect | choose the interpreter |

That is the complete set the argument parser accepts
(`scripts/install.sh:60-71`). `--help` additionally advertises `--doctor` and
`--with-build-tools`, but neither is implemented — the parser rejects both with
`unknown option`. Use `python scripts/doctor.py` for the environment check; see
[You do not need a compiler](#you-do-not-need-a-compiler) for the toolchain
question.

### What you get, and what you owe

Self-hosted open source is **not a crippled build**. You get distributed Ray
execution, the scheduler, and batch jobs, because it is your cluster and your
bill. Nothing here is rate-limited by us.

In exchange, you operate it: you provision the Ray cluster, supply cloud
credentials and LLM keys, and carry the risk. Generated code is yours to
inspect, run, and take at your own discretion — review it as you would any code
before running it against production data.

**What it is sized for.** A laptop, or a small Kubernetes cluster you run
yourself. That is what is tested. Large multi-node clusters, autoscaling node
pools, scheduled cloud workloads and multi-tenant operation are not — Avaloka
will not provision a managed GKE, EKS or AKS cluster for you, and an
open-source build refuses the attempt rather than billing your cloud account.
It will happily deploy into a cloud cluster **you** provisioned.

Support is community and best-effort, and you run it at your own risk. If you
need scale, provisioned infrastructure, or an SLA, that is what the commercial
editions are for: [avaloka.ai](https://avaloka.ai), or email
**[support@avaloka.ai](mailto:support@avaloka.ai)**.

---

## Commercial editions

```bash
./scripts/install.sh --edition professional --license-key "$AVALOKA_LICENSE_KEY"
```

`--edition` accepts `professional` or `enterprise`. Licence keys come from the
Avaloka team.

This runs the identical open-source install, then adds one package —
`avaloka-commercial`, pulled from Avaloka's private index and authenticated
with your licence-scoped token. That package carries the code for the
commercial capabilities. In an open-source install it is simply absent.

The token is stored at `~/.avaloka/license` with mode `600`. Override the
location with `AVALOKA_HOME`.

## Running Avaloka completely offline

Avaloka runs with **no internet connection at all** — no model API, no model
download, no telemetry, no licence call. This is the configuration to use on an
air-gapped host, and it is worth running once even on a connected machine,
because anything that quietly depends on the network shows up immediately.

**What makes it possible**

| Dependency | How it is satisfied offline |
| --- | --- |
| Language model | A local model server (Ollama or vLLM) — `INFERENCE_PROVIDER=local` |
| Embedding model | **Baked into the API image at build time** (`all-MiniLM-L6-v2`, ~88 MB, under `HF_HOME=/opt/hf`) |
| Object storage | SeaweedFS, deployed by the chart; its images must be mirrored for a host with no registry access — see [Object storage](#object-storage) |
| Vector / memory tiers | Redis, Chroma and Milvus, all deployed by the chart |
| Auth | Self-hosted Supabase, deployed by the chart |
| Licence check (commercial) | Offline Ed25519 signature against an embedded public key |

**Deploy it**

```bash
helm upgrade --install avaloka deploy/helm/avaloka \
  --set inference.provider=local \
  --set memory.offlineEmbeddings=true \
  --set milvus.enabled=true
```

The object store needs no flag: SeaweedFS is the chart default. Do **not** add
`--set minio.enabled=true` — MinIO is now only a rollback path, and the chart
refuses to render with both stores enabled. On a host that cannot reach Docker
Hub, mirror `chrislusf/seaweedfs` and `amazon/aws-cli` into a registry you
control and override `seaweedfs.image.repository` and
`seaweedfs.awsCliImage.repository`, or point the stack at an object store you
already run. An air-gapped install has to do this for every image anyway.

Then bring up a local model server — `deploy/inference/ollama-cpu.yaml` needs no
GPU and is the path to start with; `deploy/inference/vllm-gpu.yaml` is the GPU
equivalent.

**Verify it is genuinely offline**

Do not take the configuration's word for it. The embedding path is the one that
historically reached out, so test it with the network removed:

```bash
docker run --rm --network none --entrypoint sh avaloka-api:latest -c \
  'cd /app && HF_HUB_OFFLINE=1 PYTHONPATH=/app python -c \
   "from app.services.embedding_utils import embed_text; \
    print(len(embed_text(chr(120)*8, target_dim=384)))"'
```

A printed `384` means the model is resolved from the image. An error means the
image was built without the model baked in, and the memory plane will stall at
query time rather than fail loudly.

**What you give up offline**

Only the things that are inherently remote: hosted model providers
(Groq / OpenRouter / Bedrock / Vertex / Azure), cloud object stores, and managed
cloud Kubernetes. Every agent, the evidence layer, the memory plane and the full
analysis path run locally.

---

### Prefer not to run the infrastructure yourself?

**[avaloka.ai](https://avaloka.ai)** hosts the **Professional** and
**Enterprise** editions with auto-scaling compute, so you do not stand up
Kubernetes, Ray, object storage or a model provider yourself. Enterprise adds
shared team workspaces, the cloud scheduler, and connections to your own cloud
clusters — see [EDITIONS.md](EDITIONS.md) for what each edition includes.

The open-source edition is not a trial of those: it is the same engine, run on
your own machine, with no call home.

---

### Air-gapped and VPC deployments

Licence verification is **fully offline** — an Ed25519 signature checked against
an embedded public key. Nothing contacts Avaloka at runtime, which is what makes
Professional and Enterprise viable inside a customer VPC.

For an air-gapped host, mirror the `avaloka-commercial` wheel to your internal
index and point the installer at it:

```bash
./scripts/install.sh --edition enterprise \
  --license-key "$AVALOKA_LICENSE_KEY" \
  --index-url https://pypi.internal.example.com/simple
```

If a licence is missing, expired, or invalid, Avaloka **degrades to the
open-source capability set rather than failing to start.**

---

## Installing, and what to do when it goes wrong

```bash
./scripts/install.sh
```

That is the whole install. It creates `.venv`, installs dependencies **as
wheels**, and finishes by running an environment check that tells you whether
the result will actually work.

### You do not need a compiler

Avaloka installs from prebuilt wheels. If you have read that `cryptography`
needs a Rust toolchain, that is true only of a **source** build, and the
installer avoids one.

The mechanism is worth knowing, because it bites outside Avaloka too. `pip`
picks the *newest* version of a package. When that version has no wheel for
your platform, pip quietly falls back to the source distribution — and for
`cryptography` that means a Rust build needing `rustc >= 1.83`, which fails
with a compiler error many screens from its cause. Installing with
`--prefer-binary` picks the newest version that actually ships a wheel, so no
toolchain is involved. `scripts/install.sh` always passes it.

If you *want* a toolchain — to build something from source deliberately —
install Rust yourself from [rustup.rs](https://rustup.rs) before running the
installer. `scripts/install.sh` contains a `--with-build-tools` branch that
would do this for you, but no flag reaches it: the option is absent from the
argument parser, so passing it exits with `unknown option`.

### Check the environment any time

```bash
python scripts/doctor.py
```

It reports the Python version, the CPU architecture, whether the core packages
import, whether the `avaloka` command resolves, whether anything would need a
source build, and whether **numpy and torch agree on their ABI** — a mismatch
that announces itself only as a warning and then breaks something unrelated
later. Illustrative output:

```
Avaloka environment check

  ✓ Python 3.11.10 (x86_64, Darwin)
  ✓ core packages import cleanly (8 checked)
  ✓ avaloka 1.0.0
  ✓ cryptography 48.0.1 (wheel — no Rust toolchain needed)
  ✓ numpy 1.26.4 and torch 2.2.2 agree on their ABI

Everything checks out.
```

Every problem it reports comes with the command that fixes it. It does not
check for Graphviz — run `dot -V` separately.

### Run the tests yourself

The same suite CI runs, on your machine, needing no cluster, cloud account,
API key or network:

```bash
./scripts/ci.sh              # the full gate — what must pass before a release
./scripts/ci.sh --fast       # hermetic tests only, no benchmark (~1 minute)
./scripts/ci.sh --list       # what the stages are
```

Five stages: the environment check, the hermetic test suite, the data-science
capability suite, the CLI robustness suite, and the benchmark. The exit code is
the number of failed stages, so it works as a gate in your own CI, and a stage
that is skipped says so rather than passing silently.

---

## Configuration

Copy the template and fill in what you need:

```bash
cp .env.example .env
```

| Variable | Purpose |
| --- | --- |
| `AVALOKA_DEPLOYMENT` | `self_hosted` (default) or `avaloka_hosted` |
| `AVALOKA_EDITION` | `oss`, `professional`, `enterprise` — ignored when a licence is present |
| `AVALOKA_LICENSE_KEY` | licence token; overrides the stored file |
| `AVALOKA_HOME` | config directory (default `~/.avaloka`) |
| `AVALOKA_LICENSE_PUBLIC_KEY` | override the verification key (staging only) |

Unknown values fail closed: an unrecognised deployment resolves to
`self_hosted`, and an unrecognised edition resolves to `oss`.

Never commit `.env`. It is gitignored; keep it that way.

---

## Running it

There are two supported ways to run Avaloka. Pick one; mixing them is the most
common way to end up with an API that cannot reach its own database.

### A. Kubernetes (recommended, and what is actually tested)

This is the path the test suite exercises and the one that brings up every
dependency — Postgres, Redis, SeaweedFS, Chroma, Supabase and the LangGraph server
— in one command.

```bash
kind create cluster --name avaloka          # or use an existing cluster

# The chart's image defaults are already registry paths that exist:
#   ghcr.io/guruvaidev/avaloka-api:main      (values.yaml:59,64)
#   ghcr.io/guruvaidev/avaloka-ui:main       (values.yaml:457-458)
#   ghcr.io/guruvaidev/avaloka-ray:main      (values.yaml:498-499)
#   ghcr.io/guruvaidev/avaloka-functions:main (values.yaml:425)
# All four `:main` tags are present on GHCR and pull anonymously — verified
# against the registry. So no --set is needed for images, and no build is
# needed. If you are offline or on a commit whose content tag is not
# published, build and side-load instead:
#   make -C deploy images-status               # what would need building?
#   make -C deploy images-pull PROVIDER=local  # pull + kind-load what exists
#   make -C deploy images PROVIDER=local       # build + kind-load all four

helm upgrade --install avaloka deploy/helm/avaloka \
  --set-string secrets.groqPlanningKey="$GROQ_API_KEY_PLANNING_AGENT" \
  --set-string secrets.groqCodingKey="$GROQ_API_KEY_CODING_AGENT"

kubectl port-forward svc/avaloka 9000:9000
curl localhost:9000/health
```

Two things about that command worth knowing before you run it.

**The chart pins the provider to Groq.** `inference.provider` defaults to
`"groq"` (`deploy/helm/avaloka/values.yaml`) and the ConfigMap always emits
`INFERENCE_PROVIDER` (`deploy/helm/avaloka/templates/configmap.yaml:31`), so a
Helm deployment needs Groq keys. This is
*different* from a local install, where `INFERENCE_PROVIDER` is unset and
`app/core/inference.py` resolves the default to OpenRouter. To use OpenRouter
in-cluster, set both:

```bash
  --set inference.provider=openrouter \
  --set-string secrets.openrouterKey="$OPENROUTER_API_KEY"
```

The object store is SeaweedFS and is on by default; its two images are pulled
from Docker Hub. See [Object storage](#object-storage).

If the `avaloka`, `avaloka-webui` or `avaloka-supabase-functions` pods sit in
`ImagePullBackOff`, run `make -C deploy images-status` to see which image is
missing, then build it as above. The API image is ~4 GB (CPU-only torch plus
the baked embedding model) and takes a while to build cold; the others are
small.

`/health` reports `graph_ready`, `redis_connected` and whether the LangGraph
server upstream is reachable — check it before anything else.

> **Configure model keys through Helm values, never `kubectl patch`.** The
> secret is re-rendered from chart values on every `helm upgrade`, so a patched
> key is silently wiped and the agent quietly falls back to canned replies.
> Use `--set-string secrets.groqPlanningKey=…` / `secrets.groqCodingKey=…` for
> the planner and coder; `secrets.groqApiKey` is a separate, third key read
> only by `app/agents/visualization_agent.py`. All are listed in `values.yaml`
> under `secrets:`.

### B. Local processes

Useful for iterating on the API itself. **Each dependency has its own compose
file** — the default `docker-compose.yml` is the Milvus stack (etcd, SeaweedFS,
Milvus, Attu), *not* Postgres and Redis:

```bash
docker compose -f deploy/compose/docker-compose.postgres.yml up -d   # Postgres
docker compose -f deploy/compose/docker-compose.redis.yml up -d      # Redis
docker compose up -d                                  # Milvus stack (optional;
                                                      #   needed for episodic memory)

uvicorn app.api.server:app --reload --port 9000       # API
cd ui && npm install && npm run dev                   # UI on :5173
```

**A note on ports.** The container image serves on **9000**
(`deploy/docker/Dockerfile.api`), and the Helm service `avaloka` exposes it on
9000 as well (NodePort 30085). Uvicorn's own default is 8000, so running it
without `--port` gives you a different port from every other path — pass
`--port 9000` and everything lines up.

> One exception, and it catches people: under **kind** the NodePort route lands
> on host port **9010**, not 9000, because `deploy/clusters/kind-cluster.yaml:15-16`
> maps containerPort 30085 to hostPort 9010. `kubectl port-forward svc/avaloka
> 9000:9000` still gives you 9000. Both are correct — they are different routes
> to the same Service. See
> [deployment.md](deployment.md#2-quick-start-local-kind).

Kubernetes, Ray and cloud targets: see [deployment.md](deployment.md).

---

## The command line

`./scripts/install.sh` puts the `avaloka` command on your path. It is the
shortest route from a fresh install to a real answer, and it needs no server,
no cluster and no browser:

```bash
# These three run against datasets that ship in the repo, so they work on a
# fresh clone with no file of your own.
avaloka analyze app/sample_data/Global_YouTube_Statistics.csv \
        --goal "what separates the channels that grow from the ones that stall?"
avaloka train   app/sample_data/healthcare_dataset.csv --target "Test Results"
avaloka chat    app/sample_data/salaries.csv   # work it out conversationally
```

`avaloka chat` reads the file, says what it notices, and offers numbered next
steps you answer with a number. Reads CSV, TSV, Parquet and Excel, including
spreadsheets whose header sits below a title block.

Full reference, including what every failure message means:
**[README_CLI.md](../README_CLI.md)**.

> Not to be confused with `python -m app.interfaces.cli.main`, a separate
> planning-only interface documented in [cli.md](cli.md). The `avaloka`
> command is the one that runs analyses.

---

## First run: what to expect, and what goes wrong

Every item below was hit while bringing this stack up on a clean cluster. They
are listed because each one is silent — the symptom never names the cause.

### Upload a file, ask a question

```bash
curl -F "file=@app/sample_data/salaries.csv" -H "Authorization: Bearer $TOKEN" \
     localhost:9000/api/upload
```

Supported: **CSV, TSV, JSON, XML, Parquet, XLS/XLSX**. Spreadsheets whose
header is not on row 1 are handled — the loader finds the header under title
and subtitle rows rather than reading the first row and producing a frame of
mostly-NaN.

Ownership is enforced from the session, so dataset-scoped routes need the
`X-Avaloka-Session` header the upload returns. Without it they answer `400`,
which is correct behaviour and not a bug.

### "It answers, but the answers are canned"

The agent falls back to deterministic replies when it cannot build a model.
That fallback is intentional — an install with no key still profiles data and
answers schema questions — but it also fires when a key IS set for the *wrong*
provider, which is easiest to hit by pinning one explicitly:

```
INFERENCE_PROVIDER=groq   GROQ_API_KEY=""   OPENROUTER_API_KEY=sk-or-…
```

Leaving `INFERENCE_PROVIDER` unset avoids this: `app/core/inference.py` then
picks a provider that actually has a key, preferring OpenRouter and falling back
to Groq, and warns when it has to choose. Pinning the variable disables that
resolution. Note the chart does pin it — see
[A. Kubernetes](#a-kubernetes-recommended-and-what-is-actually-tested).

Avaloka names this explicitly rather than degrading quietly. If replies look
generic, check the API logs for `no usable reply model`
(`app/agents/avaloka_agent/agent.py:111`).

### "Memory never remembers anything"

Episodic recall needs a vector backend, and it is **off by default** because it
adds an etcd plus a ~2 GiB pod:

```bash
helm upgrade --install avaloka deploy/helm/avaloka --set milvus.enabled=true
```

With it off, runs report `memory_context_unavailable: true` rather than
pretending they recalled nothing. Chroma keeps a persistent volume, so tier-2
context survives pod restarts.

### Object storage

**The in-cluster object store is SeaweedFS, on by default**
(`seaweedfs.enabled: true`). It serves S3 on port 8333 as `avaloka-seaweedfs`,
holds uploaded datasets and MLflow artifacts on a 20Gi volume, and gets its
`avaloka` bucket from a post-install hook. Nothing in the application changes
with the store: every client reads the endpoint and credentials the chart
publishes.

It replaced MinIO because MinIO's images stopped being pullable: anonymous
manifest requests to `quay.io/minio/minio` and `quay.io/minio/mc` return `401`,
as do the Docker Hub equivalents. **MinIO is retained for one release as a
rollback path only**, off by default. To use it you must turn SeaweedFS off in
the same command — the chart refuses to render with both enabled — and supply
images from a registry of your own:

```bash
--set seaweedfs.enabled=false --set minio.enabled=true
```

Data is not copied between the two stores.

Things to know before you rely on it:

- **Both images come from Docker Hub**, pinned by digest:
  `chrislusf/seaweedfs:4.48` and `amazon/aws-cli:2.32.6` (the bucket hook). A
  digest protects against a retag, not against the repository going away, which
  is what happened to MinIO. For anything you depend on, mirror both into a
  registry you control and override `seaweedfs.image.repository` and
  `seaweedfs.awsCliImage.repository`; keep the digests, they still verify.
- **The default credentials are development values.** Override
  `seaweedfs.auth.accessKey` / `seaweedfs.auth.secretKey` for anything beyond a
  laptop.
- **A NetworkPolicy restricts the store to its S3 port and to the workloads that
  use it.** It only protects you on a cluster whose CNI enforces NetworkPolicy.
  To let another workload in, add a peer under `seaweedfs.extraIngressFrom`.
- **Or point at object storage you already run**: set `seaweedfs.enabled=false`,
  `config.storageBackend` and `mlflow.artifactsDestination`, as the cloud
  overlays in `deploy/helm/avaloka/values/` do.

`docker-compose.yml` runs the same SeaweedFS image as the object store for the
Milvus stack.

### Air-gapped

The sentence-transformers embedding model is baked into the API image at build
time, so nothing is fetched at query time. Verify it rather than trust it:

```bash
docker run --rm --network none --entrypoint sh avaloka-api:latest -c \
  'cd /app && HF_HUB_OFFLINE=1 PYTHONPATH=/app python -c \
   "from app.services.embedding_utils import embed_text; print(len(embed_text(chr(120)*8, target_dim=384)))"'
```

A printed `384` means the model resolved from the image.

---

## Verifying and troubleshooting

`./scripts/install.sh --check` prints the resolved deployment, edition, licence
status, and every capability with the reason it is on or off. Paste that output
into any support request — it never includes your licence token.

**"Python 3.11+ required, found 2.7"** — see the macOS note above.

**A capability is off and I expected it on.** Run the check. If the reason says
*not in this build*, the code is not installed — that is a commercial
capability. If it says *hosted plan limit*, self-host or upgrade.

**"could not install avaloka-commercial"** — the open-source install already
completed and is usable. Check the licence key and network reachability to the
index, then re-run with `--edition`.

**Licence reported as rejected.** The check prints the specific reason —
expired, malformed, or signature failure. An expired licence degrades to open
source rather than blocking startup.

---

## Related

- **[README_CLI.md](../README_CLI.md)** — the `avaloka` command, in full
- [EDITIONS.md](EDITIONS.md) — what each edition includes and why
- [architecture.md](architecture.md) — how the pieces fit
- [deployment.md](deployment.md) — Kubernetes and Ray
- [testing.md](testing.md) — running the test suites
