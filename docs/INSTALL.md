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

- **Docker** — the web quickstart (`scripts/quickstart.sh`, see
  [Quickstart](#quickstart-the-web-app-on-one-machine)); or, for local
  processes, Postgres, Redis and Milvus, each in its own compose file (the
  default `docker-compose.yml` is Milvus, with SeaweedFS as its object store).
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
| Embedding models | **Baked into the API image at build time** by `scripts/prefetch_models.py`: `all-MiniLM-L6-v2` (memory plane), `mxbai-embed-large-v1` (~670 MB, the data-transfer agent's doc retrieval; `--build-arg AVALOKA_PREFETCH_DTA_MODEL=0` leaves it out) and chroma's ONNX model. The image sets `HF_HUB_OFFLINE=1`, so a missing model fails at once with a plain `ModelNotCached` error instead of a download attempt; `python scripts/prefetch_models.py --check` verifies them |
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

**Memory embeddings are local by default.** `EMBEDDING_PROVIDER` defaults to
`local` (Helm: `memory.embeddingProvider: local`), which embeds with the model
baked into the image. `EMBEDDING_PROVIDER=openai` is an explicit opt-in that
also requires `EMBEDDING_OPENAI_API_KEY`; it sends memory text — user queries,
learned hints, answers, and cached schema JSON including table and column names
and sample values — to OpenAI. `OPENAI_API_KEY`, the chat key, never turns it
on. Before 2.0 it did: the old `auto` default embedded with OpenAI whenever
`OPENAI_API_KEY` was set. `auto` is now deprecated and means `local`.

**What you give up offline**

Only the things that are inherently remote: hosted model providers
(Groq / OpenRouter / Bedrock / Vertex / Azure), cloud object stores, and managed
cloud Kubernetes. Every agent, the evidence layer, the memory plane and the full
analysis path run locally.

---

### Prefer not to stand up the infrastructure yourself?

The **Professional** and **Enterprise** editions
(**[avaloka.ai](https://avaloka.ai)**) run in your own cloud, on your billing,
and provision and autoscale the Kubernetes and Ray clusters there for you, so
you do not stand them up yourself. Enterprise adds
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

One env file drives every install: `.env.quickstart` for the quickstart
(scripts/quickstart.sh writes it), `.env` for local processes and any compose
stack (`env_file: .env`), and the same names as chart values for Helm. Avaloka
creates the secrets it can create for itself; you set only what identifies
*your* accounts — a model provider key, and your Supabase sign-in secret if you
use Supabase.

### Install with the right parameters

* **Quickstart (one machine):** `scripts/quickstart.sh up`. It writes
  `.env.quickstart` (mode 600, gitignored) with every secret it needs:
  `DB_ENCRYPTION_KEY`, `MCP_SERVICE_TOKEN`, `AVALOKA_PII_KEY`, the local
  sign-in (`AVALOKA_LOCAL_AUTH_SECRET`, `AVALOKA_LOCAL_AUTH_PASSWORD`) and
  `QS_POSTGRES_PASSWORD`. A later `up` keeps them; a published value (the old
  chart default `avaloka-local-dev-encryption-key`, a placeholder) is replaced
  with a warning — see "Upgrade consequence" below. Inference defaults to a local model; set
  `INFERENCE_PROVIDER` and that provider's key in `.env.quickstart` to use a
  hosted one.
* **Local processes:** `cp .env.example .env`, set the **you set** rows below,
  and run `./scripts/install.sh` (or `python -m app.core.provisioning --env-file
  .env`): it generates the **generated automatically** rows into `.env`. The
  first start of the API, the MCP server or the onboarding service on a host
  does the same if neither ran. Every later start reuses what `.env` holds.
  Nothing is printed but "generated and saved to .env".
* **Your own compose stack:** run the provisioning step once on the host and
  give every service the same `env_file`. Containers never generate their own
  secrets (each would get a different one); one that starts without a key
  refuses to start and says how to fix it.
* **Helm:** put the **you set** values in your values file (`secrets.*`) and
  leave the generated ones empty. The chart generates `DB_ENCRYPTION_KEY`,
  `MCP_SERVICE_TOKEN`, `EXECUTE_SERVICE_TOKEN`, `LANGGRAPH_SERVICE_TOKEN` and
  `AVALOKA_PII_KEY` into the release Secret on first
  install and keeps them on every upgrade (`lookup` of the existing Secret); a
  published value, given or left in an old Secret, is replaced and
  `helm install` prints a warning. The Supabase edge functions read the same
  key from the release Secret. With `existingSecret`, that Secret must carry
  them itself (`EXECUTE_SERVICE_TOKEN` only when `executeService.enabled`).

**Upgrade consequence.** When an existing install's `DB_ENCRYPTION_KEY` is a
published value (a 2.0 Helm install with bundled Supabase used the chart's
`avaloka-local-dev-encryption-key`), the upgrade replaces it, and everything
sealed under the old key can no longer be opened: saved database connections,
cloud connection keys, MCP and GitHub tokens, and passwords typed into chat.
Register those connections again (and re-enter those tokens) after the
upgrade. They are not repaired automatically: the old key protected nothing,
so nothing sealed under it is carried forward. A replaced `MCP_SERVICE_TOKEN`
costs nothing; a replaced `AVALOKA_PII_KEY` means models trained on
pseudonymised columns should be retrained. Every replacement warning
(installer, service start, quickstart, `helm install`) says this.

**Generated secrets** (`app/core/provisioning.py` `GENERATED_SECRETS`):
`DB_ENCRYPTION_KEY` (AES-GCM key for every sealed credential; the onboarding
user assertion and the MCP admin signature are derived from it),
`MCP_SERVICE_TOKEN` (API ↔ MCP/onboarding service identity), `EXECUTE_SERVICE_TOKEN`
(graph pods ↔ `/execute` service secret), `LANGGRAPH_SERVICE_TOKEN` (API ↔ LangGraph
server secret), `AVALOKA_PII_KEY` (PII pseudonymisation HMAC key). Each is
`secrets.token_hex(32)` locally, `randAlphaNum 48` in the chart. **Back up `DB_ENCRYPTION_KEY`:** changing or
losing it makes every sealed credential unreadable.

A service refuses to start only when, after all of this, it still has no
usable `DB_ENCRYPTION_KEY` — `AVALOKA_AUTO_PROVISION=0`, a read-only `.env`,
or a container (Docker or Kubernetes) started without one. The error says how
to fix it.

### Every parameter

Kept in step with `.env.example` (`tests/test_install_configuration.py` fails
if either has a name the other lacks). Commented-out names in `.env.example`
are optional.

| Parameter | Purpose | Default | Who provides it | Where it is kept |
| --- | --- | --- | --- | --- |
| `PYTHONPATH` | module path for local runs | `.` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AVALOKA_DEPLOYMENT` | `self_hosted` or `avaloka_hosted` | `self_hosted` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AVALOKA_EDITION` | edition when no licence is present | `oss` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AVALOKA_LICENSE_KEY` | licence token (commercial editions) | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `AVALOKA_HOME` | config directory | `~/.avaloka` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AVALOKA_LICENSE_PUBLIC_KEY` | licence verification key override (staging only) | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `AVALOKA_ENV_FILE` | the file generated secrets are read from and saved to | `.env` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AVALOKA_AUTO_PROVISION` | `0` never generate secrets at service start; `1` also inside a container (never in Kubernetes); unset: only on a host | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `GROQ_API_KEY_PLANNING_AGENT` | planner model key (Groq) | — | **you set** (or pick another provider; the quickstart defaults to a local model) | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `GROQ_API_KEY_CODING_AGENT` | coder model key (Groq) | — | **you set** (or pick another provider; the quickstart defaults to a local model) | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `OPENROUTER_API_KEY` | the default model provider's key (OpenRouter); with `INFERENCE_PROVIDER=groq` it arms the backup | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `AVALOKA_OPENROUTER_ONLY` | OpenRouter provider allowlist for the default model (`none` = any provider) | `deepinfra/bf16,together,crusoe,parasail` | optional — you set it to change the routing | `.env` / `.env.quickstart` · Helm: chart env |
| `AVALOKA_OPENROUTER_ALLOW_FALLBACKS` | let OpenRouter route outside the allowlist | `false` | optional — you set it to change the routing | `.env` / `.env.quickstart` · Helm: chart env |
| `AVALOKA_OPENROUTER_DATA_COLLECTION` | OpenRouter provider data-collection policy | `deny` | optional — you set it to change the routing | `.env` / `.env.quickstart` · Helm: chart env |
| `AVALOKA_OPENROUTER_ZDR` | require zero-data-retention endpoints | `true` | optional — you set it to change the routing | `.env` / `.env.quickstart` · Helm: chart env |
| `AVALOKA_OPENROUTER_SORT` | provider ordering within the allowlist | `price` | optional — you set it to change the routing | `.env` / `.env.quickstart` · Helm: chart env |
| `AVALOKA_OPENROUTER_FALLBACK_MODEL` | second tier on OpenRouter (`none` = no fallback tier) | `openai/gpt-6-luna` | optional — you set it to change the routing | `.env` / `.env.quickstart` · Helm: chart env |
| `AVALOKA_OPENROUTER_FALLBACK_ONLY` | provider allowlist for that fallback tier | `azure` | optional — you set it to change the routing | `.env` / `.env.quickstart` · Helm: chart env |
| `AVALOKA_GROQ_FALLBACK_MODEL` | last tier, on Groq; needs a Groq key (`none` = off) | `openai/gpt-oss-120b` | optional — you set it to change the routing | `.env` / `.env.quickstart` · Helm: chart env |
| `OPENROUTER_BASE_URL` | OpenRouter endpoint | `https://openrouter.ai/api/v1` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AVALOKA_DISABLE_MODEL_FALLBACK` | `1` turns the backup provider off | `0` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AVALOKA_REPLAN_MAX_ATTEMPTS` | re-runs of a failed local analysis per turn, on repaired data or fixed code (`0` = off; `5` = ceiling) | `3` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AVALOKA_REPLAN_MEASURED_CHOICE` | `median` or `leave_out`: the up-front answer to the question about missing values in the measured column, for runs no one can answer | unset (the run asks) | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_PROVIDER` | model provider for every role | `openrouter` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_PROVIDER_PLANNING` | per-role override (planner) | `groq` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_PROVIDER_CODING` | per-role override (coder) | `local` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_LOCAL_BASE_URL` | local (Ollama/vLLM) endpoint | `http://avaloka-local-llm:11434/v1` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_LOCAL_MODEL_LARGE` | local large model id | `qwen3:4b-instruct-2507-q4_K_M` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_LOCAL_MODEL_SMALL` | local small model id | `qwen3:4b-instruct-2507-q4_K_M` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_LOCAL_API_KEY` | local server key (usually ignored) | `sk-local` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `INFERENCE_OPENROUTER_MODEL_LARGE` | OpenRouter large model | `qwen/qwen-2.5-72b-instruct` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_OPENROUTER_MODEL_SMALL` | OpenRouter small model | `qwen/qwen-2.5-7b-instruct` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `OPENAI_API_KEY` | OpenAI provider key | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `INFERENCE_OPENAI_BASE_URL` | OpenAI endpoint | `https://api.openai.com/v1` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_OPENAI_MODEL_LARGE` | OpenAI large model | `gpt-4o` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_OPENAI_MODEL_SMALL` | OpenAI small model | `gpt-4o-mini` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AWS_REGION` | AWS region (Bedrock, AWS infra) | `us-east-1` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_BEDROCK_MODEL_LARGE` | Bedrock large model | `meta.llama3-1-70b-instruct-v1:0` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_BEDROCK_MODEL_SMALL` | Bedrock small model | `meta.llama3-1-8b-instruct-v1:0` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `GOOGLE_CLOUD_PROJECT` | GCP project (Vertex, GCP infra) | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `GOOGLE_CLOUD_LOCATION` | Vertex location | `us-central1` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_VERTEX_MODEL_LARGE` | Vertex large model | `gemini-1.5-pro` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_VERTEX_MODEL_SMALL` | Vertex small model | `gemini-1.5-flash` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | `https://<resource>.openai.azure.com` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version | `2024-10-21` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AZURE_OPENAI_DEPLOYMENT_LARGE` | Azure large deployment | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AZURE_OPENAI_DEPLOYMENT_SMALL` | Azure small deployment | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `EMBEDDING_PROVIDER` | memory embeddings (`local` stays on the machine) | `local` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `EMBEDDING_OPENAI_API_KEY` | key when embeddings use OpenAI | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `REDIS_URL` | sessions/cache Redis | `redis://127.0.0.1:6379` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `REDIS_CLUSTER` | `1` for a Redis cluster | `0` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `CELERY_REDIS_URL` | scheduler broker | `redis://localhost:6380/0` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `CELERY_RESULT_EXPIRES` | seconds a finished task result stays in Redis; a miss is read from Postgres. `0` never expires | `604800` (7 days) | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `celery.resultExpires` |
| `CELERY_RESULT_RETENTION_DAYS` | days task results are kept in Postgres, purged daily. 1–365; up to **1095 on Enterprise only**. Out of range, or above 365 without Enterprise, is refused (the purge task fails with a configuration error; it never clamps) | `365` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `celery.resultRetentionDays` |
| `CACHE_DEGRADE_SECONDS` | Redis circuit-breaker window | `2` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `JWT_SECRET` | not read by 1.6 (the API verifies with `SUPABASE_JWT_SECRET`); kept for older docs | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `COOKIE_SECURE` | `true` behind HTTPS | `false` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `LANGGRAPH_API_URL` | LangGraph API endpoint (localhost by default; set explicitly in deployments) | `http://localhost:2024/` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `LANGGRAPH_SERVICE_TOKEN` | shared API ↔ LangGraph server secret; the server answers 401 without it | — (generated) | **generated automatically** | `.env` (local, compose) · release Secret (Helm, kept across upgrades) |
| `UPSTREAM_TIMEOUT` | upstream call timeout (s) | `15.0` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `LOG_LEVEL` | log level | `INFO` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `DEFAULT_SAMPLE_MAX_ROWS` | default sample size | `1000` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `TMP_ROOT` | where uploaded session input is kept (mount a volume to keep it across restarts) | `/var/tmp/avaloka` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `PLANNER_GRAPH_OUTPUT_DIR` | where planner graphs are written | `/tmp/planner_graphs/` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `LANGSMITH_API_KEY` | LangSmith tracing key | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `LANGSMITH_PROJECT` | LangSmith project | `avaloka` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `CLOUD_PROVIDER` | cloud for artifacts/infra | `gcp` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `CONTAINER_REGISTRY` | image registry | `europe-docker.pkg.dev/<project>/<repo>` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AWS_ACCOUNT_ID` | AWS account | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AZURE_RESOURCE_GROUP` | Azure resource group | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AZURE_LOCATION` | Azure region | `eastus` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AZURE_CONTAINER_REGISTRY` | ACR name | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP service-account key path | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `GCP_PROJECT_ID` | GCP project | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `GCP_CLUSTER_NAME` | GKE cluster | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `GCP_ZONE` | GKE zone | `us-central1-c` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `GCS_BUCKET` | artifact bucket | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `GCS_INFERENCE_BUCKET` | inference bucket | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `GCS_INFERENCE_PREFIX` | inference prefix | `tmp-inference` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AVALOKA_ASSET_BUCKET` | asset registry bucket | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AVALOKA_SIGNED_URL_EXPIRY_MINUTES` | signed URL lifetime | `15` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `DISABLE_MLFLOW` | turn MLflow off | `False` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `MLFLOW_BACKEND_STORE_URI` | MLflow database (holds a password: keep in .env / Secret) | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `MLFLOW_DEFAULT_ARTIFACT_ROOT` | MLflow artifacts | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `MLFLOW_TRACKING_URI` | MLflow server (unset: local SQLite) | `http://localhost:5000` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment | `avaloka-training` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_MODE` | model inference mode | `local` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_IMAGE` | inference image | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_SERVICE_DOCKER_IMAGE` | inference service image | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_SERVICE_GKE_CLUSTER_LOCATION` | inference GKE location | `us-central1` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `K8S_NAMESPACE` | namespace for inference jobs | `default` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `K8S_SERVICE_ACCOUNT` | service account for inference jobs | `gcs-writer` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `INFERENCE_JOB_TTL_SECONDS` | inference job TTL | `600` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING` | size above which training uses Ray | `10485760` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `MTA_RAY_DASHBOARD_URL` | existing RayCluster for training (empty: per-run cluster) | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `MTA_RAY_NAMESPACE` | Ray namespace for training | `default` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `MTA_RAY_TRAINING_ENTRYPOINT` | training entrypoint | `python /home/ray/app/app/agents/mta_v2/training_docker_image/ray_job.py` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `AVALOKA_PII_KEY` | HMAC key for PII pseudonymisation | — (generated) | **generated automatically** | `.env` (local) · `.env.quickstart` (quickstart) · release Secret (Helm, kept across upgrades) |
| `MCP_SERVER_URL` | MCP server endpoint | `http://localhost:8010` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `MCP_SERVICE_TOKEN` | shared API ↔ MCP/onboarding service secret | — (generated) | **generated automatically** | `.env` (local) · `.env.quickstart` (quickstart) · release Secret (Helm, kept across upgrades) |
| `EXECUTE_SERVICE_TOKEN` | shared graph pods ↔ `/execute` service secret | — (generated) | **generated automatically** | `.env` (local, compose) · release Secret (Helm, kept across upgrades) |
| `DB_ENCRYPTION_KEY` | AES-GCM key for every sealed credential | — (generated) | **generated automatically** | `.env` (local) · `.env.quickstart` (quickstart) · release Secret (Helm, kept across upgrades) |
| `SUPABASE_URL` | Supabase project URL | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service-role key | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `SUPABASE_JWT_SECRET` | verifies Supabase sign-in tokens (not used in local auth mode) | — | **you set** with Supabase sign-in — the API will not start without it (the quickstart uses local auth instead) | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `GITHUB_SYSTEM_TOKEN` | asset registry token | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |
| `GITHUB_JOB_REGISTRY_REPO` | asset registry repo | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `DEFAULT_INFRA_PLATFORM` | `gcp` / `aws` / `azure` | `gcp` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `ENABLE_AZURE_INFRA` | `1` enables Azure infra | `0` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `OPENCLAW_ENABLED` | `1` enables the openclaw sidecar | `0` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `OPENCLAW_GATEWAY_URL` | openclaw endpoint | `http://localhost:18789` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `OPENCLAW_TIMEOUT_S` | openclaw timeout | `2.0` | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: chart values, where the chart exposes it |
| `OPENCLAW_API_KEY` | openclaw key | — | optional — you set it if you use it | `.env` / `.env.quickstart` · Helm: `secrets.*` (release Secret), where the chart exposes it |

Unknown values fail closed: an unrecognised deployment resolves to
`self_hosted`, and an unrecognised edition resolves to `oss`.

Never commit `.env` or `.env.quickstart`. Both are gitignored; keep it that way.

---

## Running it

To **try the web app on a laptop**, use the quickstart: one command, no
Supabase, no cluster, and it runs offline once it is built. For a team or a
production install, use Kubernetes (A). Local processes (B) are for working
on the API itself. Pick one; mixing them is the most common way to end up
with an API that cannot reach its own database.

### Quickstart: the web app on one machine

```bash
git clone https://github.com/guruvaidev/avaloka.git && cd avaloka
ollama pull gemma4:e4b             # the default local model (or set a hosted one, below)
scripts/quickstart.sh up           # build, check models, start; prints the URL
scripts/quickstart.sh password     # the sign-in password for local@avaloka.local
# ... open http://127.0.0.1:3000/, sign in, upload a CSV or connect a database
scripts/quickstart.sh down         # stop (data kept); `down --volumes` deletes it too
```

You need Docker with Compose v2 and about 12 GB of free disk (the images are
about 6 GB). The first `up` builds two images from your checkout and takes a
while; after that, `up` with no code change starts in under a minute, and a
rebuild after pulling new code took about six minutes on an M-series Mac. The
stack uses 1.5 to 1.7 GB of memory, plus whatever the model server needs.

Uploads answer in about 3 s with charts drawn from the data alone; the
model-written profile and insights follow in the background and land minutes
later on a local model (about 3 minutes with `gemma4:e4b` on a laptop CPU/GPU
shared with everything else). A hosted model is much faster.

What it runs (`docker-compose.quickstart.yml`):

| Service | What | Host port |
| --- | --- | --- |
| `ui` | the web app | `127.0.0.1:3000` |
| `api` | the API (`app.api.server`) | `127.0.0.1:9000` |
| `langgraph` | the agent graph server the API drives | none |
| `mcp` | MCP server with database onboarding | none |
| `redis` | sessions, Celery/RedBeat broker | none |
| `postgres` | a sample database (`orders`, 2,000 rows) to try the connector on | none |

Only the UI and the API are published, and only on 127.0.0.1. If 3000 or 9000
is taken, the script picks the next free port and remembers it; set
`AVALOKA_UI_PORT` / `AVALOKA_API_PORT` to choose. Redis and Postgres have no
host port, so they cannot collide with ones you already run.

**Sign-in without Supabase.** The quickstart runs the API with
`AVALOKA_AUTH_MODE=local`: one local account (`local@avaloka.local`), password
sign-in, tokens signed by the API itself. The API refuses this mode, and will
not start, unless `AVALOKA_DEPLOYMENT=self_hosted` is set explicitly, the
edition is not a hosted one, and `SUPABASE_URL` is unset. The Helm chart never
sets it. It is a laptop convenience, not a multi-user auth system.

**Secrets.** The first `up` writes `.env.quickstart` (gitignored, mode 600):
`DB_ENCRYPTION_KEY` (the API and the MCP server share it: it seals stored
database credentials and signs requests between them), the local-auth secret
and password, and the sample Postgres password. Nothing is printed except by
`scripts/quickstart.sh password`. The data volumes belong to the machine, not
the checkout, so to start over delete both: `scripts/quickstart.sh down
--volumes`, then the file. (`up` refuses to generate new secrets while old
data exists, because the old database and sealed credentials would not open.)

**Models.** Inference goes to Ollama on the host
(`host.docker.internal:11434`, model `gemma4:e4b`). To use a hosted provider,
edit `.env.quickstart`, e.g. `INFERENCE_PROVIDER=groq` and `GROQ_API_KEY=...`.
The embedding models are baked into the API image (see below).

**Connect the sample database.** In the app, add a PostgreSQL connection with
host `postgres`, port `5432`, database `sample`, user `avaloka`, and the
`QS_POSTGRES_PASSWORD` from `.env.quickstart`; then analyse the `orders`
table. To reach a database on your own machine, use host
`host.docker.internal`.

**Offline.** `scripts/quickstart.sh up --offline` starts the same stack with
the backend network marked `internal`: the API, LangGraph, MCP, Redis and
Postgres have no route off it at all. The UI keeps a second network so your
browser can reach it, and one relay forwards a single port to the host's
Ollama. In this mode the API's own host port is not published; everything
goes through the UI, as the browser does.

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
  --set-string secrets.openrouterKey="$OPENROUTER_API_KEY"

kubectl port-forward svc/avaloka 9000:9000
curl localhost:9000/health
```

Two things about that command worth knowing before you run it.

**The chart's provider follows its keys.** `inference.provider` defaults to
`auto` (`deploy/helm/avaloka/values.yaml`): OpenRouter when
`secrets.openrouterKey` is set, Groq when only Groq keys are, OpenRouter
otherwise -- the same rule as a local install with `INFERENCE_PROVIDER` unset
(MP-01, owner decisions 2026-11-01 and 2026-11-02).
Agents run `openai/gpt-oss-120b`, pinned to the upstream providers in
`inference.openrouter.providersOnly`, with GPT-6 Luna, then gpt-oss-120b on Groq
(when a Groq key is set), as the fallback tiers. A local install with
`INFERENCE_PROVIDER` unset resolves to OpenRouter as well (`app/core/inference.py`;
MP-01, owner decision 2026-11-01). See
[privacy-and-egress.md](privacy-and-egress.md) for where prompts go.

```bash
  --set-string secrets.openrouterKey="$OPENROUTER_API_KEY"
```

To stay on Groq instead, set `--set inference.provider=groq` and the Groq keys.

**Upgrading a chart install that ran on Groq.** `inference.provider` defaults
to `auto`: the chart selects OpenRouter only when `secrets.openrouterKey` is
set, and Groq when only Groq keys are. A Groq-only install therefore keeps its
model after `helm upgrade`. Add `--set-string secrets.openrouterKey=...` to move
it to OpenRouter. With `existingSecret` the chart cannot see the keys, so it
emits no provider and the backend applies the same rule to the keys it finds.

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
POSTGRES_HOST_PORT=55432 docker compose -f deploy/compose/docker-compose.postgres.yml up -d  # Postgres, 127.0.0.1 only
REDIS_HOST_PORT=6379 docker compose -f deploy/compose/docker-compose.redis.yml up -d         # Redis, 127.0.0.1 only
docker compose up -d                                  # Milvus stack (optional;
                                                      #   needed for episodic memory)

uvicorn app.api.server:app --reload --port 9000 --loop asyncio  # API
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
