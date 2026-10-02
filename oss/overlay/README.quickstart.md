## Quickstart

Five minutes, on your own machine, no cloud account.

```bash
git clone https://github.com/guruvaidev/avaloka.git && cd avaloka
./scripts/install.sh                 # creates .venv, installs deps, prints an edition report
```

Give it one model key — any provider — and start the API:

```bash
export GROQ_API_KEY="<your key>"     # or OPENAI_API_KEY / OPENROUTER_API_KEY / a local Ollama
uvicorn app.api.server:app --port 9000
curl localhost:9000/health
```

Then upload something and ask a question:

```bash
curl -F "file=@sales.csv" localhost:9000/api/upload
```

> **Port 9000, not 8000.** The container image serves on 9000 and the Helm
> service maps `8000 → 9000`. Uvicorn's own default is 8000, which is the single
> most common reason a fresh install "cannot reach the API" — pass `--port 9000`
> and every path lines up.

Prefer a cluster, a UI, or no internet at all? That is all in the
**[Install Guide](docs/INSTALL.md)** — including
[running Avaloka completely offline](docs/INSTALL.md#running-avaloka-completely-offline)
with a `--network none` check so you can prove it rather than trust it.

**Something not working?** The install guide's
[First run: what to expect, and what goes wrong](docs/INSTALL.md#first-run-what-to-expect-and-what-goes-wrong)
section lists every failure we hit bringing this stack up on a clean cluster.
Each one is silent — the symptom never names the cause — so it is worth a read
before you start debugging.

### Two ways to run it

| | |
| --- | --- |
| **Run it yourself, fully offline** | [**Install Guide → Running Avaloka completely offline**](docs/INSTALL.md#running-avaloka-completely-offline). No internet at all: a local model server, the embedding model **baked into the image**, and storage, memory and auth all deployed by the chart. Nothing calls home. |
| **Let us run it** | [**avaloka.ai**](https://avaloka.ai) hosts the **Professional** and **Enterprise** editions with auto-scaling compute — no Kubernetes, Ray, object store or model provider to stand up. Enterprise adds shared team workspaces, the cloud scheduler, and connections to your own cloud clusters. |

The open-source edition is not a limited trial of the hosted ones. It is the
same engine on your own hardware, and it stays useful with no account, no key
and no network.

Under the hood, a Planner agent acts as the team lead: it converses with you and
delegates to Sampling, Profiling, Coder, Validator, Execution, Data-Transfer,
Model-Training, and Visualization specialists, all coordinating through one
shared state object. Generated code is authored pseudocode-first and passes a
three-layer validation gate, so it stays grounded in the source schema even when
the primary LLM is unavailable.

---

