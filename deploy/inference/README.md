# Custom inference stack — local & multi-cloud LLM backends

Avaloka's agents (planner, coder, validator, summarizer, viz, profiling, MTA,
DTA) are **provider-agnostic**. Every agent builds its LLM through
`app.core.inference.build_chat_model()`, which returns a LangChain chat model
for the configured backend. The agent code — prompts, tool definitions, control
flow — is identical across backends; only the endpoint changes.

Nothing here changes planner/agent behavior. Groq remains the default; the rest
is opt-in via `INFERENCE_PROVIDER`.

## Supported providers

| `INFERENCE_PROVIDER` | Backend | Client | Typical use |
|---|---|---|---|
| `groq` (default) | Groq Cloud | `ChatGroq` | Hosted, fast, default |
| `local` | In-cluster Ollama (CPU) / vLLM (GPU) | `ChatOpenAI` (OpenAI-spec) | Laptop/kind; on-prem; no vendor |
| `openai` | OpenAI or any OpenAI-spec endpoint | `ChatOpenAI` | Hosted OpenAI-spec |
| `openrouter` | OpenRouter | `ChatOpenAI` | Broad model catalog |
| `bedrock` | AWS Bedrock | `ChatBedrockConverse` | AWS-native |
| `vertex` | GCP Vertex AI | `ChatVertexAI` | GCP-native |
| `azure` | Azure OpenAI / Azure AI | `AzureChatOpenAI` | Azure-native |

Selection precedence (most specific wins): `AVALOKA_<AGENT>_PROVIDER` →
`INFERENCE_PROVIDER_<ROLE>` (`PLANNING`/`CODING`/`VIZ`) → `INFERENCE_PROVIDER` →
`groq`. If the selected provider is not configured, the agent falls back to its
deterministic path — exactly as it did when a `GROQ_API_KEY` was missing.

## Local model choice — Qwen2.5-3B-Instruct

The default small local model is **Qwen2.5-3B-Instruct**, chosen over
**Nemotron-Nano-3B** and **Gemma-3-4B** for *this* workload:

- The agents drive the model almost entirely through **OpenAI tool/function
  calls and strict JSON** (`bind_tools`, `tool_choice="required"`). Qwen2.5-
  Instruct has the most consistent tool-calling + JSON-schema adherence at the
  ~3B scale, which is what actually determines whether the planner routes
  correctly on a small model.
- **First-class in both Ollama and vLLM**, so the same model id works on the
  CPU (laptop) tier and scales up to the GPU tier with no agent changes.
- Permissive license and a coherent family (3B → 7B → 14B → 32B → 72B), so
  "small on my laptop, big on GPU nodes" is one knob, not a model migration.

Gemma-3-4B is strong and multimodal but historically weaker at strict
function-calling/JSON tool schemas and is larger; Nemotron-Nano-3B reasons well
but has less consistent OpenAI-style tool-call formatting and thinner
Ollama/vLLM support today. Swap the model any time — it is pure configuration
(`localLLM.model` / `INFERENCE_LOCAL_MODEL_*`); the agents do not change.

## Laptop / kind (CPU, no GPU)

Serves the 3B model via Ollama's OpenAI-compatible API.

**Standalone manifests:**

```bash
kubectl apply -f deploy/inference/ollama-cpu.yaml
kubectl rollout status deploy/avaloka-local-llm --timeout=600s   # first run pulls ~2GB
```

Then point the backend at it:

```bash
INFERENCE_PROVIDER=local
INFERENCE_LOCAL_BASE_URL=http://avaloka-local-llm:11434/v1
INFERENCE_LOCAL_MODEL_LARGE=qwen2.5:3b-instruct
INFERENCE_LOCAL_MODEL_SMALL=qwen2.5:3b-instruct
```

**Via the Helm chart** (wires the backend env automatically):

```bash
helm upgrade --install avaloka deploy/helm/avaloka \
  --set localLLM.enabled=true \
  --set inference.provider=local
```

**Smoke test** the endpoint:

```bash
kubectl port-forward svc/avaloka-local-llm 11434:11434 &
curl -s http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:3b-instruct","messages":[{"role":"user","content":"ping"}]}'
```

## GPU nodes (larger 14B / 32B models)

Serves via vLLM (OpenAI-spec) on a GPU node pool with the NVIDIA device plugin.

**Standalone:**

```bash
kubectl apply -f deploy/inference/vllm-gpu.yaml   # default Qwen2.5-14B-Instruct
```

**Via Helm:**

```bash
helm upgrade --install avaloka deploy/helm/avaloka \
  --set localLLM.enabled=true \
  --set localLLM.engine=vllm \
  --set localLLM.model=Qwen/Qwen2.5-14B-Instruct \
  --set localLLM.servedModelName=qwen-local \
  --set inference.provider=local
```

- 32B: `--set localLLM.model=Qwen/Qwen2.5-32B-Instruct` (needs an 80GB GPU, or
  add `--tensor-parallel-size 2` and 2 GPUs).
- The client-facing model id is `--served-model-name` (`qwen-local`), which is
  what `INFERENCE_LOCAL_MODEL_*` must equal — the chart sets both from
  `localLLM.servedModelName`.

## Cloud providers

Set `INFERENCE_PROVIDER` and the provider's credentials/model env. Common vars
(see `app/core/inference.py` and `.env.example` for the full list):

```bash
# OpenRouter
INFERENCE_PROVIDER=openrouter
OPENROUTER_API_KEY=...
INFERENCE_OPENROUTER_MODEL_LARGE=qwen/qwen-2.5-72b-instruct

# AWS Bedrock (creds via the standard AWS chain / IRSA)
INFERENCE_PROVIDER=bedrock
AWS_REGION=us-east-1
INFERENCE_BEDROCK_MODEL_LARGE=meta.llama3-1-70b-instruct-v1:0

# GCP Vertex AI (ADC / Workload Identity)
INFERENCE_PROVIDER=vertex
GOOGLE_CLOUD_PROJECT=my-project
INFERENCE_VERTEX_MODEL_LARGE=gemini-1.5-pro

# Azure OpenAI
INFERENCE_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://my.openai.azure.com
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT_LARGE=gpt-4o
```

Bedrock needs `langchain-aws`; Vertex needs `langchain-google-vertexai` (both in
`requirements.txt`, imported lazily so a Groq-only/local-only install skips them).

## Notes on tool-calling parity

The OpenAI-spec providers (`local`, `openai`, `openrouter`, `azure`) and `groq`
carry the planner's raw `tools=`/`tool_choice=` calls natively. Bedrock and
Vertex use their own tool schemas; they are best suited to the non-tool agents
or to models exposed through an OpenAI-spec gateway. For a fully local,
tool-driven planner, use the `local` (vLLM/Ollama) tier.
