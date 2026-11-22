# Where model prompts go

Agent prompts carry table schemas, column names and sample rows. This page
covers model-provider egress: which provider sees a prompt, under what terms,
and what happens when that provider is unavailable.

## The chain

| Tier | Model | Through | Upstream providers allowed |
|---|---|---|---|
| 1, the default (large) | `openai/gpt-oss-120b` | OpenRouter, cheapest listed provider first | DeepInfra (bf16), Together, Crusoe, Parasail |
| 1, the default (small) | `openai/gpt-oss-20b` | OpenRouter, cheapest listed provider first | DeepInfra (bf16), Parasail (the two of the four that serve it) |
| 2 | `openai/gpt-6-luna` | OpenRouter | Azure |
| 3, last | `openai/gpt-oss-120b` | Groq, directly | Groq |

Each OpenRouter request carries these
[provider routing preferences](https://openrouter.ai/docs/features/provider-routing):

```json
"provider": {
  "only": ["deepinfra/bf16", "together", "crusoe", "parasail"],
  "allow_fallbacks": false,
  "data_collection": "deny",
  "zdr": true,
  "sort": "price"
}
```

This covers every OpenRouter client: the agents' factory, the Groq primary's
backup and the visualization agent's own client.

* **`only`**: the providers allowed for this request, per model (tier 2 uses its
  own list, `azure`).
* **`allow_fallbacks: false`**: OpenRouter must not use a provider outside the list
  when the listed ones are busy. This keeps egress inside the list.
* **`data_collection: "deny"`**: only providers that do not train on prompts
  ([privacy and logging](https://openrouter.ai/docs/features/privacy-and-logging)).
* **`zdr: true`**: only Zero Data Retention endpoints
  ([ZDR](https://openrouter.ai/docs/features/zdr); live list
  `https://openrouter.ai/api/v1/endpoints/zdr`).
* **`sort: "price"`**: the cheapest listed provider first.

Tier 3 is called on Groq's own API, outside OpenRouter, so OpenRouter's
preferences do not apply to it. Groq states that it does not retain customer
data for inference requests by default and keeps data in the US
([Groq: your data](https://console.groq.com/docs/your-data)).

## How a provider is chosen

A provider is allowed when all of these hold:

* it publishes terms for API inference;
* those terms say it does not train on prompts or completions;
* it is on OpenRouter's Zero Data Retention list for the model;
* it states the jurisdiction it operates in;
* it supports tool calling for the model.

The allowed providers' own terms:
[DeepInfra](https://deepinfra.com/privacy),
[Together](https://www.together.ai/privacy),
[Crusoe](https://legal.crusoe.ai/open-router),
[Parasail](https://www.parasail.io/legal/privacy-policy) and
[Groq](https://console.groq.com/docs/your-data). Azure's endpoint for tier 2 is the
model's ZDR endpoint on OpenRouter's list.

## When a tier cannot answer

* With `allow_fallbacks: false`, OpenRouter refuses a request no listed provider
  can serve. Measured on 2026-11-02: **HTTP 404**, "No allowed providers are
  available for the selected model". OpenRouter also documents **HTTP 503** for
  "no available model provider that meets your routing requirements"
  ([errors](https://openrouter.ai/docs/api-reference/errors)).
* The request then moves to the next tier on 404 or 503, on 402 (the OpenRouter
  account is out of credit), on 429 (rate limited), and on an outage (any other 5xx
  or a connection failure). A 400, 401, 403 or 422 does not move, because it would
  fail the same way on every tier.
* When no tier can answer, the agent receives the first tier's error, which names
  the reason, for example "the openrouter account is out of credit (HTTP 402)".
  The request is never routed to a provider outside the lists above.

## A swap is never silent

When a tier other than the configured model answers, the reply says so. Under the
answer the UI shows which model answered and why. For example:

> openai/gpt-4o did not answer (no allowed zero-retention provider serves
> openai/gpt-4o); openai/gpt-6-luna via openrouter answered instead.

The reason comes from the recorded HTTP status and provider message, never from a
model's text. The model screen (Configurations, then Models, then Language models)
shows each agent role's configured model and the full chain behind it, including
a tier that is configured but has no key.

`data_collection` and `zdr` also apply to a model you choose yourself. A model
with no ZDR endpoint is refused and the next tier answers, and the reply says so.
On OpenRouter's list of 2026-11-01, `openai/gpt-4o` has only Azure, and
`anthropic/claude-haiku-5.5` has only Google Vertex and Amazon Bedrock. Set
`AVALOKA_OPENROUTER_ZDR=false` to use such a model.

## Changing it

| Setting (env / Helm `inference.*`) | Default | Effect |
|---|---|---|
| `INFERENCE_PROVIDER` / `inference.provider` | `auto` in the chart: OpenRouter with its key, Groq with only Groq keys | `openrouter`, `groq`, `local` (nothing leaves your environment), `openai`, `bedrock`, `vertex`, `azure` |
| `AVALOKA_OPENROUTER_ONLY` / `openrouter.providersOnly` | unset: each model's own list | replaces the list for the primary, whatever its model; `none` lifts it (OpenRouter's whole pool) |
| `AVALOKA_<AGENT>_OPENROUTER_ONLY` | unset | per-agent allowlist |
| `AVALOKA_OPENROUTER_ALLOW_FALLBACKS` / `openrouter.allowFallbacks` | `false` | `true` lets OpenRouter leave the list when it is busy |
| `AVALOKA_OPENROUTER_DATA_COLLECTION` / `openrouter.dataCollection` | `deny` | `allow` admits providers that train on prompts |
| `AVALOKA_OPENROUTER_ZDR` / `openrouter.zdr` | `true` | `false` admits endpoints that retain prompts |
| `AVALOKA_OPENROUTER_SORT` | `price` | `throughput` or `latency` instead |
| `AVALOKA_OPENROUTER_FALLBACK_MODEL` / `openrouter.fallbackModel` | `openai/gpt-6-luna` | tier 2; `none` disables it |
| `AVALOKA_OPENROUTER_FALLBACK_ONLY` / `openrouter.fallbackProvidersOnly` | unset: that model's own list | replaces tier 2's list |
| `AVALOKA_GROQ_FALLBACK_MODEL` | `openai/gpt-oss-120b` | tier 3, armed when a Groq key is set; `none` disables it |
| `AVALOKA_DISABLE_MODEL_FALLBACK` | `0` | `1` disables every fallback tier |
| `INFERENCE_OPENROUTER_MODEL_LARGE` | `openai/gpt-oss-120b` | `openai/gpt-6-luna` runs the higher tier as the primary; it is sent with reasoning effort `none`, which its tool calling on Chat Completions requires |

Before adding a provider to an allowlist, check it against the criteria above and
its entry in OpenRouter's ZDR list.
