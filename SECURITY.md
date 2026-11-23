# Security Policy

## Supported Versions

Avaloka is under active development. Security fixes are applied to the latest
release line and published on `main`. Older versions are not maintained.

| Version | Supported |
| ------- | --------- |
| 1.6.x (`main`) | ✅ |
| < 1.6   | ❌        |

## Advisories

### 2026-11: The /execute service ran code without authentication

**Affected:** every release up to and including 1.6.0, and `develop-1.6` before #687.
**Fixed in:** `develop-1.6` from 2026-11-03 (#687).

`app/infra/python_app` ran `exec()` on the body of every request it received,
with no authentication. It was exposed through a Service of
`type: LoadBalancer`, and its pod had the namespace's default service-account
token mounted. Anyone who could reach the load balancer could run code in that
pod and use the token with the default service account's permissions.

**When it was deployed:** the infrastructure path built and deployed it when a
chat message matched `deploy`, `aws`, `gcp` or `kubernetes`, and `docker`,
`gcloud` and `kubectl` were available to the server.

**What changed:** `app/infra/python_app` is removed. `/execute` is now served
by `executeService` in the Helm chart, which is off by default. It is
`ClusterIP` only, with a NetworkPolicy that admits only the API, LangGraph and
Celery pods. Every request needs `EXECUTE_SERVICE_TOKEN`. The pod runs
hardened, without a service-account token, and each request runs in a child
process with a scrubbed environment and time and memory limits. A chat message
that names a cloud no longer provisions anything: provisioning needs an
explicit request and a confirmation.

**How to check:** in each namespace Avaloka ran in,

```bash
kubectl get deploy infra-agent
kubectl get svc infra-agent-service
kubectl get configmap infra-agent-config
```

and look in your registry for the image `gcr.io/<project>/infra-agent-python-app`.

**What you need to do:**

1. Upgrade to a build that carries the fix (see **Fixed in** above).
2. Delete those resources and the image:
   `kubectl delete deploy infra-agent`, `kubectl delete svc infra-agent-service`,
   `kubectl delete configmap infra-agent-config`.
3. Treat everything the namespace's default service account could reach as
   exposed, and rotate it: its token, and every secret or credential it could
   read.
4. Review the load balancer's logs for requests to `/execute` since the service
   was created.

Questions about this advisory: **support@avaloka.ai**.

### 2026-10: MCP server onboarding and admin routes were unauthenticated

**Affected:** every release up to and including 1.6.0, and `main` before the
sync of 2026-10-29.
**Fixed in:** `main` from 2026-10-29.

The MCP server (port 8010 in the Helm chart) served its customer-onboarding
routes (`/customers/*`) and its admin routes (`/admin/customers`, `/connect`,
`/connections`) without authentication. Anyone who could reach the MCP server
could list the registered database connections, read their stored
credentials, and replace a connection's configuration.

**What changed:** these routes now require either the customer's own API key or
the Avaloka API's service token (`MCP_SERVICE_TOKEN`), and each caller sees
only the connections it owns. A deleted connection's key stops working
immediately.

**What you need to do:**

1. Upgrade to the current `main`.
2. Set `MCP_SERVICE_TOKEN` to the same random value on the API and on the MCP
   server (`openssl rand -hex 32`; the Helm value is `secrets.mcpServiceToken`).
   If it is unset, no credentials are released and database-to-database
   transfers stop working.
3. **Rotate the database credentials** stored in your MCP server. Treat them as
   exposed if the server was ever reachable from an untrusted network.
4. Keep the MCP server off untrusted networks. Registering a connection makes
   the server connect to the address it is given.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public issues, pull
requests, or discussion threads.**

Instead, report them privately by email to **security@avaloka.ai**. If you use
encryption, request our PGP key in your first message and we will provide it.

Please include as much of the following as you can:

- The type of issue (e.g. remote code execution, credential exposure, SSRF,
  injection, privilege escalation, path traversal).
- The affected component and file path(s) — for example a specific agent under
  `app/agents/`, an API route in `app/api/server.py`, or a deployment manifest
  under `deploy/`.
- Step-by-step instructions to reproduce, including any configuration or
  environment variables required.
- Proof-of-concept or exploit code, if available.
- The impact, and how an attacker might exploit it.

You will receive an acknowledgement within **3 business days**. We aim to
provide an initial assessment within **10 business days** and to agree a
coordinated disclosure timeline with you from there.

## Scope and Sensitive Areas

Avaloka generates and executes code, provisions cloud infrastructure, and
handles credentials. The following areas are especially security-relevant and
are prioritized for reports:

- **Generated-code execution** — the Coder and Execution agents produce Python
  that is executed locally, in Kubernetes Jobs, or on Ray. Sandbox escapes,
  arbitrary file access, and injection into generated code are in scope.
- **Credential handling** — Groq/LLM keys, cloud service-account JSON, database
  credentials, and the JWT signing secret. Leakage into logs, artifacts, error
  messages, or persisted assets is in scope.
- **Multi-tenant isolation** — the MCP server (`app/mcp_server/`) and the
  session/thread model in `app/api/server.py`. Cross-tenant data access is in
  scope.
- **Cluster/RBAC** — the Helm charts and provider abstraction under `deploy/`
  and `app/infra/`. Over-broad RBAC, privileged pods, or secrets rendered into
  plaintext manifests are in scope.
- **Supply chain** — pinned dependencies in `requirements.txt` and base images
  in `deploy/docker/`.

## Operator Guidance

If you deploy Avaloka, you are responsible for the security of your environment.
At minimum:

- **Never commit secrets.** Provide `GROQ_API_KEY_*`, cloud credentials, and
  `JWT_SECRET` at deploy time via Kubernetes Secrets or a secrets manager, not in
  values files or `.env` files that reach version control. `.env` is
  `.gitignore`d — keep it that way.
- **Treat generated code as untrusted.** Run execution workloads with the least
  privilege necessary; do not grant the execution namespace access to
  credentials it does not need.
- **Scope RBAC narrowly.** The service account only needs the Ray/Job verbs the
  workload uses.
- **Do not expose the API or Ray dashboard to the public internet** without an
  authenticating proxy. The Ray dashboard has no authentication of its own.
- **Set `MCP_SERVICE_TOKEN`** on the API and the MCP server, and keep the MCP
  server on a private network. See the 2026-10 advisory above.

## Disclosure

We follow coordinated disclosure. We will credit reporters who wish to be
acknowledged once a fix is released, unless you prefer to remain anonymous.
