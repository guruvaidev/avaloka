# Security Policy

## Supported Versions

Avaloka is under active development. Security fixes are applied to the latest
0.x release. Older versions are not maintained.

| Version | Supported |
| ------- | --------- |
| 0.2.x   | ✅        |
| < 0.2   | ❌        |

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

## Disclosure

We follow coordinated disclosure. We will credit reporters who wish to be
acknowledged once a fix is released, unless you prefer to remain anonymous.
