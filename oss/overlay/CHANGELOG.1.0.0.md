## [1.0.0] — first public release

The first tagged release of this repository. It is built from the same source
as the 1.6 development line and contains every change recorded below, under
[Unreleased] and [1.6.0]; those headings keep the numbers they were written
under. Builds of `main` published before this release carried no tag and
reported their version as `1.6.0`.

- **Security.** 1.0.0 includes the fix for the MCP server advisory in
  [SECURITY.md](SECURITY.md): the onboarding and admin routes require the
  customer's API key or the API's `MCP_SERVICE_TOKEN`. **Upgrading from a build
  of `main` from before 2026-10-29:** set `MCP_SERVICE_TOKEN` to the same value
  on the API and the MCP server, and rotate the database credentials stored in
  the MCP server.
- **Security: generated code no longer runs with the server's secrets.** The
  model-written analysis script, the data-transfer sample preview and the
  validator's trial run inherited the server's environment (provider keys,
  database passwords, `DB_ENCRYPTION_KEY`, `MCP_SERVICE_TOKEN`, Supabase keys),
  and the trial run executed inside the server process. All three now run in a
  child process with an allowlisted environment and a private home directory;
  the trial run also has a time limit (`AVALOKA_VALIDATION_TIMEOUT`) and, on
  Linux, a memory limit (`AVALOKA_VALIDATION_MEMORY_MB`).
- **Security: the `/execute` service is authenticated, cluster-internal and
  sandboxed.** Earlier builds could deploy `app/infra/python_app` at run time
  when a chat message mentioned deploying or a cloud: a service that ran
  `exec()` on the request body with no authentication, behind a `LoadBalancer`
  Service, with the namespace's default ServiceAccount token mounted. It is
  removed. The replacement (`executeService` in the Helm chart, off by default)
  requires `EXECUTE_SERVICE_TOKEN` on every request, is `ClusterIP` only with a
  NetworkPolicy, has no ServiceAccount token, and runs each request in a
  sandboxed child process. Provisioning now needs an explicit request and a
  confirmation. See the advisory in [SECURITY.md](SECURITY.md): **if an
  earlier build created the `infra-agent` Deployment, the `infra-agent-service`
  Service or the `infra-agent-config` ConfigMap in your cluster, delete them**,
  rotate what the namespace's default service account could reach, and review
  the load balancer's logs.
- **A connection saved under a different `DB_ENCRYPTION_KEY` is reported, not
  lost.** After the key changed, such connections vanished from listings and
  another user could register the same id over them. They are now listed to
  their owner as needing re-registration, and nobody else can take their id.
- **One version number.** `VERSION`, `pyproject.toml`, the Helm chart's
  `version` and `appVersion`, `avaloka --version` and `GET /version` all read
  1.0.0.
- **Images.** The `v1.0.0` tag publishes the four images under
  `ghcr.io/guruvaidev/` (`avaloka-api`, `avaloka-ui`, `avaloka-ray`,
  `avaloka-functions`) tagged `1.0.0` and `1.0`
  (`.github/workflows/images.yml`).
- **What it does**, in the words of the [README](README.md): a team of
  specialist agents that takes a dataset from raw rows to a served model and
  checks its own work before it reports a result. Generated code is validated
  before it runs, a trained model must beat a trivial baseline, and the numbers
  in each answer are checked against the results that were computed. It runs
  from a laptop CLI, fully offline, up to Ray/KubeRay on any Kubernetes
  cluster through one Helm chart, with the model provider of your choice.
- **Known issues** are listed under [Unreleased] below.

