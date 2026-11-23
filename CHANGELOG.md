# Changelog

All notable changes to Avaloka are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Changes land here first and move under a version heading when that version
is cut.

### Added
- **Connector reads feed the catalog and the graph** (`app/connectors/registry.py`;
  design part B §B.5, the catalog half of 15b). `record_read` turns a
  connector's `ReadReceipt` into the owner's catalog registration (the
  receipt's columns, row count -- a floor when the read was truncated -- and
  source version, so a re-read of an unchanged source writes no version and
  the analysis pins the version it read) and into a `LOADED_FROM` lineage edge
  carrying the source-version and query digests, never the query text or a
  credential. Connectors say where a dataset lives with `locator_for()`
  (file name and format; backend, schema and table); `try_record_read` is the
  never-raising shape for agent paths; `read_and_record` does both.
- **Object stores on the connector contract** (`avaloka/connectors/objectstore.py`;
  design part B, 15c). S3, GCS and Azure (and the local and in-memory
  filesystems the tests use) are one `ObjectStoreConnector`: a bucket and an
  optional prefix, read/schema/sample, no push-down, no data sockets beyond
  the one host `endpoints()` names (a GCS service account's token exchange
  also reaches Google's OAuth endpoint, and options that name no key leave the
  backend's default credential lookup in charge). Options (keys, service account, SAS, endpoint)
  arrive only as a sealed `Credentials` document; a signed URL or userinfo in
  the store URL is refused as a credential. `source_version()` is the object's
  ETag (size and mtime where there is none) and over a prefix the set of them.
  The conformance suite now runs on an in-memory store, a local directory and
  S3 against an in-process moto server, where the socket recorder sees a real
  socket.
- **Cloud keys are sealed, never plaintext** (D15.3). Migration
  `20261116130000` drops the plaintext branch of `cloud_datasets`'
  credentials CHECK (`NOT VALID`, so rows written earlier survive until
  sealed), retires the anon-callable `encrypt_cloud_credentials` RPC, and the
  backend seals: `POST /api/cloud-connections` is now the one place a key
  enters the table (AES-GCM under `DB_ENCRYPTION_KEY`, in process), both UI
  writers post there instead of inserting keys themselves, and rows written
  before the migration are sealed at API start-up (`seal_plaintext_rows`,
  also `python -m app.api.cloud_connections seal`). An opened row becomes one
  `Credentials` document for the connector (`object_store_connector`).
- **The connector contract, extended** (`avaloka/connectors`; design
  `docs/design/1.7-DATA-PLATFORM.md` part B, 15a). Every connector declares
  its `Capability` set (read, pushdown, write, schema, sample, stream) and a
  method it did not declare refuses the call. `endpoints()` lists every
  host:port it will open; `source_version()` is stable for an unchanged source
  and moves when it changes; `read()` and `sample()` return bounded Arrow
  results with a `ReadReceipt` (rows, bytes, `truncated`, what stopped it, the
  sample method, a query digest and never the query text). A sample returns
  min(n, rows): a Postgres `tablesample` that comes back short (block
  sampling can miss every page of a small table) is re-drawn by `random`, and
  the receipt names `random`. A push-down is
  parsed with sqlglot in the connector's dialect and refused unless it is one
  parameterised SELECT over one table made of the plan's node types
  (`avaloka/connectors/sqlguard.py`). A local-file connector (CSV, Parquet)
  joins the registry. The conformance suite
  (`tests/avaloka/connector_conformance.py`) is parametrised from the
  registry: sqlite and files natively, Postgres in the containers lane.
  `read(query)` now returns a `ReadResult`; call `.to_pandas()` for the frame.
- **The catalog on the command line, and `/datasets` over the catalog**
  (16g). `avaloka catalog list|show|use|forget` works on a local SQLite
  catalog under `~/.avaloka` (SQLAlchemy from the `sources` extra, imported
  only when used) or, with `--server`/`--token` (env `AVALOKA_API_URL`,
  `AVALOKA_API_TOKEN`), on a hosted catalog through `/catalog/datasets` over
  plain urllib -- the CLI gains no dependency and still never imports the
  application. `avaloka analyze` registers the file it reads (columns, a
  content hash) and accepts `my.<name>`, `my.<name>@<version>` or
  `analysis:<id>` in place of a path. `GET /datasets` now answers with the
  session store's datasets plus the catalog's file and derived datasets the
  sessions no longer hold; Redis-only datasets are kept. The catalog store
  itself moved to `avaloka/catalog/store.py` (no `app` imports);
  `app/catalog/store.py` wraps it with the lineage graph and keeps every name.
- **The catalog in chat** (`app/catalog/chat.py`; design part C §C.6, 16f).
  `find_datasets` searches the caller's own catalog (names, tags, terms, the
  analyses that used a dataset) and answers with each dataset's `my.<name>`
  handle and a code-written summary; `use_dataset` resolves `my.<name>`,
  `my.<name>@<version>` or `analysis:<id>` to one (dataset, version) and binds
  it to the turn exactly like a fresh upload (`active_dataset_id`,
  `datasets_context`, `uploaded_csv_columns` from the catalog's schema, the
  file from the session store), pinning the version a past analysis read when
  asked for it. A dataset whose copy Avaloka no longer holds is named and not
  bound. The planner advertises both tools, answers a bare "use my.<name>" or
  "my datasets" without a model, and never lets generate_code answer either.
- **The dataset catalog, personal** (`app/catalog`, `/catalog/datasets`;
  design `docs/design/1.7-DATA-PLATFORM.md` part C). Every dataset an
  analysis touches -- an upload, a table read through a connection, the
  output an analysis produced -- is registered to its owner with its typed
  columns, and a new version is written only when the schema fingerprint or
  the source's own version marker changes. Each analysis pins the version it
  used, so `my.<name>`, `my.<name>@3` and `analysis:<id>` resolve to one
  `(dataset, version)` for the planner. `DELETE` removes versions, terms and
  the lineage nodes, column digests and vault rows (`LineageStore.forget`),
  leaves a nameless tombstone, lists derived datasets rather than deleting
  them, and says which copies it did not reach. Locators carry names and ids
  only; a key or value that looks like a credential is refused. Catalog
  events carry kind, outcome and size buckets, never a name. Two capability
  flags: `dataset_catalog` (open, every edition) and `shared_catalog`
  (Enterprise; `POST .../share` answers 403 with the edition sentence
  elsewhere). The hosted migration adds owner-only row-level security and a
  trigger that refuses any visibility but `personal` outside Enterprise.
- **The Analysis Recipe Library** (`avaloka/recipes`): `profile`, `describe`
  and `trend` run as DuckDB SQL plus pandas statistics with no model in the
  loop, and each emits a verified Outcome Report and a parameterised
  notebook; golden and mutation tests under `tests/recipes` pin every figure.
  DuckDB, scipy and statsmodels are the `avaloka[recipes]` extra.
  - `compare_periods`, `segment`, `correlate`, `regress` and `anomalies`
    join the library: two-window comparison with a bridge by dimension,
    share and concentration by group, correlation with a within-group
    (Simpson) check, OLS with HC3 errors and VIF, and robust-z anomalies on
    the seasonal residual. Each ships goldens for its misleading-data cases
    (incomplete latest period, high-cardinality dimension, Simpson's paradox,
    leakage, collinearity, too many points flagged) and a mutation test.
  - `cohort_retention` and `ab_test` complete the ten: the cohort × period
    matrix with a size-weighted retention curve (an unfinished latest period
    left out), and the two-proportion z / Welch t comparison of experiment
    arms with intervals, power at the observed rows and the sample-ratio
    check that refuses a mis-split experiment. `ab_test` is the one recipe
    whose wording may say "effect", and only after that check passes; the
    gate vocabulary gains `min_cohorts` and `power`.
  - Every recipe's notebook now ends with an `evidence` cell that prints the
    report's cited evidence leaves as JSON, and the parity test
    (`tests/recipes/test_notebook_parity.py`) executes each reported golden's
    notebook in a fresh kernel through `avaloka.recipes.notebook_exec` and
    proves it reproduces the report's figures; nbclient, nbformat and
    ipykernel are the `avaloka[recipes-test]` extra.
  - **Outcome Report schema 1.1**: three optional fields a 1.0 reader
    ignores — `data_checks[]` (the recipe's quality gates, each with its
    verdict and its deciding number as a verified figure), `provenance.recipe`
    and `provenance.engine` (duckdb, scipy, statsmodels, pandas versions). The
    HTML and Markdown exports render them as an **Evidence** / "Source-data
    checks" section and footer lines; a 1.0 document renders as before. Every
    recipe report now carries them, and `build_report` raises on a headline,
    bullet or method template over its slot's limit instead of cutting it.
  - **Bounded execution** (`avaloka/recipes/budget.py`): every recipe run
    has a `Budget` it cannot exceed — rows and bytes (the ONLINE lane's
    100k rows / 50 MB; `AVALOKA_RECIPE_*` overrides), DuckDB `memory_limit`
    and `threads`, and a wall clock. A larger source is analysed on a
    deterministic reservoir sample and the report says so (`sampling`
    caveat, `method.sample` with the real counts); past the sampled lane's
    ceiling the run refuses `too_large` (new in `REASONS`); past its clock
    it refuses `engine_timeout` — in plain words, never a traceback.
    A sampled run is `qualified`, and the notebook it ships takes the same
    sample (`SAMPLE_ROWS` / `SAMPLE_SEED` in its `parameters` cell, the
    engine's own statement in its load cell), so it still reproduces the
    report. `RunResult.budget_report` records what was spent; `load_and_run`
    no longer reads the whole file into pandas. A `heavy` test proves a
    5,000,000-row file ends `qualified sampling` within the budget.
- **Intent parsing for Instant Analysis** (`avaloka.recipes.intent`): a
  recognizer that checks a model's JSON against the closed recipe
  vocabularies and never coerces, a deterministic heuristic parser with a
  causal-question flag, and a generic binder that returns one Ask or a named
  Refusal; `intent_model` retries a rejected reply once with the reason. 129
  golden questions and the no-spend `instant_intent` benchmark track score it.
- **The conversational probes now measure the product.** A new
  `conversational_product` benchmark family drives the eleven probes through the
  compiled agent graph and grades the reply a user would see. It replays a
  committed recording (gpt-oss through OpenRouter, the router's defaults) by
  default, so it needs no model and no network, and `--live` runs it on a real
  model (local first, then Groq, never Anthropic). Where the application is not installed it reports SKIP. CI runs the
  replay in the Python test step.
- `/instant` ("Ask" in the rail): the Instant Analysis landing surface — drop a
  CSV/XLSX/XLS or pick one of three sample datasets, see the upload's data
  profile (rows, columns, per-column type / missing / distinct, quality
  concerns from `GET /api/datasets/{id}/profile`), then ask one question. The
  question submits to a typed placeholder in `ui/src/lib/api/instant.ts`
  that answers "coming soon" until the recipe endpoint lands (PR-6).
- `/instant/:runId`: the report page a question lands on. The run's verified
  HTML export renders inside `<iframe sandbox="allow-scripts">` (one
  renderer, sized by the export's own height message); an Evidence panel
  lists each source-data check by label and status word with the report's
  caveat sentence, and carries no number; a Limitations panel shows the
  caveats, what did not verify and the method summary as the report wrote
  them; downloads for the `.html` and `.ipynb` (and `.md` when present); a
  question-back renders its options as chips that ask again; a refusal points
  at Deep Investigation. Share is disabled until saved reports (PR-13) and the
  paywall card reads its copy from one config object with no price (PR-8).
### Changed

- **The API authenticates a request before it validates it.** An anonymous
  request used to be answered by body or query validation (422 naming the
  missing fields) or by a router's own store and body checks (200, 400)
  before the handler looked at the caller. A gate in front of routing
  (`app/api/auth_gate.py`) now answers 401 first; a signed-in caller is
  validated exactly as before. Public without a token: `/health`, `/version`,
  `/api/capabilities`, `/analytics/config` and the local-auth sign-in routes.
- **`AVALOKA_DEPLOYMENT` declares who operates an install**: `self_hosted`
  (your own) or `avaloka_hosted` (the Avaloka-operated sites). It is read in
  one place (`app/core/editions.py` `deployment_state()`) by billing and plan
  caps, local auth, and the API docs gate. The Helm chart (`config.deployment`)
  and the Avaloka services in `deploy/compose/docker-compose.scheduler.yml`
  now default it to `self_hosted`; the quickstart already set it. **Upgrade
  note:** an install that sets nothing is *undeclared*: every check keeps the
  default it had before, and the server logs one warning at the first check.
  On `avaloka_hosted`, `/openapi.json`, `/docs`, `/docs/oauth2-redirect`,
  `/redoc` and `/debug/whoami` need a login; self-hosted and undeclared
  installs keep them public. See docs/deployment.md.
- **`conv_avaloka_behaviour_is_unmeasured` now passes only on a measurement**: a
  record per probe, its own question sent, the graph walked to its end, a model
  reached, no request without a recording. It used to pass once a factory with
  the right name returned.

### Fixed
- **The conversational agent passes all eleven product probes on the recorded
  model** (`openai/gpt-oss-120b` through OpenRouter; `tests/conversational/`,
  replayed through the compiled graph). An ambiguous question about results in
  hand is asked in code's words -- the planner's own measure check -- so no
  candidate figure reaches the reply; the direct reply sees the last three
  exchanges and keeps a verified figure when the user pushes another; a
  correction to the method ("you excluded refunds") always re-plans, with the
  original approach, the corrected one, the reason and the alternative written
  from code-read facts (a model's phrasing is used only when it keeps to them);
  new work asked with no dataset connected is stopped before the planner and
  the reply says what data is needed, that a prior artifact it refers to ("the
  cleaned dataset") does not exist yet, and the product's own time estimate;
  a delegated turn's hand-off line reaches the chat; a wrong premise is called
  wrong before the figures. Scorers: a digit inside a word is not a sense of
  scale, a non-breaking-hyphen "re-run" is a re-plan, saying a segment is absent
  is a refusal to invent, and the turn's every assistant message is graded.
  `known_defects.py` lists no `conv_product_*` entry any more.

## [2.0.0] — 2026-10-17

The 1.7 development line, released publicly as **2.0**. The public repository
and the internal branch now carry the same number: `VERSION`,
`pyproject.toml`, the Helm chart, `avaloka --version` and `GET /version` all
say `2.0.0`, and the OSS edition no longer reports a separate 1.0.x line.
What 2.0 adds over 1.6 — the public swarm with per-clone telemetry, the agent
trace API and UI, the planner graph drawn from data operations, the DAB agents
and the benchmark harness — is itemised in
[docs/releases/2.0.md](docs/releases/2.0.md).

Image builds from this line publish `2.0.0` and `2.0` tags alongside the
branch, `sha-*` and content tags; the version comes from `VERSION`, so the 1.6
line keeps producing 1.6 tags.

The entries below are the `develop-1.6` fixes since 1.6.0 that 2.0 carries.
Every one is a change that broke, or would have broken, a clean install or a
first run.

### Security

- **Cloud storage keys are sealed, never stored in plaintext.** The UI wrote
  `access_key`/`secret_key` straight into `cloud_datasets` from two places, the
  table's CHECK allowed it, and the `encrypt_cloud_credentials` RPC -- callable
  by `anon` -- stored whatever text it was handed as "ciphertext" with no IV.
  Migration `20261116130000` drops the plaintext branch of the CHECK (added
  `NOT VALID`, so rows written earlier survive until sealed) and retires the
  RPC. `POST /api/cloud-connections` is now the one place a key enters the
  table, sealed in process under `DB_ENCRYPTION_KEY` (503 without the key,
  never a plaintext insert); both UI writers post there. Rows written before
  the migration are sealed at API start-up and by
  `python -m app.api.cloud_connections seal` (a dry run that prints counts
  only; `--apply` seals). Every reader (`get_cloud_connection`, the Ray job
  submitter, the Kubernetes secret builder, the planner) opens the keys in
  memory as before. **Hosted databases may hold plaintext rows today; run the
  dry run, then `--apply`.**
- **The MCP server's onboarding and admin routes require authentication.**
  `/customers/*`, `/admin/customers`, `/connect` and `/connections` were open
  to anyone who could reach the MCP server. They now take the customer's API
  key or the API's `MCP_SERVICE_TOKEN`, and are scoped to the caller's own
  connections; a deleted connection's key is refused at once. **Set
  `MCP_SERVICE_TOKEN` and rotate stored database credentials**; see the
  advisory in [SECURITY.md](SECURITY.md)
  (`app/mcp_server/customer_dbs.py`, `app/mcp_server/multi_tenant_mcp_server.py`).
- **Connections registered before owners existed keep working for transfers.**
  After the change above, the API's service could no longer fetch credentials
  for a connection bound to no owner, so database-to-database transfers from
  or into it failed. The service may now use such a connection on a user's
  behalf for status, test-connection and credentials only, and the first
  connect binds it to that user. Tested against real Postgres databases
  (`tests/test_customer_dbs_dta_e2e.py`).

### Changed

- **The deployment type is declared, and a failed plan lookup on a hosted
  deployment answers 503.** `AVALOKA_DEPLOYMENT` (`self_hosted` or
  `avaloka_hosted`) is now the one signal the plan gates, billing and the
  plan-lookup fail mode read (`app/core/editions.py`). The chart renders it
  from `config.deployment`, default `self_hosted`, and the scheduler compose
  file defaults it too. On `avaloka_hosted`, a plan lookup that fails refuses
  data transfer, model training and model inference with `503 plan check
  unavailable` (chat says the check could not run) instead of allowing them;
  it never asks the user to upgrade. Self-hosted and undeclared installs are
  unchanged: premium features are allowed, and an undeclared install logs one
  warning. See [Deployment type](docs/deployment.md#deployment-type)
  (`app/core/plan_gate.py`, `app/api/server.py`, `app/agents/planner.py`).

- **The LangGraph server has its own image, `avaloka-langgraph`.** It is the API
  image's build plus `langgraph-cli[inmem]` (`requirements-langgraph-server.txt`);
  the API image no longer carries the CLI, whose langgraph-api 0.5.x capped
  cryptography below 45 and held the API at 44.0.3 with four high advisories.
  The API now resolves cryptography 50. The chart's langgraph pod derives the
  image from `image.repository` (`avaloka-api` -> `avaloka-langgraph`, same
  tag); **if you push images to your own registry, push `avaloka-langgraph`
  too**, or set `langgraph.image.repository`. Running `langgraph dev` locally
  now needs `pip install -r requirements-langgraph-server.txt`.
- **Ray 2.56.0 and pyarrow 23.0.1, everywhere Ray is pinned.** The app client
  (`ray[air,serve]==2.56.0`), the cluster image (`rayproject/ray:2.56.0-py311`),
  the RayCluster/RayService/RayJob `rayVersion` fields, the MTA training and
  inference images, the DTA runtime and the per-user clusters' default image
  all name the same release; `tests/test_ray_version_alignment.py` walks the
  tree so none can drift again. Ray Train stays on V1 (`RAY_TRAIN_V2_ENABLED=0`
  in the images that run `ray_job.py`). `getdaft`, a shim for the `daft` the
  app already pins, is no longer listed; `python-jose` leaves the MCP
  requirements. mlflow 3.17.2, fsspec/s3fs/gcsfs 2026.6.0, urllib3 2.8.0,
  dulwich 1.2.9, pytest 9.1.1; the UI moves to zod 4 and its lockfile carries
  no known critical or high advisory.
- **MLflow and Ray require authentication and are cluster-internal.** The
  chart runs the tracking server as `mlflow server --app-name basic-auth`
  (`mlflow.auth`, on by default; the API image installs `mlflow[auth]`), with
  the admin credentials generated into the release Secret and handed to every
  client by `envFrom`/`secretKeyRef`; its NetworkPolicy admits only the api,
  langgraph, celery, celery-redbeat and Ray pods. Ray runs with
  `RAY_AUTH_MODE=token` (`ray.auth`, on by default): one generated token in the
  release Secret and in `<release>-ray-auth`, read by every Ray container, the
  autoscalers and the RayJobs' own Secrets; the app's dashboard calls send it
  (`app/infra/ray_auth.py`). **KubeRay operator 1.6.2 is required** (the first
  that reads the token from a named Secret); `install_kuberay_operator()`
  applies the CRDs server-side before upgrading it. The kind-only dashboard
  NodePort stays opt-in (`RAY_DASHBOARD_EXPOSE_NODEPORT=1`) and is
  token-checked. See [Ray and MLflow authentication](docs/deployment.md).

- **The Celery worker has its own Kubernetes service account identity.** The
  chart can annotate it for cloud workload identity
  (`celery.serviceAccount.annotations`), and its Role may create and watch
  RayJobs for scheduled Ray analyses (`celery.rbac.rayJobs`, on by default).


- **Python floor raised to 3.11, ceiling pinned below 3.13.**
  `app/services/session_service.py` uses `async with asyncio.timeout(...)`,
  added in 3.11; on 3.10 it raised `AttributeError`, `save_session` swallowed
  it and returned `False` — a silent loss of session state. The upper bound
  exists because `pyarrow==16.1.0` publishes no wheel past cp312, and without
  a cap pip tries to build Arrow C++ from source. `pyproject.toml`, the CI
  matrix, `scripts/install.sh` and `scripts/doctor.py` now all state the same
  range.
- **Default inference provider is OpenRouter**, resolved key-aware: OpenRouter
  if its key is present, else Groq, else OpenAI, else OpenRouter so the choice
  still reports itself. One key now fronts many models, instead of a first run
  requiring a Groq account specifically. An explicit `INFERENCE_PROVIDER` is
  never second-guessed, and Groq remains fully supported
  (`app/core/inference.py`).
- **SQLAlchemy's upper cap removed and `psycopg[binary]` added.** SQLAlchemy
  2.1 changed which DBAPI a bare `postgresql://` URL selects — psycopg2 on
  2.0.x, psycopg v3 on 2.1.x — so the v3 driver must be installed for that URL
  to keep working (`requirements.txt`).

### Fixed

- **`avaloka train` works on a text-label target.** A classification target
  of words -- `healthcare_dataset.csv --target "Test Results"`, the example in
  `docs/INSTALL.md` -- crashed with "Cannot cast array data from dtype('O') to
  dtype('int64')". Labels were encoded only when the column's dtype was
  `object`, and the target arrived as pandas' string dtype. Any non-numeric
  label is now encoded (`avaloka/modeling.py`).
- **pandas is capped below 3** in `pyproject.toml`, `requirements.txt` and
  the five image/server requirements files. `>=2.0` let a clean install pick
  3.0.6, on which the hermetic gate fails 13 more tests (preparation agent,
  PII, null policy, local inference): pandas 3 reads text as its own `str`
  dtype and much of the code asks `dtype == object`.
  `tests/test_pandas_ceiling.py` keeps the cap in every file.
- **A null-like word in a spreadsheet stays a category in the CLI** (#479).
  The CLI's Excel reader read `None`, `NA`, `N/A` as missing in every column;
  it now follows the CSV rule from #474, and that rule also recognises
  pandas 3's `str` dtype (`avaloka/io/loader.py`,
  `file_handler/null_policy.py`).
- **The CLI no longer answers a measure it could not find with a row count**
  (#475). "Total revenue by region" over `gross_revenue` and `net_revenue`
  counted rows per region and reported `status="answered"`; it now reports
  `limited` and names the candidate columns. Count goals and goals that name
  the column are answered as before (`avaloka/goal_analysis.py`).

- **A model that does not beat its baseline is no longer announced as a success.**
  Training through the API reported "The training was successful" and offered
  predictions for a model below its trivial baseline (accuracy 0.627 vs 0.668),
  although the evaluation report said so. The report now states the comparison
  and says not to use the model; the claim verifier also rejects success wording
  for a below-baseline model (`app/agents/mta_v2/agent.py`,
  `app/agents/claim_verifier.py`).
- **The claim verifier catches a wrong "lowest" or "highest".** "The west region
  has the lowest at 68.12" passed beside a result whose minimum was 67.73. A new
  check compares each superlative with the result column the figure came from.
- **`tool_choice` is now chosen per provider**, not per client class. `local`,
  `openrouter` and `openai` all build the same `ChatOpenAI`, so the class name
  cannot tell them apart, and the two that matter need opposite values:
  OpenRouter rejects `required` (measured: 0/3 tool calls, 3/3
  `finish_reason="error"`), while Ollama returns prose instead of a tool call
  without it. `tool_choice_for_provider()` in `app/core/inference.py` keys the
  decision on the provider.
- **The planner and MTA broke when no API key was set at all** —
  `TOOL_CHOICE` was left unbound, so the code raised `NameError` instead of
  reporting a disabled LLM.
- **Training through the API works without MLflow configuration.** The trainer
  and the model endpoints defaulted to MLflow's file store, which the pinned
  MLflow refuses to write; every training turn failed and `GET /api/models`
  returned 500 unless `MLFLOW_TRACKING_URI` was set. Both now default to a
  local SQLite `mlflow.db` (`app/agents/mta_v2/tracking.py`), and the variable
  is documented in `docs/INSTALL.md`.
- **The API image runs the dependency set it pins.** An unpinned
  `pip install langchain-google-vertexai` after the requirements upgraded
  langchain-core to 1.x and pyarrow to 23 inside the published image. It is
  removed (Vertex is optional; see `requirements-cloud-llm.txt`), the build now
  runs `pip check`, and the arm64 image gets an arm64 `kubectl` instead of an
  amd64 binary it could not execute (`deploy/docker/Dockerfile.api`).
- **A cold clone could not install.** `requirements.txt` pinned plain
  `psycopg2`, which publishes no wheels, so pip built it from source and the
  install died.

### Added

- **A Ray jobs panel** in the UI, backed by a `ray_jobs` table (a new Supabase
  migration, also applied by the Helm chart). Scheduled analyses that are not
  Ray jobs run in an executor pod.
- **Auto Insights across several datasets or sheets** in one analysis.
- **The benchmark declares its known defects**, and its exit code follows
  them strictly, as `xfail(strict=True)` does: a declared run that fails
  does not fail the run; an undeclared failure does, as before; a declared
  run that passes fails it too, until its entry is removed. Before this, the
  eleven known-red runs made the pull-request Benchmark Scorecard step and
  the public CI `benchmark` job red on every pull request, so a real
  regression looked the same as the known ones. The scorecard's counts do
  not change; `--strict` restores exit 1 on any failure
  (`avaloka/benchmark/known_defects.py`).

- **A live CI tier**, because a hermetic gate cannot see provider behaviour:
  `.github/workflows/live-provider-weekly.yml` calls real hosted providers and
  `live-local-model-nightly.yml` exercises a local model server, both driving
  `tests/live/`. Neither runs on pull requests, so a provider outage cannot
  redden a PR.
- **The public tree is now generated**, not hand-maintained:
  `scripts/generate-oss.sh` builds it from `develop-1.6` per
  `oss/manifest.yaml`, which carries eight exclusion rules — several covering
  whole directories — plus an overlay and a set of assertions the generated
  tree must satisfy. Editing the public tree directly is not supported: the
  next generation overwrites it. Publishing is one squash commit on top of the
  public `main`; the intermediate `oss/1.6` branch and the generator's `--write`
  mode are retired, and the dry run compares against the published tree. See
  [`docs/oss-branch-model.md`](docs/oss-branch-model.md).

### Removed

- **Four documents retired from `docs/`**, all stale on both branches rather
  than merely internal: `OSS_LAUNCH_READINESS.md` (a launch plan for a date
  that has passed, naming internal project identifiers, with its go/no-go
  boxes unticked), `TEST_PLAN-1.5.2.md` and `TEST_REPORT-1.5.2.md` (58 KB of
  QA material for `feature/k8s-deploy-1.5.2`, three of whose four Day-0
  blockers are now false), and `TASK-conversational-friendliness.md` (an
  internal ticket measured against a harness that is not on this branch). Git
  history keeps all four. [`docs/DOCS_MAP.md`](docs/DOCS_MAP.md) is the new
  index of what remains.

### Known issues

- **Celery's soft time limit is above its hard limit** (1860 s and 1800 s in
  `app/core/celery_app.py`), so a long task is stopped at 30 minutes without
  the soft-limit warning that lets it clean up. A fix is in progress.

- **A multi-sheet workbook is read from its longest sheet**, in the upload
  path (`file_handler/excel_connector.py::_resolve_sheet`), whatever the
  question is about. On a 10-K-style workbook that is the balance sheet when
  the question is about the income statement; a workbook whose decoy sheet
  is longer than its answer sheet is read off the decoy. The frame parses
  cleanly, so nothing in the result says so. Workaround: upload the sheet on
  its own. The fix is to choose the sheet where the question is; the eight
  tests that grade it are strict xfails in
  `tests/test_spreadsheet_comprehension_upload_path.py`.
- **The CLI's spreadsheet reader handles fewer shapes than the upload
  path**: two-row and merged headers, footnote markers, subtotal and total
  rows, a second table below the first, `(1,234)` negatives and a
  "thousands" scale stated in a title are not understood. Ten benchmark
  tasks measure this and fail, as does the check that the conversational
  benchmark does not yet drive the product. The eleven are declared in
  `avaloka/benchmark/known_defects.py`: they still count as failures in the
  scorecard (39/50), but no longer fail the run. See
  [`docs/benchmarks.md`](docs/benchmarks.md).
- **MinIO's images are not currently pullable** from the registries the Helm
  chart points at, including by digest, so the chart's default in-cluster
  object store does not come up unaided. A migration to an alternative object
  store has been proposed but is **not merged**.

## [1.6.0] — 2026-09-16

The 1.6 development line. Numbered to match `VERSION`, the Helm chart and the
version `GET /version` serves — those had drifted to five different answers
(DEFECT E14.01), and a release bump now moves them together.

### Fixed

- **A forgeable auth stack shipped as the chart default** (DEFECT E9.19, P0).
  `values.yaml` enabled in-cluster Supabase with Supabase's *published* demo
  signing key, which `secret.yaml` adopts as the API's `SUPABASE_JWT_SECRET` —
  so anyone could mint a `service_role` token. Supabase is now off by default,
  credentials ship empty, and the chart refuses to render if you enable it
  without supplying your own.
- **One test file aborted the whole test suite.** A module-level `sys.exit(1)`
  in `tests/planner_graph_agent/` made pytest report INTERNALERROR and run
  nothing at all, which is why CI could not pass.
- **The API server no longer imports Daft to start.** Daft was imported at
  module scope for a single availability flag; on any interpreter whose wheel
  does not match the CPU, that import raises SIGILL rather than ImportError and
  kills the process.
- **Vendored loader copies had drifted 96 lines** — the inference image was
  missing the non-UTF-8 encoding fallback entirely, so a CSV that loaded in the
  agent failed in the deployed service.

### Changed

- Images publish to GHCR with content-addressed tags, so a deployment pulls what
  already exists instead of rebuilding it.
- `scripts/doctor.py` reports whether the environment will actually work,
  including an x86_64-under-Rosetta Python, which breaks native wheels in a way
  that looks like a hang.

## [0.2] — 2026

Avaloka as an agentic team of data scientists that takes any dataset from raw
data to model inference.

### Added

- **Multi-agent workflow** on LangGraph: Planner, Planner Graph, Summarizer,
  Sampling, Profiling, Coder, Validator, Execution, Data Transfer (DTA), Model
  Training (MTA v1 & v2), Visualization, and Scheduler agents coordinating
  through one shared `ETLState`.
- **Pseudocode-first code authoring** with a three-layer validation gate
  (syntax, schema-aware static analysis, logical LLM review) and deterministic
  fallbacks when the LLM is unavailable.
- **Multi-mode execution** — local, Kubernetes Job, or Ray cluster — chosen
  automatically from plan context and analysis fidelity.
- **Kubernetes deployment on any cluster** via a multi-cloud provider
  abstraction (local `kind`, GKE, EKS) behind a single `provision` / `connect`
  bootstrap, with a Helm chart, Ray/KubeRay integration, and a one-command
  `make up`.
- **Ray Serve inference-as-a-service** (`RayService`) and an MTA v2 managed
  inference path.
- **React.js web UI** (built with Lovable) in the `ui/` directory, backed by a
  local Supabase database, over the REST API.
- **Mission-planning CLI** (`python -m app.interfaces.cli.main`) and an **MCP
  server** sharing one core with the REST API, plus a multi-tenant MCP server
  for customer database connections.
- **Data connectors** for CSV, Excel, Parquet, Avro, Delta Lake, Apache Iceberg,
  JSON, and XML.
- **Sampling & profiling** (Daft-powered portfolio samples, semantic profiling)
  and **RAG-powered code retrieval**.
- **Asset persistence** to GCS/S3 with signed-URL retrieval; MLflow experiment
  tracking and model registry; Celery/RedBeat scheduling.
- **Open-source project setup**: Apache-2.0 license, documentation set
  (architecture, deployment, API, CLI, testing, version matrix), and
  contribution/security/conduct guides.

### Roadmap

*As stated at the time. Two of these have since moved — see Unreleased.*

- Azure AKS provider. **Done**: `app/infra/providers/azure_aks.py`, wired into
  `app/infra/providers/factory.py`.
- CLI remote mode (`--endpoint`) to drive a deployed backend. **Partly done**:
  the mission-planning CLI (`python -m app.interfaces.cli.main`) takes
  `--endpoint`, defaulting to `AVALOKA_API_URL`; the `avaloka` console script
  does not.
- Loading real trained models into the Ray Serve inference app by default.
  **Still open**: `app/serve/inference.py` reports its backend as
  `deterministic-stub` when no model is loaded, which keeps the platform
  exercisable without claiming a model is being served.
