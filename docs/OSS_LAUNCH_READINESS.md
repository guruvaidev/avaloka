# Avaloka 1.6 — Open-Source Launch Readiness

| | |
|---|---|
| **Target** | 2026-09-14 |
| **Written** | 2026-09-07 · 5 working days remain |
| **Branch cut** | `oss/1.6` — from `develop-1.6` @ `83dcd360`, `.env` untracked |
| **Engine branch cut** | `feat/starrocks-duckdb-compat` |
| **Scope** | Two separate deliverables: a public git repository, and a hosted trial at avaloka.ai |

---

## 1. The two tracks, and why they must not be confused

| | **OSS repository** | **Hosted trial (avaloka.ai)** |
|---|---|---|
| What it is | Source, Apache 2.0, self-hosted | A running instance on GCP a newcomer can try |
| Who runs it | The user | Us |
| Auth | Their own Supabase or local | Our Supabase project |
| Compute | Their machine or cluster | Our GCP bill |
| Data | Theirs, never leaves | Theirs, on our infrastructure — **a privacy commitment we must publish** |
| Blocks the other? | No | No |

These ship independently. **The repository must not contain our hosted
instance's identity, and the hosted instance must not be the only way the
product works.** Every default that points at our project is a defect in the
first track (see §3).

---

## 2. The blocker — history, not code

```
.env tracked on develop-1.6 : YES, across 49 commits
credentials in it           : Supabase service_role JWT (valid to 2035),
                              the JWT signing secret, a Postgres URI
```

`git clone` fetches all history, so publishing this branch publishes the key.
Untracking (done on `oss/1.6`) stops new exposure and does **not** remove the
blob from history.

### Decision: fresh repository, squashed initial commit

Not a history rewrite. The reasoning:

| | History rewrite | **Fresh repo, squashed** |
| --- | --- | --- |
| Review surface | 49 commits in a long merge history | **one working tree** |
| Risk of missing something | real | near zero |
| Invalidates clones, forks, open PRs | yes | not applicable |
| Drops the 21 MB embeddings DB from history | separate work | free |
| Fits five days | doubtful | yes |

`avaloka-dev` stays private with its full history and contributor record. The
public repo starts at commit one. Contributors are credited in a `CONTRIBUTORS`
file rather than by rewritten authorship.

### Rotation is separate and more urgent

The leaked credential stays valid until the **JWT secret** is regenerated —
rotating the service-role key alone is insufficient, because anyone holding the
signing secret can mint a new one. Check Supabase auth logs for `service_role`
use since 2026-03-22 **before** rotating; rotation destroys that evidence.

---

## 3. What still leaks in the tree

`.env` was the obvious one. These are not:

| Where | What | Fix |
| --- | --- | --- |
| 7 Python runtime defaults | `os.getenv("GCP_PROJECT_ID", "ai-playground-440519")` | **PR #294** — but it now conflicts on `ray_trainer.py` and needs a rebase |
| 2 YAML manifests | Same project id in `rayjob_data_transfer.yaml`, `infra-agent-deployment.yaml` | Not covered by #294 — **new work** |
| 8 docstrings and comments | Project id as provenance | Cosmetic, but it names internal infrastructure — strip |
| `ui/src/integrations/supabase/config.ts` | **Our Supabase URL and anon key as hardcoded fallbacks** | See below — this is the UI equivalent of #294 |

### The UI default is the same defect as #294

```ts
export const SUPABASE_URL = runtimeEnv.SUPABASE_URL || "https://yqsrqpfulmtplvqjckaz.supabase.co";
export const SUPABASE_ANON_KEY = runtimeEnv.SUPABASE_ANON_KEY || "eyJhbGciOi…";
```

The anon key is publishable by design, so this is not a credential leak. It is
worse in a different way: **an OSS user who does not set the env vars gets a UI
authenticating against our hosted project.** Their accounts land in our
database, and our project absorbs their traffic. The fix is the same one #294
applied to GCP — fail with a message naming the variable rather than defaulting
to something private.

---

## 4. PR triage for 09/14

Twelve PRs are open to `develop-1.6`. They will not all land safely.

**Must ship**

| PR | Why | State |
| --- | --- | --- |
| **#294** cloud config | Stops the repo shipping our GCP project | ⚠️ **conflicts — rebase needed** |
| **#261** editions | The OSS/commercial boundary itself | clean |
| **#296** user guides | An OSS release without docs is not one | clean |

**Ship if convenient** — all clean: #264 sync, #306 design docs, #290 prompt
suite, #307 lineage, and the preparation chain #297 → #300 → #303 (merge
bottom-up).

**Needs a small conflict resolution:** #288 docs sync (README), #289 k8s deploy
(requirements.txt).

**Cut to 1.6.1**

- **#273** conversational agent — 109 commits behind, 14 conflicts, #282 stacked on it. Rebasing that and shipping it in the release the world will read first is the highest-risk item on the list, and cutting it costs a feature rather than the launch.
- **#286** local fallback profiles — depends on the same module, and its defaults are OpenRouter slugs that do not resolve on Ollama.

---

## 5. The new UI

The UI is a TanStack Start / React 19 / Tailwind 4 application, 598 files,
authenticating through Supabase. Four things stand between it and a public
release.

**5.1 The hardcoded project (§3).** Blocking. Same fix as #294.

**5.2 Third-party logos.** `ui/public/assets/logos/` contains marks for Azure,
BigQuery, Cloud SQL, ClickHouse, Databricks, GCS, MariaDB, Microsoft SQL,
MongoDB, MySQL and PostgreSQL. Displaying a vendor's mark to identify their
product is ordinarily fine; **redistributing the image files under Apache 2.0 is
a different question** and needs a decision from whoever owns legal. The cheap
alternative is to reference official CDN assets or use text labels.

**5.3 Nineteen missing screenshots.** The user guide ships with placeholders
because the previous captures predate the current UI. A guide with nineteen
`[Screenshot placeholder]` markers is not a launch artifact. Half a day of
capture work against a seeded demo dataset; the manifest naming each one is
already in `docs/images/user-guide/README.md`.

**5.4 A self-hosted auth story.** `feature/webui_localsupabase` (#269, merged)
means local Supabase exists — but the OSS path needs to be *documented and
tested*, not merely present. The first-run experience for someone with no
Supabase account is the single most important path in the release.

---

## 6. The hosted trial (avaloka.ai)

Independent of the repository, and it needs its own decisions.

| Item | Note |
| --- | --- |
| Instance shape | A single GKE deployment of the same chart, or a VM. Prefer the chart — the demo then exercises the code the repo ships. |
| Cost containment | This is a public endpoint with LLM calls behind it. It needs a hard per-session budget, a request rate limit, and idle scale-down. The autoscaling policy in **#299** is directly applicable. |
| Data policy | Users will upload real data to a demo. **We must publish what happens to it and delete on a schedule.** This is the single largest reputational risk of the trial. |
| Isolation | Separate GCP project and Supabase project from anything internal. The demo must not share identity with production. |
| Abuse | Anonymous upload plus model access is an obvious target. Session caps, file size limits, no outbound network from executed code. |
| Sample data | Ship the demo pre-seeded with a Kaggle set so a visitor sees value in thirty seconds without uploading anything. |
| Content | Reset on a schedule. A demo instance accumulating strangers' datasets is a liability. |

**Recommendation:** the trial should not gate 09/14. Publish the repository on
the 14th and the trial when its data policy is written and its budget controls
are in place. Shipping a public endpoint with no spend ceiling in five days is
how a demo becomes an incident.

---

## 7. Engine compatibility — separate branch

`feat/starrocks-duckdb-compat` is cut with compose files for StarRocks and
DuckDB, plus [ENGINE_COMPATIBILITY.md](ENGINE_COMPATIBILITY.md).

The useful finding: **StarRocks speaks the MySQL wire protocol**, so registering
and querying may work today with no new driver — worth verifying before anything
is built. The work is in the sink: `BaseSQLDataSink` emits MySQL DDL, and
StarRocks rejects `CREATE TABLE` without a key and distribution clause. Loading
matters as much as DDL — row-by-row `INSERT` into a columnar engine passes a test
and is unusable in production, so the sink should use Stream Load.

DuckDB is the cheapest and the best fit for the OSS tier: in-process, no server,
which is exactly the deployment profile the release targets.

MongoDB needs an audit before a claim. `pymongo` is a dependency and five modules
reference it, but there is no sink, and the documentation review could not
substantiate the support the guide originally claimed.

**None of this gates 09/14.** It is 1.6.1 or 1.7 work.

---

## 8. Five-day plan

| Day | Work | Owner |
| --- | --- | --- |
| **Mon 8** | Check Supabase auth logs, then rotate JWT secret + service-role key + Postgres password. Rebase **#294**. Fix the UI Supabase default. | Security · Platform |
| **Tue 9** | Merge #294, #261, #296. Hand out the 126-case test plan, one sheet per team. Sheet 6 works CV-07/CV-08 first. | Release · all teams |
| **Wed 10** | Execute tests. Capture the 19 screenshots. Strip project ids from YAML and docstrings. | All · Frontend |
| **Thu 11** | Triage P1 failures. Build the clean tree. **One person reads the entire tree** — that review is the whole safety argument for the squash approach. | Release lead |
| **Fri 12** | Create the public repository, squashed initial commit. Run the P1 subset **against a clean clone** — this is what catches "works because my machine has a file the repo does not." | Release · QA |
| **Mon 14** | Publish. | — |

---

## 9. Go / no-go

**Blocking — must be true on the 14th**

- [ ] Supabase JWT secret rotated; old credential dead
- [ ] No `.env` and no history containing it in the published repo
- [ ] No default anywhere resolves to our GCP project or Supabase project (#294 + the UI fix)
- [ ] `install.sh` → upload → prompt → result works **from a clean clone** with no internal access
- [ ] LICENSE, NOTICE, SECURITY.md, CODE_OF_CONDUCT.md, CONTRIBUTING.md present *(all confirmed present)*
- [ ] Legal decision on redistributing third-party logos
- [ ] P1 test cases executed; failures either fixed or documented as known issues

**Not blocking**

- The 19 screenshots (ship with placeholders and a note if it comes to it — a placeholder is honest, a stale screenshot is not)
- The hosted trial
- StarRocks, DuckDB, MongoDB
- #273 and #286

---

## 10. The thing I would protect

An open-source release is judged almost entirely on whether a stranger can go
from `git clone` to a working analysis in ten minutes. Not on feature count.

That path is `install.sh` → upload → prompt → result, and it must work with **no
internal access, no default pointing at our infrastructure, and no undocumented
prerequisite**. Cases UI-01, UI-04, UI-08, CM-01 and CM-02 in the test plan are
exactly that path.

Run them on a clean clone, on a machine that has never had access to our
systems, before anything else.
