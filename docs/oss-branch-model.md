# How the public branch is produced

`oss/1.6` is **generated** from `develop-1.6`. It is not a parallel branch, and
editing it directly is not supported — the next generation overwrites it.

```
develop-1.6  ──(scripts/generate-oss.sh)──▶  oss/1.6  ──▶  github.com/guruvaidev/avaloka
```

The two remotes this involves are `origin`
(`git@bitbucket.org:kamaliincteam/avaloka-dev.git`, internal) and `github`
(`git@github.com:guruvaidev/avaloka.git`, public) — `git remote -v` in a
checkout of this branch shows both. `SRC_BRANCH` defaults to `develop-1.6`
(`scripts/generate-oss.sh:12`).

## Why this changed

`oss/1.6` had been maintained by hand, merging from `develop-1.6` when someone
remembered. Measured before the switch:

| | |
|---|---|
| Commits on `develop-1.6` absent from `oss/1.6` | 114 |
| Commits on `oss/1.6` absent from `develop-1.6` | 57 |
| Files differing in content under `app/`, `avaloka/`, `tests/` | 61 (+5,791 / −1,199) |
| A test merge of `develop-1.6` into `oss/1.6` | 20 conflicts across 113 files |

Drift that size is not a merge problem, it is a process problem. Three concrete
consequences had already reached the public branch:

- **Third-party LaTeX templates shipped in an Apache-2.0 tree.** `natbib.sty`
  (Patrick W Daly), `iclr2026_conference.bst` (Hal Daumé III), `fancyhdr.sty`
  and the ICLR class file sat on `oss/1.6` for weeks.
- **A stray file leaked a developer's machine.** A 7 KB file named `2.2.0` —
  captured `pip` output — carried a local Windows path and username.
- **Committed test output.** `artifacts/` is listed in `.gitignore` on both
  branches, but the files predate the rule and were never removed.

None of these were decisions. They were things nobody got round to, which is
exactly what a hand-maintained mirror accumulates.

## The failure this is designed to prevent

The public quickstart deliberately does **not** assume `SUPABASE_JWT_SECRET` or
a running `redis-server`; the internal one does. Copying `develop-1.6`'s README
onto `oss/1.6` silently reverts that, along with the scale-boundary language
(`support@avaloka.ai`, "at your own risk", "Not tested on"). That has happened.

So the README is spliced, not copied: `oss/overlay/README.quickstart.md`
replaces the `## Quickstart` section, and five README invariants are asserted
after every generation. A violation aborts before anything is written.

```
== invariants ==
   ok   README.md present: support@avaloka.ai
   ok   README.md present: at your own risk
   ok   README.md present: Not tested on
   ok   README.md absent: SUPABASE_JWT_SECRET
   ok   README.md absent: redis-server
```

Those five are the README's. The manifest asserts four more kinds besides, and
the script fails the run on any of them:

| Manifest key | What it checks |
| --- | --- |
| `assert_present` / `assert_absent` | Literal substrings in named files — the README five, plus the `factory.py` commercial gate, the `.gitignore` rules that guard the withheld paths, and the `pyproject.toml` Python range |
| `assert_no_paths` | Globs that must match nothing in the generated tree: `**/*.sty`, `**/*.bst`, `artifacts/**`, `**/.lovable/**`, `docs/research/**/*.tex`, `avaloka/arena/**`, `tests/arena/**` |
| `assert_no_secrets` | Credential prefixes (`sk-or-v1-`, `gsk_`, `sk-ant-`, `ghp_`) with at least 20 trailing credential-ish characters, plus PEM private-key headers built at runtime so the manifest does not itself trip the scanner |
| `assert_unreachable` | `.env` and `.env.bak.pre-docker-switch` must not be reachable from the ref about to be pushed — a tree check is not enough, see below |

### A literal assertion goes stale when the file it names changes

`assert_present.pyproject.toml` currently asserts the substring
`">=3.10,<3.13"`. `pyproject.toml:16` now reads `requires-python = ">=3.11,<3.13"`
— the floor was raised because `app/services/session_service.py:170` uses
`asyncio.timeout`, which is 3.11+. The substring no longer occurs:

```bash
$ grep -c '>=3.10,<3.13' pyproject.toml
0
```

So the invariant **fails and generation refuses to write** until the manifest is
updated to `">=3.11,<3.13"`. This is the mechanism working — the assertion
exists because `pyarrow==16.1.0` ships cp38–cp312 wheels only and an uncapped
floor sends a 3.13+ interpreter into an Arrow C++ source build — but it is also
the mechanism's cost: a substring assertion has to be maintained alongside the
file it guards. Fix the manifest, do not delete the assertion.

## Why exclusion works here

The public/internal boundary is **path-based, not content-based**. There is no
systematic redaction inside shared files — `internal` appears in 23 files on
both branches, `supabase.co` in 2 on both. Nothing has to be rewritten on the
way out; whole paths are withheld. That is what makes generation safe.

If that ever stops being true — if a shared source file needs different content
in public — this mechanism is the wrong one, and the right move is to split the
file, not to add a content filter.

## Two trees, one source

The goal is not a smaller public tree for its own sake. It is that
`develop-1.6` and `oss/1.6` stay as close as possible, differing only where
there is a reason:

| | `develop-1.6` | `oss/1.6` |
|---|---|---|
| Product code | full | full |
| Enterprise features | yes | withheld |
| Research paper | full conference bundle, buildable | compiled PDFs only |
| Internal planning notes | yes | **partly** — see below |

"Internal planning notes are withheld" is true of exactly one path:
`ui/.lovable/`. It is **not** a general guarantee about `docs/`. The manifest's
`exclude:` list withholds `artifacts/`, `ui/.lovable/`, `avaloka/arena/`,
`tests/arena/`, the stray `2.2.0` file,
`docs/Avaloka-1.6-Master-Test-Plan.xlsx`, `research_paper/` and the sources
under `docs/research/` — and nothing else. Every other file under `docs/` is
published, so **a document that should not be public has to be deleted from
`develop-1.6` or added to `exclude:`; there is no third state.**

Four documents were published that way until the 1.6 documentation refresh, and
they were retired from `develop-1.6` rather than added to `exclude:`, because
each was stale on *both* branches rather than merely internal:

| Retired | Why |
| --- | --- |
| `docs/OSS_LAUNCH_READINESS.md` | A launch plan for a date that has passed, naming the internal GCP project id, the hosted Supabase project URL and twelve internal PR numbers, with its go/no-go boxes still unticked — so a reader could not tell whether the credential rotation it describes ever happened |
| `docs/TEST_PLAN-1.5.2.md`, `docs/TEST_REPORT-1.5.2.md` | 58 KB of QA material for `feature/k8s-deploy-1.5.2`, a branch two releases gone. Three of its four Day-0 blockers are now false — the chart has `celery.yaml`, `mlflow.yaml` and `mcp.yaml` — and its defect register of P0 findings carried no way to tell which entries were still open |
| `docs/TASK-conversational-friendliness.md` | An internal ticket (`Owner: TBD`), whose two headline findings were measured against a `track6` harness that is not on this branch |

Git history keeps all four. The durable lessons were folded into the documents
that are maintained: [`testing.md`](testing.md) for the test tiers,
[`test-reports/three-pillar-coverage.md`](test-reports/three-pillar-coverage.md)
for the conversational probes, and [`DOCS_MAP.md`](DOCS_MAP.md) for what each
surviving document is for.

> **Retiring a QA document does not close the defects it recorded.** Two items
> from the 1.5.2 register were re-verified during the refresh and are still
> live in code; they were handed to the release lead rather than left in a
> stale register. Check the issue tracker, not this directory, for defect
> status — `docs/` has never been the right place for it.

The research split is the clearest case. `develop-1.6` keeps the whole ICLR
bundle — `.tex`, `.bib`, and the templates — so the paper can actually be
rebuilt for a submission. The public tree carries the compiled PDFs, because
the templates are third-party licensed and cannot ship under Apache-2.0. That
is expressed as `keep_globs: ["*.pdf"]` rather than as two diverging trees.

## Reconciling, not overwriting

Generation is one-way, so anything valuable that only exists on `oss/1.6` must
be moved to `develop-1.6` first or it is lost. Before switching, every
oss-only path was checked against develop rather than assumed stale — the
avatar work is the worked example, and the manifest records the date evidence
for why the older implementation goes and the newer one stays.

Run the dry run and read the deletions before every `--write`. A file
disappearing that you did not expect is the signal that something needs to go
back to `develop-1.6` first.

The first pass found eleven files where `oss/1.6` was ahead. They were sorted
by last-touched date on each branch rather than by assuming the internal branch
leads:

| | |
|---|---|
| Moved to `develop-1.6` | `avaloka/version.py`, `tests/plan_execution/host_checks.py`, three dataset tests, `tests/test_supabase_url_default.py`, `tests/k8s/README.md`, `tests/test_oss_readiness.py` |
| Left alone — equivalent fix on both sides | `tests/test_memory_semantics.py` |
| Left alone — cosmetic | `app/agents/mta_v2/training_reply_classifier.py` |
| Moved to `oss/overlay/` — commercial boundary | `app/infra/providers/factory.py` |

`avaloka/version.py` is the one worth noting: `develop-1.6` hardcoded `0.3.0`
while its own `VERSION` file and `pyproject.toml` both said `1.6.0`. The fix
had been sitting on the public branch.

`factory.py` is the shape the rest of the commercial boundary should take. The
public build gates GKE/EKS/AKS on `cloud_provisioning` — a flag
`app.core.editions` already marked `COMMERCIAL` while nothing checked it. Two
invariants assert the gate survives generation.

## Publishing is a squash, never a push of an internal branch

`git push <internal-branch> <public-remote>` carries every commit behind that
branch, no matter how clean its tip tree is. Both `develop-1.6` and `oss/1.6`
have 50 commits touching a `.env` that was force-added in July 2026 and deleted
in August. A tree-level check passes on those branches and tells you nothing.

That is not hypothetical — it happened. A test branch pushed to the public repo
on 29 Sep 2026 tripped GitHub secret scanning on a `.env` that was nowhere in
its tree, only in the history it dragged along. The leaked values were a
Supabase service-role key, the JWT secret, and an MLflow URI with an embedded
password.

So publishing builds **one commit whose only parent is the public branch tip**:

```bash
PUBLISH_REMOTE=github PUBLISH_BRANCH=main scripts/generate-oss.sh --publish
# prints the commit and the exact push command; set PUBLISH_CONFIRM=yes to push
```

Internal commits are never parents, so nothing ever force-added into the
internal history can be reached from the published ref. `assert_unreachable` in
the manifest then verifies that claim against the commit that is about to be
pushed, and refuses if any probe path is reachable.

## Using it

```bash
scripts/generate-oss.sh            # dry run: what would change
scripts/generate-oss.sh --write    # commit onto oss/1.6
```

`oss/manifest.yaml` is the whole contract: what is withheld, what is added,
what is spliced, what is asserted. Change the manifest, not the output.

## Flowing work the other way

Anything genuinely public-first — governance, the DCO, community docs — belongs
in `oss/overlay/`. Anything else that lands on `oss/1.6` and matters should be
committed to `develop-1.6` instead, or it will be erased on the next run. The
57 divergent commits are how the current gap was built.
