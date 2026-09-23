#!/usr/bin/env bash
#
# Everything CI runs, runnable by you, on a clean install.
#
#   ./scripts/ci.sh              the full gate: what must pass before a release
#   ./scripts/ci.sh --fast       hermetic tests only, no benchmark (~1 minute)
#   ./scripts/ci.sh --list       print the stages and exit
#
# Nothing here needs a cluster, a cloud account, an API key or the network.
# Stages that would are excluded by marker, not skipped silently -- if a stage
# is not run, this script says so.
#
# Exit code is the number of failed stages, so it is usable as a gate.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOLD=""; OFF=""; GREEN=""; RED=""; DIM=""
if [[ -t 1 ]]; then BOLD=$'\033[1m'; OFF=$'\033[0m'; GREEN=$'\033[32m'; RED=$'\033[31m'; DIM=$'\033[2m'; fi

FAST=0
for arg in "$@"; do
  case "$arg" in
    --fast) FAST=1 ;;
    --list) sed -n '/^# STAGES/,/^# END STAGES/p' "$0" | sed '1d;$d' | sed 's/^# \?//'; exit 0 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in "$REPO_ROOT/.venv/bin/python" python3; do
    [[ -x "$c" ]] || command -v "$c" >/dev/null 2>&1 || continue
    PY="$c"; break
  done
fi
[[ -n "$PY" ]] || { echo "no python found; run ./scripts/install.sh first" >&2; exit 2; }

FAILED=0
RESULTS=()

#: Per-stage wall-clock ceiling. A stage that wedges must be reported, not
#: endured: a script a user runs has to end, and "still going" after an hour is
#: indistinguishable from "stuck" without one of these.
STAGE_TIMEOUT="${AVALOKA_CI_STAGE_TIMEOUT:-1800}"
TIMEOUT_BIN=""
for c in timeout gtimeout; do command -v "$c" >/dev/null 2>&1 && { TIMEOUT_BIN="$c"; break; }; done

stage() {
  local name="$1"; shift
  printf '\n%s── %s %s\n' "$BOLD" "$name" "$OFF"
  printf '%s   %s%s\n' "$DIM" "$*" "$OFF"
  local rc=0
  if [[ -n "$TIMEOUT_BIN" ]]; then
    # --kill-after matters as much as the timeout. SIGTERM alone is not enough:
    # a process that has died inside a native extension can sit in
    # uninterruptible exit, where no signal reaches it and `timeout` waits
    # alongside it -- which is how a 15-minute bound still ran for an hour.
    "$TIMEOUT_BIN" --kill-after=30 "$STAGE_TIMEOUT" "$@" || rc=$?
    if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
      printf '%s   stopped after %ss — raise AVALOKA_CI_STAGE_TIMEOUT if that is too short%s\n' \
             "$DIM" "$STAGE_TIMEOUT" "$OFF"
    fi
  else
    "$@" || rc=$?
  fi
  if [[ "$rc" -eq 0 ]]; then
    RESULTS+=("${GREEN}PASS${OFF}  $name")
  else
    RESULTS+=("${RED}FAIL${OFF}  $name")
    FAILED=$((FAILED + 1))
  fi
}

# STAGES
# 1. environment      the install is coherent and will actually run
# 2. hermetic tests   no cluster, no cloud, no network, no real LLM
# 3. data science     capability suite -- planted facts, asserted findings
# 4. cli robustness   what the CLI does when the user gets it wrong
# 5. benchmark        ground-truth tasks, scoring the on-disk deliverables
# END STAGES

stage "environment"    "$PY" scripts/doctor.py
# --fast also drops `slow` and `kaggle`, which is what makes it fast: those
# markers exist for multi-minute and download-dependent tests, and the default
# selection did not exclude them.
HERMETIC_MARKERS="not cluster and not cloud and not integration"
if [[ "$FAST" -eq 1 ]]; then
  HERMETIC_MARKERS="$HERMETIC_MARKERS and not slow and not kaggle"
fi
stage "hermetic tests" "$PY" -m pytest -q -p no:cacheprovider -m "$HERMETIC_MARKERS"

# Optional suites. tests/datascience ships on develop-1.6 and not on oss/1.6 --
# the data-science suite is deliberately development-only while the code it
# exercises is synced. Naming it unconditionally meant every open-source CI run
# called pytest on a path that does not exist, which exits 4 and failed the
# gate before a single test had run. A suite that is absent by design is not a
# failure; a suite that is present and failing still is.
optional_stage() {
  local name="$1" path="$2"
  if [[ -d "$path" ]]; then
    stage "$name" "$PY" -m pytest -q -p no:cacheprovider "$path"
  else
    RESULTS+=("${DIM}SKIP${OFF}  $name (no $path in this tree)")
  fi
}

optional_stage "data science"   tests/datascience
optional_stage "cli robustness" tests/avaloka

if [[ "$FAST" -eq 0 ]]; then
  stage "benchmark" "$PY" -m avaloka.benchmark run
else
  RESULTS+=("${DIM}SKIP${OFF}  benchmark (--fast)")
fi

printf '\n%s%s%s\n' "$BOLD" "──────── summary ────────" "$OFF"
for line in "${RESULTS[@]}"; do printf '  %s\n' "$line"; done

if [[ "$FAILED" -eq 0 ]]; then
  printf '\n%sAll stages passed.%s\n' "$GREEN$BOLD" "$OFF"
else
  printf '\n%s%d stage(s) failed.%s\n' "$RED$BOLD" "$FAILED" "$OFF"
fi
exit "$FAILED"
