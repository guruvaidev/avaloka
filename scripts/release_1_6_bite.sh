#!/usr/bin/env bash
# Show that every 1.6 release-suite test can fail ("bite").
#
# Reads tests/release_1_6/bite/manifest*.tsv (one fragment per area), one row per bite:
#   id <TAB> kind <TAB> runner <TAB> patch <TAB> test
#     id      case/bug ID the test certifies (e.g. AG-01, BG13)
#     kind    revert : the patch breaks the certified behaviour (reverts the fix or
#                      mutates the line); the test must FAIL on an assertion.
#             fix    : the test is a strict xfail (pytest) or it.fails (vitest) for
#                      an open defect; the patch is the minimal fix, so the test
#                      must now report a failure (XPASS(strict) / "expected to fail").
#     runner  pytest | vitest
#     patch   file under tests/release_1_6/bite/ (git diff format, repo-root paths)
#     test    pytest node id, or vitest "<file> -t <name>" relative to ui/
#
# Applies each patch to the CURRENT tree with `git apply`, runs the one test,
# reverts with `git apply -R` (also on interrupt), and prints a TSV verdict.
# Run it in a scratch copy or a CI container, never on a tree with local edits
# to the patched files. Usage:
#   scripts/release_1_6_bite.sh [pytest|vitest|all] [ID-filter-regex]
set -u
RUNNER="${1:-all}"; FILTER="${2:-.}"
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT" || exit 2
DIR=tests/release_1_6/bite
APPLIED=""
revert() { [ -n "$APPLIED" ] && git apply -R "$APPLIED" && APPLIED=""; }
trap 'revert; exit 130' INT TERM
pass=0; fail=0
printf 'id\tkind\trunner\tverdict\tdetail\n'
while IFS=$'\t' read -r id kind runner patch test; do
  case "$id" in ''|\#*|id) continue;; esac
  [ "$RUNNER" = all ] || [ "$RUNNER" = "$runner" ] || continue
  echo "$id" | grep -Eq "$FILTER" || continue
  p="$DIR/$patch"
  if ! git apply --check "$p" 2>/dev/null; then
    printf '%s\t%s\t%s\tNO-APPLY\tpatch does not apply\n' "$id" "$kind" "$runner"; fail=$((fail+1)); continue
  fi
  git apply "$p" && APPLIED="$p"
  if [ "$runner" = pytest ]; then
    out=$(python -m pytest -p no:cacheprovider -q -rfEX --tb=short "$test" 2>&1); rc=$?
  else
    # shellcheck disable=SC2086
    out=$(cd ui && npx vitest run $test 2>&1); rc=$?
  fi
  revert
  verdict=MISS; detail=""
  if echo "$out" | grep -Eq 'ERROR collecting|ImportError|ModuleNotFoundError|SyntaxError|Failed to load|Transform failed|no tests ran|No test files found|export is defined on the|Cannot find module'; then
    detail="collection/import error, not an assertion"
  elif [ "$kind" = revert ] && [ $rc -ne 0 ] && echo "$out" | grep -Eq 'AssertionError|assert |Error: expect|expected .* to|toBe|toEqual|toContain|toMatch|FAILED'; then
    verdict=BITES
  elif [ "$kind" = fix ] && [ $rc -ne 0 ] && echo "$out" | grep -Eq 'XPASS\(strict\)|\[XPASS|Expect test to fail|expected to fail'; then
    verdict=BITES
  else
    detail="rc=$rc"
  fi
  [ $verdict = BITES ] && pass=$((pass+1)) || fail=$((fail+1))
  printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$kind" "$runner" "$verdict" "$detail"
done < <(cat "$DIR"/manifest*.tsv)
echo "# bites=$pass misses=$fail" >&2
[ $fail -eq 0 ]
