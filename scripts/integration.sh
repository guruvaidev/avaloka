#!/usr/bin/env bash
# Container-backed integration tests: a real Postgres and a real S3 store.
#
#   scripts/integration.sh                 # tests run in a container (default)
#   scripts/integration.sh --host          # tests run with this machine's python
#   scripts/integration.sh -- -k salt -x   # anything after -- goes to pytest
#
# What it does, in order:
#   1. generates a Postgres user/password/database and an S3 key pair for this
#      run only -- nothing is read from the repository or from ./.env, and
#      nothing is written back to either;
#   2. starts tests/containers/compose.yaml under a unique project name and
#      waits for both health checks;
#   3. runs tests/containers/ plus the Postgres half of tests/test_lineage.py
#      (written long ago, and deselected from every run until this script);
#   4. tears the stack down, volumes included.
#
# It exits non-zero if docker is missing, if a container does not come up, if
# any test fails, if any test SKIPS, or if nothing was collected. A skip is a
# failure here because this is the only place these tests run: one that
# skipped did not run anywhere.
#
# This is not part of the hermetic gate (scripts/ci.sh). It needs docker and
# takes about a minute; schedule it (see docs/testing.md) rather than gate on it.
#
# Environment:
#   AVALOKA_IT_IMAGE     image holding the python dependencies, for the default
#                        mode (default: avaloka-ci-deps:latest)
#   AVALOKA_IT_PLATFORM  e.g. linux/arm64. Default: what `docker info` reports,
#                        because `uname -m` lies under Rosetta.
#   AVALOKA_IT_KEEP=1    leave the stack running afterwards and print how to
#                        remove it. For debugging.
#   PYTHON               interpreter for --host (default: python)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/tests/containers/compose.yaml"
MODE=container
PYTEST_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host) MODE=host ;;
        --container) MODE=container ;;
        --) shift; PYTEST_ARGS=("$@"); break ;;
        -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
        *) echo "integration.sh: unknown argument '$1' (pytest arguments go after --)" >&2; exit 2 ;;
    esac
    shift
done

die() { echo "integration.sh: $*" >&2; exit 2; }

command -v docker >/dev/null 2>&1 || die "docker is not installed. This suite cannot run without it, and does not skip."
docker info >/dev/null 2>&1 || die "the docker daemon is not reachable. This suite cannot run without it, and does not skip."
docker compose version >/dev/null 2>&1 || die "docker compose (v2) is not available."

rand_hex() {  # $1 = number of hex characters
    LC_ALL=C tr -dc 'a-f0-9' </dev/urandom | head -c "$1" || true
}

WORK="$(mktemp -d "${TMPDIR:-/tmp}/avaloka-it.XXXXXXXX")"
chmod 700 "$WORK"
PROJECT="avaloka-it-$(rand_hex 8)"
compose() { docker compose -p "$PROJECT" -f "$COMPOSE_FILE" "$@"; }

cleanup() {
    status=$?
    if [ "${AVALOKA_IT_KEEP:-0}" = "1" ]; then
        echo "integration.sh: stack left running. Remove it with:" >&2
        echo "  AVALOKA_IT_PG_USER=x AVALOKA_IT_PG_PASSWORD=x AVALOKA_IT_PG_DB=x AVALOKA_IT_S3_CONFIG=/dev/null docker compose -p $PROJECT -f $COMPOSE_FILE down -v" >&2
    else
        compose down -v --remove-orphans >/dev/null 2>&1 || true
        rm -rf "$WORK"
    fi
    exit "$status"
}
trap cleanup EXIT

# --- 1. credentials for this run only ---------------------------------------
export AVALOKA_IT_PG_USER="it_$(rand_hex 8)"
export AVALOKA_IT_PG_PASSWORD="$(rand_hex 40)"
export AVALOKA_IT_PG_DB="it_$(rand_hex 8)"
export AVALOKA_IT_S3_ACCESS_KEY="it$(rand_hex 18)"
export AVALOKA_IT_S3_SECRET_KEY="$(rand_hex 40)"
export AVALOKA_IT_S3_REGION="us-east-1"
[ "${#AVALOKA_IT_PG_PASSWORD}" -eq 40 ] && [ "${#AVALOKA_IT_S3_SECRET_KEY}" -eq 40 ] \
    || die "could not generate credentials from /dev/urandom."

export AVALOKA_IT_S3_CONFIG="$WORK/s3.json"
cat >"$AVALOKA_IT_S3_CONFIG" <<JSON
{"identities": [{"name": "integration",
  "credentials": [{"accessKey": "$AVALOKA_IT_S3_ACCESS_KEY", "secretKey": "$AVALOKA_IT_S3_SECRET_KEY"}],
  "actions": ["Admin", "Read", "Write", "List", "Tagging"]}]}
JSON
chmod 644 "$AVALOKA_IT_S3_CONFIG"   # the directory is 0700; the container user must read the file

# --- 2. the stack ------------------------------------------------------------
if [ "$MODE" = container ]; then export AVALOKA_IT_NO_EGRESS=true; else export AVALOKA_IT_NO_EGRESS=false; fi
echo "integration.sh: starting postgres and seaweedfs (project $PROJECT, mode $MODE)"
if ! compose up -d --wait --quiet-pull; then
    compose ps >&2 || true
    compose logs --tail 40 >&2 || true
    die "the containers did not become healthy."
fi

# --- 3. where the tests find them --------------------------------------------
if [ "$MODE" = container ]; then
    PG_HOSTPORT="postgres:5432"
    S3_HOSTPORT="seaweedfs:8333"
else
    PG_HOSTPORT="$(compose port postgres 5432)" || die "postgres published no port."
    S3_HOSTPORT="$(compose port seaweedfs 8333)" || die "seaweedfs published no port."
fi
PG_SCHEME="postgresql"
export AVALOKA_IT_POSTGRES_URL="${PG_SCHEME}://${AVALOKA_IT_PG_USER}:${AVALOKA_IT_PG_PASSWORD}@${PG_HOSTPORT}/${AVALOKA_IT_PG_DB}"
export AVALOKA_IT_S3_ENDPOINT="http://${S3_HOSTPORT}"
# tests/test_lineage.py's Postgres half reads this name, and fails rather than
# skips without it when CI is set.
export AVALOKA_TEST_POSTGRES_URL="$AVALOKA_IT_POSTGRES_URL"
export CI=true

PASS_ENV=(AVALOKA_IT_POSTGRES_URL AVALOKA_IT_S3_ENDPOINT AVALOKA_IT_S3_ACCESS_KEY
          AVALOKA_IT_S3_SECRET_KEY AVALOKA_IT_S3_REGION AVALOKA_TEST_POSTGRES_URL CI)
PYTEST=(-m pytest tests/containers tests/test_lineage.py -m "containers or integration"
        -p no:cacheprovider -rs -q)
[ "${#PYTEST_ARGS[@]}" -gt 0 ] && PYTEST+=("${PYTEST_ARGS[@]}")

LOG="$WORK/pytest.log"
set +e
if [ "$MODE" = container ]; then
    IMAGE="${AVALOKA_IT_IMAGE:-avaloka-ci-deps:latest}"
    docker image inspect "$IMAGE" >/dev/null 2>&1 \
        || die "image '$IMAGE' is not present. Build or pull it, set AVALOKA_IT_IMAGE, or use --host."
    PLATFORM="${AVALOKA_IT_PLATFORM:-}"
    if [ -z "$PLATFORM" ]; then
        case "$(docker info --format '{{.Architecture}}')" in
            aarch64|arm64) PLATFORM=linux/arm64 ;;
            x86_64|amd64) PLATFORM=linux/amd64 ;;
        esac
    fi
    ENV_FLAGS=()
    for name in "${PASS_ENV[@]}"; do ENV_FLAGS+=(-e "$name"); done   # by name: values stay out of argv
    docker run --rm ${PLATFORM:+--platform "$PLATFORM"} --network "${PROJECT}_default" \
        "${ENV_FLAGS[@]}" -e PYTHONPATH=/w -e PYTHONDONTWRITEBYTECODE=1 \
        -v "$ROOT:/w:ro" -w /w "$IMAGE" python "${PYTEST[@]}" 2>&1 | tee "$LOG"
    status=${PIPESTATUS[0]}
else
    (cd "$ROOT" && PYTHONPATH="$ROOT" "${PYTHON:-python}" "${PYTEST[@]}") 2>&1 | tee "$LOG"
    status=${PIPESTATUS[0]}
fi
set -e

# --- 4. a green line is not enough --------------------------------------------
summary="$(grep -E '^(=+ )?[0-9]+ (passed|failed|error|errors|skipped|deselected)' "$LOG" | tail -1 || true)"
if [ "$status" -ne 0 ]; then
    echo "integration.sh: FAILED (pytest exit $status)" >&2
    exit 1
fi
if [ -z "$summary" ] || ! echo "$summary" | grep -q ' passed'; then
    echo "integration.sh: FAILED -- pytest exited 0 but printed no pass count; treating as not run." >&2
    exit 1
fi
if echo "$summary" | grep -q 'skipped'; then
    echo "integration.sh: FAILED -- tests skipped. In this suite a skip means the test ran nowhere." >&2
    exit 1
fi
echo "integration.sh: OK -- $summary"
