#!/usr/bin/env bash
# Avaloka web stack on this machine, with one command.
#
#   scripts/quickstart.sh up            build, check models, start; prints the URL
#   scripts/quickstart.sh up --offline  the same, and the backend gets no internet egress
#   scripts/quickstart.sh down          stop everything (keeps your data)
#   scripts/quickstart.sh down --volumes   ...and delete the data volumes too
#   scripts/quickstart.sh status | logs [service] | password
#
# First run generates .env.quickstart (gitignored, mode 600) holding this
# install's secrets: DB_ENCRYPTION_KEY and MCP_SERVICE_TOKEN (each shared by
# the API and the MCP server), the local-auth JWT secret and password, and the
# sample Postgres password.
# Later runs reuse it and only add what is missing. Delete it to start over.
#
# Needs Docker with Compose v2, about 12 GB of free disk for the images, and,
# for the default local inference, Ollama on this machine with a model pulled
# (`ollama pull gemma4:e4b`). Set INFERENCE_PROVIDER etc. in .env.quickstart
# to use a hosted model instead.
#
# Environment: AVALOKA_API_IMAGE / AVALOKA_UI_IMAGE use prebuilt images
# instead of building from this checkout (pulled; with --no-build, an image
# already on this machine is used as it is); AVALOKA_UI_PORT / AVALOKA_API_PORT
# pin the host ports (otherwise the first free one from 3000 / 9000);
# COMPOSE_PROJECT_NAME runs a second, separate stack (its own containers,
# networks and volumes) beside the default avaloka-quickstart one; LOG_LEVEL
# (default INFO) sets the backend's log level.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/.env.quickstart"
COMPOSE_FILE="$ROOT/docker-compose.quickstart.yml"
PROJECT="${COMPOSE_PROJECT_NAME:-avaloka-quickstart}"

say() { printf '%s\n' "$*"; }
die() { printf 'quickstart: %s\n' "$*" >&2; exit 1; }

compose() {
  docker compose --project-name "$PROJECT" --project-directory "$ROOT" \
    --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

rand_hex() { openssl rand -hex "$1"; }
# Letters and digits only, so the value needs no quoting anywhere.
rand_word() { LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c "$1"; }

env_get() { [ -f "$ENV_FILE" ] && sed -n "s/^$1=//p" "$ENV_FILE" | tail -n1 || true; }
env_set() {
  # Replace or append KEY=VALUE without ever echoing the value.
  local key=$1 val=$2 tmp
  tmp="$(mktemp "$ENV_FILE.XXXXXX")"
  { grep -v "^$key=" "$ENV_FILE" 2>/dev/null || true; printf '%s=%s\n' "$key" "$val"; } >"$tmp"
  chmod 600 "$tmp"; mv "$tmp" "$ENV_FILE"
}
env_default() { [ -n "$(env_get "$1")" ] || env_set "$1" "$2"; }

# Values that ship in this repository (app/core/provisioning.py PUBLISHED_VALUES);
# a secret equal to one of them protects nothing and is replaced.
PUBLISHED_SECRETS=" avaloka-local-dev-encryption-key <generate-with-openssl-rand-hex-32> <generate-and-store-a-strong-random-secret> changeme change-me secret test dev development "
is_published() { case "$PUBLISHED_SECRETS" in *" $(printf '%s' "$1" | tr '[:upper:]' '[:lower:]') "*) return 0;; esac; return 1; }
# What a replacement costs (app/core/provisioning.py REPLACEMENT_CONSEQUENCE).
replacement_consequence() {
  case "$1" in
    DB_ENCRYPTION_KEY) printf '%s' "Anything sealed under the old key can no longer be opened: saved database connections, cloud connection keys, MCP and GitHub tokens, and passwords typed into chat. Register those connections again (and re-enter those tokens)." ;;
    MCP_SERVICE_TOKEN) printf '%s' "Nothing stored depends on it; the next up starts every service with the new value." ;;
    EXECUTE_SERVICE_TOKEN) printf '%s' "Nothing stored depends on it; the next up starts the graph services and the execute service with the new value." ;;
    LANGGRAPH_SERVICE_TOKEN) printf '%s' "Nothing stored depends on it; the next up starts the API and the LangGraph server with the new value." ;;
    AVALOKA_PII_KEY) printf '%s' "Columns pseudonymised under the old key no longer match new pseudonyms; retrain models trained on pseudonymised data." ;;
  esac
}
# A generated secret: kept when set; generated when empty; replaced, with a
# warning, when it is a published value. The value is never printed.
env_secret() {
  local key=$1 cur
  cur="$(env_get "$key")"
  if [ -z "$cur" ]; then
    env_set "$key" "$(rand_hex 32)"
  elif is_published "$cur"; then
    env_set "$key" "$(rand_hex 32)"
    say "WARNING: $key in .env.quickstart was a published value (it ships in this repository); replaced with a generated one. $(replacement_consequence "$key")"
  fi
}

port_free() {
  python3 - "$1" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
PY
}

pick_port() {
  # $1 env key, $2 default. An explicit environment value wins; then the
  # saved one; then the first free port from the default upward.
  local key=$1 def=$2 want p
  want="${!key:-$(env_get "$key")}"
  if [ -n "$want" ]; then env_set "$key" "$want"; return; fi
  p=$def
  while ! port_free "$p"; do p=$((p + 1)); [ $p -lt $((def + 50)) ] || die "no free port near $def for $key"; done
  env_set "$key" "$p"
}

ensure_env() {
  umask 077
  if [ ! -f "$ENV_FILE" ]; then
    # New secrets would not open old data: Postgres keeps the password it was
    # initialised with, and the MCP registry is sealed with the old
    # DB_ENCRYPTION_KEY. Volumes are per machine (project name), not per
    # checkout, so a second clone or a deleted .env.quickstart hits this.
    local old
    old="$(docker volume ls -q --filter "name=^${PROJECT}_qs-" 2>/dev/null || true)"
    if [ -n "$old" ]; then
      die "quickstart data from an earlier install exists (volumes: $(echo $old | tr '\n' ' ')) but its .env.quickstart is gone, so the new secrets cannot open it. Restore that .env.quickstart, or delete the old data: docker volume rm $(echo $old | tr '\n' ' ')"
    fi
    {
      echo "# Generated by scripts/quickstart.sh. Secrets for this machine only."
      echo "# Never commit this file (it is gitignored). Delete it to start over."
      echo "#"
      echo "# Inference: local Ollama by default. To use a hosted model instead, e.g.:"
      echo "#   INFERENCE_PROVIDER=groq"
      echo "#   GROQ_API_KEY=..."
      echo "INFERENCE_PROVIDER=local"
      echo "INFERENCE_LOCAL_MODEL=gemma4:e4b"
    } >"$ENV_FILE"
    chmod 600 "$ENV_FILE"
    say "created .env.quickstart (secrets for this machine; gitignored)"
  fi
  env_default AVALOKA_LOCAL_AUTH_SECRET "$(rand_hex 32)"
  env_default AVALOKA_LOCAL_AUTH_EMAIL "local@avaloka.local"
  env_default AVALOKA_LOCAL_AUTH_PASSWORD "$(rand_word 20)"
  # The generated secrets (app/core/provisioning.py GENERATED_SECRETS).
  env_secret DB_ENCRYPTION_KEY
  # The API <-> MCP service identity; an install from before it existed gets
  # one on its next `up`. Rotating it needs no data migration.
  env_secret MCP_SERVICE_TOKEN
  # The graph pods <-> /execute service secret and the API <-> LangGraph server
  # secret (app/core/provisioning.py GENERATED_SECRETS): generated here like the
  # others, so a container never has to make its own; an install from before they
  # were generated gets them on its next `up`. Rotating either needs no data migration.
  env_secret EXECUTE_SERVICE_TOKEN
  env_secret LANGGRAPH_SERVICE_TOKEN
  # The PII pseudonymisation key; an install from before it was generated gets
  # one on its next `up`.
  env_secret AVALOKA_PII_KEY
  env_default QS_POSTGRES_PASSWORD "$(rand_word 24)"
}

running() { [ -n "$(compose ps -q 2>/dev/null || true)" ]; }

check_models() {
  local img=${AVALOKA_API_IMAGE:-avaloka-api:quickstart}
  say "checking the baked models with networking disabled..."
  docker run --rm --network none "$img" python scripts/prefetch_models.py --check \
    || die "the API image is missing a model; rebuild it (scripts/quickstart.sh up) or run scripts/prefetch_models.py"
}

check_ollama() {
  local provider model
  provider="$(env_get INFERENCE_PROVIDER)"; provider=${provider:-local}
  [ "$provider" = local ] || return 0
  model="$(env_get INFERENCE_LOCAL_MODEL)"; model=${model:-gemma4:e4b}
  if ! curl -fsS --max-time 3 http://127.0.0.1:11434/api/tags >/tmp/.avaloka-qs-tags.$$ 2>/dev/null; then
    say "warning: Ollama is not answering on 127.0.0.1:11434. Analyses need a model:"
    say "         install Ollama and run \`ollama pull $model\`, or set INFERENCE_PROVIDER in .env.quickstart."
  elif ! grep -q "\"$model\"" /tmp/.avaloka-qs-tags.$$; then
    say "warning: Ollama has no model '$model'. Run \`ollama pull $model\`."
  fi
  rm -f /tmp/.avaloka-qs-tags.$$
}

cmd_up() {
  local offline=false build=true t0 t1
  for a in "$@"; do
    case $a in
      --offline) offline=true ;;
      --no-build) build=false ;;
      *) die "unknown option: $a" ;;
    esac
  done
  command -v docker >/dev/null || die "docker is not installed"
  docker compose version >/dev/null 2>&1 || die "docker compose v2 is required"
  command -v openssl >/dev/null || die "openssl is required to generate secrets"
  t0=$(date +%s)
  ensure_env
  # A running stack in the other network mode must be recreated: a network's
  # internal flag cannot change in place.
  if running; then
    local cur
    cur="$(docker network inspect "${PROJECT}_backend" --format '{{.Internal}}' 2>/dev/null || echo none)"
    if [ "$cur" != "$offline" ]; then say "switching network mode; restarting the stack"; compose --profile offline down --remove-orphans; fi
  fi
  if ! running; then
    pick_port AVALOKA_UI_PORT 3000
    pick_port AVALOKA_API_PORT 9000
  fi
  if [ -n "${AVALOKA_API_IMAGE:-}${AVALOKA_UI_IMAGE:-}" ]; then
    if $build; then
      compose pull api ui
    else
      # --no-build: use what is here; pull only an image this machine lacks.
      local missing=()
      docker image inspect "${AVALOKA_API_IMAGE:-avaloka-api:quickstart}" >/dev/null 2>&1 || missing+=(api)
      docker image inspect "${AVALOKA_UI_IMAGE:-avaloka-ui:quickstart}" >/dev/null 2>&1 || missing+=(ui)
      [ ${#missing[@]} -eq 0 ] || compose pull "${missing[@]}"
    fi
  elif $build; then
    say "building images from this checkout (the first build takes a while)..."
    compose build api ui
  fi
  check_models
  check_ollama
  export AVALOKA_QS_OFFLINE=$offline
  if $offline; then
    export INFERENCE_LOCAL_BASE_URL=http://ollama.internal:11434/v1
    compose --profile offline up -d --wait --remove-orphans
  else
    compose up -d --wait --remove-orphans
  fi
  t1=$(date +%s)
  local ui api email
  ui="$(env_get AVALOKA_UI_PORT)"; api="$(env_get AVALOKA_API_PORT)"; email="$(env_get AVALOKA_LOCAL_AUTH_EMAIL)"
  say ""
  say "Avaloka is up in $((t1 - t0))s$($offline && echo ' (offline: no internet egress from the backend)')."
  say "  open      http://127.0.0.1:$ui/"
  say "  sign in   $email"
  say "  password  run: scripts/quickstart.sh password"
  $offline || say "  API       http://127.0.0.1:$api/health"
  say "  stop      scripts/quickstart.sh down"
}

cmd_down() {
  [ -f "$ENV_FILE" ] || die "no .env.quickstart here; nothing was started from this checkout"
  if [ "${1:-}" = "--volumes" ]; then
    compose --profile offline down --remove-orphans --volumes
    say "stopped, and the quickstart's data volumes are deleted"
  else
    compose --profile offline down --remove-orphans
    say "stopped (data kept; \`down --volumes\` deletes it)"
  fi
}

cmd_password() {
  [ -f "$ENV_FILE" ] || die "run scripts/quickstart.sh up first"
  env_get AVALOKA_LOCAL_AUTH_PASSWORD
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  down) shift; cmd_down "$@" ;;
  status) compose --profile offline ps ;;
  logs) shift; compose logs --tail 200 "$@" ;;
  password) cmd_password ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
