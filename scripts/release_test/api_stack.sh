#!/bin/bash
# Bring up (or tear down) an isolated API stack for release testing: its own
# network, its own Redis, the API served from a mounted checkout inside the
# arm64 CI image, with one inference provider pinned. Never touches the
# owner's avaloka-api / avaloka-redis containers.
#
#   api_stack.sh up   <label> <host-port> <checkout-dir> <state-dir> local gemma4:e4b
#   api_stack.sh up   <label> <host-port> <checkout-dir> <state-dir> groq  openai/gpt-oss-120b
#   api_stack.sh down <label>
#
# Auth: the API rejects every request when SUPABASE_JWT_SECRET is empty, even
# with AVALOKA_ALLOW_INSECURE_AUTH=1, so a throwaway secret is generated and
# written to <state-dir>/jwt_secret for the harness to sign with.
set -eu
IMAGE="${AVALOKA_CI_IMAGE:-avaloka-ci-deps:latest}"
cmd="$1"; label="$2"
case "$cmd" in
  up)
    port="$3"; checkout="$4"; state="$5"; provider="$6"; model="$7"
    mkdir -p "$state"
    [ -s "$state/jwt_secret" ] || head -c 32 /dev/urandom | base64 | tr -d '\n' > "$state/jwt_secret"
    docker network create "rt-net-$label" >/dev/null 2>&1 || true
    docker rm -f "rt-redis-$label" "rt-api-$label" >/dev/null 2>&1 || true
    docker run -d --name "rt-redis-$label" --network "rt-net-$label" redis:7 >/dev/null
    extra=()
    if [ "$provider" = local ]; then
      extra+=(-e INFERENCE_LOCAL_BASE_URL=http://host.docker.internal:11434/v1 -e INFERENCE_LOCAL_MODEL="$model")
    else
      # per-agent model variables mirror the single --model argument
      extra+=(-e GROQ_API_KEY="${GROQ_API_KEY:?GROQ_API_KEY must be in the environment}" \
              -e AVALOKA_PLANNING_AGENT_MODEL="$model" -e AVALOKA_CODING_AGENT_MODEL="$model" \
              -e AVALOKA_VIZ_AGENT_MODEL="$model" -e AVALOKA_AVALOKA_AGENT_MODEL="$model")
    fi
    docker run -d --name "rt-api-$label" --network "rt-net-$label" --platform linux/arm64 \
      -p "127.0.0.1:$port:9000" --add-host host.docker.internal:host-gateway \
      -v "$checkout:/w" -v "$state:/state" -w /w \
      -e PYTHONPATH=/w -e PYTHONUNBUFFERED=1 -e CI=true -e LOG_LEVEL=INFO \
      -e SUPABASE_JWT_SECRET="$(cat "$state/jwt_secret")" \
      -e AVALOKA_EDITION=oss -e AVALOKA_DEPLOYMENT_PROFILE=laptop \
      -e INFERENCE_PROVIDER="$provider" -e AVALOKA_DISABLE_MODEL_FALLBACK=1 \
      -e REDIS_URL="redis://rt-redis-$label:6379/0" -e STORAGE_BACKEND=local \
      -e LOCAL_STORAGE_ROOT=/state/filestore -e DATA_DIR=/state/data -e CHAT_OUTPUT_DIR=/state/tmp \
      -e PLANNER_GRAPH_OUTPUT_DIR=/state/planner_graphs -e LANGGRAPH_API_URL=http://127.0.0.1:2024 \
      -e AVALOKA_LOCAL_EXEC_TIMEOUT=300 -e LANGSMITH_TRACING=false \
      "${extra[@]}" "$IMAGE" bash -c '
        # pipefail: without it a uvicorn that dies mid-run leaves the container
        # exiting 0 through tee, indistinguishable from a clean stop.
        set -u -o pipefail
        export GROQ_API_KEY_PLANNING_AGENT="${GROQ_API_KEY_PLANNING_AGENT:-${GROQ_API_KEY:-}}"
        export GROQ_API_KEY_CODING_AGENT="${GROQ_API_KEY_CODING_AGENT:-${GROQ_API_KEY:-}}"
        mkdir -p /state/tmp /state/filestore /state/planner_graphs /state/data
        # The checkout is mounted read-write on purpose: langgraph dev writes
        # .langgraph_api/ into its cwd and the graph module creates artifacts/
        # under the repo root at import time (both fail on a read-only mount).
        ( langgraph dev --config /w/langgraph.json --host 127.0.0.1 --port 2024 \
            --no-browser --no-reload --allow-blocking > /state/langgraph.log 2>&1 ) &
        exec uvicorn app.api.server:app --host 0.0.0.0 --port 9000 --loop asyncio 2>&1 | tee /state/api.log' >/dev/null
    for i in $(seq 1 90); do
      if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        echo "rt-api-$label up on :$port ($provider/$model)"; curl -fsS "http://127.0.0.1:$port/health"; echo; exit 0
      fi; sleep 2
    done
    echo "rt-api-$label did not become healthy; last log lines:" >&2; docker logs --tail 40 "rt-api-$label" >&2; exit 1 ;;
  down)
    docker rm -f "rt-redis-$label" "rt-api-$label" >/dev/null 2>&1 || true
    docker network rm "rt-net-$label" >/dev/null 2>&1 || true
    echo "rt-$label removed" ;;
  *) echo "usage: $0 up|down ..." >&2; exit 2 ;;
esac
