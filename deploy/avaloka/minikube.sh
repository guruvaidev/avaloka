#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-shot minikube deploy for the Avaloka backend.
#
# Brings up a local Kubernetes cluster, builds the backend images straight into
# minikube's docker daemon, and installs the Helm stack with the minikube
# overlay. Idempotent: safe to re-run.
#
#   ./deploy/avaloka/minikube.sh                 # deploy with defaults
#   REPLICAS=3 ./deploy/avaloka/minikube.sh      # deploy 3 backend instances
#   SKIP_BUILD=1 ./deploy/avaloka/minikube.sh    # reuse already-built images
#
# Prereqs: minikube, kubectl, helm, docker.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Resolve repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHART_DIR="$SCRIPT_DIR"                       # deploy/avaloka
VALUES="$CHART_DIR/values/values-minikube.yaml"

NAMESPACE="${NAMESPACE:-avaloka}"
RELEASE="${RELEASE:-avaloka}"
REPLICAS="${REPLICAS:-1}"
LANGGRAPH_IMAGE="avaloka-langgraph:latest"
API_IMAGE="avaloka-api:latest"

log()  { printf '\033[1;36m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }

# 1. Cluster ────────────────────────────────────────────────────────────────
if ! minikube status >/dev/null 2>&1; then
  log "Starting minikube (docker driver)…"
  minikube start --driver=docker --cpus=4 --memory=6144 --disk-size=40g
else
  ok "minikube already running"
fi

log "Enabling ingress + metrics-server addons…"
minikube addons enable ingress >/dev/null
minikube addons enable metrics-server >/dev/null
ok "addons enabled"

# 2. Images (built directly into minikube's docker) ──────────────────────────
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  log "Pointing docker at minikube and building backend images (this is slow the first time)…"
  eval "$(minikube docker-env)"
  docker build -t "$API_IMAGE" \
    -f "$CHART_DIR/charts/avaloka-backend/docker/Dockerfile.api" "$REPO_ROOT"
  # Shares cached layers with the api image, so this is fast.
  docker build -t "$LANGGRAPH_IMAGE" \
    -f "$CHART_DIR/charts/avaloka-backend/docker/Dockerfile.langgraph" "$REPO_ROOT"
  eval "$(minikube docker-env -u)"
  ok "images built: $API_IMAGE, $LANGGRAPH_IMAGE"
else
  ok "SKIP_BUILD=1 — reusing existing images"
fi

# 3. Helm install ────────────────────────────────────────────────────────────
log "Resolving chart dependencies…"
helm dependency build "$CHART_DIR" >/dev/null

log "Installing release '$RELEASE' (replicas=$REPLICAS) into namespace '$NAMESPACE'…"
helm upgrade --install "$RELEASE" "$CHART_DIR" \
  --namespace "$NAMESPACE" --create-namespace \
  -f "$VALUES" \
  --set "avaloka-backend.replicaCount=$REPLICAS" \
  --wait --timeout 10m

# 4. Wait + report ───────────────────────────────────────────────────────────
log "Waiting for backend rollout…"
kubectl -n "$NAMESPACE" rollout status deploy -l app.kubernetes.io/component=backend --timeout=300s

ok "Deployed. Resources:"
kubectl -n "$NAMESPACE" get pods,svc,hpa -l app.kubernetes.io/component=backend 2>/dev/null || \
  kubectl -n "$NAMESPACE" get pods,svc

cat <<EOF

$(ok "Avaloka is up on minikube.")
Reach the services with port-forward:

  kubectl -n $NAMESPACE port-forward svc/${RELEASE}-avaloka-backend-api 9000:9000 &
  kubectl -n $NAMESPACE port-forward svc/${RELEASE}-avaloka-backend-langgraph 3000:3000 &

  curl localhost:9000/health      # FastAPI backend
  curl localhost:3000/info        # LangGraph server

Scale the number of instances:

  kubectl -n $NAMESPACE scale deploy/${RELEASE}-avaloka-backend --replicas=3

Tear down:  ./deploy/avaloka/stop.sh   (or: minikube delete)
EOF
