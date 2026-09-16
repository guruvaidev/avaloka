#!/bin/bash

# 🚀 Interactive Helm + K8s redeploy script for Avaloka
# ------------------------------------------------------

set -e

# === Styling helpers ===
GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
YELLOW=$(tput setaf 3)
CYAN=$(tput setaf 6)
RESET=$(tput sgr0)
BOLD=$(tput bold)

spinner() {
  local pid=$!
  local delay=0.1
  local spin='|/-\'
  while kill -0 $pid 2>/dev/null; do
    for i in $(seq 0 3); do
      printf "\r${CYAN}${spin:$i:1}${RESET} "
      sleep $delay
    done
  done
  printf "\r"
}

check_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "${RED}❌ Error:${RESET} $1 is not installed."
    exit 1
  }
}

pause() {
  read -rp "${YELLOW}→ Press [Enter] to continue...${RESET}"
}

wait_for_redis() {
  local namespace=$1
  local timeout=${2:-180} # 3 min timeout
  local start_time=$(date +%s)

  echo "${CYAN}${BOLD}⏳ Waiting for Redis pod to become Ready...${RESET}"

  while true; do
    # Detect any pod with "redis" in its name
    local redis_pod
    redis_pod=$(kubectl get pods -n "$namespace" --no-headers 2>/dev/null | grep -i redis | awk '{print $1}' | head -n 1)

    if [[ -z "$redis_pod" ]]; then
      printf "${YELLOW}⏳ No Redis pod found yet...${RESET}\n"
    else
      local status
      status=$(kubectl get pod "$redis_pod" -n "$namespace" -o jsonpath='{.status.phase}' 2>/dev/null)

      if [[ "$status" == "Running" ]]; then
        local ready
        ready=$(kubectl get pod "$redis_pod" -n "$namespace" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)
        if [[ "$ready" == "true" ]]; then
          echo "${GREEN}✅ Redis pod ${redis_pod} is Ready!${RESET}"
          return 0
        fi
      fi
      printf "${YELLOW}🕐 Redis pod ($redis_pod) status: $status${RESET}\n"
    fi

    local now=$(date +%s)
    if (( now - start_time > timeout )); then
      echo "${RED}❌ Timeout waiting for Redis pod to become ready.${RESET}"
      kubectl get pods -n "$namespace" | grep -i redis || true
      exit 1
    fi

    sleep 5
  done
}


# === Pre-flight checks ===
echo "${BOLD}${CYAN}⚙️  Checking dependencies...${RESET}"
for cmd in helm kubectl; do
  check_command "$cmd"
done
echo "${GREEN}✅ All dependencies found.${RESET}"
sleep 1

# === Namespace & paths ===
NAMESPACE="avaloka"
ENV_FILE="deploy/avaloka/charts/avaloka-backend/docker/docker.env"
GCP_KEY_PATH="/Users/$USER/.config/gcloud/application_default_credentials.json"

echo "${BOLD}${CYAN}🌐 Namespace:${RESET} ${NAMESPACE}"
pause

# === Uninstall existing release ===
echo "${BOLD}${YELLOW}🧹 Cleaning up old release...${RESET}"
(helm uninstall avaloka -n "$NAMESPACE" >/dev/null 2>&1 & spinner)
echo "${GREEN}✅ Helm release uninstalled (if existed).${RESET}"

(kubectl delete configmap avaloka-backend-env -n "$NAMESPACE" >/dev/null 2>&1 & spinner)
echo "${GREEN}✅ ConfigMap removed.${RESET}"

(kubectl delete secret gcp-key -n "$NAMESPACE" >/dev/null 2>&1 & spinner)
echo "${GREEN}✅ Secret removed.${RESET}"

sleep 1

# Ensure namespace exists
if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
  echo "${YELLOW}Namespace ${NAMESPACE} does not exist. Creating it...${RESET}"
  kubectl create namespace "$NAMESPACE"
  echo "${GREEN}✅ Namespace ${NAMESPACE} created.${RESET}"
fi

sleep 1

# === Recreate ConfigMap ===
echo "${BOLD}${CYAN}🧩 Creating new ConfigMap from:${RESET} ${ENV_FILE}"
(kubectl create configmap avaloka-backend-env \
  --from-env-file="$ENV_FILE" \
  -n "$NAMESPACE" >/dev/null 2>&1 & spinner)
echo "${GREEN}✅ ConfigMap created.${RESET}"

# === Recreate Secret ===
echo "${BOLD}${CYAN}🔑 Creating new GCP Secret...${RESET}"

if [[ ! -f "$GCP_KEY_PATH" ]]; then
  echo "${RED}❌ GCP key file not found: $GCP_KEY_PATH${RESET}"
  exit 1
fi

kubectl create secret generic gcp-key \
  --from-file=gcp-key.json="$GCP_KEY_PATH" \
  -n "$NAMESPACE" >/dev/null 2>&1

pid=$!
spinner $pid

# Verify creation
if kubectl get secret gcp-key -n "$NAMESPACE" >/dev/null 2>&1; then
  echo "${GREEN}✅ Secret created and verified.${RESET}"
else
  echo "${RED}❌ Secret creation failed.${RESET}"
  exit 1
fi

sleep 1

# === Helm install ===
echo "${BOLD}${CYAN}🚀 Deploying Avaloka Helm chart...${RESET}"
(helm install avaloka deploy/avaloka \
  --namespace "$NAMESPACE" \
  --create-namespace \
  --dependency-update >/dev/null 2>&1 & spinner)
echo "${GREEN}✅ Helm install complete.${RESET}"

sleep 3

# === Wait for Redis ===
wait_for_redis "$NAMESPACE"

# === Verify resources ===
echo "${BOLD}${CYAN}🔍 Checking deployed resources...${RESET}"
kubectl get all -n "$NAMESPACE"

# === Final output ===
echo
echo "${GREEN}${BOLD}🎉 Avaloka successfully deployed!${RESET}"
echo "🌍 Access the application at:"
echo "   ${CYAN}http://localhost${RESET}"
echo
echo "🧠 Tip: To port-forward locally, run:"
echo "   ${YELLOW}kubectl port-forward svc/avaloka-backend 8080:80 -n ${NAMESPACE}${RESET}"
echo "   Then open → ${CYAN}http://localhost:8080${RESET}"
echo
