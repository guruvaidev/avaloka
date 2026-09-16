#!/bin/bash

# 🧹 Interactive Helm + K8s stop script for Avaloka
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

confirm() {
  read -rp "${YELLOW}⚠️  Are you sure you want to stop and delete Avaloka? (y/n): ${RESET}" answer
  case "$answer" in
    [Yy]* ) echo "${CYAN}Proceeding...${RESET}" ;;
    * ) echo "${RED}Operation cancelled.${RESET}"; exit 0 ;;
  esac
}

check_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "${RED}❌ Error:${RESET} $1 is not installed."
    exit 1
  }
}

# === Pre-flight checks ===
echo "${BOLD}${CYAN}⚙️  Checking dependencies...${RESET}"
for cmd in helm kubectl; do
  check_command "$cmd"
done
echo "${GREEN}✅ All dependencies found.${RESET}"
sleep 1

# === Namespace ===
NAMESPACE="avaloka"
echo "${BOLD}${CYAN}🌐 Namespace:${RESET} ${NAMESPACE}"

confirm

# === Stop Helm release ===
echo "${BOLD}${YELLOW}🧩 Uninstalling Helm release 'avaloka'...${RESET}"
(helm uninstall avaloka -n "$NAMESPACE" >/dev/null 2>&1 & spinner)
echo "${GREEN}✅ Helm release removed.${RESET}"

# === Delete ConfigMap ===
echo "${BOLD}${YELLOW}🗑️  Deleting ConfigMap avaloka-backend-env...${RESET}"
(kubectl delete configmap avaloka-backend-env -n "$NAMESPACE" >/dev/null 2>&1 & spinner)
echo "${GREEN}✅ ConfigMap deleted.${RESET}"

# === Delete Secret ===
echo "${BOLD}${YELLOW}🔐 Deleting Secret gcp-key...${RESET}"
(kubectl delete secret gcp-key -n "$NAMESPACE" >/dev/null 2>&1 & spinner)
echo "${GREEN}✅ Secret deleted.${RESET}"

# === Optional clean up namespace (ask user) ===
read -rp "${YELLOW}🧹 Do you also want to delete the entire namespace '${NAMESPACE}'? (y/n): ${RESET}" ns_delete
if [[ "$ns_delete" =~ ^[Yy]$ ]]; then
  echo "${BOLD}${YELLOW}Deleting namespace ${NAMESPACE}...${RESET}"
  (kubectl delete namespace "$NAMESPACE" >/dev/null 2>&1 & spinner)
  echo "${GREEN}✅ Namespace deleted.${RESET}"
else
  echo "${CYAN}⏭️  Skipping namespace deletion.${RESET}"
fi

echo
echo "${GREEN}${BOLD}🧼 Avaloka stopped and cleaned up successfully!${RESET}"
echo
