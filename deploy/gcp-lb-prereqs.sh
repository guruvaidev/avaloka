#!/usr/bin/env bash
#
# Create the GCP resources the avaloka Gateway needs before `helm install`.
#
# The chart renders the Gateway/HTTPRoute/policies, and GKE's controller turns those
# into the global external Application Load Balancer. Three things must exist first
# because they outlive any single release:
#
#   1. a reserved global static IP        — so DNS keeps resolving across redeploys
#   2. a Certificate Manager map + certs  — Google-managed TLS for each hostname
#   3. (reported, not created) DNS records — A records for each host -> that IP
#
# Idempotent: every step is skipped if the resource already exists, so it is safe
# to re-run after adding a hostname.
#
# Each avaloka instance gets its OWN IP and certificate map, so run this once per
# instance — that separation is what lets test be redeployed without risking prod.
#
# Usage:
#   # test.example.com (values-gke.yaml defaults)
#   ./deploy/gcp-lb-prereqs.sh --project my-project --hosts test.example.com
#
#   # app.example.com (values-gke-app.yaml) — separate IP, separate cert map
#   ./deploy/gcp-lb-prereqs.sh --project my-project --hosts app.example.com \
#       --ip-name avaloka-app-lb-ip --cert-map avaloka-app-cert-map
#
#   ./deploy/gcp-lb-prereqs.sh --project my-project --hosts app.example.com --dry-run
#
set -euo pipefail

PROJECT=""
HOSTS=""
IP_NAME="avaloka-lb-ip"
CERT_MAP_NAME="avaloka-cert-map"
DRY_RUN=0

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)      PROJECT="$2"; shift 2 ;;
    --hosts)        HOSTS="$2"; shift 2 ;;
    --ip-name)      IP_NAME="$2"; shift 2 ;;
    --cert-map)     CERT_MAP_NAME="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=1; shift ;;
    -h|--help)      usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

[[ -n "$PROJECT" ]] || { echo "--project is required" >&2; exit 1; }
[[ -n "$HOSTS"   ]] || { echo "--hosts is required (comma-separated)" >&2; exit 1; }
command -v gcloud >/dev/null || { echo "gcloud not found on PATH" >&2; exit 1; }

run() {
  if (( DRY_RUN )); then
    printf '  [dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

# Succeeds only when the resource is really there; `gcloud ... describe` writes the
# not-found error to stderr, which we discard so the caller sees a clean boolean.
exists() { "$@" >/dev/null 2>&1; }

IFS=',' read -r -a HOST_LIST <<< "$HOSTS"

echo "project   : $PROJECT"
echo "hosts     : ${HOST_LIST[*]}"
echo "static ip : $IP_NAME"
echo "cert map  : $CERT_MAP_NAME"
echo

# --------------------------------------------------------------------- 0. APIs
echo "==> Enabling required APIs"
run gcloud services enable compute.googleapis.com certificatemanager.googleapis.com \
  --project "$PROJECT"

# ------------------------------------------------------------- 1. Static IP
echo "==> Reserving global static IP '$IP_NAME'"
if exists gcloud compute addresses describe "$IP_NAME" --global --project "$PROJECT"; then
  echo "  already exists"
else
  run gcloud compute addresses create "$IP_NAME" \
    --global --ip-version=IPV4 --network-tier=PREMIUM --project "$PROJECT"
fi

# ------------------------------------------------------- 2. Certificate Manager
echo "==> Creating certificate map '$CERT_MAP_NAME'"
if exists gcloud certificate-manager maps describe "$CERT_MAP_NAME" --project "$PROJECT"; then
  echo "  already exists"
else
  run gcloud certificate-manager maps create "$CERT_MAP_NAME" --project "$PROJECT"
fi

for host in "${HOST_LIST[@]}"; do
  # Resource names allow no dots, so app.example.com -> app-avaloka-ai.
  slug="${host//./-}"
  cert_name="avaloka-cert-${slug}"
  entry_name="avaloka-entry-${slug}"

  echo "==> Certificate for ${host}"
  if exists gcloud certificate-manager certificates describe "$cert_name" --project "$PROJECT"; then
    echo "  already exists"
  else
    # Google-managed and auto-renewing. Issuance stays PENDING until the A record
    # for this host resolves to the static IP above — that is the domain validation.
    run gcloud certificate-manager certificates create "$cert_name" \
      --domains="$host" --project "$PROJECT"
  fi

  echo "==> Map entry for ${host}"
  if exists gcloud certificate-manager maps entries describe "$entry_name" \
       --map="$CERT_MAP_NAME" --project "$PROJECT"; then
    echo "  already exists"
  else
    run gcloud certificate-manager maps entries create "$entry_name" \
      --map="$CERT_MAP_NAME" \
      --certificates="$cert_name" \
      --hostname="$host" \
      --project "$PROJECT"
  fi
done

# ------------------------------------------------------------------ 3. Next steps
echo
if (( DRY_RUN )); then
  echo "Dry run complete — nothing was created."
  exit 0
fi

IP_ADDR="$(gcloud compute addresses describe "$IP_NAME" --global \
  --project "$PROJECT" --format='value(address)')"

cat <<EOF

Done. Reserved IP: ${IP_ADDR}

Next:

  1. Point DNS at the load balancer — one A record per host:
$(for host in "${HOST_LIST[@]}"; do printf '       %-24s A     %s\n' "$host" "$IP_ADDR"; done)

     Google-managed certificates stay PENDING until these resolve; that is the
     domain-ownership check, so nothing serves HTTPS before DNS propagates.
     Validation and renewal happen over 443, so the HTTPS-only Gateway (no :80
     listener) is fine.

  2. Install the matching instance. For test.example.com:

       helm upgrade --install avaloka deploy/helm/avaloka \\
         --namespace avaloka --create-namespace \\
         -f deploy/helm/avaloka/values/values-gke.yaml \\
         --set gateway.addresses[0].value=${IP_NAME} \\
         --set gateway.tls.certMapName=${CERT_MAP_NAME}

     For app.example.com — note the SEPARATE namespace, which is required because
     the chart names resources after the chart, not the release:

       helm upgrade --install avaloka-app deploy/helm/avaloka \\
         --namespace avaloka-app --create-namespace \\
         -f deploy/helm/avaloka/values/values-gke.yaml \\
         -f deploy/helm/avaloka/values/values-gke-app.yaml

  3. Watch the load balancer come up (first provision takes a few minutes):

       kubectl get gateway avaloka-gateway -n <namespace> -w
       gcloud certificate-manager certificates describe avaloka-cert-${HOST_LIST[0]//./-} \\
         --project ${PROJECT} --format='value(managed.state)'
EOF
