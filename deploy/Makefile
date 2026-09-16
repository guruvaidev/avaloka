# deploy/Makefile — one-command avaloka k8s deployment.
#
# Usage:
#   make up PROVIDER=local                 # kind + KubeRay + avaloka + Ray Serve
#   make up PROVIDER=gcp                    # GKE (requires gcloud creds + GCP_PROJECT_ID)
#   make up PROVIDER=aws                    # EKS (requires aws creds + AWS_* vars)
#   make connect RAY_ADDRESS=ray://h:10001  # attach to an existing Ray cluster
#   make status
#   make down PROVIDER=local
#
# GROQ_API_KEY_PLANNING_AGENT / GROQ_API_KEY_CODING_AGENT are read from the env
# and injected into the avaloka Secret at install time.

PROVIDER    ?= local
NAMESPACE   ?= default
RAY_ADDRESS ?=
DATA_STACK  ?=

# Run python from the repo root (this Makefile lives in deploy/).
ROOT := $(realpath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PY   := cd $(ROOT) && python -m app.infra.cluster_bootstrap

.PHONY: up connect status down images help

up:
	$(PY) --mode provision --provider $(PROVIDER) --namespace $(NAMESPACE) \
		$(if $(DATA_STACK),--data-stack $(DATA_STACK),)

connect:
	@test -n "$(RAY_ADDRESS)" || (echo "RAY_ADDRESS is required, e.g. make connect RAY_ADDRESS=ray://host:10001"; exit 1)
	$(PY) --mode connect --ray-address $(RAY_ADDRESS) --namespace $(NAMESPACE)

images:
	cd $(ROOT) && python -c "from app.infra.deploy_stack import build_images; \
		[print(o['status'], o['step_name']) for o in build_images(load_into_kind=($(if $(filter local,$(PROVIDER)),True,False)))]"

status:
	kubectl get pods,svc -n $(NAMESPACE)
	@echo "--- Ray ---"
	-kubectl get rayclusters,rayservices -n $(NAMESPACE)

down:
	@echo "Tearing down ($(PROVIDER))..."
	-helm uninstall avaloka -n $(NAMESPACE)
	-kubectl delete -f $(ROOT)/deploy/helm/ray/rayservice.yaml -n $(NAMESPACE) --ignore-not-found
	-kubectl delete -f $(ROOT)/deploy/helm/ray/raycluster.yaml -n $(NAMESPACE) --ignore-not-found
	cd $(ROOT) && python -c "from app.infra import cloud_provisioner as c; print(c.teardown('$(PROVIDER)'))"

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) || sed -n '1,20p' $(MAKEFILE_LIST)

# --------------------------------------------------------------------------- #
# Images: pull what already exists, build only what changed.
# --------------------------------------------------------------------------- #
REGISTRY ?= ghcr.io/guruvaidev
PY_BIN   ?= python

# The content tag for each image. Same computation CI uses, so a tag that
# exists in the registry means that exact image has already been built.
api-tag:
	@$(PY_BIN) $(ROOT)/scripts/image_tag.py api --source

# Pull the images matching this checkout. Falls back to building only those
# that are genuinely absent, which is the case this exists to make rare.
images-pull:
	@set -e; 	for img in api ray ui functions; do 	  tag=$$($(PY_BIN) $(ROOT)/scripts/image_tag.py $$img --source); 	  ref="$(REGISTRY)/avaloka-$$img:$$tag"; 	  if docker manifest inspect "$$ref" >/dev/null 2>&1; then 	    echo "pull  $$ref"; docker pull -q "$$ref"; 	    if [ "$(PROVIDER)" = "local" ]; then kind load docker-image "$$ref" --name avaloka || true; fi; 	  else 	    echo "MISS  $$ref  (not published; run 'make images' to build it)"; 	  fi; 	done

# What would a deployment have to rebuild?
images-status:
	@$(PY_BIN) $(ROOT)/scripts/image_tag.py --all | while read name deps src; do 	  ref="$(REGISTRY)/avaloka-$$name:$$src"; 	  if docker manifest inspect "$$ref" >/dev/null 2>&1; then 	    printf "  %-10s up to date (%s)\n" "$$name" "$$src"; 	  else 	    printf "  %-10s NEEDS BUILD (%s)\n" "$$name" "$$src"; 	  fi; 	done
