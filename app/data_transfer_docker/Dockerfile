# This Dockerfile builds BOTH DTA runtime images:
#   - avaloka-dta-local:latest                       (the local Docker runner on the VM)
#   - gcr.io/.../avaloka-dta-ray-cluster             (the dedicated ray-gke-dta cluster)
# The running ray-gke-dta cluster is pinned by DIGEST (deploy/k8s/raycluster-image-patch.json) and
# still runs ray 2.10 + daft 0.7.14 — rebuilding/redeploying it from this file moves it
# to ray 2.53, which requires a KubeRay-operator compat check and a planned restart
# (see the teammate's cluster-change doc: do it at a quiet time). Rebuilding the LOCAL
# image is safe and isolated.
#
# Use the official Ray image as the base.
#
# 2.56.0 matches the app env (requirements.txt pins ray==2.56.0), so the injection
# scripts the planner generates and validates run on the same ray/daft/pyarrow stack
# inside this container. The old 2.10.0 base broke against modern daft: daft's
# flotilla error handler references ray.exceptions.ActorDiedError (added after 2.10),
# so any task failure crashed with AttributeError instead of surfacing the real error.
FROM rayproject/ray:2.56.0-py311
# Third-party usage beacons, and botocore's instance-metadata credential probe,
# off by default (app/core/thirdparty_optout.py says what each one silences:
# daft -> scarf.sh, onnxruntime -> events.data.microsoft.com, chroma, mlflow,
# HF hub, dvc, Ray usage stats, streamlit, langgraph-cli and its PyPI version check; 169.254.169.254). A deployment opts
# back in by setting the variable itself, e.g. AWS_EC2_METADATA_DISABLED=false
# to use the VM's instance credentials.
ENV DO_NOT_TRACK=true \
    SCARF_NO_ANALYTICS=true \
    ORT_DISABLE_TELEMETRY=1 \
    ANONYMIZED_TELEMETRY=False \
    MLFLOW_DISABLE_TELEMETRY=true \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DVC_NO_ANALYTICS=1 \
    RAY_USAGE_STATS_ENABLED=0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    LANGGRAPH_CLI_NO_ANALYTICS=1 \
    LANGGRAPH_NO_VERSION_CHECK=true \
    AWS_EC2_METADATA_DISABLED=true

# Switch to root to install system dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Fix: Set the correct working directory!
WORKDIR /home/ray/app

# Copy and install requirements first (to cache the layer)
COPY app/data_transfer_docker/runtime-requirements.txt .
RUN pip install --no-cache-dir -r runtime-requirements.txt

# Copy the app source code
COPY app/ ./app/
# Copy the entrypoint script into /home/ray/app/
COPY app/data_transfer_docker/container_file.py .

RUN mkdir -p /home/ray/app/external /home/ray/app/data /mnt/gcs/transfers

ENV PYTHONPATH="/home/ray/app:/home/ray/app/external"

# Switch back to the 'ray' user (default for the base image)
USER ray

# KubeRay overrides the entrypoint, but we can set it for manual container runs
ENTRYPOINT ["python", "container_file.py"]