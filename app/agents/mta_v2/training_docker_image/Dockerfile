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

ENV PYTHONUNBUFFERED=1
ENV RAY_BACKEND_LOG_LEVEL=error
ENV RAY_DISABLE_IMPORT_WARNING=1
# Ray 2.52 turns Train V2 on by default (2.49 had it off). ray_job.py's
# TorchTrainer / train.report / Checkpoint flow was written and tested on V1;
# keep V1 until it is ported and run on a cluster. Ray propagates this
# variable from the driver to the train workers.
ENV RAY_TRAIN_V2_ENABLED=0

WORKDIR /app

# ---- System deps ----
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- App code ----
COPY . /app

WORKDIR /app
ENV PYTHONPATH="/app:${PYTHONPATH}"
