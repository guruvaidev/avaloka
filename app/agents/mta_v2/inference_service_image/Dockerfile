FROM python:3.12-slim
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

WORKDIR /app

# Improve logging
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Runtime environment variables (overridden by GKE)
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=""
ENV MLFLOW_EXPERIMENT_NAME=""
ENV MLFLOW_RUN_ID=""

ENV GOOGLE_CLOUD_PROJECT=""
ENV RAY_GKE_CLUSTER_NAME=""
ENV RAY_GKE_CLUSTER_LOCATION=""
ENV RAY_INFERENCE_DOCKER_URI=""

ENV GCP_SERVICE_ACCOUNT_JSON=""
ENV GOOGLE_APPLICATION_CREDENTIALS=""
ENV DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING=""

# Expose the port for the inference service
ENV PORT=8080
EXPOSE 8080

# Start the inference server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]