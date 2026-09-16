FROM python:3.12-slim

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