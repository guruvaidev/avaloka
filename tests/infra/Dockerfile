FROM python:3.9-slim
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
COPY . .
RUN pip install pandas
CMD ["python", "process.py"]
