import os
import asyncio
import logging
import requests
from typing import Union, List, Dict
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

from loader.url import _is_local, _is_http, _gcs_storage_options, _s3_storage_options
from inference import InferenceInterface

logger = logging.getLogger(__name__)
DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING = int(os.getenv("DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING") or 10 * 1024 * 1024)  # 10 MB threshold for local vs Ray training
MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID", "")


def setup_gcp_credentials():
    """
    Set up GCP credentials from environment variables.

    Effective credential path priority:
      1. GOOGLE_APPLICATION_CREDENTIALS
      2. /tmp/gcp_sa.json
    """
    print("[GCP Credentials] Setting up GCP credentials for Ray worker...")
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "").strip()
    if sa_json:
        sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "/tmp/gcp_sa.json"
        Path(sa_path).write_text(sa_json, encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        return sa_path
    return None


def setup_mlflow():
    """
    Configure MLflow from environment variables.

    Effective URI priority:
      1. MLFLOW_TRACKING_URI
      2. MLFLOW_BACKEND_STORE_URI
      3. local ./mlruns
    """
    tracking_uri    = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    backend_store   = os.getenv("MLFLOW_BACKEND_STORE_URI", "").strip()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or "avaloka-training"

    effective_uri = tracking_uri or backend_store or "mlruns"
    import mlflow
    mlflow.set_tracking_uri(effective_uri)
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] tracking_uri={effective_uri}  experiment={experiment_name}")


@asynccontextmanager
async def service_lifespan(_: FastAPI):
    setup_gcp_credentials()
    setup_mlflow()
    yield

app = FastAPI(
    title="Avaloka AI Inference Service",
    lifespan=service_lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


def get_file_size(uri: str):
    """
    Return file size in bytes for local, HTTP, GCS, or S3.
    Returns None if size cannot be determined.
    """
    try:
        # -------------------------
        # Local file
        # -------------------------
        if _is_local(uri):
            path = uri.replace("file://", "")
            if os.path.isfile(path):
                return os.path.getsize(path)
            return None

        # -------------------------
        # HTTP(S)
        # -------------------------
        if _is_http(uri):
            resp = requests.head(uri, allow_redirects=True, timeout=10)
            size = resp.headers.get("Content-Length")
            return int(size) if size else None

        # -------------------------
        # GCS
        # -------------------------
        if uri.startswith("gs://"):
            import gcsfs
            fs = gcsfs.GCSFileSystem(**_gcs_storage_options())
            info = fs.info(uri)
            return info.get("size")

        # -------------------------
        # S3
        # -------------------------
        if uri.startswith("s3://"):
            import s3fs
            fs = s3fs.S3FileSystem(**_s3_storage_options())
            info = fs.info(uri)
            return info.get("size")

    except Exception as e:
        logger.error(f"Could not determine file size: {uri}")

    return None

def is_local_data_source(data_uri: str, dataset_size_bytes: int | None) -> bool:
    """
    Return True when the data source is a local path and no cloud location
    has been provided.

    A source is considered "local" when the resolved URI does NOT start with
    a recognised cloud/remote scheme. SQL connection strings (postgresql://,
    mysql://, sqlite://, mssql://) are treated as remote/server-side and
    therefore use RayTrainer, not LocalTrainer.
    """
    uri = (data_uri or "").strip()
    if not uri:
        return True

    dataset_size_bytes = get_file_size(uri) or dataset_size_bytes or 0
    if dataset_size_bytes < DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING:
        logger.info(f"Dataset size {dataset_size_bytes} bytes is below threshold for Ray training; treating as local data source.")
        return True

    REMOTE_SCHEMES = (
        "gs://", "s3://", "az://", "abfs://",
        "http://", "https://",
        "postgresql://", "mysql://", "sqlite://", "mssql://",
    )
    return not any(uri.startswith(scheme) for scheme in REMOTE_SCHEMES)


@app.post("/inference")
async def inference(
    input_data: Union[Dict, List] = Body(...)
):
    inferencer = InferenceInterface()

    try:
        # -------------------------
        # Single dict input
        # -------------------------
        if isinstance(input_data, dict):
            if "dataset_uri" in input_data:
                is_local = is_local_data_source(input_data["dataset_uri"], None)

                if is_local:
                    logger.info(
                        "Inference input data source is local based on URI: %s",
                        input_data["dataset_uri"]
                    )
                    result = await asyncio.to_thread(
                        inferencer.inference_local,
                        MLFLOW_RUN_ID,
                        input_data["dataset_uri"]
                    )
                else:
                    logger.info(
                        "Inference input data source is ray based on URI: %s",
                        input_data["dataset_uri"]
                    )
                    result = await asyncio.to_thread(
                        inferencer.inference_on_ray,
                        MLFLOW_RUN_ID,
                        {},
                        input_data["dataset_uri"]
                    )
            else:
                logger.info(
                    "Inference input data is single instance based on JSON structure."
                )
                result = await asyncio.to_thread(
                    inferencer.predict,
                    MLFLOW_RUN_ID,
                    input_data
                )

        # -------------------------
        # Batch list input
        # -------------------------
        elif isinstance(input_data, list):
            logger.info(
                "Inference input data is batch based on JSON structure."
            )
            result = await asyncio.to_thread(
                inferencer.predict_batch,
                MLFLOW_RUN_ID,
                input_data
            )

        else:
            raise HTTPException(400, "Invalid input_data format")

        return result

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            500,
            f"Error occurred during inference: {e}"
        )


@app.get("/health")
async def health():
    return {
        "status": "OK"
    }