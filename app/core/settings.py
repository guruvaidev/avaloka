from dotenv import load_dotenv
load_dotenv()
import os
from dataclasses import dataclass

@dataclass
class Settings:
    # storage
    storage_backend: str = os.getenv("STORAGE_BACKEND", "gcs").lower()   # gcs|s3
    gcs_bucket: str = os.getenv("GCS_BUCKET") or os.getenv("GCS_BUCKET_NAME", "")
    gcs_prefix: str = os.getenv("GCS_PREFIX", "").strip().strip("/")
    
    aws_region: str = os.getenv("AWS_REGION", "")
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_prefix: str = os.getenv("S3_PREFIX", "").strip().strip("/")
    # Override the S3 endpoint to talk to an S3-compatible store instead of AWS —
    # in-cluster MinIO (local/kind/on-prem), Ceph RGW, or any other. Empty means
    # real AWS S3. AWS_ENDPOINT_URL is the name botocore itself understands, so a
    # deployment that sets only that still works.
    s3_endpoint_url: str = (
        os.getenv("S3_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL", "")
    ).strip()

    # AZURE
    azure_account: str = os.getenv("AZURE_ACCOUNT", "")                       # e.g. "avalokastorage"
    azure_container: str = os.getenv("AZURE_CONTAINER", "")                   # e.g. "avaloka-test-blob-container"
    azure_prefix: str = os.getenv("AZURE_PREFIX", "").strip().strip("/")
    azure_sas: str = os.getenv("AZURE_SAS", "")                               # entire "?sp=...&sig=..." string

    # cache
    redis_url: str = os.getenv("REDIS_URL") or (
        f"redis://{os.getenv('REDISHOST')}:{os.getenv('REDISPORT')}/0"
        if os.getenv("REDISHOST") and os.getenv("REDISPORT") else ""
    )
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", str(7 * 24 * 3600)))

    # upstreams
    mcp_server_url: str = os.getenv("MCP_SERVER_URL", "http://localhost:8080")
    mcp_timeout: float = float(os.getenv("MCP_TIMEOUT", "30.0"))
