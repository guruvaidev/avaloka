# state.py

import re
from typing import Annotated, List, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import pandas as pd
from pydantic import BaseModel, Field, SecretStr

# Matches the "user:password@" segment of a URL so the password can be masked.
_CONN_PW_RE = re.compile(r"(?<=//)([^/@\s]*?):([^/@\s]*)@")


def redact_conn_str(value: Optional[str]) -> str:
    """Mask the password in a database URL so it is safe to log.

    ``mysql+pymysql://root:secret@host:3309/db`` -> ``mysql+pymysql://root:***@host:3309/db``

    URLs without a password (and non-URL values such as ``gs://bucket/obj``) are
    returned unchanged. Anything that fails to parse is masked entirely rather
    than risking a leak.
    """
    if not value:
        return ""
    try:
        return _CONN_PW_RE.sub(r"\1:***@", str(value))
    except Exception:
        return "***"


class db_credentials(BaseModel):
    db_type: str = Field(default="mysql", description="Database dialect (e.g., mysql, postgres, sqlite)")
    host: str = Field(..., description="Hostname or IP address of the database server")
    port: int = Field(..., description="Port number of the database server")
    database: str = Field(..., description="Name of the target database")
    user: str = Field(..., description="Username for authentication")
    password: SecretStr = Field(..., description="Password for authentication")
    table: Optional[str] = Field(default=None, description="Target table name, if required for the specific transfer task")
    schema_name: Optional[str] = Field(default=None, description="Database schema, primarily used in PostgreSQL or SQL Server")
    sslmode: Optional[str] = Field(default=None, description="Postgres SSL mode, e.g. 'require' or 'verify-full'")


class cloud_storage_credentials(BaseModel):
    provider: str = Field(..., description="Cloud provider (e.g., 'aws', 'gcp', 'azure')")
    bucket_name: str = Field(..., description="Name of the S3 bucket, GCP bucket, or Azure container")
    file_path: str = Field(..., description="Path to the file within the bucket, e.g. 'folder/file.parquet'")

    # AWS / S3 Compatible
    access_key: Optional[str] = Field(default=None, description="AWS Access Key ID")
    secret_key: Optional[SecretStr] = Field(default=None, description="AWS Secret Access Key")
    region: Optional[str] = Field(default=None, description="Cloud region (e.g., 'us-east-1')")

    # Google Cloud Platform (GCP)
    gcp_service_account_info: Optional[Dict[str, Any]] = Field(default=None, description="GCP Service Account JSON dictionary")

    # Azure Blob Storage
    az_storage_account: Optional[str] = Field(default=None, description="Azure Storage Account name")
    az_access_key: Optional[SecretStr] = Field(default=None, description="Azure Storage Account access key")
    az_sas_token: Optional[SecretStr] = Field(default=None, description="Azure SAS token")
    az_tenant_id: Optional[str] = Field(default=None, description="Azure tenant ID for service principal auth")
    az_client_id: Optional[str] = Field(default=None, description="Azure client ID for service principal auth")
    az_client_secret: Optional[SecretStr] = Field(default=None, description="Azure client secret for service principal auth")

    def get_cloud_uri(self, file_type: str) -> str:
        """
        Constructs the full cloud URI for the file.
        e.g. gs://my-bucket/folder/file.parquet
        """
        prefix_map = {
            "gcp": "gs",
            "aws": "s3",
            "azure": "az",
        }
        prefix = prefix_map.get(self.provider.lower())
        if not prefix:
            raise ValueError(f"Unsupported cloud provider: {self.provider}")
        return f"{prefix}://{self.bucket_name}/{self.file_path}"


class DaftCodingAgentState(TypedDict):
    """Represents the state passed between agent nodes."""
    # --- Inputs from User & Planner ---
    user_prompt: str
    data_source_location: str
    source_schema: Dict[str, str]
    input_sample_data: Optional[pd.DataFrame]
    requirements: Optional[str]
    uploaded_csv_preview: Optional[List[List[str]]]

    # --- Artifacts & Feedback Loop ---
    coder_pseudocode: Optional[str]
    pseudocode_validation_feedback: Optional[str]

    generated_code: Optional[str]
    code_validation_feedback: Optional[str]

    syntax_error: Optional[bool]
    static_semantic_error: Optional[bool]
    logical_semantic_error: Optional[bool]
    execution_stdout: Optional[str]
    execution_stderr: Optional[str]
    execution_error: Optional[str]

    execution_output_data: Optional[object]
    execution_output_preview: Optional[object]

    code_executed: Optional[bool]
    code_validated: Optional[bool]
    retry_count: Optional[int]
    llm_raw_response: Optional[str]
    primary_llm_response: Optional[str]
    rag_retrieved_docs: Optional[str]
    generated_output_schema: Optional[str]


class DaftETLState(TypedDict):

    transfer_run_id: Optional[str]
    user_query: str
    messages: Annotated[List[BaseMessage], lambda x, y: x + [m for m in y if m not in x]]

    source_type: str
    source_db_credentials: Optional[db_credentials]
    source_cloud_credentials: Optional[cloud_storage_credentials]
    source_connection_string: Optional[str]
    source_table: Optional[str]
    source_file: Optional[str]
    source_schema: Optional[dict]
    source_connect_args: Optional[dict]
    source_io_config: Optional[Any]  # daft.io.IOConfig — typed as Any to avoid hard import in state.py

    destination_type: str
    destination_db_credentials: Optional[db_credentials]
    destination_cloud_credentials: Optional[cloud_storage_credentials]
    destination_connection_string: Optional[str]
    destination_table: Optional[str]
    destination_file: Optional[str]
    write_mode: Optional[str]
    destination_schema: Optional[dict]
    destination_sslmode: Optional[str]
    destination_connect_args: Optional[dict]
    destination_io_config: Optional[Any]  # daft.io.IOConfig — typed as Any to avoid hard import in state.py

    coder_definition: DaftCodingAgentState

    code_validated: bool
    code_generated_successfully: bool

    expected_output_schema: Optional[dict]
    destination_schema_compatible: bool

    error_message: Optional[str]
    warnings: Optional[List[str]]