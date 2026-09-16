import os
import copy
import json
import re
import uuid
import logging
import requests
import pandas as pd
from dotenv import load_dotenv
from typing import Any, Optional, Sequence
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

from app.core.agent_llm import build_agent_llm
from app.graph.etl_state import ETLState
from app.agents.mta_v2.ray_trainer import RayTrainer
from app.agents.mta_v2.local_trainer import LocalTrainer
from app.agents.mta_v2.inference import InferenceInterface
from app.agents.mta_v2.inference_service_manager import InferenceServiceManager
from app.agents.mta_v2.failure_diagnostics import build_training_failure, format_training_failure
from app.agents.mta_v2.schema import HyperparameterConfig, RayConfig, DataConfig, TrainingPlan
from app.agents.mta_v2.training_docker_image.src.data_integrity import (
    classify_identifier_features,
)
from app.agents.mta_v2.utils import get_columns, load_dataset, extract_json_from_content
from app.agents.mta_v2.loader.url import _is_local, _is_http, _gcs_storage_options, _s3_storage_options
from app.agents.pii_agent import (
    MissingPIIKey,
    deidentification_contract,
    pii_key_required_columns,
    require_pii_key,
    scan_dataframe,
)

load_dotenv()

logger = logging.getLogger(__name__)

DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING = int(os.getenv("DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING", 10 * 1024 * 1024))  # 10 MB threshold for local vs Ray training
REMOTE_DATA_SOURCE_SCHEMES = (
    "gs://", "gcs://", "s3://", "az://", "abfs://",
    "http://", "https://",
    "postgresql://", "mysql://", "sqlite://", "mssql://",
)
_planner_api_key = os.environ.get("GROQ_API_KEY_PLANNING_AGENT")
# Hybrid-reasoning MTA (verified live: gpt-oss honors tool_choice="none" for
# the conversational replies and "required" for the tool-pick call). Effort
# defaults to "low" to keep natural-language replies snappy; raise via
# AVALOKA_MTA_REASONING_EFFORT. Set AVALOKA_MTA_MODEL=llama-3.3-70b-versatile
# to restore the previous non-reasoning behavior without a code change.
llm = build_agent_llm(
    agent="MTA",
    api_key=_planner_api_key,
    default_model="openai/gpt-oss-120b",
    temperature=0.2,
    default_effort="low",
)
if llm is None:
    logger.warning(
        "Planner LLM disabled; set GROQ_API_KEY_PLANNING_AGENT to re-enable remote generation."
    )

TOOLS = [
    {
        "type": "function",
        "description": "Generate training plan for model training to show to user and get confirmation before execution",
        "function": {
            "name": "generate_training_plan",
            "parameters": {}
        },
    },
    {
        "type": "function",
        "description": "Ask more details to user to update training plan (not start training confirmation) because user didn't provide enough information to make update on training plan",
        "function": {
            "name": "ask_more_detail_to_update_training_plan",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "description": "Update training plan based on user feedback and confirmation from user provided enough information to make update on training plan (not start training confirmation)",
        "function": {
            "name": "update_training_plan",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "description": "Generate code repository for model training based on the training plan",
        "function": {
            "name": "generate_code_repo",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "description": "Prepare jupyterlab env with generated code repo so that user can review and update repo",
        "function": {
            "name": "review_code_repo",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "description": "Execute model training based on the training plan and code repo",
        "function": {
            "name": "execute_training",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "description": "Configure the inference service for the trained model",
        "function": {
            "name": "configure_inference_service",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "description": "Execute inference using the trained model and its configured service when available",
        "function": {
            "name": "execute_inference",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "description": "Stop the inference service",
        "function": {
            "name": "stop_inference_service",
            "parameters": {}
        }
    },
]

tool_descriptions = "\n".join(
    f"    {idx}. {tool['function']['name']}: {tool['description']}"
    for idx, tool in enumerate(TOOLS, 1)
)
SYSTEM_PROMPT = f"""
You are a helpful and precise assistant for model training and inference tasks for Avaloka AI - AI Data Science Co-Pilot System.
Your goal is to determine the best training plan and build training code repo, execute training and inference based on user requests and feedback.
You have access to following tools to accomplish the task:
{tool_descriptions}
**NOTE**: No need any parameters for all tools for now. Just call the tools to trigger the flow, and the tools will get necessary information from the state and do the job.

The flows for training and inference tasks is as follows:
1. User request training at the first time: (only one tool) -> generate_training_plan
2. User want to update training plan (only one tool)
   2.1 User didn't provide enough information -> ask_more_detail_to_update_training_plan
   2.2 User provided enough information -> update_training_plan
3. User confirm training plan (3 tools) -> generate_code_repo -> review_code_repo -> execute_training
4. User request a sample prediction / in-chat inference (only one tool) -> execute_inference
5. User explicitly asks to configure, deploy, start, or expose an inference service/API (only one tool) -> configure_inference_service
6. User requests that the inference service be stopped (only one tool) -> stop_inference_service

Important:
1. You can't go to training without training plan and its confirmation
2. You can't execute inference without a training result and trained model. When a service is configured, chat inference must use it.
3. Configure or stop the service only when the user explicitly requests that action.
4. Do not call configure_inference_service and execute_inference together.

You should meet one of these request. So determine what flow is best for current state, and call all of tools of that flow to accomplish the task. Always think step by step and be precise about which tools to use.
You can call multiple tools in one response if necessary.
"""


class ModelTrainingAgent:
    """
    LangChain-based agent that orchestrates the full model training and
    inference lifecycle for the Avaloka AI Co-Pilot.

    The agent uses a Groq-hosted LLM (llama-3.3-70b-versatile) in
    tool-calling mode to decide which step to execute next, then delegates
    to the appropriate private method.

    Training backend selection
    --------------------------
    When ``_execute_training`` is called the agent inspects the state to
    decide which backend to use:

    - **RayTrainer** (default) — used when ``data_source_location_cloud`` is
      set, or when the resolved data URI starts with a recognised remote scheme
      (gs://, s3://, http(s)://, any SQL dialect).
    - **LocalTrainer** — used when only a local filesystem path is available
      (no cloud URI anywhere in the state). Runs in-process without Ray.

    Public entry point
    ------------------
    execute(state) → ETLState
        Inspect the conversation, call the appropriate handler(s), and
        return the updated state.
    """

    def _get_file_size(self, uri: str) -> Optional[int]:
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

    # ----------------------------------------------------------
    # Data source inspection
    # ----------------------------------------------------------

    def _is_remote_data_source(self, uri: str) -> bool:
        return (uri or "").strip().lower().startswith(REMOTE_DATA_SOURCE_SCHEMES)

    def _truthy_state_value(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def _local_path_from_uri(self, uri: str) -> str:
        return (uri or "").replace("file://", "", 1)

    def _state_dataset_size_bytes(self, state: ETLState) -> Optional[int]:
        for key in ("dataset_size_bytes", "file_size_bytes", "file_size", "data_size_bytes"):
            value = state.get(key)
            if value is None:
                continue
            try:
                size = int(value)
                # Upload/session records historically use zero as an
                # "unavailable" placeholder. Treating it as a real object size
                # prevents _training_backend_decision() from querying MinIO/GCS
                # and can incorrectly route a large remote dataset locally.
                if size > 0:
                    return size
            except (TypeError, ValueError):
                continue

        for message in reversed(self._normalize_messages(state.get("messages", []))):
            content = getattr(message, "content", "") or ""
            match = re.search(r"Dataset size bytes:\s*(\d+)", content, flags=re.IGNORECASE)
            if match:
                size = int(match.group(1))
                if size > 0:
                    return size
        return None

    def _active_dataset_candidates(self, state: ETLState) -> tuple[str, ...]:
        candidates: list[str] = []
        for key in (
            "active_data_source_location_cloud",
            "active_data_source_location",
            "active_data_source_location_local",
        ):
            uri = (state.get(key) or "").strip()
            if uri and uri not in candidates:
                candidates.append(uri)
        return tuple(candidates)

    def _modified_dataset_is_active(self, state: ETLState) -> bool:
        return (
            self._truthy_state_value(state.get("data_source_was_modified"))
            or self._truthy_state_value(state.get("activate_output_as_dataset"))
        )

    def _modified_dataset_feasibility(self, uri: str, state: ETLState) -> tuple[bool, Optional[str]]:
        uri = (uri or "").strip()
        if not uri:
            return False, "The modified dataset output is not available for training."

        if self._is_remote_data_source(uri):
            return True, None

        local_path = self._local_path_from_uri(uri)
        if not os.path.isfile(local_path):
            return False, (
                "The modified dataset output is no longer available as a readable local file. "
                "Re-run the transformation or upload the modified output as a dataset before training."
            )

        size_bytes = self._get_file_size(uri)
        if (
            size_bytes is not None
            and size_bytes >= DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING
        ):
            return False, (
                "The modified dataset is local-only and large enough that it should be trained with Ray, "
                "but Ray cannot read that local output file. Upload or persist the modified dataset to cloud storage first."
            )

        try:
            columns = get_columns(uri)
        except Exception as exc:
            return False, f"The modified dataset output could not be inspected for training columns: {exc}"
        if not columns:
            return False, "The modified dataset output has no columns to train on."

        return True, None

    def _resolve_modified_training_data_location(self, state: ETLState) -> tuple[str, Optional[str]]:
        if not self._modified_dataset_is_active(state):
            return "", None

        candidates = self._active_dataset_candidates(state)
        if not candidates:
            return "", (
                "The previous step marked its output as the active dataset, but no modified dataset path was recorded. "
                "Re-run the transformation or upload the modified output as a dataset before training."
            )

        last_reason: Optional[str] = None
        for candidate in candidates:
            feasible, reason = self._modified_dataset_feasibility(candidate, state)
            if feasible:
                return candidate, None
            last_reason = reason

        return "", last_reason or "The modified dataset output is not feasible for training."

    def _modified_dataset_unavailable_reason(self, state: ETLState) -> Optional[str]:
        _, reason = self._resolve_modified_training_data_location(state)
        return reason

    def _latest_output_columns_from_state(self, state: ETLState) -> list[str]:
        columns = state.get("latest_output_columns")
        if isinstance(columns, str):
            try:
                columns = json.loads(columns)
            except Exception:
                columns = []
        if isinstance(columns, list):
            return [str(column) for column in columns]
        return []

    def _latest_output_candidates(self, state: ETLState) -> tuple[str, ...]:
        candidates: list[str] = []
        for key in (
            "latest_output_location",
            "latest_output_location_local",
        ):
            uri = (state.get(key) or "").strip()
            if uri and uri not in candidates:
                candidates.append(uri)
        return tuple(candidates)

    def _uris_match(self, left: str, right: str) -> bool:
        return self._local_path_from_uri(left).strip() == self._local_path_from_uri(right).strip()

    def _latest_output_requested(self, state: ETLState) -> bool:
        text = self._latest_user_text(state).lower()
        if not text:
            return False

        for candidate in self._latest_output_candidates(state):
            basename = os.path.basename(self._local_path_from_uri(candidate)).lower()
            if basename and re.search(rf"(?<!\w){re.escape(basename)}(?!\w)", text):
                return True

        output_reference_patterns = (
            r"\b(?:latest|recent|generated|transformed|modified|new)\b"
            r".{0,80}\b(?:output|dataframe|dataset|data|csv|file)\b",
            r"\b(?:output|dataframe|dataset|data|csv|file)\b"
            r".{0,80}\b(?:latest|recent|generated|transformed|modified|new)\b",
        )
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in output_reference_patterns)

    def _latest_output_is_trainable(self, state: ETLState) -> bool:
        explicit_flag = state.get("latest_output_is_trainable")
        if explicit_flag is not None:
            return self._truthy_state_value(explicit_flag)

        columns = self._latest_output_columns_from_state(state)
        if len(columns) < 2:
            return False

        row_count = state.get("latest_output_row_count")
        try:
            return row_count is None or int(row_count) >= 2
        except (TypeError, ValueError):
            return True

    def _latest_output_feasibility(self, uri: str, state: ETLState) -> tuple[bool, Optional[str]]:
        uri = (uri or "").strip()
        if not uri:
            return False, "The latest generated output is not available for training."

        if not self._latest_output_is_trainable(state):
            return False, (
                "The latest generated output does not look trainable. "
                "Create or select an output dataframe with at least two rows and two columns before training."
            )

        if self._is_remote_data_source(uri):
            return True, None

        local_path = self._local_path_from_uri(uri)
        if not os.path.isfile(local_path):
            return False, (
                "The latest generated output is no longer available as a readable local file. "
                "Re-run the transformation or upload the output as a dataset before training."
            )

        size_bytes = self._get_file_size(uri)
        if (
            size_bytes is not None
            and size_bytes >= DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING
        ):
            return False, (
                "The latest generated output is local-only and large enough that it should be trained with Ray, "
                "but Ray cannot read that local output file. Upload or persist the output to cloud storage first."
            )

        columns = self._columns_for_training_data(uri, state)
        if not columns:
            return False, "The latest generated output has no columns to train on."

        return True, None

    def _resolve_latest_output_training_data_location(self, state: ETLState) -> tuple[str, Optional[str]]:
        candidates = self._latest_output_candidates(state)
        if not candidates:
            return "", "No generated output dataframe has been recorded for this session."

        last_reason: Optional[str] = None
        for candidate in candidates:
            feasible, reason = self._latest_output_feasibility(candidate, state)
            if feasible:
                return candidate, None
            last_reason = reason

        return "", last_reason or "The latest generated output is not feasible for training."

    def _latest_output_unavailable_reason(self, state: ETLState) -> Optional[str]:
        _, reason = self._resolve_latest_output_training_data_location(state)
        return reason

    def _preferred_training_data_unavailable_reason(self, state: ETLState) -> Optional[str]:
        modified_reason = self._modified_dataset_unavailable_reason(state)
        if modified_reason:
            return modified_reason
        if self._latest_output_requested(state):
            return self._latest_output_unavailable_reason(state)
        return None

    def _resolve_training_data_location(self, state: ETLState) -> str:
        """
        Pick the dataset URI for model training.

        Training prefers an activated modified dataset when one is available
        and feasible. Otherwise, it falls back to the full cloud object when
        available. Chat analysis may be running against a quick/portfolio
        sample, but model training must not silently inherit that local sample
        path when a full cloud source exists.
        """
        modified_uri, modified_reason = self._resolve_modified_training_data_location(state)
        if modified_uri:
            return modified_uri
        if modified_reason:
            logger.warning("Modified dataset is active but not feasible for training: %s", modified_reason)
            return ""

        if self._latest_output_requested(state):
            latest_uri, latest_reason = self._resolve_latest_output_training_data_location(state)
            if latest_uri:
                return latest_uri
            if latest_reason:
                logger.warning("Latest output was requested but is not feasible for training: %s", latest_reason)
                return ""

        remote_candidates = (
            state.get("full_data_source_location_cloud"),
            state.get("active_data_source_location_cloud"),
            state.get("data_source_location_cloud"),
            state.get("data_source_location"),
            state.get("active_data_source_location"),
        )
        for candidate in remote_candidates:
            uri = (candidate or "").strip()
            if self._is_remote_data_source(uri):
                return uri

        local_candidates = (
            state.get("active_data_source_location"),
            state.get("active_data_source_location_local"),
            state.get("full_data_location"),
            state.get("data_source_location"),
            state.get("data_source_location_local"),
        )
        for candidate in local_candidates:
            uri = (candidate or "").strip()
            if uri:
                return uri
        return ""

    def _schema_columns_from_state(self, state: ETLState) -> list[str]:
        schema = state.get("schema") or {}
        if isinstance(schema, dict) and schema:
            return [str(col) for col in schema.keys()]
        if isinstance(schema, list) and schema:
            return [str(col) for col in schema]
        uploaded_columns = state.get("uploaded_csv_columns")
        if isinstance(uploaded_columns, list):
            return [str(col) for col in uploaded_columns]
        return []

    def _columns_for_training_data(self, data_uri: str, state: ETLState) -> list[str]:
        try:
            columns = get_columns(data_uri)
        except Exception as exc:
            logger.warning("Could not inspect training columns from %s: %s", data_uri, exc)
            columns = []
        latest_columns = self._latest_output_columns_from_state(state)
        if latest_columns and any(
            self._uris_match(data_uri, candidate)
            for candidate in self._latest_output_candidates(state)
        ):
            return columns or latest_columns
        return columns or self._schema_columns_from_state(state)

    def _sample_for_identifier_detection(
        self,
        data_uri: str,
        state: ETLState,
        *,
        limit: int = 2_000,
    ) -> pd.DataFrame:
        """Return a bounded dataset sample for identifier-role detection."""
        for key in ("full_sample_rows", "quick_sample_rows"):
            rows = state.get(key)
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                return pd.DataFrame(rows[:limit])

        portfolio = state.get("portfolio_samples")
        if isinstance(portfolio, dict):
            selected = str(state.get("selected_sample_name") or "random_baseline")
            rows = portfolio.get(selected) or portfolio.get("random_baseline")
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                return pd.DataFrame(rows[:limit])

        preview = state.get("uploaded_csv_preview")
        preview_columns = state.get("uploaded_csv_columns")
        if (
            isinstance(preview, list)
            and preview
            and isinstance(preview_columns, list)
            and preview_columns
        ):
            return pd.DataFrame(preview[:limit], columns=preview_columns)

        # Local files and SQL sources can be sampled cheaply. Cloud object
        # samples normally come from sampler state; do not risk downloading a
        # large Parquet/JSON object merely to choose features.
        if _is_local(data_uri) or data_uri.startswith(
            ("postgresql://", "mysql://", "sqlite://", "mssql://")
        ):
            try:
                return load_dataset(data_uri, limit=limit)
            except Exception as exc:
                logger.warning(
                    "Could not sample training data for identifier detection from %s: %s",
                    data_uri,
                    exc,
                )
        return pd.DataFrame()

    @staticmethod
    def _identifier_decisions(
        columns: list[str],
        sample: Optional[pd.DataFrame] = None,
    ) -> dict[str, dict[str, Any]]:
        frame = sample if isinstance(sample, pd.DataFrame) else pd.DataFrame()
        return {
            str(decision["column"]): decision
            for decision in classify_identifier_features(frame, columns)
        }

    def _sample_for_pii_detection(
        self,
        data_uri: str,
        state: ETLState,
        columns: list[str],
        *,
        limit: int = 2_000,
    ) -> pd.DataFrame:
        """Return the bounded identifier sample restricted to PII columns."""
        sample = self._sample_for_identifier_detection(
            data_uri,
            state,
            limit=limit,
        )
        # Reindexing an empty sample retains column-name evidence, allowing the
        # PII scanner to catch customer_email/phone_number without downloading
        # a large remote object solely for validation.
        return sample.reindex(columns=columns)

    def _validate_training_pii_key(
        self,
        data_config: DataConfig,
        state: ETLState,
    ) -> None:
        """Require the configured PII key for selected key-protected columns."""
        selected_columns = list(dict.fromkeys(
            list(data_config.get("feature_columns") or [])
            + ([data_config.get("target_column")] if data_config.get("target_column") else [])
        ))
        if not selected_columns:
            data_config.pop("pii_key_required_columns", None)
            data_config.pop("pii_transformations", None)
            return

        data_uri = data_config.get("dataset_uri", "")
        sample = self._sample_for_pii_detection(
            data_uri,
            state,
            selected_columns,
        )
        report = scan_dataframe(sample, columns=selected_columns)
        required_columns = pii_key_required_columns(report)
        transformations = deidentification_contract(report)
        if required_columns:
            data_config["pii_key_required_columns"] = required_columns
        else:
            data_config.pop("pii_key_required_columns", None)
        if transformations:
            data_config["pii_transformations"] = transformations
        else:
            data_config.pop("pii_transformations", None)
        require_pii_key(report)

    @staticmethod
    def _normalize_column_token(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())

    def _canonical_training_column(self, value: Any, lower_to_actual: dict[str, str]) -> str:
        normalized = self._normalize_column_token(value)
        return lower_to_actual.get(normalized, "")

    def _explicit_feature_terms_from_text(self, text: str) -> set[str]:
        """
        Extract terms the user explicitly framed as feature/input columns.

        This is intentionally about clause structure, not metric names. A later
        sentence like "Report ..." should not become feature-column intent.
        """
        lowered = text or ""
        patterns = (
            r"\b(?:feature columns?|features?|input columns?|predictors?)\b\s*(?:only\s*)?(?:are|include|included|:|=)?\s*([^.;]+)",
            r"\buse(?: these| the)?\s+(?:feature\s+)?columns(?: only)?\s*(?:are|:|=)?\s*([^.;]+)",
            r"\busing\s+([^.;]+?)\s+\b(?:to\s+predict|for\s+predicting|as\s+features?)\b",
        )
        terms: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
                clause = match.group(1)
                clause = re.sub(r"[\[\]()`\"']", " ", clause)
                clause = re.sub(r"\b(?:and|or)\b", ",", clause, flags=re.IGNORECASE)
                for raw_part in re.split(r",|\n", clause):
                    part = re.sub(r"\s+", " ", raw_part).strip(" :-")
                    if part:
                        terms.add(self._normalize_column_token(part))
        return terms

    def _feature_was_explicitly_requested(self, feature: Any, state: ETLState) -> bool:
        explicit_terms = self._explicit_feature_terms_from_text(self._latest_user_text(state))
        return self._normalize_column_token(feature) in explicit_terms

    def _explicitly_requested_missing_target(
        self, extracted_target: str, columns: list[str], user_text: str
    ) -> str:
        """Return the column-like target name the user explicitly asked for when
        it does not exist in the dataset; "" otherwise.

        Two signals, both requiring the name to LOOK like an identifier
        (contains "_" or interior capitals) so plain-English words in phrasings
        like "predict whether a passenger survived" never trigger a refusal:
          1. the LLM echoed a non-column target that appears verbatim in the text
          2. the token immediately after a training verb is identifier-like and
             not a real column
        """
        lowered = (user_text or "").lower()
        col_tokens = {self._normalize_column_token(c) for c in columns}

        def _identifier_like(tok: str) -> bool:
            return "_" in tok or bool(re.search(r"[a-z][A-Z]", tok))

        candidate = str(extracted_target or "").strip()
        if (
            candidate
            and _identifier_like(candidate)
            and self._normalize_column_token(candidate) not in col_tokens
            and re.search(rf"(?<!\w){re.escape(candidate.lower())}(?!\w)", lowered)
        ):
            return candidate

        match = re.search(
            r"\b(?:predict|predicting|classify|classifying|classifier for|"
            r"forecast|forecasting|regress(?:ion)?\s+(?:on|for))\s+"
            r"(?:the\s+)?([A-Za-z][A-Za-z0-9_]*)",
            user_text or "",
            flags=re.IGNORECASE,
        )
        if match:
            token = match.group(1)
            if _identifier_like(token) and self._normalize_column_token(token) not in col_tokens:
                return token
        return ""

    def _requested_training_columns(self, data_config: DataConfig) -> list[str]:
        requested: list[str] = []
        target = data_config.get("target_column") or ""
        if target:
            requested.append(str(target))
        for feature in data_config.get("feature_columns", []) or []:
            feature_name = str(feature)
            if feature_name and feature_name not in requested:
                requested.append(feature_name)
        time_column = str(data_config.get("time_column") or "")
        if time_column and time_column not in requested:
            requested.append(time_column)
        return requested

    def _maybe_switch_data_config_to_latest_output(
        self,
        data_config: DataConfig,
        state: ETLState,
        current_columns: list[str],
    ) -> list[str]:
        latest_uri, _latest_reason = self._resolve_latest_output_training_data_location(state)
        if not latest_uri:
            return current_columns

        current_uri = data_config.get("dataset_uri", "")
        latest_columns = self._columns_for_training_data(latest_uri, state)
        if not latest_columns:
            return current_columns

        if current_uri and self._uris_match(current_uri, latest_uri):
            return latest_columns

        requested_columns = self._requested_training_columns(data_config)
        current_normalized = {
            self._normalize_column_token(column)
            for column in current_columns
        }
        latest_normalized = {
            self._normalize_column_token(column)
            for column in latest_columns
        }
        missing_from_current = [
            column
            for column in requested_columns
            if self._normalize_column_token(column) not in current_normalized
        ]
        requested_columns_in_latest = bool(requested_columns) and all(
            self._normalize_column_token(column) in latest_normalized
            for column in requested_columns
        )

        if self._latest_output_requested(state) or (
            missing_from_current and requested_columns_in_latest
        ):
            data_config["dataset_uri"] = latest_uri
            logger.info(
                "Using latest generated output for training data: requested columns=%s latest_uri=%s",
                requested_columns,
                latest_uri,
            )
            return latest_columns

        return current_columns

    @staticmethod
    def _default_feature_columns(
        columns: list[str],
        target: str,
        sample: Optional[pd.DataFrame] = None,
        excluded_columns: Optional[Sequence[str]] = None,
    ) -> list[str]:
        decisions = ModelTrainingAgent._identifier_decisions(columns, sample)
        excluded = set(excluded_columns or [])
        return [
            column
            for column in columns
            if column != target
            and column not in excluded
            and not column.lower().startswith("event_name")
            and decisions.get(column, {}).get("recommendation") != "exclude"
        ]

    def _infer_column_after_markers(self, columns: list[str], user_text: str, markers: tuple[str, ...]) -> str:
        lowered = (user_text or "").lower()
        if not lowered:
            return ""

        sorted_columns = sorted(columns, key=len, reverse=True)
        for marker in markers:
            for match in re.finditer(marker, lowered, flags=re.IGNORECASE):
                window = lowered[match.end():match.end() + 200]
                # The column nearest the marker is the intended one — "classifier
                # for isFraud. Use TransactionAmt" must pick isFraud, not whichever
                # feature has the longest name somewhere later in the window.
                best_column, best_index = "", len(window) + 1
                for column in sorted_columns:
                    hit = re.search(rf"(?<!\w){re.escape(column.lower())}(?!\w)", window)
                    if hit and hit.start() < best_index:
                        best_column, best_index = column, hit.start()
                if best_column:
                    return best_column
        return ""

    @staticmethod
    def _tool_use_failed_text(exc: Exception) -> str:
        """Extract the model's conversational reply from a Groq tool_use_failed error.

        Returns "" when the exception is anything else, so callers can fall
        through to their generic handler.
        """
        body = getattr(exc, "body", None)
        err = body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else body
        if isinstance(err, dict) and err.get("code") == "tool_use_failed":
            return str(err.get("failed_generation") or "").strip()
        return ""

    def _first_column_mentioned(self, columns: list[str], user_text: str) -> str:
        """Return the earliest column named verbatim (whole word) in the text."""
        lowered = (user_text or "").lower()
        if not lowered:
            return ""
        best_column, best_index = "", len(lowered) + 1
        # Longest names first so 'V339' wins over 'V3' at the same position.
        for column in sorted(columns, key=len, reverse=True):
            match = re.search(rf"(?<!\w){re.escape(column.lower())}(?!\w)", lowered)
            if match and match.start() < best_index:
                best_index, best_column = match.start(), column
        return best_column

    def _sanitize_training_data_config(self, data_config: DataConfig, state: ETLState) -> None:
        data_uri = data_config.get("dataset_uri", "")
        columns = self._columns_for_training_data(data_uri, state)
        columns = self._maybe_switch_data_config_to_latest_output(data_config, state, columns)
        if not columns:
            return

        lower_to_actual = {self._normalize_column_token(col): col for col in columns}
        latest_user_text = self._latest_user_text(state)

        target = data_config.get("target_column") or ""
        canonical_target = self._canonical_training_column(target, lower_to_actual)
        if not canonical_target:
            # Refuse instead of silently retargeting when the user explicitly
            # asked for a column-like name that does not exist in the dataset
            # ("predict house_price" -> error naming house_price, not a quiet
            # switch to some other column).
            missing_target = self._explicitly_requested_missing_target(
                target, columns, latest_user_text
            )
            if missing_target:
                raise ValueError(
                    f"Target column '{missing_target}' is not present in the "
                    "selected training dataset. Please choose one of the "
                    "existing columns as the target."
                )
            inferred_target = self._infer_target_column(columns, latest_user_text)
            canonical_target = inferred_target if inferred_target in columns else ""
        if canonical_target:
            data_config["target_column"] = canonical_target

        configured_time = str(data_config.get("time_column") or "").strip()
        canonical_time = self._canonical_training_column(configured_time, lower_to_actual)
        if configured_time and not canonical_time:
            raise ValueError(
                f"Time column {configured_time!r} is not present in the selected "
                "training dataset. Choose an existing timestamp/date column."
            )
        if not canonical_time:
            canonical_time = self._infer_time_column(columns, latest_user_text)
        if canonical_time:
            if canonical_time == data_config.get("target_column"):
                raise ValueError("The time column must be different from the target column.")
            data_config["time_column"] = canonical_time
        else:
            data_config.pop("time_column", None)

        sample = self._sample_for_identifier_detection(
            data_config.get("dataset_uri", data_uri), state
        )
        identifier_decisions = self._identifier_decisions(columns, sample)
        excluded_by_evidence = {
            column
            for column, decision in identifier_decisions.items()
            if decision.get("recommendation") == "exclude"
        }

        normalized_features: list[str] = []
        ignored_non_columns: list[str] = []
        ignored_identifier_features: list[str] = []
        existing_identifier_exclusions = [
            column for column in data_config.get("excluded_identifier_columns", []) or []
            if column in excluded_by_evidence
        ]
        missing_features: list[str] = []
        for feature in data_config.get("feature_columns", []) or []:
            actual_feature = self._canonical_training_column(feature, lower_to_actual)
            if actual_feature:
                if actual_feature in excluded_by_evidence:
                    ignored_identifier_features.append(actual_feature)
                elif actual_feature == data_config.get("time_column"):
                    logger.info(
                        "Using %s for chronological ordering rather than as a raw model feature.",
                        actual_feature,
                    )
                elif actual_feature != data_config.get("target_column") and actual_feature not in normalized_features:
                    normalized_features.append(actual_feature)
            elif self._feature_was_explicitly_requested(feature, state):
                missing_features.append(str(feature))
            else:
                ignored_non_columns.append(str(feature))

        if missing_features:
            raise ValueError(
                "Feature columns are not present in the selected training dataset: "
                f"{', '.join(missing_features)}. Please update the feature list before training."
            )

        if ignored_non_columns:
            logger.info(
                "Ignoring extracted training data-config terms that are not dataset columns: %s",
                ignored_non_columns,
            )

        identifier_exclusions = list(dict.fromkeys(
            existing_identifier_exclusions + ignored_identifier_features
        ))
        if identifier_exclusions:
            logger.info(
                "Excluding identifier-like columns from the training feature list: %s",
                identifier_exclusions,
            )
            data_config["excluded_identifier_columns"] = identifier_exclusions
        else:
            data_config.pop("excluded_identifier_columns", None)

        if not normalized_features and data_config.get("target_column") in columns:
            normalized_features = self._default_feature_columns(
                columns,
                data_config["target_column"],
                sample,
                [data_config.get("time_column")] if data_config.get("time_column") else [],
            )
        data_config["feature_columns"] = normalized_features

    def _validate_training_data_config(self, data_config: DataConfig, state: ETLState) -> None:
        self._sanitize_training_data_config(data_config, state)
        data_uri = data_config.get("dataset_uri", "")
        columns = self._columns_for_training_data(data_uri, state)
        columns = self._maybe_switch_data_config_to_latest_output(data_config, state, columns)
        if not columns:
            return

        available = set(columns)
        lower_to_actual = {self._normalize_column_token(col): col for col in columns}
        target = data_config.get("target_column") or ""
        time_column = data_config.get("time_column") or ""

        if target and target not in available and self._normalize_column_token(target) in lower_to_actual:
            data_config["target_column"] = lower_to_actual[self._normalize_column_token(target)]
            target = data_config["target_column"]

        if not target or target not in available:
            preview = ", ".join(columns[:30])
            more = "..." if len(columns) > 30 else ""
            raise ValueError(
                f"Target column {target!r} is not present in the selected training dataset. "
                f"Available columns include: {preview}{more}. "
                "Choose a labeled training dataset that contains the target column, or update the target column before training."
            )

        if time_column:
            canonical_time = (
                time_column
                if time_column in available
                else lower_to_actual.get(self._normalize_column_token(time_column), "")
            )
            if not canonical_time:
                raise ValueError(
                    f"Time column {time_column!r} is not present in the selected training dataset."
                )
            if canonical_time == target:
                raise ValueError("The time column must be different from the target column.")
            data_config["time_column"] = canonical_time

        normalized_features: list[str] = []
        missing_features: list[str] = []
        for feature in data_config.get("feature_columns", []) or []:
            if feature in available:
                normalized_features.append(feature)
            elif self._normalize_column_token(feature) in lower_to_actual:
                normalized_features.append(lower_to_actual[self._normalize_column_token(feature)])
            else:
                missing_features.append(str(feature))

        if missing_features:
            raise ValueError(
                "Feature columns are not present in the selected training dataset: "
                f"{', '.join(missing_features)}. Please update the feature list before training."
            )
        data_config["feature_columns"] = normalized_features

    def _is_local_data_source(self, data_uri: str, dataset_size_bytes: Optional[int] = None) -> bool:
        """Compatibility wrapper around the planner's backend decision."""
        backend, _, _ = self._training_backend_decision(data_uri, dataset_size_bytes)
        return backend == "local"

    def _training_backend_decision(
        self,
        data_uri: str,
        dataset_size_bytes: Optional[int] = None,
    ) -> tuple[str, str, Optional[int]]:
        """Choose an executor independently of the dataset storage provider.

        Small objects in GCS or MinIO/S3 can safely use the local trainer as
        long as the API pod has the matching credentials. Ray is reserved for
        large or unknown remote datasets so the API process is not forced to
        load an unbounded dataframe.
        """
        uri = (data_uri or "").strip()
        size = dataset_size_bytes if dataset_size_bytes is not None else self._get_file_size(uri)
        is_remote = self._is_remote_data_source(uri)

        if not is_remote:
            if size is not None and size >= DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING:
                return (
                    "local",
                    "The dataset is local-only; Ray workers cannot access the API pod filesystem, so training will run locally.",
                    size,
                )
            return "local", "The selected dataset is a local file, so training will run locally.", size
        if size is None:
            return (
                "ray",
                "The remote dataset size is unknown, so training will be scheduled on Ray to avoid exhausting API memory.",
                None,
            )
        if size >= DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING:
            return (
                "ray",
                f"The remote dataset is {size} bytes, at or above the Ray threshold of {DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING} bytes.",
                size,
            )
        return (
            "local",
            f"The remote dataset is {size} bytes, below the Ray threshold of {DATASET_SIZE_BYTES_THRESHOLD_FOR_RAY_TRAINING} bytes; the API will train it locally using the same storage URI.",
            size,
        )

    # ----------------------------------------------------------
    # Message normalisation
    # ----------------------------------------------------------

    def _normalize_messages(self, messages: list) -> list[BaseMessage]:
        """
        Convert a heterogeneous list of message objects or dicts into a
        uniform list of LangChain ``BaseMessage`` instances.

        Supported input formats:
          - Any ``BaseMessage`` subclass (passed through unchanged).
          - Dict with ``"type"`` / ``"role"`` key and ``"content"`` key.
            Content may be a string, a list of strings, or a list of dicts
            containing a ``"text"`` field.

        Unrecognised dicts are treated as ``HumanMessage``.
        Non-dict, non-BaseMessage items are coerced to ``HumanMessage``
        via ``str()``.

        Parameters
        ----------
        messages : list
            Raw messages from the ETL state.

        Returns
        -------
        list[BaseMessage]
        """
        normalized = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                normalized.append(msg)
            elif isinstance(msg, dict):
                msg_type = msg.get("type") or msg.get("role")
                content  = msg.get("content", "")

                # Unwrap list-style content (e.g. from Anthropic message format)
                if isinstance(content, list) and len(content) > 0:
                    if isinstance(content[0], dict) and "text" in content[0]:
                        content = content[0]["text"]
                    elif isinstance(content[0], str):
                        content = content[0]

                if msg_type in ("human", "user"):
                    normalized.append(HumanMessage(content=content))
                elif msg_type in ("ai", "assistant"):
                    normalized.append(AIMessage(content=content))
                else:
                    normalized.append(HumanMessage(content=content))
            else:
                normalized.append(HumanMessage(content=str(msg)))

        return normalized

    def _latest_human_text(self, state: ETLState) -> str:
        """Return the latest user text without injected analysis context."""
        for message in reversed(self._normalize_messages(state.get("messages", []))):
            if isinstance(message, HumanMessage):
                return str(message.content or "").split("\n\n[Analysis context]", 1)[0].strip().lower()
        return ""

    def _training_plan_update_clarification(self, state: ETLState) -> Optional[str]:
        """Return a deterministic clarification for menu-style update replies."""
        latest_text = self._latest_human_text(state)
        if not latest_text:
            return None

        option_text = latest_text.strip().rstrip(".")
        if option_text in {"1", "one", "option 1"}:
            return (
                "What model type or hyperparameters would you like to change? "
                "Please include the new values, for example epochs, batch size, optimizer, learning rate, or hidden layer sizes."
            )
        if option_text in {"2", "two", "option 2"}:
            return (
                "Which feature columns would you like to add or remove? "
                "Please name the columns explicitly."
            )
        if option_text in {"3", "three", "option 3"}:
            return (
                "What target column or dataset should I use instead? "
                "Please provide the new target column and, if changing datasets, the dataset name or URI."
            )
        if option_text in {"4", "four", "option 4", "something else"}:
            return "What specific change would you like me to make to the training plan?"

        # Guard against the assistant's own clarification prompt being replayed as user input.
        if (
            "to update the training plan" in latest_text
            and "what specific changes" in latest_text
            and "please provide more details" in latest_text
        ):
            return "Please tell me the specific training-plan change you want to make."

        return None

    def _apply_direct_training_plan_updates(self, state: ETLState) -> Optional[TrainingPlan]:
        training_plan = state.get("training_plan")
        if not isinstance(training_plan, dict):
            return None

        latest_text = self._latest_human_text(state)
        if not latest_text:
            return None

        updated_plan = copy.deepcopy(training_plan)
        changed = False

        hyperparameter_config = dict(updated_plan.get("hyperparameter_config") or {})
        numeric_updates = (
            ("epochs", (r"\bepochs?\b\s*(?:to|=|:)?\s*(\d+)", r"\b(\d+)\s*epochs?\b"), int),
            ("batch_size", (r"\bbatch\s*size\b\s*(?:to|=|:)?\s*(\d+)",), int),
            ("learning_rate", (r"\b(?:learning\s*rate|lr)\b\s*(?:to|=|:)?\s*([0-9]*\.?[0-9]+)",), float),
            ("early_stopping_patience", (r"\b(?:early\s*stopping\s*)?patience\b\s*(?:to|=|:)?\s*(\d+)",), int),
            ("dropout_rate", (r"\b(?:dropout|dropout\s*rate)\b\s*(?:to|=|:)?\s*([0-9]*\.?[0-9]+)",), float),
        )
        for key, patterns, caster in numeric_updates:
            for pattern in patterns:
                match = re.search(pattern, latest_text)
                if match:
                    hyperparameter_config[key] = caster(match.group(1))
                    changed = True
                    break

        optimizer_match = re.search(r"\b(?:optimizer|optimiser)\b\s*(?:to|=|:)?\s*(adam|sgd|rmsprop)\b", latest_text)
        if optimizer_match:
            hyperparameter_config["optimizer_name"] = optimizer_match.group(1)
            changed = True

        activation_match = re.search(r"\bactivation\b\s*(?:to|=|:)?\s*(relu|tanh|sigmoid)\b", latest_text)
        if activation_match:
            hyperparameter_config["activation"] = activation_match.group(1)
            changed = True

        batch_norm_match = re.search(r"\bbatch\s*norm(?:alization)?\b\s*(?:to|=|:)?\s*(true|false|on|off|enabled|disabled)\b", latest_text)
        if batch_norm_match:
            hyperparameter_config["batch_norm"] = batch_norm_match.group(1) in {"true", "on", "enabled"}
            changed = True

        hidden_match = re.search(r"\bhidden(?:\s*layer)?(?:\s*sizes?)?\b.*?\[([0-9,\s]+)\]", latest_text)
        if hidden_match:
            hidden_sizes = [int(value) for value in re.findall(r"\d+", hidden_match.group(1))]
            if hidden_sizes:
                hyperparameter_config["hidden_layer_sizes"] = hidden_sizes
                changed = True

        if changed:
            updated_plan["hyperparameter_config"] = hyperparameter_config

        data_config = dict(updated_plan.get("data_config") or {})
        feature_columns = list(data_config.get("feature_columns") or [])
        if feature_columns:
            mentioned_features = [
                column for column in feature_columns
                if re.search(rf"(?<!\w){re.escape(str(column).lower())}(?!\w)", latest_text)
            ]
            if mentioned_features and re.search(r"\b(remove|drop|exclude|without)\b", latest_text):
                data_config["feature_columns"] = [
                    column for column in feature_columns if column not in mentioned_features
                ]
                updated_plan["data_config"] = data_config
                changed = True

        if not changed:
            return None

        logger.info("Applied direct training plan update from user text: %s", latest_text)
        return TrainingPlan(**updated_plan)

    def _latest_user_requests_full_training_plan(self, state: ETLState) -> bool:
        latest_text = self._latest_human_text(state)
        if not latest_text:
            return False
        return bool(
            re.search(
                r"\b(?:full|complete|detailed|all\s+details|entire)\b.{0,40}\btraining\s+plan\b"
                r"|\btraining\s+plan\b.{0,40}\b(?:full|complete|detailed|all\s+details|entire)\b",
                latest_text,
            )
        )

    def _format_training_plan_response(
        self,
        training_plan: TrainingPlan,
        updated: bool = False,
        full: bool = False,
    ) -> str:
        hyperparameter_config = training_plan.get("hyperparameter_config") or {}
        ray_config = training_plan.get("ray_config")
        data_config = training_plan.get("data_config") or {}

        def _display(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            display_values = {
                "classification": "Classification",
                "regression": "Regression",
                "adam": "Adam",
                "sgd": "SGD",
                "rmsprop": "RMSprop",
                "relu": "ReLU",
                "tanh": "Tanh",
                "sigmoid": "Sigmoid",
            }
            return display_values.get(value.lower(), value)

        def _none_dash(value: Any) -> Any:
            return "-" if value is None else value

        model_type = _display(training_plan.get("model_type"))
        is_regression = isinstance(model_type, str) and "regression" in model_type.lower()
        metric_description = (
            "Training tracks squared loss, average error, large-error score, and explained variation."
            if is_regression
            else "Training logs loss, validation loss, accuracy, and weighted F1 score."
        )
        model_name = training_plan.get("model_name") or "model"
        model_version = training_plan.get("model_version") or "v1.0"
        target_column = data_config.get("target_column")
        feature_columns = data_config.get("feature_columns") or []
        feature_count = len(feature_columns)
        excluded_identifier_columns = data_config.get("excluded_identifier_columns") or []
        time_column = data_config.get("time_column")
        epochs = hyperparameter_config.get("epochs")
        optimizer = _display(hyperparameter_config.get("optimizer_name"))
        learning_rate = hyperparameter_config.get("learning_rate")
        batch_size = hyperparameter_config.get("batch_size")
        if not full:
            lines = [
                "## Training Plan Ready" if not updated else "## Training Plan Updated",
                "",
                f"**Model:** {model_type} (`{model_name}`, {model_version})",
                f"**Target:** `{target_column}`",
                f"**Features:** {feature_count} selected column{'s' if feature_count != 1 else ''}",
                (
                    f"**Training:** {epochs} epochs, {optimizer} optimizer, "
                    f"learning rate `{learning_rate}`, batch size `{batch_size}`"
                ),
                "",
                "Training has not started.",
                "",
                "**Start training with this plan, or describe any changes you want to make.**",
            ]
            if excluded_identifier_columns:
                lines.insert(
                    5,
                    "**Excluded identifier columns:** "
                    + ", ".join(f"`{column}`" for column in excluded_identifier_columns),
                )
            if time_column:
                lines.insert(
                    5,
                    f"**Validation:** `TimeSeriesSplit` ordered by `{time_column}`; "
                    "training rows precede validation rows",
                )
            return "\n".join(lines)

        lines = [
            f"## {'Full Updated Training Plan' if updated else 'Full Training Plan'}",
            "",
            "The current training plan is as follows:",
            "",
            "**Model Configuration**",
            "",
            f"- **Model Type:** {model_type}",
            "- **Trainer / Architecture:** DynamicMLP (PyTorch)",
            f"- **Model Name:** `{model_name}`",
            f"- **Model Description:** {_none_dash(training_plan.get('model_description'))}",
            f"- **Model Version:** {model_version}",
            "",
            "**Hyperparameter Configuration**",
            "",
            f"- **Learning Rate:** {_none_dash(learning_rate)}",
            f"- **Epochs:** {_none_dash(epochs)}",
            f"- **Batch Size:** {_none_dash(batch_size)}",
            f"- **Optimizer Name:** {_none_dash(optimizer)}",
            f"- **Random Seed:** {_none_dash(hyperparameter_config.get('random_seed'))}",
            f"- **Early Stopping Patience:** {_none_dash(hyperparameter_config.get('early_stopping_patience'))}",
            f"- **Hidden Layer Sizes:** {_none_dash(hyperparameter_config.get('hidden_layer_sizes'))}",
            f"- **Activation:** {_none_dash(_display(hyperparameter_config.get('activation')))}",
            f"- **Dropout Rate:** {_none_dash(hyperparameter_config.get('dropout_rate'))}",
            f"- **Batch Normalization:** {_none_dash(hyperparameter_config.get('batch_norm'))}",
            "",
            "**Data Configuration**",
            "",
            f"- **Dataset URI:** `{_none_dash(data_config.get('dataset_uri'))}`",
            f"- **Feature Columns:** {feature_columns}",
            f"- **Excluded Identifier Columns:** {excluded_identifier_columns or '-'}",
            f"- **Target Column:** `{_none_dash(target_column)}`",
            f"- **Time Column:** `{_none_dash(time_column)}`",
            "",
            "**Training Procedure**",
            "",
            "- **Feature Preprocessing:** MTA converts selected features into numeric tensors. "
            "The local trainer ordinal-encodes categorical columns; the Ray trainer validates numeric features "
            "and requires categorical columns to be encoded before distributed training. Both trainers fit "
            "standard scaling on the training split for numeric features; regression targets are also scaled "
            "during optimization and converted back to their original units for metrics and inference.",
            (
                f"- **Validation:** TimeSeriesSplit ordered by `{time_column}`; the final "
                "expanding-window fold trains on earlier rows and validates on later rows."
                if time_column
                else "- **Validation:** Deterministic 80/20 row-hash holdout for validation and early stopping."
            ),
            f"- **Metrics:** {metric_description}",
            "- **Artifacts:** MLflow logs the PyTorch model, `.pth` weights, `.onnx` export, model config, metrics, and metadata.",
        ]
        if ray_config:
            lines.extend(
                [
                    "",
                    "**Ray Config**",
                    "",
                    f"- **Num Workers:** `{_none_dash(ray_config.get('num_workers'))}`",
                    f"- **Use GPU:** `{_none_dash(ray_config.get('use_gpu'))}`",
                    f"- **Memory per Worker:** `{_none_dash(ray_config.get('memory_per_worker'))}`",
                    f"- **CPU per Worker:** `{_none_dash(ray_config.get('cpu_per_worker'))}`",
                ]
            )
        lines.extend(
            [
                "",
                "Training has not started.",
                "",
                "**Start training with this plan, or describe any changes you want to make.**",
            ]
        )
        return "\n".join(lines)

    # ----------------------------------------------------------
    # Tool-call parsing
    # ----------------------------------------------------------

    def parse_tool_call(self, tool_call_response) -> str:
        """
        Extract the tool function name from a raw LLM tool-call response.

        Accepts a string (returned as-is), a dict with a ``"function"``
        sub-dict, or a flat dict with a ``"name"`` key.

        Parameters
        ----------
        tool_call_response : str | dict

        Returns
        -------
        str
            The tool function name.

        Raises
        ------
        TypeError
            If *tool_call_response* is neither a string nor a dict.
        """
        if isinstance(tool_call_response, str):
            return tool_call_response

        if isinstance(tool_call_response, dict):
            if "function" in tool_call_response:
                return tool_call_response["function"].get("name")
            return tool_call_response.get("name")

        raise TypeError(
            f"Unsupported tool call response type: {type(tool_call_response)}"
        )

    # ----------------------------------------------------------
    # Config extraction helpers
    # ----------------------------------------------------------

    def _latest_user_text(self, state: ETLState) -> str:
        for message in reversed(state.get("messages", []) or []):
            if isinstance(message, HumanMessage):
                return str(message.content or "")
            if isinstance(message, dict) and (
                message.get("type") in {"human", "user"}
                or message.get("role") == "user"
            ):
                return str(message.get("content") or "")
        return ""

    def _recent_user_texts(self, state: ETLState, limit: int = 5) -> str:
        texts: list[str] = []
        for message in reversed(state.get("messages", []) or []):
            if isinstance(message, HumanMessage):
                texts.append(str(message.content or ""))
            elif isinstance(message, dict) and (
                message.get("type") in {"human", "user"}
                or message.get("role") == "user"
            ):
                texts.append(str(message.get("content") or ""))
            if len(texts) >= limit:
                break
        return "\n".join(texts)

    def _reconcile_model_type_with_target(
        self, model_type: str, state: ETLState, data_config: DataConfig,
        has_previous_plan: bool = False,
    ) -> str:
        """Cross-check the text-derived model_type against explicit user wording and
        the target column's dtype. Without this, model_type defaults to
        "classification", so a continuous target silently trains with
        CrossEntropyLoss and reports accuracy/f1 instead of RMSE/MAE.

        Only the CURRENT user message counts as explicit wording — scanning
        history let a stale "classify into buckets" ask flip a later re-train.
        The dtype heuristic only applies when creating an initial plan; once a
        plan exists its type is kept unless the user says otherwise.
        """
        text = (self._latest_user_text(state) or "").lower()
        if re.search(r"\bregress(?:ion|or)?\b", text):
            return "regression"
        if re.search(r"\bclassif(?:y|ier|ication)\b", text):
            return "classification"
        if has_previous_plan:
            return model_type
        schema = state.get("schema") or {}
        target = data_config.get("target_column") or ""
        dtype = str(schema.get(target, "")).lower() if isinstance(schema, dict) else ""
        if model_type == "classification" and dtype.startswith("float"):
            logger.info(
                "Overriding model_type classification -> regression: target %r has "
                "continuous dtype %s and the user did not ask for classification.",
                target, dtype,
            )
            return "regression"
        return model_type

    def _extract_inference_json(self, content: str) -> Any:
        """Return the first complete JSON object or array embedded in text."""
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", content or ""):
            try:
                value, _ = decoder.raw_decode(content[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(value, (dict, list)):
                return value
        raise ValueError("No JSON object or array was found in the inference request.")

    def _feature_aliases_for_inference(self, feature: str) -> list[str]:
        spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(feature)).lower()
        spaced = spaced.replace("_", " ").replace("-", " ")
        aliases = {
            str(feature).lower(),
            spaced,
            spaced.replace(" ", ""),
        }
        special_aliases = {
            "Pclass": ["pclass", "p class", "passenger class"],
            "SibSp": ["sibsp", "sib sp", "siblings spouses"],
            "Parch": ["parch", "parents children"],
            "FamilySize": ["family size", "familysize"],
            "IsAlone": ["is alone", "isalone"],
            "CabinKnown": ["cabin known", "cabinknown"],
        }
        aliases.update(alias.lower() for alias in special_aliases.get(str(feature), []))
        return sorted({alias for alias in aliases if alias}, key=len, reverse=True)

    @staticmethod
    def _parse_inference_scalar(raw_value: str, quoted: bool = False) -> Any:
        value = str(raw_value or "").strip().strip(",.;")
        if quoted:
            return value
        lowered = value.lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
        if re.fullmatch(r"[-+]?\d+", value):
            try:
                return int(value)
            except ValueError:
                return value
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", value):
            try:
                return float(value)
            except ValueError:
                return value
        return value

    @staticmethod
    def _coerce_numeric_string(value: Any) -> Any:
        if not isinstance(value, str):
            return value

        stripped = value.strip()
        if re.fullmatch(r"[-+]?\d+", stripped):
            try:
                return int(stripped)
            except ValueError:
                return value
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", stripped):
            try:
                return float(stripped)
            except ValueError:
                return value
        return value

    def _normalize_inference_payload_values(self, input_data: Any) -> Any:
        """Accept encoded numeric values even when JSON sends them as strings."""
        if isinstance(input_data, dict):
            if "dataset_uri" in input_data:
                return input_data
            return {
                key: self._normalize_inference_payload_values(value)
                for key, value in input_data.items()
            }
        if isinstance(input_data, list):
            return [self._normalize_inference_payload_values(row) for row in input_data]
        return self._coerce_numeric_string(input_data)

    def _latest_user_requests_detailed_inference_report(self, state: ETLState) -> bool:
        latest_user_text = self._latest_user_text(state).split("\n\n[Analysis context]", 1)[0].lower()
        return bool(
            re.search(
                r"\b(?:full|detailed?|details?|larger|expanded?|explain|breakdown|"
                r"feature\s+table|raw\s+logits?|logits?|class\s+probabilities|probabilities)\b",
                latest_user_text,
            )
        )

    @staticmethod
    def _format_inference_confidence(value: Any) -> Optional[str]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None

        if 0 <= numeric <= 1:
            numeric *= 100
        return f"{numeric:.3f}".rstrip("0").rstrip(".") + "%"

    @staticmethod
    def _model_detail_value(model_detail: Any, key: str, default: Any = None) -> Any:
        if isinstance(model_detail, dict):
            return model_detail.get(key, default)
        return getattr(model_detail, key, default)

    @staticmethod
    def _humanize_column_name(column_name: Any) -> str:
        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(column_name or ""))
        text = re.sub(r"[_-]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _binary_prediction_value(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return int(value)

        text = str(value).strip().lower()
        if text in {"true", "yes"}:
            return 1
        if text in {"false", "no"}:
            return 0

        try:
            numeric = float(text)
        except ValueError:
            return None
        if numeric in {0.0, 1.0}:
            return int(numeric)
        return None

    def _binary_target_meaning(self, target_column: Any, value: int) -> Optional[str]:
        target_text = self._humanize_column_name(target_column)
        if not target_text:
            return None

        lower_target = target_text.lower()
        if "surviv" in lower_target:
            return "survived" if value == 1 else "did not survive"
        if lower_target.startswith("is "):
            subject = target_text[3:].strip()
            return f"is {subject}" if value == 1 else f"is not {subject}"
        if lower_target.startswith("has "):
            subject = target_text[4:].strip()
            return f"has {subject}" if value == 1 else f"does not have {subject}"
        if lower_target.startswith("had "):
            subject = target_text[4:].strip()
            return f"had {subject}" if value == 1 else f"did not have {subject}"
        if lower_target.startswith("snap "):
            return "yes" if value == 1 else "no"
        return None

    def _inference_prediction_value(self, inference_result: dict[str, Any], model_detail: Any = None) -> Any:
        if "prediction" in inference_result:
            return inference_result.get("prediction")

        predicted_index = inference_result.get("predicted_class_index")
        class_names = (
            inference_result.get("class_names")
            or self._model_detail_value(model_detail, "class_names", [])
            or []
        )
        if predicted_index is not None and class_names:
            try:
                return class_names[int(predicted_index)]
            except (IndexError, TypeError, ValueError):
                pass
        return predicted_index

    def _format_inference_prediction(self, inference_result: dict[str, Any], model_detail: Any = None) -> Any:
        prediction = self._inference_prediction_value(inference_result, model_detail)
        target_column = self._model_detail_value(model_detail, "target_column", "")
        if not target_column:
            return prediction

        display = f"{target_column} = {prediction}"
        binary_value = self._binary_prediction_value(prediction)
        meaning = (
            self._binary_target_meaning(target_column, binary_value)
            if binary_value is not None
            else None
        )
        if meaning:
            return f"{display} ({meaning})"
        return display

    def _format_compact_inference_summary(self, inference_result: Any, model_detail: Any = None) -> str:
        """Return the default short chat response for an inference result."""
        if isinstance(inference_result, dict) and inference_result.get("output_uri"):
            return f"Inference completed. Predictions were saved to `{inference_result['output_uri']}`."

        if isinstance(inference_result, list):
            if not inference_result:
                return "Inference completed, but no prediction rows were returned."
            row_summaries = [
                f"- Row {index}: {self._format_compact_inference_summary(result, model_detail)}"
                for index, result in enumerate(inference_result, start=1)
            ]
            return "\n".join(
                [f"Inference completed for {len(inference_result)} rows:", *row_summaries]
            )

        if not isinstance(inference_result, dict):
            return f"Inference result: {inference_result}"

        prediction = self._inference_prediction_value(inference_result, model_detail)
        display_prediction = self._format_inference_prediction(inference_result, model_detail)
        probabilities = inference_result.get("probabilities") or {}

        confidence = None
        if isinstance(probabilities, dict) and probabilities:
            prediction_key = str(prediction)
            probability_value = (
                probabilities.get(prediction)
                if prediction in probabilities
                else probabilities.get(prediction_key)
            )
            if probability_value is None:
                try:
                    _, probability_value = max(
                        probabilities.items(),
                        key=lambda item: float(item[1]),
                    )
                except (TypeError, ValueError):
                    probability_value = None
            confidence = self._format_inference_confidence(probability_value)

        if prediction is None:
            return f"Inference completed. Result: {inference_result}"
        if confidence:
            return f"Prediction: {display_prediction}. Confidence: {confidence}."
        return f"Prediction: {display_prediction}."

    def _plain_inference_result(self, inference_result: Any) -> Any:
        """Convert InferenceInterface DTOs into dict/list values for chat formatting."""
        if isinstance(inference_result, list):
            return [self._plain_inference_result(item) for item in inference_result]
        to_dict = getattr(inference_result, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        return inference_result

    @staticmethod
    def _inference_sample_input_prompt(feature_names: list[str]) -> str:
        feature_text = ", ".join(feature_names) if feature_names else "the model feature values"
        return (
            "I can run a sample prediction in chat. Please send one row of values "
            f"for these model features: {feature_text}."
        )

    @staticmethod
    def _is_planner_inference_format_response(message: BaseMessage) -> bool:
        if not isinstance(message, AIMessage):
            return False

        text = str(message.content or "").lower()
        if "inference" not in text:
            return False
        if not re.search(r"\b(payload|format|expects?|expected|required|provide|resend)\b", text):
            return False
        return bool(re.search(r"\b(feature|field|value|model|request|input)\b", text))

    def _drop_planner_inference_format_response(self, state: ETLState) -> ETLState:
        """
        Remove a planner-only inference format warning before MTA summarizes
        the actual inference result.
        """
        messages = self._normalize_messages(state.get("messages", []) or [])
        latest_human_index = None
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], HumanMessage):
                latest_human_index = index
                break
        if latest_human_index is None:
            return state

        before_latest_reply = messages[: latest_human_index + 1]
        after_latest_reply = messages[latest_human_index + 1 :]
        kept_after_reply = [
            message
            for message in after_latest_reply
            if not self._is_planner_inference_format_response(message)
        ]
        if len(kept_after_reply) == len(after_latest_reply):
            return state

        new_state = state.copy()
        new_state["messages"] = before_latest_reply + kept_after_reply
        return new_state

    def _extract_named_inference_values(
        self,
        content: str,
        feature_names: list[str],
    ) -> Optional[dict[str, Any] | list[dict[str, Any]]]:
        """Parse one or more named-value inference rows without an LLM/tool call."""
        text = str(content or "").split("\n\n[Analysis context]", 1)[0]
        if not text or not feature_names:
            return None

        # A user can provide a batch by repeating the natural-language request,
        # for example: ``Run inference with: f1=1 Run inference with: f1=2``.
        # Parse each block independently so values from later rows do not get
        # discarded by the single-row ``re.search`` calls below.
        request_pattern = re.compile(
            r"\b(?:run|execute|perform|do)\s+(?:an?\s+)?inference\s+"
            r"(?:with|using)(?:\s+these\s+values)?\s*:?",
            flags=re.IGNORECASE,
        )
        request_markers = list(request_pattern.finditer(text))
        if len(request_markers) > 1:
            rows: list[dict[str, Any]] = []
            for index, marker in enumerate(request_markers):
                end = (
                    request_markers[index + 1].start()
                    if index + 1 < len(request_markers)
                    else len(text)
                )
                row = self._extract_named_inference_values(text[marker.start():end], feature_names)
                if isinstance(row, dict) and row:
                    rows.append(row)
            return rows or None

        parsed: dict[str, Any] = {}
        for feature in feature_names:
            for alias in self._feature_aliases_for_inference(feature):
                alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
                pattern = (
                    rf'(?<!\w)"?{alias_pattern}"?(?!\w)\s*'
                    r'(?:=|:|is|as|at|to)?\s*'
                    r'(?:"([^"]+)"|\'([^\']+)\'|([A-Za-z0-9_.+-]+))'
                )
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if not match:
                    continue
                quoted_value = match.group(1) if match.group(1) is not None else match.group(2)
                raw_value = quoted_value if quoted_value is not None else match.group(3)
                if raw_value is None:
                    continue
                parsed[feature] = self._parse_inference_scalar(raw_value, quoted=quoted_value is not None)
                break

        feature_lookup = {str(feature).lower(): feature for feature in feature_names}
        sex_feature = feature_lookup.get("sex")
        if sex_feature and sex_feature not in parsed:
            if re.search(r"\bfemale\b", text, flags=re.IGNORECASE):
                parsed[sex_feature] = "female"
            elif re.search(r"\bmale\b", text, flags=re.IGNORECASE):
                parsed[sex_feature] = "male"

        is_alone_feature = feature_lookup.get("isalone")
        if is_alone_feature and is_alone_feature not in parsed:
            if re.search(r"\bnot\s+alone\b", text, flags=re.IGNORECASE):
                parsed[is_alone_feature] = 0
            elif re.search(r"\balone\b", text, flags=re.IGNORECASE):
                parsed[is_alone_feature] = 1

        cabin_known_feature = feature_lookup.get("cabinknown")
        if cabin_known_feature and cabin_known_feature not in parsed:
            if re.search(r"\bcabin\s+(?:is\s+)?(?:not\s+known|unknown|not\s+available)\b", text, flags=re.IGNORECASE):
                parsed[cabin_known_feature] = 0
            elif re.search(r"\bcabin\s+(?:is\s+)?(?:known|available|present)\b", text, flags=re.IGNORECASE):
                parsed[cabin_known_feature] = 1

        return parsed or None

    def _inference_input_error(self, input_data: Any, _feature_names: list[str]) -> Optional[str]:
        """Validate inference payload shape without requiring every model feature."""
        if isinstance(input_data, dict) and "dataset_uri" in input_data:
            return None

        if isinstance(input_data, dict):
            rows = [input_data]
        elif isinstance(input_data, list) and input_data:
            rows = input_data
        else:
            return "Inference input must be a JSON object or a non-empty JSON array of objects."

        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                return f"Inference row {index + 1} must be a JSON object."
        return None

    def _latest_user_has_inference_payload(self, state: ETLState) -> bool:
        latest_user_text = self._latest_user_text(state).split("\n\n[Analysis context]", 1)[0]
        try:
            self._extract_inference_json(latest_user_text)
            return True
        except ValueError:
            return False

    def _latest_user_requests_inference(self, state: ETLState) -> bool:
        latest_user_text = self._latest_user_text(state).split("\n\n[Analysis context]", 1)[0].lower()
        if re.search(r"\b(?:configure|deploy|start|stop)\b.{0,40}\binference\s+service\b", latest_user_text):
            return False
        direct_inference = re.search(
            r"\b(?:run|execute|perform|do)\s+(?:an?\s+)?inference\b",
            latest_user_text,
        )
        predictive_inference = re.search(r"\b(infer|inference|predict|prediction|classify)\b", latest_user_text)
        if not (direct_inference or predictive_inference):
            return False
        if self._latest_user_has_inference_payload(state):
            return True
        detailed_inference = re.search(
            r"\b(?:full|detailed?|details?|expanded?|report)\b.{0,40}\binference\b|\binference\b.{0,40}\b(?:full|detailed?|details?|expanded?|report)\b",
            latest_user_text,
        )
        if detailed_inference:
            return True
        return bool(
            direct_inference
            and re.search(r"\b(?:with|using|values?|row|input)\b", latest_user_text)
        )

    def _latest_user_requests_inference_service(self, state: ETLState) -> Optional[str]:
        """Return ``start`` or ``stop`` for an explicit inference-service request."""
        text = self._latest_user_text(state).split("\n\n[Analysis context]", 1)[0].strip().lower()
        if not text:
            return None
        if re.search(
            r"\b(?:stop|disable|remove|delete|shut\s*down)\b.{0,60}"
            r"\b(?:inference\s+)?(?:service|gateway|api|endpoint)\b",
            text,
        ):
            return "stop"
        if re.search(
            r"\b(?:configure|deploy|start|create|enable|set\s+up|setup|expose)\b.{0,60}"
            r"\b(?:inference\s+)?(?:service|gateway|api|endpoint)\b",
            text,
        ):
            return "start"
        return None

    def _parse_json_or_default(self, content: str, default: dict, label: str) -> dict:
        try:
            parsed = extract_json_from_content(content)
            return parsed if isinstance(parsed, dict) else default
        except Exception as exc:
            logger.warning(
                "MTA %s extraction returned non-JSON; using deterministic fallback: %s",
                label,
                exc,
            )
            return default

    def _training_plans_equivalent(self, before: Any, after: Any) -> bool:
        """Return True when an attempted plan update made no structural change."""
        return before == after

    def _training_plan_update_has_enough_detail(self, state: ETLState) -> bool:
        """Use the LLM to decide whether a plan-update request is actionable."""
        latest_text = self._latest_human_text(state)
        if not latest_text or not llm:
            return False

        system_prompt = (
            "Decide whether the latest user reply gives enough concrete detail to update "
            "an existing machine-learning training plan. Return JSON only with this schema:\n"
            '{"has_specific_update": boolean, "reason": string}\n\n'
            "Use true only when the reply identifies at least one specific plan change, "
            "such as a different hyperparameter value, model setting, feature/target choice, "
            "dataset change, or another clear directional change. Use false when the user only "
            "indicates that they want changes but does not specify what should change."
        )
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=(
                            f"Current training plan:\n{state.get('training_plan')}\n\n"
                            f"Latest user reply:\n{latest_text}"
                        )
                    ),
                ],
                tools=[],
                tool_choice="none",
            )
            parsed = extract_json_from_content(response.content)
            return bool(parsed.get("has_specific_update"))
        except Exception as exc:
            logger.warning("Could not classify training-plan update specificity: %s", exc)
            return False

    def _infer_target_column(self, columns: list[str], user_text: str) -> str:
        marked_target = self._infer_column_after_markers(
            columns,
            user_text,
            (
                r"\b(?:predict|predicting|classify|classifying|classifier|classification|"
                r"forecast|forecasting|regress|regression|regressor)\b",
                r"\b(?:target|target column|label|label column|outcome|response)\b\s*(?:is|as|:|=|for|of)?",
            ),
        )
        if marked_target:
            return marked_target
        # A column the user names anywhere in the request beats every heuristic below
        # ("train a fraud classifier for isFraud" names isFraud without a marker word).
        named_target = self._first_column_mentioned(columns, user_text)
        if named_target:
            return named_target
        for marker in ("target", "label", "class", "type", "status", "outcome"):
            match = next((column for column in columns if marker in column.lower()), None)
            if match:
                return match
        return columns[-1] if columns else ""

    def _infer_time_column(self, columns: list[str], user_text: str) -> str:
        """Infer row-order metadata only when the request is explicitly temporal."""
        text = user_text or ""
        explicit = self._infer_column_after_markers(
            columns,
            text,
            (
                r"\b(?:ordered|sorted)\s+by\b",
                r"\b(?:time|date|timestamp|datetime)\s+column\b",
                r"\bchronological(?:ly)?\s+by\b",
            ),
        )
        if explicit:
            return explicit
        if not re.search(
            r"\b(?:time[ -]?(?:series|ordered)|forecast(?:ing)?|chronological|temporal|"
            r"future\s+(?:sales|demand|value|values|period|periods)|"
            r"next\s+(?:day|week|month|year))\b",
            text,
            re.IGNORECASE,
        ):
            return ""

        def _score(column: str) -> tuple[int, int]:
            token = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
            if re.search(r"birth|dob", token):
                return (-1, 0)
            if token in {"date", "datetime", "timestamp", "time", "ds"}:
                return (5, -len(column))
            if re.search(
                r"(?:^|_)(?:event|order|sale|transaction|pickup|dropoff)?_?"
                r"(?:date|datetime|timestamp|time)$",
                token,
            ):
                return (4, -len(column))
            if re.search(r"(?:^|_)(?:year|month|week|day)$", token):
                return (2, -len(column))
            return (0, -len(column))

        ranked = sorted(columns, key=_score, reverse=True)
        return ranked[0] if ranked and _score(ranked[0])[0] > 0 else ""

    def _fallback_data_config(self, data_location: str, columns: list[str], state: ETLState) -> DataConfig:
        target_column = self._infer_target_column(columns, self._latest_user_text(state))
        time_column = self._infer_time_column(columns, self._latest_user_text(state))
        sample = self._sample_for_identifier_detection(data_location, state)
        decisions = self._identifier_decisions(columns, sample)
        feature_columns = self._default_feature_columns(
            columns,
            target_column,
            sample,
            [time_column] if time_column else [],
        )
        excluded_identifiers = [
            column for column in columns
            if column != target_column
            and decisions.get(column, {}).get("recommendation") == "exclude"
        ]
        config = DataConfig(
            dataset_uri=data_location,
            feature_columns=feature_columns,
            target_column=target_column,
        )
        if excluded_identifiers:
            config["excluded_identifier_columns"] = excluded_identifiers
        if time_column:
            config["time_column"] = time_column
        return config

    def _extract_hyperparameter_config(self, state: ETLState) -> HyperparameterConfig:
        """
        Use the LLM to extract or update hyperparameter settings from the
        current conversation history.

        Falls back to the previously stored plan's config, or sensible
        defaults if no plan exists yet.

        Parameters
        ----------
        state : ETLState

        Returns
        -------
        HyperparameterConfig
        """
        default_config = HyperparameterConfig(
            learning_rate=0.001,
            epochs=30,
            batch_size=32,
            optimizer_name="adam",
            random_seed=42,
            early_stopping_patience=5,
            hidden_layer_sizes=[128, 64],
            activation="relu",
            dropout_rate=0.2,
            batch_norm=True,
        )
        training_plan = state.get("training_plan") or {}
        config = training_plan.get("hyperparameter_config", default_config)
        system_prompt = (
            "Please extract hyperparameter configuration for model training as following JSON schema based on user request and chat history.\n"
            f"Default hyperparameter configuration is: {config}\n"
            "JSON schema:\n"
            "```json\n"
            f"{HyperparameterConfig.__annotations__}"
            "```\n"
            "Output only extracted hyperparameter configuration as defined JSON schema."
        )
        messages      = [SystemMessage(content=system_prompt)] + self._normalize_messages(state["messages"])[-10:]
        full_response = llm.invoke(messages, tools=[], tool_choice="none")
        config_data   = self._parse_json_or_default(
            full_response.content,
            dict(config) if isinstance(config, dict) else dict(default_config),
            "hyperparameter",
        )
        return HyperparameterConfig(**config_data)

    def _extract_ray_config(self, state: ETLState) -> RayConfig:
        """
        Return the Ray cluster configuration from the current training plan,
        or sensible defaults if no plan exists yet.

        When the data source is local (no cloud URI), returns a minimal
        RayConfig placeholder — it will not be used for actual Ray submission
        but is kept in the TrainingPlan for schema consistency.

        Parameters
        ----------
        state : ETLState

        Returns
        -------
        RayConfig
        """
        default_config = RayConfig(
            num_workers=4,
            use_gpu=False,
            memory_per_worker="16Gi",
            cpu_per_worker=4,
        )
        training_plan = state.get("training_plan") or {}
        return training_plan.get("ray_config") or default_config

    def _extract_initial_data_config(self, state: ETLState) -> DataConfig:
        """
        Use the LLM to pick feature columns and a target column from the
        dataset at ``state["data_source_location_cloud"]``.

        Calls ``get_columns()`` to retrieve available column names without
        loading the full dataset.

        Parameters
        ----------
        state : ETLState

        Returns
        -------
        DataConfig
        """
        data_location = self._resolve_training_data_location(state)
        if not data_location:
            unavailable_reason = self._preferred_training_data_unavailable_reason(state)
            if unavailable_reason:
                raise ValueError(unavailable_reason)
            raise ValueError(
                "No training dataset is selected. Please upload or select a dataset before creating a training plan."
            )
        columns = self._columns_for_training_data(data_location, state)
        system_prompt = (
            "Please extract data configuration for model training as following JSON schema based on user request and chat history.\n"
            f"Available columns in the dataset: {', '.join(columns)}.\n"
            f"Data source location: {data_location}.\n"
            "Please choose feature columns and target column for model training based on user request or chat history.\n"
            "feature_columns and target_column must contain exact dataset column names from the available columns list only.\n"
            "For forecasting or explicitly time-ordered training, set time_column to the exact date, datetime, timestamp, or sequence column. Otherwise omit it. Do not include time_column in feature_columns.\n"
            "Exclude row, transaction, and person/account identifiers, but retain repeated reusable entity keys such as product_id or store_id when sample values show they are not row-unique.\n"
            "Do not include model names, task descriptions, reporting requirements, evaluation metrics, output artifacts, or submission columns unless they are real dataset columns used for training.\n"
            "JSON schema:\n"
            "```json\n"
            f"{DataConfig.__annotations__}"
            "```\n"
            "Output only extracted data configuration as defined JSON schema."
        )
        messages      = [SystemMessage(content=system_prompt)] + self._normalize_messages(state["messages"])[-10:]
        full_response = llm.invoke(messages, tools=[], tool_choice="none")
        config_data   = self._parse_json_or_default(
            full_response.content,
            self._fallback_data_config(data_location, columns, state),
            "initial data config",
        )
        config_data["dataset_uri"] = data_location
        data_config = DataConfig(**config_data)
        self._validate_training_data_config(data_config, state)
        return data_config

    def _extract_data_config(self, state: ETLState) -> DataConfig:
        """
        Use the LLM to update the data config based on the latest user
        feedback, starting from the previously stored config.

        Parameters
        ----------
        state : ETLState

        Returns
        -------
        DataConfig
        """
        training_plan = state.get("training_plan") or {}
        if "data_config" not in training_plan:
            return self._extract_initial_data_config(state)

        config = training_plan["data_config"]
        data_location = self._resolve_training_data_location(state)
        if not data_location:
            unavailable_reason = self._preferred_training_data_unavailable_reason(state)
            if unavailable_reason:
                raise ValueError(unavailable_reason)
            data_location = config.get("dataset_uri", "")
        columns = (
            self._columns_for_training_data(data_location, state)
            if data_location
            else self._schema_columns_from_state(state)
        )
        system_prompt = (
            "Please extract data configuration for model training as following JSON schema based on user request and chat history.\n"
            f"Default data configuration is: {config}\n"
            f"Available columns in the dataset: {', '.join(columns)}.\n"
            "feature_columns and target_column must contain exact dataset column names from the available columns list only.\n"
            "For forecasting or explicitly time-ordered training, set time_column to the exact ordering column. Otherwise omit it. Do not include time_column in feature_columns.\n"
            "Exclude row, transaction, and person/account identifiers, but retain repeated reusable entity keys such as product_id or store_id when sample values show they are not row-unique.\n"
            "Do not include model names, task descriptions, reporting requirements, evaluation metrics, output artifacts, or submission columns unless they are real dataset columns used for training.\n"
            "JSON schema:\n"
            "```json\n"
            f"{DataConfig.__annotations__}"
            "```\n"
            "Output only extracted data configuration as defined JSON schema."
        )
        messages      = [SystemMessage(content=system_prompt)] + self._normalize_messages(state["messages"])[-10:]
        full_response = llm.invoke(messages, tools=[], tool_choice="none")
        config_data   = self._parse_json_or_default(
            full_response.content,
            dict(config) if isinstance(config, dict) else {},
            "data config",
        )
        config_data["dataset_uri"] = data_location or config_data.get("dataset_uri")
        data_config = DataConfig(**config_data)
        self._validate_training_data_config(data_config, state)
        return data_config

    def _extract_training_plan(self, state: ETLState) -> TrainingPlan:
        """
        Assemble a complete TrainingPlan from the current conversation.

        Calls sub-extractors for hyperparameters, Ray config, and data
        config, then uses the LLM to infer model-level metadata
        (name, type, description, version).

        Parameters
        ----------
        state : ETLState

        Returns
        -------
        TrainingPlan
        """
        hyperparameter_config = self._extract_hyperparameter_config(state)
        ray_config            = self._extract_ray_config(state)
        training_plan = state.get("training_plan") or {}
        data_config = (
            self._extract_data_config(state)
            if "data_config" in training_plan
            else self._extract_initial_data_config(state)
        )

        system_prompt = (
            "Please extract model details including model type (classification or regression), model name, description and version from the user request or chat history.\n"
            f"Reference training dataset config: {data_config}\n"
            "JSON schema:\n"
            "```json\n"
            "{\n"
            "  \"model_type\": str,\n"
            "  \"model_name\": str,\n"
            "  \"model_description\": str,\n"
            "  \"model_version\": str (e.g. v1.0)\n"
            "}\n"
            "```\n"
            "Output only extracted model details as defined JSON schema."
        )
        messages      = [SystemMessage(content=system_prompt)] + self._normalize_messages(state["messages"])[-10:]
        full_response = llm.invoke(messages, tools=[], tool_choice="none")

        previous_plan = state.get("training_plan") or {}
        fallback_model_config = {
            "model_type": previous_plan.get("model_type") or "classification",
            "model_name": previous_plan.get("model_name") or f"{data_config.get('target_column') or 'Target'} Predictor",
            "model_description": (
                previous_plan.get("model_description")
                or f"Predict {data_config.get('target_column') or 'the target column'} using selected dataset features"
            ),
            "model_version": previous_plan.get("model_version") or "v1.0",
        }
        config_data = self._parse_json_or_default(
            full_response.content,
            fallback_model_config,
            "model details",
        )

        model_type = config_data.get("model_type") or previous_plan.get("model_type") or "classification"
        if model_type not in {"classification", "regression"}:
            model_type = previous_plan.get("model_type") or "classification"
        model_type = self._reconcile_model_type_with_target(
            model_type, state, data_config, has_previous_plan=bool(previous_plan)
        )
        model_name = config_data.get("model_name") or previous_plan.get("model_name") or "my_model"
        model_description = (
            config_data.get("model_description")
            or previous_plan.get("model_description")
            or "This is a model for..."
        )
        model_version = config_data.get("model_version") or previous_plan.get("model_version") or "v1.0"

        execution_backend, execution_reason, execution_size = self._training_backend_decision(
            data_config.get("dataset_uri", ""), self._state_dataset_size_bytes(state)
        )
        use_local = execution_backend == "local"
        dataset_size = self._state_dataset_size_bytes(state)
        logger.info(
            "Training backend decision: uri=%s size=%s backend=%s reason=%s",
            data_config.get("dataset_uri"),
            dataset_size if dataset_size is not None else "unknown",
            execution_backend,
            execution_reason,
        )
        if use_local:
            return TrainingPlan(
                model_type=model_type,
                model_name=model_name,
                model_description=model_description,
                model_version=model_version,
                hyperparameter_config=hyperparameter_config,
                ray_config=None,
                data_config=data_config,
                execution_backend="local",
                execution_reason=execution_reason,
                execution_dataset_size_bytes=execution_size,
                execution_dataset_uri=data_config.get("dataset_uri", ""),
            )
        return TrainingPlan(
            model_type=model_type,
            model_name=model_name,
            model_description=model_description,
            model_version=model_version,
            hyperparameter_config=hyperparameter_config,
            ray_config=ray_config,
            data_config=data_config,
            execution_backend="ray",
            execution_reason=execution_reason,
            execution_dataset_size_bytes=execution_size,
            execution_dataset_uri=data_config.get("dataset_uri", ""),
        )

    # ----------------------------------------------------------
    # Agent actions
    # ----------------------------------------------------------

    def _generate_training_plan(self, state: ETLState) -> ETLState:
        """
        Extract a new TrainingPlan from the conversation and present it to
        the user for review.

        Parameters
        ----------
        state : ETLState

        Returns
        -------
        ETLState with ``training_plan`` set and a summary appended to messages.
        """
        try:
            training_plan = self._extract_training_plan(state)
        except ValueError as exc:
            logger.info("Cannot generate training plan: %s", exc)
            new_state = state.copy()
            new_state["training_plan"] = None
            new_state["training_completed"] = False
            new_state["messages"].append(
                AIMessage(
                    content=(
                        "I can't create a training plan for this dataset yet.\n\n"
                        f"{exc}"
                    )
                )
            )
            return new_state
        logger.info("Generated initial training plan: %s", training_plan)

        new_state                  = state.copy()
        new_state["training_plan"] = training_plan
        new_state["training_completed"] = False
        new_state["messages"].append(
            AIMessage(
                content=self._format_training_plan_response(
                    training_plan,
                    full=self._latest_user_requests_full_training_plan(state),
                )
            )
        )
        return new_state

    def _update_training_plan(self, state: ETLState) -> ETLState:
        """
        Re-extract the TrainingPlan with user-requested modifications and
        present the updated plan for confirmation.

        Parameters
        ----------
        state : ETLState

        Returns
        -------
        ETLState with the updated ``training_plan`` and a summary appended.
        """
        clarification = self._training_plan_update_clarification(state)
        if clarification:
            new_state = state.copy()
            new_state["messages"].append(AIMessage(content=clarification))
            return new_state

        original_training_plan = copy.deepcopy(state.get("training_plan"))
        training_plan = self._apply_direct_training_plan_updates(state)
        if training_plan is None:
            if not self._training_plan_update_has_enough_detail(state):
                logger.info("Training plan update lacks specific change details; asking for clarification.")
                return self._ask_more_detail_to_update_training_plan(state)
            try:
                training_plan = self._extract_training_plan(state)
            except ValueError as exc:
                logger.info("Training plan update needs more detail: %s", exc)
                return self._ask_more_detail_to_update_training_plan(state)
        if self._training_plans_equivalent(original_training_plan, training_plan):
            logger.info("Training plan update produced no changes; asking user for more detail.")
            return self._ask_more_detail_to_update_training_plan(state)
        logger.info("Updated training plan: %s", training_plan)

        new_state                  = state.copy()
        new_state["training_plan"] = training_plan
        new_state["training_completed"] = False
        new_state["messages"].append(
            AIMessage(
                content=self._format_training_plan_response(
                    training_plan,
                    updated=True,
                    full=self._latest_user_requests_full_training_plan(state),
                )
            )
        )
        return new_state

    def _ask_more_detail_to_update_training_plan(self, state: ETLState) -> ETLState:
        """
        Ask the user for missing information needed to update the training plan.

        Parameters
        ----------
        state : ETLState

        Returns
        -------
        ETLState with a clarifying question appended to messages.
        """
        system_prompt = (
            "User want to update training plan but didn't provide enough information to make update on training plan.\n"
            "Please ask user for more details to update training plan.\n"
            "Ask shortly.\n"
            "Important: training is not started yet. User should confirm to start training with this updated plan."
        )
        messages      = [SystemMessage(content=system_prompt)] + self._normalize_messages(state["messages"])[-10:]
        full_response = llm.invoke(messages, tools=[], tool_choice="none")

        new_state = state.copy()
        new_state["messages"].append(AIMessage(content=full_response.content))
        return new_state

    def _ask_training_plan_reply_clarification(self, state: ETLState) -> ETLState:
        new_state = state.copy()
        new_state["messages"].append(
            AIMessage(
                content=(
                    "I could not tell whether you want to start training with the current plan "
                    "or change the plan first. You can also ask me to modify the dataset before training "
                    "or stop the pending training flow. Please clarify which action you want."
                )
            )
        )
        return new_state

    def _generate_code_repo(self, state: ETLState) -> ETLState:
        """
        Generate a training code repository from the current TrainingPlan.

        .. todo::
            Implement code repo generation based on training plan.

        Parameters
        ----------
        state : ETLState

        Returns
        -------
        ETLState (unchanged until implemented).
        """
        # TODO: Implement code repo generation based on training plan
        return state.copy()

    def _review_code_repo(self, state: ETLState) -> ETLState:
        """
        Open the generated code repo in a JupyterLab environment for review.

        .. todo::
            Implement code repo review via JupyterLab env.

        Parameters
        ----------
        state : ETLState

        Returns
        -------
        ETLState (unchanged until implemented).
        """
        # TODO: Implement code repo review via JupyterLab env
        return state.copy()

    def _format_training_result_report(self, training_result: Any, use_local: bool) -> str:
        if not isinstance(training_result, dict):
            return (
                "**Training Run Report**\n\n"
                "The training run finished, but the result format was not recognized."
            )

        failure = training_result.get("failure")
        error_detail = training_result.get("error") or training_result.get("execution_error")
        if error_detail:
            if not isinstance(failure, dict):
                failure = build_training_failure(error_detail)
                training_result["failure"] = failure
            return format_training_failure(failure)

        title = "Local Training Run Report" if use_local else "Training Run Report"
        model_type_raw = training_result.get("model_type")
        is_regression = isinstance(model_type_raw, str) and "regression" in model_type_raw.lower()
        model_type = model_type_raw.title() if isinstance(model_type_raw, str) else model_type_raw

        fields = [
            ("Model Name", "model_name"),
            ("Model Type", "model_type"),
            ("Model Version", "model_version"),
            ("MLflow Run ID", "mlflow_run_id"),
            ("Rows Processed", "rows_processed"),
            ("Number of Epochs Trained", "num_epochs_trained"),
            ("Number of Features", "num_features"),
        ]
        if is_regression:
            fields.extend([
                ("Training Loss", "final_train_loss"),
                ("Validation Loss", "final_val_loss"),
                ("Average Error", "final_mae"),
                ("Large-Error Score", "final_rmse"),
                ("Explained Variation", "final_r2_score"),
            ])
        else:
            fields.extend([
                ("Number of Classes", "num_classes"),
                ("Final Train Loss", "final_train_loss"),
                ("Final Validation Loss", "final_val_loss"),
                ("Final Accuracy", "final_accuracy"),
                ("Final F1 Score", "final_f1_score"),
            ])

        metric_lines = []
        for label, key in fields:
            value = training_result.get(key)
            if value is None:
                continue
            if key == "model_type":
                value = model_type
            elif key == "final_r2_score":
                try:
                    value = f"{float(value) * 100:.2f}%"
                except (TypeError, ValueError):
                    pass
            metric_lines.append(f"* **{label}:** {value}")

        evaluation_report = training_result.get("evaluation_report")
        if isinstance(evaluation_report, dict):
            splitter = evaluation_report.get("splitter")
            if splitter:
                metric_lines.append(f"* **Validation Splitter:** {splitter}")
            time_column = evaluation_report.get("time_column")
            if time_column:
                metric_lines.append(f"* **Time Column:** {time_column}")
            splitter_reason = evaluation_report.get("splitter_reason")
            if splitter_reason:
                metric_lines.append(f"* **Validation Reason:** {splitter_reason}")

        if not metric_lines:
            error_detail = training_result.get("error") or training_result.get("execution_error")
            if error_detail:
                return (
                    f"**{title}**\n\n"
                    "The training run failed.\n\n"
                    f"**Error:** {error_detail}"
                )
            return (
                f"**{title}**\n\n"
                "The training run failed or returned no reportable training metrics."
            )

        error_detail = training_result.get("error") or training_result.get("execution_error")
        success_keys = (
            ("mlflow_run_id", "final_val_loss", "final_mae", "final_rmse", "final_r2_score")
            if is_regression
            else ("mlflow_run_id", "final_val_loss", "final_accuracy", "final_f1_score")
        )
        succeeded = not error_detail and any(training_result.get(key) is not None for key in success_keys)

        if is_regression and succeeded:
            regression_parts = []
            for label, key in (
                ("Average Error", "final_mae"),
                ("Large-Error Score", "final_rmse"),
                ("Explained Variation", "final_r2_score"),
            ):
                value = training_result.get(key)
                if value is not None:
                    if key == "final_r2_score":
                        try:
                            value = f"{float(value) * 100:.2f}%"
                        except (TypeError, ValueError):
                            pass
                    regression_parts.append(f"{label} {value}")
            status_sentence = "The regression training was successful."
            if regression_parts:
                status_sentence += " Validation performance: " + ", ".join(regression_parts) + "."
        elif succeeded:
            final_accuracy = training_result.get("final_accuracy")
            final_f1_score = training_result.get("final_f1_score")
            status_sentence = "The training was successful"
            if final_accuracy is not None and final_f1_score is not None:
                status_sentence += f", and the model achieved an accuracy of {final_accuracy} and an F1 score of {final_f1_score}."
            else:
                status_sentence += "."
        else:
            status_sentence = "The training failed or did not return reportable metrics."

        lines = [
            f"**{title}**",
            "",
            "The training result is as follows:",
            "",
            *metric_lines,
            "",
            status_sentence,
        ]
        dashboard_url = training_result.get("dashboard_url")
        if dashboard_url:
            lines.extend(
                [
                    "",
                    f"Ray dashboard: {dashboard_url}",
                    "The dashboard remains available for 30 minutes to inspect training logs.",
                ]
            )
        if succeeded:
            lines.extend(
                [
                    "",
                    "You can view the trained model in the Configurations tab.",
                    "",
                    "You can now run a prediction in chat by sending one row or a JSON array of rows with feature values.",
                ]
            )
        return "\n".join(lines)

    def _execute_training(self, state: ETLState) -> ETLState:
        """
        Submit a training job using the stored TrainingPlan and append
        a human-readable summary of the result to the conversation.

        Backend selection
        -----------------
        - **LocalTrainer** is used when the data source is a local filesystem
          path (no cloud URI found in the state). Training runs in-process
          without Ray.
        - **RayTrainer** is used in all other cases: cloud URIs (gs://, s3://,
          http/https) or SQL connection strings, where distributed execution
          is expected.

        Guards against missing training plan and surfaces dashboard URL to
        the user for a 30-minute post-run inspection window (Ray path only).

        Parameters
        ----------
        state : ETLState

        Returns
        -------
        ETLState with ``training_result`` populated and summary appended.
        """
        training_plan = state.get("training_plan")
        if training_plan is None:
            logger.error("No training plan found in state. Cannot execute training.")
            new_state = state.copy()
            new_state["messages"].append(
                AIMessage(content="Sorry, the model training plan is not found. Please make plan at the first.")
            )
            return self._finish_mta(new_state)

        training_plan = copy.deepcopy(training_plan)
        data_config = training_plan.get("data_config") or {}
        preferred_uri = self._resolve_training_data_location(state)
        if not preferred_uri:
            unavailable_reason = self._preferred_training_data_unavailable_reason(state)
            if unavailable_reason:
                new_state = state.copy()
                new_state["training_plan"] = training_plan
                new_state["training_completed"] = False
                new_state["training_scheduled"] = False
                new_state["messages"].append(
                    AIMessage(
                        content=(
                            "I can't start training with the current plan.\n\n"
                            f"{unavailable_reason}"
                        )
                    )
                )
                return self._finish_mta(new_state)

        modified_dataset_active = self._modified_dataset_is_active(state)
        if preferred_uri and (
            not data_config.get("dataset_uri")
            or (
                modified_dataset_active
                and data_config.get("dataset_uri") != preferred_uri
            )
            or (
                not self._is_remote_data_source(data_config.get("dataset_uri", ""))
                and self._is_remote_data_source(preferred_uri)
            )
        ):
            data_config["dataset_uri"] = preferred_uri
            training_plan["data_config"] = data_config

        try:
            self._validate_training_data_config(data_config, state)
            self._validate_training_pii_key(data_config, state)
        except (ValueError, MissingPIIKey) as exc:
            logger.info("Cannot execute training plan: %s", exc)
            new_state = state.copy()
            new_state["training_plan"] = training_plan
            new_state["training_completed"] = False
            new_state["training_scheduled"] = False
            detail = (
                format_training_failure(build_training_failure(exc))
                if isinstance(exc, MissingPIIKey)
                else str(exc)
            )
            new_state["messages"].append(
                AIMessage(
                    content=(
                        "I can't start training with the current plan.\n\n"
                        f"{detail}"
                    )
                )
            )
            return self._finish_mta(new_state)

        dataset_size = self._state_dataset_size_bytes(state)
        current_dataset_uri = data_config.get("dataset_uri", "")
        planned_backend = training_plan.get("execution_backend")
        planned_uri = training_plan.get("execution_dataset_uri")
        if planned_backend in {"local", "ray"} and planned_uri == current_dataset_uri:
            execution_backend = planned_backend
            execution_reason = training_plan.get("execution_reason") or "Using the execution backend selected in the confirmed plan."
            execution_size = training_plan.get("execution_dataset_size_bytes")
        else:
            execution_backend, execution_reason, execution_size = self._training_backend_decision(
                current_dataset_uri, dataset_size
            )
        # Plans created before explicit execution metadata used ray_config as
        # an implicit request for Ray. Preserve that compatibility.
        if "execution_backend" not in training_plan and training_plan.get("ray_config"):
            execution_backend = "ray"
            execution_reason = "This existing plan already requests a Ray job."
        training_plan["execution_backend"] = execution_backend
        training_plan["execution_reason"] = execution_reason
        training_plan["execution_dataset_size_bytes"] = execution_size
        training_plan["execution_dataset_uri"] = current_dataset_uri
        use_local = execution_backend == "local"
        if not use_local and training_plan.get("ray_config") is None:
            training_plan["ray_config"] = self._extract_ray_config(state)
        elif use_local:
            training_plan["ray_config"] = None

        if use_local:
            # ---- Local path: run training in-process via LocalTrainer ----
            logger.info("Data source is local — using LocalTrainer (no Ray).")
            trainer = LocalTrainer(
                user_id=state.get("user_id", "unknown_user"),
                session_id=state.get("session_id", "unknown_session"),
            )
            training_result = trainer.train(training_plan)
            logger.info("Local training completed with result: %s", training_result)

        else:
            # ---- Ray path: submit distributed job via RayTrainer ----
            if not state.get("training_scheduled", False):
                logger.info("Training task is scheduled")
                new_state = state.copy()
                new_state["task_schedule"] = {
                    "task_type": "training",
                    "schedule_type": "relative",
                    "second": 0,
                    "max_runs": 1
                }
                new_state["training_scheduled"] = True
                new_state["training_plan"] = training_plan
                return new_state
            
            logger.info("Data source is remote/cloud — using RayTrainer.")
            trainer = RayTrainer(
                user_id=state.get("user_id", "unknown_user"),
                session_id=state.get("session_id", "unknown_session"),
            )
            training_result = trainer.train(training_plan)
            logger.info("Training completed with result: %s", training_result)

        if isinstance(training_result, dict):
            error_detail = training_result.get("error") or training_result.get("execution_error")
            if error_detail and not isinstance(training_result.get("failure"), dict):
                training_result["failure"] = build_training_failure(error_detail)

        report_content = self._format_training_result_report(training_result, use_local)
        training_failed = bool(
            isinstance(training_result, dict)
            and (
                training_result.get("status") == "error"
                or training_result.get("error")
                or training_result.get("execution_error")
            )
        )

        new_state                    = state.copy()
        new_state["training_result"] = training_result
        if isinstance(training_result, dict) and isinstance(training_result.get("integrity_report"), dict):
            new_state["integrity_report"] = training_result["integrity_report"]
            new_state["integrity_safe_to_train"] = bool(
                training_result["integrity_report"].get("safe_to_train")
            )
        if isinstance(training_result, dict) and isinstance(training_result.get("evaluation_report"), dict):
            new_state["evaluation_report"] = training_result["evaluation_report"]
            new_state["evaluation_beats_baseline"] = training_result[
                "evaluation_report"
            ].get("beats_baseline")
        if isinstance(training_result, dict) and training_result.get("mlflow_run_id"):
            new_state["mlflow_run_id"] = training_result["mlflow_run_id"]
        new_state["messages"].append(AIMessage(content=report_content))
        new_state["training_scheduled"] = False
        new_state["training_completed"] = not training_failed
        return new_state

    def _configure_inference_service(self, state: ETLState) -> ETLState:
        """Schedule, then configure, the selected trained model's service."""
        training_result = state.get("training_result") or {}
        mlflow_run_id = training_result.get("mlflow_run_id") or state.get("mlflow_run_id")
        if not mlflow_run_id:
            new_state = state.copy()
            new_state["messages"].append(
                AIMessage(content="A trained model is required before configuring an inference service.")
            )
            return self._finish_mta(new_state)

        if not state.get("configure_inference_service_scheduled", False):
            logger.info("Inference service configuration task is scheduled")
            new_state = state.copy()
            new_state["task_schedule"] = {
                "task_type": "start_inference",
                "schedule_type": "relative",
                "second": 0,
                "max_runs": 1,
            }
            new_state["configure_inference_service_scheduled"] = True
            return new_state

        try:
            details = InferenceServiceManager().configure_inference_service(mlflow_run_id)
            new_state = state.copy()
            new_state["configure_inference_service_scheduled"] = False
            new_state["task_schedule"] = None
            new_state["inference_service_details"] = details
            service_url = details.get("gateway_url") or details.get("endpoint")
            content = "Inference service setup is complete."
            if service_url:
                content += f" The service is available at: {service_url}"
            new_state["messages"].append(AIMessage(content=content))
            return self._finish_mta(new_state)
        except Exception as exc:
            logger.error("Error configuring inference service: %s", exc, exc_info=True)
            new_state = state.copy()
            new_state["configure_inference_service_scheduled"] = False
            new_state["task_schedule"] = None
            new_state["messages"].append(
                AIMessage(content="Sorry, there was an error configuring the inference service. Please try again.")
            )
            return self._finish_mta(new_state)

    def _stop_inference_service(self, state: ETLState) -> ETLState:
        """Schedule, then stop, the selected trained model's service."""
        training_result = state.get("training_result") or {}
        mlflow_run_id = training_result.get("mlflow_run_id") or state.get("mlflow_run_id")
        if not mlflow_run_id:
            new_state = state.copy()
            new_state["messages"].append(
                AIMessage(content="A trained model is required before stopping an inference service.")
            )
            return self._finish_mta(new_state)

        if not state.get("stop_inference_service_scheduled", False):
            logger.info("Inference service stopping task is scheduled")
            new_state = state.copy()
            new_state["task_schedule"] = {
                "task_type": "stop_inference",
                "schedule_type": "relative",
                "second": 0,
                "max_runs": 1,
            }
            new_state["stop_inference_service_scheduled"] = True
            return new_state

        try:
            InferenceServiceManager().stop_inference_service(mlflow_run_id)
            new_state = state.copy()
            new_state["stop_inference_service_scheduled"] = False
            new_state["task_schedule"] = None
            new_state["inference_service_details"] = None
            new_state["messages"].append(AIMessage(content="The inference service has been stopped."))
            return self._finish_mta(new_state)
        except Exception as exc:
            logger.error("Error stopping inference service: %s", exc, exc_info=True)
            new_state = state.copy()
            new_state["stop_inference_service_scheduled"] = False
            new_state["task_schedule"] = None
            new_state["messages"].append(
                AIMessage(content="Sorry, there was an error stopping the inference service. Please try again.")
            )
            return self._finish_mta(new_state)

    def _execute_inference(self, state: ETLState) -> ETLState:
        """
        Run inference on user-provided input data using the trained model.

        Extracts input features from the conversation via the LLM, calls
        :meth:`InferenceInterface.predict`, and appends a natural-language
        summary of the result.

        Parameters
        ----------
        state : ETLState
            Must contain ``training_result`` with a valid ``mlflow_run_id``.

        Returns
        -------
        ETLState with inference result summary appended to messages.
        """
        if state.get("training_result") is None or state["training_result"].get("mlflow_run_id") is None:
            logger.error("No training result found in state. Cannot execute inference.")
            new_state = state.copy()
            new_state["messages"].append(
                AIMessage(content="Sorry, the model training result is not found. Please train model at the first.")
            )
            return self._finish_mta(new_state)

        state = self._drop_planner_inference_format_response(state)

        inferencer = InferenceInterface()
        mlflow_run_id = state["training_result"]["mlflow_run_id"]
        model_detail = inferencer.get_model_details(mlflow_run_id)

        if model_detail is None:
            logger.error("Model detail not found for run '%s'.", mlflow_run_id)
            new_state = state.copy()
            new_state["messages"].append(
                AIMessage(content="Sorry, the model details could not be retrieved. Please try again.")
            )
            return self._finish_mta(new_state)

        # ModelDetail uses "feature_names", not "feature_columns".
        feature_names = self._model_detail_value(model_detail, "feature_names", []) or []

        input_data = None
        latest_user_text = self._latest_user_text(state).split("\n\n[Analysis context]", 1)[0]
        try:
            input_data = self._extract_inference_json(latest_user_text)
            logger.info("Using inference input parsed directly from the latest user message.")
        except ValueError:
            input_data = self._extract_named_inference_values(latest_user_text, feature_names)
            if input_data is not None:
                logger.info("Using inference input parsed from named feature values in the latest user message.")

            elif llm is None:
                new_state = state.copy()
                new_state["messages"].append(
                    AIMessage(content=self._inference_sample_input_prompt(feature_names))
                )
                return self._finish_mta(new_state)

            else:
                system_prompt = (
                    "Extract only the inference input explicitly provided by the user or chat history.\n"
                    "Do not invent, calculate, or default any feature values.\n"
                    "The accepted forms are one JSON object, a JSON array of row objects, or "
                    'a JSON object containing only "dataset_uri".\n'
                    f"Required model features for row input: {feature_names}.\n"
                    "Output JSON only."
                )
                messages = [SystemMessage(content=system_prompt)] + self._normalize_messages(state["messages"])[-10:]
                full_response = llm.invoke(messages, tools=[], tool_choice="none")
                try:
                    input_data = self._extract_inference_json(full_response.content)
                except ValueError:
                    new_state = state.copy()
                    new_state["messages"].append(
                        AIMessage(content=self._inference_sample_input_prompt(feature_names))
                    )
                    return self._finish_mta(new_state)

        input_data = self._normalize_inference_payload_values(input_data)
        if input_data == {}:
            new_state = state.copy()
            new_state["messages"].append(
                AIMessage(content=self._inference_sample_input_prompt(feature_names))
            )
            return self._finish_mta(new_state)

        input_error = self._inference_input_error(input_data, feature_names)
        if input_error:
            new_state = state.copy()
            new_state["messages"].append(AIMessage(content=input_error))
            return self._finish_mta(new_state)

        try:
            is_dataset_inference = isinstance(input_data, dict) and "dataset_uri" in input_data
            inference_service_details = (
                self._model_detail_value(model_detail, "inference_service_details", None)
                or state.get("inference_service_details")
                or (state.get("training_result") or {}).get("inference_service_details")
            )
            if inference_service_details and not is_dataset_inference:
                logger.info("Using configured inference gateway for run '%s'.", mlflow_run_id)
                inference_result = InferenceServiceManager().inference(
                    mlflow_run_id=mlflow_run_id,
                    input_data=input_data,
                )
            elif is_dataset_inference:
                inference_result = inferencer.inference_local(
                    mlflow_run_id=mlflow_run_id,
                    dataset_uri=input_data["dataset_uri"],
                )
            elif isinstance(input_data, list):
                inference_result = inferencer.predict_batch(
                    mlflow_run_id=mlflow_run_id,
                    rows=input_data,
                )
            else:
                inference_result = inferencer.predict(
                    mlflow_run_id=mlflow_run_id,
                    row=input_data,
                )
            inference_result = self._plain_inference_result(inference_result)
            logger.info("Inference executed with result: %s", inference_result)

            model_info = self._model_detail_value(model_detail, "model_info", {}) or {}
            if not isinstance(model_info, dict):
                model_info = {}
            categorical_encodings = (
                model_info
                    .get("data_info", {})
                    .get("preprocessing", {})
                    .get("categorical_features", {})
            )

            new_state = state.copy()
            if not self._latest_user_requests_detailed_inference_report(state) or llm is None:
                summary = self._format_compact_inference_summary(inference_result, model_detail)
                new_state["messages"].append(
                    AIMessage(content=summary)
                )
                return new_state

            system_prompt = (
                "Inference has been executed.\n"
                "The user explicitly asked for a detailed or expanded inference report.\n"
                f"Target column: {self._model_detail_value(model_detail, 'target_column', '')}\n"
                f"Known class names: {self._model_detail_value(model_detail, 'class_names', [])}\n"
                "Here are the encodings for the categorical features:\n"
                f"Encodings: {categorical_encodings}\n"
                "Here are the inference result details:\n"
                f"Inference result: {inference_result}\n"
                "Write a clear inference report. Include raw logits, class probabilities, "
                "feature handling, or encoding details only when they are present in the result "
                "or useful for the user's detailed request. Do not invent ignored fields."
            )
            messages = [SystemMessage(content=system_prompt)] + self._normalize_messages(state["messages"])[-10:]
            full_response = llm.invoke(messages, tools=[], tool_choice="none")
            new_state["messages"].append(
                AIMessage(content=full_response.content)
            )
            return new_state
        
        except Exception as e:
            logger.error("Error during inference execution: %s", e, exc_info=True)
            new_state = state.copy()
            new_state["messages"].append(
                AIMessage(content="Sorry, there was an error during inference execution. Please try again.")
            )
            return self._finish_mta(new_state)

    def _finish_mta(self, state: ETLState) -> ETLState:
        """
        Reset training-related flags in the state to signal that the MTA
        workflow has concluded (successfully or with an error).

        Parameters
        ----------
        state : ETLState

        Returns
        -------
        ETLState with ``enable_training``, ``skip_to_training``, and
        ``training_task`` cleared.
        """
        new_state                       = state.copy()
        new_state["training_scheduled"] = False
        # A planner may classify an interactive inference request as an
        # immediate ``execute`` task before this agent handles it directly.
        # Once MTA work is finished that schedule is stale; leaving it in the
        # state sends the successful result to Celery and turns it into a 500
        # when no worker is running.  Genuine async jobs return before calling
        # _finish_mta, so clearing it here does not cancel scheduled work.
        new_state["task_schedule"]      = None
        new_state["inference_scheduled"] = False
        new_state["enable_training"]    = False
        new_state["skip_to_training"]   = False
        new_state["training_task"]      = None
        new_state["training_plan_reply_action"] = "none"
        return new_state

    # ----------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------

    def execute(self, state: ETLState) -> ETLState:
        """
        Determine the next action(s) from the conversation and execute them.

        The LLM is invoked in tool-calling mode to select one or more tools
        from ``TOOLS``. Each selected tool maps to a private handler method.
        Multiple tools can be called in a single turn (e.g. generate_code_repo
        → review_code_repo → execute_training on user confirmation).

        Parameters
        ----------
        state : ETLState
            Current conversation and training state.

        Returns
        -------
        ETLState
            Updated state after all selected handlers have run.
        """
        try:
            task_type = (state.get("task_schedule") or {}).get("task_type")
            is_start_inference_task = task_type == "start_inference"
            is_stop_inference_task = task_type == "stop_inference"
            if state.get("training_scheduled", False):
                logger.info("Training task is scheduled, executing training...")
                new_state = self._execute_training(state)
                return self._finish_mta(new_state)
            if is_start_inference_task and state.get("configure_inference_service_scheduled", False):
                logger.info("Configure inference service task is scheduled, executing...")
                return self._finish_mta(self._configure_inference_service(state))
            if is_stop_inference_task and state.get("stop_inference_service_scheduled", False):
                logger.info("Stop inference service task is scheduled, executing...")
                return self._finish_mta(self._stop_inference_service(state))
            if state.get("configure_inference_service_scheduled") or state.get("stop_inference_service_scheduled"):
                logger.info("Clearing stale inference service schedule flags on an interactive turn.")
                state = state.copy()
                state["configure_inference_service_scheduled"] = False
                state["stop_inference_service_scheduled"] = False
                state["task_schedule"] = None

            training_plan_reply_action = state.get("training_plan_reply_action")
            training_confirmed = bool(
                state.get("training_plan")
                and (state.get("skip_to_training") or training_plan_reply_action == "confirm")
                and not state.get("training_completed", False)
            )
            training_plan_update_requested = bool(
                state.get("training_plan")
                and not state.get("training_completed", False)
                and training_plan_reply_action == "update"
            )
            training_plan_reply_unclear = bool(
                state.get("training_plan")
                and not state.get("training_completed", False)
                and training_plan_reply_action == "unclear"
            )
            full_training_plan_requested = bool(
                state.get("training_plan")
                and not state.get("training_completed", False)
                and self._latest_user_requests_full_training_plan(state)
            )
            if training_plan_update_requested:
                logger.info("Pending training plan update requested; updating plan before any training.")
                new_state = self._update_training_plan(state)
                return self._finish_mta(new_state)
            if full_training_plan_requested:
                logger.info("Full training plan requested; returning detailed plan without starting training.")
                new_state = state.copy()
                new_state["messages"].append(
                    AIMessage(
                        content=self._format_training_plan_response(
                            state["training_plan"],
                            full=True,
                        )
                    )
                )
                return self._finish_mta(new_state)
            if training_confirmed:
                logger.info("Confirmed training plan found; executing MTA training without tool re-selection.")
                new_state = self._generate_code_repo(state)
                new_state = self._review_code_repo(new_state)
                new_state = self._execute_training(new_state)
                # The first Ray pass does not train immediately: it returns a
                # one-shot Celery/RedBeat schedule.  Preserve those routing
                # flags so route_after_training can enter schedule_task.  The
                # scheduled second pass executes Ray training and clears the
                # flags through _finish_mta as usual.
                if new_state.get("training_scheduled", False):
                    return new_state
                return self._finish_mta(new_state)
            if training_plan_reply_unclear:
                logger.info("Pending training plan reply was unclear; asking for clarification.")
                new_state = self._ask_training_plan_reply_clarification(state)
                return self._finish_mta(new_state)

            service_action = self._latest_user_requests_inference_service(state)
            if state.get("training_result") and service_action == "start":
                new_state = self._configure_inference_service(state)
                return new_state if new_state.get("task_schedule") else self._finish_mta(new_state)
            if state.get("training_result") and service_action == "stop":
                new_state = self._stop_inference_service(state)
                return new_state if new_state.get("task_schedule") else self._finish_mta(new_state)

            if state.get("training_result") and self._latest_user_requests_inference(state):
                logger.info("Inference JSON request detected; executing inference without tool re-selection.")
                new_state = self._execute_inference(state)
                return self._finish_mta(new_state)

            if not llm:
                logger.error("Planner LLM is not available. Cannot execute training plan.")
                new_state = state.copy()
                new_state["messages"].append(
                    AIMessage(content="Sorry, the model training agent is currently unavailable. Please try again later.")
                )
                return self._finish_mta(new_state)

            messages = (
                [SystemMessage(content=SYSTEM_PROMPT)]
                + self._normalize_messages(state["messages"])[-10:]
            )
            response = llm.invoke(messages, tools=TOOLS, tool_choice="required")

            # Normalise tool_calls — some SDK versions expose different attrs
            if hasattr(response, "tool_calls"):
                tool_calls = response.tool_calls
            elif hasattr(response, "tool_call"):
                tool_calls = [response.tool_call]
            else:
                tool_calls = []

            if not tool_calls:
                logger.warning("LLM response does not contain any tool calls.")
                new_state = state.copy()
                new_state["messages"].append(
                    AIMessage(content="Sorry, the model training agent did not receive any tool calls. Please try again.")
                )
                return self._finish_mta(new_state)

            requested_tools = [
                self.parse_tool_call(tool_call)
                for tool_call in tool_calls
            ]
            had_training_plan = bool(state.get("training_plan"))

            if not had_training_plan and (
                "generate_training_plan" in requested_tools
                or "execute_training" in requested_tools
            ):
                logger.info(
                    "Initial MTA training request detected; generating plan and stopping before training."
                )
                new_state = self._generate_training_plan(state.copy())
                return self._finish_mta(new_state)

            if "update_training_plan" in requested_tools:
                logger.info("MTA training plan update detected; updating plan and stopping before training.")
                new_state = self._update_training_plan(state.copy())
                return self._finish_mta(new_state)

            if "ask_more_detail_to_update_training_plan" in requested_tools:
                logger.info("MTA training plan update is ambiguous; asking for more details.")
                new_state = self._ask_more_detail_to_update_training_plan(state.copy())
                return self._finish_mta(new_state)

            if "execute_training" in requested_tools and not training_confirmed:
                logger.warning(
                    "MTA requested execute_training without explicit confirmation; blocking execution."
                )
                new_state = state.copy()
                new_state["messages"].append(
                    AIMessage(
                        content=(
                            "I have the training plan ready, but training has not started yet. "
                            "Please confirm if you want me to start training with this plan."
                        )
                    )
                )
                return self._finish_mta(new_state)

            new_state = state.copy()
            for parsed_tool_name in requested_tools:
                logger.info("LLM called tool: %s", parsed_tool_name)

                if parsed_tool_name == "generate_training_plan":
                    new_state = self._generate_training_plan(new_state)
                elif parsed_tool_name == "ask_more_detail_to_update_training_plan":
                    new_state = self._ask_more_detail_to_update_training_plan(new_state)
                elif parsed_tool_name == "update_training_plan":
                    new_state = self._update_training_plan(new_state)
                elif parsed_tool_name == "generate_code_repo":
                    new_state = self._generate_code_repo(new_state)
                elif parsed_tool_name == "review_code_repo":
                    new_state = self._review_code_repo(new_state)
                elif parsed_tool_name == "execute_training":
                    new_state = self._execute_training(new_state)
                elif parsed_tool_name == "configure_inference_service":
                    new_state = self._configure_inference_service(new_state)
                elif parsed_tool_name == "execute_inference":
                    new_state = self._execute_inference(new_state)
                elif parsed_tool_name == "stop_inference_service":
                    new_state = self._stop_inference_service(new_state)
                else:
                    logger.warning("Unknown tool name received: '%s' — skipping.", parsed_tool_name)

            if (
                new_state.get("training_scheduled", False)
                or new_state.get("configure_inference_service_scheduled", False)
                or new_state.get("stop_inference_service_scheduled", False)
            ):
                return new_state
            else:
                return self._finish_mta(new_state)

        except Exception as exc:
            # tool_choice="required" forbids conversational replies, so when an
            # out-of-scope ask reaches the MTA the model's polite refusal comes
            # back as a Groq 400 (code=tool_use_failed) with the refusal text in
            # failed_generation. That text IS the right answer for the user —
            # surface it instead of a generic apology.
            declined_text = self._tool_use_failed_text(exc)
            if declined_text:
                logger.info(
                    "MTA got a conversational reply under tool_choice=required; "
                    "surfacing it to the user: %s", declined_text[:120],
                )
                new_state = state.copy()
                new_state["messages"].append(AIMessage(content=declined_text))
                return self._finish_mta(new_state)
            logger.error("Error during LLM invocation: %s", exc, exc_info=True)
            new_state = state.copy()
            new_state["messages"].append(
                AIMessage(content="Sorry, something went wrong while processing your request. Please try again.")
            )
            return self._finish_mta(new_state)
