"""
Temporary Inference Manager for Model Training Agent

Orchestrates temporary inference workflow:
- Detects inference requests from chat
- Extracts CSV from state (uploaded files, paths, data_source_location, output_location)
- Stages CSV to GCS
- Creates Kubernetes Jobs for inference
- Monitors job status and retrieves results
- Formats predictions for chat display
"""

import logging
import os
import re
import csv
import tempfile
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
from io import StringIO

from langchain_core.messages import BaseMessage
from app.graph.etl_state import ETLState
from app.agents.mta.k8s_job_manager import KubernetesJobManager
from app.agents.mta.mlflow_integration import MLflowManager
from app.core import cloud_config

logger = logging.getLogger(__name__)

# Import LocalInferenceManager for local mode
try:
    from app.agents.mta.local_inference import LocalInferenceManager
    LOCAL_INFERENCE_AVAILABLE = True
except ImportError:
    LOCAL_INFERENCE_AVAILABLE = False
    logger.warning("LocalInferenceManager not available - local inference mode will be disabled")

# Sentinel value to indicate "explicitly use default service account, don't check env vars"
_USE_DEFAULT_SERVICE_ACCOUNT = object()

def parse_direct_input(input_string: str) -> Optional[pd.DataFrame]:
    """
    Parse direct input string like "SepalLengthCm 5.1 SepalWidthCm 3.5 PetalLengthCm 1.4 PetalWidthCm 0.2"
    into a DataFrame.
    
    Args:
        input_string: Space-separated feature name-value pairs
        
    Returns:
        DataFrame with one row, or None if parsing fails
    """
    try:
        # Split by whitespace and parse pairs
        parts = input_string.strip().split()
        if len(parts) % 2 != 0:
            logger.warning(f"Odd number of tokens in input string: {len(parts)}")
            return None
        
        # Parse feature-value pairs
        data = {}
        for i in range(0, len(parts), 2):
            feature_name = parts[i]
            try:
                feature_value = float(parts[i + 1])
                data[feature_name] = feature_value
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse feature-value pair at index {i}: {e}")
                return None
        
        if not data:
            return None
        
        # Create DataFrame with single row
        df = pd.DataFrame([data])
        logger.info(f"Parsed direct input: {len(data)} features -> DataFrame shape {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to parse direct input: {e}")
        return None


def detect_inference_mode(messages: List[BaseMessage], state: ETLState) -> str:
    """
    Detect inference mode (local vs k8s) based on user intent.
    
    Determines mode from keywords and context, focusing on the latest message
    to match the user's most recent intent (consistent with detect_inference_request).
    
    - "k8s" for: batch, scale, production, large-scale, deploy, automated, recurring, periodic, cron, job
    - "local" for: quick, interactive, now, immediately, direct, instant
    - Default: "local" (for real-time/sample inference)
    
    Args:
        messages: List of chat messages
        state: Current ETL state
    
    Returns:
        "local" or "k8s" based on user intent
    """
    if not messages:
        # Default to local for real-time inference
        return "local"
    
    # Focus on the last message to match user's latest intent
    # This is consistent with detect_inference_request() which also checks the last message
    last_message = messages[-1].content.lower() if messages else ""
    
    # Keywords indicating Kubernetes/scheduled inference
    k8s_keywords = [
        "batch", "large scale", "large-scale",
        "production", "at scale", "scaled", "deploy", "deployment",
        "automated", "recurring", "periodic", "cron", "job"
    ]
    
    # Keywords indicating local/real-time inference
    local_keywords = [
        "quick", "interactive", "now", "immediately", "direct", "instant"
    ]
    
    # Check for k8s intent in the last message
    has_k8s_intent = any(keyword in last_message for keyword in k8s_keywords)
    
    # Check for local intent in the last message
    has_local_intent = any(keyword in last_message for keyword in local_keywords)
    
    # Check state for scheduled task indicators
    is_scheduled = (
        state.get("scheduled_task") or 
        state.get("task_scheduled") or 
        state.get("batch_inference") or
        state.get("task_type") == "scheduled"
    )
    
    # Decision logic:
    # 1. Explicit k8s keywords → k8s
    # 2. Scheduled task in state → k8s
    # 3. Explicit local keywords → local
    # 4. Default: local (for real-time/sample inference)
    
    if has_k8s_intent or is_scheduled:
        logger.info("Detected Kubernetes inference mode from user intent (keywords or scheduled task)")
        return "k8s"
    elif has_local_intent:
        logger.info("Detected local inference mode from user intent (test/try/sample keywords)")
        return "local"
    else:
        # Default to local for real-time inference (sample/testing use case)
        logger.info("Defaulting to local inference mode (real-time/sample inference)")
        return "local"


def detect_inference_request(messages: List[BaseMessage], state: ETLState) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Detect if user is requesting temporary inference and determine mode.
    
    Args:
        messages: List of chat messages
        state: Current ETL state
    
    Returns:
        (is_request, path_or_source, inference_mode)
        - is_request: True if inference request detected
        - path_or_source: Path string, "uploaded", "data_source", "output", "execution_output", or None
        - inference_mode: "local" or "k8s" based on user intent, or None if not a request
    """
    # Check 1: Training must be completed
    if not state.get("training_completed"):
        return False, None, None
    
    # Check 2: Must have MLflow run ID
    if not state.get("mlflow_run_id"):
        return False, None, None
    
    if not messages:
        return False, None, None
    
    last_message = messages[-1].content.lower() if messages else ""
    full_message = messages[-1].content if messages else ""
    
    # Check 3: Check message content for inference keywords OR direct input pattern
    inference_keywords = [
        "run inference", "run predictions", "predict", "inference",
        "test model", "use model", "apply model", "make predictions",
        "generate predictions", "what would", "what does the model predict"
    ]
    
    has_keyword = any(keyword in last_message for keyword in inference_keywords)
    
    # Check for direct input pattern: feature names followed by numbers
    # Pattern: word number word number ... (at least 2 pairs)
    # Also check if message contains quoted direct input like: "SepalLengthCm 5.1 SepalWidthCm 3.5..."
    direct_input_pattern = r'\b\w+\s+[\d.]+(?:\s+\w+\s+[\d.]+){1,}'
    has_direct_input = bool(re.search(direct_input_pattern, full_message))
    
    # Extract direct input string if present (handle quoted strings)
    direct_input_string = None
    if has_direct_input:
        # Try to extract the direct input part
        # Look for quoted strings first - match opening and closing quotes of the same type
        # Try double quotes first, then single quotes to avoid mismatched quotes
        quoted_match = re.search(r'"([^"]+)"', full_message) or re.search(r"'([^']+)'", full_message)
        if quoted_match:
            quoted_content = quoted_match.group(1)
            # Check if quoted content matches direct input pattern
            if re.search(direct_input_pattern, quoted_content):
                direct_input_string = quoted_content
        else:
            # Extract the direct input part from the message
            # Find the part that matches the pattern
            match = re.search(direct_input_pattern, full_message)
            if match:
                direct_input_string = match.group(0)
    
    if not has_keyword and not has_direct_input:
        return False, None, None
    
    # Detect inference mode from user intent
    inference_mode = detect_inference_mode(messages, state)
    
    # Check 4: Detect path references in message
    # Pattern: absolute path (/path/to/file.csv) or relative path (./file.csv, ../file.csv)
    path_patterns = [
        r'["\']?([/][^"\'\s]+\.csv)["\']?',  # Absolute path
        r'["\']?([./][^"\'\s]+\.csv)["\']?',  # Relative path
        r'["\']?([a-zA-Z]:[\\/][^"\'\s]+\.csv)["\']?',  # Windows path
    ]
    
    detected_path = None
    for pattern in path_patterns:
        match = re.search(pattern, full_message)
        if match:
            detected_path = match.group(1)
            break
    
    # Check 5: Check for state-based references
    mentions_data_source = any(phrase in last_message for phrase in [
        "data_source", "source location", "input data", "original data"
    ])
    mentions_output = any(phrase in last_message for phrase in [
        "output", "generated", "processed data", "result", "execution output"
    ])
    
    # Check 6: Check for uploaded CSV
    has_uploaded_csv = (
        state.get("uploaded_csv_preview") or 
        state.get("uploaded_csv_columns") or
        state.get("sample_data")
    )
    
    # Check 7: Detect dataset mentions in message (filename or alias)
    # Look for patterns like "Iris_info.csv", "run inference on Iris_info", etc.
    datasets_context = state.get("datasets_context") or state.get("multi_dataset_state")
    mentioned_dataset_id = None
    
    if datasets_context and isinstance(datasets_context, list):
        # Extract potential dataset names from message
        # Pattern: filename with extension (e.g., "Iris_info.csv") or just name (e.g., "Iris_info")
        # Also check for phrases like "on Iris_info.csv", "using Iris_info", etc.
        dataset_mention_patterns = [
            r'(?:on|using|from|with)\s+([^\s,\.]+\.csv)',  # "on Iris_info.csv"
            r'["\']([^\s,\.]+\.csv)["\']',  # Quoted filename
            r'\b([^\s,\.]+\.csv)\b',  # Standalone filename with extension
            r'(?:on|using|from|with)\s+([^\s,\.]+)(?:\s|$|,|\.)',  # "on Iris_info" (without extension)
        ]
        
        mentioned_names = set()
        for pattern in dataset_mention_patterns:
            matches = re.findall(pattern, full_message, re.IGNORECASE)
            for match in matches:
                # Normalize: remove quotes, trim whitespace, lowercase
                name = match.strip().strip('"\'')
                if name:
                    mentioned_names.add(name.lower())
                    # Also try without extension
                    if '.' in name:
                        name_no_ext = name.rsplit('.', 1)[0]
                        if name_no_ext:
                            mentioned_names.add(name_no_ext.lower())
        
        # Match against datasets in datasets_context
        for dataset in datasets_context:
            if not isinstance(dataset, dict):
                continue
            
            dataset_id = dataset.get("dataset_id")
            filename = dataset.get("filename", "")
            alias = dataset.get("alias", "")
            
            # Normalize for comparison
            filename_lower = filename.lower() if filename else ""
            alias_lower = alias.lower() if alias else ""
            filename_no_ext = filename_lower.rsplit('.', 1)[0] if '.' in filename_lower else filename_lower
            
            # Check if any mentioned name matches
            for mentioned_name in mentioned_names:
                if (mentioned_name == filename_lower or 
                    mentioned_name == alias_lower or 
                    mentioned_name == filename_no_ext or
                    (filename_lower and mentioned_name in filename_lower) or
                    (alias_lower and mentioned_name in alias_lower)):
                    mentioned_dataset_id = dataset_id
                    logger.info(
                        f"Detected dataset mention in message: '{mentioned_name}' "
                        f"matches dataset {dataset_id} (filename: {filename}, alias: {alias})"
                    )
                    break
            
            if mentioned_dataset_id:
                break
        
        # If dataset was mentioned, set active_dataset_id in state
        if mentioned_dataset_id:
            state["active_dataset_id"] = mentioned_dataset_id
            logger.info(f"Set active_dataset_id to {mentioned_dataset_id} based on message mention")
    
    # Determine source - prioritize direct input if detected
    if has_direct_input:
        # Return "direct_input" as the path_or_source
        # The actual string will be extracted in handle_inference_request
        return True, "direct_input", inference_mode
    
    elif detected_path:
        # Validate path exists
        if os.path.exists(detected_path) or os.path.exists(os.path.abspath(detected_path)):
            return True, detected_path, inference_mode
        else:
            # Path mentioned but doesn't exist - still return True but with warning
            return True, detected_path, inference_mode
    
    elif mentions_data_source and state.get("data_source_location"):
        return True, "data_source", inference_mode
    
    elif mentions_output and state.get("output_location"):
        return True, "output", inference_mode
    
    elif has_uploaded_csv:
        return True, "uploaded", inference_mode
    
    elif mentions_output and state.get("execution_output_data") is not None:
        return True, "execution_output", inference_mode
    
    # If inference was explicitly requested (has_keyword or has_direct_input was True)
    # but no specific data source was identified, check if we have any data available
    # for auto-detection in extract_csv_from_state
    # Note: has_keyword or has_direct_input must be True at this point (checked at line 217)
    
    # Check if we have any potential data sources available for auto-detection
    has_any_data_source = (
        has_uploaded_csv or  # Already checked above, but included for clarity
        state.get("data_source_location") or
        state.get("output_location") or
        state.get("execution_output_data") is not None or
        mentioned_dataset_id is not None  # Dataset was mentioned and found
    )
    
    if has_any_data_source:
        # We have inference keywords/direct input AND some data source available
        # Return True with None to allow auto-detection
        logger.info("Inference request detected with keywords/direct input, but no specific source identified. Will attempt auto-detection.")
        return True, None, inference_mode
    else:
        # Inference keywords/direct input detected, but no data sources available
        # This is an invalid request - return False
        logger.warning(
            "Inference request detected (keywords/direct input present) but no data sources available. "
            "Cannot proceed with inference without data."
        )
        return False, None, None


def _get_selected_dataset_data(state: ETLState, dataset_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Look up a dataset in datasets_context by dataset_id.
    
    Args:
        state: Current ETL state
        dataset_id: Dataset ID to look up. If None, uses active_dataset_id from state.
    
    Returns:
        Dataset metadata dict with keys like:
        - dataset_id, session_id, filename, alias
        - data_source_location, full_data_location, sample_data_location
        - preview, columns, schema
        Returns None if dataset not found.
    """
    if not dataset_id:
        dataset_id = state.get("active_dataset_id")
    
    if not dataset_id:
        return None
    
    # Check datasets_context (preferred) or multi_dataset_state (fallback)
    datasets_context = state.get("datasets_context") or state.get("multi_dataset_state")
    
    if not datasets_context or not isinstance(datasets_context, list):
        return None
    
    # Find dataset by dataset_id
    for dataset in datasets_context:
        if isinstance(dataset, dict) and dataset.get("dataset_id") == dataset_id:
            return dataset
    
    return None


def _load_dataset_from_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Load dataset data from session by session_id.
    
    This is a synchronous wrapper around async get_session().
    Handles both sync and async contexts (e.g., when called from FastAPI routes).
    
    Args:
        session_id: Session ID to load
    
    Returns:
        Session data dict if found, None otherwise
    """
    try:
        # Import here to avoid circular dependencies
        from app.services.session_service import get_session
        
        # Try to get the current event loop
        loop = None
        try:
            loop = asyncio.get_running_loop()
            # Loop is running - we're in an async context
            # Use run_coroutine_threadsafe to schedule the coroutine on the running loop
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(get_session(session_id), loop)
            # Wait for the result with a timeout
            try:
                session_data = future.result(timeout=10.0)  # 10 second timeout
                return session_data
            except concurrent.futures.TimeoutError:
                logger.error(f"Timeout loading session {session_id} from async context")
                return None
        except RuntimeError:
            # No running loop - we're in a sync context
            # Try to get or create an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Check if loop is running (shouldn't happen in sync context, but check anyway)
            if loop.is_running():
                # This shouldn't happen, but if it does, use run_coroutine_threadsafe
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(get_session(session_id), loop)
                try:
                    session_data = future.result(timeout=10.0)
                    return session_data
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout loading session {session_id}")
                    return None
            else:
                # Normal sync context - use run_until_complete
                session_data = loop.run_until_complete(get_session(session_id))
                return session_data
    except ImportError:
        logger.warning("session_service not available - cannot load dataset from session")
        return None
    except Exception as e:
        logger.warning(f"Failed to load session {session_id}: {e}")
        return None


def _enrich_selected_dataset_from_session(state: ETLState) -> bool:
    """
    Enrich selected dataset in state by loading missing data from session.
    
    If a selected dataset exists but its data_source_location doesn't exist,
    try to load the session and update the dataset with session data.
    
    Args:
        state: Current ETL state
    
    Returns:
        True if dataset was enriched, False otherwise
    """
    selected_dataset = _get_selected_dataset_data(state)
    if not selected_dataset:
        return False
    
    dataset_id = selected_dataset.get("dataset_id")
    session_id = selected_dataset.get("session_id")
    
    if not session_id:
        logger.debug(f"Selected dataset {dataset_id} has no session_id, skipping session lookup")
        return False
    
    # Check if data_source_location exists
    data_location = selected_dataset.get("data_source_location") or selected_dataset.get("full_data_location")
    if data_location and os.path.exists(data_location):
        logger.debug(f"Selected dataset {dataset_id} already has valid data_source_location: {data_location}")
        return False
    
    # Try to load from session
    logger.info(f"Selected dataset {dataset_id} missing data_source_location, loading from session {session_id}")
    session_data = _load_dataset_from_session(session_id)
    
    if not session_data:
        logger.warning(f"Could not load session {session_id} for dataset {dataset_id}")
        return False
    
    # Update datasets_context with session data
    datasets_context = state.get("datasets_context") or state.get("multi_dataset_state")
    if not datasets_context or not isinstance(datasets_context, list):
        return False
    
    # Find and update the dataset in datasets_context
    for i, dataset in enumerate(datasets_context):
        if isinstance(dataset, dict) and dataset.get("dataset_id") == dataset_id:
            # Update with session data (prioritize existing fields, add missing ones)
            updated_dataset = {**dataset}
            
            # Update data_source_location if missing
            if not updated_dataset.get("data_source_location") and session_data.get("data_source_location"):
                updated_dataset["data_source_location"] = session_data["data_source_location"]
                logger.info(f"Updated dataset {dataset_id} with data_source_location from session")
            
            # Update full_data_location if missing
            if not updated_dataset.get("full_data_location") and session_data.get("full_data_location"):
                updated_dataset["full_data_location"] = session_data["full_data_location"]
            
            # Update preview if missing
            if not updated_dataset.get("preview") and session_data.get("uploaded_csv_preview"):
                updated_dataset["preview"] = session_data["uploaded_csv_preview"]
            
            # Update columns if missing
            if not updated_dataset.get("columns") and session_data.get("uploaded_csv_columns"):
                updated_dataset["columns"] = session_data["uploaded_csv_columns"]
            
            # Update the dataset in the list
            datasets_context[i] = updated_dataset
            state["datasets_context"] = datasets_context
            
            logger.info(f"Successfully enriched dataset {dataset_id} from session {session_id}")
            return True
    
    return False


def extract_csv_from_state(state: ETLState, path_or_source: Optional[str] = None, 
                          direct_input_string: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Extract CSV data from state, local path, or direct input string.
    
    Args:
        state: ETLState
        path_or_source: 
            - "direct_input" - Parse from direct_input_string
            - Local file path (e.g., "/path/to/file.csv")
            - "uploaded" - Use uploaded CSV
            - "data_source" - Use data_source_location
            - "output" - Use output_location
            - "execution_output" - Use execution_output_data
            - None - Auto-detect
        direct_input_string: Raw input string for direct_input mode
    
    Returns:
        DataFrame if data found, None otherwise
    """
    # Priority 0: Selected dataset from datasets_context (if active_dataset_id is set)
    # Only check if path_or_source is None, "uploaded", or not explicitly set
    # This allows explicit paths to override dataset selection
    if (path_or_source is None or path_or_source == "uploaded"):
        selected_dataset = _get_selected_dataset_data(state)
        if selected_dataset:
            dataset_id = selected_dataset.get("dataset_id")
            logger.info(f"Found selected dataset: {dataset_id} (filename: {selected_dataset.get('filename')})")
            
            # Try to load from data_source_location first (most reliable)
            data_location = selected_dataset.get("data_source_location") or selected_dataset.get("full_data_location")
            if data_location and os.path.exists(data_location):
                try:
                    logger.info(f"Loading CSV from selected dataset data_source_location: {data_location}")
                    df = pd.read_csv(data_location)
                    return df
                except Exception as e:
                    logger.warning(f"Failed to load CSV from selected dataset location {data_location}: {e}")
            
            # Fallback: Try to use preview data from dataset context
            preview = selected_dataset.get("preview")
            columns = selected_dataset.get("columns")
            if preview and columns:
                try:
                    # Handle preview data similar to uploaded_csv_preview logic
                    if isinstance(preview, list) and len(preview) > 0:
                        # Check if first row is header
                        header_row = preview[0] if preview else []
                        header_from_preview = [str(val).strip() for val in header_row] if header_row else []
                        columns_str = [str(col).strip() for col in columns]
                        
                        start_idx = 1 if header_from_preview == columns_str else 0
                        
                        # Filter out header rows
                        header_normalized = [str(col).strip().lower() for col in columns]
                        data_rows = []
                        for row in preview[start_idx:]:
                            if not row:
                                continue
                            row_normalized = [str(val).strip().lower() for val in row]
                            if row_normalized != header_normalized:
                                data_rows.append(row)
                        
                        if data_rows:
                            df = pd.DataFrame(data_rows, columns=columns)
                            logger.info(f"Loaded CSV from selected dataset preview: {df.shape}")
                            return df
                except Exception as e:
                    logger.warning(f"Failed to create DataFrame from selected dataset preview: {e}")
            
            # If we couldn't load from selected dataset, continue to other priorities
            logger.info(f"Selected dataset {dataset_id} found but data not available, falling back to other sources")
    
    # Priority 1: Direct input string
    if path_or_source == "direct_input" and direct_input_string:
        logger.info(f"Parsing direct input string: {direct_input_string[:100]}...")
        return parse_direct_input(direct_input_string)
    
    # Priority 2: Explicit path provided
    if path_or_source and os.path.exists(path_or_source):
        try:
            logger.info(f"Loading CSV from explicit path: {path_or_source}")
            df = pd.read_csv(path_or_source)
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV from {path_or_source}: {e}")
            return None
    
    # Priority 2: Resolve path_or_source to actual path
    if path_or_source == "data_source" and state.get("data_source_location"):
        data_path = state["data_source_location"]
        if os.path.exists(data_path):
            try:
                logger.info(f"Loading CSV from data_source_location: {data_path}")
                df = pd.read_csv(data_path)
                return df
            except Exception as e:
                logger.error(f"Failed to load CSV from data_source_location: {e}")
                return None
    
    elif path_or_source == "output" and state.get("output_location"):
        output_path = state["output_location"]
        if os.path.exists(output_path):
            try:
                logger.info(f"Loading CSV from output_location: {output_path}")
                df = pd.read_csv(output_path)
                return df
            except Exception as e:
                logger.error(f"Failed to load CSV from output_location: {e}")
                return None
    
    elif path_or_source == "execution_output" and state.get("execution_output_data") is not None:
        data = state["execution_output_data"]
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, list):
            return pd.DataFrame(data)
    
    # Priority 3: Uploaded CSV (if no explicit source)
    if not path_or_source or path_or_source == "uploaded":
        # Check uploaded_csv_preview
        if state.get("uploaded_csv_preview"):
            preview = state["uploaded_csv_preview"]
            columns = state.get("uploaded_csv_columns", [])
            
            logger.info(
                f"Extracting CSV from uploaded_csv_preview: "
                f"preview_rows={len(preview) if preview else 0}, "
                f"columns={len(columns) if columns else 0}"
            )
            
            if columns and preview:
                # Use the first row as header if it matches columns, otherwise use columns directly
                # Sometimes uploaded_csv_preview[0] might not be the header
                header_row = preview[0] if preview else []
                
                # Check if first row matches the columns (it should be the header)
                header_from_preview = [str(val).strip() for val in header_row] if header_row else []
                columns_str = [str(col).strip() for col in columns]
                
                # If they don't match, the first row might already be data, not header
                if header_from_preview == columns_str:
                    # First row is the header, skip it
                    start_idx = 1
                    logger.debug("First row matches columns - treating as header row")
                else:
                    # First row might be data, or columns don't match - use columns as header
                    start_idx = 0
                    logger.warning(
                        f"First row doesn't match columns. "
                        f"Header from preview: {header_from_preview[:5]}, "
                        f"Columns: {columns_str[:5]}"
                    )
                
                # Normalize header for comparison (use columns as the canonical header)
                header_normalized = [str(col).strip().lower() for col in columns]
                
                # Filter out rows that match the header (duplicate headers)
                data_rows = []
                duplicate_count = 0
                skipped_rows = 0
                
                for i, row in enumerate(preview[start_idx:], start=start_idx):
                    if not row:  # Skip empty rows
                        skipped_rows += 1
                        continue
                    
                    # Normalize row for comparison: convert to strings, strip whitespace, lowercase
                    row_normalized = [str(val).strip().lower() for val in row]
                    
                    # Only include row if it doesn't match the header
                    if row_normalized != header_normalized:
                        data_rows.append(row)
                    else:
                        duplicate_count += 1
                        logger.debug(f"Filtered duplicate header at row {i}")
                
                if not data_rows:
                    logger.error(
                        f"No data rows found after filtering. "
                        f"Total preview rows: {len(preview)}, "
                        f"Start index: {start_idx}, "
                        f"Duplicate headers filtered: {duplicate_count}, "
                        f"Empty rows skipped: {skipped_rows}, "
                        f"Preview sample (first 3 rows): {preview[:3] if len(preview) >= 3 else preview}"
                    )
                    return None
                
                # Create DataFrame
                try:
                    df = pd.DataFrame(data_rows, columns=columns)
                except Exception as e:
                    logger.error(
                        f"Failed to create DataFrame from data_rows. "
                        f"Error: {e}, "
                        f"Columns: {columns}, "
                        f"Data rows count: {len(data_rows)}, "
                        f"First row length: {len(data_rows[0]) if data_rows else 0}"
                    )
                    return None
                
                # Final safety check: verify the DataFrame doesn't contain only header rows
                if len(df) > 0:
                    header_normalized_check = [str(col).strip().lower() for col in columns]
                    rows_that_are_headers = 0
                    for idx, row in df.iterrows():
                        row_normalized = [str(val).strip().lower() for val in row.values]
                        if row_normalized == header_normalized_check:
                            rows_that_are_headers += 1
                    
                    if rows_that_are_headers == len(df):
                        logger.error(
                            f"DataFrame created from uploaded_csv_preview contains only header rows. "
                            f"Preview had {len(preview)} rows, start_idx={start_idx}, "
                            f"filtered {duplicate_count} duplicates, "
                            f"created DataFrame with {len(df)} rows but all are headers. "
                            f"First few rows of preview: {preview[:min(5, len(preview))]}"
                        )
                        return None
                    elif rows_that_are_headers > 0:
                        logger.warning(
                            f"DataFrame contains {rows_that_are_headers} additional header rows "
                            f"that weren't filtered. Filtering them now."
                        )
                        # Filter out remaining header rows
                        mask = df.apply(
                            lambda row: [str(val).strip().lower() for val in row.values] != header_normalized_check,
                            axis=1
                        )
                        df = df[mask].reset_index(drop=True)
                        
                        if df.empty:
                            logger.error("After final filtering, DataFrame is empty")
                            return None
                
                logger.info(
                    f"Loaded CSV from uploaded_csv_preview: {df.shape} "
                    f"(filtered {duplicate_count} duplicate header rows from {len(preview)} total rows, "
                    f"start_idx={start_idx})"
                )
                return df
            else:
                logger.warning(
                    f"Cannot extract from uploaded_csv_preview: "
                    f"columns={columns is not None and len(columns) > 0}, "
                    f"preview={preview is not None and len(preview) > 0}"
                )
        
        # Check sample_data
        if state.get("sample_data"):
            try:
                sample_data = state["sample_data"]
                # Handle both string and list formats
                if isinstance(sample_data, list):
                    # If it's a list, convert to string (join rows with newlines)
                    if all(isinstance(row, str) for row in sample_data):
                        csv_string = "\n".join(sample_data)
                    else:
                        # If it's a list of lists, convert to CSV format
                        output = StringIO()
                        writer = csv.writer(output)
                        writer.writerows(sample_data)
                        csv_string = output.getvalue()
                elif isinstance(sample_data, str):
                    csv_string = sample_data
                else:
                    logger.warning(f"sample_data has unsupported type: {type(sample_data)}")
                    return None
                
                df = pd.read_csv(StringIO(csv_string))
                logger.info(f"Loaded CSV from sample_data: {df.shape}")
                return df
            except Exception as e:
                logger.warning(f"Failed to parse sample_data as CSV: {e}")
    
    # Priority 4: Auto-detect from state (fallback)
    # Try data_source_location
    if state.get("data_source_location"):
        data_path = state["data_source_location"]
        if os.path.exists(data_path) and data_path.endswith(('.csv', '.tsv')):
            try:
                logger.info(f"Auto-detected CSV from data_source_location: {data_path}")
                df = pd.read_csv(data_path)
                return df
            except Exception as e:
                logger.warning(f"Auto-detection failed for data_source_location: {e}")
    
    # Try output_location
    if state.get("output_location"):
        output_path = state["output_location"]
        if os.path.exists(output_path) and output_path.endswith(('.csv', '.tsv')):
            try:
                logger.info(f"Auto-detected CSV from output_location: {output_path}")
                df = pd.read_csv(output_path)
                return df
            except Exception as e:
                logger.warning(f"Auto-detection failed for output_location: {e}")
    
    return None


def validate_inference_path(file_path: str, allowed_directories: List[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate that the inference path is safe and accessible.
    
    Args:
        file_path: Path to validate
        allowed_directories: List of allowed base directories (for security)
    
    Returns:
        (is_valid, error_message)
    """
    # Resolve to absolute path
    abs_path = os.path.abspath(file_path)
    path_obj = Path(abs_path)
    
    # Check file exists
    if not path_obj.exists():
        return False, f"File not found: {abs_path}"
    
    # Check is file (not directory)
    if not path_obj.is_file():
        return False, f"Path is not a file: {abs_path}"
    
    # Check extension
    if path_obj.suffix.lower() not in ['.csv', '.tsv']:
        return False, f"File must be CSV or TSV, got: {path_obj.suffix}"
    
    # Security: Check if path is within allowed directories
    if allowed_directories:
        is_allowed = any(
            abs_path.startswith(os.path.abspath(allowed_dir))
            for allowed_dir in allowed_directories
        )
        if not is_allowed:
            return False, f"Path outside allowed directories: {abs_path}"
    
    # Check readable
    if not os.access(abs_path, os.R_OK):
        return False, f"File is not readable: {abs_path}"
    
    return True, None


class TemporaryInferenceManager:
    """Manages temporary inference workflow for unregistered models"""
    
    def __init__(self,
                 mlflow_manager: Optional[MLflowManager] = None,
                 blob_store = None,
                 k8s_namespace: str = None,
                 inference_image: str = None,
                 gcs_bucket: str = None,
                 gcs_prefix: str = None,
                 keep_jobs: bool = None):
        """
        Initialize Temporary Inference Manager.
        
        Inference mode (local vs k8s) is determined per-request based on user intent
        in the chat messages, not at initialization time.
        
        Args:
            mlflow_manager: MLflowManager instance
            blob_store: GCSBlobStore or S3BlobStore instance (for staging files)
            k8s_namespace: Kubernetes namespace (default: from env or 'default')
            inference_image: Docker image for inference (default: from env or 'avaloka-inference:latest')
            gcs_bucket: GCS bucket for staging (default: from env)
            gcs_prefix: GCS prefix for staging paths (default: 'tmp-inference')
            keep_jobs: If True, keep jobs after completion for debugging (default: from KEEP_INFERENCE_JOBS env or False)
        """
        self.mlflow_manager = mlflow_manager
        self.blob_store = blob_store
        
        # Kubernetes configuration
        self.k8s_namespace = k8s_namespace or os.getenv("K8S_NAMESPACE", "default")
        self.inference_image = inference_image or cloud_config.image(
            "INFERENCE_IMAGE", "avaloka-inference", "latest"
        )
        
        # GCS configuration
        # Priority: explicit parameter > GCS_INFERENCE_BUCKET > GCS_BUCKET (from settings)
        self.gcs_bucket = gcs_bucket or os.getenv("GCS_INFERENCE_BUCKET")
        if not self.gcs_bucket:
            # Try to get from settings if available
            try:
                from app.core.settings import Settings
                settings = Settings()
                self.gcs_bucket = settings.gcs_bucket
            except Exception:
                pass
        
        # Job cleanup configuration
        # Priority: explicit parameter > KEEP_INFERENCE_JOBS env > False (default: delete jobs)
        if keep_jobs is None:
            keep_jobs_env = os.getenv("KEEP_INFERENCE_JOBS", "").lower()
            self.keep_jobs = keep_jobs_env in ("true", "1", "yes")
        else:
            self.keep_jobs = keep_jobs
        
        self.gcs_prefix = gcs_prefix or os.getenv("GCS_INFERENCE_PREFIX", "tmp-inference")
        
        # Initialize both inference managers (mode determined per-request based on user intent)
        self.k8s_manager = None
        self.local_inference = None
        
        # Initialize Local Inference Manager (if available)
        if LOCAL_INFERENCE_AVAILABLE:
            try:
                self.local_inference = LocalInferenceManager(mlflow_manager=self.mlflow_manager)
                logger.info("Local Inference Manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Local Inference Manager: {e}. Local inference will not be available.")
                self.local_inference = None
        else:
            logger.info("Local Inference Manager not available (LOCAL_INFERENCE_AVAILABLE=False)")
        
        # Initialize Kubernetes Job Manager (if available)
        # Validate storage configuration for Kubernetes mode
        if not self.blob_store:
            logger.warning("blob_store not provided - Kubernetes inference will require blob_store to be set later")
        if not self.gcs_bucket:
            logger.warning("GCS bucket not configured - Kubernetes inference staging may fail. Set GCS_INFERENCE_BUCKET or GCS_BUCKET environment variable.")
        
        try:
            self.k8s_manager = KubernetesJobManager(namespace=self.k8s_namespace)
            logger.info("Kubernetes Job Manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Kubernetes Job Manager: {e}. Kubernetes inference will not be available.")
            self.k8s_manager = None
    
    def stage_input_csv(self, df: pd.DataFrame, run_id: str) -> str:
        """
        Stage input CSV to GCS for Kubernetes Job.
        
        Args:
            df: DataFrame to stage
            run_id: MLflow run ID (for organizing paths)
        
        Returns:
            GCS path to staged CSV (always a full gs:// URI)
        """
        if not self.blob_store:
            raise ValueError("blob_store not configured - cannot stage CSV to GCS")
        
        # Pre-extract bucket name from blob_store for use in fallback scenarios
        # This ensures we always have the bucket name available
        self._cached_bucket_name = None
        blob_store_bucket_obj = getattr(self.blob_store, '_bucket', None)
        if blob_store_bucket_obj:
            if hasattr(blob_store_bucket_obj, 'name'):
                self._cached_bucket_name = blob_store_bucket_obj.name
            elif isinstance(blob_store_bucket_obj, str):
                self._cached_bucket_name = blob_store_bucket_obj
        
        # Validate DataFrame
        if df is None or df.empty:
            raise ValueError("Cannot stage empty DataFrame")
        
        # Check if DataFrame contains only header-like rows (all rows match column names)
        # This can happen if uploaded_csv_preview had duplicate headers
        column_names = list(df.columns)
        if len(df) > 0:
            # Normalize column names for comparison (lowercase, strip whitespace)
            column_names_normalized = [str(col).strip().lower() for col in column_names]
            
            # Check if any row exactly matches the column names
            rows_matching_headers = 0
            for idx, row in df.iterrows():
                # Normalize row values for comparison (convert to string, strip, lowercase)
                row_values_normalized = [str(val).strip().lower() for val in row.values]
                if row_values_normalized == column_names_normalized:
                    rows_matching_headers += 1
            
            if rows_matching_headers == len(df):
                # Provide helpful error message with suggestions
                error_msg = (
                    f"DataFrame contains only header rows (no actual data rows found). "
                    f"This usually indicates the uploaded CSV file had duplicate headers or no data rows. "
                    f"\n\nTroubleshooting:"
                    f"\n1. Check your CSV file - it should have a header row followed by data rows"
                    f"\n2. Ensure your CSV file has actual data, not just column headers"
                    f"\n3. Try uploading the file again or use a file path instead"
                    f"\n4. If using uploaded_csv_preview, ensure it contains data rows after the header"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            elif rows_matching_headers > 0:
                logger.warning(
                    f"DataFrame contains {rows_matching_headers} rows that match headers. "
                    f"These will be filtered out before uploading."
                )
                # Filter out rows that match headers (using normalized comparison)
                def row_does_not_match_header(row):
                    row_values_normalized = [str(val).strip().lower() for val in row.values]
                    return row_values_normalized != column_names_normalized
                
                mask = df.apply(row_does_not_match_header, axis=1)
                df = df[mask].reset_index(drop=True)
                
                if df.empty:
                    error_msg = (
                        "After filtering header rows, DataFrame is empty. "
                        "This means all rows in your CSV matched the header row. "
                        "Please check your input CSV file and ensure it contains actual data rows."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{run_id}_{timestamp}_input.csv"
        
        # Check if blob_store has a prefix - if so, don't include gcs_prefix in object_name
        # If blob_store doesn't have a prefix, include gcs_prefix in object_name
        # This mirrors the logic in create_inference_job to ensure consistency
        blob_prefix = getattr(self.blob_store, '_prefix', '') if self.blob_store else ''
        if blob_prefix:
            # Blob store has prefix, don't add gcs_prefix again
            object_name = f"inputs/{run_id}/{filename}"
        else:
            # Blob store has no prefix, include gcs_prefix
            object_name = f"{self.gcs_prefix}/inputs/{run_id}/{filename}" if self.gcs_prefix else f"inputs/{run_id}/{filename}"
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
            df.to_csv(temp_path, index=False)
        
        try:
            # Upload to GCS
            blob_prefix = getattr(self.blob_store, '_prefix', '') if self.blob_store else ''
            blob_store_type = type(self.blob_store).__name__ if self.blob_store else 'None'
            logger.info(
                f"Staging input CSV to GCS: object_name='{object_name}', "
                f"blob_store prefix='{blob_prefix}', "
                f"blob_store type={blob_store_type}"
            )
            
            # CRITICAL: LocalBlobStore stores files locally, not in GCS
            # For inference jobs, we MUST upload to GCS so the pod can access it
            # If LocalBlobStore is being used, upload directly to GCS
            if blob_store_type == 'LocalBlobStore':
                logger.warning(
                    f"LocalBlobStore detected - files are stored locally, not in GCS. "
                    f"For inference jobs, we need GCS access. Uploading directly to GCS..."
                )
                
                # Upload directly to GCS using the GCS client
                gcs_bucket = (
                    getattr(self, '_cached_bucket_name', None) or
                    self.gcs_bucket or 
                    os.getenv("GCS_BUCKET") or 
                    os.getenv("GCS_INFERENCE_BUCKET")
                )
                
                if not gcs_bucket:
                    raise ValueError(
                        f"Cannot upload to GCS: bucket name is unknown. "
                        f"Cached bucket: {getattr(self, '_cached_bucket_name', None)}, "
                        f"self.gcs_bucket: {self.gcs_bucket}, "
                        f"env GCS_BUCKET: {os.getenv('GCS_BUCKET')}, "
                        f"env GCS_INFERENCE_BUCKET: {os.getenv('GCS_INFERENCE_BUCKET')}. "
                        f"Please configure GCS_BUCKET or GCS_INFERENCE_BUCKET environment variable."
                    )
                
                # Construct the GCS key (with prefix if blob_store has one)
                if blob_prefix:
                    gcs_key = f"{blob_prefix}/{object_name}" if not object_name.startswith(blob_prefix + "/") else object_name
                else:
                    gcs_key = object_name
                
                # Upload directly to GCS
                from google.cloud import storage
                storage_client = storage.Client()
                bucket = storage_client.bucket(gcs_bucket)
                blob = bucket.blob(gcs_key)
                blob.upload_from_filename(temp_path)
                
                # Construct full GCS URI
                gcs_path = f"gs://{gcs_bucket}/{gcs_key}"
                logger.info(
                    f"✓ Input CSV uploaded directly to GCS: {gcs_path} "
                    f"(bypassed LocalBlobStore)"
                )
            else:
                # Use blob_store.put_file (should work for GCSBlobStore, S3BlobStore, etc.)
                # put_file should return the full GCS URI: gs://bucket/prefix/object_name (if prefix exists)
                # or gs://bucket/object_name (if no prefix)
                gcs_path = self.blob_store.put_file(Path(temp_path), object_name)
                logger.info(
                    f"✓ Input CSV staged: put_file returned '{gcs_path}' "
                    f"(type: {type(gcs_path)}, starts with gs://: {gcs_path.startswith('gs://') if isinstance(gcs_path, str) else 'N/A'})"
                )
            
            # Validate return value
            if not isinstance(gcs_path, str):
                raise ValueError(f"blob_store.put_file returned non-string: {type(gcs_path)}")
            
            # GCSBlobStore.put_file should always return a full GCS URI (gs://bucket/key)
            # where key = _key(object_name) = prefix/object_name if prefix exists, else object_name
            # If it doesn't return gs://, something is wrong - but handle it gracefully
            if not gcs_path.startswith("gs://"):
                logger.error(
                    f"CRITICAL: blob_store.put_file returned path without gs:// prefix: '{gcs_path}'. "
                    f"This should not happen with GCSBlobStore. "
                    f"Blob store type: {type(self.blob_store).__name__}, "
                    f"blob_store._prefix: '{blob_prefix}'. "
                    f"object_name passed to put_file: '{object_name}'. "
                    f"Attempting to construct full URI..."
                )
                
                # Emergency fallback: construct full URI
                # CRITICAL: The actual object in GCS is at bucket/prefix/object_name (if prefix exists)
                # put_file's _key() method does: prefix/object_name if prefix exists, else object_name
                # So we MUST include the prefix when reconstructing the path
                gcs_bucket = (
                    getattr(self, '_cached_bucket_name', None) or
                    self.gcs_bucket or 
                    os.getenv("GCS_BUCKET") or 
                    os.getenv("GCS_INFERENCE_BUCKET")
                )
                
                if not gcs_bucket:
                    error_msg = (
                        f"Cannot construct full GCS URI: blob_store.put_file returned '{gcs_path}' "
                        f"but bucket name is unknown. "
                        f"Cached bucket: {getattr(self, '_cached_bucket_name', None)}, "
                        f"self.gcs_bucket: {self.gcs_bucket}, "
                        f"env GCS_BUCKET: {os.getenv('GCS_BUCKET')}, "
                        f"env GCS_INFERENCE_BUCKET: {os.getenv('GCS_INFERENCE_BUCKET')}. "
                        f"Please configure GCS_BUCKET or ensure blob_store is properly initialized."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Reconstruct the path that was actually uploaded
                # CRITICAL: put_file's _key() method does: prefix/object_name if prefix exists, else object_name
                # So the actual object in GCS is ALWAYS at: bucket/prefix/object_name (if prefix exists)
                # or bucket/object_name (if no prefix)
                # 
                # If put_file returned a relative path (without gs://), it might be:
                # 1. Just object_name (missing prefix) - we need to add prefix
                # 2. prefix/object_name (has prefix) - we can use as-is
                # 3. Some other format
                #
                # The safest approach: if blob_prefix exists, ALWAYS prepend it unless the path already starts with it
                if blob_prefix:
                    # Check if the returned path already includes the prefix
                    if gcs_path.startswith(blob_prefix + "/"):
                        # Path already includes prefix - use as-is
                        full_gcs_path = f"gs://{gcs_bucket}/{gcs_path.lstrip('/')}"
                        logger.info(f"Path already includes prefix '{blob_prefix}', using as-is: {full_gcs_path}")
                    else:
                        # Path doesn't include prefix - add it (this is what _key() does)
                        # This is the most common case when put_file returns just object_name
                        full_gcs_path = f"gs://{gcs_bucket}/{blob_prefix}/{gcs_path.lstrip('/')}"
                        logger.info(
                            f"Adding prefix '{blob_prefix}' to path '{gcs_path}': {full_gcs_path} "
                            f"(object_name was: '{object_name}')"
                        )
                else:
                    # No prefix, use path as-is
                    full_gcs_path = f"gs://{gcs_bucket}/{gcs_path.lstrip('/')}"
                    logger.info(f"No prefix, using path as-is: {full_gcs_path}")
                
                logger.warning(
                    f"Constructed full GCS URI: '{full_gcs_path}' "
                    f"(blob_prefix: '{blob_prefix}', object_name: '{object_name}', "
                    f"put_file returned: '{gcs_path}')"
                )
                return full_gcs_path
            
            # put_file returned a full GCS URI - use it directly (this is the correct path)
            # CRITICAL: put_file's return value includes the prefix if it exists, so we MUST use it as-is
            # The path format is: gs://bucket/prefix/object_name (if prefix exists)
            # or gs://bucket/object_name (if no prefix)
            # This is the exact path where the object was uploaded, so use it directly
            
            # Verify the bucket matches what we expect (for logging/debugging)
            path_bucket = gcs_path.replace("gs://", "").split("/")[0]
            expected_bucket = getattr(self, '_cached_bucket_name', None) or self.gcs_bucket
            if expected_bucket and path_bucket != expected_bucket:
                logger.warning(
                    f"Bucket mismatch: path says '{path_bucket}' but expected '{expected_bucket}'. "
                    f"Using path bucket from put_file return value: {path_bucket}"
                )
            
            # Verify the path includes the prefix if blob_store has one
            # This is just for validation - the path from put_file should already be correct
            # Ensure blob_prefix is a string (handles mock objects in tests)
            blob_prefix_str = str(blob_prefix) if blob_prefix else ''
            if blob_prefix_str:
                path_after_bucket = gcs_path.replace(f"gs://{path_bucket}/", "")
                if not path_after_bucket.startswith(blob_prefix_str + "/"):
                    logger.warning(
                        f"WARNING: GCS path from put_file doesn't start with blob_store prefix! "
                        f"Path: '{gcs_path}', expected prefix: '{blob_prefix_str}'. "
                        f"This might cause 404 errors when downloading. "
                        f"Using path as-is from put_file, but this may be incorrect."
                    )
                else:
                    logger.info(
                        f"✓ GCS path includes blob_store prefix correctly: '{gcs_path}'"
                    )
            
            logger.info(f"✓ Returning GCS path from put_file: {gcs_path}")
            return gcs_path
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def retrieve_output_csv(self, gcs_path: str) -> pd.DataFrame:
        """
        Retrieve output CSV from GCS after inference.
        
        Args:
            gcs_path: GCS path to output CSV (should be a full gs:// URI)
        
        Returns:
            DataFrame with predictions
        """
        if not gcs_path:
            raise ValueError("gcs_path is required")
        
        blob_store_type = type(self.blob_store).__name__ if self.blob_store else 'None'
        
        # Parse GCS path to get bucket and object name
        if not gcs_path.startswith("gs://"):
            raise ValueError(
                f"Invalid GCS path: '{gcs_path}'. Expected format: 'gs://bucket/path/to/file.csv'. "
                f"If using LocalBlobStore, the output should still be in GCS for inference jobs."
            )
        
        # Extract bucket and object name from GCS URI
        path_parts = gcs_path.replace("gs://", "").split("/", 1)
        if len(path_parts) < 2:
            raise ValueError(f"Invalid GCS path: '{gcs_path}'. Missing object path after bucket name.")
        
        bucket_name = path_parts[0]
        object_name = path_parts[1]
        
        # Download to temp file
        fd, temp_path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)  # Close the file descriptor, we only need the path
        
        try:
            # CRITICAL: If LocalBlobStore is being used, download directly from GCS
            # because the file is in GCS (uploaded directly), not in local filesystem
            if blob_store_type == 'LocalBlobStore':
                logger.info(
                    f"LocalBlobStore detected - downloading output CSV directly from GCS: {gcs_path}"
                )
                
                # Download directly from GCS using the GCS client
                from google.cloud import storage
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(object_name)
                blob.download_to_filename(temp_path)
                
                logger.info(f"✓ Downloaded output CSV from GCS: {gcs_path}")
            else:
                # Use blob_store.get_file() for GCSBlobStore, S3BlobStore, etc.
                # If blob_store has a prefix, we need to remove it from the object_name
                # because blob_store.get_file() will add the prefix again
                blob_prefix = getattr(self.blob_store, '_prefix', '') if self.blob_store else ''
                # Ensure blob_prefix is a string (handles mock objects in tests)
                blob_prefix_str = str(blob_prefix) if blob_prefix else ''
                if blob_prefix_str and object_name.startswith(blob_prefix_str + "/"):
                    # Remove the prefix since blob_store will add it
                    blob_store_object_name = object_name[len(blob_prefix_str) + 1:]
                else:
                    blob_store_object_name = object_name
                
                logger.info(
                    f"Retrieving output CSV from blob_store: {blob_store_object_name} "
                    f"(full GCS path: {gcs_path}, blob_store prefix: '{blob_prefix_str}')"
                )
                
                if not self.blob_store:
                    raise ValueError("blob_store not configured - cannot retrieve CSV")
                
                self.blob_store.get_file(blob_store_object_name, Path(temp_path))
                logger.info(f"✓ Retrieved output CSV from blob_store: {gcs_path}")
            
            # Load CSV
            df = pd.read_csv(temp_path)
            logger.info(f"✓ Loaded output CSV with shape: {df.shape}")
            return df
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def create_inference_job(self,
                            run_id: str,
                            model_uri: str,
                            input_gcs_path: str,
                            mlflow_tracking_uri: str,
                            metadata_path: str = None,
                            service_account: Optional[str] = _USE_DEFAULT_SERVICE_ACCOUNT,
                            gcs_bucket: Optional[str] = None) -> Dict[str, Any]:
        """
        Create Kubernetes Job for inference.
        
        Args:
            run_id: MLflow run ID
            model_uri: MLflow model URI (e.g., "runs:/run_42/model")
            input_gcs_path: GCS path to input CSV
            mlflow_tracking_uri: MLflow tracking server URI
            metadata_path: Optional path to metadata JSON file
        
        Returns:
            Job creation result dictionary
        """
        if not self.k8s_manager:
            raise ValueError("Kubernetes Job Manager not initialized")
        
        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{run_id}_{timestamp}_output.csv"
        
        # Check if blob_store has a prefix - if so, don't include gcs_prefix in object_name
        # If blob_store doesn't have a prefix, include gcs_prefix in object_name
        blob_prefix = getattr(self.blob_store, '_prefix', '') if self.blob_store else ''
        if blob_prefix:
            # Blob store has prefix, don't add gcs_prefix again
            output_object_name = f"outputs/{run_id}/{output_filename}"
            # Construct full path for return (blob_store will add prefix when uploading)
            full_object_name = f"{blob_prefix}/{output_object_name}"
        else:
            # Blob store has no prefix, include gcs_prefix
            output_object_name = f"{self.gcs_prefix}/outputs/{run_id}/{output_filename}" if self.gcs_prefix else f"outputs/{run_id}/{output_filename}"
            full_object_name = output_object_name
        
        # Construct GCS output path
        if self.gcs_bucket:
            output_gcs_path = f"gs://{self.gcs_bucket}/{full_object_name}"
        else:
            # Extract bucket from input path
            if input_gcs_path.startswith("gs://"):
                bucket = input_gcs_path.replace("gs://", "").split("/")[0]
                # Use full_object_name (not output_object_name) to include blob_prefix if it exists
                # This ensures consistency with the case when self.gcs_bucket is set
                output_gcs_path = f"gs://{bucket}/{full_object_name}"
            else:
                raise ValueError("Cannot determine GCS bucket for output path")
        
        # Generate unique job name
        job_name = f"temp-inference-{run_id}-{timestamp}".replace("_", "-").lower()
        # Kubernetes names must be lowercase and contain only alphanumeric and hyphens
        job_name = re.sub(r'[^a-z0-9-]', '-', job_name)
        job_name = job_name[:63]  # Kubernetes name length limit
        
        # Get service account for GCS access (if configured)
        # If service_account is the sentinel value, it means "explicitly use default, don't check env vars"
        # If service_account is None (from old API), check env vars for backward compatibility
        # If service_account is a string, use it directly
        if service_account is _USE_DEFAULT_SERVICE_ACCOUNT:
            # Explicitly requested to use default - don't check environment variables
            service_account = None
        elif service_account is None:
            # None was passed (for backward compatibility) - check environment variables
            service_account = (
                os.getenv("GCS_WRITER") or  # Check for gcs-writer first
                os.getenv("K8S_SERVICE_ACCOUNT") or 
                os.getenv("INFERENCE_SERVICE_ACCOUNT")
            )
        # If service_account is a string, use it as-is
        
        # If service account is specified, verify it exists before using it
        if service_account:
            try:
                # Check if service account exists in the namespace
                self.k8s_manager.core_api.read_namespaced_service_account(
                    name=service_account,
                    namespace=self.k8s_manager.namespace
                )
                logger.info(f"Using service account for inference job: {service_account}")
            except Exception as e:
                logger.warning(
                    f"Service account '{service_account}' not found in namespace '{self.k8s_manager.namespace}': {e}. "
                    f"Will use default service account instead. "
                    f"To use a custom service account, create it first: "
                    f"kubectl create serviceaccount {service_account} -n {self.k8s_manager.namespace}"
                )
                service_account = None  # Use default instead
        else:
            logger.info(
                "No service account specified. Pod will use default service account for namespace. "
                "Ensure it has GCS permissions (Storage Object Viewer/Creator roles) or Workload Identity configured."
            )
        
        # Create job
        logger.info(
            f"Creating Kubernetes Job: {job_name} "
            f"(input_path={input_gcs_path}, gcs_bucket={gcs_bucket})"
        )
        
        # CRITICAL: Verify gcs_bucket is provided - it's required for the pod to construct full GCS URIs
        if not gcs_bucket:
            logger.error(
                f"CRITICAL: gcs_bucket is None when creating inference job! "
                f"This will cause the pod to fail when trying to load CSV. "
                f"input_gcs_path: {input_gcs_path}"
            )
            # Try to extract from input_path as last resort
            if input_gcs_path.startswith("gs://"):
                gcs_bucket = input_gcs_path.replace("gs://", "").split("/")[0]
                logger.warning(f"Extracted bucket from input_path as fallback: {gcs_bucket}")
            else:
                # Try environment variables
                gcs_bucket = os.getenv("GCS_BUCKET") or os.getenv("GCS_INFERENCE_BUCKET")
                if gcs_bucket:
                    logger.warning(f"Using GCS_BUCKET from environment as fallback: {gcs_bucket}")
                else:
                    raise ValueError(
                        f"Cannot create inference job: gcs_bucket is required but not provided. "
                        f"input_gcs_path: {input_gcs_path}. "
                        f"Please ensure gcs_bucket parameter is passed or GCS_BUCKET environment variable is set."
                    )
        
        result = self.k8s_manager.create_inference_job(
            job_name=job_name,
            image=self.inference_image,
            model_uri=model_uri,
            input_path=input_gcs_path,
            output_path=output_gcs_path,
            mlflow_tracking_uri=mlflow_tracking_uri,
            metadata_path=metadata_path,
            service_account=service_account,
            gcs_bucket=gcs_bucket  # Pass bucket name to ensure it's available in the pod
        )
        
        result["output_gcs_path"] = output_gcs_path
        return result
    
    def format_predictions_message(self, df: pd.DataFrame, job_status: Dict[str, Any] = None) -> str:
        """
        Format prediction results for chat display.
        
        Args:
            df: DataFrame with predictions
            job_status: Optional job status information
        
        Returns:
            Formatted message string
        """
        message_parts = [
            "**Inference Results:**",
            f"- Total samples: {len(df)}",
        ]
        
        # Show prediction columns
        if "Predicted" in df.columns:
            message_parts.append(f"- Predictions column: Predicted")
            
            # Show class distribution if classification
            if df["Predicted"].dtype == 'object':
                class_dist = df["Predicted"].value_counts().to_dict()
                message_parts.append(f"- Class distribution:")
                for class_name, count in class_dist.items():
                    percentage = (count / len(df)) * 100
                    message_parts.append(f"  - {class_name}: {count} ({percentage:.1f}%)")
        
        if "Confidence" in df.columns:
            avg_confidence = df["Confidence"].mean()
            message_parts.append(f"- Average confidence: {avg_confidence:.3f}")
        
        # Show sample predictions
        message_parts.append("\n**Sample Predictions:**")
        sample_size = min(10, len(df))
        sample_df = df[["Predicted", "Confidence"]].head(sample_size) if "Predicted" in df.columns and "Confidence" in df.columns else df.head(sample_size)
        
        # Format as table
        message_parts.append(sample_df.to_string(index=False))
        
        if len(df) > sample_size:
            message_parts.append(f"\n... and {len(df) - sample_size} more predictions")
        
        # Add job info if available
        if job_status:
            message_parts.append(f"\n**Job Status:** {job_status.get('status', 'Unknown')}")
            if job_status.get('completion_time'):
                message_parts.append(f"- Completed at: {job_status['completion_time']}")
        
        return "\n".join(message_parts)
    
    def handle_inference_request(self, 
                                state: ETLState,
                                messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        Main entry point for handling inference requests.
        
        Args:
            state: Current ETL state
            messages: List of chat messages
        
        Returns:
            Result dictionary with success status and message
        """
        try:
            # Detect request
            is_request, path_or_source, inference_mode = detect_inference_request(messages, state)
            
            # Log detected inference mode for debugging
            if is_request:
                logger.info(f"Detected inference request: path_or_source={path_or_source}, inference_mode={inference_mode}")
            
            if not is_request:
                return {
                    "success": False,
                    "error": "No inference request detected",
                    "message": None
                }
            
            # Extract direct input string if needed
            direct_input_string = None
            if path_or_source == "direct_input" and messages:
                # Extract the input string from the last message
                full_message = messages[-1].content.strip()
                
                # Try to extract quoted content first (e.g., "SepalLengthCm 5.1 ...")
                # Match opening and closing quotes of the same type to avoid mismatched quotes
                # Try double quotes first, then single quotes
                quoted_match = re.search(r'"([^"]+)"', full_message) or re.search(r"'([^']+)'", full_message)
                if quoted_match:
                    direct_input_string = quoted_match.group(1)
                    logger.info(f"Extracted direct input from quoted string: {direct_input_string[:100]}...")
                else:
                    # Extract the direct input pattern from the message
                    direct_input_pattern = r'\b\w+\s+[\d.]+(?:\s+\w+\s+[\d.]+){1,}'
                    match = re.search(direct_input_pattern, full_message)
                    if match:
                        direct_input_string = match.group(0)
                        logger.info(f"Extracted direct input from message: {direct_input_string[:100]}...")
                    else:
                        # Fallback: use the whole message (might contain inference keywords)
                        # Remove common inference keywords to get just the data
                        cleaned = full_message
                        for keyword in ["run inference on", "predict", "inference on"]:
                            cleaned = re.sub(keyword, "", cleaned, flags=re.IGNORECASE).strip()
                        # Remove quotes if present
                        cleaned = cleaned.strip('"\'')
                        direct_input_string = cleaned
                        logger.info(f"Using cleaned message as direct input: {direct_input_string[:100]}...")
            
            # Step 3: Enrich selected dataset from session if needed (before extraction)
            # This ensures selected datasets have their data_source_location available
            if path_or_source is None or path_or_source == "uploaded":
                _enrich_selected_dataset_from_session(state)
            
            # Extract CSV or parse direct input
            logger.info(f"Extracting data from state, path_or_source: {path_or_source}")
            df = extract_csv_from_state(state, path_or_source, direct_input_string=direct_input_string)
            
            # Helper function to check if DataFrame contains only headers
            def df_has_only_headers(df_check):
                """Check if DataFrame contains only header rows"""
                if df_check is None or df_check.empty:
                    return False
                column_names = list(df_check.columns)
                column_names_normalized = [str(col).strip().lower() for col in column_names]
                rows_that_are_headers = 0
                for idx, row in df_check.iterrows():
                    row_values_normalized = [str(val).strip().lower() for val in row.values]
                    if row_values_normalized == column_names_normalized:
                        rows_that_are_headers += 1
                return rows_that_are_headers == len(df_check)
            
            # Check if DataFrame has only headers (this is also a failure case)
            if df is not None and not df.empty and df_has_only_headers(df):
                logger.warning(
                    f"DataFrame extracted from {path_or_source} contains only header rows. "
                    f"This is invalid. Will try fallback methods..."
                )
                df = None  # Treat as failure to trigger fallbacks
            
            # If extraction from uploaded_csv_preview failed, try multiple fallbacks
            if (df is None or df.empty) and path_or_source == "uploaded":
                logger.warning(
                    "Extraction from uploaded_csv_preview failed or returned empty. "
                    "This might indicate all rows in preview are headers or preview is malformed. "
                    "Trying fallback methods..."
                )
                
                # Fallback 1: Try data_source_location (actual file path) - read directly
                if state.get("data_source_location"):
                    data_path = state["data_source_location"]
                    if os.path.exists(data_path):
                        logger.info(f"Fallback 1: Reading directly from data_source_location: {data_path}")
                        try:
                            # Read directly from file, bypassing extract_csv_from_state
                            df_fallback = pd.read_csv(data_path)
                            logger.info(f"  Read {len(df_fallback)} rows from file")
                            
                            # Check if fallback result is valid (not only headers)
                            if df_fallback is not None and not df_fallback.empty and not df_has_only_headers(df_fallback):
                                df = df_fallback
                                logger.info(f"✓ Successfully extracted from data_source_location: {df.shape}")
                            else:
                                logger.warning(
                                    f"Fallback 1 failed: File read returned DataFrame with only headers. "
                                    f"Shape: {df_fallback.shape if df_fallback is not None else 'None'}"
                                )
                        except Exception as e:
                            logger.error(f"Fallback 1 failed: Error reading file {data_path}: {e}")
                    else:
                        logger.warning(f"Fallback 1 skipped: data_source_location file does not exist: {data_path}")
                
                # Fallback 2: Try sample_data
                if (df is None or df.empty or df_has_only_headers(df)) and state.get("sample_data"):
                    logger.info("Fallback 2: Attempting to extract from sample_data")
                    df_fallback = extract_csv_from_state(state, None)  # Try auto-detect which might use sample_data
                    # Check if fallback result is valid (not only headers)
                    if df_fallback is not None and not df_fallback.empty and not df_has_only_headers(df_fallback):
                        df = df_fallback
                        logger.info(f"✓ Successfully extracted from sample_data: {df.shape}")
                    else:
                        logger.warning(f"Fallback 2 failed: sample_data returned invalid DataFrame")
                
                # Fallback 3: Try output_location
                if (df is None or df.empty or df_has_only_headers(df)) and state.get("output_location"):
                    output_path = state["output_location"]
                    if os.path.exists(output_path):
                        logger.info(f"Fallback 3: Trying to extract from output_location: {output_path}")
                        df_fallback = extract_csv_from_state(state, output_path)
                        # Check if fallback result is valid (not only headers)
                        if df_fallback is not None and not df_fallback.empty and not df_has_only_headers(df_fallback):
                            df = df_fallback
                            logger.info(f"✓ Successfully extracted from output_location: {df.shape}")
                        else:
                            logger.warning(f"Fallback 3 failed: output_location returned invalid DataFrame")
                
                # Final check after fallbacks
                if df is not None and not df.empty and df_has_only_headers(df):
                    logger.error("All fallback methods returned DataFrames with only headers. This is a critical error.")
                    df = None  # Set to None to trigger error handling
            
            # Final validation: ensure DataFrame is valid (not None, not empty, and not only headers)
            if df is None or df.empty or df_has_only_headers(df):
                # Check if a selected dataset was requested but failed
                active_dataset_id = state.get("active_dataset_id")
                if active_dataset_id:
                    selected_dataset = _get_selected_dataset_data(state, active_dataset_id)
                    datasets_context = state.get("datasets_context") or state.get("multi_dataset_state")
                    
                    if not selected_dataset:
                        # Selected dataset not found - list available datasets
                        available_datasets = []
                        if datasets_context and isinstance(datasets_context, list):
                            for dataset in datasets_context:
                                if isinstance(dataset, dict):
                                    dataset_id = dataset.get("dataset_id", "unknown")
                                    filename = dataset.get("filename", "unknown")
                                    alias = dataset.get("alias", "")
                                    available_datasets.append({
                                        "dataset_id": dataset_id,
                                        "filename": filename,
                                        "alias": alias
                                    })
                        
                        error_msg = (
                            f"Selected dataset '{active_dataset_id}' not found in available datasets.\n\n"
                        )
                        
                        if available_datasets:
                            error_msg += "**Available datasets:**\n"
                            for ds in available_datasets:
                                display_name = ds["alias"] or ds["filename"] or ds["dataset_id"]
                                error_msg += f"- {display_name} (ID: {ds['dataset_id']})\n"
                            error_msg += (
                                "\nYou can specify a dataset by:\n"
                                "- Mentioning the filename in your message (e.g., 'run inference on Iris_info.csv')\n"
                                "- Using the dataset alias\n"
                                "- Or use the dataset ID"
                            )
                        else:
                            error_msg += "No datasets are currently available. Please upload a CSV file first."
                        
                        return {
                            "success": False,
                            "error": f"Selected dataset '{active_dataset_id}' not found",
                            "message": error_msg
                        }
                    else:
                        # Selected dataset exists but data extraction failed
                        filename = selected_dataset.get("filename", "unknown")
                        alias = selected_dataset.get("alias", "")
                        display_name = alias or filename or active_dataset_id
                        
                        error_msg = (
                            f"Could not extract data from selected dataset '{display_name}' (ID: {active_dataset_id}).\n\n"
                        )
                        
                        # Check what's missing
                        data_location = selected_dataset.get("data_source_location") or selected_dataset.get("full_data_location")
                        preview = selected_dataset.get("preview")
                        columns = selected_dataset.get("columns")
                        
                        if not data_location or not os.path.exists(data_location):
                            error_msg += f"- Data file not found or not accessible\n"
                        if not preview:
                            error_msg += f"- Preview data not available\n"
                        if not columns:
                            error_msg += f"- Column information not available\n"
                        
                        # List available datasets as alternatives
                        if datasets_context and isinstance(datasets_context, list):
                            other_datasets = [
                                ds for ds in datasets_context
                                if isinstance(ds, dict) and ds.get("dataset_id") != active_dataset_id
                            ]
                            if other_datasets:
                                error_msg += "\n**Other available datasets:**\n"
                                for ds in other_datasets:
                                    ds_id = ds.get("dataset_id", "unknown")
                                    ds_filename = ds.get("filename", "unknown")
                                    ds_alias = ds.get("alias", "")
                                    ds_display = ds_alias or ds_filename or ds_id
                                    error_msg += f"- {ds_display} (ID: {ds_id})\n"
                        
                        return {
                            "success": False,
                            "error": f"Could not extract data from selected dataset '{display_name}'",
                            "message": error_msg
                        }
                
                # Provide helpful error message with debugging info
                debug_info = []
                if state.get("uploaded_csv_preview"):
                    preview = state.get("uploaded_csv_preview", [])
                    debug_info.append(f"uploaded_csv_preview: {len(preview)} rows")
                    if preview:
                        # Show first few rows to help debug
                        debug_info.append(f"  First 3 rows: {preview[:min(3, len(preview))]}")
                if state.get("uploaded_csv_columns"):
                    debug_info.append(f"uploaded_csv_columns: {state.get('uploaded_csv_columns')}")
                if state.get("sample_data"):
                    sample = state.get("sample_data")
                    if isinstance(sample, str):
                        debug_info.append(f"sample_data: string, length={len(sample)}, first 200 chars: {sample[:200]}")
                    else:
                        debug_info.append(f"sample_data: type={type(sample)}")
                if state.get("data_source_location"):
                    data_path = state.get("data_source_location")
                    debug_info.append(f"data_source_location: {data_path}")
                    debug_info.append(f"  File exists: {os.path.exists(data_path) if data_path else False}")
                    if data_path and os.path.exists(data_path):
                        # Try to read a few lines to verify it's a valid CSV
                        try:
                            with open(data_path, 'r') as f:
                                first_lines = [f.readline().strip() for _ in range(3)]
                            debug_info.append(f"  First 3 lines of file: {first_lines}")
                        except Exception as e:
                            debug_info.append(f"  Error reading file: {e}")
                
                if path_or_source and not os.path.exists(path_or_source):
                    return {
                        "success": False,
                        "error": f"File not found: {path_or_source}",
                        "message": f"File not found: {path_or_source}\nPlease check the path and ensure the file exists."
                    }
                elif path_or_source == "data_source" and not state.get("data_source_location"):
                    return {
                        "success": False,
                        "error": "No data_source_location found in state",
                        "message": "No data source found. Please specify a data source or upload a CSV file."
                    }
                elif path_or_source == "uploaded":
                    debug_msg = "\n".join(debug_info) if debug_info else "No uploaded data found in state"
                    
                    # Check if preview exists but only has headers
                    preview = state.get("uploaded_csv_preview", [])
                    if preview:
                        # Check if all rows are headers
                        columns = state.get("uploaded_csv_columns", [])
                        if columns:
                            header_normalized = [str(col).strip().lower() for col in columns]
                            all_headers = all(
                                [str(val).strip().lower() for val in row] == header_normalized
                                for row in preview
                            )
                            if all_headers:
                                error_msg = (
                                    f"All rows in uploaded_csv_preview appear to be header rows (no data rows found). "
                                    f"This usually indicates an issue with how the CSV preview was generated. "
                                    f"\n\nTroubleshooting:"
                                    f"\n1. The uploaded CSV file itself is likely correct"
                                    f"\n2. Try using the file path directly if data_source_location is available"
                                    f"\n3. Check if sample_data contains the full CSV content"
                                    f"\n\nDebug info: {debug_msg}"
                                )
                            else:
                                error_msg = (
                                    f"Could not extract CSV from uploaded data. "
                                    f"Preview has {len(preview)} rows but no valid data rows were found after filtering. "
                                    f"\n\nDebug info: {debug_msg}"
                                )
                        else:
                            error_msg = (
                                f"Could not extract CSV from uploaded data. "
                                f"Preview exists but columns are missing. "
                                f"\n\nDebug info: {debug_msg}"
                            )
                    else:
                        error_msg = (
                            f"Could not extract CSV from uploaded data. "
                            f"No preview data found in state. "
                            f"\n\nDebug info: {debug_msg}"
                        )
                    
                    return {
                        "success": False,
                        "error": "No CSV data found in uploaded data",
                        "message": error_msg
                    }
                else:
                    debug_msg = "\n".join(debug_info) if debug_info else "No data sources found"
                    return {
                        "success": False,
                        "error": "No CSV data found for inference",
                        "message": f"No CSV data found for inference.\nDebug info: {debug_msg}\n\nPlease upload a CSV file or specify a valid file path."
                    }
            
            # Final check right before staging - ensure DataFrame is valid
            if df_has_only_headers(df):
                logger.error(
                    f"CRITICAL: DataFrame still contains only headers after all filtering and fallbacks. "
                    f"DataFrame shape: {df.shape}, columns: {list(df.columns)}"
                )
                # Last resort: try to read directly from file if available
                if state.get("data_source_location"):
                    file_path = state["data_source_location"]
                    if os.path.exists(file_path):
                        logger.info(f"Last resort: Reading directly from file: {file_path}")
                        try:
                            df = pd.read_csv(file_path)
                            if df_has_only_headers(df):
                                logger.error(f"Even direct file read contains only headers. File: {file_path}")
                            else:
                                logger.info(f"✓ Successfully read from file: {df.shape}")
                        except Exception as e:
                            logger.error(f"Failed to read from file {file_path}: {e}")
                
                # If still invalid, return error
                if df_has_only_headers(df):
                    return {
                        "success": False,
                        "error": "DataFrame contains only header rows",
                        "message": (
                            f"Unable to extract valid data from uploaded CSV. "
                            f"The DataFrame contains only header rows (no actual data). "
                            f"This usually means:\n"
                            f"1. The CSV preview was generated incorrectly\n"
                            f"2. All rows in the preview match the header\n"
                            f"3. The file might need to be re-uploaded\n\n"
                            f"Please try uploading the CSV file again or use a file path directly."
                        )
                    }
            
            # Resolve MLflow run
            run_id = state.get("mlflow_run_id")
            if not run_id:
                return {
                    "success": False,
                    "error": "No trained model found",
                    "message": "No trained model found. Please train a model first before running inference."
                }
            
            # Get MLflow tracking URI
            mlflow_tracking_uri = os.getenv("MLFLOW_BACKEND_STORE_URI") or os.getenv("MLFLOW_TRACKING_URI")
            if not mlflow_tracking_uri:
                # Try to get from MLflow manager
                if self.mlflow_manager:
                    mlflow_tracking_uri = self.mlflow_manager.tracking_uri
                else:
                    return {
                        "success": False,
                        "error": "MLflow tracking URI not configured",
                        "message": "MLflow tracking URI not configured. Cannot access model artifacts."
                    }
            
            # Construct model URI
            model_uri = f"runs:/{run_id}/model"
            
            # Final safety check before inference
            if df_has_only_headers(df):
                logger.error(
                    f"CRITICAL ERROR: DataFrame contains only headers right before inference. "
                    f"This should have been caught earlier. DataFrame shape: {df.shape}, "
                    f"columns: {list(df.columns)}, first row: {df.iloc[0].to_dict() if len(df) > 0 else 'empty'}"
                )
                return {
                    "success": False,
                    "error": "DataFrame contains only header rows",
                    "message": (
                        f"Unable to proceed with inference: DataFrame contains only header rows. "
                        f"This indicates a problem with the CSV data extraction. "
                        f"Please check the uploaded CSV file and try again."
                    )
                }
            
            # Route to local or Kubernetes inference based on detected mode
            # Use detected mode from user intent, default to "local" if not set
            use_local_mode = (inference_mode == "local") if inference_mode else True
            logger.info(f"Routing inference: detected_mode={inference_mode}, use_local_mode={use_local_mode}")
            
            if use_local_mode:
                # Local inference path
                if not self.local_inference:
                    # Fallback to Kubernetes if local is not available
                    if self.k8s_manager:
                        logger.warning("Local inference requested but not available. Falling back to Kubernetes inference.")
                        use_local_mode = False  # Fall through to k8s path
                    else:
                        return {
                            "success": False,
                            "error": "No inference manager available",
                            "message": "Local inference mode was requested but Local Inference Manager is not available, and Kubernetes Job Manager is also not available. Please check configuration."
                        }
                
                if use_local_mode and self.local_inference:
                    logger.info(f"Running local inference for run_id: {run_id}, DataFrame shape: {df.shape}")
                    try:
                        # Run inference (predict() internally handles model loading via cache)
                        predictions_df = self.local_inference.predict(df, run_id, model_uri)
                        
                        # Format message
                        message = self.format_predictions_message(predictions_df, job_status=None)
                        
                        return {
                            "success": True,
                            "message": message,
                            "predictions_df": predictions_df,
                            "job_name": None,  # No job in local mode
                            "output_gcs_path": None  # No GCS path in local mode
                        }
                    except Exception as e:
                        logger.error(f"Local inference failed: {e}", exc_info=True)
                        # Fallback to Kubernetes if local inference execution fails
                        if self.k8s_manager:
                            logger.warning("Local inference execution failed. Falling back to Kubernetes inference.")
                            use_local_mode = False  # Fall through to k8s path
                        else:
                            return {
                                "success": False,
                                "error": str(e),
                                "message": f"Local inference failed: {str(e)}"
                            }
            
            # Kubernetes inference path (fallback or primary)
            if not use_local_mode:
                # Check that k8s_manager is available
                if not self.k8s_manager:
                    # Final fallback: try local if k8s is not available
                    if self.local_inference:
                        logger.warning("Kubernetes inference requested but not available. Falling back to local inference.")
                        try:
                            logger.info(f"Running local inference (fallback) for run_id: {run_id}, DataFrame shape: {df.shape}")
                            predictions_df = self.local_inference.predict(df, run_id, model_uri)
                            message = self.format_predictions_message(predictions_df, job_status=None)
                            return {
                                "success": True,
                                "message": message,
                                "predictions_df": predictions_df,
                                "job_name": None,
                                "output_gcs_path": None
                            }
                        except Exception as e:
                            logger.error(f"Local inference fallback also failed: {e}", exc_info=True)
                            return {
                                "success": False,
                                "error": str(e),
                                "message": f"Both inference modes failed. Local inference fallback error: {str(e)}"
                            }
                    else:
                        return {
                            "success": False,
                            "error": "No inference manager available",
                            "message": "Kubernetes inference mode was requested but Kubernetes Job Manager is not available, and Local Inference Manager is also not available. Please check configuration."
                        }
                
                # k8s_manager is available, proceed with Kubernetes inference
                # Stage input CSV to GCS
                logger.info(f"Staging input CSV for inference (run_id: {run_id}, DataFrame shape: {df.shape})")
                input_gcs_path = self.stage_input_csv(df, run_id)
                
                # CRITICAL: Validate that stage_input_csv returned a full GCS URI
                # This should never happen if stage_input_csv is working correctly, but we check anyway
                if not input_gcs_path or not isinstance(input_gcs_path, str):
                    raise ValueError(
                        f"stage_input_csv returned invalid path: {input_gcs_path} (type: {type(input_gcs_path)}). "
                        f"Expected a full GCS URI starting with 'gs://'."
                    )
                
                # stage_input_csv should always return a full GCS URI
                # If it doesn't, that's a critical error
                if not input_gcs_path.startswith("gs://"):
                    error_details = (
                        f"CRITICAL: stage_input_csv returned path without gs:// prefix: '{input_gcs_path}'. "
                        f"This should never happen - stage_input_csv should always return a full GCS URI. "
                        f"This indicates a bug in blob_store.put_file or stage_input_csv."
                    )
                    logger.error(error_details)
                    raise ValueError(
                        f"{error_details} "
                        f"Please check blob_store configuration and ensure put_file returns a full GCS URI."
                    )
                
                # Final validation: ensure we have a valid GCS URI
                path_parts = input_gcs_path.replace("gs://", "").split("/")
                if len(path_parts) < 2:
                    raise ValueError(
                        f"Invalid GCS URI format: '{input_gcs_path}'. "
                        f"Expected format: 'gs://bucket-name/path/to/file'"
                    )
                
                logger.info(f"✓ Input CSV staged to GCS: {input_gcs_path}")
                
                # Get metadata path from artifacts if available
                # Only pass GCS paths - local paths won't be accessible in the container
                # The inference job will load metadata from MLflow artifacts if metadata_path is None
                metadata_path = None
                if state.get("model_artifacts"):
                    artifacts = state["model_artifacts"]
                    potential_path = artifacts.get("metadata_path") or artifacts.get("onnx_metadata_path")
                    # Only use if it's a GCS path (container can't access local paths)
                    if potential_path and potential_path.startswith("gs://"):
                        metadata_path = potential_path
                    # Otherwise, let the inference job load from MLflow artifacts (which it now does automatically)
                
                # Extract bucket name from input_gcs_path to pass to Kubernetes job
                # This ensures GCS_BUCKET env var is available even if path format changes
                # CRITICAL: We MUST extract and pass the bucket name to the pod
                gcs_bucket_for_job = None
                
                # ALWAYS extract bucket from blob_store first (most reliable source)
                # This ensures we have it even if path format is unexpected
                blob_store_bucket = getattr(self.blob_store, '_bucket', None)
                blob_store_bucket_name = None
                if blob_store_bucket:
                    if hasattr(blob_store_bucket, 'name'):
                        blob_store_bucket_name = blob_store_bucket.name
                        logger.info(f"✓ Extracted bucket from blob_store._bucket.name: {blob_store_bucket_name}")
                    elif isinstance(blob_store_bucket, str):
                        blob_store_bucket_name = blob_store_bucket
                        logger.info(f"✓ Extracted bucket from blob_store._bucket (string): {blob_store_bucket_name}")
            
                # Priority 1: Use blob_store bucket (most reliable)
                gcs_bucket_for_job = blob_store_bucket_name
            
                # Priority 2: Extract from input_gcs_path if it's a full GCS URI (verify/override)
                if input_gcs_path.startswith("gs://"):
                    path_bucket = input_gcs_path.replace("gs://", "").split("/")[0]
                    if gcs_bucket_for_job and gcs_bucket_for_job != path_bucket:
                        logger.warning(
                            f"Bucket mismatch: blob_store says '{gcs_bucket_for_job}' but path says '{path_bucket}'. "
                            f"Using blob_store value for reliability."
                        )
                    elif not gcs_bucket_for_job:
                        gcs_bucket_for_job = path_bucket
                        logger.info(f"✓ Extracted bucket from input_gcs_path: {gcs_bucket_for_job}")
                else:
                    logger.warning(f"input_gcs_path does not start with gs://: '{input_gcs_path}'. Using blob_store bucket.")
            
                # Priority 3: Try from TemporaryInferenceManager's gcs_bucket attribute
                if not gcs_bucket_for_job:
                    gcs_bucket_for_job = self.gcs_bucket
                    if gcs_bucket_for_job:
                        logger.info(f"Using self.gcs_bucket: {gcs_bucket_for_job}")
                
                # Priority 4: Try environment variables
                if not gcs_bucket_for_job:
                    gcs_bucket_for_job = os.getenv("GCS_BUCKET") or os.getenv("GCS_INFERENCE_BUCKET")
                    if gcs_bucket_for_job:
                        logger.info(f"Using GCS_BUCKET from environment: {gcs_bucket_for_job}")
                
                # Priority 5: Try cached bucket name from stage_input_csv
                if not gcs_bucket_for_job:
                    cached_bucket = getattr(self, '_cached_bucket_name', None)
                    if cached_bucket:
                        gcs_bucket_for_job = cached_bucket
                        logger.info(f"Using cached bucket name from stage_input_csv: {gcs_bucket_for_job}")
            
                # Priority 6: Try to extract from storage service settings
                if not gcs_bucket_for_job:
                    try:
                        from app.core.settings import Settings
                        settings = Settings()
                        if hasattr(settings, 'gcs_bucket') and settings.gcs_bucket:
                            gcs_bucket_for_job = settings.gcs_bucket
                            logger.info(f"Using bucket from Settings.gcs_bucket: {gcs_bucket_for_job}")
                    except Exception as e:
                        logger.debug(f"Could not get bucket from Settings: {e}")
                
                # CRITICAL: We MUST have a bucket name to pass to the inference job
                # If we still don't have it, try one more aggressive extraction
                if not gcs_bucket_for_job:
                    # Try to get from storage_service if available (same blob_store instance)
                    try:
                        from app.services import storage_service
                        if hasattr(storage_service, 'blob_store') and storage_service.blob_store:
                            blob_store = storage_service.blob_store
                            if hasattr(blob_store, '_bucket'):
                                bucket_obj = blob_store._bucket
                                if hasattr(bucket_obj, 'name'):
                                    gcs_bucket_for_job = bucket_obj.name
                                    logger.info(f"Extracted bucket from storage_service.blob_store: {gcs_bucket_for_job}")
                                elif isinstance(bucket_obj, str):
                                    gcs_bucket_for_job = bucket_obj
                                    logger.info(f"Extracted bucket from storage_service.blob_store (string): {gcs_bucket_for_job}")
                    except Exception as e:
                        logger.debug(f"Could not get bucket from storage_service: {e}")
                    
                    # Also try to use cached bucket name from stage_input_csv if available
                    if not gcs_bucket_for_job:
                        cached_bucket = getattr(self, '_cached_bucket_name', None)
                        if cached_bucket:
                            gcs_bucket_for_job = cached_bucket
                            logger.info(f"Using cached bucket name from stage_input_csv: {gcs_bucket_for_job}")
                
                if gcs_bucket_for_job:
                    logger.info(f"✓ Will pass GCS_BUCKET={gcs_bucket_for_job} to inference job")
                else:
                    # This is a critical error - we MUST have a bucket name
                    error_details = (
                        f"input_gcs_path: '{input_gcs_path}', "
                        f"blob_store type: {type(self.blob_store)}, "
                        f"blob_store._bucket: {getattr(self.blob_store, '_bucket', 'N/A')}, "
                        f"cached_bucket_name: {getattr(self, '_cached_bucket_name', None)}, "
                        f"self.gcs_bucket: {self.gcs_bucket}, "
                        f"env GCS_BUCKET: {os.getenv('GCS_BUCKET')}, "
                        f"env GCS_INFERENCE_BUCKET: {os.getenv('GCS_INFERENCE_BUCKET')}"
                    )
                    logger.error(f"✗ CRITICAL: Could not determine GCS bucket for inference job! {error_details}")
                    # Raise error to prevent creating a job that will definitely fail
                    raise ValueError(
                        f"Cannot determine GCS bucket for inference job. "
                        f"This is required to pass GCS_BUCKET environment variable to the pod. "
                        f"Details: {error_details}. "
                        f"Please ensure GCS_BUCKET environment variable is set or blob_store is properly configured."
                    )
                
                # Create Kubernetes Job
                logger.info(f"Creating inference job for run: {run_id}")
            
                # Get service account - only use if explicitly set and exists
                # Priority: GCS_WRITER > K8S_SERVICE_ACCOUNT > INFERENCE_SERVICE_ACCOUNT
                # Otherwise, let it use the default service account
                service_account = (
                    os.getenv("GCS_WRITER") or  # Check for gcs-writer first
                    os.getenv("K8S_SERVICE_ACCOUNT") or 
                    os.getenv("INFERENCE_SERVICE_ACCOUNT")
                )
                # If service account is set but doesn't exist, don't use it (will use default)
                if service_account:
                    try:
                        # Check if service account exists
                        if self.k8s_manager and hasattr(self.k8s_manager, 'core_api'):
                            try:
                                self.k8s_manager.core_api.read_namespaced_service_account(
                                    name=service_account,
                                    namespace=self.k8s_manager.namespace
                                )
                                logger.info(f"Service account '{service_account}' exists, will use it")
                            except Exception as e:
                                logger.warning(
                                    f"Service account '{service_account}' not found or error checking: {e}. "
                                    f"Will use default service account instead."
                                )
                                service_account = None  # Use default
                    except Exception as e:
                        logger.warning(f"Could not check service account: {e}. Will use default.")
                        service_account = None
            
                # Pass _USE_DEFAULT_SERVICE_ACCOUNT sentinel if service_account is None
                # This explicitly tells create_inference_job to use default and NOT check env vars
                service_account_param = _USE_DEFAULT_SERVICE_ACCOUNT if service_account is None else service_account
            
                # CRITICAL: Verify we have a bucket before creating the job
                if not gcs_bucket_for_job:
                    raise ValueError(
                        f"Cannot create inference job: GCS bucket name is required but could not be determined. "
                        f"input_gcs_path: '{input_gcs_path}', "
                        f"blob_store type: {type(self.blob_store)}. "
                        f"Please ensure GCS_BUCKET environment variable is set or blob_store is properly configured."
                    )
                
                logger.info(
                    f"Creating inference job with: "
                    f"run_id={run_id}, "
                    f"input_gcs_path={input_gcs_path}, "
                    f"gcs_bucket={gcs_bucket_for_job}"
                )
                
                job_result = self.create_inference_job(
                    run_id=run_id,
                    model_uri=model_uri,
                    input_gcs_path=input_gcs_path,
                    mlflow_tracking_uri=mlflow_tracking_uri,
                    metadata_path=metadata_path,
                    service_account=service_account_param,  # Use sentinel to explicitly request default
                    gcs_bucket=gcs_bucket_for_job  # Pass bucket name explicitly - CRITICAL for pod to construct full GCS URIs
                )
                
                if not job_result.get("created"):
                    return {
                        "success": False,
                        "error": job_result.get("error", "Failed to create inference job"),
                        "message": f"Failed to create inference job: {job_result.get('error', 'Unknown error')}"
                    }
                
                job_name = job_result["job_name"]
                output_gcs_path = job_result["output_gcs_path"]
                
                # Monitor job until completion
                logger.info(f"Monitoring inference job: {job_name}")
                job_status = self.k8s_manager.wait_for_completion(
                    job_name=job_name,
                    timeout=3600  # 1 hour timeout
                )
                
                if job_status.get("status") == "Timeout":
                    return {
                        "success": False,
                        "error": "Inference job timed out",
                        "message": "Inference job did not complete within the timeout period. Please check the job logs."
                    }
                
                if job_status.get("status") != "Succeeded":
                    # Get logs for debugging
                    logs = self.k8s_manager.get_job_logs(job_name)
                    return {
                        "success": False,
                        "error": f"Inference job failed with status: {job_status.get('status')}",
                        "message": f"Inference job failed.\nStatus: {job_status.get('status')}\n\nJob logs:\n{logs[:1000]}",  # First 1000 chars
                        "logs": logs
                    }
                
                # Retrieve results
                logger.info(f"Retrieving inference results from: {output_gcs_path}")
                df_results = self.retrieve_output_csv(output_gcs_path)
                
                # Format message
                message = self.format_predictions_message(df_results, job_status)
                
                # Clean up job (optional - TTL will handle it, but we can clean up immediately)
                # Only delete if keep_jobs is False
                if not self.keep_jobs:
                    try:
                        logger.info(f"Cleaning up job {job_name} (set KEEP_INFERENCE_JOBS=true to keep jobs for debugging)")
                        self.k8s_manager.delete_job(job_name, wait=False)
                    except Exception as e:
                        logger.warning(f"Failed to delete job {job_name}: {e}")
                else:
                    logger.info(f"Keeping job {job_name} for debugging (KEEP_INFERENCE_JOBS is enabled)")
                
                return {
                    "success": True,
                    "message": message,
                    "predictions_df": df_results,
                    "job_name": job_name,
                    "output_gcs_path": output_gcs_path
                }
            
        except Exception as e:
            logger.error(f"Inference request handling failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Inference failed: {str(e)}"
            }

