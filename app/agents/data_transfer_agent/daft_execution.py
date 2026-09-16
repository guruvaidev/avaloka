import base64
import logging
import os
import subprocess
import sys
import textwrap
import time
from io import StringIO
from typing import Any, Dict, Optional
import uuid
import tempfile
import re

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage
from app.agents.execution_agent import _strip_markdown_code_fences, get_service_details
from app.agents.data_transfer_agent.dta_state import DaftETLState
from pathlib import Path 

from app.graph.etl_state import ETLState
from app.utils import generate_filename_timestamp

logger = logging.getLogger(__name__)
AVALOKA_LOCAL_EXEC_TIMEOUT=120

def execution_agent_node_local(state: DaftETLState) -> DaftETLState:
    """LangGraph node that executes Python code locally"""
    logger.info("Execution Agent: Starting local execution process")

    new_state = state.copy()

    # Get code from the coding agent
    coder_def = new_state.get("coder_definition", {}).copy()
    code = coder_def.get("generated_code", "")
    if not code:
        message = "No code provided for execution"
        logger.error(message)

        # Update state with error
        new_state = state.copy()
        coder_def["execution_error"] = message
        new_state["coder_definition"] = coder_def
        new_state["code_validated"] = False
        new_state["generated_output_schema"] = None
        return new_state
    
    #update chat history
    coder_explanation = coder_def.get("llm_raw_response", "")
    if coder_explanation:
        existing_messages = state.get("messages", [])
        if not any(getattr(msg, "content", None) == coder_explanation for msg in existing_messages):
            state["messages"] = existing_messages + [AIMessage(content=coder_explanation)]

    # Get input data from state
    input_sample_data = coder_def.get("input_sample_data")
    if input_sample_data is None:
        error_msg = "No input sample data provided for execution"
        logger.error(error_msg)
        # Flag the failure like the no-code path, else Step 8 reports a false
        # success on code that never executed.
        coder_def["execution_error"] = error_msg
        coder_def["code_validation_feedback"] = error_msg
        new_state["coder_definition"] = coder_def
        new_state["code_executed"] = False
        new_state["generated_output_schema"] = None
        new_state["messages"] = new_state["messages"] + [AIMessage(content=error_msg)]
        return new_state

    # Execute code locally
    execution_result = execute_code_on_local(
        code=code,
        sample_data=input_sample_data,
    )

    coder_def['execution_stdout'] = execution_result.get("execution_stdout")
    coder_def['execution_stderr'] = execution_result.get("execution_stderr")
    coder_def['generated_output_schema'] = execution_result.get("generated_output_schema")

    if execution_result["status"] == "success":
        preview_table = execution_result.get("execution_stdout", "")

        coder_def["execution_output_preview"] = preview_table
        coder_def["execution_error"] = None
        coder_def["code_validation_feedback"] = None 
        new_state["code_executed"] = True
        coder_def["generated_output_schema"] = execution_result.get("generated_output_schema")

        success_msg = (
            f"**Validation Successful**\n"
            f"Here is a preview of the transformed data:\n\n"
            f"```text\n{preview_table}\n```"
        )
        new_state["messages"] = new_state["messages"] + [AIMessage(content=success_msg)]
        
    else:
        # execution_error holds the real cause; stderr is empty on timeout and
        # sample-serialization failures, so prefer it over stderr.
        error_msg = (
            execution_result.get("execution_error")
            or execution_result.get("execution_stderr")
            or "Unknown execution error occurred."
        )

        # Store error details
        coder_def["execution_error"] = error_msg
        coder_def["execution_output_preview"] = None
                
        # Mark as not validated
        new_state["code_executed"] = False

        # Show error message to User
        failure_msg = (
            f"**Execution Failed**\n"
            f"Error Details:\n\n"
            f"```text\n{error_msg}\n```"
        )
        new_state["messages"] = new_state["messages"] + [AIMessage(content=failure_msg)]
        

    new_state["coder_definition"] = coder_def
    
    return new_state  
    

def execute_code_on_local(
    code: str,
    sample_data: pd.DataFrame
) -> Dict[str, Any]:

    out  = {}
    # Clean up markdown code fences if ``` is present
    extracted_code = _strip_markdown_code_fences(code)

    run_id = str(uuid.uuid4())[:8]
    temp_data_path = f"temp_data_{run_id}.parquet"

    # We save the Pandas DataFrame to Parquet. This preserves strict data types 
    try:
        sample_data.to_parquet(temp_data_path, index=False)
        logger.info(f"Serialized sample data to {temp_data_path}")
    except Exception as e:
        return {
            "status": "error",
            "execution_error": f"System Error: Failed to serialize sample data: {e}"
        }

    start = time.time()

    #harnessing template for the script to execute as daft code
    
    harness_template = textwrap.dedent(f"""\
import daft
import sys

# 1. SETUP: Load the data from the bridge file
try:
    df = daft.read_parquet(r"{temp_data_path}")
except Exception as e:
    print(f"System Error: Failed to load input data: {{e}}", file=sys.stderr)
    sys.exit(1)

# 2. INJECTION: Run the User's Code
# ---------------------------------------------------------
""")

    script_to_run = harness_template + extracted_code + textwrap.dedent(f"""
# ---------------------------------------------------------

# 3. TRIGGER: Force Execution & Preview
try:
    target_df = None

    # CHECK 1: Did the user define a 'transform_data' function?
    if 'transform_data' in locals() and callable(locals()['transform_data']):
        print("--- Invoking transform_data(df) ---")
        target_df = transform_data(df)
        
    # CHECK 2: Fallback to 'main' just in case
    elif 'main' in locals() and callable(locals()['main']):
        print("--- Invoking main(df) ---")
        target_df = main(df)

    # CHECK 3: Fallback to 'df'
    elif 'df' in locals():
        target_df = df
    
    else:
        raise ValueError("Code ran, but no DataFrame variable ('df') or transformation function was found.")

    # FINAL CHECK: Ensure we actually have a dataframe
    if target_df is None:
         raise ValueError("The transformation function returned None. It must return a Daft DataFrame.")

    # Trigger execution
    print(target_df.show())
    
    # Print schema for validation
    schema_output = target_df.schema()
    print("\\n---SCHEMA_START---")
    print(schema_output)
    print("---SCHEMA_END---")

except Exception as e:
    print(e, file=sys.stderr)
    sys.exit(1)
""")

    # Save the script content to a temporary file (optional, but good for complex scripts)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(script_to_run)
        temp_file_path = temp_file.name

    try:
        # Run the script in a separate process
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            encoding='utf-8',      # ← Decode subprocess output as UTF-8
            errors='replace',  
            check=False, 
            timeout=int(os.getenv("AVALOKA_LOCAL_EXEC_TIMEOUT", "60")),
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        
        # Store raw output for logging
        out["returncode"] = result.returncode
        out["execution_stdout"] = stdout
        out["execution_stderr"] = stderr

        # Status Logic ---
        if result.returncode == 0:
            logger.info("Output from the new interpreter:\n%s", stdout)
            if stderr.strip():
                # warnings are not errors
                logger.warning("Warnings from the new interpreter:\n%s", stderr)

            match = re.search(r"---SCHEMA_START---\n(.*?)\n---SCHEMA_END---", stdout, re.DOTALL)
            if match:
                schema_variable = match.group(1).strip()
                print("Successfully captured schema!")
                out["generated_output_schema"] = str(schema_variable)
            else:
                print("Execution succeeded, but schema markers were not found.")
                out["generated_output_schema"] = "Execution succeeded, but schema markers were not found."

            out["status"] = "success"
            out["execution_error"] = None
        else:
            logger.error("Interpreter failed (rc=%s)", result.returncode)
            logger.error("Stdout:\n%s", stdout)
            logger.error("Stderr:\n%s", stderr)

            out["status"] = "error"
            out["execution_error"] = f"Script failed with exit status {result.returncode}. Error: {stderr.strip()}"

    except subprocess.TimeoutExpired as e:
        out["status"] = "error"
        out["execution_error"] = "Timeout Error: The transformation took too long to generate a preview."
        out["execution_stderr"] = e.stderr or ""

    except FileNotFoundError:
        out["status"] = "error"
        out["execution_error"] = "System Error: Python executable not found."

    finally:
        # Cleanup: Delete both the script and the temporary data file to keep the server clean
        for file_to_remove in [temp_file_path, temp_data_path]:
            if os.path.exists(file_to_remove):
                os.remove(file_to_remove)
                
    return out


