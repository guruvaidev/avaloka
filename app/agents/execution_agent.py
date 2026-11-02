"""Execute python code from the coding agent via the infra agent."""

import ast
import inspect
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import time
import tempfile
from io import StringIO
from typing import Any, Dict, Optional
import re
import numpy as np
import pandas as pd
import requests
from langchain_core.messages import AIMessage
from app.agents.infra_agent import infra_agent_node
from app.agents.result_renderer import render_response
from app.agents.lineage_hooks import record_analysis_lineage
from app.analytics.schema import column_count_bucket, row_count_bucket
from pathlib import Path

from app.graph.etl_state import ETLState
from app.utils import generate_filename_timestamp
from app.infra.ray_job_runner import run_rayjob_from_yaml
from app.agents.output_contract import (
    check_output,
    frame_from_output_file,
    local_reference,
    parse_reference,
    reference_script,
    refusal_text,
    spec_from_state,
)

import asyncio
import concurrent.futures
from urllib.parse import urlparse

from app.api.cloud_connections import get_cloud_connection
from app.infra.k8s_secrets import cloud_secret_env
from app.executor.env import scrubbed_env

logger = logging.getLogger(__name__)
AVALOKA_LOCAL_EXEC_TIMEOUT = 120


import os


HELPER_FUNCTIONS = r'''
# Headless server: force a non-GUI matplotlib backend BEFORE any user code
# imports pyplot, so generated plotting code can never pop a window or block
# the process on plt.show() (which would hang until the 60s timeout and surface
# as an "execution failure"). Must run before "import matplotlib.pyplot".
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass
import pandas as pd
import json as _avaloka_json

def avaloka_result(value, kind=None, columns=None, **kw):
    """Emit a structured result sentinel that the response renderer picks up.
    Call this for analytical queries (mean, correlation, count, etc.) even when
    an output file is also written — the sentinel takes priority over the table.
    value   — the scalar result. Usually a number, but a single-value answer
              is not always numeric ("which category is the biggest?" -> "Music"),
              so a non-numeric scalar is passed through unchanged rather than
              crashing the run.
    kind    — "mean"|"sum"|"count"|"correlation"|"std"|"max"|"min"
    columns — list of column names involved, e.g. ["TransactionAmt"] or
              ["P_emaildomain_length","TransactionAmt"]
    """
    if value is None:
        _coerced = None
    else:
        try:
            _coerced = float(value)
        except (TypeError, ValueError):
            # A categorical answer ("Music", a channel name, a date). float()
            # used to raise here and fail the entire execution.
            _coerced = value
    obj = {"kind": kind or "scalar", "value": _coerced,
           "columns": list(columns or []), **kw}
    print(f"<<<AVALOKA_RESULT>>>{_avaloka_json.dumps(obj)}<<<END_AVALOKA_RESULT>>>", flush=True)

# <<AVALOKA_CSV_READER>>


def safe_groupby_agg(df, group_cols, agg_col, agg_func='sum'):
    """Safely perform groupby aggregation with conflict resolution."""
    group_cols = [col for col in group_cols if col != agg_col]
    if not group_cols:
        print("Warning: No valid grouping columns found. Using categorical columns.")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        group_cols = categorical_cols[:2]
    print(f"Grouping by: {group_cols}")
    print(f"Aggregating column: {agg_col}")
    if group_cols:
        result = df.groupby(group_cols)[agg_col].agg(agg_func).reset_index()
        if agg_func == 'sum':
            result = result.rename(columns={agg_col: f'{agg_col}_sum'})
        elif agg_func == 'mean':
            result = result.rename(columns={agg_col: f'{agg_col}_mean'})
        elif agg_func == 'count':
            result = result.rename(columns={agg_col: f'{agg_col}_count'})
    else:
        result = df[agg_col].agg(agg_func).to_frame().T
        if agg_func == 'sum':
            result = result.rename(columns={agg_col: f'{agg_col}_sum'})
    return result


def safe_groupby_multiple_agg(df, group_cols, agg_cols, agg_funcs=['sum']):
    """Safely perform groupby with multiple aggregations."""
    group_cols = [col for col in group_cols if col not in agg_cols]
    if not group_cols:
        print("Warning: No valid grouping columns found. Using categorical columns.")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        group_cols = categorical_cols[:2]
    print(f"Grouping by: {group_cols}")
    print(f"Aggregating columns: {agg_cols}")
    agg_dict = {col: agg_funcs for col in agg_cols}
    result = df.groupby(group_cols).agg(agg_dict)
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            f"{c[0]}_{c[1]}" if len(c) > 1 else str(c[0])
            for c in result.columns
        ]
    return result.reset_index()
'''


def _resolve_read_paths(state) -> dict:
    """basename/alias -> real per-dataset path, for multi-file code."""
    mapping = {}
    for meta in state.get("multi_dataset_state") or []:
        path = meta.get("data_source_location")
        if not path:
            continue
        if meta.get("filename"):
            mapping[os.path.basename(str(meta["filename"]))] = path
        if meta.get("alias"):
            mapping[str(meta["alias"])] = path
    return mapping

def _brief_error(detail, limit: int = 220) -> str:
    """One customer-readable line from an arbitrary error/traceback.

    Customers must never see a wall of technical output in chat — the full
    detail goes to the server logs. A Python traceback's LAST line is the
    actual error, so surface that; otherwise the first non-empty line.
    """
    text = str(detail or "").strip()
    if not text:
        return "an unknown error occurred"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    line = lines[-1] if "Traceback (most recent call last)" in text else lines[0]
    if len(line) > limit:
        line = line[:limit].rstrip() + "…"
    return line


def _friendly_failure_message(reason_source) -> str:
    """The standard short chat reply for a failed execution step."""
    return (
        "❌ I couldn't complete this step.\n\n"
        f"Reason: {_brief_error(reason_source)}\n\n"
        "Please try again or rephrase the request — the technical details "
        "have been recorded in the logs."
    )

RAYJOB_TEMPLATE_PATH = os.getenv("RAYJOB_TEMPLATE_PATH", "app/infra/rayjob.yaml")


class InteractiveInputVisitor(ast.NodeVisitor):
    """Detect generated code that would block waiting for interactive input."""

    def __init__(self):
        self.errors: list[str] = []

    def _add_error(self, message: str):
        if message not in self.errors:
            self.errors.append(message)

    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "input":
            self._add_error(
                f"SafetyError on line {node.lineno}: input() is not allowed in generated code."
            )
        elif isinstance(func, ast.Attribute) and func.attr in {"getpass", "read", "readline", "readlines"}:
            value = func.value
            if isinstance(value, ast.Name) and value.id in {"getpass", "stdin"}:
                self._add_error(
                    f"SafetyError on line {node.lineno}: interactive stdin/getpass reads are not allowed."
                )
            elif (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "sys"
                and value.attr == "stdin"
            ):
                self._add_error(
                    f"SafetyError on line {node.lineno}: sys.stdin reads are not allowed."
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "sys" and node.attr == "stdin":
            self._add_error(
                f"SafetyError on line {node.lineno}: sys.stdin reads are not allowed."
            )
        self.generic_visit(node)


def _find_interactive_input_errors(code: str) -> list[str]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []
    visitor = InteractiveInputVisitor()
    visitor.visit(tree)
    return visitor.errors


# ---------------------------------------------------------------------------
# RayJob template loading
# ---------------------------------------------------------------------------

def _load_rayjob_template() -> str:
    p = Path(RAYJOB_TEMPLATE_PATH)
    if p.exists():
        return p.read_text(encoding="utf-8")

    here = Path(__file__).resolve()
    candidate = here.parents[1] / "infra" / "rayjob.yaml"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")

    raise FileNotFoundError(
        f"RayJob template not found. Tried {p} and {candidate}. "
        f"Set RAYJOB_TEMPLATE_PATH env var."
    )


BASE_RAYJOB_YAML = _load_rayjob_template()


def _indent(text: str, spaces: int = 4) -> str:
    pad = " " * spaces
    lines = (text or "").splitlines()
    return "\n".join(pad + ln for ln in lines) + "\n"


def _run_coro_sync(coro):
    """
    Run an async coroutine from sync code.
    Works even if we're already inside an event loop (FastAPI etc.)
    by running the coroutine in a separate thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


def _dispatch_execution_memory_record(
    session_id: str,
    content: str,
    agent_role: str = "assistant",
    notebook_id: str | None = None,
) -> None:
    """Import the Celery task lazily to avoid a celery_app/execution_agent cycle."""
    try:
        from app.services.milvus_recorder import record_execution_to_milvus

        record_execution_to_milvus.delay(
            session_id,
            content[:2000],
            agent_role,
            notebook_id=notebook_id,
        )
    except Exception as e:
        logger.error(f"Failed to dispatch milvus recording task: {e}")


def _normalize_secret_provider(raw: str, data_uri: str) -> str:
    p = (raw or "").lower().strip()
    if p in ("aws", "s3"):
        return "s3"
    if p in ("gcp", "gcs", "google"):
        return "gcs"
    if p in ("azure", "az"):
        return "azure"

    u = (data_uri or "").lower().strip()
    if u.startswith("s3://"):
        return "s3"
    if u.startswith(("gs://", "gcs://")):
        return "gcs"
    if u.startswith(("az://", "azure://")):
        return "azure"

    return p or "s3"


def _dir_of_uri(uri: str) -> str:
    u = (uri or "").rstrip("/")
    if "://" not in u:
        return u
    if "/" in u.split("://", 1)[1]:
        return u.rsplit("/", 1)[0]
    return u


def render_rayjob_yaml(
    base_yaml: str,
    name: str,
    namespace: str,
    workers: int,
    data_uri: str,
    script_text: str,
    cloud_secret_name: Optional[str] = None,
    output_artifact_uri: str = "",
    output_metrics_uri: str = "",
) -> str:
    name = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")[:63]

    y = base_yaml

    y = y.replace("rayjob-sample-3w", name)
    y = y.replace("rayjob-smoke-1", name)
    y = y.replace("__RAYJOB_NAME__", name)
    y = y.replace("__RAY_NAMESPACE__", namespace)

    y = y.replace("rayjob-sample-3w-code", f"{name}-code")
    y = y.replace("rayjob-smoke-1-code", f"{name}-code")
    y = y.replace("__RAYJOB_NAME__-code", f"{name}-code")

    y = re.sub(r"(?m)^(\s*)namespace:\s*\S+\s*$", rf"\1namespace: {namespace}", y)

    # NEW-RS-01: count lines may end in a comment (the shipped rayjob.yaml's
    # do), so anchor on the key and allow one; the comment goes with the old
    # number. Anchored at end of line, the requested count was ignored.
    # K8-10/K8-11: idle workers scale to zero. minReplicas used to be pinned to
    # the worker count (== maxReplicas), which left the autoscaler no room.
    y = re.sub(r"(?m)^([ \t]*)replicas:[ \t]*\d+[ \t]*(?:#.*)?$", rf"\1replicas: {workers}", y)
    y = re.sub(r"(?m)^([ \t]*)minReplicas:[ \t]*\d+[ \t]*(?:#.*)?$", r"\1minReplicas: 0", y)
    y = re.sub(r"(?m)^([ \t]*)maxReplicas:[ \t]*\d+[ \t]*(?:#.*)?$", rf"\1maxReplicas: {workers}", y)

    y = y.replace("__DATA_SOURCE_URI__", data_uri or "")
    y = y.replace("__OUTPUT_ARTIFACT_URI__", output_artifact_uri or "")
    y = y.replace("__OUTPUT_METRICS_URI__", output_metrics_uri or "")

    if not cloud_secret_name:
        cloud_secret_name = os.getenv("CLOUD_SECRET_NAME", "cloud-creds")
    y = y.replace("__CLOUD_SECRET_NAME__", cloud_secret_name)

    code = _strip_markdown_code_fences(script_text).strip()
    if not code:
        raise ValueError("Empty generated script_text (coder_definition.code missing)")

    pat = re.compile(r"(?ms)(^\s*sample_code\.py:\s*\|\s*\n)(?:^\s{4}.*\n?)*")
    if not pat.search(y):
        raise ValueError("Could not find sample_code.py block in RayJob YAML template")

    y = pat.sub(lambda m: m.group(1) + _indent(code, 4), y, count=1)

    leftovers = re.findall(r"__([A-Z0-9_]+)__", y)
    if leftovers:
        raise ValueError(f"Template render incomplete; leftover placeholders: {sorted(set(leftovers))}")

    return y


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# The ONE CSV reader. It lives in file_handler/null_policy.py and is imported
# here for in-process callers. The scripts built below run where this
# repository may not exist, so they cannot import it: they get that module's
# source spliced in verbatim (_with_csv_reader). There used to be three
# hand-kept copies of this function -- this one and two inside script
# templates -- and a fix to one left the two that actually run the user's
# analysis untouched. Do not paste the function back into a template.
from file_handler import null_policy as _null_policy
from file_handler.null_policy import read_csv_best_effort  # noqa: E402,F401

_CSV_READER_SOURCE = inspect.getsource(_null_policy)
_CSV_READER_MARKER = "# <<AVALOKA_CSV_READER>>"


def _with_csv_reader(script_template: str) -> str:
    """``script_template`` with the reader module's source in place of its marker."""
    if script_template.count(_CSV_READER_MARKER) != 1:
        raise ValueError("script template must contain the CSV reader marker exactly once")
    return script_template.replace(_CSV_READER_MARKER, _CSV_READER_SOURCE)


def _csv_has_data_rows(path: Optional[str]) -> bool:
    if not path:
        return False
    try:
        path_obj = Path(path)
        if not path_obj.exists() or not path_obj.is_file() or path_obj.stat().st_size == 0:
            return False
        return not pd.read_csv(path_obj, nrows=1).empty
    except Exception:
        return False


def _resolve_local_execution_input(state: Dict[str, Any], input_path: Optional[str], output_path: Optional[str]) -> Optional[str]:
    """
    Pick a readable input for local execution. If the selected input is the same
    header-only output.csv that a previous empty run produced, fall back to a
    stable sample/full input instead of repeatedly filtering an empty file.
    """
    input_str = str(input_path) if input_path else None
    output_str = str(output_path) if output_path else None
    if input_str and (input_str != output_str or _csv_has_data_rows(input_str)):
        return input_str

    fallback_keys = (
        "active_data_source_location_local",
        "full_data_location",
        "sample_data_location",
        "data_source_location_local",
    )
    for key in fallback_keys:
        candidate = state.get(key)
        candidate_str = str(candidate) if candidate else None
        if candidate_str and candidate_str != output_str and _csv_has_data_rows(candidate_str):
            logger.info(
                "Using fallback local execution input because selected input is empty or unsafe: %s -> %s",
                input_str,
                candidate_str,
            )
            return candidate_str

    return input_str



def _make_candidate_output_path(
    output_path: Optional[str],
) -> Optional[str]:
    if not output_path:
        return None

    output_obj = Path(output_path)
    output_obj.parent.mkdir(parents=True, exist_ok=True)

    fd, candidate = tempfile.mkstemp(
        suffix=output_obj.suffix or ".csv",
        prefix=f".{output_obj.stem}_candidate_",
        dir=str(output_obj.parent),
    )
    os.close(fd)

    # mkstemp creates a zero-byte file. Remove it so file existence means
    # that the executed script actually produced fresh output.
    try:
        os.unlink(candidate)
    except FileNotFoundError:
        pass

    return candidate


def _replace_first_literal_arg(code: str, call_prefix_pattern: str, new_path: Optional[str]) -> str:
    if not new_path:
        return code
    return re.sub(
        call_prefix_pattern,
        lambda match: f"{match.group(1)}{repr(str(new_path))}",
        code,
        flags=re.MULTILINE,
    )

def _strip_inprocess_scheduling(code: str) -> str:
    """Remove self-scheduling scaffolding from generated ETL code.

    Recurrence is Celery/RedBeat's job (task_scheduler_node). The coder sometimes
    emits the `schedule` pip package plus a `while True: schedule.run_pending()`
    driver, which isn't installed (ModuleNotFoundError) and would block the worker
    forever. Convert both failing shapes into a one-shot `main()` run.
    """
    if "schedule" not in code:
        return code
    # remove the import
    code = re.sub(r"(?m)^\s*import\s+schedule\s*$\n?", "", code)
    # remove any `def schedule_job(...): ...` driver
    code = re.sub(r"(?ms)^def\s+schedule_job\s*\(.*?\):.*?(?=^\S|\Z)", "", code)
    # if a __main__ guard drives the scheduler, replace it with a direct main() call
    if re.search(r"schedule\.(every|run_pending)", code):
        code = re.sub(
            r'(?ms)^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*.*\Z',
            'if __name__ == "__main__":\n    main()\n',
            code,
        )
    return code


# def _strip_markdown_code_fences(code_text: str) -> str:
#     """Return code without surrounding Markdown ```."""
def _strip_markdown_code_fences(code_text: str) -> str:
    """Return code without surrounding Markdown ```."""
    if not code_text:
        return code_text

    stripped = code_text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)

    return code_text


def _sanitize_k8s_name(name: str, max_len: int = 63) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return (s[:max_len].strip("-")) or "rayjob"


# ---------------------------------------------------------------------------
# K8s service helpers
# ---------------------------------------------------------------------------

def get_service_details(infrastructure_provisioned: Dict[str, Any]) -> Dict[str, Any]:
    """Extract service IP and port from infrastructure details"""
    if not infrastructure_provisioned or infrastructure_provisioned.get("status") != "provisioned":
        return {
            "status": "failed",
            "message": "Infrastructure not properly provisioned"
        }

    service_ip = None
    service_port = None

    for outcome in infrastructure_provisioned.get("details", []):
        if outcome.get("step_name") == "Get infra-agent-service IP/Port" and outcome.get("status") == "SUCCESS":
            service_ip = outcome.get("details", {}).get("ip")
            service_port = outcome.get("details", {}).get("port")
            break

    if not service_ip or not service_port:
        return {
            "status": "failed",
            "message": "Service IP and port not found in infrastructure details"
        }

    return {
        "status": "success",
        "service_ip": service_ip,
        "service_port": service_port
    }


def execute_code_on_k8s(
    code: str,
    service_ip: str,
    service_port: str,
    data_source_location: Optional[str] = None,
    output_location: Optional[str] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """POST user code and CSV data to the k8s service for execution."""
    logger.info("Running application on K8s...")
    logger.info(f"Original code:\n{code}")
    print(f"Original data_source_location: \n{data_source_location}")

    modified_code = code

    if data_source_location:
        logger.info("Replacing data_source_location")
        path_str = str(data_source_location)
        path_with_double_backslashes = path_str.replace("\\", "\\\\")
        path_with_forward_slashes = path_str.replace("\\", "/")

        modified_code = modified_code.replace(f"'{path_with_double_backslashes}'", "input_file")
        modified_code = modified_code.replace(f'"{path_with_double_backslashes}"', "input_file")
        modified_code = modified_code.replace(f"'{path_with_forward_slashes}'", "input_file")
        modified_code = modified_code.replace(f'"{path_with_forward_slashes}"', "input_file")
        modified_code = modified_code.replace(f"'{path_str}'", "input_file")
        modified_code = modified_code.replace(f'"{path_str}"', "input_file")
        logger.info("Replaced input path with input_file variable")

    if output_location:
        logger.info("Replacing output_location")
        path_str = str(output_location)
        path_with_double_backslashes = path_str.replace("\\", "\\\\")
        path_with_forward_slashes = path_str.replace("\\", "/")

        modified_code = modified_code.replace(f"'{path_with_double_backslashes}'", "output_file")
        modified_code = modified_code.replace(f'"{path_with_double_backslashes}"', "output_file")
        modified_code = modified_code.replace(f"'{path_with_forward_slashes}'", "output_file")
        modified_code = modified_code.replace(f'"{path_with_forward_slashes}"', "output_file")
        modified_code = modified_code.replace(f"'{path_str}'", "output_file")
        modified_code = modified_code.replace(f'"{path_str}"', "output_file")
        logger.info("Replaced output path with output_file variable")

    payload = {"code": base64.b64encode(modified_code.encode()).decode()}

    if data_source_location:
        data_source_str = str(data_source_location)
        if os.path.exists(data_source_str):
            try:
                with open(data_source_str, "r") as f:
                    csv_str = f.read()
                payload["data"] = base64.b64encode(csv_str.encode()).decode()
                logger.info(f"Added CSV data from {data_source_str}")
            except Exception as e:
                logger.error(f"Unable to load CSV input from {data_source_str}: {e}")

    url = f"http://{service_ip}:{service_port}/execute"
    start = time.time()

    try:
        logger.info(f"Sending code execution request to {url}")
        logger.info(f"Modified code being sent:\n{modified_code}")
        res = requests.post(url, json=payload, timeout=timeout)
        logger.info(f"Response status code: {res.status_code}")
        logger.info(f"Response headers: {dict(res.headers)}")
        res.raise_for_status()
        out = res.json()
        logger.info(out)
        logger.info("Code execution completed successfully")
    except requests.exceptions.RequestException as exc:
        logger.error(f"Request failed: {exc}")
        out = {"status": "error", "message": str(exc)}

    out["execution_time"] = round(time.time() - start, 2)
    return out


# ---------------------------------------------------------------------------
# The execution_result log line  (every backend, persisted or not)
# ---------------------------------------------------------------------------

#: Fields of the ``execution_result`` log line, in the order they are written.
#: Behaviour only: every value is a literal from this module, an exception
#: class name, a number, or a band from app.analytics.schema. Nothing here is
#: read out of the data, the code, the question or a path, so a column name, a
#: file name or a cell value has no way into it.
EXECUTION_RESULT_LOG_KEYS = (
    "backend", "outcome", "error_class", "duration_ms",
    "row_count_bucket", "column_count_bucket", "fresh_output", "output_kind",
)


def _log_execution_result(
    backend: str,
    outcome: str,
    started: float,
    *,
    error_class: str = "none",
    output_data: Any = None,
    output_kind: str = "none",
) -> None:
    """Write the one line that says how an execution ended.

    The result is logged whether or not anything downstream persists it, so a
    run is visible in ``kubectl logs`` on every backend. ``started`` is a
    ``time.monotonic()`` reading taken when the node was entered.
    """
    has_output = output_data is not None
    fields = {
        "backend": backend,
        "outcome": outcome,
        "error_class": error_class,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "row_count_bucket": row_count_bucket(len(output_data)) if has_output else "none",
        "column_count_bucket": (
            column_count_bucket(len(output_data.columns)) if has_output else "none"
        ),
        "fresh_output": has_output and outcome == "success",
        "output_kind": output_kind,
    }
    logger.info(
        "execution_result " + " ".join(f"{key}=%s" for key in EXECUTION_RESULT_LOG_KEYS),
        *(fields[key] for key in EXECUTION_RESULT_LOG_KEYS),
    )


# ---------------------------------------------------------------------------
# execution_agent_node  (k8s service path)
# ---------------------------------------------------------------------------

def execution_agent_node(state: ETLState) -> ETLState:
    """LangGraph node that executes Python code using Kubernetes infrastructure"""
    logger.info("Execution Agent: Starting k8s execution process")
    started = time.monotonic()
    code = state.get("coder_definition", {}).get("code", "")
    if not code:
        message = "No code provided for execution"
        logger.error(message)
        new_state = state.copy()
        new_state.update({
            "messages": new_state["messages"] + [AIMessage(content=message)],
            "execution_result": {"status": "failed", "message": message}
        })
        _log_execution_result("k8s", "failed", started, error_class="NoCode")
        return new_state

    cleaned_code = _strip_markdown_code_fences(code)

    infrastructure_provisioned = state.get("infrastructure_provisioned", {})
    service_details = get_service_details(infrastructure_provisioned)

    if service_details.get("message") == "Infrastructure not properly provisioned":
        logger.warning("Infrastructure not properly provisioned; attempting reprovision on GCP python-docker.")
        state.update({
            "infrastructure_request": {"type": "gcp", "app_type": "python-docker"}
        })
        state = infra_agent_node(state)
        infrastructure_provisioned = state.get("infrastructure_provisioned", {})
        service_details = get_service_details(infrastructure_provisioned)
        logger.debug("Service details after reprovision attempt: %s", service_details)

    if service_details.get("status") != "success":
        message = f"Infrastructure service not available: {service_details.get('message')}"
        new_state = state.copy()
        new_state.update({
            "messages": new_state.get("messages", []) + [AIMessage(content=message)],
            "execution_result": {"status": "failed", "message": message},
        })
        _log_execution_result(
            "k8s", "failed", started, error_class="InfrastructureUnavailable"
        )
        return new_state

    data_source_location = state.get("data_source_location")
    output_location = state.get("output_location")
    service_ip = service_details["service_ip"]
    service_port = service_details["service_port"]
    logger.info(f"Using service at {service_ip}:{service_port}")

    execution_result = execute_code_on_k8s(
        code=cleaned_code,
        service_ip=service_ip,
        service_port=service_port,
        data_source_location=data_source_location,
        output_location=output_location,
    )

    new_state = state.copy()

    file_data = None
    output_data = None
    # Why this run is not a success, as a class name; None while it still is.
    error_class = None
    returned_output = bool(
        execution_result.get("status") == "success" and execution_result.get("output")
    )
    if returned_output:
        try:
            output_data_str = execution_result["output"]
            if output_data_str.startswith("data:text/csv;base64,"):
                csv_str = base64.b64decode(output_data_str.split("base64,")[1]).decode()
                output_data = pd.read_csv(StringIO(csv_str))
                if not output_data.empty:
                    ai_response = AIMessage(content=output_data.to_markdown())
                else:
                    logger.info(f"Empty dataset returned")
                    ai_response = AIMessage(f"The resulting output dataset is empty")

                if execution_result.get("output_file"):
                    output_file_info = execution_result["output_file"]
                    file_data = {
                        "filename": output_file_info["filename"],
                        "content": output_file_info["content"],
                        "size": output_file_info["size"]
                    }
                    logger.info(f"Prepared file data for download: {output_file_info['filename']}")
                else:
                    # The pod's filesystem is not ours, so the returned CSV is
                    # the only copy. Carry it as the file, as the local path
                    # does, rather than leave a success with nothing attached.
                    file_data = {
                        "filename": f"{generate_filename_timestamp()}_output.csv",
                        "content": output_data_str,
                        "size": len(csv_str.encode("utf-8")),
                    }
            else:
                failure_message = "Code executed successfully but no valid CSV output received"
                ai_response = AIMessage(content=failure_message)
                execution_error = failure_message
                error_class = "NoOutput"
        except Exception as e:
            logger.error(f"Failed to process output: {e}")
            ai_response = AIMessage(content=(
            f"The step ran, but I couldn't prepare the output: {_brief_error(e)}"
        ))
            failure_message = str(e)
            execution_error = failure_message
            error_class = type(e).__name__
            file_data = None
    else:
        # Full stderr to the logs; the chat gets one readable line.
        error_msg = execution_result.get("message", "Unknown error")
        stderr = execution_result.get("stderr") or ""
        if stderr:
            logger.error("Execution failed. stderr:\n%s", stderr)
        ai_response = AIMessage(content=_friendly_failure_message(stderr or error_msg))
        failure_message = f"{error_msg}\n\n{stderr.strip()}" if stderr.strip() else error_msg
        execution_error = error_msg
        error_class = "ExecutionFailed"

    succeeded = error_class is None

    # The same keys execution_agent_node_local writes on each outcome, so a
    # reader of execution_result never has to know which backend ran.
    if succeeded:
        result_for_state = {
            "status": "success",
            "message": "ETL job completed successfully",
            # A path inside the executor pod, not on this host: nothing here
            # can open it. The content travels in output_file_data.
            "output_file": str(output_location) if output_location else file_data["filename"],
            "fresh_output": True,
            "output_row_count": len(output_data),
        }
    elif returned_output:
        result_for_state = {
            "status": "failed",
            "message": failure_message,
            "output_file": None,
            "fresh_output": False,
            "output_row_count": None,
        }
    else:
        result_for_state = {
            "status": "failed",
            "message": failure_message,
            "returncode": execution_result.get("returncode"),
            "output_file": None,
        }

    new_state.update({
        "messages": new_state["messages"] + [ai_response],
        "output_location": output_location,
        "output_file_data": file_data,
        "output_json": output_data.to_dict(orient="records") if succeeded else None,
        "fresh_output_produced": succeeded,
        "execution_stdout": execution_result.get("stdout"),
        "execution_stderr": execution_result.get("stderr"),
        "execution_error": None if succeeded else execution_error,
        "execution_result": result_for_state,
    })
    _log_execution_result(
        "k8s",
        "success" if succeeded else "failed",
        started,
        error_class=error_class or "none",
        output_data=output_data if succeeded else None,
        output_kind="inline" if succeeded else "none",
    )
    if succeeded:
        lineage_dataset_id = record_analysis_lineage(new_state, output_data)
        if lineage_dataset_id:
            new_state["latest_output_dataset_id"] = lineage_dataset_id

    # Store result in Layer 4 memory if successful
    if file_data is not None or "successfully" in getattr(ai_response, "content", ""):
        session_id = new_state.get("session_id", "default")
        content_to_embed = getattr(ai_response, "content", "Execution completed successfully.")

        # Determine the textual intent of the query for layer 3 hashing
        query_intent = ""
        user_msgs = [m.content for m in new_state.get("messages", []) if getattr(m, "type", "") == "human" or type(m).__name__ == "HumanMessage"]
        if user_msgs:
            query_intent = str(user_msgs[-1])

        milvus_context_id = new_state.get("milvus_context_id", "default_notebook")
        _dispatch_execution_memory_record(
            session_id,
            content_to_embed,
            "assistant",
            notebook_id=milvus_context_id,
        )

        try:
            from app.services.db.postgres_client import postgres_client
            postgres_client.connect()
            postgres_client.record_artifact(
                query=query_intent,
                artifact_metadata={"file_data": file_data, "status": "success", "session": session_id,
                                   "thread_id": new_state.get("thread_id")},
                user_id=new_state.get("user_id", "default"),
            )
        except Exception as e:
            logger.error(f"Failed to cleanly record Layer 3 artifact redundancy context: {e}")

    return new_state


# ---------------------------------------------------------------------------
# execution_agent_node_local
# ---------------------------------------------------------------------------

def execution_agent_node_local(state: ETLState) -> ETLState:
    """LangGraph node that executes Python code locally.

    When the turn was meant for Ray (``ray_fallback_reason`` is set), the reply
    and the run record say that it ran locally, on how many rows, and why.
    """
    reason = state.get("ray_fallback_reason")
    new_state = _execution_agent_node_local(state)
    if reason:
        new_state = _label_ray_local_fallback(new_state, state, reason)
    return new_state


def _execution_agent_node_local(state: ETLState) -> ETLState:
    logger.info("Execution Agent: Starting local execution process")
    started = time.monotonic()

    code = state.get("coder_definition", {}).get("code", "")
    if not code:
        message = "No code provided for execution"
        logger.error(message)
        new_state = state.copy()
        new_state.update({
            "messages": new_state["messages"] + [AIMessage(content=message)],
            "execution_result": {"status": "failed", "message": message}
        })
        _log_execution_result("local", "failed", started, error_class="NoCode")
        return new_state

    input_data_location = state.get("data_source_location")
    output_location = state.get("output_location")
    resolved_input_location = _resolve_local_execution_input(state, input_data_location, output_location)
    candidate_output_location = _make_candidate_output_path(output_location)

    read_path_map = _resolve_read_paths(state)
    if read_path_map:
        logger.info("[local-exec] multi-file read map: %s", read_path_map)

    execution_result = execute_code_on_local(
        code=code,
        input_data_location=resolved_input_location,
        output_location=candidate_output_location,
        read_path_map=read_path_map,
    )

    new_state = state.copy()

    if execution_result.get("status") == "error":
        stderr = execution_result.get("execution_stderr") or ""
        stdout = execution_result.get("execution_stdout") or ""
        error_msg = execution_result.get("execution_error") or "Code execution failed"
        details = stderr.strip() or stdout.strip()
        if details:
            # Keep the full detail in state/logs for the repair loop; the chat
            # message below carries only a one-line reason.
            logger.error("Execution failed. Details:\n%s", details)
            error_msg = f"{error_msg}\n\n{details}"
        ai_response = AIMessage(content=_friendly_failure_message(details or error_msg))
        new_state.update({
            "messages": new_state.get("messages", []) + [ai_response],
            "output_location": output_location,
            "output_file_data": None,
            "output_json": None,
            "fresh_output_produced": False,
            "execution_stdout": stdout,
            "execution_stderr": stderr,
            "execution_error": execution_result.get("execution_error"),
            "execution_result": {
                "status": "failed",
                "message": error_msg,
                "returncode": execution_result.get("returncode"),
                "output_file": None,
            },
        })
        _log_execution_result("local", "failed", started, error_class="ExecutionFailed")
        return new_state

    coder_message = state.get("coder_raw_response")
    if coder_message:
        existing_messages = new_state.get("messages", [])
        if not any(getattr(msg, "content", None) == coder_message for msg in existing_messages):
            new_state["messages"] = existing_messages + [AIMessage(content=coder_message)]

    if not new_state.get("generated_code"):
        generated_code = state.get("coder_definition", {}).get("code")
        if generated_code:
            new_state["generated_code"] = generated_code

    file_data = None
    ai_response = None
    output_data = None

    # Inputs for result rendering.
    _user_msgs = [
        m.content
        for m in state.get("messages", [])
        if type(m).__name__ == "HumanMessage"
    ]
    _user_question = str(_user_msgs[-1]) if _user_msgs else ""
    _render_schema = state.get("schema")

    result_output_location = candidate_output_location

    try:
        if not result_output_location:
            raise RuntimeError(
                "No candidate output path was configured"
            )

        if not os.path.exists(result_output_location):
            raise RuntimeError(
                "Execution finished without creating a fresh output file"
            )

        output_size = os.path.getsize(result_output_location)

        if output_size == 0:
            raise RuntimeError(
                "Execution created a zero-byte output file"
            )

        logger.info(
            "[local-output] candidate exists=True size=%d",
            output_size,
        )

        output_data = pd.read_csv(result_output_location)

        logger.info(
            "[local-output] fresh candidate rows=%d columns=%d",
            len(output_data),
            len(output_data.columns),
        )

        ai_response = AIMessage(
            content=render_response(
                execution_result,
                output_data,
                _user_question,
                _render_schema,
            )
        )

        # Atomically replace the previous canonical output.csv.
        if output_location:
            os.replace(
                result_output_location,
                output_location,
            )
            result_output_location = output_location

        if not result_output_location or not os.path.exists(
            result_output_location
        ):
            raise RuntimeError(
                "Fresh output could not be promoted to the canonical path"
            )

        with open(
            result_output_location,
            "r",
            encoding="utf-8",
        ) as output_file:
            csv_content = output_file.read()

        timestamp = generate_filename_timestamp()
        filename = f"{timestamp}_output.csv"

        csv_bytes = csv_content.encode("utf-8")
        base64_content = base64.b64encode(
            csv_bytes
        ).decode("utf-8")

        file_data = {
            "filename": filename,
            "content": (
                "data:text/csv;base64,"
                f"{base64_content}"
            ),
            "size": len(csv_bytes),
        }

        output_json = output_data.to_dict(
            orient="records"
        )

        execution_status = "success"
        execution_message = "ETL job completed successfully"

        logger.info(
            "[local-output] promoted fresh output: rows=%d",
            len(output_data),
        )

    except Exception as exc:
        logger.exception(
            "[local-output] failed to process fresh candidate"
        )

        ai_response = AIMessage(
            content=(
                "The code ran, but the new output could not "
                f"be prepared: {_brief_error(exc)}"
            )
        )

        new_state.update({
            "messages": (
                new_state.get("messages", [])
                + [ai_response]
            ),
            "output_location": output_location,
            "output_file_data": None,
            "output_json": None,
            "fresh_output_produced": False,
            "execution_stdout": execution_result.get(
                "execution_stdout"
            ),
            "execution_stderr": execution_result.get(
                "execution_stderr"
            ),
            "execution_error": str(exc),
            "execution_result": {
                "status": "failed",
                "message": str(exc),
                "output_file": None,
                "fresh_output": False,
                "output_row_count": None,
            },
        })
        _log_execution_result(
            "local", "failed", started, error_class=type(exc).__name__
        )

        return new_state

    finally:
        # Delete only an unpromoted temporary candidate.
        # After os.replace(), the candidate path no longer exists.
        if (
            candidate_output_location
            and candidate_output_location != output_location
            and os.path.exists(candidate_output_location)
        ):
            try:
                os.remove(candidate_output_location)
            except Exception:
                logger.warning(
                    "Could not remove candidate output %s",
                    candidate_output_location,
                    exc_info=True,
                )


    new_state.update({
        "messages": (
            new_state.get("messages", [])
            + [ai_response]
        ),
        "output_location": output_location,
        "output_file_data": file_data,
        "output_json": output_json,
        "fresh_output_produced": True,
        "execution_stdout": execution_result.get(
            "execution_stdout"
        ),
        "execution_stderr": execution_result.get(
            "execution_stderr"
        ),
        "execution_error": None,
        "execution_result": {
            "status": execution_status,
            "message": execution_message,
            "output_file": (
                str(output_location)
                if output_location
                else result_output_location
            ),
            "fresh_output": True,
            "output_row_count": len(output_data),
        },
    })
    _log_execution_result(
        "local", "success", started, output_data=output_data, output_kind="file"
    )
    if execution_status == "success":
        lineage_dataset_id = record_analysis_lineage(new_state, output_data)
        if lineage_dataset_id:
            new_state["latest_output_dataset_id"] = lineage_dataset_id

    # Store result in Layer 4 memory if successful
    #if execution_status == "success" or execution_status == "completed":
    if execution_status == "success":
        session_id = new_state.get("session_id", "default")
        content_to_embed = getattr(ai_response, "content", "Execution completed successfully.")

        query_intent = ""
        user_msgs = [m.content for m in new_state.get("messages", []) if getattr(m, "type", "") == "human" or type(m).__name__ == "HumanMessage"]
        if user_msgs:
            query_intent = str(user_msgs[-1])

        milvus_context_id = new_state.get("milvus_context_id", "default_notebook")
        _dispatch_execution_memory_record(
            session_id,
            content_to_embed,
            "assistant",
            notebook_id=milvus_context_id,
        )

        try:
            from app.services.db.postgres_client import postgres_client
            postgres_client.connect()
            postgres_client.record_artifact(
                query=query_intent,
                artifact_metadata={"file_data": file_data, "status": execution_status, "session": session_id,
                                   "thread_id": new_state.get("thread_id")},
                user_id=new_state.get("user_id", "default"),
            )
        except Exception as e:
            logger.error(f"Failed to cleanly record Layer 3 artifact redundancy context: {e}")

    return new_state


_SCHEDULED_EXECUTION_DEPLOYMENT = os.getenv(
    "AVALOKA_SCHEDULED_EXECUTION_DEPLOYMENT", "avaloka-celery-executor"
)
_SCHEDULED_EXECUTION_SELECTOR = os.getenv(
    "AVALOKA_SCHEDULED_EXECUTION_SELECTOR",
    "app.kubernetes.io/component=celery-executor",
)
_SCHEDULED_EXECUTION_SCRIPT = "\n".join([
    "import contextlib, io, json, sys",
    "from app.agents.execution_agent import execution_agent_node_local",
    "from app.utils import convert_message_dicts_to_objects, convert_message_objects_to_dicts",
    "state = json.load(sys.stdin)",
    "state['messages'] = convert_message_dicts_to_objects(state.get('messages') or [])",
    "with contextlib.redirect_stdout(io.StringIO()):",
    "    result = execution_agent_node_local(state)",
    "result['messages'] = convert_message_objects_to_dicts(result.get('messages') or [])",
    "print('AVALOKA_EXECUTION_RESULT:' + json.dumps(result, default=str))",
])


def _run_scheduled_pod_command(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["kubectl", *args], capture_output=True, text=True, **kwargs
    )
    if result.returncode:
        detail = (result.stderr or result.stdout or "kubectl command failed").strip()
        raise RuntimeError(detail)
    return result


def _get_scheduled_execution_pod() -> tuple[str, str]:
    namespace = os.getenv(
        "AVALOKA_SCHEDULED_EXECUTION_NAMESPACE",
        os.getenv("POD_NAMESPACE", os.getenv("K8S_NAMESPACE", "default")),
    )
    _run_scheduled_pod_command(
        [
            "rollout", "status", f"deployment/{_SCHEDULED_EXECUTION_DEPLOYMENT}",
            "-n", namespace, "--timeout=180s",
        ],
        timeout=190,
    )
    result = _run_scheduled_pod_command(
        [
            "get", "pods", "-n", namespace, "-l", _SCHEDULED_EXECUTION_SELECTOR,
            "-o", "jsonpath={.items[0].metadata.name}",
        ],
        timeout=30,
    )
    pod_name = result.stdout.strip()
    if not pod_name:
        raise RuntimeError("Scheduled execution Deployment has no ready pod")
    return namespace, pod_name


_SCHEDULED_INPUT_PATH_FIELDS = (
    "data_source_location",
    "active_data_source_location",
    "active_data_source_location_local",
    "full_data_location",
    "sample_data_location",
    "data_source_location_local",
)


def _is_shared_storage_uri(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme.lower() in {"gs", "s3", "az", "azure"} or (
        parsed.hostname or ""
    ).endswith(".blob.core.windows.net")


def _scheduled_cloud_input_pairs(state: dict) -> list[tuple[Optional[str], str]]:
    pairs: list[tuple[Optional[str], str]] = []
    for local_key, cloud_key in (
        ("data_source_location_local", "data_source_location_cloud"),
        ("active_data_source_location_local", "active_data_source_location_cloud"),
        ("full_data_location", "full_data_source_location_cloud"),
    ):
        uri = state.get(cloud_key)
        if _is_shared_storage_uri(uri):
            pairs.append((state.get(local_key), uri))

    for key in ("data_source_location", "active_data_source_location"):
        value = state.get(key)
        if _is_shared_storage_uri(value):
            pairs.append((None, value))

    for dataset in state.get("multi_dataset_state") or []:
        if not isinstance(dataset, dict):
            continue
        uri = dataset.get("active_data_source_location_cloud")
        if not _is_shared_storage_uri(uri):
            uri = next((
                value for value in (
                    dataset.get("active_data_source_location"),
                    dataset.get("data_source_location"),
                ) if _is_shared_storage_uri(value)
            ), None)
        if _is_shared_storage_uri(uri):
            local_path = dataset.get("data_source_location")
            if _is_shared_storage_uri(local_path):
                local_path = None
            pairs.append((local_path, uri))
    return pairs


def _replace_scheduled_input_paths(state: dict, replacements: dict[str, str]) -> None:
    for key in _SCHEDULED_INPUT_PATH_FIELDS:
        value = state.get(key)
        if isinstance(value, str) and value in replacements:
            state[key] = replacements[value]

    for dataset in state.get("multi_dataset_state") or []:
        if not isinstance(dataset, dict):
            continue
        for key in _SCHEDULED_INPUT_PATH_FIELDS:
            value = dataset.get(key)
            if isinstance(value, str) and value in replacements:
                dataset[key] = replacements[value]

    coder_definition = dict(state.get("coder_definition") or {})
    code = coder_definition.get("code")
    if isinstance(code, str):
        for old_path, new_path in sorted(
            replacements.items(), key=lambda item: len(item[0]), reverse=True
        ):
            code = code.replace(old_path, new_path)
        coder_definition["code"] = code
    if coder_definition:
        state["coder_definition"] = coder_definition


def _restore_scheduled_input_paths(result: dict, original: dict) -> None:
    for key in _SCHEDULED_INPUT_PATH_FIELDS:
        if key in original:
            result[key] = original[key]
    original_datasets = original.get("multi_dataset_state") or []
    result_datasets = result.get("multi_dataset_state") or []
    for index, dataset in enumerate(original_datasets):
        if index >= len(result_datasets) or not isinstance(dataset, dict):
            continue
        for key in _SCHEDULED_INPUT_PATH_FIELDS:
            if key in dataset:
                result_datasets[index][key] = dataset[key]
    if "coder_definition" in original:
        result["coder_definition"] = original["coder_definition"]


def _validate_scheduled_local_inputs(state: dict, staged_paths: set[str]) -> None:
    candidates = [("dataset", state.get("data_source_location"))]
    candidates.extend(
        ("multi-dataset input", dataset.get("data_source_location"))
        for dataset in state.get("multi_dataset_state") or []
        if isinstance(dataset, dict)
    )
    for label, path in candidates:
        if (
            path
            and not _is_shared_storage_uri(path)
            and not os.path.isfile(str(path))
            and path not in staged_paths
        ):
            raise RuntimeError(
                f"Scheduled {label} is not available in the Celery pod: {path}. "
                "Use a shared object-storage URI (gs://, s3://, or Azure Blob) "
                "so the scheduled worker can stage the input."
            )


def execution_agent_node_scheduled_pod(state: ETLState) -> ETLState:
    """Run scheduled local code in a persistent Kubernetes pod."""
    new_state = state.copy()
    staging = tempfile.TemporaryDirectory(prefix="avaloka-scheduled-")
    namespace = None
    pod_name = None
    remote_root = f"/tmp/avaloka-scheduled-inputs/{os.path.basename(staging.name)}"
    original_output_path = state.get("output_location")
    remote_output_path = None
    try:
        namespace, pod_name = _get_scheduled_execution_pod()

        from app.services import storage_service

        pod_state = json.loads(json.dumps(state, default=str))
        if original_output_path:
            remote_output_path = (
                f"{remote_root}/output/{os.path.basename(str(original_output_path))}"
            )
            pod_state["output_location"] = remote_output_path
            coder_definition = dict(pod_state.get("coder_definition") or {})
            code = coder_definition.get("code")
            if isinstance(code, str):
                coder_definition["code"] = code.replace(
                    str(original_output_path), remote_output_path
                )
                pod_state["coder_definition"] = coder_definition

        replacements: dict[str, str] = {}
        copy_specs: list[tuple[str, str]] = []
        staged_uris: dict[str, str] = {}
        for index, (local_path, cloud_uri) in enumerate(
            _scheduled_cloud_input_pairs(state)
        ):
            if local_path and os.path.isfile(str(local_path)):
                continue
            remote_path = staged_uris.get(cloud_uri)
            if remote_path is None:
                store, object_key = storage_service._store_and_key_from_uri(
                    cloud_uri, None
                )
                if store is None or not object_key:
                    raise RuntimeError(
                        f"Could not resolve scheduled input storage URI: {cloud_uri}"
                    )
                filename = os.path.basename(urlparse(cloud_uri).path) or f"input-{index}"
                staged_path = Path(staging.name) / f"{index}-{filename}"
                store.get_file(object_key, staged_path)
                remote_path = f"{remote_root}/{index}-{filename}"
                staged_uris[cloud_uri] = remote_path
                copy_specs.append((str(staged_path), remote_path))
            replacements[cloud_uri] = remote_path
            if local_path:
                replacements[str(local_path)] = remote_path

        _replace_scheduled_input_paths(pod_state, replacements)
        _validate_scheduled_local_inputs(pod_state, set(replacements.values()))

        input_paths = [
            state.get("data_source_location"),
            state.get("active_data_source_location_local"),
            state.get("full_data_location"),
            state.get("sample_data_location"),
            state.get("data_source_location_local"),
        ]
        input_paths.extend(
            dataset.get("data_source_location")
            for dataset in (state.get("multi_dataset_state") or [])
            if isinstance(dataset, dict)
        )
        for input_path in dict.fromkeys(path for path in input_paths if path):
            if _is_shared_storage_uri(input_path):
                continue
            local_path = os.path.abspath(str(input_path))
            if not os.path.isfile(local_path):
                continue
            copy_specs.append((local_path, local_path))

        copied_destinations = set()
        for source_path, remote_path in copy_specs:
            if remote_path in copied_destinations:
                continue
            copied_destinations.add(remote_path)
            _run_scheduled_pod_command(
                ["exec", "-n", namespace, pod_name, "--", "mkdir", "-p", os.path.dirname(remote_path)],
                timeout=30,
            )
            _run_scheduled_pod_command(
                ["cp", "-n", namespace, source_path, f"{pod_name}:{remote_path}"],
                timeout=300,
            )

        from app.utils import convert_message_objects_to_dicts

        pod_state["messages"] = convert_message_objects_to_dicts(
            state.get("messages") or []
        )
        result = _run_scheduled_pod_command(
            [
                "exec", "-i", "-n", namespace, pod_name, "--", "python", "-c",
                _SCHEDULED_EXECUTION_SCRIPT,
            ],
            input=json.dumps(pod_state, default=str),
            timeout=int(os.getenv("AVALOKA_SCHEDULED_EXECUTION_TIMEOUT", "300")),
        )
        result_prefix = "AVALOKA_EXECUTION_RESULT:"
        result_line = next(
            (line for line in reversed(result.stdout.splitlines()) if line.startswith(result_prefix)),
            None,
        )
        if result_line is None:
            raise RuntimeError("Scheduled execution pod returned no execution result")

        from app.utils import convert_message_dicts_to_objects

        pod_result = json.loads(result_line[len(result_prefix):])
        pod_result["messages"] = convert_message_dicts_to_objects(
            pod_result.get("messages") or []
        )
        _restore_scheduled_input_paths(pod_result, state)

        output_path = original_output_path or pod_result.get("output_location")
        if output_path and remote_output_path:
            remote_path = os.path.abspath(remote_output_path)
            exists = subprocess.run(
                ["kubectl", "exec", "-n", namespace, pod_name, "--", "test", "-f", remote_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if exists.returncode == 0:
                os.makedirs(os.path.dirname(os.path.abspath(str(output_path))), exist_ok=True)
                _run_scheduled_pod_command(
                    ["cp", "-n", namespace, f"{pod_name}:{remote_path}", str(output_path)],
                    timeout=300,
                )
        if original_output_path:
            pod_result["output_location"] = original_output_path
            generated_code = pod_result.get("generated_code")
            if isinstance(generated_code, str) and remote_output_path:
                pod_result["generated_code"] = generated_code.replace(
                    remote_output_path, str(original_output_path)
                )
        return pod_result
    except Exception as exc:
        logger.exception("Scheduled pod execution failed")
        new_state.update({
            "messages": new_state.get("messages", []) + [
                AIMessage(content=f"Scheduled execution pod failed: {exc}")
            ],
            "execution_result": {"status": "failed", "message": str(exc)},
            "output_file_data": None,
        })
        return new_state
    finally:
        if namespace and pod_name:
            try:
                _run_scheduled_pod_command(
                    ["exec", "-n", namespace, pod_name, "--", "rm", "-rf", remote_root],
                    timeout=30,
                )
            except Exception:
                logger.warning("Failed to clean scheduled input staging directory %s", remote_root)
        staging.cleanup()


# ---------------------------------------------------------------------------
# execution_agent_node_ssh  (from develop-1.2)
# ---------------------------------------------------------------------------

def execution_agent_node_ssh(state: ETLState, server: str) -> ETLState:
    """LangGraph node that executes Python code via SSH"""
    logger.info("Execution Agent: Starting ssh execution process")

    code = state.get("coder_definition", {}).get("code", "")
    if not code:
        message = "No code provided for execution"
        logger.error(message)
        new_state = state.copy()
        new_state.update({
            "messages": new_state["messages"] + [AIMessage(content=message)],
            "execution_result": {"status": "failed", "message": message}
        })
        return new_state

    input_data_location = state.get("data_source_location")
    output_location = state.get("output_location")

    execution_result = execute_code_on_ssh(
        code=code,
        server=server,
        input_data_location=os.path.dirname(input_data_location),
        output_location=os.path.dirname(output_location),
    )

    p = subprocess.run(["scp", f"{server}:{output_location}", output_location], capture_output=True)
    if p.returncode != 0:
        stdout = p.stdout or ""
        stderr = p.stderr or ""
        logger.info("Failed To Copy Output file")
        logger.error("Interpreter failed (rc=%s)", p.returncode)
        logger.error("Stdout:\n%s", stdout)
        logger.error("Stderr:\n%s", stderr)

    new_state = state.copy()

    coder_message = state.get("coder_raw_response")
    if coder_message:
        existing_messages = new_state.get("messages", [])
        if not any(getattr(msg, "content", None) == coder_message for msg in existing_messages):
            new_state["messages"] = existing_messages + [AIMessage(content=coder_message)]

    if not new_state.get("generated_code"):
        generated_code = state.get("coder_definition", {}).get("code")
        if generated_code:
            new_state["generated_code"] = generated_code

    file_data = None
    ai_response = None
    # Inputs for type-directed result rendering (conclusion only when warranted).
    _user_msgs = [m.content for m in state.get("messages", []) if type(m).__name__ == "HumanMessage"]
    _user_question = str(_user_msgs[-1]) if _user_msgs else ""
    _render_schema = state.get("schema")
    try:
        if output_location and os.path.exists(output_location):
            output_data = pd.read_csv(output_location)
            if not output_data.empty:
                logger.info(f"output data is {output_data.head().to_markdown()}")
                ai_response = AIMessage(content=render_response(
                    execution_result, output_data, _user_question, _render_schema
                ))
            else:
                logger.info(f"Empty dataset returned")
                ai_response = AIMessage(content=render_response(
                    execution_result, output_data, _user_question, _render_schema
                ))

            with open(output_location, 'r') as f:
                csv_content = f.read()

            timestamp = generate_filename_timestamp()
            filename = f"{timestamp}_output.csv"
            file_size = len(csv_content.encode('utf-8'))
            base64_content = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')

            file_data = {
                "filename": filename,
                "content": f"data:text/csv;base64,{base64_content}",
                "size": file_size
            }
            logger.info(f"Prepared file data for download: {filename}")
        else:
            ai_response = AIMessage(content=render_response(
                execution_result, None, _user_question, _render_schema
            ))

    except Exception as e:
        logger.error(f"Failed to process output file: {e}")
        ai_response = AIMessage(content=(
            f"The step ran, but I couldn't prepare the output: {_brief_error(e)}"
        ))
        new_state.update({
            "messages": new_state["messages"] + [ai_response],
            "output_location": output_location,
            "execution_result": {
                "status": "error",
                "message": f"Failed to process output file: {str(e)}",
                "output_file": str(output_location) if output_location else None
            },
            "output_file_data": file_data,
        })
        return new_state

    if ai_response is None:
        logger.warning("ai_response is None after output file processing - using fallback message")
        ai_response = AIMessage(content="Code executed but output processing result is unknown")

    if file_data is not None:
        execution_status = "success"
        execution_message = "ETL job completed successfully"
    else:
        execution_status = "completed"
        execution_message = "Code executed but no output file was generated"

    new_state.update({
        "messages": new_state["messages"] + [ai_response],
        "output_location": output_location,
        "execution_result": {
            "status": execution_status,
            "message": execution_message,
            "output_file": str(output_location) if output_location and file_data is not None else None
        },
        "output_file_data": file_data,
    })

    # Store result in Layer 4 memory if successful
    if execution_status == "success" or execution_status == "completed":
        session_id = new_state.get("session_id", "default")
        content_to_embed = getattr(ai_response, "content", "Execution completed successfully.")

        query_intent = ""
        user_msgs = [m.content for m in new_state.get("messages", []) if getattr(m, "type", "") == "human" or type(m).__name__ == "HumanMessage"]
        if user_msgs:
            query_intent = str(user_msgs[-1])

        milvus_context_id = new_state.get("milvus_context_id", "default_notebook")
        _dispatch_execution_memory_record(
            session_id,
            content_to_embed,
            "assistant",
            notebook_id=milvus_context_id,
        )

        try:
            from app.services.db.postgres_client import postgres_client
            postgres_client.connect()
            postgres_client.record_artifact(
                query=query_intent,
                artifact_metadata={"file_data": file_data, "status": execution_status, "session": session_id,
                                   "thread_id": new_state.get("thread_id")},
                user_id=new_state.get("user_id", "default"),
            )
        except Exception as e:
            logger.error(f"Failed to cleanly record Layer 3 artifact redundancy context: {e}")

    return new_state


# ---------------------------------------------------------------------------
# Ray-specific helpers
# ---------------------------------------------------------------------------

RAY_NO_CONNECTION_REASON = "no cloud storage connection is linked to this dataset"


def _is_scheduled_run(state: Dict[str, Any]) -> bool:
    """True inside a Celery scheduled run (the scheduler tags schedule_id)."""
    from app.core.scheduled_run_context import current_scheduled_run
    return bool(state.get("schedule_id")) or current_scheduled_run() is not None


def _local_input_row_count(path: Optional[str]) -> Optional[int]:
    """Data rows in a local CSV or Parquet file; None when not cheaply known."""
    try:
        p = str(path or "")
        if not p or not os.path.isfile(p):
            return None
        if p.lower().endswith(".parquet"):
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(p).metadata.num_rows)
        if p.lower().endswith((".csv", ".tsv", ".txt")):
            breaks, last = 0, b"\n"
            with open(p, "rb") as fh:
                for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                    breaks += block.count(b"\n")
                    last = block[-1:]
            return max(0, breaks - 1 + (0 if last == b"\n" else 1))
    except Exception:
        logger.debug("row count failed for %s", path, exc_info=True)
    return None


def ray_local_fallback_note(reason: str, rows: Optional[int]) -> str:
    """The line a reply carries when a turn meant for Ray ran locally."""
    where = f"on {rows:,} rows" if rows is not None else "on the local copy of the data"
    return f"\n\n> Note: this ran locally {where}, not on Ray, because {reason}."


def _label_ray_local_fallback(new_state: dict, state: dict, reason: str) -> dict:
    """Put a Ray-to-local fallback in the reply and on the run record."""
    input_path = _resolve_local_execution_input(
        state, state.get("data_source_location"), state.get("output_location")
    )
    rows = _local_input_row_count(input_path)
    note = ray_local_fallback_note(reason, rows)
    new_state = dict(new_state)
    er = dict(new_state.get("execution_result") or {})
    er["ray_fallback"] = {"from": "k8s-ray", "to": "local", "reason": reason,
                          "rows_used": rows, "note": note}
    new_state["execution_result"] = er
    msgs = list(new_state.get("messages") or [])
    before = len(state.get("messages") or [])
    for i in range(len(msgs) - 1, before - 1, -1):
        if isinstance(msgs[i], AIMessage):
            msgs[i] = AIMessage(content=str(msgs[i].content).rstrip() + note)
            break
    else:
        msgs.append(AIMessage(content=note.strip()))
    new_state["messages"] = msgs
    # Consumed: a later turn must not inherit this turn's fallback.
    new_state["ray_fallback_reason"] = None
    logger.warning("ray_fallback to=local reason=%r rows=%s", reason, rows)
    return new_state


# The batch Ray script runs the generated analysis on this many rows (the
# first rows of the file) and reports that scope; it does not analyse the
# whole file. See _build_ray_wrapper and _ray_data_scope.
RAY_ANALYSIS_SAMPLE_ROWS = 50_000


def _size_bytes(uri: str) -> int:
    """Best-effort file size from cloud URI. Returns 0 on failure."""
    try:
        import fsspec
        with fsspec.open(uri, "rb") as f:
            f.seek(0, 2)
            return f.tell()
    except Exception:
        return 0


def _build_ray_wrapper(data_uri: str, output_artifact_uri: str, output_metrics_uri: str) -> str:
    lines = [
        "import os, time, json, math",
        "import ray",
        "import pandas as pd",
        "import fsspec",
        "from ray.util import get_node_ip_address",
        "from pathlib import Path",
        "",
        "sa_json = os.getenv('GCP_SERVICE_ACCOUNT_JSON', '').strip()",
        "if sa_json and not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):",
        "    sa_json = sa_json.replace('\\\\n', '\\n')",
        "    try:",
        "        import json as _j0; sa_json = _j0.dumps(_j0.loads(sa_json))",
        "    except Exception: pass",
        "    sa_path = '/tmp/gcp_sa.json'",
        "    Path(sa_path).write_text(sa_json, encoding='utf-8')",
        "    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = sa_path",
        "",
        "ray.init(address='auto')",
        "",
        f"DATA_SOURCE_URI = {repr(data_uri)}",
        f"OUTPUT_ARTIFACT_URI = {repr(output_artifact_uri)}",
        f"OUTPUT_METRICS_URI = {repr(output_metrics_uri)}",
        "print('DATA_SOURCE_URI =', repr(DATA_SOURCE_URI))",
        "",
        "def _gcs_token():",
        "    bundled = 'gcp_sa.json'",
        "    if os.path.exists(bundled): return os.path.abspath(bundled)",
        "    s = os.getenv('GCP_SERVICE_ACCOUNT_JSON', '').strip()",
        "    if not s: return None",
        "    try:",
        "        import json as _j",
        "        s = _j.dumps(_j.loads(s.replace('\\\\n', '\\n')))",
        "    except Exception: pass",
        "    p = '/tmp/gcp_sa.json'",
        "    Path(p).write_text(s)",
        "    return p",
        "",
        "def _storage_options(uri):",
        "    if uri.startswith('gs://') or uri.startswith('gcs://'):",
        "        t = _gcs_token()",
        "        if t: return {'token': t}",
        "    return {}",
        "",
        "def _size_bytes(uri):",
        "    try:",
        "        opts = _storage_options(uri)",
        "        with fsspec.open(uri, 'rb', **opts) as fh:",
        "            fh.seek(0, 2)",
        "            return fh.tell()",
        "    except Exception:",
        "        return 0",
        "",
        "# ---- wait for workers ----",
        "deadline = time.time() + 240",
        "while True:",
        "    alive = [n for n in ray.nodes() if n.get('Alive')]",
        "    workers = [n for n in alive if float((n.get('Resources') or {}).get('CPU', 0)) > 0]",
        "    worker_ips = sorted({n.get('NodeManagerAddress') for n in workers if n.get('NodeManagerAddress')})",
        "    print('worker_nodes =', worker_ips, 'count =', len(workers))",
        "    if len(workers) >= 1: break",
        "    if time.time() > deadline: raise RuntimeError(f'No workers. workers={worker_ips}')",
        "    time.sleep(3)",
        "",
        "num_workers = len(workers)",
        "print('cluster_resources =', ray.cluster_resources())",
        "",
        "# ---- row count: each worker counts the lines in its own byte range ----",
        "# The ranges tile the file exactly, so every byte is read once and the",
        "# total is the file's real row count (a quoted field that holds a line",
        "# break counts as an extra row). Nothing here is analysed; see SAMPLE_ROWS.",
        "@ray.remote(num_cpus=1, max_retries=3)",
        "def process_partition(uri, partition_id, total, byte_start, byte_end):",
        "    import fsspec, os, time",
        "    from pathlib import Path",
        "    opts = {}",
        "    if uri.startswith('gs://') or uri.startswith('gcs://'):",
        "        bundled = 'gcp_sa.json'",
        "        if os.path.exists(bundled):",
        "            opts = {'token': os.path.abspath(bundled)}",
        "        else:",
        "            sa = os.getenv('GCP_SERVICE_ACCOUNT_JSON','').strip()",
        "            if sa:",
        "                try:",
        "                    import json as _j2",
        "                    sa = _j2.dumps(_j2.loads(sa.replace('\\\\n','\\n')))",
        "                except Exception: pass",
        "                p = f'/tmp/gcp_sa_{partition_id}.json'",
        "                Path(p).write_text(sa)",
        "                opts = {'token': p}",
        "    from ray.util import get_node_ip_address",
        "    node_ip = get_node_ip_address()",
        "    t0 = time.time()",
        "    print(f'Partition {partition_id}/{total} on node {node_ip}: bytes {byte_start}-{byte_end}')",
        "    newlines, read = 0, 0",
        "    with fsspec.open(uri, 'rb', **opts) as fh:",
        "        fh.seek(byte_start)",
        "        while read < byte_end - byte_start:",
        "            block = fh.read(min(8 * 1024 * 1024, byte_end - byte_start - read))",
        "            if not block: break",
        "            newlines += block.count(b'\\n')",
        "            read += len(block)",
        "    elapsed = time.time() - t0",
        "    print(f'Partition {partition_id} done: {newlines} line breaks in {read} bytes in {elapsed:.1f}s on {node_ip}')",
        "    return {'partition_id': partition_id, 'rows': newlines, 'bytes': read,",
        "            'node_ip': node_ip, 'elapsed_s': elapsed}",
        "",
        "def _ends_with_newline(uri, size):",
        "    try:",
        "        with fsspec.open(uri, 'rb', **_storage_options(uri)) as fh:",
        "            fh.seek(size - 1)",
        "            return fh.read(1) == b'\\n'",
        "    except Exception:",
        "        return True",
        "",
        "# ---- partitions by bytes ----",
        "total_size = _size_bytes(DATA_SOURCE_URI)",
        "print(f'Total file size: {total_size:,} bytes')",
        "PARTITION_BYTES = 64 * 1024 * 1024",
        "total_partitions = max(1, math.ceil(total_size / PARTITION_BYTES)) if total_size else 0",
        "print(f'Total partitions to submit: {total_partitions}')",
        "",
        "t0 = time.time()",
        "futures = []",
        "for i in range(total_partitions):",
        "    byte_start = i * PARTITION_BYTES",
        "    byte_end = min(total_size, byte_start + PARTITION_BYTES)",
        "    futures.append(process_partition.remote(DATA_SOURCE_URI, i+1, total_partitions, byte_start, byte_end))",
        "",
        "print(f'Submitted {total_partitions} partitions across workers')",
        "partition_results = ray.get(futures) if futures else []",
        "scan_runtime_s = max(0.001, time.time() - t0)",
        "scan_bytes = sum(r['bytes'] for r in partition_results)",
        "",
        "nodes_used = sorted({r['node_ip'] for r in partition_results})",
        "if total_size and scan_bytes == total_size:",
        "    # data rows = line breaks - header line + a last line with no trailing break",
        "    ROWS_TOTAL = max(0, sum(r['rows'] for r in partition_results) - 1",
        "                     + (0 if _ends_with_newline(DATA_SOURCE_URI, total_size) else 1))",
        "else:",
        "    ROWS_TOTAL = None   # size unknown or a range came back short: no total is claimed",
        "print(f'NODES USED = {nodes_used}')",
        "print(f'NUM NODES = {len(nodes_used)}')",
        "print(f'ROWS IN FILE = {ROWS_TOTAL}')",
        "for r in partition_results:",
        "    print(f'  Partition {r[\"partition_id\"]}: {r[\"rows\"]} line breaks on {r[\"node_ip\"]} in {r[\"elapsed_s\"]:.1f}s')",
        "",
        "# ---- the analysis runs on the first SAMPLE_ROWS rows, not the whole file ----",
        "# main(df) runs in this driver process. Reading a >=1 GB CSV whole into one",
        "# pandas frame here needs several times the file size in memory, so the",
        "# analysis is scoped to a sample and the scope is reported (rows_used,",
        "# rows_total, sample) in the metrics, the artifact and the reply.",
        f"SAMPLE_ROWS = {RAY_ANALYSIS_SAMPLE_ROWS}",
        "t_analysis = time.time()",
        "opts = _storage_options(DATA_SOURCE_URI)",
        "try:",
        "    df = next(iter(pd.read_csv(DATA_SOURCE_URI, nrows=SAMPLE_ROWS, encoding='utf-8', storage_options=opts, chunksize=SAMPLE_ROWS)))",
        "except UnicodeDecodeError:",
        "    df = next(iter(pd.read_csv(DATA_SOURCE_URI, nrows=SAMPLE_ROWS, encoding='latin-1', storage_options=opts, chunksize=SAMPLE_ROWS)))",
        "ROWS_USED = int(len(df))",
        "IS_SAMPLE = ROWS_TOTAL is None or ROWS_USED < ROWS_TOTAL",
        "print('SHAPE (sample):' if IS_SAMPLE else 'SHAPE (all rows):', df.shape)",
    ]
    return "\n".join(lines) + "\n"


def _build_ray_footer(data_uri: str) -> str:
    lines = [
        "",
        "if 'main' in globals() and callable(globals().get('main')):",
        "    df = globals()['main'](df)",
        "",
        "t1 = time.time()",
        "runtime_s = max(0.001, t1 - t0)",
        "analysis_runtime_s = max(0.001, t1 - t_analysis)",
        "worker_count = len(set(r['node_ip'] for r in partition_results))",
        "# Each rate is over what was actually processed: the workers read every",
        "# byte to count rows (scan_*); the analysis code saw ROWS_USED rows.",
        "metrics = {",
        "    'rows_used': ROWS_USED,",
        "    'rows_total': ROWS_TOTAL,",
        "    'sample': int(IS_SAMPLE),",
        "    'analysed_rows_per_s': round(ROWS_USED / analysis_runtime_s, 2),",
        "    'analysis_runtime_s': round(float(analysis_runtime_s), 3),",
        "    'scan_bytes': int(scan_bytes),",
        "    'scan_mb_s': round((scan_bytes / (1024 * 1024)) / scan_runtime_s, 4),",
        "    'runtime_s': round(float(runtime_s), 3),",
        "    'worker_count': int(worker_count),",
        "}",
        "_output_df = df if ('df' in globals() and hasattr(df, 'to_dict')) else None",
        "_artifact_df = _output_df.head(1000) if _output_df is not None else None",
        "artifact = {",
        "    'metrics': metrics,",
        "    'output_rows': _artifact_df.to_dict(orient='records') if _artifact_df is not None else [],",
        "    'output_columns': list(_artifact_df.columns) if _artifact_df is not None else [],",
        "    'output_row_count': len(_output_df) if _output_df is not None else 0,",
        "}",
        "",
        "def _write_json(uri, obj):",
        "    if not uri: return",
        "    try:",
        "        opts = _storage_options(uri)",
        "        with fsspec.open(uri, 'w', **opts) as fout:",
        "            json.dump(obj, fout)",
        "        print(f'Wrote {uri}')",
        "    except Exception as e:",
        "        print(f'Warning: could not write {uri}: {e}')",
        "",
        "_write_json(OUTPUT_METRICS_URI, metrics)",
        "_write_json(OUTPUT_ARTIFACT_URI, artifact)",
        "for _k, _v in metrics.items():",
        "    if _v is not None:",
        "        print(f'METRIC {_k}={_v}')",
        "print(f'ARTIFACT_URI={OUTPUT_ARTIFACT_URI}')",
        "print(f'METRICS_URI={OUTPUT_METRICS_URI}')",
    ]
    return "\n".join(lines) + "\n"


def _build_interactive_sample_wrapper(data_uri: str, output_artifact_uri: str, output_metrics_uri: str) -> str:
    lines = [
        "import json, os, time",
        "from pathlib import Path",
        "import pandas as pd",
        "import fsspec",
        "",
        "sa_json = os.getenv('GCP_SERVICE_ACCOUNT_JSON', '').strip()",
        "if sa_json and not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):",
        "    sa_json = sa_json.replace('\\\\n', '\\n')",
        "    try:",
        "        import json as _j0; sa_json = _j0.dumps(_j0.loads(sa_json))",
        "    except Exception: pass",
        "    sa_path = '/tmp/gcp_sa.json'",
        "    Path(sa_path).write_text(sa_json, encoding='utf-8')",
        "    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = sa_path",
        "",
        f"DATA_SOURCE_URI = {repr(data_uri)}",
        f"OUTPUT_ARTIFACT_URI = {repr(output_artifact_uri)}",
        f"OUTPUT_METRICS_URI = {repr(output_metrics_uri)}",
        "",
        "def _gcs_token():",
        "    bundled = 'gcp_sa.json'",
        "    if os.path.exists(bundled): return os.path.abspath(bundled)",
        "    s = os.getenv('GCP_SERVICE_ACCOUNT_JSON', '').strip()",
        "    if not s: return None",
        "    try:",
        "        import json as _j",
        "        s = _j.dumps(_j.loads(s.replace('\\\\n', '\\n')))",
        "    except Exception: pass",
        "    p = '/tmp/gcp_sa.json'",
        "    Path(p).write_text(s)",
        "    return p",
        "",
        "def _storage_options(uri):",
        "    if uri.startswith('gs://') or uri.startswith('gcs://'):",
        "        t = _gcs_token()",
        "        if t: return {'token': t}",
        "    return {}",
        "",
        "def _size_bytes(uri):",
        "    try:",
        "        opts = _storage_options(uri)",
        "        with fsspec.open(uri, 'rb', **opts) as fh:",
        "            fh.seek(0, 2)",
        "            return fh.tell()",
        "    except Exception:",
        "        return 0",
        "",
        "def _read_dataframe(uri):",
        "    opts = _storage_options(uri)",
        "    lower = uri.lower()",
        "    if lower.endswith('.parquet'):",
        "        return pd.read_parquet(uri, storage_options=opts)",
        "    if lower.endswith('.json') or lower.endswith('.jsonl'):",
        "        try:",
        "            return pd.read_json(uri, lines=lower.endswith('.jsonl'), storage_options=opts)",
        "        except TypeError:",
        "            with fsspec.open(uri, 'r', **opts) as fin:",
        "                return pd.read_json(fin, lines=lower.endswith('.jsonl'))",
        "    try:",
        "        return pd.read_csv(uri, storage_options=opts, encoding='utf-8')",
        "    except UnicodeDecodeError:",
        "        return pd.read_csv(uri, storage_options=opts, encoding='latin-1')",
        "",
        "def _write_json(uri, obj):",
        "    if not uri: return",
        "    opts = _storage_options(uri)",
        "    with fsspec.open(uri, 'w', **opts) as fout:",
        "        json.dump(obj, fout)",
        "",
        "t0 = time.time()",
        "print('INTERACTIVE_SAMPLE_ANALYSIS_START')",
        "print(f'DATA_SOURCE_URI={DATA_SOURCE_URI}')",
        "df = _read_dataframe(DATA_SOURCE_URI)",
        "print(f'INPUT_DF_SHAPE={getattr(df, \"shape\", None)}')",
        "print(f'INPUT_COLUMNS={list(getattr(df, \"columns\", []))}')",
        "print('USER_CODE_START')",
    ]
    return "\n".join(lines) + "\n"


def _build_interactive_sample_footer() -> str:
    lines = [
        "",
        "print('USER_CODE_END')",
        "if 'main' in globals() and callable(globals().get('main')):",
        "    df = globals()['main'](df)",
        "print(f'OUTPUT_DF_SHAPE={getattr(df, \"shape\", None)}')",
        "",
        "bytes_processed = _size_bytes(DATA_SOURCE_URI)",
        "runtime_s = max(0.001, time.time() - t0)",
        "throughput_mb_s = (bytes_processed / (1024 * 1024)) / runtime_s if runtime_s else 0.0",
        "metrics = {",
        "    'bytes_processed': int(bytes_processed),",
        "    'throughput_mb_s': round(float(throughput_mb_s), 4),",
        "    'runtime_s': round(float(runtime_s), 3),",
        "    'worker_count': 1,",
        "}",
        "_output_df = df if ('df' in globals() and hasattr(df, 'to_dict')) else None",
        "_artifact_df = _output_df.head(1000) if _output_df is not None else None",
        "artifact = {",
        "    'metrics': metrics,",
        "    'output_rows': _artifact_df.to_dict(orient='records') if _artifact_df is not None else [],",
        "    'output_columns': list(_artifact_df.columns) if _artifact_df is not None else [],",
        "    'output_row_count': len(_output_df) if _output_df is not None else 0,",
        "}",
        "_write_json(OUTPUT_METRICS_URI, metrics)",
        "_write_json(OUTPUT_ARTIFACT_URI, artifact)",
        "print(f'OUTPUT_ROW_COUNT={artifact[\"output_row_count\"]}')",
        "print(f\"METRIC bytes_processed={metrics['bytes_processed']}\")",
        "print(f\"METRIC throughput_mb_s={metrics['throughput_mb_s']}\")",
        "print(f\"METRIC runtime_s={metrics['runtime_s']}\")",
        "print(f\"METRIC worker_count={metrics['worker_count']}\")",
        "print(f'ARTIFACT_URI={OUTPUT_ARTIFACT_URI}')",
        "print(f'METRICS_URI={OUTPUT_METRICS_URI}')",
        "print('INTERACTIVE_SAMPLE_ANALYSIS_DONE')",
    ]
    return "\n".join(lines) + "\n"


def _apply_interactive_profile_to_yaml(yaml_text: str) -> str:
    """Shrink the RayJob footprint for lightweight portfolio-sample analysis."""
    y = yaml_text
    # Same rule as render_rayjob_yaml (NEW-RS-01): a count line may carry a
    # trailing comment. This used to work only because the renderer had
    # already stripped them.
    y = re.sub(r'(?m)^([ \t]*)replicas:[ \t]*\d+[ \t]*(?:#.*)?$', r'\g<1>replicas: 1', y, count=1)
    y = re.sub(r'(?m)^([ \t]*)minReplicas:[ \t]*\d+[ \t]*(?:#.*)?$', r'\g<1>minReplicas: 0', y, count=1)
    y = re.sub(r'(?m)^([ \t]*)maxReplicas:[ \t]*\d+[ \t]*(?:#.*)?$', r'\g<1>maxReplicas: 1', y, count=1)
    y = y.replace('cpu: "2"', 'cpu: "1"', 2)
    y = y.replace('memory: "8Gi"', 'memory: "2Gi"', 2)
    y = y.replace('memory: "6Gi"', 'memory: "2Gi"', 2)
    y = y.replace('cpu: "500m"', 'cpu: "250m"', 1)
    return y


def _patch_csv_encoding(code: str) -> str:
    """
    Add encoding='latin-1' fallback to any pd.read_csv calls in user-generated
    code that don't already specify an encoding. Prevents UnicodeDecodeError when
    Ray workers read non-UTF-8 CSVs from S3/GCS (e.g. Excel-exported files).
    """
    def _add_encoding(m):
        full = m.group(0)
        args = m.group(1)
        if "encoding" in args:
            return full
        return f"pd.read_csv({args.rstrip()}, encoding='latin-1')"

    # Match the full read_csv(...) call, allowing one level of nested parentheses
    # (e.g. pd.read_csv(os.path.join(base_dir, 'f.csv'))) so the encoding is added
    # to read_csv itself, not an inner call. The old pattern [^)]+ stopped at the
    # first ')', injecting encoding into the inner call and raising TypeError.
    # Anything more deeply nested simply isn't matched (left untouched) rather
    # than corrupted.
    return re.sub(
        r"pd\.read_csv\(((?:[^()]|\([^()]*\))*)\)",
        _add_encoding,
        code,
    )


def _autogenerate_code(state: dict) -> str:
    """
    Generate a pandas script from the user message + schema.
    Uses semantic parsing to detect intent — no hardcoded column names.
    """
    import re as _re
    import json as _json

    # ── Pull last human message ───────────────────────────────────────
    msgs = state.get("messages", [])
    user_text = ""
    for m in reversed(msgs):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", "")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if role in ("human", "user"):
            user_text = content or ""
            break

    schema = state.get("schema") or {}
    if isinstance(schema, str):
        try:
            schema = _json.loads(schema)
        except Exception:
            schema = {}

    columns = list(schema.keys()) if isinstance(schema, dict) else []
    columns_lower = {c.lower(): c for c in columns}
    user_lower = user_text.lower()

    # ── Detect limit (top N) ──────────────────────────────────────────
    # Only an explicit "top N" truncates; a per-group listing keeps every group.
    limit = None
    m = _re.search(r"top\s+(\d+)", user_lower)
    if m:
        limit = int(m.group(1))

    # ── Detect aggregation(s) and the measure they apply to ───────────
    # MAJ-031: the aggregation used to be detected here and then ignored (the
    # script always emitted groupby(col).size()). The question is read by the
    # same parser the output contract checks the result with.
    from app.agents.output_contract import parse_question
    spec = parse_question(user_text, schema)
    aggs = list(spec.aggs) if spec else ["count"]
    measure = spec.measure if spec else None
    if any(a != "count" for a in aggs) and not measure:
        # A min/max/mean/sum with no column to apply it to: emitting a count
        # under the measure's name is the defect, so emit nothing.
        logger.warning("[_autogenerate_code] %s requested but no measure column found; no code generated", aggs)
        return ""

    # ── Detect sort order ─────────────────────────────────────────────
    sort_asc = "asc" in user_lower and "desc" not in user_lower

    # ── Detect output column names from "Return X, Y, Z" pattern ─────
    # e.g. "Return product_id, brand, category_code, purchase_count"
    requested_output_cols = []
    return_match = _re.search(
        r"(?:return|show|output|select|include)\s+([\w_,\s]+?)(?:\.|sort|order|limit|$)",
        user_lower,
        _re.IGNORECASE,
    )
    if return_match:
        tokens = [t.strip() for t in return_match.group(1).split(",")]
        for token in tokens:
            if token in columns_lower:
                requested_output_cols.append(columns_lower[token])

    # ── Detect group-by column using semantic context ─────────────────
    # Priority 1: "by <col>", "per <col>", "for each <col>", "group by <col>"
    group_col = spec.group_by[0] if spec and spec.group_by else None
    context_patterns = [
        r"(?:group(?:ed)?\s+by|by|per|for\s+each)\s+([\w_]+)",
        r"([\w_]+)\s+(?:with\s+the\s+most|with\s+highest|ranked\s+by)",
    ]
    for pat in ([] if group_col else context_patterns):
        cm = _re.search(pat, user_lower)
        if cm:
            candidate = cm.group(1).strip()
            if candidate in columns_lower:
                group_col = columns_lower[candidate]
                break

    # Priority 2: first column from "Return X, Y, Z" that is in schema
    # and is not the aggregation result column (e.g. "purchase_count" not in schema)
    if not group_col and requested_output_cols:
        for col in requested_output_cols:
            if col in columns:  # must exist in actual dataset
                group_col = col
                break

    # Priority 3: any column name mentioned in the user message,
    # ranked by specificity — prefer columns whose full name appears as a
    # standalone word (word boundary match) over partial substring matches
    if not group_col:
        best_col = None
        best_score = -1
        for col in columns:
            col_l = col.lower()
            # Full word boundary match scores higher
            if _re.search(rf"\b{_re.escape(col_l)}\b", user_lower):
                score = len(col_l)  # longer = more specific
                if score > best_score:
                    best_score = score
                    best_col = col
        group_col = best_col

    # Priority 4: fallback to first non-numeric column in schema
    if not group_col and columns:
        dtype_map = schema if isinstance(schema, dict) else {}
        for col in columns:
            dtype = str(dtype_map.get(col, "")).lower()
            if "int" not in dtype and "float" not in dtype:
                group_col = col
                break
        if not group_col:
            group_col = columns[0]

    # ── Determine aggregation result column name ──────────────────────
    # Use name from requested_output_cols if it's not in schema (it's derived)
    agg_col_name = "count"
    for col_name in requested_output_cols:
        if col_name not in columns:  # not a real column → it's the aggregation result
            agg_col_name = col_name
            break
    if agg_col_name == "count" and "purchase" in user_lower:
        agg_col_name = "purchase_count"

    # ── Extra columns to join in (from requested_output_cols minus group_col & agg) ──
    # Only columns the question asked for. A first-row value of some other
    # column, attached to each group, reads as that group's figure (MAJ-031).
    extra_cols = [
        c for c in requested_output_cols
        if c != group_col and c in columns and c != measure
    ]
    head = f"\n        .head({limit})" if limit else ""

    # ── Build the script ──────────────────────────────────────────────
    if any(a != "count" for a in aggs):
        # The requested aggregation(s) of the requested measure; .size() only
        # for a count that was asked for alongside them.
        named = {
            ("count" if a == "count" else f"{a}_{measure}"): (measure, "size" if a == "count" else a)
            for a in aggs
        }
        sort_col = next(iter(named))
        agg_args = "**{" + ", ".join(f"{k!r}: ({v[0]!r}, {v[1]!r})" for k, v in named.items()) + "}"
        code = f"""
import pandas as pd

def main(df):
    result = (
        df.groupby({repr(group_col)}, as_index=False)
        .agg({agg_args})
        .sort_values({repr(sort_col)}, ascending={sort_asc}){head}
    )
    return result
"""
        logger.info("[_autogenerate_code] group_col=%s measure=%s aggs=%s", group_col, measure, aggs)
        logger.info("[_autogenerate_code] generated code:\n%s", code)
        return code.strip()

    code = f"""
import pandas as pd

def main(df):
    result = (
        df.groupby({repr(group_col)})
        .size()
        .reset_index(name={repr(agg_col_name)})
        .sort_values({repr(agg_col_name)}, ascending={sort_asc}){head}
    )
    # Join extra columns from original dataframe
    extra_cols = {repr(extra_cols)}
    for col in extra_cols:
        if col in df.columns and col not in result.columns:
            mapping = df.drop_duplicates(subset=[{repr(group_col)}]).set_index({repr(group_col)})[col]
            result[col] = result[{repr(group_col)}].map(mapping)
    return result
"""
    logger.info("[_autogenerate_code] group_col=%s agg_col=%s extra_cols=%s",
                group_col, agg_col_name, extra_cols)
    logger.info("[_autogenerate_code] generated code:\n%s", code)
    return code.strip()


def _apply_output_contract(new_state: dict, spec, report: dict) -> bool:
    """Record an output-contract report on the run; on a refusal, replace the
    reply with the plain reason and withdraw the table. Returns True if refused."""
    er = dict(new_state.get("execution_result") or {})
    er["output_contract"] = report
    new_state["execution_result"] = er
    if report.get("status") != "refused":
        return False
    text = refusal_text(spec, report)
    logger.warning("[output-contract] refused: %s", "; ".join(v["rule"] for v in report["violations"]))
    er.update(status="refused", message=text, error=text)
    new_state["execution_error"] = text
    new_state["output_file_data"] = None
    msgs = list(new_state.get("messages") or [])
    if msgs and isinstance(msgs[-1], AIMessage):
        msgs[-1] = AIMessage(content=text)
    else:
        msgs.append(AIMessage(content=text))
    new_state["messages"] = msgs
    return True


def apply_scheduled_output_contract(final_state: dict) -> dict:
    """The output contract for a scheduled run that did not go through Ray
    (the Ray node checks its own result). The reference is recomputed from the
    run's local input when it is small enough to re-read."""
    er = final_state.get("execution_result") or {}
    if not isinstance(er, dict) or "output_contract" in er:
        return final_state
    if str(er.get("status", "")).lower() not in ("success", "succeeded", "completed", "done"):
        return final_state
    spec = spec_from_state(final_state)
    frame = frame_from_output_file(final_state.get("output_file_data")) if spec else None
    if frame is None:
        return final_state
    input_path = _resolve_local_execution_input(
        final_state, final_state.get("data_source_location"), final_state.get("output_location")
    )
    code = (final_state.get("coder_definition") or {}).get("code")
    report = check_output(spec, frame, local_reference(spec, input_path), code=code)
    final_state = dict(final_state)
    messages = final_state.get("messages") or []
    if messages and isinstance(messages[-1], dict):
        # agent_task serialises messages before this runs; keep the shape.
        from app.utils import convert_message_dicts_to_objects, convert_message_objects_to_dicts
        final_state["messages"] = convert_message_dicts_to_objects(messages)
        _apply_output_contract(final_state, spec, report)
        final_state["messages"] = convert_message_objects_to_dicts(final_state["messages"])
    else:
        _apply_output_contract(final_state, spec, report)
    return final_state


# ---------------------------------------------------------------------------
# execution_agent_node_ray
# ---------------------------------------------------------------------------

def execution_agent_node_ray(state: ETLState) -> ETLState:
    # ── FIX: default namespace updated to ray-training (GKE cluster) ──
    ns        = state.get("ray_namespace") or state.get("ray_job_namespace") or "ray-training"
    workers   = int(state.get("ray_workers", 3))
    timeout_s = int(state.get("ray_timeout_s", 1800))
    execution_profile = state.get("ray_execution_profile") or "batch_heavy"
    is_dryrun = os.getenv("RAYJOB_DRYRUN", "0") == "1"

    # ── Validate required inputs ──────────────────────────────────────
    data_uri = state.get("data_source_location_cloud") or state.get("data_source_location")
    if not data_uri:
        msg = "Missing data_source_location_cloud for k8s-ray execution"
        ns_ = state.copy()
        ns_["messages"] = state.get("messages", []) + [AIMessage(content=msg)]
        ns_["execution_result"] = {"mode": "k8s-ray", "status": "error", "message": msg}
        ns_["execution_error"] = msg
        return ns_


    code = (state.get("coder_definition") or {}).get("code") or ""
    if not code.strip():
        logger.info("[execution_agent_node_ray] coder_definition.code missing — auto-generating from user message + schema")
        code = _autogenerate_code(state)

    if not code.strip():
        msg = "Missing coder_definition.code for k8s-ray execution"
        ns_ = state.copy()
        ns_["messages"] = state.get("messages", []) + [AIMessage(content=msg)]
        ns_["execution_result"] = {"mode": "k8s-ray", "status": "error", "message": msg}
        ns_["execution_error"] = msg
        return ns_

    connection_id = (
        state.get("connection_id")
        or state.get("cloud_connection_id")
        or state.get("storage_connection_id")
    )

    # ── No connection_id: refuse a scheduled run, fall back visibly otherwise ──
    if not connection_id and not is_dryrun:
        if _is_scheduled_run(state):
            msg = (
                f"This scheduled run was not started: it needs Ray, but "
                f"{RAY_NO_CONNECTION_REASON}. Link the dataset's cloud connection "
                f"and the next run will use Ray."
            )
            logger.warning("execution_agent_node_ray: scheduled run refused: %s", RAY_NO_CONNECTION_REASON)
            ns_ = state.copy()
            ns_["messages"] = state.get("messages", []) + [AIMessage(content=msg)]
            ns_["execution_result"] = {
                "mode": "k8s-ray", "status": "error", "message": msg, "error": msg,
                "refused_reason": RAY_NO_CONNECTION_REASON,
            }
            ns_["execution_error"] = msg
            return ns_
        logger.warning(
            "execution_agent_node_ray: k8s-ray but connection_id=None "
            "and data_source_location_cloud=%s -> falling back to local execution",
            state.get("data_source_location_cloud"),
        )
        fallback_state = {
            **state,
            "execution_mode": "local",
            "ray_fallback_reason": RAY_NO_CONNECTION_REASON,
            "data_source_location": (
                state.get("data_source_location_local")
                or state.get("full_data_location")
                or state.get("data_source_location")
            ),
        }
        return execution_agent_node_local(fallback_state)

    dataset_id = (
        state.get("dataset_id")
        or state.get("active_dataset_id")
        or state.get("cloud_dataset_id")
        or (state.get("dataset") or {}).get("id")
        or f"ds-{int(time.time())}"
    )

    # ── RayJob name ───────────────────────────────────────────────────
    raw_name = (
        state.get("ray_job_name")
        or state.get("rayjob_name")
        or (
            f"interactive-sample-analysis-{str(dataset_id)[:8]}-{int(time.time())}"
            if execution_profile == "interactive_sample_analysis"
            else f"batch-heavy-{str(dataset_id)[:8]}-{int(time.time())}"
        )
    )
    rayjob_name = _sanitize_k8s_name(raw_name, max_len=63)

    # ── Output URIs ───────────────────────────────────────────────────
    base                = _dir_of_uri(str(data_uri))
    run_prefix          = f"{base}/_avaloka_runs/{rayjob_name}"
    output_artifact_uri = state.get("output_artifact_uri") or f"{run_prefix}/artifact.json"
    output_metrics_uri  = state.get("output_metrics_uri")  or f"{run_prefix}/metrics.json"

    # ── Build full script ─────────────────────────────────────────────
    user_code = _strip_markdown_code_fences(code).strip()

    # Strip local __main__ guard (hardcoded Windows paths etc.)
    user_code = re.sub(
        r'(?ms)^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*.*\Z',
        '',
        user_code,
    )

    # Patch any pd.read_csv in user code to use encoding fallback
    user_code = _patch_csv_encoding(user_code)
    helper_code = _with_csv_reader(textwrap.dedent(HELPER_FUNCTIONS).lstrip())

    # Output contract (MAJ-030): what the question asks for, and the script
    # lines that recompute it on the rows main(df) is given.
    output_spec = spec_from_state(state)
    reference_block = reference_script(output_spec) if output_spec else ""

    if execution_profile == "interactive_sample_analysis":
        workers = 1
        timeout_s = min(timeout_s, 600)
        script_text = (
            _build_interactive_sample_wrapper(str(data_uri), output_artifact_uri, output_metrics_uri)
            + "\n\n"
            + reference_block
            + helper_code
            + "\n\n"
            + user_code
            + "\n\n"
            + _build_interactive_sample_footer()
        )
    else:
        script_text = (
            _build_ray_wrapper(str(data_uri), output_artifact_uri, output_metrics_uri)
            + "\n\n"
            + reference_block
            + helper_code
            + "\n\n"
            + user_code
            + "\n\n"
            + _build_ray_footer(str(data_uri))
        )

    # ── The job's own Secret (KubeRay mode), named after the RayJob ──────
    secret_name_for_yaml = _sanitize_k8s_name(f"{rayjob_name}-env")

    yaml_text_dry = render_rayjob_yaml(
        base_yaml=BASE_RAYJOB_YAML,
        name=rayjob_name,
        namespace=ns,
        workers=workers,
        data_uri=str(data_uri),
        script_text=script_text,
        cloud_secret_name=secret_name_for_yaml,
        output_artifact_uri=output_artifact_uri,
        output_metrics_uri=output_metrics_uri,
    )
    if execution_profile == "interactive_sample_analysis":
        yaml_text_dry = _apply_interactive_profile_to_yaml(yaml_text_dry)

    # ── DRYRUN — return without hitting Supabase or kubectl ───────────
    if is_dryrun:
        new_state = state.copy()
        new_state["ray_job_name"]      = rayjob_name
        new_state["ray_job_namespace"] = ns
        new_state["execution_result"]  = {
            "mode":              "k8s-ray",
            "status":            "dryrun",
            "rayjob_name":       rayjob_name,
            "namespace":         ns,
            "worker_count":      workers,
            "timeout_s":         timeout_s,
            "execution_profile": execution_profile,
            "cloud_secret_name": secret_name_for_yaml,
            "rayjob_yaml":       yaml_text_dry,
            "artifact_uri":      output_artifact_uri,
            "metrics":           output_metrics_uri,
        }
        new_state["execution_error"] = None
        new_state["messages"] = state.get("messages", []) + [
            AIMessage(content=f"[DRYRUN] RayJob YAML rendered for {rayjob_name} (not submitted)")
        ]
        out_path = os.getenv("RAYJOB_DRYRUN_OUTPUT_PATH")
        if out_path:
            Path(out_path).write_text(yaml_text_dry, encoding="utf-8")
        return new_state

    # ── Real run path ─────────────────────────────────────────────────
    secret_name  = None
    outcome      = None
    logs         = ""
    metrics      = {}
    artifact_uri = output_artifact_uri
    conn         = None

    try:
        # 1) Fetch creds from Supabase (with retry for stale connections)
        last_err = None
        for attempt in range(3):
            try:
                conn = _run_coro_sync(get_cloud_connection(connection_id))
                break
            except Exception as e:
                last_err = e
                logger.warning("[execution_agent_node_ray] Supabase attempt %d failed: %s", attempt + 1, e)
                time.sleep(1)
        if conn is None:
            raise Exception(f"Supabase lookup failed after 3 attempts: {last_err}")

        raw_provider = conn.get("provider") or conn.get("backend") or ""
        provider     = _normalize_secret_provider(raw_provider, str(data_uri))

        # 2) The connection's keys, and only those, for this job. The runner
        #    puts them in a Secret owned by the RayJob (KubeRay mode) or in the
        #    job's env (direct mode); the shared Ray pods never hold them.
        _is_direct_mode = bool(os.getenv("RAY_DASHBOARD_URL"))
        secret_name = secret_name_for_yaml
        try:
            secret_env = cloud_secret_env(provider, conn)
        except ValueError:
            if not _is_direct_mode:
                raise
            # Direct mode can still run on the cluster's own identity (GKE
            # Workload Identity) or the bundled gcp_sa.json.
            secret_env = {}

        # 3) Re-render YAML with real secret name
        yaml_text = render_rayjob_yaml(
            base_yaml=BASE_RAYJOB_YAML,
            name=rayjob_name,
            namespace=ns,
            workers=workers,
            data_uri=str(data_uri),
            script_text=script_text,
            cloud_secret_name=secret_name,
            output_artifact_uri=output_artifact_uri,
            output_metrics_uri=output_metrics_uri,
        )
        if execution_profile == "interactive_sample_analysis":
            yaml_text = _apply_interactive_profile_to_yaml(yaml_text)

        # 4) Submit + wait + collect logs
        outcome = run_rayjob_from_yaml(
            yaml_text=yaml_text,
            namespace=ns,
            rayjob_name=rayjob_name,
            timeout_s=timeout_s,
            poll_interval_s=5,
            cloud_creds=conn,
            secret_env=secret_env,
            secret_name=secret_name,
        )

        logs         = outcome.logs or ""
        metrics      = _parse_kv_metrics(logs)
        artifact_uri = _parse_marker(logs, "ARTIFACT_URI") or output_artifact_uri

    except Exception as e:
        logger.exception("[execution_agent_node_ray] Job failed: %s", e)
        new_state = state.copy()
        new_state["ray_job_name"]      = rayjob_name
        new_state["ray_job_namespace"] = ns
        new_state["execution_result"]  = {
            "mode":         "k8s-ray",
            "status":       "error",
            "rayjob_name":  rayjob_name,
            "namespace":    ns,
            "worker_count": workers,
            "execution_profile": execution_profile,
            "runtime_s":    None,
            "logs":         logs,
            "metrics":      metrics,
            "artifact_uri": artifact_uri,
            "message":      str(e),
        }
        new_state["execution_error"] = str(e)
        new_state["messages"] = state.get("messages", []) + [
            AIMessage(content=_friendly_failure_message(e))
        ]
        return new_state


    new_state = state.copy()
    new_state["ray_job_name"]      = rayjob_name
    new_state["ray_job_namespace"] = ns
    new_state["ray_job_logs"]      = logs
    new_state["execution_result"]  = {
        "mode":         "k8s-ray",
        "status":       (outcome.status.lower() if outcome else "error"),
        "rayjob_name":  rayjob_name,
        "namespace":    ns,
        "worker_count": workers,
        "execution_profile": execution_profile,
        "runtime_s":    (outcome.runtime_s if outcome else None),
        "logs":         logs,
        "metrics":      metrics,
        "artifact_uri": artifact_uri,
    }
    data_scope = _ray_data_scope(metrics)
    if data_scope is not None:
        new_state["execution_result"]["data_scope"] = data_scope
    # execution_error is a declared channel, so a value left by an earlier node
    # or turn stays until something clears it. It is set exactly when the
    # status is one route_after_execution already treats as a failure.
    _ray_status = new_state["execution_result"]["status"]
    new_state["execution_error"] = (
        f"RayJob {rayjob_name} finished with {_ray_status}"
        if _ray_status in ("error", "failed") else None
    )

    # ── Fetch the output artifact from cloud storage ──────────────────
    output_file_data = None
    output_df = None
    ai_message_content = f"RayJob {rayjob_name} finished with {outcome.status if outcome else 'ERROR'}"

    if artifact_uri and (outcome and outcome.status.upper() == "SUCCEEDED"):
        try:
            import fsspec, json as _json

            storage_opts = {}
            uri_lower = artifact_uri.lower()

            if uri_lower.startswith("s3://"):
                if conn:
                    ak  = conn.get("aws_access_key_id")    or conn.get("access_key")    or conn.get("access_key_id")
                    sk  = conn.get("aws_secret_access_key") or conn.get("secret_key")    or conn.get("secret_access_key")
                    rgn = conn.get("aws_region")            or conn.get("region")        or "us-east-1"
                    if ak and sk:
                        storage_opts = {
                            "key":    ak,
                            "secret": sk,
                            "client_kwargs": {"region_name": rgn},
                        }
                        logger.info("[execution_agent_node_ray] Using S3 creds from conn for artifact fetch")
                    else:
                        logger.warning("[execution_agent_node_ray] conn present but no AWS keys found; trying default creds")

            elif uri_lower.startswith(("gs://", "gcs://")):
                import json as _j
                sa_file_path = os.getenv("GCP_SERVICE_ACCOUNT_JSON_PATH", "").strip()
                if sa_file_path and os.path.exists(sa_file_path):
                    storage_opts = {"token": sa_file_path}
                else:
                    sa_json = (
                        (conn or {}).get("service_account_json")
                        or (conn or {}).get("gcp_service_account_json")
                        or os.getenv("GCP_SERVICE_ACCOUNT_JSON", "").strip()
                    )
                    if sa_json:
                        try:
                            sa_json = _j.dumps(_j.loads(sa_json.replace("\\n", "\n")))
                        except Exception:
                            pass
                        sa_path = "/tmp/_avaloka_sa_read.json"
                        Path(sa_path).write_text(sa_json, encoding="utf-8")
                        storage_opts = {"token": sa_path}

            logger.info("[execution_agent_node_ray] Fetching artifact from %s", artifact_uri)
            with fsspec.open(artifact_uri, "r", **storage_opts) as fh:
                artifact_data = _json.load(fh)

            output_rows    = artifact_data.get("output_rows", [])
            output_columns = artifact_data.get("output_columns", [])
            row_count      = artifact_data.get("output_row_count", len(output_rows))

            if output_rows:
                output_df = pd.DataFrame(output_rows, columns=output_columns if output_columns else None)
                logger.info("[execution_agent_node_ray] Fetched %d output rows from artifact", len(output_df))

                csv_content    = output_df.to_csv(index=False)
                timestamp      = generate_filename_timestamp()
                filename       = f"{timestamp}_ray_output.csv"
                file_size      = len(csv_content.encode("utf-8"))
                base64_content = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")

                output_file_data = {
                    "filename": filename,
                    "content":  f"data:text/csv;base64,{base64_content}",
                    "size":     file_size,
                }

                ai_message_content = output_df.to_markdown(index=False)
            else:
                logger.warning("[execution_agent_node_ray] Artifact had no output_rows — job may not have called main(df)")

        except Exception as e:
            logger.warning("[execution_agent_node_ray] Could not fetch artifact from %s: %s", artifact_uri, e)

    if data_scope and data_scope.get("note") and outcome and outcome.status.upper() == "SUCCEEDED":
        ai_message_content = ai_message_content.rstrip() + data_scope["note"]

    new_state["output_file_data"] = output_file_data
    new_state["messages"] = state.get("messages", []) + [
        AIMessage(content=ai_message_content)
    ]
    if output_df is not None and outcome and outcome.status.upper() == "SUCCEEDED" and output_spec:
        # Checked before lineage and memory see the result: a refused table is
        # not recorded or remembered as an answer.
        report = check_output(output_spec, output_df, parse_reference(logs), code=code)
        if _apply_output_contract(new_state, output_spec, report):
            output_df = None
            ai_message_content = new_state["messages"][-1].content
    if output_df is not None and outcome and outcome.status.upper() == "SUCCEEDED":
        lineage_dataset_id = record_analysis_lineage(new_state, output_df)
        if lineage_dataset_id:
            new_state["latest_output_dataset_id"] = lineage_dataset_id

    # Store result in Layer 4 memory if successful
    if outcome and outcome.status.upper() == "SUCCEEDED":
        session_id = new_state.get("session_id", "default")
        _dispatch_execution_memory_record(session_id, ai_message_content, "assistant")

    return new_state


def _ray_data_scope(metrics: dict) -> Optional[dict]:
    """The rows a batch Ray run analysed, read from the script's METRIC lines.

    Returns None when the script reported no scope (an older script, or the
    interactive profile, which reads its whole input). ``note`` is the line
    the reply carries when the analysis saw fewer rows than the file has.
    """
    if not isinstance(metrics, dict) or "rows_used" not in metrics:
        return None
    rows_used = int(metrics.get("rows_used") or 0)
    rows_total = metrics.get("rows_total")
    rows_total = int(rows_total) if rows_total is not None else None
    sample = bool(metrics.get("sample", rows_total is None or rows_used < rows_total))
    note = None
    if sample:
        if rows_total:
            scope = f"a {rows_used:,}-row sample of the {rows_total:,}-row file (its first rows)"
        else:
            scope = f"a {rows_used:,}-row sample (the first rows of the file)"
        note = (
            f"\n\n> Note: these figures were computed on {scope}, not the entire dataset. "
            f"Ray counted the rows of the whole file, but the analysis itself ran on "
            f"that sample, so counts, rates, distinct values and min/max may differ "
            f"from the full file."
        )
    return {"rows_used": rows_used, "rows_total": rows_total, "sample": sample, "note": note}


# ---------------------------------------------------------------------------
# Metric / marker parsers
# ---------------------------------------------------------------------------

def _parse_kv_metrics(logs: str) -> dict:
    metrics = {}
    for m in re.finditer(r"METRIC\s+([a-zA-Z0-9_]+)=([0-9.]+)", logs or ""):
        k, v = m.group(1), m.group(2)
        metrics[k] = float(v) if "." in v else int(v)
    return metrics


def _parse_marker(logs: str, key: str):
    m = re.search(rf"{key}=([^\s]+)", logs or "")
    return m.group(1) if m else None

def _strip_top_level_main_guard(code: str) -> str:
    """
    Remove user/generated `if __name__ == "__main__":` blocks.

    Avaloka supplies its own driver that loads the actual input DataFrame,
    calls main(df), and writes the fresh output CSV.
    """
    if not code:
        return code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    def _is_main_guard(test: ast.AST) -> bool:
        try:
            compact = ast.unparse(test).replace(" ", "")
        except Exception:
            return False

        return compact in {
            "__name__=='__main__'",
            '__name__=="__main__"',
            "'__main__'==__name__",
            '"__main__"==__name__',
        }

    lines = code.splitlines(keepends=True)
    lines_to_remove: set[int] = set()

    # Only remove top-level __main__ guards.
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and _is_main_guard(node.test)
        ):
            end_lineno = getattr(
                node,
                "end_lineno",
                node.lineno,
            )

            lines_to_remove.update(
                range(node.lineno, end_lineno + 1)
            )

    if not lines_to_remove:
        return code

    cleaned = "".join(
        line
        for line_number, line in enumerate(lines, start=1)
        if line_number not in lines_to_remove
    )

    return cleaned.rstrip() + "\n"

def _needs_local_main_harness(code: str) -> bool:
    """Return True when code defines main() but never invokes it."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return False

    defines_main = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
        for node in ast.walk(tree)
    )

    invokes_main = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "main"
        for node in ast.walk(tree)
    )

    return defines_main and not invokes_main


def _build_local_main_harness(
    input_path: Optional[str],
    output_path: Optional[str],
) -> str:
    if not input_path:
        raise ValueError("Local execution input path is missing")
    if not output_path:
        raise ValueError("Local execution output path is missing")

    return f'''
# Avaloka local execution driver.
if __name__ == "__main__":
    import os as _avaloka_os
    import numpy as _avaloka_np

    _avaloka_input_path = {str(input_path)!r}
    _avaloka_output_path = {str(output_path)!r}

    _avaloka_df = read_csv_best_effort(_avaloka_input_path)
    _avaloka_result_value = main(_avaloka_df)

    if _avaloka_result_value is None:
        # Some generated functions write the output themselves.
        if not (
            _avaloka_os.path.exists(_avaloka_output_path)
            and _avaloka_os.path.getsize(_avaloka_output_path) > 0
        ):
            raise RuntimeError(
                "main(df) returned None and did not create an output file"
            )
    else:
        if isinstance(_avaloka_result_value, pd.DataFrame):
            _avaloka_output_df = _avaloka_result_value

        elif isinstance(_avaloka_result_value, pd.Series):
            _avaloka_output_df = (
                _avaloka_result_value
                .to_frame()
                .reset_index()
            )

        elif isinstance(_avaloka_result_value, dict):
            try:
                _avaloka_output_df = pd.DataFrame(
                    _avaloka_result_value
                )
            except ValueError:
                _avaloka_output_df = pd.DataFrame(
                    [_avaloka_result_value]
                )

        elif isinstance(
            _avaloka_result_value,
            (list, tuple, _avaloka_np.ndarray),
        ):
            _avaloka_output_df = pd.DataFrame(
                _avaloka_result_value
            )

        else:
            _avaloka_output_df = pd.DataFrame(
                {{"result": [_avaloka_result_value]}}
            )

        _avaloka_os.makedirs(
            _avaloka_os.path.dirname(_avaloka_output_path)
            or ".",
            exist_ok=True,
        )

        _avaloka_output_df.to_csv(
            _avaloka_output_path,
            index=False,
        )

        print(
            "AVALOKA_OUTPUT_ROWS="
            f"{{len(_avaloka_output_df)}}",
            flush=True,
        )
'''

# ---------------------------------------------------------------------------
# execute_code_on_local
# ---------------------------------------------------------------------------

def execute_code_on_local(
    code: str,
    input_data_location: Optional[np.ndarray] = None,
    output_location: Optional[str] = None,
    read_path_map: Optional[dict] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # 1) Strip ``` fences if present
    #extracted_code = _strip_markdown_code_fences(code)
    extracted_code = _strip_markdown_code_fences(code)
    # Generated ETL code must never schedule itself — recurrence is handled by
    # Celery/RedBeat in task_scheduler_node. The coder occasionally emits the
    # `schedule` pip package + a `while True: schedule.run_pending()` driver,
    # which isn't installed (ModuleNotFoundError) and would block the worker
    # forever. Strip that scaffolding so the script runs as a one-shot transform.
    extracted_code = _strip_inprocess_scheduling(extracted_code)
    extracted_code = _strip_top_level_main_guard(
        extracted_code
    )

    # 2) Replace ONLY pd.read_csv("literal_path", ...) with read_csv_best_effort(...)
    extracted_code = re.sub(
        r"pd\.read_csv\(\s*([\"'][^\"']+[\"'])([^\)]*)\)",
        r"read_csv_best_effort(\1\2)",
        extracted_code,
        flags=re.MULTILINE,
    )

    # Resolve each read to its OWN dataset path by basename/alias. Falling back
    # to the single input path only on a miss preserves single-file behaviour;
    # for a multi-file join this is what stops orders.csv and customers.csv
    # collapsing onto one file (the self-join that produced KeyError: 'country').
    _fallback_input = str(input_data_location) if input_data_location is not None else None

    def _resolve_read(match):
        prefix = match.group(1)
        quoted = match.group(2)
        orig = quoted[1:-1]                      # strip the surrounding quotes
        base = os.path.basename(orig)
        target = None
        if read_path_map:
            target = read_path_map.get(base) or read_path_map.get(orig)
        if not target:
            target = _fallback_input
        if not target:
            return match.group(0)                # nothing to substitute
        return f"{prefix}{repr(str(target))}"

    if read_path_map or _fallback_input:
        extracted_code = re.sub(
            r"(read_csv_best_effort\(\s*)([\"'][^\"']+[\"'])",
            _resolve_read,
            extracted_code,
            flags=re.MULTILINE,
        )
    extracted_code = _replace_first_literal_arg(
        extracted_code,
        r"(\.to_csv\(\s*)([\"'][^\"']+[\"'])",
        str(output_location) if output_location is not None else None,
    )

    logger.info("input data location is %s", input_data_location)
    logger.info("candidate output location is %s", output_location)
    logger.info("extracted code is %s", extracted_code)


    # 3) Build final script, ensuring both parts are left-aligned
    script_parts = [
        _with_csv_reader(textwrap.dedent(HELPER_FUNCTIONS).lstrip()),
        textwrap.dedent(extracted_code).lstrip(),
    ]

    if _needs_local_main_harness(extracted_code):
        # A missing input/output path is a routing problem, not a crash: the
        # node is a LangGraph node, so letting ValueError escape here tears
        # down the whole graph run instead of surfacing a failed
        # execution_result the repair loop can act on. Report it the same way
        # every other pre-flight failure below is reported.
        try:
            harness = _build_local_main_harness(
                input_data_location,
                output_location,
            )
        except ValueError as exc:
            message = str(exc)
            logger.error(message)
            return {
                "returncode": None,
                "execution_stdout": "",
                "execution_stderr": "",
                "status": "error",
                "execution_error": message,
            }
        script_parts.append(textwrap.dedent(harness).lstrip())

    script_to_run = "\n\n".join(
        part for part in script_parts if part.strip()
    )
    

    logger.info("\n\nscript to run is : \n\n%s\n\n", script_to_run)

    interactive_errors = _find_interactive_input_errors(script_to_run)
    if interactive_errors:
        message = (
            "\n".join(interactive_errors)
            + "\nGenerated code runs unattended and cannot ask for terminal input."
        )
        logger.error(message)
        return {
            "returncode": None,
            "execution_stdout": "",
            "execution_stderr": "",
            "status": "error",
            "execution_error": message,
        }

    # 4) Write temp script securely
    fd, temp_script_path = tempfile.mkstemp(suffix=".py", prefix="avaloka_exec_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(script_to_run)

    # Model-written code: the child gets the allowlisted environment and a
    # private HOME/TMPDIR, never this process's provider keys, database
    # passwords, DB_ENCRYPTION_KEY or MCP_SERVICE_TOKEN.
    exec_home = tempfile.mkdtemp(prefix="avaloka_exec_home_")
    try:
        result = subprocess.run(
            [sys.executable, temp_script_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=int(os.getenv("AVALOKA_LOCAL_EXEC_TIMEOUT", "60")),
            env=scrubbed_env(extra={"HOME": exec_home, "TMPDIR": exec_home}),
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        out["returncode"] = result.returncode
        out["execution_stdout"] = stdout
        out["execution_stderr"] = stderr

        if result.returncode == 0:
            out["status"] = "success"
            out["execution_error"] = None
            out["output_location"] = output_location
            if stdout:
                logger.info("Output from interpreter:\n%s", stdout)
            if stderr.strip():
                logger.warning("Warnings from interpreter:\n%s", stderr)
        else:
            out["status"] = "error"
            out["execution_error"] = f"Script failed with exit status {result.returncode}"
            logger.error("Interpreter failed (rc=%s)", result.returncode)
            logger.error("Stdout:\n%s", stdout)
            logger.error("Stderr:\n%s", stderr)

    except subprocess.TimeoutExpired as e:
        out["returncode"] = None
        out["execution_stdout"] = (e.stdout or "")
        out["execution_stderr"] = (e.stderr or "")
        out["status"] = "error"
        out["execution_error"] = f"Script execution timed out after {int(os.getenv('AVALOKA_LOCAL_EXEC_TIMEOUT', '60'))} seconds"
        logger.error(out["execution_error"])

    except FileNotFoundError:
        out["status"] = "error"
        out["execution_error"] = f"Python executable not found at {sys.executable}"
        logger.error(out["execution_error"])

    finally:
        try:
            if os.path.exists(temp_script_path):
                os.remove(temp_script_path)
        except Exception:
            pass
        shutil.rmtree(exec_home, ignore_errors=True)

    return out


# ---------------------------------------------------------------------------
# execute_code_on_ssh  (from develop-1.2)
# ---------------------------------------------------------------------------

def execute_code_on_ssh(code: str, server: str, input_data_location: str, output_location: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # 1) Strip ``` fences if present
    extracted_code = _strip_markdown_code_fences(code)

    # 2) Replace ONLY pd.read_csv("literal_path", ...) with read_csv_best_effort(...)
    extracted_code = re.sub(
        r"pd\.read_csv\(\s*([\"'][^\"']+[\"'])([^\)]*)\)",
        r"read_csv_best_effort(\1\2)",
        extracted_code,
        flags=re.MULTILINE,
    )

    logger.info("input data location is %s", input_data_location)
    logger.info("extracted code is %s", extracted_code)


    # 3) Build final script
    script_to_run = (
        _with_csv_reader(textwrap.dedent(HELPER_FUNCTIONS).lstrip())
        + "\n\n" + textwrap.dedent(extracted_code).lstrip()
    )

    logger.info("\n\nscript to run is : \n\n%s\n\n", script_to_run)

    # 4) Write temp script securely
    fd, temp_script_path = tempfile.mkstemp(suffix=".py", prefix="avaloka_exec_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(script_to_run)
    remote_script_name = os.path.basename(temp_script_path)

    try:
        path = os.path.join(output_location, remote_script_name)
        p = subprocess.run(["scp", temp_script_path, f"{server}:{path}"], capture_output=True)
        if p.returncode != 0:
            stdout = p.stdout or ""
            stderr = p.stderr or ""
            out["returncode"] = p.returncode
            out["execution_stdout"] = stdout
            out["execution_stderr"] = stderr
            out["status"] = "error"
            out["execution_error"] = f"Script failed with exit status {p.returncode}"
            logger.error("Interpreter failed (rc=%s)", p.returncode)
            logger.error("Stdout:\n%s", stdout)
            logger.error("Stderr:\n%s", stderr)
            return out

        result = subprocess.run(
            ["ssh", server, f"cd {output_location} && bash execute.sh {remote_script_name}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=int(os.getenv("AVALOKA_LOCAL_EXEC_TIMEOUT", "60")),
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        out["returncode"] = result.returncode
        out["execution_stdout"] = stdout
        out["execution_stderr"] = stderr

        if result.returncode == 0:
            out["status"] = "success"
            out["execution_error"] = None
            if stdout:
                logger.info("Output from interpreter:\n%s", stdout)
            if stderr.strip():
                logger.warning("Warnings from interpreter:\n%s", stderr)
        else:
            out["status"] = "error"
            out["execution_error"] = f"Script failed with exit status {result.returncode}"
            logger.error("Interpreter failed (rc=%s)", result.returncode)
            logger.error("Stdout:\n%s", stdout)
            logger.error("Stderr:\n%s", stderr)

    except subprocess.TimeoutExpired as e:
        out["returncode"] = None
        out["execution_stdout"] = (e.stdout or "")
        out["execution_stderr"] = (e.stderr or "")
        out["status"] = "error"
        out["execution_error"] = f"Script execution timed out after {int(os.getenv('AVALOKA_LOCAL_EXEC_TIMEOUT', '60'))} seconds"
        logger.error(out["execution_error"])

    except FileNotFoundError:
        out["status"] = "error"
        out["execution_error"] = f"Python executable not found at {sys.executable}"
        logger.error(out["execution_error"])

    finally:
        try:
            if os.path.exists(temp_script_path):
                os.remove(temp_script_path)
        except Exception:
            pass

    return out
