import json
import logging
import os
import re
from pathlib import Path
from string import Template
from typing import Any
import gcsfs
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from app.core.agent_llm import build_agent_llm
from app.agents.data_transfer_agent.dta_state import DaftCodingAgentState as CodingAgentState
from app.agents.validator import (
    DataFrameColumnVisitor,
    StrAccessorOnDfVisitor,
    _coerce_message_content,
    _extract_json_dict,
    syntactic_validator_node,
)

# Constants
# Logic-review enforcement. Blocking a WRONG verdict is the default; set this env var
# to "0" to opt OUT into advisory mode (warn but proceed). See strict_mode below.
STRICT_LOGIC_ENV = "AVALOKA_STRICT_LOGIC_VALIDATION"
#: Where the DTA system documentation lives.
#:
#: This used to be hardcoded to a private Google Cloud bucket
#: (gs://avaloka-test-user-filestore/...). Every install -- including every
#: open-source one -- listed that bucket on each validation run, got
#: "Anonymous caller does not have storage.objects.list access" back, logged a
#: full traceback, and then validated WITHOUT the documentation the validator
#: was designed to use. The degradation was silent and the quality loss
#: invisible.
#:
#: It now defaults to the MinIO/S3 object store this stack already deploys, and
#: is overridable. Both s3:// and gs:// are honoured so an existing cloud
#: deployment can keep pointing at its own bucket.
SYSDOCS_PATH = os.getenv(
    "AVALOKA_SYSDOCS_PATH",
    f"s3://{os.getenv('AVALOKA_ARTIFACT_BUCKET', 'avaloka')}/dta_md_files/sysdocs_markdown",
)
#: Kept as an alias so any out-of-tree caller importing the old name still works.
SYSDOCS_GCS_PATH = SYSDOCS_PATH
SUPPORTED_DOC_EXTENSIONS = (".md", ".txt")

logger = logging.getLogger(__name__)

# LLM initialisation
def _build_validator_llm() -> ChatGroq | None:
    """Construct the validator LLM from environment variables, or return None."""
    api_key = (os.environ.get("GROQ_API_KEY_CODING_AGENT")
               or os.environ.get("GROQ_API_KEY"))
    if not api_key:
        logger.warning(
            "GROQ_API_KEY_CODING_AGENT is not set — logical validation will be skipped."
        )
        return None

    # Mechanical verdict check: reasoning stays at "low".
    # DTA_VALIDATOR_MODEL=llama-3.3-70b-versatile restores the old behavior.
    return build_agent_llm(
        agent="DTA_VALIDATOR",
        api_key=api_key,
        default_model="openai/gpt-oss-120b",
        temperature=0.0,
        default_effort="low",
        env_model_var="DTA_VALIDATOR_MODEL",
        role="coding",
        tier="small",
    )


validator_llm: ChatGroq | None = _build_validator_llm()

# GCS sysdoc loading
def build_gcs_filesystem() -> gcsfs.GCSFileSystem:
    """Create a GCSFileSystem, preferring a service-account JSON from env."""
    sa_json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GCP_SERVICE_ACCOUNT_JSON", "")

    if sa_json_str:
        # Normalise escape sequences and strip control characters
        sa_json_str = sa_json_str.replace("\\n", "\n").replace("\\r", "")
        sa_json_str = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", sa_json_str)
        token_info = json.loads(sa_json_str)
        return gcsfs.GCSFileSystem(token=token_info)

    logger.info("No service-account JSON env var found; falling back to GOOGLE_APPLICATION_CREDENTIALS.")
    return gcsfs.GCSFileSystem()

def _build_sysdocs_filesystem():
    """Return ``(filesystem, path_without_scheme)`` for SYSDOCS_PATH.

    s3:// goes to the object store this stack already runs (MinIO in the
    default deployment, any S3 endpoint via S3_ENDPOINT_URL). gs:// keeps the
    original Google path for deployments that have one. Anything else, or a
    missing client library, returns ``(None, "")`` so the caller degrades with
    a single informative line instead of a traceback.
    """
    path = SYSDOCS_PATH or ""
    if path.startswith("gs://"):
        try:
            return build_gcs_filesystem(), path[len("gs://"):]
        except Exception as exc:  # noqa: BLE001
            logger.warning("GCS filesystem unavailable for sysdocs (%s: %s)",
                           type(exc).__name__, exc)
            return None, ""
    if path.startswith("s3://"):
        try:
            import s3fs  # imported lazily; not every install needs it
        except ImportError:
            logger.info("s3fs is not installed; sysdocs will be skipped.")
            return None, ""
        endpoint = (os.getenv("S3_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL") or "").strip()
        key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER")
        secret = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
        kwargs = {"key": key, "secret": secret}
        if endpoint:
            kwargs["client_kwargs"] = {"endpoint_url": endpoint}
        return s3fs.S3FileSystem(**kwargs), path[len("s3://"):]
    logger.info("SYSDOCS_PATH %r has no supported scheme (expected s3:// or gs://).", path)
    return None, ""


def load_sysdocs_from_gcs() -> dict[str, str]:
    """
    Load all .md and .txt documentation files from the configured GCS path.

    Returns a mapping of ``filename -> file_content``.
    On failure, logs a warning and returns an empty dict so callers can
    degrade gracefully.
    """
    try:
        fs, bucket_path = _build_sysdocs_filesystem()
        if fs is None:
            logger.info(
                "Sysdocs are not configured (AVALOKA_SYSDOCS_PATH=%s); logical "
                "validation will run without them.", SYSDOCS_PATH)
            return {}
        files = fs.ls(bucket_path)

        docs: dict[str, str] = {}
        for file_path in files:
            filename = file_path.split("/")[-1]
            if not filename.endswith(SUPPORTED_DOC_EXTENSIONS):
                continue
            try:
                with fs.open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                    docs[filename] = fh.read()
            except Exception:
                logger.exception("Failed to read sysdoc file: %s", file_path)

        logger.info("Loaded %d sysdoc file(s) from GCS.", len(docs))
        return docs

    except FileNotFoundError:
        # An absent sysdocs prefix is the normal state of a fresh install, not
        # an error worth a traceback.
        logger.info("No sysdocs found at %s; proceeding without them.", SYSDOCS_PATH)
        return {}
    except Exception as exc:
        # One line naming the path and the cause. The previous logger.exception
        # printed a full traceback on every validation run of every install
        # that could not reach the hardcoded bucket.
        logger.warning("Could not load sysdocs from %s (%s: %s); proceeding without them.",
                       SYSDOCS_PATH, type(exc).__name__, exc)
        return {}

def index_sysdocs(raw: dict[str, str]) -> dict[str, str]:
    """Re-key the raw filename->content map using the file stem (no extension)."""
    return {Path(filename).stem: content for filename, content in raw.items()}

# Module-level load — happens once at import time.
daft_docs: dict[str, str] = index_sysdocs(load_sysdocs_from_gcs())

# Prompt template
_REVIEW_PROMPT_TEMPLATE = Template("""
You are an expert Python code reviewer specialising in the Daft DataFrame library.

### SCOPE — WHAT YOU ARE REVIEWING
The code below is ONLY the in-memory transformation step: a single
`transform_data(df)` function. Reading the source table and writing to the
destination table are performed by the surrounding runtime (the injection script),
NOT by this code, and the destination is NOT addressable from here.

Therefore:
- NEVER require a read, write, insert, save, connection or sink step. Their absence
  is CORRECT, not a defect. Do not suggest df.write_parquet / df.write_sql / any
  write method — a write added here would corrupt the pipeline.
- The user prompt names a destination connection and table (e.g. "transfer to X into
  table Y"). That destination is handled by the runtime. Treat it as CONTEXT ONLY —
  never as a requirement this code must implement.
- Judge ONLY the transformation the user asked for: row filters, column selection,
  renames, type casts, derived columns.
- If the user asked for a plain copy/move/transfer with NO filter, NO column
  selection and NO derived columns, then a `transform_data` that returns `df`
  unchanged is FULLY CORRECT — respond is_logically_correct: true.

IMPORTANT RULES:
- Only validate EXPLICIT user requirements. Do NOT invent new requirements.
- Use the execution output preview/columns.
- Re-confirm if there are any missing columns, or newly invented columns.
- Strictly stick to the schema provided.
- Check if any columns are newly invented outside the specified user requirement and immediately flag them.
- Ensure the newly created columns align with the instructions given in the user prompt and how they are described in the pseudocode.
- Reference the "Daft Framework Documentation" below to ensure the code uses correct functions. If you think the currently used function is not doing the right job, traverse the system documentation again to see if there is a better fit; if there isn't, flag it as requiring a UDF.
- Reference the "Retrieved Documentation" to check if syntax and semantic rules are being followed and if the currently used functions are appropriate for the given transformational task.

### CRITICAL INSTRUCTION: PRIORITY ON EXECUTION LOGS
You must check the "Execution Results" section below FIRST.
1. If "STDERR" contains any error messages, the code is **BROKEN**.
2. If "Exception" contains a traceback, the code is **BROKEN**.
3. In these cases, return `is_logically_correct: false` and quote the error message in your rationale.
4. DO NOT IGNORE ERRORS even if the Python syntax looks correct.

User Requirements:
$user_prompt

Original Input Schema:
$schema

System documents – Daft Fundamentals Reference:
2.1 DataType System
$datatypes

2.2 Type Conversion
$type_conversions

2.3 Type Casting
$casting

2.4 Expression Patterns
$expressions

2.5 DataFrame Operations
$dataframe

2.6 User Defined Functions
$udf_documentation_md

Context Documentation:
$rag_docs

Code to Review:
```python
$generated_code
```

Execution Results:

STDOUT: $stdout

Output Data Preview (first rows of the TRANSFORMED result — these are the actual
output columns and values; use them to verify columns are present/correct and that
no column was invented or dropped):
$preview

STDERR: $stderr

Exception: $error

Respond ONLY with a JSON object:
{ "is_logically_correct": <true_or_false>, "rationale": "<Your reasoning. If false, provide clear, actionable feedback for the developer.>" }
""")


# Validator node
def logical_semantic_validator_node(state: CodingAgentState) -> dict[str, Any]:
    """
    Use an LLM to check whether the generated code logically fulfils the user
    prompt.

    Short-circuits without calling the LLM when:
    - A syntax or static-semantic error was already detected upstream.
    - The validator LLM is not configured.

    Returns a partial state dict consumed by the LangGraph runner.
    """
    if state.get("syntax_error") or state.get("static_semantic_error"):
        logger.debug("Skipping logical validation — upstream error already present.")
        return {}

    if validator_llm is None:
        logger.info("Skipping LLM review — GROQ_API_KEY_CODING_AGENT is not configured.")
        return {"logical_semantic_error": False, "code_validation_feedback": "APPROVED"}

    logger.info("--- Checking Logic with LLM Reviewer ---")
    logger.debug("Code to validate:\n%s", state.get("generated_code"))

    # Blocking on a WRONG verdict is the SAFE DEFAULT: a transform the reviewer judges
    # logically incorrect must not silently write bad data to the destination. The
    # reviewer runs inside a repair loop, so a block first triggers regeneration and
    # only aborts if repair can't fix it. Set AVALOKA_STRICT_LOGIC_VALIDATION=0 to
    # downgrade to advisory mode (surface the feedback as a warning but proceed).
    strict_mode: bool = os.getenv(STRICT_LOGIC_ENV, "1") != "0"

    output_preview_text = _serialise_output_preview(state.get("execution_output_preview") or [])

    prompt = _REVIEW_PROMPT_TEMPLATE.safe_substitute(
        user_prompt=state.get("user_prompt", ""),
        # DTA stores the schema under "source_schema"; fall back to "schema".
        schema=state.get("source_schema") or state.get("schema") or "",
        datatypes=daft_docs.get("datatypes", ""),
        type_conversions=daft_docs.get("type_conversions", ""),
        casting=daft_docs.get("casting", ""),
        expressions=daft_docs.get("expressions", ""),
        dataframe=daft_docs.get("dataframe", ""),
        rag_docs=state.get("rag_retrieved_docs", "No context documents retrieved."),
        udf_documentation_md=daft_docs.get("udf", ""),
        generated_code=state.get("generated_code", ""),
        stdout=state.get("execution_stdout", "No output."),
        stderr=state.get("execution_stderr", "No errors."),
        error=state.get("execution_error", "No exception."),
        preview=output_preview_text,
    )

    try:
        response = validator_llm.invoke([SystemMessage(content=prompt)])
        logger.debug("Raw LLM review response: %s", response)
    except Exception:
        logger.exception("LLM invocation failed during logical validation.")
        return {
            "logical_semantic_error": False,
            "code_validation_feedback": "Reviewer invocation failed; skipping logical validation.",
        }

    return _parse_review_response(response, strict_mode)

# Internal helpers
def _serialise_output_preview(output_preview) -> str:
    """Serialise the execution output preview for the reviewer prompt.

    The preview arrives in two shapes depending on the execution path:
      * a STRING — the DTA's local executor stores the transformed-data preview as
        the already-rendered stdout table (daft_execution.py); use it verbatim.
        (Slicing a string as ``[:2]`` would hand the reviewer its first two
        CHARACTERS — the original bug this guards against.)
      * a LIST of row dicts — the main validator stores structured records; show the
        first two rows as JSON.
    Returns a clear placeholder when empty so the reviewer isn't handed a
    misleading blank value.
    """
    if not output_preview:
        return "No output preview available."
    if isinstance(output_preview, str):
        return output_preview.strip() or "No output preview available."
    try:
        return json.dumps(output_preview[:2], indent=2)
    except Exception:
        logger.warning("Could not JSON-serialise output_preview; falling back to str().")
        return str(output_preview[:2])

def _parse_review_response(response: Any, strict_mode: bool) -> dict[str, Any]:
    """
    Parse the LLM reviewer's JSON response and map it to a LangGraph state
    update dict.
    """
    try:
        raw_content = _coerce_message_content(response.content)
        review = _extract_json_dict(raw_content)

        if review is None:
            raise ValueError("Reviewer output was not valid JSON.")

        logger.debug("Parsed LLM review: %s", review)

        if review.get("is_logically_correct"):
            logger.info("Logical review passed.")
            return {
                "logical_semantic_error": False,
                "code_validation_feedback": "APPROVED",
                "code_validated": True,
            }

        feedback = review.get("rationale", "No rationale provided.")
        logger.warning("Logic reviewer flagged an issue: %s", feedback)

        if strict_mode:
            # logical_review_feedback is what build_composite_feedback feeds the coder;
            # code_validation_feedback alone gets overwritten before the coder sees it.
            return {
                "logical_semantic_error": True,
                "code_validation_feedback": feedback,
                "logical_review_feedback": feedback,
                "code_validated": False,
            }

        # Advisory mode (AVALOKA_STRICT_LOGIC_VALIDATION=0, explicit opt-out only):
        # surface the reviewer's feedback as a warning but do NOT block. This is a
        # deliberate escape hatch — the default blocks (see strict_mode above), so a
        # WRONG verdict cannot silently ship a bad transform unless someone opted out.
        logger.warning(
            "Advisory mode: reviewer flagged the transform but the transfer will "
            "proceed (AVALOKA_STRICT_LOGIC_VALIDATION=0)."
        )
        return {
            "logical_semantic_error": False,
            "code_validation_feedback": "APPROVED",
            "logical_review_feedback": feedback,
            "code_validated": True,
        }

    except (json.JSONDecodeError, ValueError, AttributeError):
        logger.exception("Failed to parse reviewer LLM response; skipping logical validation.")
        return {
            "logical_semantic_error": False,
            "code_validation_feedback": "Reviewer response unparsable; skipping logical validation.",
        }