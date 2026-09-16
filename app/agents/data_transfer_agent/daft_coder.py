# daft_coder.py
import os
import logging
import json
import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional
import gcsfs
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from app.core.log_utils import describe_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.core.agent_llm import build_agent_llm
from sqlalchemy import create_engine, text
from app.agents.coder import (
    _coerce_response_text,
    _append_unique_message,
    _patch_df_str_accessor,
    extract_code_block,
    generate_filename_timestamp,
)
from app.agents.data_transfer_agent.dta_state import DaftCodingAgentState as CodingAgentState
from app.agents.data_transfer_agent.dta_state import redact_conn_str
from app.rag.daft_retrieval import retrieve_docs, load_chroma_collection, format_chunks_for_prompt
from app.agents.data_transfer_agent.daft_validator import daft_docs

# Logger
logger = logging.getLogger(__name__)

# Prompt templates (loaded once at import time)

_PROMPT_DIR = Path(__file__).parent / "prompts"

pseudo_code_prompt_path = _PROMPT_DIR / "pseudocode_prompt.md"
with open(pseudo_code_prompt_path, "r", encoding="utf-8") as _f:
    PSEUDO_CODE_PROMPT_TEMPLATE = _f.read()

daft_coder_prompt_path = _PROMPT_DIR / "daft_coder_prompt.md"
with open(daft_coder_prompt_path, "r", encoding="utf-8") as _f:
    DAFT_CODER_SYSTEM_PROMPT_TEMPLATE = _f.read()

# Chroma collection (cached)

@lru_cache(maxsize=1)
def _get_cached_chroma_collection(db_path: str, collection_name: str, embedding_model: str):
    return load_chroma_collection(db_path, collection_name, embedding_model)

# LLM initialisation
def _build_coder_llm() -> ChatGroq | None:
    """Construct the coder LLM from environment variables, or return None."""
    api_key = os.environ.get("GROQ_API_KEY_CODING_AGENT")
    if not api_key:
        logger.warning(
            "GROQ_API_KEY_CODING_AGENT is not set — code generation will be skipped."
        )
        return None

    # Hybrid-reasoning code generation (effort "medium": correctness of the
    # transfer code benefits from deliberation; verified live to be fast and
    # AST-clean). DTA_CODER_MODEL=llama-3.3-70b-versatile restores the
    # previous non-reasoning behavior.
    return build_agent_llm(
        agent="DTA_CODER",
        api_key=api_key,
        default_model="openai/gpt-oss-120b",
        temperature=0.0,
        default_effort="medium",
        env_model_var="DTA_CODER_MODEL",
        role="coding",
    )

coder_llm = _build_coder_llm()

# Helpers
# Shared constant so the pipeline's success gate can recognise the stub and
# refuse to ship untransformed data as a success.
_DAFT_NOOP_STUB_CODE = "\n".join([
    "def main(df):",
    "    # Daft stub: no-op transformation",
    "    return df",
])


def _generate_daft_stub_code(state: CodingAgentState) -> str:
    """Return a no-op Daft stub that passes the dataframe through unchanged."""
    return _DAFT_NOOP_STUB_CODE


def _normalise_code_lines(code: str) -> List[str]:
    """Non-blank, non-comment, stripped lines — for structural comparison."""
    return [
        s for line in code.strip().splitlines()
        if (s := line.strip()) and not s.startswith("#")
    ]


def is_stub_code(code: Optional[str]) -> bool:
    """``True`` if *code* is the deterministic no-op stub (or empty).

    Detects the stub structurally, so the success gate stays correct even if the
    ``stub_fallback_used`` flag is dropped. A legitimate ``transform_data`` copy
    is intentionally NOT treated as a stub.
    """
    if not code or not code.strip():
        return True
    return _normalise_code_lines(code) == _normalise_code_lines(_DAFT_NOOP_STUB_CODE)

# The chat endpoint appends a runtime-metadata suffix to EVERY user message, e.g.
#   "\n\n[Analysis context] Current fidelity: portfolio_samples; Selected sample:
#    random_baseline; Resolved analysis source: quick_sample; Dataset size bytes: …"
# It describes how the ANALYSIS view was sampled — it is NOT a transformation the
# user asked for. Left in a transfer prompt, the sanitiser/coder LLM reads the
# "fidelity"/"sample" words as an instruction and emits df.sample(fraction=…),
# silently dropping ~half the rows from the transfer. The training agent (MTA)
# already strips this marker; the planner does too (_strip_runtime_context). Strip
# everything from the marker onwards here as well (case-insensitive, single or
# double newline) so the DTA code-gen only ever sees the real transformation intent.
_ANALYSIS_CONTEXT_RE = re.compile(r"\n\s*\[analysis context\]", re.IGNORECASE)


def strip_runtime_analysis_context(prompt: str) -> str:
    """Remove the chat endpoint's appended '[Analysis context] …' metadata suffix."""
    return _ANALYSIS_CONTEXT_RE.split(prompt or "", maxsplit=1)[0].strip()


def sanitize_user_prompt_for_transformations(raw_prompt: str, llm) -> str:
    """Strip I/O operations from the user prompt, returning only transformation intent."""
    # Deterministically drop the runtime "[Analysis context]" metadata BEFORE the
    # LLM (or the passthrough below) ever sees it — otherwise its fidelity/sample
    # words get misread as a df.sample() transform. See _ANALYSIS_CONTEXT_RE above.
    raw_prompt = strip_runtime_analysis_context(raw_prompt)
    # No coding LLM configured — pass the prompt through instead of crashing on
    # None.invoke; the coder node's stub fallback then surfaces a clear error.
    if llm is None:
        logger.info("Coder LLM not configured; skipping prompt sanitisation.")
        return raw_prompt
    system = (
        "You are a data engineering assistant. "
        "Extract ONLY the data transformation requirements from the user's request. "
        "Remove any mention of: reading files, loading data, writing output, saving results, "
        "creating new files, or any I/O operations. "
        "Return only a clean description of what transformations should be applied to the existing data. "
        "Output only the cleaned prompt text. No preamble, no explanation."
    )
    messages = [SystemMessage(content=system), HumanMessage(content=raw_prompt)]
    response = llm.invoke(messages)
    return getattr(response, "content", raw_prompt).strip()


def parse_pseudo_code(pseudo_code_block: str) -> List[str]:
    """Parse a pseudocode block into a list of non-empty, stripped lines."""
    return [line.strip() for line in pseudo_code_block.split("\n") if line.strip()]


# Pipeline nodes
# Empty source guard
# Natural-language → SQL operator mapping.
# Pseudocode uses English phrases; these are translated to valid SQL before
# the COUNT query is executed.
_NL_TO_SQL_OPS = [
    (re.compile(r"\bis\s+greater\s+than\s+or\s+equal\s+to\b", re.IGNORECASE), ">="),
    (re.compile(r"\bis\s+less\s+than\s+or\s+equal\s+to\b",    re.IGNORECASE), "<="),
    (re.compile(r"\bis\s+greater\s+than\b",                       re.IGNORECASE), ">"),
    (re.compile(r"\bis\s+less\s+than\b",                          re.IGNORECASE), "<"),
    (re.compile(r"\bis\s+not\s+equal\s+to\b",                    re.IGNORECASE), "!="),
    (re.compile(r"\bis\s+equal\s+to\b",                           re.IGNORECASE), "="),
    (re.compile(r"\bequals\b",                                       re.IGNORECASE), "="),
    (re.compile(r"\bis\s+not\b",                                    re.IGNORECASE), "!="),
    (re.compile(r"\bis\b",                                           re.IGNORECASE), "="),
    (re.compile(r"\band\b",                                          re.IGNORECASE), "AND"),
    (re.compile(r"\bor\b",                                           re.IGNORECASE), "OR"),
]

# Strips Daft-style suffixes like "[Function: filter()]" that sometimes leak
# into the pseudocode description.
_DAFT_SUFFIX_PATTERN = re.compile(r"\s*\[Function:[^\]]+\]", re.IGNORECASE)

# Matches pseudocode filter lines in any of these forms:
#   "Filter rows where ocean_proximity = 'INLAND'"
#   "Keep only rows where age < 30"
#   "Select rows where status is 'active'"
_FILTER_PSEUDOCODE_PATTERN = re.compile(
    r"(?:filter|keep only|select)\s+rows?\s+where\s+(.+)",
    re.IGNORECASE,
)

# Value-modification operations (fill / impute / set-null) whose pseudocode
# "where …" clause selects rows to MODIFY, not rows to KEEP. The empty-source
# COUNT guard must skip these — their filter is not row-reducing, so a 0 count is
# NOT an empty source. Detection is deliberately BROAD / fail-open: over-skipping
# merely forgoes an early-abort optimization, whereas under-skipping FALSELY aborts
# a valid transfer (the bug where "fill missing values with the median" — which has
# no literal "null" — slipped past the old ``"null" AND "fill"`` check and ran a
# mis-parsed COUNT).
_IMPUTE_VERB_RE = re.compile(
    r"\b(?:impute\w*|fillna|coalesce|interpolat\w+|ffill|bfill"
    r"|(?:forward|backward|back)[\s-]?fill)\b",
    re.IGNORECASE,
)
_MISSING_CONCEPT_RE = re.compile(
    r"\b(?:null|nulls|missing|nan|n/?a|empty|blank|absent)\b", re.IGNORECASE,
)
# Setting/replacing values to NULL (then usually filling) is also a modification,
# not a filter — e.g. "Set 'age' to NULL where …", "replace 0 with NULL".
_NULL_SET_RE = re.compile(
    r"\bset\b[^.\n]*\bnull\b|\breplace\b[^.\n]*\bnull\b|\b(?:to|as)\s+null\b",
    re.IGNORECASE,
)


def _is_impute_or_fill_pseudocode(pseudocode: str) -> bool:
    """True when the pseudocode FILLS/IMPUTES/SETS values rather than filtering rows.

    Covers "fill missing values with the median", "fillna", "forward fill",
    "impute", "coalesce", "interpolate", and "set X to NULL where …" — none of which
    reduce the row count, so the empty-source COUNT guard must not treat their
    ``where`` clause as a row-reducing filter.
    """
    text = pseudocode or ""
    if _IMPUTE_VERB_RE.search(text):
        return True
    if re.search(r"\bfill\w*\b", text, re.IGNORECASE) and _MISSING_CONCEPT_RE.search(text):
        return True
    if _NULL_SET_RE.search(text):
        return True
    return False


def _nl_condition_to_sql(condition: str) -> str:
    """
    Translate a natural-language filter condition extracted from pseudocode
    into a valid SQL WHERE expression.

    Examples
    --------
    "median_income is greater than 8.0 and total_bedrooms is less than 50"
      → "median_income > 8.0 AND total_bedrooms < 50"

    "ocean_proximity = 'INLAND'"
      → "ocean_proximity = 'INLAND'"   (already SQL — returned unchanged)
    """
    # Remove Daft artefacts like "[Function: filter()]"
    condition = _DAFT_SUFFIX_PATTERN.sub("", condition).strip()

    for pattern, sql_op in _NL_TO_SQL_OPS:
        condition = pattern.sub(sql_op, condition)

    logger.debug("Translated filter condition to SQL: %s", condition)
    return condition.strip()

def _extract_filter_condition_from_pseudocode(pseudocode: str):
    """
    Scan the pseudocode for a filter/where clause, translate any natural-
    language operators to SQL, and return the result.
    Returns None if no filter line is found.
    """
    for line in pseudocode.splitlines():
        match = _FILTER_PSEUDOCODE_PATTERN.search(line.strip())
        if match:
            raw_condition = match.group(1).strip().rstrip(".")
            sql_condition = _nl_condition_to_sql(raw_condition)
            logger.debug(
                "Pseudocode filter: raw='%s' → sql='%s'",
                raw_condition,
                sql_condition,
            )
            return sql_condition
    return None

def _detect_db_dialect(connection_string: str) -> str:
    """
    Infer the database dialect from the SQLAlchemy connection string prefix.

    Returns one of: 'mysql', 'postgresql', 'mssql', 'sqlite', 'unknown'.
    Used to apply the correct identifier-quoting style in COUNT queries.
    """
    cs = connection_string.lower()
    if cs.startswith("mysql"):
        return "mysql"
    if cs.startswith(("postgresql", "postgres")):
        return "postgresql"
    if cs.startswith("mssql"):
        return "mssql"
    if cs.startswith("sqlite"):
        return "sqlite"
    return "unknown"

def _quote_identifier(name: str, dialect: str) -> str:
    """
    Wrap a table name in the correct dialect-specific quote characters.

      mysql       → `table`
      postgresql  → "table"
      mssql       → [table]
      sqlite      → "table"   (ANSI-compatible)
      unknown     → "table"   (safe default)
    """
    if dialect == "mysql":
        return f"`{name}`"
    if dialect == "mssql":
        return f"[{name}]"
    # postgresql / sqlite / unknown — ANSI double-quotes
    return f'"{name}"'

def _run_source_count_query(
    connection_string: str,
    source_table: str,
    filter_condition,
    connect_args: Optional[Dict] = None,
) -> Optional[int]:
    """
    Execute COUNT(*) against the source table with an optional WHERE clause.

    Works for MySQL, PostgreSQL, MSSQL, and SQLite by:
      - Detecting the dialect from the connection string prefix.
      - Quoting the table name with the dialect-correct characters.
      - Forwarding connect_args (SSL contexts etc.) to SQLAlchemy.

    Returns the integer row count, or None if the query fails so the
    pipeline can proceed rather than abort on a guard infrastructure error.
    """
    try:
        dialect = _detect_db_dialect(connection_string)
        quoted_table = _quote_identifier(source_table, dialect)

        where_clause = f"WHERE {filter_condition}" if filter_condition else ""
        query = f"SELECT COUNT(*) AS cnt FROM {quoted_table} {where_clause}".strip()

        logger.info(
            "Empty source guard | dialect=%s | query: %s",
            dialect,
            query,
        )

        engine_kwargs: Dict[str, Any] = {}
        if connect_args:
            engine_kwargs["connect_args"] = connect_args

        engine = create_engine(connection_string, **engine_kwargs)
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()

        count = int(result[0]) if result else 0
        logger.info("Empty source guard COUNT(*) result: %d row(s).", count)
        return count

    except Exception as exc:
        logger.error(
            "Empty source guard COUNT query failed — guard will be skipped: %s",
            exc,
            exc_info=True,
        )
        return None

def empty_source_guard_node(state: CodingAgentState) -> dict:
    """
    Pipeline node: runs AFTER pseudocode generation, BEFORE code generation.

    Extracts any filter condition from the pseudocode, runs COUNT(*) against
    the source table, and aborts the pipeline early when the filtered result
    set is empty — avoiding pointless code generation and an empty write.

    Supports MySQL, PostgreSQL, MSSQL, and SQLite via dialect-aware table
    quoting.  Skipped automatically for non-DB sources (CSV, Parquet, JSON).

    State keys read:
        data_source_location  — SQLAlchemy connection string for the source DB
        source_table          — table name to query
        source_type           — 'mysql' | 'postgresql' | 'mssql' | 'csv' | etc.
        source_connect_args   — optional dict forwarded to SQLAlchemy (SSL etc.)
        coder_pseudocode      — pseudocode produced by daft_pseudocode_node

    State keys written:
        empty_source          — True when the filter returns 0 rows
        pipeline_abort_reason — human-readable message explaining the abort
    """
    connection_string = state.get("data_source_location", "")
    source_table = state.get("source_table", "")
    source_type = (state.get("source_type") or "").lower()
    connect_args = state.get("source_connect_args") or {}
    pseudocode = state.get("coder_pseudocode", "")

    # Only meaningful for DB sources — file sources have no SQL engine
    _DB_SOURCE_TYPES = {"mysql", "postgresql", "postgres", "mssql", "sqlite"}
    if source_type and source_type not in _DB_SOURCE_TYPES:
        logger.info(
            "Empty source guard skipped — source type '%s' is not a database.",
            source_type,
        )
        return {"empty_source": False}

    if not connection_string or not source_table:
        logger.warning(
            "Empty source guard skipped — 'data_source_location' or 'source_table' "
            "not present in state."
        )
        return {"empty_source": False}

    filter_condition = _extract_filter_condition_from_pseudocode(pseudocode)

    if filter_condition and _is_impute_or_fill_pseudocode(pseudocode):
        logger.info(
            "Empty source guard skipped — pseudocode fills/imputes/sets values "
            "(its `where` selects rows to modify, not to keep), not a pure filter.")
        return {"empty_source": False}
    if filter_condition is None:
        logger.info(
            "Empty source guard: no filter condition found in pseudocode — "
            "skipping COUNT query."
        )
        return {"empty_source": False}

    logger.info(
        "Empty source guard | source_type=%s | filter='%s' | table='%s'",
        source_type or "unknown",
        filter_condition,
        source_table,
    )

    row_count = _run_source_count_query(
        connection_string,
        source_table,
        filter_condition,
        connect_args=connect_args or None,
    )

    if row_count is None:
        logger.warning(
            "Empty source guard: COUNT query could not complete — proceeding with pipeline."
        )
        return {"empty_source": False}

    if row_count == 0:
        abort_reason = (
            f"Filter condition `{filter_condition}` matched 0 rows in source table "
            f"`{source_table}`. There is no data to transform or transfer — "
            "pipeline aborted early to avoid writing an empty dataset to the destination."
        )
        logger.warning("Empty source detected: %s", abort_reason)
        return {
            "empty_source": True,
            "pipeline_abort_reason": abort_reason,
        }

    logger.info(
        "Empty source guard passed: %d row(s) match filter \'%s\' on table \'%s\'.",
        row_count,
        filter_condition,
        source_table,
    )
    return {"empty_source": False}

def daft_pseudocode_node(state: CodingAgentState) -> dict:
    """Generate a pseudocode interpretation of the user prompt."""
    logger.info("Generating pseudocode interpretation of user prompt.")

    user_prompt = state.get("user_prompt", "")
    primary_response = state.get("primary_llm_response", "")
    retry_count = state.get("retry_count") or 0
    messages = state.get("messages") or []

    if not user_prompt:
        default_msg = "No prompt provided. Defaulting to row count."
        logger.warning(default_msg)
        return {
            "coder_pseudocode": "",
            "code_validation_feedback": default_msg,
            "retry_count": retry_count + 1,
            "llm_raw_response": default_msg,
            "messages": _append_unique_message(messages, default_msg),
            "primary_llm_response": primary_response or default_msg,
        }

    if coder_llm is None:
        logger.warning("Coder LLM not configured; skipping pseudocode generation.")
        return {"coder_pseudocode": ""}

    # Resolve instruction from state or message history
    instruction: str = user_prompt
    logger.debug("User prompt for pseudocode generation: %s", instruction)

    if not instruction:
        for message in reversed(messages):
            candidate = getattr(message, "content", None)
            if candidate:
                instruction = candidate
                break

    if not instruction:
        logger.warning("No instruction found for pseudocode generation.")
        return {"coder_pseudocode": ""}

    # DTA stores the deduced schema under "source_schema"; fall back to "schema".
    schema = state.get("source_schema") or state.get("schema") or {}
    logger.debug("Schema for pseudocode generation: %s", schema)

    psc_validation = state.get("pseudocode_validation_feedback", "")
    if psc_validation:
        logger.info("Applying pseudocode validation feedback to refine pseudocode.")
        instruction += (
            f"\n\n**Previous Pseudocode Validation Feedback:**\n{psc_validation}\n\n"
            "Please refine the pseudocode based ONLY on this feedback."
        )

    system_prompt = PSEUDO_CODE_PROMPT_TEMPLATE.format(
        datatypes_core_md=daft_docs.get("datatypes", ""),
        type_conversion_md=daft_docs.get("type_conversions", ""),
        expressions_core_md=daft_docs.get("expressions", ""),
        dataframe_core_md=daft_docs.get("dataframe", ""),
        casting_md=daft_docs.get("casting", ""),
        functions_core_md=daft_docs.get("functions", ""),
    )

    human_prompt = (
        f"{instruction}\n\n"
        f"Schema (column: dtype):\n{schema}\n"
    )

    try:
        response = coder_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
    except Exception as exc:
        logger.error("Failed to generate pseudocode interpretation: %s", exc)
        return {"coder_pseudocode": ""}

    pseudocode = _coerce_response_text(response)
    logger.info("Generated pseudocode blueprint:\n%s", pseudocode)
    return {"coder_pseudocode": pseudocode}


def daft_retrieval_node(state: CodingAgentState) -> dict:
    """Retrieve relevant documentation based on the pseudocode to assist the coder."""
    pseudocode = state.get("coder_pseudocode", "")
    if not pseudocode:
        logger.debug("No pseudocode available; skipping RAG retrieval.")
        return {"rag_retrieved_docs": ""}

    current_file = Path(__file__).resolve()
    app_dir = current_file.parents[2]
    db_path = str(app_dir / "rag" / "daft_embeddings")

    collection_name = "daft_documentation"
    embedding_model = "mixedbread-ai/mxbai-embed-large-v1"

    try:
        collection = _get_cached_chroma_collection(db_path, collection_name, embedding_model)
        rag_docs = retrieve_docs(parse_pseudo_code(pseudocode), collection, 2)
        formatted_rag_docs = format_chunks_for_prompt(rag_docs)
        logger.info("RAG retrieval returned %d document chunk(s).", len(rag_docs))
        return {"rag_retrieved_docs": formatted_rag_docs}
    except Exception as exc:
        logger.error("RAG retrieval failed; continuing without docs: %s", exc)
        return {"rag_retrieved_docs": ""}


def daft_coder_node(state: CodingAgentState) -> dict:
    """Generate or refine Daft Python code based on the plan and feedback."""
    _sample = state.get("input_sample_data")
    try:
        _sample_desc = f"{len(_sample)} rows" if _sample is not None else "none"
    except TypeError:
        _sample_desc = "unsized"
    logger.debug("Coder node state: keys=%s | input_sample_data=%s", list(state.keys()), _sample_desc)

    user_prompt = state.get("user_prompt", "")
    retry_count = state.get("retry_count") or 0
    messages = state.get("messages", []) or []
    primary_response = state.get("primary_llm_response") or ""

    logger.debug("DTA coder received %s", describe_messages(messages))

    # --- Stub fallback: LLM not configured ---
    if coder_llm is None:
        logger.info("Coder LLM not configured; returning stub code.")
        stub_code = _generate_daft_stub_code(state)
        fallback_msg = "Generated deterministic stub code."
        return {
            "generated_code": stub_code,
            "stub_fallback_used": True,
            "code_validation_feedback": None,
            "retry_count": retry_count + 1,
            "llm_raw_response": fallback_msg,
            "messages": _append_unique_message(messages, fallback_msg),
            "primary_llm_response": primary_response or fallback_msg,
            "coder_pseudocode": state.get("coder_pseudocode"),
        }

    data_source_location = state.get("data_source_location", "~")
    # Never log the raw value: for a DB source it is a SQLAlchemy URL containing
    # the password in plaintext.
    logger.info("Entering coder node. data_source_location=%s", redact_conn_str(data_source_location))

    feedback = state.get("code_validation_feedback")
    previous_code = state.get("generated_code")
    hard_fail = bool(state.get("syntax_error") or state.get("static_semantic_error"))

    # --- Stub fallback: repeated hard validation failures ---
    if hard_fail and retry_count >= 2:
        logger.warning(
            "Falling back to deterministic stub after %d hard validation failure(s).", retry_count
        )
        stub_code = _generate_daft_stub_code(state)
        fallback_msg = "Generated deterministic stub code after repeated HARD validation failures."
        return {
            "generated_code": stub_code,
            "stub_fallback_used": True,
            "code_validation_feedback": None,
            "retry_count": retry_count + 1,
            "llm_raw_response": fallback_msg,
            "messages": _append_unique_message(messages, fallback_msg),
            "primary_llm_response": primary_response or fallback_msg,
            "coder_pseudocode": state.get("coder_pseudocode"),
        }

    # --- System prompt ---
    system_prompt = DAFT_CODER_SYSTEM_PROMPT_TEMPLATE.format(
        datatypes_core_md=daft_docs.get("datatypes", ""),
        type_conversion_md=daft_docs.get("type_conversions", ""),
        expressions_core_md=daft_docs.get("expressions", ""),
        dataframe_core_md=daft_docs.get("dataframe", ""),
        casting_md=daft_docs.get("casting", ""),
        udf_documentation_md=daft_docs.get("udf", ""),
    )

    # --- Human prompt ---
    human_prompt = (
        f"**Data Schema:**\n{state.get('source_schema', 'No schema available')}\n\n"
        f"**User prompt:**\n{user_prompt}\n\n"
        "**Important:** Keep all existing column names unchanged unless the plan or pseudocode "
        "explicitly requests a rename. If you aggregate a column (such as summing `Price`), "
        "keep the resulting column labeled with the original name (`Price`).\n\n"
    )

    if state.get("coder_pseudocode"):
        human_prompt += f"**Pseudocode blueprint (follow exactly):**\n{state['coder_pseudocode']}\n\n"

    formatted_rag_docs = state.get("rag_retrieved_docs", "")
    if formatted_rag_docs:
        human_prompt += (
            "**Use this document as reference (retrieved for each pseudocode line):**\n"
            f"{formatted_rag_docs}\n\n"
        )

    if state.get("source_schema"):
        human_prompt += (
            "**Strictly adhere to the provided schema. "
            "Do not create new columns that are not in the schema unless explicitly instructed.**\n\n"
        )

    if feedback and previous_code:
        logger.info("Applying validation feedback to refine previous code.")
        human_prompt += (
            f"**Previous Code (with errors):**\n```python\n{previous_code}\n```\n"
            f"**Validation Feedback:**\n{feedback}\n\n"
            "Please fix the previous code based ONLY on the feedback."
        )
    else:
        logger.info("Generating initial code from plan.")
        human_prompt += "Please write the Python script to execute the plan."

    # --- Invoke LLM ---
    llm_messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    try:
        response = coder_llm.invoke(llm_messages)
        raw_response = getattr(response, "content", str(response))

        generated_code = extract_code_block(raw_response)
        generated_code = re.sub(r"('Price'\s*:\s*)'[^']+'", r"\1'Price'", generated_code)
        generated_code = _patch_df_str_accessor(generated_code)

        logger.info("Code generated successfully:\n%s", generated_code)

        return {
            "generated_code": generated_code,
            # Real code — clear any stub flag left by a prior repair attempt.
            "stub_fallback_used": False,
            "retry_count": retry_count + 1,
            "code_validation_feedback": None,
        }
    except Exception as exc:
        logger.error("LLM invocation failed in daft_coder_node: %s", exc)
        stub_code = _generate_daft_stub_code(state)
        fallback_msg = f"LLM invocation failed: {exc}. Falling back to stub code."
        return {
            "generated_code": stub_code,
            "stub_fallback_used": True,
            "code_validation_feedback": None,
            "retry_count": retry_count + 1,
            "llm_raw_response": fallback_msg,
            "messages": _append_unique_message(messages, fallback_msg),
            "primary_llm_response": primary_response or fallback_msg,
            "coder_pseudocode": state.get("coder_pseudocode"),
        }