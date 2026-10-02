"""
Data Profiling Agent - Semantic analysis of datasets using LLM

Workflow:
1. Receives statistical profile from sampling_agent_daft
2. Analyzes domain, column meanings, data quality
3. Ranks columns by predictive power
4. Provides actionable insights
5. Extension point for feature engineering agent

Two entry points:
  profile_quick()  - Phase A: only domain + column semantics (LLM calls 1+2, fast ~3s)
  profile_full()   - Phase B: all insights including red flags + suggestions (all 3 LLM calls)
"""

import json
import logging
import os
import re
import concurrent.futures
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.model_config import resolve as resolve_model
from app.core.model_fallback import attach_fallback
from app.core.log_utils import describe_response

from app.core.inference import build_chat_model

# Load environment variables
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, '.env'))

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no"}


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


# Run column semantics (call 2) and insights (call 3) concurrently after the
# domain call. Set AVALOKA_PROFILE_PARALLEL=0 to restore the sequential flow.
_PROFILE_PARALLEL = _env_flag("AVALOKA_PROFILE_PARALLEL", "1")

# Ask the provider for a JSON object response (Groq `response_format`). If the
# model/provider rejects it, that call silently falls back to plain mode.
_PROFILE_JSON_MODE = _env_flag("AVALOKA_PROFILE_JSON_MODE", "1")

# Attempts per LLM call when the reply can't be parsed as JSON (1 = no retry).
_PROFILE_JSON_ATTEMPTS = _env_int("AVALOKA_PROFILE_JSON_ATTEMPTS", 2)

# Columns per column-semantics call. Smaller batches = shorter replies, so
# they are less likely to be cut off by the output-token limit.
_SEMANTICS_BATCH_SIZE = _env_int("AVALOKA_PROFILE_SEMANTICS_BATCH", 10)

# Max column names listed in the insights prompt (wide schemas are truncated).
_INSIGHTS_MAX_COLUMNS = _env_int("AVALOKA_PROFILE_INSIGHTS_MAX_COLS", 30)

# Long cell values (free text, JSON blobs) are shortened in prompts.
_MAX_CELL_CHARS = _env_int("AVALOKA_PROFILE_MAX_CELL_CHARS", 80)

# Concurrent column-semantics calls. With the old cap of 4, an 85-column file
# (9 batches of 10) ran in 3 waves; 10 runs up to 100 columns in one wave.
# Lower it if Groq starts returning 429 rate-limit errors.
_SEMANTICS_MAX_WORKERS = _env_int("AVALOKA_PROFILE_MAX_WORKERS", 10)

_RETRY_NOTE = (
    "Your previous reply could not be parsed as JSON (it may have been cut off). "
    "Reply again with ONLY the JSON object and no other text. "
    "Keep every string under 20 words."
)

# LLM setup (Groq Llama - same as planner)
_PROFILING_API_KEY = os.environ.get("GROQ_API_KEY_PLANNING_AGENT") or os.environ.get("GROQ_API_KEY")
profiling_llm = build_chat_model(
    role="planning",
    agent="PROFILING",
    tier="large",
    temperature=0.2,
    groq_model=resolve_model("profiling"),
    groq_api_key=_PROFILING_API_KEY,
)
if profiling_llm is not None:
    logger.info("Profiling LLM enabled (model=%s)", resolve_model("profiling"))
else:
    logger.warning("Profiling LLM disabled; set GROQ_API_KEY_PLANNING_AGENT to enable semantic analysis")


def round_floats(obj):
    if isinstance(obj, float):
        return round(obj, 2)
    elif isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(v) for v in obj]
    return obj


def _trim_rows(
    rows: Optional[List[Dict[str, Any]]],
    columns: Optional[List[str]] = None,
    n: int = 5,
    max_chars: int = _MAX_CELL_CHARS,
) -> List[Dict[str, Any]]:
    """First `n` rows, restricted to `columns` (if given), long strings shortened."""
    out: List[Dict[str, Any]] = []
    for row in (rows or [])[:n]:
        if not isinstance(row, dict):
            continue
        items = row.items() if columns is None else ((c, row.get(c)) for c in columns)
        trimmed: Dict[str, Any] = {}
        for key, value in items:
            if isinstance(value, str) and len(value) > max_chars:
                value = value[:max_chars] + "..."
            trimmed[key] = value
        out.append(trimmed)
    return round_floats(out)


# =========================
# LLM CALL HELPERS
# Three focused calls
# =========================

def _call_llm(system: str, user: str) -> Optional[str]:
    """Run a single LLM call and return the raw text response (kept for callers outside this module)."""
    if profiling_llm is None:
        return None
    try:
        response = profiling_llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None


_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _parse_json(text: str, warn: bool = True) -> Optional[Dict]:
    """Parse a JSON object from an LLM response.

    Strips reasoning blocks (<think>...</think>) and markdown fences, including
    an opening fence with no closing one (a cut-off reply).
    """
    if not text:
        return None
    cleaned = _THINK_RE.sub("", text).strip()
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    parsed: Any = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find the JSON object inside the response
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                parsed = None
    if isinstance(parsed, dict):
        return parsed
    if warn:
        logger.warning("Failed to parse LLM response as JSON")
    return None


def _invoke(messages: List[Any]):
    """Invoke the profiling model, in JSON mode when enabled.

    If the provider/model rejects JSON mode (or JSON-mode validation fails
    server-side), the same call is retried once in plain mode.
    """
    if _PROFILE_JSON_MODE:
        try:
            return profiling_llm.bind(response_format={"type": "json_object"}).invoke(messages)
        except Exception as e:
            logger.warning("[profiling] JSON-mode call failed, retrying in plain mode: %s", str(e)[:200])
    return profiling_llm.invoke(messages)


def _call_json(system: str, user: str, label: str) -> Optional[Dict]:
    """Run one LLM call that must return a JSON object.

    Retries (up to _PROFILE_JSON_ATTEMPTS) with a stricter instruction when the
    reply can't be parsed, and logs finish_reason/length so truncation is visible.
    Returns None when every attempt fails.
    """
    if profiling_llm is None:
        return None

    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    for attempt in range(1, _PROFILE_JSON_ATTEMPTS + 1):
        try:
            response = _invoke(messages)
        except Exception as e:
            logger.warning("[profiling] %s: LLM call failed (attempt %d/%d): %s",
                           label, attempt, _PROFILE_JSON_ATTEMPTS, e)
            continue

        text = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_json(text or "", warn=False)
        if parsed is not None:
            if attempt > 1:
                logger.info("[profiling] %s: parsed on attempt %d", label, attempt)
            return parsed

        try:
            meta = describe_response(response)
        except Exception:
            meta = "n/a"
        logger.warning(
            "[profiling] %s: reply is not valid JSON (attempt %d/%d): %s, %d chars, ends with %r",
            label, attempt, _PROFILE_JSON_ATTEMPTS, meta, len(text or ""), (text or "")[-120:],
        )
        messages = [SystemMessage(content=system), HumanMessage(content=f"{user}\n\n{_RETRY_NOTE}")]

    logger.error("[profiling] %s: no valid JSON after %d attempts", label, _PROFILE_JSON_ATTEMPTS)
    return None


# --- Call 1: Domain classification ---

def _call_domain(column_names: List[str], sample_rows: List[Dict]) -> Optional[Dict]:
    """
    LLM Call 1: Identify the dataset domain from column names and a few sample rows.
    Fast and reliable — just needs column names, no stats required.
    """
    system = (
        "You are a data domain classifier. Given column names and a few sample rows from a dataset, "
        "identify the business or scientific domain it belongs to.\n\n"
        "Respond with ONLY a valid JSON object:\n"
        "{\n"
        "  \"domain\": {\n"
        "    \"category\": \"string (be specific; not limited to examples. Prefer '<industry>/<subdomain>' like 'Retail/E-commerce', 'Marketing/Ads Attribution', 'FinTech/Payments', 'Healthcare/Claims', 'Media/YouTube Analytics', 'Scientific/Bioinformatics')\",\n"
        "    \"confidence\": float (0.0-1.0),\n"
        "    \"reasoning\": \"string (1-2 sentences)\"\n"
        "  }\n"
        "}"
    )
    # Column names + first 3 rows (long values shortened) keep the prompt small
    user = (
        f"Column names: {column_names}\n\n"
        f"Sample rows (first 3):\n{json.dumps(_trim_rows(sample_rows, n=3), default=str)}\n\n"
        "Classify the domain."
    )
    parsed = _call_json(system, user, "domain")
    domain = parsed.get("domain") if parsed else None
    return domain if isinstance(domain, dict) else None


# --- Call 2: Column semantics (batched for wide schemas) ---

def _call_column_semantics_batch(
    column_batch: List[str],
    domain_category: str,
    col_stats: Dict[str, Any],
    sample_rows: List[Dict],
) -> Dict[str, Any]:
    """
    LLM Call 2 (per batch): Explain what each column represents and estimate its predictive value.
    Works on a batch of _SEMANTICS_BATCH_SIZE columns to keep prompt and reply short.
    """
    batch_stats = {}
    for col in column_batch:
        stats = col_stats.get(col, {})
        batch_stats[col] = {
            "type": stats.get("type", "unknown"),
            "missing_pct": round(stats.get("missing_ratio", 0) * 100, 1),
            "n_unique": stats.get("n_unique", 0),
            "top_values": stats.get("top_values", [])[:3],
        }

    system = (
        f"You are analyzing a {domain_category} dataset. For each column provided, give a one-sentence "
        "explanation (under 20 words) of what it represents and rate its predictive value (high/medium/low).\n\n"
        "Respond with ONLY a valid JSON object covering every column provided:\n"
        "{\n"
        "  \"column_explanations\": { \"<col_name>\": \"one-sentence explanation\" },\n"
        "  \"predictive_values\": { \"<col_name>\": \"high|medium|low\" }\n"
        "}"
    )
    # Only this batch's columns in the sample rows: an 85-column row repeated in
    # every batch used to dominate the prompt.
    user = (
        f"Column stats:\n{json.dumps(batch_stats, default=str)}\n\n"
        f"Sample rows (first 3, these columns only):\n"
        f"{json.dumps(_trim_rows(sample_rows, columns=column_batch, n=3), default=str)}\n\n"
        "Explain each column and rate its predictive value."
    )
    parsed = _call_json(system, user, f"column_semantics[{column_batch[0]}..]")
    if not parsed:
        return {}
    return {
        "column_explanations": parsed.get("column_explanations", {}) or {},
        "predictive_values": parsed.get("predictive_values", {}) or {},
    }


def _call_all_column_semantics(
    schema: List[str],
    domain_category: str,
    col_stats: Dict[str, Any],
    sample_rows: List[Dict],
) -> Dict:
    """
    Run Call 2 across all columns in batches of _SEMANTICS_BATCH_SIZE, in parallel
    when there is more than one batch. Merges results back into a single dict and
    reports which columns got no explanation.
    """
    batches = [schema[i:i + _SEMANTICS_BATCH_SIZE] for i in range(0, len(schema), _SEMANTICS_BATCH_SIZE)]
    all_explanations: Dict[str, str] = {}
    all_predictive: Dict[str, str] = {}

    def _run(batch: List[str]) -> Dict:
        return _call_column_semantics_batch(batch, domain_category, col_stats, sample_rows) or {}

    if len(batches) <= 1:
        results = [_run(b) for b in batches]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(batches), _SEMANTICS_MAX_WORKERS)
        ) as pool:
            results = list(pool.map(_run, batches))

    for result in results:
        all_explanations.update(result.get("column_explanations", {}))
        all_predictive.update(result.get("predictive_values", {}))

    missing = [c for c in schema if c not in all_explanations]
    return {
        "column_explanations": all_explanations,
        "predictive_values": all_predictive,
        "columns_without_semantics": missing,
    }


# --- Call 3: Insights (needs real portfolio stats) ---

def _call_insights(
    domain_category: str,
    data_quality: Dict,
    high_value_cols: List[str],
    sample_rows: List[Dict],
    columns_label: str = "High-value columns",
) -> Optional[Dict]:
    """
    LLM Call 3: Generate actionable insights — red flags, analysis suggestions, readiness score.
    Only runs in Phase B once we have real portfolio statistics.
    Wide schemas are truncated to _INSIGHTS_MAX_COLUMNS names so prompt and reply stay short.
    """
    system = (
        f"You are analyzing a {domain_category} dataset and generating actionable insights.\n"
        "Keep every list to at most 5 items and every item under 25 words.\n\n"
        "Respond with ONLY a valid JSON object:\n"
        "{\n"
        "  \"data_quality_insights\": {\n"
        "    \"strengths\": [\"list\"],\n"
        "    \"concerns\": [\"list\"],\n"
        "    \"recommendations\": [\"list\"]\n"
        "  },\n"
        "  \"analysis_suggestions\": [\"specific analysis idea\"],\n"
        "  \"quick_insights\": {\n"
        "    \"data_readiness_score\": float (0-100),\n"
        "    \"readiness_reasoning\": \"string\",\n"
        "    \"key_relationships\": [{\"columns\": [\"col1\", \"col2\"], \"relationship\": \"string\"}],\n"
        "    \"red_flags\": [\"list\"],\n"
        "    \"quick_wins\": [\"list\"]\n"
        "  }\n"
        "}"
    )
    cols = list(high_value_cols or [])
    shown = cols[:_INSIGHTS_MAX_COLUMNS]
    hidden = len(cols) - len(shown)
    cols_text = f"{shown}" + (f" (and {hidden} more columns not listed)" if hidden > 0 else "")
    rows = _trim_rows(sample_rows, columns=shown if hidden > 0 else None, n=5)

    user = (
        f"Data quality summary:\n{json.dumps(data_quality, default=str)}\n\n"
        f"{columns_label} ({len(cols)} total): {cols_text}\n\n"
        f"Sample rows (first 5):\n{json.dumps(rows, default=str)}\n\n"
        "Generate insights."
    )
    return _call_json(system, user, "insights")


# =========================
# PUBLIC ENTRY POINTS
# =========================

def _clean_schema(schema: List[str]) -> List[str]:
    """Drop blank column names (e.g. from a trailing comma in a CSV header)."""
    return [c for c in (schema or []) if str(c).strip()]


def profile_quick(
    schema: List[str],
    sample_rows: List[Dict[str, Any]],
    col_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Phase A profiling: domain classification + column semantics.
    Runs on quick sample rows — no portfolio stats needed.
    Returns quick_profiling_result and profiling_status.
    """
    schema = _clean_schema(schema)
    logger.info(f"profile_quick: starting for {len(schema)} columns")

    if profiling_llm is None:
        return {
            "quick_profiling_result": {"status": "skipped", "reason": "LLM not available"},
            "profiling_status": "skipped",
        }

    try:
        # Call 1: domain
        domain = _call_domain(schema, sample_rows) or {
            "category": "Unknown", "confidence": 0.0, "reasoning": "Domain detection skipped"
        }
        domain_category = domain.get("category", "Unknown")

        # Call 2: column semantics (using basic stats from quick sample if available)
        semantics = _call_all_column_semantics(
            schema=schema,
            domain_category=domain_category,
            col_stats=col_stats or {},
            sample_rows=sample_rows,
        )

        quick_result = {
            "domain": domain,
            "column_explanations": semantics.get("column_explanations", {}),
            "predictive_values": semantics.get("predictive_values", {}),
            "columns_without_semantics": semantics.get("columns_without_semantics", []),
            "llm_calls": ["domain", "column_semantics"],
        }

        logger.info(f"profile_quick: done, domain={domain_category}")
        return {
            "quick_profiling_result": quick_result,
            "profiling_status": "quick_profile",
        }

    except Exception as e:
        logger.error(f"profile_quick failed: {e}", exc_info=True)
        return {
            "quick_profiling_result": {"status": "error", "reason": str(e)},
            "profiling_status": "error",
        }


def profile_full(
    schema: List[str],
    sample_rows: List[Dict[str, Any]],
    sample_statistics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Phase B profiling: domain + column semantics (re-run with real stats) + insights.
    Needs portfolio statistics from sample_with_profiling() for accurate results.
    Returns full_profiling_result and profiling_status.

    full_profiling_result["incomplete_sections"] lists any part whose LLM call
    never returned valid JSON, so callers/UI can tell a partial profile apart.
    """
    schema = _clean_schema(schema)
    logger.info(f"profile_full: starting for {len(schema)} columns")

    if profiling_llm is None:
        return {
            "full_profiling_result": {"status": "skipped", "reason": "LLM not available"},
            "profiling_status": "skipped",
        }

    try:
        col_stats = sample_statistics.get("column_statistics", {})
        data_quality = sample_statistics.get("data_quality", {})

        # Call 1: domain
        domain_raw = _call_domain(schema, sample_rows)
        domain = domain_raw or {
            "category": "Unknown", "confidence": 0.0, "reasoning": "Domain detection skipped"
        }
        domain_category = domain.get("category", "Unknown")

        if _PROFILE_PARALLEL:
            # Calls 2 and 3 only depend on the domain from Call 1, so run them
            # concurrently. Call 3 gets the column list instead of Call 2's
            # "high" columns, since it no longer waits for Call 2.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                sem_future = pool.submit(
                    _call_all_column_semantics,
                    schema=schema,
                    domain_category=domain_category,
                    col_stats=col_stats,
                    sample_rows=sample_rows,
                )
                ins_future = pool.submit(
                    _call_insights,
                    domain_category,
                    data_quality,
                    list(schema),
                    sample_rows,
                    columns_label="Columns",
                )
                semantics = sem_future.result() or {}
                insights_raw = ins_future.result()
            predictive_values = semantics.get("predictive_values", {})
        else:
            # Call 2: column semantics with real stats
            semantics = _call_all_column_semantics(
                schema=schema,
                domain_category=domain_category,
                col_stats=col_stats,
                sample_rows=sample_rows,
            )

            # Identify high-value columns from Call 2 results for insights prompt
            predictive_values = semantics.get("predictive_values", {})
            high_value_cols = [col for col, val in predictive_values.items() if val == "high"]

            # Call 3: insights (needs real stats)
            insights_raw = _call_insights(domain_category, data_quality, high_value_cols, sample_rows)

        insights = insights_raw or {}
        missing_cols = semantics.get("columns_without_semantics", [])

        incomplete: List[str] = []
        if domain_raw is None:
            incomplete.append("domain")
        if missing_cols:
            incomplete.append("column_semantics")
        if insights_raw is None:
            incomplete.append("insights")

        full_result = {
            "domain": domain,
            "column_explanations": semantics.get("column_explanations", {}),
            "predictive_values": predictive_values,
            "data_quality_insights": insights.get("data_quality_insights", {}),
            "analysis_suggestions": insights.get("analysis_suggestions", []),
            "quick_insights": insights.get("quick_insights", {}),
            "llm_calls": ["domain", "column_semantics", "insights"],
            "incomplete_sections": incomplete,
            "columns_without_semantics": missing_cols,
        }

        if incomplete:
            logger.warning(
                "profile_full: PARTIAL profile, missing %s (%d/%d columns without semantics)",
                incomplete, len(missing_cols), len(schema),
            )
        logger.info(
            "profile_full: done, domain=%s, readiness_score=%s, complete=%s",
            domain_category,
            full_result.get("quick_insights", {}).get("data_readiness_score"),
            not incomplete,
        )
        return {
            "full_profiling_result": full_result,
            "profiling_status": "full_profile",
        }

    except Exception as e:
        logger.error(f"profile_full failed: {e}", exc_info=True)
        return {
            "full_profiling_result": {"status": "error", "reason": str(e)},
            "profiling_status": "error",
        }


def format_profile_as_table(
    profiling_result: Optional[Dict[str, Any]] = None,
    sample_statistics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert profiling_result + sample_statistics into a flat list of row-dicts
    that the UI can render as a table via output_json.
    """
    profiling_result = profiling_result or {}
    sample_statistics = sample_statistics or {}

    col_stats = sample_statistics.get("column_statistics", {})
    explanations = profiling_result.get("column_explanations", {})
    predictive = profiling_result.get("predictive_values", {})

    rows = []
    for col, stats in col_stats.items():
        rows.append({
            "column": col,
            "type": stats.get("type", "unknown"),
            "missing_%": stats.get("missing_pct", 0),
            "unique": stats.get("n_unique", 0),
            "explanation": explanations.get(col, ""),
            "predictive_value": predictive.get(col, ""),
        })

    return rows


# =========================
# BACKWARD COMPAT: kept for existing callers (profile_dataset, profiling_agent_node)
# =========================

def _build_system_prompt() -> str:
    """Legacy: build the original monolithic system prompt. Still used by profile_dataset()."""
    return """You are an expert data profiler and domain classifier for Avaloka, an AI data science platform.

Your role is to analyze dataset characteristics and provide semantic insights that help data scientists understand their data quickly and make informed decisions.

**Your Responsibilities:**
1. **Domain Classification**: Identify the business/scientific domain (e.g., HR, Finance, E-commerce, Healthcare, Scientific)
2. **Column Explanations**: Describe what each column represents in clear, natural language
3. **Data Quality Assessment**: Interpret statistical metrics into actionable insights about data quality
4. **Predictive Power Analysis**: Rank columns by their potential value for modeling and analysis
5. **Analysis Suggestions**: Recommend specific analyses that would be valuable for this dataset
6. **Quick Insights**: Provide at-a-glance assessment including data readiness score, relationships, red flags, and quick wins

**Guidelines:**
- Be fact-based: Ground insights in the provided statistics
- Be concise: Each explanation should be 1-2 sentences maximum
- Be domain-aware: Use appropriate terminology for the identified domain
- Be actionable: Focus on insights that inform next steps
- Be specific: Reference actual column names and values from the data

**Important Context:**
- This profile will be used by both the planner and coder agents
- The profile may extend into feature engineering recommendations (future)
- Users rely on this to understand uploaded data immediately after upload
- Quick insights should be actionable and help users make immediate decisions

**Quick Insights Guidelines:**
- **Data Readiness Score (0-100)**: Rate how ready this data is for analysis
  * 90-100: Excellent - ready for immediate analysis
  * 70-89: Good - minor issues that don't block analysis
  * 50-69: Fair - needs attention before serious analysis
  * 0-49: Poor - significant issues must be addressed first
- **Key Relationships**: Identify obvious relationships (e.g., "event_time and user_session form session boundaries")
- **Red Flags**: Critical blockers (e.g., "50% missing in target column", "duplicate keys detected")
- **Quick Wins**: Easy fixes with high impact (e.g., "Parse event_time as datetime for time-series analysis")

**Output Format:**
You MUST respond with ONLY a valid JSON object (no markdown, no explanation) with this exact structure:

{
  "domain": {
    "category": "string (e.g., 'HR/Employee', 'Finance/Banking', 'Healthcare', 'E-commerce', 'Scientific/Research')",
    "confidence": float (0.0-1.0),
    "reasoning": "string (2-3 sentences explaining why this domain was identified)"
  },
  "column_explanations": {
    "{column_name}": "string (concise explanation of what this column represents and its purpose)"
  },
  "data_quality_insights": {
    "strengths": ["list of positive quality aspects found in the data"],
    "concerns": ["list of data quality issues that should be addressed"],
    "recommendations": ["list of suggested data cleaning or preparation steps"]
  },
  "predictive_power": {
    "high_value_columns": [
      {"column": "name", "reason": "why this column is valuable for analysis/modeling"}
    ],
    "low_value_columns": [
      {"column": "name", "reason": "why this column has limited value (e.g., constant, high missing ratio)"}
    ]
  },
  "analysis_suggestions": [
    "string (specific analysis idea based on this data, e.g., 'Analyze salary distribution by department', 'Time-series forecasting on sales data')"
  ],
  "quick_insights": {
    "data_readiness_score": float (0-100, overall readiness for analysis),
    "readiness_reasoning": "string (why this score was given)",
    "key_relationships": [
      {"columns": ["col1", "col2"], "relationship": "string describing the relationship"}
    ],
    "red_flags": ["list of critical issues that must be addressed before analysis"],
    "quick_wins": ["list of easy improvements that would significantly enhance data quality"]
  }
}

Respond with ONLY the JSON object. Do not include markdown code blocks, explanations, or any other text."""


def _build_user_prompt(
    profiling_result: Dict[str, Any],
    sample_rows: List[Dict[str, Any]],
    schema: List[str],
    dataset_name: Optional[str] = None
) -> str:
    """
    Legacy: Build user prompt with statistical context and sample data.
    Still used by profile_dataset() for backward compat.
    """
    data_shape = profiling_result.get("data_shape", {})
    column_statistics = profiling_result.get("column_statistics", {})
    data_quality = profiling_result.get("data_quality", {})
    portfolio_meta = profiling_result.get("portfolio_metadata", {})

    total_rows = data_shape.get("rows", 0)
    total_columns = data_shape.get("columns", 0)
    completeness = data_quality.get("overall_completeness", 1.0) * 100
    sizing_tier = portfolio_meta.get("sizing_tier", "UNKNOWN")

    prompt_parts = []

    dataset_info = f"Dataset: {dataset_name}" if dataset_name else "Dataset Analysis"
    prompt_parts.append(f"**{dataset_info}**")
    prompt_parts.append(f"- Total rows: {total_rows:,}")
    prompt_parts.append(f"- Total columns: {total_columns}")
    prompt_parts.append(f"- Data completeness: {completeness:.1f}%")
    prompt_parts.append(f"- Size tier: {sizing_tier}")
    prompt_parts.append("")

    prompt_parts.append("**Column Profiles:**")
    for col_name in schema:
        col_stats = column_statistics.get(col_name, {})
        col_type = col_stats.get("type", "unknown")
        missing_ratio = col_stats.get("missing_ratio", 0.0) * 100
        n_unique = col_stats.get("n_unique", 0)

        prompt_parts.append(f"\\n{col_name}:")
        prompt_parts.append(f"  - Type: {col_type}")
        prompt_parts.append(f"  - Missing: {missing_ratio:.1f}%")
        prompt_parts.append(f"  - Unique values: {n_unique:,}")

        if col_type == "numeric":
            mean = col_stats.get("mean", 0)
            std = col_stats.get("std", 0)
            min_val = col_stats.get("min", 0)
            max_val = col_stats.get("max", 0)
            skewness = col_stats.get("skewness", 0)
            prompt_parts.append(f"  - Range: [{min_val:.2f}, {max_val:.2f}]")
            prompt_parts.append(f"  - Mean: {mean:.2f}, Std: {std:.2f}")
            prompt_parts.append(f"  - Skewness: {skewness:.2f}")

        elif col_type == "categorical":
            top_values = col_stats.get("top_values", [])[:5]
            if top_values:
                top_str = ", ".join([f"{v['value']} ({v['count']})" for v in top_values])
                prompt_parts.append(f"  - Top values: {top_str}")

    prompt_parts.append("")

    prompt_parts.append("**Sample Rows (first 10):**")
    for i, row in enumerate(sample_rows[:10], 1):
        prompt_parts.append(f"Row {i}: {json.dumps(round_floats(row), default=str)}")

    prompt_parts.append("")

    if data_quality.get("columns_with_high_missing"):
        prompt_parts.append("**Data Quality Alerts:**")
        prompt_parts.append(f"- High missing ratio (>30%): {', '.join(data_quality['columns_with_high_missing'])}")
    if data_quality.get("constant_columns"):
        prompt_parts.append(f"- Constant columns: {', '.join(data_quality['constant_columns'])}")

    prompt_parts.append("")
    prompt_parts.append("Analyze this dataset and provide your insights in the JSON format specified in the system prompt.")

    return "\\n".join(prompt_parts)


# =========================
# LLM RESPONSE PARSING (legacy)
# =========================

def _parse_llm_response(response_text: str) -> Optional[Dict]:
    """Parse and validate LLM JSON response (legacy monolithic call)."""
    if not response_text:
        return None
    result = _parse_json(response_text)
    if result and "domain" in result:
        return result
    return None


# =========================
# MAIN PROFILING FUNCTION (legacy — kept for backward compat)
# =========================

def profile_dataset(
    profiling_result: Dict[str, Any],
    sample_rows: List[Dict[str, Any]],
    schema: List[str],
    dataset_name: Optional[str] = None,
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Legacy entry point: enhance statistical profile with semantic analysis using one monolithic LLM call.
    Kept for backward compat. New code should use profile_quick() or profile_full() instead.
    """
    logger.info("=== Data Profiling Agent: Starting semantic analysis ===")

    enhanced_profile = profiling_result.copy()

    enhanced_profile["semantic_analysis"] = {
        "status": "pending",
        "llm_used": False
    }

    if not use_llm or profiling_llm is None:
        logger.info("LLM analysis skipped (disabled or not available)")
        enhanced_profile["semantic_analysis"]["status"] = "skipped"
        enhanced_profile["semantic_analysis"]["reason"] = "LLM not available or disabled"
        return enhanced_profile

    try:
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(profiling_result, sample_rows, schema, dataset_name)

        logger.info(f"Invoking LLM for semantic analysis ({len(sample_rows)} sample rows, {len(schema)} columns)")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = profiling_llm.invoke(messages)
        logger.info("Profiling LLM answered: %s", describe_response(response))
        response_text = response.content if hasattr(response, 'content') else str(response)

        semantic_insights = _parse_llm_response(response_text)

        if semantic_insights:
            enhanced_profile["semantic_analysis"] = {
                "status": "completed",
                "llm_used": True,
                **semantic_insights
            }

            column_explanations = semantic_insights.get("column_explanations", {})
            column_statistics = enhanced_profile.get("column_statistics", {})

            for col_name, explanation in column_explanations.items():
                if col_name in column_statistics:
                    column_statistics[col_name]["explanation"] = explanation

            high_value_cols = {item["column"]: item["reason"] for item in semantic_insights.get("predictive_power", {}).get("high_value_columns", [])}
            low_value_cols = {item["column"]: item["reason"] for item in semantic_insights.get("predictive_power", {}).get("low_value_columns", [])}

            for col_name in column_statistics:
                if col_name in high_value_cols:
                    column_statistics[col_name]["predictive_value"] = "high"
                    column_statistics[col_name]["predictive_reason"] = high_value_cols[col_name]
                elif col_name in low_value_cols:
                    column_statistics[col_name]["predictive_value"] = "low"
                    column_statistics[col_name]["predictive_reason"] = low_value_cols[col_name]
                else:
                    column_statistics[col_name]["predictive_value"] = "medium"

            logger.info(f"Semantic analysis completed: domain={semantic_insights.get('domain', {}).get('category', 'unknown')}")

        else:
            logger.warning("LLM returned invalid response, using statistical profile only")
            enhanced_profile["semantic_analysis"] = {
                "status": "failed",
                "llm_used": True,
                "reason": "LLM returned invalid JSON response",
                "fallback": "statistical_profile_only"
            }

    except Exception as e:
        logger.error(f"Semantic analysis failed: {e}", exc_info=True)
        enhanced_profile["semantic_analysis"] = {
            "status": "error",
            "llm_used": True,
            "reason": str(e),
            "fallback": "statistical_profile_only"
        }

    return enhanced_profile


# =========================
# LANGGRAPH INTEGRATION
# =========================

def profiling_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for data profiling.

    Reads sample_statistics (set by sampler agent) and writes back
    quick_profiling_result or full_profiling_result depending on sample_status.
    Falls back to reading legacy profiling_result if new keys aren't present.
    """
    try:
        logger.info("--- Entering Data Profiling Agent ---")

        sample_status = state.get("sample_status")
        sample_statistics = state.get("sample_statistics", {})

        # Determine which sample rows to use
        if sample_status == "full_sample" and state.get("full_sample_rows"):
            sample_rows = state["full_sample_rows"]
        elif state.get("quick_sample_rows"):
            sample_rows = state["quick_sample_rows"]
        else:
            # Legacy fallback
            sample_rows = state.get("uploaded_csv_preview") or state.get("sample_rows", [])

        schema = state.get("uploaded_csv_columns") or state.get("schema", [])
        if not schema and sample_statistics:
            schema = list(sample_statistics.get("column_statistics", {}).keys())

        if not sample_rows:
            logger.warning("No sample rows in state, skipping profiling agent")
            return state

        # Convert list-of-lists to list-of-dicts if needed (legacy format)
        if sample_rows and isinstance(sample_rows[0], list):
            dict_rows = []
            for row in sample_rows[1:]:
                row_dict = {col: (row[i] if i < len(row) else None) for i, col in enumerate(schema)}
                dict_rows.append(row_dict)
            sample_rows = dict_rows

        new_state = state.copy()

        if sample_status == "full_sample":
            # Phase B: run full profiling
            result = profile_full(
                schema=schema,
                sample_rows=sample_rows,
                sample_statistics=sample_statistics,
            )
            new_state["full_profiling_result"] = result.get("full_profiling_result")
            new_state["profiling_status"] = result.get("profiling_status")
        else:
            # Phase A: run quick profiling
            col_stats = sample_statistics.get("column_statistics", {})
            result = profile_quick(
                schema=schema,
                sample_rows=sample_rows,
                col_stats=col_stats,
            )
            new_state["quick_profiling_result"] = result.get("quick_profiling_result")
            new_state["profiling_status"] = result.get("profiling_status")

        logger.info(f"Data profiling completed: status={new_state.get('profiling_status')}")
        return new_state

    except Exception as e:
        logger.error(f"Profiling agent node failed: {e}", exc_info=True)
        return state


# =========================
# EXTENSION POINT: Feature Engineering
# =========================

def extract_feature_engineering_context(profiling_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract context for the feature engineering agent.
    Prefers full_profiling_result if available, falls back to quick_profiling_result.
    """
    # Support both new separate state keys and legacy combined key
    full = profiling_result.get("full_profiling_result") or profiling_result.get("semantic_analysis", {})
    column_stats = profiling_result.get("sample_statistics", {}).get("column_statistics", {}) \
        or profiling_result.get("column_statistics", {})

    predictive_values = full.get("predictive_values", {})
    high_value_columns = []
    for col_name, col_stat in column_stats.items():
        pv = predictive_values.get(col_name) or col_stat.get("predictive_value")
        if pv == "high":
            high_value_columns.append({
                "column": col_name,
                "type": col_stat.get("type"),
                "reason": col_stat.get("predictive_reason", ""),
                "stats": col_stat
            })

    return {
        "high_value_columns": high_value_columns,
        "data_quality_concerns": full.get("data_quality_insights", {}).get("concerns", []),
        "domain_context": full.get("domain", {}),
        "suggested_transformations": full.get("analysis_suggestions", []),
        "feature_engineering_ready": len(high_value_columns) > 0
    }