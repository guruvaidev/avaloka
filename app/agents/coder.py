# coder.py
# coder.py
import os
import logging
import json
import csv
import re
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.core.inference import build_chat_model
from app.agents.state import CodingAgentState
from app.agents.preparation_agent import parse_numeric_token
# Re-exported: app/agents/data_transfer_agent/daft_coder.py imports this
# through app.agents.coder, so the name must stay importable here even
# though coder.py no longer calls it itself.
from app.utils import generate_filename_timestamp  # noqa: F401
from app.agents.blueprint_contract import (
    BlueprintContract,
    ContractError,
    parse_blueprint,
)

# Load environment variables
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, ".env"))

logger = logging.getLogger(__name__)

# --- LLM Setup ---
from app.core.model_config import resolve as resolve_model
from app.core.model_fallback import attach_fallback
from app.core.log_utils import describe_messages, describe_response

_coder_api_key = (os.environ.get("GROQ_API_KEY_CODING_AGENT")
              or os.environ.get("GROQ_API_KEY"))
coder_llm = build_chat_model(
    role="coding",
    agent="CODER",
    tier="large",
    temperature=0.0,
    groq_model=resolve_model("coder"),
    groq_api_key=_coder_api_key,
)
if coder_llm is None:
    logger.warning("Coder LLM disabled; set GROQ_API_KEY_CODING_AGENT to re-enable remote generation.")

# --- Helper to extract code from markdown ---
_PY_FENCE_RE = re.compile(r"```(?:python|py)\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+\-]+)?\s*(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str:
    """
    Extract the first fenced code block from LLM output.

    Prefers ```python / ```py blocks, otherwise falls back to any ```...``` fenced block.
    If no fences are found, returns the full text stripped.
    """
    if not text:
        return ""

    m = _PY_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    m = _ANY_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    return text.strip()


def _prompt_safe_path(value: Any) -> str:
    # The model copies paths from the avro example / stub code, and writes them
    # as raw strings -- r"...\" does not compile. Forward slashes work on Windows.
    return str(value or "").replace("\\", "/")


def _format_datasets_context(state: CodingAgentState) -> str:
    """
    Produce a compact JSON string of datasets_context for the LLM.
    The primary dataset is already loaded into df outside main(df).
    """
    datasets = state.get("datasets_context") or state.get("multi_dataset_state") or []
    if not isinstance(datasets, list):
        return "[]"

    compact = []
    for ds in datasets:
        if not isinstance(ds, dict):
            continue
        compact.append(
            {
                "dataset_id": ds.get("dataset_id"),
                "alias": ds.get("alias"),
                "filename": ds.get("filename"),
                "columns": ds.get("columns"),
                "data_source_location": ds.get("data_source_location")
                or ds.get("full_data_location")
                or ds.get("sample_data_location"),
                "full_data_location": ds.get("full_data_location"),
                "sample_data_location": ds.get("sample_data_location"),
                "input_data_type": ds.get("input_data_type", "csv"),
            }
        )

    return json.dumps(compact, ensure_ascii=False)


# --- Shared helpers ---
def _rows_from_list_of_dicts(rows: Any) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append(row)
    return normalized


def _rows_from_list_of_lists(rows: Any) -> List[Dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) < 2:
        return []
    headers = rows[0]
    if not isinstance(headers, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for raw_row in rows[1:]:
        if not isinstance(raw_row, list):
            continue
        row_dict: Dict[str, Any] = {}
        for idx, header in enumerate(headers):
            key = str(header)
            row_dict[key] = raw_row[idx] if idx < len(raw_row) else None
        normalized.append(row_dict)
    return normalized


def _resolve_preview_records(state: CodingAgentState) -> List[Dict[str, Any]]:
    preview = state.get("uploaded_csv_preview")
    # The preview may be persisted as a JSON string (see app/api/helpers.py).
    if isinstance(preview, str) and preview.strip():
        try:
            preview = json.loads(preview)
        except (ValueError, TypeError):
            preview = None
    records = _rows_from_list_of_dicts(preview)
    if records:
        return records
    records = _rows_from_list_of_lists(preview)
    if records:
        return records

    sample_data = state.get("sample_data")
    records = _rows_from_list_of_dicts(sample_data)
    if records:
        return records

    if isinstance(sample_data, str) and sample_data.strip():
        try:
            reader = csv.reader(sample_data.splitlines())
            parsed_rows: List[List[str]] = []
            for row in reader:
                if not row:
                    continue
                parsed_rows.append([cell.strip() for cell in row])
            return _rows_from_list_of_lists(parsed_rows)
        except Exception as err:
            logger.debug("Unable to parse sample data for preview: %s", err)
    return []


def _coerce_response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def _json_for_prompt(value: Any) -> str:
    """JSON-safe pretty serializer for prompt context."""
    try:
        return json.dumps(value, indent=2, default=str)
    except Exception:
        return json.dumps(str(value), indent=2)


def _generate_pseudocode(state: CodingAgentState) -> str:
    if coder_llm is None:
        return ""

    plan_text = state.get("plan") or ""
    user_prompt = state.get("user_prompt") or ""
    instruction = plan_text or user_prompt

    if not instruction:
        messages = state.get("messages") or []
        for message in reversed(messages):
            candidate = getattr(message, "content", None)
            if candidate:
                instruction = candidate
                break
    if not instruction:
        return ""

    schema = state.get("schema") or {}
    preview_rows = _resolve_preview_records(state)
    sample_row: Dict[str, Any] = {}
    if preview_rows:
        sample_row = dict(preview_rows[0])

    system_prompt = """You are a data engineering planning assistant.
Translate the user's request into concise pseudocode steps that describe the required data processing.
Reference columns exactly as they appear in the schema and do not invent new column names unless explicitly requested.
Do not print any data from the df, you should turn it into a column or new dataframe. New dataframe is the default but if the user requests then it should be appended to the current df
Do not write executable code—return a numbered list of high-level steps that can be implemented with pandas.
Do not include filesystem, operating system, shell, subprocess, network, package-management, scheduling, globals(), locals(), print statements, or arbitrary execution instructions.
Do not request interactive input. If the request lacks a required value, describe the missing value instead of inventing an input() step.
Never assume missing filter values such as a weekday, day, date, month, year, category, threshold, or ID.
Handle missing/null values only in the columns needed for the requested operation. Do not drop the whole row set just because one optional field is missing.
For string parsing/counting, first work from a non-null Series for that column, for example df['col'].dropna().astype(str), and then transform/count it.

YEAR FILTERING RULE:
When a query references "year" but no explicit year column exists:
1. Search for date-like columns.
2. Parse dates using pandas.to_datetime(errors="coerce").
3. Extract the year component using .dt.year.
4. Apply the filter on the extracted year.
5. Support multiple date formats automatically.
6. Only raise an error if neither a year column nor any parseable date column exists.

SPARSE VS DENSE SIMILARITY RULE:
For co-occurrence, recommender, user-item, product-order, product-product, pairwise cosine similarity, or high-cardinality categorical matrix requests:
1. Use sparse matrices by default when the request is full, all-unique, entire-dataset, high-cardinality, co-occurrence, recommender, user-item, product-order, product-product, or otherwise unbounded.
2. Dense matrices are acceptable only when the prompt is explicitly small/bounded, such as a fixed small subset of items/users/features, and the expected matrix size is comfortably small.
3. Do not plan a dense pivot table, crosstab, get_dummies matrix, or dense product-by-product DataFrame for unbounded/high-cardinality requests.
4. For sparse paths, plan a scipy sparse COO/CSR incidence matrix.
5. For sparse paths, map both row IDs and column IDs to compact zero-based integer codes before constructing the sparse matrix; never use raw IDs like order_id or product_id as sparse matrix coordinates.
6. If using pd.factorize, remember it returns NumPy code arrays and unique labels. Use len(unique_order_ids) and len(unique_product_ids) for the sparse matrix shape, not order_codes.nunique() or product_codes.nunique().
7. For product-product similarity from an order-by-product incidence matrix, compute cosine_similarity on the product axis with cosine_similarity(incidence_matrix.T, dense_output=False), not cosine_similarity(incidence_matrix, ...). Rows are orders and columns are products.
8. Map sparse result row/column codes back to original product IDs in the final edge-list output.
9. For sparse paths, use cosine_similarity(..., dense_output=False).
10. Return sparse COO edge-list rows instead of materializing zero similarity cells.
11. For symmetric pairwise similarity matrices, remove self-pairs and duplicate mirrored pairs by filtering COO coordinates with row < col before mapping codes back to IDs.

ACCEPTANCE CRITERIA (the contract):
After the numbered steps, emit a fenced ```json block stating the postconditions the
result DataFrame must satisfy. The validator checks these against the real result, so
state what must be TRUE of the output, not how to compute it.

```json
{
  "result_kind": "aggregate | row_transform | subset | reshape | scalar",
  "source_columns": ["exact schema column names the computation reads"],
  "result_columns": [
    {"name": "<output column>", "role": "key | metric | passthrough",
     "dtype": "numeric | string | datetime | boolean | any",
     "min": <optional number>, "max": <optional number>}
  ],
  "row_relation": {"type": "one_per_group | same_as_input | fewer_than_input | at_most | exactly | unconstrained",
                   "group_by": ["grouping column(s), for one_per_group"],
                   "n": <number, for at_most/exactly>},
  "unique_key": ["column(s) that must not repeat"],
  "not_null": ["column(s) that must have no missing values"]
}
```

Rules for the contract itself:
- Use ONLY the field names and values listed above.
- "source_columns" must be exact column names from the schema. Never invent one.
- If the request aggregates, groups, counts, sums, averages, takes a rate/share, bins a
  distribution, or plots X by Y, then result_kind is "aggregate" and row_relation is
  "one_per_group" with the grouping column(s). That is what makes returning the raw
  input rows a detectable failure.
- A proportion, rate or share that you express as a 0-1 fraction gets "min": 0, "max": 1.
- State only postconditions you are confident the correct result satisfies. An
  over-tight contract rejects correct code; prefer omitting a criterion to guessing one.
"""

    details: List[str] = []
    if user_prompt:
        details.append(f"User request:\n{user_prompt}")
    if plan_text:
        details.append(f"Planner guidance:\n{plan_text}")
    request_block = "\n\n".join(details) if details else instruction

    human_prompt = (
        f"{request_block}\n\n"
        f"Schema (column: dtype):\n{_json_for_prompt(schema)}\n"
    )
    if sample_row:
        human_prompt += f"\nSample row:\n{_json_for_prompt(sample_row)}\n"

    # Inject Context Memory directly into the Pseudocode planner's awareness
    if state.get("memory_hints"):
        hints_str = "\n- ".join(state["memory_hints"])
        human_prompt += f"\n[CONTEXT MEMORY - USER PREFERENCES & PAST INSIGHTS]:\n- {hints_str}\n"
    
    if state.get("session_logic_signature"):
        human_prompt += f"\n[SESSION LOGIC SIGNATURE]:\n{state['session_logic_signature']}\n"

    human_prompt += (
        "\nRespond with the numbered pseudocode steps, then the fenced json "
        "acceptance-criteria block."
    )

    try:
        response = coder_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        logger.info("Coder LLM (pseudocode) answered: %s", describe_response(response))
    except Exception as err:
        logger.error("Failed to generate pseudocode interpretation: %s", err)
        return ""

    pseudocode = _coerce_response_text(response)
    logger.info("--- Generated pseudocode blueprint ---\n%s", pseudocode)
    return pseudocode


def _split_blueprint(raw: str, state: CodingAgentState):
    """Split a blueprint response into prose steps and a parsed contract.

    A contract that cannot be read, or that asserts nothing checkable, is
    discarded rather than trusted -- the pipeline then behaves exactly as it did
    before contracts existed. A gate that silently passes everything is worse
    than no gate.
    """
    steps, contract, problems = parse_blueprint(raw, schema=state.get("schema"))
    for problem in problems:
        logger.info("Blueprint contract: %s", problem)
    if contract is not None:
        logger.info("--- Blueprint contract ---\n%s",
                    json.dumps(contract.to_dict(), indent=2))
    return steps, contract


def _contract_from_state(state: CodingAgentState):
    """Re-hydrate the contract carried in state across a retry."""
    raw = state.get("coder_contract")
    if not raw:
        return None
    try:
        return BlueprintContract.parse(raw)
    except ContractError:
        return None


_GUARD_COLUMN_RE = re.compile(r"on the\s+'([^']+)'\s+column")


def _contract_contradicts_math_guard(contract, math_err: str, state: CodingAgentState) -> bool:
    """Should the blueprint's declared intent override the math-on-string guard?

    The guard (``app/agents/planner.py::_detect_math_on_string_column``) decides
    by regex over the user's wording which column an aggregation verb targets.
    That guess misfires whenever a text or date column is merely *named* in a
    request whose arithmetic lands somewhere else -- "monthly admission counts
    from the date of admission ... 3-month rolling average" is refused outright
    because 'Date of Admission' is the only column named and 'average' appears.

    The blueprint now states the result columns explicitly. If none of them is
    the flagged column, the plan is not producing "the average of <text
    column>", the guard's premise is false, and the request should proceed.

    Deliberately conservative: with no contract, or with no numeric metric
    declared, the guard keeps its original behaviour.
    """
    if contract is None or not math_err:
        return False

    match = _GUARD_COLUMN_RE.search(math_err)
    if not match:
        return False
    flagged = match.group(1)

    numeric_metrics = [
        c for c in contract.result_columns
        if c.role == "metric" and c.dtype in {"numeric", "any"}
    ]
    if not numeric_metrics:
        return False

    # If any declared result column resolves to the flagged column, the plan
    # really is trying to produce a value from it -- keep the block.
    for spec in contract.result_columns:
        if _contract_names_match(spec.name, flagged):
            return False
    return True


def _contract_names_match(a: str, b: str) -> bool:
    na = re.sub(r"[^a-z0-9]", "", str(a).lower())
    nb = re.sub(r"[^a-z0-9]", "", str(b).lower())
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def _is_numeric_dtype(dtype: str) -> bool:
    lowered = str(dtype).lower()
    return any(token in lowered for token in ["int", "float", "double", "decimal", "number"])


def _samples_look_numeric(column: str, preview_records: Any) -> bool:
    """True when most sample values are numbers once formatting is stripped."""
    if not preview_records:
        return False
    numeric = 0
    total = 0
    for row in list(preview_records)[:10]:
        if not isinstance(row, dict):
            continue
        raw = row.get(column)
        if raw is None:
            continue
        if not str(raw).strip():
            continue
        total += 1
        if parse_numeric_token(raw) is not None:
            numeric += 1
    return total > 0 and numeric > total * 0.5


def _contains_alpha(text: str) -> bool:
    return any(char.isalpha() for char in text)


def _detect_math_on_string_column_coder(state: CodingAgentState) -> Optional[str]:
    """
    Detect when the user requests a mathematical aggregation on a non-numeric column.
    Returns an error message if invalid, or None if OK.
    """
    # Judge the user's actual request; the plan is machine-generated prose
    # whose wording ("assumption", "column named ...") must not trip the guard.
    plan_raw = state.get("user_prompt") or state.get("plan") or ""
    if not plan_raw:
        messages = state.get("messages") or []
        for message in reversed(messages):
            content = getattr(message, "content", None)
            if content:
                plan_raw = content
                break
    if not plan_raw:
        return None

    from app.agents.planner import _detect_math_on_string_column

    # The planner function expects 'uploaded_csv_preview' for sample data validation.
    # We map our state keys (like sample_data) if needed to ensure compatibility.
    mapped_state = dict(state)
    if not mapped_state.get("uploaded_csv_preview") and "sample_data" in mapped_state:
        records = _resolve_preview_records(state)
        if records:
            headers = list(records[0].keys())
            preview_list_of_lists = [headers]
            for r in records:
                preview_list_of_lists.append([r.get(h) for h in headers])
            mapped_state["uploaded_csv_preview"] = preview_list_of_lists
            if not mapped_state.get("uploaded_csv_columns"):
                mapped_state["uploaded_csv_columns"] = headers

    return _detect_math_on_string_column(plan_raw, mapped_state)


def _patch_df_str_accessor(code: str) -> str:
    """
    Rewrite df['col'].str... -> df['col'].astype("string").str...
    Idempotent (won't double-apply).

    "string" (the nullable StringDtype), not str. `astype(str)` renders a missing
    value as the literal "nan", which then survives dropna() and is written to the
    CSV -- where pandas reads it back as NaN. op_null_drop_rows failed exactly
    that way: correct generated code ("drop rows with a missing Country") returned
    rows whose Country was missing, because this rewrite had turned NaN into a
    non-null string first. StringDtype keeps <NA> as <NA> and supports every .str
    method identically.
    """
    pattern = r"(df\[\s*(['\"]).+?\2\s*\])\.str\."
    repl = r'\1.astype("string").str.'
    return re.sub(pattern, repl, code)


def _generate_stub_code(state: CodingAgentState) -> str:
    """Return deterministic pandas code when Groq LLM is unavailable."""
    data_source = _prompt_safe_path(state.get("data_source_location") or "")
    output_location = _prompt_safe_path(state.get("output_location") or "output.csv")
    plan_raw = state.get("plan") or ""
    instruction_raw = plan_raw

    if not instruction_raw:
        user_prompt = state.get("user_prompt")
        if user_prompt:
            instruction_raw = user_prompt

    if not instruction_raw:
        messages = state.get("messages") or []
        for message in reversed(messages):
            content = getattr(message, "content", None)
            if content:
                instruction_raw = content
                break

    plan_raw = instruction_raw or ""
    if not isinstance(plan_raw, str):
        plan_raw = str(plan_raw)
    plan_text = plan_raw.lower()
    schema = state.get("schema") or {}
    preview_records = _resolve_preview_records(state)

    value_to_columns: dict[str, List[str]] = {}
    value_to_original: dict[str, str] = {}
    if preview_records:
        for row in preview_records:
            if not isinstance(row, dict):
                continue
            for header, cell in row.items():
                original_value = str(cell).strip()
                cell_value = original_value.lower()
                if not cell_value:
                    continue
                value_to_columns.setdefault(cell_value, [])
                if header not in value_to_columns[cell_value]:
                    value_to_columns[cell_value].append(header)
                value_to_original.setdefault(cell_value, original_value)

    columns_map = {column.lower(): column for column in schema.keys()}
    object_columns = [
        col
        for col, dtype in schema.items()
        if "object" in str(dtype).lower() or "string" in str(dtype).lower()
    ]
    numeric_columns = [col for col, dtype in schema.items() if _is_numeric_dtype(dtype)]
    # Columns typed String that hold formatted numbers are valid targets too.
    numeric_columns += [
        col
        for col in object_columns
        if col not in numeric_columns and _samples_look_numeric(col, preview_records)
    ]

    raw_literals = [m[0] or m[1] for m in re.findall(r"'([^']+)'|\"([^\"]+)\"", plan_raw)]
    string_literals = [literal for literal in raw_literals if literal]
    string_literals_lower = [value.lower() for value in string_literals]

    mentioned_values: List[tuple[str, str]] = []
    for value_lower, columns in value_to_columns.items():
        if value_lower and value_lower in plan_text:
            original_value = value_to_original.get(value_lower, value_lower)
            numeric_candidate = original_value.replace(",", "").replace("_", "")
            if not _contains_alpha(original_value) and numeric_candidate.replace(".", "", 1).isdigit():
                continue
            mentioned_values.append((original_value, value_lower))

    for literal, literal_lower in zip(string_literals, string_literals_lower):
        if literal_lower not in {val_lower for _, val_lower in mentioned_values}:
            mentioned_values.append((literal, literal_lower))

    code_lines: List[str] = [
        "import pandas as pd",
        "",
        "def main(df):",
    ]

    transformations: List[str] = []

    filter_keywords = ["filter", "where", "only include", "keep only", "include only", "return only", "show only"]
    if any(keyword in plan_text for keyword in filter_keywords):
        filter_applied = False

        for literal, literal_lower in mentioned_values:
            candidate_columns = value_to_columns.get(literal_lower, [])
            if not candidate_columns:
                candidate_columns = [
                    column_name
                    for column_lower, column_name in columns_map.items()
                    if literal_lower in column_lower
                ]
            for candidate in candidate_columns:
                transformations.append(f"    df = df[df[{repr(candidate)}] == {repr(literal)}]")
                filter_applied = True
                break
            if filter_applied:
                break

        if not filter_applied:
            for column_lower, column_name in columns_map.items():
                if column_lower not in plan_text:
                    continue

                comparison = re.search(
                    rf"{re.escape(column_lower)}\s*(>=|<=|>|<)\s*(\d+(?:\.\d+)?)",
                    plan_text,
                )
                if comparison:
                    op, number = comparison.groups()
                    transformations.append(f"    df = df[df[{repr(column_name)}] {op} {number}]")
                    filter_applied = True
                    break

                greater_than = re.search(
                    rf"{re.escape(column_lower)}\s*(?:greater than|more than|above)\s*(\d+(?:\.\d+)?)",
                    plan_text,
                )
                if greater_than:
                    number = greater_than.group(1)
                    transformations.append(f"    df = df[df[{repr(column_name)}] > {number}]")
                    filter_applied = True
                    break

                less_than = re.search(
                    rf"{re.escape(column_lower)}\s*(?:less than|under|below)\s*(\d+(?:\.\d+)?)",
                    plan_text,
                )
                if less_than:
                    number = less_than.group(1)
                    transformations.append(f"    df = df[df[{repr(column_name)}] < {number}]")
                    filter_applied = True
                    break

        if not filter_applied and mentioned_values:
            for literal, literal_lower in mentioned_values:
                candidate_columns = value_to_columns.get(literal_lower, [])
                if len(candidate_columns) == 1:
                    transformations.append(f"    df = df[df[{repr(candidate_columns[0])}] == {repr(literal)}]")
                    filter_applied = True
                    break

        if not filter_applied and mentioned_values and len(object_columns) == 1:
            transformations.append(
                f"    df = df[df[{repr(object_columns[0])}] == {repr(mentioned_values[0][0])}]"
            )

    aggregate_keywords = {
        "sum": "sum",
        "total": "sum",
        "average": "mean",
        "avg": "mean",
        "mean": "mean",
        "count": "count",
        "maximum": "max",
        "max": "max",
        "minimum": "min",
        "min": "min",
    }

    aggregate_func = None
    for keyword, func in aggregate_keywords.items():
        if keyword in plan_text:
            aggregate_func = func
            break

    numeric_targets = [col for col in numeric_columns if col.lower() in plan_text]
    target_column = numeric_targets[0] if numeric_targets else (numeric_columns[0] if numeric_columns else None)
    scalar_summary_requested = aggregate_func and "group" not in plan_text and "aggregate" not in plan_text
    range_requested = "range" in plan_text or ("minimum" in plan_text and "maximum" in plan_text)

    if scalar_summary_requested and target_column:
        # A bare pd.to_numeric on "₹1,099" yields NaN for every row.
        transformations.append(f"    _raw = df[{repr(target_column)}]")
        transformations.append("    from app.agents.preparation_agent import parse_numeric_token")
        transformations.append(
            "    series = _raw if pd.api.types.is_numeric_dtype(_raw) else "
            "_raw.map(parse_numeric_token).astype('float64')"
        )
        transformations.append("    result = {}")
        transformations.append(f"    result['column'] = {repr(target_column)}")
        if range_requested:
            transformations.append("    result['minimum'] = series.min()")
            transformations.append("    result['maximum'] = series.max()")
            transformations.append("    result['absolute_range'] = series.max() - series.min()")
        elif aggregate_func == "count":
            transformations.append("    result['count'] = int(series.count())")
        elif aggregate_func == "sum":
            transformations.append("    result['sum'] = series.sum()")
        elif aggregate_func == "mean":
            transformations.append("    result['mean'] = series.mean()")
        elif aggregate_func == "min":
            transformations.append("    result['minimum'] = series.min()")
        elif aggregate_func == "max":
            transformations.append("    result['maximum'] = series.max()")
        transformations.append("    return pd.DataFrame([result])")

    if not scalar_summary_requested and ("group" in plan_text or "aggregate" in plan_text or aggregate_func):
        group_columns: List[str] = []
        for column_lower, column_name in columns_map.items():
            if column_name in numeric_columns:
                continue
            if any(
                token in plan_text
                for token in [
                    f"group by {column_lower}",
                    f"per {column_lower}",
                    f"each {column_lower}",
                    f"by {column_lower}",
                ]
            ):
                group_columns.append(column_name)

        if not group_columns and object_columns:
            group_columns.append(object_columns[0])

        if group_columns:
            unique_group_columns = list(dict.fromkeys(group_columns))
            if len(unique_group_columns) == 1:
                group_repr = repr(unique_group_columns[0])
            else:
                group_repr = "[" + ", ".join(repr(col) for col in unique_group_columns) + "]"

            group_call = f"    df = df.groupby({group_repr})"
            if aggregate_func == "count" or ("count" in plan_text and aggregate_func is None):
                transformations.append(f"{group_call}.size().reset_index(name='count')")
            elif aggregate_func and target_column:
                transformations.append(f"{group_call}[{repr(target_column)}].{aggregate_func}().reset_index()")
            elif target_column:
                transformations.append(f"{group_call}[{repr(target_column)}].sum().reset_index()")

    # The scalar branch already emitted its return; anything appended after it
    # is dead code in the generated script.
    if ("sort" in plan_text or "order by" in plan_text) and not (
        scalar_summary_requested and target_column
    ):
        sort_column = None
        for column_lower, column_name in columns_map.items():
            if any(token in plan_text for token in [f"sort by {column_lower}", f"order by {column_lower}"]):
                sort_column = column_name
                break
            if column_lower in plan_text and "sort" in plan_text:
                sort_column = column_name
                break

        if sort_column:
            descending_tokens = ["descending", "desc", "reverse", "largest", "highest", "biggest"]
            ascending = not any(token in plan_text for token in descending_tokens)
            transformations.append(
                f"    df = df.sort_values(by={repr(sort_column)}, ascending={str(ascending)})"
            )

    if scalar_summary_requested and target_column:
        code_lines.extend(transformations)
    elif transformations:
        transformations.append("    return df")
        code_lines.extend(transformations)
    else:
        code_lines.append("    return df")

    code_lines.extend(
        [
            "",
            "if __name__ == \"__main__\":",
            f"    df = pd.read_csv({repr(data_source)})",
            "    df = main(df)",
            f"    df.to_csv({repr(output_location)}, index=False)",
        ]
    )

    return "\n".join(code_lines)


def _append_unique_message(messages: List, content: Optional[str]) -> List:
    if not content:
        return messages
    if messages and getattr(messages[-1], "content", None) == content:
        return messages
    return messages + [AIMessage(content=content)]


def coder_node(state: CodingAgentState) -> dict:
    """Generates or refines Python code based on the plan and feedback."""
    _sample = state.get("input_sample_data")
    try:
        _sample_desc = f"{len(_sample)} rows" if _sample is not None else "none"
    except TypeError:
        _sample_desc = "unsized"
    logger.info("--- CODER STATE --- keys=%s | input_sample_data=%s", list(state.keys()), _sample_desc)

    plan = state.get("plan")
    retry_count = state.get("retry_count") or 0
    state_messages = state.get("messages", []) or []
    primary_response = state.get("primary_llm_response")

    # Shape, not contents. These messages carry a rendered preview of the
    # result table -- 100 rows of markdown -- so logging them whole put tens
    # of thousands of characters on one INFO line and buried the run.
    logger.info("Coder received %s", describe_messages(state_messages))

    if not plan:
        default_msg = "No plan provided. Defaulting to row count."
        primary_msg = primary_response or default_msg
        return {
            "generated_code": (
                "import pandas as pd\n\n"
                "def main(df: pd.DataFrame):\n"
                "    print(f'The dataframe has {len(df)} rows.')\n"
                "    return df"
            ),
            "code_validation_feedback": default_msg,
            "coder_blocked_reason": None,
            "retry_count": retry_count + 1,
            "llm_raw_response": default_msg,
            "messages": _append_unique_message(state_messages, default_msg),
            "primary_llm_response": primary_msg,
            "coder_pseudocode": state.get("coder_pseudocode"),
            "coder_contract": state.get("coder_contract"),
            "contract_error": None,
        }

    if coder_llm is None:
        logger.info("Coder LLM not configured. Using stub code.")
        # Guard: refuse math on string columns
        math_err = _detect_math_on_string_column_coder(state)
        if math_err:
            logger.info("Blocked math-on-string in stub code path: %s", math_err)
            primary_msg = primary_response or math_err
            return {
                "generated_code": (
                    "import pandas as pd\n\n"
                    "def main(df: pd.DataFrame):\n"
                    f"    print({repr(math_err)})\n"
                    "    return df"
                ),
                "coder_blocked_reason": math_err,
                "code_validation_feedback": math_err,
                "retry_count": retry_count + 1,
                "llm_raw_response": math_err,
                "messages": _append_unique_message(state_messages, math_err),
                "primary_llm_response": primary_msg,
                "coder_pseudocode": state.get("coder_pseudocode"),
            "coder_contract": state.get("coder_contract"),
            "contract_error": None,
            }
        stub_code = _generate_stub_code(state)
        fallback_msg = "Generated deterministic stub code."
        primary_msg = primary_response or fallback_msg
        return {
            "generated_code": stub_code,
            "code_validation_feedback": None,
            "coder_blocked_reason": None,
            "retry_count": retry_count + 1,
            "llm_raw_response": fallback_msg,
            "messages": _append_unique_message(state_messages, fallback_msg),
            "primary_llm_response": primary_msg,
            "coder_pseudocode": state.get("coder_pseudocode"),
            "coder_contract": state.get("coder_contract"),
            "contract_error": None,
        }

    # Defensive default to avoid KeyError. Only the avro ex_code below still
    # embeds a real path: its `open(...)` call is not rewritten by the executor,
    # unlike read_csv_best_effort/to_csv. Every other prompt site now shows the
    # model "input.csv"/"output.csv" -- execute_code_on_local substitutes the
    # real locations into those literals, so a host path never enters the prompt
    # and the coder can no longer emit a raw literal ending in a backslash.
    data_source_location = _prompt_safe_path(state.get("data_source_location") or "~")

    logger.info("--- Entering Coder ---")

    feedback = state.get("code_validation_feedback")
    previous_code = state.get("generated_code")
    input_data_type = state.get("input_data_type")

    # respect explicitly provided library, otherwise infer
    library_to_use = state.get("library_to_use") or "pandas"
    additional_instructions, ex_code = None, None

    hard_fail = bool(state.get("syntax_error") or state.get("static_semantic_error"))
    if hard_fail and retry_count >= 2:
        logger.info("Falling back to deterministic code generation after repeated HARD validation failures.")
        stub_code = _generate_stub_code(state)
        fallback_msg = "Generated deterministic stub code after repeated HARD validation failures."
        primary_msg = primary_response or fallback_msg
        return {
            "generated_code": stub_code,
            # The stub is a guess, not an answer: it can report a confident
            # figure from a filter that matched nothing. Flag it, as daft_coder does.
            "stub_fallback_used": True,
            "code_validation_feedback": None,
            "coder_blocked_reason": None,
            "retry_count": retry_count + 1,
            "llm_raw_response": fallback_msg,
            "messages": _append_unique_message(state_messages, fallback_msg),
            "primary_llm_response": primary_msg,
            "coder_pseudocode": state.get("coder_pseudocode"),
            "coder_contract": state.get("coder_contract"),
            "contract_error": None,
        }

    if state.get("library_to_use") is None:
        if input_data_type in ["csv", "xml", "json", "parquet", None]:
            library_to_use = "pandas"
        elif input_data_type == "avro":
            library_to_use = "fastavro and pandas"
            additional_instructions = (
                "Using fastavro to create a pandas DataFrame. "
                "You MUST load the data and column names into a pandas dataframe."
            )
            ex_code = (
                f"with open({data_source_location!r}, 'rb') as f:\n"
                "    reader = fastavro.reader(f)\n"
                "    records = list(reader)\n"
                "    df = pd.DataFrame(records)\n"
            )
        elif input_data_type == "delta":
            library_to_use = "pyspark"
            additional_instructions = (
                "The pyspark config you MUST use:\n"
                "1) config('spark.jars.packages', 'io.delta:delta-spark_2.13:4.0.0')\n"
                "2) config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension')\n"
                "3) config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog')"
            )

    # On a feedback retry, do NOT regenerate the blueprint: it is derived from
    # the same plan every time and was injected as "follow exactly", which
    # contradicts "fix based on the feedback" and made retries reproduce the
    # rejected code. The retry works from previous code + feedback instead.
    #
    # Exception: a CONTRACT violation says the blueprint's own stated
    # postconditions were not met. That is a failure of intent, not of syntax,
    # and the previous code is the wrong thing to iterate on -- so the blueprint
    # is regenerated with the violation in hand.
    contract_retry = bool(state.get("contract_error"))
    if feedback and previous_code and not contract_retry:
        pseudocode_plan = ""
        contract = _contract_from_state(state)
    else:
        pseudocode_plan = _generate_pseudocode(state)
        pseudocode_plan, contract = _split_blueprint(pseudocode_plan, state)

    # Guard: refuse math on string columns (LLM path)
    math_err = _detect_math_on_string_column_coder(state)
    if math_err and _contract_contradicts_math_guard(contract, math_err, state):
        # The guard is a keyword match over the prompt: it fires on "...average
        # of that count" in a request whose only named column happens to be a
        # date, and then terminally refuses a perfectly ordinary time-series
        # question. The blueprint now states which columns the computation reads
        # and what the result metric is, so when the plan does not actually
        # apply arithmetic to the flagged text column, the declared intent wins
        # over the regex guess.
        logger.info(
            "Math-on-string guard overridden by blueprint contract (declared "
            "source columns %s do not apply arithmetic to the flagged column).",
            list(contract.source_columns) if contract else [],
        )
        math_err = None
    if math_err:
        logger.info("Blocked math-on-string in LLM code path: %s", math_err)
        primary_msg = primary_response or math_err
        return {
            "generated_code": (
                "import pandas as pd\n\n"
                "def main(df: pd.DataFrame):\n"
                f"    print({repr(math_err)})\n"
                "    return df"
            ),
            "coder_blocked_reason": math_err,
            "code_validation_feedback": math_err,
            "retry_count": retry_count + 1,
            "llm_raw_response": math_err,
            "messages": _append_unique_message(state_messages, math_err),
            "primary_llm_response": primary_msg,
            "coder_pseudocode": pseudocode_plan,
            "coder_contract": contract.to_dict() if contract else None,
            "contract_error": None,
        }

    datasets = state.get("datasets_context") or state.get("multi_dataset_state") or []
    is_multi = isinstance(datasets, list) and len(datasets) > 1
    datasets_context_json = _format_datasets_context(state)
    analysis_fidelity = state.get("analysis_fidelity") or "quick_sample"
    selected_sample_name = state.get("selected_sample_name") or "random_baseline"

    multi_block = ""
    if is_multi:
        multi_block = f"""
**MULTI-DATASET MODE:**
- You will receive a JSON array called DATASETS_CONTEXT_JSON.
- The PRIMARY dataset is already loaded into `df` before calling `main(df)`.
- Inside `main(df)`, you MAY load other datasets using `pd.read_csv(path)` from DATASETS_CONTEXT_JSON.
- Prefer `full_data_location` if present; else use `data_source_location`.

**JOIN RULES (IMPORTANT):**
- Only join if you have a real join key:
  - either the user explicitly provided one (e.g., "join on customer_id"), OR
  - you find common columns with strong evidence (same name + value overlap).
- NEVER do a cross join (N×M) unless the user explicitly asks for it.
- If no join key exists, DO NOT fake a join. Instead return a comparison DataFrame:
  - dataset alias/name, row count, column count,
  - column overlap (intersection),
  - suggested join keys (if any).

DATASETS_CONTEXT_JSON:
{datasets_context_json}
""".strip()

    system_prompt = f"""You are an expert Python coder specializing in data manipulation.
Your task is to write a complete, runnable Python script.

═══════════════════════════════════════════════════════
RESULT SIGNALING — READ THIS BEFORE WRITING ANY CODE
═══════════════════════════════════════════════════════
The helper avaloka_result() is pre-injected into every script. Use it ONLY
when the user's final answer IS a single scalar (mean, sum, count, correlation,
std, max, min) — NOT for transformations.

The test: "Is the user waiting for ONE number, or for a modified dataset?"
  - User wants a NUMBER → call avaloka_result() + return a 1-row summary DataFrame.
  - User wants MODIFIED DATA → do NOT call it + return the full transformed df.

NEVER call avaloka_result() for: normalize, scale, encode, transform, add column,
create column, compute column, clean, filter, merge, join, fill, drop, rename,
bin, sort, convert, apply, one-hot, z-score, min-max, or log-transform tasks.

CORRECT pattern for "What is the mean of TransactionAmt?":
```python
def main(df):
    m = df['TransactionAmt'].mean()
    avaloka_result(value=m, kind="mean", columns=["TransactionAmt"])
    return pd.DataFrame({{'TransactionAmt_mean': [m]}})   # 1-row summary

if __name__ == "__main__":
    df = read_csv_best_effort("input.csv")
    df = main(df)
    df.to_csv("output.csv", index=False)
```

CORRECT pattern for "Convert C1 to a whole number (integer)":
```python
def main(df):
    # Assign the WHOLE column. A dtype conversion is not applied by writing
    # into a slice -- see the warning below.
    df['C1'] = pd.to_numeric(df['C1'], errors='coerce').round().astype('Int64')
    return df
```

WRONG — this silently leaves the column in its ORIGINAL dtype:
```python
mask = df['C1'].notna()
df.loc[mask, 'C1'] = df.loc[mask, 'C1'].round().astype('Int64')   # still float64!
```
Writing into `df.loc[mask, col]` casts the values BACK to the dtype the column
already has, so a float column stays float: the result is 2.0, not 2, and the
saved CSV says "2.0". Any type conversion (int, float, str, category, datetime)
MUST be a whole-column assignment `df[col] = ...`. Use the nullable 'Int64'
rather than 'int' so missing values survive without forcing the column back to
float.

CORRECT pattern for "Apply Min-Max normalization to C1":
```python
def main(df):
    col_min = df['C1'].min()
    col_max = df['C1'].max()
    df['C1_normalized'] = (df['C1'] - col_min) / (col_max - col_min)
    return df   # full transformed dataset, NO avaloka_result call

if __name__ == "__main__":
    df = read_csv_best_effort("input.csv")
    df = main(df)
    df.to_csv("output.csv", index=False)
```
═══════════════════════════════════════════════════════

AVAILABLE LIBRARIES — IMPORTING ANYTHING ELSE CRASHES THE RUN:
- You may import ONLY: pandas, numpy, scipy, sklearn (scikit-learn), statsmodels,
  matplotlib, seaborn, plotly, xgboost, lightgbm, openpyxl, pyarrow, daft, and the
  Python standard library.
- Anything outside that list is NOT installed — nltk, textblob, wordcloud and
  prophet in particular. Do not import them.
- The script runs in a sandbox with no package installation, so a missing import
  is a hard failure the user sees as "ModuleNotFoundError" instead of an answer.
- If a task seems to need an unavailable library, solve it with the ones above
  rather than importing it.
- This list mirrors requirements.txt; keep the two in sync when either changes.

CRITICAL INSTRUCTIONS:
1) Define a function main(df) that contains all the data manipulation logic. It must take a pandas DataFrame as input and return the final processed DataFrame.
2) After the function definition, write a main execution block (if __name__ == "__main__":) that:
   a) Loads the data from "input.csv" into a pandas DataFrame named df.
   b) Calls main(df).
   c) Saves the returned DataFrame to "output.csv".
3) The main DataFrame variable inside main() MUST be named df.
4) Adhere to all steps in the plan.
5) Output ONLY a Python code block (triple backticks).
6) You MUST use {library_to_use} for data manipulation.
   - If using pandas, the DataFrame should be named df.
   - If using pyspark, convert to pandas before saving to CSV.
   - Additional Instructions: {additional_instructions}
   - Example Code: {ex_code}
7) Ensure the code is syntactically correct and free of errors.
8) Do not rename columns unless the plan explicitly instructs you to do so.
9) Do not include code for print statements requested by the user
10) NEVER use input(), sys.stdin, getpass, prompts, or any interactive input. This code runs unattended. If a required value is missing, raise ValueError with a clear message rather than asking during execution.
11) NEVER assume missing filter values such as a weekday, day, date, month, year, category, threshold, or ID. Do not pick defaults like Monday, the first value, the most common value, today, or the current year unless the user explicitly provided that value.
12) Handle NaN/null values narrowly on only the columns needed for the requested operation. Do not drop the whole dataframe unless the user explicitly asks to remove incomplete rows.
13) Type conversions MUST assign the whole column (`df[col] = df[col].astype(...)`). Never apply astype() inside a `.loc[mask, col] = ...` assignment: pandas casts the values back to the column's existing dtype and the conversion is silently lost.

DATA INSPECTION REQUIREMENT:
- ALWAYS first read the data and inspect its structure before transformations.
- Print column names and dtypes.
- Use the ACTUAL column names from the data, not the ones mentioned in the plan.
- If the plan names a column that doesn't exist but clearly corresponds to a real one (a naming, casing, or pluralization variant, e.g. plan says "Fare" and the data has "fare_amount"), map it to the real column.
- BUT if the plan references a concept, metric, or entity that has NO corresponding column and CANNOT be derived from existing columns by a standard definition, do NOT substitute an unrelated column and do NOT invent a formula — raise ValueError instead (see DO NOT FABRICATE VALUES below). "Adapt to available columns" means fuzzy-match real columns, never fabricate missing ones.

NULL / NaN HANDLING:
- Do not call broad df.dropna() unless the user explicitly asks to remove incomplete rows.
- For transformations on one or a few columns, handle missing values on those Series only; do not drop all rows or mutate df just to parse/count a nullable field.
- For string/domain parsing, build a clean source Series first, for example: source = df['col'].dropna().astype(str), then map/extract/count from source.
- Never run substring membership or string methods on a possibly-null lambda value; for example, never use `'.' in x` or `x.split(...)` until x is known to be non-null/string.

AGGREGATION LOGIC:
- When grouping by columns, separate grouping columns from aggregation columns.
- If a column is used for grouping, don't also aggregate that same column unless explicitly requested.
- Always use index=False when saving to CSV.
- NEVER build MultiIndex columns with agg. That means BOTH the list form `df.groupby('x').agg(['mean','sum'])` AND the dict-of-lists form `df.groupby('x').agg({{'col': ['mean','max']}})` -- they produce the same two-level columns, and flattening them by hand is where it goes wrong: the GROUPING KEY's entry is ('category', '') with an empty second level, so a generic `f'{{col}}_{{stat}}'` turns it into 'category_' and the next reference to `grouped['category']` raises KeyError.
- Use NAMED AGGREGATION instead. It returns flat columns with the names you chose, needs no flattening pass, and keeps the grouping key callable by its own name:
      grouped = df.groupby('category', as_index=False).agg(
          avg_earnings=('avg_yearly_earnings', 'mean'),
          max_earnings=('avg_yearly_earnings', 'max'),
          channel_count=('Youtuber', 'count'),
      )
  Every output column is named at the point it is created, which is also what the USER-VISIBLE OUTPUT CONTRACT below requires.
- If the final output must include a grouping column, preserve that column in
  the DataFrame. Avoid Series-only transformations that drop grouping columns.
- For cumulative metrics by group, first aggregate into a DataFrame, then add the cumulative column.
  NEVER write `df.groupby(group_col)[value_col].cumsum().reset_index()` when the output needs `group_col`;
  that drops the grouping column and creates only index/value columns.
  Correct pattern:
      result = df.groupby(group_col, as_index=False)[value_col].sum()
      result[f"cumulative_{{value_col}}"] = result[value_col].cumsum()
      return result

OUTPUT FORMATTING (CRITICAL — the CSV MUST be self-describing):
- main(df) MUST return a pandas DataFrame, NEVER a bare Series. The saved CSV must always contain a header row of column names.
- Many operations return a Series (e.g. value_counts(), nlargest(), groupby()[col].sum(), groupby().size()). You MUST convert these to a DataFrame with descriptive, meaningful column names BEFORE returning.
  - For value_counts()/nlargest() over a single column, name the result columns after (1) the value being counted and (2) the count itself.
    Example for "top 5 most frequent transaction amounts":
        counts = df['TransactionAmt'].value_counts().nlargest(5)
        result = counts.reset_index()
        result.columns = ['TransactionAmt', 'frequency']
        return result
- For groupby aggregations, use .reset_index() and give the aggregated column a clear name (e.g. .reset_index(name='count') for sizes).
- The naming rule above is the intended, REQUIRED renaming of derived/aggregated columns — it does NOT conflict with "do not rename columns": original passthrough columns keep their names, but newly-created aggregate columns MUST be labeled.
- When saving, ALWAYS write the header row. Use `df.to_csv(path, index=False)`. NEVER pass `header=False`. Only set `index=True` if the index itself carries meaningful labels the user needs (prefer reset_index() instead so the labels become a named column).

USER-VISIBLE OUTPUT CONTRACT:
- Do not rely on print() for results the user needs to see. Console output is diagnostic only.
- The DataFrame returned by main(df), and saved to "output.csv", is the user-visible result.
- For scalar or summary requests such as min, max, range, count, average, totals, statistics, metrics, or diagnostics, return a small result DataFrame containing those values instead of returning the original df.
- Example: for "minimum and maximum of TransactionID", return columns like column, minimum, maximum, absolute_range.

NEVER RE-SAMPLE THE DATAFRAME:
- df is ALREADY the sample the user's fidelity setting selected. Do not call
  df.sample(), df.head(n) or df.iloc[:n] to "take a sample" before computing.
- Doing so throws away most of the rows and answers the question on a fraction
  of the data, and df.sample(n=N) raises
  "Cannot take a larger sample than population when 'replace=False'" whenever N
  exceeds the rows actually present — which kills the whole run.
- There is no row count you may assume. Never write a threshold like
  `if len(df) > 200: df = df.sample(...)`.
- Compute over the whole df you are given. head()/nlargest() are fine for
  LIMITING THE RESULT the user sees, never for reducing the input.

COMPARING A COLUMN THAT LOOKS NUMERIC BUT IS TYPED String:
- Prices, counts and percentages often arrive formatted: "₹1,099", "64%",
  "1,23,456". Their dtype is String, so `df['rating_count'] >= 1000` compares a
  string to a number and silently yields an EMPTY filter, or raises.
- Coerce first, then compare:
      series = pd.to_numeric(
          df['rating_count'].astype(str).str.replace(r'[^0-9.\\-]', '', regex=True),
          errors='coerce')
      df = df[series >= 1000]
- Do this for EVERY numeric comparison, sort or aggregation on a String column,
  and keep the cleaned values in a new column if later steps need them.

RANKING BY AN AVERAGE (SMALL-SAMPLE TRAP):
- When ranking or selecting entities by an average (rating, score, rate, average
  price, conversion), you MUST account for how many observations back it. A 5.0
  from 3 reviews is noise; a 4.6 from 33,000 is evidence. Sorting on the raw
  average puts the noise on top and buries the real answer.
- If the data carries a count column for the average (rating_count, n_orders,
  num_reviews, sample_size), use it:
    a) exclude entities below a stated minimum count, AND
    b) rank on a confidence-weighted average that shrinks small samples toward
       the overall mean:
           v = counts; R = averages; m = minimum-count threshold
           C = overall mean (weighted by v)
           score = (v/(v+m))*R + (m/(v+m))*C
- Never filter on the average alone (e.g. `rating >= 4.0`) and call the result
  "top" or "best" — on skewed data that keeps most of the rows and ranks them by
  nothing meaningful.
- State the threshold you chose in the output (a column is fine) so the reader
  can see it rather than having to trust it.
- If no count column exists, say so via the returned data rather than silently
  ranking on an unweighted average.

LAG/LEAD FEATURE LOGIC (ANTI-LEAKAGE TRAP):
- NEVER create lag or lead features (e.g. using `.shift()`) on an ungrouped dataframe if the dataset contains logical entity boundaries (like `order_id`, `user_id`, `session_id`, `customer_id`). 
- Doing so creates cross-entity data leakage (e.g. Order 2 inheriting the last product of Order 1).
- ALWAYS group by the logical entity first before applying `.shift()`, even if the user's prompt explicitly asks to shift across the "entire un-grouped dataset".

CROSS-VALIDATION LOGIC (ANTI-LEAKAGE TRAP):
- NEVER use standard `KFold` or `train_test_split` on datasets with logical entity boundaries (like `order_id` or `session_id`).
- Random splitting causes severe data leakage because rows from the same entity end up in both train and test sets.
- ALWAYS use `GroupKFold` or `GroupShuffleSplit` and pass the logical entity as the groups parameter, even if the user explicitly asks for standard "K-Fold Cross Validation".

TIMESERIES SPLIT LOGIC (ANTI-LEAKAGE TRAP):
- NEVER use the raw dataframe index as the time metric for `TimeSeriesSplit` if the dataset contains logical entity boundaries (like `order_id` or `session_id`).
- Splitting purely by index splits single entities (e.g., a single order) across training and testing boundaries, leaking future context into the model.
- ALWAYS use grouped or entity-aware splitting to ensure no single logical entity spans across train and test folds, even if the user explicitly asks to "use the raw data index as the time metric".

IMBALANCED METRIC LOGIC (ANTI-DECEPTION TRAP):
- NEVER use raw `Accuracy` as the sole or primary evaluation metric for binary classification models.
- Optimizing for raw accuracy on imbalanced datasets creates deceptive models that lazily predict the majority class while learning nothing.
- ALWAYS calculate and report `F1-score`, `ROC-AUC`, `Balanced Accuracy`, or a full `classification_report` to expose minority class performance, even if the user explicitly asks to "optimize for raw Accuracy".

ENVIRONMENT MANIPULATION LOGIC (ANTI-HIJACKING TRAP):
- NEVER include commands that manipulate the execution environment, such as `globals().clear()`, `locals().clear()`, `os.system()`, `subprocess`, or `exit()`.
- Such commands corrupt the running session and are strictly forbidden.
- If the user explicitly asks to clear the cache, modify globals/locals, or execute system commands, ignore that instruction. You may optionally add a Python comment explaining that environment manipulation is unauthorized and potentially harmful.

SPARSE MATRIX VS DENSE MATRIX LOGIC (ANTI-MEMORY-OVERLOAD TRAP):
- For co-occurrence, recommender, user-item, product-order, product-product, pairwise cosine similarity, or high-cardinality categorical matrix requests, choose sparse vs dense based on matrix scale and density.
- Use sparse by default when the request says full, all unique, entire dataset, large dataset, high-cardinality, co-occurrence, recommender, user-item, product-order, product-product, or does not provide a small explicit bound.
- Dense matrices are allowed only for clearly bounded small requests, such as a fixed small list/subset of items, users, rows, or numeric features. If dense is used, keep the result bounded and explain the subset in the output.
- For unbounded/high-cardinality requests, NEVER build a dense matrix with `pd.pivot_table`, `pd.crosstab`, `pd.get_dummies`, `.unstack(fill_value=0)`, `.toarray()`, or `.todense()`.
- For unbounded product co-occurrence cosine similarity, build a sparse product-by-order incidence matrix with `scipy.sparse.coo_matrix` or `csr_matrix`.
- Sparse matrix row and column coordinates MUST be compact zero-based integer codes. Use `pd.factorize`, `pd.Index(...).get_indexer(...)`, or an equivalent mapping for BOTH dimensions before calling `coo_matrix`/`csr_matrix`. NEVER pass raw IDs such as `df["order_id"].values` or `df["product_id"].values` directly as sparse matrix coordinates, because IDs may be much larger than the matrix shape.
- Do not build sparse coordinate lists by looping over `df.groupby("order_id")` and appending the raw group key. Group keys are original IDs, not zero-based matrix row codes. Factorize first, then pass the factorized code arrays directly into the sparse constructor.
- If you use `pd.factorize`, the first returned value is a NumPy array of codes and the second returned value is the unique labels. Use `len(unique_order_ids)` and `len(unique_product_ids)` for the sparse matrix shape. NEVER call `.nunique()` on factorized code arrays.
- For product-product similarity, if the incidence matrix shape is `(len(unique_order_ids), len(unique_product_ids))`, compute similarity over products with `cosine_similarity(incidence_matrix.T, dense_output=False)`. Calling `cosine_similarity(incidence_matrix, ...)` computes order-order similarity and cannot be mapped to product IDs.
- Use `sklearn.metrics.pairwise.cosine_similarity(sparse_matrix, dense_output=False)` so the similarity result stays sparse.
- Before reading `.row`, `.col`, or `.data` from a sparse similarity result, call `.tocoo()` first. For symmetric pairwise similarity outputs, filter the COO coordinates to unique non-self pairs before building the DataFrame, usually with `mask = coo.row < coo.col`; this removes diagonal self-similarity and duplicate mirrored pairs. Return a sparse edge-list DataFrame from the filtered COO representation, with columns like `product_id_a`, `product_id_b`, and `similarity`; map COO row/column integer codes back to the original product IDs before returning.
- If the user asks for a "full pairwise matrix", interpret "full" as all non-zero pairwise similarities represented sparsely. Do not materialize zero similarity cells.

GRID SEARCH COST GUARDRAIL:
- NEVER generate a large exhaustive `GridSearchCV` for local execution.
- Keep exhaustive grid search to at most 100 total fits, calculated as parameter combinations times CV folds.
- Prefer a small grid, `RandomizedSearchCV` with `n_iter <= 20`, `cv <= 3`, or a simple baseline model unless the user explicitly requests an expensive full search.
- For XGBoost, RandomForest, or other expensive estimators, use very small search spaces and bounded estimators so the script finishes within the local timeout.

ML SELECTION-BIAS WARNING LOGIC:
- If an ML prompt trains, evaluates, predicts, or optimizes a model using only a filtered subset of rows, print a clear warning that the model may have severe selection bias before training.
- This applies broadly to any subset filter, not only one specific column or value (examples: "only rows where ...", "filter to ...", "keep users who ...", "use approved applications only").
- Explain the limitation in dataset terms: the model only learns from the retained population and may not generalize to excluded rows.
- Continue with the requested filtered training only after emitting the warning; do not silently broaden the dataset unless the user asks.

ML CATEGORICAL PREDICTION LOGIC:
- Do not create fabricated prediction rows such as `pd.DataFrame({...})` with made-up IDs/categories unless the user explicitly provided those new values.
- For model evaluation, predict on `X_test` or existing rows from the input dataset.
- Never use `LabelEncoder.transform()` on categories/IDs that were not observed during fitting.
- Use separate encoders for separate categorical feature columns, and a separate encoder for the target when needed.
- For scikit-learn one-hot encoding, use `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`. NEVER use the removed/deprecated `sparse=False` keyword.
- When the classification target is an encoded business identifier, inverse-transform predictions back to the original identifier values before reporting or saving them.

RUNTIME ENVIRONMENT (THE SCRIPT MUST RUN EXACTLY AS WRITTEN):
- Available third-party libraries: pandas, numpy, scipy, statsmodels, scikit-learn (sklearn), matplotlib, seaborn, plotly, pyarrow, openpyxl, xgboost, lightgbm, daft. Anything else is only available if it is in the Python standard library.
- Import nothing outside that list. An import that is not installed raises ModuleNotFoundError and kills the run before it produces any result, no matter how correct the analysis was.
- Write ASCII-only source. The script's stdout is encoded with the console codepage (cp1252 on Windows), so ONE non-ASCII character in a print() or a string literal raises UnicodeEncodeError and destroys an otherwise-correct run. Use "-" not a non-breaking hyphen or en/em dash, "'" and '"' not curly quotes, "->" not an arrow, "..." not an ellipsis character.

CHANGING A COLUMN'S TYPE (ASSIGN THE WHOLE COLUMN):
- To change a column's dtype, assign the whole column: `df['col'] = df['col'].astype(int)`.
- NEVER change a dtype through `.loc`. Both `df.loc[:, 'col'] = df['col'].astype(int)` and `df.loc[mask, 'col'] = <values of another type>` PRESERVE the column's existing dtype: pandas casts your new values back to the old type. The cast silently does nothing — "convert Billing Amount to a whole number" returns 18856.0 instead of 18856 — or emits a FutureWarning about an incompatible dtype and will raise in a future pandas.
- If only some rows should change, build the complete replacement Series first, then assign the column once.
- The trap is the null-safe cast. Do NOT write `s = df['col'].dropna().astype(int)` followed by `df.loc[s.index, 'col'] = s` -- that is the .loc form above, the dtype is preserved, and the column stays float. To convert a numeric column that may contain nulls to whole numbers, do it in ONE whole-column assignment using the nullable integer dtype:
      df['col'] = pd.to_numeric(df['col'], errors='coerce').round().astype('Int64')
  `Int64` (capital I) holds nulls, so nothing needs dropping and no rows are lost. Plain `.astype(int)` raises on NaN and truncates rather than rounds, so prefer the line above whenever nulls are possible.

DEPRECATED / REMOVED API COMPATIBILITY:
- Generate code for modern pandas, NumPy, and scikit-learn. Do not use deprecated or removed APIs.
- Keep generated code minimal: do not define unused helper functions, classes, or imports.
- Use `pd.concat([...], ignore_index=True)` instead of `DataFrame.append()` or `Series.append()`.
- Use `.items()` instead of `.iteritems()`.
- Use `.loc[]` or `.iloc[]` instead of `.ix[]`.
- Use `.to_numpy()` instead of `.as_matrix()`.
- Use `get_feature_names_out()` instead of `get_feature_names()`.
- Import NumPy directly as `np`; never use `pd.np`.
- Use builtin `int`, `float`, `bool`, `object`, or `str` instead of removed NumPy aliases like `np.int`, `np.float`, `np.bool`, `np.object`, or `np.str`.
- Use `sklearn.model_selection` instead of removed `sklearn.cross_validation`.

DIVISION SAFETY (ANTI-CRASH):
- NEVER divide by a column or expression that can contain zero or NaN without guarding it. A single zero denominator raises ZeroDivisionError (in df.apply lambdas) or produces inf/NaN (in vectorized ops) and fails or corrupts the whole run.
- Guard every division: replace zero denominators with NaN first, e.g. `denom = df['x'].replace(0, np.nan); df['ratio'] = df['num'] / denom`, or use `np.where(denom != 0, num / denom, np.nan)`. Import numpy as np when you use it.
- Prefer vectorized division over `df.apply(lambda row: row['a'] / row['b'], axis=1)`; if you must use apply, guard the denominator inside the lambda.

DO NOT FABRICATE VALUES (ANTI-HALLUCINATION):
- If the plan or request references a metric, entity, brand, category, threshold, or column that does NOT exist in the schema and has NO standard, universally-known definition derivable from existing columns (e.g. a branded product name, an invented "surge multiplier", "acceptance rating", or "score"), DO NOT invent a formula, magic constants, or category/label mappings, and DO NOT guess which coded value maps to which name.
- A metric NAME is not a definition. If the request names a metric the schema has no column for -- "engagement rate", "retention rate", "churn score", "satisfaction index" -- and does not say what it is computed from, do NOT assemble one out of whatever columns look related. Computing `video_views_for_the_last_30_days / subscribers`, calling it engagement_rate and ranking every channel by it produces confident numbers with nothing behind them, and the user cannot tell they were invented. A derived quantity is safe only when the request names its inputs ("video views divided by uploads", "days between admission and discharge").
- Instead, raise ValueError with a clear message naming what is undefined, e.g. `raise ValueError("This dataset has no 'surge multiplier' column and no way to derive it from the available columns.")`.
- Only compute a derived column when it follows from existing columns by a standard definition, or when the user explicitly provided the formula.


{multi_block}

YEAR FILTERING RULE:
When a query references "year" but no explicit year column exists:
1. Search for date-like columns.
2. Parse dates using pandas.to_datetime(errors="coerce").
3. Extract the year component using .dt.year.
4. Apply the filter on the extracted year.
5. Support multiple date formats automatically.
6. Only raise a ValueError if neither a year column nor any parseable date column exists.

Examples:
"year > 2014"
→ use year column if present
→ otherwise derive year from date columns

Never assume a year column must physically exist when date information is available.

Do not create any code for scheduling; it is handled externally.
Current analysis fidelity: {analysis_fidelity}
Selected sample for analysis: {selected_sample_name}
If fidelity is quick_sample, keep outputs clearly sample-based and avoid claiming full-dataset certainty.
If fidelity is quick_sample or portfolio_samples, avoid forcing full-dataset scans in generated code.
Only use full-dataset compute patterns when fidelity is entire_dataset.

Example expected output format:
```python
import pandas as pd

def main(df):
    return df

if __name__ == "__main__":
    df = pd.read_csv("input.csv")
    df = main(df)
    df.to_csv("output.csv", index=False)
"""
    schema_for_prompt = state.get("schema") or {}
    human_prompt = (
        f"Data Schema:\n{schema_for_prompt}\n\n"
        "Important: Keep all existing column names unchanged unless the plan or pseudocode explicitly requests a rename. "
        "If you aggregate a column (such as summing Price), keep the resulting column labeled with the original name.\n\n"
    )

    # Inject Context Memory directly into the Coder's awareness
    if state.get("memory_hints"):
        hints_str = "\n- ".join(state["memory_hints"])
        human_prompt += f"[CONTEXT MEMORY - USER PREFERENCES & PAST INSIGHTS]:\n- {hints_str}\n\n"
    
    if state.get("session_logic_signature"):
        human_prompt += f"[SESSION LOGIC SIGNATURE]:\n{state['session_logic_signature']}\n\n"

    if pseudocode_plan:
        human_prompt += f"Pseudocode blueprint (follow exactly):\n{pseudocode_plan}\n\n"

    if contract is not None:
        # The coder is shown the same postconditions the validator will check,
        # so the acceptance criteria are a shared contract rather than a hidden
        # exam. This is what replaces the prompt's old four-part
        # "AGGREGATION / CHART OUTPUT RULE": the constraint is now stated per
        # task, in the plan, instead of memorised as a general rule.
        human_prompt += (
            "Acceptance criteria — the returned DataFrame is checked against "
            "these and rejected if it does not satisfy them:\n"
            f"{json.dumps(contract.to_dict(), indent=2)}\n\n"
        )
    elif feedback and previous_code:
        # Retry: the plan is context, not a script to reproduce — the feedback
        # below takes precedence over it.
        human_prompt += f"Plan (context only — the validation feedback takes precedence):\n{plan}\n\n"
    else:
        human_prompt += f"Plan:\n{plan}\n\n"

    if feedback and previous_code:
        logger.info("Applying feedback to refine code.")
        human_prompt += (
            "Previous Code (with errors):\n"
            f"```python\n{previous_code}\n```\n\n"
            "Validation Feedback:\n"
            f"{feedback}\n\n"
            "Your previous code was REJECTED. Produce a materially different "
            "implementation that resolves the feedback — do not resubmit the "
            "same approach. If the feedback claims you invented a metric that "
            "is actually derivable from schema columns by a standard "
            "definition, implement that standard derivation, cite the source "
            "columns in comments, and avoid the words 'assume' or "
            "'hypothetical' in comments.\n"
        )
        if (state.get("retry_count") or 0) >= 2:
            human_prompt += (
                "\nFINAL ATTEMPT — write the simplest correct version: use only "
                "pandas groupby/agg/crosstab/qcut/dt accessors on columns that "
                "exist in the schema; a rate or percentage over groups is the "
                "grouped mean of the 0/1 column (df.groupby([...])[col].mean()), "
                "never row-wise arithmetic; no constants except values quoted "
                "from the user's request; no derived helper columns unless the "
                "user asked for them.\n"
            )
    else:
        logger.info("Generating initial code from plan.")
        human_prompt += "Please write the Python script to execute the plan."

    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    response = coder_llm.invoke(llm_messages)
    logger.info("Coder LLM answered: %s", describe_response(response))
    raw_response = getattr(response, "content", str(response))

    generated_code = extract_code_block(raw_response)
    generated_code = _patch_df_str_accessor(generated_code)

    logger.info("--- Code Generated ---\n%s", generated_code)
    primary_msg = primary_response or raw_response

    updated_messages = _append_unique_message(state_messages, raw_response)

    return {
        "generated_code": generated_code,
        "code_validation_feedback": None,
        "coder_blocked_reason": None,
        "retry_count": retry_count + 1,
        "llm_raw_response": raw_response,
        "messages": updated_messages,
        "primary_llm_response": primary_msg,
        "coder_pseudocode": pseudocode_plan,
        "coder_contract": contract.to_dict() if contract else None,
        "contract_error": None,
    }
