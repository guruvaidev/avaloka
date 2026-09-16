# validator.py
import ast
import json
import logging
import re
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.inference import build_chat_model
from app.agents.state import CodingAgentState
import os
STRICT_LOGIC_ENV = "AVALOKA_STRICT_LOGIC_VALIDATION"
logger = logging.getLogger(__name__)
from app.core.model_config import resolve as resolve_model
from app.core.model_fallback import attach_fallback
from app.core.log_utils import describe_response

_validator_api_key = os.environ.get("GROQ_API_KEY_CODING_AGENT")
validator_llm = build_chat_model(
    role="coding",
    agent="VALIDATOR",
    tier="small",
    temperature=0.0,
    groq_model=resolve_model("validator"),
    groq_api_key=_validator_api_key,
)


class SafetyGuardrailError(Exception):
    pass


class InteractiveInputVisitor(ast.NodeVisitor):
    """Detect code that would block unattended execution by waiting for input."""

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


_WEEKDAY_VALUE_TERMS = {
    "monday",
    "mon",
    "tuesday",
    "tue",
    "wednesday",
    "wed",
    "thursday",
    "thu",
    "friday",
    "fri",
    "saturday",
    "sat",
    "sunday",
    "sun",
}


def _prompt_has_explicit_weekday(prompt: str) -> bool:
    lowered = (prompt or "").lower()
    return any(re.search(rf"\b{re.escape(value)}\b", lowered) for value in _WEEKDAY_VALUE_TERMS)


def _find_assumed_weekday_errors(code: str, user_prompt: str) -> list[str]:
    lowered_prompt = (user_prompt or "").lower()
    # "per day" / "each day" / "group by day" is aggregation phrasing, not a
    # filter with a missing value; weekday literals (e.g. ordering lists) are
    # legitimate there.
    distributive_day = re.search(
        r"\b(?:per|each|every|group(?:ed)?\s+by)\s+(?:day|weekday|week day)\b",
        lowered_prompt,
    )
    asks_for_weekday_filter = (
        re.search(r"\b(filter|where|only|keep|include|return)\b", lowered_prompt)
        and re.search(r"\b(day|weekday|week day)\b", lowered_prompt)
        and not distributive_day
        and not _prompt_has_explicit_weekday(lowered_prompt)
    )
    if not asks_for_weekday_filter:
        return []

    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip().lower()
            if value in _WEEKDAY_VALUE_TERMS:
                errors.append(
                    f"SafetyError on line {node.lineno}: generated code assumed weekday '{node.value}' "
                    "without the user providing that value."
                )
    return errors


def validate_join_safety(df_a_metadata: dict, df_b_metadata: dict, join_type: str, max_allowed_rows: int = 1000000):
    if join_type != 'cross':
        return
    
    n = df_a_metadata.get('row_count')
    m = df_b_metadata.get('row_count')
    
    if n is None or m is None:
        raise SafetyGuardrailError(
            "A cross join was attempted but dataset row counts are unknown. "
            "To prevent an OOM (Out of Memory) crash, cross joins are strictly blocked. "
            "Please refine your logic to avoid cross joins and use a specific join key."
        )
        
    result_rows = n * m
    if result_rows > max_allowed_rows:
        raise SafetyGuardrailError(
            f"A cross join between {n} rows and {m} rows will yield {result_rows} rows, "
            f"exceeding the maximum allowed of {max_allowed_rows}. "
            "This will cause an OOM crash. Please refine your logic to avoid a cross join."
        )

def check_join_keys(df_a_schema: list, df_b_schema: list, provided_keys: list):
    for key in provided_keys:
        if key not in df_a_schema or key not in df_b_schema:
             raise SafetyGuardrailError(f"Join key '{key}' not found in either dataset schema.")


def _is_numeric_dtype(dtype: str) -> bool:
    d = str(dtype).lower()
    return any(tok in d for tok in ["int", "float", "double", "decimal", "number"])


def _is_string_like_dtype(dtype: Any) -> bool:
    d = str(dtype).lower()
    return any(tok in d for tok in ["object", "string", "str", "category"])


def _is_numeric_literal(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class StrAccessorOnDfVisitor(ast.NodeVisitor):
    """
    Flags df['col'].str.... when schema says col is numeric.
    This catches the exact crash you're seeing.
    """
    def __init__(self, schema: dict[str, Any]):
        self.schema = schema or {}
        self.errors: list[str] = []

    def _extract_str_key(self, slice_node):
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return slice_node.value
        if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
            val = slice_node.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                return val.value
        return None

    def visit_Attribute(self, node: ast.Attribute):
        # Match: (df['col']).str
        if node.attr == "str" and isinstance(node.value, ast.Subscript):
            sub = node.value
            if isinstance(sub.value, ast.Name) and sub.value.id == "df":
                col = self._extract_str_key(sub.slice)
                if col:
                    dtype = self.schema.get(col)
                    if _is_numeric_dtype(dtype):
                        self.errors.append(
                            f"SemanticError on line {node.lineno}: df[{col!r}] is numeric ({dtype}); "
                            f"do not use .str on it. Use pd.to_numeric(df[{col!r}], errors='coerce') "
                            f"or cast to string first (df[{col!r}].astype(str).str...)."
                        )
        self.generic_visit(node)

def _coerce_message_content(content: Any) -> str:
    """Normalize LangChain message content into a plain string."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)

    return str(content)


def _extract_json_dict(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from arbitrary text."""
    stripped = text.strip()
    if not stripped:
        return None

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None

def syntactic_validator_node(state: CodingAgentState) -> dict:
    """Checks for valid Python syntax using the built-in compile() function."""
    logger.info("--- Checking Syntax ---")
    logger.info(f"--- CODE TO VALIDATE ---\n{state.get('generated_code')}\n")
    code = state.get("generated_code")
    if not code:
         logger.warning("❌ No code found to validate.")
         # Return error state immediately if no code
         return {
            "syntax_error": True,
            "code_validation_feedback": "No code was generated."
         }

    try:
        # Use compile() to check syntax without execution
        # '<string>' is a placeholder filename for error messages
        # 'exec' mode checks syntax for a sequence of statements
        compile(code, '<string>', 'exec')
        interactive_errors = _find_interactive_input_errors(code)
        if interactive_errors:
            feedback = (
                "\n".join(interactive_errors)
                + "\nGenerated code runs unattended. Ask the user for clarification before coding, "
                "or raise ValueError for missing required values; do not use interactive input."
            )
            logger.warning("❌ Interactive input guardrail violation: %s", feedback)
            return {
                "syntax_error": True,
                "code_validation_feedback": feedback,
            }
        assumed_weekday_errors = _find_assumed_weekday_errors(code, state.get("user_prompt", ""))
        if assumed_weekday_errors:
            feedback = (
                "\n".join(assumed_weekday_errors)
                + "\nAsk the user which weekday/day to use instead of choosing one."
            )
            logger.warning("❌ Assumed weekday guardrail violation: %s", feedback)
            return {
                "syntax_error": True,
                "code_validation_feedback": feedback,
            }
        logger.info("✅ Syntax is valid (compilable).")
        return {"syntax_error": False}
    except (SyntaxError, ValueError) as e: # Catch compile-time errors
        logger.warning(f"❌ Invalid Syntax or Compile Error: {e}")
        # Provide specific feedback for the coder LLM
        return {
            "syntax_error": True,
            "code_validation_feedback": f"SyntaxError or CompileError: {e}. Please fix the Python syntax/structure."
        }
    except Exception as e: # Catch unexpected errors during compile check
        logger.error(f"❌ Unexpected error during syntax check: {e}")
        return {
            "syntax_error": True,
            "code_validation_feedback": f"Unexpected validation error: {e}. Cannot check syntax."
        }

class DataFrameColumnVisitor(ast.NodeVisitor):
    """AST visitor to find invalid column access on df, allowing new columns created in code."""
    def __init__(self, schema_columns: set[str]):
        self.schema_columns = set(schema_columns)
        self.created_columns: set[str] = set()
        self.errors: list[str] = []

    def _extract_str_key(self, slice_node):
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return slice_node.value
        if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
            val = slice_node.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                return val.value
        return None

    def _record_created_col(self, target):
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "df":
            col = self._extract_str_key(target.slice)
            if col:
                self.created_columns.add(col)

    def visit_Assign(self, node: ast.Assign):
        for t in node.targets:
            self._record_created_col(t)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self._record_created_col(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        self._record_created_col(node.target)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == "df":
            col = self._extract_str_key(node.slice)
            ctx = getattr(node, "ctx", None)
            is_read = (ctx is None) or isinstance(ctx, ast.Load)
            if col and is_read:
                if col not in self.schema_columns and col not in self.created_columns:
                    self.errors.append(
                        f"SemanticError on line {node.lineno}: Column '{col}' not in schema."
                    )
        self.generic_visit(node)

class JoinSafetyVisitor(ast.NodeVisitor):
    """AST visitor to intercept and validate merge/join operations to prevent OOM errors."""
    def __init__(self, schemas: dict, metadata: list):
        self.schemas = schemas
        self.metadata = metadata
        self.errors: list[str] = []

    def visit_Call(self, node: ast.Call):
        is_merge_call = False
        join_type = 'inner'
        provided_keys = []

        # Extract keyword arguments
        kwargs = {}
        for kw in node.keywords:
            if kw.arg:
                if isinstance(kw.value, ast.Constant):
                    kwargs[kw.arg] = kw.value.value
                elif isinstance(kw.value, ast.List):
                    kwargs[kw.arg] = [el.value for el in kw.value.elts if isinstance(el, ast.Constant)]

        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'pd' and node.func.attr == 'merge':
                is_merge_call = True
            elif node.func.attr in ('merge', 'join'):
                is_merge_call = True

        if is_merge_call:
            join_type = kwargs.get('how', None)
            
            # Check positional arguments for 'how'
            if join_type is None:
                is_pd_merge = isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'pd'
                how_idx = 2 if is_pd_merge else 1
                if len(node.args) > how_idx and isinstance(node.args[how_idx], ast.Constant):
                    join_type = node.args[how_idx].value
                    
            if join_type is None:
                join_type = 'inner'

            if 'on' in kwargs:
                on_val = kwargs['on']
                provided_keys = on_val if isinstance(on_val, list) else [on_val]
            
            try:
                # We attempt to find the metadata. For a static AST, accurately mapping variables to
                # datasets is complex, so we will use a conservative global check for cross joins.
                # If there are multiple datasets, we just check if any combination exceeds the limit,
                # or if row counts are missing, we block the cross join outright.
                if join_type == 'cross':
                    # Find max row counts among all datasets in metadata
                    max_n = None
                    if self.metadata:
                        for ds in self.metadata:
                            rc = ds.get('row_count')
                            if rc is not None:
                                max_n = max(max_n, rc) if max_n is not None else rc
                                
                    if max_n is not None:
                        validate_join_safety({'row_count': max_n}, {'row_count': max_n}, join_type)
                    else:
                        # Missing row counts, triggers strict block
                        validate_join_safety({}, {}, join_type)
                        
                if provided_keys:
                    all_columns = set(self.schemas.keys())
                    for ds in self.metadata or []:
                        if not isinstance(ds, dict):
                            continue
                        cols = ds.get('columns')
                        if isinstance(cols, list):
                            all_columns.update(str(c) for c in cols)
                        ds_schema = ds.get('schema')
                        if isinstance(ds_schema, dict):
                            all_columns.update(str(c) for c in ds_schema.keys())
                    all_columns = list(all_columns)
                    check_join_keys(all_columns, all_columns, provided_keys)
                    
            except SafetyGuardrailError as e:
                self.errors.append(f"SemanticError on line {node.lineno}: {str(e)}")

        self.generic_visit(node)


class NumericColumnStringComparisonVisitor(ast.NodeVisitor):
    """
    Flags comparisons like df['numeric_col'] == '6' 
    where the column is numeric but compared to a string literal.
    """
    def __init__(self, schema: dict[str, Any]):
        self.schema = schema or {}
        self.errors: list[str] = []

    def _extract_str_key(self, slice_node):
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return slice_node.value
        if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
            val = slice_node.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                return val.value
        return None

    def visit_Compare(self, node: ast.Compare):
        # Match: df['col'] == 'some_string' or 'some_string' == df['col']
        left = node.left
        comparators = node.comparators

        operands = [left] + comparators
        for i in range(len(operands) - 1):
            op_left, op_right = operands[i], operands[i+1]

            # Find which operand is the df subscript and which is the constant
            df_sub, const_val_node = None, None
            if isinstance(op_left, ast.Subscript) and isinstance(op_left.value, ast.Name) and op_left.value.id == 'df':
                df_sub, const_val_node = op_left, op_right
            elif isinstance(op_right, ast.Subscript) and isinstance(op_right.value, ast.Name) and op_right.value.id == 'df':
                df_sub, const_val_node = op_right, op_left

            if df_sub and const_val_node:
                col = self._extract_str_key(df_sub.slice)
                if col and _is_numeric_dtype(self.schema.get(col)):
                    # Any string literal compared to a numeric column yields an
                    # empty filter. Numeric-looking strings are usually rewritten
                    # by NumericStringComparisonFixer before we get here; if one
                    # slips through, or the string is non-numeric (which the fixer
                    # cannot auto-convert), flag it so the coder retries.
                    if isinstance(const_val_node, ast.Constant) and isinstance(const_val_node.value, str):
                        if const_val_node.value.replace('.', '', 1).isdigit():
                            self.errors.append(
                                f"SemanticError on line {node.lineno}: "
                                f"Column '{col}' is numeric ({self.schema.get(col)}) but is being compared to "
                                f"a string literal '{const_val_node.value}'. This will likely result in an empty filter. "
                                f"Use a numeric comparison instead: df['{col}'] == {const_val_node.value}"
                            )
                        else:
                            self.errors.append(
                                f"SemanticError on line {node.lineno}: "
                                f"Column '{col}' is numeric ({self.schema.get(col)}) but is being compared to "
                                f"the non-numeric string literal '{const_val_node.value}'. This comparison can never "
                                f"match and will silently produce an empty result. Compare against a numeric value, "
                                f"or map the label to its numeric encoding first."
                            )
        self.generic_visit(node)


class ObjectColumnNumericComparisonVisitor(ast.NodeVisitor):
    """
    Flags comparisons like df['month'] == 6 when schema says the column is
    object/string. CSV reads can store numeric-looking values as strings, so
    raw numeric comparisons may silently return an empty dataset.
    """
    def __init__(self, schema: dict[str, Any]):
        self.schema = schema or {}
        self.errors: list[str] = []

    def _extract_str_key(self, slice_node):
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return slice_node.value
        if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
            val = slice_node.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                return val.value
        return None

    def visit_Compare(self, node: ast.Compare):
        operands = [node.left] + node.comparators
        for i in range(len(operands) - 1):
            op_left, op_right = operands[i], operands[i + 1]

            df_sub, const_val_node = None, None
            if (
                isinstance(op_left, ast.Subscript)
                and isinstance(op_left.value, ast.Name)
                and op_left.value.id == "df"
            ):
                df_sub, const_val_node = op_left, op_right
            elif (
                isinstance(op_right, ast.Subscript)
                and isinstance(op_right.value, ast.Name)
                and op_right.value.id == "df"
            ):
                df_sub, const_val_node = op_right, op_left

            if not df_sub or not isinstance(const_val_node, ast.Constant):
                continue

            col = self._extract_str_key(df_sub.slice)
            dtype = self.schema.get(col)
            if col and _is_string_like_dtype(dtype) and _is_numeric_literal(const_val_node.value):
                self.errors.append(
                    f"SemanticError on line {node.lineno}: "
                    f"Column '{col}' is typed as {dtype}, but it is being compared to numeric literal "
                    f"{const_val_node.value!r}. This can silently produce an empty filter when values are stored "
                    "as strings. Coerce the column for the comparison, for example: "
                    f"series = pd.to_numeric(df[{col!r}], errors='coerce'); df = df[series == {const_val_node.value!r}]"
                )

        self.generic_visit(node)


class StringEqualityWhitespaceFixer(ast.NodeTransformer):
    """
    Rewrites equality/inequality of a string-like column against a string literal
    so leading/trailing whitespace in the stored data does not silently produce
    an empty result:

        df['sex'] == 'Female'  ->  df['sex'].astype(str).str.strip() == 'Female'

    Only string/object/category columns are transformed (numeric columns are left
    unchanged). The strip affects ONLY the comparison; it does not mutate the
    dataframe or any values written out.
    """
    def __init__(self, schema):
        self.schema = schema or {}

    def _extract_str_key(self, slice_node):
        # Pull the column name string out of a df['col'] subscript slice.
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return slice_node.value
        if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):  # py<3.9 compat
            val = slice_node.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                return val.value
        return None

    def _df_col_name(self, node):
        # Return the column name iff node is exactly df['<str>'], else None.
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "df"
        ):
            return self._extract_str_key(node.slice)
        return None

    def _wrap_strip(self, subscript_node):
        # Build the AST for: <subscript>.astype(str).str.strip()
        astype_call = ast.Call(
            func=ast.Attribute(value=subscript_node, attr="astype", ctx=ast.Load()),
            args=[ast.Name(id="str", ctx=ast.Load())],
            keywords=[],
        )
        str_attr = ast.Attribute(value=astype_call, attr="str", ctx=ast.Load())
        return ast.Call(
            func=ast.Attribute(value=str_attr, attr="strip", ctx=ast.Load()),
            args=[],
            keywords=[],
        )

    @staticmethod
    def _is_str_const(n):
        return isinstance(n, ast.Constant) and isinstance(n.value, str)

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)

        # Only single == / != comparisons against a string literal.
        if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            return node

        left, right = node.left, node.comparators[0]

        # Order A: df['col'] == 'literal'
        col = self._df_col_name(left)
        if col and self._is_str_const(right) and _is_string_like_dtype(self.schema.get(col)):
            node.left = self._wrap_strip(left)
            return node

        # Order B: 'literal' == df['col']
        col_r = self._df_col_name(right)
        if col_r and self._is_str_const(left) and _is_string_like_dtype(self.schema.get(col_r)):
            node.comparators[0] = self._wrap_strip(right)
            return node

        return node


class DeprecatedApiFixer(ast.NodeTransformer):
    """Rewrite removed APIs when there is a safe mechanical replacement."""

    REMOVED_NUMPY_ALIASES = {
        "bool": "bool",
        "float": "float",
        "int": "int",
        "object": "object",
        "str": "str",
    }

    RENAMED_METHODS = {
        "as_matrix": "to_numpy",
        "get_feature_names": "get_feature_names_out",
        "iteritems": "items",
    }

    #: Keywords only pandas' append ever took. Their presence identifies a
    #: DataFrame/Series append even when the result is discarded.
    PANDAS_APPEND_KWARGS = frozenset({"ignore_index", "verify_integrity", "sort"})

    @staticmethod
    def _looks_like_append(node) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and len(node.args) >= 1
        )

    @classmethod
    def _as_concat(cls, call: ast.Call) -> ast.Call:
        """`recv.append(other, ...)` -> `pd.concat([recv, other], ignore_index=True)`.

        A dict or a bare Series row is wrapped in a DataFrame first: concat takes
        frames, and `df.append({...}, ignore_index=True)` was the idiom pandas
        removed, so it is the shape most likely to turn up here.
        """
        receiver = call.func.value
        other = call.args[0]
        if isinstance(other, (ast.Dict, ast.DictComp)):
            other = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="pd", ctx=ast.Load()),
                    attr="DataFrame", ctx=ast.Load(),
                ),
                args=[ast.List(elts=[other], ctx=ast.Load())],
                keywords=[],
            )
        ignore_index = next(
            (kw.value for kw in call.keywords if kw.arg == "ignore_index"),
            ast.Constant(value=True),
        )
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="pd", ctx=ast.Load()),
                    attr="concat", ctx=ast.Load(),
                ),
                args=[ast.List(elts=[receiver, other], ctx=ast.Load())],
                keywords=[ast.keyword(arg="ignore_index", value=ignore_index)],
            ),
            call,
        )

    def visit_Assign(self, node: ast.Assign):
        """`x = y.append(z)` is pandas: list.append returns None."""
        self.generic_visit(node)
        if self._looks_like_append(node.value):
            node.value = self._as_concat(node.value)
        return node

    @staticmethod
    def _is_one_hot_encoder_call(node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id == "OneHotEncoder"
        if isinstance(func, ast.Attribute):
            return func.attr == "OneHotEncoder"
        return False

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "np"
            and node.attr in self.REMOVED_NUMPY_ALIASES
        ):
            return ast.copy_location(
                ast.Name(id=self.REMOVED_NUMPY_ALIASES[node.attr], ctx=node.ctx),
                node,
            )
        if node.attr in self.RENAMED_METHODS:
            node.attr = self.RENAMED_METHODS[node.attr]
        return node

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        # A pandas-only keyword identifies DataFrame.append even when the result
        # is thrown away. Checked before the OneHotEncoder early-return below.
        if self._looks_like_append(node) and any(
            kw.arg in self.PANDAS_APPEND_KWARGS for kw in node.keywords
        ):
            return self._as_concat(node)
        if not self._is_one_hot_encoder_call(node):
            return node

        has_sparse_output = any(keyword.arg == "sparse_output" for keyword in node.keywords)
        updated_keywords = []
        for keyword in node.keywords:
            if keyword.arg == "sparse":
                if not has_sparse_output:
                    keyword.arg = "sparse_output"
                    updated_keywords.append(keyword)
                continue
            updated_keywords.append(keyword)
        node.keywords = updated_keywords
        return node


class DeprecatedApiVisitor(ast.NodeVisitor):
    """Reject deprecated APIs whose replacement is too ambiguous to rewrite."""

    def __init__(self):
        self.errors: list[str] = []

    @staticmethod
    def _is_attr(node: ast.AST, base: str, attr: str) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == base
            and node.attr == attr
        )

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        if module == "sklearn.cross_validation" or module.startswith("sklearn.cross_validation."):
            self.errors.append(
                f"SemanticError on line {node.lineno}: sklearn.cross_validation is removed. "
                "Use sklearn.model_selection instead."
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr == "ix":
            self.errors.append(
                f"SemanticError on line {node.lineno}: pandas .ix[] is removed. "
                "Use .loc[] for label-based selection or .iloc[] for positional selection."
            )
        if self._is_attr(node, "pd", "np"):
            self.errors.append(
                f"SemanticError on line {node.lineno}: pd.np is removed. Import numpy as np directly."
            )
        if self._is_attr(node, "pd", "Panel"):
            self.errors.append(
                f"SemanticError on line {node.lineno}: pd.Panel is removed. Use a DataFrame, xarray, or another explicit structure."
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "df"
        ):
            self.errors.append(
                f"SemanticError on line {node.lineno}: df.append() is removed in modern pandas. "
                "Use pd.concat([...], ignore_index=True) instead."
            )
        self.generic_visit(node)


class NumericStringComparisonFixer(ast.NodeTransformer):
    def __init__(self, schema):
        self.schema = schema or {}

    def _extract_str_key(self, slice_node):
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return slice_node.value

        if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
            val = slice_node.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                return val.value

        return None

    def visit_Compare(self, node):
        self.generic_visit(node)

        operands = [node.left] + node.comparators

        for i in range(len(operands) - 1):
            left = operands[i]
            right = operands[i + 1]

            df_sub = None
            const = None

            if (
                isinstance(left, ast.Subscript)
                and isinstance(left.value, ast.Name)
                and left.value.id == "df"
            ):
                df_sub = left
                const = right

            elif (
                isinstance(right, ast.Subscript)
                and isinstance(right.value, ast.Name)
                and right.value.id == "df"
            ):
                df_sub = right
                const = left

            if not df_sub:
                continue

            col = self._extract_str_key(df_sub.slice)

            if (
                col
                and _is_numeric_dtype(self.schema.get(col))
                and isinstance(const, ast.Constant)
                and isinstance(const.value, str)
            ):
                val = const.value

                if val.replace(".", "", 1).isdigit():
                    const.value = (
                        float(val) if "." in val else int(val)
                    )

        return node


class InvalidPivotAggVisitor(ast.NodeVisitor):
    """AST visitor to prevent aggregating non-numeric columns in pivot_table using numeric aggregation functions."""
    def __init__(self, schema: dict[str, Any]):
        self.schema = schema or {}
        self.errors: list[str] = []

    def visit_Call(self, node: ast.Call):
        is_pivot = False
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'pivot_table':
            is_pivot = True
            
        if is_pivot:
            def _extract_val(v_node):
                if isinstance(v_node, ast.Constant):
                    return v_node.value
                elif isinstance(v_node, ast.List):
                    return [_extract_val(el) for el in v_node.elts]
                elif isinstance(v_node, ast.Name):
                    return v_node.id
                elif isinstance(v_node, ast.Attribute):
                    base = _extract_val(v_node.value)
                    if base:
                        return f"{base}.{v_node.attr}"
                    return v_node.attr
                return None

            kwargs = {}
            for kw in node.keywords:
                if kw.arg:
                    extracted = _extract_val(kw.value)
                    if extracted is not None:
                        kwargs[kw.arg] = extracted
            
            aggfunc = kwargs.get('aggfunc', 'mean')
            aggfunc_str = str(aggfunc).lower()
            
            numeric_aggs = ['sum', 'mean', 'median', 'std', 'var']
            is_numeric_agg = any(na in aggfunc_str for na in numeric_aggs)
                    
            if is_numeric_agg:
                values = kwargs.get('values')
                if not values:
                    # try to extract from positional arguments
                    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'pd':
                        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                            values = node.args[1].value
                    else:
                        if len(node.args) > 0 and isinstance(node.args[0], ast.Constant):
                            values = node.args[0].value
                            
                if values:
                    vals = values if isinstance(values, list) else [values]
                    for val in vals:
                        dtype = self.schema.get(val)
                        if dtype and not _is_numeric_dtype(dtype):
                            self.errors.append(
                                f"SemanticError on line {node.lineno}: Cannot aggregate non-numeric column '{val}' (dtype: {dtype}) using aggfunc '{aggfunc}' in pivot_table. Use a valid numeric column or a different aggregation function."
                            )

        self.generic_visit(node)

class InvalidRegexVisitor(ast.NodeVisitor):
    """AST visitor to prevent invalid regular expressions in pandas str accessors and re module."""
    def __init__(self):
        self.errors: list[str] = []
        self.string_vars: dict[str, str] = {}

    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.string_vars[target.id] = node.value.value
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        is_regex_call = False
        func_name = ""
        
        skip_check = any(kw.arg == 'regex' and isinstance(kw.value, ast.Constant) and kw.value.value is False for kw in node.keywords)
        has_regex_true = any(kw.arg == 'regex' and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in node.keywords)

        if not skip_check and isinstance(node.func, ast.Attribute):
            if node.func.attr in ('extract', 'extractall', 'contains', 'match', 'findall', 'replace'):
                if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'str':
                    is_regex_call = True
                    func_name = f".str.{node.func.attr}"
                elif node.func.attr == 'replace' and has_regex_true:
                    is_regex_call = True
                    func_name = ".replace"
            elif isinstance(node.func.value, ast.Name) and node.func.value.id == 're':
                if node.func.attr in ('compile', 'match', 'search', 'findall', 'sub', 'split'):
                    is_regex_call = True
                    func_name = f"re.{node.func.attr}"
        
        if is_regex_call:
            pattern = None
            for kw in node.keywords:
                if kw.arg in ('pat', 'pattern', 'to_replace'):
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        pattern = kw.value.value
                    elif isinstance(kw.value, ast.Name) and kw.value.id in self.string_vars:
                        pattern = self.string_vars[kw.value.id]
                    
            if pattern is None and len(node.args) > 0:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    pattern = first_arg.value
                elif isinstance(first_arg, ast.Name) and first_arg.id in self.string_vars:
                    pattern = self.string_vars[first_arg.id]
                    
            if pattern is not None:
                try:
                    import re
                    re.compile(pattern)
                except re.error as e:
                    self.errors.append(
                        f"SemanticError on line {node.lineno}: Invalid regular expression '{pattern}' in {func_name}(): {e}"
                    )
        
        self.generic_visit(node)


def _prompt_requests_missing_value_removal(prompt: str) -> bool:
    lowered = (prompt or "").lower()
    return bool(
        re.search(
            r"\b(drop|remove|filter\s+out|exclude|delete)\b.*\b("
            r"na|nan|null|missing|blank|empty\s+values?)\b",
            lowered,
        )
        or re.search(
            r"\b(non[-\s]?null|not\s+null|not\s+missing|without\s+missing|complete\s+cases?)\b",
            lowered,
        )
    )


class BlanketDropnaVisitor(ast.NodeVisitor):
    """Detect broad row deletion from df.dropna() when missing-value removal was not requested."""

    def __init__(self, user_prompt: str):
        self.errors: list[str] = []
        self.missing_removal_requested = _prompt_requests_missing_value_removal(user_prompt)

    def visit_Call(self, node: ast.Call):
        if (
            not self.missing_removal_requested
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "dropna"
        ):
            is_direct_df_dropna = (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "df"
            )
            if not is_direct_df_dropna:
                self.generic_visit(node)
                return

            has_subset = any(keyword.arg == "subset" for keyword in node.keywords)
            if not has_subset:
                self.errors.append(
                    f"SemanticError on line {node.lineno}: Blanket dropna() removes valid rows with sparse optional fields. "
                    "Do not drop missing values unless the user explicitly asks for it; apply the requested filter directly."
                )
        self.generic_visit(node)


class UnsafeNullableStringApplyVisitor(ast.NodeVisitor):
    """Detect nullable df['col'].apply(lambda x: string ops on x) without a null/type guard."""

    STRING_METHODS = {
        "capitalize",
        "casefold",
        "count",
        "endswith",
        "find",
        "index",
        "isalnum",
        "isalpha",
        "isdigit",
        "islower",
        "isnumeric",
        "isspace",
        "istitle",
        "isupper",
        "lower",
        "lstrip",
        "partition",
        "replace",
        "rfind",
        "rpartition",
        "rsplit",
        "rstrip",
        "split",
        "startswith",
        "strip",
        "title",
        "upper",
    }

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema or {}
        self.errors: list[str] = []

    def _extract_str_key(self, slice_node):
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return slice_node.value
        if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
            val = slice_node.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                return val.value
        return None

    def _extract_direct_df_column(self, node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "df"
        ):
            return self._extract_str_key(node.slice)
        return None

    def _references_lambda_arg(self, node: ast.AST, arg_names: set[str]) -> bool:
        return any(isinstance(child, ast.Name) and child.id in arg_names for child in ast.walk(node))

    def _is_isinstance_str_guard(self, node: ast.Call, arg_names: set[str]) -> bool:
        if not (isinstance(node.func, ast.Name) and node.func.id == "isinstance"):
            return False
        if len(node.args) < 2 or not self._references_lambda_arg(node.args[0], arg_names):
            return False

        type_arg = node.args[1]
        if isinstance(type_arg, ast.Name) and type_arg.id == "str":
            return True
        if isinstance(type_arg, ast.Tuple):
            return any(isinstance(elt, ast.Name) and elt.id == "str" for elt in type_arg.elts)
        return False

    def _is_pd_null_check(self, node: ast.Call, arg_names: set[str], safe_when_true: bool) -> bool:
        safe_names = {"notna", "notnull"} if safe_when_true else {"isna", "isnull"}
        func = node.func
        if isinstance(func, ast.Attribute):
            func_name = func.attr
        elif isinstance(func, ast.Name):
            func_name = func.id
        else:
            return False

        return func_name in safe_names and any(
            self._references_lambda_arg(arg, arg_names) for arg in node.args
        )

    def _has_null_or_type_guard(self, node: ast.AST, arg_names: set[str]) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if self._is_isinstance_str_guard(child, arg_names):
                    return True
                if self._is_pd_null_check(child, arg_names, safe_when_true=True):
                    return True
            elif (
                isinstance(child, ast.UnaryOp)
                and isinstance(child.op, ast.Not)
                and isinstance(child.operand, ast.Call)
                and self._is_pd_null_check(child.operand, arg_names, safe_when_true=False)
            ):
                return True
        return False

    def _uses_string_membership_or_method(self, node: ast.AST, arg_names: set[str]) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Compare) and any(
                isinstance(op, (ast.In, ast.NotIn)) for op in child.ops
            ):
                operands = [child.left] + list(child.comparators)
                if any(self._references_lambda_arg(operand, arg_names) for operand in operands):
                    return True
            elif (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in self.STRING_METHODS
                and self._references_lambda_arg(child.func.value, arg_names)
            ):
                return True
        return False

    def visit_Call(self, node: ast.Call):
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "apply"
            and node.args
            and isinstance(node.args[0], ast.Lambda)
        ):
            self.generic_visit(node)
            return

        col = self._extract_direct_df_column(node.func.value)
        lambda_node = node.args[0]
        arg_names = {arg.arg for arg in lambda_node.args.args}
        dtype = self.schema.get(col)

        if (
            col
            and arg_names
            and self._uses_string_membership_or_method(lambda_node.body, arg_names)
            and not self._has_null_or_type_guard(lambda_node.body, arg_names)
        ):
            dtype_text = f" ({dtype})" if dtype else ""
            self.errors.append(
                f"SemanticError on line {node.lineno}: Column '{col}'{dtype_text} may contain null/NaN values, "
                "but this lambda uses string parsing before null/type handling. Build a clean Series for only this "
                f"column, for example df[{col!r}].dropna().astype(str), or guard the lambda with pd.notna(x) "
                "or isinstance(x, str). Do not drop the whole dataframe for this."
            )

        self.generic_visit(node)


def _requires_sparse_similarity(prompt: str) -> bool:
    prompt_lower = (prompt or "").lower()
    if not ("cosine similarity" in prompt_lower or "similarity matrix" in prompt_lower):
        return False

    bounded_small_terms = [
        "small subset",
        "sample of",
        "top 10",
        "top 20",
        "top 50",
        "first 10",
        "first 20",
        "first 50",
        "10 products",
        "20 products",
        "50 products",
        "10 users",
        "20 users",
        "50 users",
        "numeric columns",
    ]
    unbounded_large_terms = [
        "full",
        "all unique",
        "entire",
        "large",
        "high-cardinality",
        "high cardinality",
        "co-occurrence",
        "cooccurrence",
        "recommender",
        "user-item",
        "rating matrix",
        "product-order",
        "product-product",
    ]

    if any(term in prompt_lower for term in bounded_small_terms) and not any(
        term in prompt_lower for term in unbounded_large_terms
    ):
        return False

    return any(
        term in prompt_lower
        for term in [
            "pairwise",
            "co-occurrence",
            "cooccurrence",
            "product",
            "user-item",
            "rating matrix",
            "recommender",
        ]
    )


class DenseSimilarityMatrixVisitor(ast.NodeVisitor):
    """Blocks dense matrix construction for high-cardinality cosine/co-occurrence prompts."""
    def __init__(self):
        self.errors: list[str] = []
        self.saw_cosine_similarity = False
        self.saw_sparse_constructor = False
        self.saw_dense_matrix_builder = False
        self.saw_dense_conversion = False
        self.cosine_dense_output_false = False
        self.factorize_code_vars: set[str] = set()
        self.saw_factorize_nunique_error = False
        self.raw_coordinate_list_vars: set[str] = set()
        self.saw_similarity_coo_coordinates = False
        self.saw_unique_pair_filter = False
        self.coo_row_vars: set[str] = set()
        self.coo_col_vars: set[str] = set()
        self.product_label_vars: set[str] = set()
        self.cosine_called_on_untransposed_incidence = False
        self.saw_product_label_mapping = False

    def _call_name(self, node: ast.Call) -> str:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts = [func.attr]
            value = func.value
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                parts.append(value.id)
            return ".".join(reversed(parts))
        return ""

    def _is_dataframe_column_access(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in {"df", "data", "dataset"}
        )

    def _is_raw_dataframe_column_coordinate(self, node: ast.AST) -> bool:
        """Detects raw df['id'].values-style sparse coordinates that can exceed matrix shape."""
        if self._is_dataframe_column_access(node):
            return True

        if isinstance(node, ast.Name) and node.id in self.raw_coordinate_list_vars:
            return True

        if isinstance(node, ast.Attribute):
            if node.attr in {"values", "array"}:
                return self._is_raw_dataframe_column_coordinate(node.value)
            return False

        if isinstance(node, ast.Call):
            name = self._call_name(node)
            if name.lower().endswith((".to_numpy", ".tolist")) and isinstance(node.func, ast.Attribute):
                return self._is_raw_dataframe_column_coordinate(node.func.value)
            return False

        return False

    def _groupby_key_var(self, node: ast.For) -> str | None:
        if not isinstance(node.target, ast.Tuple) or not node.target.elts:
            return None
        if not isinstance(node.target.elts[0], ast.Name):
            return None
        if not isinstance(node.iter, ast.Call):
            return None

        iter_name = self._call_name(node.iter).lower()
        if not iter_name.endswith(".groupby"):
            return None
        return node.target.elts[0].id

    def _list_append_name(self, node: ast.Call) -> str | None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
        ):
            return node.func.value.id
        return None

    def _find_raw_group_key_appends(self, statements: list[ast.stmt], key_var: str) -> set[str]:
        raw_lists: set[str] = set()
        for child in ast.walk(ast.Module(body=statements, type_ignores=[])):
            if not isinstance(child, ast.Call) or not child.args:
                continue
            list_name = self._list_append_name(child)
            if list_name and isinstance(child.args[0], ast.Name) and child.args[0].id == key_var:
                raw_lists.add(list_name)
        return raw_lists

    def _sparse_constructor_uses_raw_coordinates(self, node: ast.Call) -> bool:
        if not node.args:
            return False

        data_arg = node.args[0]
        if not isinstance(data_arg, ast.Tuple) or len(data_arg.elts) < 2:
            return False

        coords_arg = data_arg.elts[1]
        if not isinstance(coords_arg, ast.Tuple) or len(coords_arg.elts) < 2:
            return False

        return any(self._is_raw_dataframe_column_coordinate(coord) for coord in coords_arg.elts[:2])

    def _is_coo_coordinate_attr(self, node: ast.AST, attr: str | None = None) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr in {"row", "col"}
            and (attr is None or node.attr == attr)
        )

    def _is_row_expr(self, node: ast.AST) -> bool:
        return (
            self._is_coo_coordinate_attr(node, "row")
            or (isinstance(node, ast.Name) and node.id in self.coo_row_vars)
            or self._is_named_product_pair_column(node, "a")
        )

    def _is_col_expr(self, node: ast.AST) -> bool:
        return (
            self._is_coo_coordinate_attr(node, "col")
            or (isinstance(node, ast.Name) and node.id in self.coo_col_vars)
            or self._is_named_product_pair_column(node, "b")
        )

    def _is_named_product_pair_column(self, node: ast.AST, side: str) -> bool:
        if isinstance(node, ast.Name):
            return node.id.lower() in {f"product_id_{side}", f"item_id_{side}", f"{side}_id"}

        if not isinstance(node, ast.Subscript):
            return False

        slice_node = node.slice
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            column_name = slice_node.value.lower()
            return column_name in {
                f"product_id_{side}",
                f"item_id_{side}",
                f"{side}_id",
            }
        return False

    def _is_incidence_matrix_name(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Name)
            and any(term in node.id.lower() for term in ["incidence", "matrix", "sparse_matrix"])
        )

    def _is_product_label_mapping(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in self.product_label_vars

    def _comparison_is_unique_pair_filter(self, node: ast.Compare) -> bool:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return False

        left = node.left
        right = node.comparators[0]
        op = node.ops[0]
        compares_row_col = (self._is_row_expr(left) and self._is_col_expr(right)) or (
            self._is_col_expr(left) and self._is_row_expr(right)
        )
        if not compares_row_col:
            return False

        return isinstance(op, (ast.Lt, ast.Gt, ast.NotEq))

    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.Call) and self._call_name(node.value).lower().endswith("factorize"):
            for target in node.targets:
                if isinstance(target, ast.Tuple) and target.elts and isinstance(target.elts[0], ast.Name):
                    self.factorize_code_vars.add(target.elts[0].id)
                    if len(target.elts) > 1 and isinstance(target.elts[1], ast.Name):
                        label_var = target.elts[1].id
                        if "product" in label_var.lower() or "item" in label_var.lower():
                            self.product_label_vars.add(label_var)

        if isinstance(node.value, ast.Call) and self._call_name(node.value).lower().endswith("unique"):
            for target in node.targets:
                if isinstance(target, ast.Name) and ("product" in target.id.lower() or "item" in target.id.lower()):
                    self.product_label_vars.add(target.id)

        if self._is_coo_coordinate_attr(node.value):
            self.saw_similarity_coo_coordinates = True
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if node.value.attr == "row":
                        self.coo_row_vars.add(target.id)
                    elif node.value.attr == "col":
                        self.coo_col_vars.add(target.id)

        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        key_var = self._groupby_key_var(node)
        if key_var:
            self.raw_coordinate_list_vars.update(self._find_raw_group_key_appends(node.body, key_var))

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        name = self._call_name(node)
        name_lower = name.lower()

        if (
            name_lower.endswith(".nunique")
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.factorize_code_vars
            and not self.saw_factorize_nunique_error
        ):
            self.saw_factorize_nunique_error = True
            self.errors.append(
                "SemanticError: pd.factorize returns NumPy code arrays, so .nunique() is invalid on factorized codes. "
                "Use len(unique_labels) from the second pd.factorize return value for sparse matrix shape."
            )

        if name_lower.endswith("cosine_similarity"):
            self.saw_cosine_similarity = True
            if node.args and self._is_incidence_matrix_name(node.args[0]):
                self.cosine_called_on_untransposed_incidence = True
            for kw in node.keywords:
                if (
                    kw.arg == "dense_output"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is False
                ):
                    self.cosine_dense_output_false = True

        if name_lower in {
            "pd.pivot_table",
            "pandas.pivot_table",
            "pd.crosstab",
            "pandas.crosstab",
            "pd.get_dummies",
            "pandas.get_dummies",
        } or name_lower.endswith((".pivot_table", ".crosstab", ".get_dummies")):
            self.saw_dense_matrix_builder = True

        if name_lower.endswith((".toarray", ".todense")):
            self.saw_dense_conversion = True

        if name_lower.endswith((
            "csr_matrix",
            "coo_matrix",
            "csc_matrix",
            "lil_matrix",
            "dok_matrix",
        )):
            self.saw_sparse_constructor = True
            if self._sparse_constructor_uses_raw_coordinates(node):
                self.errors.append(
                    "SemanticError: Sparse matrix coordinates must use compact zero-based integer codes. "
                    "Do not pass raw dataframe IDs like df['order_id'].values or df['product_id'].values "
                    "directly into coo_matrix/csr_matrix; factorize or map both dimensions first."
                )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in {"row", "col"}:
            self.saw_similarity_coo_coordinates = True

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if self._is_product_label_mapping(node):
            self.saw_product_label_mapping = True

        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        if self._comparison_is_unique_pair_filter(node):
            self.saw_unique_pair_filter = True

        self.generic_visit(node)

    def finalize(self):
        if self.saw_dense_matrix_builder:
            self.errors.append(
                "SemanticError: This pairwise cosine/co-occurrence request must use sparse matrices. "
                "Do not build dense matrices with pivot_table, crosstab, or get_dummies. "
                "Build a scipy.sparse COO/CSR incidence matrix instead."
            )
        if self.saw_dense_conversion:
            self.errors.append(
                "SemanticError: This pairwise cosine/co-occurrence request must stay sparse. "
                "Do not call .toarray() or .todense()."
            )
        if self.saw_cosine_similarity and not self.cosine_dense_output_false:
            self.errors.append(
                "SemanticError: cosine_similarity must be called with dense_output=False for sparse pairwise similarity requests."
            )
        if self.saw_cosine_similarity and not self.saw_sparse_constructor:
            self.errors.append(
                "SemanticError: Build the product/order incidence matrix with scipy.sparse.coo_matrix or csr_matrix before cosine_similarity."
            )
        if self.saw_cosine_similarity and self.saw_similarity_coo_coordinates and not self.saw_unique_pair_filter:
            self.errors.append(
                "SemanticError: Symmetric pairwise similarity output must remove self-pairs and duplicate mirrored pairs. "
                "Filter COO coordinates with row < col, or an equivalent product_id_a/product_id_b comparison, before returning the edge-list."
            )
        if self.cosine_called_on_untransposed_incidence and self.saw_product_label_mapping:
            self.errors.append(
                "SemanticError: Product-product similarity from an order-by-product incidence matrix must compute "
                "cosine_similarity(incidence_matrix.T, dense_output=False). Calling cosine_similarity on the untransposed "
                "incidence matrix computes order-order similarity, so its row/column indexes cannot be mapped to product IDs."
            )
class GroupbyCumsumResetIndexVisitor(ast.NodeVisitor):
    """Reject groupby-selected cumsum reset_index chains that drop grouping columns."""
    def __init__(self):
        self.errors: list[str] = []

    def _extract_constant(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        return None

    def _extract_groupby_columns(self, node: ast.Call) -> list[str]:
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "groupby"
        ):
            return []

        if not node.args:
            return []

        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            return [first_arg.value]
        if isinstance(first_arg, ast.List):
            return [
                elt.value
                for elt in first_arg.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            ]
        return []

    def _find_groupby_cumsum_chain(self, node: ast.AST) -> tuple[list[str], str | None] | None:
        # Match: df.groupby(group_col)[value_col].cumsum()
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "cumsum"
            and isinstance(node.func.value, ast.Subscript)
        ):
            return None

        selected = node.func.value
        value_col = self._extract_constant(selected.slice)
        groupby_call = selected.value

        if not isinstance(groupby_call, ast.Call):
            return None

        group_cols = self._extract_groupby_columns(groupby_call)
        if not group_cols:
            return None

        return group_cols, value_col if isinstance(value_col, str) else None

    def visit_Call(self, node: ast.Call):
        # Match: df.groupby(group_col)[value_col].cumsum().reset_index()
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "reset_index"
        ):
            chain = self._find_groupby_cumsum_chain(node.func.value)
            if chain:
                group_cols, value_col = chain
                group_desc = ", ".join(repr(col) for col in group_cols)
                value_desc = repr(value_col) if value_col else "the value column"
                self.errors.append(
                    "SemanticError on line "
                    f"{node.lineno}: Do not use groupby({group_desc})[{value_desc}].cumsum().reset_index(). "
                    "That returns index/value columns and drops the grouping column. "
                    "First aggregate into a DataFrame, then add the cumulative column, e.g. "
                    f"result = df.groupby({group_desc}, as_index=False)[{value_desc}].sum(); "
                    f"result['cumulative_{value_col or 'value'}'] = result[{value_desc}].cumsum(); "
                    "return result."
                )

        self.generic_visit(node)


class LargeGridSearchVisitor(ast.NodeVisitor):
    """Blocks exhaustive GridSearchCV calls that are too large for local execution."""
    MAX_TOTAL_FITS = 100

    def __init__(self):
        self.assigned_grids: dict[str, int] = {}
        self.errors: list[str] = []

    def _range_len(self, node: ast.Call) -> int | None:
        if not isinstance(node.func, ast.Name) or node.func.id != "range":
            return None

        args: list[int] = []
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                args.append(arg.value)
            else:
                return None

        try:
            return len(range(*args))
        except TypeError:
            return None

    def _value_options_count(self, node: ast.AST) -> int | None:
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return len(node.elts)
        if isinstance(node, ast.Call):
            return self._range_len(node)
        if isinstance(node, ast.Constant):
            return 1
        return None

    def _grid_size(self, node: ast.AST) -> int | None:
        if isinstance(node, ast.Dict):
            total = 1
            for value in node.values:
                count = self._value_options_count(value)
                if count is None:
                    return None
                total *= max(count, 1)
                if total > self.MAX_TOTAL_FITS:
                    return total
            return total

        if isinstance(node, (ast.List, ast.Tuple)):
            total = 0
            for elt in node.elts:
                size = self._grid_size(elt)
                if size is None:
                    return None
                total += size
                if total > self.MAX_TOTAL_FITS:
                    return total
            return total

        return None

    def _is_grid_search_call(self, node: ast.Call) -> bool:
        if isinstance(node.func, ast.Name):
            return node.func.id == "GridSearchCV"
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == "GridSearchCV"
        return False

    def _extract_grid_size_from_call(self, node: ast.Call) -> int | None:
        param_grid_node = None
        if len(node.args) >= 2:
            param_grid_node = node.args[1]

        for kw in node.keywords:
            if kw.arg == "param_grid":
                param_grid_node = kw.value
                break

        if param_grid_node is None:
            return None

        if isinstance(param_grid_node, ast.Name):
            return self.assigned_grids.get(param_grid_node.id)

        return self._grid_size(param_grid_node)

    def _extract_cv(self, node: ast.Call) -> int:
        for kw in node.keywords:
            if (
                kw.arg == "cv"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, int)
            ):
                return max(kw.value.value, 1)
        return 5

    def visit_Assign(self, node: ast.Assign):
        grid_size = self._grid_size(node.value)
        if grid_size is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.assigned_grids[target.id] = grid_size

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if self._is_grid_search_call(node):
            grid_size = self._extract_grid_size_from_call(node)
            cv = self._extract_cv(node)

            if grid_size is not None:
                total_fits = grid_size * cv
                if total_fits > self.MAX_TOTAL_FITS:
                    self.errors.append(
                        f"SemanticError on line {node.lineno}: GridSearchCV would run "
                        f"{total_fits} fits ({grid_size} parameter combinations * cv={cv}), "
                        f"which is too large for local execution. Use a smaller grid, "
                        f"RandomizedSearchCV with n_iter <= 20, cv <= 3, or skip exhaustive search."
                    )

        self.generic_visit(node)

def _subscript_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if hasattr(ast, "Index") and isinstance(node, ast.Index):
        return _subscript_key(node.value)
    return None


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts = [node.func.attr]
        value = node.func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return ""


def _extract_column_reference(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript):
        key = _subscript_key(node.slice)
        if key:
            base = node.value
            if isinstance(base, ast.Name):
                return f"{base.id}[{key}]"
            if isinstance(base, ast.Subscript):
                nested = _extract_column_reference(base)
                if nested:
                    return f"{nested}[{key}]"
    if isinstance(node, ast.Name):
        return node.id
    return None


class LabelEncoderMisuseVisitor(ast.NodeVisitor):
    """Flags one LabelEncoder being fitted to multiple distinct fields."""
    def __init__(self):
        self.encoder_vars: set[str] = set()
        self.fit_sources: dict[str, set[str]] = {}
        self.errors: list[str] = []

    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.Call) and _call_name(node.value).endswith("LabelEncoder"):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.encoder_vars.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"fit", "fit_transform"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.encoder_vars
            and node.args
        ):
            encoder_name = node.func.value.id
            source = _extract_column_reference(node.args[0]) or ast.unparse(node.args[0])
            sources = self.fit_sources.setdefault(encoder_name, set())
            sources.add(source)
            if len(sources) > 1:
                self.errors.append(
                    f"SemanticError on line {node.lineno}: LabelEncoder '{encoder_name}' is fitted to multiple fields "
                    f"({', '.join(sorted(sources))}). Use one encoder per categorical feature and a separate target encoder."
                )
        self.generic_visit(node)


class SyntheticPredictionDataVisitor(ast.NodeVisitor):
    """
    Blocks fabricated DataFrames being passed into model prediction or encoder transforms.
    Evaluation should use X_test/existing rows unless user-provided new data exists.
    """
    def __init__(self):
        self.synthetic_dataframe_vars: set[str] = set()
        self.errors: list[str] = []

    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.Call) and _call_name(node.value) == "pd.DataFrame":
            has_literal_payload = bool(node.value.args) or any(
                kw.arg in {"data", None} for kw in node.value.keywords
            )
            if has_literal_payload:
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id != "df":
                        self.synthetic_dataframe_vars.add(target.id)
        self.generic_visit(node)

    def _uses_synthetic_dataframe(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name) and node.id in self.synthetic_dataframe_vars:
            return node.id
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in self.synthetic_dataframe_vars:
                return node.value.id
        return None

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"predict", "predict_proba", "transform"}:
            for arg in node.args:
                synthetic_var = self._uses_synthetic_dataframe(arg)
                if synthetic_var:
                    self.errors.append(
                        f"SemanticError on line {node.lineno}: Do not pass fabricated DataFrame '{synthetic_var}' "
                        "to model prediction or encoder transformation. Use X_test/existing rows, or only user-provided new rows."
                    )
        self.generic_visit(node)


class FilteredMLTrainingVisitor(ast.NodeVisitor):
    """Detects ML training on a row-filtered dataset."""
    def __init__(self):
        self.has_row_filter = False
        self.has_model_fit = False

    def _is_row_filter(self, value: ast.AST) -> bool:
        if not isinstance(value, ast.Subscript):
            return False
        slice_node = value.slice
        return isinstance(slice_node, (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.Call, ast.BinOp))

    def visit_Assign(self, node: ast.Assign):
        if self._is_row_filter(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"df", "X", "y"}:
                    self.has_row_filter = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
            self.has_model_fit = True
        self.generic_visit(node)


def _prompt_requests_filtered_ml(user_prompt: str) -> bool:
    prompt = (user_prompt or "").lower()
    ml_terms = (
        "train", "model", "predict", "classification", "classifier", "regression",
        "machine learning", "ml", "fit", "forecast",
    )
    subset_terms = (
        "only", "where", "filter", "filtered", "subset", "rows where", "keep",
        "include", "exclude", "using rows", "use rows",
    )
    return any(term in prompt for term in ml_terms) and any(term in prompt for term in subset_terms)


def static_semantic_validator_node(state: CodingAgentState) -> dict:
    """Uses AST traversal to check for schema violations + dtype-safe .str usage."""
    if state.get("syntax_error"):
        return {}

    logger.info("--- Checking Static Semantics (Schema) ---")
    logger.info(f"--- CODE TO VALIDATE ---\n{state.get('generated_code')}\n")

    code = state.get("generated_code") or ""
    # Shared node: non-DTA coder uses "schema", DTA coder uses "source_schema".
    schema = state.get("schema") or state.get("source_schema") or {}
    # Defensive: a non-dict schema (e.g. a raw Daft/PyArrow Schema object) has
    # no .keys(); coerce it so this shared node can never crash on it.
    if not isinstance(schema, dict):
        schema = {str(f.name): str(f.dtype) for f in schema} if hasattr(schema, "__iter__") else {}

    logger.info("=== VALIDATOR SCHEMA ===")
    logger.info("%s", schema)
    logger.info("Schema type: %s", type(schema))
    known_columns = set(schema.keys())

    try:
        tree = ast.parse(code)

        # Auto-fix numeric column comparisons
        fixer = NumericStringComparisonFixer(schema)
        tree = fixer.visit(tree)
        ast.fix_missing_locations(tree)

        # Auto-fix string equality filters so whitespace-padded values still match
        # (df['sex'] == 'Female' -> df['sex'].astype(str).str.strip() == 'Female').
        ws_fixer = StringEqualityWhitespaceFixer(schema)
        tree = ws_fixer.visit(tree)
        ast.fix_missing_locations(tree)

        # Auto-fix removed/deprecated APIs when the replacement is mechanical.
        deprecated_api_fixer = DeprecatedApiFixer()
        tree = deprecated_api_fixer.visit(tree)
        ast.fix_missing_locations(tree)

        # Convert AST back to source
        code = ast.unparse(tree)

        # Persist corrected code
        state["generated_code"] = code

        logger.info("--- CODE AFTER AUTO-FIX ---")
        logger.info("\n%s", code)

        # Re-parse corrected code
        tree = ast.parse(code)

        # 1) Column existence check
        col_visitor = DataFrameColumnVisitor(known_columns)
        col_visitor.visit(tree)

        # 2) Dtype-safe check: block df['numeric_col'].str...
        str_visitor = StrAccessorOnDfVisitor(schema)
        str_visitor.visit(tree)

        # 3) Join Safety Guardrail Check
        # Try to extract multi-dataset metadata if present, else fallback
        datasets_meta = state.get("datasets_context") or state.get("multi_dataset_state") or []
        join_visitor = JoinSafetyVisitor(schema, datasets_meta)
        join_visitor.visit(tree)

        # 4) Invalid Pivot Aggregation Guardrail
        pivot_visitor = InvalidPivotAggVisitor(schema)
        pivot_visitor.visit(tree)

        # 5) Numeric column compared to string literal
        type_cmp_visitor = NumericColumnStringComparisonVisitor(schema)
        type_cmp_visitor.visit(tree)

        # 6) Object/string column compared to numeric literal
        object_numeric_cmp_visitor = ObjectColumnNumericComparisonVisitor(schema)
        object_numeric_cmp_visitor.visit(tree)

        # 7) Invalid Regex Guardrail
        regex_visitor = InvalidRegexVisitor()
        regex_visitor.visit(tree)

        # 8) Missing-value guardrail: don't silently drop sparse optional fields
        blanket_dropna_visitor = BlanketDropnaVisitor(state.get("user_prompt", ""))
        blanket_dropna_visitor.visit(tree)

        # 9) Nullable string parsing guardrail
        nullable_string_apply_visitor = UnsafeNullableStringApplyVisitor(schema)
        nullable_string_apply_visitor.visit(tree)

        # 10) Dense pairwise similarity OOM guardrail
        dense_similarity_visitor = DenseSimilarityMatrixVisitor()
        if _requires_sparse_similarity(state.get("user_prompt", "")):
            dense_similarity_visitor.visit(tree)
            dense_similarity_visitor.finalize()

        # 11) Grouped cumulative metrics must preserve grouping columns
        groupby_cumsum_visitor = GroupbyCumsumResetIndexVisitor()
        groupby_cumsum_visitor.visit(tree)

        # 12) Block large exhaustive GridSearchCV searches before execution
        grid_search_visitor = LargeGridSearchVisitor()
        grid_search_visitor.visit(tree)

        # 13) ML guardrails: categorical encoders and fabricated prediction rows
        label_encoder_visitor = LabelEncoderMisuseVisitor()
        label_encoder_visitor.visit(tree)

        synthetic_prediction_visitor = SyntheticPredictionDataVisitor()
        synthetic_prediction_visitor.visit(tree)

        # 14) ML guardrail: filtered training must warn about selection bias
        filtered_ml_visitor = FilteredMLTrainingVisitor()
        filtered_ml_visitor.visit(tree)

        # 15) Deprecated APIs that cannot be safely auto-rewritten
        deprecated_api_visitor = DeprecatedApiVisitor()
        deprecated_api_visitor.visit(tree)

        # Combine errors from both checks
        all_errors: list[str] = []
        all_errors.extend(col_visitor.errors)
        all_errors.extend(str_visitor.errors)
        all_errors.extend(join_visitor.errors)
        all_errors.extend(pivot_visitor.errors)
        all_errors.extend(type_cmp_visitor.errors)
        all_errors.extend(object_numeric_cmp_visitor.errors)
        all_errors.extend(regex_visitor.errors)
        all_errors.extend(blanket_dropna_visitor.errors)
        all_errors.extend(nullable_string_apply_visitor.errors)
        all_errors.extend(dense_similarity_visitor.errors)
        all_errors.extend(groupby_cumsum_visitor.errors)
        all_errors.extend(grid_search_visitor.errors)
        all_errors.extend(label_encoder_visitor.errors)
        all_errors.extend(synthetic_prediction_visitor.errors)
        all_errors.extend(deprecated_api_visitor.errors)

        if (
            _prompt_requests_filtered_ml(state.get("user_prompt", ""))
            and filtered_ml_visitor.has_row_filter
            and filtered_ml_visitor.has_model_fit
            and "selection bias" not in code.lower()
        ):
            all_errors.append(
                "SemanticError: This ML prompt trains on a filtered subset of rows, but the code does not warn about "
                "severe selection bias. Print a clear selection-bias warning before fitting the model."
            )

        if all_errors:
            feedback = "\n".join(all_errors)
            logger.warning("❌ Schema/Dtype Violation Found: %s", feedback)
            return {"static_semantic_error": True, "code_validation_feedback": feedback}

        logger.info("✅ Schema check passed.")
        # The executors read code from coder_definition["code"], not
        # generated_code — so return the auto-fixed source in BOTH keys,
        # otherwise the auto-fixes above (numeric + whitespace) are discarded
        # at execution time. Returning it (rather than mutating state in place)
        # is required because the graph checkpoints state between nodes.
        result = {"static_semantic_error": False, "generated_code": code}
        if isinstance(state.get("coder_definition"), dict):
            updated_def = dict(state["coder_definition"])   # shallow copy; preserves other keys
            updated_def["code"] = code                      # the key executors read
            result["coder_definition"] = updated_def
        return result

    except Exception as e:
        logger.error(f"❌ Error during AST parsing/visit for static check: {e}")
        return {
            "static_semantic_error": True,
            "code_validation_feedback": f"AST Analysis Error: {e}. Failed static semantic checks."
        }

import io
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr


def _build_sample_dataframe(state: CodingAgentState) -> pd.DataFrame | None:
    """Construct a small DataFrame from preview/sample data for execution validation."""
    sample_text = state.get("sample_data") or ""
    dataframe: pd.DataFrame | None = None

    if sample_text:
        try:
            dataframe = pd.read_csv(io.StringIO(sample_text))
        except Exception as exc:
            logger.warning("Failed to parse sample_data CSV string: %s", exc)

    if (dataframe is None or dataframe.empty) and state.get("uploaded_csv_preview"):
        preview_rows = state["uploaded_csv_preview"]

        # The preview may be persisted as a JSON string (see app/api/helpers.py).
        if isinstance(preview_rows, str):
            try:
                preview_rows = json.loads(preview_rows)
            except (ValueError, TypeError):
                preview_rows = []

        if isinstance(preview_rows, list) and preview_rows:
            # uploaded_csv_preview comes in two shapes:
            #   1. list[dict] -> each row already keyed by column name.
            #   2. list[list] -> a header row followed by positional value rows.
            if isinstance(preview_rows[0], dict):
                dataframe = pd.DataFrame(
                    [row for row in preview_rows if isinstance(row, dict) and row]
                )
            else:
                headers = preview_rows[0]
                records = preview_rows[1:]
                dataframe = pd.DataFrame(records, columns=headers)

    if dataframe is None or dataframe.empty:
        return None

    # Limit to two rows to keep execution deterministic.
    return dataframe.head(2).reset_index(drop=True)

def execute_code_node(state: CodingAgentState) -> dict:
    """Executes the 'main' function from the generated code against the sample data."""
    if state.get("syntax_error") or state.get("static_semantic_error"):
        return {
            "execution_stdout": "",
            "execution_stderr": "Skipped execution due to validation errors.",
            "execution_error": "Validation failed.",
            "execution_output_data": None
        }
    logger.info("--- Executing Code ---")
    code = state["generated_code"]
    sample_df = _build_sample_dataframe(state)

    if sample_df is None or sample_df.empty:
        logger.error("Sample data unavailable for execution validation.")
        return {
            "execution_stdout": "",
            "execution_stderr": "Sample data unavailable for execution.",
            "execution_error": "Sample data unavailable for execution",
            "execution_output_data": None
        }

    # Single exec scope so top-level imports/helpers are visible inside main();
    # __name__ is deliberately not "__main__" so the script's entry block
    # (which reads the full dataset and writes the real output) stays inert here.
    exec_scope = {"pd": pd, "__name__": "avaloka_sample_execution"}

    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        # Execute the entire script to define functions, including main()
        exec(code, exec_scope)

        # Now, call the main function with the dataframe
        main_func = exec_scope.get('main')
        if not callable(main_func):
            raise NameError("main() function not found in the generated code.")

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            output_data = main_func(sample_df.copy())
        
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        
        if not isinstance(output_data, pd.DataFrame):
            message = f"main() must return a pandas DataFrame (got {type(output_data).__name__})."
            logger.error(message)
            return {
                "execution_stdout": stdout,
                "execution_stderr": stderr,
                "execution_error": message,
                "execution_output_data": None
            }

        if output_data.empty:
            message = "Execution returned an empty DataFrame."
            logger.warning(message)
            # Graph state must stay serializable: a raw DataFrame here crashes
            # the checkpointer (msgpack) and 500s the whole request.
            return {
                "execution_stdout": stdout,
                "execution_stderr": stderr,
                "execution_error": message,
                "execution_output_data": [],
                "execution_output_preview": [],
            }

        logger.info("✅ Code executed successfully.")
        logger.info(f"STDOUT:\n{stdout}")
        if stderr:
            logger.warning(f"STDERR:\n{stderr}")

        # Provide serializable artifacts for downstream consumers.
        preview = output_data.head(2).reset_index(drop=True)
        output_records = output_data.to_dict(orient="records")
        preview_records = preview.to_dict(orient="records")

        return {
            "execution_stdout": stdout,
            "execution_stderr": stderr,
            "execution_error": None,
            "execution_output_data": output_records,
            "execution_output_preview": preview_records,
        }
    except Exception as e:
        logger.error(f"❌ Code execution failed: {e}")
        return {
            "execution_stdout": stdout_capture.getvalue(),
            "execution_stderr": stderr_capture.getvalue(),
            "execution_error": str(e),
            "execution_output_data": None
        }

def logical_semantic_validator_node(state: CodingAgentState) -> dict:
    """Uses an LLM to check if the code logically fulfills the user prompt."""
    if state.get("syntax_error") or state.get("static_semantic_error"): return {}
    logger.info("--- Checking Logic with LLM Reviewer ---")
    logger.info(f"--- CODE TO VALIDATE ---\n{state.get('generated_code')}\n")

    if validator_llm is None:
        logger.info("Skipping LLM review because GROQ_API_KEY_CODING_AGENT is not configured.")
        return {"logical_semantic_error": False, "code_validation_feedback": "APPROVED"}
    
    # strict_mode = os.getenv(STRICT_LOGIC_ENV, "0") == "1"
    # #  include output preview to reduce hallucinations
    # output_preview = state.get("execution_output_preview") or []
    # output_cols = list(output_preview[0].keys()) if output_preview else []

    strict_mode = os.getenv(STRICT_LOGIC_ENV, "0") == "1"

    # Include execution output info to reduce hallucinations (safe guards added)
    output_preview = state.get("execution_output_preview") or []
    output_cols: list[str] = []

    if (
        isinstance(output_preview, list)
        and len(output_preview) > 0
        and isinstance(output_preview[0], dict)
    ):
        output_cols = list(output_preview[0].keys())

    output_preview_text = "[]"
    try:
        # Keep it bounded and JSON formatted for the reviewer
        output_preview_text = json.dumps(output_preview[:2], indent=2)
    except Exception:
        output_preview_text = str(output_preview[:2])

    
    # prompt = f"""You are an expert Python code reviewer. 
    # Validate if the code logically and completely fulfills the user's requirements, taking into account the code's output.
    # **User Requirements:**
    # {state['user_prompt']}

    # **Code to Review:**
    # ```python
    prompt = f"""
    You are an expert Python code reviewer.

    IMPORTANT RULES:
    - Only validate EXPLICIT user requirements. Do NOT demand extra defensive handling the user did not ask for WHEN THE CODE ALREADY RAN CLEANLY.
    - BUT a run that actually crashed is always a failure: if the Exception or STDERR field below shows a real error (e.g. ZeroDivisionError, KeyError, ValueError from a real bug), set is_logically_correct=false and explain the fix.
    - If the Exception field says sample data was unavailable, the code was never executed — judge it on its own merits and do NOT count the missing sample as a failure.
    - FABRICATION IS A LOGICAL ERROR. Set is_fabricated=true AND is_logically_correct=false when the code invents a formula, magic constants, thresholds, or category/label mappings for a metric, brand, entity, or column that is NOT in the dataset, NOT defined by the user or by the planner-approved interpretation below, AND not derivable from existing columns by a standard, universally-known definition. Fabricated logic is wrong even if the code runs without error, because it emits confident but meaningless numbers.
    - NOT fabrication (never flag these): a rate computed as the mean of a 0/1 column; deciles/quantiles via qcut; hour/day-of-week/month derived from an existing timestamp column; pivot/crosstab/groupby over existing columns; descriptive bin labels for a binning the user requested; comments that restate the plan's stated interpretation; code that raises ValueError instead of computing an undefined metric (judge whether refusing was appropriate — a refusal is not an invented formula).
    - Use the execution output preview/columns to confirm claims about missing columns.

    User Requirements:
    {state.get("user_prompt", "")}

    Planner-approved interpretation (assumptions stated here are sanctioned, not fabricated):
    {state.get("plan", "")}

    Code to Review:
    ```python
    {state['generated_code']}
    ```
    **Execution Results:**
    - STDOUT:
    ```
    {state.get('execution_stdout', 'No output.')}
    ```
    - STDERR:
    ```
    {state.get('execution_stderr', 'No errors.')}
    ```
    - Exception:
    ```
    {state.get('execution_error', 'No exception.')}
    ```
    Does the code meet all requirements, avoid inventing undefined logic, and run without unexpected errors?
    Respond ONLY with a JSON object:
    {{
      "is_logically_correct": <true_or_false>,
      "is_fabricated": <true_or_false>,
      "rationale": "<Your reasoning. If false, provide clear, actionable feedback for the developer.>"
    }}
    """
    messages = [SystemMessage(content="You are a python code reviewer."), HumanMessage(content=prompt)]
    response = validator_llm.invoke(messages)
    logger.info("Validator LLM answered: %s", describe_response(response))
    logger.info(f"--- LLM CODE REVIEW RESPONSE ---\n{response}\n")
    try:
        raw_content = _coerce_message_content(response.content)
        review = _extract_json_dict(raw_content)
        if review is None:
            raise json.JSONDecodeError("Reviewer output was not valid JSON.", raw_content, 0)
        logger.info(f"--- LLM REVIEW ---\n{review}\n")
        if review.get("is_logically_correct"):
            logger.info("✅ Logic is correct.")
            return {"logical_semantic_error": False, "code_validation_feedback": "APPROVED"}
        # else:
        #     feedback = review.get("rationale", "No rationale provided.")
        #     logger.warning(f"❌ Logic is flawed: {feedback}")
        #     return {"logical_semantic_error": True, "code_validation_feedback": feedback}
        else:
            feedback = review.get("rationale", "No rationale provided.")
            logger.warning("❌ Logic reviewer flagged: %s", feedback)

            # Fabricated logic always blocks, even in default mode: invented formulas
            # / magic constants / category mappings must never be silently approved
            # and shipped as confident-but-meaningless numbers. (Runtime crashes are
            # handled by the execution feedback loop, not this pre-execution node.)
            is_fabricated = bool(review.get("is_fabricated"))

            # A fabrication block must survive a second independent judge pass:
            # the small reviewer model flakes on standard derivations (grouped
            # rates, crosstabs), and a misfire here is terminal for the user.
            # Genuine fabrication (invented mappings, magic constants) verdicts
            # are stable across calls and still confirm; flaky misfires do not.
            if is_fabricated and not strict_mode:
                try:
                    recheck_resp = validator_llm.invoke(messages)
                    recheck = _extract_json_dict(_coerce_message_content(recheck_resp.content)) or {}
                    if not bool(recheck.get("is_fabricated")):
                        logger.warning(
                            "Fabrication verdict NOT confirmed on recheck; treating as soft flag."
                        )
                        is_fabricated = False
                        if recheck.get("is_logically_correct"):
                            logger.info("✅ Recheck approved the code outright.")
                            return {"logical_semantic_error": False, "code_validation_feedback": "APPROVED"}
                except Exception:
                    logger.exception("Fabrication recheck failed; keeping original verdict.")

            # STRICT MODE, or a fabricated result: block and send feedback to coder.
            if strict_mode or is_fabricated:
                reason = "fabricated logic" if is_fabricated else "strict mode"
                logger.warning("Blocking code (%s) and returning feedback to coder.", reason)
                return {
                    "logical_semantic_error": True,
                    "code_validation_feedback": feedback,
                }

            # DEFAULT MODE (soft flags only): do NOT block pipeline; don't trigger stub fallback
            return {
                "logical_semantic_error": False,
                "code_validation_feedback": "APPROVED",
                "logical_review_feedback": feedback,  # store notes separately
            }
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.error(f"Reviewer LLM failed to return valid JSON: {exc}")
        return {
            "logical_semantic_error": False,
            "code_validation_feedback": "Reviewer response unparsable; skipping logical validation."
        }



def run_code_validation(
    state: CodingAgentState,
    *,
    run_sample_execution: bool = True,
) -> dict:
    """
    Validate code the same way the coding subgraph validates generated code:
        1) syntax check (also blocks input()/assumed-weekday filters)
        2) static-semantic AST checks (also applies numeric/whitespace/deprecated-API
           auto-fixes and returns the fixed source)
        3) optional sample run of main(df) on a tiny frame to catch runtime errors

    Returns:
        {"ok": bool, "feedback": Optional[str], "code": str, "stage": Optional[str]}
    """
    code = state.get("generated_code") or (state.get("coder_definition") or {}).get("code") or ""
    if not code.strip():
        return {"ok": False, "feedback": "No code was provided.", "code": code, "stage": "input"}

    working = dict(state)
    working["generated_code"] = code
    if not isinstance(working.get("coder_definition"), dict):
        working["coder_definition"] = {"code": code}
    working["syntax_error"] = False
    working["static_semantic_error"] = False

    syn = syntactic_validator_node(working)
    working.update(syn)
    if syn.get("syntax_error"):
        return {"ok": False, "feedback": syn.get("code_validation_feedback"),
                "code": working.get("generated_code", code), "stage": "syntax"}

    sem = static_semantic_validator_node(working)
    working.update(sem)
    if sem.get("static_semantic_error"):
        return {"ok": False, "feedback": sem.get("code_validation_feedback"),
                "code": working.get("generated_code", code), "stage": "static_semantic"}

    fixed_code = working.get("generated_code") or code

    if run_sample_execution:
        exec_res = execute_code_node(working)
        err = exec_res.get("execution_error")
        _benign = {
            "Execution returned an empty DataFrame.",
            "Sample data unavailable for execution",
            "Sample data unavailable for execution.",
            "Validation failed.",
        }
        if err and err not in _benign:
            return {"ok": False, "feedback": err, "code": fixed_code, "stage": "sample_execution"}

    return {"ok": True, "feedback": None, "code": fixed_code, "stage": None}