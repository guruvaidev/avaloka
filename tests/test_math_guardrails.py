import pytest
from app.agents.planner import _detect_math_on_string_column
from app.agents.coder import _detect_math_on_string_column_coder

PATIENT_SCHEMA = {
    "name": "String", "age": "Int64", "dob": "Date",
    "price": "Float64", "country": "String",
}


def test_keyword_match_requires_whole_word():
    # "assumption" contains "sum"; "summarize" contains "sum" — neither is a
    # math aggregation request.
    state = {"schema": PATIENT_SCHEMA}
    assert _detect_math_on_string_column(
        "state your assumption about the name column", state) is None
    assert _detect_math_on_string_column(
        "summarize the country column", state) is None


def test_column_match_requires_whole_word():
    # keyword "total" is real, but 'name' only appears inside "renamed" and
    # must not be resolved as the target column.
    state = {"schema": PATIENT_SCHEMA}
    assert _detect_math_on_string_column(
        "compute the total for each renamed group", state) is None


def test_machine_plan_prose_does_not_trip_coder_guard():
    # Post plan-channel fix the coder sees the planner's real plan; its
    # wording ("assumption", "column named ...") must not be read as
    # "sum of the name column".
    state = {
        "user_prompt": "give me 4 random rows",
        "plan": (
            "1. State your assumption about the dataset.\n"
            "2. Create a column named 'billing_category'.\n"
            "3. Sample 4 random rows and return them."
        ),
        "schema": PATIENT_SCHEMA,
    }
    assert _detect_math_on_string_column_coder(state) is None


def test_coder_guard_prefers_user_prompt_over_plan():
    state = {
        "user_prompt": "show me 4 rows",
        "plan": "calculate sum of the 'country' field",
        "schema": PATIENT_SCHEMA,
    }
    assert _detect_math_on_string_column_coder(state) is None


def test_coder_guard_still_blocks_math_on_string_prompt():
    state = {"user_prompt": "what is the mean of country", "schema": PATIENT_SCHEMA}
    result = _detect_math_on_string_column_coder(state)
    assert result is not None
    assert "'country' column" in result


def test_fabrication_extraction_template_renders(monkeypatch):
    # The JSON example in the extraction prompt must be brace-escaped or
    # LangChain raises INVALID_PROMPT_INPUT and the guard silently dies.
    import app.agents.planner as planner_mod

    calls = {}

    def fake_llm(prompt_value):
        calls["rendered"] = True

        class R:
            content = '{"field_references": ["price"]}'

        return R()

    monkeypatch.setattr(planner_mod, "llm", fake_llm)
    result = planner_mod._extract_fab_reference_candidates_with_llm(
        "sum of price", ["price", "name"], [{"price": 1.5, "name": "a"}]
    )
    assert calls.get("rendered"), "prompt template failed to render"
    assert result == ["price"]

def test_planner_guardrail_numeric_column_schema():
    """Test that mathematical operations on numeric columns (defined via schema) are allowed."""
    state = {
        "schema": {
            "age": "int",
            "name": "string"
        }
    }
    user_input = "Calculate the mean of the 'age' column"
    result = _detect_math_on_string_column(user_input, state)
    assert result is None, "Expected no error for numeric column"


def test_planner_guardrail_string_column_schema():
    """Test that mathematical operations on string columns (defined via schema) return an error."""
    state = {
        "schema": {
            "age": "int",
            "name": "string"
        }
    }
    user_input = "Calculate the average of the name column"
    result = _detect_math_on_string_column(user_input, state)
    assert result is not None, "Expected an error message"
    assert "Cannot perform mathematical operation" in result
    assert "'name' column" in result


def test_planner_guardrail_string_column_preview_data():
    """Test that mathematical operations on string columns (detected via preview data) return an error."""
    state = {
        "schema": {"weekday": "string", "value": "int"},
        "uploaded_csv_columns": ["weekday", "value"],
        "uploaded_csv_preview": [
            ["weekday", "value"],
            ["Monday", 10],
            ["Tuesday", 20],
            ["Wednesday", 30],
        ]
    }
    user_input = "What is the standard deviation of weekday?"
    result = _detect_math_on_string_column(user_input, state)
    assert result is not None, "Expected an error message"
    assert "Cannot perform mathematical operation" in result
    assert "'weekday' column" in result


def test_planner_guardrail_no_match():
    """Test that operations not targeting a specific recognized column pass through (let LLM handle it)."""
    state = {
        "schema": {
            "age": "int",
        }
    }
    user_input = "Calculate the mean of the height column" # height is not in schema
    result = _detect_math_on_string_column(user_input, state)
    assert result is None, "Expected no error because column is not found"


def test_coder_guardrail_string_column_sample_data():
    """Test that the coder wrapper correctly maps sample_data and catches errors for string columns."""
    state = {
        "plan": "calculate sum of the 'category' field",
        "sample_data": [
            {"category": "A", "val": 1},
            {"category": "B", "val": 2},
            {"category": "C", "val": 3},
        ],
        "schema": {"category": "object", "val": "int64"}
    }
    result = _detect_math_on_string_column_coder(state)
    assert result is not None, "Expected an error message"
    assert "Cannot perform mathematical operation" in result
    assert "'category' column" in result


def test_coder_guardrail_valid_operation():
    """Test that the coder wrapper allows valid operations on numeric columns."""
    state = {
        "plan": "calculate sum of the 'val' field",
        "sample_data": [
            {"category": "A", "val": 1},
            {"category": "B", "val": 2},
            {"category": "C", "val": 3},
        ],
        "schema": {"category": "object", "val": "int64"}
    }
    result = _detect_math_on_string_column_coder(state)
    assert result is None, "Expected no error for numeric column"


def test_coder_guardrail_fallback_messages():
    """Test that the coder wrapper correctly parses the plan from recent messages if 'plan' is absent."""
    class DummyMessage:
        def __init__(self, content):
            self.content = content

    state = {
        "messages": [
            DummyMessage("Some context message"),
            DummyMessage("find the median of the 'status' column")
        ],
        "schema": {"status": "string", "id": "int"}
    }
    result = _detect_math_on_string_column_coder(state)
    assert result is not None, "Expected an error message based on message parsing"
    assert "Cannot perform mathematical operation" in result
    assert "'status' column" in result
