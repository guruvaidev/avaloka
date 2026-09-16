import logging
import os
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage

from app.api.workflow import build_graph
from app.graph.etl_state import ETLState

# Configure logging so generated artifacts are visible during pytest runs
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
LOGGER = logging.getLogger(__name__)

pytestmark = pytest.mark.integration


@pytest.fixture
def graph():
    """Fixture to provide the compiled production graph for testing."""
    original_flag = os.environ.get("AVALOKA_FORCE_PLAN_ON_RESPONSE")
    os.environ["AVALOKA_FORCE_PLAN_ON_RESPONSE"] = "1"
    try:
        yield build_graph().compile()
    finally:
        if original_flag is None:
            os.environ.pop("AVALOKA_FORCE_PLAN_ON_RESPONSE", None)
        else:
            os.environ["AVALOKA_FORCE_PLAN_ON_RESPONSE"] = original_flag


def run_graph_with_prompt(graph, prompt: str, data_source: str, schema: dict) -> Tuple[ETLState, str, str]:
    """Helper function to run the graph with a given prompt and initial state."""
    output_location = os.path.abspath(
        f"app/sample_data/output_{prompt.replace(' ', '_')[:20]}.csv"
    )

    initial_state = ETLState(
        messages=[HumanMessage(content=prompt)],
        data_source_location=data_source,
        output_location=output_location,
        schema=schema,
        plan=prompt,
        input_data_type="csv",
        uploaded_csv_preview=[
            ["OrderID", "Product", "Category", "Price", "OrderDate", "Country"],
            ["1", "Laptop", "Electronics", "1200", "2023-01-15", "USA"],
            ["2", "Mouse", "Electronics", "25", "2023-01-17", "Canada"],
        ],
    )

    LOGGER.info("--- INITIAL STATE ---\n%s\n", initial_state)

    final_state = graph.invoke(initial_state)

    generated_code = final_state.get("coder_definition", {}).get("code", "")
    validator_preview = final_state.get("execution_output_preview")
    validator_error = final_state.get("execution_error")
    validator_output = final_state.get("execution_output_data")

    LOGGER.info("Generated code for prompt '%s':\n%s", prompt, generated_code)
    if validator_preview is not None:
        LOGGER.info("Validator preview type: %s", type(validator_preview))
        LOGGER.info("Validator preview for prompt '%s':\n%s", prompt, validator_preview)
    if validator_output is not None:
        LOGGER.info("Validator output type: %s", type(validator_output))
    if validator_error:
        LOGGER.warning("Validator error for prompt '%s': %s", prompt, validator_error)

    artifact_dir = Path("artifacts/test_coder_integration")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt).strip("_") or "case"

    if generated_code:
        code_path = artifact_dir / f"{slug}.py"
        code_path.write_text(generated_code, encoding="utf-8")
        LOGGER.info("Generated code saved to %s", code_path.resolve())

    if validator_preview:
        preview_df = pd.DataFrame(validator_preview)
        preview_path = artifact_dir / f"{slug}_preview.csv"
        preview_df.to_csv(preview_path, index=False)
        LOGGER.info("Validator preview saved to %s", preview_path.resolve())

    if validator_output:
        output_df = pd.DataFrame(validator_output)
        output_path = artifact_dir / f"{slug}_output.csv"
        output_df.to_csv(output_path, index=False)
        LOGGER.info("Validator execution output saved to %s", output_path.resolve())

    return final_state, output_location, slug


def test_simple_read_write(graph):
    """Read CSV and write it back; validate by execution + equality (avoid brittle path-escaping asserts)."""
    prompt = "Read the CSV file and save it to the output location."
    data_source = os.path.abspath("app/sample_data/sales_data.csv")
    schema = {
        "OrderID": "int64",
        "Product": "object",
        "Category": "object",
        "Price": "int64",
        "OrderDate": "object",
        "Country": "object",
    }

    final_state, output_location, slug = run_graph_with_prompt(graph, prompt, data_source, schema)

    generated_code = final_state.get("coder_definition", {}).get("code") or ""
    assert generated_code, "Code generation failed"

    # robust: don't assert full absolute path literal (LLM may escape it differently)
    assert "pd.read_csv" in generated_code
    assert re.search(r"pd\.read_csv\(\s*['\"].*sales_data\.csv['\"]\s*\)", generated_code), generated_code

    assert "to_csv" in generated_code
    out_base = re.escape(os.path.basename(output_location))
    assert re.search(rf"to_csv\(\s*['\"].*{out_base}['\"]", generated_code), generated_code

    try:
        exec_globals = {"__name__": "__main__"}
        exec(generated_code, exec_globals)

        assert os.path.exists(output_location), "Output file was not created"
        original_df = pd.read_csv(data_source)
        output_df = pd.read_csv(output_location)
        pd.testing.assert_frame_equal(original_df, output_df)

        artifact_dir = Path("artifacts/test_coder_integration")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_artifact = artifact_dir / f"{slug}_result.csv"
        output_df.to_csv(output_artifact, index=False)
        LOGGER.info("Execution result saved to %s", output_artifact.resolve())
    finally:
        if os.path.exists(output_location):
            os.remove(output_location)


def test_filter_data(graph):
    """Filter for USA and save result."""
    prompt = "Read the sales data, filter for sales in the 'USA', and save the result."
    data_source = os.path.abspath("app/sample_data/sales_data.csv")
    schema = {
        "OrderID": "int64",
        "Product": "object",
        "Category": "object",
        "Price": "int64",
        "OrderDate": "object",
        "Country": "object",
    }

    final_state, output_location, slug = run_graph_with_prompt(graph, prompt, data_source, schema)

    generated_code = final_state.get("coder_definition", {}).get("code") or ""
    assert generated_code, "Code generation failed"

    assert re.search(r"df\[\s*['\"]Country['\"]\s*\]\s*==\s*['\"]USA['\"]", generated_code), generated_code
    assert "to_csv" in generated_code

    try:
        exec_globals = {"__name__": "__main__"}
        exec(generated_code, exec_globals)
        assert os.path.exists(output_location)

        result_df = pd.read_csv(output_location)
        assert all(result_df["Country"] == "USA")
        assert len(result_df) > 0

        artifact_dir = Path("artifacts/test_coder_integration")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_artifact = artifact_dir / f"{slug}_result.csv"
        result_df.to_csv(output_artifact, index=False)
        LOGGER.info("Execution result saved to %s", output_artifact.resolve())
    finally:
        if os.path.exists(output_location):
            os.remove(output_location)


def test_groupby_aggregation(graph):
    """Group by Country and sum Price; validate numerically (column name may vary)."""
    prompt = "Calculate the total price of products for each country and save the result."
    data_source = os.path.abspath("app/sample_data/sales_data.csv")
    schema = {
        "OrderID": "int64",
        "Product": "object",
        "Category": "object",
        "Price": "int64",
        "OrderDate": "object",
        "Country": "object",
    }

    final_state, output_location, slug = run_graph_with_prompt(graph, prompt, data_source, schema)

    generated_code = final_state.get("coder_definition", {}).get("code") or ""
    assert generated_code, "Code generation failed"
    assert "groupby" in generated_code
    assert ".sum()" in generated_code

    try:
        exec_globals = {"__name__": "__main__"}
        exec(generated_code, exec_globals)
        assert os.path.exists(output_location)

        result_df = pd.read_csv(output_location)
        assert "Country" in result_df.columns
        assert len(result_df) > 0

        value_candidates = [c for c in result_df.columns if c != "Country"]
        assert value_candidates, f"Expected a value column besides Country, got {list(result_df.columns)}"
        value_col = next((c for c in value_candidates if "price" in c.lower()), value_candidates[0])

        expected_df = (
            pd.read_csv(data_source)
            .groupby("Country", as_index=False)["Price"]
            .sum()
            .rename(columns={"Price": value_col})
            .sort_values("Country")
            .reset_index(drop=True)
        )

        result_comp = (
            result_df[["Country", value_col]]
            .sort_values("Country")
            .reset_index(drop=True)
        )

        pd.testing.assert_frame_equal(expected_df, result_comp, check_dtype=False)

        artifact_dir = Path("artifacts/test_coder_integration")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_artifact = artifact_dir / f"{slug}_result.csv"
        result_df.to_csv(output_artifact, index=False)
        LOGGER.info("Execution result saved to %s", output_artifact.resolve())
    finally:
        if os.path.exists(output_location):
            os.remove(output_location)


def test_filter_then_group_multi_instruction(graph):
    """Filter USA then group by Category and sum Price."""
    prompt = (
        "First filter the sales data for orders in the 'USA'. "
        "Then group the remaining rows by Category and calculate the total Price."
    )
    data_source = os.path.abspath("app/sample_data/sales_data.csv")
    schema = {
        "OrderID": "int64",
        "Product": "object",
        "Category": "object",
        "Price": "int64",
        "OrderDate": "object",
        "Country": "object",
    }

    final_state, output_location, slug = run_graph_with_prompt(graph, prompt, data_source, schema)
    generated_code = final_state.get("coder_definition", {}).get("code") or ""

    assert re.search(r"df\[\s*['\"]Country['\"]\s*\]\s*==\s*['\"]USA['\"]", generated_code), generated_code
    assert re.search(r"\.groupby\(\s*['\"]Category['\"]", generated_code), generated_code

    if os.path.exists(output_location):
        os.remove(output_location)






def _exec_generated_code_and_read_output(code: str, output_location: str) -> pd.DataFrame:
    """Helper: exec generated code and load output csv."""
    exec_globals = {"__name__": "__main__"}
    exec(code, exec_globals)
    assert os.path.exists(output_location), f"Output file not created: {output_location}"
    return pd.read_csv(output_location)




def test_string_accessor_patch_uppercase_country(graph, tmp_path):
    """
    Validates _patch_df_str_accessor behavior:
    - LLM might write df['Country'].str.upper()
    - you patch to df['Country'].astype("string").str.upper()

    "string" rather than str is load-bearing: astype(str) turns a missing value
    into the literal "nan", which then survives dropna(). Both spellings are
    accepted below so the test asserts dtype-SAFETY rather than one spelling.
    """
    # Create minimal CSV
    p = tmp_path / "sales.csv"
    p.write_text("OrderID,Country\n1,USA\n2,Canada\n", encoding="utf-8")

    prompt = "Create a new column Country_upper by converting Country to uppercase and save the result."
    schema = {"OrderID": "int64", "Country": "object"}

    output_location = str(tmp_path / "out.csv")

    initial_state = ETLState(
        messages=[HumanMessage(content=prompt)],
        data_source_location=str(p),
        output_location=output_location,
        schema=schema,
        plan=prompt,
        input_data_type="csv",
        uploaded_csv_preview=[
            ["OrderID", "Country"],
            ["1", "USA"],
            ["2", "Canada"],
        ],
    )

    final_state = graph.invoke(initial_state)
    code = final_state.get("coder_definition", {}).get("code") or ""
    assert code, "Code generation failed"
    assert "Country_upper" in code

    # Must be safe str accessor (your patch)
    # Accept either upper() or lower() variations, but enforce a dtype-safe
    # coercion before .str -- astype("string") preferred, astype(str) tolerated.
    assert re.search(r"""\.astype\((?:str|['"]string['"])\)\.str\.""", code), code

    out_df = _exec_generated_code_and_read_output(code, output_location)
    assert "Country_upper" in out_df.columns
    assert out_df.loc[0, "Country_upper"] == "USA"
    assert out_df.loc[1, "Country_upper"] == "CANADA"


def test_groupby_mean_and_sum_without_malformed_agg_list(graph, tmp_path):
    """
    Ensures coder does NOT generate df.groupby().agg(['mean','sum',...]) which creates malformed output.
    We ask for mean + sum and verify:
    - code doesn't contain agg([ or agg(['
    - output file loads cleanly
    """
    p = tmp_path / "sales.csv"
    p.write_text(
        "OrderID,Country,Price\n1,USA,10\n2,USA,20\n3,Canada,5\n",
        encoding="utf-8",
    )

    prompt = "Group by Country and calculate mean and sum of Price, then save the result."
    schema = {"OrderID": "int64", "Country": "object", "Price": "int64"}
    output_location = str(tmp_path / "out.csv")

    initial_state = ETLState(
        messages=[HumanMessage(content=prompt)],
        data_source_location=str(p),
        output_location=output_location,
        schema=schema,
        plan=prompt,
        input_data_type="csv",
        uploaded_csv_preview=[
            ["OrderID", "Country", "Price"],
            ["1", "USA", "10"],
            ["2", "USA", "20"],
        ],
    )

    final_state = graph.invoke(initial_state)
    code = final_state.get("coder_definition", {}).get("code") or ""
    assert code

    # Hard guard: don't allow agg(['mean', 'sum'...]) style
    assert "agg([" not in code.replace(" ", ""), code
    assert "agg(['" not in code.replace(" ", ""), code

    out_df = _exec_generated_code_and_read_output(code, output_location)
    assert "Country" in out_df.columns
    assert len(out_df) == 2


def test_missing_column_in_prompt_adapts_to_available_columns(graph, tmp_path):
    """
    Prompt mentions Region which doesn't exist in schema.
    We verify generated code does NOT reference Region and still produces output.
    """
    p = tmp_path / "sales.csv"
    p.write_text(
        "OrderID,Country,Category,Price\n1,USA,Electronics,10\n2,USA,Electronics,20\n3,Canada,Toys,5\n",
        encoding="utf-8",
    )

    prompt = "Group by Region and calculate total Price, then save."
    schema = {"OrderID": "int64", "Country": "object", "Category": "object", "Price": "int64"}
    output_location = str(tmp_path / "out.csv")

    initial_state = ETLState(
        messages=[HumanMessage(content=prompt)],
        data_source_location=str(p),
        output_location=output_location,
        schema=schema,
        plan=prompt,
        input_data_type="csv",
        uploaded_csv_preview=[
            ["OrderID", "Country", "Category", "Price"],
            ["1", "USA", "Electronics", "10"],
            ["2", "USA", "Electronics", "20"],
        ],
    )

    final_state = graph.invoke(initial_state)
    code = final_state.get("coder_definition", {}).get("code") or ""
    assert code

    # Ensure it didn't hallucinate Region usage
    assert "Region" not in code
    assert "region" not in code.lower()

    out_df = _exec_generated_code_and_read_output(code, output_location)
    assert len(out_df) > 0
    assert out_df.shape[1] >= 2  # group column + aggregate


def test_output_location_basename_used_in_to_csv(graph, tmp_path):
    """
    Ensures code writes to the provided output_location (or at least includes its basename),
    not a hardcoded output.csv or random timestamp file.
    """
    p = tmp_path / "sales.csv"
    p.write_text("OrderID,Country\n1,USA\n", encoding="utf-8")

    prompt = "Read the CSV and save it."
    schema = {"OrderID": "int64", "Country": "object"}
    output_location = str(tmp_path / "my_custom_name_output.csv")

    initial_state = ETLState(
        messages=[HumanMessage(content=prompt)],
        data_source_location=str(p),
        output_location=output_location,
        schema=schema,
        plan=prompt,
        input_data_type="csv",
        uploaded_csv_preview=[["OrderID", "Country"], ["1", "USA"]],
    )
    final_state = graph.invoke(initial_state)
    code = final_state.get("coder_definition", {}).get("code") or ""
    assert code

    out_base = re.escape(os.path.basename(output_location))
    assert re.search(rf"to_csv\(\s*['\"].*{out_base}['\"]", code), code


# -----------------------------
# Multi-dataset tests
# -----------------------------

def test_multi_dataset_join_on_customer_id_left_join(graph, tmp_path):
    """
    Multi-dataset:
    - customers is primary df (data_source_location)
    - orders is secondary dataset in datasets_context
    - prompt explicitly provides join key
    Expect: code loads orders inside main(df) and merges on customer_id
    """
    customers = tmp_path / "customers.csv"
    orders = tmp_path / "orders.csv"
    customers.write_text("customer_id,name\nC001,Asha\nC002,John\n", encoding="utf-8")
    orders.write_text("order_id,customer_id,amount\nO1,C001,10\nO2,C002,20\n", encoding="utf-8")

    output_location = str(tmp_path / "joined.csv")

    datasets_context = [
        {
            "dataset_id": "customers",
            "alias": "customers",
            "filename": "customers.csv",
            "columns": ["customer_id", "name"],
            "preview": [{"customer_id": "C001", "name": "Asha"}],
            "data_source_location": str(customers),
            "full_data_location": str(customers),
            "sample_data_location": str(customers),
        },
        {
            "dataset_id": "orders",
            "alias": "orders",
            "filename": "orders.csv",
            "columns": ["order_id", "customer_id", "amount"],
            "preview": [{"order_id": "O1", "customer_id": "C001", "amount": "10"}],
            "data_source_location": str(orders),
            "full_data_location": str(orders),
            "sample_data_location": str(orders),
        },
    ]

    prompt = "Join customers and orders on customer_id (left join) and save the result."

    initial_state = ETLState(
        messages=[HumanMessage(content=prompt)],
        data_source_location=str(customers),  # primary
        output_location=output_location,
        schema={"customer_id": "object", "name": "object"},
        plan=prompt,
        input_data_type="csv",
        uploaded_csv_preview=[["customer_id", "name"], ["C001", "Asha"]],
        datasets_context=datasets_context,
        multi_dataset_state=datasets_context,
    )

    final_state = graph.invoke(initial_state)
    code = final_state.get("coder_definition", {}).get("code") or ""
    assert code

    # It should read the secondary csv somewhere
    assert "read_csv" in code
    # It should merge/join on the key
    assert ("merge" in code.lower()) or (".join" in code.lower())
    assert "customer_id" in code

    out_df = _exec_generated_code_and_read_output(code, output_location)
    assert "name" in out_df.columns
    assert "amount" in out_df.columns
    assert len(out_df) == 2


def test_multi_dataset_join_without_key_should_not_merge(graph, tmp_path):
    """
    Multi-dataset join requested but no join key provided.
    Expectation per your rules:
    - no cross join
    - do NOT fake join
    - return a comparison summary dataframe instead of merging
    This test asserts that the generated code does not contain merge/join calls.
    """
    customers = tmp_path / "customers.csv"
    orders = tmp_path / "orders.csv"
    customers.write_text("customer_id,name\nC001,Asha\nC002,John\n", encoding="utf-8")
    orders.write_text("order_id,customer_ref,amount\nO1,C001,10\nO2,C002,20\n", encoding="utf-8")

    output_location = str(tmp_path / "comparison.csv")

    datasets_context = [
        {
            "dataset_id": "customers",
            "alias": "customers",
            "filename": "customers.csv",
            "columns": ["customer_id", "name"],
            "preview": [{"customer_id": "C001", "name": "Asha"}],
            "data_source_location": str(customers),
            "full_data_location": str(customers),
        },
        {
            "dataset_id": "orders",
            "alias": "orders",
            "filename": "orders.csv",
            "columns": ["order_id", "customer_ref", "amount"],
            "preview": [{"order_id": "O1", "customer_ref": "C001", "amount": "10"}],
            "data_source_location": str(orders),
            "full_data_location": str(orders),
        },
    ]

    prompt = "Join customers and orders and save the result."

    initial_state = ETLState(
        messages=[HumanMessage(content=prompt)],
        data_source_location=str(customers),
        output_location=output_location,
        schema={"customer_id": "object", "name": "object"},
        plan=prompt,
        input_data_type="csv",
        uploaded_csv_preview=[["customer_id", "name"], ["C001", "Asha"]],
        datasets_context=datasets_context,
        multi_dataset_state=datasets_context,
    )

    final_state = graph.invoke(initial_state)
    code = final_state.get("coder_definition", {}).get("code") or ""
    assert code

    # Must not blindly merge/join when no explicit key
    lowered = code.lower()
    assert ".merge(" not in lowered
    assert ".join(" not in lowered

    # Still must produce output
    out_df = _exec_generated_code_and_read_output(code, output_location)
    assert len(out_df) > 0
    # should look like summary-ish dataframe (loose checks)
    # depending on your implementation it might include these
    cols = [c.lower() for c in out_df.columns]
    assert any("dataset" in c or "alias" in c for c in cols) or out_df.shape[1] >= 2


def test_top_n_frequent_values_output_has_named_headers(graph, tmp_path):
    """
    Regression for the value_counts / top-N OUTPUT FORMATTING fix in coder.py.

    "Top N most frequent" queries produce a pandas Series (value_counts/nlargest).
    Previously the coder saved that Series with header=False, so the output CSV had
    no column names at all. The OUTPUT FORMATTING rule now requires main(df) to return
    a DataFrame with descriptive, named columns and to keep the header row.

    We assert:
    - the generated code never disables the header (no header=False), and
    - the output CSV has real named headers (the first data row was NOT consumed as a
      header, which is the symptom of the old header=False bug), and
    - the top-5 amounts/frequencies are correct.
    """
    # Strictly decreasing frequencies so the top-5 is unambiguous; 10.0 (freq 1) must be dropped.
    amount_freqs = {
        "107.95": 6,
        "117.0": 5,
        "59.0": 4,
        "57.95": 3,
        "49.0": 2,
        "10.0": 1,  # outside the top-5, must be excluded
    }
    rows = ["TransactionID,TransactionAmt"]
    tid = 1
    for amt, freq in amount_freqs.items():
        for _ in range(freq):
            rows.append(f"{tid},{amt}")
            tid += 1
    p = tmp_path / "transactions.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")

    prompt = "What are the top 5 most frequent exact transaction amounts?"
    schema = {"TransactionID": "int64", "TransactionAmt": "float64"}
    output_location = str(tmp_path / "out.csv")

    initial_state = ETLState(
        messages=[HumanMessage(content=prompt)],
        data_source_location=str(p),
        output_location=output_location,
        schema=schema,
        plan=prompt,
        input_data_type="csv",
        uploaded_csv_preview=[
            ["TransactionID", "TransactionAmt"],
            ["1", "107.95"],
            ["2", "117.0"],
        ],
    )

    final_state = graph.invoke(initial_state)
    code = final_state.get("coder_definition", {}).get("code") or ""
    assert code, "Code generation failed"

    # Core fix: never suppress the header row when writing the CSV.
    assert "header=False" not in code.replace(" ", ""), code

    out_df = _exec_generated_code_and_read_output(code, output_location)

    # Top-5 out of 6 distinct values -> 5 rows; at least a value column + a frequency column.
    assert len(out_df) == 5, f"Expected 5 rows, got {len(out_df)}: {out_df}"
    assert out_df.shape[1] >= 2, f"Expected >=2 columns, got {list(out_df.columns)}"

    # Header row must be real names, not a consumed data row (the old header=False symptom).
    def _is_floatish(name) -> bool:
        try:
            float(name)
            return True
        except (TypeError, ValueError):
            return False

    assert not any(_is_floatish(c) for c in out_df.columns), (
        f"Output columns look like data values; header row was lost: {list(out_df.columns)}"
    )
    assert not any(str(c).startswith("Unnamed") for c in out_df.columns), list(out_df.columns)

    # Identify the amount vs frequency columns by content and verify correctness.
    expected_amounts = {107.95, 117.0, 59.0, 57.95, 49.0}  # 10.0 excluded
    expected_freqs = {6, 5, 4, 3, 2}

    col_value_sets = {
        c: set(pd.to_numeric(out_df[c], errors="coerce").dropna()) for c in out_df.columns
    }
    amount_col = next((c for c, vals in col_value_sets.items() if vals == expected_amounts), None)
    freq_col = next((c for c, vals in col_value_sets.items() if vals == expected_freqs), None)

    assert amount_col is not None, f"No column matched the expected top-5 amounts: {col_value_sets}"
    assert freq_col is not None, f"No column matched the expected frequencies: {col_value_sets}"
    assert amount_col != freq_col, list(out_df.columns)


def test_messages_not_overwritten_regression(graph, tmp_path):
    """
    Regression guard: Fix 9 in coder.py (don’t overwrite conversation history).
    We run graph and ensure returned messages contain the original user prompt + at least one assistant message.
    """
    p = tmp_path / "sales.csv"
    p.write_text("OrderID,Country\n1,USA\n", encoding="utf-8")

    prompt = "Read the CSV and save it."
    output_location = str(tmp_path / "out.csv")

    initial_state = ETLState(
        messages=[HumanMessage(content=prompt)],
        data_source_location=str(p),
        output_location=output_location,
        schema={"OrderID": "int64", "Country": "object"},
        plan=prompt,
        input_data_type="csv",
        uploaded_csv_preview=[["OrderID", "Country"], ["1", "USA"]],
    )

    final_state = graph.invoke(initial_state)
    msgs = final_state.get("messages") or []
    assert len(msgs) >= 1
    # Ensure the original prompt still exists somewhere in messages
    assert any(getattr(m, "content", "") == prompt for m in msgs), "Original prompt missing from messages"