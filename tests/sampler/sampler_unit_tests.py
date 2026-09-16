from pathlib import Path
import os, sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import pytest

#TEST_DATA_DIR = project_root / "avaloka-agentic-workflow" / "app" / "sample_data"
TEST_DATA_DIR = project_root / "app" / "sample_data"


# ===========================================================
#  UNIT TESTS – Focused on isolated CSV parsing and logic
# ===========================================================

@pytest.mark.parametrize(
    "filename, transform_fn, expected_rows_len",
    [
        ("TC-CSV-016_derived_columns.csv", lambda df: df.assign(Total=df["Price"] * df["Quantity"]), 5),
        ("TC-CSV-017_drop_columns.csv", lambda df: df.drop(columns=["Email"]), 3),
        ("TC-CSV-018_duplicates.csv", lambda df: df.drop_duplicates(), 3),
    ]
)
def test_transformations_before_sampling(filename, transform_fn, expected_rows_len):
    """Unit Test: Checks if transformations produce correct sample sizes."""
    path = TEST_DATA_DIR / filename
    df = pd.read_csv(path)
    df = transform_fn(df)
    if df.empty:
        pytest.skip("Transformed dataframe is empty")
    df_sampled = df.sample(n=min(len(df), 5), random_state=42)
    assert len(df_sampled) == expected_rows_len


@pytest.mark.parametrize(
    "filename, transform_fn, expected_columns, expected_rows, expected_error",
    [
        ("TC-CSV-019_inconsistent_rows.csv", None, None, None, "Failed to sample"),
        ("TC-CSV-020_whitespace_encoded.csv",
         lambda df: df.rename(columns=lambda x: x.strip()).map(lambda x: x.strip() if isinstance(x, str) else x),
         3, 4, None),
        ("TC-CSV-021_wide_columns.csv", None, 120, 10, None),
    ]
)
def test_extended_csv_cases(filename, transform_fn, expected_columns, expected_rows, expected_error):
    """Unit Test: Edge cases for CSV parsing and schema validation."""
    path = TEST_DATA_DIR / filename
    try:
        df = pd.read_csv(path)
        if transform_fn:
            df = transform_fn(df)
        assert expected_error is None, "Expected error but read CSV successfully"
        if expected_columns:
            assert len(df.columns) == expected_columns
        if expected_rows:
            assert len(df) == expected_rows
    except Exception as e:
        assert expected_error is not None and expected_error in str(e)


@pytest.mark.parametrize(
    "filename, transform_fn, expected_columns, expected_rows, expected_error",
    [
        ("TC-CSV-022_summary_stats.csv", None, 4, 5, None),
        ("TC-CSV-023_multiline_header.csv", None, 4, 3, None),
        ("TC-CSV-024_mixed_types_column.csv", None, 2, 5, None),
    ]
)
def test_additional_csv_edge_cases(filename, transform_fn, expected_columns, expected_rows, expected_error):
    """Unit Test: Tests multi-line headers and mixed column types."""
    path = TEST_DATA_DIR / filename
    try:
        header_row = 3 if "multiline_header" in filename else 0
        df = pd.read_csv(path, header=header_row)
        if transform_fn:
            df = transform_fn(df)
        assert expected_error is None
        assert len(df.columns) == expected_columns
        assert len(df) == expected_rows
    except Exception as e:
        assert expected_error is not None and expected_error in str(e)


@pytest.mark.parametrize(
    "filename, transform_fn, expected_columns, expected_rows, expected_error",
    [
        ("TC-CSV-025_summary_group_by.csv",
         lambda df: df.groupby("Region").agg({"Sales": "sum"}).reset_index(), 2, 3, None),
        ("TC-CSV-026_chained_operations.csv",
         lambda df: df.rename(columns=lambda x: x.strip().lower()).pipe(
             lambda df: df[df["quantity"] > 1]
             .assign(total=df["quantity"] * df["price"])
             .drop(columns=["price", "dropcol"])
         ), 4, 4, None),
        ("TC-CSV-027_utf8_bom_headers.csv",
         lambda df: df.rename(columns=lambda x: x.strip("\ufeff")), 3, 3, None),
        ("TC-CSV-028_mixed_encoding.csv", None, 2, 5, None),
        ("TC-CSV-029_numeric_string_mix.csv", None, 2, 5, None),
    ]
)
def test_extra_csv_edge_cases(filename, transform_fn, expected_columns, expected_rows, expected_error):
    """Unit Test: Covers BOM headers, encoding issues, and mixed types."""
    path = TEST_DATA_DIR / filename
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="python")
        if transform_fn:
            df = transform_fn(df)
        assert expected_error is None
        assert len(df.columns) == expected_columns
        assert len(df) == expected_rows
    except Exception as e:
        assert expected_error is not None and expected_error in str(e)
