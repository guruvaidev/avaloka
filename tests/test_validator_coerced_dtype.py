"""D2 (rc9 web check): a column the code coerces is no longer string-typed.

"What is the average age of patients who had a stroke?" on a PostgreSQL table
thread (truth 67.73) failed 4 of 5 times. The coder's code was right: it made
``stroke`` numeric with ``pd.to_numeric`` and then compared it to 1. The static
check still said "Column 'stroke' is typed as string, but it is being compared
to numeric literal 1", because (a) it ignored the coercion on the line before,
and (b) the database-table path typed every column "string" although the
query returns ``stroke`` as an integer. After two HARD blocks the deterministic
stub answered a different question: it filtered ``bmi == 'None'`` and summed
``id``, dropping the stroke filter.

Ported from 2.0 #571 (parts 1 and 2: the coercion-aware check and the typed
database schema). Part 3, the blueprint-contract refusal in the coder's stub
fallback, depends on 2.0's contract and is not part of this hotfix.
"""

from __future__ import annotations

import textwrap

import pytest

from app.agents.validator import static_semantic_validator_node

# The validator schema rc9 logged for the patients table: every column "string".
RC9_SCHEMA = {c: "string" for c in (
    "id", "gender", "age", "hypertension", "heart_disease", "ever_married", "work_type",
    "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke",
)}

# The coder's code, verbatim from rweb9 (rw9-groq-api-full.log, 01:21:29).
RC9_CODE = '''import pandas as pd

def main(df):
    # Inspect data structure
    print("Columns:", df.columns.tolist())
    print("Dtypes:\\n", df.dtypes)

    # Cast relevant columns to numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['stroke'] = pd.to_numeric(df['stroke'], errors='coerce')

    # Filter for stroke patients (stroke == 1)
    df_stroke = df[df['stroke'] == 1]

    # Exclude rows with missing ages and compute average age
    ages = df_stroke['age'].dropna()
    avg_age = ages.mean()

    # Report the scalar result
    avaloka_result(value=avg_age, kind="mean", columns=["age", "stroke"])

    # Return a one‑row DataFrame with the metric
    return pd.DataFrame({'avg_age': [avg_age]})

if __name__ == "__main__":
    df = read_csv_best_effort("input.csv")
    df = main(df)
    df.to_csv("output.csv", index=False)
'''

# The second variant rc9 blocked (01:24:20): a combined filter after the coercion.
RC9_CODE_COMBINED = '''import pandas as pd

def main(df):
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['stroke'] = pd.to_numeric(df['stroke'], errors='coerce')
    df_stroke = df[(df['stroke'] == 1) & (df['age'].notna())]
    avg_age = df_stroke['age'].mean()
    return pd.DataFrame({'avg_age': [avg_age]})
'''


def _check(code: str, schema=None) -> dict:
    return static_semantic_validator_node({
        "generated_code": textwrap.dedent(code),
        "schema": dict(schema or RC9_SCHEMA),
        "user_prompt": "What is the average age of patients who had a stroke?",
    })


def _flagged(out: dict) -> bool:
    return bool(out.get("static_semantic_error")) and "compared to numeric literal" in (
        out.get("code_validation_feedback") or "")


def test_rc9_coders_exact_code_passes():
    out = _check(RC9_CODE)
    assert out.get("static_semantic_error") is False, out.get("code_validation_feedback")


def test_rc9_combined_filter_passes():
    out = _check(RC9_CODE_COMBINED)
    assert out.get("static_semantic_error") is False, out.get("code_validation_feedback")


@pytest.mark.parametrize("coercion", [
    "df['stroke'] = pd.to_numeric(df['stroke'], errors='coerce')",
    "df['stroke'] = pd.to_numeric(df['stroke'], errors='coerce').fillna(0)",
    "df['stroke'] = df['stroke'].astype(int)",
    "df['stroke'] = df['stroke'].astype(float)",
    "df['stroke'] = df['stroke'].astype('Int64')",
    "df['stroke'] = df['stroke'].astype(np.int64)",
    "df.loc[:, 'stroke'] = pd.to_numeric(df['stroke'])",
    "df[['age', 'stroke']] = df[['age', 'stroke']].apply(pd.to_numeric, errors='coerce')",
    "df = df.assign(stroke=pd.to_numeric(df['stroke'], errors='coerce'))",
    "df = df.assign(stroke=lambda d: d['stroke'].astype(int))",
    "df = df.astype({'stroke': int})",
])
def test_a_column_coerced_earlier_is_not_string_typed(coercion):
    out = _check(f"""
        import numpy as np
        import pandas as pd
        def main(df):
            {coercion}
            return df[df['stroke'] == 1]
    """)
    assert out.get("static_semantic_error") is False, (coercion, out.get("code_validation_feedback"))


@pytest.mark.parametrize("compare", [
    "s = pd.to_numeric(df['stroke'], errors='coerce')\n    return df[s == 1]",
    "return df[pd.to_numeric(df['stroke'], errors='coerce') == 1]",
])
def test_a_coerced_local_or_inline_coercion_passes(compare):
    out = _check("import pandas as pd\ndef main(df):\n    " + compare + "\n")
    assert out.get("static_semantic_error") is False, out.get("code_validation_feedback")


# ---- true positives stay flagged ----

def test_a_raw_comparison_with_no_coercion_is_still_flagged():
    out = _check("""
        import pandas as pd
        def main(df):
            return df[df['stroke'] == 1]
    """)
    assert _flagged(out), out


def test_a_comparison_before_the_coercion_is_still_flagged():
    out = _check("""
        import pandas as pd
        def main(df):
            first = df[df['stroke'] == 1]
            df['stroke'] = pd.to_numeric(df['stroke'], errors='coerce')
            return first
    """)
    assert _flagged(out), out


@pytest.mark.parametrize("not_numeric", [
    "df['stroke'] = df['stroke'].astype(str)",
    "df['stroke'] = df['stroke'].str.strip()",
    "df['age'] = pd.to_numeric(df['age'], errors='coerce')",   # another column
])
def test_a_non_numeric_reassignment_does_not_clear_the_flag(not_numeric):
    out = _check(f"""
        import pandas as pd
        def main(df):
            {not_numeric}
            return df[df['stroke'] == 1]
    """)
    assert _flagged(out), (not_numeric, out)


# ---- D2.2: the database-table path types columns from the values it read ----

def test_db_rows_are_typed_from_their_values():
    from app.api.helpers import _schema_from_db_rows
    rows = [
        {"id": 9046, "age": 67.0, "stroke": 1, "bmi": 36.6, "gender": "Male", "flag": True, "empty": None},
        {"id": 51676, "age": 61, "stroke": 0, "bmi": None, "gender": "Female", "flag": False, "empty": None},
    ]
    assert _schema_from_db_rows(list(rows[0]), rows) == {
        "id": "int64", "age": "float64", "stroke": "int64", "bmi": "float64",
        "gender": "string", "flag": "bool", "empty": "string",
    }


def test_db_typed_schema_lets_the_raw_comparison_through():
    # With the table's real types the plain comparison is correct as written.
    from app.api.helpers import _schema_from_db_rows
    schema = _schema_from_db_rows(["stroke", "age"], [{"stroke": 1, "age": 67.0}])
    out = _check("""
        import pandas as pd
        def main(df):
            return df[df['stroke'] == 1]['age'].mean()
    """, schema=schema)
    assert out.get("static_semantic_error") is False, out.get("code_validation_feedback")


def test_db_ddl_maps_pandas_dtypes():
    from app.api.helpers import _ddl_from_schema
    ddl = _ddl_from_schema("t", {"stroke": "int64", "age": "float64", "g": "string"})
    assert '"stroke" INTEGER' in ddl and '"age" DOUBLE' in ddl and '"g" TEXT' in ddl, ddl
