"""A database write happens through the transfer path, or not at all.

rc10 web check, D1 e_expiry: the user cancelled a transfer, asked "What is the
stroke rate by gender?", then said "yes" twice. The planner took the second yes
as a job, and the analysis coder -- fed the thread history and the memory
plane's "Transfer patients table to e2ewhgroq as d1_e" hint -- wrote a
blueprint that opened a connection to the warehouse and appended the table
with ``to_sql``. Nothing was written only because the model then chose to
raise "...not installed" instead (the drivers are installed). No confirmation
had been shown.

Each layer is tested on its own here: the validator rule (HARD), the coder's
blueprint guard, and the shared prose check the planner uses. The planner's
own guard and the end-to-end turn sequence are driven against the SQLite world
in tests/test_dta_transfer_integrity.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

import app.agents.coder as coder
from app.agents.db_write_guard import DB_WRITE_REFUSAL, DB_WRITE_RULE, describes_database_write
from app.agents.validator import static_semantic_validator_node

# Verbatim from rweb10/logs/groq-phase/rw10-groq-api-full.log (rc10 web check, D1 e_expiry).

# The coder's pseudocode blueprint for the second "yes" (log lines 8539-8590).
RC10_BLUEPRINT = r"""**Pseudocode steps**

1. **Open source connection** – Create a PostgreSQL connection object (`src_conn`) to the database that contains the `patients` table.  
2. **Read full table into a DataFrame** – Execute `SELECT * FROM patients` via `pd.read_sql` (or equivalent) using `src_conn`; store the result in `df_patients`.  
3. **Validate row count** – Assert that `len(df_patients) == 5110`; raise an error if the count differs.  
4. **Open target connection** – Create a PostgreSQL connection object (`tgt_conn`) to the target database `e2ewhgroq`.  
5. **Create target table with alias** – Using the schema of `df_patients`, issue a `CREATE TABLE d1_new ( … )` statement on `tgt_conn` that matches each column name and dtype exactly (e.g., `INTEGER`, `TEXT`, `DOUBLE PRECISION`).  
6. **Write DataFrame to target table** – Use `df_patients.to_sql(name='d1_new', con=tgt_conn, if_exists='append', index=False, method='multi')` (or a bulk‑insert loop) to copy all rows, preserving `NULL` values.  
7. **Verify target row count** – Run `SELECT COUNT(*) FROM d1_new` on `tgt_conn` and assert the result equals 5110.  
8. **(Optional) Create alias view** – If a second alias `d1_d` is required, execute `CREATE VIEW d1_d AS SELECT * FROM d1_new` on `tgt_conn`.  
9. **Commit and close** – Commit the transaction on `tgt_conn`, then close both `src_conn` and `tgt_conn`.  

---

```json
{
  "result_kind": "row_transform",
  "source_columns": [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke"
  ],
  "result_columns": [
    {"name": "id", "role": "key", "dtype": "numeric"},
    {"name": "gender", "role": "passthrough", "dtype": "string"},
    {"name": "age", "role": "passthrough", "dtype": "numeric"},
    {"name": "hypertension", "role": "passthrough", "dtype": "numeric"},
    {"name": "heart_disease", "role": "passthrough", "dtype": "numeric"},
    {"name": "ever_married", "role": "passthrough", "dtype": "string"},
    {"name": "work_type", "role": "passthrough", "dtype": "string"},
    {"name": "Residence_type", "role": "passthrough", "dtype": "string"},
    {"name": "avg_glucose_level", "role": "passthrough", "dtype": "numeric"},
    {"name": "bmi", "role": "passthrough", "dtype": "numeric"},
    {"name": "smoking_status", "role": "passthrough", "dtype": "string"},
    {"name": "stroke", "role": "passthrough", "dtype": "numeric"}
  ],
  "row_relation": {
    "type": "same_as_input"
  },
  "unique_key": ["id"],
  "not_null": ["id"]
}
```"""

# The code the coder then generated (log lines 8684-8702): it raised instead of writing.
RC10_CODE = r"""import pandas as pd

def main(df):
    # The requested operation involves copying a PostgreSQL table between databases.
    # This requires external libraries such as `sqlalchemy` or `psycopg2`,
    # which are not available in the execution environment.
    raise ValueError(
        "PostgreSQL database operations require external libraries (e.g., sqlalchemy or psycopg2) "
        "that are not installed. Unable to perform the requested table transfer."
    )
    # The function would normally return a DataFrame, but execution stops at the error above.
    return df

if __name__ == "__main__":
    # Load the input CSV as a placeholder; the script will raise an error before any
    # further processing.
    df = pd.read_csv("input.csv")
    df = main(df)
    df.to_csv("output.csv", index=False)
"""

# RC10_BLUEPRINT carried out as written -- the code the coder would have run had
# it not chosen to raise: open both connections, read, CREATE TABLE, append.
RC10_BLUEPRINT_AS_CODE = '''
import pandas as pd
from sqlalchemy import create_engine


def main(df):
    src_conn = create_engine("postgresql://u:p@db:5432/e2eclinicgroq")
    df_patients = pd.read_sql("SELECT * FROM patients", src_conn)
    assert len(df_patients) == 5110
    tgt_conn = create_engine("postgresql://u:p@db:5432/e2ewhgroq")
    with tgt_conn.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE d1_new (id INTEGER, gender TEXT, stroke INTEGER)")
    df_patients.to_sql(name='d1_new', con=tgt_conn, if_exists='append', index=False, method='multi')
    return df_patients
'''

SCHEMA = {"id": "int64", "gender": "string", "age": "float64", "stroke": "int64"}


def _static(code: str, **extra) -> dict:
    state = {"generated_code": code, "schema": SCHEMA, "user_prompt": "", **extra}
    return static_semantic_validator_node(state)


def _rule_errors(out: dict) -> list[str]:
    return [line for line in (out.get("code_validation_feedback") or "").splitlines()
            if f"[{DB_WRITE_RULE}]" in line]


# ---- layer 2: the static validator rule (HARD) -------------------------------

def test_validator_rejects_the_rc10_blueprint_carried_out():
    out = _static(RC10_BLUEPRINT_AS_CODE)
    assert out["static_semantic_error"] is True
    errors = _rule_errors(out)
    joined = "\n".join(errors)
    for what in ("sqlalchemy", "create_engine()", "read_sql()", "CREATE SQL", ".to_sql()"):
        assert what in joined, joined
    assert all("transfer the user confirms" in e for e in errors)


def test_the_rc10_code_itself_wrote_nothing_and_is_not_what_the_rule_is_about():
    # The code rc10 actually ran raised before touching anything; "sqlalchemy"
    # and "psycopg2" appear only inside its error message. The rule reads code,
    # not strings, so it does not fire here -- what stopped this turn is the
    # coder's blueprint guard (below) and the planner's bare-yes guard.
    assert _rule_errors(_static(RC10_CODE)) == []
    assert describes_database_write(RC10_BLUEPRINT)


def _in_main(body: str, imports: str = "") -> str:
    lines = "\n".join("    " + ln for ln in body.strip().splitlines())
    return f"import pandas as pd\n{imports}\n\ndef main(df):\n{lines}\n    return df\n"


@pytest.mark.parametrize("imports, body, expect", [
    ("", "df.to_sql('d1_new', con, if_exists='append')", ".to_sql()"),
    ("from sqlalchemy import create_engine", "e = create_engine('postgresql://u:p@db/wh')", "create_engine()"),
    ("import sqlalchemy as sa", "e = sa.create_engine('mysql://u:p@db/wh')", "'sqlalchemy'"),
    ("import psycopg2", "c = psycopg2.connect(host='db')", "'psycopg2'"),
    ("import psycopg", "c = psycopg.connect('dbname=wh')", "'psycopg'"),
    ("import pymysql", "c = pymysql.connect(host='db')", "'pymysql'"),
    ("from pymongo import MongoClient", "MongoClient('mongodb://db')['wh']['t'].insert_many([])", "'pymongo'"),
    ("", "coll.insert_many(df.to_dict('records'))", "MongoDB write"),
    ("import sqlite3", "con = sqlite3.connect('warehouse.db')", "sqlite3.connect()"),
    ("import sqlite3 as lite", "con = lite.connect('warehouse.db')", "sqlite3.connect()"),
    ("from sqlite3 import connect", "con = connect('warehouse.db')", "sqlite3.connect()"),
    ("", "cur.execute('INSERT INTO d1_new VALUES (1, 2)')", "INSERT SQL"),
    ("", "cur.executemany('insert into t values (?)', rows)", "INSERT SQL"),
    ("", "q = 'DROP TABLE patients'\ncur.execute(q)", "DROP SQL"),
    ("", "t = 'patients'\ncur.execute(f'DELETE FROM {t} WHERE 1=1')", "DELETE SQL"),
    ("", "conn.execute(text('UPDATE patients SET stroke = 0'))", "UPDATE SQL"),
    ("", "cur.executescript('CREATE TABLE x (a INT); ALTER TABLE x ADD b INT')", "CREATE SQL"),
    ("", "cur.execute('TRUNCATE patients')", "TRUNCATE SQL"),
    ("", "cur.execute('-- tidy up\\nALTER TABLE patients DROP COLUMN bmi')", "ALTER SQL"),
    ("", "cur.execute('WITH s AS (SELECT 1) INSERT INTO t SELECT * FROM s')", "INSERT SQL"),
    ("", "m = __import__('psycopg2')", "'psycopg2'"),
    ("import importlib", "m = importlib.import_module('pymysql')", "'pymysql'"),
    ("", "x = pd.read_sql('SELECT * FROM patients', 'postgresql://u:p@db/clinic')", "read_sql()"),
])
def test_validator_rejects_each_way_into_a_database(imports, body, expect):
    out = _static(_in_main(body, imports))
    assert out["static_semantic_error"] is True, out
    errors = _rule_errors(out)
    assert errors and any(expect in e for e in errors), out.get("code_validation_feedback")


@pytest.mark.parametrize("body", [
    # Ordinary analysis on the provided df.
    "rate = df.groupby('gender')['stroke'].mean()\navaloka_result(value=rate.max(), kind='max', columns=['stroke'])",
    "df = df[df['stroke'] == 1]\nmean_age = df['age'].mean()",
    "df['gender'] = df['gender'].str.replace('F', 'Female')",
    "df = df.query('age > 50')",
    "df.to_csv('output.csv', index=False)",
    "df.to_parquet('output.parquet')",
    "df = df.copy()\ndf.update(df.fillna(0))",
    "summary = df.describe()\nsummary.to_json('summary.json')",
    # A scratch in-memory SQL engine over the df writes nothing anywhere.
    "import sqlite3\ncon = sqlite3.connect(':memory:')\n"
    "rows = con.execute(\"SELECT REPLACE('a-b', '-', '') AS x\").fetchall()",
    # Spark/DuckDB temp views live in the session.
    "spark.sql('CREATE OR REPLACE TEMP VIEW pts AS SELECT 1')",
])
def test_normal_analysis_code_still_passes(body):
    out = _static(_in_main(body))
    assert _rule_errors(out) == [], out.get("code_validation_feedback")


def test_rc10_stroke_answer_code_passes_whole():
    # The stroke question's real rc10 code (log line 9731), unchanged.
    code = (
        "import pandas as pd\n\n"
        "def main(df):\n"
        "    df = df[df['stroke'] == 1]\n"
        "    df = df[df['age'].notna()]\n"
        "    mean_age = df['age'].mean()\n"
        "    avaloka_result(value=mean_age, kind=\"mean\", columns=[\"age\"])\n"
        "    return pd.DataFrame({'average_age': [mean_age]})\n\n"
        "if __name__ == \"__main__\":\n"
        "    df = read_csv_best_effort(\"input.csv\")\n"
        "    df = main(df)\n"
        "    df.to_csv(\"output.csv\", index=False)\n"
    )
    out = _static(code)
    assert out["static_semantic_error"] is False, out.get("code_validation_feedback")


def test_the_transfer_agents_own_code_is_exempt_only_when_its_pipeline_says_so():
    code = _in_main("df.to_sql('dest', con, if_exists='append')")
    assert _rule_errors(_static(code))                                   # fail closed
    assert _rule_errors(_static(code, dta_transfer_codegen=False))
    assert _rule_errors(_static(code, dta_transfer_codegen=True)) == []


def test_the_transfer_pipeline_marks_its_code(monkeypatch):
    import app.agents.data_transfer_agent.data_transfer_agent as dta

    seen = {}

    class _Stop(Exception):
        pass

    def _pseudocode(cd):
        seen.update(cd)
        raise _Stop

    monkeypatch.setattr(dta, "sanitize_user_prompt_for_transformations", lambda p, llm: p)
    monkeypatch.setattr(dta, "daft_pseudocode_node", _pseudocode)
    with pytest.raises(_Stop):
        dta.daft_code_generation_pipeline({"coder_definition": {"user_prompt": "copy it"}})
    assert seen.get("dta_transfer_codegen") is True


# ---- layer 1b: the coder's blueprint guard -----------------------------------

def _coder_state(plan: str) -> dict:
    return {"plan": plan, "user_prompt": "yes", "messages": [], "schema": SCHEMA,
            "retry_count": 0, "input_data_type": "csv"}


def test_the_coder_refuses_the_rc10_blueprint_before_writing_any_code(monkeypatch):
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=RC10_BLUEPRINT)
    monkeypatch.setattr(coder, "coder_llm", llm)
    # rc10's planner plan was about the stroke question; the write came from the blueprint.
    out = coder.coder_node(_coder_state("Group by gender and count rows; sum strokes per gender."))
    assert out["coder_blocked_reason"] == DB_WRITE_REFUSAL
    assert llm.invoke.call_count == 1, "only the blueprint was asked for; no code was generated"
    assert "to_sql" not in out["generated_code"] and "sqlalchemy" not in out["generated_code"]
    assert _rule_errors(_static(out["generated_code"])) == []


def test_the_coder_refuses_a_plan_that_writes_even_with_a_clean_blueprint(monkeypatch):
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="1. Group by gender.\n2. Compute the mean of stroke.")
    monkeypatch.setattr(coder, "coder_llm", llm)
    out = coder.coder_node(_coder_state(
        "Compute the stroke rate by gender, then append the result to the warehouse database."))
    assert out["coder_blocked_reason"] == DB_WRITE_REFUSAL


def test_the_coder_still_codes_an_analysis_blueprint(monkeypatch):
    blueprint = (
        "**Pseudocode steps**\n\n"
        "1. **Start with the original DataFrame** `df` that contains the columns `age` and `stroke`.\n"
        "2. **Select rows where a stroke occurred**: keep only rows with `df['stroke'] == 1`.\n"
        "3. **Compute the mean age** of the remaining rows using the `age` column.\n"
    )
    code = "```python\nimport pandas as pd\n\ndef main(df):\n    return df[df['stroke'] == 1]\n```"
    llm = MagicMock()
    llm.invoke.side_effect = [AIMessage(content=blueprint), AIMessage(content=code)]
    monkeypatch.setattr(coder, "coder_llm", llm)
    out = coder.coder_node(_coder_state("Average age of patients with stroke."))
    assert not out.get("coder_blocked_reason"), out.get("coder_blocked_reason")
    assert llm.invoke.call_count == 2
    assert "def main" in out["generated_code"]


# ---- the shared prose check (planner + coder) --------------------------------

@pytest.mark.parametrize("text", [
    RC10_BLUEPRINT,
    "Transfer the patients table to the warehouse e2ewhgroq as d1_new.",
    "Use df.to_sql to append the rows.",
    "Issue a CREATE TABLE d1_new statement on the target connection.",
    "Write the per-gender stroke rate to the target table.",
    "Load the cleaned frame into the Postgres database.",
    "Export the summary to Snowflake.",
    "Open a psycopg2 connection to the warehouse.",
])
def test_plans_that_write_are_recognised(text):
    assert describes_database_write(text), text


@pytest.mark.parametrize("text", [
    "Group by gender and count rows; sum strokes per gender.",
    "Read the full patients table from the database into a DataFrame and compute the mean age.",
    "Load the data from the warehouse into pandas, then group by gender.",
    "Push the aggregation down to the database and return the counts.",
    "Create a table of counts per gender; truncate long labels.",
    "Save the result to output.csv.",
    "Copy the dataframe to avoid mutating the original.",
    "Store the mean in a new column called avg_age.",
    "Group by transfer type to compute totals per database_id.",
    "How many rows did each database contribute? Count rows per source_db.",
    "",
    None,
])
def test_analysis_plans_are_not(text):
    assert describes_database_write(text) is None, text


# ---- rc11: a read plan is not a write ----------------------------------------
#
# rc11 web check (929a140a): 5 of 9 READ questions on the generate_code path
# were refused with DB_WRITE_REFUSAL; "average age of patients with stroke" 3 of
# 5 times (rc10: answered 5 of 5). The guard matched 'psycopg2', 'sqlalchemy'
# and 'CREATE TABLE' in plans that only read: the memory plane's session
# signature ("Use SQL to copy and rename the table ...") leads the planner to
# open an engine and read_sql the table, or stage it in a scratch table.
#
# The rweb11 log kept only the matched word, not the plan, and the refused
# turn's state has plan=None. So these plans are regenerated, not copied: the
# planner's plan prompt verbatim, the same model (openai/gpt-oss-120b on Groq,
# temperature 0.2, effort medium), and each refused turn's own inputs -- the
# question with its [Analysis context] line, the patients dataset as
# tables-to-analysis returned it, and that turn's memory hints and session
# signature from its "Memory Payload" log line. Each one kept here was refused
# by the guard at 60782703 (match_at_60782703) and only reads.

_RC11_PLANS = json.loads(
    (Path(__file__).parent / "fixtures" / "rc11_refused_read_plans.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _RC11_PLANS,
                         ids=[f"{c['rweb11_turn']}-{c['logged_match']}" for c in _RC11_PLANS])
def test_rc11_refused_read_plans_are_not_writes(case):
    assert describes_database_write(case["plan"]) is None, case["question"]


@pytest.mark.parametrize("text", [
    # A driver or library named, to read.
    "Import psycopg2 and read the patients table.",
    "Use a DB-API like `sqlite3`, `psycopg2`, etc. or the SQL console to obtain the count.",
    "1. Import pandas and sqlalchemy.\n2. Create a SQLAlchemy engine from the DB URI.\n"
    "3. df = pd.read_sql_table('patients', con=engine)",
    "Run SELECT COUNT(*) FROM patients WHERE bmi IS NULL with pd.read_sql.",
    "Load it into a DataFrame using the connection.",
    # A scratch or temporary table a read plan stages its query in.
    "CREATE TABLE d1_b_copy AS SELECT * FROM patients;\n... DROP TABLE d1_b_copy;",
    "CREATE TABLE d1_b_copy AS\nSELECT *\nFROM e2eclinicgroq.patients\nWITH (FIDELITY='full');",
    "CREATE TEMP TABLE stroke_pts AS SELECT * FROM patients WHERE stroke = 1",
    "CREATE TEMPORARY TABLE x AS SELECT 1",
    "con = sqlite3.connect(':memory:'); CREATE TABLE t AS SELECT * FROM df",
    "CREATE VIEW d1_stroke AS SELECT * FROM d1_b WHERE stroke = 1;",
    "| 1 | **Copy the table** | ```sql<br>CREATE TABLE d1_b AS SELECT * FROM patients;<br>``` |",
    "Drop the temporary copy table to keep the database tidy.",
])
def test_read_plans_that_name_drivers_or_stage_tables_are_not_writes(text):
    assert describes_database_write(text) is None, text


@pytest.mark.parametrize("text", [
    # Write calls and write SQL.
    "df.to_sql('d1_new', con=tgt_conn, if_exists='append', index=False)",
    "Upload it with pandas_gbq: df.to_gbq('ds.t').",
    "coll.insert_many(records)",
    "INSERT INTO d1_new SELECT * FROM patients",
    "UPDATE patients SET stroke = 0",
    "DELETE FROM patients WHERE age < 0",
    "MERGE INTO wh.patients t USING src s ON t.id = s.id",
    "TRUNCATE TABLE patients",
    "ALTER TABLE patients ADD COLUMN risk DOUBLE",
    # CREATE/DROP of a persistent table aimed at a connection.
    "CREATE TABLE e2ewhgroq.patients_copy AS SELECT * FROM e2ewhgroq.patients",
    "| 10 | **Clean-up** | ```sql\\nDROP TABLE e2eclinicgroq.patients_copy;\\n``` |",
    "CREATE TABLE d1_new (id INT) on the target connection.",
    "CREATE SCHEMA analytics",
    # Prose: copy/save/export INTO a database, connection or warehouse table.
    "TRANSFER d1_b TO db://e2ewhgroq AS d1_b_copy WITH fidelity=full;",
    "TRANSFER TABLE patients FROM db://e2eclinicgroq/c7527ead TO db://e2ewhgroq/patients_copy;",
    "Copy the full `patients` table from the source database to a working database.",
    "Save the per-gender rates into the warehouse table d1_rates.",
    "1. Create a SQLAlchemy engine for the destination.\n2. Append the rows.",
])
def test_plans_that_write_still_refuse(text):
    assert describes_database_write(text), text


def test_the_coder_codes_an_rc11_read_blueprint(monkeypatch):
    # The coder runs the same check over its blueprint and the planner's plan;
    # a read named through a driver must reach code generation there too.
    plan = next(c["plan"] for c in _RC11_PLANS if c["logged_match"] == "sqlalchemy"
                and "sqlalchemy" in c["plan"].lower())
    code = "```python\nimport pandas as pd\n\ndef main(df):\n    return df[df['bmi'].isna()]\n```"
    llm = MagicMock()
    llm.invoke.side_effect = [AIMessage(content=plan), AIMessage(content=code)]
    monkeypatch.setattr(coder, "coder_llm", llm)
    out = coder.coder_node(_coder_state(plan))
    assert not out.get("coder_blocked_reason"), out.get("coder_blocked_reason")
    assert llm.invoke.call_count == 2
    assert "def main" in out["generated_code"]
