import sys
from pathlib import Path
import types
import pandas as pd
import pytest
import inspect
import os

# Make repo root importable
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Under test
import app.agents.sampling_agent as sa
from app.agents.sampling_agent import (
    sample_data_from_source,
    infer_best_stratify_column,
    stratified_sample,
    generate_ddl_from_spark_schema,
    get_spark_schema,
)

# ---------- Tiny helpers used by a couple of tests ----------

class _DT:
    def __init__(self, s): self._s = s
    def simpleString(self): return self._s

class _Field:
    def __init__(self, name, typ): self.name, self.dataType = name, _DT(typ)

def _fake_schema_from_columns(cols, typ="string"):
    return types.SimpleNamespace(fields=[_Field(c, typ) for c in cols])

class _FakeSchema: fields = []

class _FakeReader:
    def __init__(self): self.opts = {}
    def option(self, k, v): self.opts[k] = v; return self
    def csv(self, _):
        class _FakeDF: schema = _FakeSchema()
        return _FakeDF()

class _FakeSpark:
    def __init__(self): self.read = _FakeReader()

class _FakeBuilder:
    def __init__(self): self._spark = _FakeSpark()
    def appName(self, _): return self
    def getOrCreate(self): return self._spark


# ============================================================
# Self-contained sample CSVs (no app/sample_data dependency)
# ============================================================

@pytest.fixture(scope="module")
def sample_csvs(tmp_path_factory):
    """
    Creates all CSV fixtures under a temp dir and returns a dict: {filename: Path}.
    Anything not created (e.g., not_exist.csv) will simply be absent.
    """
    base = tmp_path_factory.mktemp("sample_data")

    def _w(name: str, text: str):
        p = base / name
        p.write_text(text, encoding="utf-8")
        return p

    files = {
        # Basic/valid datasets
        "Dummy_Sales_Data.csv": _w("Dummy_Sales_Data.csv",
                                   "Region,Sales\nNorth,100\nSouth,200\nEast,150\nWest,120\n"),
        "TC-CSV-001_valid.csv": _w("TC-CSV-001_valid.csv",
                                   "Product,Price\nA,10\nB,20\n"),
        "TC-CSV-002_only_header.csv": _w("TC-CSV-002_only_header.csv",
                                         "Product,Price\n"),
        "TC-CSV-003_empty.csv": _w("TC-CSV-003_empty.csv", ""),  # empty file

        # Special chars / malformed but should succeed leniently
        "TC-CSV-006_special_characters.csv": _w("TC-CSV-006_special_characters.csv",
                                                "id,Product(Name),Price$\n1,Apple,1.2\n2,B@nana,0.8\n"),
        "TC-CSV-007_malformed.csv": _w("TC-CSV-007_malformed.csv",
                                       "Product,Qty\nApple,10\nBanana,20,EXTRA\nCherry,30\n"),

        # Error case: stratify column missing
        "TC-CSV-010_no_stratify_column.csv": _w("TC-CSV-010_no_stratify_column.csv",
                                                "Product,Price\nA,10\nB,20\n"),

        # Single-row per class / stratum
        "TC-CSV-011_single_row_per_class.csv": _w("TC-CSV-011_single_row_per_class.csv",
                                                  "Segment,Value\nA,1\nB,2\nC,3\n"),
        "TC-CSV-011_single_row_per_stratum.csv": _w("TC-CSV-011_single_row_per_stratum.csv",
                                                    "Category,Item,Amount\nA,Item1,10\nB,Item2,20\nC,Item3,30\n"),

        # Imbalanced distribution
        "TC-CSV-012_imbalanced_distribution.csv": _w("TC-CSV-012_imbalanced_distribution.csv",
                                                     "Segment,Value\nA,1\nA,2\nA,3\nB,4\n"),

        # Numeric stratify
        "TC-CSV-013_numeric_stratify.csv": _w("TC-CSV-013_numeric_stratify.csv",
                                              "Age,Score\n20,80\n25,85\n20,90\n30,70\n"),

        
        "Dummy_Sales_Data.json": _w("Dummy_Sales_Data.json",
                                    '[{"Region":"North","Sales":100},{"Region":"South","Sales":200},{"Region":"East","Sales":150},{"Region":"West","Sales":120}]'),
        "TC-JSON-001_valid.json": _w("TC-JSON-001_valid.json",
                                   '[{"Product":"A","Price":10},{"Product":"B","Price":20}]'),
        "TC-JSON-002_only_header.json": _w("TC-JSON-002_only_header.json",
                                         "[]"),
        "TC-JSON-003_empty.json": _w("TC-JSON-003_empty.json", ""),  # empty file


        "Dummy_Sales_Data.xml": _w("Dummy_Sales_Data.xml",
                                    '<Orders><Order><Region>North</Region><Sales>100</Sales></Order><Order><Region>South</Region><Sales>200</Sales></Order><Order><Region>East</Region><Sales>150</Sales></Order><Order><Region>West</Region><Sales>120</Sales></Order></Orders>'),
        "TC-XML-001_valid.xml": _w("TC-XML-001_valid.xml",
                                   '<Orders><Order><Product>A</Product><Price>10</Price></Order><Order><Product>B</Product><Price>20</Price></Order></Orders>'),
        "TC-XML-002_only_header.xml": _w("TC-XML-002_only_header.xml",
                                         "<Orders></Orders>"),
        "TC-XML-003_empty.xml": _w("TC-XML-003_empty.xml", ""),  # empty file
    }

    # NOTE: "not_exist.csv" is intentionally NOT created.
    return files


# ============================================================
# Integration tests that now call sample_data_from_source()
# using the temp CSVs created above
# ============================================================

def test_valid_sampling(sample_csvs):
    p = sample_csvs["TC-CSV-001_valid.csv"]
    out = sample_data_from_source(str(p), "csv", stratify_by="Product", sample_size=0.2)
    assert out["error"] is None
    assert len(out["rows"]) > 0
    assert "CREATE TABLE" in out["ddl_schema"]

def test_invalid_stratify_column(sample_csvs):
    p = sample_csvs["TC-CSV-010_no_stratify_column.csv"]
    out = sample_data_from_source(str(p), "csv", stratify_by="NonExistentColumn", sample_size=1)
    assert out["error"] is not None
    assert "Stratify column" in out["error"]

def test_one_row_per_class_and_stratum(sample_csvs):
    p1 = sample_csvs["TC-CSV-011_single_row_per_class.csv"]
    out1 = sample_data_from_source(str(p1), "csv", stratify_by="Segment", sample_size=1)
    assert out1["error"] is None

    p2 = sample_csvs["TC-CSV-011_single_row_per_stratum.csv"]
    out2 = sample_data_from_source(str(p2), "csv", stratify_by="Category", sample_size=1)
    assert out2["error"] is None

def test_imbalanced_stratify_column(sample_csvs):
    p = sample_csvs["TC-CSV-012_imbalanced_distribution.csv"]
    out = sample_data_from_source(str(p), "csv", stratify_by="Segment", sample_size=1)
    assert out["error"] is None

def test_large_file_simulation(monkeypatch, tmp_path):
    # Create the path the code will try to open for delimiter sniffing
    p = tmp_path / "ignored.csv"
    p.write_text("Region,Sales\n", encoding="utf-8")

    def mock_read_csv(*args, **kwargs):
        return pd.DataFrame({
            "Region": ["North"] * 100_000,
            "Sales": [float(i) for i in range(100_000)],
        })
    monkeypatch.setattr("pandas.read_csv", mock_read_csv)

    monkeypatch.setattr(sa, "get_spark_schema",
                        lambda csv_path, pandas_encoding="utf-8":
                        _fake_schema_from_columns(["Region", "Sales"]))

    out = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=0.1)
    assert out["error"] is None


def test_numeric_stratify_column(sample_csvs):
    p = sample_csvs["TC-CSV-013_numeric_stratify.csv"]
    out = sample_data_from_source(str(p), "csv", stratify_by="Age", sample_size=0.5)
    assert out["error"] is None

def test_random_sampling_fallback(sample_csvs):
    p = sample_csvs["TC-CSV-001_valid.csv"]
    out = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=0.2)
    assert out["error"] is None

@pytest.mark.parametrize(
    "filename, expected_error, stratify_by",
    [
        ("Dummy_Sales_Data.csv", None, "Region"),
        ("TC-CSV-001_valid.csv", None, "Product"),
        ("TC-CSV-002_only_header.csv", None, "Product"),
        ("TC-CSV-003_empty.csv", "Failed to sample data", "Product"),

        ("Dummy_Sales_Data.json", None, "Region"),
        ("TC-JSON-001_valid.json", None, "Product"),
        ("TC-JSON-002_only_header.json", "", "Product"),
        ("TC-JSON-003_empty.json", "Failed to sample data", "Product"),

        ("Dummy_Sales_Data.xml", None, "Region"),
        ("TC-XML-001_valid.xml", None, "Product"),
        ("TC-XML-002_only_header.xml", "", "Product"),
        ("TC-XML-003_empty.xml", "Failed to sample data", "Product"),

        ("not_exist.csv", "Failed to sample data", "Product"),  # intentionally absent
        ("TC-CSV-006_special_characters.csv", None, "Product(Name)"),
        ("TC-CSV-007_malformed.csv", None, "Product"),
    ]
)
def test_sample_csv_cases(filename, expected_error, stratify_by, sample_csvs, tmp_path):
    if filename in sample_csvs:
        p = sample_csvs[filename]
    else:
        # Point to a non-existent path for "not_exist.csv"
        p = tmp_path / filename

    _, ext = os.path.splitext(filename)
    ext = ext.lstrip(".")
    out = sample_data_from_source(str(p), ext, stratify_by=stratify_by, sample_size=1)
    if expected_error:
        assert out["error"] is not None
        assert expected_error in out["error"]
    else:
        assert out["error"] is None
        assert isinstance(out["rows"], list)
        assert isinstance(out["ddl_schema"], str)

def test_integration_csv_with_special_chars(tmp_path):
    csv_file = tmp_path / "special_chars.csv"
    csv_file.write_text("id,Product(Name),Price$\n1,Apple,1.2\n2,B@nana,0.8")
    out = sample_data_from_source(str(csv_file), "csv", stratify_by="Product(Name)", sample_size=1)
    assert out["error"] is None
    assert any("Product(Name)" in col for col in out["schema"])

def test_integration_csv_all_nulls_in_stratify(tmp_path):
    csv_file = tmp_path / "nulls.csv"
    csv_file.write_text("id,category\n1,\n2,\n3,")
    out = sample_data_from_source(str(csv_file), "csv", stratify_by="category", sample_size=1)
    assert out["error"] is not None
    assert "contains only null" in out["error"]

def test_integration_csv_duplicate_rows(tmp_path):
    csv_file = tmp_path / "duplicates.csv"
    csv_file.write_text("id,region\n1,East\n1,East\n2,West")
    out = sample_data_from_source(str(csv_file), "csv", stratify_by="region", sample_size=1)
    assert out["error"] is None
    assert len(out["rows"]) > 0

def test_integration_csv_numeric_only(tmp_path):
    csv_file = tmp_path / "numeric_only.csv"
    csv_file.write_text("value\n10\n20\n30")
    out = sample_data_from_source(str(csv_file), "csv", stratify_by=None, sample_size=1)
    assert out["error"] is None
    assert "CREATE TABLE" in out["ddl_schema"]

def test_integration_csv_one_column(tmp_path):
    csv_file = tmp_path / "one_column.csv"
    csv_file.write_text("name\nAlice\nBob")
    out = sample_data_from_source(str(csv_file), "csv", stratify_by=None, sample_size=1)
    assert out["error"] is None
    assert len(out["schema"]) == 1

# --- Malformed-row behaviors (your reader skips bad lines) ---
def test_skips_extra_columns_in_row_instead_of_error(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b,c\n1,2,3\n4,5,6,7,8,9\n")
    out = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=1)
    # With on_bad_lines='skip' your function should succeed with the valid row
    assert out["error"] is None
    assert len(out["rows"]) == 1

def test_skips_too_few_fields_and_succeeds(tmp_path, monkeypatch):
    p = tmp_path / "few.csv"
    p.write_text("id,name\n1,Alice\n2\n")
    # Provide a simple fake Spark schema (two columns)
    monkeypatch.setattr(sa, "get_spark_schema",
                        lambda csv_path, pandas_encoding="utf-8": _fake_schema_from_columns(["id","name"]))
    out = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=1)
    assert out["error"] is None
    assert len(out["rows"]) == 1

def test_quotes_with_commas_and_newlines(tmp_path, monkeypatch):
    p = tmp_path / "quotes.csv"
    p.write_text('id,text\n1,"hello, world\nnext"\n')
    monkeypatch.setattr(sa, "get_spark_schema",
                        lambda csv_path, pandas_encoding="utf-8": _fake_schema_from_columns(["id","text"]))
    out = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=1)
    assert out["error"] is None
    assert any(r["text"].startswith("hello") for r in out["rows"])

def test_utf8_bom_file(tmp_path, monkeypatch):
    p = tmp_path / "bom.csv"
    raw = "id,name\n1,Alice\n2,Bob\n".encode("utf-8-sig")
    p.write_bytes(raw)
    monkeypatch.setattr(sa, "get_spark_schema",
                        lambda csv_path, pandas_encoding="utf-8": _fake_schema_from_columns(["id","name"]))
    out = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=2)
    # or: sample_size=1.0
    assert out["error"] is None
    assert len(out["rows"]) == 2

def test_generate_ddl_types_without_spark():
    schema = types.SimpleNamespace(fields=[
        _Field("name","string"), _Field("age","int"), _Field("income","double")
    ])
    ddl = generate_ddl_from_spark_schema("people", schema)
    assert "name VARCHAR" in ddl
    assert "age INT" in ddl
    assert "income DOUBLE" in ddl

def test_sqlite_source(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "ENABLE_SQLITE", True, raising=False)
    db = tmp_path / "db.sqlite"
    import sqlite3
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("create table t(id int, name text)")
    cur.executemany("insert into t values(?,?)", [(1,"A"), (2,"B")])
    conn.commit(); conn.close()

    out = sample_data_from_source(str(db), "sql", stratify_by=None, sample_size=1)
    assert out["error"] is None
    assert out["table_name"] == "t"
    assert len(out["rows"]) >= 1

def test_reproducible_random_sample(tmp_path, monkeypatch):
    p = tmp_path / "rand.csv"
    p.write_text("id\n1\n2\n3\n4\n5\n")
    monkeypatch.setattr(sa, "get_spark_schema",
                        lambda csv_path, pandas_encoding="utf-8": _fake_schema_from_columns(["id"]))
    out1 = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=2)
    out2 = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=2)
    assert out1["rows"] == out2["rows"]

def test_auto_stratify_skips_high_cardinality(tmp_path):
    p = tmp_path / "high_card.csv"
    p.write_text("u\n" + "\n".join(str(i) for i in range(30)))
    df = pd.read_csv(p)
    assert infer_best_stratify_column(df) is None

def test_strict_fails_on_extra_columns(tmp_path, monkeypatch):
    # CSV with overlong second row
    p: Path = tmp_path / "bad.csv"
    p.write_text("a,b,c\n1,2,3\n4,5,6,7,8,9\n")

    # Avoid real Spark; schema from header
    monkeypatch.setattr(
        sa, "get_spark_schema",
        lambda csv_path, pandas_encoding="utf-8": _fake_schema_from_columns(["a", "b", "c"])
    )

    sig = inspect.signature(sa.sample_data_from_source)
    kwargs = {
        "source_type": "csv",
        "stratify_by": None,
        "sample_size": 1,   # small to avoid sampling noise
    }
    supported_strict = False
    if "allow_lenient" in sig.parameters:
        kwargs["allow_lenient"] = False
        supported_strict = True
    if "strict_bad_lines" in sig.parameters:
        kwargs["strict_bad_lines"] = True
        supported_strict = True

    out = sa.sample_data_from_source(str(p), **kwargs)

    if supported_strict and out.get("error"):
        # Strict path: should error
        assert any(x in out["error"] for x in [
            "Malformed CSV", "too many fields", "too many columns", "Expected 3 fields", "CSV parsing failed"
        ])
        return

    # Lenient path: ensure only valid row remains
    assert out.get("error") is None
    rows = out["rows"]
    df = pd.DataFrame(rows)
    assert len(df) == 1
    assert set(df.columns) == {"a", "b", "c"}
    assert df.iloc[0].to_dict() == {"a": 1, "b": 2, "c": 3}

# ============================================================
# End-to-End tests (kept as before, but use tmp files)
# ============================================================

def test_e2e_csv_to_ddl_integration(tmp_path):
    p = tmp_path / "etl_data.csv"
    p.write_text("id,name,sales\n1,Alice,100\n2,Bob,200\n3,Charlie,300")
    out = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=2)
    assert out["error"] is None
    assert "CREATE TABLE" in out["ddl_schema"]
    #spark_schema = get_spark_schema(str(p))
    spark_schema = get_spark_schema(str(p), "csv")
    from pyspark.sql.types import StructType
    assert isinstance(spark_schema, StructType)

def test_e2e_csv_auto_infer_stratify(tmp_path):
    p = tmp_path / "auto_infer.csv"
    p.write_text("region,sales\nEast,100\nWest,200\nEast,300")
    out = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=0.5)
    assert out["error"] is None
    assert "CREATE TABLE" in out["ddl_schema"]

def test_e2e_malformed_csv_error(tmp_path):
    p = tmp_path / "malformed.csv"
    p.write_text("id,name\n1,Alice\n2")  # too few fields on last line
    out = sample_data_from_source(str(p), "csv", stratify_by=None)
    assert out["error"] is not None
    assert "Failed to sample data" in out["error"]

def test_e2e_csv_single_row_flow(tmp_path):
    p = tmp_path / "small.csv"
    p.write_text("id,region\n1,East")
    out = sample_data_from_source(str(p), "csv", stratify_by="region", sample_size=1)
    assert out["error"] is None
    assert len(out["rows"]) == 1

def test_e2e_float_sample_size(tmp_path):
    p = tmp_path / "percentage.csv"
    p.write_text("id,region\n1,East\n2,East\n3,West\n4,West")
    out = sample_data_from_source(str(p), "csv", stratify_by="region", sample_size=0.5)
    assert out["error"] is None
    assert len(out["rows"]) > 0

def test_e2e_csv_missing_values_flow(tmp_path):
    p = tmp_path / "missing_values.csv"
    p.write_text("id,region\n1,East\n2,\n3,West")
    out = sample_data_from_source(str(p), "csv", stratify_by="region", sample_size=1)
    assert "CREATE TABLE" in out["ddl_schema"]

def test_e2e_csv_high_cardinality_auto_stratify(tmp_path):
    p = tmp_path / "high_card.csv"
    data = "\n".join([f"{i},{i}" for i in range(1, 51)])
    p.write_text("id,unique\n" + data)
    out = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=0.2)
    assert out["error"] is None
    assert "CREATE TABLE" in out["ddl_schema"]

def test_e2e_csv_no_header(tmp_path):
    p = tmp_path / "no_header.csv"
    p.write_text("1,Alice\n2,Bob")
    out = sample_data_from_source(str(p), "csv", stratify_by=None)
    assert "rows" in out or "error" in out

def test_e2e_empty_stratify_fallback(tmp_path):
    p = tmp_path / "empty_stratify.csv"
    p.write_text("id,category\n1,\n2,\n3,")
    out = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=2)
    assert out["error"] is None or "contains only null" in str(out["error"])


# ---------- Small fakes ----------
class _DT:
    def __init__(self, s): self._s = s
    def simpleString(self): return self._s

class _Field:
    def __init__(self, name, typ): self.name, self.dataType = name, _DT(typ)

def _fake_schema_from_columns(cols, typ="string"):
    return types.SimpleNamespace(fields=[_Field(c, typ) for c in cols])

class _FakeFileHandler:
    """Fake FileHandler for non-CSV formats."""
    def __init__(self, path, source_fmt, **kwargs):
        self._cols = ["a", "b", "grp"]
        self._rows = [
            (1, "x", "G1"),
            (2, "y", "G1"),
            (3, "z", "G2"),
            (4, "x", "G2"),
            (5, "y", "G3"),
            (6, "z", "G3"),
        ]
    def load_data(self): return self._rows
    def get_columns(self): return self._cols


# ============================================================
# Delimiter + encoding
# ============================================================

@pytest.mark.parametrize("payload, expected_cols", [
    ("id;name\n1;Alice\n2;Bob\n", ["id", "name"]),   # semicolon
    ("id\tname\n1\tAlice\n2\tBob\n", ["id", "name"]), # tab
    ("id|name\n1|Alice\n2|Bob\n", ["id", "name"]),   # pipe
])
def test_delimiter_sniffing_variants(tmp_path, monkeypatch, payload, expected_cols):
    p = tmp_path / "mix.csv"
    p.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(sa, "get_spark_schema",
                        lambda *_, **__: _fake_schema_from_columns(expected_cols))
    out = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=1)
    assert out["error"] is None
    assert out["schema"] == expected_cols

def test_cp1252_file_parses(tmp_path, monkeypatch):
    p = tmp_path / "latin.csv"
    text = "id,name\n1,André\n2,Zoë\n3,Price €\n"
    p.write_bytes(text.encode("cp1252", errors="ignore"))
    monkeypatch.setattr(sa, "get_spark_schema",
                        lambda *_, **__: _fake_schema_from_columns(["id","name"]))
    out = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=3)
    assert out["error"] is None
    assert len(out["rows"]) == 3


# ============================================================
# Env cap + sample_size validation
# ============================================================

def test_fraction_respects_env_hard_cap(tmp_path, monkeypatch):
    p = tmp_path / "big.csv"
    p.write_text("id,grp\n" + "\n".join(f"{i},A" for i in range(50_000)))
    monkeypatch.setattr(sa, "get_spark_schema",
                        lambda *_, **__: _fake_schema_from_columns(["id","grp"]))
    monkeypatch.setenv("DEFAULT_SAMPLE_MAX_ROWS", "100")

    # Reload module to re-read env-derived constants
    import importlib
    sa_r = importlib.reload(sa)

    out = sa_r.sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=0.4)
    assert out["error"] is None
    assert len(out["rows"]) == 100

@pytest.mark.parametrize("bad", [0, -1, -0.1, 1.1, "two"])
def test_invalid_sample_size_inputs(tmp_path, bad):
    p = tmp_path / "data.csv"
    p.write_text("id\n1\n2\n3\n")
    out = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=bad)
    assert out["error"] is not None
    assert "sample_size" in out["error"]


# ============================================================
# Auto-stratify & proportional allocation
# ============================================================

def test_infer_best_stratify_picks_low_cardinality():
    df = pd.DataFrame({"a": ["A","B","C","D"], "b": ["X","X","Y","Y"]})
    assert infer_best_stratify_column(df) == "b"

def test_stratified_sample_is_deterministic():
    df = pd.DataFrame({"g": ["A"]*6 + ["B"]*4, "v": range(10)})
    s1 = stratified_sample(df, "g", sample_size=5, cap=10)
    s2 = stratified_sample(df, "g", sample_size=5, cap=10)
    pd.testing.assert_frame_equal(s1, s2)

def test_stratified_sample_proportional_and_capped():
    # 8:2 split → target=5 should roughly allocate 4 to A, 1 to B,
    # then cap=4 trims down to 4 total.
    df = pd.DataFrame({"g": ["A"]*8 + ["B"]*2, "v": range(10)})
    out = stratified_sample(df, "g", sample_size=5, cap=4)
    assert len(out) == 4
    assert out["g"].tolist().count("A") >= out["g"].tolist().count("B")


# ============================================================
# DDL generation (fallback + spark mapping)
# ============================================================

def test_pandas_ddl_fallback_type_map(tmp_path, monkeypatch):
    # Force pandas DDL by making get_spark_schema raise
    monkeypatch.setattr(sa, "get_spark_schema", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no spark")))
    p = tmp_path / "types.csv"
    p.write_text("i64,i32,f64,f32,b,s,c\n1,1,1.0,1.0,True,hello,cat\n")
    out = sample_data_from_source(str(p), "csv", stratify_by=None, sample_size=1)
    ddl = out["ddl_schema"]
    assert all(t in ddl for t in ["BIGINT","INT","DOUBLE","FLOAT","BOOLEAN","VARCHAR"])

def test_spark_ddl_decimal_array_and_dates():
    schema = types.SimpleNamespace(fields=[
        _Field("price", "decimal(12,4)"),
        _Field("tags", "array<string>"),
        _Field("when_ts", "timestamp"),
        _Field("when_d", "date"),
        _Field("blob", "binary"),
    ])
    ddl = generate_ddl_from_spark_schema("t", schema)
    assert "price DECIMAL(12,4)" in ddl
    assert "tags TEXT" in ddl
    assert "when_ts TIMESTAMP" in ddl
    assert "when_d DATE" in ddl
    assert "blob BLOB" in ddl


# ============================================================
# Non-CSV formats via FileHandler (+ TSV path)
# ============================================================

@pytest.mark.parametrize("fmt", ["json","parquet","avro","orc","delta","xml"])
def test_non_csv_formats_routed_through_filehandler(tmp_path, monkeypatch, fmt):
    monkeypatch.setattr(sa, "FileHandler", _FakeFileHandler)
    monkeypatch.setattr(sa, "get_spark_schema",
                        lambda *_, **__: _fake_schema_from_columns(["a","b","grp"]))
    p = tmp_path / f"data.{fmt}"
    p.write_text("placeholder")
    out = sample_data_from_source(str(p), fmt, stratify_by="grp", sample_size=0.5)
    assert out["error"] is None
    assert set(out["schema"]) == {"a","b","grp"}
    assert "CREATE TABLE" in out["ddl_schema"]

def test_tsv_through_filehandler(tmp_path, monkeypatch):
    # In this codebase TSV goes through FileHandler (non-CSV branch)
    monkeypatch.setattr(sa, "FileHandler", _FakeFileHandler)
    monkeypatch.setattr(sa, "get_spark_schema",
                        lambda *_, **__: _fake_schema_from_columns(["a","b","grp"]))
    p = tmp_path / "data.tsv"
    p.write_text("a\tb\tgrp\n1\tx\tG1\n")
    out = sample_data_from_source(str(p), "tsv", stratify_by="grp", sample_size=1)
    assert out["error"] is None
    assert set(out["schema"]) == {"a","b","grp"}


# ============================================================
# Spark builder configuration (packages/extensions)
# ============================================================

def test_build_spark_for_adds_required_packages(monkeypatch):
    captured = {}
    class _ObsBuilder:
        def __init__(self): self._options = {}
        def appName(self, _): return self
        def config(self, k, v):
            self._options[k] = v; captured[k] = v; return self
        def getOrCreate(self):
            class _S: read = types.SimpleNamespace()
            return _S()
    # Replace builder object
    monkeypatch.setattr(sa.SparkSession, "builder", _ObsBuilder())

    sa._build_spark_for("avro")
    sa._build_spark_for("delta")
    sa._build_spark_for("xml")

    assert "spark.jars.packages" in captured
    assert any(s in captured["spark.jars.packages"] for s in ["spark-avro", "delta-spark", "spark-xml"])
    # Delta-specific keys should also be set at least once
    assert any(k.startswith("spark.sql.") for k in captured.keys())

