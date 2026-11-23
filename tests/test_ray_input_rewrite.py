"""The Ray path swaps the placeholder 'input.csv' for the real data source.

Regression: `INPUT_PATH = 'input.csv'` used to become
`INPUT_PATH = DATA_SOURCE_URI, storage_options=opts`, a chained assignment that
crashed every Ray ETL job with "not enough values to unpack (expected 2, got 1)".
"""
from app.agents.execution_agent import _point_input_at_source


def test_inline_reader_call_gets_uri_and_storage_options():
    out = _point_input_at_source("df = pd.read_csv('input.csv')")
    assert out == "df = pd.read_csv(DATA_SOURCE_URI, storage_options=opts)"


def test_path_held_in_a_variable_stays_valid_python():
    src = (
        "INPUT_PATH = 'input.csv'\n"
        "CHUNKSIZE = 500000\n"
        "for chunk in pd.read_csv(INPUT_PATH, chunksize=CHUNKSIZE):\n"
        "    pass\n"
    )
    out = _point_input_at_source(src)
    assert "INPUT_PATH = DATA_SOURCE_URI\n" in out
    assert "pd.read_csv(INPUT_PATH, storage_options=opts, chunksize=CHUNKSIZE)" in out
    compile(out, "<generated>", "exec")


def test_other_readers_and_double_quotes():
    out = _point_input_at_source('df = pd.read_parquet("input.csv", columns=["a"])')
    assert out == 'df = pd.read_parquet(DATA_SOURCE_URI, storage_options=opts, columns=["a"])'


def test_storage_options_is_not_added_twice():
    src = "INPUT_PATH = 'input.csv'\nx = pd.read_csv(INPUT_PATH, storage_options=opts)\n"
    assert _point_input_at_source(src).count("storage_options") == 1


def test_other_files_are_left_alone():
    assert _point_input_at_source("df = pd.read_csv('other.csv')") == "df = pd.read_csv('other.csv')"