"""The upload path must actually read the nine formats /api/upload declares.

``SUPPORTED_UPLOAD_EXTS`` in ``app/api/server.py`` lists csv, tsv, json, xml,
parquet, avro, orc, xls and xlsx. Three of those declarations were not backed
by the reader:

* ``orc`` had no connector at all, so ``FileHandler(path, "orc")`` raised
  ``TypeError`` from the fall-through at the end of ``get_connector``.
* ``tsv`` rode the CSV connector, which never sniffed the delimiter, so a
  tab-separated file came back as one column whose name was the whole header
  line. Nothing raised -- the profile of a one-column frame looks like the
  profile of a real table, which is why this survived.
* the fall-through itself named ``Path(path).stem`` -- the filename -- where it
  meant the extension, so the one error a user does see pointed at the wrong
  thing and did not say what is supported.

Each test below was run against the unfixed code first and observed to fail on
the real symptom; a check that cannot fail is not a check.
"""

from __future__ import annotations

import datetime as dt

import pytest

from file_handler.csv_connector import CSVConnector
from file_handler.handler import FileHandler

pa = pytest.importorskip("pyarrow")
orc = pytest.importorskip("pyarrow.orc")


# ---------------------------------------------------------------------------
# Fixtures: small files written with the libraries already in the runtime
# image (pyarrow for ORC), not with the connector under test, so a reader bug
# cannot be hidden by a matching writer bug.
# ---------------------------------------------------------------------------
ORC_COLUMNS = ["region", "units_sold", "revenue_usd", "reported_on", "channel"]
ORC_ROWS = [
    ["north", 10, 100.5, "2024-01-01", "web"],
    ["south", 20, 200.25, "2024-01-02", "retail"],
    ["east", 30, 300.75, "2024-01-03", "web"],
    ["west", 40, 400.0, "2024-01-04", "partner"],
]


@pytest.fixture
def orc_path(tmp_path):
    table = pa.table(
        {
            "region": pa.array([r[0] for r in ORC_ROWS], pa.string()),
            "units_sold": pa.array([r[1] for r in ORC_ROWS], pa.int64()),
            "revenue_usd": pa.array([r[2] for r in ORC_ROWS], pa.float64()),
            "reported_on": pa.array([r[3] for r in ORC_ROWS], pa.string()),
            "channel": pa.array([r[4] for r in ORC_ROWS], pa.string()),
        }
    )
    path = tmp_path / "quarterly_sales.orc"
    orc.write_table(table, str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Defect 1 -- ORC has no connector.
# ---------------------------------------------------------------------------
def test_orc_is_readable_through_file_handler(orc_path):
    """Before the fix this raised TypeError: 't files aren't supported'."""
    handler = FileHandler(orc_path, "orc")

    assert handler.get_columns() == ORC_COLUMNS
    assert handler.get_row_count() == len(ORC_ROWS)
    assert handler.get_row(0) == ORC_ROWS[0]
    assert handler.get_row(3) == ORC_ROWS[3]
    assert handler.load_data() == ORC_ROWS


def test_orc_dtypes_survive_the_round_trip(orc_path):
    """An int column must not come back as text.

    This is the failure that a column-name assertion alone would miss: a reader
    that stringifies everything still reports the right five columns, and the
    DDL generated downstream would then type a numeric measure as VARCHAR.
    """
    handler = FileHandler(orc_path, "orc")
    row = handler.get_row(0)

    assert isinstance(row[1], int) and not isinstance(row[1], bool), (
        f"units_sold came back as {type(row[1]).__name__} ({row[1]!r})")
    assert isinstance(row[2], float), (
        f"revenue_usd came back as {type(row[2]).__name__} ({row[2]!r})")
    assert isinstance(row[0], str)

    # load_data goes through a different code path (whole-table read) than
    # get_row (one stripe), so assert the types on both.
    loaded = handler.load_data()
    assert [type(v) for v in loaded[0]] == [str, int, float, str, str]


def test_orc_sample_rows_is_bounded_and_typed(orc_path):
    handler = FileHandler(orc_path, "orc")

    head = handler.sample_rows(num_rows=2, random_rows=False)
    assert head == ORC_ROWS[:2]

    # Asking for more rows than exist must clamp rather than raise -- the
    # sampling agent asks for a fixed n without knowing the row count.
    everything = handler.sample_rows(num_rows=50, random_rows=False)
    assert everything == ORC_ROWS

    drawn = handler.sample_rows(num_rows=3, random_rows=True)
    assert len(drawn) == 3
    assert all(row in ORC_ROWS for row in drawn)


def test_orc_timestamps_do_not_leak_pandas_objects(tmp_path):
    """Datetime columns are stringified, as the parquet connector does, so the
    rows stay JSON-serialisable for the API response."""
    table = pa.table(
        {
            "event_at": pa.array(
                [dt.datetime(2024, 5, 1, 12, 30), dt.datetime(2024, 5, 2, 8, 0)],
                pa.timestamp("us"),
            ),
            "amount": pa.array([1, 2], pa.int64()),
        }
    )
    path = tmp_path / "events.orc"
    orc.write_table(table, str(path))

    rows = FileHandler(str(path), "orc").load_data()
    assert [type(r[0]) for r in rows] == [str, str]
    assert rows[0][0].startswith("2024-05-01")


def test_orc_write_data_round_trips_through_the_handler(tmp_path):
    """``BaseConnector.write_data`` is abstract, so an ORC connector has to
    implement it to be instantiable at all; ``orc`` is also a declared
    destination format in ``planner._DEST_EXT_TO_FORMAT``."""
    out = tmp_path / "written.orc"
    FileHandler.write_data(
        {"columns": ORC_COLUMNS, "rows": ORC_ROWS}, str(out), "orc"
    )

    assert out.exists(), "FileHandler.write_data did not dispatch output_type='orc'"
    handler = FileHandler(str(out), "orc")
    assert handler.get_columns() == ORC_COLUMNS
    assert handler.load_data() == ORC_ROWS


# ---------------------------------------------------------------------------
# Defect 2 -- the CSV connector did not sniff the delimiter.
# ---------------------------------------------------------------------------
TSV_HEADER = "region\tunits_sold\trevenue_usd\treported_on\tchannel\n"


def test_tsv_gives_five_separate_columns(tmp_path):
    """Before the fix get_columns() returned the single tab-joined name
    ['region\\tunits_sold\\trevenue_usd\\treported_on\\tchannel']."""
    path = tmp_path / "quarterly_sales.tsv"
    path.write_text(
        TSV_HEADER
        + "north\t10\t100.5\t2024-01-01\tweb\n"
        + "south\t20\t200.25\t2024-01-02\tretail\n"
    )

    handler = FileHandler(str(path), "csv")
    assert handler.get_columns() == [
        "region", "units_sold", "revenue_usd", "reported_on", "channel",
    ]
    assert handler.get_row_count() == 2
    assert handler.get_row(0) == ["north", 10, 100.5, "2024-01-01", "web"]
    assert handler.load_data() == [
        ["north", 10, 100.5, "2024-01-01", "web"],
        ["south", 20, 200.25, "2024-01-02", "retail"],
    ]


def test_tsv_is_also_reachable_by_its_own_file_type(tmp_path):
    """``tsv`` is in SUPPORTED_UPLOAD_EXTS, so FileHandler should accept it as
    a file_type and not only via the 'csv' alias."""
    path = tmp_path / "quarterly_sales.tsv"
    path.write_text(TSV_HEADER + "north\t10\t100.5\t2024-01-01\tweb\n")

    assert FileHandler(str(path), "tsv").get_columns()[0] == "region"


def test_comma_csv_is_unaffected_by_the_sniffer(tmp_path):
    """Regression: a comma file whose *values* contain tabs must still read as
    comma-separated, and the tab must stay inside the field."""
    path = tmp_path / "notes.csv"
    path.write_text(
        "region,note,units_sold\n"
        'north,"first\tsecond\tthird",10\n'
        'south,"a\tb",20\n'
    )

    handler = FileHandler(str(path), "csv")
    assert handler.get_columns() == ["region", "note", "units_sold"]
    assert handler.get_row(0) == ["north", "first\tsecond\tthird", 10]
    assert handler.get_row_count() == 2


def test_plain_comma_csv_still_round_trips(tmp_path):
    """The shape tests/test_connectors.py already relies on."""
    path = tmp_path / "simple.csv"
    CSVConnector.write_data({"columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}, str(path))

    conn = CSVConnector(str(path))
    assert conn.get_columns() == ["a", "b"]
    assert conn.get_row_count() == 2
    assert conn.load_data() == [[1, 2], [3, 4]]


def test_semicolon_csv_matches_the_cli_sniffer(tmp_path):
    """``avaloka/io/loader.py::_sniff_delimiter`` counts , ; \\t | in the header
    and takes the most frequent; the app-side reader must agree, or the same
    European-export file profiles differently through the API than the CLI."""
    path = tmp_path / "export.csv"
    path.write_text("region;units_sold;revenue_usd\nnorth;10;100.5\n")

    handler = FileHandler(str(path), "csv")
    assert handler.get_columns() == ["region", "units_sold", "revenue_usd"]
    assert handler.get_row(0) == ["north", 10, 100.5]


def test_pipe_delimited_file_matches_the_cli_sniffer(tmp_path):
    path = tmp_path / "export.csv"
    path.write_text("region|units_sold|revenue_usd\nnorth|10|100.5\n")

    assert FileHandler(str(path), "csv").get_columns() == [
        "region", "units_sold", "revenue_usd",
    ]


def test_single_column_file_falls_back_to_comma(tmp_path):
    """No candidate delimiter in the header: the CLI defaults to ',' rather
    than guessing, and so must this. csv.Sniffer raises here instead."""
    path = tmp_path / "one.csv"
    path.write_text("region\nnorth\nsouth\n")

    handler = FileHandler(str(path), "csv")
    assert handler.get_columns() == ["region"]
    assert handler.load_data() == [["north"], ["south"]]


def test_the_app_reader_does_not_import_the_cli_package():
    """file_handler must not depend on avaloka; the sniffer is duplicated on
    purpose, not imported across the app/CLI boundary."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "file_handler"
    offenders = [
        p.name
        for p in root.glob("*.py")
        if "import avaloka" in p.read_text() or "from avaloka" in p.read_text()
    ]
    assert offenders == [], f"file_handler modules importing avaloka: {offenders}"


# ---------------------------------------------------------------------------
# Defect 3 -- the unsupported-format error named the filename.
# ---------------------------------------------------------------------------
def test_unsupported_extension_error_names_the_extension(tmp_path):
    """Before the fix a file named 'annual_summary.sav' reported
    "annual_summary files aren't supported" -- the stem, not the extension."""
    path = tmp_path / "annual_summary.sav"
    path.write_text("whatever")

    with pytest.raises(TypeError) as excinfo:
        FileHandler(str(path), "sav")

    message = str(excinfo.value)
    assert "sav" in message, message
    assert "annual_summary" not in message, (
        f"the error still names the filename rather than the extension: {message}")


def test_unsupported_extension_error_says_what_is_supported(tmp_path):
    path = tmp_path / "annual_summary.sav"
    path.write_text("whatever")

    with pytest.raises(TypeError) as excinfo:
        FileHandler(str(path), "sav")

    message = str(excinfo.value)
    for expected in ("csv", "parquet", "orc"):
        assert expected in message, (
            f"the error does not tell the user {expected!r} is supported: {message}")


def test_a_genuinely_unsupported_type_still_raises(tmp_path):
    """The fix must not have made everything 'supported'."""
    path = tmp_path / "archive.zip"
    path.write_text("not a table")

    with pytest.raises(TypeError):
        FileHandler(str(path), "zip")

    assert FileHandler.get_connector(str(path), "zip") is None


def test_error_falls_back_to_the_file_type_when_there_is_no_extension(tmp_path):
    """delta and iceberg tables are directories, so a path can carry no
    suffix at all; the message must still name something useful."""
    target = tmp_path / "some_table"
    target.mkdir()

    with pytest.raises(TypeError) as excinfo:
        FileHandler(str(target), "hudi")

    assert "hudi" in str(excinfo.value), str(excinfo.value)
