import os
import json
import pytest
import unittest

from file_handler.csv_connector import CSVConnector
from file_handler.json_connector import JSONConnector
from file_handler.avro_connector import AvroConnector
from file_handler.excel_connector import ExcelConnector
from file_handler.parquet_connector import ParquetConnector
from file_handler.delta_connector import DeltaConnector
from file_handler.iceberg_connector import IcebergConnector
from file_handler.xml_connector import XMLConnector


TEMP_DIR = "/tmp"


class TestConnectors(unittest.TestCase):
    # --- CSV ---
    def test_csv_connector(self):
        file = os.path.join(TEMP_DIR, "test.csv")

        columns = ["a", "b"]
        rows = [[1, 2], [3, 4]]
        CSVConnector.write_data({"columns": columns, "rows": rows}, str(file))

        conn = CSVConnector(str(file))
        assert conn.get_columns() == columns
        assert conn.get_row_count() == 2
        assert conn.get_row(0) == [1, 2]
        assert conn.load_data() == rows

    # --- EXCEL ---
    def test_excel_connector(self):
        file = os.path.join(TEMP_DIR, "test.xlsx")

        columns = ["a", "b"]
        rows = [[1, 2], [3, 4]]

        ExcelConnector.write_data({"columns": columns, "rows": rows}, str(file))

        conn = ExcelConnector(str(file))
        assert conn.get_columns() == columns
        assert conn.get_row_count() == 2
        assert conn.get_row(0) == [1, 2]
        assert conn.load_data() == rows

    # --- JSON ---
    def test_json_connector(self):
        file = os.path.join(TEMP_DIR, "test.json")
        columns = ["a", "b"]
        rows = [[1, 2], [3, 4]]
        JSONConnector.write_data({"columns": columns, "rows": rows}, str(file))

        conn = JSONConnector(str(file))
        assert set(conn.get_columns()) == set(columns)
        # The row count used to be asserted as "== 1 or == 2" ("ijson may count
        # from 0 or 1"), which accepted the MAJ-114 off-by-one (get_row_count
        # returns the last enumerate index, 1 for two rows). The exact count is
        # now a strict xfail tied to MAJ-114: test_json_connector_row_count_is_exact.
        assert set(conn.get_row(0)) == set([1, 2])
        assert all(set(row) <= set([1, 2, 3, 4]) for row in conn.load_data())

    # --- AVRO ---
    def test_avro_connector(self):
        # write_data takes OutputData -- {"columns": [names], "rows": [[values]]}
        # -- exactly as every other connector above. This test used to hand it a
        # raw Avro record schema as "columns" and dicts as "rows", which is an
        # older Avro-only signature; the current code iterated the schema dict
        # and wrote a file whose columns were ['doc','name','namespace','type',
        # 'fields']. The assertions below had been loosened to tolerate that
        # ("== 1 or == 2", subset checks), which hid it further.
        file = os.path.join(TEMP_DIR, "test.avro")
        columns = ["a", "b"]
        rows = [[1, 2], [3, 4]]
        AvroConnector.write_data({"columns": columns, "rows": rows}, str(file))

        conn = AvroConnector(str(file))
        assert conn.get_columns() == columns
        assert conn.get_row_count() == 2
        assert conn.get_row(0) == [1, 2]
        assert conn.load_data() == rows

    # --- PARQUET ---
    def test_parquet_connector(self):
        import pandas as pd

        file = os.path.join(TEMP_DIR, "test.parquet")
        columns = ["a", "b"]
        rows = [[1, 2], [3, 4]]

        ParquetConnector.write_data({"columns": columns, "rows": rows}, str(file))

        conn = ParquetConnector(str(file))
        assert conn.get_columns() == columns
        assert conn.get_row_count() == 2
        assert set(conn.get_row(0)) == set([1, 2])
        assert all(set(row) <= set([1, 2, 3, 4]) for row in conn.load_data())

    # --- XML ---
    def test_xml_connector(self):
        file = os.path.join(TEMP_DIR, "test.xml")
        columns = ["orderid", "ordername"]
        rows = [["1", "foo"], ["2", "bar"]]

        XMLConnector.write_data({"columns": columns, "rows": rows}, str(file))

        conn = XMLConnector(str(file))
        assert set(conn.get_columns()) == set(["orderid", "ordername"])
        assert conn.get_row_count() == 2
        assert conn.get_row(0) == [1, "foo"]
        assert conn.load_data() == [[1, "foo"], [2, "bar"]]

    # --- DELTA ---
    def test_delta_connector(self):
        file = os.path.join(TEMP_DIR, "delta")
        columns = ["a", "b"]
        rows = [[1, 2], [3, 4]]
        DeltaConnector.write_data({"columns": columns, "rows": rows}, str(file))

        conn = DeltaConnector(str(file))
        assert conn.get_columns() == columns
        assert conn.get_row_count() == 2
        assert set(conn.get_row(0)) == set([1, 2])
        assert all(set(row) <= set([1, 2, 3, 4]) for row in conn.load_data())


@pytest.mark.defect
@pytest.mark.xfail(strict=True, reason=(
    "MAJ-114: JSONConnector.get_row_count returns the last enumerate index "
    "(file_handler/json_connector.py:26-30), one less than the number of rows"))
def test_json_connector_row_count_is_exact(tmp_path):
    file = tmp_path / "rows.json"
    JSONConnector.write_data({"columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}, str(file))
    assert JSONConnector(str(file)).get_row_count() == 2


if __name__ == "__main__":
    unittest.main(exit=False)
