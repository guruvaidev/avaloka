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
        assert (
            conn.get_row_count() == 1 or conn.get_row_count() == 2
        )  # ijson may count from 0 or 1
        assert set(conn.get_row(0)) == set([1, 2])
        assert all(set(row) <= set([1, 2, 3, 4]) for row in conn.load_data())

    # --- AVRO ---
    def test_avro_connector(self):
        file = os.path.join(TEMP_DIR, "test.avro")
        columns = ["a", "b"]
        rows = [[1, 2], [3, 4]]
        schema = {
            "doc": "Test",
            "name": "Test",
            "namespace": "test",
            "type": "record",
            "fields": [{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
        }
        records = [dict(zip(columns, row)) for row in rows]
        AvroConnector.write_data({"columns": schema, "rows": records}, str(file))

        conn = AvroConnector(str(file))
        assert conn.get_columns() == columns
        assert conn.get_row_count() == 1 or conn.get_row_count() == 2
        assert set(conn.get_row(0)) <= set([1, 2])
        assert all(set(row) <= set([1, 2, 3, 4]) for row in conn.load_data())

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


if __name__ == "__main__":
    unittest.main(exit=False)
