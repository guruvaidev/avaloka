import os
from pathlib import Path
from typing import Any

from .base_connector import BaseConnector, OutputData
from .csv_connector import CSVConnector
from .excel_connector import ExcelConnector
from .avro_connector import AvroConnector
from .json_connector import JSONConnector
from .parquet_connector import ParquetConnector
from .delta_connector import DeltaConnector
from .iceberg_connector import IcebergConnector
from .xml_connector import XMLConnector 


class FileHandler:
    def __init__(self, path: str, file_type: str, *args, **kwargs) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError

        self.path = path
        self.connector: BaseConnector | None = FileHandler.get_connector(
            path, file_type, *args, **kwargs
        )
        if self.connector is None:
            ext = Path(path).stem
            raise TypeError(f"{ext} files aren't supported")

    @staticmethod
    def get_connector(
        path: str,
        file_type: str,
        iceberg_properties: dict = None,
        sheet_name: str | int | None = None,
    ) -> BaseConnector | None:
        if file_type == "csv":
            return CSVConnector(path)
        if file_type in ("excel", "xlsx", "xls"):
            return ExcelConnector(path, sheet_name=sheet_name)
        if file_type == "avro":
            return AvroConnector(path)
        if file_type == "json":
            return JSONConnector(path)
        if file_type == "parquet":
            return ParquetConnector(path)
        if file_type == "delta":
            return DeltaConnector(path)
        if file_type == "iceberg":
            return IcebergConnector(path, iceberg_properties)
        if file_type == "xml":                     # ← add this case
            return XMLConnector(path)
        return None

    def list_tables(self) -> list[str]:
        """Returns a list of tables if the file is an sql-type file"""
        return self.connector.list_tables()

    def select_table(self, table: str) -> bool:
        """Selects a table if the file is an sql-type file"""
        return self.connector.select_table(table=table)

    def get_columns(self) -> list[str]:
        """Get the column names"""
        return self.connector.get_columns()

    def get_row_count(self) -> int:
        """Get the total row count"""
        return self.connector.get_row_count()

    def get_row(self, row: int) -> list[Any]:
        """Gets a specific row"""
        return self.connector.get_row(row=row)

    def sample_rows(
        self, num_rows: int = 10, random_rows: bool = True
    ) -> list[list[Any]]:
        """Samples rows from the given dataset"""
        return self.connector.sample_rows(num_rows=num_rows, random_rows=random_rows)

    def load_data(self) -> list[list[Any]]:
        """Loads all the data"""
        return self.connector.load_data()

    @staticmethod
    def write_data(data: OutputData, output_path: str, output_type: str) -> None:
        """Writes data to the given output path"""

        if output_type == "csv":
            return CSVConnector.write_data(data, output_path)
        if output_type in ("excel", "xlsx", "xls"):
            return ExcelConnector.write_data(data, output_path)
        if output_type == "avro":
            return AvroConnector.write_data(data, output_path)
        if output_type == "json":
            return JSONConnector.write_data(data, output_path)
        if output_type == "parquet":
            return ParquetConnector.write_data(data, output_path)
        if output_type == "delta":
            return DeltaConnector.write_data(data, output_path)
