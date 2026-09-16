import os
from abc import ABC, abstractmethod
from typing import Any
from typing_extensions import TypedDict


class OutputData(TypedDict):
    columns: list[str]
    rows: list[list[Any]]


class BaseConnector(ABC):
    def __init__(self, path: str, is_sql: bool = False) -> None:
        self.path = path
        self.is_sql = is_sql

    def get_file_size(self) -> int:
        """Gets the file size"""
        return os.path.getsize(self.path)

    def list_tables(self) -> list[str]:
        """Returns a list of tables if the file is an sql-type file"""
        if self.is_sql:
            raise NotImplemented

    def select_table(self, table: str) -> bool:
        """Selects a table if the file is an sql-type file"""
        if self.is_sql:
            raise NotImplemented
        return True

    @abstractmethod
    def get_columns(self) -> list[str]:
        """Get the column names"""
        pass

    @abstractmethod
    def get_row_count(self) -> int:
        """Get the total row count"""
        pass

    @abstractmethod
    def get_row(self, row: int) -> list[Any]:
        """Gets a specific row"""
        pass

    @abstractmethod
    def sample_rows(
        self, num_rows: int = 10, random_rows: bool = True
    ) -> list[list[Any]]:
        """Samples rows from the given dataset"""
        pass

    @abstractmethod
    def load_data(self) -> list[list[Any]]:
        """Loads all the data"""
        pass
    @staticmethod
    @abstractmethod
    def write_data(data: OutputData, output_path: str) -> None:
        """Writes data to the given output path"""
        pass
