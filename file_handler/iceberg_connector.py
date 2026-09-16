import random
from copy import copy
from typing import Any

from pyiceberg.catalog import load_catalog

from file_handler.base_connector import OutputData

from .base_connector import BaseConnector


class IcebergConnector(BaseConnector):
    def __init__(self, path: str, properties: dict) -> None:
        super().__init__(path, is_sql=True)
        self._catalog = load_catalog("local", **properties)
        self._current_table = None

    def list_tables(self) -> list[str]:
        tables = []
        for namespace in self._catalog.list_namespaces():
            for table in self._catalog.list_tables(namespace):
                tables.append(".".join(table))
        return tables

    def select_table(self, table: str) -> bool:
        try:
            self._current_table = self._catalog.load_table(table)
        except Exception:
            return False

        return True

    def get_columns(self) -> list[str]:
        return [field.name for field in self._current_table.schema().fields]

    def get_row_count(self) -> int:
        if self._current_table is not None:
            return self._current_table.scan().count()
        return -1

    def get_row(self, row: int) -> list[Any]:
        if not self._current_table:
            return []

        cumulative_rows = 0
        for batch in self._current_table.scan().to_arrow().to_batches():
            df = batch.to_pandas()
            if row < cumulative_rows + len(df):
                for col in df.select_dtypes(include=["datetime"]):
                    df[col] = df[col].astype(str)
                return (
                    df.iloc[row - cumulative_rows]
                    .map(lambda x: x.item() if hasattr(x, "item") else x)
                    .tolist()
                )
            cumulative_rows += len(df)

        return []

    def sample_rows(self, num_rows: int = 10, random_rows: bool = True):
        total_rows = self.get_row_count()

        rows_to_sample = set(
            random.sample(range(total_rows - 1), num_rows)
            if random_rows
            else list(range(num_rows))
        )

        data = []
        cumulative_rows = 0
        for batch in self._current_table.scan().to_arrow().to_batches():
            df = batch.to_pandas()
            for col in df.select_dtypes(include=["datetime"]):
                df[col] = df[col].astype(str)

            size = len(df)
            cumulative_size = size + cumulative_rows
            for row in copy(rows_to_sample):
                if row < cumulative_size:
                    data.append(
                        df.iloc[row - cumulative_rows]
                        .map(lambda x: x.item() if hasattr(x, "item") else x)
                        .tolist()
                    )
                    rows_to_sample.discard(row)

            if not rows_to_sample:
                break
            cumulative_rows += len(df)

        return data

    def load_data(self) -> None:
        return self._current_table.scan().to_pandas().values.tolist()

    @staticmethod
    def write_data(self, data: OutputData, output_path: str) -> None:
        raise NotImplementedError("Iceberg write_data is not implemented yet.")
