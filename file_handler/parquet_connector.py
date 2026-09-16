import random
from copy import copy
from typing import Any

import pandas as pd
from fastparquet import ParquetFile
import pyarrow as pa
import pyarrow.parquet as pq

from .base_connector import BaseConnector, OutputData


class ParquetConnector(BaseConnector):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self._parquet_file = ParquetFile(self.path)

    def get_columns(self) -> list[str]:
        return self._parquet_file.columns

    def get_row_count(self) -> int:
        return self._parquet_file.count()

    def get_row(self, row: int) -> list[Any]:
        cumulative_rows = 0

        for df in self._parquet_file.iter_row_groups():
            num_rows = len(df)
            if cumulative_rows + num_rows > row:
                for col in df.select_dtypes(include=["datetime"]):
                    df[col] = df[col].astype(str)

                return (
                    df.iloc[row - cumulative_rows]
                    .map(lambda x: x.item() if hasattr(x, "item") else x)
                    .tolist()
                )

            cumulative_rows += num_rows
        return []

    def sample_rows(
        self,
        num_rows: int = 10,
        random_rows: bool = True,
    ):
        total_rows = self.get_row_count()

        rows_to_sample = set(
            random.sample(range(total_rows - 1), k=num_rows)
            if random_rows
            else range(num_rows)
        )

        data = []
        cumulative_rows = 0
        for df in self._parquet_file.iter_row_groups():
            size = len(df)
            cumulative_size = size + cumulative_rows
            for row in copy(rows_to_sample):
                for col in df.select_dtypes(include=["datetime"]):
                    df[col] = df[col].astype(str)

                if row < cumulative_size:
                    data.append(
                        df.iloc[row - cumulative_rows]
                        .map(lambda x: x.item() if hasattr(x, "item") else x)
                        .tolist()
                    )
                    rows_to_sample.discard(row)

                if not rows_to_sample:
                    break
            cumulative_rows += size
        return data

    def load_data(self) -> None:
        df = self._parquet_file.to_pandas()
        for col in df.select_dtypes(include=["datetime"]):
            df[col] = df[col].astype(str)
        return df.map(lambda x: x.item() if hasattr(x, "item") else x).values.tolist()

    @staticmethod
    def write_data(data: OutputData, output_path: str) -> None:
        out = []
        for row in data["rows"]:
            out.append(
                {
                    col: row[i] 
                    for i, col in enumerate(data["columns"])
                }
            )
        table = pa.Table.from_pylist(out)
        pq.write_table(table, output_path)
