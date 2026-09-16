import random
from copy import copy
from typing import Any

from deltalake import DeltaTable
from deltalake.writer import write_deltalake
import pandas as pd

from .base_connector import BaseConnector, OutputData


class DeltaConnector(BaseConnector):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.table = DeltaTable(path)

    def get_columns(self) -> list[str]:
        return [field.name for field in self.table.schema().fields]

    def get_row_count(self) -> int:
        return self.table.to_pyarrow_dataset().count_rows()

    def get_row(self, row: int) -> list[Any]:
        cumulative_rows = 0
        for batch in self.table.to_pyarrow_dataset().to_batches(batch_size=10_000):
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

    def sample_rows(
        self,
        num_rows: int = 10,
        random_rows: bool = True,
    ) -> list[list[Any]]:
        total_rows = self.get_row_count()

        rows_to_sample = set(
            random.sample(range(total_rows - 1), num_rows)
            if random_rows
            else list(range(num_rows))
        )

        data = []
        cumulative_rows = 0
        for batch in self.table.to_pyarrow_dataset().to_batches(batch_size=10_000):
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

    def load_data(self) -> list[list[Any]]:
        data = []
        for batch in self.table.to_pyarrow_dataset().to_batches(batch_size=50_000):
            for row in batch.to_pylist():
                data.append(row.values())
        return data

    @staticmethod
    def write_data(data: OutputData, output_path: str) -> None:
        out = {
            col: [row[i] for row in data["rows"]]
            for i, col in enumerate(data["columns"])
        }
        write_deltalake(output_path, pd.DataFrame(out), mode="overwrite")
