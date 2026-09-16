import csv
import random
import os
from itertools import islice
from typing import Any

from .base_connector import BaseConnector, OutputData


class CSVConnector(BaseConnector):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self._row_count = None

    @staticmethod
    def _convert_type(data: str) -> int | float | str:
        if data.isnumeric():
            return int(data)

        try:
            result = float(data)
        except ValueError:
            return data
        else:
            return result

    def get_columns(self) -> list[str]:
        with open(self.path) as f:
            reader = csv.reader(f)
            return next(reader, [])

    def get_row_count(self) -> int:
        if self._row_count is None:
            with open(self.path) as f:
                reader = csv.reader(f)
                for _row_count, _ in enumerate(reader):
                    ...
        return _row_count

    def get_row(self, row: int) -> list[Any]:
        with open(self.path) as f:
            reader = csv.reader(f)
            next(reader, None)
            try:
                row: list[str] = next(islice(reader, row, row + 1))
            except StopIteration:
                return None

        return list(map(CSVConnector._convert_type, row))

    def sample_rows(
        self, num_rows: int = 10, random_rows: bool = True
    ) -> list[list[Any]]:
        total_rows = self.get_row_count()

        rows_to_sample = sorted(
            random.sample(range(1, total_rows), k=num_rows)
            if random_rows
            else range(1, num_rows + 1)
        )

        data = []
        with open(self.path) as f:
            reader = csv.reader(f)
            next(reader, None)

            prev = 0
            for i in rows_to_sample:
                start = i - prev
                try:
                    row: list[str] = next(islice(reader, start - 1, start))
                except StopIteration:
                    break
                data.append(list(map(CSVConnector._convert_type, row)))
                prev = i

        return data

    def load_data(self) -> list[list[Any]]:
        with open(self.path) as f:
            reader = csv.reader(f)
            next(reader, None)
            data = [list(map(CSVConnector._convert_type, row)) for row in reader]
        return data

    @staticmethod
    def write_data(data: OutputData, output_path: str) -> None:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data["columns"])
            writer.writerows(data["rows"])
