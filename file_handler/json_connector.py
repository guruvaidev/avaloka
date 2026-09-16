import random
import json
from itertools import islice
from typing import Any

import ijson

from .base_connector import BaseConnector, OutputData


class JSONConnector(BaseConnector):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self._row_count = None

    def get_columns(self) -> list[str]:
        cols = {}
        with open(self.path) as f:
            parser = ijson.parse(f)
            for _, event, value in parser:
                if event == "map_key":
                    cols[value] = None

        return list(cols.keys())

    def get_row_count(self) -> int:
        if self._row_count is None:
            with open(self.path) as f:
                for self._row_count, _ in enumerate(ijson.items(f, "item")):
                    ...

        return self._row_count

    def get_row(self, row: int) -> list[Any]:
        with open(self.path) as f:
            reader = ijson.items(f, "item")
            try:
                row: list[str] = next(islice(reader, row , row+1))
            except StopIteration:
                return None

        return list(row.values())

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
            reader = ijson.items(f, "item")

            prev = 0
            for i in rows_to_sample:
                start = i - prev
                try:
                    row: list[str] = next(islice(reader, start - 1, start))
                except StopIteration:
                    break
                data.append(list(row.values()))
                prev = i

        return data

    def load_data(self) -> list[list[Any]]:
        with open(self.path) as f:
            data = [list(row.values()) for row in ijson.items(f, "item")]
        return data

    @staticmethod
    def write_data(data: OutputData, output_path: str) -> None:
        converted_data = [dict(zip(data["columns"], row)) for row in data["rows"]]
        with open(output_path, "w") as f:
            json.dump(converted_data, f, indent=4)
