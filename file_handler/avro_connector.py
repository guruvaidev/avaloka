import os
import random
from itertools import islice
from typing import Any

import fastavro

from .base_connector import BaseConnector, OutputData


class AvroConnector(BaseConnector):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self._row_count = None

    def get_columns(self) -> list[str]:
        with open(self.path, "rb") as f:
            reader = fastavro.reader(f)
            return [field["name"] for field in reader.schema["fields"]]

    def get_row_count(self) -> int:
        if self._row_count is None:
            with open(self.path, "rb") as f:
                self._row_count = sum(1 for _ in fastavro.reader(f))
        return self._row_count

    def get_row(self, row: int) -> list[Any]:
        with open(self.path, "rb") as f:
            reader = fastavro.reader(f)
            try:
                row = next(islice(reader, row, row+1))
            except StopIteration:
                return None

            return list(row.values())

    def sample_rows(self, num_rows: int = 10, random_rows: bool = True):
        total_rows = self.get_row_count()

        # 1-based indices 1..total_rows (the streaming loop below maps index i to
        # the (i-1)-th record). Cover every row, and never draw more than exist.
        k = min(num_rows, total_rows)
        rows_to_sample = sorted(
            random.sample(range(1, total_rows + 1), k)
            if random_rows
            else range(1, k + 1)
        )

        data = []
        with open(self.path, "rb") as f:
            reader = fastavro.reader(f)

            prev = 0
            for i in rows_to_sample:
                start = i - prev
                try:
                    row = next(islice(reader, start - 1, start))
                except StopIteration:
                    break
                data.append(list(row.values()))
                prev = i

        return data

    def load_data(self) -> list[Any]:
        with open(self.path, "rb") as f:
            data = [list(row.values()) for row in fastavro.reader(f)]
        return data

    @staticmethod
    def write_data(data: OutputData, output_path: str) -> None:
        columns = [str(c) for c in data["columns"]]
        rows = [list(r) for r in data["rows"]]

        def _col_type(idx: int) -> str:
            # Infer one Avro primitive for the column from its non-null values.
            # Mixed int/float -> double; anything else mixed -> string.
            t = None
            for row in rows:
                v = row[idx] if idx < len(row) else None
                if v is None:
                    continue
                if isinstance(v, bool):
                    cur = "boolean"
                elif isinstance(v, int):
                    cur = "long"
                elif isinstance(v, float):
                    cur = "double"
                else:
                    cur = "string"
                if t is None:
                    t = cur
                elif t != cur:
                    t = "double" if {t, cur} <= {"long", "double"} else "string"
            return t or "string"

        col_types = [_col_type(i) for i in range(len(columns))]

        schema = fastavro.parse_schema({
            "type": "record",
            "name": "Root",
            # ["null", <type>] so blank cells are allowed.
            "fields": [
                {"name": col, "type": ["null", col_types[i]]}
                for i, col in enumerate(columns)
            ],
        })

        def _coerce(v, t):
            if v is None:
                return None
            if t == "string":
                return str(v)
            if t == "double":
                return float(v)
            if t == "long":
                return int(v)
            if t == "boolean":
                return bool(v)
            return v

        records = [
            {
                columns[i]: _coerce(row[i] if i < len(row) else None, col_types[i])
                for i in range(len(columns))
            }
            for row in rows
        ]

        with open(output_path, "wb") as out:
            fastavro.writer(out, schema, records)
