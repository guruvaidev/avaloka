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
        self._delimiter = None

    @staticmethod
    def _sniff_delimiter(path: str) -> str:
        """Pick the most likely delimiter from the header line.

        Without this every reader below used csv.reader's default comma, so a
        tab-separated upload -- .tsv is one of the nine extensions /api/upload
        declares -- came back as a single column whose name was the entire
        header line. Nothing raised; a one-column frame profiles like a real
        table, which is how it went unnoticed.

        The rule is deliberately the same one as
        avaloka/io/loader.py::_sniff_delimiter -- count , ; \t | in the header
        and take the most frequent, defaulting to ',' -- so the same file reads
        the same way through the API and through the CLI. It is duplicated
        rather than imported because file_handler must not depend on the CLI
        package. csv.Sniffer was the alternative and was rejected: it raises
        _csv.Error on a genuinely single-column file, and it infers from a data
        sample, so it can pick a delimiter out of punctuation inside free-text
        values. Counting a fixed candidate set fails in the direction that
        leaves existing comma behaviour alone.

        Only the header is inspected, so a tab or semicolon inside a quoted
        *value* cannot change the choice. A quoted header field containing one
        still can; the CLI has the same blind spot, and keeping the two in step
        matters more than papering over it on one side.
        """
        with open(path, errors="replace") as f:
            header = f.readline()
        candidates = {d: header.count(d) for d in [",", ";", "\t", "|"]}
        best = max(candidates, key=candidates.get)
        return best if candidates[best] > 0 else ","

    @property
    def delimiter(self) -> str:
        """The sniffed delimiter, read from the header once on first use.

        Lazy so that constructing a connector still touches no I/O.
        """
        if self._delimiter is None:
            self._delimiter = CSVConnector._sniff_delimiter(self.path)
        return self._delimiter

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
            reader = csv.reader(f, delimiter=self.delimiter)
            return next(reader, [])

    def get_row_count(self) -> int:
        if self._row_count is None:
            with open(self.path) as f:
                reader = csv.reader(f, delimiter=self.delimiter)
                for _row_count, _ in enumerate(reader):
                    ...
        return _row_count

    def get_row(self, row: int) -> list[Any]:
        with open(self.path) as f:
            reader = csv.reader(f, delimiter=self.delimiter)
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
            reader = csv.reader(f, delimiter=self.delimiter)
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
            reader = csv.reader(f, delimiter=self.delimiter)
            next(reader, None)
            data = [list(map(CSVConnector._convert_type, row)) for row in reader]
        return data

    @staticmethod
    def write_data(data: OutputData, output_path: str) -> None:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data["columns"])
            writer.writerows(data["rows"])
