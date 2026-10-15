import random
from typing import Any, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc

from .base_connector import BaseConnector, OutputData


class OrcConnector(BaseConnector):
    """Reads ORC files, which /api/upload declares in SUPPORTED_UPLOAD_EXTS.

    ORC groups rows into stripes the way Parquet groups them into row groups,
    so this follows ParquetConnector: the footer answers the metadata questions
    without reading any data, and row access walks one stripe at a time so
    fetching a single row never costs a full-file read.

    pyarrow reads and writes ORC, and pyarrow is already a dependency of both
    the app and the data-transfer image, so no new library is involved.
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self._orc_file = orc.ORCFile(self.path)

    def _iter_stripes(self) -> Iterator[pd.DataFrame]:
        """Yields each stripe as a DataFrame.

        The ORC analogue of ``ParquetFile.iter_row_groups``, which pyarrow does
        not provide: stripes are only reachable by index.
        """
        for stripe in range(self._orc_file.nstripes):
            yield self._orc_file.read_stripe(stripe).to_pandas()

    @staticmethod
    def _stringify_datetimes(df: pd.DataFrame) -> pd.DataFrame:
        """Timestamps become strings, as in the parquet connector, so the rows
        stay JSON-serialisable for the API response."""
        for col in df.select_dtypes(include=["datetime", "datetimetz"]):
            df[col] = df[col].astype(str)
        return df

    def get_columns(self) -> list[str]:
        return list(self._orc_file.schema.names)

    def get_row_count(self) -> int:
        return self._orc_file.nrows

    def get_row(self, row: int) -> list[Any]:
        cumulative_rows = 0

        for df in self._iter_stripes():
            num_rows = len(df)
            if cumulative_rows + num_rows > row:
                df = OrcConnector._stringify_datetimes(df)

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
    ) -> list[list[Any]]:
        total_rows = self.get_row_count()
        if not total_rows:
            return []

        # Clamped, unlike the parquet connector's `range(total_rows - 1)`: the
        # sampling agent asks for a fixed n without knowing the row count, and
        # on a short file that call raises instead of returning what exists.
        k = min(num_rows, total_rows)
        rows_to_sample = set(
            random.sample(range(total_rows), k=k) if random_rows else range(k)
        )

        data = []
        cumulative_rows = 0
        for df in self._iter_stripes():
            size = len(df)
            cumulative_size = size + cumulative_rows
            df = OrcConnector._stringify_datetimes(df)

            for row in sorted(r for r in rows_to_sample if r < cumulative_size):
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

    def load_data(self) -> list[list[Any]]:
        df = self._orc_file.read().to_pandas()
        df = OrcConnector._stringify_datetimes(df)
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
        orc.write_table(table, output_path)
