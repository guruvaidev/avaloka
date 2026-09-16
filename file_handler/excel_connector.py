import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base_connector import BaseConnector, OutputData
from .utils import convert_type


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the CSV path: blank/whitespace-only cells -> NA, and
    best-effort numeric coercion for text columns.

    Text columns are ``object`` dtype on pandas < 3 but the dedicated ``str``
    dtype on pandas >= 3, so match both — an object-only check silently skips
    every text column on pandas 3.
    """
    for col in df.columns:
        s = df[col]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            s = s.replace(r"^\s*$", pd.NA, regex=True)
            try:
                s = pd.to_numeric(s)
            except (ValueError, TypeError):
                pass
            df[col] = s
    return df


class ExcelConnector(BaseConnector):
    def __init__(self, path: str, sheet_name: str | int | None = None) -> None:
        super().__init__(path)
        self._df: pd.DataFrame | None = None
        # The sheet the caller explicitly asked for (name or index). When None we
        # auto-resolve to the sheet that actually holds the data (see below).
        self._requested_sheet = sheet_name
        self._selected_sheet: str | None = None
        self._sheet_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Sheet resolution
    # ------------------------------------------------------------------
    def list_sheets(self) -> list[str]:
        """Return every sheet name in the workbook."""
        if self._sheet_names is None:
            self._sheet_names = pd.ExcelFile(self.path).sheet_names
        return self._sheet_names

    @property
    def selected_sheet(self) -> str:
        """Name of the sheet actually being read (resolves lazily)."""
        if self._selected_sheet is None:
            self._selected_sheet = self._resolve_sheet()
        return self._selected_sheet

    def _resolve_sheet(self) -> str:
        """Decide which sheet to read.

        A workbook uploaded for analysis often has several sheets (raw data,
        pivots, notes, a cover/summary sheet, …). ``pd.read_excel`` defaults to
        the *first* sheet, which is frequently NOT the data sheet and silently
        makes the analysis run on the wrong table. When the caller does not pin
        a sheet we pick the one with the most data rows, which is almost always
        the real dataset.
        """
        sheets = self.list_sheets()

        # Caller pinned a sheet explicitly.
        if self._requested_sheet is not None:
            if isinstance(self._requested_sheet, int):
                return sheets[self._requested_sheet]
            if self._requested_sheet in sheets:
                return self._requested_sheet
            raise ValueError(
                f"Sheet '{self._requested_sheet}' not found. "
                f"Available sheets: {sheets}"
            )

        if len(sheets) == 1:
            return sheets[0]

        # Auto-pick the sheet with the most data rows.
        best_sheet, best_rows = sheets[0], -1
        xl = pd.ExcelFile(self.path)
        for sheet in sheets:
            try:
                n_rows = len(pd.read_excel(xl, sheet_name=sheet))
            except Exception:
                n_rows = -1
            if n_rows > best_rows:
                best_sheet, best_rows = sheet, n_rows
        return best_sheet

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_excel(self.path, sheet_name=self.selected_sheet)
        return self._df

    def load_dataframe(self, normalize: bool = True) -> pd.DataFrame:
        """Return the sheet as a typed DataFrame (dtypes preserved).

        Unlike :meth:`load_data`, this keeps pandas' inferred dtypes instead of
        flattening everything to a list of Python scalars with stringified
        dates. That preserved typing is what downstream profiling/analysis
        relies on to reason about numeric and datetime columns.
        """
        df = self._load_df().copy()
        if normalize:
            df = _normalize_dataframe(df)
        return df

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, float) and np.isnan(value):
            return None

        # Convert pandas datetime, timestamp, and date-like to string
        if hasattr(value, "isoformat") and not isinstance(value, str):
            return str(value)

        return value

    def get_columns(self) -> list[str]:
        df = self._load_df()
        return [str(c) for c in df.columns.tolist()]

    def get_row_count(self) -> int:
        return len(self._load_df())

    def get_row(self, row: int) -> list[Any] | None:
        df = self._load_df()
        if row < 0 or row >= len(df):
            return None

        values = df.iloc[row].tolist()
        return [ExcelConnector._normalize_value(v) for v in values]

    def sample_rows(
        self, num_rows: int = 10, random_rows: bool = True
    ) -> list[list[Any]]:
        total_rows = self.get_row_count()
        if total_rows == 0:
            return []

        n = min(num_rows, total_rows)
        rows_to_sample = sorted(
            random.sample(range(total_rows), k=n)
            if random_rows
            else list(range(n))
        )
        return [self.get_row(i) for i in rows_to_sample if self.get_row(i) is not None]

    def load_data(self) -> list[list[Any]]:
        total_rows = self.get_row_count()
        return [self.get_row(i) for i in range(total_rows)]

    @staticmethod
    def write_data(data: OutputData, output_path: str) -> None:
        out = {
            col: [row[i] for row in data["rows"]]
            for i, col in enumerate(data["columns"])
        }

        df = pd.DataFrame(out)
        df.to_excel(output_path, index=False)


# ----------------------------------------------------------------------
# Workbook -> per-sheet CSV materialization
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SheetCsvExport:
    sheet_name: str
    csv_path: str
    n_rows: int


def export_workbook_sheets_to_csv(
    path: str,
    out_dir: str | Path,
    max_sheets: int | None = None,
) -> tuple[list[SheetCsvExport], list[str]]:
    """Materialize every non-empty sheet of a workbook as its own CSV file.

    This is how multi-sheet Excel uploads become multiple datasets: each sheet
    is read as a typed DataFrame (same normalization as ``load_dataframe``) and
    written to a CSV under ``out_dir``, so everything downstream (sampling,
    codegen, execution) runs on the CSV-native path instead of silently
    analyzing only the first sheet of the workbook.

    Returns ``(exports, notes)``. Exports are ordered largest sheet first, so
    callers can treat ``exports[0]`` as the primary dataset. Empty or unreadable
    sheets are skipped and recorded in ``notes``; if ``max_sheets`` caps the
    workbook, only the largest ``max_sheets`` sheets are kept and the dropped
    ones are recorded in ``notes``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    exports: list[SheetCsvExport] = []

    xl = pd.ExcelFile(path)
    for i, sheet in enumerate(xl.sheet_names):
        try:
            df = pd.read_excel(xl, sheet_name=sheet)
        except Exception as e:
            notes.append(f"Sheet '{sheet}' could not be read and was skipped ({e}).")
            continue
        if df.empty or len(df.columns) == 0:
            notes.append(f"Sheet '{sheet}' is empty and was skipped.")
            continue
        df = _normalize_dataframe(df)
        csv_path = out_dir / f"sheetcsv_{uuid.uuid4().hex[:12]}_{i}.csv"
        df.to_csv(csv_path, index=False)
        exports.append(SheetCsvExport(sheet_name=str(sheet), csv_path=str(csv_path), n_rows=len(df)))
        del df

    # Largest sheet first: it is almost always the real dataset, and callers
    # use exports[0] as the primary dataset of the upload.
    exports.sort(key=lambda e: e.n_rows, reverse=True)

    if max_sheets is not None and max_sheets > 0 and len(exports) > max_sheets:
        dropped = exports[max_sheets:]
        exports = exports[:max_sheets]
        for exp in dropped:
            try:
                Path(exp.csv_path).unlink(missing_ok=True)
            except OSError:
                pass
        notes.append(
            f"Workbook has {len(exports) + len(dropped)} non-empty sheets; only the "
            f"{max_sheets} largest were imported. Not imported: "
            f"{[e.sheet_name for e in dropped]}."
        )

    return exports, notes
