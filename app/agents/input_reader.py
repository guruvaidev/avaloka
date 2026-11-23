"""The analysis input, read by its format.

The execution agent used to read every input with the CSV reader, whatever the
file was. Parquet and JSONL uploads were accepted, sampled and profiled
correctly -- the upload path and the sampler read them by format since #523 --
and then the analysis itself failed with "ParserError: Expected 2 fields in
line 4, saw 4", because the script handed Parquet bytes to ``pd.read_csv``.
Every Parquet and JSONL analysis failed that way.

So the analysis reads its input with the readers the upload path already uses,
chosen by suffix:

    suffix                                  reader
    --------------------------------------  -----------------------------------
    .csv .tsv                               null_policy.read_csv_best_effort
                                            (the one CSV reader; tab for .tsv)
    .xlsx .xlsm .xls                        excel_connector.ExcelConnector, the
                                            sampler's reader, honouring a sheet
    .parquet .pq .json .jsonl .ndjson       avaloka.io.load_dataset, the loader
    .xml .orc .avro .txt                    #523 taught every declared format,
    .pdf, images, videos                    including the bounded complex ones
    a directory of parts                    load_dataset (parts stacked), or
                                            pyarrow with hive partitioning when
                                            the session recorded that layout

It lives in the app, not beside the CSV reader in file_handler/, because it
reaches the CLI loader and file_handler must not depend on avaloka
(tests/test_upload_format_connectors.py); the app may.

``HANDLED_SUFFIXES`` is the set this module dispatches. A contract test fails
if any suffix ``/api/upload`` or the CLI loader declares is missing from it,
so a format can no longer be accepted at the door and refused by the analysis.

THIS FILE IS ALSO SHIPPED AS TEXT
    Like null_policy.py, the execution agent splices this module's source into
    the scripts it runs. The same rules hold: standard library and pandas at
    module level, no relative imports, no ``from __future__`` line. The readers
    outside pandas are imported inside the functions that need them, so a CSV
    analysis never needs anything but this text.
"""

import os

import pandas as pd

#: Read by the CSV reader the caller passes in, with this separator.
CSV_FAMILY_SUFFIXES = {".csv": ",", ".tsv": "\t"}
#: Read by ExcelConnector, the reader the sampler uses for workbooks.
EXCEL_SUFFIXES = frozenset({".xlsx", ".xlsm", ".xls"})
#: Read by avaloka.io.load_dataset. ``.txt`` goes there, not to the CSV
#: reader, because its delimiter is unknown and the loader sniffs it.
LOADER_SUFFIXES = frozenset({
    ".txt", ".parquet", ".pq", ".json", ".jsonl", ".ndjson", ".xml", ".orc", ".avro",
    ".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
})
HANDLED_SUFFIXES = frozenset(CSV_FAMILY_SUFFIXES) | EXCEL_SUFFIXES | LOADER_SUFFIXES


def input_kind(path):
    """Which reader *path* goes to: directory, csv, excel, loader or unknown."""
    text = str(path)
    if os.path.isdir(text):
        return "directory"
    suffix = os.path.splitext(text)[1].lower()
    if suffix in CSV_FAMILY_SUFFIXES:
        return "csv"
    if suffix in EXCEL_SUFFIXES:
        return "excel"
    if suffix in LOADER_SUFFIXES:
        return "loader"
    return "unknown"


def read_input_frame(path, csv_reader, sheet_name=None, hive_partitioning=False, **csv_kwargs):
    """Read the analysis input at *path* as one DataFrame, by its format.

    ``csv_reader`` reads the CSV family (and a path with no recognised suffix,
    which is how the CSV-only reader treated everything before). Its keyword
    arguments apply to that reader only; a Parquet file has no ``sep``.
    """
    text = str(path)
    kind = input_kind(text)
    if kind == "csv":
        if text.lower().endswith(".tsv"):
            csv_kwargs.setdefault("sep", "\t")
        return csv_reader(text, **csv_kwargs)
    if kind == "excel":
        from file_handler.excel_connector import ExcelConnector

        return ExcelConnector(text, sheet_name=sheet_name).load_dataframe()
    if kind == "directory" and hive_partitioning:
        # key=value/ folders carry a column the files themselves do not; the
        # sampler recovers it with Daft's hive_partitioning, pyarrow does the same.
        return pd.read_parquet(text)
    if kind in ("loader", "directory"):
        from avaloka.io.loader import load_dataset

        return load_dataset(text).frame
    return csv_reader(text, **csv_kwargs)


def format_aware(csv_reader):
    """*csv_reader*, extended to read every format this module dispatches.

    The execution scripts rebind ``read_csv_best_effort`` to this, so a
    ``read_csv_best_effort("<input>")`` in generated code reads a Parquet
    input as Parquet.
    """
    def read(path, **kwargs):
        return read_input_frame(path, csv_reader, **kwargs)

    read.__wrapped__ = csv_reader
    return read
