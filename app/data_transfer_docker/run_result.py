"""Shared result type for the DTA runners (Local Docker + GKE Ray).

Both runners historically returned a bare ``bool`` for success. They now return
a :class:`TransferResult`, which still behaves like that bool (``__bool__``) so
existing ``if not success`` / ``bool(success)`` callers keep working, while also
carrying the number of rows inserted and the target table parsed from the job
logs. The row count is produced *inside* the execution container by the SQL
sink, which prints a stable marker line:

    AVALOKA_RESULT rows_inserted=<n> table=<name>
"""

import re
from dataclasses import dataclass
from typing import Optional

# Matches the marker emitted by BaseSQLDataSink.finalize(). ``table`` may contain
# a schema-qualified name (e.g. ``public.housing_transformed``), so accept any
# non-whitespace run.
_MARKER_RE = re.compile(r"AVALOKA_RESULT\s+rows_inserted=(\d+)\s+table=(\S+)")


@dataclass
class TransferResult:
    """Outcome of a DTA run. Truthy iff the run succeeded."""

    success: bool
    rows_inserted: Optional[int] = None
    table: Optional[str] = None

    def __bool__(self) -> bool:
        return self.success


def parse_transfer_result(success: bool, logs: Optional[str]) -> TransferResult:
    """Build a :class:`TransferResult`, extracting the row count/table from logs.

    If multiple markers are present (e.g. retries or multiple sinks), the last
    one wins. Missing/garbled markers leave ``rows_inserted``/``table`` as None,
    and callers simply omit the count line.
    """
    rows: Optional[int] = None
    table: Optional[str] = None
    if logs:
        for match in _MARKER_RE.finditer(logs):
            rows = int(match.group(1))
            table = match.group(2)
    return TransferResult(success=success, rows_inserted=rows, table=table)
