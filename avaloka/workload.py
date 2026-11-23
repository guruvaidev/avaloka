"""Workload-based routing.

Avaloka sizes the job before she touches it and picks the honest lane:

    ONLINE          small enough to analyse live, in full, right now.
    SAMPLED_ONLINE  too big for instant full analysis — she shows a defensible
                    sampled analysis immediately and offers the full batch.
    BATCH           large — she recommends running the full analysis as a swarm
                    over Apache Ray on Kubernetes (``avaloka batch``).

The size is read *cheaply* (file bytes + a fast row estimate, without loading
the whole file) so she can speak about the plan before doing heavy work.
"""

from __future__ import annotations

import enum
import errno
from dataclasses import dataclass
from pathlib import Path


class Lane(str, enum.Enum):
    ONLINE = "online"
    SAMPLED_ONLINE = "sampled_online"
    BATCH = "batch"


# Thresholds (tunable). Rows dominate the decision; bytes are a backstop for
# very wide tables.
ONLINE_MAX_ROWS = 100_000
ONLINE_MAX_BYTES = 50 * 1024 * 1024          # 50 MB
SAMPLED_MAX_ROWS = 5_000_000
SAMPLED_MAX_BYTES = 1024 * 1024 * 1024        # 1 GB


@dataclass
class WorkloadPlan:
    lane: Lane
    n_rows: int
    n_bytes: int
    estimated: bool          # True when n_rows is an estimate, not exact
    rationale: str
    spoken: str              # Avaloka's first-person framing
    recommend_batch: bool

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["lane"] = self.lane.value
        return d


def _count_csv_rows(path: Path, sample_bytes: int = 4 * 1024 * 1024) -> tuple[int, bool]:
    """Exact row count for small files; a byte-extrapolated estimate for big ones."""
    n_bytes = path.stat().st_size
    if n_bytes <= sample_bytes:
        with path.open("rb") as fh:
            rows = sum(1 for _ in fh) - 1  # minus header
        return max(0, rows), False
    # Estimate: average line length from the first chunk, extrapolated.
    with path.open("rb") as fh:
        chunk = fh.read(sample_bytes)
    lines = chunk.count(b"\n")
    if lines <= 1:
        return 0, True
    avg_line = len(chunk) / lines
    est = int(n_bytes / avg_line) - 1
    return max(0, est), True


def peek(source: str) -> tuple[int, int, bool]:
    """Return ``(n_rows, n_bytes, estimated)`` without loading the dataset.

    A directory of parts is the sum of its parts: a directory's own ``st_size``
    is a few hundred bytes whatever it holds, which routed every partitioned
    dataset to the smallest lane.
    """
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(source)
    if path.is_dir():
        from avaloka.io.complex import detect_complex_format
        from avaloka.io.loader import _part_files
        parts = _part_files(path)
        if not parts:
            if detect_complex_format(path) is not None:
                return 0, 0, True
            # Nothing in it the loader could read as parts, so it is a folder
            # rather than a dataset. Saying so here, before the mission starts,
            # is what lets the CLI refuse it instead of running on no data.
            raise IsADirectoryError(errno.EISDIR, "a directory with no data files", source)
        peeks = [peek(str(part)) for part in parts]
        return (sum(rows for rows, _, _ in peeks), sum(size for _, size, _ in peeks),
                any(estimated for _, _, estimated in peeks))
    n_bytes = path.stat().st_size
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq

            return pq.ParquetFile(path).metadata.num_rows, n_bytes, False
        except Exception:
            return 0, n_bytes, True
    if suffix == ".orc":
        try:
            import pyarrow.orc as orc

            return orc.ORCFile(str(path)).nrows, n_bytes, False
        except Exception:
            return 0, n_bytes, True
    n_rows, estimated = _count_csv_rows(path)
    # A line is a row only of delimited text and JSON lines. In a workbook,
    # an Avro container or a JSON/XML document the newline count is no
    # count at all (an online_retail workbook "had" 181,115 rows; its sheet
    # holds 541,910), so it is never reported as exact.
    if suffix in _NOT_LINE_ROWS:
        estimated = True
    return n_rows, n_bytes, estimated


#: Suffixes whose newline count is not their row count.
_NOT_LINE_ROWS = frozenset({".xlsx", ".xlsm", ".xls", ".avro", ".json", ".xml"})


def rows_phrase(n_rows: int, estimated: bool) -> str:
    """A row count with its true basis: "541,910 rows" when it was counted,
    "about 181,115 rows (estimated)" when it was extrapolated from bytes.

    The estimate is a planner's guess (a workbook's is the newline count of
    its zip), so it is never printed as if it were the count."""
    return f"about {n_rows:,} rows (estimated)" if estimated else f"{n_rows:,} rows"


def classify(n_rows: int, n_bytes: int, *, estimated: bool = False) -> WorkloadPlan:
    mb = n_bytes / (1024 * 1024)
    rows = rows_phrase(n_rows, estimated)

    if n_rows <= ONLINE_MAX_ROWS and n_bytes <= ONLINE_MAX_BYTES:
        return WorkloadPlan(
            lane=Lane.ONLINE, n_rows=n_rows, n_bytes=n_bytes, estimated=estimated,
            rationale=f"{rows} / {mb:.1f} MB fits the live full-analysis budget.",
            spoken=(f"This one's compact — {rows}. I'll analyse the whole thing "
                    "live, right now, nothing held back."),
            recommend_batch=False,
        )
    if n_rows <= SAMPLED_MAX_ROWS and n_bytes <= SAMPLED_MAX_BYTES:
        return WorkloadPlan(
            lane=Lane.SAMPLED_ONLINE, n_rows=n_rows, n_bytes=n_bytes, estimated=estimated,
            rationale=f"{rows} / {mb:.1f} MB is large for instant full analysis; "
                      "a stratified sample gives a faithful answer now.",
            spoken=(f"This is sizeable — {rows}. I'll give you a statistically "
                    "honest sampled read immediately, and when you want the whole dataset I can "
                    "run the full pass as a swarm on Apache Ray (`avaloka batch`)."),
            recommend_batch=True,
        )
    return WorkloadPlan(
        lane=Lane.BATCH, n_rows=n_rows, n_bytes=n_bytes, estimated=estimated,
        rationale=f"{rows} / {mb:.0f} MB is genuinely big-data; "
                  "the full analysis belongs on a Ray cluster.",
        spoken=(f"That's a lot of data — {rows}. I'll still show you a sampled "
                "preview so you're not waiting, but the real answer should run as a Ray swarm on "
                "Kubernetes. Use `avaloka batch` and I'll fan myself out across the cluster."),
        recommend_batch=True,
    )


def route(source: str) -> WorkloadPlan:
    n_rows, n_bytes, estimated = peek(source)
    return classify(n_rows, n_bytes, estimated=estimated)
