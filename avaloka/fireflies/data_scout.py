"""Data Scout — replaces the data analyst.

Economic output: a schema, distribution and data-quality report. The Scout
loads the dataset, profiles every column, scores quality and records concrete
issues. Its output (``data_quality.json`` + the in-memory profile) is the
factual foundation every downstream firefly reasons about.
"""

from __future__ import annotations

from typing import Any

from avaloka.fireflies.base import Firefly
from avaloka.io import load_dataset
from avaloka.stats import correlation_pairs, profile_column, quality_report
from avaloka.util import write_json


class DataScout(Firefly):
    name = "data_scout"
    human_role = "Data analyst"

    def perform(self) -> tuple[str, float, dict[str, Any]]:
        handle = load_dataset(self.ctx.data_source)
        frame = handle.frame
        n_rows = handle.n_rows

        columns = [profile_column(frame[c], n_rows) for c in frame.columns]
        quality = quality_report(frame, columns)
        correlations = correlation_pairs(frame)

        profile = {
            "dataset": {
                "source": handle.source,
                "format": handle.fmt,
                "sha256": handle.sha256,
                "n_rows": handle.n_rows,
                "n_cols": handle.n_cols,
                "delimiter": handle.delimiter,
            },
            "columns": columns,
            "quality": quality,
            "correlations": correlations,
        }

        # Hand the loaded frame + profile to downstream fireflies.
        self.ctx.blackboard["handle"] = handle
        self.ctx.blackboard["frame"] = frame
        self.ctx.blackboard["profile"] = profile

        write_json(self.ctx.path("data_quality.json"), profile)

        manual_minutes = 45 + 6 * handle.n_cols
        return (
            f"Profiled {handle.n_cols} columns x {handle.n_rows} rows; quality score "
            f"{quality['score']}/100 with {len(quality['issues'])} issues flagged.",
            manual_minutes,
            {"quality_score": quality["score"], "n_issues": len(quality["issues"])},
        )
