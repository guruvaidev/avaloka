"""Data Engineer — replaces the data engineer.

Economic output: conservative, documented transformations *and* the
reproducible pipeline that re-creates them. It writes the transformed dataset
(``transformed_dataset.parquet``) plus runnable, human-readable code
(``analysis.py`` and ``analysis.ipynb``) so the analysis is not a black box — a
practitioner can re-run or adapt it. Transformations are deliberately
non-destructive: it cleans obvious defects and *records* judgement calls
(e.g. imputation) rather than silently applying them.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from avaloka.fireflies.base import Firefly
from avaloka.notebook import code, md, notebook
from avaloka.util import write_json, write_text


class DataEngineer(Firefly):
    name = "data_engineer"
    human_role = "Data engineer"

    def perform(self) -> tuple[str, float, dict[str, Any]]:
        frame: pd.DataFrame = self.ctx.blackboard["working_frame"]
        profile = self.ctx.blackboard["profile"]
        transformations: list[dict[str, Any]] = []

        out = frame.copy()

        # 1) Drop constant columns — no predictive content.
        const_cols = profile["quality"]["constant_columns"]
        const_present = [c for c in const_cols if c in out.columns]
        if const_present:
            out = out.drop(columns=const_present)
            transformations.append({"op": "drop_constant_columns", "columns": const_present})

        # 2) Drop exact duplicate rows.
        dupes = int(out.duplicated().sum())
        if dupes:
            out = out.drop_duplicates().reset_index(drop=True)
            transformations.append({"op": "drop_duplicate_rows", "rows_removed": dupes})

        # 3) Trim whitespace on string columns.
        str_cols = [c for c in out.columns if out[c].dtype == object]
        if str_cols:
            for c in str_cols:
                out[c] = out[c].astype("string").str.strip()
            transformations.append({"op": "strip_whitespace", "columns": str_cols})

        # Imputation is a judgement call — flagged, not silently applied.
        high_missing = profile["quality"]["high_missing_columns"]
        if high_missing:
            transformations.append({
                "op": "flag_high_missingness",
                "columns": high_missing,
                "applied": False,
                "note": "Imputation deferred to a human decision; see assumptions.yaml.",
            })

        out_path = self.ctx.path("transformed_dataset.parquet")
        out.to_parquet(out_path, index=False)

        self.ctx.blackboard["transformed_frame"] = out
        self.ctx.blackboard["transformations"] = transformations

        # Whether the generated code can repeat the read is decided once, here,
        # and told to everything that would otherwise promise it can.
        _, declined = self._load_src()
        self.ctx.blackboard["script_declines_to_reread"] = declined

        # Reproducible code (no LLM — generated from the actual decisions above).
        self._write_python(transformations)
        self._write_notebook(transformations)

        return (
            f"Applied {len(transformations)} documented transformations; wrote "
            f"transformed_dataset.parquet ({out.shape[0]}x{out.shape[1]}) + reproducible "
            "analysis.py / analysis.ipynb.",
            90 + 8 * profile["dataset"]["n_cols"],
            {"n_transformations": len(transformations)},
        )

    # --- code generation --------------------------------------------------
    def _pipeline_src(self, transformations: list[dict[str, Any]]) -> str:
        steps = []
        for t in transformations:
            if t["op"] == "drop_constant_columns":
                steps.append(f"    df = df.drop(columns={t['columns']!r})  # constant columns")
            elif t["op"] == "drop_duplicate_rows":
                steps.append("    df = df.drop_duplicates().reset_index(drop=True)  # exact duplicates")
            elif t["op"] == "strip_whitespace":
                steps.append(
                    f"    for _c in {t['columns']!r}:\n"
                    f"        df[_c] = df[_c].astype('string').str.strip()"
                )
        return "\n".join(steps) if steps else "    # (no transformations were necessary)"

    # --- the read ---------------------------------------------------------
    def _load_src(self) -> tuple[str, str | None]:
        """Source of ``load()`` for the generated code, and why it declines.

        The mission read the data through avaloka/io/loader.py: a sniffed
        delimiter, an encoding fallback, null-like words kept as labels in text
        columns. A bare ``pd.read_csv`` here read a different frame from the one
        that was analysed. So the read is rendered from what the loader
        recorded on the handle, argument for argument, with pandas alone.

        Where pandas alone cannot repeat it -- a spreadsheet whose header row
        was searched for, a database, an object store -- ``load()`` says so and
        stops. A read that fails at the user's desk, or one that half-works, is
        worse than a plain refusal. Returns ``(source, None)`` or
        ``(source, the kind of source that was declined)``.
        """
        read = getattr(self.ctx.blackboard["handle"], "read", None) or {}
        resolved = self.ctx.blackboard.get("resolved_source") or {}
        declined = None
        if resolved.get("kind") == "database":
            declined = "a database table or query"
        elif resolved.get("kind") == "object_storage":
            declined = "an object in object storage"
        elif read.get("kind") == "excel":
            declined = "an Excel workbook"
        elif read.get("kind") not in {"csv", "parquet"}:
            declined = "a source whose read was not recorded"
        if declined:
            message = (
                f"This script cannot re-read the data: the source was {declined}, and "
                "reading it the way the analysis did takes more than pandas alone. "
                "The prepared data the analysis used is in the bundle, next to this "
                "script: pd.read_parquet('transformed_dataset.parquet').")
            return (
                "def load() -> pd.DataFrame:\n"
                f"    raise SystemExit({message!r})\n"
            ), declined

        if read["kind"] == "parquet":
            call = "pd.read_parquet(DATA_SOURCE)"
            preamble = ""
        else:
            args = [f"sep={read['sep']!r}", f"encoding={read['encoding']!r}"]
            if read.get("encoding_errors"):
                args.append(f"encoding_errors={read['encoding_errors']!r}")
            preamble = ""
            if read.get("na_values") is not None:
                # Two kinds of column: text, where only an empty cell is missing
                # and words like None or NA are labels; and the rest, where
                # pandas' usual list applies. Grouped so a wide file stays short.
                groups: dict[tuple[str, ...], list[Any]] = {}
                for column, strings in read["na_values"].items():
                    groups.setdefault(tuple(strings), []).append(column)
                lines = ["# The strings read as missing, per column, exactly as the analysis read them.",
                         "MISSING_STRINGS = {"]
                for strings, columns in groups.items():
                    lines.append(f"    **dict.fromkeys({columns!r}, {list(strings)!r}),")
                lines.append("}")
                preamble = "\n".join(lines) + "\n\n\n"
                args += ["keep_default_na=False", "na_values=MISSING_STRINGS"]
            call = "pd.read_csv(\n        DATA_SOURCE, " + ", ".join(args) + ")"
        return (
            preamble
            + "def load() -> pd.DataFrame:\n"
            + f"    df = {call}\n"
            + "    return df.dropna(axis=1, how=\"all\")  # all-empty columns, as the analysis dropped them\n"
        ), None

    def _write_python(self, transformations: list[dict[str, Any]]) -> None:
        src = self.ctx.blackboard["handle"].source
        goal = self.ctx.goal.replace('"', "'")
        body = self._pipeline_src(transformations)
        load_src = self._load_src()[0].rstrip("\n")
        script = f'''"""Reproducible analysis pipeline generated by Avaloka.

Mission goal: {goal}
This script reproduces the data preparation performed during the Avaloka
mission. It is dependency-light (pandas only) and deterministic.
"""

import pandas as pd

DATA_SOURCE = {src!r}


{load_src}

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the transformations Avaloka selected for this dataset."""
{body}
    return df


def main() -> None:
    df = load()
    print("Loaded:", df.shape)
    df = transform(df)
    print("Transformed:", df.shape)
    df.to_parquet("transformed_dataset.parquet", index=False)
    print("Wrote transformed_dataset.parquet")


if __name__ == "__main__":
    main()
'''
        write_text(self.ctx.path("analysis.py"), script)

    def _write_notebook(self, transformations: list[dict[str, Any]]) -> None:
        profile = self.ctx.blackboard["profile"]
        src = self.ctx.blackboard["handle"].source
        body = self._pipeline_src(transformations)
        load_src, _ = self._load_src()             # the same text analysis.py gets
        n_rows = profile["dataset"]["n_rows"]
        cells = [
            md(f"# Avaloka analysis\n\n**Goal:** {self.ctx.goal}\n\n"
               f"Reproducible notebook generated by Avaloka mission `{self.ctx.mission_id}`."),
            code("import pandas as pd\nimport numpy as np\npd.set_option('display.max_columns', 100)"),
            md("## Load"),
            code(f"DATA_SOURCE = {src!r}\n\n\n"
                 + load_src
                 + "\n\ndf = load()\nprint(df.shape)\ndf.head()"),
            md("## Profile"),
            code("df.describe(include='all').T"),
            code("df.isna().mean().sort_values(ascending=False).head(20)  # missingness"),
            md("## Transform (Avaloka-selected)"),
            code("def transform(df):\n" + (body.replace("    ", "    ", 1) or "    return df")
                 + "\n    return df\n\ndf = transform(df)\nprint(df.shape)"),
            md("## Distributions"),
            code("num = df.select_dtypes('number')\n"
                 "if num.shape[1]:\n    num.hist(figsize=(12, 8))"),
            md(f"## Notes\n\n- Working rows analysed: {n_rows:,}\n"
               "- See `validation_report.json` for leakage / stability checks.\n"
               "- See `assumptions.yaml` for every judgement call."),
        ]
        write_json(self.ctx.path("analysis.ipynb"), notebook(cells))
        # ensure valid JSON round-trips (defensive)
        json.loads(self.ctx.path("analysis.ipynb").read_text())
