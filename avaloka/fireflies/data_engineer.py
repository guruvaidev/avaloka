"""Data Engineer — replaces the data engineer.

Economic output: conservative, documented transformations *and* the
reproducible pipeline that re-creates them. It writes the transformed dataset
(``transformed_dataset.parquet``) plus runnable, human-readable code
(``analysis.py`` and ``analysis.ipynb``) so the analysis is not a black box — a
practitioner can re-run or adapt it. Transformations are deliberately
non-destructive: it cleans obvious defects and *records* judgement calls
(e.g. imputation) rather than silently applying them.

The first transformation is the Workload Optimizer's plan. Its projection and
predicates are pushed into the Parquet reader when the source is a local
Parquet file and the working frame is the whole dataset; otherwise they are
applied to the working frame in memory. Either way the rows and columns in
and out are measured and recorded under ``applied`` in
``workload_optimization.json``, beside the estimate, so the plan's claimed
gain is checked against what happened, not assumed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from avaloka.fireflies.base import Firefly
from avaloka.goal_analysis import apply_filters, describe_filters
from avaloka.notebook import code, md, notebook
from avaloka.util import write_json, write_text

#: How the plan was executed, recorded as ``applied.strategy_applied``.
PUSHDOWN_PYARROW = "pyarrow_pushdown"      # columns= and filters= in the Parquet read
IN_MEMORY = "in_memory"                    # predicates and projection on the working frame
NO_PLAN = "none"                           # no plan, or nothing to apply


class DataEngineer(Firefly):
    name = "data_engineer"
    human_role = "Data engineer"

    def perform(self) -> tuple[str, float, dict[str, Any]]:
        frame: pd.DataFrame = self.ctx.blackboard["working_frame"]
        profile = self.ctx.blackboard["profile"]
        transformations: list[dict[str, Any]] = []

        # 0) The optimizer's plan: the scan this analysis needs, measured.
        out, applied = self._apply_plan(frame, transformations)

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
        applied["columns_out"] = int(out.shape[1])
        plan = self.ctx.blackboard.get("workload_optimization")
        if plan is not None:
            # The measurement lands beside the estimate it is checked against.
            plan["applied"] = applied
            write_json(self.ctx.path("workload_optimization.json"), plan)

        # Whether the generated code can repeat the read is decided once, here,
        # and told to everything that would otherwise promise it can.
        _, declined = self._load_src()
        self.ctx.blackboard["script_declines_to_reread"] = declined

        # Reproducible code (no LLM — generated from the actual decisions above).
        self._write_python(transformations)
        self._write_notebook(transformations)

        scan = ""
        if applied["strategy_applied"] != NO_PLAN:
            scan = (f" Scan: {applied['rows_in']:,} -> {applied['rows_out']:,} rows, "
                    f"{applied['columns_in']} -> {applied['columns_out']} columns via "
                    f"{applied['strategy_applied']} (optimizer estimated "
                    f"{applied['estimated_rows_after_filters']:,} rows).")
        return (
            f"Applied {len(transformations)} documented transformations; wrote "
            f"transformed_dataset.parquet ({out.shape[0]}x{out.shape[1]}) + reproducible "
            f"analysis.py / analysis.ipynb.{scan}",
            90 + 8 * profile["dataset"]["n_cols"],
            {"n_transformations": len(transformations), "workload": applied},
        )

    # --- the optimizer's plan ----------------------------------------------
    def _apply_plan(self, frame: pd.DataFrame,
                    transformations: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Execute ``workload_optimization`` and measure it.

        Returns the frame the rest of the pipeline works on and the ``applied``
        record: ``rows_in``/``rows_out``, ``columns_in``/``columns_out`` (the
        latter is completed once the pipeline has run), ``strategy_applied``,
        the predicates applied, what was pushed into the reader, and the
        optimizer's ``estimated_rows_after_filters`` so the two can be read
        side by side. ``selectivity_measured`` is ``rows_out / rows_in``;
        ``selectivity_estimated`` is the optimizer's product of filter factors.
        """
        plan = self.ctx.blackboard.get("workload_optimization") or {}
        filters = [
            {"column": f["column"], "operator": f["operator"], "value": f.get("value")}
            for f in plan.get("filter_factors", []) if f.get("column") in frame.columns
        ]
        projection = [c for c in plan.get("projection_columns", []) if c in frame.columns]
        estimated = int(plan.get("estimated_rows_after_filters", len(frame)))
        n_rows = int((self.ctx.blackboard.get("profile") or {}).get("dataset", {}).get("n_rows", len(frame)))
        applied: dict[str, Any] = {
            "rows_in": int(len(frame)), "rows_out": int(len(frame)),
            "columns_in": int(frame.shape[1]), "columns_out": int(frame.shape[1]),
            "strategy_planned": plan.get("strategy"), "strategy_applied": NO_PLAN,
            "filters_applied": filters, "projection_applied": False, "pushed_down": None,
            "estimated_rows_after_filters": estimated,
            "selectivity_estimated": round(estimated / n_rows, 6) if n_rows else 1.0,
            "selectivity_measured": 1.0,
        }
        if not plan or (not filters and not projection):
            return frame.copy(), applied

        out = self._pushdown(frame, projection, filters, applied, transformations)
        if out is None:
            out = frame.copy()
            if projection and len(projection) < out.shape[1]:
                self._record_projection(list(out.columns), projection, False, transformations)
                out = out[projection]
                applied["projection_applied"] = True
            if filters:
                kept = apply_filters(out, filters).reset_index(drop=True)
                transformations.append({"op": "apply_filters", "filters": filters,
                                        "rows_removed": int(len(out) - len(kept)),
                                        "pushed_down": False})
                out = kept
            applied["strategy_applied"] = IN_MEMORY
        applied["rows_out"] = int(len(out))
        applied["selectivity_measured"] = (
            round(applied["rows_out"] / applied["rows_in"], 6) if applied["rows_in"] else 1.0)
        applied["estimate_error_rows"] = int(applied["rows_out"] - round(
            applied["rows_in"] * applied["selectivity_estimated"]))
        return out, applied

    def _record_projection(self, columns: list[str], projection: list[str], pushed_down: bool,
                           transformations: list[dict[str, Any]]) -> None:
        """Record the projection, naming the constant columns it removes.

        The plan projects constant columns away, which would otherwise leave
        the record and the generated script saying only "projection" for them.
        They are recorded first as ``drop_constant_columns`` (the reason), and
        ``project_columns`` carries only what the goal itself left out.
        """
        constants = set(self.ctx.blackboard["profile"]["quality"].get("constant_columns", []))
        dropped = [c for c in columns if c not in projection]
        const_dropped = [c for c in dropped if c in constants]
        other = [c for c in dropped if c not in constants]
        if const_dropped:
            transformations.append({"op": "drop_constant_columns", "columns": const_dropped,
                                    "via": "projection", "pushed_down": pushed_down})
        if other:
            transformations.append({"op": "project_columns", "columns": projection,
                                    "dropped": other, "pushed_down": pushed_down})

    def _pushdown(self, frame: pd.DataFrame, projection: list[str], filters: list[dict[str, Any]],
                  applied: dict[str, Any], transformations: list[dict[str, Any]]) -> pd.DataFrame | None:
        """Read only the planned columns and rows from a local Parquet file.

        Only when the source is a Parquet file on disk, the working frame is
        the whole dataset (a sample must stay the sample the Sampling
        Specialist drew) and every planned column is in the file's schema.
        Returns ``None`` when the plan has to be applied in memory instead;
        the reason is recorded under ``applied.pushed_down``. A database or
        object-storage source is read whole here: pushing predicates into SQL
        is the connector contract's work, not this firefly's.
        """
        handle = self.ctx.blackboard.get("handle")
        read = getattr(handle, "read", None) or {}
        resolved = self.ctx.blackboard.get("resolved_source") or {}
        sampling = self.ctx.blackboard.get("sampling") or {}
        source = Path(str(getattr(handle, "source", ""))).expanduser()
        if read.get("kind") != "parquet" or resolved.get("kind", "file") != "file":
            applied["pushed_down"] = {"applied": False, "reason": "not a local Parquet file"}
            return None
        if sampling.get("sampled") or len(frame) != int(getattr(handle, "n_rows", len(frame))):
            applied["pushed_down"] = {"applied": False, "reason": "working frame is a sample"}
            return None
        if not source.is_file():
            applied["pushed_down"] = {"applied": False, "reason": "source file is not on disk"}
            return None
        try:
            import pyarrow.dataset as ds
            import pyarrow.parquet as pq

            schema_names = set(pq.read_schema(source).names)
            if any(c not in schema_names for c in projection + [f["column"] for f in filters]):
                applied["pushed_down"] = {"applied": False,
                                          "reason": "a planned column is derived, not stored"}
                return None
            expression = None
            for predicate in filters:
                term = self._arrow_term(ds.field(predicate["column"]), predicate)
                if term is None:
                    applied["pushed_down"] = {"applied": False,
                                              "reason": f"operator {predicate['operator']!r} has no reader form"}
                    return None
                expression = term if expression is None else expression & term
            table = pq.read_table(source, columns=projection or None, filters=expression)
        except Exception as exc:  # noqa: BLE001 - the in-memory path is always available
            applied["pushed_down"] = {"applied": False, "reason": type(exc).__name__}
            return None
        out = table.to_pandas()
        out.columns = [str(c) for c in out.columns]
        # Pandas-side equality on text is case-insensitive; the reader's is
        # not. The optimizer canonicalised values against the profile, so the
        # two agree, and the in-memory mask over the pushed-down rows proves it.
        if filters and len(apply_filters(out, filters)) != len(out):
            applied["pushed_down"] = {"applied": False, "reason": "reader and pandas disagreed"}
            return None
        applied["pushed_down"] = {"applied": True, "engine": "pyarrow", "columns": projection,
                                  "filters": filters, "rows_read": int(len(out))}
        applied["projection_applied"] = bool(projection) and len(projection) < frame.shape[1]
        applied["strategy_applied"] = PUSHDOWN_PYARROW
        if applied["projection_applied"]:
            self._record_projection(list(frame.columns), projection, True, transformations)
        if filters:
            transformations.append({"op": "apply_filters", "filters": filters,
                                    "rows_removed": int(len(frame) - len(out)), "pushed_down": True})
        return out

    @staticmethod
    def _arrow_term(field: Any, predicate: dict[str, Any]) -> Any:
        operator, value = str(predicate["operator"]).lower(), predicate.get("value")
        if operator in {"eq", "=", "=="}:
            return field == value
        if operator == "in":
            return field.isin(list(value or []))
        if operator in {"is_null", "null"}:
            return field.is_null()
        return {"lt": field < value, "<": field < value, "lte": field <= value, "<=": field <= value,
                "gt": field > value, ">": field > value, "gte": field >= value, ">=": field >= value,
                }.get(operator) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    # --- code generation --------------------------------------------------
    @staticmethod
    def _filter_src(predicate: dict[str, Any]) -> str:
        """One predicate as a pandas boolean expression over ``df``."""
        column, operator, value = predicate["column"], str(predicate["operator"]).lower(), predicate.get("value")
        if operator in {"is_null", "null"}:
            return f"df[{column!r}].isna()"
        if operator in {"lt", "lte", "gt", "gte"}:
            symbol = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}[operator]
            return f"(pd.to_numeric(df[{column!r}], errors='coerce') {symbol} {value!r})"
        wanted = list(value) if operator == "in" else [value]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in wanted):
            return f"pd.to_numeric(df[{column!r}], errors='coerce').isin({wanted!r})"
        return (f"df[{column!r}].astype('string').str.lower()"
                f".isin({[str(v).lower() for v in wanted]!r})")

    def _pipeline_src(self, transformations: list[dict[str, Any]]) -> str:
        steps = []
        for t in transformations:
            if t["op"] == "project_columns":
                steps.append(f"    df = df[{t['columns']!r}]  # the optimizer's projection")
            elif t["op"] == "apply_filters":
                masks = " & ".join(self._filter_src(f) for f in t["filters"])
                steps.append(f"    df = df[{masks}].reset_index(drop=True)  # the goal's predicates")
            elif t["op"] == "drop_constant_columns":
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
        elif read.get("kind") == "directory":
            declined = f"a directory of {read.get('parts')} {read.get('part_kind')} part files"
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
        # Gap tokens in typed and mixed columns were read as missing after the
        # read -- those exact strings and nothing else; the script does the
        # same, column by column, so the frames match. No errors="coerce": a
        # real value the analysis did not see must stop the script, not
        # quietly become missing.
        repair = ""
        for column, info in (read.get("coerced") or {}).items():
            gaps = f"df[{column!r}].mask(df[{column!r}].isin({list(info.get('words') or [])!r}))"
            if info.get("kind") == "numeric":
                repair += (f"    df[{column!r}] = pd.to_numeric({gaps}.str.strip())"
                           f"  # {info['cells']} gap word(s) read as missing\n")
            elif info.get("kind") == "mixed":
                repair += (f"    df[{column!r}] = {gaps}"
                           f"  # {info['cells']} gap word(s) read as missing; {info.get('kept', 0)}"
                           " real non-numeric value(s) kept, so the column stays text\n")
            else:
                repair += (f"    df[{column!r}] = {gaps}"
                           f"  # {info['cells']} gap word(s) in a date column read as missing\n")
        return (
            preamble
            + "def load() -> pd.DataFrame:\n"
            + f"    df = {call}\n"
            + repair
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
