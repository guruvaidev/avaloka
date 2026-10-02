"""Structured planning and deterministic execution for multi-metric questions.

The model interprets language into a typed plan. Execution never matches words
in the user's prompt or guesses column names: every field is checked against
the uploaded schema before an action runs.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

Operation = Literal["mean", "median", "sum", "count", "count_distinct", "min", "max", "std"]
_NUMERIC_OPERATIONS = {"mean", "median", "sum", "min", "max", "std"}


class MetricAction(BaseModel):
    id: str
    operation: Operation
    column: str | None = None
    group_by: list[str] = Field(default_factory=list)
    output_name: str

    @property
    def label(self) -> str:
        subject = self.column or "records"
        grouping = f" by {', '.join(self.group_by)}" if self.group_by else ""
        return f"{self.operation} of {subject}{grouping}"


class ActionPlan(BaseModel):
    mode: Literal["aggregate", "code", "clarify"]
    actions: list[MetricAction] = Field(default_factory=list)
    clarification: str | None = None


@dataclass
class ActionResult:
    tables: list[dict[str, Any]]
    missing: list[dict[str, str]]
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "incomplete" if self.missing else "complete"


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")


def _json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text[:16000]):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _output_name(operation: str, column: str | None) -> str:
    subject = re.sub(r"[^a-z0-9]+", "_", (column or "records").lower()).strip("_")
    return f"{operation}_{subject or 'records'}"


def plan_metric_actions(prompt: str, schema: dict[str, Any], model: Any) -> ActionPlan | None:
    """Ask the model for a complete action list and validate it against schema.

    None means the existing coding path should handle the request. The model is
    asked about every eligible coding turn because prompt vocabulary is not
    used as a routing gate. Only plans with two or more actions take this path.
    """
    if model is None or not prompt or not isinstance(schema, dict) or not schema:
        return None
    system = (
        "Translate the user's entire data request into JSON. Return only one JSON object with "
        "mode ('aggregate', 'code', or 'clarify'), actions, and clarification. "
        "Use aggregate only when EVERY requested task is an independent simple aggregate on this one table, "
        "with no filters, joins, ratios, derived measures, rankings, transforms, or time windows. "
        "For aggregate, list EVERY requested calculation as a separate action. Each action has "
        "operation, column (null only for count of rows), and group_by (an array of exact schema column names). "
        "Allowed operations are mean, median, sum, count, count_distinct, min, max, std. "
        "Use min and max here only for numeric measures; date parsing or textual ordering needs code. "
        "Interpret the user's phrasing semantically; do not omit an action or combine actions with different "
        "groupings. Use code for any request that needs a calculation outside this set. "
        "Use clarify only when the requested source field or grouping is genuinely ambiguous. "
        "Never invent a dataset column."
    )
    user = json.dumps({"request": prompt, "schema": schema}, default=str)
    try:
        raw = _json_object(_response_text(model.invoke([SystemMessage(content=system), HumanMessage(content=user)])))
    except Exception as exc:
        logger.warning("Structured metric planning unavailable: %s", exc)
        return None
    if not raw:
        return None
    if raw.get("mode") == "clarify":
        return None
    if raw.get("mode") != "aggregate" or not isinstance(raw.get("actions"), list):
        return None
    if len(raw["actions"]) > 16:
        return None

    actions: list[MetricAction] = []
    seen: set[tuple[str, str | None, tuple[str, ...]]] = set()
    output_names: dict[tuple[str, ...], set[str]] = {}
    try:
        for index, item in enumerate(raw["actions"]):
            if not isinstance(item, dict):
                return None
            if set(item) - {"operation", "column", "group_by"}:
                return None
            operation = item.get("operation")
            column = item.get("column")
            groups = item.get("group_by") or []
            if isinstance(groups, str):
                groups = [groups]
            if not isinstance(groups, list) or any(not isinstance(group, str) for group in groups):
                return None
            if len(set(groups)) != len(groups):
                return None
            if (column is None and operation != "count") or (column is not None and column not in schema):
                return ActionPlan(
                    mode="clarify",
                    clarification=f"I couldn't match the requested field {column!r} to a dataset column. Which column should I use?",
                )
            if any(group not in schema for group in groups):
                unknown = next(group for group in groups if group not in schema)
                return ActionPlan(
                    mode="clarify",
                    clarification=f"I couldn't find grouping column {unknown!r}. Which column should I group by?",
                )
            signature = (operation, column, tuple(groups))
            if signature in seen:
                return None
            seen.add(signature)
            name = _output_name(str(operation), column)
            taken = output_names.setdefault(tuple(groups), set())
            if name in taken:
                return None
            taken.add(name)
            actions.append(MetricAction(
                id=f"metric_{index + 1}", operation=operation, column=column,
                group_by=groups, output_name=name,
            ))
    except (ValidationError, TypeError, ValueError):
        return None
    if len(actions) < 2:
        return None
    plan = ActionPlan(mode="aggregate", actions=actions)
    review_system = (
        "Independently audit this proposed analysis plan against the original request. "
        "Return only JSON with complete (boolean) and missing (array of short descriptions). "
        "Complete is true only if EVERY requested metric, grouping, filter, and condition is represented exactly. "
        "If any request needs more than a simple aggregate, complete must be false. "
        "Do not infer coverage merely because there are two actions."
    )
    review_input = json.dumps({"request": prompt, "schema": schema, "plan": plan.model_dump()}, default=str)
    try:
        review = _json_object(_response_text(model.invoke([
            SystemMessage(content=review_system), HumanMessage(content=review_input),
        ])))
    except Exception as exc:
        logger.warning("Structured metric plan review unavailable: %s", exc)
        return None
    if not review or review.get("complete") is not True or review.get("missing") != []:
        logger.info("Structured metric plan was incomplete; using the coding agent")
        return None
    return plan


def _numeric_series(series: pd.Series, approach: int) -> pd.Series:
    if approach == 1:
        return series.astype(float)
    if approach == 2:
        return pd.to_numeric(series, errors="raise")
    cleaned = series.astype("string").str.strip()
    valid = cleaned.dropna().str.fullmatch(r"\$?\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
    if not valid.all():
        raise ValueError("Numeric formatting is ambiguous and needs data cleansing")
    cleaned = cleaned.str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="raise")


def _calculate(df: pd.DataFrame, action: MetricAction, approach: int) -> pd.DataFrame:
    groups = action.group_by
    if action.operation == "count" and action.column is None:
        if groups:
            if approach == 1:
                values = df.groupby(groups, dropna=False).size()
            elif approach == 2:
                values = df.assign(_avaloka_row=1).groupby(groups, dropna=False)["_avaloka_row"].sum()
            else:
                values = df.value_counts(subset=groups, dropna=False)
            return values.reset_index(name=action.output_name)
        count = len(df) if approach == 1 else df.shape[0] if approach == 2 else df.index.size
        return pd.DataFrame([{action.output_name: int(count)}])

    assert action.column is not None
    values = df[action.column]
    if action.operation in _NUMERIC_OPERATIONS:
        values = _numeric_series(values, approach)
        if values.isin([math.inf, -math.inf]).any():
            raise ValueError("Non-finite numeric values need data cleansing")
    if groups:
        working = df[groups].copy()
        working["_avaloka_metric_value"] = values
        grouped = working.groupby(groups, dropna=False)["_avaloka_metric_value"]
        if action.operation == "count_distinct":
            if approach == 1:
                result = grouped.nunique(dropna=True)
            elif approach == 2:
                result = grouped.agg(lambda series: len(series.dropna().drop_duplicates()))
            else:
                result = grouped.agg(lambda series: pd.Index(series.dropna()).nunique())
        elif action.operation == "count":
            if approach == 1:
                result = grouped.count()
            elif approach == 2:
                result = grouped.agg(lambda series: int(series.notna().sum()))
            else:
                working["_avaloka_present"] = values.notna().astype(int)
                result = working.groupby(groups, dropna=False)["_avaloka_present"].sum()
        else:
            result = grouped.agg(action.operation)
        return result.reset_index(name=action.output_name)

    if action.operation == "count_distinct":
        if approach == 1:
            value = values.nunique(dropna=True)
        elif approach == 2:
            value = len(values.dropna().drop_duplicates())
        else:
            value = pd.Index(values.dropna()).nunique()
    elif action.operation == "count":
        value = values.count() if approach == 1 else values.notna().sum() if approach == 2 else len(values.dropna())
    else:
        value = getattr(values, action.operation)()
    return pd.DataFrame([{action.output_name: value}])


def _json_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def execute_action_plan(df: pd.DataFrame, plan: ActionPlan) -> ActionResult:
    """Execute actions independently, preserving every valid table and metric."""
    tables_by_measure: dict[tuple[tuple[str, ...], str | None], dict[str, Any]] = {}
    missing: list[dict[str, str]] = []
    notes: list[str] = []
    for action in plan.actions:
        result: pd.DataFrame | None = None
        errors: list[str] = []
        for approach in (1, 2, 3):
            try:
                result = _calculate(df, action, approach)
                if result.empty or (
                    action.operation not in {"count", "count_distinct", "sum"}
                    and result[action.output_name].isna().all()
                ):
                    raise ValueError("No usable values were available for this calculation")
                break
            except (TypeError, ValueError, KeyError, OverflowError) as exc:
                result = None
                errors.append(str(exc))
        if result is None:
            reason = (
                f"Some values in {action.column} could not be parsed as numbers"
                if action.operation in _NUMERIC_OPERATIONS
                else "The source or grouping values could not be aggregated safely"
            )
            if errors and all("No usable values" in error for error in errors):
                reason = "No usable values were available for this calculation"
            missing.append({
                "id": action.id, "label": action.label, "reason": reason,
            })
            continue
        if approach == 2:
            notes.append(f"Parsed numeric text in {action.column} for {action.label}.")
        elif approach == 3:
            notes.append(f"Removed currency or thousands separators in {action.column} for {action.label}.")
        # Actions on different source fields can have different units. Keep
        # their tables (and therefore their charts) separate even when both
        # are ungrouped or use the same category columns.
        key = (tuple(action.group_by), action.column)
        if key not in tables_by_measure:
            tables_by_measure[key] = {
                "data": result,
                "action_ids": [action.id],
                "metric_columns": [action.output_name],
                "actions": [action],
            }
        else:
            table = tables_by_measure[key]
            if action.group_by:
                table["data"] = table["data"].merge(result, on=action.group_by, how="outer", validate="one_to_one")
            else:
                table["data"] = pd.concat([table["data"].reset_index(drop=True), result.reset_index(drop=True)], axis=1)
            table["action_ids"].append(action.id)
            table["metric_columns"].append(action.output_name)
            table["actions"].append(action)

    tables: list[dict[str, Any]] = []
    for (group_by, column), table in tables_by_measure.items():
        rows = [
            {str(field_name): _json_value(value) for field_name, value in row.items()}
            for row in table["data"].to_dict(orient="records")
        ]
        operations = " and ".join(action.operation.replace("_", " ") for action in table["actions"])
        subject = (column or "records").replace("_", " ")
        grouping = f" by {', '.join(group_by)}" if group_by else ""
        tables.append({
            "title": f"{operations.capitalize()} of {subject}{grouping}",
            "rows": rows, "action_ids": table["action_ids"],
            "group_by": list(group_by), "metric_columns": table["metric_columns"],
        })
    return ActionResult(tables=tables, missing=missing, notes=notes)
