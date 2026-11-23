"""Recursive, value-safe schemas for tabular and complex datasets.

The validator deliberately operates on Python values instead of tying the core
package to Pydantic, TensorFlow, PyTorch or a media runtime.  The emitted shape
is JSON-Schema-like, with one Avaloka extension (``type: tensor``) for dense
numeric arrays.  It is intended for dataset inspection and execution gates,
not as a replacement for a serving API's formal JSON Schema contract.
"""

from __future__ import annotations
import ast

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SchemaIssue:
    """One actionable difference between a value and its expected schema."""

    path: str
    code: str
    message: str
    severity: str = "error"
    expected: Any = None
    observed: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SchemaValidationResult:
    """Serializable validation outcome."""

    issues: tuple[SchemaIssue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "error_count": sum(i.severity == "error" for i in self.issues),
            "warning_count": sum(i.severity == "warning" for i in self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scalar_type(value: Any) -> str:
    value = _plain(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "bytes"
    if isinstance(value, (datetime, date)):
        return "datetime"
    if isinstance(value, Decimal):
        return "number"
    return "string"


def _tensor_signature(value: Any) -> dict[str, Any] | None:
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)) and value:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if array.ndim == 0 or array.dtype.kind not in "biufc":
        return None
    return {
        "type": "tensor",
        "dtype": str(array.dtype),
        "shape": [int(size) for size in array.shape],
    }


def _infer_node(value: Any, *, depth: int, max_depth: int) -> dict[str, Any]:
    value = _plain(value)
    if depth >= max_depth:
        return {"type": "opaque", "reason": "max_depth"}
    if isinstance(value, Mapping):
        properties = {
            str(key): _infer_node(child, depth=depth + 1, max_depth=max_depth)
            for key, child in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
        return {
            "type": "object",
            "properties": properties,
            "required": sorted(properties),
            "additionalProperties": False,
        }
    tensor = _tensor_signature(value)
    if tensor is not None:
        return tensor
    if isinstance(value, (list, tuple)):
        return {
            "type": "array",
            "items": _merge_nodes([
                _infer_node(item, depth=depth + 1, max_depth=max_depth)
                for item in value
            ]),
        }
    return {"type": _scalar_type(value)}


def _merge_shapes(shapes: list[list[int | None]]) -> list[int | None]:
    if not shapes:
        return []
    ranks = {len(shape) for shape in shapes}
    if len(ranks) != 1:
        return []
    return [
        values[0] if len(set(values)) == 1 else None
        for values in zip(*shapes)
    ]


def _merge_object_nodes(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({name for node in nodes for name in node.get("properties", {})})
    properties: dict[str, Any] = {}
    required: list[str] = []
    presence: dict[str, float] = {}
    for name in names:
        children = [node["properties"][name] for node in nodes
                    if name in node.get("properties", {})]
        properties[name] = _merge_nodes(children)
        presence[name] = round(len(children) / len(nodes), 6)
        if len(children) == len(nodes):
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
        "x-avaloka-presence": presence,
    }


def _merge_nodes(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not nodes:
        return {}
    nullable = any(node.get("type") == "null" for node in nodes)
    active = [node for node in nodes if node.get("type") != "null"]
    if not active:
        return {"type": "null"}

    types = {node.get("type") for node in active}
    if types <= {"integer", "number"}:
        merged: dict[str, Any] = {
            "type": "number" if "number" in types else "integer"
        }
    elif len(types) == 1 and "object" in types:
        merged = _merge_object_nodes(active)
    elif len(types) == 1 and "array" in types:
        merged = {
            "type": "array",
            "items": _merge_nodes([
                node["items"] for node in active if node.get("items")
            ]),
        }
    elif len(types) == 1 and "tensor" in types:
        dtypes = sorted({str(node.get("dtype")) for node in active})
        shapes = [node.get("shape", []) for node in active]
        merged = {
            "type": "tensor",
            "dtype": dtypes[0] if len(dtypes) == 1 else dtypes,
            "shape": _merge_shapes(shapes),
        }
    elif len(types) == 1:
        merged = {"type": next(iter(types))}
    else:
        unique: dict[str, dict[str, Any]] = {}
        for node in active:
            unique[str(node)] = node
        merged = {"anyOf": list(unique.values())}
    if nullable:
        merged["nullable"] = True
    return merged


def infer_complex_schema(
    values: Any,
    *,
    max_samples: int = 256,
    max_depth: int = 12,
) -> dict[str, Any]:
    """Infer a bounded recursive schema from records, tensors or scalars."""
    if hasattr(values, "to_dict") and hasattr(values, "columns"):
        values = values.head(max_samples).to_dict(orient="records")
    if isinstance(values, Iterable) and not isinstance(
        values, (Mapping, str, bytes, bytearray, np.ndarray)
    ):
        sampled = []
        for value in values:
            sampled.append(value)
            if len(sampled) >= max_samples:
                break
        node = {
            "type": "array",
            "items": _merge_nodes([
                _infer_node(value, depth=1, max_depth=max_depth)
                for value in sampled
            ]),
        }
        node["x-avaloka-sampled-items"] = len(sampled)
        return node
    return _infer_node(values, depth=0, max_depth=max_depth)


def _matches_type(value: Any, expected: str) -> bool:
    value = _plain(value)
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "bytes":
        return isinstance(value, (bytes, bytearray, memoryview))
    if expected == "datetime":
        return isinstance(value, (datetime, date))
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, (list, tuple))
    if expected == "tensor":
        return _tensor_signature(value) is not None
    if expected == "opaque":
        return True
    return False


def _validate(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str,
    issues: list[SchemaIssue],
    unexpected_severity: str,
) -> None:
    if value is None and schema.get("nullable"):
        return
    alternatives = schema.get("anyOf")
    if alternatives:
        attempts: list[list[SchemaIssue]] = []
        for alternative in alternatives:
            candidate: list[SchemaIssue] = []
            _validate(value, alternative, path=path, issues=candidate,
                      unexpected_severity=unexpected_severity)
            if not any(issue.severity == "error" for issue in candidate):
                return
            attempts.append(candidate)
        issues.append(SchemaIssue(
            path, "union_mismatch", "Value does not match any allowed schema.",
            expected=[item.get("type") for item in alternatives],
            observed=_scalar_type(value),
        ))
        return
    expected = schema.get("type")
    if expected is None:
        return
    if not _matches_type(value, str(expected)):
        issues.append(SchemaIssue(
            path, "type_mismatch", "Value has a different type.",
            expected=expected, observed=_scalar_type(value),
        ))
        return

    if expected == "object":
        properties = schema.get("properties", {})
        missing = sorted(set(schema.get("required", [])) - set(map(str, value)))
        for name in missing:
            issues.append(SchemaIssue(
                f"{path}.{name}", "required_field_missing",
                "Required field is missing.", expected="present", observed="missing",
            ))
        for raw_name, child in value.items():
            name = str(raw_name)
            if name in properties:
                _validate(child, properties[name], path=f"{path}.{name}",
                          issues=issues, unexpected_severity=unexpected_severity)
            elif schema.get("additionalProperties") is False:
                issues.append(SchemaIssue(
                    f"{path}.{name}", "unexpected_field",
                    "Field was not present in the inferred contract.",
                    severity=unexpected_severity, observed="present",
                ))
    elif expected == "array":
        item_schema = schema.get("items", {})
        for index, item in enumerate(value):
            _validate(item, item_schema, path=f"{path}[{index}]", issues=issues,
                      unexpected_severity=unexpected_severity)
    elif expected == "tensor":
        signature = _tensor_signature(value)
        if signature is None:
            return
        expected_shape = schema.get("shape", [])
        observed_shape = signature["shape"]
        shape_matches = len(expected_shape) == len(observed_shape) and all(
            required is None or required == observed
            for required, observed in zip(expected_shape, observed_shape)
        )
        if not shape_matches:
            issues.append(SchemaIssue(
                path, "tensor_shape_mismatch", "Tensor has a different shape.",
                expected=expected_shape, observed=observed_shape,
            ))
        expected_dtype = schema.get("dtype")
        allowed_dtypes = (expected_dtype if isinstance(expected_dtype, list)
                          else [expected_dtype])
        if expected_dtype and signature["dtype"] not in allowed_dtypes:
            issues.append(SchemaIssue(
                path, "tensor_dtype_mismatch", "Tensor has a different dtype.",
                expected=expected_dtype, observed=signature["dtype"],
            ))


def validate_complex_schema(
    values: Any,
    schema: Mapping[str, Any],
    *,
    unexpected_severity: str = "warning",
) -> SchemaValidationResult:
    """Validate nested values, including tensor ranks, shapes and dtypes."""
    if unexpected_severity not in {"warning", "error"}:
        raise ValueError("unexpected_severity must be 'warning' or 'error'.")
    if hasattr(values, "to_dict") and hasattr(values, "columns"):
        values = values.to_dict(orient="records")
    issues: list[SchemaIssue] = []
    _validate(values, schema, path="$", issues=issues,
              unexpected_severity=unexpected_severity)
    return SchemaValidationResult(tuple(issues))


def validate_rlds_episodes(episodes: Sequence[Mapping[str, Any]]) -> SchemaValidationResult:
    """Validate RLDS episode/step invariants independent of TensorFlow."""
    issues: list[SchemaIssue] = []
    canonical_fields: set[str] | None = None
    episode_ids: list[str] = []
    for episode_index, episode in enumerate(episodes):
        base = f"$[{episode_index}]"
        episode_id = str(episode.get("episode_id", episode_index))
        episode_ids.append(episode_id)
        steps = episode.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            issues.append(SchemaIssue(
                f"{base}.steps", "rlds_steps_missing",
                "RLDS episode must contain a sequence of steps.",
                expected="sequence", observed=_scalar_type(steps),
            ))
            continue
        if not steps:
            issues.append(SchemaIssue(
                f"{base}.steps", "rlds_empty_episode", "RLDS episode is empty."
            ))
            continue
        for step_index, step in enumerate(steps):
            path = f"{base}.steps[{step_index}]"
            if not isinstance(step, Mapping):
                issues.append(SchemaIssue(
                    path, "rlds_step_type", "RLDS step must be an object.",
                    expected="object", observed=_scalar_type(step),
                ))
                continue
            fields = set(map(str, step))
            if canonical_fields is None:
                canonical_fields = fields
            elif fields != canonical_fields:
                issues.append(SchemaIssue(
                    path, "rlds_step_schema_drift",
                    "All RLDS steps in a dataset must expose the same fields.",
                    expected=sorted(canonical_fields), observed=sorted(fields),
                ))
            for flag in ("is_first", "is_last"):
                if flag not in step or not isinstance(_plain(step.get(flag)), bool):
                    issues.append(SchemaIssue(
                        f"{path}.{flag}", "rlds_flag_invalid",
                        f"RLDS {flag} must be present and boolean.",
                        expected="boolean", observed=_scalar_type(step.get(flag)),
                    ))
        first_positions = [
            i for i, step in enumerate(steps)
            if isinstance(step, Mapping) and _plain(step.get("is_first")) is True
        ]
        last_positions = [
            i for i, step in enumerate(steps)
            if isinstance(step, Mapping) and _plain(step.get("is_last")) is True
        ]
        if first_positions != [0]:
            issues.append(SchemaIssue(
                f"{base}.steps", "rlds_first_step_invalid",
                "Exactly the first step must set is_first=true.",
                expected=[0], observed=first_positions,
            ))
        expected_last = [len(steps) - 1]
        if last_positions != expected_last:
            issues.append(SchemaIssue(
                f"{base}.steps", "rlds_last_step_invalid",
                "Exactly the final step must set is_last=true.",
                expected=expected_last, observed=last_positions,
            ))
        if episode.get("invalid") is True:
            issues.append(SchemaIssue(
                base, "rlds_episode_marked_invalid",
                "Episode metadata marks this episode invalid.", severity="warning",
            ))
    duplicates = sorted(name for name, count in Counter(episode_ids).items() if count > 1)
    if duplicates:
        issues.append(SchemaIssue(
            "$", "rlds_duplicate_episode_id", "Episode identifiers are not unique.",
            expected="unique", observed=duplicates,
        ))
    return SchemaValidationResult(tuple(issues))


def validate_lerobot_frames(frame: Any) -> SchemaValidationResult:
    """Validate the temporal identity columns in a LeRobot frame table."""
    issues: list[SchemaIssue] = []
    required = {"episode_index", "frame_index", "timestamp"}
    columns = set(map(str, getattr(frame, "columns", [])))
    for name in sorted(required - columns):
        issues.append(SchemaIssue(
            f"$.{name}", "lerobot_field_missing",
            "LeRobot frame table is missing a temporal identity field.",
            expected="present", observed="missing",
        ))
    if required - columns:
        return SchemaValidationResult(tuple(issues))
    duplicated = frame.duplicated(subset=["episode_index", "frame_index"])
    if bool(duplicated.any()):
        issues.append(SchemaIssue(
            "$", "lerobot_duplicate_frame",
            "Episode/frame identity pairs must be unique.",
            expected="unique", observed=int(duplicated.sum()),
        ))
    for episode_id, group in frame.groupby("episode_index", sort=False):
        frame_ids = group["frame_index"].tolist()
        if frame_ids != sorted(frame_ids):
            issues.append(SchemaIssue(
                f"$.episode[{episode_id}]", "lerobot_frame_order",
                "Frame indices must be monotonic within an episode.",
                expected="monotonic", observed=frame_ids[:20],
            ))
        timestamps = group["timestamp"].tolist()
        if timestamps != sorted(timestamps):
            issues.append(SchemaIssue(
                f"$.episode[{episode_id}]", "lerobot_timestamp_order",
                "Timestamps must be monotonic within an episode.",
                expected="monotonic", observed=timestamps[:20],
            ))
    return SchemaValidationResult(tuple(issues))


_COMPLEX_SOURCE_TYPES = {
    "pdf", "jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp",
    "mp4", "mov", "mkv", "avi", "webm", "m4v", "lerobot", "rlds",
}


def validate_complex_loader_code(
    code: str,
    source_type: str | None,
    *,
    primary_source: str | None = None,
) -> SchemaValidationResult:
    """Ensure generated code uses the bounded loader for a complex primary source."""
    if (source_type or "").lower() not in _COMPLEX_SOURCE_TYPES:
        return SchemaValidationResult(())
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        # The syntax validator owns this failure class.
        return SchemaValidationResult(())

    uses_loader = False
    imports_loader = False
    unsafe_primary_reads: list[int] = []
    tabular_readers = {
        "read_csv", "read_json", "read_parquet", "read_excel",
        "read_csv_best_effort",
    }
    primary_names = {
        "DATA_SOURCE", "DATA_SOURCE_URI", "data_source",
        "data_source_location", "source_path",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "avaloka.io":
            if any(
                alias.name == "load_dataset"
                and alias.asname in {None, "load_dataset"}
                for alias in node.names
            ):
                imports_loader = True
            continue
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            name = ""
        if name == "load_dataset":
            uses_loader = True
        if name not in tabular_readers or not node.args:
            continue
        argument = node.args[0]
        reads_primary = (
            isinstance(argument, ast.Name) and argument.id in primary_names
        )
        if (
            isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and primary_source
            and argument.value == primary_source
        ):
            reads_primary = True
        if reads_primary:
            unsafe_primary_reads.append(node.lineno)

    issues: list[SchemaIssue] = []
    if not (imports_loader and uses_loader):
        issues.append(SchemaIssue(
            "$.generated_code",
            "complex_loader_missing",
            "Complex sources must be loaded with Avaloka load_dataset(source).frame "
            "so bounds and schema validation apply.",
            expected="load_dataset(source).frame",
            observed="missing load_dataset import or call",
        ))
    for line in unsafe_primary_reads:
        issues.append(SchemaIssue(
            f"$.generated_code.line[{line}]",
            "unsafe_complex_tabular_reader",
            "The primary complex source is passed to a tabular pandas reader. "
            "Use Avaloka load_dataset instead.",
            expected="load_dataset(source).frame",
            observed="tabular reader",
        ))
    return SchemaValidationResult(tuple(issues))
