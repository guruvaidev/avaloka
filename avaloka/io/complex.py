"""Bounded adapters for document, media and robotics datasets.

The adapters materialize *inspection frames*: text, temporal identity, sensor
values small enough to handle safely, and references/metadata for large binary
payloads.  They do not pretend that OCR, video decoding or TensorFlow exists;
optional capabilities are reported and required capabilities fail with an
actionable installation message.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from avaloka.io.schema import (
    infer_complex_schema,
    validate_complex_schema,
    validate_lerobot_frames,
    validate_rlds_episodes,
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


class ComplexDataError(ValueError):
    """A complex source is malformed, unsafe or cannot be decoded."""


class MissingCapabilityError(ComplexDataError):
    """An optional runtime required for the requested operation is absent."""


@dataclass(frozen=True)
class ComplexDataOptions:
    """Hard materialization limits for potentially large or hostile inputs."""

    ocr: str = "auto"  # auto, always, never
    ocr_timeout_seconds: int = 20
    max_input_bytes: int = 4 * 1024 * 1024 * 1024
    max_pdf_pages: int = 100
    max_text_chars_per_page: int = 200_000
    max_image_pixels: int = 40_000_000
    max_video_frames: int = 32
    max_rows: int = 100_000
    max_episodes: int = 100
    max_steps_per_episode: int = 2_000
    max_inline_tensor_values: int = 4_096
    max_nested_depth: int = 32
    rlds_split: str = "all"

    def __post_init__(self) -> None:
        if self.ocr not in {"auto", "always", "never"}:
            raise ValueError("ocr must be 'auto', 'always' or 'never'.")
        numeric = {
            name: value for name, value in vars(self).items()
            if name.startswith("max_") or name == "ocr_timeout_seconds"
        }
        if any(not isinstance(value, int) or value <= 0
               for value in numeric.values()):
            raise ValueError("Complex-data limits must be positive integers.")


@dataclass
class ComplexLoadResult:
    frame: pd.DataFrame
    fmt: str
    structure: dict[str, Any] = field(default_factory=dict)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def complex_data_capabilities() -> dict[str, dict[str, Any]]:
    """Return feature-level availability without importing heavyweight stacks."""
    pillow = _module_available("PIL")
    tesseract = _module_available("pytesseract") and shutil.which("tesseract") is not None
    return {
        "pdf_text": {"available": _module_available("pypdf"), "extra": "documents"},
        "pdf_ocr": {
            "available": _module_available("pymupdf") and pillow and tesseract,
            "extra": "documents",
        },
        "image": {"available": pillow, "extra": "documents"},
        "image_ocr": {"available": pillow and tesseract, "extra": "documents"},
        "video": {"available": _module_available("cv2"), "extra": "video"},
        "lerobot": {
            "available": _module_available("pyarrow"),
            "extra": None,
        },
        "rlds": {
            "available": (_module_available("tensorflow_datasets")
                          and _module_available("tensorflow")),
            "extra": "rlds",
        },
    }


def _require(module: str, *, feature: str, extra: str) -> Any:
    try:
        return importlib.import_module(module)
    except (ImportError, ModuleNotFoundError) as exc:
        raise MissingCapabilityError(
            f"{feature} requires optional dependency {module!r}. "
            f"Install Avaloka with `pip install 'avaloka[{extra}]'`."
        ) from exc


def detect_complex_format(path: Path) -> str | None:
    """Detect a supported complex format from a local path and layout."""
    if path.is_dir():
        if (path / "meta" / "info.json").is_file() and (path / "data").is_dir():
            return "lerobot"
        if ((path / "dataset_info.json").is_file()
                or (path / "features.json").is_file()
                or any(path.glob("*.tfrecord*"))):
            return "rlds"
        return None
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _VIDEO_SUFFIXES:
        return "video"
    return None


def _check_file_size(path: Path, options: ComplexDataOptions) -> None:
    if path.stat().st_size > options.max_input_bytes:
        raise ComplexDataError(
            f"Input is {path.stat().st_size} bytes; limit is {options.max_input_bytes}."
        )


def _bound_text(value: Any, maximum: int) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    if len(text) <= maximum:
        return text, False
    marker = "\n[TRUNCATED]"
    return (text[:max(0, maximum - len(marker))] + marker)[:maximum], True


def _ocr_ready(*, pdf: bool = False) -> bool:
    caps = complex_data_capabilities()
    return bool(caps["pdf_ocr" if pdf else "image_ocr"]["available"])


def _ocr_image(image: Any, options: ComplexDataOptions) -> str:
    pytesseract = _require("pytesseract", feature="OCR", extra="documents")
    if shutil.which("tesseract") is None:
        raise MissingCapabilityError(
            "OCR requires the Tesseract executable in PATH in addition to pytesseract."
        )
    return str(pytesseract.image_to_string(
        image, timeout=options.ocr_timeout_seconds
    ))


def _load_pdf(path: Path, options: ComplexDataOptions) -> ComplexLoadResult:
    _check_file_size(path, options)
    pypdf = _require("pypdf", feature="PDF text extraction", extra="documents")
    try:
        reader = pypdf.PdfReader(str(path), strict=False)
        page_count = len(reader.pages)
    except Exception as exc:
        raise ComplexDataError(f"Unable to open PDF {path.name!r}: {exc}") from exc
    warnings: list[str] = []
    if page_count > options.max_pdf_pages:
        warnings.append(
            f"PDF has {page_count} pages; inspected first {options.max_pdf_pages}."
        )
    rows: list[dict[str, Any]] = []
    text_truncated = False
    for page_index in range(min(page_count, options.max_pdf_pages)):
        try:
            extracted = reader.pages[page_index].extract_text() or ""
        except Exception as exc:
            extracted = ""
            warnings.append(f"Page {page_index + 1} text extraction failed: {exc}")
        method = "embedded_text"
        should_ocr = options.ocr == "always" or (
            options.ocr == "auto" and not extracted.strip()
        )
        if should_ocr:
            if _ocr_ready(pdf=True):
                extracted = _ocr_pdf_page(path, page_index, options)
                method = "ocr"
            elif options.ocr == "always":
                raise MissingCapabilityError(
                    "PDF OCR requires PyMuPDF, Pillow, pytesseract and the "
                    "Tesseract executable. Install `avaloka[documents]`."
                )
            else:
                method = "no_text_ocr_unavailable"
                warnings.append(
                    f"Page {page_index + 1} has no embedded text and OCR is unavailable."
                )
        bounded, truncated = _bound_text(extracted, options.max_text_chars_per_page)
        text_truncated = text_truncated or truncated
        rows.append({
            "page_number": page_index + 1,
            "text": bounded,
            "char_count": len(bounded),
            "extraction_method": method,
        })
    if text_truncated:
        warnings.append("At least one page exceeded the configured text character limit.")
    frame = pd.DataFrame.from_records(rows)
    validation = validate_complex_schema(rows, infer_complex_schema(rows))
    return ComplexLoadResult(frame, "pdf", {
        "normalization": "document_pages_v1",
        "page_count": page_count,
        "materialized_pages": len(rows),
        "truncated": page_count > len(rows) or text_truncated,
        "warnings": sorted(set(warnings)),
        "capabilities": complex_data_capabilities(),
        "complex_schema": infer_complex_schema(rows),
        "schema_validation": validation.to_dict(),
        "text_columns": ["text"],
    })


def _ocr_pdf_page(path: Path, page_index: int, options: ComplexDataOptions) -> str:
    fitz = _require("pymupdf", feature="PDF OCR rendering", extra="documents")
    image_module = _require("PIL.Image", feature="PDF OCR", extra="documents")
    document = fitz.open(str(path))
    try:
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(dpi=150, alpha=False)
        if pixmap.width * pixmap.height > options.max_image_pixels:
            raise ComplexDataError(
                f"Rendered PDF page exceeds {options.max_image_pixels} pixels."
            )
        image = image_module.frombytes(
            "RGB", (pixmap.width, pixmap.height), pixmap.samples
        )
        return _ocr_image(image, options)
    finally:
        document.close()


def _load_image(path: Path, options: ComplexDataOptions) -> ComplexLoadResult:
    _check_file_size(path, options)
    image_module = _require("PIL.Image", feature="Image ingestion", extra="documents")
    warnings: list[str] = []
    with image_module.open(str(path)) as image:
        width, height = map(int, image.size)
        if width * height > options.max_image_pixels:
            raise ComplexDataError(
                f"Image has {width * height} pixels; limit is {options.max_image_pixels}."
            )
        text = ""
        method = "metadata_only"
        if options.ocr != "never":
            if _ocr_ready():
                text = _ocr_image(image, options)
                method = "ocr"
            elif options.ocr == "always":
                raise MissingCapabilityError(
                    "Image OCR requires Pillow, pytesseract and the Tesseract executable."
                )
            else:
                warnings.append("OCR is unavailable; image metadata was loaded without text.")
        text, truncated = _bound_text(text, options.max_text_chars_per_page)
        row = {
            "file_name": path.name,
            "format": str(getattr(image, "format", "") or path.suffix[1:]).lower(),
            "width": width,
            "height": height,
            "mode": str(image.mode),
            "text": text,
            "extraction_method": method,
        }
    schema = infer_complex_schema([row])
    return ComplexLoadResult(pd.DataFrame.from_records([row]), "image", {
        "normalization": "image_metadata_v1",
        "truncated": truncated,
        "warnings": warnings,
        "capabilities": complex_data_capabilities(),
        "complex_schema": schema,
        "schema_validation": validate_complex_schema([row], schema).to_dict(),
        "text_columns": ["text"],
    })


def _video_indices(frame_count: int, maximum: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= maximum:
        return list(range(frame_count))
    return sorted(set(int(value) for value in np.linspace(
        0, frame_count - 1, num=maximum
    )))


def _load_video(path: Path, options: ComplexDataOptions) -> ComplexLoadResult:
    _check_file_size(path, options)
    cv2 = _require("cv2", feature="Video ingestion", extra="video")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ComplexDataError(f"Unable to decode video {path.name!r}.")
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    try:
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps_value = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps_value if math.isfinite(fps_value) and fps_value > 0 else None
        indices = _video_indices(frame_count, options.max_video_frames)
        if not indices:
            indices = list(range(options.max_video_frames))
            sequential = True
        else:
            sequential = False
        for frame_index in indices:
            if not sequential:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, pixels = capture.read()
            if not ok:
                if sequential:
                    break
                warnings.append(f"Frame {frame_index} could not be decoded.")
                continue
            height, width = map(int, pixels.shape[:2])
            text = ""
            method = "sampled_frame"
            if options.ocr != "never":
                if _ocr_ready():
                    image_module = _require(
                        "PIL.Image", feature="Video-frame OCR", extra="documents"
                    )
                    rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
                    text = _ocr_image(image_module.fromarray(rgb), options)
                    method = "sampled_frame_ocr"
                elif options.ocr == "always":
                    raise MissingCapabilityError(
                        "Video OCR requires the video and documents optional extras."
                    )
            text, _ = _bound_text(text, options.max_text_chars_per_page)
            rows.append({
                "file_name": path.name,
                "frame_index": int(frame_index),
                "timestamp_ms": (round(frame_index * 1000 / fps, 3) if fps else None),
                "width": width,
                "height": height,
                "text": text,
                "extraction_method": method,
            })
    finally:
        capture.release()
    if frame_count > len(rows):
        warnings.append(
            f"Video has {frame_count} frames; uniformly sampled {len(rows)}."
        )
    schema = infer_complex_schema(rows)
    return ComplexLoadResult(pd.DataFrame.from_records(rows), "video", {
        "normalization": "video_frame_index_v1",
        "frame_count": frame_count,
        "sampled_frame_count": len(rows),
        "fps": fps,
        "duration_ms": (round(frame_count * 1000 / fps, 3)
                        if fps and frame_count else None),
        "truncated": frame_count > len(rows),
        "warnings": warnings,
        "capabilities": complex_data_capabilities(),
        "complex_schema": schema,
        "schema_validation": validate_complex_schema(rows, schema).to_dict(),
        "text_columns": ["text"],
    })


def _bounded_value(
    value: Any,
    options: ComplexDataOptions,
    *,
    _depth: int = 0,
) -> Any:
    if _depth > options.max_nested_depth:
        raise ComplexDataError(
            f"Nested value exceeds maximum depth {options.max_nested_depth}."
        )
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= options.max_inline_tensor_values:
            return value.tolist()
        return {
            "__tensor__": True,
            "dtype": str(value.dtype),
            "shape": [int(size) for size in value.shape],
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return {
            "__binary__": True,
            "byte_length": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(key): _bounded_value(child, options, _depth=_depth + 1)
                for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            array = None
        if (array is not None and array.dtype.kind in "biufc"
                and array.size > options.max_inline_tensor_values):
            return {
                "__tensor__": True,
                "dtype": str(array.dtype),
                "shape": [int(size) for size in array.shape],
            }
        return [_bounded_value(child, options, _depth=_depth + 1)
                for child in value]
    return value


def _normalize_records(records: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    from avaloka.io.loader import normalize_records
    return normalize_records(records)


def _read_parquet_records(
    root: Path,
    files: Sequence[Path],
    options: ComplexDataOptions,
) -> tuple[list[dict[str, Any]], bool]:
    parquet = _require("pyarrow.parquet", feature="LeRobot ingestion", extra="complex")
    records: list[dict[str, Any]] = []
    truncated = False
    for path in files:
        remaining = options.max_rows - len(records)
        if remaining <= 0:
            truncated = True
            break
        parquet_file = parquet.ParquetFile(str(path))
        for batch in parquet_file.iter_batches(batch_size=min(4096, remaining)):
            for record in batch.to_pylist():
                bounded = _bounded_value(record, options)
                bounded["_source_file"] = str(path.relative_to(root))
                records.append(bounded)
                if len(records) >= options.max_rows:
                    truncated = True
                    break
            if len(records) >= options.max_rows:
                break
    return records, truncated


def _load_lerobot(path: Path, options: ComplexDataOptions) -> ComplexLoadResult:
    info_path = path / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComplexDataError(f"Invalid LeRobot metadata {info_path}: {exc}") from exc
    files = sorted((path / "data").glob("**/*.parquet"))
    unsafe = [item for item in files if item.is_symlink()]
    if unsafe:
        raise ComplexDataError(
            f"LeRobot dataset contains symbolic-link data files: {unsafe[:3]}."
        )
    if not files:
        raise ComplexDataError("LeRobot dataset has no data/**/*.parquet files.")
    records, truncated = _read_parquet_records(path, files, options)
    frame, normalization = _normalize_records(records)
    validation = validate_lerobot_frames(frame)
    video_files = sorted((path / "videos").glob("**/*")) if (path / "videos").is_dir() else []
    video_files = [item for item in video_files
                   if item.is_file() and not item.is_symlink()
                   and item.suffix.lower() in _VIDEO_SUFFIXES]
    schema = infer_complex_schema(records)
    structure = {
        **normalization,
        "normalization": "lerobot_v3_inspection_v1",
        "format_version": info.get("codebase_version", info.get("format_version")),
        "declared_features": info.get("features", {}),
        "declared_total_episodes": info.get("total_episodes"),
        "declared_total_frames": info.get("total_frames"),
        "parquet_file_count": len(files),
        "video_file_count": len(video_files),
        "video_keys": sorted({item.relative_to(path / "videos").parts[0]
                              for item in video_files}),
        "materialized_rows": len(records),
        "truncated": truncated,
        "warnings": [],
        "capabilities": complex_data_capabilities(),
        "complex_schema": schema,
        "schema_validation": validation.to_dict(),
    }
    return ComplexLoadResult(frame, "lerobot", structure)


def _python_value(
    value: Any,
    *,
    max_depth: int = 32,
    _depth: int = 0,
) -> Any:
    if _depth > max_depth:
        raise ComplexDataError(
            f"Nested value exceeds maximum depth {max_depth}."
        )
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        return _python_value(
            value.numpy(), max_depth=max_depth, _depth=_depth + 1
        )
    if isinstance(value, Mapping):
        return {str(key): _python_value(
            child, max_depth=max_depth, _depth=_depth + 1
        ) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python_value(
            child, max_depth=max_depth, _depth=_depth + 1
        ) for child in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _steps_from_value(
    value: Any,
    tfds: Any,
    *,
    max_depth: int = 32,
) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        converted = {
            str(key): _python_value(child, max_depth=max_depth)
            for key, child in value.items()
        }
        lengths = [len(child) for child in converted.values()
                   if isinstance(child, (list, tuple, np.ndarray))]
        if not lengths:
            return [converted]
        count = min(lengths)
        return [{
            key: (
                _python_value(child[index], max_depth=max_depth)
                if isinstance(child, (list, tuple, np.ndarray)) else child
            )
            for key, child in converted.items()
        } for index in range(count)]
    try:
        iterable = tfds.as_numpy(value)
    except (TypeError, AttributeError):
        iterable = value
    return [_python_value(step, max_depth=max_depth) for step in iterable]


def normalize_rlds_episodes(
    episodes: Iterable[Mapping[str, Any]],
    *,
    options: ComplexDataOptions | None = None,
) -> ComplexLoadResult:
    """Normalize already-decoded RLDS episodes without requiring TensorFlow."""
    options = options or ComplexDataOptions()
    materialized: list[dict[str, Any]] = []
    truncated = False
    for episode_index, episode in enumerate(episodes):
        if episode_index >= options.max_episodes:
            truncated = True
            break
        converted = _python_value(
            episode, max_depth=options.max_nested_depth
        )
        raw_steps = converted.get("steps", [])
        all_steps = list(raw_steps)
        steps = all_steps[:options.max_steps_per_episode]
        if len(all_steps) > len(steps):
            truncated = True
        converted["steps"] = steps
        converted.setdefault("episode_id", str(episode_index))
        materialized.append(converted)

    contract = infer_complex_schema(materialized)
    validation = validate_rlds_episodes(materialized)
    rows: list[dict[str, Any]] = []
    for episode in materialized:
        metadata = {key: value for key, value in episode.items()
                    if key not in {"steps", "episode_id"}}
        for step_index, step in enumerate(episode["steps"]):
            record = {
                "episode_id": episode["episode_id"],
                "step_index": step_index,
                "episode_metadata": metadata,
                **dict(step),
            }
            rows.append(_bounded_value(record, options))
            if len(rows) >= options.max_rows:
                truncated = True
                break
        if len(rows) >= options.max_rows:
            break
    frame, normalization = _normalize_records(rows)
    return ComplexLoadResult(frame, "rlds", {
        **normalization,
        "normalization": "rlds_episode_steps_v1",
        "materialized_episodes": len(materialized),
        "materialized_steps": len(rows),
        "truncated": truncated,
        "warnings": [],
        "capabilities": complex_data_capabilities(),
        "complex_schema": contract,
        "schema_validation": validation.to_dict(),
    })


def _load_rlds(path: Path, options: ComplexDataOptions) -> ComplexLoadResult:
    tfds = _require("tensorflow_datasets", feature="RLDS ingestion", extra="rlds")
    _require("tensorflow", feature="RLDS ingestion", extra="rlds")
    try:
        builder = tfds.builder_from_directory(str(path))
        dataset = builder.as_dataset(split=options.rlds_split)
        decoded_episodes = tfds.as_numpy(dataset)
    except Exception as exc:
        raise ComplexDataError(f"Unable to open RLDS directory {path}: {exc}") from exc

    def convert() -> Iterable[Mapping[str, Any]]:
        for raw_episode in decoded_episodes:
            episode = dict(raw_episode)
            raw_steps = episode.pop("steps", [])
            converted = _python_value(
                episode, max_depth=options.max_nested_depth
            )
            converted["steps"] = _steps_from_value(
                raw_steps, tfds,
                max_depth=options.max_nested_depth,
            )
            yield converted

    return normalize_rlds_episodes(convert(), options=options)


def load_complex_data(
    path: Path,
    *,
    options: ComplexDataOptions | None = None,
) -> ComplexLoadResult:
    """Load a detected complex source into a bounded inspection frame."""
    options = options or ComplexDataOptions()
    fmt = detect_complex_format(path)
    if fmt is None:
        raise ComplexDataError(f"Unsupported complex data source: {path}.")
    if fmt == "pdf":
        return _load_pdf(path, options)
    if fmt == "image":
        return _load_image(path, options)
    if fmt == "video":
        return _load_video(path, options)
    if fmt == "lerobot":
        return _load_lerobot(path, options)
    if fmt == "rlds":
        return _load_rlds(path, options)
    raise AssertionError(f"Unhandled complex format: {fmt}")
