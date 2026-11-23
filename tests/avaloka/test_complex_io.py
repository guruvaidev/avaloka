"""Import-light coverage for document, media and robotics ingestion."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import avaloka.io.complex as complex_module
from avaloka.io import (
    ComplexDataError,
    ComplexDataOptions,
    MissingCapabilityError,
    infer_complex_schema,
    load_dataset,
    normalize_rlds_episodes,
    validate_complex_schema,
    validate_complex_loader_code,
    validate_lerobot_frames,
    validate_rlds_episodes,
)


def _rlds_episode(episode_id: str = "episode-1"):
    return {
        "episode_id": episode_id,
        "operator": "human",
        "steps": [
            {
                "is_first": True,
                "is_last": False,
                "is_terminal": False,
                "observation": np.zeros((8, 8, 3), dtype=np.uint8),
                "action": np.array([0.1, 0.2], dtype=np.float32),
            },
            {
                "is_first": False,
                "is_last": True,
                "is_terminal": True,
                "observation": np.ones((8, 8, 3), dtype=np.uint8),
                "action": np.array([0.0, 0.0], dtype=np.float32),
            },
        ],
    }


def test_recursive_schema_detects_missing_fields_and_tensor_shape_drift():
    baseline = [
        {"id": 1, "meta": {"label": "left"},
         "observation": np.zeros((2, 3), dtype=np.float32)},
        {"id": 2, "meta": {"label": "right"},
         "observation": np.ones((2, 3), dtype=np.float32)},
    ]
    schema = infer_complex_schema(baseline)
    candidate = [{
        "meta": {"label": "changed"},
        "observation": np.zeros((4, 3), dtype=np.float32),
        "new_field": True,
    }]

    result = validate_complex_schema(candidate, schema)
    codes = {issue.code for issue in result.issues}

    assert result.valid is False
    assert "required_field_missing" in codes
    assert "tensor_shape_mismatch" in codes
    assert "unexpected_field" in codes


def test_nullable_and_optional_nested_fields_validate():
    values = [{"id": 1, "note": None}, {"id": 2, "note": "ready"}]
    schema = infer_complex_schema(values)

    assert schema["items"]["properties"]["note"]["nullable"] is True
    assert validate_complex_schema(values, schema).valid is True


def test_rlds_normalization_bounds_large_tensors_and_preserves_contract():
    options = ComplexDataOptions(
        ocr="never", max_inline_tensor_values=8, max_steps_per_episode=10
    )

    result = normalize_rlds_episodes([_rlds_episode()], options=options)

    assert result.fmt == "rlds"
    assert len(result.frame) == 2
    assert result.structure["schema_validation"]["valid"] is True
    assert result.structure["materialized_episodes"] == 1
    assert result.frame["episode_metadata.operator"].tolist() == ["human", "human"]
    assert result.frame["observation.__tensor__"].tolist() == [True, True]
    assert json.loads(result.frame.loc[0, "observation.shape"]) == [8, 8, 3]
    assert json.loads(result.frame.loc[0, "action"]) == pytest.approx([0.1, 0.2])


def test_rlds_validator_rejects_bad_boundaries_and_duplicate_ids():
    invalid = [
        {"episode_id": "duplicate", "steps": [
            {"is_first": False, "is_last": False}
        ]},
        {"episode_id": "duplicate", "steps": [
            {"is_first": True, "is_last": True}
        ]},
    ]

    result = validate_rlds_episodes(invalid)
    codes = {issue.code for issue in result.issues}

    assert result.valid is False
    assert "rlds_first_step_invalid" in codes
    assert "rlds_last_step_invalid" in codes
    assert "rlds_duplicate_episode_id" in codes


def test_lerobot_directory_loads_bounded_parquet_and_validates(tmp_path):
    root = tmp_path / "robot"
    data = root / "data" / "chunk-000"
    meta = root / "meta"
    data.mkdir(parents=True)
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0",
        "total_episodes": 2,
        "total_frames": 3,
        "features": {"action": {"dtype": "float32", "shape": [2]}},
    }))
    pd.DataFrame({
        "episode_index": [0, 0, 1],
        "frame_index": [0, 1, 0],
        "timestamp": [0.0, 0.1, 0.0],
        "action": [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]],
    }).to_parquet(data / "file-000.parquet", index=False)

    dataset = load_dataset(
        str(root), complex_options=ComplexDataOptions(ocr="never", max_rows=2)
    )

    assert dataset.fmt == "lerobot"
    assert dataset.n_rows == 2
    assert dataset.structure["truncated"] is True
    assert dataset.structure["schema_validation"]["valid"] is True
    assert dataset.structure["parquet_file_count"] == 1
    assert set(dataset.frame["_source_file"]) == {"data/chunk-000/file-000.parquet"}
    assert len(dataset.sha256) == 64


def test_lerobot_rejects_symbolic_link_data_files(tmp_path):
    outside = tmp_path / "outside.parquet"
    pd.DataFrame({
        "episode_index": [0],
        "frame_index": [0],
        "timestamp": [0.0],
    }).to_parquet(outside, index=False)
    root = tmp_path / "robot"
    data = root / "data"
    meta = root / "meta"
    data.mkdir(parents=True)
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    (data / "file-000.parquet").symlink_to(outside)

    with pytest.raises(ComplexDataError, match="symbolic-link"):
        load_dataset(str(root))



def test_lerobot_validator_detects_duplicate_and_out_of_order_frames():
    frame = pd.DataFrame({
        "episode_index": [1, 1, 1],
        "frame_index": [1, 0, 0],
        "timestamp": [1.0, 0.0, -1.0],
    })

    codes = {issue.code for issue in validate_lerobot_frames(frame).issues}

    assert "lerobot_duplicate_frame" in codes
    assert "lerobot_frame_order" in codes
    assert "lerobot_timestamp_order" in codes


def test_pdf_embedded_text_adapter_is_bounded_without_real_pdf_runtime(
    tmp_path, monkeypatch
):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"fake-pdf")

    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    fake_pypdf = SimpleNamespace(
        PdfReader=lambda *_args, **_kwargs: SimpleNamespace(
            pages=[Page("first"), Page("second"), Page("third")]
        )
    )
    original_require = complex_module._require

    def fake_require(module, **kwargs):
        if module == "pypdf":
            return fake_pypdf
        return original_require(module, **kwargs)

    monkeypatch.setattr(complex_module, "_require", fake_require)

    dataset = load_dataset(str(path), complex_options=ComplexDataOptions(
        ocr="never", max_pdf_pages=2
    ))

    assert dataset.fmt == "pdf"
    assert dataset.frame["text"].tolist() == ["first", "second"]
    assert dataset.structure["page_count"] == 3
    assert dataset.structure["truncated"] is True


def test_image_metadata_adapter_does_not_claim_ocr_when_disabled(
    tmp_path, monkeypatch
):
    path = tmp_path / "scan.jpeg"
    path.write_bytes(b"fake-image")

    class FakeImage:
        size = (640, 480)
        format = "JPEG"
        mode = "RGB"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    fake_image_module = SimpleNamespace(open=lambda _path: FakeImage())
    original_require = complex_module._require

    def fake_require(module, **kwargs):
        if module == "PIL.Image":
            return fake_image_module
        return original_require(module, **kwargs)

    monkeypatch.setattr(complex_module, "_require", fake_require)

    dataset = load_dataset(
        str(path), complex_options=ComplexDataOptions(ocr="never")
    )

    assert dataset.fmt == "image"
    assert dataset.frame.loc[0, "extraction_method"] == "metadata_only"
    assert dataset.frame.loc[0, "width"] == 640
    assert dataset.frame.loc[0, "text"] == ""

    monkeypatch.setenv("AVALOKA_COMPLEX_OCR", "never")
    from app.agents.sampling_agent_daft import _sample_complex_source

    sampled = _sample_complex_source(str(path), "jpeg", 10)
    assert sampled["error"] is None
    assert sampled["portfolio_samples"]["random_baseline"][0]["width"] == 640
    assert sampled["complex_structure"]["normalization"] == "image_metadata_v1"


def test_video_adapter_uniformly_samples_frames(tmp_path, monkeypatch):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fake-video")

    class Capture:
        def __init__(self):
            self.position = 0

        def isOpened(self):
            return True

        def get(self, key):
            return {1: 10, 2: 5.0}.get(key, 0)

        def set(self, _key, value):
            self.position = int(value)

        def read(self):
            return True, np.zeros((12, 20, 3), dtype=np.uint8)

        def release(self):
            return None

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FRAME_COUNT=1,
        CAP_PROP_FPS=2,
        CAP_PROP_POS_FRAMES=3,
        VideoCapture=lambda _path: Capture(),
    )
    original_require = complex_module._require

    def fake_require(module, **kwargs):
        if module == "cv2":
            return fake_cv2
        return original_require(module, **kwargs)

    monkeypatch.setattr(complex_module, "_require", fake_require)

    dataset = load_dataset(str(path), complex_options=ComplexDataOptions(
        ocr="never", max_video_frames=3
    ))

    assert dataset.fmt == "video"
    assert dataset.frame["frame_index"].tolist() == [0, 4, 9]
    assert dataset.frame["timestamp_ms"].tolist() == [0.0, 800.0, 1800.0]
    assert dataset.structure["truncated"] is True


def test_rlds_tfds_directory_adapter_can_decode_nested_steps(
    tmp_path, monkeypatch
):
    root = tmp_path / "rlds"
    root.mkdir()
    (root / "dataset_info.json").write_text("{}")

    class Builder:
        def as_dataset(self, *, split):
            assert split == "all"
            return [_rlds_episode()]

    fake_tfds = SimpleNamespace(
        builder_from_directory=lambda _path: Builder(),
        as_numpy=lambda value: value,
    )
    original_require = complex_module._require

    def fake_require(module, **kwargs):
        if module == "tensorflow_datasets":
            return fake_tfds
        if module == "tensorflow":
            return SimpleNamespace()
        return original_require(module, **kwargs)

    monkeypatch.setattr(complex_module, "_require", fake_require)

    dataset = load_dataset(str(root), complex_options=ComplexDataOptions(
        ocr="never", max_inline_tensor_values=8
    ))

    assert dataset.fmt == "rlds"
    assert dataset.n_rows == 2
    assert dataset.structure["schema_validation"]["valid"] is True


def test_missing_pdf_runtime_fails_with_actionable_capability_error(
    tmp_path, monkeypatch
):
    path = tmp_path / "missing.pdf"
    path.write_bytes(b"fake")

    def unavailable(module, **_kwargs):
        raise MissingCapabilityError(f"missing {module}; install avaloka[documents]")

    monkeypatch.setattr(complex_module, "_require", unavailable)

    with pytest.raises(MissingCapabilityError, match="documents"):
        load_dataset(str(path), complex_options=ComplexDataOptions(ocr="never"))


def test_complex_options_reject_unbounded_or_unknown_modes():
    with pytest.raises(ValueError, match="ocr"):
        ComplexDataOptions(ocr="sometimes")
    with pytest.raises(ValueError, match="positive"):
        ComplexDataOptions(max_video_frames=0)



def test_rlds_conversion_rejects_excessive_nesting():
    with pytest.raises(ComplexDataError, match="maximum depth"):
        normalize_rlds_episodes(
            [_rlds_episode()],
            options=ComplexDataOptions(ocr="never", max_nested_depth=1),
        )



def test_generated_code_for_complex_source_must_use_bounded_loader():
    invalid = validate_complex_loader_code(
        "import pandas as pd\ndf = pd.read_csv(DATA_SOURCE_URI)",
        "pdf",
        primary_source="/tmp/report.pdf",
    )
    codes = {issue.code for issue in invalid.issues}

    assert invalid.valid is False
    assert codes == {"complex_loader_missing", "unsafe_complex_tabular_reader"}

    valid = validate_complex_loader_code(
        "from avaloka.io import load_dataset\n"
        "df = load_dataset('/tmp/report.pdf').frame\n"
        "other = pd.read_csv('/tmp/labels.csv')",
        "pdf",
        primary_source="/tmp/report.pdf",
    )
    assert valid.valid is True


def test_generated_code_loader_guard_is_noop_for_tabular_source():
    result = validate_complex_loader_code(
        "import pandas as pd\ndf = pd.read_csv('/tmp/data.csv')",
        "csv",
        primary_source="/tmp/data.csv",
    )
    assert result.valid is True
