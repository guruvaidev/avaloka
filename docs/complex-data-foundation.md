# Complex Data Foundation

> Status: implemented foundation for `develop-1.7` benchmark preparation.
> Scope: bounded inspection, recursive schema contracts, validation, product upload
> routing and code-execution loading. This is not a claim that Avaloka already has a
> trained vision-language model or robot-policy training pipeline.

## Decision

Benchmark work must not treat every source as a flat table. Avaloka now has a common
inspection contract for documents, images, videos and episodic robot data before broader
multimodal benchmarks are run.

The contract has four rules:

1. Materialization is bounded before parsing: input bytes, pages, frames, rows, episodes,
   steps, image pixels, OCR time and inline tensor values all have explicit ceilings.
2. Heavy runtimes are optional capabilities. Missing PDF, OCR, video or TensorFlow support
   produces an actionable capability error; it never silently claims that extraction ran.
3. Binary payloads and large tensors are not copied into arbitrary DataFrame cells. Avaloka
   retains hashes, dtypes, shapes, timestamps and source references; small sensor vectors may
   be inlined.
4. Every loaded source emits a recursive schema and a machine-readable validation result.

## Supported surfaces

| Source | SDK / local path | `/api/upload` | Materialized inspection unit | Required capability |
| --- | --- | --- | --- | --- |
| PDF | Yes | Yes | One row per bounded page, embedded text or OCR text | `avaloka[documents]`; OCR also needs the Tesseract executable |
| JPEG/PNG/TIFF/BMP/WebP | Yes | Yes | Image metadata plus optional OCR text | `avaloka[documents]` |
| MP4/MOV/MKV/AVI/WebM/M4V | Yes | Yes | Uniformly sampled frame index, dimensions, timestamp and optional OCR | `avaloka[video]`; OCR also needs `avaloka[documents]` |
| LeRobot v3 directory | Yes | Not as a multipart directory | Bounded Parquet frame rows plus episode/video/feature manifest | Core PyArrow dependency |
| RLDS / TFDS directory | Yes | Not as a multipart directory | Bounded episode-step rows plus RLDS invariant report | `avaloka[rlds]` |
| CSV/Parquet/JSON/JSONL | Existing support | Existing support | Tabular rows | Core dependencies |

LeRobot v3 parsing follows its file-based layout: `meta/info.json`, chunked
`data/**/*.parquet`, optional `meta/episodes/**/*.parquet`, and camera streams under
`videos/<camera-key>/**`. It does not require importing the full LeRobot training package.

RLDS parsing follows the canonical episode → steps structure. `is_first` and `is_last` are
mandatory boolean step fields. Episode identifiers, step-field consistency, first/last
boundaries and invalid-episode markers are validated independently of TensorFlow after
decoding.

## Install and runtime capabilities

```bash
# Embedded PDF text, image metadata and OCR Python bindings
pip install 'avaloka[documents]'

# Video frame decoding
pip install 'avaloka[video]'

# Local RLDS / TFDS directories
pip install 'avaloka[rlds]'

# All optional Python runtimes
pip install 'avaloka[complex]'
```

`pytesseract` is only a Python binding. OCR also requires the Tesseract executable to be
installed in the runtime image and available on `PATH`.

Capability discovery is explicit:

```python
from avaloka.io import complex_data_capabilities

print(complex_data_capabilities())
```

## Loading and hard limits

```python
from avaloka.io import ComplexDataOptions, load_dataset

dataset = load_dataset(
    "/data/robot_run",
    complex_options=ComplexDataOptions(
        ocr="auto",                 # auto | always | never
        max_pdf_pages=50,
        max_video_frames=24,
        max_episodes=100,
        max_steps_per_episode=2_000,
        max_rows=100_000,
        max_inline_tensor_values=4_096,
        max_nested_depth=32,
    ),
)

inspection_frame = dataset.frame
recursive_contract = dataset.structure["complex_schema"]
validation = dataset.structure["schema_validation"]
```

Object-store URIs are streamed into a bounded local cache. The default complex input ceiling
is 4 GiB and can be lowered through `ComplexDataOptions.max_input_bytes`. Lineage retains a
redacted source URI and a deterministic content or directory hash.

Product upload uses these environment settings:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `AVALOKA_COMPLEX_OCR` | `auto` | Use OCR when available; `always` fails if OCR is unavailable; `never` disables it |
| `AVALOKA_MAX_PDF_PAGES` | `100` | Maximum inspected PDF pages |
| `AVALOKA_MAX_VIDEO_FRAMES` | `32` | Maximum uniformly sampled video frames |
| `AVALOKA_MAX_TEXT_CHARS_PER_PAGE` | `200000` | Maximum retained text per document page/frame |

## Recursive schema and validators

`infer_complex_schema()` understands:

- nested objects, optional fields and nullable values;
- arrays with recursively inferred item contracts;
- integer, number, boolean, string, byte and datetime scalars;
- dense numeric tensors with dtype, rank and dimension constraints;
- heterogeneous values through explicit union alternatives.

`validate_complex_schema()` reports stable paths and issue codes for missing required fields,
unexpected fields, type changes, tensor dtype changes and tensor shape changes. Unexpected
fields default to warnings so schema evolution is visible without making every additive field
fatal; callers can elevate them to errors.

Format-specific validators add semantic invariants:

- `validate_lerobot_frames()` checks required temporal identity fields, unique
  `(episode_index, frame_index)` pairs, and monotonic frame/timestamp ordering.
- `validate_rlds_episodes()` checks step-object shape, consistent fields, mandatory boolean
  flags, exactly one first/final boundary, unique episode IDs and invalid-episode markers.

## Product execution path

For PDF, image and video uploads:

1. `/api/upload` applies existing byte and count limits and persists the original object.
2. The sampler bypasses Daft's tabular readers and creates one deterministic bounded complex
   inspection portfolio.
3. `complex_structure` is stored with the dataset session, returned additively in upload
   responses and restored when the active dataset changes.
4. The Avaloka planner receives a capped summary of format, capabilities, schema and
   validation state.
5. Deterministic coder, LLM coder and interactive execution paths use
   `avaloka.io.load_dataset(...).frame`; none falls back to `pandas.read_csv()` for a complex
   file.

Existing UI clients can ignore `complex_structure`; the field is optional and additive.

## What this enables in benchmark work

This foundation enables benchmark fixtures to grade separately:

- ingestion success and explicit capability failure;
- OCR text coverage and character error rate;
- PDF page coverage and extraction method;
- video temporal sampling coverage and timestamp correctness;
- recursive schema accuracy and schema-drift detection;
- LeRobot episode/frame identity and sensor tensor integrity;
- RLDS episode-boundary, step-schema and terminal-state integrity;
- cost, latency, bytes read and truncation behavior under configured limits.

The first release gate is the deterministic adapter/validator suite. The next gates should use
licensed public fixtures with hidden mutations, then add multimodal reasoning tasks only after
the extraction ground truth is stable.

## Deliberate non-goals and next transport work

- OCR extracts text; it does not provide general visual question answering, object detection
  or chart understanding.
- Video support indexes uniformly sampled frames and optional frame OCR; it does not yet
  provide audio transcription, shot detection or VLM embeddings.
- LeRobot/RLDS support inspects and validates dataset structure; it does not train or evaluate
  a robot policy.
- Multipart upload cannot represent a directory. LeRobot/RLDS product ingestion still needs a
  safe archive/repository transport, archive-bomb controls, Hub authentication and a background
  materialization job before it should be exposed in the UI.
- Native codec/connector deadlines are still required because Python cannot forcibly cancel
  every blocked C extension or remote filesystem driver.

Those items are follow-up capabilities. They are not silently classified as supported by this
foundation.
