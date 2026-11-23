#!/usr/bin/env python3
"""Put the memory plane's embedding model, pinned, into the HF cache for the test gate.

    python scripts/prefetch_test_embedding_model.py

tests/test_embedding_local_by_default.py runs the REAL bundled model with the
network refused, to prove the default embedding path makes no request. Under
CI it fails, rather than skips, when the model is not cached, so the gate must
supply it: bitbucket-pipelines.yml and .github/workflows/ci.yml run this before
pytest, and a local gate image runs it at build time.

Pinned by commit, unlike scripts/prefetch_models.py (which the API image uses
and which follows the Hub's main): a test that passes on one revision and fails
on the next should be a change someone made, not one the Hub made. The app
loads the model by name, which resolves through refs/main, so that ref is
pointed at the pinned commit.

Self-contained (no ``app`` imports), like scripts/prefetch_models.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = "sentence-transformers/all-MiniLM-L6-v2"
# The revision the API image bakes today (its refs/main), so the gate tests the
# same weights users run.
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
# Only what sentence-transformers loads with the safetensors weights; the
# ONNX/OpenVINO/TF/legacy-bin copies would multiply the download for nothing.
IGNORE = ["onnx/*", "openvino/*", "*.h5", "*.ot", "*.msgpack", "pytorch_model.bin"]


def main() -> int:
    if (os.environ.get("HF_HUB_OFFLINE") or "").strip().lower() in ("1", "true", "yes"):
        print("HF_HUB_OFFLINE is set; unset it to download (this script must reach huggingface.co).")
        return 2
    from huggingface_hub import constants, snapshot_download

    path = snapshot_download(REPO, revision=REVISION, ignore_patterns=IGNORE)
    refs = Path(constants.HF_HUB_CACHE) / ("models--" + REPO.replace("/", "--")) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(REVISION)
    print(f"fetched  {REPO}@{REVISION[:12]} -> {path}")

    # Prove it loads the way the app loads it: by name, offline.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    dim = len(SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True).encode("prefetch"))
    print(f"ok       all-MiniLM-L6-v2 loads offline by name ({dim}d)")
    return 0 if dim == 384 else 1


if __name__ == "__main__":
    sys.exit(main())
