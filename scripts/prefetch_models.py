#!/usr/bin/env python3
"""Download every model the stack embeds with, so it can run with no network.

    python scripts/prefetch_models.py          # download (needs network once)
    python scripts/prefetch_models.py --check  # verify, offline; exit 1 if any is missing

The API image runs this at build time (deploy/docker/Dockerfile.api), and the
quickstart runs ``--check`` inside the image before it starts anything. A
host install ("local processes") runs it once by hand.

Models, and who needs them:

* ``SENTENCE_TRANSFORMER_MODEL`` (default ``all-MiniLM-L6-v2``): the memory
  plane, on every query.
* ``mixedbread-ai/mxbai-embed-large-v1``: the data-transfer agent's Daft
  documentation retrieval. Set ``AVALOKA_PREFETCH_DTA_MODEL=0`` to skip it
  (about 670 MB); a transfer then runs without retrieved docs.
* chroma's ONNX ``all-MiniLM-L6-v2``: chroma's default embedding function,
  into ``CHROMA_ONNX_MODEL_DIR`` when set.

Self-contained on purpose (no ``app`` imports): the image's builder stage has
the Python environment but not the source tree.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _models() -> list[str]:
    names = [os.environ.get("SENTENCE_TRANSFORMER_MODEL") or "all-MiniLM-L6-v2"]
    if (os.environ.get("AVALOKA_PREFETCH_DTA_MODEL") or "1").strip().lower() not in ("0", "false", "no"):
        names.append("mixedbread-ai/mxbai-embed-large-v1")
    return names


def _chroma_ef():
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    ef = ONNXMiniLM_L6_V2()
    configured = os.environ.get("CHROMA_ONNX_MODEL_DIR")
    if configured:
        ef.DOWNLOAD_PATH = Path(configured)
    return ef


def check() -> int:
    # Offline for real: a check that may download proves nothing.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    missing = []
    from sentence_transformers import SentenceTransformer

    for name in _models():
        try:
            SentenceTransformer(name)
            print(f"ok       {name}")
        except Exception as exc:  # noqa: BLE001
            missing.append(name)
            print(f"MISSING  {name}: {type(exc).__name__}")
    ef = _chroma_ef()
    onnx = Path(ef.DOWNLOAD_PATH) / ef.EXTRACTED_FOLDER_NAME / "model.onnx"
    if onnx.exists():
        print(f"ok       chroma onnx all-MiniLM-L6-v2 ({onnx.parent})")
    else:
        missing.append("chroma onnx")
        print(f"MISSING  chroma onnx all-MiniLM-L6-v2 (expected {onnx})")
    if missing:
        print(f"{len(missing)} model(s) missing; run scripts/prefetch_models.py with network access.")
        return 1
    print("all models cached; the stack can run offline")
    return 0


def fetch() -> int:
    if (os.environ.get("HF_HUB_OFFLINE") or "").strip().lower() in ("1", "true", "yes"):
        print("HF_HUB_OFFLINE is set; unset it to download (this script must reach huggingface.co).")
        return 2
    from sentence_transformers import SentenceTransformer

    for name in _models():
        SentenceTransformer(name).encode("prefetch")
        print(f"fetched  {name}")
    ef = _chroma_ef()
    ef(["prefetch"])  # downloads and verifies the archive's SHA-256, then extracts it
    print(f"fetched  chroma onnx all-MiniLM-L6-v2 ({ef.DOWNLOAD_PATH})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify offline that every model is cached")
    args = ap.parse_args(argv)
    return check() if args.check else fetch()


if __name__ == "__main__":
    sys.exit(main())
