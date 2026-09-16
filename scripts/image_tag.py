#!/usr/bin/env python3
"""Compute the tag for an Avaloka image from the inputs that actually build it.

Answers one question: **does this need rebuilding?**

Each image has two costs. The expensive one is the dependency layer -- the
virtualenv and the baked embedding model, minutes and hundreds of megabytes --
which changes only when requirements.txt or the Dockerfile changes. The cheap
one is `COPY . /app`, which changes on every commit.

Hashing the expensive inputs separately gives a tag that stays the same across
a hundred source-only commits. If `deps-<hash>` already exists in the registry,
the dependency layer does not need rebuilding and the build reuses it --
which is the difference between a two-minute deploy and a twenty-minute one.

    python scripts/image_tag.py api            # -> deps-3f9a1c2e
    python scripts/image_tag.py api --source   # -> src-8b21d07f  (includes app/)
    python scripts/image_tag.py --all
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: What each image's *dependency* layer is built from. Deliberately narrow: add
#: a path here only if changing it invalidates the venv, or every commit will
#: appear to need a full rebuild and the tag becomes useless.
DEPENDENCY_INPUTS: dict[str, list[str]] = {
    "api": ["requirements.txt", "deploy/docker/Dockerfile.api"],
    "ray": ["requirements.txt", "deploy/docker/Dockerfile.ray"],
    "ui": ["ui/package-lock.json", "ui/package.json", "ui/Dockerfile"],
    "functions": ["deploy/docker/Dockerfile.functions"],
}

#: Added on top of the dependency inputs for a full source tag.
SOURCE_INPUTS: dict[str, list[str]] = {
    "api": ["app", "avaloka", "file_handler"],
    "ray": ["app", "avaloka"],
    "ui": ["ui/src", "ui/public"],
    "functions": ["deploy/docker/functions-main"],
}


def _hash_paths(paths: list[str]) -> str:
    """A stable digest of the given files and directories.

    Uses `git hash-object` where possible so the result matches what git would
    record, and falls back to reading bytes for anything untracked.
    """
    digest = hashlib.sha256()
    for rel in sorted(paths):
        target = REPO / rel
        if not target.exists():
            # A missing input is itself a fact worth hashing: it distinguishes
            # "this image has no UI sources" from "the UI sources are empty".
            digest.update(f"{rel}:absent\n".encode())
            continue
        files = sorted(target.rglob("*")) if target.is_dir() else [target]
        for path in files:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(REPO)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:12]


def tag_for(image: str, include_source: bool) -> str:
    if image not in DEPENDENCY_INPUTS:
        raise SystemExit(f"unknown image {image!r}; expected one of "
                         f"{', '.join(sorted(DEPENDENCY_INPUTS))}")
    paths = list(DEPENDENCY_INPUTS[image])
    prefix = "deps"
    if include_source:
        paths += SOURCE_INPUTS.get(image, [])
        prefix = "src"
    return f"{prefix}-{_hash_paths(paths)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", choices=sorted(DEPENDENCY_INPUTS))
    parser.add_argument("--source", action="store_true",
                        help="include application sources (a per-commit tag)")
    parser.add_argument("--all", action="store_true", help="print every image's tag")
    args = parser.parse_args()

    if args.all:
        for name in sorted(DEPENDENCY_INPUTS):
            print(f"{name}\t{tag_for(name, False)}\t{tag_for(name, True)}")
        return 0
    if not args.image:
        parser.error("give an image name, or --all")
    print(tag_for(args.image, args.source))
    return 0


if __name__ == "__main__":
    sys.exit(main())
