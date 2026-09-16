"""The single source of the Avaloka version.

Read from the repository's VERSION file so the package, the CLI banner and the
release tag cannot drift apart. They already had: pyproject said 0.3.0, VERSION
said 1.0.0, and `avaloka --version` reported a third answer depending on which
one you asked.
"""
from __future__ import annotations

from pathlib import Path

_FALLBACK = "1.0.0"


def _read_version() -> str:
    # Installed wheels have no VERSION file next to the package, so fall back
    # to the packaged metadata and finally to the literal above.
    candidate = Path(__file__).resolve().parent.parent / "VERSION"
    try:
        text = candidate.read_text().strip()
        if text:
            return text
    except OSError:
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("avaloka")
    except (ImportError, PackageNotFoundError):
        return _FALLBACK


__version__ = _read_version()
