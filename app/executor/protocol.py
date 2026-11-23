"""Wire format between the API and the code executor.

One POST per run, a tar stream each way. Tar because the inputs are files
(sometimes a partitioned directory) of up to the in-memory analysis cap, and a
tar stream carries them, their paths and the request in one body without
base64 inflation or a multipart parser.

Request members, in order:
    request.json         {"timeout": seconds, "outputs": [absolute paths]}
    script.py            the script to run
    in/<abs path>        each input file or directory, at its absolute path
                         minus the leading slash

Reply members:
    result.json          {"returncode", "stdout", "stderr", "timed_out", "error"}
    out/<abs path>       each declared output the script wrote, as a regular file

Paths are absolute so the script runs unchanged: the executor recreates each
input at the same path it has in the API container, inside its own tmpfs.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from typing import Optional

REQUEST_MEMBER = "request.json"
SCRIPT_MEMBER = "script.py"
RESULT_MEMBER = "result.json"
INPUT_PREFIX = "in/"
OUTPUT_PREFIX = "out/"
RUN_PATH = "/run"
HEALTH_PATH = "/health"


@dataclass
class ScriptResult:
    """What running a script produced. ``returncode`` is None when it never ran
    to an exit status (timed out, or the executor could not be reached)."""

    returncode: Optional[int]
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: Optional[str] = None
    outputs: list = field(default_factory=list)


def member_for(prefix: str, path: str) -> str:
    """Tar member name for absolute ``path`` under ``prefix``."""
    return prefix + normalized_abs(path).lstrip("/")


def path_for(prefix: str, member: str) -> Optional[str]:
    """The absolute path a member names, or None if it is not a clean one."""
    if not member.startswith(prefix):
        return None
    rel = member[len(prefix):]
    if not rel or rel.startswith("/"):
        return None
    path = "/" + rel
    try:
        return path if normalized_abs(path) == path else None
    except ValueError:
        return None


def normalized_abs(path: str) -> str:
    """``path`` made absolute-normal; raises ValueError for a relative path or one with '..'."""
    p = str(path)
    if not p.startswith("/"):
        raise ValueError(f"not an absolute path: {p!r}")
    if any(part == ".." for part in p.split("/")):
        raise ValueError(f"path climbs out with '..': {p!r}")
    return posixpath.normpath(p)


def under(path: str, roots) -> bool:
    """True when ``path`` is strictly inside one of ``roots``."""
    for root in roots:
        r = posixpath.normpath(root).rstrip("/")
        if r and path.startswith(r + "/"):
            return True
    return False
