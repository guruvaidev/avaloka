"""What a user may be shown when something fails.

Owner rule: the user sees a plain, code-written message. The detail -- a
traceback, a raw exception, the script's stderr -- goes to the server log and
nowhere else. These helpers sit at the edges where error text leaves the
server (API responses, run logs, the CLI) and replace anything that looks like
raw error output with plain words.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

#: What raw error output looks like: a traceback header, a stack frame, an
#: exception repr (``ValueError('...')``) or an exception line
#: (``KeyError: 'col'``).
RAW_ERROR_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r'File "[^"]*", line \d+'),
    re.compile(r"\b[A-Z]\w*(?:Error|Exception)\((?:'|\")"),
    re.compile(r"\b[A-Z]\w*(?:Error|Exception)\s*:\s*\S"),
)

#: The line a run log shows in place of raw error output.
RUN_LOG_ERROR_LINE = "[error] The run stopped with an error. The technical details are in the server log."


def looks_like_raw_error(text: Any) -> bool:
    """True when *text* carries a traceback, a stack frame or a raw exception."""
    if text is None:
        return False
    if not isinstance(text, str):
        try:
            text = json.dumps(text, default=str)
        except (TypeError, ValueError):
            text = str(text)
    return any(rx.search(text) for rx in RAW_ERROR_PATTERNS)


def plain_or(text: Any, fallback: str) -> str:
    """*text* when it is plain words, else *fallback*."""
    if not isinstance(text, str) or not text.strip() or looks_like_raw_error(text):
        return fallback
    return text


def public_agent_errors(errors: Optional[Iterable[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """``agent_errors`` as the API may return them: which agent failed, and the
    exception's class name. The message and traceback stay in the server log
    (contract.run_agent logs them where they happen)."""
    if errors is None:
        return None
    out: List[Dict[str, Any]] = []
    for entry in errors:
        if not isinstance(entry, dict):
            continue
        error_class = entry.get("error_class")
        if not error_class:
            # 1.6 records carry "Type: message" only; keep the type, drop the message.
            m = re.match(r"([A-Za-z_][\w.]*)\s*:", str(entry.get("error") or ""))
            error_class = m.group(1).rsplit(".", 1)[-1] if m else None
        public = {"agent": entry.get("agent"), "error_class": error_class}
        out.append({k: v for k, v in public.items() if isinstance(v, str) and v})
    return out


def scrub_run_logs(lines: Any) -> Any:
    """A run's log with every raw error entry replaced by one plain line.

    Each entry is judged whole: a "[stderr] ..." entry that holds a traceback
    goes entirely, code lines and all, not just the lines that match.
    """
    if isinstance(lines, str):
        return RUN_LOG_ERROR_LINE if looks_like_raw_error(lines) else lines
    if not isinstance(lines, list):
        return lines
    out: List[Any] = []
    for line in lines:
        if looks_like_raw_error(line):
            if not out or out[-1] != RUN_LOG_ERROR_LINE:
                out.append(RUN_LOG_ERROR_LINE)
            continue
        out.append(line)
    return out
