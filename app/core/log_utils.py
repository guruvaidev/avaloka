"""Compact, useful log lines for payloads that are too big to log whole.

Two problems this exists for.

**Volume.** Several agents logged whole message lists at INFO. A single coder
turn carried a 100-row markdown preview of the result table, so one line of log
ran to tens of thousands of characters and buried everything around it. The
size of a payload is nearly always the interesting part; its contents almost
never are. :func:`describe_messages` reports the shape.

**Provenance.** Once an agent can fail over to a second provider, "which model
actually answered this?" stops being a constant you can read off the config and
becomes a per-response fact. :func:`describe_response` reads it back out of the
response the provider returned, so the log reflects what served the request
rather than what was configured to.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

#: How much of a long string is worth keeping in a log line.
DEFAULT_PREVIEW_CHARS = 160


def _ascii(text: str) -> str:
    """Make a fragment safe to write to a cp1252 console.

    Log records are formatted by the handler, and on Windows that handler
    encodes to cp1252. A character it cannot represent raises
    UnicodeEncodeError *inside logging*, so a log line becomes a crash. Payloads
    here routinely carry em-dashes and multiplication signs ("190 rows x 6
    columns" is written with real Unicode in the agents' own output), which is
    exactly the case that bites.

    ``backslashreplace`` rather than ``replace`` so the original character is
    still recoverable from the log instead of collapsing to "?".
    """
    return text.encode("ascii", "backslashreplace").decode("ascii")


def preview(value: Any, limit: int = DEFAULT_PREVIEW_CHARS) -> str:
    """A short, length-annotated stand-in for a possibly huge value.

    Keeps the head (which usually identifies the payload) and always states the
    true size, so a truncated line can never be mistaken for a small one.
    """
    if value is None:
        return "None"
    text = value if isinstance(value, str) else str(value)
    n = len(text)
    if n <= limit:
        return _ascii(text)
    head = _ascii(text[:limit].replace("\n", "\\n"))
    # ASCII only, deliberately. These lines land on a Windows console using
    # cp1252, where a non-ASCII character raises UnicodeEncodeError inside the
    # logging handler -- turning a log line into a crash.
    return f"{head}... [{n:,} chars total]"


def describe_messages(messages: Optional[Iterable[Any]]) -> str:
    """Shape of a message list: how many, how big, of what kind.

    Deliberately reports no content. When a message list contains a rendered
    data table, logging the content is what made these lines unreadable in the
    first place.
    """
    if not messages:
        return "0 messages"
    items = list(messages)
    parts = []
    total = 0
    for m in items:
        content = getattr(m, "content", m)
        length = len(content) if isinstance(content, str) else len(str(content))
        total += length
        parts.append(f"{type(m).__name__}:{length:,}c")
    return f"{len(items)} messages, {total:,} chars [{', '.join(parts)}]"


def response_model(response: Any) -> Optional[str]:
    """The model id that produced *response*, as reported by the provider.

    Providers disagree on the key -- Groq sends ``model_name``, OpenAI-spec
    servers (OpenRouter, vLLM, Ollama) usually send ``model`` -- so check both
    rather than assuming the primary provider's shape. Returns None when the
    response carries no metadata, which is the case for a stubbed or
    deterministic reply and is worth being able to tell apart.
    """
    meta = getattr(response, "response_metadata", None) or {}
    if not isinstance(meta, dict):
        return None
    for key in ("model_name", "model", "model_id"):
        value = meta.get(key)
        if value:
            return str(value)
    return None


def describe_response(response: Any) -> str:
    """One line: which model answered, at what reasoning effort, for what cost.

    Replaces logging the whole response object. With a failover chain in play
    the model id is the load-bearing part: it is how you tell from a log that a
    turn was served by the backup rather than the configured primary.
    """
    if response is None:
        return "no response"
    model = response_model(response) or "unknown-model"
    meta = getattr(response, "response_metadata", None) or {}
    bits = [f"model={model}"]

    if isinstance(meta, dict):
        effort = meta.get("reasoning_effort")
        if effort:
            bits.append(f"effort={effort}")
        finish = meta.get("finish_reason")
        if finish:
            bits.append(f"finish={finish}")

    usage = getattr(response, "usage_metadata", None) or {}
    if isinstance(usage, dict) and usage:
        bits.append(f"tokens={usage.get('input_tokens', '?')}->{usage.get('output_tokens', '?')}")

    calls = getattr(response, "tool_calls", None)
    if calls:
        bits.append("tools=" + ",".join(str(c.get("name")) for c in calls if isinstance(c, dict)))

    content = getattr(response, "content", None)
    if isinstance(content, str) and content:
        bits.append(f"content={len(content):,}c")

    return " ".join(bits)
