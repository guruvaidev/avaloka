"""Which sentences of a reply Avaloka checked, and which she only wrote.

2.0 shows the user, sentence by sentence, who wrote what (the owner's
option "c-strict", 2026-10-26). Text that code or a template wrote from the
data -- a table, the scope line, a verifier note, the drivers answer, a
transfer confirmation -- is *plain*. Every sentence a language model wrote --
the result narrator, a conversational reply, an explanation, a suggestion --
is *commentary*: Avaloka wrote it and nothing checked it against the data.
A sentence that puts code-inserted figures into LLM text is commentary. There
is no "verified" mark: nothing here says a sentence is true.

The contract (C3, the UI, and C4, the CLI, build against it)
--------------------------------------------------------------

Wherever a reply is returned -- the assistant entry of ``messages`` in a chat
response, an entry of ``GET /threads/{id}/messages``, an MCP tool reply -- it
carries, next to ``content``::

    "reply_segments": [{"text": str, "kind": "plain" | "commentary"}, ...]

* ``content``, the reply *string*, carries a suffix after every commentary
  segment, so a consumer that reads only text still sees which sentences
  are commentary. Exactly::

      content == "".join(s["text"] + (suffix_for(s["text"]) if s["kind"] == "commentary" else "")
                         for s in reply_segments)

  where ``suffix_for(text)`` is :data:`TABLE_SUFFIX` when ``text`` is a
  markdown table -- the only commentary segment whose text contains a
  newline -- and :data:`COMMENTARY_SUFFIX` otherwise. That is, the segments' texts concatenate to ``content`` with
  one suffix removed after each commentary segment (:func:`render`;
  :func:`check` asserts it). A client that renders the segments renders
  ``text`` as it is and never shows the suffix: it marks a commentary
  segment its own way (muted, italic, a left rule, a tooltip).
* A commentary segment is exactly one sentence, one line, or one list item
  of the reply, with no surrounding whitespace. Sentences are cut where the
  claim verifier cuts them (#587: never at "vs.", "e.g." and the like). A
  list item (a line starting ``-``, ``*``, ``+``, ``•`` or ``1.``/``1)``) is
  one segment with its marker, its suffix at the item's end. The whitespace
  and newlines between them are plain segments. Nothing is ever cut inside a
  sentence or an item, so no segment is a fragment.
* Two exceptions keep markdown intact. A markdown table a model wrote, from
  its header row through the separator and every row, is ONE commentary
  segment; its suffix, :data:`TABLE_SUFFIX`, puts the label on its own line
  after a blank line, so the table still parses. A fenced code block is one
  plain segment (code is not a statement about the data, and a suffix after
  the closing fence would leave the fence open).
* Adjacent plain segments are merged; no segment is empty.
* A reply without ``reply_segments`` (history written before 2.0, a reply no
  path tagged) is rendered from ``content`` as it is.
* ``assistant_prompts`` entries carry ``message_segments`` for ``message`` in
  the same shape and under the same rules.

Provenance is recorded where the text is created, never guessed afterwards:
whoever writes the text says what wrote it (:func:`tag`, :func:`plain`,
:func:`commentary`), and the claim verifier's presenter keeps each kept
sentence's kind and adds its own notes as plain. Text no path tagged is
commentary: a code line marked commentary costs the user a muted line, an
LLM sentence shown plain would claim a check that never happened.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Literal, Mapping, Optional, Sequence, TypedDict

#: Appended to every commentary sentence of the reply string. C3 and C4 mirror
#: this exact string (U+2014 em dash, one leading space) to strip it.
COMMENTARY_SUFFIX = " (commentary — not checked)"
#: The suffix after a markdown table a model wrote: the same label on its own
#: line after a blank line, so the table above it still parses as a table.
TABLE_SUFFIX = "\n\n" + COMMENTARY_SUFFIX.strip()

PLAIN = "plain"
COMMENTARY = "commentary"
KINDS = (PLAIN, COMMENTARY)

#: Where an AIMessage carries its segments (``additional_kwargs``), from the
#: node that wrote it to the server, the thread history and back.
MESSAGE_KEY = "reply_segments"


class ReplySegment(TypedDict):
    text: str
    kind: Literal["plain", "commentary"]


def segment(text: str, kind: str) -> ReplySegment:
    if kind not in KINDS:
        raise ValueError(f"unknown reply segment kind {kind!r}; one of {KINDS}")
    return {"text": text, "kind": kind}  # type: ignore[typeddict-item]


def plain(text: Any) -> List[ReplySegment]:
    """*text* as code or a template wrote it."""
    return normalize([segment(str(text), PLAIN)]) if text else []


def commentary(text: Any) -> List[ReplySegment]:
    """*text* as a language model wrote it, one segment per sentence."""
    return normalize([segment(str(text), COMMENTARY)]) if text else []


def tag(message: Any, kind_or_segments: Any) -> Any:
    """*message* (an AIMessage) with its provenance recorded on it.

    *kind_or_segments* is :data:`PLAIN`, :data:`COMMENTARY` or a segment list
    whose texts concatenate to the message's content. Returns the message.
    """
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return message
    if isinstance(kind_or_segments, str):
        segs = normalize([segment(content, kind_or_segments)]) if content else []
    else:
        segs = normalize(kind_or_segments)
        if text_of(segs) != content:
            raise ValueError("reply segments do not concatenate to the message content")
    message.additional_kwargs = {**(getattr(message, "additional_kwargs", None) or {}), MESSAGE_KEY: segs}
    return message


def of_message(message: Any) -> List[ReplySegment]:
    """The segments a node recorded on *message*, unsuffixed. A message no
    node tagged is commentary, whole."""
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return []
    recorded = (getattr(message, "additional_kwargs", None) or {}).get(MESSAGE_KEY)
    if isinstance(recorded, list) and _well_formed(recorded):
        if text_of(recorded) == content:
            return normalize(recorded)
        if render(recorded) == content:      # already presented (thread history)
            return normalize(recorded)
    return commentary(content)


def text_of(segments: Iterable[Mapping[str, Any]]) -> str:
    """The segments' texts, concatenated: the reply without suffixes."""
    return "".join(str(s["text"]) for s in segments)


def is_table(text: str) -> bool:
    """*text*, a commentary segment, is a markdown table: the only
    commentary segment that spans lines (a sentence or an item never does)."""
    return "\n" in text


def suffix_for(text: str) -> str:
    """The suffix the reply string carries after commentary *text*."""
    return TABLE_SUFFIX if is_table(text) else COMMENTARY_SUFFIX


def render(segments: Iterable[Mapping[str, Any]]) -> str:
    """The reply string: each commentary segment followed by its suffix
    (:func:`suffix_for`)."""
    return "".join(str(s["text"]) + (suffix_for(str(s["text"])) if s["kind"] == COMMENTARY else "")
                   for s in segments)


def check(content: str, segments: Sequence[Mapping[str, Any]]) -> None:
    """Raise unless *segments* are a well-formed segmentation of *content*."""
    if not _well_formed(segments):
        raise AssertionError("malformed reply segments")
    if render(segments) != content:
        raise AssertionError("reply segments do not render to the reply")
    for s in segments:
        if s["kind"] == COMMENTARY and s["text"] != s["text"].strip():
            raise AssertionError(f"commentary segment carries whitespace: {s['text']!r}")
        if s["kind"] == COMMENTARY and _LIST_ITEM.match(s["text"]) is None and \
                re.fullmatch(r"\d{1,3}[.)]|[-*+\u2022]", s["text"]):
            raise AssertionError(f"a list marker apart from its item: {s['text']!r}")


def _well_formed(segments: Any) -> bool:
    return all(isinstance(s, Mapping) and isinstance(s.get("text"), str) and s.get("kind") in KINDS
               for s in segments)


_FENCE = re.compile(r"(```.*?```)", re.DOTALL)
_TABLE_ROW = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
_TABLE_SEPARATOR = re.compile(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$")
_LIST_ITEM = re.compile(r"^[ \t]*(?:[-*+\u2022]|\d{1,3}[.)])[ \t]+\S")
_WORDY = re.compile(r"\w")


def _boundary() -> "re.Pattern[str]":
    # The claim verifier's own sentence boundary (#587), so a commentary
    # segment is exactly a piece the verifier checks and withholds by.
    from app.agents.claim_verifier import _BOUNDARY
    return _BOUNDARY


def _sentences(text: str) -> List[ReplySegment]:
    """Commentary *text* cut into sentences, whitespace between them plain."""
    out: List[ReplySegment] = []
    for block in _FENCE.split(text):
        if not block:
            continue
        if block.startswith("```") and block.endswith("```") and len(block) >= 6:
            out.append(segment(block, PLAIN))
            continue
        out.extend(_table_aware(block))
    return out


def _table_aware(block: str) -> List[ReplySegment]:
    out: List[ReplySegment] = []
    lines = block.split("\n")
    i = 0
    prose: List[str] = []

    def flush_prose(with_newline: bool) -> None:
        if prose:
            out.extend(_prose("\n".join(prose) + ("\n" if with_newline else "")))
            prose.clear()

    while i < len(lines):
        if _LIST_ITEM.match(lines[i]):
            flush_prose(True)
            out.extend(_whole(lines[i]))
            if i + 1 < len(lines):
                out.append(segment("\n", PLAIN))
            i += 1
            continue
        if _TABLE_ROW.match(lines[i]) and i + 1 < len(lines) and _TABLE_SEPARATOR.match(lines[i + 1]):
            j = i
            while j < len(lines) and _TABLE_ROW.match(lines[j]):
                j += 1
            if j - i >= 2:
                flush_prose(True)
                table = "\n".join(lines[i:j])
                lead = len(table) - len(table.lstrip())
                if lead:
                    out.append(segment(table[:lead], PLAIN))
                out.append(segment(table[lead:].rstrip(), COMMENTARY))
                trail = table[len(table.rstrip()):]
                if trail:
                    out.append(segment(trail, PLAIN))
                if j < len(lines):
                    out.append(segment("\n", PLAIN))
                i = j
                continue
        prose.append(lines[i])
        i += 1
    flush_prose(False)
    return out


def _whole(line: str) -> List[ReplySegment]:
    """One line as one commentary segment, its edge whitespace plain."""
    core = line.strip()
    if not core:
        return [segment(line, PLAIN)] if line else []
    start = line.index(core)
    out = [segment(line[:start], PLAIN)] if start else []
    out.append(segment(core, COMMENTARY))
    if start + len(core) < len(line):
        out.append(segment(line[start + len(core):], PLAIN))
    return out


def _prose(text: str) -> List[ReplySegment]:
    out: List[ReplySegment] = []
    for i, piece in enumerate(_boundary().split(text)):
        if not piece:
            continue
        if i % 2:
            out.append(segment(piece, PLAIN))
            continue
        core = piece.strip()
        if not _WORDY.search(core):
            out.append(segment(piece, PLAIN))  # a rule, a bullet marker alone
            continue
        start = piece.index(core)
        if start:
            out.append(segment(piece[:start], PLAIN))
        out.append(segment(core, COMMENTARY))
        if start + len(core) < len(piece):
            out.append(segment(piece[start + len(core):], PLAIN))
    return out


def _strip_suffixes(text: str) -> str:
    # A model that read an earlier reply may copy the label; the label is
    # ours to place, never the model's.
    return (text.replace(TABLE_SUFFIX, "").replace(COMMENTARY_SUFFIX, "")
            .replace(COMMENTARY_SUFFIX.strip(), ""))


def normalize(segments: Iterable[Mapping[str, Any]]) -> List[ReplySegment]:
    """Commentary cut into sentences, adjacent plain text merged, empty
    segments dropped. Idempotent."""
    # Commentary pieces with only whitespace between them are cut again as
    # one text: the presenter hands back a table row by row and "1." apart
    # from its item, and only the whole text shows the table and the item.
    runs: List[ReplySegment] = []
    for s in segments:
        text, kind = str(s.get("text") or ""), s.get("kind")
        if not text:
            continue
        kind = COMMENTARY if kind == COMMENTARY else PLAIN
        if runs and kind == COMMENTARY and runs[-1]["kind"] == COMMENTARY:
            runs[-1] = segment(runs[-1]["text"] + text, COMMENTARY)
        elif (kind == PLAIN and not text.strip() and runs and runs[-1]["kind"] == COMMENTARY):
            runs[-1] = segment(runs[-1]["text"] + text, COMMENTARY)   # held: may sit between two pieces
        else:
            runs.append(segment(text, kind))
    cut: List[ReplySegment] = []
    for s in runs:
        if s["kind"] == COMMENTARY:
            cut.extend(_sentences(_strip_suffixes(s["text"])))
        else:
            cut.append(s)
    out: List[ReplySegment] = []
    for s in cut:
        if not s["text"]:
            continue
        if out and s["kind"] == PLAIN and out[-1]["kind"] == PLAIN:
            out[-1] = segment(out[-1]["text"] + s["text"], PLAIN)
        else:
            out.append(s)
    return out


def join(*parts: Optional[Sequence[Mapping[str, Any]]]) -> List[ReplySegment]:
    """Segment lists one after another."""
    return normalize([s for p in parts if p for s in p])


def strip(segments: Sequence[Mapping[str, Any]]) -> List[ReplySegment]:
    """*segments* without leading and trailing whitespace, as ``str.strip``
    would leave their text."""
    segs = [dict(s) for s in normalize(segments)]
    while segs and not segs[0]["text"].strip():
        segs.pop(0)
    while segs and not segs[-1]["text"].strip():
        segs.pop()
    if segs:
        segs[0]["text"] = segs[0]["text"].lstrip()
        segs[-1]["text"] = segs[-1]["text"].rstrip()
    return normalize(segs)  # type: ignore[arg-type]


def rstrip(segments: Sequence[Mapping[str, Any]]) -> List[ReplySegment]:
    """*segments* without trailing whitespace, as ``str.rstrip`` would leave
    their text."""
    segs = [dict(s) for s in normalize(segments)]
    while segs and not segs[-1]["text"].strip():
        segs.pop()
    if segs:
        segs[-1]["text"] = segs[-1]["text"].rstrip()
    return normalize(segs)  # type: ignore[arg-type]


def message(segments: Sequence[Mapping[str, Any]]) -> Any:
    """An AIMessage of *segments*' text that records them: what a node
    returns when it wrote some of a reply and a model wrote the rest."""
    from langchain_core.messages import AIMessage

    segs = normalize(segments)
    return AIMessage(content=text_of(segs), additional_kwargs={MESSAGE_KEY: segs})


def plain_message(*args: Any, **kwargs: Any) -> Any:
    """``AIMessage(...)`` for a reply code or a template wrote."""
    from langchain_core.messages import AIMessage

    return tag(AIMessage(*args, **kwargs), PLAIN)


def commentary_message(*args: Any, **kwargs: Any) -> Any:
    """``AIMessage(...)`` for a reply a language model wrote."""
    from langchain_core.messages import AIMessage

    return tag(AIMessage(*args, **kwargs), COMMENTARY)


def as_message(segments: Sequence[Mapping[str, Any]]) -> Any:
    """An AIMessage whose content is the presented reply (:func:`render`) and
    which carries its segments, for the thread history."""
    from langchain_core.messages import AIMessage

    segs = normalize(segments)
    return AIMessage(content=render(segs), additional_kwargs={MESSAGE_KEY: segs})


def unmarked(message: Any) -> str:
    """*message*'s reply without the commentary marks."""
    content = getattr(message, "content", None)
    return text_of(of_message(message)) if isinstance(content, str) else content


def presented(segments: Sequence[Mapping[str, Any]]) -> dict:
    """``{"content": the reply string, "reply_segments": the segments}``."""
    segs = normalize(segments)
    return {"content": render(segs), MESSAGE_KEY: segs}


__all__ = [
    "COMMENTARY", "COMMENTARY_SUFFIX", "KINDS", "MESSAGE_KEY", "PLAIN", "ReplySegment", "as_message",
    "TABLE_SUFFIX", "check", "commentary", "commentary_message", "is_table", "join", "message", "normalize", "of_message", "plain", "plain_message", "presented", "render", "rstrip",
    "segment", "strip", "suffix_for", "tag", "text_of", "unmarked",
]
