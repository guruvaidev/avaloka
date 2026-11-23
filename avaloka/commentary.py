"""Marking model-written text in the CLI as "commentary — not checked" (2.0, c-strict).

Every sentence a model wrote that the user sees is marked; text the code or a
template wrote stays plain. In the CLI the model's only words are the ``--llm``
rephrasings of Avaloka's lines (:meth:`avaloka.persona.Avaloka._voice`): a line
the model reworded is commentary, the deterministic line it replaced is not.
When the guard falls back to the original line, nothing the model wrote is
shown, so nothing is marked.

The suffixes are the ones the API appends (``COMMENTARY_SUFFIX``,
``TABLE_SUFFIX`` and ``suffix_for`` in ``app/api/reply_segments.py``); this
package ships without ``app/``, so they are repeated here and a test pins
them together. No check marks:
2.0 marks what is not checked and claims nothing about the rest.
"""

from __future__ import annotations

import re

COMMENTARY_SUFFIX = " (commentary — not checked)"
#: After a table a model wrote: the label on its own line, so the table still parses.
TABLE_SUFFIX = "\n\n" + COMMENTARY_SUFFIX.strip()


def is_table(text: str) -> bool:
    """Commentary *text* is a markdown table: the only commentary that spans lines."""
    return "\n" in text


def suffix_for(text: str) -> str:
    """The suffix after commentary *text*: TABLE_SUFFIX for a table, COMMENTARY_SUFFIX otherwise."""
    return TABLE_SUFFIX if is_table(text) else COMMENTARY_SUFFIX


def strip_suffixes(text: str) -> str:
    """*text* without either suffix or the bare label: a model that read an
    earlier reply may copy the label, and the label is ours to place."""
    return (text.replace(TABLE_SUFFIX, "").replace(COMMENTARY_SUFFIX, "")
            .replace(COMMENTARY_SUFFIX.strip(), ""))

# Sentence ends: terminal punctuation (optionally closed by markup, a quote or
# a bracket) followed by whitespace. Decimals ("0.7134") have no space after
# the point, so they never split.
_SENTENCE_END = re.compile(r"(?<=[.!?…])((?:\[/[a-z ]+\]|[\"'”’)])*)\s+")


def split_sentences(text: str) -> list[str]:
    """``text`` as sentences, closing markup kept with the sentence it closes."""
    out: list[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        out.append(text[start:m.end(1)])
        start = m.end()
    tail = text[start:]
    if tail.strip():
        out.append(tail)
    return [s.strip() for s in out if s.strip()]


def mark_commentary(text: str) -> str:
    """``text`` as rich markup: dim, with its suffix after every sentence (a
    table, kept whole, gets the table suffix once)."""
    text = strip_suffixes(text).strip()
    if not text:
        return text
    if all(line.strip().startswith("|") for line in text.splitlines() if line.strip()) and is_table(text):
        return f"[dim]{text}{suffix_for(text)}[/dim]"
    return " ".join(f"[dim]{s}{suffix_for(s)}[/dim]" for s in split_sentences(text))
