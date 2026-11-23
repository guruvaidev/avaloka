"""2.0 (c-strict) in the CLI: what a model wrote is marked, what the code wrote is not.

The CLI's only model-written text is the ``--llm`` rewording of Avaloka's
lines. A reworded line is printed ``[dim]`` with "(commentary — not checked)"
after each sentence; the deterministic line, and a rewording the guard threw
away, stay plain. The model replies below are recorded: the two gemma4:e4b
outputs quoted in avaloka/llm/guard.py, and the persona prompt's own example.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest
from rich.console import Console

import avaloka.chat as chat
from avaloka.commentary import (COMMENTARY_SUFFIX, TABLE_SUFFIX, mark_commentary, split_sentences,
                                suffix_for)
from avaloka.persona import Avaloka
from avaloka.swarm import SwarmNarrator

CHURN = Path(__file__).resolve().parents[2] / "avaloka" / "examples" / "churn.csv"
CHECKS = re.compile("[✓✔✅☑\U0001F5F8]")

# Recorded gemma4:e4b replies (avaloka/llm/guard.py): pure meta-narration.
GEMMA_META = ("I'm going to rephrase this to cleanly communicate that no single variable "
              "shows dominance. Essentially, the distribution is quite even.")


class _Recorded:
    """An LLM client that answers with a fixed reply, or a rule, and costs nothing."""

    available = True
    last_cost_usd = 0.0
    calls = 0

    def __init__(self, reply):
        self._reply = reply

    def complete(self, system, prompt):
        self.calls += 1
        line = prompt.split("Line: ", 1)[1].rsplit("\nReply:", 1)[0]
        return self._reply(line) if callable(self._reply) else self._reply


def _voiced(reply) -> Avaloka:
    ava = Avaloka(llm=False)
    ava._client = _Recorded(reply)
    return ava


def test_a_reworded_line_is_dim_with_the_api_suffix_after_every_sentence():
    # The persona prompt's example reply, for the line it was written for.
    ava = _voiced("Ooh, 1,200 rows across 9 columns. Give me a second to really take it in.")
    out = ava.first_glance({"n_rows": 1200, "n_cols": 9})
    assert "[dim]Ooh, 1,200 rows across 9 columns. (commentary — not checked)[/dim]" in out
    assert "[dim]Give me a second to really take it in. (commentary — not checked)[/dim]" in out
    # The very suffix the API appends (this package ships without app/, so it repeats it).
    api = pytest.importorskip("app.api.reply_segments")
    assert COMMENTARY_SUFFIX == api.COMMENTARY_SUFFIX and out.count(api.COMMENTARY_SUFFIX) == 2
    # Without --llm the same line is the deterministic one, plain.
    plain = Avaloka(llm=False).first_glance({"n_rows": 1200, "n_cols": 9})
    assert COMMENTARY_SUFFIX not in plain and "[dim]" not in plain


def test_what_survives_the_guard_is_marked_and_what_it_rejects_is_not():
    # gemma4:e4b's meta-narration: the guard drops the preamble and keeps
    # "The distribution is quite even." -- the model's words, so commentary.
    out = _voiced(GEMMA_META).greet()
    assert "[dim]The distribution is quite even. (commentary — not checked)[/dim]" in out
    assert out.count(COMMENTARY_SUFFIX) == 1
    # A rewording that adds a number is thrown away (9 columns is not 7): the
    # deterministic line is printed, plain.
    out = _voiced("Ooh, 1,200 rows across 9 columns.").first_glance({"n_rows": 1200, "n_cols": 7})
    assert COMMENTARY_SUFFIX not in out and "[dim]" not in out and "1,200 rows across 7 columns" in out
    # A model that echoes the line wrote nothing new.
    out = _voiced(lambda line: line).reflect_swarm(4)
    assert COMMENTARY_SUFFIX not in out and "[dim]" not in out


def test_the_swarm_narrator_marks_reworded_lines():
    lines: list[str] = []
    SwarmNarrator(lines.append, voice=_voiced(lambda line: "Right — " + line)).announce(4)
    assert lines and lines[0].count(COMMENTARY_SUFFIX) == 1 and "[dim]Right" in lines[0]
    plain: list[str] = []
    SwarmNarrator(plain.append).announce(4)
    assert COMMENTARY_SUFFIX not in plain[0]


def test_decimals_and_closing_markup_do_not_split_or_break_sentences():
    text = "This is a [bold]regression[/bold] problem. It scored 0.7134 on holdout! Done?"
    assert split_sentences(text) == ["This is a [bold]regression[/bold] problem.",
                                     "It scored 0.7134 on holdout!", "Done?"]
    marked = mark_commentary(text)
    assert marked.count(COMMENTARY_SUFFIX) == 3
    console = Console(file=io.StringIO(), width=300, color_system=None)
    console.print(marked)  # valid rich markup
    assert console.file.getvalue().count("(commentary — not checked)") == 3


def _chat_transcript(monkeypatch, voice: Avaloka | None) -> str:
    """``avaloka chat --data churn.csv`` then ``quit``, as the user sees it (markup kept)."""
    printed: list[str] = []
    console = Console(file=io.StringIO(), width=200, color_system=None)
    monkeypatch.setattr(console, "print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    monkeypatch.setattr(console, "input", lambda *a, **k: "quit")
    if voice is not None:
        monkeypatch.setattr(chat, "Avaloka", lambda **kw: voice)
    chat.run_chat(console, data=str(CHURN))
    return "\n".join(printed)


def test_chat_on_churn_marks_exactly_the_reworded_lines(monkeypatch):
    voiced = _chat_transcript(monkeypatch, _voiced(lambda line: "Right — " + line)).splitlines()
    plain = _chat_transcript(monkeypatch, Avaloka(llm=False)).splitlines()
    assert len(voiced) == len(plain)
    reworded = [v for v, p in zip(voiced, plain) if v != p]
    assert len(reworded) >= 3  # greeting, first glance, workload read, proposal
    for line in reworded:
        assert "[dim]Right" in line and COMMENTARY_SUFFIX in line, line
    # Everything the code wrote (schema lines, the banner, unvoiced remarks
    # like the domain read) is printed the same in both runs, and plain.
    same = [v for v, p in zip(voiced, plain) if v == p]
    assert same and not any(COMMENTARY_SUFFIX in line for line in same)
    assert not CHECKS.search("\n".join(voiced))
    # The deterministic run (no --llm) is unmarked throughout.
    assert "1,200 rows" in "\n".join(plain)
    assert not any("commentary" in line for line in plain)


def test_both_suffixes_mirror_the_api_and_a_copied_label_is_not_marked_twice():
    assert TABLE_SUFFIX == "\n\n(commentary — not checked)"
    api = pytest.importorskip("app.api.reply_segments")
    if hasattr(api, "suffix_for"):  # C1 6fffcc8c and later
        assert TABLE_SUFFIX == api.TABLE_SUFFIX
        for text in ("Spend rose.", "| a | b |\n|---|---|\n| 1 | 2 |"):
            assert suffix_for(text) == api.suffix_for(text)
    # A rewording that copied the label (either suffix) is marked once, by us.
    copied = f"Ooh, 1,200 rows across 9 columns.{COMMENTARY_SUFFIX}{TABLE_SUFFIX}"
    out = _voiced(copied).first_glance({"n_rows": 1200, "n_cols": 9})
    assert out.count("(commentary — not checked)") == 1
    # A table stays whole and is labelled once, on its own line.
    table = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert mark_commentary(table) == f"[dim]{table}{TABLE_SUFFIX}[/dim]"
