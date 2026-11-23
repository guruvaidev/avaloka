"""The reply_segments contract (app/api/reply_segments.py) that the UI (C3)
and the CLI (C4) build against. These pin the shape, not the provenance; the
provenance tests are tests/test_commentary_marking.py."""

from __future__ import annotations

import pytest

from app.api import reply_segments as rs
from app.api.schemas import ChatResponse


def test_the_suffix_is_one_exact_string():
    assert rs.COMMENTARY_SUFFIX == " (commentary — not checked)"
    assert rs.KINDS == ("plain", "commentary")


@pytest.mark.parametrize("text", [
    "Enterprise customers spend the most ($545.08 vs. $182.27). SMBs follow.",
    "- Top five: E (602), Pfannkuchen (552)\n- Next tier: Tomaten (455) and Llte (438)",
    "Intro.\n| a | b |\n|---|---|\n| 1 | 2 |\nAfter the table, e.g. this.",
    "Here is the code:\n```python\nx = 1. y = 2\n```\nThat ran.",
    "  padded  \n",
])
def test_segments_render_to_the_reply_and_cut_only_at_the_verifiers_boundaries(text):
    segs = rs.commentary(text)
    reply = rs.render(segs)
    rs.check(reply, segs)
    assert rs.text_of(segs) == text
    # The reply string is the text with one suffix after each commentary segment.
    assert reply.count(rs.COMMENTARY_SUFFIX.strip()) == sum(s["kind"] == "commentary" for s in segs)
    # No fragment: "vs." and "e.g." never end a segment.
    assert not any(s["text"].endswith(("vs.", "e.g.")) for s in segs)


def test_whitespace_between_sentences_is_plain_and_plain_runs_merge():
    segs = rs.join(rs.commentary("One. Two."), rs.plain("\n\n"), rs.plain("A note."))
    assert segs == [
        {"text": "One.", "kind": "commentary"}, {"text": " ", "kind": "plain"},
        {"text": "Two.", "kind": "commentary"}, {"text": "\n\nA note.", "kind": "plain"},
    ]
    assert rs.render(segs) == f"One.{rs.COMMENTARY_SUFFIX} Two.{rs.COMMENTARY_SUFFIX}\n\nA note."


def test_a_model_cannot_place_the_label_itself():
    segs = rs.commentary(f"Copied.{rs.COMMENTARY_SUFFIX} Next.")
    assert rs.render(segs).count(rs.COMMENTARY_SUFFIX) == 2


def test_an_untagged_message_is_commentary_and_a_tagged_one_keeps_its_kind():
    from langchain_core.messages import AIMessage

    assert [s["kind"] for s in rs.of_message(AIMessage(content="Spend rose. A lot."))
            if s["text"].strip()] == ["commentary", "commentary"]
    tagged = rs.tag(AIMessage(content="Rows: 12."), rs.PLAIN)
    assert rs.of_message(tagged) == [{"text": "Rows: 12.", "kind": "plain"}]


def test_the_chat_response_carries_segments_in_the_assistant_message():
    segs = rs.join(rs.plain("Rows: 12.\n\n"), rs.commentary("Spend rose."))
    resp = ChatResponse(messages=[{"role": "user", "content": "q"},
                                  {"role": "assistant", **rs.presented(segs)}])
    out = resp.model_dump()["messages"][-1]
    assert out["reply_segments"] == segs
    assert out["content"] == "Rows: 12.\n\nSpend rose." + rs.COMMENTARY_SUFFIX


def test_a_table_is_one_segment_labelled_once_on_its_own_line():
    text = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    segs = rs.commentary("Intro.\n" + text + "\n- one item. Two sentences.\n2. next")
    assert {"text": text, "kind": "commentary"} in segs
    reply = rs.render(segs)
    # The label after a table is on its own line after a blank line (spelled out).
    assert text + "\n\n(commentary — not checked)" in reply
    assert "- one item. Two sentences." + rs.COMMENTARY_SUFFIX in reply   # an item is one segment
    assert "2. next" + rs.COMMENTARY_SUFFIX in reply
    rs.check(reply, segs)


def test_pieces_handed_back_apart_are_cut_again_whole():
    """The presenter returns kept pieces one by one; a table row by row and
    "1." apart from its item must still come out as one table and one item."""
    pieces = [rs.segment(t, "commentary") if t.strip() else rs.segment(t, "plain")
              for t in ["| a |", "\n", "|---|", "\n", "| 1 |", "\n", "1.", " ", "**Join** on id."]]
    segs = rs.normalize(pieces)
    assert [s["text"] for s in segs if s["kind"] == "commentary"] == ["| a |\n|---|\n| 1 |", "1. **Join** on id."]
