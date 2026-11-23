"""Every sentence a model wrote reaches the user marked "commentary — not
checked"; what code wrote reaches them plain (2.0, option c-strict).

Each test drives the text from the node that writes it -- the result
narrator, the claim verifier, the drivers route, the planner's transfer
replies -- through the server's own assembly (POST /threads/{id}/messages)
to the API response, and checks what the user is shown: the reply string,
where every commentary sentence carries the suffix, and ``reply_segments``,
which says who wrote each piece. The live-shape turns are rweb15's own
(tests/fixtures/filter_disclosure_live/rc15_turns.json, #588).

Nothing here imports the reply_segments module: the suffix is spelled out
and every check reads the HTTP response, so on a build without the marking
these fail on what the user sees, not on a missing name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.api.server as server
from app.agents import result_narrator
from app.agents.claim_verifier import claim_verifier_node

try:
    from tests.server_harness import (  # noqa: F401  (fixtures are used by name)
        FakeGraph, SESSIONS, THREAD_TO_SESSION, client, make_auth_headers, _do_upload, _quiet_send)
    from tests.test_filter_disclosure_live import FIXTURES, TURNS, _live_state, _related_data
except ImportError:  # tests/ on sys.path
    from server_harness import (  # noqa: F401
        FakeGraph, SESSIONS, THREAD_TO_SESSION, client, make_auth_headers, _do_upload, _quiet_send)
    from test_filter_disclosure_live import FIXTURES, TURNS, _live_state, _related_data

#: The contract's suffixes (app/api/reply_segments.py), spelled out: after a
#: sentence or an item, and after a table (its own line, after a blank line).
SUFFIX = " (commentary — not checked)"
TABLE_SUFFIX = "\n\n(commentary — not checked)"
GOLDEN = Path(__file__).parent / "fixtures" / "commentary_marking" / "false_sentences_rc14_rc15.json"


# --------------------------------------------------------------------------- #
# The path: node -> verifier -> server -> response
# --------------------------------------------------------------------------- #

class _FinalGraph(FakeGraph):
    """The graph, standing in only for its routing: it returns the turn's
    final state, whose messages the real nodes wrote."""

    def __init__(self, final: Dict[str, Any]):
        self.final = final

    def invoke(self, state_in, config=None):
        out = dict(super().invoke(state_in, config))
        out.update({k: v for k, v in self.final.items() if k != "messages"})
        out["ready_to_code"] = False
        out["coder_definition"] = self.final.get("coder_definition") or {}
        out["messages"] = state_in["messages"] + list(self.final["messages"])
        return out


def _narrated(state: Dict[str, Any], narrative: str, monkeypatch) -> AIMessage:
    """What result_narrator_node writes when the model returns *narrative*."""
    monkeypatch.setattr(result_narrator, "narrator_llm", object(), raising=False)
    monkeypatch.setattr(result_narrator, "_narrate", lambda question, rows: narrative)
    out = result_narrator.result_narrator_node(state)
    message = out["messages"][-1]
    assert isinstance(message, AIMessage) and message.content == narrative
    return message


def _ask(client, monkeypatch, final: Dict[str, Any], user: str = "c1-user") -> Dict[str, Any]:
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _FinalGraph(final), raising=False)
    upload = _do_upload(client, user_id=user)
    resp = client.post(f"/threads/{upload['thread_id']}/messages",
                       json={"content": "the question", "metadata": {"dataset_id": upload["dataset_id"]}},
                       headers=make_auth_headers(user))
    assert resp.status_code == 200, resp.text
    entry = [m for m in resp.json()["messages"] if m["role"] == "assistant"][-1]
    entry["_thread_id"] = upload["thread_id"]
    return entry


def _segments(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """The entry's segments, checked against the contract."""
    segs = entry.get("reply_segments")
    assert isinstance(segs, list) and segs, f"the reply carries no reply_segments: {entry['content'][:200]!r}"
    assert all(s["kind"] in ("plain", "commentary") and s["text"] for s in segs), segs
    rendered = "".join(s["text"] + ((TABLE_SUFFIX if "\n" in s["text"] else SUFFIX)
                                    if s["kind"] == "commentary" else "") for s in segs)
    assert rendered == entry["content"], "reply_segments do not render to the reply string"
    for s in segs:
        if s["kind"] == "commentary":
            assert s["text"] == s["text"].strip(), s
    return segs


def _kind_of(segs: List[Dict[str, str]], fragment: str) -> str:
    hits = {s["kind"] for s in segs if fragment in s["text"]}
    assert hits, f"{fragment!r} is in no segment"
    assert len(hits) == 1, f"{fragment!r} is in segments of both kinds"
    return hits.pop()


# --------------------------------------------------------------------------- #
# Live shape: rweb15's own turns
# --------------------------------------------------------------------------- #

def _sentences(narrative: str) -> List[str]:
    """The narrative's sentences, cut where the verifier cuts (#587)."""
    from app.agents.claim_verifier import _SENTENCE_END

    return [p.strip() for p in re.split(_SENTENCE_END + r"|\n+", narrative) if p.strip()]


@pytest.mark.parametrize("run", ["rw7", "rw7c"])
def test_related_turn_the_narration_is_commentary_and_the_scope_line_is_plain(client, monkeypatch, tmp_path, run):
    """rc15's "average order amount by customer segment": the narrator's
    "$545.08 / $182.27 / $59.77" is the model's (commentary, marked); the
    scope line the verifier adds (#586/#588) is code (plain, unmarked)."""
    turn = TURNS[run]
    state = _live_state(turn, _related_data(tmp_path))
    state["messages"] = state["messages"][:1]            # the narrator writes the reply
    state["execution_result"]["status"] = "success"
    message = _narrated(state, turn["narrative"], monkeypatch)
    state["messages"] = state["messages"] + [message]
    report = claim_verifier_node(state)["verification_report"]
    entry = _ask(client, monkeypatch, {**state, "messages": [message], "verification_report": report})
    content = entry["content"]
    # What a text-only client sees: each model sentence marked, the scope line not.
    shown = [s for s in _sentences(turn["narrative"]) if s in content]
    assert shown and all(s + SUFFIX in content for s in shown), content
    segs = _segments(entry)

    scope = [line for line in content.splitlines() if line.startswith("Scope:")]
    assert len(scope) == 1, content
    assert _kind_of(segs, scope[0]) == "plain"
    assert SUFFIX not in scope[0]
    assert "$545.08" in content and _kind_of(segs, "$545.08") == "commentary"
    for sentence in _sentences(turn["narrative"]):
        if sentence in content:
            assert _kind_of(segs, sentence) == "commentary", sentence
            assert content.count(sentence + SUFFIX) >= 1, sentence


def test_a_withheld_sentence_note_is_plain_and_the_kept_narration_commentary(client, monkeypatch, tmp_path):
    """rc15 rw7c nullxlsx: the verifier holds one statement back. Its note
    ("I've held back ...") and its "Why:" are code; every sentence it kept
    is the model's."""
    turn = TURNS["rw7c_nullxlsx"]
    code, reads = turn["code"], []
    for sheet, live in re.findall(r"path_(\d{4}) = '([^']+)'", code):
        path = tmp_path / f"{sheet}.csv"
        pd.read_excel(FIXTURES / "null_words" / "three_sheets.xlsx", sheet_name=sheet, dtype=str,
                      keep_default_na=False).to_csv(path, index=False)
        code = code.replace(live, str(path))
        reads.append({"path": str(path), "sheet_name": None, "via": "code"})
    state = {"messages": [HumanMessage(content=turn["question"])], "coder_definition": {"code": code},
             "output_json": turn["output_json"], "execution_result": {"status": "success", "reads": reads}}
    # The stored narrative is what rc15 showed; the claim it held back is in
    # the stored report, verbatim. Put it back where it stood.
    held = turn["verification_report_on_rc15"]["findings"][0]["claim"]
    month = "- The month‑to‑outcome correlation is also identical (–0.501).  \n"
    assert month in turn["narrative"] and held not in turn["narrative"]
    narrative = turn["narrative"].replace(month, month + held + "  \n")
    message = _narrated(state, narrative, monkeypatch)
    state["messages"] = state["messages"] + [message]
    report = claim_verifier_node(state)["verification_report"]
    entry = _ask(client, monkeypatch, {**state, "messages": [message], "verification_report": report})
    content = entry["content"]
    shown = [s for s in _sentences(narrative) if s in content]
    assert shown and all(s + SUFFIX in content for s in shown), content
    segs = _segments(entry)

    note = next(line for line in content.splitlines() if line.startswith("I've held back"))
    assert _kind_of(segs, note) == "plain" and SUFFIX not in note
    for line in content.splitlines():
        if line.startswith(("Why:", "Scope:")):
            assert _kind_of(segs, line) == "plain" and SUFFIX not in line, line
    assert held not in content
    kept = [s for s in _sentences(narrative) if s in content]
    assert kept, content
    for sentence in kept:
        assert _kind_of(segs, sentence) == "commentary", sentence
        assert sentence + SUFFIX in content, sentence


def test_the_drivers_template_is_plain_throughout(client, monkeypatch, tmp_path):
    """#581's drivers answer: every word is the template's, so none is marked."""
    from app.agents.drivers_route import drivers_route_node
    try:
        from tests.test_predictors_deterministic_route import QUESTION, patients
    except ImportError:
        from test_predictors_deterministic_route import QUESTION, patients

    csv = tmp_path / "stroke.csv"
    patients().to_csv(csv, index=False)
    out = drivers_route_node({"messages": [HumanMessage(content=QUESTION)],
                              "data_source_location_local": str(csv)})
    assert out["drivers_answered"] is True
    final = {k: v for k, v in out.items() if k != "drivers_answered"}
    entry = _ask(client, monkeypatch, final)
    segs = _segments(entry)
    assert {s["kind"] for s in segs} == {"plain"}, [s for s in segs if s["kind"] != "plain"]
    assert SUFFIX not in entry["content"]
    assert "has the strongest association with" in entry["content"]


def test_the_web_chats_drivers_answer_is_plain_throughout(client, monkeypatch, tmp_path):
    """The same answer from the web chat's own drivers path (before the graph)."""
    try:
        from tests.test_predictors_deterministic_route import QUESTION, _MustNotRunGraph, _upload_thread
    except ImportError:
        from test_predictors_deterministic_route import QUESTION, _MustNotRunGraph, _upload_thread
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", _MustNotRunGraph(), raising=False)
    thread = _upload_thread(client, "c1-drv", tmp_path)
    resp = client.post(f"/threads/{thread['thread_id']}/messages",
                       json={"content": QUESTION, "metadata": {"dataset_id": thread["dataset_id"]}},
                       headers=make_auth_headers("c1-drv"))
    assert resp.status_code == 200, resp.text
    entry = [m for m in resp.json()["messages"] if m["role"] == "assistant"][-1]
    assert {s["kind"] for s in _segments(entry)} == {"plain"}
    assert SUFFIX not in entry["content"]


# --------------------------------------------------------------------------- #
# The planner's own replies
# --------------------------------------------------------------------------- #

try:
    from tests.test_dta_transfer_integrity import (  # noqa: F401  (fixtures are used by name)
        _ASK, _no_ray, _planner_rebuilds_the_old_plan, _turn, sealing_key, world)
except ImportError:
    from test_dta_transfer_integrity import (  # noqa: F401
        _ASK, _no_ray, _planner_rebuilds_the_old_plan, _turn, sealing_key, world)


def test_no_transfer_waiting_is_plain(world, client, monkeypatch):
    """A bare "yes" after a cancel: the planner answers "there's no transfer
    waiting" from its own template; nothing in it is the model's."""
    # The world's registry carries passwords the old way; since #599 the
    # planner opens credentials by handle only, so they live in the stand-in
    # sealed store (tests/dta_sealed.py), as for the transfer tests it reuses.
    try:
        from tests import dta_sealed
    except ImportError:  # pragma: no cover
        import dta_sealed
    dta_sealed.install(monkeypatch)
    _planner_rebuilds_the_old_plan(monkeypatch)
    state, _src, _dst = world
    _turn(state, _ASK)
    _turn(state, "no")
    reply = _turn(state, "yes")
    assert reply.startswith("There's no transfer waiting")
    message = [m for m in state["messages"] if isinstance(m, AIMessage)][-1]
    entry = _ask(client, monkeypatch, {"messages": [message]})
    segs = _segments(entry)
    assert {s["kind"] for s in segs} == {"plain"}
    assert entry["content"] == reply


def test_a_conversational_reply_is_commentary(client, monkeypatch):
    """respond_to_user is the model's own answer: every sentence is marked."""
    from app.agents import planner

    text = "Your data covers 2023. The busiest month was March, e.g. 412 orders."
    monkeypatch.setattr(planner, "llm", object(), raising=False)
    monkeypatch.setattr(planner, "_invoke_planner_with_recovery",
                        lambda msgs, tool_choice, kw: AIMessage(content="", tool_calls=[
                            {"name": "respond_to_user", "id": "r", "args": {"response_text": text}}]))
    state = {"messages": [HumanMessage(content="what period does my data cover?")]}
    out = planner.plan_etl_job(state)
    message = [m for m in out["messages"] if isinstance(m, AIMessage)][-1]
    assert message.content == text
    entry = _ask(client, monkeypatch, {"messages": [message]})
    assert entry["content"] == ("Your data covers 2023." + SUFFIX + " The busiest month was March, "
                                "e.g. 412 orders." + SUFFIX)
    segs = _segments(entry)
    assert _kind_of(segs, "Your data covers 2023.") == "commentary"
    assert _kind_of(segs, "The busiest month was March, e.g. 412 orders.") == "commentary"  # no cut at "e.g."
    assert entry["content"] == ("Your data covers 2023." + SUFFIX + " The busiest month was March, "
                                "e.g. 412 orders." + SUFFIX)


# --------------------------------------------------------------------------- #
# A reopened thread renders as the turn did
# --------------------------------------------------------------------------- #

def test_the_thread_history_keeps_the_segments(client, monkeypatch, tmp_path):
    import asyncio

    from app.api import helpers

    turn = TURNS["rw7"]
    state = _live_state(turn, _related_data(tmp_path))
    state["messages"] = state["messages"][:1]
    message = _narrated(state, turn["narrative"], monkeypatch)
    report = claim_verifier_node({**state, "messages": state["messages"] + [message]})["verification_report"]
    entry = _ask(client, monkeypatch, {**state, "messages": [message], "verification_report": report})
    stored = [m for m in server.THREAD_META[entry["_thread_id"]]["lc_msgs"] if isinstance(m, AIMessage)][-1]
    # Through the durable store (Redis JSON) and back, then served as history.
    revived = helpers._jsonable_to_lc_msgs(json.loads(json.dumps(helpers._lc_msgs_to_jsonable([stored]))))
    server.THREAD_META["reopened"] = {"lc_msgs": revived, "metadata": {}}
    history = asyncio.run(helpers.read_thread_msgs("reopened"))
    assert history[-1]["content"] == entry["content"]
    assert history[-1].get("reply_segments") == entry["reply_segments"]


# --------------------------------------------------------------------------- #
# Golden: the false sentences rc14 and rc15 showed unmarked
# --------------------------------------------------------------------------- #

_GOLDEN = json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.mark.parametrize("group", _GOLDEN["narratives"], ids=lambda g: g["id"])
def test_every_false_sentence_from_rc14_and_rc15_is_commentary(client, monkeypatch, group):
    """The list is RECONSTRUCTED (see the fixture's _source): the original
    40-sentence labelled set was lost to a /tmp cleanup. Every one of these
    sentences was model-written and shown as fact; each must now come out
    marked."""
    state = {"messages": [HumanMessage(content=group["question"])],
             "output_json": [{"x": 1}, {"x": 2}], "execution_result": {"status": "success"}}
    message = _narrated(state, group["narrative"], monkeypatch)
    entry = _ask(client, monkeypatch, {"messages": [message]})
    content = entry["content"]
    sentences = _sentences(group["narrative"])
    for item in group["sentences"]:
        sentence = next(s for s in sentences if item["quoted"][0] in s)
        assert sentence + SUFFIX in content, (item["id"], sentence)
    segs = _segments(entry)
    for item in group["sentences"]:
        for fragment in item["quoted"]:
            assert _kind_of(segs, fragment) == "commentary", (item["id"], fragment)
        sentence = next(s["text"] for s in segs if item["quoted"][0] in s["text"])
        assert sentence + SUFFIX in content, (item["id"], sentence)


# --------------------------------------------------------------------------- #
# Live shape: rweb16's stored turns (rc16, the base this marks)
# --------------------------------------------------------------------------- #

RWEB16 = json.loads((Path(__file__).parent / "fixtures" / "commentary_marking" / "rweb16_turns.json")
                    .read_text(encoding="utf-8"))["turns"]


def _shown_after_602(shown: str) -> str:
    """What rc16 showed, with #602's one intended change to the layout: a
    paragraph's withheld last sentence no longer runs the next paragraph into
    it (rc16 related run 3: "... key-to-key match). **Next actions**")."""
    return shown.replace("key‑to‑key match). **Next actions**", "key‑to‑key match).\n\n**Next actions**")


def _unmarked(entry: Dict[str, Any]) -> str:
    return "".join(s["text"] for s in _segments(entry))


@pytest.mark.parametrize("key", ["related_scope", "withheld"])
def test_rweb16_analysis_turns_mark_the_narration_only(client, monkeypatch, key):
    """The narrator's text from rc16's verify-time state, with rc16's own
    verification report, through the server: what rc16 showed is shown again
    word for word, with each model sentence marked and the verifier's notes
    (the scope line; the held-back note and its "Why:") plain."""
    turn = RWEB16[key]
    state = {"messages": [HumanMessage(content=turn["augmented_human_message"])],
             "output_json": turn["output_json"], "execution_result": {"status": "success"}}
    message = _narrated(state, turn["narrative"], monkeypatch)
    entry = _ask(client, monkeypatch, {"messages": [message], "output_json": turn["output_json"],
                                       "verification_report": turn["verification_report"]})
    content = entry["content"]
    shown = [s for s in _sentences(turn["narrative"]) if s in turn["shown_text"]]
    assert shown and all(s + SUFFIX in content for s in shown), content
    segs = _segments(entry)
    assert _unmarked(entry) == turn["shown_text"]
    notes = [line for line in turn["shown_text"].splitlines()
             if line.startswith(("Scope:", "I've held back", "Why:"))]
    assert notes, turn["shown_text"]
    if key == "related_scope":
        assert any(n.startswith("Scope:") for n in notes)
    else:
        assert any(n.startswith("I've held back") for n in notes)
    for line in notes:
        assert _kind_of(segs, line) == "plain" and line + SUFFIX not in content, line
    for sentence in shown:
        assert _kind_of(segs, sentence) == "commentary", sentence


def test_rweb16_drivers_turn_is_plain_throughout(client, monkeypatch):
    """rc16's "What drives stroke?" answer and table, through the web chat's
    drivers branch: the template's every line stays plain and unmarked."""
    turn = RWEB16["drivers"]
    _quiet_send(monkeypatch)
    monkeypatch.setattr(server, "GRAPH", FakeGraph(), raising=False)
    monkeypatch.setattr(server, "answer_drivers_turn",
                        lambda *a, **k: (turn["shown_text"], turn["output_table"]), raising=False)
    upload = _do_upload(client, user_id="c1-drv16")
    resp = client.post(f"/threads/{upload['thread_id']}/messages",
                       json={"content": turn["question"], "metadata": {"dataset_id": upload["dataset_id"]}},
                       headers=make_auth_headers("c1-drv16"))
    assert resp.status_code == 200, resp.text
    entry = [m for m in resp.json()["messages"] if m["role"] == "assistant"][-1]
    assert entry["content"] == turn["shown_text"]
    assert {s["kind"] for s in _segments(entry)} == {"plain"}


def test_rweb16_no_transfer_waiting_is_plain(client, monkeypatch):
    """rc16's cancel cell: transfer, "cancel", another question, then "yes".
    The planner answers from its template; the reply stays plain."""
    from app.agents import planner

    turn = RWEB16["no_transfer"]
    calls = _planner_rebuilds_the_old_plan(monkeypatch)
    history = [HumanMessage(content=m["content"]) if m["role"] == "human" else AIMessage(content=m["content"])
               for m in turn["messages"]]
    out = planner.plan_etl_job({"messages": history})
    message = [m for m in out["messages"] if isinstance(m, AIMessage)][-1]
    assert message.content == turn["shown_text"] and calls == []
    entry = _ask(client, monkeypatch, {"messages": [message]})
    assert entry["content"] == turn["shown_text"]
    assert {s["kind"] for s in _segments(entry)} == {"plain"}


_ITEM = re.compile(r"^[ \t]*(?:[-*+•]|\d{1,3}[.)])[ \t]+")


@pytest.mark.parametrize("key", ["table_and_list", "list_withheld"])
def test_rweb16_a_model_table_and_list_stay_valid_markdown(client, monkeypatch, key):
    """rc16's "What tables do I have and how do they relate?": the model
    answered with a markdown table and numbered lists. A string-only reader
    (the API, MCP, an old thread) must still get a table that parses, the
    label once after it on its own line, and each list item with its marker
    and one label at its end -- never "1. (commentary ...)"."""
    from markdown_it import MarkdownIt

    from app.agents import planner

    turn = RWEB16[key]
    monkeypatch.setattr(planner, "llm", object(), raising=False)
    monkeypatch.setattr(planner, "_invoke_planner_with_recovery",
                        lambda msgs, tool_choice, kw: AIMessage(content="", tool_calls=[
                            {"name": "respond_to_user", "id": "r", "args": {"response_text": turn["narrative"]}}]))
    out = planner.plan_etl_job({"messages": [HumanMessage(content=turn["augmented_human_message"])]})
    message = [m for m in out["messages"] if isinstance(m, AIMessage)][-1]
    assert message.content == turn["narrative"]
    entry = _ask(client, monkeypatch, {"messages": [message], "verification_report": turn["verification_report"]})
    content = entry["content"]

    # The table first: it parses, no cell carries the label, the label follows it once on its own line.
    tokens = MarkdownIt("commonmark").enable("table").parse(content)
    rows = [line.rstrip() for line in turn["narrative"].splitlines() if line.strip().startswith("|")]
    if rows:
        assert [t.type for t in tokens].count("table_open") == 1, content
        cells = [tokens[i + 1].content for i, t in enumerate(tokens) if t.type in ("th_open", "td_open")]
        assert cells and not any("commentary" in c for c in cells), cells
        assert [t.type for t in tokens].count("tr_open") == len(rows) - 1      # the separator is no row
        assert content.count(rows[-1] + TABLE_SUFFIX) == 1 and content.count(TABLE_SUFFIX) == 1, content

    # Then each list item: its marker with its text, one label at its end.
    items = [line for line in content.splitlines() if _ITEM.match(line)]
    assert len(items) >= 3, content
    for line in items:
        assert not re.match(r"^[ \t]*(?:[-*+•]|\d{1,3}[.)])[ \t]+\(commentary", line), line
        assert line.count(SUFFIX) == 1 and line.rstrip().endswith(SUFFIX), line
    assert [t.type for t in tokens].count("list_item_open") == len(items)

    segs = _segments(entry)
    assert _unmarked(entry) == _shown_after_602(turn["shown_text"])
    for line in items:
        assert _kind_of(segs, line.rstrip()[: -len(SUFFIX)].strip()) == "commentary", line
