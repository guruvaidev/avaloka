"""C2 commentary marking on the upload responses rweb16 stored on rc16.

The fixture holds only the upload-visible text of 12 live upload+profile
turns (rw7, rw7b, rw7c × taxi, recipes, stroke, messy, nullxlsx; related's
profile turn did not return): chart titles, findings, headlines, the data
sentences, the profile overview and the upload message, as the server sent
them before C2. Each cell is replayed through the assembly's last step, and
the replay must rebuild the stored upload message word for word before any
marking is judged.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.agents import visualization_agent as va
from app.api import reply_segments as rs

SUFFIX = " (commentary — not checked)"
FIXTURE = Path(__file__).parent / "fixtures" / "commentary_upload" / "rweb16_rc16_upload_text.json"
CELLS = json.loads(FIXTURE.read_text(encoding="utf-8"))
LLM_LISTS = ("red_flags", "concerns", "strengths", "quick_wins", "analysis_suggestions")


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    monkeypatch.setattr(va, "viz_llm", None)


def _replay(cell):
    stored = CELLS[cell]
    viz = copy.deepcopy({"charts": stored["charts"], "profile_overview": stored["profile_overview"]})
    total, n_cols = stored["dataset"]["total_rows"], stored["dataset"]["n_columns"]
    step = getattr(va, "mark_upload_commentary", None)
    if step is not None:
        step(viz, total, n_cols)
    else:  # before C2 the assembly's last step only wrote the message
        viz["upload_message"] = va.build_upload_message(viz["profile_overview"], viz["charts"], total, n_cols)
    return stored, viz


def _model_sentences(stored):
    """What the models wrote in this turn, as the server showed it."""
    out = []
    for ch in stored["charts"]:
        if ch["insight_source"] == "llm":
            out.append(ch["headline"])
        if ch.get("llm_title") and ch["title"] == ch["llm_title"]:
            out.append(ch["title"])
    ov = stored["profile_overview"]
    out += [ov[k] for k in ("readiness_reasoning", "domain_reasoning") if ov.get(k)]
    out += [t for k in LLM_LISTS for t in ov.get(k) or []]
    return [s for s in out if s]


def test_fixture_covers_the_cells():
    assert len(CELLS) == 12
    assert {c.split("/")[1] for c in CELLS} == {"taxi_rg", "recipes_jsonl", "stroke", "messy_xlsx", "nullxlsx"}
    assert sum(ch["insight_source"] == "llm" for v in CELLS.values() for ch in v["charts"]) >= 30


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_model_card_text_is_commentary_and_data_text_plain(cell):
    stored, viz = _replay(cell)
    for before, ch in zip(stored["charts"], viz["charts"]):
        facts = before["insight_facts"]
        if before["insight_source"] == "llm":
            # Lead and headline: the model's words, suffixed in the strings old readers show.
            # One suffix per sentence (C1's contract), so a two-sentence lead gets two.
            assert ch["headline"] == rs.render(rs.commentary(before["headline"]))
            assert ch["headline"].endswith(SUFFIX)
            assert rs.text_of(ch["insight_segments"]) == before["insight"]
            assert "commentary" in {s["kind"] for s in ch["insight_segments"]}
            assert SUFFIX in ch["insight"] and ch["reason"] == ch["insight"]
            assert [s["kind"] for s in ch["headline_segments"]].count("commentary") >= 1
            assert rs.text_of(ch["headline_segments"]) == before["headline"]
            # The numbers the code wrote under it stay plain.
            if facts and before["insight"].endswith(facts):
                assert ch["insight_segments"][-1] == {"text": " " + facts, "kind": "plain"}
                assert ch["insight"].endswith(SUFFIX + " " + facts)
        else:
            assert ch["insight"] == before["insight"] and SUFFIX not in ch["insight"]
            assert ch["insight_segments"] == [{"text": before["insight"], "kind": "plain"}]
            assert ch["headline_segments"] is None
        assert facts is None or SUFFIX not in facts
        # A model title the checks kept is commentary; a neutral or code title plain.
        if before.get("llm_title") and before["title"] == before["llm_title"]:
            assert ch["title"] == rs.render(rs.commentary(before["title"])) and ch["title"].endswith(SUFFIX)
            assert {s["kind"] for s in ch["title_segments"]} >= {"commentary"}
        else:
            assert ch["title"] == before["title"]
            assert ch["title_segments"] == [{"text": before["title"], "kind": "plain"}]
        for key in ("insight", "headline", "title"):
            if ch.get(key + "_segments"):
                rs.check(ch[key], ch[key + "_segments"])


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_profile_watch_outs_and_readiness_are_commentary(cell):
    stored, viz = _replay(cell)
    before, ov = stored["profile_overview"], viz["profile_overview"]
    for key in ("readiness_reasoning", "domain_reasoning"):
        if before.get(key):
            assert ov[key] == rs.render(rs.commentary(before[key])) and ov[key].endswith(SUFFIX), key
            assert rs.text_of(ov[key + "_segments"]) == before[key]
            assert "plain" not in {s["kind"] for s in ov[key + "_segments"] if s["text"].strip()}
            rs.check(ov[key], ov[key + "_segments"])
    assert any(before.get(k) for k in LLM_LISTS), "a live profile has watch-outs"
    for key in LLM_LISTS:
        for text, shown, segs in zip(before.get(key) or [], ov[key], ov[key + "_segments"]):
            assert shown.endswith(SUFFIX) and rs.text_of(segs) == text, key
            rs.check(shown, segs)
    assert ov["readiness_score"] == before["readiness_score"] and ov["domain"] == before["domain"]


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_upload_message_marks_every_model_sentence(cell):
    stored, viz = _replay(cell)
    msg = viz["upload_message"]
    for sentence in _model_sentences(stored):
        if sentence in stored["upload_message"]:
            assert rs.render(rs.commentary(sentence)) in msg, sentence
    segs = viz["upload_message_segments"]
    # The replay is the real assembly: unmarked, it is the message the server sent.
    assert rs.text_of(segs) == stored["upload_message"]
    rs.check(msg, segs)
    plain = "".join(s["text"] for s in segs if s["kind"] == "plain")
    for sentence in _model_sentences(stored):
        assert sentence not in plain, sentence
    for ch in stored["charts"]:
        if ch["insight_source"] == "data" and ch["insight"] in stored["upload_message"]:
            assert ch["insight"] in plain


# Sentences of the false-unmarked shapes (golden list: "markedly higher", "dominate",
# "bulk") that rweb16 showed on rc16 because #585's checks passed them.
GOLDEN_SHAPES = {
    "rw7/stroke": "a markedly higher stroke rate",
    "rw7/taxi_rg": "Trip distances are dominated by short rides",
    "rw7b/messy_xlsx": "the bulk of items are low‑priced",
    "rw7b/nullxlsx": "Clinician notes are dominated by placeholder entries",
}


@pytest.mark.parametrize("cell", sorted(GOLDEN_SHAPES))
def test_golden_shaped_sentences_that_passed_the_checks_are_commentary(cell):
    stored, viz = _replay(cell)
    phrase = GOLDEN_SHAPES[cell]
    card = next(ch for ch in viz["charts"] if phrase in (ch.get("headline") or ""))
    assert card["headline"].endswith(SUFFIX)
    assert all(s["kind"] == "commentary" for s in card["headline_segments"] if s["text"].strip())
    commentary = [s["text"] for s in viz["upload_message_segments"] if s["kind"] == "commentary"]
    if phrase in stored["upload_message"]:
        assert any(phrase in t for t in commentary)


def test_idempotent_on_live_shapes():
    for cell in sorted(CELLS):
        stored, viz = _replay(cell)
        again = copy.deepcopy(viz)
        va.mark_upload_commentary(again, stored["dataset"]["total_rows"], stored["dataset"]["n_columns"])
        assert again == viz, cell
