"""A named source must survive the LLM door, or be asked about — never guessed away.

The fast-path parsers are gated on _TRANSFER_VERBS. A transfer phrased with an
unlisted verb ("SHIFT the data from A to B") skips them entirely and is routed by
the planner LLM instead — and that door strips source_alias/source_object
unconditionally, so _handle_initiate_transfer fell through to
_resolve_implicit_source and transferred whatever dataset was OPEN. The user named
patient_data.csv and got their active sales dataset, reported as success. One word
("transfer" vs "shift") silently changed which data moved — the "two doors, one
room" invariant (see the fast-path's _normalize_llm_transfer_params) broken by a
feature that only door #1 could honor.

The fix keeps the LLM's *value* untrusted (it invents aliases) while using its
*signal* — that the user pointed at a source at all — to re-derive the source
deterministically from the user's own words, with the verb gate dropped because
intent is already established. Unparseable → ask, never guess.
"""
import pytest

from app.agents.planner import (
    _extract_named_source_transfer,
    _extract_transfer_aliases,
)

_NAMED = ("shift the data from C2C_Source_GCP to C2C_Destination_GCP, "
          "from patient_data.csv to out/result.json")


# ── the verb gate is intent-only, and droppable once intent is known ──────────

def test_unlisted_verb_defeats_the_verb_gated_parser():
    """Precondition: this is exactly why the prompt reaches the LLM door."""
    assert _extract_transfer_aliases(_NAMED) is None
    assert _extract_named_source_transfer(_NAMED) is None


def test_dropping_the_verb_gate_recovers_the_named_source():
    """Same prompt, verb gate dropped → the user's own words still parse cleanly."""
    params = _extract_named_source_transfer(_NAMED, require_verb=False)

    assert params is not None
    assert params.source_alias == "C2C_Source_GCP"
    assert params.source_object == "patient_data.csv"
    assert params.destination_alias == "C2C_Destination_GCP"
    assert params.dest_object == "out/result.json"


def test_listed_verb_is_unaffected_by_the_new_flag():
    """The fast-path's behaviour must not shift; require_verb=True stays the default."""
    p = ("transfer from C2C_Source_GCP to C2C_Destination_GCP, "
         "from patient_data.csv to out/result.json")
    with_verb = _extract_named_source_transfer(p)
    assert with_verb is not None
    assert with_verb.source_alias == "C2C_Source_GCP"
    # Default is require_verb=True — the fast-path keeps its deterministic gate.
    assert _extract_named_source_transfer(p, require_verb=False).source_alias == "C2C_Source_GCP"


# ── the gate still has to gate ────────────────────────────────────────────────

@pytest.mark.parametrize("prompt", [
    "fill nulls from the median to keep the column numeric",
    "rename the column from id to customer_id",
])
def test_verb_free_parsing_is_never_used_as_an_intent_gate(prompt):
    """Bare "from <a> to <b>" is far too loose to detect a transfer.

    These are transformation requests. They only ever reach require_verb=False if the
    LLM already called initiate_transfer, which is the actual gate — this pins that
    the verb-gated default (what the fast-path uses) does NOT claim them.
    """
    assert _extract_named_source_transfer(prompt) is None


def test_object_only_form_is_not_a_named_source():
    """"from in.csv to out.json" names objects, not connections → implicit source."""
    p = "shift from in.csv to out.json"
    assert _extract_named_source_transfer(p, require_verb=False) is None


def test_named_source_without_a_source_object_is_not_claimed():
    """No object to read → not the named-source grammar; the caller asks instead."""
    p = "shift the data from C2C_Source_GCP to C2C_Destination_GCP"
    assert _extract_named_source_transfer(p, require_verb=False) is None
