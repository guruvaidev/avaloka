"""Integration: scan, plan and explain — as one step, proposing only."""

import numpy as np
import pandas as pd
import pytest

from app.agents.pii_agent import Strategy
from app.agents.preparation_agent import Objective
from app.agents.preparation_narrator import narrate_headline, narrate_preparation
from app.agents.preparation_pipeline import (apply_proposal, preparation_pipeline_node,
                                             propose_preparation)

KEY = b"integration-test-key"


@pytest.fixture
def messy():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "customer_email": [f"user{i}@example.com" for i in range(200)],
        "revenue": ["$1,204.50", "$980", "$2,340.25", None] * 50,
        "signup_date": ["2024-01-15", "2023-06-30", "2022-12-01", "2024-03-22"] * 50,
        "notes": [None] * 160 + ["late payment"] * 40,
        "region": ["north", "south", "east", "west"] * 50,
        "date_of_birth": ["1985-04-12", "1990-11-30", "1972-01-05", "1999-07-19"] * 50,
        "spend": rng.lognormal(3, 1, 200),
    })
    df.loc[df.sample(60, random_state=1).index, "spend"] = np.nan
    return df


# --------------------------------------------------------------------------- #
# The regression the narration exposed
# --------------------------------------------------------------------------- #

def test_a_numeric_column_is_never_personal_data(messy):
    """A float renders as digits and a dot — the phone pattern's alphabet."""
    p = propose_preparation(messy)
    flagged = {f["column"] for f in p.pii["findings"]}
    assert "spend" not in flagged, "a revenue column must not be pseudonymised into noise"
    assert "revenue" not in flagged


def test_real_identifiers_are_still_found(messy):
    p = propose_preparation(messy)
    direct = {f["column"] for f in p.pii["findings"] if f["sensitivity"] == "direct"}
    assert "customer_email" in direct


# --------------------------------------------------------------------------- #
# Policy: scan always, apply never
# --------------------------------------------------------------------------- #

def test_pii_is_scanned_without_being_asked(messy):
    assert propose_preparation(messy).pii["contains_pii"] is True


def test_proposing_changes_nothing(messy):
    before = messy.copy()
    propose_preparation(messy)
    pd.testing.assert_frame_equal(messy, before)


def test_applying_does_not_deidentify_unless_asked(messy):
    p = propose_preparation(messy)
    out = apply_proposal(messy, p)
    assert out["customer_email"].iloc[0] == "user0@example.com"
    assert pd.api.types.is_numeric_dtype(out["revenue"]), "cleaning still happened"


def test_deidentification_is_opt_in_and_keyed(messy):
    p = propose_preparation(messy)
    out = apply_proposal(messy, p, deidentify=True, pii_key=KEY)
    assert out["customer_email"].iloc[0] != "user0@example.com"


def test_needs_attention_when_something_is_unfixable_or_personal(messy):
    assert propose_preparation(messy).needs_attention is True


def test_a_clean_dataset_needs_no_attention():
    clean = pd.DataFrame({"region": ["north", "south"] * 50,
                          "units": list(range(100))})
    p = propose_preparation(clean)
    assert p.needs_attention is False


# --------------------------------------------------------------------------- #
# The explanation
# --------------------------------------------------------------------------- #

def test_explanation_is_plain_text_a_person_can_read(messy):
    text = propose_preparation(messy, dataset_name="customers.csv").explanation
    assert "customers.csv" in text
    for jargon in ("coerce_numeric", "impute", "missing_rate", "{", "}", "None"):
        assert jargon not in text, f"{jargon!r} is a data structure, not an explanation"


def test_explanation_states_the_consequence_not_just_the_measurement(messy):
    text = propose_preparation(messy).explanation
    assert "one row in three" in text or "one row in four" in text
    assert "quietly change the answers" in text


def test_refusals_are_explained_rather_than_silent(messy):
    text = propose_preparation(messy).explanation
    assert "notes" in text
    assert "inventing most of the column" in text


def test_currency_is_explained_with_a_real_example(messy):
    text = propose_preparation(messy).explanation
    assert "currency symbols" in text and "$1,204.50" in text


def test_personal_data_is_reported_and_says_it_will_not_act(messy):
    text = propose_preparation(messy).explanation
    assert "customer_email" in text
    assert "won't change anything until you ask" in text


def test_every_figure_is_attributed_to_measured_rows(messy):
    text = propose_preparation(messy).explanation
    assert "200 rows I measured" in text and "not an estimate" in text


def test_a_clean_dataset_gets_an_honest_all_clear_not_invented_concerns():
    clean = pd.DataFrame({"region": ["north", "south"] * 50, "units": list(range(100))})
    text = propose_preparation(clean, dataset_name="tidy.csv").explanation
    assert "nothing that needs cleaning up" in text
    assert "worth fixing" not in text


def test_headline_is_one_line():
    clean = pd.DataFrame({"region": ["a", "b"] * 50, "units": list(range(100))})
    assert "\n" not in narrate_headline(propose_preparation(clean).plan)


def test_narrator_tolerates_an_empty_report():
    assert "nothing that needs cleaning" in narrate_preparation({"decisions": []})


# --------------------------------------------------------------------------- #
# Graph node
# --------------------------------------------------------------------------- #

def test_node_writes_everything_the_planner_needs(messy):
    out = preparation_pipeline_node({"dataframe": messy, "dataset_name": "customers.csv",
                                     "objective": "analysis"})
    assert set(out) == {"preparation_proposal", "preparation_explanation",
                        "preparation_headline", "pii_report",
                        "preparation_needs_attention"}
    assert out["preparation_explanation"].startswith("Before I analyse customers.csv")
    assert out["preparation_needs_attention"] is True


def test_node_without_a_dataframe_is_a_no_op():
    assert preparation_pipeline_node({}) == {}


def test_node_declares_its_contract():
    from app.agents.contract import registry
    spec = registry()["preparation_pipeline"]
    assert spec.stage.value == "prepare"
    assert "preparation_explanation" in spec.writes


def test_modelling_objective_fits_on_train_rows_only(messy):
    train = list(range(100))
    p = propose_preparation(messy, objective=Objective.MODELLING, train_index=train)
    assert p.plan["fitted_rows"] == 100


def test_applying_an_unfitted_proposal_is_refused():
    from app.agents.preparation_pipeline import PreparationProposal
    bare = PreparationProposal(plan={}, pii={}, explanation="", headline="")
    with pytest.raises(ValueError, match="no fitted plan"):
        apply_proposal(pd.DataFrame({"a": [1]}), bare)
