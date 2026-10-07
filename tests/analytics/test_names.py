"""The membership test. Deterministic: a planted name is either gone or the test fails."""
import json

import pytest

from app.analytics.names import MASK, Vocabulary, names_from_session_blobs, strip_names

NAMES = ["patient_hiv_status", "order_date", "region", "status", "id", "customerLifetimeValue",
         "public.sales_fact", "q3_layoffs_final.csv", "Unit Price", "employees_terminated_2024"]


def out(text, names=NAMES):
    result = strip_names(text, names)
    assert result is not None
    return result[0]


# (question, fragments that must be gone -- compared case-insensitively)
PLANTED = [
    # the planted column, in every spelling an analyst might use
    ("average patient_hiv_status by clinic", ["hiv", "patient"]),
    ("average PATIENT_HIV_STATUS by clinic", ["hiv", "patient"]),
    ("average Patient Hiv Status by clinic", ["hiv", "patient"]),
    ("average patient-hiv-status by clinic", ["hiv"]),
    ("average patientHivStatus by clinic", ["hiv"]),
    ("average patienthivstatus by clinic", ["hiv"]),
    ("average `patient_hiv_status` by clinic", ["hiv"]),
    ("average [patient_hiv_status] by clinic", ["hiv"]),
    ("sum(patient_hiv_status), max(order_date)", ["hiv", "order_date", "order"]),
    ("df['patient_hiv_status'].mean()", ["hiv"]),
    ("t.patient_hiv_status = 1", ["hiv"]),
    ("patient_hiv_status?", ["hiv"]),
    ("ｐａｔｉｅｎｔ_ｈｉｖ_ｓｔａｔｕｓ by clinic", ["ｈｉｖ", "hiv"]),          # full-width
    # camelCase column typed four ways
    ("plot customerLifetimeValue", ["lifetime"]),
    ("plot customer_lifetime_value", ["lifetime"]),
    ("plot customer lifetime value over time", ["lifetime"]),
    ("plot CUSTOMERLIFETIMEVALUE", ["lifetime"]),
    # a column with a space in its name
    ("mean Unit Price per month", ["unit price", "unit"]),
    ("mean unit_price per month", ["unit_price", "unit"]),
    # table names, qualified and not
    ("count rows in public.sales_fact", ["sales_fact", "sales"]),
    ("count rows in sales_fact", ["sales_fact", "sales"]),
    ("count rows in Sales Fact", ["sales fact"]),
    ("join SALES_FACT to it", ["sales_fact"]),
    ("how many employees_terminated_2024 rows", ["terminated", "employees"]),
    # dataset / file names, with and without the extension
    ("summarise q3_layoffs_final", ["layoffs"]),
    ("summarise q3 layoffs final", ["layoffs"]),
    # names that are ordinary English words
    ("what is the status of the pipeline", ["status"]),
    ("which region grew fastest", ["region"]),
    ("Status by Region", ["status", "region"]),
    # names inside longer tokens
    ("group by region_id", ["region"]),
    ("list the regions", ["region"]),
    ("per-region totals", ["region"]),
    ("the byRegion view", ["region"]),
    ("subregional split", ["region"]),
    ("statuses over time", ["status"]),
    ("count ids", [" ids"]),
]


@pytest.mark.parametrize("text,gone", PLANTED, ids=[p[0][:36] for p in PLANTED])
def test_planted_names_are_removed(text, gone):
    stripped = out(text).casefold()
    assert MASK in stripped
    for fragment in gone:
        assert fragment.casefold() not in stripped, f"{fragment!r} survived in {stripped!r}"


def test_short_names_match_whole_words_only():
    # ``id`` is a column. It must go; "valid", "video" and "paid" must not.
    stripped = out("is the id valid for paid video rows")
    assert stripped == f"is the {MASK} valid for paid video rows"


def test_words_that_are_not_names_survive():
    assert out("what is the average revenue by month in 2024") == "what is the average revenue by month in 2024"


def test_reports_how_many_tokens_were_removed():
    text, hits = strip_names("status by region and month", NAMES)
    assert text == f"{MASK} by {MASK} and month" and sum(hits.values()) == 2


# Over-strip, and count precisely: every removal lands in exactly one counter.
@pytest.mark.parametrize("text,expected", [
    ("average patient_hiv_status", {"exact": 1}),                 # an identifier, whole
    ("group by region_id", {"exact": 1}),                         # name on a word boundary
    ("the byRegion view", {"exact": 1}),                          # camelCase boundary
    ("order date range", {"exact": 2}),                           # a name typed as two words
    ("sum(order_date), max(region_id)", {"exact": 2}),
    ("what is the status of Q3", {"common_word": 1}),             # English, also a column
    ("Status by Region", {"common_word": 2}),
    ("list the regions", {"substring": 1}),                       # no boundary
    ("subregional statuses", {"substring": 2}),
    ("status of patient_hiv_status in regions", {"common_word": 1, "exact": 1, "substring": 1}),
    ("average revenue by month", {}),
])
def test_each_removal_is_classified(text, expected):
    _, hits = strip_names(text, NAMES)
    assert hits == {"exact": 0, "common_word": 0, "substring": 0, **expected}


def test_classification_never_decides_removal():
    # A common word that is a column is still removed; the list only labels it.
    from app.analytics.names import COMMON_WORDS
    assert "status" in COMMON_WORDS and "status" not in out("what is the status").casefold()


def test_adjacent_masks_collapse():
    assert out("show Unit Price now") == f"show {MASK} now"


# -- fail closed --------------------------------------------------------------

@pytest.mark.parametrize("names", [None, [], set(), [""], ["___", "  "], [None, 7], Vocabulary([])])
def test_no_usable_vocabulary_means_no_text_at_all(names):
    assert strip_names("average patient_hiv_status by clinic", names) is None


@pytest.mark.parametrize("text", [None, "", "   ", 7, b"bytes"])
def test_no_text_means_nothing(text):
    assert strip_names(text, NAMES) is None


def test_an_internal_error_yields_nothing_rather_than_the_input():
    class Broken:
        tuples = {("x",)}
        max_len = 1
        def __bool__(self): return True
        def hits_word(self, word): raise RuntimeError("boom")
    assert strip_names("average patient_hiv_status", Broken()) is None


# -- building the vocabulary from dataset sessions ----------------------------

def blob(user="u1", **extra):
    return json.dumps({"user_id": user, **extra})


def test_names_are_collected_from_every_schema_shape_the_sessions_use():
    names = names_from_session_blobs([
        blob(alias="Q3 Layoffs", filename="q3_layoffs_final.csv",
             uploaded_csv_columns=json.dumps(["patient_hiv_status", "region"])),
        blob(schema={"order_date": "datetime64", "amount": "float"}),
        blob(schema=[{"name": "salary_band", "type": "str"}]),
        blob(schema={"tables": {"sales_fact": {"columns": [{"name": "unit_price"}]}}}),
        blob(filename="exports/payroll.parquet"),
    ], "u1")
    for expected in ("patient_hiv_status", "region", "order_date", "amount", "salary_band",
                     "sales_fact", "unit_price", "Q3 Layoffs", "q3_layoffs_final.csv", "payroll.parquet"):
        assert expected in names
    assert "datetime64" not in names and "str" not in names        # types are not names


def test_another_users_sessions_contribute_nothing():
    names = names_from_session_blobs([blob(user="someone-else", uploaded_csv_columns=["secret_col"])], "u1")
    assert names == set()


@pytest.mark.parametrize("blobs", [
    [blob(uploaded_csv_columns=["a"]), None],          # an expired session
    [blob(uploaded_csv_columns=["a"]), ""],
    [blob(uploaded_csv_columns=["a"]), "{not json"],
    [blob(uploaded_csv_columns=["a"]), "[1, 2]"],
])
def test_a_partial_vocabulary_is_no_vocabulary(blobs):
    assert names_from_session_blobs(blobs, "u1") is None


def test_fewer_sessions_returned_than_asked_for_is_no_vocabulary():
    assert names_from_session_blobs([blob(uploaded_csv_columns=["a"])], "u1", expected=2) is None


# -- the session-store provider: tested on fakes, wired nowhere ----------------

import asyncio

from app.analytics.names import MAX_SESSIONS, session_store_provider


class FakeCache:
    def __init__(self, sids):
        self.sids = sids

    async def smembers(self, key):
        return set(self.sids)


def run_provider(sids, blobs_by_key, user="u1", cache=True):
    async def mget(keys):
        return [blobs_by_key.get(k) for k in keys]
    provider = session_store_provider(lambda: FakeCache(sids) if cache else None,
                                      lambda u: f"user:{u}", lambda s: f"sess:{s}", mget)
    return asyncio.run(provider(None, user))


def test_provider_collects_names_across_a_users_sessions():
    names = run_provider(["a", "b"], {"sess:a": blob(uploaded_csv_columns=["region"]),
                                      "sess:b": blob(schema={"order_date": "date"})})
    assert {"region", "order_date"} <= names


def test_provider_fails_closed():
    assert run_provider(["a"], {}, cache=False) is None                       # no session store
    assert run_provider(["a", "b"], {"sess:a": blob(uploaded_csv_columns=["x"])}) is None   # one expired
    assert run_provider([str(i) for i in range(MAX_SESSIONS + 1)], {}) is None             # too many
    assert run_provider([], {}) == set()          # no datasets: empty, which strip_names refuses


def test_provider_is_not_wired_into_the_server():
    from pathlib import Path
    server = (Path(__file__).resolve().parents[2] / "app" / "api" / "server.py").read_text("utf-8")
    assert "names_provider" not in server and "session_store_provider" not in server
