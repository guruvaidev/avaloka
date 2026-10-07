"""Redaction is a control the owner relies on. Both halves are pinned here:
what it removes, and -- just as deliberately -- what it does not."""
import time

import pytest

from app.analytics.redact import scrub
from app.analytics.schema import MAX_PROMPT_CHARS

# Credential-shaped fixtures are assembled at import time. The repository's own
# secret scanner (tests/contract/test_c1_repo_hygiene.py) reads tracked files as
# text, and a literal key shape here would -- correctly -- fail it.
_AWS = "AKIA" + "IOSFODNN7EXAMPLE"
_GSK = "gsk" + "_Zx81kPq92LmNw0Vb7TtYy"
_GHP = "ghp" + "_" + "16C7e42F292c6912E7710c838347Ae178B4a"
_PEM = "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIEvQIBADANBg\n-----END RSA " + "PRIVATE KEY-----"
_JWT = "eyJhbGciOiJIUzI1NiJ9" + ".eyJzdWIiOiIxMjM0NTYifQ" + ".SflKxwRJSMeKKF2QT4"
_DSN = "postgres://admin:" + "hunter2" + "@db.internal:5432/prod"

# (input, substrings that must be gone)
REMOVED = [
    ("email ravi.kumar@acme.co.in the churn list", ["ravi.kumar", "acme.co.in"]),
    ("call me on +44 7700 900123 about it", ["7700", "900123"]),
    ("call me on 07700 900123", ["07700", "900123"]),                 # national form, no '+'
    ("phone (415) 555-0132 please", ["555-0132", "415"]),
    ("SSN 078-05-1120 for this customer", ["078-05-1120", "1120"]),   # 9 digits
    ("NHS number 943 476 5919", ["943 476 5919", "5919"]),
    ("account 12345678 sort code 20-00-00", ["12345678", "20-00-00"]),
    ("card 4111 1111 1111 1111 declined", ["4111", "1111 1111"]),
    ("IBAN GB29NWBK60161331926819", ["GB29NWBK60161331926819", "60161331926819"]),
    ("IBAN GB29 NWBK 6016 1331 9268 19", ["NWBK", "9268"]),
    ("customers in Bengaluru 560001", ["560001"]),
    ("patient born 12/03/1981", ["12/03/1981", "1981"]),
    ("born 1981.03.12 in", ["1981.03.12"]),
    ("my password is hunter2", ["hunter2"]),                          # labelled, no ':' or '='
    ("password=hunter2 and go", ["hunter2"]),
    ("api key: abcd1234efgh", ["abcd1234efgh"]),
    ("token is s3cr3tvalue", ["s3cr3tvalue"]),
    ("use sk-proj-AbCdEf0123456789xyz to call", ["AbCdEf0123456789xyz"]),
    (_GSK + " here", ["Zx81kPq92LmNw0Vb7TtYy"]),
    (_GHP, ["16C7e42F292c"]),
    ("aws " + _AWS + " key", ["IOSFODNN7EXAMPLE"]),
    ("jwt " + _JWT, ["eyJhbGci", "SflKxw"]),
    ("Authorization: Bearer abcdef123456ghijkl", ["abcdef123456ghijkl"]),
    (_PEM, ["MIIEvQIBADANBg"]),
    ("hash 5d41402abc4b2a76b9719d911017c592 dup", ["5d41402abc4b2a76b9719d911017c592"]),
    ("row 123e4567-e89b-12d3-a456-426614174000 is odd", ["123e4567", "426614174000"]),
    ("load https://acme.sharepoint.com/sites/hr/q3_layoffs_final.xlsx", ["sharepoint", "layoffs", "acme"]),
    (_DSN, ["hunter2", "db.internal", "admin"]),
    ("gs://acme-finance/exports/payroll.parquet summary", ["acme-finance", "payroll"]),
    ("see www.acme-internal.example/payroll", ["acme-internal"]),
    ("summarise q3_layoffs_final.csv", ["q3_layoffs_final", "layoffs"]),
    ("open /home/ravi/data/salaries.parquet", ["ravi", "salaries"]),
    (r"open C:\Users\ravi\Desktop\salaries.xlsx now", ["ravi", "salaries", "Desktop"]),
    (r"read \\fileserver\hr\payroll now", ["fileserver", "payroll"]),
    ("where status = 'churned' and tier = \"platinum\"", ["churned", "platinum"]),
    ("customers named “Ravi Kumar”", ["Ravi Kumar"]),
    ("call ０７７００ ９００１２３", ["９００１２３", "900123"]),               # full-width digits
    ("mail ravi\u200b@acme.example", ["acme.example"]),                # zero-width split
]


@pytest.mark.parametrize("text,gone", REMOVED, ids=[t[0][:32] for t in REMOVED])
def test_removed(text, gone):
    cleaned, counts = scrub(text)
    assert cleaned is not None and sum(counts.values()) >= 1
    for fragment in gone:
        assert fragment not in cleaned, f"{fragment!r} survived in {cleaned!r}"


# Questions that must survive intact, or the data is useless for feedback analysis.
KEPT = [
    "what is the average revenue by region in 2024",
    "top 10 customers by churn_rate over the last 90 days",
    "show orders where amount > 50000 grouped by month",
    "compare Q3 vs Q4 for the sales_fact table",
    "Ravi's and Meena's teams: who closed more deals?",
    "plot a histogram of age with 20 bins",
    "why did the model's accuracy drop to 0.82?",
]


@pytest.mark.parametrize("text", KEPT)
def test_ordinary_questions_pass_through_unchanged(text):
    cleaned, counts = scrub(text)
    assert cleaned == text and counts == {}


# The honest half. These are NOT removed. If one of these tests starts failing
# because redaction got better, good -- update the docs and the egress inventory
# description in the same change. Until then nobody may claim otherwise.
@pytest.mark.parametrize("text,survives", [
    ("average of patient_hiv_status by clinic", "patient_hiv_status"),   # a column name is a diagnosis
    ("join salary_band to employees_terminated_2024", "employees_terminated_2024"),
    ("churn for patient Ravi Kumar", "Ravi Kumar"),                       # a personal name in prose
    ("customers at Nehru Road, Bengaluru", "Nehru Road, Bengaluru"),     # an address without digits
    ("orders from Acme Corp last year", "Acme Corp"),                     # an organisation
    ("rows where diagnosis is diabetes", "diabetes"),                     # an unquoted value
])
def test_documented_gaps_what_redaction_does_not_remove(text, survives):
    cleaned, _ = scrub(text)
    assert survives in cleaned


def test_idempotent():
    for text, _ in REMOVED:
        once, _ = scrub(text)
        assert scrub(once)[0] == once


def test_truncates_and_never_leaves_half_a_mask():
    cleaned, _ = scrub("word " * 99 + "x" + " ravi@acme.example" * 20)
    assert len(cleaned) <= MAX_PROMPT_CHARS + 1 and cleaned.endswith("…")
    assert "acme" not in cleaned and not cleaned.rstrip("…").endswith("[")


@pytest.mark.parametrize("bad", [None, 7, 4.2, b"bytes", ["a"], {"a": 1}, "", "   \n\t "])
def test_unusable_input_yields_nothing(bad):
    assert scrub(bad)[0] is None


def test_text_past_the_work_bound_is_discarded_not_stored_raw():
    cleaned, _ = scrub("a " * 3000 + "ravi@acme.example 07700 900123")
    assert "acme" not in cleaned and "900123" not in cleaned


@pytest.mark.parametrize("hostile", ["1 " * 5000, "a" * 20000, "'" * 5000, "/" * 5000,
                                     "A1a-" * 3000, "(" * 4000 + "1", "x@" * 4000])
def test_hostile_input_is_bounded_in_time(hostile):
    # CPU time, not wall time: a loaded CI node must not fail this, a
    # catastrophic pattern must.
    start = time.process_time()
    scrub(hostile)
    assert time.process_time() - start < 2.0
