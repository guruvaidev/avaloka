"""PII detection and vault-free de-identification."""

import pandas as pd
import pytest

from app.agents.pii_agent import (MissingPIIKey, PIIKind, PIIReport, Sensitivity, Strategy,
                                  apply_deidentification, classify_column,
                                  deidentification_contract,
                                  generalise_date_to_age_band, generalise_postcode,
                                  pii_key_required_columns, pseudonym, require_pii_key,
                                  scan_dataframe, tokenise_preserving_tail,
                                  transform_pii_value)

KEY = b"test-key-not-a-real-secret"


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def test_emails_detected_from_values_not_the_column_name():
    f = classify_column("contact_field", ["a@b.com", "c@d.co.uk", "e@f.org", "g@h.net"])
    assert f.kind is PIIKind.EMAIL and f.sensitivity is Sensitivity.DIRECT


def test_credit_card_requires_the_luhn_checksum():
    real = classify_column("num", ["4242424242424242"] * 5)
    assert real.kind is PIIKind.CREDIT_CARD
    # A 16-digit order id is not a card number.
    fake = classify_column("order_ref", ["1234567890123456"] * 5)
    assert fake.kind is not PIIKind.CREDIT_CARD


def test_column_name_alone_still_raises_a_finding():
    """A column called ssn deserves attention even with an unreadable sample."""
    f = classify_column("customer_ssn", ["REDACTED", "REDACTED"])
    assert f.kind is PIIKind.NATIONAL_ID
    assert f.confidence < 0.9, "a name hint is weaker evidence than matching values"


def test_name_plus_agreeing_values_is_high_confidence():
    f = classify_column("date_of_birth", ["1985-04-12", "1990-11-30", "1972-01-05"])
    assert f.kind is PIIKind.DATE_OF_BIRTH and f.confidence >= 0.9


def test_ordinary_columns_are_not_pii():
    for name, vals in [("revenue", [1.0, 2.0]), ("region", ["north", "south"]),
                       ("n_orders", [3, 4])]:
        assert classify_column(name, vals).kind is PIIKind.NONE


def test_ip_addresses_detected():
    assert classify_column("src", ["10.0.0.1", "192.168.1.7", "8.8.8.8"]).kind is PIIKind.IP_ADDRESS


def test_scan_reports_residual_risk_from_quasi_identifiers():
    df = pd.DataFrame({"date_of_birth": ["1985-04-12"], "postcode": ["SW1A 1AA"],
                       "gender": ["f"], "spend": [10.0]})
    report = scan_dataframe(df)
    assert report.contains_pii
    assert report.residual_risk, "quasi-identifiers in combination must be called out"
    assert "identify most individuals uniquely" in report.residual_risk[0]


def test_a_single_quasi_identifier_is_not_flagged_as_combination_risk():
    df = pd.DataFrame({"postcode": ["SW1A 1AA"], "spend": [10.0]})
    assert not scan_dataframe(df).residual_risk


# --------------------------------------------------------------------------- #
# The rule that matters: no unkeyed hashing
# --------------------------------------------------------------------------- #

def test_pseudonymising_without_a_key_is_refused(monkeypatch):
    monkeypatch.delenv("AVALOKA_PII_KEY", raising=False)
    with pytest.raises(MissingPIIKey, match="wordlist"):
        pseudonym("alice@example.com")


def test_sensitive_training_columns_require_the_pii_key(monkeypatch):
    monkeypatch.delenv("AVALOKA_PII_KEY", raising=False)
    report = scan_dataframe(pd.DataFrame({
        "customer_email": ["alice@example.com", "bob@example.com"],
        "spend": [10.0, 20.0],
    }))

    assert pii_key_required_columns(report) == ["customer_email"]
    with pytest.raises(MissingPIIKey, match="customer_email"):
        require_pii_key(report)


def test_generalisation_only_pii_does_not_require_a_key(monkeypatch):
    monkeypatch.delenv("AVALOKA_PII_KEY", raising=False)
    report = scan_dataframe(pd.DataFrame({
        "date_of_birth": ["1985-04-12", "1990-11-30"],
        "postcode": ["SW1A 1AA", "M5V 1A1"],
    }))

    assert pii_key_required_columns(report) == []
    assert require_pii_key(report) == []


def test_configured_key_allows_sensitive_training_columns(monkeypatch):
    monkeypatch.setenv("AVALOKA_PII_KEY", "test-only-key")
    report = scan_dataframe(pd.DataFrame({
        "contact": ["alice@example.com", "bob@example.com"],
    }))

    assert require_pii_key(report) == ["contact"]


def test_the_key_is_never_in_the_output():
    out = pseudonym("alice@example.com", salt="email", key=KEY)
    assert KEY.decode() not in out
    assert "alice" not in out and "example" not in out


# --------------------------------------------------------------------------- #
# Deterministic, join-preserving, vault-free
# --------------------------------------------------------------------------- #

def test_same_input_and_key_always_give_the_same_pseudonym():
    a = pseudonym("alice@example.com", salt="email", key=KEY)
    b = pseudonym("alice@example.com", salt="email", key=KEY)
    assert a == b, "joins and group-bys depend on this"


def test_different_keys_give_different_pseudonyms():
    a = pseudonym("alice@example.com", salt="email", key=b"key-one")
    b = pseudonym("alice@example.com", salt="email", key=b"key-two")
    assert a != b


def test_the_same_value_in_two_columns_does_not_collide():
    """Otherwise the output leaks that two fields held the same value."""
    a = pseudonym("alice@example.com", salt="work_email", key=KEY)
    b = pseudonym("alice@example.com", salt="personal_email", key=KEY)
    assert a != b


def test_group_by_survives_pseudonymisation():
    df = pd.DataFrame({"email": ["a@x.com", "a@x.com", "b@x.com"], "spend": [1.0, 2.0, 5.0]})
    report = scan_dataframe(df)
    out = apply_deidentification(df, report, key=KEY)
    assert out.groupby("email")["spend"].sum().tolist() == [3.0, 5.0]


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

def test_tokenisation_keeps_the_recognisable_tail():
    out = tokenise_preserving_tail("4242424242424242", keep=4, salt="card", key=KEY)
    assert out.endswith("4242")
    assert not out.startswith("4242")
    assert len(out) == len("4242424242424242")


def test_short_values_are_fully_replaced_not_left_intact():
    out = tokenise_preserving_tail("42", keep=4, salt="card", key=KEY)
    assert out != "42"


def test_birth_date_becomes_a_band_not_an_age():
    """An exact age plus a postcode is close to an identifier again."""
    assert generalise_date_to_age_band("1985-04-12", reference_year=2026) == "40-49"
    assert generalise_date_to_age_band("not a date", reference_year=2026) is None


def test_postcode_is_truncated_to_a_district():
    assert generalise_postcode("SW1A 1AA", keep=3) == "SW1"


def test_redaction_removes_utility_entirely():
    df = pd.DataFrame({"email": ["a@x.com", "b@x.com"]})
    report = scan_dataframe(df)
    out = apply_deidentification(df, report, overrides={"email": Strategy.REDACT}, key=KEY)
    assert set(out["email"]) == {"__redacted__"}


def test_apply_does_not_mutate_the_original_frame():
    df = pd.DataFrame({"email": ["a@x.com", "b@x.com"]})
    original = df.copy()
    apply_deidentification(df, scan_dataframe(df), key=KEY)
    pd.testing.assert_frame_equal(df, original)


def test_non_pii_columns_are_untouched():
    df = pd.DataFrame({"email": ["a@x.com"], "revenue": [123.45]})
    out = apply_deidentification(df, scan_dataframe(df), key=KEY)
    assert out["revenue"].iloc[0] == 123.45
    assert out["email"].iloc[0] != "a@x.com"


def test_nulls_stay_null_rather_than_becoming_a_pseudonym_of_none():
    df = pd.DataFrame({"email": ["a@x.com", None]})
    out = apply_deidentification(df, scan_dataframe(df), key=KEY)
    assert out["email"].iloc[1] is None


def test_report_records_what_was_applied():
    df = pd.DataFrame({"email": ["a@x.com"], "date_of_birth": ["1985-04-12"]})
    report = scan_dataframe(df)
    apply_deidentification(df, report, key=KEY)
    assert report.applied["email"] == "pseudonymise"
    assert report.applied["date_of_birth"] == "generalise"
    assert report.as_dict()["contains_pii"] is True


def test_contract_contains_no_key_or_source_values():
    df = pd.DataFrame({"customer_email": ["alice@example.com"]})
    contract = deidentification_contract(scan_dataframe(df))

    assert contract == {
        "customer_email": {"kind": "email", "strategy": "pseudonymise"}
    }
    rendered = str(contract)
    assert "alice@example.com" not in rendered
    assert KEY.decode() not in rendered


def test_persisted_contract_repeats_the_training_transform_at_inference():
    expected = pseudonym("alice@example.com", salt="customer_email", key=KEY)
    actual = transform_pii_value(
        "alice@example.com",
        column="customer_email",
        kind="email",
        strategy="pseudonymise",
        key=KEY,
    )

    assert actual == expected


def test_an_operator_can_override_the_recommendation():
    df = pd.DataFrame({"email": ["a@x.com"]})
    report = scan_dataframe(df)
    out = apply_deidentification(df, report, overrides={"email": Strategy.KEEP}, key=KEY)
    assert out["email"].iloc[0] == "a@x.com"


def test_a_date_is_not_mistaken_for_a_phone_number():
    """1985-04-12 is digits and dashes of phone-like length."""
    f = classify_column("some_column", ["1985-04-12", "1990-11-30", "1972-01-05"])
    assert f.kind is not PIIKind.PHONE


def test_real_phone_numbers_still_detected():
    f = classify_column("contact", ["+44 20 7946 0958", "+1 (555) 123-4567",
                                    "020 7946 0321"])
    assert f.kind is PIIKind.PHONE


# --------------------------------------------------------------------------- #
# Redaction must not leak which rows held a value
# --------------------------------------------------------------------------- #

def test_redaction_covers_null_rows_too():
    """A redacted column must be uniformly opaque, nulls included.

    REDACT is the strategy chosen for the most sensitive data, so the column
    must not reveal *which* rows held a value. Missingness is itself
    informative -- "this person had an SSN on file" can correlate with the
    target and survives into the trained model -- so a redacted column that
    preserves nulls hands back a bit of exactly what redaction removed.
    """
    df = pd.DataFrame({"email": ["a@x.com", None, "b@y.com", float("nan")]})
    report = scan_dataframe(df)
    out = apply_deidentification(df, report,
                                 overrides={"email": Strategy.REDACT}, key=KEY)

    distinct = set(out["email"].tolist())
    assert distinct == {"__redacted__"}, (
        f"redacted column still distinguishes rows: {distinct}. A reader can "
        f"recover which rows held a value.")


def test_transform_pii_value_redacts_a_null():
    """The per-value transform is where the null escapes.

    apply_deidentification now routes every strategy through
    transform_pii_value, which early-returns None for null input. That is
    correct for pseudonymisation -- hashing a missing value would invent one --
    but wrong for REDACT, where the point is that every row looks the same.
    """
    assert transform_pii_value(None, column="email", kind=PIIKind.EMAIL,
                               strategy=Strategy.REDACT, key=KEY) == "__redacted__"
    assert transform_pii_value(float("nan"), column="email", kind=PIIKind.EMAIL,
                               strategy=Strategy.REDACT, key=KEY) == "__redacted__"


def test_non_redacting_strategies_still_pass_nulls_through():
    """Guard the fix: only REDACT changes: a missing value must stay missing.

    Pseudonymising a null would mint a stable token for "no value", which both
    invents data and lets absence be joined on.
    """
    for strategy in (Strategy.PSEUDONYMISE, Strategy.TOKENISE, Strategy.GENERALISE):
        assert transform_pii_value(None, column="email", kind=PIIKind.EMAIL,
                                   strategy=strategy, key=KEY) is None, (
            f"{strategy} must leave a null as null")


# --------------------------------------------------------------------------- #
# The key requirement is scoped to strategies that actually need a secret
# --------------------------------------------------------------------------- #

def test_redaction_alone_does_not_require_a_key(monkeypatch):
    """Redaction throws the value away; there is nothing to key.

    Demanding a key for a column that will simply be blanked would train
    operators to set a dummy key, which is worse than not asking.
    """
    monkeypatch.delenv("AVALOKA_PII_KEY", raising=False)
    report = PIIReport()
    report.findings = [f for f in scan_dataframe(
        pd.DataFrame({"email": ["a@x.com"]})).findings]
    for finding in report.findings:
        finding.recommended = Strategy.REDACT
    assert require_pii_key(report) == []


def test_missing_key_names_the_offending_columns(monkeypatch):
    """The error has to be actionable without reading the source.

    An operator sees this at the moment training refuses to start; it must say
    which columns caused it and what to do, not merely that a variable is
    unset.
    """
    monkeypatch.delenv("AVALOKA_PII_KEY", raising=False)
    df = pd.DataFrame({"customer_email": ["a@x.com"], "ssn": ["123-45-6789"]})
    report = scan_dataframe(df)

    with pytest.raises(MissingPIIKey) as excinfo:
        require_pii_key(report)

    message = str(excinfo.value)
    assert "AVALOKA_PII_KEY" in message, "the message must name the variable to set"
    required = pii_key_required_columns(report)
    assert required, "fixture should produce at least one key-requiring column"
    for column in required:
        assert column in message, f"{column!r} missing from the refusal message"


# --------------------------------------------------------------------------- #
# The persisted contract must stay free of secrets and source values
# --------------------------------------------------------------------------- #

def test_contract_excludes_kept_columns():
    """A KEEP column has no transform to repeat, so it must not appear.

    Listing it would imply at inference that something was applied.
    """
    df = pd.DataFrame({"email": ["a@x.com"], "region": ["north"]})
    report = scan_dataframe(df)
    contract = deidentification_contract(report)
    assert "region" not in contract


def test_contract_never_contains_the_key_or_a_source_value():
    df = pd.DataFrame({"customer_email": ["alice@example.com"]})
    report = scan_dataframe(df)
    contract = deidentification_contract(report)

    blob = repr(contract)
    assert "alice@example.com" not in blob, "a source value leaked into the contract"
    assert KEY.decode("utf-8", "ignore") not in blob
    for entry in contract.values():
        assert set(entry) == {"kind", "strategy"}, (
            f"contract entry carries unexpected fields: {sorted(entry)}")


# --------------------------------------------------------------------------- #
# Detection over a bounded sample
# --------------------------------------------------------------------------- #

def test_detection_by_name_survives_an_empty_sample():
    """The MTA guard reindexes a sample that may hold no rows for a column.

    Name evidence is what keeps the guard useful there, so it must not depend
    on seeing values.
    """
    df = pd.DataFrame({"customer_email": pd.Series([], dtype="object")})
    report = scan_dataframe(df)
    assert any(f.column == "customer_email" for f in report.findings)


def test_pii_beyond_the_value_sample_is_not_detected_by_values_alone():
    """Records the LIMIT, so a sampled pass is never read as a guarantee.

    classify_column inspects at most CLASSIFY_SAMPLE_ROWS non-null values. A
    neutrally named column whose PII appears only after that window classifies
    clean, no key is demanded, and training proceeds on unprotected data. This
    test documents that boundary rather than asserting it is acceptable -- if
    detection is ever made exhaustive, it should fail and be deleted.
    """
    from app.agents.pii_agent import CLASSIFY_SAMPLE_ROWS

    values = ["not-an-email"] * CLASSIFY_SAMPLE_ROWS + ["hidden@example.com"]
    finding = classify_column("ref", values)
    assert finding.kind is PIIKind.NONE, (
        "detection now reaches past the sample window -- good; delete this test "
        "and the docstring caveat it guards")
