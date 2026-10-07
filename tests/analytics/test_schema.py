"""The schema is a permit-list. These tests check that it refuses, not that it accepts."""
import json
from pathlib import Path

import pytest

from app.analytics import schema
from tests.analytics.conftest import ev

ARTIFACT = Path(__file__).resolve().parents[2] / "docs" / "analytics" / "event-schema.v1.json"


def _minimal(event_type):
    required, _ = schema.EVENTS[event_type]
    sample = {"enum": None, "int": 1, "bool": True, "id": "a" * 16, "class": "ValueError", "text": "q"}
    fields = {}
    for name, spec in required.items():
        fields[name] = sorted(spec.values)[0] if spec.kind == "enum" else sample[spec.kind]
    return ev(event_type, **fields)


def test_published_artifact_matches_the_code():
    assert json.loads(ARTIFACT.read_text("utf-8")) == json.loads(json.dumps(schema.describe())), (
        "docs/analytics/event-schema.v1.json is stale; run `python -m app.analytics.schema`"
    )


@pytest.mark.parametrize("event_type", sorted(schema.EVENT_TYPES))
def test_every_event_type_has_a_valid_minimal_form(event_type):
    assert schema.validate_event(_minimal(event_type))["event_type"] == event_type


@pytest.mark.parametrize("event_type", sorted(schema.EVENT_TYPES))
@pytest.mark.parametrize("forbidden", sorted(schema.NEVER_ACCEPTED))
def test_named_temptations_are_refused_on_every_event_type(event_type, forbidden):
    event = _minimal(event_type)
    event[forbidden] = "x"
    with pytest.raises(schema.Rejected):
        schema.validate_event(event)


def test_prompt_is_the_only_free_text_field_in_the_whole_schema():
    text_fields = {
        (etype, name)
        for etype, (req, opt) in schema.EVENTS.items()
        for name, spec in {**req, **opt}.items() if spec.kind == "text"
    }
    assert text_fields == {("question.submitted", "prompt")}
    assert all(spec.kind != "text" for spec in {**schema.COMMON_REQUIRED, **schema.COMMON_OPTIONAL}.values())


def test_prompt_is_refused_on_any_other_event():
    event = _minimal("question.completed")
    event["prompt"] = "what is churn"
    with pytest.raises(schema.Rejected):
        schema.validate_event(event)


@pytest.mark.parametrize("route", ["/shared/8fa2c1d09e", "/reports/42", "/analysis?file=q3_layoffs.csv",
                                   "https://acme.example/x", ""])
def test_concrete_paths_are_not_routes(route):
    with pytest.raises(schema.Rejected):
        schema.validate_event(ev("page.viewed", route=route))


@pytest.mark.parametrize("bad", ["KeyError: 'patient_hiv_status'", "boom at /data/q3.csv", "a b", "x" * 80, 7])
def test_error_class_admits_a_class_name_and_never_a_message(bad):
    with pytest.raises(schema.Rejected):
        schema.validate_event(ev("error.shown", surface="analysis", error_class=bad))


def test_missing_required_field_is_refused():
    with pytest.raises(schema.Rejected):
        schema.validate_event(ev("question.submitted", input_method="typed"))


def test_out_of_vocabulary_value_is_refused_not_coerced():
    with pytest.raises(schema.Rejected):
        schema.validate_event(ev("question.completed", turn_id="7" * 16, outcome="kinda_worked"))


def test_a_flag_is_not_a_measurement():
    with pytest.raises(schema.Rejected):
        schema.validate_event(ev("question.completed", turn_id="7" * 16, outcome="success", duration_ms=True))


def test_rejection_reason_never_echoes_the_rejected_value():
    secret = "ravi.kumar@acme.example"
    with pytest.raises(schema.Rejected) as info:
        schema.validate_event(ev("page.viewed", route=secret))
    assert secret not in info.value.reason


@pytest.mark.parametrize("version,ok", [("1.0", True), ("1.7", True), ("2.0", False), ("0.9", False),
                                        ("1", False), ("1.0.0", False), (1.0, False), (None, False)])
def test_version_rule_same_major_accepted_other_major_refused(version, ok):
    if ok:
        assert schema.parse_version(version)[0] == 1
    else:
        with pytest.raises(schema.Rejected):
            schema.parse_version(version)


def test_envelope_refuses_unknown_keys_and_identity():
    good = {"schema_version": "1.0", "session_id": "5" * 16, "events": []}
    assert schema.validate_envelope(good)[1] == "5" * 16
    for extra in ("user_id", "email", "install_id"):
        with pytest.raises(schema.Rejected):
            schema.validate_envelope({**good, extra: "x"})
    with pytest.raises(schema.Rejected):
        schema.validate_envelope({**good, "session_id": "ravi kumar <ravi@acme.example>"})


@pytest.mark.parametrize("version,ok", [("1.6.2", True), ("1.7.0-rc1", True), ("1.6.2+3f9a1c0", True),
                                        ("ravi_kumar", False), ("v1", False), ("1.6 beta for acme", False)])
def test_ui_version_is_a_version_not_a_text_slot(version, ok):
    body = {"schema_version": "1.0", "session_id": "5" * 16, "ui_version": version, "events": []}
    if ok:
        assert schema.validate_envelope(body)[2] == version
    else:
        with pytest.raises(schema.Rejected):
            schema.validate_envelope(body)


@pytest.mark.parametrize("bad", ["ravi_kumar_acme", "ravi.kumar@acme.example", "patient_hiv_status",
                                 "short", "z" * 16, "a" * 65])
def test_an_id_slot_cannot_carry_words(bad):
    with pytest.raises(schema.Rejected):
        schema.validate_event(ev("question.submitted", turn_id=bad, input_method="typed"))


def test_uuid_in_both_forms_is_an_id():
    import uuid
    u = uuid.uuid4()
    for form in (u.hex, str(u)):
        assert schema.validate_event(ev("question.submitted", turn_id=form, input_method="typed"))
