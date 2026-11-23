"""T16.8: the telemetry guard. Every catalog event is emitted with planted
names in the dataset (``patient_hiv_status``, ``q3_layoffs_final``, a topic
name, a tag, a term, a connection id, a bucket) and none of them appears in
any event, nor in the event stream as a whole. Bite: add ``dataset_name``
or the locator to an event.

The harness is checked against itself: a sink that leaks one planted
string goes red here, so a green run is one in which the scan found nothing,
not one that scanned nothing.
"""

from __future__ import annotations

import json

import pytest

from app.catalog.store import CatalogStore, catalog_event

PLANTED = ["patient_hiv_status", "q3_layoffs_final", "payroll-events-prod", "acme-finance-bucket",
           "conn-acme-redshift", "hiv_clinic_visits.csv", "Zanzibar", "layoffs", "hiv", "acme"]


def _scan(text: str, where: str) -> None:
    for plant in PLANTED:
        assert plant.casefold() not in text.casefold(), f"{plant!r} reached {where}"


@pytest.fixture
def stream(db_url):
    events = []
    store = CatalogStore("user-a", on_event=events.append)
    store.register("file", {"dataset_id": "u1", "filename": "hiv_clinic_visits.csv"},
                   [("patient_hiv_status", "int64"), ("q3_layoffs_final", "object")], dataset_id="u1",
                   display_name="hiv_clinic_visits.csv", tags=["Zanzibar"], analysis_id="an:layoffs",
                   row_count=3, row_count_kind="exact")
    store.add_terms("u1", ["layoffs", "hiv"], column="patient_hiv_status")
    store.register("kafka_window", {"connection_id": "conn-acme-redshift", "topic": "payroll-events-prod",
                                    "window": {"start": "2026-11-01T00:00:00Z", "end": "2026-11-01T03:00:00Z"}},
                   [("patient_hiv_status", "string")], source_version={"offsets": {"0": [10, 20]}})
    store.register("object", {"connection_id": "conn-acme-redshift", "bucket": "acme-finance-bucket",
                              "key": "q3_layoffs_final.parquet"}, [("q3_layoffs_final", "object")],
                   source_version={"etag": "e1"})
    store.register("object", {"connection_id": "conn-acme-redshift", "bucket": "acme-finance-bucket",
                              "key": "q3_layoffs_final.parquet"}, [("q3_layoffs_final", "object")],
                   source_version={"etag": "e2"})
    store.forget("u1")
    return events


def test_the_catalog_itself_does_carry_the_plants(db_url, stream):
    """The local layer is the user's own data and must keep it; otherwise the
    scan below would be passing on an empty catalog."""
    store = CatalogStore("user-a")
    local = json.dumps(store.list(), default=str) + json.dumps([store.get(d["dataset_id"]) for d in store.list()],
                                                              default=str)
    assert "payroll-events-prod" in local and "q3_layoffs_final" in local and "acme-finance-bucket" in local


def test_no_planted_name_reaches_any_event(stream):
    assert len(stream) == 5, [e["event"] for e in stream]
    for event in stream:
        assert set(event) == {"event", "kind", "outcome", "rows", "columns"}
        assert all(isinstance(v, str) for v in event.values())
        _scan(json.dumps(event), event["event"])
    _scan(json.dumps(stream), "the event stream")
    assert [e["outcome"] for e in stream] == ["new_dataset", "new_dataset", "new_dataset", "new_version",
                                              "deleted"]
    kinds = {e["kind"] for e in stream}
    assert kinds == {"file", "kafka_window", "object"}


def test_the_harness_bites_when_a_field_leaks():
    leaked = dict(catalog_event("register", kind="file", outcome="new_dataset"))
    leaked["dataset_name"] = "q3_layoffs_final.csv"
    with pytest.raises(AssertionError, match="q3_layoffs_final"):
        _scan(json.dumps(leaked), "a leaking event")
