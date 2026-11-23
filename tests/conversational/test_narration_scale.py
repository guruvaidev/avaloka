"""A digit inside a word is not a sense of scale.

The narration probe credits a reply that says roughly how long the work will
take, and counted ANY digit as that. The reply below -- the product's own, from
the recording, for a long job asked with no dataset connected -- gives no
duration at all; it passed on the "3" in "s3://". Only a quantity counts now.
"""
from tests.conversational.probes import BRILLIANT, PROBES, eval_narrates_before_long_work

PROBE = next(p for p in PROBES if p.id == "resp_long_job_narrated")

NO_DATA_REPLY = "There is no dataset connected to this conversation yet, so I haven't run anything. To do this I need the data this request is about.\n\nYou can provide it by uploading a file (CSV, Parquet, Excel or JSON), by giving a bucket path such as s3://\u2026 or gs://\u2026, or by connecting a database. As soon as it is connected I'll say what I'm about to run and roughly how long it will take, then run it."


def test_a_bucket_scheme_is_not_a_duration():
    """Fails on the base: gave_a_sense_of_scale was 1.0 on "s3://"."""
    metrics = eval_narrates_before_long_work(NO_DATA_REPLY, PROBE.grading_context)
    assert metrics["gave_a_sense_of_scale"] == 0.0 and metrics["pass"] == 0.0, metrics


def test_a_quantity_still_is():
    for text in (BRILLIANT[PROBE.id], "Running it now, about 2 minutes.",
                 "Starting the scan; expect 90s.", "I'll run it across 32,000,000 rows."):
        assert eval_narrates_before_long_work(text, PROBE.grading_context)["gave_a_sense_of_scale"] == 1.0, text
