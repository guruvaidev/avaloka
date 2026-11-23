import json
from pathlib import Path

from benchmarks.external.dab.adapter import _answer_text
from benchmarks.external.dab.submission import export
from benchmarks.run import build_parser

def test_answer_text_prefers_explicit_answer():
    assert _answer_text({"answer": "2020s", "last_message": "ignored"}) == "2020s"

def test_submission_export_rejects_partial_record(tmp_path: Path):
    record = {"suite": "dab", "results": {"cfg": {"tasks": {
        "dab:bookreview:1": {"trials": [{"trial_index": 0,
            "outcome": {"answer": "2020s", "trajectory": []}}]}}}}}
    source = tmp_path / "record.json"
    source.write_text(json.dumps(record))
    try:
        export(source, tmp_path / "submission.json", tmp_path / "traces")
    except ValueError as exc:
        assert "270 unique trials" in str(exc)
    else:
        raise AssertionError("partial DAB record must not be exportable")

def test_runner_accepts_dataset_and_task_filters():
    args = build_parser().parse_args(["run", "dab", "--dataset", "bookreview",
                                      "--task", "dab:bookreview:1"])
    assert args.dataset == ["bookreview"]
    assert args.task == ["dab:bookreview:1"]
