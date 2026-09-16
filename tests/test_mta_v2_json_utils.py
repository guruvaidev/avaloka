import importlib

import pandas as pd
import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "app.agents.mta_v2.utils",
        "app.agents.mta_v2.inference_service_image.utils",
    ],
)
def test_json_array_columns_and_limited_load(module_name, tmp_path):
    utils = importlib.import_module(module_name)
    path = tmp_path / "records.json"
    pd.DataFrame(
        [
            {"alpha": 1, "beta": "x"},
            {"alpha": 2, "beta": "y"},
        ]
    ).to_json(path, orient="records")

    assert utils.detect_format_from_url(str(path)) == "json"
    assert utils.get_columns(str(path)) == ["alpha", "beta"]

    loaded = utils.load_dataset(str(path), limit=1)
    assert loaded.to_dict(orient="records") == [{"alpha": 1, "beta": "x"}]


@pytest.mark.parametrize(
    "module_name",
    [
        "app.agents.mta_v2.utils",
        "app.agents.mta_v2.inference_service_image.utils",
    ],
)
def test_jsonl_columns_and_limited_load(module_name, tmp_path):
    utils = importlib.import_module(module_name)
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"alpha": 1, "beta": "x"}\n{"alpha": 2, "beta": "y"}\n',
        encoding="utf-8",
    )

    assert utils.detect_format_from_url(str(path)) == "jsonl"
    assert utils.get_columns(str(path)) == ["alpha", "beta"]

    loaded = utils.load_dataset(str(path), limit=1)
    assert loaded.to_dict(orient="records") == [{"alpha": 1, "beta": "x"}]


@pytest.mark.parametrize(
    "module_name",
    [
        "app.agents.mta_v2.utils",
        "app.agents.mta_v2.inference_service_image.utils",
    ],
)
def test_ndjson_detects_jsonl(module_name, tmp_path):
    utils = importlib.import_module(module_name)
    path = tmp_path / "records.ndjson"
    path.write_text('{"alpha": 1}\n', encoding="utf-8")

    assert utils.detect_format_from_url(str(path)) == "jsonl"
