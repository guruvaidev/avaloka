import pytest

from unittest.mock import patch

from app.agents.mta_v2.loader.url import _load_pandas, load_dataset_from_url


def test_csv_loader_retries_legacy_encoding_without_losing_characters(tmp_path):
    dataset = tmp_path / "legacy.csv"
    dataset.write_bytes("name,city\nAlice,Frýdek\n".encode("cp1252"))

    result = _load_pandas(str(dataset), "csv")

    assert result.to_dict("records") == [{"name": "Alice", "city": "Frýdek"}]


def test_csv_loader_does_not_override_explicit_encoding(monkeypatch, tmp_path):
    dataset = tmp_path / "legacy.csv"
    dataset.write_bytes("name,city\nAlice,Frýdek\n".encode("cp1252"))
    monkeypatch.setenv("URL_READ_OPTIONS_CSV", '{"encoding": "utf-8"}')

    with pytest.raises(UnicodeDecodeError):
        _load_pandas(str(dataset), "csv")


def test_explicit_local_load_ignores_initialized_global_ray(tmp_path):
    dataset = tmp_path / "small.csv"
    dataset.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")

    with (
        patch("app.agents.mta_v2.loader.url.ray.is_initialized", return_value=True),
        patch("app.agents.mta_v2.loader.url._load_ray") as load_ray,
    ):
        result = load_dataset_from_url(str(dataset), use_ray=False)

    load_ray.assert_not_called()
    assert result.to_dict("records") == [
        {"feature": 1, "target": 0},
        {"feature": 2, "target": 1},
    ]


def test_explicit_ray_load_requires_an_initialized_driver(tmp_path):
    dataset = tmp_path / "small.csv"
    dataset.write_text("feature,target\n1,0\n", encoding="utf-8")

    with patch("app.agents.mta_v2.loader.url.ray.is_initialized", return_value=False):
        with pytest.raises(RuntimeError, match="Ray is not initialized"):
            load_dataset_from_url(str(dataset), use_ray=True)
