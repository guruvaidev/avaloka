# Copyright 2026 Avaloka.ai Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CSV samples include every data row and tolerate small datasets."""

import pytest

from file_handler.csv_connector import CSVConnector


@pytest.mark.parametrize("random_rows", [True, False])
def test_sampling_all_rows_includes_the_last_row(tmp_path, random_rows):
    path = tmp_path / "data.csv"
    path.write_text("value\n10\n20\n30\n")

    sample = CSVConnector(str(path)).sample_rows(3, random_rows=random_rows)

    assert sample == [[10], [20], [30]]


@pytest.mark.parametrize("random_rows", [True, False])
@pytest.mark.parametrize("values", [[], [10], [10, 20, 30]])
def test_default_sample_returns_all_available_rows(tmp_path, random_rows, values):
    path = tmp_path / "small.csv"
    CSVConnector.write_data(
        {"columns": ["value"], "rows": [[value] for value in values]}, str(path)
    )

    sample = CSVConnector(str(path)).sample_rows(random_rows=random_rows)

    assert sample == [[value] for value in values]


@pytest.mark.parametrize("random_rows", [True, False])
def test_empty_file_has_no_rows_to_sample(tmp_path, random_rows):
    path = tmp_path / "empty.csv"
    path.write_text("")
    connector = CSVConnector(str(path))

    assert connector.get_row_count() == 0
    assert connector.sample_rows(random_rows=random_rows) == []


@pytest.mark.parametrize("random_rows", [True, False])
def test_zero_requested_rows_returns_an_empty_sample(tmp_path, random_rows):
    path = tmp_path / "data.csv"
    path.write_text("value\n10\n20\n30\n")

    assert CSVConnector(str(path)).sample_rows(0, random_rows=random_rows) == []


def test_partial_random_sample_has_unique_rows_in_source_order(tmp_path):
    path = tmp_path / "data.csv"
    rows = [[value] for value in range(20)]
    CSVConnector.write_data({"columns": ["value"], "rows": rows}, str(path))

    sample = CSVConnector(str(path)).sample_rows(5)

    assert len(sample) == 5
    assert len({row[0] for row in sample}) == 5
    assert all(row in rows for row in sample)
    assert sample == sorted(sample)


def test_sampling_counts_csv_records_not_physical_lines(tmp_path):
    path = tmp_path / "multiline.csv"
    rows = [[1, "first\nsecond"], [2, "last"]]
    CSVConnector.write_data({"columns": ["id", "note"], "rows": rows}, str(path))
    connector = CSVConnector(str(path))

    assert connector.get_row_count() == 2
    assert connector.get_row_count() == 2
    assert connector.sample_rows(2) == rows
