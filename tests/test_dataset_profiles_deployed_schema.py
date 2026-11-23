"""
Every column upsert_profile writes exists in the deployed dataset_profiles table.

PostgREST rejects an upsert whose row names a column the table does not have
(PGRST204, "Could not find the '<col>' column ... in the schema cache"), and it
rejects the whole row, not just that key. upsert_profile logs and swallows the
error, so a single misnamed key means the profile silently never persists.

upsert_profile wrote quick_elapsed_s / full_elapsed_s, the names in the
module's own MIGRATION_SQL. The schema Supabase is actually deployed from
(the helm chart's supabase-migrations/0-schema.sql, copied to
ui/supabase/migrations/0-schema.sql) names them phase_a_elapsed_s /
phase_b_elapsed_s.

The fake client below enforces the deployed column list, parsed from that file,
the same way PostgREST does. No database or network is needed.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents import sampling_persistence as sp

DEPLOYED_SCHEMAS = [
    ROOT / "deploy/helm/avaloka/files/supabase-migrations/0-schema.sql",
    ROOT / "ui/supabase/migrations/0-schema.sql",
]


def _columns_from_create(sql: str, table: str, quoted: bool) -> set:
    """Column names in the CREATE TABLE block for `table`."""
    name = rf'"public"\."{table}"' if quoted else rf"\b{table}\b"
    m = re.search(rf"CREATE TABLE IF NOT EXISTS {name}\s*\((.*?)\n\);", sql, re.S)
    assert m, f"no CREATE TABLE for {table}"
    cols = set()
    for line in m.group(1).splitlines():
        line = line.split("--")[0].strip()
        cm = re.match(r'"?([a-z_][a-z0-9_]*)"?\s+\S', line)
        if cm and cm.group(1).upper() not in {"CONSTRAINT", "PRIMARY", "UNIQUE"}:
            cols.add(cm.group(1))
    return cols


def _deployed_columns(table: str = "dataset_profiles") -> set:
    """Columns of `table` in the deployed schema, plus any later ADD COLUMN."""
    cols = None
    for path in DEPLOYED_SCHEMAS:
        sql = path.read_text()
        found = _columns_from_create(sql, table, quoted=True)
        found |= set(re.findall(
            rf'ALTER TABLE (?:ONLY )?"public"\."{table}"\s+ADD COLUMN (?:IF NOT EXISTS )?"?(\w+)"?', sql))
        assert cols is None or found == cols, f"{path} disagrees with {DEPLOYED_SCHEMAS[0]}"
        cols = found
    return cols


class PostgrestUnknownColumn(Exception):
    """What postgrest-py raises (as APIError) for a row naming a missing column."""


class _FakeTable:
    def __init__(self, client, name):
        self.client, self.name, self._row = client, name, None

    def upsert(self, row, on_conflict=None):
        self._row = dict(row)
        return self

    def execute(self):
        unknown = sorted(set(self._row) - self.client.columns[self.name])
        if unknown:
            self.client.rejected.append((self.name, unknown))
            raise PostgrestUnknownColumn(
                f"PGRST204: Could not find the '{unknown[0]}' column of '{self.name}' in the schema cache")
        self.client.written.append((self.name, self._row))
        return type("Resp", (), {"data": [self._row]})()


class FakeSupabase:
    def __init__(self, columns):
        self.columns, self.rejected, self.written = columns, [], []

    def table(self, name):
        return _FakeTable(self, name)


@pytest.fixture
def fake(monkeypatch):
    client = FakeSupabase({sp.TABLE_PROFILES: _deployed_columns()})
    monkeypatch.setattr(sp, "get_supabase_client", lambda: client)
    return client


# Every keyword upsert_profile accepts, set to a non-None value so every
# optional key reaches the row.
ALL_KWARGS = dict(
    user_id="u1", source_path="s3://b/k.csv", source_type="csv", tier="SMALL",
    estimated_rows=10, exact_rows=10, file_size_mb=1.5, sample_pct=100.0,
    phase_a_elapsed_s=0.4, phase_b_elapsed_s=2.5, column_count=3,
    completeness_pct=99.0, profiling_result={"k": 1}, error="none",
)


def test_deployed_schema_parses():
    cols = _deployed_columns()
    assert {"dataset_id", "phase", "phase_a_elapsed_s", "phase_b_elapsed_s"} <= cols


def test_fake_rejects_unknown_column(fake):
    """The fake has teeth: an unknown key fails the whole upsert, like PostgREST."""
    with pytest.raises(PostgrestUnknownColumn):
        fake.table(sp.TABLE_PROFILES).upsert({"dataset_id": "d", "not_a_column": 1}).execute()


def test_kwargs_cover_upsert_profile_signature():
    import inspect
    params = set(inspect.signature(sp.upsert_profile).parameters) - {"dataset_id", "phase"}
    assert params == set(ALL_KWARGS), "update ALL_KWARGS to cover every upsert_profile keyword"


def test_upsert_profile_writes_only_deployed_columns(fake):
    result = sp.upsert_profile("ds-1", "full_profile", **ALL_KWARGS)
    assert fake.rejected == [], f"PostgREST would reject the upsert; unknown columns: {fake.rejected}"
    assert result is not None
    row = fake.written[0][1]
    assert row["phase_a_elapsed_s"] == 0.4 and row["phase_b_elapsed_s"] == 2.5


def test_persist_full_profile_with_elapsed_times_persists(fake, monkeypatch):
    """The caller path that forwards both elapsed times reaches the table."""
    monkeypatch.setattr(sp, "_PERSIST_WORKERS", 1)
    sp.persist_full_profile(
        "ds-2", {"profiling_result": {}, "schema": {"a": "int"}, "rows": []},
        phase_a_elapsed_s=0.4, phase_b_elapsed_s=2.5,
    )
    assert fake.rejected == [], f"PostgREST would reject the upsert; unknown columns: {fake.rejected}"
    assert [t for t, _ in fake.written] == [sp.TABLE_PROFILES]


def test_persist_quick_profile_with_elapsed_time_persists(fake):
    sp.persist_quick_profile("ds-3", {"profiling_result": {}, "schema": {}}, phase_a_elapsed_s=0.4)
    assert fake.rejected == [], f"PostgREST would reject the upsert; unknown columns: {fake.rejected}"
    assert len(fake.written) == 1


def test_in_file_migration_matches_deployed_columns():
    """MIGRATION_SQL must not create a dataset_profiles column the deployed table lacks."""
    in_file = _columns_from_create(sp.MIGRATION_SQL, "dataset_profiles", quoted=False)
    in_file |= set(re.findall(r"ALTER TABLE dataset_profiles ADD COLUMN IF NOT EXISTS (\w+)", sp.MIGRATION_SQL))
    assert in_file - _deployed_columns() == set(), "MIGRATION_SQL names columns the deployed schema lacks"
