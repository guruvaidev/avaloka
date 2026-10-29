"""HMDA-style mortgage columns (sheet rows G33-G36): codes are never measures."""
import pytest

from app.agents import visualization_agent as va


def _rows(name, values):
    return [{name: v, "loan_amount": 100000 + i * 1000} for i, v in enumerate(values)]


@pytest.mark.parametrize("name", ["derived_msa_md", "msa_md", "MSA MD", "derived_msa-md"])
def test_msa_md_codes_are_identifiers(name):
    values = [31080, 41740, 41860, 19100, 26420, 12420, 16980, 35620, 33100, 42660] * 5
    prof = {c["name"]: c for c in va.profile_columns(_rows(name, values))}
    assert prof[name]["role"] == "identifier"


def test_hmda_style_table_charts_no_codes():
    rows = []
    for i in range(300):
        rows.append({
            "activity_year": 2018 + i % 7,
            "census_tract": 6037000000 + i * 137,
            "county_code": 6037 + (i % 3) * 22,
            "derived_msa_md": [31080, 41740, 19100][i % 3],
            "loan_amount": 50000 + (i * 7919) % 900000 if i % 50 else 6_000_000,
            "interest_rate": 3 + (i % 40) / 10,
        })
    cols = va.profile_columns(rows)
    prof = {"dataset": {"rows_sampled": len(rows)}, "columns": cols,
            "feature_ranking": va.compute_unsupervised_feature_ranking(cols)}
    charted = set()
    for ch in va.select_charts(prof, rows):
        for enc in ch["encodings"].values():
            if isinstance(enc, dict) and enc.get("field"):
                charted.add(enc["field"])
    assert not charted & {"census_tract", "county_code", "derived_msa_md"}
    by_name = {c["name"]: c for c in cols}
    assert by_name["activity_year"].get("is_year") is True