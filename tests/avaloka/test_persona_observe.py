"""Avaloka's first-glance observations must be about signal, not identifiers.

Reported on integration/2.0: ``avaloka analyze`` printed "InvoiceNo leans hard
on 537434 (3% of rows) — that's a segment worth naming" because ``observe``
took the first categorical column with any top value and never asked whether
the share was a lean or the column an identifier, and "a pull between
encounter_id and patient_nbr" for two id columns that happen to correlate.
"""

from __future__ import annotations

from avaloka.persona import (ID_LIKE_UNIQUE_PCT, SEGMENT_MAX_CARDINALITY, SEGMENT_MIN_LEAD,
                             SEGMENT_MIN_SHARE, Avaloka)


def _profile(columns, correlations=(), id_columns=(), n_rows=1000):
    return {
        "dataset": {"n_rows": n_rows, "n_cols": len(columns)},
        "quality": {"score": 90, "high_missing_columns": [], "id_columns": list(id_columns)},
        "columns": columns,
        "correlations": list(correlations),
    }


def _cat(name, top_count, n_unique, n_rows=1000, role="categorical", second=None):
    """``second`` is the runner-up's count; by default a clear minority."""
    second = top_count // 4 if second is None else second
    return {"name": name, "role": role, "n_unique": n_unique, "unique_pct": n_unique / n_rows,
            "top_values": [{"value": f"{name}-top", "count": top_count},
                           {"value": f"{name}-second", "count": second}]}


def _lines(profile):
    return "\n".join(Avaloka(llm=False).observe(profile))


def _plain(text):
    return text.replace("[cyan]", "").replace("[/cyan]", "")


def _observe_frame(frame):
    """Through the real profiler, exactly as the data scout builds the profile."""
    import pandas as pd  # noqa: F401 - the fixtures are frames

    from avaloka.stats import correlation_pairs, profile_column, quality_report

    n_rows = len(frame)
    columns = [profile_column(frame[c], n_rows) for c in frame.columns]
    profile = {"dataset": {"n_rows": n_rows, "n_cols": len(frame.columns)},
               "columns": columns, "quality": quality_report(frame, columns),
               "correlations": correlation_pairs(frame)}
    return _plain("\n".join(Avaloka(llm=False).observe(profile)))


def test_when_nothing_leans_she_says_so_instead_of_inventing_a_segment():
    text = _lines(_profile([_cat("Gender", 27_774, 2, n_rows=55_500, second=27_726),
                            _cat("Blood Type", 6_969, 8, n_rows=55_500, second=6_956)], n_rows=55_500))
    assert "leans hard" not in text
    assert "Nothing dominates a column — the closest is Gender at 50%" in _plain(text)
    # Identifier-only categoricals are not "the closest" to anything.
    assert "Nothing dominates" not in _lines(_profile([_cat("InvoiceNo", 1200, 24_000, n_rows=40_000)],
                                                      n_rows=40_000))


def test_an_invoice_number_at_three_percent_is_not_a_segment():
    # 40,000 rows, 24,000 distinct invoice numbers, the commonest on 3% of rows.
    profile = _profile([_cat("InvoiceNo", 1200, 24_000, n_rows=40_000),
                        _cat("Description", 900, 3_900, n_rows=40_000)], n_rows=40_000)
    text = _lines(profile)
    assert "leans hard" not in text
    assert "InvoiceNo" not in text


def test_a_real_lean_is_named_and_the_strongest_column_wins_not_the_first():
    profile = _profile([_cat("region", 260, 4),          # 26%: four regions, no lean
                        _cat("contract", 560, 3)])        # 56%: month-to-month dominates
    text = _lines(profile)
    assert "contract" in text and "leans hard on “contract-top” (56% of rows)" in text
    assert "region leans hard" not in text


def test_the_share_floor_is_the_published_constant():
    just_under = _cat("plan", int(SEGMENT_MIN_SHARE * 1000) - 1, 5)
    just_over = _cat("plan", int(SEGMENT_MIN_SHARE * 1000) + 1, 5)
    assert "leans hard" not in _lines(_profile([just_under]))
    assert "leans hard" in _lines(_profile([just_over]))


def test_a_high_cardinality_title_column_is_an_identifier_even_if_its_top_value_leans():
    # 60% distinct values but one title repeated on 40% of rows: still not a segment.
    title = _cat("Name", 400, int(ID_LIKE_UNIQUE_PCT * 1000) + 100)
    assert "leans hard" not in _lines(_profile([title]))


def test_more_distinct_values_than_sqrt_rows_is_a_label_not_a_segment():
    # 40,000 rows: the bound is max(50, 200) = 200 distinct values.
    bound = max(SEGMENT_MAX_CARDINALITY, int(40_000 ** 0.5))
    label = _cat("InvoiceNo", 16_000, bound + 1, n_rows=40_000)   # 40% on one value, 201 distinct
    segment = _cat("Country", 16_000, bound, n_rows=40_000)       # 40% on one value, 200 distinct
    assert "leans hard" not in _lines(_profile([label], n_rows=40_000))
    assert "Country leans hard" in _plain(_lines(_profile([segment], n_rows=40_000)))


def test_a_correlation_between_two_identifiers_is_not_a_relationship():
    columns = [{"name": "encounter_id", "role": "id", "unique_pct": 1.0},
               {"name": "patient_nbr", "role": "numeric", "unique_pct": 0.71},
               {"name": "num_medications", "role": "numeric", "unique_pct": 0.05},
               {"name": "time_in_hospital", "role": "numeric", "unique_pct": 0.01}]
    pairs = [{"a": "encounter_id", "b": "patient_nbr", "corr": 0.512},
             {"a": "num_medications", "b": "time_in_hospital", "corr": 0.47}]
    text = _lines(_profile(columns, pairs, id_columns=["encounter_id"]))
    assert "encounter_id" not in text.split("treating")[0]  # not in the "pull" line
    assert "move together" not in text                      # 0.47 is below the bar
    pairs[1]["corr"] = 0.61
    text = _lines(_profile(columns, pairs, id_columns=["encounter_id"]))
    assert "pull between num_medications and time_in_hospital" in text.replace("[cyan]", "").replace("[/cyan]", "")


def test_a_fifty_fifty_split_is_not_a_lean():
    # healthcare_dataset.csv: Gender is 27,774 Male to 27,726 Female.
    gender = _cat("Gender", 27_774, 2, n_rows=55_500, second=27_726)
    assert "leans hard" not in _lines(_profile([gender], n_rows=55_500))
    # The lead factor is the published constant: just under it, no lean; at it, a lean.
    top = 600
    under = _cat("plan", top, 3, second=int(top / SEGMENT_MIN_LEAD) + 1)
    at = _cat("plan", top, 3, second=int(top / SEGMENT_MIN_LEAD))
    assert "leans hard" not in _lines(_profile([under]))
    assert "leans hard" in _lines(_profile([at]))


# --------------------------------------------------------------------------- #
# The three shapes release-test hit, built as frames and run through the real
# profiler, plus the shipped example keeping the insight it really has.
# --------------------------------------------------------------------------- #

def _rng():
    import numpy as np
    return np.random.default_rng(7)


def test_fixture_invoice_ids_like_the_ecommerce_workbook():
    import numpy as np
    import pandas as pd

    rng = _rng()
    n = 40_000
    # ~24k invoice numbers; the commonest one on ~3% of rows (a big basket).
    invoices = np.concatenate([np.full(1_200, "537434"),
                               rng.integers(536_000, 560_000, n - 1_200).astype(str)])
    rng.shuffle(invoices)
    frame = pd.DataFrame({"InvoiceNo": invoices,
                          "Description": rng.choice([f"ITEM {i}" for i in range(3_900)], n),
                          "Quantity": rng.integers(1, 50, n),
                          "UnitPrice": rng.random(n) * 20})
    text = _observe_frame(frame)
    assert "leans hard" not in text and "537434" not in text


def test_fixture_recipe_titles_like_the_german_recipes_json():
    import numpy as np
    import pandas as pd

    rng = _rng()
    n = 12_000
    titles = np.concatenate([np.full(360, "Herzhafte Pfannkuchen"),
                             np.array([f"Rezept {i}" for i in rng.integers(0, 9_000, n - 360)])])
    rng.shuffle(titles)
    frame = pd.DataFrame({"Name": titles,
                          "Day": rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], n),
                          "Prep": rng.integers(5, 120, n)})
    text = _observe_frame(frame)
    assert "Herzhafte Pfannkuchen" not in text and "Name leans" not in text


def test_fixture_two_id_columns_like_diabetes130():
    import numpy as np
    import pandas as pd

    rng = _rng()
    n = 20_000
    patient = rng.integers(1_000_000, 90_000_000, n)
    encounter = np.sort(patient) * 3 + rng.integers(0, 1000, n)   # ids that correlate
    frame = pd.DataFrame({"encounter_id": encounter, "patient_nbr": np.sort(patient),
                          "time_in_hospital": rng.integers(1, 14, n),
                          "num_medications": rng.integers(1, 80, n)})
    text = _observe_frame(frame)
    assert "pull between encounter_id and patient_nbr" not in text
    assert "move together" not in text  # time_in_hospital and num_medications are independent here


def test_the_shipped_churn_example_keeps_its_real_insight():
    from pathlib import Path

    import pandas as pd

    churn = Path(__file__).resolve().parents[2] / "avaloka" / "examples" / "churn.csv"
    text = _observe_frame(pd.read_csv(churn))
    assert "contract leans hard on “month-to-month”" in text


def test_a_twenty_percent_top_value_with_a_clear_lead_is_still_not_a_segment():
    """The share floor, with literal numbers so that a floor of 0 cannot pass.

    One categorical column, 4 distinct values, top 200 vs runner-up 50 of
    1,000 rows: a 4x lead, low cardinality, 20% share. Only the floor says no.
    """
    plan = {"name": "plan", "role": "categorical", "n_unique": 4, "unique_pct": 0.004,
            "top_values": [{"value": "basic", "count": 200}, {"value": "plus", "count": 50},
                           {"value": "pro", "count": 40}, {"value": "team", "count": 10}]}
    text = _plain(_lines(_profile([plan], n_rows=1000)))
    assert "leans hard" not in text
    assert "Nothing dominates a column — the closest is plan at 20% “basic”" in text


def test_a_constant_column_is_not_a_segment():
    """diabetic_data.csv printed "acetohexamide leans hard on 'No' (100% of rows)"."""
    constant = {"name": "acetohexamide", "role": "categorical", "n_unique": 1, "unique_pct": 1 / 101_766,
                "top_values": [{"value": "No", "count": 101_766}]}
    text = _plain(_lines(_profile([constant], n_rows=101_766)))
    assert "leans hard" not in text
    assert "acetohexamide" not in text
