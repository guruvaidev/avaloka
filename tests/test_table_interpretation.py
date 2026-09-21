"""What a table is, and saying it back before acting on it."""
import pandas as pd
import pytest

from file_handler.interpret import (apply_suggested_names, confirm_understanding,
                                    interpret, render_for_model)

DF = pd.DataFrame({
    "Role": ["Engineer", "Analyst"],
    "Rate": [120.0, 90.0],
    "Region": ["US", "EU"],
})


class TestHeuristicFallback:
    """No model configured is the normal case in OSS installs; it must still work."""

    def test_types_columns_without_a_model(self, monkeypatch):
        monkeypatch.setenv("AVALOKA_DISABLE_TABLE_INTERPRETER", "1")
        out = interpret(DF)
        assert out["source"] == "heuristic"
        kinds = {c["name"]: c["semantic_type"] for c in out["columns"]}
        assert kinds["Rate"] == "quantity"

    def test_does_not_guess_a_domain_without_a_model(self, monkeypatch):
        """Claiming a sheet is a balance sheet on no evidence is worse than
        saying nothing: everything downstream treats the guess as established."""
        monkeypatch.setenv("AVALOKA_DISABLE_TABLE_INTERPRETER", "1")
        assert interpret(DF)["kind"] == "other"

    def test_empty_frame_is_safe(self, monkeypatch):
        monkeypatch.setenv("AVALOKA_DISABLE_TABLE_INTERPRETER", "1")
        assert interpret(pd.DataFrame())["columns"] == []


class TestRendering:
    def test_shows_columns_and_a_sample(self):
        text = render_for_model(DF)
        assert "Role | Rate | Region" in text
        assert "Engineer" in text

    def test_truncates_long_tables(self):
        big = pd.DataFrame({"a": range(100)})
        assert "more rows" in render_for_model(big)


class TestSuggestedNames:
    def test_renames_only_placeholders(self):
        df = pd.DataFrame({"column_1": [1], "Role": ["Eng"]})
        out = apply_suggested_names(df, {"columns": [
            {"name": "column_1", "suggested_name": "Service Name"},
            {"name": "Role", "suggested_name": "job_title"},
        ]})
        assert list(out.columns) == ["service_name", "Role"], (
            "a name the sheet gave must survive: it is the word the user will "
            "use when they ask about that column"
        )

    def test_no_suggestions_is_a_no_op(self):
        df = pd.DataFrame({"a": [1]})
        assert list(apply_suggested_names(df, {"columns": []}).columns) == ["a"]


class TestConfirmUnderstanding:
    def test_names_the_user_the_domain_and_the_goal(self):
        line = confirm_understanding(
            {"kind": "rate_card", "title": "Rate Card", "description": None},
            user="Leela", goal="the blended cost per video hour",
            dataset="pricing.xlsx")
        assert "So Leela," in line
        assert "rate card" in line
        assert "blended cost per video hour" in line
        assert line.rstrip().endswith("?"), "it has to actually ask"

    def test_falls_back_when_the_domain_is_unknown(self):
        line = confirm_understanding({"kind": "other"}, dataset="notes.csv")
        assert "this dataset" in line and "notes.csv" in line
        assert "?" in line

    def test_works_without_a_user_name(self):
        line = confirm_understanding({"kind": "budget"}, goal="next year's spend")
        assert line.startswith("So you want me to read")

    @pytest.mark.parametrize("kind,phrase", [
        ("balance_sheet", "balance sheet"), ("tax_form", "tax form"),
        ("project_plan", "project plan"), ("cost_estimate", "cost estimate"),
    ])
    def test_speaks_each_domain_in_plain_words(self, kind, phrase):
        assert phrase in confirm_understanding({"kind": kind})
