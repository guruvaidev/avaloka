"""The O(N^2) loop guard rejects the two quadratic shapes and nothing else.

A static complexity rule that rejects legitimate code costs more than the
timeouts it prevents: the coder is sent back with a false reason and the user
waits for a rewrite of code that was fine. So every row here is a shape the
coder legitimately writes on a full-dataset run (a loop over a few groups that
filters ``df`` once per group, ``iterrows`` over a small aggregate, a filter
outside any loop) and must pass, and the two quadratic shapes must not.
"""
import textwrap

import pytest

from app.agents.validator import static_semantic_validator_node

SCHEMA = {"id": "int64", "parent_id": "int64", "category": "object", "amount": "float64",
          "created_at": "object"}


def _validate(code: str, fidelity: str = "entire_dataset") -> dict:
    return static_semantic_validator_node({
        "generated_code": textwrap.dedent(code), "schema": SCHEMA, "analysis_fidelity": fidelity,
    })


def _performance_errors(res: dict) -> list:
    return [ln for ln in str(res.get("code_validation_feedback") or "").splitlines() if "PerformanceError" in ln]


LEGITIMATE = {
    "loop over a few categories, one filter per group (O(k*N))": '''
        import pandas as pd

        def main(df):
            rows = []
            for cat in df["category"].unique():
                sub = df[df["category"] == cat]
                rows.append({"category": cat, "total": sub["amount"].sum()})
            return pd.DataFrame(rows)
    ''',
    "loop over column names, one mask per column": '''
        import pandas as pd

        def main(df):
            out = {}
            for col in ["amount", "parent_id"]:
                out[col] = int((df[col] > 0).sum())
            return pd.DataFrame([out])
    ''',
    "groupby iteration with a filter per group": '''
        import pandas as pd

        def main(df):
            rows = []
            for cat, group in df.groupby("category"):
                big = df[df["amount"] > group["amount"].mean()]
                rows.append({"category": cat, "above_mean": len(big)})
            return pd.DataFrame(rows)
    ''',
    "iterrows over a small aggregate, not the frame": '''
        import pandas as pd

        def main(df):
            summary = df.groupby("category", as_index=False)["amount"].sum()
            lines = []
            for _, row in summary.iterrows():
                lines.append(f"{row['category']}: {row['amount']:.2f}")
            summary["label"] = lines
            return summary
    ''',
    "a filter outside any loop, and a vectorised merge": '''
        import pandas as pd

        def main(df):
            parents = df[df["parent_id"] == 0]
            joined = df.merge(parents[["id", "amount"]], left_on="parent_id", right_on="id",
                              how="left", suffixes=("", "_parent"))
            return joined
    ''',
    "a loop over the top rows only": '''
        import pandas as pd

        def main(df):
            top = df.nlargest(5, "amount")
            rows = []
            for _, r in top.iterrows():
                rows.append({"id": r["id"], "share": r["amount"] / df["amount"].sum()})
            return pd.DataFrame(rows)
    ''',
    "a while loop that does not scan the frame": '''
        import pandas as pd

        def main(df):
            bins, edge = [], 0
            while edge < 100:
                bins.append(edge)
                edge += 25
            df["bin"] = pd.cut(df["amount"], bins=bins + [float("inf")], labels=False)
            return df
    ''',
}

QUADRATIC = {
    "iterrows over the frame with a filter per row": '''
        import pandas as pd

        def main(df):
            rows = []
            for idx, row in df.iterrows():
                sub = df[df["id"] == row["parent_id"]]
                rows.append(len(sub))
            return pd.DataFrame({"matches": rows})
    ''',
    "itertuples over the frame": '''
        import pandas as pd

        def main(df):
            total = 0.0
            for r in df.itertuples():
                total += r.amount
            return pd.DataFrame({"total": [total]})
    ''',
    "index loop with a .loc filter per row": '''
        import pandas as pd

        def main(df):
            out = []
            for i in range(len(df)):
                parent = df.loc[df["id"] == df["parent_id"].iloc[i]]
                out.append(len(parent))
            return pd.DataFrame({"n": out})
    ''',
    "column loop with a query per row": '''
        import pandas as pd

        def main(df):
            out = []
            for pid in df["parent_id"]:
                out.append(len(df.query("id == @pid")))
            return pd.DataFrame({"n": out})
    ''',
}


@pytest.mark.parametrize("name", sorted(LEGITIMATE), ids=sorted(LEGITIMATE))
def test_legitimate_full_dataset_code_is_not_rejected(name):
    res = _validate(LEGITIMATE[name])
    assert _performance_errors(res) == [], f"false positive: {name}"


@pytest.mark.parametrize("name", sorted(QUADRATIC), ids=sorted(QUADRATIC))
def test_the_quadratic_shapes_are_rejected_on_the_entire_dataset(name):
    res = _validate(QUADRATIC[name])
    assert res["static_semantic_error"] is True and _performance_errors(res), name


@pytest.mark.parametrize("name", sorted(QUADRATIC), ids=sorted(QUADRATIC))
def test_a_sample_run_is_never_held_to_the_rule(name):
    """On a sample the quadratic code finishes in time; the rule is for the full data."""
    res = _validate(QUADRATIC[name], fidelity="quick_sample")
    assert _performance_errors(res) == [], name
