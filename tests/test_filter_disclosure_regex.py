"""The filter replay never runs a pattern that can pin a worker, nor more rows than its cap.

app/agents/filter_disclosure replays model-written filters in the API
process. ``str.contains('(a+)+$')`` on 'aaaa...ab' backtracks for hours, and a
running ``re`` match cannot be interrupted, so such a pattern is never run:
the filter is disclosed uncounted instead. A frame over the replay cap is
counted on a fixed-seed sample and the line says "about".
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from app.agents.claim_verifier import DataUsed
from app.agents.filter_disclosure import detect_row_filters, scope_filters

ROOT = pathlib.Path(__file__).resolve().parents[1]
UNCOUNTED = "computed on a filtered subset of the orders (only where"


def _code(condition: str) -> str:
    return (f"def main(orders):\n    orders = orders[{condition}]\n"
            "    return orders.groupby('segment')['amount'].mean()\n")


def _orders(n: int = 40) -> pd.DataFrame:
    status = ["returned" if i % 10 == 0 else "refunded" if i % 10 == 1 else "shipped" for i in range(n)]
    return pd.DataFrame({"status": status, "amount": [float(i) for i in range(n)],
                         "segment": ["smb" if i % 2 else "consumer" for i in range(n)]})


def _used(frame: pd.DataFrame) -> DataUsed:
    return DataUsed(frames=[("orders.csv", frame)], sources=["/x/orders.csv"])


def test_a_catastrophic_pattern_is_not_run_and_the_filter_is_disclosed_uncounted():
    """``(a+)+$`` on 'a' * 40 + 'b' is about 2**40 steps in ``re``. Run in a child
    process with a hard timeout, so a version that runs it fails here instead of
    hanging the suite."""
    script = textwrap.dedent("""
        import time
        import pandas as pd
        from app.agents.claim_verifier import DataUsed
        from app.agents.filter_disclosure import scope_filters
        frame = pd.DataFrame({"note": ["a" * 40 + "b"] * 3 + ["ok"], "amount": [1.0, 2.0, 3.0, 4.0],
                              "segment": ["smb"] * 4})
        code = ("def main(orders):\\n    orders = orders[orders['note'].str.contains(r'(a+)+$')]\\n"
                "    return orders.groupby('segment')['amount'].mean()\\n")
        start = time.monotonic()
        lines = [f.line() for f in scope_filters(code, "Average amount by segment?", "Smb averages 2.0.",
                                                 DataUsed(frames=[("orders.csv", frame)], sources=["/x"]))]
        print(round(time.monotonic() - start, 3))
        print(lines[0] if lines else "")
    """)
    try:
        done = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("the replay ran the catastrophic pattern and hung (killed after 60 s)")
    assert done.returncode == 0, done.stderr[-2000:]
    elapsed, line = done.stdout.strip().splitlines()[-2:]
    assert float(elapsed) < 1.0
    assert UNCOUNTED in line and "(a+)+$" in line


@pytest.mark.parametrize("pattern", [r"(?<=re)turned", r"(?<!x)returned", r"(re)turned\1?", r"(r)\1"])
def test_lookarounds_and_backreferences_are_not_run(pattern):
    code = _code(f"~orders['status'].str.contains(r'{pattern}')")
    filters = scope_filters(code, "Average amount by segment?", "Smb averages 20.", _used(_orders()))
    assert len(filters) == 1 and filters[0].rows_dropped is None
    assert UNCOUNTED in filters[0].line()


def test_a_class_or_quantifier_is_uncounted_rather_than_counted_as_something_else():
    code = _code("~orders['status'].str.contains(r're[tf]\\w+ed')")
    (flt,) = detect_row_filters(code, _used(_orders()))
    assert flt.rows_dropped is None


def test_a_plain_alternation_still_counts_exactly():
    code = _code("~orders['status'].str.contains('returned|refunded')")
    (flt,) = detect_row_filters(code, _used(_orders()))
    assert (flt.rows_before, flt.rows_dropped) == (40, 8)
    literal = _code("~orders['status'].str.contains('returned')")
    assert detect_row_filters(literal, _used(_orders()))[0].rows_dropped == 4


def test_a_frame_over_the_replay_cap_is_counted_on_a_sample_and_said_to_be_about():
    n = 250_000
    frame = pd.DataFrame({"returned": [i % 13 == 0 for i in range(n)], "amount": [1.0] * n,
                          "segment": ["smb"] * n})
    truth = sum(1 for i in range(n) if i % 13 == 0)                       # 19,231
    code = _code("orders['returned'] == False")
    (flt,) = scope_filters(code, "Average amount by segment?", "Smb averages 1.", _used(frame))
    line = flt.line()
    match = re.search(r"excludes about ([\d,]+) returned orders", line)
    assert match, line
    assert abs(int(match.group(1).replace(",", "")) - truth) <= 0.03 * truth
    assert f"of {n:,} orders" in line
