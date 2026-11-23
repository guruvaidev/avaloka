"""rweb18b: each clone's line said "cost $0.0001" above "No model was called".

The figure was the nominal local-CPU charge (fireflies.base.LOCAL_CPU_USD_PER_SECOND),
not money and not a model: the clone line now says which. Records come from a real
coordinator run (the fireflies, not a stub).
"""

from __future__ import annotations

import re

import pytest

from avaloka.coordinator import MainAvaloka
from avaloka.swarm_limits import SwarmLimits
from avaloka.swarm_narration import VERBOSE, narrate
from avaloka.swarm_trace import SwarmTraceRecorder


@pytest.fixture(autouse=True)
def _no_edition(monkeypatch):
    monkeypatch.delenv("AVALOKA_EDITION", raising=False)


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    from tests.avaloka.conftest import write_churn_csv

    base = tmp_path_factory.mktemp("cost")
    data = write_churn_csv(base / "churn.csv")
    recorder = SwarmTraceRecorder()
    MainAvaloka(swarm_limits=SwarmLimits("oss", 2, 2)).coordinate(
        str(data), goal="what drives churn", output_dir=base / "fleet", workers=2, recorder=recorder)
    return recorder.records


def test_a_clone_line_never_calls_a_nominal_cpu_charge_a_cost(records):
    text = "\n".join(narrate(records, VERBOSE))
    assert "No model was called" in text, text
    clones = [line for line in text.splitlines() if " is back after " in line]
    assert len(clones) == 2, text
    for line in clones:
        assert not re.search(r"cost \$", line), line
        assert "at a nominal CPU rate" in line, line
    back = [r for r in records if r.get("type") == "clone.returned"]
    assert all(r.get("model_cost_usd") == 0 and r.get("compute_cost_usd", 0) > 0 for r in back), back


def test_a_tiny_charge_is_never_rounded_to_zero(records):
    tiny = [dict(r, compute_cost_usd=0.00003) if r.get("type") == "clone.returned" else r for r in records]
    text = "\n".join(narrate(tiny, VERBOSE))
    assert "compute under $0.0001 at a nominal CPU rate" in text, text
    assert not re.search(r"\$0(?![.\d])|\$0\.(?!\d)", text.split("Finished in")[0]), text
