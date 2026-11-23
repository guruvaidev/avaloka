"""CI forwarder so `pytest tests/` exercises the benchmark harness smoke tests.

The real tests live in ``benchmarks/tests/test_harness_smoke.py`` (kept next to the
harness). This module re-exports them so the existing CI pipeline
(`pytest --continue-on-collection-errors tests/`) runs them without new config.
"""

from benchmarks.tests.test_checkpointing import *  # noqa: F401,F403
from benchmarks.tests.test_dab_boundary import *  # noqa: F401,F403
from benchmarks.tests.test_dab_adapter import *  # noqa: F401,F403
from benchmarks.tests.test_dab_agent_fixes import *  # noqa: F401,F403
from benchmarks.tests.test_data_agent import *  # noqa: F401,F403
from benchmarks.tests.test_harness_smoke import *  # noqa: F401,F403
from benchmarks.tests.test_swarm_ablation import *  # noqa: F401,F403
from benchmarks.tests.test_reliability import *  # noqa: F401,F403
from benchmarks.tests.test_instant_intent import *  # noqa: F401,F403
