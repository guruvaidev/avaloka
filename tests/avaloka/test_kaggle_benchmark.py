"""Opt-in end-to-end benchmark over real Kaggle datasets.

Reuses the repo's *existing* download tooling (``tests/download_kaggle_datasets``)
so the same datasets that exercise the coding agents also exercise the Avaloka
CLI. Skips cleanly when the Kaggle CLI/credentials or the datasets are absent, so
CI never flakes on network.

Run explicitly::

    python tests/download_kaggle_datasets.py          # fetch the coding-agent datasets
    pytest tests/avaloka/test_kaggle_benchmark.py -m kaggle -v
"""

import sys
from pathlib import Path

import pytest

from avaloka.benchmark.datasets import ensure_kaggle_dataset
from avaloka.benchmark.runner import BenchmarkRunner
from avaloka.benchmark.suite import kaggle_tasks

pytestmark = pytest.mark.kaggle

REPO_ROOT = Path(__file__).resolve().parents[2]


def _reuse_existing_downloader(slug: str) -> bool:
    """Best-effort: drive the repo's existing kaggle download script for ``slug``.

    The existing script lives in ``tests/`` and imports the heavy app graph; if
    that import fails we simply fall back to the benchmark's own downloader.
    """
    tests_dir = REPO_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    try:
        from download_kaggle_datasets import download_dataset  # type: ignore
    except Exception:
        return False
    try:
        download_dataset(slug)
        return True
    except SystemExit:
        return False


@pytest.mark.parametrize("task", kaggle_tasks(), ids=lambda t: t.name)
def test_kaggle_dataset_benchmark(task, tmp_path):
    slug = task.dataset.kaggle_slug
    # Prefer the repo's existing downloader (the user's kaggle test harness).
    _reuse_existing_downloader(slug)
    local = ensure_kaggle_dataset(slug, task.dataset.preferred_file)
    if local is None:
        pytest.skip(f"Kaggle dataset {slug} unavailable — run tests/download_kaggle_datasets.py")

    card = BenchmarkRunner(tmp_path).run([task])
    results = [r for r in card.results if r.error is None]
    assert results, f"no gradable runs for {task.name}"
    for r in results:
        assert r.passed, r.as_dict()
