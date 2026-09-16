"""Utility script to download all Kaggle datasets used by tests/test_e2e_kaggle.py."""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from test_e2e_kaggle import DATA_DIR, load_all_questions

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def download_dataset(dataset_slug: str) -> None:
    target_dir = DATA_DIR / dataset_slug.replace("/", "__")
    csv_files = sorted(target_dir.rglob("*.csv"))
    if csv_files:
        LOGGER.info("Dataset %s already available at %s; skipping.", dataset_slug, target_dir)
        return

    kaggle_cli = shutil.which("kaggle")
    if not kaggle_cli:
        raise SystemExit(
            "Kaggle CLI not found on PATH. Install it (`pip install kaggle`) and configure ~/.kaggle/kaggle.json."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s ...", dataset_slug)
    result = subprocess.run(
        [
            kaggle_cli,
            "datasets",
            "download",
            "-d",
            dataset_slug,
            "--unzip",
            "-p",
            str(target_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        LOGGER.error("Failed to download %s. STDERR:\n%s", dataset_slug, result.stderr.strip())
        raise SystemExit(1)
    LOGGER.info("Dataset %s downloaded to %s", dataset_slug, target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kaggle datasets required for the E2E tests.")
    parser.add_argument(
        "--filter",
        metavar="SUBSTRING",
        help="Download only datasets whose slug contains this substring.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_all_questions()
    slugs = sorted({q.dataset_slug for q in questions})
    if args.filter:
        slugs = [slug for slug in slugs if args.filter in slug]
        if not slugs:
            LOGGER.warning("No datasets matched filter '%s'.", args.filter)
            return

    for slug in slugs:
        download_dataset(slug)

    LOGGER.info("All requested datasets are available in %s", DATA_DIR)


if __name__ == "__main__":
    main()
