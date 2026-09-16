import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import pytest

from app.api.workflow import build_graph
from app.graph.etl_state import ETLState
from langchain_core.messages import HumanMessage


pytestmark = pytest.mark.e2e


# Detailed logging setup
log_file = Path(__file__).parent / "e2e_kaggle_tests.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    filename=log_file,
    filemode='w',  # Overwrite the log file on each run
    force=True
)

# Console handler for INFO level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

LOGGER = logging.getLogger(__name__)

DATA_DIR = Path("app/sample_data")
DATASET_QUESTION_CSV = DATA_DIR / "Dataset Questions - Sheet1.csv"
KAGGLE_ANALYSIS_CSV = DATA_DIR / "Kaggle Data Analysis Questions - Sheet1.csv"


@dataclass
class KaggleQuestion:
    dataset_slug: str
    question: str
    preferred_file: Optional[str] = None


def _load_dataset_questions_sheet() -> List[KaggleQuestion]:
    if not DATASET_QUESTION_CSV.exists():
        return []
    df = pd.read_csv(DATASET_QUESTION_CSV, header=None)
    if df.empty:
        return []
    df[0] = df[0].ffill()
    questions: List[KaggleQuestion] = []
    for dataset_url, question in df.itertuples(index=False, name=None):
        if not isinstance(question, str) or not question.strip():
            continue
        dataset_slug = None
        preferred_file = None
        if isinstance(dataset_url, str) and "kaggle.com/datasets" in dataset_url:
            parts = dataset_url.split("/datasets/")
            if len(parts) == 2:
                slug_part = parts[1]
                if "?select=" in slug_part:
                    slug, file_part = slug_part.split("?select=", maxsplit=1)
                    preferred_file = file_part.strip()
                else:
                    slug = slug_part
                dataset_slug = slug.strip("/")
        if not dataset_slug:
            continue
        questions.append(KaggleQuestion(dataset_slug=dataset_slug, question=question.strip(), preferred_file=preferred_file))
    return questions


def _load_analysis_questions_sheet() -> List[KaggleQuestion]:
    if not KAGGLE_ANALYSIS_CSV.exists():
        return []
    df = pd.read_csv(KAGGLE_ANALYSIS_CSV, engine="python")
    questions: List[KaggleQuestion] = []
    for _, row in df.iterrows():
        dataset_path = row.get("Kaggle Dataset Path")
        question_blob = row.get("Questions for Data Analysis")
        if not isinstance(dataset_path, str) or not isinstance(question_blob, str):
            continue
        match = re.search(r'"([^"]+)"', dataset_path)
        if not match:
            continue
        dataset_slug = match.group(1)
        for question in filter(None, [segment.strip() for segment in question_blob.splitlines()]):
            questions.append(KaggleQuestion(dataset_slug=dataset_slug, question=question))
    return questions


def load_all_questions() -> List[KaggleQuestion]:
    questions = _load_dataset_questions_sheet() + _load_analysis_questions_sheet()
    # Deduplicate by slug + question text
    seen = set()
    unique: List[KaggleQuestion] = []
    for q in questions:
        key = (q.dataset_slug, q.question)
        if key in seen:
            continue
        seen.add(key)
        unique.append(q)
    return unique


def ensure_dataset_file(dataset_slug: str, preferred_file: Optional[str]) -> Path:
    if preferred_file:
        root_preferred = DATA_DIR / preferred_file
        if root_preferred.exists():
            return root_preferred
    dataset_dir = DATA_DIR / dataset_slug.replace("/", "__")
    if preferred_file:
        preferred_path = DATA_DIR / preferred_file
        if preferred_path.exists():
            return preferred_path
        preferred_path = dataset_dir / preferred_file
        if preferred_path.exists():
            return preferred_path

    if dataset_dir.exists():
        csv_files = sorted(dataset_dir.rglob("*.csv"))
        if csv_files:
            return csv_files[0]

    LOGGER.warning(
        "Dataset %s not found under %s. Run 'python tests/download_kaggle_datasets.py' to fetch it before executing the tests.",
        dataset_slug,
        dataset_dir,
    )
    pytest.skip(f"Dataset {dataset_slug} unavailable locally. Run 'python tests/download_kaggle_datasets.py' first.")


def get_csv_schema(file_path: Path) -> dict:
    df = pd.read_csv(file_path, nrows=0)
    return {col: "object" for col in df.columns}


@pytest.fixture(scope="session")
def compiled_graph():
    return build_graph().compile()


def run_graph_with_prompt(graph, prompt: str, data_source: Path, output_location: Path, schema: dict):
    initial_state = ETLState(
        messages=[HumanMessage(content=prompt)],
        planner_definition=prompt,
        ready_to_code=True,
        data_source_location=str(data_source),
        output_location=str(output_location),
        schema=schema,
        input_data_type="csv",
        uploaded_csv_preview=[list(schema.keys())],
    )
    return graph.invoke(initial_state, config={"recursion_limit": 50})


def _sanitize_id(text: str, max_length: int = 60) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return token[:max_length] if len(token) > max_length else token


ALL_QUESTIONS = load_all_questions()
QUESTION_INDEX = {
    (entry.dataset_slug, entry.question): idx + 1 for idx, entry in enumerate(ALL_QUESTIONS)
}
TOTAL_QUESTIONS = len(ALL_QUESTIONS)


@pytest.mark.parametrize(
    "entry",
    ALL_QUESTIONS,
    ids=lambda e: f"{e.dataset_slug}-{_sanitize_id(e.question)}"
)
def test_kaggle_questions(entry: KaggleQuestion, compiled_graph, tmp_path):
    position = QUESTION_INDEX[(entry.dataset_slug, entry.question)]
    LOGGER.info(
        "[%d/%d] START: Processing dataset '%s' | question: %s",
        position,
        TOTAL_QUESTIONS,
        entry.dataset_slug,
        entry.question,
    )

    dataset_path = ensure_dataset_file(entry.dataset_slug, entry.preferred_file)

    output_location = tmp_path / f"output_{_sanitize_id(entry.question)}.csv"
    schema = get_csv_schema(dataset_path)

    LOGGER.info("Running graph for question: %s", entry.question)
    final_state = run_graph_with_prompt(
        compiled_graph,
        entry.question,
        dataset_path,
        output_location,
        schema,
    )
    LOGGER.info("Graph execution finished.")

    status = "SUCCESS"
    error_message = ""

    generated_code = (
        final_state.get("generated_code")
        or final_state.get("coder_definition", {}).get("code")
    )
    if not generated_code:
        status = "FAILURE"
        error_message = "Code generation failed"
    elif final_state.get("syntax_error"):
        status = "FAILURE"
        error_message = "Syntax error detected in generated code."
    elif final_state.get("static_semantic_error"):
        status = "FAILURE"
        error_message = "Static semantic error detected in generated code."
    elif final_state.get("execution_error") is not None:
        status = "FAILURE"
        error_message = f"Execution error: {final_state.get('execution_error')}"

    LOGGER.info(
        "[%d/%d] END: Status: %s | Dataset: '%s' | Question: %s",
        position,
        TOTAL_QUESTIONS,
        status,
        entry.dataset_slug,
        entry.question,
    )
    if status == "FAILURE":
        LOGGER.error(error_message)
        # Also log the full final_state for debugging
        LOGGER.error("Final state: %s", final_state)

    assert status == "SUCCESS", error_message
