"""The coder prompt's OOM guidance, and the one thing it must never say.

The block tells the model not to fit memory-unbounded estimators on large
data. An earlier wording also told it to "aggressively downsample the dataset",
which contradicted NEVER RE-SAMPLE THE DATAFRAME in the same prompt. These
tests pin the reconciled wording: constrain or replace the model, tell the
analyst when nothing fits, and never shrink the input.

They read the RENDERED prompt (the f-string evaluated with dummy values, the
same way tests/test_coder_sample_data.py checks it for unescaped braces), not
the module source, so text in a comment or another prompt does not satisfy
them. No LLM: this proves what the model is told, not that it obeys.
"""

import ast
import pathlib

_CODER = pathlib.Path(__file__).resolve().parent.parent / "app" / "agents" / "coder.py"
_HEADER = "SECURITY AND STABILITY (OOM TRAP PREVENTION):"
# The block runs from its header to the next blank line.
_BLOCK_END = "\n\n"


def _rendered_coder_prompts():
    src = _CODER.read_text(encoding="utf-8")
    rendered = []
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.JoinedStr) and len(ast.dump(node)) > 2000):
            continue
        segment = ast.get_source_segment(src, node)
        if not segment:
            continue
        names = {
            n.id
            for fv in ast.walk(node)
            if isinstance(fv, ast.FormattedValue)
            for n in ast.walk(fv.value)
            if isinstance(n, ast.Name)
        }
        rendered.append(eval(segment, {"__builtins__": {}}, {name: "x" for name in names}))  # noqa: S307
    assert rendered, "no large f-string found in coder.py; did the prompt move?"
    return rendered


def _oom_block() -> str:
    holders = [p for p in _rendered_coder_prompts() if _HEADER in p]
    assert len(holders) == 1, (
        f"expected the OOM guidance in exactly one coder prompt, found {len(holders)}"
    )
    tail = holders[0].split(_HEADER, 1)[1]
    block = tail.split(_BLOCK_END, 1)[0]
    assert block.strip(), "the OOM guidance header has no rules under it"
    return block


def test_oom_block_names_the_unbounded_estimators():
    block = _oom_block()
    assert "DecisionTreeClassifier(max_depth=None)" in block
    assert "`SVC`" in block
    assert "`KNeighborsClassifier`" in block


def test_oom_block_offers_a_scalable_alternative():
    block = _oom_block()
    assert "SGDClassifier" in block or "LinearSVC" in block, (
        "a prohibition with no sanctioned alternative invites the model to "
        "invent one, most likely by sampling"
    )


def test_oom_block_says_to_tell_the_analyst_when_nothing_fits():
    block = _oom_block()
    assert "say so to the analyst" in block
    assert "rather than silently reducing the input" in block


def test_oom_block_never_tells_the_model_to_shrink_the_dataframe():
    # Scoped to the block: NEVER RE-SAMPLE THE DATAFRAME names df.sample() and
    # head() elsewhere in the prompt precisely in order to forbid them.
    block = _oom_block().lower()
    for forbidden in ("downsample", "subsample", "df.sample", ".sample(", ".head(", ".iloc["):
        assert forbidden not in block, (
            f"the OOM guidance mentions {forbidden!r}; it must not tell the model "
            "to reduce the input (see NEVER RE-SAMPLE THE DATAFRAME)"
        )


def test_oom_block_and_the_resampling_rule_live_in_the_same_prompt():
    prompt = next(p for p in _rendered_coder_prompts() if _HEADER in p)
    assert "NEVER RE-SAMPLE THE DATAFRAME" in prompt
    assert prompt.index("NEVER RE-SAMPLE THE DATAFRAME") < prompt.index(_HEADER), (
        "the OOM block refers to the re-sampling rule as 'above'"
    )
