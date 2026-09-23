"""The key the docs tell you to set has to be the key that works.

README and docs/INSTALL.md name GROQ_API_KEY. On oss/1.6 a user who set exactly
that got a planner, a validator and a memory plane that were all silently
disabled -- each agent read only its own GROQ_API_KEY_<ROLE>_AGENT variable, and
the only sign was an INFO line in a log nobody reads.

This is a release-blocking class of bug precisely because nothing fails: the
product starts, answers, and is quietly stupid.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP = REPO_ROOT / "app"

#: Role-scoped keys. Any of these may be read, but never as the only source.
_ROLE_KEYS = re.compile(r"^GROQ_API_KEY_(?:PLANNING|CODING)_AGENT$")


def _python_files() -> List[Path]:
    return sorted(p for p in APP.rglob("*.py") if "__pycache__" not in p.parts)


def _env_key(node) -> str | None:
    """The literal of an ``os.environ.get("X")`` call, if that is what this is."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr != "get":
        return None
    target = node.func.value
    if not (isinstance(target, ast.Attribute) and target.attr == "environ"):
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return None
    value = node.args[0].value
    return value if isinstance(value, str) else None


def _offenders() -> List[str]:
    """Statements that read a role key without also accepting the generic one.

    Parsed rather than regexed: the lookup is usually a multi-line parenthesised
    ``or`` chain, and reading one physical line of it reports a fallback that is
    sitting on the next line as missing.
    """
    bad: List[str] = []
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text("utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for statement in ast.walk(tree):
            if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Return)):
                continue
            keys = {k for node in ast.walk(statement) if (k := _env_key(node))}
            role = {k for k in keys if _ROLE_KEYS.match(k)}
            if role and "GROQ_API_KEY" not in keys:
                bad.append(
                    f"{path.relative_to(REPO_ROOT).as_posix()}:{statement.lineno} "
                    f"({', '.join(sorted(role))})"
                )
    return bad


def test_c7_01_role_keys_always_fall_back_to_the_documented_key() -> None:
    """Every GROQ_API_KEY_<ROLE>_AGENT lookup also accepts GROQ_API_KEY."""
    offenders = _offenders()
    assert not offenders, (
        "these read a role-scoped Groq key with no fallback to GROQ_API_KEY, so a "
        "user who followed the README gets a silently disabled agent:\n  "
        + "\n  ".join(offenders)
    )


def test_c7_02_the_documented_key_is_actually_documented() -> None:
    """If the docs stop naming GROQ_API_KEY, this contract is measuring nothing."""
    readme = (REPO_ROOT / "README.md")
    install = (REPO_ROOT / "docs" / "INSTALL.md")
    named = any(p.is_file() and "GROQ_API_KEY" in p.read_text("utf-8", errors="ignore")
                for p in (readme, install))
    assert named, "neither README.md nor docs/INSTALL.md mentions GROQ_API_KEY"
