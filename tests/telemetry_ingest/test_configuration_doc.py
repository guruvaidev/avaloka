"""The configuration table in the design document against the service's code.

``docs/design/TELEMETRY.md`` lists every environment variable the ingest reads.
A list in prose drifts the first time someone adds a variable and not a row, or
removes one and leaves the row, so both directions are checked here.

The code side is found by parsing ``telemetry_service/*.py``, not by importing
it: a variable counts when its name reaches ``os.environ.get``, ``os.getenv``
or ``os.environ[...]``, as a literal or through a module-level constant.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "design" / "TELEMETRY.md"
SERVICE = REPO / "telemetry_service"
PREFIX = "TELEMETRY_"

_HEADING_RE = re.compile(r"^#### Configuration\s*$", re.M)
_ROW_RE = re.compile(r"^\| `([A-Z][A-Z0-9_]*)` \|", re.M)


def _documented() -> set[str]:
    """Variable names in the first table under the Configuration heading."""
    text = DOC.read_text(encoding="utf-8")
    heading = _HEADING_RE.search(text)
    assert heading, f"{DOC.relative_to(REPO)} has no '#### Configuration' heading"
    rows: list[str] = []
    in_table = False
    for line in text[heading.end():].splitlines():
        if line.startswith("|"):
            in_table = True
            rows.append(line)
        elif in_table:
            break
    return set(_ROW_RE.findall("\n".join(rows)))


def _is_environ(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "environ"


def _env_name_nodes(tree: ast.AST) -> list[ast.AST]:
    """The expression naming the variable, for every environment read."""
    found: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
            reads_environ = node.func.attr == "get" and _is_environ(node.func.value)
            if reads_environ or node.func.attr == "getenv":
                found.append(node.args[0])
        elif isinstance(node, ast.Subscript) and _is_environ(node.value):
            found.append(node.slice)
    return found


def _read_by_the_service() -> set[str]:
    names: set[str] = set()
    for path in sorted(SERVICE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = {
            target.id: stmt.value.value
            for stmt in tree.body
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
            for target in stmt.targets
            if isinstance(target, ast.Name)
        }
        for node in _env_name_nodes(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.add(node.value)
            elif isinstance(node, ast.Name) and node.id in constants:
                names.add(constants[node.id])
            else:
                raise AssertionError(
                    f"{path.relative_to(REPO)}:{node.lineno} reads an environment variable "
                    "whose name this test cannot resolve; name it with a literal or a "
                    "module-level constant"
                )
    return {name for name in names if name.startswith(PREFIX)}


class TestTheConfigurationTable:
    def test_both_sides_were_found(self):
        """An empty set on either side would make the comparisons vacuous."""
        assert len(_documented()) >= 5
        assert len(_read_by_the_service()) >= 5

    def test_every_variable_the_service_reads_is_documented(self):
        missing = _read_by_the_service() - _documented()
        assert not missing, (
            f"read by telemetry_service/ but not in the Configuration table of "
            f"{DOC.relative_to(REPO)}: {sorted(missing)}"
        )

    def test_every_documented_variable_is_read_by_the_service(self):
        stale = _documented() - _read_by_the_service()
        assert not stale, (
            f"in the Configuration table of {DOC.relative_to(REPO)} but read nowhere in "
            f"telemetry_service/: {sorted(stale)}"
        )
