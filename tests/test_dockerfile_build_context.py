import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO_ROOT / "deploy"

# COPY/ADD <src> where a source path escapes the build context (starts with ..).
_PARENT_SRC = re.compile(r"(?:^|\s)\.\.(?:/|\s|$)")


def _dockerfiles():
    return list(DEPLOY_DIR.rglob("Dockerfile*")) if DEPLOY_DIR.exists() else []


def test_dockerfiles_present():
    assert _dockerfiles(), "expected Dockerfiles under deploy/"


def test_no_dockerfile_copies_from_outside_build_context():
    offenders = []
    for df in _dockerfiles():
        for lineno, raw in enumerate(df.read_text().splitlines(), 1):
            line = raw.strip()
            if line.startswith("#"):
                continue
            head = line.split(None, 1)[0].upper() if line else ""
            if head not in ("COPY", "ADD"):
                continue
            # Ignore --from=<stage> multi-stage sources, which are not context paths.
            args = re.sub(r"--from=\S+", "", line[len(head):])
            if _PARENT_SRC.search(args):
                offenders.append(f"{df.relative_to(REPO_ROOT)}:{lineno}: {line}")

    assert not offenders, (
        "Dockerfile COPY/ADD references a path outside the build context "
        "(Docker forbids this):\n" + "\n".join(offenders)
    )
