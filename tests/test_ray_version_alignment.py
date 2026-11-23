"""Every Ray pin in the repository names the same release.

The app talks to the cluster over Ray Client, which refuses a version mismatch,
and the cluster images, the KubeRay CRs and the image requirement files each
carry their own copy of the version. tests/k8s/test_t1_deploy_logic.py checks a
subset of them, but only where helm and kind are present; this one runs in the
hermetic gate. It also pins RAY_TRAIN_V2_ENABLED=0 in the images that run
ray_job.py: Ray 2.52 enables Train V2 by default, and the training loop was
written against V1.

The runtime templates matter as much as the images: every analysis RayJob ran
rayproject/ray:2.10.0 from app/infra/rayjob.yaml long after the app moved on,
because nothing compared it with anything. test_every_ray_reference_* walks the
source tree for every `rayproject/ray:` tag and `rayVersion` field instead of
listing files, so a new template cannot slip past it.
"""
from __future__ import annotations

import ast
import os
import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

REQUIREMENT_FILES = (
    "requirements.txt",
    "app/agents/mta_v2/training_docker_image/requirements.txt",
    "app/agents/mta_v2/inference_service_image/requirements.txt",
    "app/agents/mta_v2/inference_docker_image/requirements.txt",
)
RAY_BASE_IMAGES = (
    "deploy/docker/Dockerfile.ray",
    "app/agents/mta_v2/training_docker_image/Dockerfile",
    "app/data_transfer_docker/Dockerfile",
)
TRAIN_V1_IMAGES = (
    "deploy/docker/Dockerfile.ray",
    "app/agents/mta_v2/training_docker_image/Dockerfile",
)


# Not shipped or not current: test fixtures, and engineering write-ups that quote
# historical versions on purpose.
SCAN_SKIP_DIRS = {".git", "node_modules", "__pycache__", "tests", "research", "evaluations",
                  ".venv", "venv", "dist", "build"}
SCAN_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".sh", ".toml", ".txt", ".cfg", ""}
RAY_IMAGE_RE = re.compile(r"rayproject/ray:([A-Za-z0-9._-]+)")
VERSION_LITERAL_RE = re.compile(r"\d+\.\d+\.\d+")
RAY_VERSION_RE = re.compile(r"""["']?rayVersion["']?\s*:\s*["']?([^"'\s,}]+)""")


def _source_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SCAN_SKIP_DIRS and not d.startswith(".venv")]
        for name in filenames:
            path = pathlib.Path(dirpath, name)
            if path.suffix in SCAN_SUFFIXES or name.startswith("Dockerfile"):
                try:
                    yield path.relative_to(ROOT).as_posix(), path.read_text(errors="ignore")
                except OSError:
                    continue


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def _app_pin() -> str:
    m = re.search(r"^ray\[[^\]]*\]==([0-9.]+)\s*$", _read("requirements.txt"), re.M)
    assert m, "requirements.txt has no exact ray pin"
    return m.group(1)


def test_requirement_files_pin_the_app_ray_version():
    want = _app_pin()
    found = {}
    for rel in REQUIREMENT_FILES:
        pins = re.findall(r"^ray(?:\[[^\]]*\])?==([0-9.]+)\s*$", _read(rel), re.M)
        assert pins, f"{rel} has no exact ray pin"
        found[rel] = pins
    assert all(p == [want] for p in found.values()), f"ray pins disagree with requirements.txt=={want}: {found}"


def test_cluster_images_and_crs_match_the_app_pin():
    want = _app_pin()
    tags = {}
    for rel in RAY_BASE_IMAGES:
        m = re.search(r"^FROM rayproject/ray:([0-9.]+)-py311\s*$", _read(rel), re.M)
        assert m, f"{rel} does not build FROM a pinned rayproject/ray:<version>-py311"
        tags[rel] = m.group(1)
    rc = next(d for d in yaml.safe_load_all(_read("deploy/helm/ray/raycluster.yaml")) if d and d.get("kind") == "RayCluster")
    rs = next(d for d in yaml.safe_load_all(_read("deploy/helm/ray/rayservice.yaml")) if d and d.get("kind") == "RayService")
    tags["raycluster.yaml rayVersion"] = rc["spec"]["rayVersion"]
    tags["rayservice.yaml rayVersion"] = rs["spec"]["rayClusterConfig"]["rayVersion"]
    assert set(tags.values()) == {want}, f"app pins ray=={want}, cluster side has {tags}"


def test_training_images_keep_ray_train_v1():
    for rel in TRAIN_V1_IMAGES:
        assert re.search(r"^ENV RAY_TRAIN_V2_ENABLED=0\s*$", _read(rel), re.M), (
            f"{rel} must set RAY_TRAIN_V2_ENABLED=0: Ray 2.52 defaults to Train V2 and "
            "ray_job.py's TorchTrainer flow has only been run on V1"
        )


def test_every_ray_image_reference_names_the_app_version():
    want = _app_pin()
    refs, wrong = [], []
    for rel, text in _source_files():
        for m in RAY_IMAGE_RE.finditer(text):
            tag = m.group(1)
            refs.append(f"{rel}: {tag}")
            if tag != want and not tag.startswith(want + "-"):
                wrong.append(f"{rel}: rayproject/ray:{tag}")
    assert len(refs) >= 6, f"found only {len(refs)} rayproject/ray references; is the walk broken? {refs}"
    assert not wrong, f"app pins ray=={want}; these images run something else: {wrong}"


def test_every_ray_version_field_names_the_app_version():
    want = _app_pin()
    refs, wrong = [], []
    for rel, text in _source_files():
        for m in RAY_VERSION_RE.finditer(text):
            if not VERSION_LITERAL_RE.match(m.group(1)):
                # A computed value ("rayVersion": _ray_version(image) in
                # app/infra/user_ray_clusters.py): its default is pinned by
                # test_user_ray_cluster_defaults_name_the_app_version below.
                continue
            refs.append(f"{rel}: {m.group(1)}")
            if m.group(1) != want:
                wrong.append(f"{rel}: rayVersion {m.group(1)}")
    assert len(refs) >= 5, f"found only {len(refs)} rayVersion fields; is the walk broken? {refs}"
    assert not wrong, f"app pins ray=={want}; these clusters declare another version: {wrong}"


USER_CLUSTERS = "app/infra/user_ray_clusters.py"


def test_user_ray_cluster_defaults_name_the_app_version():
    """Per-user RayClusters are built from a default image and a fallback
    rayVersion in Python, not from a template. Ray Client refuses a version
    mismatch, so both must name the app's Ray (RAY_USER_CLUSTER_IMAGE overrides)."""
    want = _app_pin()
    text = (ROOT / USER_CLUSTERS).read_text()
    image = re.search(r'^DEFAULT_IMAGE = "rayproject/ray:([^"]+)"', text, re.M)
    version = re.search(r'^DEFAULT_RAY_VERSION = "([^"]+)"', text, re.M)
    assert image and version, f"{USER_CLUSTERS} must define DEFAULT_IMAGE and DEFAULT_RAY_VERSION"
    assert image.group(1) == want or image.group(1).startswith(want + "-"), (
        f"{USER_CLUSTERS}: default image runs rayproject/ray:{image.group(1)}, the app pins {want}")
    assert version.group(1) == want, f"{USER_CLUSTERS}: fallback rayVersion {version.group(1)}, the app pins {want}"
    assert re.search(r'or DEFAULT_IMAGE\)', text) and re.search(r'else DEFAULT_RAY_VERSION', text), (
        f"{USER_CLUSTERS}: _image()/_ray_version() must fall back to the two constants")


RUNTIME_ENV_TEMPLATES = ("app/infra/rayjob.yaml", "app/infra/rayjob_data_transfer.yaml")
CODE_ROOTS = ("app", "avaloka", "scripts")
PIN_RE = re.compile(r"[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*")


def _assert_runtime_pins(where: str, pip: list) -> None:
    unpinned = [p for p in pip if not isinstance(p, str) or not PIN_RE.fullmatch(p)]
    assert not unpinned, f"{where}: runtime_env packages must be pinned exactly, with no extras: {unpinned}"
    names = {p.split("==")[0].lower() for p in pip}
    assert "ray" not in names and not any(n.startswith("getdaft") for n in names), (
        f"{where}: a runtime_env must never install Ray, or getdaft (a shim that pulls an unpinned daft): {sorted(names)}")


@pytest.mark.parametrize("rel", RUNTIME_ENV_TEMPLATES)
def test_rayjob_template_runtime_env_is_pinned(rel):
    """Every package a RayJob template installs at start is pinned, and none can replace Ray."""
    # The templates carry render placeholders (e.g. __SECRET_ENV_REFS__) that are
    # not YAML until rendered, so cut the literal block out of the text.
    lines = _read(rel).splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "runtimeEnvYAML: |") + 1
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = []
    for l in lines[start:]:
        if l.strip() and len(l) - len(l.lstrip()) < indent:
            break
        block.append(l[indent:])
    pip = yaml.safe_load("\n".join(block))["pip"]
    assert pip, f"{rel}: runtimeEnvYAML has no pip list"
    _assert_runtime_pins(rel, pip)


def _code_runtime_pip_lists():
    """Yield (location, [entries]) for every pip list a Python runtime_env is built from.

    Two shapes occur: a dict literal {"pip": [...]} and a default in
    <runtime_env>.get("pip", [...]). A list built at run time is reported as
    such, so it cannot hide from the check.
    """
    for root in CODE_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(errors="ignore"))
            except SyntaxError:
                continue
            rel = path.relative_to(ROOT).as_posix()
            for node in ast.walk(tree):
                lists = []
                if isinstance(node, ast.Dict):
                    lists += [v for k, v in zip(node.keys, node.values)
                              if isinstance(k, ast.Constant) and k.value == "pip"]
                elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "get" and len(node.args) == 2
                      and isinstance(node.args[0], ast.Constant) and node.args[0].value == "pip"):
                    lists.append(node.args[1])
                for v in lists:
                    where = f"{rel}:{v.lineno}"
                    if isinstance(v, (ast.List, ast.Tuple)) and all(isinstance(e, ast.Constant) for e in v.elts):
                        yield where, [e.value for e in v.elts]
                    elif not (isinstance(v, ast.Name)):
                        continue
                    else:
                        yield where, None   # a variable: checked where it is built


def test_runtime_envs_built_in_code_are_pinned():
    """Python code that builds a runtime_env pip list pins every entry to the app's version.

    sampling_agent_daft once connected with {"pip": ["getdaft", "ray"]}: every
    sampling job pip-installed the newest Ray into a cluster running another
    one, plus getdaft, a 0.5.0 shim that pulls whatever daft is newest.
    """
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    specs = {}
    for line in _read("requirements.txt").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            req = Requirement(line)
        except Exception:
            continue
        if req.marker is None or req.marker.evaluate():
            specs[canonicalize_name(req.name)] = req.specifier
    found = list(_code_runtime_pip_lists())
    assert any(w.startswith("app/agents/sampling_agent_daft.py") for w, _ in found), (
        f"the sampling runtime_env was not found; is the scan broken? {found}")
    literal = [(w, pip) for w, pip in found if pip is not None]
    for where, pip in literal:
        _assert_runtime_pins(where, pip)
        for entry in pip:
            name, version = entry.split("==")
            spec = specs.get(canonicalize_name(name))
            assert spec is None or spec.contains(version, prereleases=True), (
                f"{where}: {entry} is outside requirements.txt's {name}{spec}")


def test_no_requirements_file_lists_getdaft():
    """getdaft is the pre-rename name of daft; 0.5.0 is a shim that requires an unversioned daft.

    Listed beside daft>=0.7.1,<0.8 it looked like a second, older daft, and
    anywhere it is installed on its own it pulls daft past the app's cap.
    """
    offenders = []
    for rel, text in _source_files():
        if pathlib.Path(rel).name.startswith("requirements") or rel.endswith("pyproject.toml"):
            offenders += [f"{rel}: {l.strip()}" for l in text.splitlines()
                          if re.match(r"\s*[\"']?getdaft\b", l)]
    assert not offenders, f"list daft, not getdaft: {offenders}"
