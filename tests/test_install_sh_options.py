"""scripts/install.sh accepts every option it advertises.

Its usage block listed ``--doctor`` and ``--with-build-tools`` and the doctor
told users to run ``./scripts/install.sh --with-build-tools``; both died with
"unknown option" because the argument parser never had them. Nothing here
installs anything: the cases stop at ``--check`` / ``--doctor`` or source the
one helper they test.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "install.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _usage_options() -> set[str]:
    header = []
    for line in SCRIPT.read_text().splitlines()[1:]:
        if not line.startswith("#"):
            break
        header.append(line)
    # The option table is indented three spaces; prose that mentions a pip
    # flag (--prefer-binary) is not an option of this script.
    return set(re.findall(r"^#\s{3}(--[a-z][a-z-]+)", "\n".join(header), re.M))


def _run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True,
                          timeout=120, env={**os.environ, "PYTHON": sys.executable, **(env or {})})


def test_h_every_advertised_option_is_parsed():
    advertised = _usage_options()
    assert {"--doctor", "--with-build-tools", "--check", "--edition"} <= advertised
    parser = SCRIPT.read_text()
    missing = sorted(o for o in advertised if not re.search(rf"^\s*{re.escape(o)}\)", parser, re.M)
                     and not re.search(rf"\|{re.escape(o)}\)", parser))
    assert not missing, f"install.sh advertises {missing} but its parser rejects them"


def test_h_help_prints_the_whole_option_list():
    out = _run("--help")
    assert out.returncode == 0
    for opt in ("--doctor", "--with-build-tools", "AVALOKA_TORCH_INDEX"):
        assert opt in out.stdout


def test_h_doctor_runs_the_environment_check():
    out = _run("--doctor", "--no-venv")
    assert "unknown option" not in out.stderr, out.stderr
    assert "Avaloka environment check" in out.stdout, out.stdout + out.stderr


def test_h_with_build_tools_is_accepted():
    # --check exits before any install step, so this only proves the parse.
    out = _run("--with-build-tools", "--check", "--no-venv")
    assert "unknown option" not in out.stderr, out.stderr


def test_h_a_typo_is_still_refused():
    out = _run("--docter", "--no-venv")
    assert out.returncode != 0 and "unknown option: --docter" in out.stderr


def _torch_index(tmp_path: Path, *, uname: str, nvidia: bool, override: str | None = None) -> str:
    fake = tmp_path / "bin"
    fake.mkdir(parents=True, exist_ok=True)
    (fake / "uname").write_text(f"#!/bin/sh\necho {uname}\n")
    (fake / "uname").chmod(0o755)
    if nvidia:
        (fake / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
        (fake / "nvidia-smi").chmod(0o755)
    func = re.search(r"^torch_index_url\(\) \{.*?^\}", SCRIPT.read_text(), re.M | re.S).group(0)
    env = {"PATH": f"{fake}:/usr/bin:/bin"}
    if override is not None:
        env["AVALOKA_TORCH_INDEX"] = override
    out = subprocess.run(["bash", "-c", func + "\ntorch_index_url"], capture_output=True,
                         text=True, env=env, timeout=30)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_h_cpu_torch_index_only_on_linux_without_a_gpu(tmp_path):
    if Path("/proc/driver/nvidia/version").exists():
        pytest.skip("this host has an NVIDIA driver, which the helper reads directly")
    cpu = "https://download.pytorch.org/whl/cpu"
    assert _torch_index(tmp_path, uname="Linux", nvidia=False) == cpu
    assert _torch_index(tmp_path / "g", uname="Linux", nvidia=True) == ""
    assert _torch_index(tmp_path / "m", uname="Darwin", nvidia=False) == ""
    assert _torch_index(tmp_path / "o", uname="Linux", nvidia=False, override="") == ""
    assert _torch_index(tmp_path / "u", uname="Darwin", nvidia=False, override="http://x/") == "http://x/"


def test_h_the_cpu_torch_install_runs_before_the_requirements():
    text = SCRIPT.read_text()
    assert text.index('--index-url "$TORCH_INDEX"') < text.index("-r requirements.txt")
