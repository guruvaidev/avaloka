#!/usr/bin/env python3
"""Check that an Avaloka install is actually usable, and say what to do if not.

Run this any time:

    python scripts/doctor.py

``scripts/install.sh`` runs it at the end, so a broken environment is reported
at install time rather than discovered later inside a traceback.

Every check answers one question a user would otherwise have to answer by
reading a stack trace. The two that matter most are invisible to `pip list`:

  * **A source build that wanted a Rust toolchain.** pip prefers the newest
    version of a package. When the newest `cryptography` has no wheel for your
    platform, pip quietly falls back to the sdist, which needs Rust >= 1.83 and
    fails with a compiler error many pages from its cause. Installing with
    ``--prefer-binary`` picks the newest version that *has* a wheel instead, so
    no toolchain is needed at all.

  * **numpy and torch disagreeing about their ABI.** torch wheels are compiled
    against a specific numpy generation. An unbounded ``numpy>=1.24`` resolves
    to 2.x, and a torch built against 1.x then fails at import with
    "Failed to initialize NumPy: _ARRAY_API not found" -- as a *warning*, after
    which things break later and elsewhere.
"""
from __future__ import annotations

import contextlib
import importlib
import platform
import subprocess
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

GREEN, YELLOW, RED, BOLD, OFF = "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = YELLOW = RED = BOLD = OFF = ""

_failures: list[str] = []
_warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{OFF} {msg}")


def warn(msg: str, fix: str = "") -> None:
    print(f"  {YELLOW}!{OFF} {msg}")
    if fix:
        print(f"      {fix}")
    _warnings.append(msg)


def bad(msg: str, fix: str = "") -> None:
    print(f"  {RED}✗{OFF} {msg}")
    if fix:
        print(f"      {fix}")
    _failures.append(msg)


# --------------------------------------------------------------------------- #

def check_python() -> None:
    major, minor = sys.version_info[:2]
    label = f"Python {major}.{minor}.{sys.version_info[2]} ({platform.machine()}, {platform.system()})"
    if major == 3 and 10 <= minor <= 12:
        ok(label)
    elif major == 3 and minor > 12:
        warn(f"{label} — untested above 3.12",
             "Some dependencies have no wheel yet for this version. "
             "3.11 is what CI and the shipped image use.")
    else:
        bad(f"{label} — Avaloka needs Python 3.10 to 3.12",
            "Re-run the installer as: PYTHON=/path/to/python3.11 ./scripts/install.sh")


def check_architecture() -> None:
    """Catch an x86_64 Python running under Rosetta on Apple Silicon.

    This one is worth its own check because the failure it causes looks like
    nothing else: Rosetta 2 does not implement AVX, so a wheel compiled with it
    -- Daft's, among others -- does not raise ImportError. It raises SIGILL.
    The process dies with "Fatal Python error: Illegal instruction", the
    corpse sits in uninterruptible exit where no signal can reach it, and
    anything waiting on it looks like a hang rather than a crash.

    Nothing in Avaloka can work around that. Installing a native arm64 Python
    makes it go away entirely, so the useful thing is to say so.
    """
    if platform.system() != "Darwin":
        return
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        translated = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(translated))
        rc = libc.sysctlbyname(b"sysctl.proc_translated",
                               ctypes.byref(translated), ctypes.byref(size), None, 0)
    except Exception:                                   # noqa: BLE001
        return
    if rc != 0 or not translated.value:
        return

    bad(
        "this Python is x86_64 running under Rosetta on an Apple Silicon Mac",
        "Some dependencies -- Daft among them -- ship x86_64 wheels built with\n"
        "      AVX instructions that Rosetta does not implement. They do not fail to\n"
        "      import; they crash the process with SIGILL, and the crash looks like a\n"
        "      hang. Install a native arm64 Python and rebuild the environment:\n"
        "        arch -arm64 /opt/homebrew/bin/brew install python@3.11\n"
        "        rm -rf .venv && PYTHON=/opt/homebrew/bin/python3.11 ./scripts/install.sh",
    )


def check_core_imports() -> None:
    """The packages without which nothing works."""
    required = {
        "pandas": "tabular data",
        "numpy": "numerics",
        "sklearn": "modelling",
        "pyarrow": "parquet",
        "typer": "the CLI",
        "rich": "terminal output",
        "yaml": "config",
        "jinja2": "report templates",
    }
    missing = []
    for module, why in required.items():
        try:
            importlib.import_module(module)
        except Exception as exc:                       # noqa: BLE001 - report anything
            missing.append(f"{module} ({why}): {type(exc).__name__}")
    if missing:
        bad("core packages missing or broken: " + "; ".join(missing),
            "Run ./scripts/install.sh, or: pip install --prefer-binary -r requirements.txt")
    else:
        ok(f"core packages import cleanly ({len(required)} checked)")


def check_numpy_torch_abi() -> None:
    """The mismatch that reports itself as a warning and breaks something later."""
    try:
        import numpy
    except Exception:
        return                                          # already reported above
    # torch prints its own wall of text to stderr before raising, which buries
    # the one line that says what to do. Capture it; we report the diagnosis.
    try:
        with _quiet():
            import torch
    except ModuleNotFoundError:
        warn("torch is not installed — model training is unavailable",
             "Optional. Install with: pip install --prefer-binary torch")
        return
    except Exception as exc:                            # noqa: BLE001
        bad(f"torch is installed but will not import: {type(exc).__name__}: {exc}")
        return

    try:
        with _quiet():
            import torch.nn                             # noqa: F401 - the import that fails
            arr = torch.from_numpy(numpy.zeros(1, dtype="float32"))
            _ = arr.numpy()
        ok(f"numpy {numpy.__version__} and torch {torch.__version__} agree on their ABI")
    except Exception as exc:                            # noqa: BLE001
        bad(f"torch {torch.__version__} was built against a different numpy than "
            f"the installed {numpy.__version__}: {type(exc).__name__}",
            "Pin them together. On this platform the usual fix is:\n"
            '      pip install --prefer-binary "numpy<2"\n'
            "      ...or upgrade torch to a build that supports numpy 2.")


def check_no_source_builds_needed() -> None:
    """Report packages that would compile rather than install a wheel."""
    try:
        import cryptography
        ok(f"cryptography {cryptography.__version__} (wheel — no Rust toolchain needed)")
    except ModuleNotFoundError:
        warn("cryptography is not installed — the API server and cloud connectors need it",
             "pip install --prefer-binary cryptography\n"
             "      --prefer-binary matters: without it pip picks the newest version,\n"
             "      finds no wheel, and falls back to a source build that needs Rust.")
    except Exception as exc:                            # noqa: BLE001
        bad(f"cryptography will not import: {type(exc).__name__}: {exc}")


def check_build_tools() -> None:
    """Informational: what a from-source build would find if one were needed."""
    rust = _version_of(["rustc", "--version"])
    if rust:
        ok(f"rust toolchain present ({rust}) — source builds available if ever needed")
    else:
        print(f"  {BOLD}·{OFF} no rust toolchain — not required, "
              f"install with ./scripts/install.sh --with-build-tools if you want one")


def check_cli() -> None:
    try:
        from avaloka.version import __version__
    except Exception as exc:                            # noqa: BLE001
        bad(f"the avaloka package will not import: {type(exc).__name__}: {exc}",
            "Run: pip install -e .")
        return
    declared = (REPO_ROOT / "VERSION").read_text().strip() if (REPO_ROOT / "VERSION").exists() else ""
    if declared and declared != __version__:
        bad(f"version mismatch: VERSION says {declared}, package says {__version__}")
    else:
        ok(f"avaloka {__version__}")


@contextlib.contextmanager
def _quiet():
    """Silence a library that writes diagnostics we are about to replace."""
    import io
    import os

    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
        with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def _version_of(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def main() -> int:
    print(f"\n{BOLD}Avaloka environment check{OFF}\n")
    check_python()
    check_architecture()
    check_core_imports()
    check_cli()
    check_no_source_builds_needed()
    check_numpy_torch_abi()
    check_build_tools()

    print()
    if _failures:
        print(f"{RED}{BOLD}{len(_failures)} problem(s) will stop Avaloka from working.{OFF}")
        print("Each is listed above with what to do about it.")
        return 1
    if _warnings:
        print(f"{YELLOW}Usable, with {len(_warnings)} optional thing(s) missing.{OFF}")
        return 0
    print(f"{GREEN}{BOLD}Everything checks out.{OFF}")
    print("  Try:  avaloka analyze <your.csv> --goal \"what should I look at?\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
