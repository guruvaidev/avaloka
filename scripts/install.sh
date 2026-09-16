#!/usr/bin/env bash
#
# Avaloka installer.
#
# Open source and commercial editions install through the SAME path. The only
# difference is one extra package:
#
#   open source   avaloka + its dependencies, from PyPI / this checkout
#   commercial    the above, plus `avaloka-commercial` from Avaloka's private
#                 index, authenticated with your licence-scoped token
#
# That is the whole delivery model, and it is deliberate: the commercial code
# is a separate distribution that the open-source install can never pull. There
# is no flag to flip and no build variant to choose. If `avaloka-commercial` is
# not installed, the commercial capabilities are not on disk.
#
# Usage:
#   ./scripts/install.sh                              # open source
#   ./scripts/install.sh --edition professional \
#       --license-key "$AVALOKA_LICENSE_KEY"          # commercial
#
#   --edition        oss | professional | enterprise   (default: oss)
#   --license-key    licence token; required for commercial editions
#   --venv PATH      virtualenv location               (default: .venv)
#   --index-url URL  private index for the commercial package
#   --no-venv        install into the active environment
#   --check          verify an existing install and exit
#   --with-build-tools  install a Rust toolchain before installing packages.
#                       Not normally needed: see "Wheels, not builds" below.
#   --doctor         run the environment check and exit
#
# Wheels, not builds
# ------------------
# Dependencies install with --prefer-binary, which is the difference between a
# one-command install and a compiler error. pip otherwise picks the NEWEST
# version of a package; when that version has no wheel for your platform it
# quietly falls back to the source distribution. For `cryptography` that means
# a Rust build needing rustc >= 1.83, and a failure many pages from its cause.
# --prefer-binary picks the newest version that actually ships a wheel, so no
# toolchain is needed at all.
#
set -euo pipefail

WITH_BUILD_TOOLS=0
DOCTOR_ONLY=0
EDITION="oss"
LICENSE_KEY="${AVALOKA_LICENSE_KEY:-}"
VENV_PATH=".venv"
INDEX_URL="${AVALOKA_INDEX_URL:-https://pypi.avaloka.ai/simple}"
USE_VENV=1
CHECK_ONLY=0
MIN_PY_MINOR=10

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; OFF=$'\033[0m'
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s✓%s %s\n' "$GREEN" "$OFF" "$*"; }
warn() { printf '%s!%s %s\n' "$YELLOW" "$OFF" "$*"; }
die()  { printf '%s✗%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --edition)     EDITION="${2:?--edition needs a value}"; shift 2 ;;
    --license-key) LICENSE_KEY="${2:?--license-key needs a value}"; shift 2 ;;
    --venv)        VENV_PATH="${2:?--venv needs a path}"; shift 2 ;;
    --index-url)   INDEX_URL="${2:?--index-url needs a URL}"; shift 2 ;;
    --no-venv)     USE_VENV=0; shift ;;
    --check)       CHECK_ONLY=1; shift ;;
    -h|--help)     sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)             die "unknown option: $1 (try --help)" ;;
  esac
done

case "$EDITION" in
  oss|professional|enterprise) ;;
  *) die "unknown edition: $EDITION (expected oss, professional, or enterprise)" ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
say "${BOLD}Avaloka installer${OFF} — edition: ${BOLD}${EDITION}${OFF}"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
  done
fi
[[ -n "$PYTHON" ]] || die "no python3 found on PATH; install Python 3.${MIN_PY_MINOR}+ first"

PY_MAJOR=$("$PYTHON" -c 'import sys; print(sys.version_info[0])')
PY_MINOR=$("$PYTHON" -c 'import sys; print(sys.version_info[1])')
if [[ "$PY_MAJOR" -ne 3 || "$PY_MINOR" -lt "$MIN_PY_MINOR" ]]; then
  # Caught here on purpose: `python` still resolves to 2.7 on many macOS setups,
  # and the failure that produces further downstream is deeply unhelpful.
  die "Python 3.${MIN_PY_MINOR}+ required, found $("$PYTHON" -V 2>&1). Set PYTHON=/path/to/python3"
fi
ok "using $("$PYTHON" -V 2>&1) at $(command -v "$PYTHON")"

if [[ "$EDITION" != "oss" && -z "$LICENSE_KEY" && "$CHECK_ONLY" -eq 0 ]]; then
  die "edition '$EDITION' needs --license-key (or AVALOKA_LICENSE_KEY). Contact the Avaloka team to obtain one."
fi

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
if [[ "$USE_VENV" -eq 1 ]]; then
  if [[ ! -d "$VENV_PATH" ]]; then
    "$PYTHON" -m venv "$VENV_PATH" || die "could not create a virtualenv at $VENV_PATH"
    ok "created virtualenv at $VENV_PATH"
  else
    ok "reusing virtualenv at $VENV_PATH"
  fi
  VENV_PY="$VENV_PATH/bin/python"
  [[ -x "$VENV_PY" ]] || VENV_PY="$VENV_PATH/Scripts/python.exe"   # Windows
  [[ -x "$VENV_PY" ]] || die "virtualenv at $VENV_PATH looks broken"
else
  VENV_PY="$PYTHON"
  warn "installing into the active environment (--no-venv)"
fi

if [[ "$DOCTOR_ONLY" -eq 1 ]]; then
  exec "$VENV_PY" "$REPO_ROOT/scripts/doctor.py"
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  exec "$VENV_PY" "$REPO_ROOT/scripts/check_edition.py"
fi

# --------------------------------------------------------------------------- #
# Build tools — opt in, and almost never needed
# --------------------------------------------------------------------------- #
if [[ "$WITH_BUILD_TOOLS" -eq 1 ]]; then
  say ""
  say "${BOLD}Installing build tools${OFF}"
  if command -v rustc >/dev/null 2>&1; then
    ok "rust already present ($(rustc --version))"
  elif command -v rustup >/dev/null 2>&1; then
    rustup toolchain install stable >/dev/null 2>&1 && ok "installed the stable rust toolchain via rustup" \
      || warn "rustup could not install a toolchain; continuing without one"
  else
    say "  fetching rustup from https://sh.rustup.rs"
    if curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path >/dev/null 2>&1; then
      export PATH="$HOME/.cargo/bin:$PATH"
      ok "installed rust ($(rustc --version 2>/dev/null || echo 'restart your shell to use it'))"
    else
      warn "could not install rust automatically" \
        "Install it yourself from https://rustup.rs, then re-run. Note that the
    install below does not need it -- --prefer-binary avoids source builds."
    fi
  fi
fi

# --------------------------------------------------------------------------- #
# Open-source dependencies
# --------------------------------------------------------------------------- #
say ""
say "${BOLD}Installing open-source dependencies${OFF}"
"$VENV_PY" -m pip install --quiet --upgrade pip setuptools wheel \
  || die "could not upgrade pip"
if [[ -f requirements.txt ]]; then
  # --prefer-binary is load-bearing. See "Wheels, not builds" at the top.
  if "$VENV_PY" -m pip install --quiet --prefer-binary -r requirements.txt; then
    ok "installed from requirements.txt (wheels preferred)"
  else
    warn "the quiet install failed; retrying so you can see why"
    PIP_LOG="$(mktemp)"
    "$VENV_PY" -m pip install --prefer-binary -r requirements.txt 2>&1 | tee "$PIP_LOG"
    if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
      say ""
      if grep -qiE "rustc|cargo|maturin" "$PIP_LOG"; then
        die "a package fell back to building from source because no wheel matched
    your platform, and the build needs a Rust toolchain. Two ways out, in order
    of preference:

      1. Use a Python version with wider wheel coverage -- 3.11 is what CI and
         the shipped image use:   PYTHON=python3.11 ./scripts/install.sh
      2. Install a toolchain and let it compile:
                                  ./scripts/install.sh --with-build-tools"
      elif grep -qiE "No matching distribution|Could not find a version" "$PIP_LOG"; then
        die "a pinned dependency has no release for this platform or Python version.
    The line above beginning 'ERROR: Could not find a version' names it, and
    the 'from versions:' list shows what does exist here.

    Most often this is a Python version with thin wheel coverage. Try:
      PYTHON=python3.11 ./scripts/install.sh

    If a pin in requirements.txt is genuinely unsatisfiable on your platform,
    that is a bug in Avaloka -- please open an issue quoting that ERROR line."
      else
        die "dependency install failed. The pip output above says why.
    Run ./scripts/doctor.py once resolved to confirm the result."
      fi
    fi
    rm -f "$PIP_LOG"
  fi
fi
if [[ -f pyproject.toml || -f setup.py ]]; then
  "$VENV_PY" -m pip install --quiet -e . || die "editable install failed"
  ok "installed avaloka (editable)"
fi

# --------------------------------------------------------------------------- #
# Commercial package — the only difference between editions
# --------------------------------------------------------------------------- #
if [[ "$EDITION" != "oss" ]]; then
  say ""
  say "${BOLD}Installing commercial package${OFF} (${EDITION})"
  # The token authenticates to the private index AND scopes what it may pull.
  # Passed via env so it never lands in shell history or the process list.
  if AVALOKA_LICENSE_KEY="$LICENSE_KEY" "$VENV_PY" -m pip install --quiet \
        --index-url "https://__token__:${LICENSE_KEY}@${INDEX_URL#https://}" \
        "avaloka-commercial"; then
    ok "installed avaloka-commercial from the private index"
  else
    die "could not install avaloka-commercial. Check the licence key and that
    your network can reach ${INDEX_URL}. The open-source install above is
    complete and usable; re-run with --edition ${EDITION} once resolved."
  fi

  LICENSE_FILE="${AVALOKA_HOME:-$HOME/.avaloka}/license"
  mkdir -p "$(dirname "$LICENSE_FILE")"
  umask 077
  printf '%s\n' "$LICENSE_KEY" > "$LICENSE_FILE"
  chmod 600 "$LICENSE_FILE"
  ok "licence stored at $LICENSE_FILE (mode 600)"
fi

# --------------------------------------------------------------------------- #
# Verify
# --------------------------------------------------------------------------- #
say ""
"$VENV_PY" "$REPO_ROOT/scripts/check_edition.py" || die "post-install check failed"

# The edition report says what you are licensed for. The doctor says whether
# the environment will actually run -- a broken numpy/torch pairing passes the
# first and fails the second, which is why both run.
say ""
if ! "$VENV_PY" "$REPO_ROOT/scripts/doctor.py"; then
  die "the environment check found problems. Each is listed above with a fix.
    Re-run it any time with:  ./scripts/install.sh --doctor"
fi

say ""
ok "${BOLD}Install complete.${OFF}"
if [[ "$USE_VENV" -eq 1 ]]; then
  say "  Activate with:  source $VENV_PATH/bin/activate"
fi
if [[ "$EDITION" == "oss" ]]; then
  say "  Self-hosted open source: bring your own Ray cluster, cloud credentials"
  say "  and LLM keys. You run it, you pay for it, you carry the risk."
  say "  Docs: docs/INSTALL.md · Editions: docs/EDITIONS.md"
fi
