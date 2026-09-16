#!/usr/bin/env python3
"""Report the resolved edition and capability set for this install.

Run this to answer "why can't I do X?" without guessing, and paste its output
into a support request. It reads the environment and the installed packages; it
never contacts Avaloka and never prints the licence token.

    python scripts/check_edition.py
    ./scripts/install.sh --check
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Import from the checkout regardless of how this is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from app.core.editions import (GATE_REASON, GateReason, commercial_build,
                                   normalize_deployment, normalize_edition,
                                   resolve_capabilities)
except ImportError as exc:  # pragma: no cover - only when run outside the repo
    print(f"cannot import app.core.editions: {exc}", file=sys.stderr)
    print("Run this from the Avaloka checkout, after installing.", file=sys.stderr)
    raise SystemExit(2)


GREEN, RED, DIM, BOLD, OFF = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = DIM = BOLD = OFF = ""


def _license_token() -> str | None:
    """Read the licence token, from the environment or the stored file."""
    token = os.getenv("AVALOKA_LICENSE_KEY")
    if token:
        return token.strip()
    path = Path(os.getenv("AVALOKA_HOME", Path.home() / ".avaloka")) / "license"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def main() -> int:
    deployment = normalize_deployment(os.getenv("AVALOKA_DEPLOYMENT"))
    commercial = commercial_build()

    token = _license_token()
    edition = normalize_edition(os.getenv("AVALOKA_EDITION"))
    license_note = "none"
    if token:
        # Resolve the edition from the licence rather than trusting the env var.
        try:
            from app.core.license import verify_license
            claims = verify_license(token)
            edition = claims.edition
            license_note = (f"valid — {claims.organization}, {claims.seats} seats, "
                            f"expires {claims.expires_at}")
        except Exception as exc:  # noqa: BLE001 - report, never crash the check
            license_note = f"{RED}rejected{OFF} ({type(exc).__name__}: {exc})"
            edition = "oss"

    caps = resolve_capabilities(deployment, edition, has_commercial_build=commercial)

    print(f"{BOLD}Avaloka edition report{OFF}")
    print(f"  deployment        {deployment}")
    print(f"  edition           {edition}")
    print(f"  commercial build  {'yes' if commercial else 'no'} "
          f"{DIM}(avaloka_commercial {'installed' if commercial else 'not installed'}){OFF}")
    print(f"  licence           {license_note}")
    print(f"  python            {sys.version.split()[0]}")

    print(f"\n{BOLD}Capabilities{OFF}")
    width = max(len(name) for name in GATE_REASON)
    for name in sorted(GATE_REASON):
        on = getattr(caps, name)
        mark = f"{GREEN}on {OFF}" if on else f"{RED}off{OFF}"
        reason = GATE_REASON[name]
        if on:
            note = ""
        elif reason is GateReason.COMMERCIAL and not commercial:
            note = f"{DIM}not in this build — Professional/Enterprise{OFF}"
        elif reason is GateReason.COST:
            note = f"{DIM}hosted plan limit — self-host or upgrade{OFF}"
        else:
            note = f"{DIM}{reason.value}{OFF}"
        print(f"  {mark}  {name:<{width}}  {note}")

    if deployment == "self_hosted" and not commercial:
        print(f"\n{DIM}Self-hosted open source: you bring the Ray cluster, cloud"
              f" credentials and LLM keys.\nNothing here is rate-limited by"
              f" Avaloka — you pay for the compute and carry the risk.{OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
