#!/usr/bin/env python3
"""Check every configured model against the provider's live catalogue.

Why this exists: on 2026-08-17 the entire Llama line disappeared from the
project's Groq account. ``llama-3.3-70b-versatile`` and
``llama-3.1-8b-instant`` both returned HTTP 404, and six agents — Coder,
Validator, Summarizer, Profiling, Visualization and the MTA task builder — were
still pointing at them. Nothing detected it, because a model id is only
validated when an agent actually calls it, mid-analysis, in front of a user.

Providers deprecate models on their own schedule. This turns that into a
startup check or a CI step instead of a production incident.

    python scripts/ops/verify_models.py                 # check Groq
    python scripts/ops/verify_models.py --json          # machine-readable
    python scripts/ops/verify_models.py --probe         # also send 1 token

Exit codes: 0 all good · 1 a configured model is missing · 2 cannot reach the
provider (network/credentials), which is deliberately distinct from "the model
is gone".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.model_config import DEFAULT_MODELS, describe_all  # noqa: E402

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
#: Groq rejects urllib's default user-agent with HTTP 403, so send an explicit
#: one. Found the hard way: the same request succeeded under curl.
USER_AGENT = "avaloka-verify-models/1.0"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

#: Checked in order; the first one set wins. The generic key is the fallback
#: because per-agent keys may be scoped to a subset of models.
GROQ_KEY_VARS = (
    "GROQ_API_KEY",
    "GROQ_API_KEY_GENERIC",
    "GROQ_API_KEY_PLANNING_AGENT",
    "GROQ_API_KEY_CODING_AGENT",
)

GREEN, RED, YEL, DIM, BOLD, OFF = (
    ("\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")
    if sys.stdout.isatty() else ("",) * 6
)


def _api_key() -> Optional[str]:
    for var in GROQ_KEY_VARS:
        value = (os.getenv(var) or "").strip()
        if value:
            return value
    return None


def fetch_catalogue(api_key: str, *, timeout: int = 30) -> List[str]:
    """Model ids the provider currently serves. Raises on transport failure."""
    request = urllib.request.Request(
        GROQ_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return sorted(m["id"] for m in payload.get("data", []) if m.get("id"))


def probe(model: str, api_key: str, *, timeout: int = 45) -> Tuple[bool, str]:
    """Send a one-token completion. Catches access issues a listing misses."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "ok"}],
        "max_tokens": 1,
    }).encode()
    request = urllib.request.Request(
        GROQ_CHAT_URL, data=body,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json",
                 "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True, "ok"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:160]
        return False, f"HTTP {exc.code}: {detail}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--probe", action="store_true",
                        help="also send a 1-token completion per model")
    args = parser.parse_args(argv)

    key = _api_key()
    if not key:
        print(f"{RED}No Groq API key found.{OFF} Set one of: {', '.join(GROQ_KEY_VARS)}",
              file=sys.stderr)
        return 2

    try:
        catalogue = fetch_catalogue(key)
    except Exception as exc:  # noqa: BLE001
        print(f"{RED}Could not reach Groq:{OFF} {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    configured = describe_all()
    results: Dict[str, Dict[str, object]] = {}
    missing: List[str] = []

    for agent, info in sorted(configured.items()):
        model = str(info["model"])
        listed = model in catalogue
        entry: Dict[str, object] = {
            "model": model,
            "source": info["source"],
            "in_catalogue": listed,
        }
        if args.probe and listed:
            ok, detail = probe(model, key)
            entry["probe_ok"] = ok
            entry["probe_detail"] = detail
            if not ok:
                missing.append(agent)
        elif not listed:
            missing.append(agent)
        results[agent] = entry

    if args.json:
        print(json.dumps({
            "catalogue_size": len(catalogue),
            "catalogue": catalogue,
            "agents": results,
            "missing": missing,
        }, indent=2))
        return 1 if missing else 0

    print(f"{BOLD}Groq catalogue:{OFF} {len(catalogue)} model(s)")
    for model in catalogue:
        print(f"  {DIM}{model}{OFF}")
    print(f"\n{BOLD}Configured agents{OFF}")
    width = max(len(a) for a in results)
    for agent, entry in results.items():
        ok = entry["in_catalogue"] and entry.get("probe_ok", True)
        mark = f"{GREEN}ok {OFF}" if ok else f"{RED}GONE{OFF}"
        note = "" if ok else f"  {RED}<- not served by this account{OFF}"
        src = "" if entry["source"] == "default" else f"  {DIM}({entry['source']}){OFF}"
        print(f"  {mark} {agent:<{width}}  {entry['model']}{src}{note}")

    if missing:
        print(f"\n{RED}{BOLD}{len(missing)} agent(s) point at a model this account "
              f"cannot serve:{OFF} {', '.join(missing)}")
        print("Set the matching AVALOKA_<AGENT>_MODEL env var, or update "
              "DEFAULT_MODELS in app/core/model_config.py.")
        return 1

    print(f"\n{GREEN}All {len(results)} configured models are available.{OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
