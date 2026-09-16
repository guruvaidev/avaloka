#!/usr/bin/env python3
"""Find, size, and fetch local models — so 'gemma5 ships' is a command, not a PR.

The problem this solves: model families turn over faster than releases. When
gemma5 appears, an open-source developer on a laptop should not need to read
our source to know (a) that it exists, (b) which size their machine can hold,
and (c) how to get it. This tool answers all three:

    python scripts/ops/discover_models.py                    # what's new, what fits
    python scripts/ops/discover_models.py --family gemma     # one family
    python scripts/ops/discover_models.py --pull             # fetch the best fit
    python scripts/ops/discover_models.py --json             # for tooling

How it works, and the honest limits of each source:

* **Ollama local** (http://localhost:11434) — what you already have. Ground truth.
* **Ollama registry** — probed by tag (manifest HEAD request), because Ollama
  publishes no search API. We test candidate tags for known families; a brand
  new family becomes visible the moment its name is added to FAMILIES or passed
  via --family. This is deliberate: guessing family names is how tools invent
  models that do not exist.
* **OpenRouter catalogue** (public, no key needed for listing) — the broadest
  live index of what exists at all; new ids appear here within days of release.
  Used to DISCOVER new versions; serving them locally still goes through Ollama.

Sizing: a quantized (Q4) model needs roughly 0.7 GB of memory per billion
parameters, plus headroom for context. The tool reads the machine's memory and
recommends the largest variant that fits inside a budget (default: half of
RAM), stating the arithmetic — a recommendation whose reasoning is hidden is
just another thing to debug.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
UA = {"User-Agent": "avaloka-discover-models/1.0"}

#: Families we probe by default. Add a name here (or pass --family) when a new
#: line ships; everything else is automatic.
FAMILIES = ("gemma", "qwen", "llama", "deepseek", "phi", "mistral")

#: GB of memory per billion params at Q4-ish quantization, incl. overhead.
GB_PER_B = 0.7


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #

def _get_json(url: str, timeout: int = 30) -> Optional[Any]:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def local_models() -> List[Dict[str, Any]]:
    """What Ollama already serves on this machine."""
    data = _get_json(f"{OLLAMA_URL}/api/tags", timeout=5)
    if not data:
        return []
    return [{"name": m.get("name", ""),
             "size_gb": round((m.get("size") or 0) / 1e9, 1)}
            for m in data.get("models", [])]


def openrouter_catalogue(families: tuple) -> Dict[str, List[str]]:
    """Model ids per family, from the public OpenRouter listing.

    This is the discovery feed: when gemma5 ships, ids like
    'google/gemma-5-...' appear here without us changing anything.
    """
    data = _get_json(OPENROUTER_MODELS_URL, timeout=30)
    out: Dict[str, List[str]] = {f: [] for f in families}
    if not data:
        return out
    for m in data.get("data", []):
        mid = m.get("id", "")
        for fam in families:
            if re.search(rf"\b{fam}[-_]?\d", mid.lower()):
                out[fam].append(mid)
    return out


def ollama_tag_exists(tag: str, timeout: int = 20) -> bool:
    """Probe the Ollama registry for a tag without downloading it."""
    name, _, variant = tag.partition(":")
    url = f"https://registry.ollama.ai/v2/library/{name}/manifests/{variant or 'latest'}"
    req = urllib.request.Request(url, method="HEAD", headers={
        **UA, "Accept": "application/vnd.docker.distribution.manifest.v2+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError as exc:
        return exc.code not in (404,)
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #

def machine_memory_gb() -> float:
    override = os.getenv("AVALOKA_MEM_GB")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    try:  # macOS
        raw = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=5)
        return int(raw.stdout.strip()) / 1e9
    except Exception:  # noqa: BLE001
        pass
    try:  # Linux
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1e6
    except Exception:  # noqa: BLE001
        pass
    return 8.0   # conservative unknown


_PARAMS = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)


def params_b(model_id: str) -> Optional[float]:
    """Billions of params parsed from an id/tag. 'e4b'/'a4b' = active params."""
    active = re.search(r"[ea](\d+(?:\.\d+)?)b", model_id.lower())
    if active:
        return float(active.group(1))
    hits = _PARAMS.findall(model_id.lower())
    return float(hits[-1]) if hits else None


def fits(model_id: str, budget_gb: float) -> Optional[bool]:
    p = params_b(model_id)
    return None if p is None else (p * GB_PER_B) <= budget_gb


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def discover(families: tuple, *, budget_gb: Optional[float] = None) -> Dict[str, Any]:
    mem = machine_memory_gb()
    budget = budget_gb if budget_gb is not None else mem / 2
    have = local_models()
    have_names = {m["name"] for m in have}
    catalogue = openrouter_catalogue(families)

    report: Dict[str, Any] = {
        "machine_memory_gb": round(mem, 1),
        "budget_gb": round(budget, 1),
        "sizing_rule": f"{GB_PER_B} GB per B params (Q4 + headroom)",
        "local": have,
        "families": {},
    }
    for fam, ids in catalogue.items():
        versions = sorted({m.group(0) for i in ids
                           if (m := re.search(rf"{fam}[-_]?(\d+(?:\.\d+)?)", i.lower()))})
        fitting = [i for i in ids if fits(i, budget) is True]
        best = max(fitting, key=lambda i: params_b(i) or 0, default=None)
        report["families"][fam] = {
            "known_versions": versions,
            "catalogue_count": len(ids),
            "largest_that_fits": best,
            "fits_count": len(fitting),
        }
    report["already_serving"] = sorted(have_names)
    return report


def suggest_pull(report: Dict[str, Any], family: str) -> Optional[str]:
    """An Ollama tag worth pulling for *family*, verified to exist upstream."""
    info = report["families"].get(family) or {}
    budget = report["budget_gb"]
    versions = info.get("known_versions") or []
    newest = versions[-1] if versions else None
    if not newest:
        return None
    base = newest.replace("_", "").replace("-", "")   # e.g. gemma4
    for variant in ("27b", "e4b", "12b", "7b", "4b", "3b", "2b", "latest"):
        candidate = f"{base}:{variant}"
        p = params_b(candidate)
        if p is not None and p * GB_PER_B > budget:
            continue
        if ollama_tag_exists(candidate):
            return candidate
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", action="append", help="limit/add families")
    ap.add_argument("--budget-gb", type=float, help="memory budget override")
    ap.add_argument("--pull", action="store_true",
                    help="ollama pull the best verified fit for each family")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    families = tuple(args.family) if args.family else FAMILIES
    report = discover(families, budget_gb=args.budget_gb)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"machine: {report['machine_memory_gb']} GB · budget "
              f"{report['budget_gb']} GB ({report['sizing_rule']})")
        print(f"already serving: {', '.join(report['already_serving']) or 'nothing'}")
        for fam, info in report["families"].items():
            print(f"\n{fam}: versions seen {info['known_versions'] or '—'} "
                  f"({info['catalogue_count']} catalogue ids, "
                  f"{info['fits_count']} fit the budget)")
            if info["largest_that_fits"]:
                print(f"  largest hosted id that fits: {info['largest_that_fits']}")

    if args.pull:
        for fam in families:
            tag = suggest_pull(report, fam)
            if not tag:
                print(f"\n{fam}: no verified pullable tag within budget")
                continue
            print(f"\n{fam}: pulling {tag} …")
            rc = subprocess.run(["ollama", "pull", tag]).returncode
            if rc != 0:
                print(f"{fam}: pull failed (rc={rc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
