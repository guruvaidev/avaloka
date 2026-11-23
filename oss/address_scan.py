#!/usr/bin/env python3
"""Fail when a tree names a publicly routable IPv4 address.

    python3 oss/address_scan.py <root> [oss/manifest.yaml] [--git]

Reports, as ``path:line address``, every IPv4 literal that is publicly
routable (``ipaddress`` ``is_global``) and not on the ``allow`` list of the
manifest's ``assert_no_addresses:`` block. What passes without being listed:
loopback (127.0.0.1), 0.0.0.0, the RFC 5737 documentation ranges
(192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24), the RFC 1918 private ranges
and the Kubernetes cluster-local defaults (10.96.0.0/12), none of which route.
Examples and tests use those.

It exists because app/data_transfer_docker/Dockerfile.train-plus-dta named the
shared GKE Ray cluster's dashboard IP:port in a comment and reached the public
tree. scripts/generate-oss.sh runs it on the generated tree; with ``--git`` it
scans the tracked files of a checkout (tests/test_oss_no_addresses.py).
Stdlib only, like the generator.
"""

from __future__ import annotations

import ipaddress
import pathlib
import re
import subprocess
import sys

_IPV4 = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.]|\w)")
_SKIP_DIRS = {".git", "node_modules"}


def allowlist(manifest: str) -> set:
    blk = re.search(r"^assert_no_addresses:\n((?:(?![a-z_]+:).*\n)*)", manifest, re.M)
    if not blk:
        return set()
    return {m.group(2) for m in re.finditer(r"^    - *(['\"])(.*?)\1", blk.group(1), re.M)}


def scan_text(text: str, allow=frozenset()) -> list:
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        for m in _IPV4.finditer(line):
            try:
                ip = ipaddress.ip_address(m.group(1))
            except ValueError:
                continue  # 34.123.456.789, or a version string with a leading zero
            if ip.is_global and str(ip) not in allow:
                hits.append((n, str(ip)))
    return hits


def _files(root: pathlib.Path, use_git: bool):
    if use_git:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
        for rel in out.decode().split("\0"):
            if rel:
                yield root / rel
        return
    for f in root.rglob("*"):
        if f.is_file() and not (_SKIP_DIRS & set(f.relative_to(root).parts)):
            yield f


def scan_tree(root: pathlib.Path, allow=frozenset(), use_git: bool = False) -> list:
    found = []
    for f in _files(root, use_git):
        try:
            text = f.read_text(errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        for n, ip in scan_text(text, allow):
            found.append(f"{f.relative_to(root)}:{n} {ip}")
    return found


def main(argv) -> int:
    args = [a for a in argv if not a.startswith("--")]
    root = pathlib.Path(args[0])
    manifest = pathlib.Path(args[1] if len(args) > 1 else root / "oss" / "manifest.yaml")
    hits = scan_tree(root, allowlist(manifest.read_text()), use_git="--git" in argv)
    print("== address scan ==")
    for h in hits[:50]:
        print(f"   FAIL {h}")
    print(f"   {'FAIL' if hits else 'ok  '} no publicly routable IPv4 address"
          + (f" ({len(hits)})" if hits else ""))
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
