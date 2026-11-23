"""No publicly routable IPv4 address in the repository, and so none in the public tree.

app/data_transfer_docker/Dockerfile.train-plus-dta named the shared GKE Ray
cluster's dashboard IP:port in its first line, and the public tree carried it,
with other cluster and load-balancer addresses in docstrings and tests. The
rule lives in oss/manifest.yaml (assert_no_addresses); scripts/generate-oss.sh
refuses to generate a tree that breaks it, and this test applies the same rule
to every tracked file, so a leak fails here before it reaches the generator.
"""
from __future__ import annotations

import importlib.util
import pathlib
import shutil

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCANNER = REPO / "oss" / "address_scan.py"
MANIFEST = REPO / "oss" / "manifest.yaml"


def _scanner():
    assert SCANNER.is_file(), "no address scan (oss/address_scan.py)"
    spec = importlib.util.spec_from_file_location("oss_address_scan", SCANNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _allow(mod):
    assert "assert_no_addresses:" in MANIFEST.read_text(), "oss/manifest.yaml has no assert_no_addresses"
    return mod.allowlist(MANIFEST.read_text())


def _j(*parts):
    """Joined at runtime, so this file does not itself carry the addresses it tests."""
    return "".join(parts)


@pytest.mark.parametrize("text", [
    _j("# Merged image for the SHARED Ray cluster (dashboard 34.132", ".134.104:8265)."),
    _j("e.g. http://136.119", ".229.5:8265"),
    _j('print(f"   Check Ray dashboard at: http://34.30', '.130.231:8265")'),
    _j('When it is a raw IPv4 address (e.g. ``"35.188', '.70.201"``)'),
], ids=["dockerfile", "ray-dashboard-doc", "ray-dashboard-test", "load-balancer"])
def test_the_scan_catches_each_leak_that_happened(text):
    mod = _scanner()
    assert mod.scan_text(text, _allow(mod)), f"not caught: {text}"


@pytest.mark.parametrize("text", [
    "http://127.0.0.1:9000", "0.0.0.0:8080", "http://203.0.113.10:8265", "198.51.100.7", "192.0.2.9",
    "redis://10.0.0.5:6379", "192.168.1.7", "172.16.4.2", "10.96.0.10",
    "image: supabase/postgres:15.8.1.060", _j("1.2", ".3.4.5"), "34.123.456.789",
], ids=["loopback", "any", "doc-1", "doc-2", "doc-3", "rfc1918-10", "rfc1918-192", "rfc1918-172",
        "kube-dns", "version", "five-octets", "not-an-address"])
def test_the_scan_passes_what_does_not_route(text):
    mod = _scanner()
    assert not mod.scan_text(text, _allow(mod)), f"flagged: {text}"


def test_no_tracked_file_names_a_routable_address():
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    mod = _scanner()
    hits = mod.scan_tree(REPO, _allow(mod), use_git=True)
    assert not hits, "publicly routable IPv4 addresses in tracked files:\n" + "\n".join(hits)
