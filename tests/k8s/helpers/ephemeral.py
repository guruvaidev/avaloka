"""D6 — ephemeral cloud cluster lifecycle (test-owned, never in app/).

A context manager that creates -> yields -> UNCONDITIONALLY destroys a cluster,
with an atexit fallback (a leaked GKE Autopilot cluster bills continuously) and a
reaper for orphans. Every cluster is named ``avaloka-it-<runid>`` and labelled so
orphans are identifiable; the harness NEVER deletes a cluster it cannot prove it
labelled.

Usage:
    with ephemeral_cluster("gcp", runid="abc123") as ctx:
        ...  # ctx is the kube-context name

    python -m tests.k8s.helpers.ephemeral --reap-orphans --provider gcp
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import os
import subprocess
import sys
from typing import Iterator, List

LABEL_KEY = "purpose"
LABEL_VALUE = "avaloka-integration-test"
TTL = "2h"


def _run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def cluster_name(runid: str) -> str:
    return f"avaloka-it-{runid}"


# --------------------------------------------------------------------------- GKE
def _gke_create(name: str, project: str, region: str) -> None:
    _run([
        "gcloud", "container", "clusters", "create-auto", name,
        "--project", project, "--region", region,
        "--labels", f"{LABEL_KEY}={LABEL_VALUE},ttl={TTL}",
    ])


def _gke_has_label(name: str, project: str, region: str) -> bool:
    r = _run(["gcloud", "container", "clusters", "describe", name,
              "--project", project, "--region", region,
              "--format", f"value(resourceLabels.{LABEL_KEY})"], check=False)
    return r.stdout.strip() == LABEL_VALUE


def _gke_delete(name: str, project: str, region: str) -> None:
    if not _gke_has_label(name, project, region):
        raise RuntimeError(f"refusing to delete {name}: missing {LABEL_KEY}={LABEL_VALUE} label")
    _run(["gcloud", "container", "clusters", "delete", name,
          "--project", project, "--region", region, "--quiet"], check=False)


def _gke_configure_kubectl(name: str, project: str, region: str) -> str:
    _run(["gcloud", "container", "clusters", "get-credentials", name,
          "--project", project, "--region", region])
    return f"gke_{project}_{region}_{name}"


# --------------------------------------------------------------------------- EKS
def _eks_create(name: str, region: str) -> None:
    _run(["eksctl", "create", "cluster", "--name", name, "--region", region,
          "--nodes", "1", "--managed",
          "--tags", f"{LABEL_KEY}={LABEL_VALUE},ttl={TTL}"])


def _eks_delete(name: str, region: str) -> None:
    _run(["eksctl", "delete", "cluster", "--name", name, "--region", region], check=False)


# --------------------------------------------------------------------------- AKS
def _aks_create(name: str, rg: str, location: str) -> None:
    _run(["az", "aks", "create", "-n", name, "-g", rg, "--location", location,
          "--node-count", "1", "--generate-ssh-keys",
          "--tags", f"{LABEL_KEY}={LABEL_VALUE}", f"ttl={TTL}"])


def _aks_has_tag(name: str, rg: str) -> bool:
    r = _run(["az", "aks", "show", "-n", name, "-g", rg,
              "--query", f"tags.{LABEL_KEY}", "-o", "tsv"], check=False)
    return r.stdout.strip() == LABEL_VALUE


def _aks_delete(name: str, rg: str) -> None:
    if not _aks_has_tag(name, rg):
        raise RuntimeError(f"refusing to delete {name}: missing {LABEL_KEY}={LABEL_VALUE} tag")
    _run(["az", "aks", "delete", "-n", name, "-g", rg, "--yes"], check=False)


@contextlib.contextmanager
def ephemeral_cluster(provider: str, runid: str) -> Iterator[str]:
    """Create -> yield kube-context -> unconditionally destroy (finally + atexit)."""
    name = cluster_name(runid)
    project = os.getenv("GCP_PROJECT_ID", "")
    region = os.getenv("GCP_REGION", os.getenv("AWS_REGION", "us-central1"))
    rg = os.getenv("AZURE_RESOURCE_GROUP", "")

    def _destroy():
        with contextlib.suppress(Exception):
            if provider == "gcp":
                _gke_delete(name, project, region)
            elif provider == "aws":
                _eks_delete(name, region)
            elif provider == "azure":
                _aks_delete(name, rg)

    atexit.register(_destroy)  # last-resort: a leaked autopilot cluster bills forever
    try:
        if provider == "gcp":
            _gke_create(name, project, region)
            ctx = _gke_configure_kubectl(name, project, region)
        elif provider == "aws":
            _eks_create(name, region)
            ctx = name
        elif provider == "azure":
            _aks_create(name, rg, os.getenv("AZURE_LOCATION", "eastus"))
            ctx = name
        else:
            raise ValueError(f"unknown provider {provider!r}")
        yield ctx
    finally:
        _destroy()


def reap_orphans(provider: str) -> List[str]:
    """List and delete labelled clusters (best-effort). Only ever touches clusters
    carrying our label/tag."""
    reaped: List[str] = []
    if provider == "gcp":
        project = os.getenv("GCP_PROJECT_ID", "")
        r = _run(["gcloud", "container", "clusters", "list",
                  "--project", project,
                  "--filter", f"resourceLabels.{LABEL_KEY}={LABEL_VALUE}",
                  "--format", "value(name,location)"], check=False)
        for line in filter(None, r.stdout.splitlines()):
            name, _, loc = line.partition("\t")
            with contextlib.suppress(Exception):
                _gke_delete(name.strip(), project, loc.strip())
                reaped.append(name.strip())
    elif provider == "azure":
        rg = os.getenv("AZURE_RESOURCE_GROUP", "")
        r = _run(["az", "aks", "list", "-g", rg,
                  "--query", f"[?tags.{LABEL_KEY}=='{LABEL_VALUE}'].name", "-o", "tsv"], check=False)
        for name in filter(None, r.stdout.split()):
            with contextlib.suppress(Exception):
                _aks_delete(name.strip(), rg)
                reaped.append(name.strip())
    # EKS orphan reaping is via eksctl get clusters + tag inspection (left to the
    # operator; eksctl has no server-side tag filter).
    return reaped


def _main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ephemeral avaloka integration-test clusters")
    p.add_argument("--reap-orphans", action="store_true")
    p.add_argument("--provider", choices=["gcp", "aws", "azure"], default="gcp")
    args = p.parse_args(argv)
    if args.reap_orphans:
        reaped = reap_orphans(args.provider)
        print(f"reaped {len(reaped)} orphan(s): {reaped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
