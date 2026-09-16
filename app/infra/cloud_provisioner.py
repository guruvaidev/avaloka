# app/infra/cloud_provisioner.py
"""Thin wrapper that provisions a cluster via the provider abstraction.

Delegates to ``app.infra.providers`` so callers (cluster_bootstrap, tests) get a
single entry point for "make sure a cluster exists and kubectl points at it".
"""
from __future__ import annotations

from typing import List

from app.infra.providers import get_provider


def provision(platform: str) -> List[dict]:
    """Provision (or reuse) a cluster, configure kubectl, and health-check it.

    Returns the ordered list of structured outcome dicts. Stops early if a step
    fails so the caller can report the failing step.
    """
    provider = get_provider(platform)
    outcomes: List[dict] = []

    step = provider.provision_cluster()
    outcomes.append(step)
    if step["status"] == "FAILED":
        return outcomes

    step = provider.configure_kubectl()
    outcomes.append(step)
    if step["status"] == "FAILED":
        return outcomes

    outcomes.append(provider.health_check())
    return outcomes


def teardown(platform: str) -> dict:
    """Delete the cluster for the given platform."""
    return get_provider(platform).teardown()


if __name__ == "__main__":  # pragma: no cover
    print("cloud_provisioner: use provision(<local|gcp|aws>) / teardown(...)")
