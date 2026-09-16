# app/infra/providers/base.py
"""Cluster provider abstraction.

A ``ClusterProvider`` knows how to ensure a Kubernetes cluster exists (or connect
to an existing one), point ``kubectl`` at it, health-check it, and tear it down.

Every method returns the same structured outcome dict used throughout the infra
layer so reporting stays uniform with the legacy ``k8s_invoker`` flow::

    {"step_name": str, "status": "SUCCESS"|"FAILED"|"SKIPPED"|"WARNING", "message": str, "details": ...}
"""
from __future__ import annotations

import re
import subprocess
from abc import ABC, abstractmethod
from typing import List


_SENSITIVE_NAME = re.compile(
    r"(?:api[_-]?key|secret|password|token|credential|service[_-]?role|access[_-]?key)",
    re.IGNORECASE,
)


def _safe_command(command: List[str]) -> tuple[str, List[str]]:
    """Render argv for diagnostics without exposing credential values."""
    rendered: List[str] = []
    secret_values: List[str] = []
    for arg in command:
        text = str(arg)
        if "=" in text:
            name, value = text.split("=", 1)
            if _SENSITIVE_NAME.search(name):
                if value:
                    secret_values.append(value)
                rendered.append(f"{name}=<redacted>")
                continue
        rendered.append(text)
    return " ".join(rendered), secret_values


def _safe_output(value: object, secret_values: List[str]) -> str:
    """Redact known argv secrets and credential-shaped assignments in output."""
    text = str(value or "")
    for secret in sorted(set(secret_values), key=len, reverse=True):
        if secret:
            text = text.replace(secret, "<redacted>")
    return re.sub(
        r"(?i)((?:api[_-]?key|secret|password|token|credential|service[_-]?role|access[_-]?key)"
        r"\s*[=:]\s*)([^\s,;]+)",
        r"\1<redacted>",
        text,
    ).strip()


def run_command(command: List[str], step_name: str) -> dict:
    """Run a shell command and return a structured outcome dict.

    Shared by every provider (and by ``k8s_invoker``) so command execution and
    error reporting behave identically everywhere.
    """
    safe_command, secret_values = _safe_command(command)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return {
            "step_name": step_name,
            "status": "SUCCESS",
            "message": f"Command '{safe_command}' executed successfully.",
            "details": _safe_output(result.stdout, secret_values),
        }
    except subprocess.CalledProcessError as e:
        return {
            "step_name": step_name,
            "status": "FAILED",
            "message": f"Command '{safe_command}' failed.",
            "details": (
                f"Stdout: {_safe_output(e.stdout, secret_values)}\n"
                f"Stderr: {_safe_output(e.stderr, secret_values)}"
            ),
        }
    except FileNotFoundError:
        return {
            "step_name": step_name,
            "status": "FAILED",
            "message": f"Command '{command[0]}' not found. Ensure it's installed and in PATH.",
            "details": "",
        }


class ClusterProvider(ABC):
    """Abstract base for a Kubernetes cluster provider (local kind / GKE / EKS)."""

    #: short provider key, e.g. "local", "gcp", "aws"
    name: str = "base"

    @abstractmethod
    def provision_cluster(self) -> dict:
        """Create the cluster if it does not already exist (idempotent)."""

    @abstractmethod
    def configure_kubectl(self) -> dict:
        """Point the local kubeconfig/context at this cluster."""

    @abstractmethod
    def teardown(self) -> dict:
        """Delete the cluster (best-effort; SKIPPED if it does not exist)."""

    def health_check(self) -> dict:
        """Default health check: ``kubectl get nodes`` and look for Ready nodes."""
        outcome = run_command(["kubectl", "get", "nodes"], "Cluster Health Checks")
        if outcome["status"] == "SUCCESS" and " Ready" in outcome["details"]:
            outcome["message"] = "Cluster health checks passed. Nodes are ready."
        elif outcome["status"] == "SUCCESS":
            outcome["status"] = "WARNING"
            outcome["message"] = "Cluster health checks completed, but not all nodes are 'Ready'."
        return outcome
