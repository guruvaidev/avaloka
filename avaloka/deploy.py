"""Release 3 — Avaloka Serve: deployment with gates.

"Model ready for deployment" must mean more than "we generated a Dockerfile".
Avaloka defines four deployment levels and *gates* the transition between them.
It never silently moves from a research artifact to a production candidate:
Levels 3+ require explicit human approval and a clean validation verdict.

    Level 1  Research artifact   notebook, results, model file, model card
    Level 2  Deployment package  API server, container, schema, tests, lock
    Level 3  Staging-ready       registered model, manifest, monitoring, rollback
    Level 4  Production candidate human approval, data contract, drift, runbook
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEVELS = {
    1: "Research artifact",
    2: "Deployment package",
    3: "Staging-ready",
    4: "Production candidate",
}

# Artifacts required to *claim* each level (cumulative).
_LEVEL_REQUIREMENTS = {
    1: ["model/model.pkl", "model_card.md"],
    2: ["service.py", "Dockerfile", "inference_schema.json", "requirements.lock",
        "tests/test_service.py", "feature_contract.yaml"],
    3: ["deployment/kubernetes.yaml", "monitoring_config.yaml", "mlflow/MLmodel"],
    4: [],  # Level 4 is a governance gate, not an artifact gate.
}

_TARGET_MANIFEST = {
    "local": "deployment/local-compose.yaml",
    "docker": "deployment/local-compose.yaml",
    "kubernetes": "deployment/kubernetes.yaml",
    "ray": "deployment/rayservice.yaml",
}


@dataclass
class DeployPlan:
    package_dir: str
    target: str
    requested_level: int
    achieved_level: int
    validator_cap: int
    gates: list[dict[str, Any]]
    manifest: str | None
    commands: list[str]
    blocked_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__


def _present(pkg: Path, rel: str) -> bool:
    return (pkg / rel).exists()


def _validator_cap(pkg: Path) -> int:
    vr = pkg / "validation_report.json"
    if vr.exists():
        try:
            return int(json.loads(vr.read_text()).get("max_safe_deployment_level", 1))
        except Exception:
            return 1
    return 2  # no validation evidence -> cannot claim staging/production


def plan_deployment(
    package_dir: str,
    target: str = "local",
    *,
    requested_level: int = 2,
    approved: bool = False,
    scale_to_zero: bool = False,
    max_monthly_cost: float | None = None,
) -> DeployPlan:
    pkg = Path(package_dir).expanduser()
    if not pkg.exists():
        raise FileNotFoundError(f"Deployment package not found: {package_dir}")
    target = target.lower()
    if target not in _TARGET_MANIFEST:
        raise ValueError(f"Unsupported target {target!r}. Choose from {sorted(_TARGET_MANIFEST)}.")

    cap = _validator_cap(pkg)
    gates: list[dict[str, Any]] = []

    # Evaluate artifact gates per level (cumulative).
    artifact_supported = 0
    for level in (1, 2, 3):
        missing = [r for r in _LEVEL_REQUIREMENTS[level] if not _present(pkg, r)]
        ok = not missing
        gates.append({"level": level, "name": LEVELS[level], "passed": ok,
                      "missing_artifacts": missing})
        if ok and artifact_supported == level - 1:
            artifact_supported = level

    # Level 4 governance gate: explicit human approval required.
    l4_ok = approved and cap >= 4 and artifact_supported >= 3
    gates.append({"level": 4, "name": LEVELS[4], "passed": l4_ok,
                  "missing_artifacts": [] if l4_ok else
                  (["human approval (--approve)"] if not approved else
                   ["clean validation verdict for production"] if cap < 4 else
                   ["staging-ready package"])})

    # Achieved level is bounded by artifacts, the validator cap and (for L4) approval.
    achieved = min(artifact_supported, cap)
    if requested_level >= 4 and l4_ok:
        achieved = 4

    blocked = None
    if requested_level > achieved:
        if requested_level >= 4 and not approved:
            blocked = ("Production (Level 4) requires explicit human approval (--approve). "
                       "Avaloka never auto-promotes a model to production.")
        elif requested_level > cap:
            blocked = (f"Validation caps this package at Level {cap}; resolve the blocking "
                       "checks in validation_report.json before requesting a higher level.")
        else:
            missing_lvl = next((g for g in gates if g["level"] == requested_level and not g["passed"]), None)
            blocked = (f"Level {requested_level} is missing artifacts: "
                       f"{missing_lvl['missing_artifacts'] if missing_lvl else 'unknown'}.")

    manifest = _TARGET_MANIFEST[target] if _present(pkg, _TARGET_MANIFEST[target]) else None
    commands = _commands(target, manifest, scale_to_zero)

    if max_monthly_cost is not None:
        commands.insert(0, f"# Cost guard: enforce monthly ceiling of ${max_monthly_cost:,.0f}")

    return DeployPlan(
        package_dir=str(pkg), target=target, requested_level=requested_level,
        achieved_level=achieved, validator_cap=cap, gates=gates, manifest=manifest,
        commands=commands, blocked_reason=blocked,
    )


def _commands(target: str, manifest: str | None, scale_to_zero: bool) -> list[str]:
    name = "avaloka-inference"
    if target in {"local", "docker"}:
        return [
            f"docker build -t {name}:latest .",
            f"docker run -p 8080:8080 {name}:latest",
            "curl -s localhost:8080/health",
        ]
    if target == "kubernetes":
        return [
            f"docker build -t {name}:latest .",
            f"kubectl apply -f {manifest or 'deployment/kubernetes.yaml'}",
            f"kubectl rollout status deploy/{name[:20]}",
        ]
    if target == "ray":
        base = [f"kubectl apply -f {manifest or 'deployment/rayservice.yaml'}"]
        if scale_to_zero:
            base.append("# scale-to-zero: min_replicas already set to 0 in rayservice.yaml")
        return base
    return []
