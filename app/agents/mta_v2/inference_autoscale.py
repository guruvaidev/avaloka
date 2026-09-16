"""Idle-aware scaling for the shared inference Ray cluster.

Three problems this solves:

1. **A new namespace per deployment.** The legacy gateway backend minted
   ``inference-service-{uuid}`` for every model, each with its own
   LoadBalancer, and nothing reclaimed them. Eighteen accumulated between April
   and August 2026 and were still billing while serving no traffic. Inference
   now targets one long-lived cluster.

2. **Capacity that never scales down.** Replicas stayed up indefinitely after
   the last request. They now scale to zero once idle past a threshold
   (default two hours) and ramp back on demand.

3. **On-demand pricing for interruptible work.** Inference replicas are
   restartable, so they belong on Spot.

**On Spot and Autopilot.** The target cluster (``ray-gke-trainer``) is GKE
Autopilot, which has no user-managed node pools -- ``gcloud container
node-pools`` refuses outright. Spot is therefore requested *per pod*, via the
``cloud.google.com/gke-spot`` nodeSelector plus the matching toleration, and
Autopilot provisions Spot capacity to match. Any instruction to "create a spot
node pool" cannot be carried out on this cluster; :func:`spot_pod_overrides`
is the equivalent that does work.

The policy functions here are pure and take an explicit clock, so the scaling
decisions are testable without a cluster. Only :class:`RayServiceScaler`
touches kubectl.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: The single shared inference cluster. Nothing in this module creates a
#: cluster; if this one is absent the caller fails loudly rather than
#: provisioning a replacement.
RAY_CLUSTER_NAME = os.getenv("RAY_GKE_CLUSTER_NAME", "ray-gke-trainer")
RAY_CLUSTER_LOCATION = os.getenv("RAY_GKE_CLUSTER_LOCATION", "us-central1")
RAYSERVICE_NAME = os.getenv("RAY_SERVICE_NAME", "avaloka-inference")
RAYSERVICE_NAMESPACE = os.getenv("RAY_SERVICE_NAMESPACE", "default")

#: Where the last-activity marker lives. A file rather than in-process state so
#: the reaper can run as a separate CronJob from the serving process.
ACTIVITY_FILE = Path(
    os.getenv("AVALOKA_INFERENCE_ACTIVITY_FILE", "/tmp/avaloka-inference-activity.json")
)

DEFAULT_IDLE_TIMEOUT_S = int(os.getenv("INFERENCE_IDLE_TIMEOUT_S", str(2 * 60 * 60)))


@dataclass(frozen=True)
class AutoscalePolicy:
    """When to scale the shared inference service down and back up."""

    #: Seconds without a request before scaling to zero. Two hours by default:
    #: long enough that an interactive session is not interrupted, short enough
    #: that an abandoned deployment stops costing money the same day.
    idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S
    #: Idle floor. Zero is the point -- Autopilot bills per running pod.
    min_replicas: int = 0
    #: Ceiling when serving.
    max_replicas: int = 4
    #: Replicas to bring up on the first request after an idle scale-down.
    warm_replicas: int = 1

    def __post_init__(self) -> None:
        if self.idle_timeout_s <= 0:
            raise ValueError("idle_timeout_s must be positive")
        if self.min_replicas < 0:
            raise ValueError("min_replicas cannot be negative")
        if self.max_replicas < 1:
            raise ValueError("max_replicas must be at least 1")
        if not self.min_replicas <= self.warm_replicas <= self.max_replicas:
            raise ValueError(
                f"warm_replicas ({self.warm_replicas}) must fall between "
                f"min_replicas ({self.min_replicas}) and max_replicas ({self.max_replicas})"
            )


# --------------------------------------------------------------------------- #
# Policy — pure, clock injected, no cluster required
# --------------------------------------------------------------------------- #

def idle_seconds(last_activity_epoch: Optional[float], *, now: float) -> float:
    """Seconds since the last request.

    A missing marker is treated as *infinitely idle*. That is the safe
    direction: an unknown service scales down (costing a cold start) rather
    than staying up forever on the assumption it might be busy.
    """
    if last_activity_epoch is None:
        return float("inf")
    return max(0.0, now - last_activity_epoch)


def should_scale_to_zero(
    last_activity_epoch: Optional[float],
    current_replicas: int,
    policy: AutoscalePolicy,
    *,
    now: float,
) -> bool:
    """True when the service is up and has been idle past the threshold."""
    if current_replicas <= policy.min_replicas:
        return False
    return idle_seconds(last_activity_epoch, now=now) >= policy.idle_timeout_s


def should_ramp_up(current_replicas: int, policy: AutoscalePolicy) -> bool:
    """True when a request arrives and nothing is running to serve it."""
    return current_replicas < policy.warm_replicas


def target_replicas(
    last_activity_epoch: Optional[float],
    current_replicas: int,
    policy: AutoscalePolicy,
    *,
    now: float,
    serving_request: bool = False,
) -> int:
    """The replica count this service should be at right now.

    ``serving_request`` biases toward availability: an in-flight request always
    wins over an idle scale-down, so a request arriving exactly at the timeout
    boundary is served rather than dropped into a cold start.
    """
    if serving_request:
        return max(current_replicas, policy.warm_replicas)
    if should_scale_to_zero(last_activity_epoch, current_replicas, policy, now=now):
        return policy.min_replicas
    return current_replicas


# --------------------------------------------------------------------------- #
# Activity marker
# --------------------------------------------------------------------------- #

def mark_activity(*, now: Optional[float] = None, path: Optional[Path] = None) -> None:
    """Record that a request was served. Never raises into the request path."""
    target = path or ACTIVITY_FILE
    stamp = now if now is not None else time.time()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"last_activity": stamp}), encoding="utf-8")
    except OSError as exc:
        # Losing the marker costs an early scale-down, not a failed inference.
        logger.warning("[inference] could not record activity marker: %s", exc)


def read_last_activity(path: Optional[Path] = None) -> Optional[float]:
    """Last activity timestamp, or ``None`` if unknown/unreadable."""
    target = path or ACTIVITY_FILE
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        value = payload.get("last_activity")
        return float(value) if value is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Spot scheduling (Autopilot: per-pod, not per-node-pool)
# --------------------------------------------------------------------------- #

def spot_pod_overrides() -> Dict[str, Any]:
    """nodeSelector and toleration that place a pod on Spot capacity.

    On Autopilot this is the *only* way to request Spot -- there are no node
    pools to configure. Inference replicas are safe here because they are
    stateless and restartable; a preempted replica is replaced and the next
    request re-warms it.

    Do not apply this to the Ray head, which holds cluster state.
    """
    return {
        "nodeSelector": {"cloud.google.com/gke-spot": "true"},
        "tolerations": [
            {
                "key": "cloud.google.com/gke-spot",
                "operator": "Equal",
                "value": "true",
                "effect": "NoSchedule",
            }
        ],
    }


def apply_spot_to_worker_groups(ray_cluster_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a RayCluster spec with worker groups moved to Spot.

    The head group is deliberately left on standard capacity -- preempting the
    head restarts the whole Ray cluster, which costs far more than the Spot
    discount saves.
    """
    spec = json.loads(json.dumps(ray_cluster_spec))  # deep copy, plain data
    overrides = spot_pod_overrides()
    for group in spec.get("workerGroupSpecs", []) or []:
        pod_spec = group.setdefault("template", {}).setdefault("spec", {})
        pod_spec.setdefault("nodeSelector", {}).update(overrides["nodeSelector"])
        existing = pod_spec.setdefault("tolerations", [])
        if not any(t.get("key") == "cloud.google.com/gke-spot" for t in existing):
            existing.extend(overrides["tolerations"])
    return spec


# --------------------------------------------------------------------------- #
# Effector — the only part that touches the cluster
# --------------------------------------------------------------------------- #

class RayServiceScaler:
    """Reads and sets replica counts on the shared RayService via kubectl."""

    def __init__(
        self,
        name: str = RAYSERVICE_NAME,
        namespace: str = RAYSERVICE_NAMESPACE,
        policy: Optional[AutoscalePolicy] = None,
    ) -> None:
        self.name = name
        self.namespace = namespace
        self.policy = policy or AutoscalePolicy()

    def _kubectl(self, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["kubectl", "-n", self.namespace, *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )

    def current_replicas(self) -> Optional[int]:
        """Current worker replica count, or ``None`` if it cannot be read.

        ``None`` is distinct from ``0``: unknown means take no scaling action,
        whereas zero means already scaled down.
        """
        result = self._kubectl(
            "get", "rayservice", self.name,
            "-o", "jsonpath={.spec.rayClusterConfig.workerGroupSpecs[0].replicas}",
        )
        if result.returncode != 0:
            logger.warning("[inference] could not read replicas: %s", result.stderr.strip())
            return None
        raw = (result.stdout or "").strip()
        try:
            return int(raw)
        except ValueError:
            logger.warning("[inference] unexpected replica value %r", raw)
            return None

    def set_replicas(self, replicas: int) -> bool:
        """Patch the worker replica count. Returns True on success."""
        patch = {
            "spec": {"rayClusterConfig": {"workerGroupSpecs": [{"replicas": replicas}]}}
        }
        result = self._kubectl(
            "patch", "rayservice", self.name, "--type", "merge",
            "-p", json.dumps(patch),
        )
        if result.returncode != 0:
            logger.warning("[inference] scale to %s failed: %s", replicas, result.stderr.strip())
            return False
        logger.info("[inference] %s scaled to %s replicas", self.name, replicas)
        return True

    def ensure_capacity(self) -> bool:
        """Ramp up before serving if the service is scaled to zero.

        Called on the request path, so it is a no-op when capacity already
        exists and never raises.
        """
        current = self.current_replicas()
        if current is None or not should_ramp_up(current, self.policy):
            return True
        logger.info("[inference] cold start: %s -> %s replicas",
                    current, self.policy.warm_replicas)
        return self.set_replicas(self.policy.warm_replicas)

    def reap_if_idle(self, *, now: Optional[float] = None) -> bool:
        """Scale to zero when idle past the threshold. True if it scaled down.

        Intended for a periodic job (CronJob or Celery beat), not the request
        path.
        """
        current = self.current_replicas()
        if current is None:
            return False
        last = read_last_activity()
        clock = now if now is not None else time.time()
        if not should_scale_to_zero(last, current, self.policy, now=clock):
            return False
        idle_h = idle_seconds(last, now=clock) / 3600
        logger.info("[inference] idle %.1fh (threshold %.1fh) — scaling to %s",
                    idle_h, self.policy.idle_timeout_s / 3600, self.policy.min_replicas)
        return self.set_replicas(self.policy.min_replicas)


def policy_from_env() -> AutoscalePolicy:
    """Build a policy from environment, falling back to defaults on bad input."""
    base = AutoscalePolicy()
    try:
        return replace(
            base,
            idle_timeout_s=int(os.getenv("INFERENCE_IDLE_TIMEOUT_S", base.idle_timeout_s)),
            min_replicas=int(os.getenv("INFERENCE_MIN_REPLICAS", base.min_replicas)),
            max_replicas=int(os.getenv("INFERENCE_MAX_REPLICAS", base.max_replicas)),
            warm_replicas=int(os.getenv("INFERENCE_WARM_REPLICAS", base.warm_replicas)),
        )
    except ValueError as exc:
        logger.warning("[inference] invalid autoscale env (%s); using defaults", exc)
        return base
