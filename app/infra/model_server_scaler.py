"""Scale the model-server Deployment 0 <-> 1 through the Kubernetes API.

The HPA owns 1..N under load; it does nothing at zero replicas, so this is the
only thing that puts the server to sleep after ``MODEL_SERVER_IDLE_TIMEOUT_S``
(2 h) and wakes it on demand. Plain ``kubernetes`` client, in-cluster config
first, kubeconfig second; no cloud SDK, so it runs the same on GKE, EKS, AKS
and a private cluster. KEDA, where installed, can take over via
``modelServer.scaleToZero: keda``; it is never a dependency.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from app.agents.mta_v2.inference_autoscale import AutoscalePolicy, idle_seconds, should_scale_to_zero

logger = logging.getLogger(__name__)

IDLE_TIMEOUT_S = int(os.getenv("MODEL_SERVER_IDLE_TIMEOUT_S", os.getenv("INFERENCE_IDLE_TIMEOUT_S", "7200")))
DEPLOYMENT = os.getenv("MODEL_SERVER_DEPLOYMENT", "avaloka-model-server")
GPU_DEPLOYMENT = os.getenv("MODEL_SERVER_GPU_DEPLOYMENT", "avaloka-model-server-gpu")
NAMESPACE = os.getenv("MODEL_SERVER_NAMESPACE", os.getenv("POD_NAMESPACE", "default"))


def policy() -> AutoscalePolicy:
    return AutoscalePolicy(idle_timeout_s=IDLE_TIMEOUT_S, min_replicas=0, max_replicas=1, warm_replicas=1)


def _apps_api() -> Any:
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except Exception:  # noqa: BLE001 - not in a pod
        config.load_kube_config()
    return client.AppsV1Api()


class ModelServerScaler:
    """get/patch on one Deployment's scale subresource (the Role is scoped to it)."""

    def __init__(self, name: str = DEPLOYMENT, namespace: str = NAMESPACE, api: Any = None,
                 idle_timeout_s: int = IDLE_TIMEOUT_S) -> None:
        self.name = name
        self.namespace = namespace
        self._api = api
        # idle_timeout_s <= 0 disables scale-to-zero: the floor is 1 and the reaper never patches (TI-19).
        self.scale_to_zero = idle_timeout_s > 0
        self.policy = AutoscalePolicy(idle_timeout_s=idle_timeout_s if idle_timeout_s > 0 else 1,
                                      min_replicas=0 if idle_timeout_s > 0 else 1, max_replicas=1, warm_replicas=1)

    @property
    def api(self) -> Any:
        if self._api is None:
            self._api = _apps_api()
        return self._api

    def replicas(self) -> Optional[int]:
        """Desired replicas, or None when the Deployment cannot be read."""
        try:
            scale = self.api.read_namespaced_deployment_scale(self.name, self.namespace)
            return int(scale.spec.replicas or 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[model-server] cannot read scale of %s/%s: %s", self.namespace, self.name, exc)
            return None

    def ready_replicas(self) -> Optional[int]:
        try:
            dep = self.api.read_namespaced_deployment(self.name, self.namespace)
            return int((dep.status.ready_replicas or 0))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[model-server] cannot read status of %s/%s: %s", self.namespace, self.name, exc)
            return None

    def scale(self, replicas: int) -> bool:
        try:
            self.api.patch_namespaced_deployment_scale(self.name, self.namespace, {"spec": {"replicas": int(replicas)}})
            logger.info("[model-server] %s/%s scaled to %s", self.namespace, self.name, replicas)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("[model-server] scale %s/%s to %s failed: %s", self.namespace, self.name, replicas, exc)
            return False

    def wake(self) -> bool:
        """0 -> 1. A Deployment already at >= 1 is left alone (the HPA owns it)."""
        current = self.replicas()
        if current is None:
            return False
        if current >= 1:
            return True
        return self.scale(1)

    def wait_ready(self, timeout_s: int = 180, poll_s: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ready = self.ready_replicas()
            if ready and ready >= 1:
                return True
            time.sleep(poll_s)
        return False

    def sleep_if_idle(self, last_request_epoch: Optional[float], *, now: Optional[float] = None) -> bool:
        """Scale to zero when idle past the timeout. True when it scaled."""
        if not self.scale_to_zero:
            logger.info("[model-server] scale-to-zero disabled for %s: the floor of 1 holds", self.name)
            return False
        current = self.replicas()
        if current is None:
            return False
        clock = now if now is not None else time.time()
        if not should_scale_to_zero(last_request_epoch, current, self.policy, now=clock):
            return False
        logger.info("[model-server] idle %.1fh >= %.1fh: scaling %s to 0",
                    idle_seconds(last_request_epoch, now=clock) / 3600, self.policy.idle_timeout_s / 3600, self.name)
        return self.scale(0)
