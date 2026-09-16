"""`kubectl port-forward` as a context manager, so cluster tests can hit a
ClusterIP Service from the test process without a LoadBalancer."""
from __future__ import annotations

import contextlib
import socket
import subprocess
import time
from typing import Iterator

from tests.k8s.helpers.k8s import CONTEXT, NAMESPACE


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextlib.contextmanager
def port_forward(target: str, remote_port: int, local_port: int | None = None,
                 ready_timeout_s: int = 30) -> Iterator[int]:
    """Port-forward ``svc/<name>`` (or ``pod/<name>``) and yield the local port.

    ``target`` e.g. "svc/avaloka" or "pod/foo". The tunnel is torn down on exit.
    """
    local_port = local_port or _free_port()
    proc = subprocess.Popen(
        ["kubectl", "--context", CONTEXT, "-n", NAMESPACE, "port-forward",
         target, f"{local_port}:{remote_port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.time() + ready_timeout_s
        while time.time() < deadline:
            with contextlib.suppress(OSError):
                with socket.create_connection(("127.0.0.1", local_port), timeout=1):
                    break
            if proc.poll() is not None:
                raise RuntimeError(f"port-forward for {target} exited early")
            time.sleep(0.5)
        else:
            raise RuntimeError(f"port-forward for {target} not ready in {ready_timeout_s}s")
        yield local_port
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
