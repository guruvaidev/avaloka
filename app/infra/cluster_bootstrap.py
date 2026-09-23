# app/infra/cluster_bootstrap.py
"""End-to-end avaloka cluster bootstrap.

Two modes:

  provision  connect-or-create a cluster (local kind / GKE / EKS), install KubeRay,
             apply the RayCluster, deploy the avaloka app, deploy Ray Serve.
  connect    attach to an existing Ray-on-k8s cluster (RAY_ADDRESS) and deploy the
             avaloka app pointed at it (skips cluster/operator creation).

CLI:
  python -m app.infra.cluster_bootstrap --provider local --mode provision
  python -m app.infra.cluster_bootstrap --mode connect --ray-address ray://host:10001
"""
from __future__ import annotations

import argparse
import sys
from typing import List

from app.infra import cloud_provisioner, deploy_stack, install_k8s, ray_manager

_FAIL = "FAILED"


def _print(outcome: dict) -> dict:
    icon = {"SUCCESS": "✅", "SKIPPED": "⏭️ ", "WARNING": "⚠️ ", "FAILED": "❌"}.get(outcome["status"], "•")
    print(f"  {icon} [{outcome['status']}] {outcome['step_name']}: {outcome['message']}")
    return outcome


def _any_failed(outcomes: List[dict]) -> bool:
    return any(o["status"] == _FAIL for o in outcomes)


def provision(args) -> int:
    results: List[dict] = []

    print("\n[1/6] Preflight tool check")
    pre = _print(install_k8s.check_tools_for(args.provider))
    results.append(pre)
    if pre["status"] == _FAIL:
        return _finish(results)

    if args.provider == "local":
        print("\n[2/6] Build + side-load images into kind")
        for o in deploy_stack.build_images(load_into_kind=False):  # build first (cluster not up yet)
            results.append(_print(o))
        if _any_failed(results):
            return _finish(results)

    print("\n[3/6] Provision cluster + configure kubectl")
    prov = cloud_provisioner.provision(args.provider)
    for o in prov:
        results.append(_print(o))
    if _any_failed(prov):
        return _finish(results)

    if args.provider == "local":
        print("\n[3b/6] Load images into kind node(s)")
        from app.infra.providers.local_kind import LocalKindProvider
        kp = LocalKindProvider()
        for img in (
            deploy_stack.API_IMAGE,
            deploy_stack.RAY_IMAGE,
            deploy_stack.WEBUI_IMAGE,
            # supabase.functions pulls this IfNotPresent, so it must reach the node.
            deploy_stack.FUNCTIONS_IMAGE,
        ):
            results.append(_print(kp.load_image(img)))

    print("\n[4/6] Install KubeRay operator")
    results.append(_print(install_k8s.install_kuberay(namespace=args.namespace)))
    if _any_failed(results):
        return _finish(results)

    print("\n[5/6] Deploy avaloka app")
    results.append(_print(deploy_stack.deploy_avaloka(
        namespace=args.namespace,
        connect_existing=False,
        service_type=args.service_type,
        provider=args.provider,
        # Local/on-prem clusters have no cloud bucket to write to, and no
        # ReadWriteMany volume the Ray workers could share instead — so give them
        # the in-cluster MinIO. Cloud providers keep their own object storage.
        minio=args.provider not in ("gcp", "aws", "azure"),
    )))
    if _any_failed(results):
        return _finish(results)

    # Ray pods consume avaloka-config, avaloka-secrets and the chart-managed KSA
    # at creation time. Apply them only after Helm has created those resources;
    # optional envFrom references do not trigger a restart when a ConfigMap or
    # Secret appears later.
    print("\n[6/6] Deploy RayCluster + Ray Serve")
    results.append(_print(ray_manager.apply_ray_cluster(namespace=args.namespace)))
    if not _any_failed(results) and not args.skip_serve:
        results.append(_print(ray_manager.deploy_ray_serve(namespace=args.namespace)))

    if args.data_stack:
        print("\n[+] Optional data stack")
        for o in deploy_stack.deploy_data_stack(args.data_stack, namespace=args.namespace):
            results.append(_print(o))

    return _finish(results, provider=args.provider)


def connect(args) -> int:
    results: List[dict] = []

    print("\n[1/3] Preflight tool check")
    results.append(_print(install_k8s.check_tools_for("connect")))
    if _any_failed(results):
        return _finish(results)

    print("\n[2/3] Validate Ray connectivity")
    conn = _print(ray_manager.connect(args.ray_address))
    results.append(conn)
    if conn["status"] == _FAIL:
        return _finish(results)
    ray_address = ray_manager.resolve_ray_address(args.ray_address)

    print("\n[3/3] Deploy avaloka app (connect mode)")
    results.append(_print(deploy_stack.deploy_avaloka(
        namespace=args.namespace,
        connect_existing=True,
        ray_address=ray_address or "",
        service_type=args.service_type,
        provider=args.provider,
        minio=args.provider not in ("gcp", "aws", "azure"),
        supabase=args.provider == "local",
    )))
    return _finish(results)


def _finish(results: List[dict], provider: str | None = None) -> int:
    failed = _any_failed(results)
    print("\n" + ("=" * 60))
    if failed:
        print("Bootstrap FAILED. See the ❌ step(s) above.")
    else:
        print("Bootstrap complete. 🎉")
        print("\nNext steps:")
        if provider == "local":
            print("  • Avaloka API:    http://localhost:9010  (kind NodePort; try /health, /docs)")
            print("  • Ray dashboard:  http://localhost:8265  (kind NodePort)")
        else:
            print("  • Avaloka API:    kubectl port-forward svc/avaloka 9000:9000  (try /health, /docs)")
            print("  • Ray dashboard:  kubectl port-forward svc/avaloka-raycluster-head-svc 8265:8265")
        print("  • Inference:      kubectl port-forward svc/avaloka-inference-serve-svc 8000:8000")
        print("  • Pods:           kubectl get pods")
    print("=" * 60)
    return 1 if failed else 0


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bootstrap avaloka on Kubernetes (provision or connect).")
    p.add_argument("--mode", choices=["provision", "connect"], default="provision")
    p.add_argument("--provider", choices=["local", "gcp", "aws", "azure"], default="local")
    p.add_argument("--namespace", default="default")
    p.add_argument("--ray-address", default=None, help="Existing Ray address for connect mode (or set RAY_ADDRESS).")
    p.add_argument("--service-type", default=None, help="Override avaloka Service type (ClusterIP/NodePort/LoadBalancer).")
    p.add_argument("--skip-serve", action="store_true", help="Skip deploying the Ray Serve inference app.")
    p.add_argument("--data-stack", nargs="*", default=[], help="Optional data-stack components (postgres kafka milvus neo4j opensearch).")
    args = p.parse_args(argv)

    return connect(args) if args.mode == "connect" else provision(args)


if __name__ == "__main__":
    sys.exit(main())
