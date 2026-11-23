import { beforeEach, describe, expect, it, vi } from "vitest";

// The inference panel's endpoint lifecycle: `endpoint.status` from Postgres
// drives the badges, and startService follows the API's 202 by polling the
// endpoint until it is ready. On the pre-2026-11 panel these fail on an
// assertion: `endpoint` is ignored (serviceStarting undefined, serviceRunning
// false) and startService resolves on the POST alone.
const mocks = vi.hoisted(() => ({ proxy: vi.fn(), token: vi.fn() }));
vi.mock("@/lib/api/backendApi", () => ({ proxyFetch: mocks.proxy, getAccessToken: mocks.token }));
import { inferenceApi, normalizeModelDetails, normalizeModels } from "@/api/inference";

describe("endpoint lifecycle in the panel", () => {
  beforeEach(() => { mocks.proxy.mockReset(); mocks.token.mockResolvedValue("tok"); });

  it("T-25 reads serviceRunning/serviceStarting from endpoint.status, not the artifact flag", () => {
    const [deploying] = normalizeModels([{ mlflow_run_id: "r1", model_name: "n", endpoint: { status: "deploying" }, inference_service_details: { stale: true } }]);
    expect(deploying.serviceRunning).toBe(false);
    expect(deploying.serviceStarting).toBe(true);
    expect(deploying.endpointStatus).toBe("deploying");
    const [ready] = normalizeModels([{ mlflow_run_id: "r2", model_name: "n", endpoint: { status: "ready" }, inference_service_details: null }]);
    expect(ready.serviceRunning).toBe(true);
    expect(ready.serviceStarting).toBe(false);
    const [failed] = normalizeModels([{ mlflow_run_id: "r3", model_name: "n", endpoint: { status: "failed", error: "could not load" } }]);
    expect(failed.serviceRunning).toBe(false);
    expect(failed.endpointError).toBe("could not load");
  });

  it("TI-10 badges ready (sleeping) when the endpoint is ready but its pod is scaled to zero", () => {
    const [asleep] = normalizeModels([{ mlflow_run_id: "r", model_name: "n", endpoint: { status: "ready", asleep: true } }]);
    expect(asleep.serviceRunning).toBe(true);
    expect(asleep.serviceAsleep).toBe(true);
    const [awake] = normalizeModels([{ mlflow_run_id: "r", model_name: "n", endpoint: { status: "ready", asleep: false } }]);
    expect(awake.serviceAsleep).toBe(false);
    const [stopped] = normalizeModels([{ mlflow_run_id: "r", model_name: "n", endpoint: { status: "stopped", asleep: true } }]);
    expect(stopped.serviceAsleep).toBe(false);
  });

  it("falls back to the legacy artifact flag when no endpoint state is present", () => {
    const [legacy] = normalizeModels([{ mlflow_run_id: "r", model_name: "n", inference_service_details: { url: "x" } }]);
    expect(legacy.serviceRunning).toBe(true);
    expect(legacy.serviceStarting).toBe(false);
    expect(legacy.endpointStatus).toBeUndefined();
  });

  it("surfaces the registry name and version", () => {
    const m = normalizeModelDetails({ mlflow_run_id: "r", model_name: "n", registry: { registered_name: "avaloka.u.n", version: 3 } });
    expect(m?.registeredName).toBe("avaloka.u.n");
    expect(m?.registryVersion).toBe(3);
  });

  it("T-26 startService resolves only after the endpoint polls ready", async () => {
    const calls: string[] = [];
    const states = ["pending", "deploying", "ready"];
    mocks.proxy.mockImplementation(async (path: string) => {
      calls.push(path);
      if (path.endsWith("/configure-inference-service")) return Response.json({ status: "deploying", job_id: "job-1" }, { status: 202 });
      if (path.endsWith("/endpoint")) return Response.json({ status: states.shift() ?? "ready" });
      throw new Error(`unexpected ${path}`);
    });
    const out = await inferenceApi.startService("abc", { intervalMs: 1 });
    expect(out.endpoint?.status).toBe("ready");
    expect(calls.filter((c) => c.endsWith("/endpoint"))).toHaveLength(3);
  });

  it("startService throws with the server's reason when the deploy fails", async () => {
    mocks.proxy.mockImplementation(async (path: string) => {
      if (path.endsWith("/configure-inference-service")) return Response.json({ status: "deploying", job_id: "job-1" }, { status: 202 });
      return Response.json({ status: "failed", error: "ONNX artifact missing" });
    });
    await expect(inferenceApi.startService("abc", { intervalMs: 1 })).rejects.toMatchObject({ status: 502, detail: "ONNX artifact missing" });
  });

  // Compatibility guard, not a feature test: it also passes on the pre-2026-11
  // panel (which always resolved on the POST). Counted as vacuous in the report.
  it("startService resolves at once on a pre-2026-11 server that answers running", async () => {
    mocks.proxy.mockImplementation(async () => Response.json({ status: "running", inference_service_details: {} }));
    const out = await inferenceApi.startService("abc", { intervalMs: 1 });
    expect(out.status).toBe("running");
    expect(mocks.proxy).toHaveBeenCalledTimes(1);
  });

  it("predict posts to the canonical /predict route", async () => {
    mocks.proxy.mockResolvedValue(Response.json({ result: [1] }));
    await inferenceApi.predict("abc", { x: 1 });
    expect(mocks.proxy.mock.calls[0][0]).toBe("/api/models/abc/predict");
  });
});
