// The swarm API bindings: every call goes through the same-origin proxy with the
// backend path in X-Backend-Path and a Bearer token, exactly like the trace calls.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/integrations/supabase/external-client", () => ({
  supabase: { auth: { getSession: vi.fn(async () => ({ data: { session: { access_token: "tok-123" } } })) } },
}));

import { backendApi, DEV_TOKEN_KEY } from "@/lib/api/backendApi";

function ok(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

describe("swarm API bindings", () => {
  const fetchSpy = vi.fn<typeof fetch>();
  beforeEach(() => {
    fetchSpy.mockReset();
    vi.stubGlobal("fetch", fetchSpy);
    localStorage.removeItem(DEV_TOKEN_KEY);
  });
  afterEach(() => vi.unstubAllGlobals());

  const lastCall = () => {
    const [url, init] = fetchSpy.mock.calls.at(-1)!;
    const headers = new Headers((init as RequestInit).headers);
    return { url: String(url), init: init as RequestInit, headers };
  };

  it("starts a mission with a POST body and a Bearer token", async () => {
    fetchSpy.mockResolvedValueOnce(ok({ mission_id: "m1", status: "running", grant: { clones: 2 }, limits: {} }));
    const res = await backendApi.startSwarmMission({ source: { sample: "churn" }, goal: "why churn", kind: "analyze", workers: 2 });
    expect(res.mission_id).toBe("m1");
    const { url, init, headers } = lastCall();
    expect(url).toMatch(/\/api\/public\/backend-proxy$/);
    expect(init.method).toBe("POST");
    expect(headers.get("x-backend-path")).toBe("/api/swarm/missions");
    expect(headers.get("authorization")).toBe("Bearer tok-123");
    expect(headers.get("content-type")).toBe("application/json");
    expect(JSON.parse(String(init.body))).toEqual({ source: { sample: "churn" }, goal: "why churn", kind: "analyze", workers: 2 });
  });

  it("surfaces the server's reason for a refused mission", async () => {
    fetchSpy.mockResolvedValueOnce(new Response(JSON.stringify({ detail: "a training swarm requires a target column" }), { status: 400 }));
    await expect(backendApi.startSwarmMission({ source: { sample: "churn" }, goal: "g", kind: "train" })).rejects.toThrow(
      /a training swarm requires a target column/,
    );
  });

  it("polls records after a cursor, and reads the mission, clones, episodes and telemetry", async () => {
    fetchSpy.mockImplementation(async () => ok({ mission_id: "m1", status: "running", records: [], cursor: 7, done: false }));
    await backendApi.getSwarmRecords("m1", 7);
    expect(lastCall().headers.get("x-backend-path")).toBe("/api/swarm/missions/m1/records?after=7&limit=500");
    await backendApi.getSwarmRecords("m1");
    expect(lastCall().headers.get("x-backend-path")).toBe("/api/swarm/missions/m1/records?after=0&limit=500");
    await backendApi.getSwarmMission("m1", { records: true });
    expect(lastCall().headers.get("x-backend-path")).toBe("/api/swarm/missions/m1?records=1");
    await backendApi.getSwarmMission("m1");
    expect(lastCall().headers.get("x-backend-path")).toBe("/api/swarm/missions/m1");
    await backendApi.listSwarmMissions();
    expect(lastCall().headers.get("x-backend-path")).toBe("/api/swarm/missions");
    await backendApi.getSwarmClone("m1", "ava-00");
    expect(lastCall().headers.get("x-backend-path")).toBe("/api/swarm/missions/m1/clones/ava-00");
    await backendApi.getSwarmEpisodes("m1");
    expect(lastCall().headers.get("x-backend-path")).toBe("/api/swarm/missions/m1/episodes");
    await backendApi.getSwarmTelemetry("m1");
    expect(lastCall().headers.get("x-backend-path")).toBe("/api/swarm/missions/m1/telemetry");
    for (const [, init] of fetchSpy.mock.calls) {
      expect(new Headers((init as RequestInit).headers).get("authorization")).toBe("Bearer tok-123");
      expect((init as RequestInit).method).toBe("GET");
    }
  });

  it("carries a 404 as a status the panel can read", async () => {
    fetchSpy.mockResolvedValueOnce(new Response("nope", { status: 404, statusText: "Not Found" }));
    await expect(backendApi.getSwarmMission("missing")).rejects.toMatchObject({ status: 404 });
  });

  it("uses a pasted dev token when one is set, without touching the session", async () => {
    localStorage.setItem(DEV_TOKEN_KEY, "pasted-token");
    fetchSpy.mockResolvedValueOnce(ok({ missions: [] }));
    await backendApi.listSwarmMissions();
    expect(lastCall().headers.get("authorization")).toBe("Bearer pasted-token");
  });
});
