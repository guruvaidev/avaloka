// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { rayDashboardBase, rayJobIdFromHref, rayJobMarkdown, rayJobUrl } from "../../src/lib/rayDashboard";
import { BACKEND_API_BASE } from "../../src/lib/api/backend-config";
import { openRayDashboard, rayApi, RayRequestError, stopRayDashboardSessions } from "../../src/lib/ray-dashboard-session";

const transport = vi.hoisted(() => ({ token: vi.fn(), proxy: vi.fn() }));
vi.mock("../../src/lib/api/backendApi", () => ({ getAccessToken: transport.token, proxyFetch: transport.proxy }));
const flush = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };
const tab = () => ({ closed: false, opener: {}, location: { href: "about:blank" }, document: { title: "", body: { textContent: "" } }, close: vi.fn() });
beforeEach(() => { vi.useFakeTimers(); transport.token.mockResolvedValue("test-token"); });
afterEach(() => { stopRayDashboardSessions(); vi.useRealTimers(); vi.unstubAllEnvs(); vi.unstubAllGlobals(); vi.restoreAllMocks(); vi.clearAllMocks(); });

describe("Ray URL and transport", () => {
  // /ray/ is served from the app's own origin (the UI's nginx), never from
  // BACKEND_API_BASE, which is an in-cluster address the browser cannot reach.
  // On localhost the dashboard proxy is on :28265 (src/lib/rayDashboard.ts).
  it("defaults to the dashboard proxy on localhost, not the backend", () => {
    vi.stubEnv("VITE_RAY_DASHBOARD_URL", "");
    expect(window.location.hostname).toBe("localhost");
    expect(rayDashboardBase()).toBe("http://localhost:28265/ray/");
    expect(rayDashboardBase()).not.toBe(new URL("/ray/", BACKEND_API_BASE).toString());
  });
  it("defaults to /ray/ on the app's own origin elsewhere", () => {
    vi.stubEnv("VITE_RAY_DASHBOARD_URL", "");
    vi.stubGlobal("location", new URL("https://app.example.test/analysis?x=1"));
    expect(rayDashboardBase()).toBe("https://app.example.test/ray/");
  });
  it("normalizes the override and encodes job IDs", () => {
    vi.stubEnv("VITE_RAY_DASHBOARD_URL", "https://ray.example/ray///");
    expect(rayDashboardBase()).toBe("https://ray.example/ray/");
    expect(rayJobUrl("a/b ?")).toBe("https://ray.example/ray/#/jobs/a%2Fb%20%3F");
    expect(rayJobIdFromHref(rayJobUrl("a/b ?"))).toBe("a/b ?");
  });
  it("preserves markdown/code and recognizes legacy job links", () => {
    const id = "dta-4f1c0a9e72b35d18";
    expect(rayJobMarkdown(`Job ${id}`)).toBe(`Job [${id}](${rayJobUrl(id)})`);
    expect(rayJobMarkdown(`\`${id}\` [${id}](/existing)`)).toBe(`\`${id}\` [${id}](/existing)`);
    expect(rayJobIdFromHref(`/ray-jobs?job=${id}`)).toBe(id);
  });
  it("uses the normal bearer header and includes credentials on direct session/heartbeat requests", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ state: "running", url: "/ray/" })));
    vi.stubGlobal("fetch", fetch);
    await rayApi.session(); await rayApi.heartbeat();
    // Same host as /ray/, so the session cookie lands where the dashboard is served.
    expect(fetch.mock.calls.map(([url]) => String(url))).toEqual(["http://localhost:28265/api/ray/session", "http://localhost:28265/api/ray/heartbeat"]);
    expect(fetch.mock.calls[0][1]).toMatchObject({ method: "POST", credentials: "include", headers: { Authorization: "Bearer test-token" } });
  });
  it("retains the upgrade reason from status", async () => {
    transport.proxy.mockResolvedValue(new Response(JSON.stringify({ reason: "plan_upgrade_required" }), { status: 403 }));
    await expect(rayApi.status()).rejects.toMatchObject({ status: 403, reason: "plan_upgrade_required" });
    expect(transport.proxy).toHaveBeenCalledWith("/api/ray/status", expect.objectContaining({ method: "GET" }));
  });
});

describe("Ray session lifecycle", () => {
  it("opens a blank tab synchronously then navigates the running job", async () => {
    const target = tab(); const open = vi.spyOn(window, "open").mockReturnValue(target as unknown as Window);
    const session = vi.spyOn(rayApi, "session").mockResolvedValue({ state: "running", url: "/ray/" });
    openRayDashboard(vi.fn(), "job/a");
    expect(open).toHaveBeenCalledExactlyOnceWith("about:blank", "_blank");
    expect(session).toHaveBeenCalledOnce();
    await flush(); expect(target.location.href).toBe(rayJobUrl("job/a"));
  });
  it.each(["starting", "stopped", "stopping"] as const)("polls %s at 5 seconds then runs", async (state) => {
    const target = tab(); vi.spyOn(window, "open").mockReturnValue(target as unknown as Window);
    const session = vi.spyOn(rayApi, "session").mockResolvedValueOnce({ state, url: null }).mockResolvedValue({ state: "running", url: "/ray/" });
    openRayDashboard(vi.fn()); await flush();
    expect(target.document.body.textContent).toBe("Starting your Ray cluster. This usually takes a few minutes. (0m 0s)");
    await vi.advanceTimersByTimeAsync(4_999); expect(session).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1); expect(session).toHaveBeenCalledTimes(2);
    expect(target.location.href).toBe(rayDashboardBase());
  });
  // A per-user cluster can take minutes to schedule; the tab waits 10 minutes,
  // as the backend does (START_TIMEOUT_MS in src/lib/ray-dashboard-session.ts).
  it("ends startup at 10 minutes without further requests", async () => {
    const target = tab(); vi.spyOn(window, "open").mockReturnValue(target as unknown as Window);
    const session = vi.spyOn(rayApi, "session").mockResolvedValue({ state: "starting", url: null });
    openRayDashboard(vi.fn()); await flush();
    await vi.advanceTimersByTimeAsync(180_000);
    expect(target.document.body.textContent).toBe("Starting your Ray cluster. This usually takes a few minutes. (3m 0s)");
    await vi.advanceTimersByTimeAsync(420_000);
    expect(target.document.body.textContent).toBe("Your Ray cluster did not start. Try again from Avaloka.");
    expect(session).toHaveBeenCalledTimes(120);
    await vi.advanceTimersByTimeAsync(60_000); expect(session).toHaveBeenCalledTimes(120);
  });
  it.each([401, 403])("closes the tab and reports %s", async (status) => {
    const target = tab(); vi.spyOn(window, "open").mockReturnValue(target as unknown as Window);
    vi.spyOn(rayApi, "session").mockRejectedValue(new RayRequestError(status)); const error = vi.fn();
    openRayDashboard(error); await flush(); expect(target.close).toHaveBeenCalledOnce(); expect(error).toHaveBeenCalledWith(expect.objectContaining({ status }));
  });
  it("has one 60-second heartbeat and stops at 15 minutes idle", async () => {
    const target = tab(); vi.spyOn(window, "open").mockReturnValue(target as unknown as Window);
    vi.spyOn(rayApi, "session").mockResolvedValue({ state: "running", url: "/ray/" });
    const beat = vi.spyOn(rayApi, "heartbeat").mockResolvedValue(new Response());
    openRayDashboard(vi.fn()); await flush(); openRayDashboard(vi.fn()); await flush();
    await vi.advanceTimersByTimeAsync(60_000); expect(beat).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(840_000); expect(beat).toHaveBeenCalledTimes(14);
    await vi.advanceTimersByTimeAsync(60_000); expect(beat).toHaveBeenCalledTimes(14);
  });
  it.each(["mousemove", "keydown", "click", "scroll"])("%s resets inactivity and closing stops the heartbeat", async (event) => {
    const target = tab(); vi.spyOn(window, "open").mockReturnValue(target as unknown as Window);
    vi.spyOn(rayApi, "session").mockResolvedValue({ state: "running", url: "/ray/" });
    const beat = vi.spyOn(rayApi, "heartbeat").mockResolvedValue(new Response());
    openRayDashboard(vi.fn()); await flush(); await vi.advanceTimersByTimeAsync(840_000);
    window.dispatchEvent(new Event(event)); await vi.advanceTimersByTimeAsync(60_000);
    expect(beat).toHaveBeenCalledTimes(15);
    target.closed = true; await vi.advanceTimersByTimeAsync(60_000); expect(beat).toHaveBeenCalledTimes(15);
  });
  it("stops startup calls when the tab is closed", async () => {
    const target = tab(); vi.spyOn(window, "open").mockReturnValue(target as unknown as Window);
    const session = vi.spyOn(rayApi, "session").mockResolvedValue({ state: "starting", url: null });
    openRayDashboard(vi.fn()); await flush(); target.closed = true;
    await vi.advanceTimersByTimeAsync(10_000); expect(session).toHaveBeenCalledTimes(1);
  });
});