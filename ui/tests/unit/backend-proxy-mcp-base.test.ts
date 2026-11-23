import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The same-origin proxy forwards an MCP call to the MCP base the request names
// (X-Backend-Base), when that base is one the server is configured with. It
// used to forward every request to the main backend, so the connections pill's
// GET /health read the API's answer instead of the MCP query service's.
const API = "http://api.test:9000";
const TOOL = "http://mcp.test:8010";

type Handler = (ctx: { request: Request }) => Promise<Response>;

async function proxyGet(): Promise<Handler> {
  vi.resetModules();
  vi.doMock("@/lib/api/backend-config", () => ({ BACKEND_API_BASE: API }));
  const { Route } = await import("@/routes/api/public/backend-proxy");
  return (Route.options as any).server.handlers.GET as Handler;
}

const req = (headers: Record<string, string>) =>
  new Request("http://ui.test/api/public/backend-proxy", { method: "GET", headers });

let upstream: string[];
beforeEach(() => {
  upstream = [];
  vi.stubEnv("MCP_TOOL_BASE", TOOL);
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      upstream.push(String(url));
      return new Response(JSON.stringify({ status: "healthy" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
});
afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.doUnmock("@/lib/api/backend-config");
});

describe("backend proxy picks the upstream from X-Backend-Base", () => {
  it("forwards to the configured MCP tool base when the request names it", async () => {
    const GET = await proxyGet();
    const res = await GET({ request: req({ "X-Backend-Path": "/health", "X-Backend-Base": `${TOOL}/` }) });
    expect(res.status).toBe(200);
    expect(upstream).toEqual([`${TOOL}/health`]);
  });

  it("still forwards to the main backend when no base is named", async () => {
    const GET = await proxyGet();
    await GET({ request: req({ "X-Backend-Path": "/health" }) });
    expect(upstream).toEqual([`${API}/health`]);
  });

  it("refuses a base it is not configured with, and calls nothing", async () => {
    const GET = await proxyGet();
    const res = await GET({ request: req({ "X-Backend-Path": "/health", "X-Backend-Base": "http://evil.test" }) });
    expect(res.status).toBe(400);
    expect(upstream).toEqual([]);
  });
});
