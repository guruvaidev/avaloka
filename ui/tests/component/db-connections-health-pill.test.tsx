import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";

// Supabase as a normal install has it: auth and the mcp_connections table
// work, and there is no "check-mcp-health" edge function (ui/supabase/functions
// ships none, so the functions runtime -- when there is one -- 404s it).
const { invoke, supabaseStub } = vi.hoisted(() => {
  const invoke = vi.fn(async (_name: string, _opts?: unknown) => ({
    data: null,
    error: new Error("FunctionsHttpError: function check-mcp-health not found"),
  }));
  const supabaseStub = {
    auth: {
      getUser: async () => ({ data: { user: { id: "user-1" } } }),
      getSession: async () => ({ data: { session: { access_token: "t" } } }),
    },
    from: () => ({
      select: () => ({ eq: () => ({ order: async () => ({ data: [] }) }) }),
    }),
    channel: () => ({ on: () => ({ subscribe: () => ({}) }) }),
    removeChannel: () => {},
    functions: { invoke },
  };
  return { invoke, supabaseStub };
});
vi.mock("@/integrations/supabase/client", () => ({ supabase: supabaseStub }));
vi.mock("@/integrations/supabase/external-client", () => ({ supabase: supabaseStub }));
vi.mock("@/components/database/DatabaseQueryModal", () => ({ DatabaseQueryModal: () => null }));

import { DatabaseConnectionsListModal } from "../../src/components/database/DatabaseConnectionsListModal";
import { MCP_TOOL_BASE } from "../../src/lib/api/mcp-config";

type Probe = { status: number; body: unknown };

// The same-origin proxy: answers only GET /health on the MCP tool server.
// Anything else is a 404, as an unknown route would be.
function installProxy(health: Probe) {
  const calls: { path: string | null; base: string | null }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: RequestInit = {}) => {
      const h = new Headers(init.headers);
      const path = h.get("X-Backend-Path");
      const base = h.get("X-Backend-Base");
      calls.push({ path, base });
      if (String(url).endsWith("/api/public/backend-proxy") && path === "/health" && base === MCP_TOOL_BASE) {
        return new Response(JSON.stringify(health.body), {
          status: health.status,
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response("not found", { status: 404 });
    }),
  );
  return calls;
}

beforeEach(() => invoke.mockClear());
afterEach(() => vi.unstubAllGlobals());

// The status pill's text once the probe has answered (null until it renders).
function pillText(): string | null {
  const pill = [...document.querySelectorAll("span.rounded-full")].find((s) =>
    /service/i.test(s.textContent ?? ""),
  );
  return pill?.textContent?.trim() ?? null;
}
const pillSays = (text: string) => waitFor(() => expect(pillText()).toBe(text), { timeout: 3000 });

const openModal = () =>
  render(<DatabaseConnectionsListModal open dbType="postgresql" onOpenChange={() => {}} />);

describe("DB connections pill follows the MCP query service's own health check", () => {
  it("reads online when the MCP server's /health says healthy", async () => {
    const calls = installProxy({ status: 200, body: { status: "healthy", connections_registered: 1 } });
    openModal();
    await pillSays("Query service online");
    expect(calls).toContainEqual({ path: "/health", base: MCP_TOOL_BASE });
    // The edge function no install ships is no longer what decides the pill.
    expect(invoke).not.toHaveBeenCalledWith("check-mcp-health", expect.anything());
  });

  it("names the query service as unreachable when the proxy cannot reach it", async () => {
    installProxy({
      status: 200,
      body: { error: "SERVICE_UNAVAILABLE", fallback: true, message: "fetch failed" },
    });
    openModal();
    await pillSays("Query service unreachable");
  });

  it("names the query service and the HTTP status when /health fails", async () => {
    installProxy({ status: 503, body: { detail: "down" } });
    openModal();
    await pillSays("Query service unhealthy (HTTP 503)");
  });
});
