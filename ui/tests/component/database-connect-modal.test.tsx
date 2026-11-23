// The database connect modal and connections list talk to the API's
// /api/connections routes, never to the mcp_connections table. In local auth
// mode (the quickstart) that table does not exist: the stand-in answers 501 to
// supabase.from(...).insert, which is how "Connect the sample database" used
// to end at "Saving connection failed" after the registration had succeeded.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const supabaseFrom = vi.fn(() => {
  throw new Error("the UI must not read or write Supabase tables for connections");
});

vi.mock("@/integrations/supabase/client", () => ({
  supabase: {
    from: (...args: unknown[]) => supabaseFrom(...(args as [])),
    auth: {
      getSession: async () => ({ data: { session: { access_token: "local-token" } } }),
      getUser: async () => ({ data: { user: { id: "local-user" } } }),
      onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => undefined } } }),
    },
    channel: () => {
      throw new Error("no realtime channel on mcp_connections");
    },
    removeChannel: () => undefined,
  },
}));
vi.mock("@/lib/api/backendApi", () => ({
  backendApi: { checkMcpHealth: async () => ({ status: "ok", healthy: true }) },
  // The real one, minus the session header: path in X-Backend-Path, same-origin proxy.
  proxyFetch: (path: string, init: RequestInit = {}) => {
    const headers = new Headers(init.headers as HeadersInit | undefined);
    headers.set("X-Backend-Path", path);
    return fetch("/api/public/backend-proxy", { ...init, headers });
  },
  mcpProxyFetch: () => {
    throw new Error("unexpected mcpProxyFetch");
  },
}));
vi.mock("@tanstack/react-router", () => ({ useNavigate: () => () => undefined }));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

import { DEFAULT_MAX_ROWS, DatabaseConnectModal } from "@/components/database/DatabaseConnectModal";
import { DatabaseConnectionsListModal } from "@/components/database/DatabaseConnectionsListModal";
import { DatabaseQueryModal } from "@/components/database/DatabaseQueryModal";

const DB_PASSWORD = "Sample-Db-Pw-q7";
const SAVED = {
  id: "conn-1",
  connection_name: "Sample orders",
  customer_id: "sample_db",
  customer_name: "Sample",
  database_type: "postgresql",
  host: "postgres",
  port: 5432,
  database_name: "sample",
  mcp_endpoint: "http://mcp:8010/sse",
  status: "active",
  query_timeout: 30,
  max_rows: 1000,
  created_at: "2026-10-19T00:00:00Z",
};

type Call = { path: string | null; method: string; body: any };
let calls: Call[] = [];
let respond: (c: Call) => { status: number; body: unknown };

beforeEach(() => {
  calls = [];
  supabaseFrom.mockClear();
  window.localStorage.clear();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_url: string, init: RequestInit) => {
      const headers = new Headers(init.headers as HeadersInit);
      const call = {
        path: headers.get("X-Backend-Path"),
        method: init.method || "GET",
        body: init.body ? JSON.parse(String(init.body)) : null,
      };
      calls.push(call);
      const { status, body } = respond(call);
      return new Response(JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
});

afterEach(() => vi.unstubAllGlobals());

function field(label: string): HTMLInputElement {
  const input = screen.getByText(label).parentElement?.querySelector("input");
  if (!input) throw new Error(`no input for ${label}`);
  return input as HTMLInputElement;
}

function fillSampleDatabase() {
  const set = (label: string, value: string) =>
    fireEvent.change(field(label), { target: { value } });
  set("Connection Name *", "Sample orders");
  set("Customer ID *", "sample_db");
  set("Customer Name *", "Sample");
  set("Contact Email *", "me@example.com");
  set("Host *", "postgres");
  set("Database Name *", "sample");
  set("Username *", "avaloka");
  set("Password *", DB_PASSWORD);
}

describe("DatabaseConnectModal", () => {
  it("saves through POST /api/connections in one call, never supabase.from", async () => {
    respond = () => ({ status: 201, body: SAVED });
    const onConnected = vi.fn();
    const onOpenChange = vi.fn();
    render(
      <DatabaseConnectModal
        open
        dbType="postgresql"
        onOpenChange={onOpenChange}
        onConnected={onConnected}
      />,
    );
    fillSampleDatabase();
    fireEvent.click(screen.getByText("Test & Register"));

    await waitFor(() => expect(onConnected).toHaveBeenCalledWith("conn-1"));
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ path: "/api/connections", method: "POST" });
    expect(calls[0].body).toMatchObject({
      connection_name: "Sample orders",
      customer_id: "sample_db",
      database_type: "postgresql",
      host: "postgres",
      port: 5432,
      database_name: "sample",
      username: "avaloka",
    });
    // The per-query row cap the database registers with. At 1,000 the
    // quickstart's 2,000-row orders table was analysed as a sample of 1,000.
    expect(calls[0].body.max_rows).toBe(DEFAULT_MAX_ROWS);
    expect(calls[0].body.max_rows).toBeGreaterThanOrEqual(2000);
    expect(supabaseFrom).not.toHaveBeenCalled();
    expect(onOpenChange).toHaveBeenCalledWith(false);
    // The password went out in that one request and was kept nowhere.
    expect(JSON.stringify({ ...window.localStorage })).not.toContain(DB_PASSWORD);
    expect(field("Password *").value).toBe("");
  });

  it("shows the API's save failure (registration rolled back) and lets a retry succeed", async () => {
    let attempt = 0;
    respond = () =>
      ++attempt === 1
        ? {
            status: 500,
            body: {
              detail:
                "Saving connection failed: the connection could not be stored. The database registration was rolled back; you can try again.",
            },
          }
        : { status: 201, body: SAVED };
    const onConnected = vi.fn();
    render(
      <DatabaseConnectModal open dbType="postgresql" onOpenChange={() => {}} onConnected={onConnected} />,
    );
    fillSampleDatabase();
    fireEvent.click(screen.getByText("Test & Register"));
    await waitFor(() => expect(screen.getByText(/rolled back; you can try again/)).toBeTruthy());
    expect(onConnected).not.toHaveBeenCalled();

    fireEvent.click(screen.getByText("Test & Register"));
    await waitFor(() => expect(onConnected).toHaveBeenCalledWith("conn-1"));
    expect(calls.map((c) => `${c.method} ${c.path}`)).toEqual([
      "POST /api/connections",
      "POST /api/connections",
    ]);
    expect(supabaseFrom).not.toHaveBeenCalled();
  });

  it("keeps the already-registered message for a 409", async () => {
    respond = () => ({ status: 409, body: { detail: "Customer sample_db already registered" } });
    render(<DatabaseConnectModal open dbType="postgresql" onOpenChange={() => {}} />);
    fillSampleDatabase();
    fireEvent.click(screen.getByText("Test & Register"));
    await waitFor(() =>
      expect(screen.getByText(/"sample_db" is already registered/)).toBeTruthy(),
    );
  });
});

describe("DatabaseConnectionsListModal", () => {
  it("lists through GET /api/connections, never supabase.from", async () => {
    respond = () => ({ status: 200, body: { connections: [SAVED] } });
    render(<DatabaseConnectionsListModal open dbType="postgresql" onOpenChange={() => {}} />);
    await waitFor(() => expect(screen.getByText("Sample orders")).toBeTruthy());
    expect(calls[0]).toMatchObject({ path: "/api/connections", method: "GET" });
    expect(supabaseFrom).not.toHaveBeenCalled();
  });
});

describe("DatabaseQueryModal", () => {
  it("lists tables through the API by connection_id, with no api_key in the browser", async () => {
    respond = () => ({
      status: 200,
      body: { query_result: { tables: [{ name: "orders", columns: [{ name: "id", type: "integer" }] }] } },
    });
    render(<DatabaseQueryModal open connection={SAVED} onOpenChange={() => {}} />);
    fireEvent.click(screen.getByText("List tables"));
    await waitFor(() => expect(screen.getByText("orders")).toBeTruthy());
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ path: "/api/v1/database/query", method: "POST" });
    expect(calls[0].body).toMatchObject({ content: "list tables", connection_id: "conn-1" });
    expect(JSON.stringify(calls[0].body)).not.toMatch(/api_key|apiKey/);
    expect(supabaseFrom).not.toHaveBeenCalled();
  });
});
