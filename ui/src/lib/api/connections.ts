import { supabase } from "@/integrations/supabase/client";
import { BACKEND_API_BASE as API_BASE } from "@/lib/api/backend-config";

/**
 * Saved database connections (the API's /api/connections routes).
 *
 * The only way the UI lists, saves or deletes a database connection, in every
 * auth mode: the API decides where rows live (Supabase when hosted, its own
 * store in local mode), so the UI never touches the mcp_connections table.
 *
 * `create` registers the database AND saves the connection in one server-side
 * step; if saving fails the server rolls the registration back. The password
 * travels in that one request body and is never kept here, and no response
 * carries a secret.
 */

export type SavedConnection = {
  id: string;
  connection_name: string;
  customer_id: string;
  customer_name: string | null;
  database_type: string;
  host: string;
  port: number;
  database_name: string;
  mcp_endpoint: string | null;
  status: string | null;
  query_timeout: number | null;
  max_rows: number | null;
  created_at: string;
};

export type NewConnection = {
  connection_name: string;
  customer_id: string;
  customer_name: string;
  contact_email: string;
  environment: string;
  database_type: string;
  host: string;
  port: number;
  database_name: string;
  username: string;
  password: string;
  query_timeout: number;
  max_rows: number;
  connection_string?: string;
};

export class ConnectionsApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

function proxyUrl(): string {
  return typeof window !== "undefined"
    ? `${window.location.origin}/api/public/backend-proxy`
    : `/api/public/backend-proxy`;
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const {
    data: { session },
  } = await supabase.auth.getSession();
  const token = session?.access_token;
  if (!token) throw new ConnectionsApiError("Not signed in.", 401);
  const headers = new Headers(init.headers as HeadersInit | undefined);
  headers.set("X-Backend-Path", path);
  headers.set("X-Backend-Base", API_BASE);
  headers.set("Authorization", `Bearer ${token}`);
  if (init.body) headers.set("Content-Type", "application/json");
  return fetch(proxyUrl(), { ...init, headers });
}

/** The parsed body, or a ConnectionsApiError carrying the API's own `detail`. */
async function parse<T>(res: Response): Promise<T> {
  const body: any = await res.json().catch(() => ({}));
  // The backend proxy answers 200 + {fallback:true} when the API is unreachable.
  if (body?.fallback) {
    throw new ConnectionsApiError(body?.message || "The API is unreachable.", 503);
  }
  if (!res.ok) {
    const d = body?.detail;
    const detail =
      typeof d === "string"
        ? d
        : Array.isArray(d)
          ? d.map((x: any) => x?.msg || JSON.stringify(x)).join("; ")
          : res.statusText || `HTTP ${res.status}`;
    throw new ConnectionsApiError(detail, res.status);
  }
  return body as T;
}

export const connectionsApi = {
  async list(): Promise<SavedConnection[]> {
    const body = await parse<{ connections?: SavedConnection[] }>(
      await request("/api/connections", { method: "GET" }),
    );
    return Array.isArray(body?.connections) ? body.connections : [];
  },

  async create(input: NewConnection): Promise<SavedConnection> {
    return parse<SavedConnection>(
      await request("/api/connections", { method: "POST", body: JSON.stringify(input) }),
    );
  },

  async update(
    id: string,
    patch: { connection_name?: string; status?: "active" | "inactive" },
  ): Promise<SavedConnection> {
    return parse<SavedConnection>(
      await request(`/api/connections/${encodeURIComponent(id)}`, {
        method: "PUT",
        body: JSON.stringify(patch),
      }),
    );
  },

  async remove(id: string): Promise<void> {
    await parse(
      await request(`/api/connections/${encodeURIComponent(id)}`, { method: "DELETE" }),
    );
  },
};
