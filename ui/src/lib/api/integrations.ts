import { supabase } from "@/integrations/supabase/external-client";
import { BACKEND_API_BASE as API_BASE } from "@/lib/api/backend-config";

/**
 * Integrations API client.
 *
 * All calls go through the same-origin backend proxy with the Supabase
 * bearer token, exactly like `backendApi`. Tokens (PATs) are only ever
 * passed through in-flight — never stored, cached or logged here.
 */

export type IntegrationProvider = "github" | "outlook" | "slack" | "jira";

export interface IntegrationState {
  provider: IntegrationProvider | string;
  enabled: boolean;
  has_token: boolean;
  using_system_default: boolean;
  config: { repo?: string } & Record<string, unknown>;
}

async function getAccessToken(): Promise<string> {
  const {
    data: { session },
  } = await supabase.auth.getSession();
  if (!session?.access_token) throw new Error("Not authenticated - please sign in");
  return session.access_token;
}

function proxyUrl(): string {
  return typeof window !== "undefined"
    ? `${window.location.origin}/api/public/backend-proxy`
    : `/api/public/backend-proxy`;
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const accessToken = await getAccessToken();
  const headers = new Headers(init.headers as HeadersInit | undefined);
  headers.set("X-Backend-Path", path);
  headers.set("X-Backend-Base", API_BASE);
  headers.set("Authorization", `Bearer ${accessToken}`);
  if (init.body) headers.set("Content-Type", "application/json");
  return fetch(proxyUrl(), { ...init, headers });
}

async function fail(res: Response): Promise<never> {
  let detail = `Request failed (${res.status})`;
  try {
    const body = await res.json();
    if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    /* ignore */
  }
  const err = new Error(detail) as Error & { status?: number };
  err.status = res.status;
  throw err;
}

export const integrationsApi = {
  async list(): Promise<IntegrationState[]> {
    const res = await request("/api/integrations", { method: "GET" });
    if (!res.ok) return fail(res);
    const data = await res.json();
    return Array.isArray(data?.integrations) ? data.integrations : [];
  },

  /** Validates AND saves the GitHub token + repo in one call. */
  async connectGithub(token: string, repo: string): Promise<{
    success: boolean;
    provider: string;
    repo: string;
    enabled: boolean;
    push_access: boolean;
  }> {
    const res = await request("/api/integrations/github/connect", {
      method: "POST",
      body: JSON.stringify({ token, repo }),
    });
    if (!res.ok) return fail(res);
    return res.json();
  },

  async patchGithub(patch: { enabled?: boolean; repo?: string }): Promise<{
    success: boolean;
    provider: string;
    enabled: boolean;
    repo: string;
  }> {
    const res = await request("/api/integrations/github", {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
    if (!res.ok) return fail(res);
    return res.json();
  },

  async disconnectGithub(): Promise<void> {
    const res = await request("/api/integrations/github", { method: "DELETE" });
    if (!res.ok && res.status !== 204) return fail(res);
  },
};

export const REPO_PATTERN = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
