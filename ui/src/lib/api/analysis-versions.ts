import { supabase } from "@/integrations/supabase/client";
import { BACKEND_API_BASE } from "@/lib/api/backend-config";

export type AnalysisVersion = {
  prompt_ts: string;
  created_at?: string | null;
  prompt?: string | null;
  is_current?: boolean;
  has_code?: boolean;
  has_output?: boolean;
  has_viz?: boolean;
  code_object_key?: string | null;
  output_object_key?: string | null;
  viz_object_key?: string | null;
};

export type RestoreResult = {
  available?: boolean;
  detail?: string;
  code?: string | null;
  output_signed_url?: string | null;
  viz_signed_url?: string | null;
  restored_prompt_ts?: string | null;
};

/**
 * Same transport the other /analysis/{id}/... calls use: the backend proxy with
 * the Supabase bearer token attached.
 */
async function proxyBackend(path: string, init?: { method?: string; body?: unknown }): Promise<Response> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  if (!token) throw new Error("Please sign in.");
  const headers: Record<string, string> = {
    Authorization: `Bearer ${token}`,
    "X-Backend-Path": path,
    "X-Backend-Base": BACKEND_API_BASE,
  };
  if (init?.body !== undefined) headers["Content-Type"] = "application/json";
  return fetch("/api/public/backend-proxy", {
    method: init?.method ?? "GET",
    headers,
    body: init?.body !== undefined ? JSON.stringify(init.body) : undefined,
  });
}

/** GET /analysis/{analysis_id}/versions — tolerates bare array or wrapped payload. */
export async function listAnalysisVersions(analysisId: string): Promise<AnalysisVersion[]> {
  const res = await proxyBackend(`/analysis/${analysisId}/versions`);
  let body: any = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    throw new Error(body?.detail || body?.error || `Failed to load versions (${res.status})`);
  }
  const list = Array.isArray(body)
    ? body
    : Array.isArray(body?.versions)
      ? body.versions
      : Array.isArray(body?.checkpoints)
        ? body.checkpoints
        : [];
  return list.filter((v: any) => v && typeof v.prompt_ts === "string") as AnalysisVersion[];
}

/** POST /analysis/{analysis_id}/restore */
export async function restoreAnalysisVersion(analysisId: string, promptTs: string): Promise<RestoreResult> {
  const res = await proxyBackend(`/analysis/${analysisId}/restore`, {
    method: "POST",
    body: { prompt_ts: promptTs },
  });
  let body: any = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok || body?.available === false) {
    throw new Error(body?.detail || body?.error || `Restore failed (${res.status})`);
  }
  return (body ?? {}) as RestoreResult;
}

/** ts format: YYYYMMDD_HHMMSS */
export function parsePromptTs(ts: string): Date | null {
  const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/.exec(ts);
  if (!m) return null;
  const [, y, mo, d, h, mi, s] = m;
  return new Date(Number(y), Number(mo) - 1, Number(d), Number(h), Number(mi), Number(s));
}

export function versionDate(v: { created_at?: string | null; prompt_ts: string }): Date | null {
  if (v.created_at) {
    const d = new Date(v.created_at);
    if (!Number.isNaN(d.getTime())) return d;
  }
  return parsePromptTs(v.prompt_ts);
}

export function formatAbsolute(d: Date | null, fallback: string): string {
  if (!d) return fallback;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function formatRelative(d: Date | null): string {
  if (!d) return "";
  const secs = Math.round((Date.now() - d.getTime()) / 1000);
  if (secs < 45) return "just now";
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.round(days / 30);
  if (months < 12) return `${months}mo ago`;
  return `${Math.round(months / 12)}y ago`;
}

/** Event used to tell an open analysis page that a version was restored elsewhere. */
export const VERSION_RESTORED_EVENT = "avaloka:version-restored";

export type VersionRestoredDetail = { analysisId: string; promptTs: string; result: RestoreResult };

export function emitVersionRestored(detail: VersionRestoredDetail) {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent<VersionRestoredDetail>(VERSION_RESTORED_EVENT, { detail }));
}
