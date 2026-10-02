import { supabase } from "@/integrations/supabase/external-client";
import { BACKEND_API_BASE } from "@/lib/api/backend-config";
import type { RenderedPoint } from "@/lib/chart-points";

export type ExplainChart = {
  id: string;
  title: string;
  type: "bar" | "line" | "pie" | "scatter" | "histogram";
  x_field: string | null;
  y_field: string | null;
  aggregate: string | null;
  reason: string | null;
  points: RenderedPoint[];
};

export type ExplainRequest = {
  question: string;
  dataset_id: string | null;
  dataset_name: string | null;
  focus_chart_id: string | null;
  history: { role: "user" | "assistant"; text: string }[];
  charts: ExplainChart[];
};

export type ExplainResponse = { answer: string; focus_chart_ids: string[] };

export const EXPLAIN_MESSAGES = {
  401: "Your session expired. Please refresh the page and sign in again.",
  404: "I couldn't find this analysis. Try reopening it from your analyses list.",
  422: "I couldn't read that question. Try asking it a different way.",
  network: "I'm having trouble reaching the server. Please try again in a moment.",
} as const;

export class ExplainError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
  }
}

const TIMEOUT_MS = 30_000;

/** POST /analysis/{aid}/insights/explain via the same proxy + auth as /analysis/{aid}/code. */
export async function explainInsights(aid: string, body: ExplainRequest, signal: AbortSignal): Promise<ExplainResponse> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  if (!token) throw new ExplainError(EXPLAIN_MESSAGES[401], 401);

  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctrl.abort();
  }, TIMEOUT_MS);
  const onAbort = () => ctrl.abort();
  if (signal.aborted) ctrl.abort();
  else signal.addEventListener("abort", onAbort, { once: true });

  try {
    const res = await fetch(`${window.location.origin}/api/public/backend-proxy`, {
      method: "POST",
      signal: ctrl.signal,
      headers: {
        "X-Backend-Path": `/analysis/${encodeURIComponent(aid)}/insights/explain`,
        "X-Backend-Base": BACKEND_API_BASE,
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    if (res.status === 401 || res.status === 404 || res.status === 422) {
      throw new ExplainError(EXPLAIN_MESSAGES[res.status], res.status);
    }
    if (!res.ok) throw new ExplainError(EXPLAIN_MESSAGES.network, res.status);
    const json = (await res.json()) as Partial<ExplainResponse>;
    return {
      answer: String(json.answer ?? "").trim() || "I don't have an answer for that yet.",
      focus_chart_ids: Array.isArray(json.focus_chart_ids) ? json.focus_chart_ids.map(String) : [],
    };
  } catch (err) {
    if (err instanceof ExplainError) throw err;
    if (signal.aborted && !timedOut) throw new DOMException("Aborted", "AbortError");
    throw new ExplainError(EXPLAIN_MESSAGES.network);
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", onAbort);
  }
}
