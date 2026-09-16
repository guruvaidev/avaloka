// Read-only viewer for shared reports via /shared-report/:token.
// Mirrors the authenticated report detail page rendering (widgets + key insights).

import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { DashboardMiniChart } from "@/components/dashboard/dashboardMiniChart";
import { dashboardFromVizConfig, type DashboardWidget } from "@/lib/api/project-dashboard";
import { Button } from "@/components/base/buttons/button";
import { toast } from "sonner";

export const Route = createFileRoute("/shared-report/$token")({
  head: () => ({
    meta: [
      { title: "Shared report · Avaloka" },
      { name: "robots", content: "noindex" },
    ],
  }),
  component: SharedReportPage,
});

type SharedComment = {
  id: string;
  text: string;
  created_at: string;
  parent_id: string | null;
  attachments: Array<{ name: string; path: string; mime_type?: string; size?: number }>;
  author: { id: string; full_name: string | null; avatar_url: string | null } | null;
};

type Payload = {
  access_level: string;
  is_public: boolean;
  report: {
    id: string;
    title: string | null;
    key_insights: unknown;
    analysis_id: string | null;
    project_id: string | null;
    created_at: string;
  };
  analysis: {
    id: string;
    name: string | null;
    viz_config: unknown;
    samples: unknown;
    filename: string | null;
  } | null;
  comments?: SharedComment[];
};

type ReportPayload = {
  widgets?: DashboardWidget[];
  samples?: Record<string, unknown>[];
};

type LegacyReportWidget = Partial<DashboardWidget> & {
  graph_key?: string;
  chart_type?: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function parseJsonValue(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}

function normalizeSamples(value: unknown): Record<string, unknown>[] {
  const parsed = parseJsonValue(value);
  return Array.isArray(parsed) ? parsed.filter(isRecord) : [];
}

function parsePayload(value: unknown): ReportPayload {
  if (!value) return {};
  if (typeof value === "string") {
    try {
      return parsePayload(JSON.parse(value));
    } catch {
      return {};
    }
  }
  if (typeof value === "object" && !Array.isArray(value)) {
    const v = value as ReportPayload;
    if (Array.isArray(v.widgets)) return v;
  }
  return {};
}

function parseLegacyReportWidgets(value: unknown): LegacyReportWidget[] {
  const parsed = parseJsonValue(value);
  if (!Array.isArray(parsed)) return [];
  return parsed.filter(isRecord) as LegacyReportWidget[];
}

function coerceReportWidget(widget: unknown, index: number, analysisId: string | undefined): DashboardWidget | null {
  if (!isRecord(widget)) return null;
  const graphKey = String(widget.graphKey ?? widget.graph_key ?? widget.id ?? `widget-${index}`);
  const config = isRecord(widget.config) ? widget.config : {};
  return {
    id: String(widget.id ?? `widget-${index}`),
    tabId: String(widget.tabId ?? `analysis-${analysisId ?? "report"}`),
    analysisId: String(widget.analysisId ?? analysisId ?? ""),
    title: String(widget.title ?? "Chart"),
    chartType: String(widget.chartType ?? (widget as { chart_type?: string }).chart_type ?? "bar"),
    config,
    position: typeof widget.position === "number" ? widget.position : index,
    placed: widget.placed !== false,
    graphKey,
  };
}

function resolveReportWidgets(
  reportKeyInsights: unknown,
  analysisId: string | undefined,
  analysisVizConfig: unknown,
): DashboardWidget[] {
  const payload = parsePayload(reportKeyInsights);
  if (Array.isArray(payload.widgets)) {
    return payload.widgets
      .map((w, i) => coerceReportWidget(w, i, analysisId))
      .filter((w): w is DashboardWidget => Boolean(w));
  }
  const legacyWidgets = parseLegacyReportWidgets(reportKeyInsights);
  if (legacyWidgets.length) {
    const analysisWidgets = analysisId ? dashboardFromVizConfig(analysisId, analysisVizConfig).widgets : [];
    return legacyWidgets
      .map((legacy, index) => {
        const legacyGraphKey = legacy.graphKey ?? legacy.graph_key;
        const match = analysisWidgets.find(
          (widget) =>
            (legacyGraphKey && widget.graphKey === legacyGraphKey) ||
            (legacy.id && widget.id === legacy.id) ||
            (legacy.title && widget.title === legacy.title),
        );
        if (match) return { ...match, position: typeof legacy.position === "number" ? legacy.position : match.position };
        return coerceReportWidget(legacy, index, analysisId);
      })
      .filter((widget): widget is DashboardWidget => Boolean(widget));
  }
  return [];
}

function insightsFromWidget(widget: DashboardWidget): string[] {
  const cfg = widget.config as Record<string, unknown> | undefined;
  if (!cfg) return [];
  const charts = Array.isArray((cfg as { charts?: unknown }).charts)
    ? ((cfg as { charts: Record<string, unknown>[] }).charts ?? [])
    : [];
  const focusId = (cfg as { focusChartId?: unknown }).focusChartId;
  const chart =
    (focusId != null && charts.find((c) => String(c.id) === String(focusId))) ||
    charts.find((c) => String(c.id) === widget.graphKey) ||
    charts[0] ||
    (cfg as Record<string, unknown>);
  const raw =
    (chart as { insights?: unknown; key_insights?: unknown; takeaways?: unknown; notes?: unknown }) ?? {};
  const src =
    raw.insights ??
    raw.key_insights ??
    raw.takeaways ??
    raw.notes ??
    (chart as { reason?: unknown; description?: unknown }).reason ??
    (chart as { description?: unknown }).description ??
    [];
  const arr = Array.isArray(src) ? src : src ? [src] : [];
  return arr
    .map((it) => {
      if (typeof it === "string") return it;
      if (it && typeof it === "object") {
        const r = it as { text?: string; insight?: string; message?: string };
        return r.text ?? r.insight ?? r.message ?? "";
      }
      return "";
    })
    .filter((s): s is string => Boolean(s));
}

function SharedReportPage() {
  const { token } = Route.useParams();
  const [status, setStatus] = useState<"loading" | "unauth" | "ok" | "error">("loading");
  const [payload, setPayload] = useState<Payload | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [email, setEmail] = useState("");
  const [sending, setSending] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const consumeHashSession = async () => {
      if (typeof window === "undefined") return;
      const hash = window.location.hash;
      if (hash && hash.includes("access_token=")) {
        const params = new URLSearchParams(hash.replace(/^#/, ""));
        const access_token = params.get("access_token");
        const refresh_token = params.get("refresh_token");
        if (access_token && refresh_token) {
          await supabase.auth.setSession({ access_token, refresh_token });
          history.replaceState(null, "", window.location.pathname + window.location.search);
        }
      }
    };

    const load = async () => {
      await consumeHashSession();
      const { data: sess } = await supabase.auth.getSession();
      const accessToken = sess.session?.access_token;
      try {
        const res = await fetch(`/api/public/shared-report/${token}`, {
          headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : undefined,
        });
        if (res.status === 401 || res.status === 403) {
          if (!cancelled) setStatus("unauth");
          return;
        }
        if (res.status === 404) {
          if (!cancelled) {
            setStatus("error");
            setErrorMsg("This link is no longer available.");
          }
          return;
        }
        if (!res.ok) throw new Error(`Request failed (${res.status})`);
        const data = (await res.json()) as Payload;
        if (!cancelled) {
          setPayload(data);
          setStatus("ok");
        }
      } catch (err) {
        if (!cancelled) {
          setStatus("error");
          setErrorMsg(err instanceof Error ? err.message : "Failed to load shared report");
        }
      }
    };

    void load();
    const { data: sub } = supabase.auth.onAuthStateChange((_e, session) => {
      if (session?.access_token) void load();
    });
    return () => {
      cancelled = true;
      sub.subscription.unsubscribe();
    };
  }, [token]);

  const sendMagicLink = async () => {
    if (!email.trim()) return;
    setSending(true);
    try {
      const { error } = await supabase.auth.signInWithOtp({
        email: email.trim().toLowerCase(),
        options: { emailRedirectTo: window.location.href },
      });
      if (error) throw error;
      toast.success("Check your email for the sign-in link");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to send link");
    } finally {
      setSending(false);
    }
  };

  const parsedPayload = useMemo(() => parsePayload(payload?.report.key_insights), [payload?.report.key_insights]);
  const widgets = useMemo(
    () =>
      resolveReportWidgets(
        payload?.report.key_insights,
        payload?.analysis?.id ?? undefined,
        payload?.analysis?.viz_config,
      )
        .slice()
        .sort((a, b) => (a.position ?? 0) - (b.position ?? 0)),
    [payload?.report.key_insights, payload?.analysis?.id, payload?.analysis?.viz_config],
  );
  const samples = useMemo(
    () =>
      parsedPayload.samples?.length
        ? parsedPayload.samples
        : normalizeSamples(payload?.analysis?.samples),
    [parsedPayload.samples, payload?.analysis?.samples],
  );

  if (status === "loading") {
    return <div className="grid min-h-screen place-items-center bg-primary text-sm text-tertiary">Loading…</div>;
  }
  if (status === "unauth") {
    return (
      <div className="grid min-h-screen place-items-center bg-primary px-6">
        <div className="w-full max-w-sm rounded-2xl border border-secondary bg-primary p-6 shadow-sm">
          <h1 className="text-lg font-semibold text-primary">Sign in to view shared report</h1>
          <p className="mt-1 text-sm text-tertiary">This link requires sign-in. Ask the owner to make it public.</p>
          <form
            className="mt-4 space-y-3"
            onSubmit={(e) => {
              e.preventDefault();
              void sendMagicLink();
            }}
          >
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-primary shadow-xs placeholder:text-tertiary focus:border-[#1565ef] focus:outline-none"
            />
            <Button type="submit" color="primary" size="md" isDisabled={sending} isLoading={sending}>
              {sending ? "Sending…" : "Send magic link"}
            </Button>
          </form>
        </div>
      </div>
    );
  }
  if (status === "error" || !payload) {
    return (
      <div className="grid min-h-screen place-items-center bg-primary px-6">
        <div className="max-w-md text-center">
          <h1 className="text-lg font-semibold text-primary">Unavailable</h1>
          <p className="mt-2 text-sm text-tertiary">{errorMsg}</p>
        </div>
      </div>
    );
  }

  const title = payload.report.title ?? payload.analysis?.name ?? "Report";
  const datasetLabel = payload.analysis?.filename ?? "Dataset";

  return (
    <div className="flex min-h-screen flex-col bg-primary text-primary">
      <header className="border-b border-secondary bg-primary px-6 py-4">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-3">
          <div className="leading-tight">
            <p className="text-xs uppercase tracking-wide text-tertiary">Shared report · Read-only</p>
            <div className="mt-1 flex items-center gap-2">
              <h1 className="text-md font-semibold text-primary">{title}</h1>
              <span className="inline-flex items-center rounded-full bg-success-secondary px-2 py-0.5 text-xs font-medium text-fg-success-primary ring-1 ring-inset ring-success-subtle">
                {datasetLabel}
              </span>
            </div>
            {payload.analysis?.name ? (
              <p className="text-xs text-tertiary">{payload.analysis.name}</p>
            ) : null}
          </div>
        </div>
      </header>
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl px-6 py-6">
          {widgets.length === 0 ? (
            <div className="mt-4 rounded-xl border border-secondary bg-secondary/30 p-4 text-sm text-secondary">
              No charts or insights available for this report.
            </div>
          ) : (
            <div className="mt-4 space-y-4">
              {widgets.map((widget) => {
                const widgetInsights = insightsFromWidget(widget);
                return (
                  <div key={widget.id} className="rounded-xl border border-secondary bg-primary p-4">
                    <p className="text-sm font-semibold text-primary">{widget.title}</p>
                    <div className="mt-3 min-h-[220px]">
                      <DashboardMiniChart widget={widget} samples={samples} height={220} />
                    </div>
                    {widgetInsights.length > 0 && (
                      <div className="mt-4 rounded-lg border border-secondary bg-secondary/30 p-3">
                        <p className="text-xs font-semibold text-primary">Key Insights</p>
                        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-secondary">
                          {widgetInsights.map((insight, index) => (
                            <li key={index}>{insight}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          <section className="mt-6 rounded-xl border border-secondary bg-primary p-4">
            <h2 className="text-sm font-semibold text-primary">
              Comments {payload.comments?.length ? `(${payload.comments.length})` : ""}
            </h2>
            {!payload.comments || payload.comments.length === 0 ? (
              <p className="mt-2 text-sm text-tertiary">No comments yet.</p>
            ) : (
              <ol className="mt-3 space-y-4">
                {payload.comments.map((c) => {
                  const name = c.author?.full_name || "Unknown user";
                  const initials =
                    name
                      .split(/\s+/)
                      .map((p) => p[0])
                      .filter(Boolean)
                      .slice(0, 2)
                      .join("")
                      .toUpperCase() || "?";
                  const when = new Date(c.created_at).toLocaleString();
                  return (
                    <li key={c.id} className={"flex gap-3 " + (c.parent_id ? "pl-8" : "")}>
                      <div className="flex size-9 shrink-0 items-center justify-center overflow-hidden rounded-full bg-[#1565ef]/10 text-xs font-semibold text-[#1565ef]">
                        {c.author?.avatar_url ? (
                          <img src={c.author.avatar_url} alt={name} className="size-full object-cover" />
                        ) : (
                          initials
                        )}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-sm font-semibold text-primary truncate">{name}</span>
                          <span className="shrink-0 text-[11px] text-tertiary">{when}</span>
                        </div>
                        {c.text ? (
                          <p className="mt-0.5 whitespace-pre-wrap break-words text-sm text-secondary">
                            {c.parent_id ? "↳ " : ""}
                            {c.text}
                          </p>
                        ) : null}
                        {c.attachments.length > 0 && (
                          <ul className="mt-1 list-disc pl-5 text-xs text-tertiary">
                            {c.attachments.map((a, i) => (
                              <li key={i}>{a.name}</li>
                            ))}
                          </ul>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ol>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
