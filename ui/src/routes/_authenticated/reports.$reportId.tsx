import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import {
  ArrowLeft,
  Bell01,
  SearchLg,
  Copy01,
  Stars02,
} from "@untitledui/icons";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { getCachedAuthUser } from "@/lib/auth-user";

import { Input } from "@/components/base/input/input";
import { Button } from "@/components/base/buttons/button";
import { ButtonUtility } from "@/components/base/buttons/button-utility";
import { ReportCommentsPanel } from "@/components/dashboard/ReportCommentsPanel";
import { ShareCollaboratorsButton } from "@/components/dashboard/ShareCollaboratorsButton";
import { InsightCommentCard } from "@/components/dashboard/AddCommentPopover";
import { DashboardMiniChart } from "@/components/dashboard/dashboardMiniChart";
import { dashboardFromVizConfig, type DashboardWidget } from "@/lib/api/project-dashboard";
import { fetchReportById } from "@/lib/reports";
import { toast } from "sonner";
import { PlanGate } from "@/components/dashboard/PlanGate";


export const Route = createFileRoute("/_authenticated/reports/$reportId")({
  head: ({ params }) => ({
    meta: [
      { title: `Report ${params.reportId} · Avaloka AI` },
      {
        name: "description",
        content: "Detailed report view with charts, key insights and collaboration tools.",
      },
      { property: "og:title", content: `Report ${params.reportId} · Avaloka AI` },
      {
        property: "og:description",
        content: "Detailed report view with charts, key insights and collaboration tools.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: () => (
    <PlanGate>
      <ReportDetailPage />
    </PlanGate>
  ),
});

type ReportPayload = {
  widgets?: DashboardWidget[];
  samples?: Record<string, unknown>[];
  source?: string;
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

  // Modern reports store the exact dashboard-tab widgets. Use them verbatim —
  // never fall back to the analysis's full chart list, which would leak
  // charts the user did not place on the dashboard.
  if (Array.isArray(payload.widgets)) {
    return payload.widgets
      .map((w, i) => coerceReportWidget(w, i, analysisId))
      .filter((w): w is DashboardWidget => Boolean(w));
  }

  // Legacy reports: key_insights is a raw array of widget snapshots.
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

function ReportDetailPage() {
  const navigate = useNavigate();
  const { reportId } = Route.useParams();
  const [query, setQuery] = useState("");
  
  const [shareOpen, setShareOpen] = useState(false);
  const [commentsOpen, setCommentsOpen] = useState(false);
  const [insightCommentOpen, setInsightCommentOpen] = useState(false);

  const { data: report, isPending } = useQuery({
    queryKey: ["report", reportId],
    queryFn: () => fetchReportById(reportId),
  });

  // Everyone with access to this route can comment. Only the report creator
  // may share it with others.
  const { data: currentUserId } = useQuery({
    queryKey: ["auth", "user-id"],
    queryFn: async () => {
      const user = await getCachedAuthUser();
      return user?.id ?? null;
    },
    staleTime: 5 * 60_000,
  });

  const creatorId =
    (typeof report?.created_by === "string" ? report.created_by : report?.created_by?.id) ??
    report?.created_by_id ??
    null;
  const canShare = !!(currentUserId && creatorId && currentUserId === creatorId);
  const canComment = true;



  const title = report?.title ?? "Report";
  const datasetLabel = report?.dataset_label ?? "Dataset";
  const projectName = report?.projects?.name ?? "Project";
  const analysisName = report?.analyses?.name ?? "Analysis";
  const payload = useMemo(() => parsePayload(report?.key_insights), [report?.key_insights]);
  const widgets = useMemo(
    () =>
      resolveReportWidgets(report?.key_insights, report?.analysis_id, report?.analyses?.viz_config)
        .slice()
        .sort((a, b) => (a.position ?? 0) - (b.position ?? 0)),
    [report?.key_insights, report?.analysis_id, report?.analyses?.viz_config],
  );
  const samples = payload.samples?.length ? payload.samples : normalizeSamples(report?.analyses?.samples);




  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col bg-primary text-primary">
      <header className="flex items-center justify-between gap-3 border-b border-secondary bg-primary px-6 py-3">
        <div className="flex items-center gap-3">
          <ButtonUtility
            size="sm"
            color="tertiary"
            icon={ArrowLeft}
            tooltip="Back to reports"
            onClick={() => navigate({ to: "/reports" })}
          />
          <div className="leading-tight">
            <div className="flex items-center gap-2">
              <h1 className="text-md font-semibold text-primary">{title}</h1>
              <span className="inline-flex items-center rounded-full bg-success-secondary px-2 py-0.5 text-xs font-medium text-fg-success-primary ring-1 ring-inset ring-success-subtle">
                {datasetLabel}
              </span>
            </div>
            <p className="text-xs text-tertiary">
              {projectName} · {analysisName} · Report {reportId}
            </p>
          </div>
        </div>
        <div className="relative flex items-center gap-2">
          {canShare && (
            <ShareCollaboratorsButton
              analysisId={report?.analysis_id ?? null}
              projectId={report?.project_id ?? null}
              reportId={report?.id ?? null}
              open={shareOpen}
              onOpenChange={setShareOpen}
            />
          )}
          {canComment && (
            <Button
              color={commentsOpen ? "primary" : "secondary"}
              size="sm"
              iconLeading={Bell01}
              onClick={() => setCommentsOpen((v) => !v)}
            >
              Notifications
            </Button>
          )}

          {commentsOpen && canComment && (
            <div className="absolute right-0 top-12 z-30">
              <ReportCommentsPanel
                onClose={() => setCommentsOpen(false)}
                reportId={report?.id ?? null}
              />
            </div>
          )}

        </div>



      </header>


      <div className="flex flex-1 overflow-hidden">
        <div className="relative flex flex-1 flex-col overflow-hidden">
          <div className="flex items-center justify-end gap-2 border-b border-secondary bg-primary px-6 py-3">
            <div className="w-[280px]">
              <Input
                aria-label="Search"
                placeholder="Search"
                size="sm"
                icon={SearchLg}
                shortcut="⌘K"
                value={query}
                onChange={setQuery}
              />
            </div>
          </div>
          <div className="flex-1 overflow-y-auto">


        <div className="mx-auto max-w-6xl px-6 py-6">
          <p className="mb-2 text-right text-xs text-tertiary">Thursday 10:16am</p>

          {isPending ? (
            <div className="mt-4 rounded-xl border border-secondary bg-secondary/30 p-4 text-sm text-secondary">
              Loading report…
            </div>
          ) : widgets.length === 0 ? (
            <div className="mt-4 rounded-xl border border-secondary bg-secondary/30 p-4 text-sm text-secondary">
              No charts or insights available for this report.
            </div>
          ) : (
            <div className="mt-4 space-y-4">
              {widgets.map((widget) => {
                const widgetInsights = insightsFromWidget(widget);
                return (
                  <div
                    key={widget.id}
                    className="rounded-xl border border-secondary bg-primary p-4"
                  >
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

          <div className="relative mt-3 flex items-center gap-3 px-1 text-fg-quaternary">
            <button
              type="button"
              className="hover:text-secondary"
              aria-label="Copy insights"
              onClick={() => {
                const text = widgets
                  .map((w) => {
                    const list = insightsFromWidget(w);
                    return list.length ? `${w.title}\n${list.map((i) => `• ${i}`).join("\n")}` : "";
                  })
                  .filter(Boolean)
                  .join("\n\n");
                if (!text) {
                  toast.message("Nothing to copy");
                  return;
                }
                void navigator.clipboard
                  ?.writeText(text)
                  .then(() => toast.success("Insights copied"))
                  .catch(() => toast.error("Copy failed"));
              }}
            >
              <Copy01 className="size-4" />
            </button>
            {canComment && (
              <button
                type="button"
                className={insightCommentOpen ? "text-primary" : "hover:text-secondary"}
                aria-label="Comment"
                onClick={() => setInsightCommentOpen((v) => !v)}
              >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="size-4">
                  <path d="M14 8C14 11.3137 11.3137 14 8 14C7.2019 14 6.4402 13.8442 5.74366 13.5613C5.61035 13.5072 5.54369 13.4801 5.48981 13.468C5.43711 13.4562 5.3981 13.4519 5.34409 13.4519C5.28887 13.4519 5.22872 13.4619 5.10843 13.4819L2.73651 13.8772C2.48812 13.9186 2.36393 13.9393 2.27412 13.9008C2.19552 13.8671 2.13289 13.8045 2.09917 13.7259C2.06065 13.6361 2.08135 13.5119 2.12275 13.2635L2.51807 10.8916C2.53812 10.7713 2.54814 10.7111 2.54814 10.6559C2.54813 10.6019 2.54381 10.5629 2.532 10.5102C2.51992 10.4563 2.49285 10.3897 2.43871 10.2563C2.15582 9.5598 2 8.7981 2 8C2 4.68629 4.68629 2 8 2C11.3137 2 14 4.68629 14 8Z" stroke="currentColor" strokeWidth="1.33" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            )}
            <button type="button" className="hover:text-secondary" aria-label="Insights">
              <Stars02 className="size-4" />
            </button>
          </div>

          {insightCommentOpen && canComment && (
            <div className="mt-4">
              <InsightCommentCard analysisId={null} reportId={report?.id ?? null} />
            </div>
          )}



        </div>
        </div>
        </div>
        
      </div>
    </div>

  );
}
