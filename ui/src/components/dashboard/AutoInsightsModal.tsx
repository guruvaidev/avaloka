import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, ChevronRight } from "lucide-react";
import { ArrowLeft, ArrowRight, ChevronDown, X, Stars02, Table } from "@untitledui/icons";
import { Code02, Copy01, ThumbsDown, ThumbsUp } from "@untitledui/icons";
import { toast } from "sonner";
import { DynamicChart, normalizeVizConfig, type Slide } from "./dynamicChart";
import { AnalysisOutputTable } from "./AnalysisOutputTable";
import { backendApi } from "@/lib/api/backendApi";
import { buildInsightGroups } from "@/lib/insight-groups";
import { normalizePlan, planIncludes } from "@/lib/edition-comparison";
import { useActivePlan } from "@/lib/use-active-plan";
import { assistantOpenOnView } from "@/lib/insights-assistant-visibility";
import { InsightsAvatar, ExplainChartButton } from "@/components/insights-avatar/InsightsAvatar";
import { InsightsAvatarProvider, useInsightsAvatar, HIGHLIGHT_CLASS } from "@/components/insights-avatar/InsightsAvatarContext";
import { Button } from "@/components/base/buttons/button";
import { Badge } from "@/components/base/badges/badges";
import { Toggle } from "@/components/base/toggle/toggle";
import { CommentaryLabel, CommentaryText, commentaryPlainText } from "@/components/chat/CommentaryText";
import { COMMENTARY_TOOLTIP } from "@/lib/reply-segments";
import { AutoInsightsHeader, InsightNavigator, ChartStatsRail, ColumnStatisticsTab, ChartNavigation, type InsightView } from "./auto-insights/AutoInsightsPanels";
import { ChartScopeStrip } from "./auto-insights/ChartScopeStrip";

function chartIdOf(slide: Slide, groupId: string, si: number) {
  return slide.id ?? `${groupId}-${si}`;
}


function stripMarkdown(s: string) {
  return s
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\*(.*?)\*/g, "$1");
}

/** One short takeaway line shown under each Overview chart. */
function chartPoint(slide: Slide): string {
  const first = slide.insights?.find((i) => i?.trim());
  if (first) return stripMarkdown(first.trim());
  if (slide.subtitle?.trim()) return stripMarkdown(slide.subtitle.trim());
  const metrics = (slide.series ?? []).map((s) => s.name || s.dataKey).filter(Boolean);
  if (metrics.length && slide.xKey) return `${metrics.join(", ")} compared across ${slide.xKey}.`;
  if (metrics.length) return `Distribution of ${metrics.join(", ")}.`;
  return "Visual summary generated from this dataset.";
}

export interface InsightDataset {
  id: string;
  name: string;
  config?: any;
  status?: string;
  samples?: any[];
}

interface AutoInsightsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  config?: any;
  status?: string;
  datasetName?: string;
  samples?: any[];
  outputRows?: Record<string, unknown>[] | null;
  defaultView?: "table" | "chart";
  title?: string;
  subtitle?: string;
  aid?: string | null;
  /** All datasets from a grouped upload; renders a per-dataset tab switcher. */
  datasets?: InsightDataset[];
  primaryDatasetId?: string | null;
  onPrefillChat?: (text: string) => boolean | void;
}

type ColumnStat = {
  name?: string;
  dtype?: string;
  role?: string;
  missing_pct?: number;
  n_unique?: number;
  min?: unknown;
  max?: unknown;
  mean?: unknown;
  median?: unknown;
  first?: unknown;
  last?: unknown;
  top_values?: Array<{ value?: unknown; pct?: number }>;
  meaning?: string;
  meaning_segments?: unknown;
  predictive_value?: "high" | "medium" | "low";
  predictive_value_source?: string;
};

const finiteNumber = (value: unknown) => {
  if (value == null || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
};

const formatStatNumber = (value: unknown) => {
  const number = finiteNumber(value);
  return number == null ? "—" : number.toLocaleString("en-US", { maximumFractionDigits: 2 });
};

function columnSummary(column: ColumnStat) {
  const dtype = String(column.dtype ?? "").toLowerCase();
  if (/date|time/.test(dtype)) {
    const firstValue = column.first ?? column.min;
    const lastValue = column.last ?? column.max;
    const first = firstValue == null ? "—" : String(firstValue);
    const last = lastValue == null ? "—" : String(lastValue);
    return `${first} – ${last}`;
  }
  if (/int|float|double|decimal|number|numeric/.test(dtype)) {
    return `${formatStatNumber(column.min)} – ${formatStatNumber(column.max)} · median ${formatStatNumber(column.median)} · mean ${formatStatNumber(column.mean)}`;
  }
  return (Array.isArray(column.top_values) ? column.top_values : [])
    .slice(0, 3)
    .map((item) => `${String(item?.value ?? "—")} (${formatStatNumber(item?.pct)}%)`)
    .join(" · ");
}

export function overviewMetrics(config: any) {
  const dataset = config?.dataset ?? {};
  const columns = Array.isArray(config?.column_stats) ? config.column_stats : Array.isArray(config?.columns) ? config.columns : [];
  const rows = finiteNumber(dataset.total_rows ?? dataset.rows_sampled);
  const count = finiteNumber(dataset.column_count) ?? (Array.isArray(dataset.columns) ? dataset.columns.length : columns.length || null);
  const types: { num: number; cat: number; date: number } = columns.reduce((result: { num: number; cat: number; date: number }, column: any) => {
    const dtype = String(column.dtype ?? column.type ?? "").toLowerCase();
    if (/date|time/.test(dtype) || column.role === "time") result.date++;
    else if (/int|float|double|decimal|number|numeric/.test(dtype)) result.num++;
    else if (dtype || column.role) result.cat++;
    return result;
  }, { num: 0, cat: 0, date: 0 });
  const missingCount = finiteNumber(dataset.missing_cells ?? config?.profile_overview?.missing_cells);
  const missingPct = finiteNumber(dataset.missing_pct ?? config?.profile_overview?.missing_pct) ?? (missingCount != null && rows != null && count != null && rows * count > 0 ? missingCount / (rows * count) * 100 : null);
  const allRowsUsed = dataset.total_rows != null && Number(dataset.rows_sampled) === Number(dataset.total_rows);
  return { rows, count, types, missingCount, missingPct, allRowsUsed };
}

/** Shared, read-only profile data for the Overview summary cards. */
function overviewSummaryData(config: any) {
  const profile = config?.profile_overview;
  if (!profile || typeof profile !== "object") return null;
  const score = finiteNumber(profile.readiness_score);
  const metrics = overviewMetrics(config);
  // Each list item travels with its segments (2.0 commentary marking), when sent.
  const withSegments = (key: string, textOf: (item: any) => unknown) =>
    (Array.isArray(profile[key]) ? (profile[key] as unknown[]) : []).map((item: any, i) => ({
      text: typeof item === "string" ? item : textOf(item),
      segments: Array.isArray(profile[`${key}_segments`]) ? profile[`${key}_segments`][i] : undefined,
    }));
  const warningKey = Array.isArray(profile.red_flags) && profile.red_flags.length ? "red_flags" : "concerns";
  const warnings = withSegments(warningKey, (item) => item?.text ?? item?.message).filter((item): item is { text: string; segments: unknown } => typeof item.text === "string" && item.text.length > 0);
  const suggestions = [
    ...withSegments("analysis_suggestions", (item) => item?.text ?? item?.question).map((item) => ({ ...item, type: "Q" })),
    ...withSegments("quick_wins", (item) => item?.text ?? item?.message).map((item) => ({ ...item, type: "C" })),
  ].filter((item): item is { text: string; segments: unknown; type: string } => typeof item.text === "string" && item.text.trim().length > 0);
  const readiness = score == null ? null : Math.max(0, Math.min(100, score));
  // c-strict: the readiness score is the profiling model's, always; the domain
  // is plain only when the server says code decided it. Without segments (an
  // older payload) both are the model's.
  const scoreSegments = profile.readiness_score_segments ?? (readiness == null ? undefined : [{ text: `${readiness}/100`, kind: "commentary" }]);
  const domainSegments = profile.domain_segments ?? (profile.domain ? [{ text: String(profile.domain), kind: "commentary" }] : undefined);
  return { profile, metrics, warnings, suggestions, readiness, scoreSegments, domainSegments };
}

/** Domain tag, description and the readiness / rows / columns tiles. */
function DatasetOverviewCard({ config }: { config: any }) {
  const data = overviewSummaryData(config);
  if (!data) return null;
  const { profile, metrics, readiness, scoreSegments, domainSegments } = data;
  return <section className="ai-surface ai-overview-dataset">
    <div className="ai-overview-description"><h3>Dataset overview</h3>{profile.domain && <span className="ai-overview-domain"><CommentaryText text={String(profile.domain)} segments={domainSegments} /></span>}{profile.domain_reasoning && <p><CommentaryText text={profile.domain_reasoning} segments={profile.domain_reasoning_segments} /></p>}</div>
    <div className="ai-overview-tiles">
      {readiness != null && <div className="ai-overview-tile ai-overview-readiness"><div className="ai-readiness-ring" role="progressbar" aria-label="Data readiness" aria-valuemin={0} aria-valuemax={100} aria-valuenow={readiness}><svg viewBox="0 0 72 72" aria-hidden="true"><circle cx="36" cy="36" r="32" className="ai-ring-track"/><circle cx="36" cy="36" r="32" className="ai-ring-progress" pathLength="100" strokeDasharray={`${readiness} 100`} transform="rotate(-90 36 36)"/></svg><strong>{readiness}</strong></div><span>Data readiness <CommentaryText text={`${readiness}/100`} segments={scoreSegments} /></span></div>}
      {metrics.rows != null && <div className="ai-overview-tile"><span>Rows</span><strong>{formatStatNumber(metrics.rows)}</strong>{metrics.allRowsUsed && <small>all rows used</small>}{config?.dataset?.total_rows == null && <small>sample rows</small>}</div>}
      {metrics.count != null && <div className="ai-overview-tile"><span>Columns</span><strong>{metrics.count}</strong>{Object.values(metrics.types).some(Boolean) && <small>{Object.entries(metrics.types).filter(([, value]) => value > 0).map(([type, value]) => `${value} ${type}`).join(" · ")}</small>}</div>}
      {(metrics.missingPct != null || metrics.missingCount != null) && <div className="ai-overview-tile"><span>Missing</span>{metrics.missingPct != null && <strong>{formatStatNumber(metrics.missingPct)}%</strong>}{metrics.missingCount != null && <small>{metrics.missingCount === 0 ? "no empty cells" : `${formatStatNumber(metrics.missingCount)} empty cells`}</small>}</div>}
    </div>
  </section>;
}

/** Wraps the chat hand-off so Auto Insights closes (like its × button) unless the hand-off is refused. */
export function askInChatAndClose(onAsk: (text: string) => boolean | void, onOpenChange: (open: boolean) => void) {
  return (text: string) => {
    const ok = onAsk(text);
    if (ok !== false) onOpenChange(false);
    return ok;
  };
}

/** Suggested steps: each whole row asks its text in chat once (spinner + all rows disabled while handing off). */
export function SuggestedStepList({ items, onAsk }: { items: { type: string; text: string; segments?: unknown }[]; onAsk?: (text: string) => boolean | void }) {
  const [pendingIndex, setPendingIndex] = useState<number | null>(null);
  const busyRef = useRef(false);
  const ask = (i: number, raw: string) => {
    const text = String(raw ?? "").trim();
    if (!text || busyRef.current || !onAsk) return;
    busyRef.current = true;
    setPendingIndex(i);
    const ok = onAsk(text);
    if (ok === false) {
      busyRef.current = false;
      setPendingIndex(null);
    }
  };
  return <div className="ai-suggestion-list">{items.map((item, i) => {
    // The chat gets the suggestion's words, never its "commentary" label or suffix.
    const text = commentaryPlainText(String(item.text ?? ""), item.segments).trim();
    return <button key={i} type="button" className="ai-suggestion-row cursor-pointer text-foreground hover:bg-muted disabled:cursor-default" aria-label={`Ask in chat: ${text}`} disabled={pendingIndex !== null} onClick={() => ask(i, text)}><span className={`ai-step-tag ai-step-${item.type}`}>{item.type}</span><span className="ai-step-text"><CommentaryText text={String(item.text ?? "")} segments={item.segments} /></span>{pendingIndex === i ? <span role="status" aria-label="Sending" className="inline-block size-4 shrink-0 animate-spin rounded-full border-2 border-border border-t-primary" /> : <ChevronRight size={16} aria-hidden="true"/>}</button>;
  })}</div>;
}

/** "Watch out for" and "Suggested next steps", side by side. */
function DatasetOverviewActions({ config, onPrefillChat }: { config: any; onPrefillChat?: (text: string) => boolean | void }) {
  const data = overviewSummaryData(config);
  if (!data) return null;
  const { profile, warnings, suggestions } = data;
  if (!(warnings.length > 0 || profile.readiness_reasoning || suggestions.length > 0)) return null;
  return <div className="ai-overview-actions">
    {(warnings.length > 0 || profile.readiness_reasoning) && <section className="ai-surface ai-overview-action"><header><h3>Watch out for</h3><span className="ai-issues-count">{warnings.length} issues</span></header>{profile.readiness_reasoning && <p className="ai-overview-reason"><CommentaryText text={profile.readiness_reasoning} segments={profile.readiness_reasoning_segments} /></p>}<ul>{warnings.map((item, i) => <li className="ai-issue-row" key={i}><AlertTriangle size={16}/><span><CommentaryText text={item.text} segments={item.segments} /></span></li>)}</ul></section>}
    {suggestions.length > 0 && <section className="ai-surface ai-overview-action"><header><h3>Suggested next steps</h3><span className="ai-overview-reason">Click to ask in chat</span></header><SuggestedStepList items={suggestions} onAsk={onPrefillChat} /></section>}
  </div>;
}

/** Dataset overview card, followed by the action cards. */
export function DatasetOverview({ config, onPrefillChat }: { config: any; onPrefillChat?: (text: string) => boolean | void }) {
  if (!overviewSummaryData(config)) return null;
  return <div className="ai-overview-summary">
    <DatasetOverviewCard config={config} />
    <DatasetOverviewActions config={config} onPrefillChat={onPrefillChat} />
  </div>;
}


function OverviewInsight({ slide }: { slide: Slide }) {
  const [expanded, setExpanded] = useState(false);
  const [overflow, setOverflow] = useState(false);
  const insights = slide.insights.filter((text) => text?.trim());
  const text = insights.length ? insights.map((text) => stripMarkdown(text.trim())).join("\n") : chartPoint(slide);
  // A single insight keeps its segments; several are joined, so only the text can be shown.
  const segments = insights.length === 1 ? slide.insightSegments : undefined;
  return <div className="ai-overview-insight" onClick={(event) => event.stopPropagation()} onKeyDown={(event) => event.stopPropagation()}><p ref={(node) => { if (node && !expanded) { const next = node.scrollHeight > node.clientHeight + 1; if (next !== overflow) setOverflow(next); } }} className={expanded ? "" : "ai-insight-clamped"}><CommentaryText text={text} segments={segments} renderText={stripMarkdown} /></p>{(overflow || expanded) && <Button color="link-gray" className="ai-insight-more" onPress={() => setExpanded(!expanded)}>{expanded ? "Show less" : "Show more"}</Button>}</div>;
}

function slugify(s: string) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

function chartKey(slide: Slide) {
  return String(slide.title ?? "").trim().toLowerCase() || JSON.stringify({
    type: slide.type,
    xKey: slide.xKey,
    series: slide.series.map((s) => s.dataKey),
  });
}

function dedupeSlides(slides: Slide[]) {
  const seen = new Set<string>();
  return slides.filter((slide) => {
    const key = chartKey(slide);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}


const FALLBACK_SLIDES: Slide[] = [
  {
    title: "Revenue Trend",
    type: "area",
    xKey: "month",
    series: [
      { dataKey: "y2020", name: "2020", color: "#1565ef" },
      { dataKey: "y2021", name: "2021", color: "#f97066" },
    ],
    data: [
      { month: "Jan", y2020: 48, y2021: -38 },
      { month: "Feb", y2020: 76, y2021: 58 },
      { month: "March", y2020: -20, y2021: 68 },
      { month: "April", y2020: 98, y2021: -50 },
      { month: "May", y2020: 40, y2021: 90 },
      { month: "June", y2020: 72, y2021: 98 },
      { month: "July", y2020: 88, y2021: 78 },
      { month: "Aug", y2020: 42, y2021: -22 },
      { month: "Sept", y2020: 18, y2021: 22 },
      { month: "Oct", y2020: -96, y2021: 80 },
    ],
    insights: [
      "Revenue increased 8.2% year-over-year.",
      "Peak month: November. Weakest month: March.",
      "Growth driven primarily by Branch North and Electronics category.",
    ],
  },
];

function InsightCard({ slide, config, outputRows, samples, defaultView = "chart", chartId, index, total, onSelect }: {
  slide: Slide; config?: any; outputRows?: Record<string, unknown>[] | null; samples?: any[];
  defaultView?: "table" | "chart"; chartId: string; index: number; total: number; onSelect: (index: number) => void;
}) {
  const avatar = useInsightsAvatar();
  const [view, setView] = useState<"table" | "chart">(defaultView);
  const [expanded, setExpanded] = useState(false);
  const [plotHeight, setPlotHeight] = useState(360);
  const [plotNode, setPlotNode] = useState<HTMLDivElement | null>(null);
  const tableRows = useMemo<Record<string, unknown>[] | null>(() => {
    if (Array.isArray(outputRows) && outputRows.length > 0) return outputRows;
    const sourceRows = Array.isArray(samples) && samples.length > 0 ? samples : slide.data;
    if (!Array.isArray(sourceRows) || !sourceRows.length) return null;
    const first = sourceRows[0];
    if (first && typeof first === "object" && !Array.isArray(first)) return sourceRows;
    if (Array.isArray(first)) {
      const cols = first.map((_, i) => `col_${i + 1}`);
      return sourceRows.map((row) => Object.fromEntries(cols.map((column, i) => [column, row[i]])));
    }
    return null;
  }, [outputRows, samples, slide.data]);
  useEffect(() => { setView(defaultView); setExpanded(false); }, [defaultView, chartId]);
  useEffect(() => {
    if (!plotNode) return;
    const measure = () => setPlotHeight(Math.max(320, Math.floor(plotNode.getBoundingClientRect().height) - (slide.type === "line" || slide.type === "area" ? 24 : 0)));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(plotNode);
    return () => observer.disconnect();
  }, [plotNode, slide.type, chartId]);
  const takeaway = stripMarkdown(slide.headline || slide.subtitle || slide.insights[0] || "");
  // The takeaway is marked as the model's when its source field carries segments.
  const takeawaySegments = slide.headline ? slide.headlineSegments : slide.subtitle ? undefined : slide.insights.length === 1 ? slide.insightSegments : undefined;
  return <section {...avatar.bindChart(chartId)} className={"ai-chart-card ai-surface " + (avatar.isHighlighted(chartId) ? HIGHLIGHT_CLASS : "")}>
    <div className="ai-card-heading"><div className="ai-chart-title"><h3><CommentaryText text={slide.title} segments={slide.titleSegments} /></h3>{takeaway && <div className="ai-takeaway"><p className={expanded ? "" : "line-clamp-2"}><CommentaryText text={takeaway} segments={takeawaySegments} renderText={stripMarkdown} /></p>{takeaway.length > 100 && <Button color="link-gray" className="ai-more" onPress={() => setExpanded((value) => !value)}>{expanded ? "Less" : "More"}</Button>}</div>}</div>
      <div className="ai-card-tools"><div className="ai-view-toggle">{(['table', 'chart'] as const).map((mode) => <Button key={mode} color="tertiary" className={`ai-control ${view === mode ? 'ai-view-active' : ''}`} isDisabled={mode === 'table' && !tableRows?.length} onPress={() => setView(mode)}>{mode === 'table' ? 'Data table' : 'Chart'}</Button>)}</div><ExplainChartButton chartId={chartId} /></div>
    </div>
    <ChartScopeStrip dataset={config?.dataset} />
    <div ref={setPlotNode} className="ai-plot">{view === 'table' && tableRows?.length ? <div className="ai-data-table"><AnalysisOutputTable rows={tableRows} title="Output" filename={`${slugify(slide.title || 'output')}.csv`} /></div> : <DynamicChart key={chartId} slide={slide} height={plotHeight} stackPresentation chartKey={chartId} />}</div>
    <ChartNavigation index={index} total={total} onSelect={onSelect} />
  </section>;
}


function CodeViewerModal({
  open,
  onClose,
  code,
  loading,
}: {
  open: boolean;
  onClose: () => void;
  code: string;
  loading: boolean;
}) {
  if (!open) return null;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      toast.success("Copied");
    } catch {
      toast.error("Copy failed");
    }
  };
  return (
    <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/50 p-4" role="dialog" aria-modal="true">
      <div className="flex max-h-[85vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-[var(--color-ai-line)] bg-[var(--color-ai-surface)] shadow-2xl">
        <div className="flex items-center justify-between border-b border-[var(--color-ai-line)] px-5 py-3">
          <h3 className="text-[15px] font-semibold text-[var(--color-ai-heading)]">Analysis code</h3>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={copy}
              disabled={loading || !code}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--color-ai-line)] bg-[var(--color-ai-surface)] px-2.5 py-1.5 text-[12px] font-semibold text-[var(--color-ai-body)] hover:bg-[var(--color-ai-thead)] disabled:opacity-50"
            >
              <Copy01 className="size-3.5" />
              Copy
            </button>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="grid size-8 place-items-center rounded-lg border border-[var(--color-ai-line)] bg-[var(--color-ai-surface)] text-[var(--color-ai-body)] hover:bg-[var(--color-ai-thead)]"
            >
              <X className="size-4" />
            </button>
          </div>
        </div>
        <div className="min-h-0 flex-1 overflow-auto bg-[#0b1020] p-4">
          {loading ? (
            <p className="text-sm text-[#cbd5e1]">Loading…</p>
          ) : (
            <pre className="whitespace-pre text-xs leading-relaxed text-[#e6edf3]"><code>{code}</code></pre>
          )}
        </div>
      </div>
    </div>
  );
}


function AutoInsightsModalInner({ open, onOpenChange, config, status, datasetName, samples, outputRows, defaultView = "chart", title, subtitle, aid, datasets, primaryDatasetId, onPrefillChat }: AutoInsightsModalProps) {
  const [index, setIndex] = useState(0);
  const [layout, setLayout] = useState<InsightView>("stack");
  const [codeOpen, setCodeOpen] = useState(false);
  const [codeText, setCodeText] = useState("");
  const [codeLoading, setCodeLoading] = useState(false);
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null);
  const avatar = useInsightsAvatar();
  // Insights Assistant visibility: each tab applies its default; closing it in
  // Stack keeps it closed for this Auto Insights view (reset per dataset).
  const [assistantOpen, setAssistantOpen] = useState(true);
  const [assistantDismissed, setAssistantDismissed] = useState(false);
  const handleAssistantOpenChange = (next: boolean) => {
    setAssistantOpen(next);
    if (!next && layout === "stack") setAssistantDismissed(true);
  };

  // One entry per DISTINCT dataset_id — each keeps a reference to its own entry
  // so two sections can never share the same visualization_config object.
  const tabs = useMemo(() => {
    const byId = new Map<string, InsightDataset>();
    (Array.isArray(datasets) ? datasets : []).forEach((d) => {
      if (!d?.id || byId.has(d.id)) return;
      byId.set(d.id, d);
    });
    return Array.from(byId.values());
  }, [datasets]);


  // Build ONE group per distinct dataset. Each group resolves its own config +
  // samples so a chart spec is never paired with another dataset's schema.
  const groups = useMemo(
    () =>
      buildInsightGroups({
        datasets: tabs,
        config,
        status,
        samples,
        outputRows,
        datasetName,
        primaryDatasetId,
        fallbackSlides: FALLBACK_SLIDES,
      }),
    [tabs, primaryDatasetId, status, config, samples, outputRows, datasetName],
  );


  const multi = groups.length > 1;

  useEffect(() => {
    setAssistantOpen(assistantOpenOnView(layout, assistantDismissed));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout]);

  useEffect(() => {
    setAssistantDismissed(false);
    setAssistantOpen(assistantOpenOnView(layout, false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeDatasetId]);

  // Default to the first dataset whenever the available groups change.
  useEffect(() => {
    setActiveDatasetId((prev) => {
      if (groups.some((g) => g.id === prev)) return prev;
      return groups[0]?.id ?? null;
    });
  }, [groups]);

  // With several datasets, default to the combined Overview so every dataset's
  // charts are visible together (user can still switch to Stack).
  useEffect(() => {
    if (multi) setLayout("grid");
  }, [multi]);

  const activeGroup = useMemo(
    () => groups.find((g) => g.id === activeDatasetId) || groups[0],
    [groups, activeDatasetId],
  );
  const visibleGroups = useMemo(
    () => (activeGroup ? [activeGroup] : groups),
    [activeGroup, groups],
  );

  // Flat, ordered index space for the stack view — scoped to the active dataset.
  const flat = useMemo(
    () =>
      visibleGroups.flatMap((g, gi) =>
        g.slides.map((slide, si) => ({ slide, group: g, gi, si })),
      ),
    [visibleGroups],
  );

  // Total across ALL datasets (used in the subtitle), not just the active one.
  const totalAll = useMemo(
    () => groups.reduce((sum, g) => sum + g.slides.length, 0),
    [groups],
  );


  const handleOpenCode = async () => {
    if (!aid) {
      toast.error("Save the analysis before viewing code");
      return;
    }
    setCodeOpen(true);
    setCodeLoading(true);
    setCodeText("");
    try {
      const res = (await backendApi.getAnalysisCode(aid)) as
        | { available?: boolean; code?: string | null; signed_url?: string }
        | null;
      if (!res) {
        setCodeOpen(false);
        toast.message("No code available for this analysis yet.");
        return;
      }
      if (res.available === false || !res.code) {
        setCodeOpen(false);
        toast.error("Code unavailable");
        return;
      }
      setCodeText(res.code);
    } catch (err: any) {
      setCodeOpen(false);
      toast.error(err?.message || "Failed to load code");
    } finally {
      setCodeLoading(false);
    }

  };


  const total = flat.length;
  const isPending = Boolean(activeGroup?.pending);
  const noInsights = !isPending && total === 0;

  const overviewScrollRef = useRef<HTMLDivElement>(null);

  // Opening the Overview tab always starts at the top, at Dataset overview.
  useEffect(() => {
    if (!open || layout !== "grid") return;
    // Optional call: jsdom (the component tests) has no Element.scrollTo.
    overviewScrollRef.current?.scrollTo?.({ top: 0 });
  }, [open, layout, activeDatasetId]);

  useEffect(() => {
    setIndex(0);
  }, [groups, activeDatasetId]);


  useEffect(() => {
    if (!open) return;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prevOverflow;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onOpenChange(false);
      else if (layout === "stack" && total > 0 && !(e.target instanceof HTMLElement && (e.target.closest('input, textarea, select, [contenteditable="true"]')))) {
        if (e.key === "ArrowRight") { e.preventDefault(); setIndex((i) => Math.min(i + 1, total - 1)); }
        else if (e.key === "ArrowLeft") { e.preventDefault(); setIndex((i) => Math.max(i - 1, 0)); }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onOpenChange, total, layout]);

  if (!open) return null;

  const avatarCharts = visibleGroups.flatMap((g) => g.slides.map((slide, si) => ({ id: chartIdOf(slide, g.id, si), slide })));
  const avatarDatasetId = activeGroup ? (activeGroup.id === "__single__" ? primaryDatasetId ?? null : activeGroup.id) : null;
  const avatarDatasetName = activeGroup?.name || datasetName || null;

  const activeEntry = total ? flat[Math.min(index, total - 1)] : null;
  const activeSlide = activeEntry?.slide;

  return (
    <div className="ai-modal fixed inset-0 z-[100] flex flex-col overflow-hidden" role="dialog" aria-modal="true" aria-labelledby="auto-insights-title">
      <AutoInsightsHeader groups={groups} active={activeGroup} view={layout} onDataset={(id) => { setActiveDatasetId(id); setIndex(0); }} onView={setLayout} onClose={() => onOpenChange(false)} />

      {layout === "grid" ? (
        <div className="ai-overview-scroll" ref={overviewScrollRef}>
          {isPending ? (
            <div className="grid h-full place-items-center text-sm text-[var(--color-ai-body)]">Generating insights…</div>
          ) : (
            <div className="ai-overview-content">
              <DatasetOverview config={activeGroup?.config} onPrefillChat={onPrefillChat ? askInChatAndClose(onPrefillChat, onOpenChange) : undefined} />
              {visibleGroups.map((g) => {
                const offset = flat.findIndex((f) => f.group.id === g.id);
                return (
                  <section key={g.id}>
                    <div className="ai-overview-charts-heading"><h3 title={g.name}>{visibleGroups.length > 1 ? g.name : "Charts"}</h3><span className="ai-overview-chart-count">{g.slides.length}</span><Button color="link-gray" className="ai-open-stack" onPress={() => { setIndex(Math.max(0, offset)); setLayout("stack"); }}>Open in Stack view →</Button></div>
                    {g.slides.length === 0 ? (
                      <div className="rounded-2xl border border-dashed border-[var(--color-ai-line)] bg-[var(--color-ai-surface)] px-6 py-6 text-center text-sm text-[var(--color-ai-body)]">
                        {g.pending ? "Generating insights…" : "No auto-insights available for this dataset"}
                      </div>
                    ) : (
                      <div className="ai-overview-chart-grid">
                        {g.slides.map((slide, si) => {
                          const globalIdx = offset + si;
                          const cid = chartIdOf(slide, g.id, si);
                          const lit = avatar.isHighlighted(cid);
                          return (
                            <div
                              {...avatar.bindChart(cid)}
                              key={`grid-${g.id}-${si}-${slugify(slide.title || String(si))}`}
                              role="button"
                              tabIndex={0}
                              onKeyDown={(event) => { if (event.target === event.currentTarget && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); setIndex(globalIdx); setLayout("stack"); } }}
                              onClick={() => { setIndex(globalIdx); setLayout("stack"); }}
                              className={
                                "group flex flex-col overflow-hidden rounded-2xl border bg-[var(--color-ai-surface)] p-4 text-left shadow-[0_12px_28px_-12px_rgba(16,24,40,0.18)] transition hover:-translate-y-1 hover:shadow-[0_24px_48px_-12px_rgba(16,24,40,0.28)] " +
                                (lit ? HIGHLIGHT_CLASS : globalIdx === index ? "border-[#1565ef] ring-2 ring-[#1565ef]/25" : "border-[var(--color-ai-line)]")
                              }
                            >
                              {slide.type !== "scatter" && <div className="ai-overview-chart-header"><Table className="size-4 shrink-0"/><span><CommentaryText text={slide.title} segments={slide.titleSegments} truncate /></span><ExplainChartButton chartId={cid}/></div>}
                              <div className="ai-overview-chart-body">
                                <DynamicChart slide={slide} height={slide.type === "scatter" ? 320 : 290} chartKey={`${g.id}-${si}`} header={slide.type === "scatter" ? <div className="ai-overview-chart-header"><span><CommentaryText text={slide.title} segments={slide.titleSegments} truncate /></span><ExplainChartButton chartId={cid}/></div> : undefined}/>
                              </div>
                              <OverviewInsight slide={slide}/>

                            </div>
                          );
                        })}
                      </div>
                    )}
                  </section>
                );
              })}
            </div>
          )}
        </div>
      ) : layout === "columns" ? (
        <ColumnStatisticsTab key={activeGroup?.id} config={activeGroup?.config} slides={activeGroup?.slides ?? []} onChart={(position) => { setIndex(position); setLayout("stack"); }} />
      ) : (
        <div className={`ai-stack-grid${assistantOpen ? " xl:mr-[380px]" : ""}`}>
          <InsightNavigator slides={activeGroup?.slides ?? []} index={index} config={activeGroup?.config} onSelect={setIndex} />
          {isPending || !activeEntry || !activeSlide ? <div className="ai-surface ai-empty-card">{isPending ? "Generating insights…" : "No auto-insights available for this dataset"}</div> : <>
            <InsightCard slide={activeSlide} config={activeEntry.group.config} outputRows={activeEntry.group.outputRows} samples={activeEntry.group.samples} defaultView={defaultView} chartId={chartIdOf(activeSlide, activeEntry.group.id, activeEntry.si)} index={index} total={total} onSelect={setIndex} />
            <ChartStatsRail config={activeEntry.group.config} slide={activeSlide} onAllColumns={() => setLayout("columns")} />
          </>}
        </div>
      )}

      <InsightsAvatar
        aid={aid ?? null}
        datasetId={avatarDatasetId}
        datasetName={avatarDatasetName}
        charts={avatarCharts}
        open={assistantOpen}
        onOpenChange={handleAssistantOpenChange}
      />

      <CodeViewerModal
        open={codeOpen}
        onClose={() => setCodeOpen(false)}
        code={codeText}
        loading={codeLoading}
      />
    </div>
  );
}

export function AutoInsightsModal(props: AutoInsightsModalProps) {
  // Every way into Auto Insights opens this modal, so this is its plan gate. It
  // reads the edition matrix's auto_insights capability, as the Billing tab's
  // comparison does. While the plan is unknown it stays open, as before.
  const { data } = useActivePlan();
  const plan = normalizePlan(data?.plan);
  const blocked = plan !== null && !planIncludes(plan, "auto_insights");
  // Keyed by aid so the assistant's transcript resets when the analysis changes;
  // unmounting on close stops speech, recognition, requests and highlights.
  if (!props.open || blocked) return null;
  return (
    <InsightsAvatarProvider key={props.aid ?? "no-aid"}>
      <AutoInsightsModalInner {...props} />
    </InsightsAvatarProvider>
  );
}