import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, ChevronDown, X, Stars02, Table } from "@untitledui/icons";
import { Code02, Copy01, ThumbsDown, ThumbsUp } from "@untitledui/icons";
import { toast } from "sonner";
import { DynamicChart, normalizeVizConfig, type Slide } from "./dynamicChart";
import { AnalysisOutputTable } from "./AnalysisOutputTable";
import { backendApi } from "@/lib/api/backendApi";
import { buildInsightGroups } from "@/lib/insight-groups";
import { InsightsAvatar, ExplainChartButton } from "@/components/insights-avatar/InsightsAvatar";
import { InsightsAvatarProvider, useInsightsAvatar, HIGHLIGHT_CLASS } from "@/components/insights-avatar/InsightsAvatarContext";
import { Button } from "@/components/base/buttons/button";
import { Badge } from "@/components/base/badges/badges";
import { Toggle } from "@/components/base/toggle/toggle";

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
  onPrefillChat?: (text: string) => void;
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
  predictive_value?: "high" | "medium" | "low";
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

function DatasetOverview({ config, onPrefillChat }: { config: any; onPrefillChat?: (text: string) => void }) {
  const profile = config?.profile_overview;
  if (!profile || typeof profile !== "object") return null;
  const score = finiteNumber(profile.readiness_score);
  const warnings = (Array.isArray(profile.red_flags) && profile.red_flags.length ? profile.red_flags : profile.concerns) ?? [];
  const suggestions = [...(Array.isArray(profile.analysis_suggestions) ? profile.analysis_suggestions : []), ...(Array.isArray(profile.quick_wins) ? profile.quick_wins : [])].filter((value) => typeof value === "string" && value.trim()).slice(0, 5);
  const scoreTone = score == null ? "bg-tertiary" : score >= 80 ? "bg-utility-success-500" : score >= 50 ? "bg-utility-warning-500" : "bg-utility-error-500";

  return (
    <section className="rounded-2xl border border-[#eaecf0] bg-white p-5 shadow-[0_12px_28px_-12px_rgba(16,24,40,0.18)]">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-[15px] font-semibold text-[#101828]">Dataset overview</h3>
          {profile.domain && <Badge className="mt-2" size="sm" color="blue">{profile.domain}</Badge>}
        </div>
        {score != null && (
          <div className="w-full max-w-[240px]">
            <p className="text-xs font-semibold text-[#344054]">Data readiness {Math.max(0, Math.min(100, score))}/100</p>
            <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-[#eaecf0]">
              <div className={`h-full rounded-full ${scoreTone}`} style={{ width: `${Math.max(0, Math.min(100, score))}%` }} />
            </div>
            {profile.readiness_reasoning && <p className="mt-1 line-clamp-1 text-xs text-[#667085]">{profile.readiness_reasoning}</p>}
          </div>
        )}
      </div>
      {profile.domain_reasoning && <p className="mt-3 text-sm leading-6 text-[#475467]">{profile.domain_reasoning}</p>}
      {(warnings.length > 0 || suggestions.length > 0) && (
        <div className="mt-4 grid gap-5 md:grid-cols-2">
          {warnings.length > 0 && <div><h4 className="text-sm font-semibold text-[#101828]">Watch out for</h4><ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-[#475467]">{warnings.map((item: string, index: number) => <li key={`${item}-${index}`}>{item}</li>)}</ul></div>}
          {suggestions.length > 0 && <div><h4 className="text-sm font-semibold text-[#101828]">Suggested next steps</h4><div className="mt-2 flex flex-col items-start gap-1">{suggestions.map((item: string, index: number) => <Button key={`${item}-${index}`} color="link-gray" size="sm" onPress={() => onPrefillChat?.(item)} className="h-auto justify-start whitespace-normal text-left">{item}</Button>)}</div></div>}
        </div>
      )}
    </section>
  );
}

function ColumnStatistics({ config }: { config: any }) {
  const [showIds, setShowIds] = useState(false);
  const source = Array.isArray(config?.column_stats) ? config.column_stats as ColumnStat[] : [];
  if (!source.length) return null;
  const highValue = new Set(Array.isArray(config?.profile_overview?.high_value_columns) ? config.profile_overview.high_value_columns : []);
  const ordered = source.map((column, index) => ({ column, index })).sort((a, b) => Number(highValue.has(b.column.name)) - Number(highValue.has(a.column.name)) || a.index - b.index).map(({ column }) => column);
  const hasIds = ordered.some((column) => String(column.role ?? "").toLowerCase() === "identifier");
  const visible = showIds ? ordered : ordered.filter((column) => String(column.role ?? "").toLowerCase() !== "identifier");
  const badgeColor = (value: ColumnStat["predictive_value"]) => value === "high" ? "success" : value === "medium" ? "gray" : "slate";

  return (
    <details className="group mt-3 rounded-lg border border-[#eaecf0] bg-white">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-3.5 py-3 text-[13px] font-semibold text-[#101828]">
        Column statistics <ChevronDown className="size-4 text-[#667085] transition-transform group-open:rotate-180" />
      </summary>
      <div className="border-t border-[#eaecf0] p-3.5">
        {hasIds && <div className="mb-3"><Toggle slim size="sm" label="Show ID columns" isSelected={showIds} onChange={setShowIds} /></div>}
        <div className="overflow-x-auto rounded-lg border border-[#eaecf0]">
          <table className="w-full min-w-[900px] text-left text-xs">
            <thead className="bg-[#f9fafb] text-[#667085]"><tr>{["Column", "Type", "Meaning", "Predictive value", "Missing %", "Distinct", "Summary"].map((heading) => <th key={heading} className="whitespace-nowrap px-3 py-2 font-medium">{heading}</th>)}</tr></thead>
            <tbody>{visible.map((column, index) => <tr key={`${column.name}-${index}`} className="border-t border-[#eaecf0] text-[#475467]"><td className="px-3 py-2 font-medium text-[#101828]">{column.name ?? "—"}</td><td className="px-3 py-2">{column.dtype ?? "—"}</td><td className="max-w-[220px] px-3 py-2">{column.meaning ?? ""}</td><td className="px-3 py-2">{column.predictive_value ? <Badge size="sm" color={badgeColor(column.predictive_value)}>{column.predictive_value}</Badge> : null}</td><td className="px-3 py-2">{formatStatNumber(column.missing_pct)}</td><td className="px-3 py-2">{formatStatNumber(column.n_unique)}</td><td className="max-w-[320px] px-3 py-2">{columnSummary(column)}</td></tr>)}</tbody>
          </table>
        </div>
      </div>
    </details>
  );
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

function InsightCard({
  slide,
  datasetName,
  aid,
  insightId,
  outputRows,
  samples,
  defaultView = "chart",
  onOpenCode,
  chartId,
  config,
}: {
  slide: Slide;
  datasetName?: string;
  aid?: string | null;
  insightId: string;
  outputRows?: Record<string, unknown>[] | null;
  samples?: any[];
  defaultView?: "table" | "chart";
  onOpenCode: () => void;
  chartId: string;
  config?: any;
}) {
  const avatar = useInsightsAvatar();
  const [view, setView] = useState<"table" | "chart">(defaultView);
  const [feedback, setFeedback] = useState<"positive" | "negative" | null>(null);
  const [submitting, setSubmitting] = useState<"positive" | "negative" | null>(null);
  const [note, setNote] = useState("");
  const [savingNote, setSavingNote] = useState(false);
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const disabled = !aid;

  // Prefer per-insight output rows, then the uploaded dataset samples. Some
  // persisted analyses only retain the chart's derived rows, so use those as
  // the final fallback rather than leaving the Data table control disabled.
  const tableRows = useMemo<Record<string, unknown>[] | null>(() => {
    if (Array.isArray(outputRows) && outputRows.length > 0) return outputRows;
    const sourceRows = Array.isArray(samples) && samples.length > 0 ? samples : slide.data;
    if (Array.isArray(sourceRows) && sourceRows.length > 0) {
      const first = sourceRows[0];
      if (first && typeof first === "object" && !Array.isArray(first)) {
        return sourceRows as Record<string, unknown>[];
      }
      if (Array.isArray(first)) {
        const cols = (first as unknown[]).map((_, i) => `col_${i + 1}`);
        return (sourceRows as unknown[][]).map((row) => {
          const obj: Record<string, unknown> = {};
          cols.forEach((c, i) => { obj[c] = row[i]; });
          return obj;
        });
      }
    }
    return null;
  }, [outputRows, samples, slide.data]);
  const hasOutputRows = !!tableRows && tableRows.length > 0;

  useEffect(() => {
    setView(defaultView);
  }, [defaultView, insightId]);

  const sendThumb = async (type: "positive" | "negative") => {
    if (!aid || submitting) return;
    setSubmitting(type);
    try {
      await backendApi.sendInsightFeedback(aid, {
        insight_id: insightId,
        feedback_type: type,
        comment: null,
      });
      setFeedback(type);
    } catch (err: any) {
      toast.error(err?.message || "Failed to submit feedback");
    } finally {
      setSubmitting(null);
    }
  };

  const saveNote = async () => {
    if (!aid || savingNote || !note.trim()) return;
    setSavingNote(true);
    try {
      await backendApi.sendInsightFeedback(aid, {
        insight_id: insightId,
        feedback_type: "positive",
        comment: note.trim(),
      });
      setSavedAt(Date.now());
      setTimeout(() => setSavedAt((t) => (t && Date.now() - t >= 1900 ? null : t)), 2000);
    } catch (err: any) {
      toast.error(err?.message || "Failed to save note");
    } finally {
      setSavingNote(false);
    }
  };

  return (
    <div
      {...avatar.bindChart(chartId)}
      className={"flex h-full flex-col overflow-y-auto rounded-2xl border border-[#eaecf0] bg-white p-5 shadow-[0_24px_48px_-12px_rgba(16,24,40,0.18)] transition-shadow " + (avatar.isHighlighted(chartId) ? HIGHLIGHT_CLASS : "")}
    >
      <div className="flex items-center gap-2">
        <Table className="size-4 text-[#344054]" />
        <h3 className="text-[15px] font-semibold text-[#101828]">{slide.title}</h3>
        <ExplainChartButton chartId={chartId} />
      </div>
      {(slide.subtitle || datasetName) && (
        <p className="mt-0.5 text-xs text-[#667085]">
          {slide.subtitle ? stripMarkdown(slide.subtitle) : datasetName}
        </p>
      )}

      <div className="mt-3">
        <div className="flex items-center justify-end">
          <div className="inline-flex rounded-lg border border-[#eaecf0] bg-white p-0.5 text-xs">
            <button
              type="button"
              onClick={() => setView("table")}
              disabled={!hasOutputRows}
              className={
                "rounded-md px-2.5 py-1 disabled:opacity-50 " +
                (view === "table"
                  ? "bg-white font-medium text-[#101828] shadow-sm ring-1 ring-[#eaecf0]"
                  : "text-[#475467]")
              }
            >
              Data table
            </button>
            <button
              type="button"
              onClick={() => setView("chart")}
              className={
                "rounded-md px-2.5 py-1 " +
                (view === "chart"
                  ? "bg-white font-medium text-[#101828] shadow-sm ring-1 ring-[#eaecf0]"
                  : "text-[#475467]")
              }
            >
              Chart
            </button>
          </div>
        </div>
        <p className="mt-1 text-right text-[11px] text-[#667085]">Auto-generated</p>
        <div className="mt-1">
          {view === "table" && hasOutputRows ? (
            <AnalysisOutputTable
              rows={tableRows!}
              title="Output"
              filename={`${slugify(slide.title || "output")}.csv`}
            />
          ) : (
            <DynamicChart slide={slide} />
          )}
        </div>
      </div>

      {(slide.insights.length > 0 || slide.headline || slide.insightFacts || config?.dataset_summary?.insights?.length > 0) && (
        <div className="mt-3 rounded-lg border border-[#eaecf0] bg-white p-3.5">
          <p className="text-[13px] font-semibold text-[#101828]">Key Insights</p>
          <ul className="mt-1.5 list-disc space-y-0.5 pl-5 text-[13px] text-[#475467]">
            {slide.headline ? <li><strong className="block text-[#101828]">{slide.headline}</strong>{slide.insightFacts && <span className="mt-0.5 block">{slide.insightFacts}</span>}</li> : slide.insights.map((line, i) => <li key={i}>{line}</li>)}
          </ul>
          {Array.isArray(config?.dataset_summary?.insights) && config.dataset_summary.insights.length > 0 && <div className="mt-3 border-t border-[#eaecf0] pt-3"><p className="text-[13px] font-semibold text-[#101828]">About this dataset</p><ul className="mt-1.5 list-disc space-y-0.5 pl-5 text-[13px] text-[#475467]">{config.dataset_summary.insights.map((line: string, index: number) => <li key={`${line}-${index}`}>{line}</li>)}</ul></div>}
        </div>
      )}

      <ColumnStatistics config={config} />

    </div>
  );
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
      <div className="flex max-h-[85vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-[#eaecf0] bg-white shadow-2xl">
        <div className="flex items-center justify-between border-b border-[#eaecf0] px-5 py-3">
          <h3 className="text-[15px] font-semibold text-[#101828]">Analysis code</h3>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={copy}
              disabled={loading || !code}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[#eaecf0] bg-white px-2.5 py-1.5 text-[12px] font-semibold text-[#344054] hover:bg-[#f9fafb] disabled:opacity-50"
            >
              <Copy01 className="size-3.5" />
              Copy
            </button>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="grid size-8 place-items-center rounded-lg border border-[#eaecf0] bg-white text-[#475467] hover:bg-[#f9fafb]"
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
  const [layout, setLayout] = useState<"stack" | "grid">("stack");
  const [codeOpen, setCodeOpen] = useState(false);
  const [codeText, setCodeText] = useState("");
  const [codeLoading, setCodeLoading] = useState(false);
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null);
  const avatar = useInsightsAvatar();

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
  const isPending = groups.some((g) => g.pending) && totalAll === 0;
  const noInsights = !isPending && total === 0;

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
    if (!open || total === 0) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onOpenChange(false);
      else if (e.key === "ArrowRight") setIndex((i) => (i + 1) % total);
      else if (e.key === "ArrowLeft") setIndex((i) => (i - 1 + total) % total);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onOpenChange, total]);

  if (!open) return null;

  const avatarCharts = visibleGroups.flatMap((g) => g.slides.map((slide, si) => ({ id: chartIdOf(slide, g.id, si), slide })));
  const avatarDatasetId = activeGroup ? (activeGroup.id === "__single__" ? primaryDatasetId ?? null : activeGroup.id) : null;
  const avatarDatasetName = activeGroup?.name || datasetName || null;

  const prev = () => setIndex((i) => (i - 1 + total) % total);
  const next = () => setIndex((i) => (i + 1) % total);

  const activeEntry = total ? flat[Math.min(index, total - 1)] : null;
  const activeSlide = activeEntry?.slide;
  const back1 = total ? flat[(index + 1) % total].slide : null;
  const back2 = total ? flat[(index + 2) % total].slide : null;
  const headerName = multi ? "" : groups[0]?.name;


  return (
    <div
      className="fixed inset-0 z-[100] flex flex-col overflow-hidden animate-in fade-in duration-200"
      style={{
        background:
          "linear-gradient(180deg, #ffffff 0%, #ffffff 40%, #dfe8fa 100%)",
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="auto-insights-title"
    >
      <button
        type="button"
        onClick={() => { onOpenChange(false); setIndex(0); }}
        aria-label="Close"
        className="absolute right-8 top-8 z-30 grid size-9 place-items-center rounded-lg border border-[#eaecf0] bg-white text-[#475467] shadow-xs transition hover:bg-[#f9fafb]"
      >
        <X className="size-4" />
      </button>

      <div className="pt-12 pb-2 text-center">
        <h2 id="auto-insights-title" className="text-3xl font-bold text-[#101828]">{title ?? "Auto Insights"}</h2>
        <p className="mt-1.5 text-sm text-[#475467]">
          {subtitle ?? "Showing selected insights from the dataset"}
        </p>
      </div>

      <div className="flex justify-center pb-2">
        <div className="inline-flex rounded-lg border border-[#eaecf0] bg-white p-0.5 text-xs shadow-xs">
          <button
            type="button"
            onClick={() => setLayout("grid")}
            className={
              "rounded-md px-3 py-1.5 " +
              (layout === "grid" ? "bg-[#1565ef] font-semibold text-white" : "text-[#475467]")
            }
          >
            Overview
          </button>
          <button
            type="button"
            onClick={() => setLayout("stack")}
            className={
              "rounded-md px-3 py-1.5 " +
              (layout === "stack" ? "bg-[#1565ef] font-semibold text-white" : "text-[#475467]")
            }
          >
            Stack
          </button>
        </div>
      </div>

      {multi && (
        <div className="flex justify-center px-8 pb-2">
          <div className="flex max-w-full gap-1 overflow-x-auto rounded-xl border border-[#eaecf0] bg-white p-1 shadow-xs" role="tablist" aria-label="Datasets">
            {groups.map((g) => {
              const active = activeDatasetId === g.id;
              return (
                <button
                  key={g.id}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => { setActiveDatasetId(g.id); setIndex(0); }}
                  title={g.name}
                  className={
                    "max-w-[220px] truncate rounded-lg px-3 py-1.5 text-xs transition " +
                    (active
                      ? "bg-[#eff8ff] font-semibold text-[#175cd3] ring-1 ring-[#1565ef]/30"
                      : "text-[#475467] hover:bg-[#f9fafb]")
                  }
                >
                  {g.name} · {g.slides.length}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {layout === "grid" ? (
        <div className="flex-1 overflow-y-auto px-8 pb-8">
          {isPending ? (
            <div className="grid h-full place-items-center text-sm text-[#475467]">Generating insights…</div>
          ) : (
            <div className="mx-auto max-w-[1200px] space-y-8">
              <DatasetOverview config={activeGroup?.config} onPrefillChat={onPrefillChat} />
              {visibleGroups.map((g) => {
                const offset = flat.findIndex((f) => f.group.id === g.id);
                return (
                  <section key={g.id}>
                    <div className="mb-3 flex items-center gap-3">
                      <h3 className="truncate text-[15px] font-semibold text-[#101828]" title={g.name}>
                        {g.name}
                      </h3>
                      <span className="rounded-full bg-[#eff8ff] px-2 py-0.5 text-[11px] font-semibold text-[#175cd3]">
                        {g.slides.length} chart{g.slides.length === 1 ? "" : "s"}
                      </span>
                      <span className="h-px flex-1 bg-[#eaecf0]" />
                    </div>
                    {g.slides.length === 0 ? (
                      <div className="rounded-2xl border border-dashed border-[#eaecf0] bg-white px-6 py-6 text-center text-sm text-[#475467]">
                        {g.pending ? "Generating insights…" : "No auto-insights available for this dataset"}
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 items-start gap-5 sm:grid-cols-2 lg:grid-cols-3">
                        {g.slides.map((slide, si) => {
                          const globalIdx = offset + si;
                          const cid = chartIdOf(slide, g.id, si);
                          const lit = avatar.isHighlighted(cid);
                          return (
                            <button
                              {...avatar.bindChart(cid)}
                              key={`grid-${g.id}-${si}-${slugify(slide.title || String(si))}`}
                              type="button"
                              onClick={() => { setIndex(globalIdx); setLayout("stack"); }}
                              className={
                                "group flex flex-col overflow-hidden rounded-2xl border bg-white p-4 text-left shadow-[0_12px_28px_-12px_rgba(16,24,40,0.18)] transition hover:-translate-y-1 hover:shadow-[0_24px_48px_-12px_rgba(16,24,40,0.28)] " +
                                (lit ? HIGHLIGHT_CLASS : globalIdx === index ? "border-[#1565ef] ring-2 ring-[#1565ef]/25" : "border-[#eaecf0]")
                              }
                            >
                              <div className="flex items-center gap-2">
                                <Table className="size-4 text-[#344054]" />
                                <span className="truncate text-[13px] font-semibold text-[#101828]">{slide.title}</span>
                                <ExplainChartButton chartId={cid} />
                              </div>
                              <div className={slide.type === "pie" || slide.type === "donut" ? "pointer-events-none mt-2 h-auto min-h-[180px] overflow-visible" : "pointer-events-none mt-2 h-[180px] overflow-hidden"}>
                                <DynamicChart slide={slide} chartKey={`${g.id}-${si}`} />
                              </div>
                              <p className="mt-2 flex gap-1.5 text-[12px] leading-5 text-[#667085]">
                                <span className="mt-[7px] size-1 shrink-0 rounded-full bg-[#1565ef]" />
                                <span className="line-clamp-2">{chartPoint(slide)}</span>
                              </p>
                            </button>
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
      ) : noInsights ? (
        <div className="flex flex-1 items-center justify-center px-8 pb-8">
          <div className="rounded-2xl border border-[#eaecf0] bg-white px-6 py-8 text-center text-sm text-[#475467] shadow-[0_12px_28px_-12px_rgba(16,24,40,0.18)]">
            No auto-insights available for this dataset
          </div>
        </div>
      ) : (
      <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-6">
        <div className="mx-auto mb-5 max-w-[720px]"><DatasetOverview config={activeGroup?.config} onPrefillChat={onPrefillChat} /></div>
      <div className="relative flex min-h-[560px] items-center justify-center">
        {total > 1 && (
          <button
            type="button"
            onClick={prev}
            aria-label="Previous"
            className="absolute left-10 top-1/2 z-30 grid size-10 -translate-y-1/2 place-items-center rounded-lg border border-[#eaecf0] bg-white text-[#98a2b3] shadow-xs transition hover:text-[#475467]"
          >
            <ArrowLeft className="size-5" />
          </button>
        )}

        <div className="relative h-[560px] w-full max-w-[720px]">
          {total > 1 && back1 && back2 && (
            <>
              <div
                className="absolute inset-0 origin-center transition-all duration-500 ease-out"
                style={{
                  transform: "translateX(-64px) translateY(10px) scale(0.94) rotate(-5deg)",
                  zIndex: 1,
                }}
                aria-hidden
              >
                <div className="h-full rounded-2xl border border-[#eaecf0] bg-white shadow-[0_16px_32px_-12px_rgba(16,24,40,0.12)]">
                  <div className="flex items-center gap-2 p-5">
                    <Table className="size-4 text-[#d0d5dd]" />
                    <span className="text-[15px] font-semibold text-[#d0d5dd]">{back2.title}</span>
                  </div>
                </div>
              </div>

              <div
                className="absolute inset-0 origin-center transition-all duration-500 ease-out"
                style={{
                  transform: "translateX(-34px) translateY(5px) scale(0.97) rotate(-2.5deg)",
                  zIndex: 2,
                }}
                aria-hidden
              >
                <div className="h-full rounded-2xl border border-[#eaecf0] bg-white shadow-[0_18px_36px_-12px_rgba(16,24,40,0.14)]">
                  <div className="flex items-center gap-2 p-5">
                    <Table className="size-4 text-[#98a2b3]" />
                    <span className="text-[15px] font-semibold text-[#98a2b3]">{back1.title}</span>
                  </div>
                </div>
              </div>
            </>
          )}

          <div
            key={`active-${index}`}
            className="absolute inset-0 transition-all duration-500 ease-out animate-in fade-in slide-in-from-right-6"
            style={{ zIndex: 10 }}
          >
            {isPending || !activeEntry || !activeSlide ? (
              <div className="flex h-full items-center justify-center rounded-2xl border border-[#eaecf0] bg-white text-sm text-[#475467] shadow-[0_24px_48px_-12px_rgba(16,24,40,0.18)]">
                Generating insights…
              </div>
            ) : (
              <InsightCard
                slide={activeSlide}
                datasetName={activeEntry.group.name}
                aid={aid}
                insightId={`${activeEntry.group.id}-${activeEntry.si}-${slugify(activeSlide.title || `insight-${index}`)}`}
                outputRows={activeEntry.group.outputRows}
                samples={activeEntry.group.samples}
                defaultView={defaultView}
                onOpenCode={handleOpenCode}
                chartId={chartIdOf(activeSlide, activeEntry.group.id, activeEntry.si)}
                config={activeEntry.group.config}
              />

            )}
          </div>
        </div>

        {total > 1 && (
          <button
            type="button"
            onClick={next}
            aria-label="Next"
            className="absolute right-10 top-1/2 z-30 grid size-10 -translate-y-1/2 place-items-center rounded-lg bg-[#1565ef] text-white shadow-md transition hover:bg-[#1257d6]"
          >
            <ArrowRight className="size-5" />
          </button>
        )}
      </div>
      </div>
      )}

      {layout === "stack" && !noInsights && (
        <div className="pb-8 text-center text-sm font-medium text-[#475467]">
          {index + 1} / {total}
        </div>
      )}


      <InsightsAvatar
        aid={aid ?? null}
        datasetId={avatarDatasetId}
        datasetName={avatarDatasetName}
        charts={avatarCharts}
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
  // Keyed by aid so the assistant's transcript resets when the analysis changes;
  // unmounting on close stops speech, recognition, requests and highlights.
  if (!props.open) return null;
  return (
    <InsightsAvatarProvider key={props.aid ?? "no-aid"}>
      <AutoInsightsModalInner {...props} />
    </InsightsAvatarProvider>
  );
}

