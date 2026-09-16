import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, X, Stars02, Table } from "@untitledui/icons";
import { Code02, Copy01, ThumbsDown, ThumbsUp } from "@untitledui/icons";
import { toast } from "sonner";
import { DynamicChart, normalizeVizConfig, type Slide } from "./dynamicChart";
import { AnalysisOutputTable } from "./AnalysisOutputTable";
import { backendApi } from "@/lib/api/backendApi";
import { buildInsightGroups } from "@/lib/insight-groups";


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
}: {
  slide: Slide;
  datasetName?: string;
  aid?: string | null;
  insightId: string;
  outputRows?: Record<string, unknown>[] | null;
  samples?: any[];
  defaultView?: "table" | "chart";
  onOpenCode: () => void;
}) {
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
    <div className="flex h-full flex-col overflow-y-auto rounded-2xl border border-[#eaecf0] bg-white p-5 shadow-[0_24px_48px_-12px_rgba(16,24,40,0.18)]">
      <div className="flex items-center gap-2">
        <Table className="size-4 text-[#344054]" />
        <h3 className="text-[15px] font-semibold text-[#101828]">{slide.title}</h3>
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

      {slide.insights.length > 0 && (
        <div className="mt-3 rounded-lg border border-[#eaecf0] bg-white p-3.5">
          <p className="text-[13px] font-semibold text-[#101828]">Key Insights</p>
          <ul className="mt-1.5 list-disc space-y-0.5 pl-5 text-[13px] text-[#475467]">
            {slide.insights.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        </div>
      )}

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


export function AutoInsightsModal({ open, onOpenChange, config, status, datasetName, samples, outputRows, defaultView = "chart", title, subtitle, aid, datasets, primaryDatasetId }: AutoInsightsModalProps) {
  const [index, setIndex] = useState(0);
  const [layout, setLayout] = useState<"stack" | "grid">("stack");
  const [codeOpen, setCodeOpen] = useState(false);
  const [codeText, setCodeText] = useState("");
  const [codeLoading, setCodeLoading] = useState(false);
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null);

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
                      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
                        {g.slides.map((slide, si) => {
                          const globalIdx = offset + si;
                          return (
                            <button
                              key={`grid-${g.id}-${si}-${slugify(slide.title || String(si))}`}
                              type="button"
                              onClick={() => { setIndex(globalIdx); setLayout("stack"); }}
                              className={
                                "group flex flex-col overflow-hidden rounded-2xl border bg-white p-4 text-left shadow-[0_12px_28px_-12px_rgba(16,24,40,0.18)] transition hover:-translate-y-1 hover:shadow-[0_24px_48px_-12px_rgba(16,24,40,0.28)] " +
                                (globalIdx === index ? "border-[#1565ef] ring-2 ring-[#1565ef]/25" : "border-[#eaecf0]")
                              }
                            >
                              <div className="flex items-center gap-2">
                                <Table className="size-4 text-[#344054]" />
                                <span className="truncate text-[13px] font-semibold text-[#101828]">{slide.title}</span>
                              </div>
                              <div className="pointer-events-none mt-2 h-[180px] overflow-hidden">
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
      <div className="relative flex flex-1 items-center justify-center px-6">
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
      )}

      {layout === "stack" && !noInsights && (
        <div className="pb-8 text-center text-sm font-medium text-[#475467]">
          {index + 1} / {total}
        </div>
      )}


      <CodeViewerModal
        open={codeOpen}
        onClose={() => setCodeOpen(false)}
        code={codeText}
        loading={codeLoading}
      />
    </div>
  );
}

