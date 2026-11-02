// Chart carousel with the "Maximize chart" dialog (UI-B29) and the inline
// markdown renderer it uses, moved verbatim out of routes/analysis.tsx (which
// re-imports both) so ui/tests/release_1_6/maximize-chart.test.tsx can render it.
import { useCallback, useEffect, useState } from "react";
import { Link } from "@tanstack/react-router";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChevronLeft, ChevronRight, X } from "@untitledui/icons";
import { Maximize2 } from "lucide-react";
import { TagPicker } from "@/components/analysis/TagPicker";
import { DynamicChart, type Slide } from "@/components/dashboard/dynamicChart";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { rayJobMarkdown } from "@/lib/ray-jobs";
import { cx } from "@/lib/utils/cx";

export function InlineMarkdown({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <>{children}</>,
        a: ({ children, href }) =>
          href?.startsWith("/ray-jobs?job=") ? (
            <Link
              to="/ray-jobs"
              search={{ job: href.slice("/ray-jobs?job=".length), status: "", kind: "" }}
              className="text-brand-secondary underline"
            >
              {children}
            </Link>
          ) : (
            <a href={href} className="text-[#1565ef] underline" target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
        code: ({ children }) => <code className="rounded bg-secondary/60 px-1 py-0.5 text-[12px]">{children}</code>,
        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
        em: ({ children }) => <em className="italic">{children}</em>,
      }}
    >
      {rayJobMarkdown(content)}
    </ReactMarkdown>
  );
}

export function ChartCarousel({
  items,
  onDismiss,
  activeRunId,
  onRunChange,
}: {
  items: { slide: Slide; index: number; cardKey: string; runId?: string }[];
  onDismiss?: (cardKey: string) => void;
  activeRunId?: string | null;
  onRunChange?: (runId: string) => void;
}) {
  const [active, setActive] = useState(0);
  const [fullscreenOpen, setFullscreenOpen] = useState(false);
  const total = items.length;
  const safeActive = Math.min(active, total - 1);
  const runIdsKey = items.map((item) => item.runId ?? "").join("\u001f");
  const selectIndex = useCallback(
    (i: number) => {
      setActive((current) => (current === i ? current : i));
      const rid = items[i]?.runId;
      if (rid && rid !== activeRunId) onRunChange?.(rid);
    },
    [items, activeRunId, onRunChange],
  );
  const go = (dir: -1 | 1) => selectIndex((safeActive + dir + total) % total);
  const activeItem = items[safeActive]!;
  const s = activeItem.slide;
  const cardKey = activeItem.cardKey;

  // Keep the carousel on the run selected elsewhere (e.g. the Data table pager),
  // without fighting the user when they page within the same run's charts.
  useEffect(() => {
    if (!activeRunId) return;
    if (items[safeActive]?.runId === activeRunId) return;
    const i = items.findIndex((it) => it.runId === activeRunId);
    if (i >= 0) setActive((current) => (current === i ? current : i));
  }, [activeRunId, runIdsKey, safeActive]);

  // Offset from center: -1 = prev, 0 = active, 1 = next. Only render 3 neighbours.
  const getOffset = (i: number) => {
    if (total <= 1) return 0;
    let d = i - safeActive;
    if (d > total / 2) d -= total;
    if (d < -total / 2) d += total;
    return d;
  };

  const CHART_H = 320;
  const STAGE_H = CHART_H + 140; // header + chart + padding

  return (
    <div className="space-y-3 rounded-2xl bg-secondary/50 p-4">
      <div className="relative">
        {total > 1 ? (
          <button
            type="button"
            aria-label="Previous chart"
            onClick={() => go(-1)}
            className="absolute left-0 top-1/2 z-30 -translate-y-1/2 rounded-full bg-brand-600 p-2.5 text-white shadow-md hover:bg-brand-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
          >
            <ChevronLeft className="size-5" strokeWidth={2.5} />
          </button>
        ) : null}
        {total > 1 ? (
          <button
            type="button"
            aria-label="Next chart"
            onClick={() => go(1)}
            className="absolute right-0 top-1/2 z-30 -translate-y-1/2 rounded-full bg-brand-600 p-2.5 text-white shadow-md hover:bg-brand-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
          >
            <ChevronRight className="size-5" strokeWidth={2.5} />
          </button>
        ) : null}

        {/* Arrows (40px) sit in 52px gutters: 12px clear of the active card on each side. */}
        <div
          className="relative mx-auto overflow-hidden"
          style={{ height: STAGE_H, perspective: "1600px", width: total > 1 ? "calc(100% - 104px)" : "100%" }}
        >
          {items.map((it, i) => {
            const off = getOffset(i);
            if (Math.abs(off) > 1) return null; // only render 3 nearest
            const isActive = off === 0;
            const translateX = off * 52; // percent
            const translateY = isActive ? 0 : -10;
            const scale = isActive ? 1 : 0.86;
            const rotY = off * -10; // subtle 3D tilt toward center
            const opacity = isActive ? 1 : 0.4;
            const z = isActive ? 20 : 10;
            const filter = isActive ? "none" : "grayscale(1) blur(0.3px)";

            return (
              <div
                key={it.cardKey}
                aria-hidden={!isActive}
                onClick={() => {
                  if (!isActive) selectIndex(i);
                }}
                className="absolute left-1/2 top-0 w-[88%] will-change-transform"
                style={{
                  transform: isActive
                    ? `translate3d(-50%, 0, 0)`
                    : `translate3d(calc(-50% + ${translateX}%), ${translateY}px, 0) scale(${scale}) rotateY(${rotY}deg)`,
                  transformStyle: "preserve-3d",
                  opacity,
                  zIndex: z,
                  filter,
                  transition: "transform 550ms cubic-bezier(0.22, 1, 0.36, 1), opacity 400ms ease, filter 400ms ease",
                  pointerEvents: "auto",
                  cursor: isActive ? "default" : "pointer",
                }}
              >
                <div
                  className={cx(
                    "relative rounded-xl p-4",
                    isActive ? "border border-secondary bg-primary shadow-sm" : "pointer-events-none select-none border border-primary bg-muted/90",
                  )}
                >
                  {isActive && onDismiss ? (
                    <button
                      type="button"
                      aria-label="Close chart"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDismiss(cardKey);
                      }}
                      className="absolute right-3 top-3 z-10 rounded-md p-1 text-fg-quaternary hover:bg-primary_hover hover:text-fg-secondary"
                    >
                      <X className="size-4" />
                    </button>
                  ) : null}
                  <div className="flex flex-wrap items-center gap-2 pr-8">
                    <h3 className="text-sm font-semibold text-primary">{it.slide.title}</h3>
                    {(it.slide as Slide & { sourceLabel?: string }).sourceLabel ? (
                      <span className="rounded-full bg-secondary px-2 py-0.5 text-[11px] font-medium text-secondary">
                        {(it.slide as Slide & { sourceLabel?: string }).sourceLabel}
                      </span>
                    ) : null}
                    {isActive ? <TagPicker size="sm" /> : null}
                    {isActive ? (
                      <button
                        type="button"
                        aria-label={`Maximize ${it.slide.title} chart`}
                        title="Maximize chart"
                        onClick={(event) => {
                          event.stopPropagation();
                          setFullscreenOpen(true);
                        }}
                        className="ml-auto rounded-md border border-secondary bg-primary p-1.5 text-fg-secondary shadow-sm transition hover:bg-secondary hover:text-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
                      >
                        <Maximize2 className="size-4" aria-hidden="true" />
                      </button>
                    ) : null}
                    {isActive && total > 1 ? (
                      <span className="text-xs text-tertiary">
                        {safeActive + 1} / {total}
                      </span>
                    ) : null}
                  </div>
                  {isActive && it.slide.subtitle ? (
                    <p className="text-xs text-tertiary">
                      <InlineMarkdown content={it.slide.subtitle} />
                    </p>
                  ) : null}
                  <div className="mt-3 h-[320px] w-full min-w-0 shrink-0">
                    <DynamicChart slide={it.slide} height={CHART_H} chartKey={it.cardKey} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      {total > 1 ? (
        <div className="flex items-center justify-center gap-1.5">
          {items.map((it, i) => (
            <button
              key={it.cardKey}
              type="button"
              aria-label={`Go to chart ${i + 1}`}
              onClick={() => selectIndex(i)}
              className={
                i === safeActive
                  ? "h-2 w-6 rounded-full bg-brand-solid transition-all"
                  : "h-2 w-2 rounded-full bg-muted-foreground/40 hover:bg-muted-foreground/60 transition-all"
              }
            />
          ))}
        </div>
      ) : null}
      <Dialog open={fullscreenOpen} onOpenChange={setFullscreenOpen}>
        <DialogContent className="grid h-[90vh] w-[94vw] max-w-none grid-rows-[auto_auto_1fr] gap-3 overflow-hidden rounded-xl border-secondary bg-primary p-5 shadow-2xl">
          <DialogTitle className="pr-10 text-base font-semibold text-primary">{s.title}</DialogTitle>
          {s.subtitle ? (
            <p className="text-sm text-tertiary">
              <InlineMarkdown content={s.subtitle} />
            </p>
          ) : null}
          <div className="min-h-0 w-full overflow-auto">
            <DynamicChart
              slide={s}
              height={typeof window === "undefined" ? 590 : Math.max(360, window.innerHeight * 0.68)}
              chartKey={`${cardKey}-fullscreen`}
            />
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
