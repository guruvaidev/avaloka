import { useMemo, useState, type ReactNode } from "react";
import { ChevronDown, SearchLg, X } from "@untitledui/icons";
import { cx } from "@/lib/utils/cx";
import { setInsightDragData } from "@/lib/dashboard-drag";
import { DynamicChart, normalizeVizConfig, type Slide } from "./dynamicChart";

export type InsightSource = {
  analysisId: string;
  analysisName: string;
  vizConfig?: unknown;
  userVizConfigs?: unknown[];
  samples?: unknown[];
};

type Props = {
  sources?: InsightSource[];
  vizConfig?: unknown;
  userVizConfigs?: unknown[];
  samples?: unknown[];
  onClose?: () => void;
  onSelectSlide?: (slide: Slide, index: number, analysisId?: string) => void;
  className?: string;
};

type InsightSlideItem = {
  slide: Slide;
  chartIndex: number;
  vizConfig: unknown;
  source: "auto" | "user";
  analysisId: string;
  analysisName: string;
};

function formatTimeAgo(value: unknown): string {
  if (typeof value !== "string") return "Just now";
  const ts = new Date(value).getTime();
  if (Number.isNaN(ts)) return "Just now";
  const mins = Math.max(0, Math.floor((Date.now() - ts) / 60_000));
  if (mins < 1) return "Just now";
  if (mins < 60) return `${String(mins).padStart(2, "0")} min Ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} hr Ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} Ago`;
}

function insightDescription(slide: Slide): string {
  if (slide.subtitle?.trim()) return slide.subtitle.trim();
  if (slide.insights[0]?.trim()) return slide.insights[0].trim();
  return "Created a New visualisation cell.";
}

function buildSlideItems(
  vizConfig: unknown,
  source: "auto" | "user",
  analysisId: string,
  analysisName: string,
  samples?: unknown[],
): InsightSlideItem[] {
  if (!vizConfig) return [];
  const seen = new Set<string>();
  const slides = normalizeVizConfig(vizConfig, samples).filter((slide) => {
    const key = String(slide.title ?? "").trim().toLowerCase() || JSON.stringify({
      type: slide.type,
      xKey: slide.xKey,
      series: slide.series.map((s) => s.dataKey),
    });
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return slides.map((slide, chartIndex) => ({ slide, chartIndex, vizConfig, source, analysisId, analysisName }));
}

function normalizeSources(props: Props): InsightSource[] {
  if (props.sources?.length) return props.sources;
  if (!props.vizConfig && !props.userVizConfigs?.length) return [];
  return [
    {
      analysisId: "active",
      analysisName: "Analysis",
      vizConfig: props.vizConfig,
      userVizConfigs: props.userVizConfigs,
      samples: props.samples,
    },
  ];
}

export function AllInsightsPanel({
  sources: sourcesProp,
  vizConfig,
  userVizConfigs = [],
  samples,
  onClose,
  onSelectSlide,
  className,
}: Props) {
  const [query, setQuery] = useState("");
  const [layout, setLayout] = useState<"grid" | "list">("grid");
  const [activeKey, setActiveKey] = useState("auto-0");
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({});

  const sources = useMemo(
    () => normalizeSources({ sources: sourcesProp, vizConfig, userVizConfigs, samples }),
    [sourcesProp, vizConfig, userVizConfigs, samples],
  );

  const groupedItems = useMemo(() => {
    return sources.map((source) => {
      const autoItems = buildSlideItems(
        source.vizConfig,
        "auto",
        source.analysisId,
        source.analysisName,
        source.samples,
      );
      const userItems = (source.userVizConfigs ?? []).flatMap((cfg) =>
        buildSlideItems(cfg, "user", source.analysisId, source.analysisName, source.samples),
      ).filter((item) => !autoItems.some((auto) => auto.slide.title.trim().toLowerCase() === item.slide.title.trim().toLowerCase()));
      return {
        source,
        autoItems,
        userItems,
      };
    });
  }, [sources]);

  const filterItems = (items: InsightSlideItem[]) => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter(
      ({ slide, analysisName }) =>
        slide.title.toLowerCase().includes(q) ||
        analysisName.toLowerCase().includes(q) ||
        (slide.subtitle?.toLowerCase().includes(q) ?? false) ||
        slide.insights.some((insight) => insight.toLowerCase().includes(q)),
    );
  };

  const filteredGroups = useMemo(
    () =>
      groupedItems.map((group) => ({
        ...group,
        autoItems: filterItems(group.autoItems),
        userItems: filterItems(group.userItems),
      })),
    [groupedItems, query],
  );

  const hasAnyItems = filteredGroups.some((g) => g.autoItems.length > 0 || g.userItems.length > 0);

  const itemKey = (item: InsightSlideItem, index: number) =>
    `${item.analysisId}-${item.source}-${index}-${item.slide.title}`;

  const renderDragItem = (item: InsightSlideItem, index: number, children: ReactNode) => {
    const key = itemKey(item, index);
    return (
      <div
        key={key}
        draggable
        onDragStart={(event) => {
          setInsightDragData(event, item.chartIndex, item.analysisId);
        }}
        onClick={() => {
          setActiveKey(key);
          onSelectSlide?.(item.slide, item.chartIndex, item.analysisId);
        }}
        className={cx("cursor-grab active:cursor-grabbing", activeKey === key && "rounded-xl ring-1 ring-[#1565ef]/30")}
      >
        {children}
      </div>
    );
  };

  const renderItems = (items: InsightSlideItem[]) => {
    if (layout === "grid") {
      return (
        <div className="flex flex-col gap-3">
          {items.map((item, index) =>
            renderDragItem(
              item,
              index,
              <div
                className={cx(
                  "overflow-hidden rounded-xl border bg-primary transition hover:border-tertiary",
                  activeKey === itemKey(item, index) ? "border-[#1565ef]" : "border-secondary",
                )}
              >
                <DynamicChart slide={item.slide} height={112} compact />
                <p className="truncate px-3 py-2 text-xs font-medium text-primary">{item.slide.title}</p>
              </div>,
            ),
          )}
        </div>
      );
    }

    return (
      <div className="flex flex-col gap-2">
        {items.map((item, index) =>
          renderDragItem(
            item,
            index,
            <div
              className={cx(
                "rounded-lg border bg-primary px-3 py-3 transition hover:border-tertiary",
                activeKey === itemKey(item, index) ? "border-[#1565ef]" : "border-secondary",
              )}
            >
              <div className="flex items-start justify-between gap-2">
                <p className="text-sm font-semibold text-primary">{item.slide.title}</p>
                <span className="shrink-0 text-xs text-tertiary">
                  {formatTimeAgo((item.slide as Slide & { created_at?: string }).created_at)}
                </span>
              </div>
              <p className="mt-1 text-xs leading-5 text-tertiary">{insightDescription(item.slide)}</p>
            </div>,
          ),
        )}
      </div>
    );
  };

  const isCollapsed = (key: string) => collapsedGroups[key] === true;

  const renderSection = (title: string, items: InsightSlideItem[], groupKey: string) => {
    if (!items.length) {
      return (
        <div className="mb-3">
          <p className="py-1.5 text-xs font-semibold text-secondary">{title}</p>
          <p className="text-xs text-tertiary">None yet.</p>
        </div>
      );
    }
    const open = !isCollapsed(groupKey);

    return (
      <div className="mb-3">
        <button
          type="button"
          onClick={() => setCollapsedGroups((prev) => ({ ...prev, [groupKey]: !prev[groupKey] }))}
          className="flex w-full items-center gap-1 py-1.5 text-xs font-semibold text-secondary"
        >
          <ChevronDown className={cx("size-3.5 transition", !open && "-rotate-90")} />
          {title}
        </button>
        {open ? renderItems(items) : null}
      </div>
    );
  };

  return (
    <aside
      className={cx(
        "flex h-full w-[300px] shrink-0 flex-col overflow-hidden rounded-xl border border-secondary bg-primary",
        className,
      )}
    >
      <div className="flex items-center justify-between border-b border-secondary px-4 py-4">
        <h2 className="text-md font-semibold text-primary">All Insights</h2>
        <div className="flex items-center gap-1">
          <div className="inline-flex items-center rounded-md border border-secondary bg-primary p-0.5">
            <button
              type="button"
              onClick={() => setLayout("grid")}
              className={cx(
                "rounded px-1.5 py-1",
                layout === "grid" ? "bg-secondary text-primary" : "text-fg-quaternary",
              )}
              aria-label="Grid view"
            >
              <svg viewBox="0 0 16 16" className="size-3.5" fill="currentColor">
                <rect x="1" y="1" width="6" height="6" rx="1" />
                <rect x="9" y="1" width="6" height="6" rx="1" />
                <rect x="1" y="9" width="6" height="6" rx="1" />
                <rect x="9" y="9" width="6" height="6" rx="1" />
              </svg>
            </button>
            <button
              type="button"
              onClick={() => setLayout("list")}
              className={cx(
                "rounded px-1.5 py-1",
                layout === "list" ? "bg-secondary text-primary" : "text-fg-quaternary",
              )}
              aria-label="List view"
            >
              <svg viewBox="0 0 16 16" className="size-3.5" fill="currentColor">
                <rect x="1" y="2" width="14" height="2" rx="1" />
                <rect x="1" y="7" width="14" height="2" rx="1" />
                <rect x="1" y="12" width="14" height="2" rx="1" />
              </svg>
            </button>
          </div>
          {onClose ? (
            <button
              type="button"
              onClick={onClose}
              aria-label="Close insights panel"
              className="rounded-md p-1 text-fg-quaternary hover:bg-primary_hover"
            >
              <X className="size-5" />
            </button>
          ) : null}
        </div>
      </div>

      <div className="px-4 py-3">
        <div className="flex items-center gap-2 rounded-lg border border-secondary bg-primary px-3 py-2 shadow-xs">
          <SearchLg className="size-4 text-fg-quaternary" />
          <input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActiveKey("auto-0");
            }}
            placeholder="Search"
            aria-label="Search insights"
            className="flex-1 bg-transparent text-sm text-primary placeholder:text-tertiary focus:outline-none"
          />
          <span className="rounded border border-secondary px-1 text-[10px] text-tertiary">⌘K</span>
        </div>
      </div>

      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto px-4 pb-4">
        {!hasAnyItems ? (
          <p className="px-2 py-4 text-center text-sm text-tertiary">
            Charts from this analysis thread will appear here once visualization is ready.
          </p>
        ) : (
          filteredGroups.map((group) => {
            if (!group.autoItems.length && !group.userItems.length) return null;
            const groupOpen = !isCollapsed(`analysis-${group.source.analysisId}`);
            return (
              <div key={group.source.analysisId} className="mb-4 border-b border-secondary pb-4 last:border-b-0">
                <button
                  type="button"
                  onClick={() =>
                    setCollapsedGroups((prev) => ({
                      ...prev,
                      [`analysis-${group.source.analysisId}`]: !prev[`analysis-${group.source.analysisId}`],
                    }))
                  }
                  className="flex w-full items-center gap-1 py-2 text-sm font-semibold text-primary"
                >
                  <ChevronDown className={cx("size-4 transition", !groupOpen && "-rotate-90")} />
                  {group.source.analysisName}
                </button>
                {groupOpen ? (
                  <>
                    {renderSection("Auto Insights", group.autoItems, `${group.source.analysisId}-auto`)}
                    {renderSection("User Insights", group.userItems, `${group.source.analysisId}-user`)}
                  </>
                ) : null}
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
}
