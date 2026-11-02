// Tab strip of the Data View card, moved verbatim out of routes/analysis.tsx
// (which renders it) so ui/tests/release_1_6/bg43-data-view-tabs.test.tsx can
// check the tab names. BG43: the structural view is "Column Summary", distinct
// from the data table, never an ambiguous "Data Preview".
import { cx } from "@/lib/utils/cx";

export type DataViewKey = "sample" | "results" | "chart" | "preview";

export function DataViewTabs({
  view,
  resultCount,
  onSelect,
}: {
  view: DataViewKey;
  /** Number of analysis results (tableRuns.length in the route). */
  resultCount: number;
  onSelect: (key: DataViewKey) => void;
}) {
  return (
    <div
      role="tablist"
      aria-label="Dataset views"
      className="inline-flex items-center rounded-lg bg-[var(--workspace-tablist-bg)] p-0.5"
    >
      {(
        [
          ["sample", "Data sample"],
          [
            "results",
            resultCount
              ? `Analysis results (${resultCount})`
              : "Analysis results",
          ],
          ["chart", "Chart"],
          ["preview", "Column Summary"],
        ] as const
      ).map(([key, label]) => (
        <button
          key={key}
          role="tab"
          aria-selected={view === key}
          onClick={() => onSelect(key)}
          title={
            key === "preview" ? "Structure of the dataset: one row per column" : undefined
          }
          className={cx(
            "rounded-md px-3 py-1.5 text-sm font-semibold transition",
            view === key
              ? "bg-primary text-primary shadow-[0_1px_2px_rgba(16,24,40,0.12)]"
              : "text-tertiary hover:text-primary",
          )}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
