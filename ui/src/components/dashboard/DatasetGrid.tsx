import { Database01 } from "@untitledui/icons";
import { cx } from "@/lib/utils/cx";

export type DatasetItem = {
  id: string;
  name: string;
  date: string;
};

interface DatasetGridProps {
  datasets: DatasetItem[];
  maxVisible?: number;
  onSelect?: (id: string) => void;
  onViewMore?: () => void;
}

export function DatasetGrid({
  datasets,
  maxVisible = 5,
  onSelect,
  onViewMore,
}: DatasetGridProps) {
  const visible = datasets.slice(0, maxVisible);
  const showViewMore = datasets.length > maxVisible;

  return (
    <div className="grid grid-cols-3 gap-3">
      {visible.map((d) => (
        <button
          key={d.id}
          type="button"
          onClick={() => onSelect?.(d.id)}
          className={cx(
            "flex h-[72px] items-center gap-3 rounded-xl border border-secondary bg-primary px-4 text-left shadow-xs",
            "transition hover:border-brand hover:bg-primary_hover",
          )}
        >
          <div className="flex size-10 shrink-0 items-center justify-center rounded-lg border border-secondary bg-primary">
            <Database01 className="size-5 text-fg-secondary" />
          </div>
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-primary">{d.name}</p>
            <p className="text-xs text-tertiary">{d.date}</p>
          </div>
        </button>
      ))}
      {showViewMore && (
        <button
          type="button"
          onClick={onViewMore}
          className="flex h-[72px] items-center justify-center rounded-xl border border-secondary bg-primary text-sm font-semibold text-primary shadow-xs transition hover:border-brand hover:bg-primary_hover"
        >
          View More
        </button>
      )}
    </div>
  );
}
