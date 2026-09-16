import { useEffect, useMemo, useState } from "react";
import { ChevronDown, SearchLg, X } from "@untitledui/icons";
import { cx } from "@/lib/utils/cx";
import {
  formatHistoryClockTime,
  groupHistoryEntriesByDate,
  isHistoryEntryViewable,
  loadAnalysisHistory,
  type AnalysisHistoryEntry,
} from "@/lib/analysis-history";

export type HistoryVersionItem = {
  id: string;
  name: string;
  initials: string;
  time: string;
  createdAt: string;
  description: string;
  avatarClassName?: string;
  output?: unknown | null;
  viewable?: boolean;
};

const DEFAULT_ITEMS: HistoryVersionItem[] = [
  {
    id: "h1",
    name: "Olivia Rhye",
    initials: "OR",
    time: "10:16 AM",
    createdAt: new Date().toISOString(),
    description: "Created a visualisation cell for data_set.",
    avatarClassName: "bg-[#eff4ff] text-[#175cd3]",
  },
  {
    id: "h2",
    name: "Avaloka AI",
    initials: "AI",
    time: "10:10 AM",
    createdAt: new Date(Date.now() - 6 * 60_000).toISOString(),
    description: "Created a New visualisation cell.",
    avatarClassName: "bg-[#ecfdf3] text-[#067647]",
  },
];

function toHistoryItem(row: AnalysisHistoryEntry): HistoryVersionItem {
  return {
    id: row.id,
    name: row.actor_name,
    initials: row.actor_initials ?? row.actor_name.slice(0, 2).toUpperCase(),
    time: formatHistoryClockTime(row.created_at),
    createdAt: row.created_at,
    description: row.description,
    output: row.output,
    viewable: isHistoryEntryViewable(row.output),
    avatarClassName: row.actor_kind === "ai" ? "bg-[#ecfdf3] text-[#067647]" : "bg-[#eff4ff] text-[#175cd3]",
  };
}

function HistoryItemCard({ item, onView }: { item: HistoryVersionItem; onView?: (item: HistoryVersionItem) => void }) {
  return (
    <div className="rounded-xl border border-secondary bg-primary p-3 shadow-xs">
      <div className="flex items-start gap-3">
        <span
          className={cx(
            "grid size-9 shrink-0 place-items-center rounded-full text-xs font-semibold",
            item.avatarClassName ?? "bg-secondary text-secondary",
          )}
        >
          {item.initials}
        </span>
        <div className="min-w-0 flex-1 overflow-hidden">
          <div className="flex items-start justify-between gap-2">
            <p className="min-w-0 flex-1 truncate text-sm font-semibold text-primary">{item.name}</p>
            <span className="shrink-0 text-[11px] text-tertiary">{item.time}</span>
          </div>
          <p className="mt-1 line-clamp-3 text-sm text-secondary">{item.description}</p>
          {item.viewable ? (
            <button
              type="button"
              onClick={() => onView?.(item)}
              className="mt-2 text-sm font-semibold text-[#1565ef] hover:text-[#1257d6]"
            >
              View
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}

type Props = {
  analysisId?: string | null;
  active?: boolean;
  refreshKey?: number | string;
  liveChat?: Array<{ id: string; role: string; content: string }>;
  items?: HistoryVersionItem[];
  onView?: (item: HistoryVersionItem) => void;
  onClose?: () => void;
  className?: string;
};

export function HistoryVersionPanel({
  analysisId = null,
  active = true,
  refreshKey,
  liveChat,
  items: itemsProp,
  onView,
  onClose,
  className,
}: Props) {
  const [query, setQuery] = useState("");
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});
  const [loadedItems, setLoadedItems] = useState<HistoryVersionItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!active) return;
    if (itemsProp) return;
    if (!analysisId) {
      setLoadedItems([]);
      return;
    }
    let cancelled = false;
    setLoading(true);
    loadAnalysisHistory(analysisId, liveChat)
      .then((rows) => {
        if (!cancelled) setLoadedItems(rows.map(toHistoryItem));
      })
      .catch((err) => {
        console.error("[HistoryVersionPanel] load failed", err);
        if (!cancelled) setLoadedItems([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [active, analysisId, itemsProp, refreshKey, liveChat]);

  const items = useMemo(() => {
    if (itemsProp) return itemsProp;
    if (!analysisId) return DEFAULT_ITEMS;
    return loadedItems;
  }, [itemsProp, analysisId, loadedItems]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter(
      (item) =>
        item.name.toLowerCase().includes(q) ||
        item.description.toLowerCase().includes(q) ||
        item.time.toLowerCase().includes(q),
    );
  }, [items, query]);

  const dateGroups = useMemo(() => {
    const entries: AnalysisHistoryEntry[] = filtered.map((item) => ({
      id: item.id,
      analysis_id: analysisId ?? "",
      actor_name: item.name,
      actor_initials: item.initials,
      actor_kind: item.avatarClassName?.includes("067647") ? "ai" : "user",
      description: item.description,
      created_at: item.createdAt,
      output: item.output ?? null,
    }));
    return groupHistoryEntriesByDate(entries).map((group) => ({
      ...group,
      items: group.entries
        .map((entry) => filtered.find((i) => i.id === entry.id))
        .filter((i): i is HistoryVersionItem => Boolean(i)),
    }));
  }, [filtered, analysisId]);

  useEffect(() => {
    if (!dateGroups.length) return;
    setOpenGroups((prev) => {
      const next = { ...prev };
      for (const group of dateGroups) {
        if (next[group.dateKey] === undefined) {
          next[group.dateKey] = group.label === "Today" || group.label === "Yesterday";
        }
      }
      return next;
    });
  }, [dateGroups]);

  const toggleGroup = (dateKey: string) => {
    setOpenGroups((prev) => ({ ...prev, [dateKey]: !prev[dateKey] }));
  };

  return (
    <aside
      className={cx(
        "flex h-full w-[300px] shrink-0 flex-col overflow-hidden rounded-xl border border-secondary bg-primary",
        className,
      )}
    >
      <div className="flex items-center justify-between border-b border-secondary px-4 py-4">
        <h2 className="text-md font-semibold text-primary">History</h2>
        {onClose ? (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close history panel"
            className="rounded-md p-1 text-fg-quaternary hover:bg-primary_hover"
          >
            <X className="size-5" />
          </button>
        ) : null}
      </div>

      <div className="px-4 py-3">
        <div className="flex items-center gap-2 rounded-lg border border-secondary bg-primary px-3 py-2 shadow-xs">
          <SearchLg className="size-4 text-fg-quaternary" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search"
            aria-label="Search history"
            className="flex-1 bg-transparent text-sm text-primary placeholder:text-tertiary focus:outline-none"
          />
          <span className="rounded border border-secondary px-1 text-[10px] text-tertiary">⌘K</span>
        </div>
      </div>

      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-4 pb-4">
        {loading && items.length === 0 ? (
          <p className="py-6 text-center text-sm text-tertiary">Loading history…</p>
        ) : dateGroups.length === 0 ? (
          <p className="py-6 text-center text-sm text-tertiary">No history entries found.</p>
        ) : (
          dateGroups.map((group) => {
            const isOpen = openGroups[group.dateKey] ?? true;
            return (
              <section key={group.dateKey}>
                <button
                  type="button"
                  onClick={() => toggleGroup(group.dateKey)}
                  className="flex w-full items-center gap-2 py-2 text-left text-sm font-semibold text-primary"
                >
                  <ChevronDown className={cx("size-4 text-fg-quaternary transition", !isOpen && "-rotate-90")} />
                  {group.label}
                </button>
                {isOpen ? (
                  <div className="flex flex-col gap-3">
                    {group.items.map((item) => (
                      <HistoryItemCard key={item.id} item={item} onView={onView} />
                    ))}
                  </div>
                ) : null}
              </section>
            );
          })
        )}
      </div>
    </aside>
  );
}
