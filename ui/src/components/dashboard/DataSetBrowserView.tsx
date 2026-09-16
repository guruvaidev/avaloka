import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { SearchLg, Database01, Database02, Folder, ArrowLeft, ArrowRight, Calendar } from "@untitledui/icons";
import { cx } from "@/lib/utils/cx";
import { toast } from "sonner";
import { dataSetCardsKey, fetchDataSetCards, type DataSetCard } from "@/lib/api/data-set-cards";
import { useDataSources } from "@/lib/data-sources";
import { openCloudDatasetInAnalysis } from "@/lib/cloud-dataset-analysis";
import { PROVIDER_LOGOS } from "@/components/database/logos";
import type { Provider } from "@/lib/data-sources";

const PAGE_SIZE = 6;

function pageRange(page: number, total: number): (number | "…")[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
  const out: (number | "…")[] = [1, 2, 3];
  if (page > 4) out.push("…");
  if (total > 3) out.push(total - 2, total - 1, total);
  return out.filter((v, i, arr) => i === 0 || v !== arr[i - 1]);
}

function CardIcon({ kind }: { kind: DataSetCard["icon"] }) {
  if (kind === "dollar") {
    return <span className="text-base font-semibold text-fg-quaternary">$</span>;
  }
  if (kind === "image") {
    return (
      <svg viewBox="0 0 24 24" className="size-4 text-fg-quaternary" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <circle cx="9" cy="9" r="2" />
        <path d="m21 15-5-5L5 21" />
      </svg>
    );
  }
  return <Folder className="size-4 text-fg-quaternary" />;
}

function ProviderBadge({ card }: { card: DataSetCard }) {
  if (card.providerKey && PROVIDER_LOGOS[card.providerKey as Provider]) {
    return (
      <img
        src={PROVIDER_LOGOS[card.providerKey as Provider]}
        alt={card.provider}
        className="h-4 w-auto max-w-[72px] object-contain"
      />
    );
  }
  return (
    <span className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: card.providerColor }}>
      {card.provider}
    </span>
  );
}

type DataSetBrowserViewProps = {
  /** Called after navigating away (e.g. close a parent modal). */
  onNavigateAway?: () => void;
  className?: string;
};

export function DataSetBrowserView({ onNavigateAway, className }: DataSetBrowserViewProps) {
  const navigate = useNavigate();
  const {
    data: cards = [],
    isLoading,
    refetch,
    isFetching,
  } = useQuery({
    queryKey: dataSetCardsKey,
    queryFn: fetchDataSetCards,
  });
  const { data: dataSources = [] } = useDataSources();

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [opening, setOpening] = useState(false);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return cards;
    return cards.filter(
      (d) =>
        d.title.toLowerCase().includes(q) ||
        d.useCase.toLowerCase().includes(q) ||
        d.provider.toLowerCase().includes(q) ||
        d.author.toLowerCase().includes(q),
    );
  }, [cards, search]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageItems = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [page, totalPages]);

  const latestDateLabel = useMemo(() => {
    if (!cards.length) {
      return new Date().toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
      });
    }
    return new Date(cards[0].createdAt).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  }, [cards]);

  const selected = cards.find((c) => c.id === selectedId) ?? null;

  const openCard = async (card: DataSetCard) => {
    setOpening(true);
    try {
      if (card.source === "upload" && card.analysisId) {
        onNavigateAway?.();
        navigate({ to: "/analysis", search: { aid: card.analysisId } });
        return;
      }
      if (card.source === "cloud" && card.dataSourceId) {
        const source = dataSources.find((s) => s.id === card.dataSourceId);
        if (!source) {
          toast.error("Connection not found. Try refreshing the list.");
          await refetch();
          return;
        }
        onNavigateAway?.();
        await openCloudDatasetInAnalysis(source, navigate);
        return;
      }
      if (card.source === "database") {
        onNavigateAway?.();
        navigate({ to: "/database" });
        return;
      }
      toast.error("Unable to open this dataset.");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to open dataset");
    } finally {
      setOpening(false);
    }
  };

  return (
    <div
      className={cx(
        "grid min-h-0 flex-1 grid-rows-[auto_auto_1fr_auto] gap-0 overflow-hidden bg-primary",
        className,
      )}
    >
      <div className="flex items-center gap-2 border-b border-secondary px-5 py-3.5">
        <Database01 className="size-4 text-fg-secondary" />
        <h1 className="text-sm font-semibold text-primary">Data Set</h1>
      </div>

      <div className="flex items-center justify-between gap-3 px-5 py-4">
        <div className="flex max-w-[360px] flex-1 items-center gap-2 rounded-lg border border-secondary bg-primary px-3 py-2 shadow-xs">
          <SearchLg className="size-4 text-fg-quaternary" />
          <input
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            placeholder="Search for trades"
            className="flex-1 bg-transparent text-sm text-primary placeholder:text-tertiary focus:outline-none"
          />
          <span className="rounded border border-secondary px-1 text-[10px] text-tertiary">⌘K</span>
        </div>
        <button
          type="button"
          onClick={() => void refetch()}
          disabled={isFetching}
          className="inline-flex items-center gap-2 rounded-lg border border-secondary bg-primary px-3 py-2 text-sm font-medium text-secondary shadow-xs hover:bg-secondary disabled:opacity-60"
        >
          <Calendar className="size-4 text-fg-quaternary" />
          {latestDateLabel}
        </button>
      </div>

      <div className="overflow-y-auto px-5 pb-4">
        {isLoading ? (
          <p className="py-16 text-center text-sm text-tertiary">Loading databases and datasets…</p>
        ) : filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-secondary py-16 text-center">
            <Database01 className="size-8 text-fg-quaternary" />
            <p className="mt-3 text-sm font-semibold text-primary">No databases or datasets yet</p>
            <p className="mt-1 max-w-sm text-sm text-tertiary">
              Connect a cloud database from the Database page or upload a file from the dashboard.
            </p>
            <button
              type="button"
              onClick={() => {
                onNavigateAway?.();
                navigate({ to: "/database" });
              }}
              className="mt-4 rounded-lg bg-[#1565ef] px-4 py-2 text-sm font-semibold text-white hover:bg-[#1257d6]"
            >
              Connect database
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {pageItems.map((d) => {
              const isSelected = selectedId === d.id;
              return (
                <button
                  key={d.id}
                  type="button"
                  onClick={() => setSelectedId(d.id)}
                  onDoubleClick={() => void openCard(d)}
                  className={cx(
                    "flex h-fit flex-col gap-3 rounded-xl border bg-primary p-4 text-left transition",
                    isSelected
                      ? "border-[#1565ef] ring-2 ring-[#1565ef]/40"
                      : "border-secondary hover:border-tertiary",
                  )}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex size-8 shrink-0 items-center justify-center rounded-md border border-secondary bg-primary">
                      <CardIcon kind={d.icon} />
                    </div>
                    <ProviderBadge card={d} />
                  </div>
                  <div className="space-y-0.5">
                    <h4 className="text-sm font-semibold text-primary">{d.title}</h4>
                    <p className="text-xs text-tertiary">Use case: {d.useCase}</p>
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span
                      className={cx(
                        "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium",
                        d.connectionType === "live" ? "bg-[#eff4ff] text-[#3538cd]" : "bg-[#f4f3ff] text-[#5925dc]",
                      )}
                    >
                      <span
                        className={cx(
                          "size-1.5 rounded-full",
                          d.connectionType === "live" ? "bg-[#3538cd]" : "bg-[#5925dc]",
                        )}
                      />
                      {d.connectionType === "live" ? "Live Connection" : "Synced Data"}
                    </span>
                    <span className="inline-flex items-center gap-1 rounded-md bg-[#ecfdf3] px-1.5 py-0.5 text-[11px] font-medium text-[#067647]">
                      <Database02 className="size-3" />
                      {d.fileCount} Dataset{d.fileCount === 1 ? "" : "s"}
                    </span>
                  </div>
                  <div className="mt-1 flex items-center gap-2 border-t border-secondary pt-2.5">
                    <span className="grid size-6 shrink-0 place-items-center rounded-full bg-[#eff4ff] text-[10px] font-semibold text-[#175cd3]">
                      {d.authorInitials}
                    </span>
                    <p className="min-w-0 flex-1 truncate text-xs font-medium text-primary">{d.author}</p>
                    <p className="shrink-0 text-[10px] text-tertiary">Updated on {d.updated}</p>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>

      <div className="flex items-center justify-between border-t border-secondary px-5 py-3">
        <button
          type="button"
          disabled={page <= 1}
          onClick={() => setPage((p) => Math.max(1, p - 1))}
          className="inline-flex items-center gap-1.5 rounded-lg border border-secondary bg-primary px-3 py-1.5 text-sm font-medium text-secondary shadow-xs hover:bg-secondary disabled:cursor-not-allowed disabled:opacity-50"
        >
          <ArrowLeft className="size-4" /> Previous
        </button>

        <div className="flex items-center gap-1 text-sm text-tertiary">
          {pageRange(page, totalPages).map((p, i) =>
            p === "…" ? (
              <span key={`ellipsis-${i}`} className="px-1">
                …
              </span>
            ) : (
              <button
                key={p}
                type="button"
                onClick={() => setPage(p)}
                className={cx(
                  "grid size-8 place-items-center rounded-md",
                  page === p ? "bg-secondary font-semibold text-primary" : "hover:bg-secondary",
                )}
              >
                {p}
              </button>
            ),
          )}
        </div>

        <div className="flex items-center gap-2">
          {selected ? (
            <button
              type="button"
              disabled={opening}
              onClick={() => void openCard(selected)}
              className="inline-flex items-center gap-1.5 rounded-lg bg-[#1565ef] px-3 py-1.5 text-sm font-semibold text-white shadow-xs hover:bg-[#1257d6] disabled:opacity-60"
            >
              {opening ? "Opening…" : "Open"}
            </button>
          ) : null}
          <button
            type="button"
            disabled={page >= totalPages}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            className="inline-flex items-center gap-1.5 rounded-lg border border-secondary bg-primary px-3 py-1.5 text-sm font-medium text-secondary shadow-xs hover:bg-secondary disabled:cursor-not-allowed disabled:opacity-50"
          >
            Next <ArrowRight className="size-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
