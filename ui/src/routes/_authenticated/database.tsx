import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Loader2, BarChart3, Eye } from "lucide-react";
import { backendApi } from "@/lib/api/backendApi";
import { DatasetPreviewModal } from "@/components/database/DatasetPreviewModal";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

import {
  Database,
  Link2,
  RefreshCw,
  Search,
  Briefcase,
  DollarSign,
  Image as ImageIcon,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Check,
  MoreVertical,
  Link2Off,
  Trash2,
  FileText,
  Upload,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

import { ConnectCloudModal } from "@/components/database/ConnectCloudModal";
import { DisconnectConfirmModal } from "@/components/database/DisconnectConfirmModal";
import { useDataSources, useDataSourceMutations, STORAGE_PROVIDER_LABELS, type DataSource } from "@/lib/data-sources";
import { getProvider } from "@/components/database/providers";
import { PROVIDER_LOGOS, STORAGE_PROVIDER_LOGOS } from "@/components/database/logos";
import { openCloudDatasetInAnalysis } from "@/lib/cloud-dataset-analysis";
import { openDatasetsInProAnalysis } from "@/lib/open-analysis-from-datasets";
import {
  useUploadedDatasets,
  formatBytes,
  groupUploadedDatasets,
  type UploadedDataset,
  type UploadedDatasetGroup,
} from "@/lib/uploaded-datasets";
import { toast } from "sonner";

import { cn } from "@/lib/utils";
import { useUpgradeGate } from "@/components/dashboard/UpgradeGate";

export const Route = createFileRoute("/_authenticated/database")({
  head: () => ({
    meta: [
      { title: "Database · Avaloka AI" },
      { name: "description", content: "Connect and manage your cloud data sources." },
    ],
  }),
  component: DatabasePage,
});


type Tab = "all" | "connected" | "disconnected";
type Section = "cloud" | "database" | "uploads";
const PAGE_SIZE = 9;
const CATEGORY_OPTIONS = ["All", "Product", "Finance", "Marketing", "Accounts", "Human Resource"] as const;

function DatabasePage() {
  const navigate = useNavigate();
  const { data: sources, isLoading } = useDataSources();
  const { update, remove } = useDataSourceMutations();
  const [connectOpen, setConnectOpen] = useState(false);
  const [disconnect, setDisconnect] = useState<DataSource | null>(null);
  const [openingAnalysisId, setOpeningAnalysisId] = useState<string | null>(null);
  const [section, setSection] = useState<Section>("cloud");
  const [tab, setTab] = useState<Tab>("all");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [category, setCategory] = useState<string>("All");
  const { blocked, dialog: upgradeDialog } = useUpgradeGate();

  const { data: uploads, isLoading: uploadsLoading } = useUploadedDatasets();

  const cloudSources = useMemo(
    () => (sources ?? []).filter((s) => s.connectionType === "storage"),
    [sources],
  );
  const dbSources = useMemo(
    () => (sources ?? []).filter((s) => s.connectionType === "database"),
    [sources],
  );




  const filtered = useMemo(() => {
    const list = section === "cloud" ? cloudSources : section === "database" ? dbSources : [];
    return list.filter((s) => {
      if (tab === "connected" && s.status !== "connected") return false;
      if (tab === "disconnected" && s.status === "connected") return false;
      if (query && !`${s.name} ${s.domain ?? ""}`.toLowerCase().includes(query.toLowerCase()))
        return false;
      if (category !== "All") {
        const hay = `${s.name} ${s.domain ?? ""} ${s.use_cases ?? ""}`.toLowerCase();
        if (!hay.includes(category.toLowerCase())) return false;
      }
      return true;
    });
  }, [section, cloudSources, dbSources, tab, query, category]);

  const uploadGroups = useMemo(
    () =>
      groupUploadedDatasets(uploads ?? []).sort(
        (a, b) =>
          new Date(b.createdAt ?? 0).getTime() - new Date(a.createdAt ?? 0).getTime(),
      ),
    [uploads],
  );

  /** Key of the single most recently created group (for the "Newest" badge). */
  const newestGroupKey = useMemo(
    () => uploadGroups.find((g) => !!g.createdAt)?.key ?? null,
    [uploadGroups],
  );

  const filteredUploadGroups = useMemo(() => {
    if (!query) return uploadGroups;
    const q = query.toLowerCase();
    return uploadGroups.filter((g) => {
      if (g.title.toLowerCase().includes(q)) return true;
      return g.members.some((m) =>
        `${m.filename ?? ""} ${m.alias ?? ""} ${m.dataset_id}`.toLowerCase().includes(q),
      );
    });
  }, [uploadGroups, query]);

  // --- Uploaded datasets multi-select ---
  const MAX_SELECTED_DATASETS = 10;
  const [selectedGroupKeys, setSelectedGroupKeys] = useState<string[]>([]);
  const [openingSelection, setOpeningSelection] = useState(false);

  const selectedGroups = useMemo(
    () =>
      selectedGroupKeys
        .map((k) => uploadGroups.find((g) => g.key === k))
        .filter((g): g is UploadedDatasetGroup => !!g),
    [selectedGroupKeys, uploadGroups],
  );
  const selectedMembers = useMemo(
    () => selectedGroups.flatMap((g) => g.members),
    [selectedGroups],
  );
  const selectedCount = selectedMembers.length;
  const atSelectionCap = selectedCount >= MAX_SELECTED_DATASETS;

  const toggleGroupSelection = (g: UploadedDatasetGroup) => {
    setSelectedGroupKeys((prev) => {
      if (prev.includes(g.key)) return prev.filter((k) => k !== g.key);
      const currentCount = prev
        .map((k) => uploadGroups.find((x) => x.key === k))
        .reduce((n, x) => n + (x?.members.length ?? 0), 0);
      if (currentCount + g.members.length > MAX_SELECTED_DATASETS) {
        toast.error("Up to 10 datasets at a time");
        return prev;
      }
      return [...prev, g.key];
    });
  };

  const openSelectedForAnalysis = async () => {
    if (openingSelection || selectedMembers.length === 0) return;
    if (blocked("Opening datasets in analysis")) return;
    setOpeningSelection(true);
    try {
      await openUploadedDatasetsInAnalysis(selectedMembers, navigate);
      setSelectedGroupKeys([]);
    } catch (e) {
      toast.error("Couldn't open the selected datasets", {
        description: (e as Error).message,
      });
    } finally {
      setOpeningSelection(false);
    }
  };

  const activeCount = section === "uploads" ? filteredUploadGroups.length : filtered.length;
  const totalPages = Math.max(1, Math.ceil(activeCount / PAGE_SIZE));
  const pageItems = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const uploadPageGroups = filteredUploadGroups.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const listLoading = section === "uploads" ? uploadsLoading : isLoading;



  const handleOpenAnalysis = async (source: DataSource) => {
    if (openingAnalysisId) return;
    if (blocked("Opening datasets in analysis")) return;
    setOpeningAnalysisId(source.id);

    try {
      await openCloudDatasetInAnalysis(source, navigate);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setOpeningAnalysisId(null);
    }
  };

  return (
    <div className="flex min-h-0 min-w-0 flex-1 bg-muted/40 text-foreground dark:bg-background">
      <main className="flex flex-1 flex-col overflow-y-auto p-4 sm:p-6">
        <div className="flex w-full flex-1 flex-col rounded-2xl border border-border bg-card text-card-foreground shadow-sm">
          {/* Top header */}
          <header className="flex items-center justify-between border-b border-border px-6 py-4">
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5 text-foreground" />
              <h1 className="text-lg font-semibold tracking-tight">Cloud and Database Connections</h1>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                className="gap-2"
                onClick={() => blocked("Syncing data")}
              >
                <RefreshCw className="h-4 w-4" />
                Sync Data
              </Button>
              <Button
                onClick={() => {
                  if (blocked("Connecting data sources")) return;
                  setConnectOpen(true);
                }}
                className="gap-2 bg-[#1565EF] text-white hover:bg-[#1257cf]"
              >
                <Link2 className="h-4 w-4" />
                Connect to cloud
              </Button>

            </div>
          </header>

          {/* Data Set panel */}
          <section className="flex flex-1 flex-col px-6 pb-6">
            <div className="flex items-center gap-2 pt-4">
              <Database className="h-4 w-4 text-foreground" />
              <h2 className="text-sm font-semibold">Data Set</h2>
            </div>

            {/* Section tabs */}
            <div className="mt-3 flex items-center gap-1 border-b border-border">
              {([
                { id: "cloud" as const, label: "Cloud Connections", count: cloudSources.length },
                { id: "database" as const, label: "Database Connections", count: dbSources.length },
                { id: "uploads" as const, label: "Uploaded Datasets", count: uploads?.length ?? 0 },
              ]).map((s) => (
                <button
                  key={s.id}
                  onClick={() => {
                    setSection(s.id);
                    setPage(1);
                    if (s.id === "uploads") setTab("all");
                  }}
                  className={cn(
                    "-mb-px border-b-2 px-3 py-2 text-sm transition",
                    section === s.id
                      ? "border-[#1565EF] font-semibold text-[#1565EF]"
                      : "border-transparent font-medium text-muted-foreground hover:text-foreground",
                  )}
                >
                  {s.label} ({s.count})
                </button>
              ))}
            </div>

            {section !== "uploads" && (
              <div className="flex items-center justify-end py-4">
                <Tabs tab={tab} onChange={(t) => { setTab(t); setPage(1); }} />
              </div>
            )}

            <div className="flex items-center justify-between gap-4 py-4">
              <div className="relative w-full max-w-sm">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={query}
                  onChange={(e) => { setQuery(e.target.value); setPage(1); }}
                  placeholder={section === "uploads" ? "Search datasets" : "Search for trades"}
                  className="h-9 pl-9 pr-12"
                />
                <kbd className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                  ⌘K
                </kbd>
              </div>
              {section !== "uploads" && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button className="flex h-9 min-w-[180px] items-center justify-between gap-2 rounded-md border border-border bg-background px-3 text-sm text-foreground hover:border-[#1565EF]">
                      {category} <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-[180px]">
                    {CATEGORY_OPTIONS.map((opt) => (
                      <DropdownMenuItem
                        key={opt}
                        onClick={() => { setCategory(opt); setPage(1); }}
                        className="flex items-center justify-between"
                      >
                        <span>{opt}</span>
                        {category === opt && <Check className="h-4 w-4 text-[#1565EF]" />}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
            </div>


            {/* Grid */}
            <div>
              {listLoading ? (
                <div className="rounded-lg border border-dashed border-border p-10 text-center text-sm text-muted-foreground">
                  {section === "uploads" ? "Loading datasets…" : "Loading data sources…"}
                </div>
              ) : activeCount === 0 ? (
                <EmptyState
                  section={section}
                  onConnect={() => {
                    if (blocked("Connecting data sources")) return;
                    setConnectOpen(true);
                  }}
                  onUpload={() => navigate({ to: "/dashboard" })}
                />
              ) : section === "uploads" ? (
                <div>
                  {selectedCount > 0 && (
                    <div className="sticky top-0 z-20 mb-4 flex flex-wrap items-center gap-3 rounded-xl border border-[#1565EF]/40 bg-card px-4 py-3 shadow-sm">
                      <span className="text-sm font-semibold">{selectedCount} selected</span>
                      {atSelectionCap && (
                        <span className="text-xs text-muted-foreground">
                          Up to 10 datasets at a time
                        </span>
                      )}
                      <div className="ml-auto flex items-center gap-2">
                        <Button
                          variant="ghost"
                          onClick={() => setSelectedGroupKeys([])}
                          disabled={openingSelection}
                        >
                          Clear selection
                        </Button>
                        <Button
                          onClick={() => void openSelectedForAnalysis()}
                          disabled={openingSelection}
                          className="gap-2 bg-[#1565EF] text-white hover:bg-[#1257cf]"
                        >
                          {openingSelection ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <BarChart3 className="h-4 w-4" />
                          )}
                          Open for analysis
                        </Button>
                      </div>
                    </div>
                  )}
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {uploadPageGroups.map((g) => (
                      <UploadedDatasetCard
                        key={g.key}
                        group={g}
                        isNewest={!!newestGroupKey && g.key === newestGroupKey}
                        selected={selectedGroupKeys.includes(g.key)}
                        selectDisabled={
                          selectedCount + g.members.length > 10 &&
                          !selectedGroupKeys.includes(g.key)
                        }
                        onToggleSelect={() => toggleGroupSelection(g)}
                      />
                    ))}
                  </div>
                </div>

              ) : (
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  {pageItems.map((s) => (
                    <DatasetCard
                      key={s.id}
                      source={s}
                      openingAnalysis={openingAnalysisId === s.id}
                      onOpenAnalysis={() => handleOpenAnalysis(s)}
                      onDisconnect={() => {
                        if (blocked("Managing data sources")) return;
                        setDisconnect(s);
                      }}
                      onReconnect={() => {
                        if (blocked("Managing data sources")) return;
                        update.mutate(
                          { id: s.id, patch: { status: "connected" }, type: s.connectionType },
                          { onSuccess: () => toast.success("Reconnected") },
                        );
                      }}
                      onDelete={() => {
                        if (blocked("Deleting data sources")) return;
                        remove.mutate(
                          { id: s.id, type: s.connectionType },
                          { onSuccess: () => toast.success("Removed") },
                        );
                      }}
                    />
                  ))}
                </div>
              )}


              {activeCount > 0 && (
                <Pagination page={page} totalPages={totalPages} onChange={setPage} />
              )}
            </div>

          </section>
        </div>
      </main>

      <ConnectCloudModal open={connectOpen} onOpenChange={setConnectOpen} />
      <DisconnectConfirmModal
        open={!!disconnect}
        name={disconnect?.name ?? ""}
        onOpenChange={(o) => !o && setDisconnect(null)}
        onConfirm={() => {
          if (!disconnect) return;
          update.mutate(
            { id: disconnect.id, patch: { status: "disconnected" } },
            {
              onSuccess: () => {
                toast.success("Disconnected");
                setDisconnect(null);
              },
            },
          );
        }}
      />
      {upgradeDialog}

    </div>
  );
}

function Tabs({ tab, onChange }: { tab: Tab; onChange: (t: Tab) => void }) {
  const items: { id: Tab; label: string }[] = [
    { id: "all", label: "All" },
    { id: "connected", label: "Connected" },
    { id: "disconnected", label: "Disconnected" },
  ];
  return (
    <div className="inline-flex items-center gap-6">
      {items.map((it) => (
        <button
          key={it.id}
          onClick={() => onChange(it.id)}
          className={cn(
            "text-sm transition",
            tab === it.id
              ? "font-semibold text-foreground"
              : "font-medium text-muted-foreground hover:text-foreground",
          )}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}

function categoryIcon(name: string, domain: string | null) {
  const s = `${name} ${domain ?? ""}`.toLowerCase();
  if (s.includes("sales") || s.includes("revenue"))
    return <DollarSign className="h-4 w-4 text-muted-foreground" />;
  if (s.includes("market") || s.includes("media") || s.includes("brand"))
    return <ImageIcon className="h-4 w-4 text-muted-foreground" />;
  return <Briefcase className="h-4 w-4 text-muted-foreground" />;
}

function DatasetCard({
  source,
  openingAnalysis,
  onOpenAnalysis,
  onDisconnect,
  onReconnect,
  onDelete,
}: {
  source: DataSource;
  openingAnalysis?: boolean;
  onOpenAnalysis?: () => void;
  onDisconnect: () => void;
  onReconnect: () => void;
  onDelete: () => void;
}) {
  const isStorage = source.connectionType === "storage";
  const storageProvider = source.storageProvider;
  const providerLabel = isStorage && storageProvider
    ? STORAGE_PROVIDER_LABELS[storageProvider]
    : getProvider(source.provider)?.name ?? source.provider;
  const logo = isStorage && storageProvider
    ? STORAGE_PROVIDER_LOGOS[storageProvider]
    : PROVIDER_LOGOS[source.provider];
  const live = source.status === "connected";
  const selectedFiles = (source.config.selected_files as string[]) ?? [];
  const canOpenAnalysis =
    live && selectedFiles.length > 0 && (isStorage || source.provider === "bigquery");

  return (
    <div className="group flex flex-col rounded-xl border border-border bg-card text-card-foreground p-4 transition hover:shadow-md dark:hover:border-[#1565EF]/50">
      <div className="flex items-start justify-between">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-border bg-background">
          {categoryIcon(source.name, source.domain)}
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex h-9 items-center justify-center rounded-md border border-border bg-background px-2 dark:bg-white">
            <img src={logo} alt={providerLabel} title={providerLabel} className="h-7 w-auto max-w-[96px] object-contain" />
          </span>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                className="rounded p-1 text-muted-foreground opacity-0 transition hover:bg-muted group-hover:opacity-100"
                aria-label="Card actions"
              >
                <MoreVertical className="h-4 w-4" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {live ? (
                <DropdownMenuItem onClick={onDisconnect}>
                  <Link2Off className="mr-2 h-4 w-4" /> Disconnect
                </DropdownMenuItem>
              ) : (
                <DropdownMenuItem onClick={onReconnect}>
                  <Link2 className="mr-2 h-4 w-4" /> Reconnect
                </DropdownMenuItem>
              )}
              <DropdownMenuItem
                className="text-destructive focus:text-destructive"
                onClick={onDelete}
              >
                <Trash2 className="mr-2 h-4 w-4" /> Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {canOpenAnalysis ? (
        <button
          type="button"
          onClick={onOpenAnalysis}
          disabled={openingAnalysis}
          className="mt-3 text-left text-sm font-semibold text-foreground transition hover:text-[#1565EF] hover:underline disabled:cursor-wait disabled:opacity-70"
        >
          {openingAnalysis ? "Opening…" : source.name}
        </button>
      ) : (
        <h3 className="mt-3 text-sm font-semibold">{source.name}</h3>
      )}
      <p className="mt-0.5 text-xs text-muted-foreground">
        Use case: {source.use_cases || source.domain || "—"}
      </p>

      <div className="mt-3 flex flex-wrap gap-2">
        <Chip
          dotClass={live ? "bg-[#1565EF]" : "bg-violet-500"}
          textClass={live ? "text-[#1565EF] bg-[#eef3ff] dark:bg-[#1565EF]/15 dark:text-[#7aa9ff]" : "text-violet-600 bg-violet-50 dark:bg-violet-500/15 dark:text-violet-300"}
        >
          {live ? "Live Connection" : "Synced Data"}
        </Chip>
        <Chip
          dotClass="bg-emerald-500"
          textClass="text-emerald-700 bg-emerald-50 dark:bg-emerald-500/15 dark:text-emerald-300"
          icon={<Database className="h-3 w-3" />}
        >
          {selectedFiles.length || 0} Dataset{selectedFiles.length === 1 ? "" : "s"}
        </Chip>
      </div>

      <div className="mt-4 flex items-center justify-end border-t border-border pt-3 text-xs">
        <span className="text-muted-foreground">
          {source.created_at
            ? `Updated on ${new Date(source.created_at).toLocaleDateString("en-GB", {
                day: "2-digit",
                month: "short",
                year: "numeric",
              })}`
            : "—"}
        </span>
      </div>

    </div>
  );
}

function Chip({
  children,
  textClass,
  dotClass,
  icon,
}: {
  children: React.ReactNode;
  textClass: string;
  dotClass: string;
  icon?: React.ReactNode;
}) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium", textClass)}>
      {icon ?? <span className={cn("h-1.5 w-1.5 rounded-full", dotClass)} />}
      {children}
    </span>
  );
}

function Pagination({
  page,
  totalPages,
  onChange,
}: {
  page: number;
  totalPages: number;
  onChange: (p: number) => void;
}) {
  const pages = pageRange(page, totalPages);
  return (
    <div className="mt-5 flex items-center justify-between">
      <Button
        variant="outline"
        size="sm"
        className="gap-1"
        disabled={page === 1}
        onClick={() => onChange(page - 1)}
      >
        <ChevronLeft className="h-4 w-4" /> Previous
      </Button>
      <div className="flex items-center gap-1 text-sm">
        {pages.map((p, i) =>
          p === "…" ? (
            <span key={`e-${i}`} className="px-2 text-muted-foreground">…</span>
          ) : (
            <button
              key={p}
              onClick={() => onChange(p)}
              className={cn(
                "h-8 w-8 rounded-md text-sm",
                p === page ? "bg-[#eef3ff] text-[#1565EF] font-medium dark:bg-[#1565EF]/20 dark:text-[#7aa9ff]" : "text-muted-foreground hover:bg-muted",
              )}
            >
              {p}
            </button>
          ),
        )}
      </div>
      <Button
        variant="outline"
        size="sm"
        className="gap-1"
        disabled={page === totalPages}
        onClick={() => onChange(page + 1)}
      >
        Next <ChevronRight className="h-4 w-4" />
      </Button>
    </div>
  );
}

function pageRange(page: number, total: number): (number | "…")[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
  const out: (number | "…")[] = [1, 2, 3];
  if (page > 4) out.push("…");
  out.push(total - 2, total - 1, total);
  return out;
}

/** Shared multi-dataset open path (same one a multi-file upload uses). */
async function openUploadedDatasetsInAnalysis(
  members: UploadedDataset[],
  navigate: unknown,
) {
  const previews = await Promise.all(
    members.map(async (m) => {
      try {
        return await backendApi.previewUploadedDataset(m.dataset_id);
      } catch {
        return null;
      }
    }),
  );
  const datasets = previews.map((res, i) => {
    const m = members[i];
    return {
      dataset_id: m.dataset_id,
      filename: res?.filename ?? m.filename ?? m.dataset_id,
      alias: m.alias ?? null,
      schema: res?.schema ?? null,
      samples: res?.samples ?? [],
      rows: res?.samples ?? [],
      rows_sampled: res?.rows_sampled ?? (res?.samples?.length ?? 0),
      visualization_config: res?.visualization_config,
      visualization_status: res?.visualization_status,
    };
  });
  if (datasets.length === 0) throw new Error("Could not load dataset preview.");
  // Use ONE representative session/thread for the whole group (the first
  // member that produced one) so every chat turn stays in the same session
  // and carries all dataset_ids, instead of per-file ad-hoc sessions.
  const representative =
    previews.find((r) => r?.session_id && r?.thread_id) ??
    previews.find((r) => r?.session_id || r?.thread_id) ??
    null;
  const sessionId = representative?.session_id ?? null;
  const threadId = representative?.thread_id ?? null;
  await openDatasetsInProAnalysis({
    navigate: navigate as (o: Record<string, unknown>) => void,
    sessionId,
    threadId,
    datasets,
    showAutoInsights: true,
  });
}

/** "24 Aug 2026, 2:16 PM" — strictly from the API timestamp. */
function formatDateTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return `${d.toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  })}, ${d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`;
}

/** "just now", "5 min ago", "3 h ago", "2 d ago". */
function relativeTimeLabel(iso: string): string {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "";
  const mins = Math.floor((Date.now() - t) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days} d ago`;
  return "";
}

function UploadedDatasetCard({
  group,
  selected = false,
  selectDisabled = false,
  isNewest = false,
  onToggleSelect,
}: {
  group: UploadedDatasetGroup;
  selected?: boolean;
  selectDisabled?: boolean;
  isNewest?: boolean;
  onToggleSelect?: () => void;
}) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [opening, setOpening] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewData, setPreviewData] = useState<any>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const isMulti = group.members.length > 1;
  const first = group.members[0];

  const openForAnalysis = async () => {
    if (opening) return;
    setOpening(true);
    try {
      await openUploadedDatasetsInAnalysis(group.members, navigate);
    } catch (e) {
      toast.error(`Couldn't open "${group.title}" for analysis`, {
        description: (e as Error).message,
        action: { label: "Retry", onClick: () => void openForAnalysis() },
      });
    } finally {
      setOpening(false);
    }
  };


  const openPreview = async () => {
    if (isMulti) return;
    setPreviewOpen(true);
    setPreviewLoading(true);
    setPreviewError(null);
    setPreviewData(null);
    try {
      const res = await backendApi.previewUploadedDataset(first.dataset_id);
      setPreviewData(res ?? null);
    } catch (e) {
      setPreviewError((e as Error)?.message || "Unknown error");
    } finally {
      setPreviewLoading(false);
    }
  };

  const doDelete = async () => {
    setDeleting(true);
    try {
      await Promise.all(group.members.map((m) => backendApi.deleteDataset(m.dataset_id)));
      const ids = new Set(group.members.map((m) => m.dataset_id));
      qc.setQueryData(["uploaded-datasets"], (old: UploadedDataset[] | undefined) =>
        (old ?? []).filter((d) => !ids.has(d.dataset_id)),
      );
      qc.invalidateQueries({ queryKey: ["uploaded-datasets"] });
      setConfirmDelete(false);
      toast.success(isMulti ? "Datasets deleted" : "Dataset deleted");
    } catch (e) {
      toast.error("Couldn't delete dataset", { description: (e as Error).message });
    } finally {
      setDeleting(false);
    }
  };

  const columnCount = isMulti ? group.totalColumnCount : first.columnCount;
  const sizeBytes = isMulti ? group.totalSizeBytes : first.sizeBytes;
  const createdAt = group.createdAt;

  return (
    <div
      onClick={() => { if (!opening) void openForAnalysis(); }}
      className={cn(
        "group flex cursor-pointer flex-col rounded-xl border bg-card p-4 text-card-foreground transition hover:shadow-md dark:hover:border-[#1565EF]/50",
        selected ? "border-[#1565EF] ring-1 ring-[#1565EF]/40" : "border-border",
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={selected}
            disabled={selectDisabled && !selected}
            aria-label={`Select ${group.title}`}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => { e.stopPropagation(); onToggleSelect?.(); }}
            className={cn(
              "h-4 w-4 shrink-0 cursor-pointer accent-[#1565EF] transition-opacity disabled:cursor-not-allowed disabled:opacity-40",
              selected ? "opacity-100" : "opacity-0 group-hover:opacity-100 focus:opacity-100",
            )}
          />
          <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-border bg-background">
            <FileText className="h-4 w-4 text-muted-foreground" />
          </div>
        </div>

        <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
          {isNewest && (
            <span className="rounded-full bg-[#1565EF]/10 px-2 py-0.5 text-[11px] font-medium text-[#1565EF] dark:bg-[#1565EF]/20">
              Newest
            </span>
          )}
          <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
            {opening ? "Opening…" : "Uploaded"}
          </span>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-7 w-7" aria-label="Dataset actions">
                {opening ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <MoreVertical className="h-4 w-4" />
                )}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuItem onClick={() => void openForAnalysis()} disabled={opening}>
                <BarChart3 className="mr-2 h-4 w-4" /> Open for analysis
              </DropdownMenuItem>
              {!isMulti && (
                <DropdownMenuItem onClick={() => void openPreview()}>
                  <Eye className="mr-2 h-4 w-4" /> Preview
                </DropdownMenuItem>
              )}
              <DropdownMenuItem
                onClick={() => setConfirmDelete(true)}
                className="text-red-600 focus:text-red-600"
              >
                <Trash2 className="mr-2 h-4 w-4" /> Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      <div onClick={(e) => e.stopPropagation()}>
      <DatasetPreviewModal

        open={previewOpen}
        onOpenChange={setPreviewOpen}
        loading={previewLoading}
        data={previewData}
        error={previewError}
        title={first.name}
      />

      <AlertDialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete {isMulti ? "these datasets" : "this dataset"}?</AlertDialogTitle>
            <AlertDialogDescription>This can't be undone.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={(e) => {
                e.preventDefault();
                void doDelete();
              }}
              disabled={deleting}
              className="bg-red-600 text-white hover:bg-red-700"
            >
              {deleting ? "Deleting…" : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      </div>


      <div className="mt-3 flex items-start gap-2">
        {isMulti && (
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); setExpanded((v) => !v); }}
            className="mt-0.5 rounded p-0.5 text-muted-foreground hover:bg-muted"
            aria-label={expanded ? "Collapse files" : "Expand files"}
          >
            <ChevronRight className={cn("h-4 w-4 transition-transform", expanded && "rotate-90")} />
          </button>
        )}
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-sm font-semibold" title={group.title}>
            {group.title}
          </h3>
          <p className="mt-0.5 truncate text-xs text-muted-foreground" title={group.subtitle}>
            {group.subtitle}
          </p>
        </div>
        {isMulti && (
          <Chip
            dotClass="bg-emerald-500"
            textClass="text-emerald-700 bg-emerald-50 dark:bg-emerald-500/15 dark:text-emerald-300"
            icon={<Database className="h-3 w-3" />}
          >
            {group.members.length} file{group.members.length === 1 ? "" : "s"}
          </Chip>
        )}
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        <Chip
          dotClass="bg-emerald-500"
          textClass="text-emerald-700 bg-emerald-50 dark:bg-emerald-500/15 dark:text-emerald-300"
          icon={<Database className="h-3 w-3" />}
        >
          {columnCount ?? "—"} Column{columnCount === 1 ? "" : "s"}
        </Chip>
        <Chip dotClass="bg-slate-400" textClass="bg-muted text-muted-foreground">
          {formatBytes(sizeBytes)}
        </Chip>
      </div>

      {isMulti && expanded && (
        <div className="mt-3 rounded-lg border border-border bg-muted/30 p-2">
          {group.members.map((m) => (
            <div
              key={m.dataset_id}
              className="flex items-center justify-between gap-3 py-1.5"
            >
              <div className="flex min-w-0 items-center gap-2">
                <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                <span
                  className="truncate text-xs font-medium text-foreground"
                  title={m.filename ?? m.alias ?? m.dataset_id}
                >
                  {m.filename ?? m.alias ?? "Untitled file"}
                </span>
              </div>
              <div className="flex shrink-0 items-center gap-2 text-[11px] text-muted-foreground">
                <span>{m.columns.length} Column{m.columns.length === 1 ? "" : "s"}</span>
                <span>·</span>
                <span>{formatBytes(m.sizeBytes)}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-3 flex items-center justify-between gap-2 border-t border-border pt-2 text-xs leading-none">
        <span className="text-muted-foreground">
          {createdAt ? relativeTimeLabel(createdAt) : ""}
        </span>
        <span className="text-muted-foreground">
          {createdAt ? `Created on ${formatDateTime(createdAt)}` : "—"}
        </span>
      </div>
    </div>
  );
}

function EmptyState({
  section,
  onConnect,
  onUpload,
}: {
  section: Section;
  onConnect: () => void;
  onUpload: () => void;
}) {
  const isUploads = section === "uploads";
  const title = isUploads
    ? "No uploaded datasets yet"
    : section === "database"
      ? "No database connections yet"
      : "No cloud connections yet";
  const body = isUploads
    ? "Upload a file to create your first dataset, then analyse it here."
    : section === "database"
      ? "Connect a database to query your tables and build analyses."
      : "Connect a cloud bucket to start importing files and building analyses.";

  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border bg-muted/20 px-6 py-16 text-center">
      <div className="rounded-full bg-[#eef3ff] p-4 text-[#1565EF] dark:bg-[#1565EF]/15 dark:text-[#7aa9ff]">
        {isUploads ? <FileText className="h-8 w-8" /> : <Database className="h-8 w-8" />}
      </div>
      <h2 className="mt-4 text-lg font-semibold">{title}</h2>
      <p className="mt-1 max-w-md text-sm text-muted-foreground">{body}</p>
      <Button
        onClick={isUploads ? onUpload : onConnect}
        className="mt-6 gap-2 bg-[#1565EF] text-white hover:bg-[#1257cf]"
      >
        {isUploads ? <Upload className="h-4 w-4" /> : <Link2 className="h-4 w-4" />}
        {isUploads ? "Upload a dataset" : "Connect to cloud"}
      </Button>
    </div>
  );
}

