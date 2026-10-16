import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { SearchLg, Database01, ArrowLeft, ArrowRight, Upload01, X, Check } from "@untitledui/icons";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { cx } from "@/lib/utils/cx";
import { dataSetCardsKey, fetchDataSetCards, type DataSetCard } from "@/lib/api/data-set-cards";
import { backendApi, type MultiUploadResponse } from "@/lib/api/backendApi";
import { supabase } from "@/integrations/supabase/client";

export const MAX_CHAT_DATASETS = 10;
const LIMIT_MSG = "A chat can have at most 10 datasets";
const PAGE_SIZE = 6;
const ACCEPT = ".csv,.xlsx,.xls,.json";
const ALLOWED = /\.(csv|xlsx|xls|json)$/i;

type Props = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  threadId: string | null;
  currentAnalysisId: string | null;
  /** dataset_ids already attached to this chat (chips). */
  chipDatasetIds: string[];
  /** Called after a successful attach; parent reloads chips. */
  onAttached: () => Promise<void> | void;
};

type Tab = "existing" | "upload";
type FileStatus = { name: string; state: "pending" | "ready" | "skipped"; sheets?: string[] };

export function attachErrorMessage(err: unknown): string {
  const status = (err as any)?.status;
  if (status === 404) return "That dataset is no longer available";
  if (status === 413) return LIMIT_MSG;
  if (status === 401) return "Please sign in again";
  return err instanceof Error && err.message ? err.message : "Something went wrong. Please retry.";
}

export function AttachDatasetsModal({ open, onOpenChange, threadId, currentAnalysisId, chipDatasetIds, onAttached }: Props) {
  const [tab, setTab] = useState<Tab>("existing");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [picked, setPicked] = useState<string[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [statuses, setStatuses] = useState<FileStatus[]>([]);
  const [progress, setProgress] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const { data: cards = [], isLoading } = useQuery({
    queryKey: dataSetCardsKey,
    queryFn: fetchDataSetCards,
    enabled: open,
  });
  const uploadCards = useMemo(() => cards.filter((c) => c.source === "upload" && c.analysisId), [cards]);

  // Map analysis -> dataset_id so cards already in this chat show as disabled.
  const analysisIds = useMemo(() => uploadCards.map((c) => c.analysisId!), [uploadCards]);
  const { data: analysisDatasetMap = {} } = useQuery({
    queryKey: ["attach-modal-analysis-datasets", analysisIds],
    enabled: open && analysisIds.length > 0,
    queryFn: async () => {
      const { data } = await supabase.from("analyses").select("id, dataset_id").in("id", analysisIds);
      const map: Record<string, string | null> = {};
      for (const row of (data ?? []) as any[]) map[row.id] = row.dataset_id ?? null;
      return map;
    },
  });

  useEffect(() => {
    if (!open) {
      setPicked([]);
      setFiles([]);
      setStatuses([]);
      setProgress(null);
      setSearch("");
      setPage(1);
      setTab("existing");
    }
  }, [open]);

  const inChat = (c: DataSetCard) => {
    if (c.analysisId && c.analysisId === currentAnalysisId) return true;
    const ds = c.analysisId ? analysisDatasetMap[c.analysisId] : null;
    return !!ds && chipDatasetIds.includes(ds);
  };

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return uploadCards;
    return uploadCards.filter(
      (d) => d.title.toLowerCase().includes(q) || d.useCase.toLowerCase().includes(q) || d.author.toLowerCase().includes(q),
    );
  }, [uploadCards, search]);
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageItems = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [page, totalPages]);

  const selectionCount = tab === "existing" ? picked.length : files.length;
  const overLimit = chipDatasetIds.length + selectionCount > MAX_CHAT_DATASETS;
  const primaryDisabled = !threadId || busy || selectionCount === 0 || overLimit;

  const togglePick = (id: string) =>
    setPicked((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  const addFiles = (list: FileList | File[] | null) => {
    if (!list) return;
    const arr = Array.from(list);
    const bad = arr.filter((f) => !ALLOWED.test(f.name));
    if (bad.length) toast.error(`Unsupported file type: ${bad.map((f) => f.name).join(", ")}`);
    const good = arr.filter((f) => ALLOWED.test(f.name));
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name));
      return [...prev, ...good.filter((f) => !names.has(f.name))];
    });
    setStatuses([]);
  };

  const finishSuccess = async () => {
    await onAttached();
    onOpenChange(false);
  };

  const submitExisting = async () => {
    if (!threadId) return;
    setBusy(true);
    try {
      await backendApi.attachAnalysesToThread(threadId, picked);
      await finishSuccess();
    } catch (err) {
      toast.error(attachErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const submitUpload = async () => {
    if (!threadId) return;
    setBusy(true);
    setProgress(0);
    setStatuses(files.map((f) => ({ name: f.name, state: "pending" })));
    try {
      const res: MultiUploadResponse = await backendApi.uploadFiles(
        files,
        undefined,
        undefined,
        (p) => setProgress(p.total ? Math.round((p.loaded / p.total) * 100) : null),
        threadId,
      );
      const datasets = (res?.datasets ?? []) as any[];
      const next: FileStatus[] = files.map((f) => {
        const matches = datasets.filter((d) => d?.filename === f.name || d?.source_workbook === f.name);
        return matches.length
          ? { name: f.name, state: "ready", sheets: matches.map((d) => d?.sheet_name).filter(Boolean) }
          : { name: f.name, state: "skipped" };
      });
      setStatuses(next);
      const skipped = next.filter((s) => s.state === "skipped");
      if (skipped.length === next.length) {
        toast.error("Skipped by the server — no datasets were added.");
        return;
      }
      if (skipped.length) toast.warning(`Skipped by the server: ${skipped.map((s) => s.name).join(", ")}`);
      await finishSuccess();
    } catch (err) {
      setStatuses([]);
      toast.error(attachErrorMessage(err));
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(v) => !busy && onOpenChange(v)}>
      <DialogContent className="grid h-[min(720px,calc(100vh-64px))] w-[min(1080px,calc(100vw-48px))] max-w-none grid-rows-[auto_auto_1fr_auto] gap-0 overflow-hidden rounded-2xl border-secondary bg-primary p-0 shadow-2xl">
        <div className="flex items-center gap-2 border-b border-secondary px-5 py-3.5">
          <Database01 className="size-4 text-fg-secondary" />
          <DialogTitle className="text-sm font-semibold text-primary">Add datasets to this chat</DialogTitle>
        </div>

        <div className="flex items-center justify-between gap-3 px-5 pt-4">
          <div className="inline-flex rounded-lg border border-secondary bg-secondary p-0.5">
            {(["existing", "upload"] as Tab[]).map((t) => (
              <button
                key={t}
                type="button"
                disabled={busy}
                onClick={() => setTab(t)}
                className={cx(
                  "rounded-md px-3 py-1.5 text-sm font-medium transition",
                  tab === t ? "bg-primary text-primary shadow-xs" : "text-tertiary hover:text-secondary",
                )}
              >
                {t === "existing" ? "Existing datasets" : "Upload new"}
              </button>
            ))}
          </div>
          {tab === "existing" ? (
            <div className="flex max-w-[360px] flex-1 items-center gap-2 rounded-lg border border-secondary bg-primary px-3 py-2 shadow-xs">
              <SearchLg className="size-4 text-fg-quaternary" />
              <input
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value);
                  setPage(1);
                }}
                placeholder="Search datasets"
                className="flex-1 bg-transparent text-sm text-primary placeholder:text-tertiary focus:outline-none"
              />
            </div>
          ) : null}
        </div>

        <div className="min-h-0 overflow-y-auto px-5 py-4">
          {tab === "existing" ? (
            isLoading ? (
              <p className="py-16 text-center text-sm text-tertiary">Loading datasets…</p>
            ) : filtered.length === 0 ? (
              <p className="py-16 text-center text-sm text-tertiary">No datasets found.</p>
            ) : (
              <>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                  {pageItems.map((d) => {
                    const disabled = inChat(d);
                    const checked = picked.includes(d.analysisId!);
                    return (
                      <button
                        key={d.id}
                        type="button"
                        disabled={disabled || busy}
                        onClick={() => togglePick(d.analysisId!)}
                        aria-pressed={checked}
                        className={cx(
                          "flex h-fit flex-col gap-2 rounded-xl border bg-primary p-4 text-left transition",
                          disabled
                            ? "cursor-not-allowed border-secondary opacity-60"
                            : checked
                              ? "border-brand ring-2 ring-brand"
                              : "border-secondary hover:border-tertiary",
                        )}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <span
                            className={cx(
                              "grid size-4 shrink-0 place-items-center rounded border",
                              checked ? "border-brand bg-brand-solid text-white" : "border-primary bg-primary",
                            )}
                          >
                            {checked ? <Check className="size-3" /> : null}
                          </span>
                          {disabled ? (
                            <span className="rounded-md bg-secondary px-1.5 py-0.5 text-[11px] font-medium text-tertiary">
                              In this chat
                            </span>
                          ) : null}
                        </div>
                        <h4 className="text-sm font-semibold text-primary">{d.title}</h4>
                        <p className="text-xs text-tertiary">Use case: {d.useCase}</p>
                        <p className="text-[11px] text-tertiary">
                          {d.author} · Updated on {d.updated}
                        </p>
                      </button>
                    );
                  })}
                </div>
                {totalPages > 1 ? (
                  <div className="mt-4 flex items-center justify-center gap-2 text-sm text-tertiary">
                    <button
                      type="button"
                      disabled={page <= 1}
                      onClick={() => setPage((p) => p - 1)}
                      className="rounded-md border border-secondary p-1.5 disabled:opacity-50"
                      aria-label="Previous page"
                    >
                      <ArrowLeft className="size-4" />
                    </button>
                    <span>
                      Page {page} of {totalPages}
                    </span>
                    <button
                      type="button"
                      disabled={page >= totalPages}
                      onClick={() => setPage((p) => p + 1)}
                      className="rounded-md border border-secondary p-1.5 disabled:opacity-50"
                      aria-label="Next page"
                    >
                      <ArrowRight className="size-4" />
                    </button>
                  </div>
                ) : null}
              </>
            )
          ) : (
            <div className="space-y-4">
              <input
                ref={inputRef}
                type="file"
                multiple
                accept={ACCEPT}
                className="hidden"
                onChange={(e) => {
                  addFiles(e.target.files);
                  e.target.value = "";
                }}
              />
              <button
                type="button"
                disabled={busy}
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  addFiles(e.dataTransfer.files);
                }}
                className={cx(
                  "flex w-full flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-10 text-center transition",
                  dragging ? "border-brand bg-secondary" : "border-secondary hover:border-tertiary",
                )}
              >
                <Upload01 className="size-6 text-fg-quaternary" />
                <p className="text-sm font-semibold text-primary">Click to upload or drag and drop</p>
                <p className="text-xs text-tertiary">CSV, Excel (.xlsx, .xls) or JSON — multiple files allowed</p>
              </button>
              {progress !== null ? (
                <div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
                    <div className="h-full bg-brand-solid transition-all" style={{ width: `${progress}%` }} />
                  </div>
                  <p className="mt-1 text-xs text-tertiary">Uploading… {progress}%</p>
                </div>
              ) : null}
              {files.length ? (
                <ul className="divide-y divide-secondary rounded-xl border border-secondary">
                  {files.map((f) => {
                    const st = statuses.find((s) => s.name === f.name);
                    return (
                      <li key={f.name} className="flex items-center gap-3 px-4 py-2.5 text-sm">
                        <span className="min-w-0 flex-1 truncate text-primary">{f.name}</span>
                        {st?.state === "ready" ? (
                          <span className="text-xs font-medium text-success-primary">
                            Ready{st.sheets && st.sheets.length > 1 ? ` · ${st.sheets.length} sheets` : ""}
                          </span>
                        ) : st?.state === "skipped" ? (
                          <span className="text-xs font-medium text-error-primary">Skipped by the server</span>
                        ) : st?.state === "pending" ? (
                          <Loader2 className="size-4 animate-spin text-fg-quaternary" />
                        ) : (
                          <button
                            type="button"
                            disabled={busy}
                            onClick={() => setFiles((prev) => prev.filter((x) => x !== f))}
                            className="text-fg-quaternary hover:text-secondary"
                            aria-label={`Remove ${f.name}`}
                          >
                            <X className="size-4" />
                          </button>
                        )}
                      </li>
                    );
                  })}
                </ul>
              ) : null}
            </div>
          )}
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-secondary px-5 py-3">
          <p className="text-xs text-error-primary">{overLimit ? LIMIT_MSG : ""}</p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={busy}
              onClick={() => onOpenChange(false)}
              className="rounded-lg border border-secondary bg-primary px-3 py-1.5 text-sm font-medium text-secondary shadow-xs hover:bg-secondary disabled:opacity-60"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={primaryDisabled}
              onClick={() => void (tab === "existing" ? submitExisting() : submitUpload())}
              className="inline-flex items-center gap-1.5 rounded-lg bg-brand-solid px-3 py-1.5 text-sm font-semibold text-white shadow-xs hover:bg-brand-solid_hover disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy ? <Loader2 className="size-4 animate-spin" /> : null}
              {tab === "existing" ? `Add to chat (${picked.length})` : "Upload and add"}
            </button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
