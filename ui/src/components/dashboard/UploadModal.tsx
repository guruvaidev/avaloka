import { useEffect, useRef, useState, type DragEvent } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { UploadCloud02, X, Play, CheckCircle, Loading01, File02, AlertTriangle } from "@untitledui/icons";
import { Dialog, DialogContent, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { analysesKey } from "@/lib/analyses";
import { createAnalysis as persistCreateAnalysis } from "@/lib/analysis-messages";
import { projectDashboardKey } from "@/lib/api/project-dashboard";
import { backendApi } from "@/lib/api/backendApi";
// import { setPendingDashboardAnalysis } from "@/lib/pending-dashboard-analysis";
import { cx } from "@/lib/utils/cx";
import { finishAutoInsightsBot, startAutoInsightsBot } from "@/lib/auto-insights-bot";
import { toast } from "sonner";

const ACCEPTED = [".csv", ".xlsx", ".xls", ".json",".parquet"];
const MAX_BYTES = 100 * 1024 * 1024;

type Status = "idle" | "selected" | "uploading" | "complete";

interface UploadResponse {
  dataset_id: string;
  session_id: string;
  thread_id: string;
  schema: string[];
  samples: unknown[];
  rows_sampled: number;
  visualization_config?: unknown;
  visualization_status?: string;
  [k: string]: unknown;
}

function validate(file: File): string | null {
  const ext = "." + (file.name.split(".").pop() ?? "").toLowerCase();
  if (!ACCEPTED.includes(ext)) return "File type not supported. Use CSV, Excel or JSON.";
  if (file.size > MAX_BYTES) return "File too large. Maximum 100MB.";
  return null;
}

function mapError(err: unknown): string {
  if (err instanceof TypeError) return "Connection failed. Check your internet and try again.";
  const e = err as { status?: number; message?: string };
  if (e?.status === 415) return "File type not supported. Use CSV, Excel or JSON.";
  if (e?.status === 413) return "File too large. Maximum 100MB.";
  return e?.message || "Upload failed. Please try again.";
}

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function fileExtBadge(name: string) {
  const ext = (name.split(".").pop() ?? "").toLowerCase();
  if (["xls", "xlsx"].includes(ext)) return { label: "XLS", color: "bg-[#079455]" };
  if (ext === "csv") return { label: "CSV", color: "bg-[#079455]" };
  if (ext === "json") return { label: "JSON", color: "bg-[#1565ef]" };
  if (ext === "pdf") return { label: "PDF", color: "bg-[#b42318]" };
  return { label: ext.toUpperCase() || "FILE", color: "bg-[#3f3f46]" };
}

const METRIC_PALETTE = [
  { bg: "bg-[#eff5ff]", text: "text-[#175cd3]", border: "border-[#b2ceff]" },
  { bg: "bg-[#fef3f2]", text: "text-[#b42318]", border: "border-[#fecdca]" },
  { bg: "bg-[#eef4ff]", text: "text-[#3538cd]", border: "border-[#c7d7fe]" },
  { bg: "bg-[#ecfdf3]", text: "text-[#067647]", border: "border-[#abefc6]" },
  { bg: "bg-[#f4f4f5]", text: "text-[#3f3f46]", border: "border-[#e4e4e7]" },
];

interface UploadModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId?: string | null;
  projectName?: string | null;
  redirectTo?: "analysis" | "preview";
  /** When set, creates a child analysis under this mother. */
  parentAnalysisId?: string | null;
  /** Keep ?aid= on the mother when creating a child. */
  motherAnalysisId?: string | null;
  /** File already chosen outside the modal (e.g. dropped on the dropzone). */
  initialFile?: File | null;
  /** Multiple files already chosen outside the modal. */
  initialFiles?: File[] | null;
  /** Called when upload completes without a project (dashboard flow). */
  // onStandaloneAnalysisSaved?: (analysisId: string) => void;
}

type NormalizedDataset = {
  dataset_id: string;
  filename?: string;
  alias?: string;
  schema: unknown;
  samples: unknown[];
  rows_sampled: number;
  visualization_config?: unknown;
  visualization_status?: string;
  [k: string]: unknown;
};

const fileKey = (f: File) => `${f.name}:${f.size}`;

/**
 * Soft-failure envelope check. The upstream may answer 200 with
 * `{ fallback: true }` or `{ error: "UPSTREAM_TIMEOUT" }` instead of real data.
 * Such a payload must never be treated as a dataset (it would render as
 * "0 rows / 0 columns" and wipe a previously good insight card).
 */
function softFailure(obj: any): string | null {
  if (!obj || typeof obj !== "object") return null;
  const err = obj.error ?? obj.error_code ?? obj.errorCode;
  if (err) return String(obj.message ?? obj.detail ?? err);
  if (obj.fallback === true) return String(obj.message ?? obj.detail ?? "Upstream unavailable — please retry");
  return null;
}

/** Map a MultiUploadResponse dataset item to the shape the analysis page reads. */
function normalizeDataset(d: any): NormalizedDataset {
  const samples = Array.isArray(d?.rows) ? d.rows : Array.isArray(d?.samples) ? d.samples : [];
  return {
    ...d,
    dataset_id: d?.dataset_id,
    filename: d?.filename,
    alias: d?.alias,
    schema: d?.columns ?? d?.schema ?? null,
    samples,
    rows_sampled: d?.rows_sampled ?? samples.length,
    visualization_config: d?.visualization_config ?? d?.visualization_configs ?? null,
    visualization_status: d?.visualization_status,
  };
}

export function UploadModal({
  open,
  onOpenChange,
  projectId,
  projectName,
  redirectTo = "analysis",
  parentAnalysisId,
  motherAnalysisId,
  initialFile,
  initialFiles,
  // onStandaloneAnalysisSaved,
}: UploadModalProps) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [batch, setBatch] = useState<{ session_id: string; thread_id: string; datasets: NormalizedDataset[] } | null>(
    null,
  );
  const [fileErrors, setFileErrors] = useState<Record<string, string>>({});
  const [batchError, setBatchError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const attemptedRef = useRef<string>("");

  /** Dataset returned for a given selected file (matched by filename, else by order). */
  const datasetFor = (f: File, index: number): NormalizedDataset | null => {
    if (!batch) return null;
    const byName = batch.datasets.find((d) => d.filename === f.name || d.alias === f.name);
    if (byName) return byName;
    if (batch.datasets.length === selectedFiles.length) return batch.datasets[index] ?? null;
    // If a single file produced multiple datasets (e.g., Excel with multiple sheets),
    // consider the file completed and associate the first dataset for UI completion state.
    if (selectedFiles.length === 1 && batch.datasets.length > 0) return batch.datasets[index] ?? batch.datasets[0] ?? null;
    return null;
  };

  const completedFiles = selectedFiles.filter((f, i) => !!datasetFor(f, i));
  const failedFiles = batch || batchError ? selectedFiles.filter((f, i) => !datasetFor(f, i)) : [];
  const datasets = batch?.datasets ?? [];
  const file = completedFiles[0] ?? selectedFiles[0] ?? null;
  const primary = datasets[0] ?? null;
  const status: Status = uploading
    ? "uploading"
    : selectedFiles.length === 0
      ? "idle"
      : primary
        ? "complete"
        : "selected";
  const response = primary;

  const errorFor = (f: File, index: number): string | null => {
    if (datasetFor(f, index)) return null;
    if (fileErrors[f.name]) return fileErrors[f.name];
    if (batchError) return batchError;
    // Only mark as skipped when the server response has no datasets.
    if (batch && (!batch.datasets || batch.datasets.length === 0)) return "Skipped by the server";
    return null;
  };

  const uploadAll = async (files: File[]) => {
    if (!files.length) return;
    setUploading(true);
    setProgress(8);
    setBatchError(null);
    setFileErrors({});
    const timer = setInterval(() => setProgress((p) => (p < 90 ? p + Math.max(1, (90 - p) / 12) : p)), 300);
    try {
      const res = (await backendApi.uploadFiles(files)) as any;

      // Whole-response soft failure: keep whatever good insights we already have.
      const envelope = softFailure(res);
      if (envelope) {
        setBatchError(envelope);
        toast.error(envelope);
        setProgress(0);
        return;
      }

      const errs: Record<string, string> = {};
      const rawErrors = Array.isArray(res?.errors) ? res.errors : Array.isArray(res?.failed) ? res.failed : [];
      for (const e of rawErrors) {
        const name = e?.filename ?? e?.file ?? e?.name;
        if (name) errs[String(name)] = String(e?.error ?? e?.detail ?? e?.message ?? "Upload failed");
      }

      const good: NormalizedDataset[] = [];
      for (const d of Array.isArray(res?.datasets) ? res.datasets : []) {
        const perFile = softFailure(d);
        const name = d?.filename ?? d?.alias;
        if (perFile || typeof d?.dataset_id !== "string" || !d.dataset_id) {
          if (name) errs[String(name)] = perFile ?? "Upload failed";
          continue;
        }
        good.push(normalizeDataset(d));
      }
      setFileErrors(errs);

      // Merge by dataset_id so a partial retry never drops earlier good cards.
      setBatch((prev) => {
        const byId = new Map<string, NormalizedDataset>();
        for (const d of prev?.datasets ?? []) byId.set(d.dataset_id, d);
        for (const d of good) byId.set(d.dataset_id, d);
        return {
          session_id: res?.session_id ?? prev?.session_id,
          thread_id: res?.thread_id ?? prev?.thread_id,
          datasets: Array.from(byId.values()),
        };
      });
      setProgress(100);
    } catch (err) {
      const msg = mapError(err);
      setBatchError(msg);
      toast.error(msg);
      setProgress(0);

    } finally {
      clearInterval(timer);
      setUploading(false);
    }
  };

  const addFiles = (incoming: File[]) => {
    if (!incoming.length) return;
    setSelectedFiles((prev) => {
      const seen = new Set(prev.map(fileKey));
      const accepted: File[] = [];
      for (const f of incoming) {
        const v = validate(f);
        if (v) {
          toast.error(`${f.name}: ${v}`);
          continue;
        }
        const key = fileKey(f);
        if (seen.has(key)) continue;
        seen.add(key);
        accepted.push(f);
      }
      return accepted.length ? [...prev, ...accepted] : prev;
    });
  };

  // One request for the whole selection: re-runs whenever the selection changes.
  useEffect(() => {
    if (!open || uploading) return;
    const sig = selectedFiles.map(fileKey).join("|");
    if (!sig || sig === attemptedRef.current) return;
    attemptedRef.current = sig;
    void uploadAll(selectedFiles);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, selectedFiles]);

  const retryAll = () => {
    attemptedRef.current = "";
    setBatchError(null);
    setFileErrors({});
  };

  useEffect(() => {
    if (!open) return;
    const list = initialFiles?.length ? initialFiles : initialFile ? [initialFile] : [];
    if (list.length) addFiles(list);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, initialFile, initialFiles]);

  const reset = () => {
    setSelectedFiles([]);
    setBatch(null);
    setFileErrors({});
    setBatchError(null);
    setProgress(0);
    attemptedRef.current = "";
  };

  const removeFile = (f: File) => {
    if (uploading) return;
    const key = fileKey(f);
    setSelectedFiles((prev) => prev.filter((x) => fileKey(x) !== key));
  };

  const handleClose = (next: boolean) => {
    if (uploading) return;
    if (!next) reset();
    onOpenChange(next);
  };

  const openPicker = () => {
    if (uploading) return;
    if (inputRef.current) inputRef.current.value = "";
    inputRef.current?.click();
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    if (uploading) return;
    addFiles(Array.from(e.dataTransfer.files ?? []));
  };




  const start = async () => {
    if (!file || !response || !batch) return;
    // Close instantly so the dataset details are not visible while we persist.
    startAutoInsightsBot();
    onOpenChange(false);
    const vizConfig = response.visualization_config ?? null;
    const primaryName = response.filename ?? response.alias ?? file.name;
    try {
      sessionStorage.setItem(
        "analysis:dataset",
        JSON.stringify({
          filename: primaryName,
          schema: response.schema,
          samples: response.samples,
          rows_sampled: response.rows_sampled,
          visualization_config: vizConfig,
          visualization_status: response.visualization_status,
        }),
      );
      sessionStorage.setItem(
        "analysis:session",
        JSON.stringify({
          threadId: batch.thread_id,
          sessionId: batch.session_id,
          datasetChips: datasets.map((d) => ({
            id: d.dataset_id,
            name: d.alias || d.filename || d.dataset_id,
          })),
        }),
      );
    } catch {}
    const name = primaryName.replace(/\.[^.]+$/, "");
    const contextPatch = {
      dataset_id: response.dataset_id ?? null,
      thread_id: batch.thread_id ?? null,
      session_id: batch.session_id ?? null,
      filename: primaryName,
      schema: (response.schema ?? null) as unknown,
      samples: (response.samples ?? null) as unknown,
      viz_config: vizConfig,
    };


    let newAid: string | null = null;
    const resolvedProjectId = redirectTo === "preview" ? null : (projectId ?? null);
    const isChildUpload = !!parentAnalysisId;
    try {
      const created = await persistCreateAnalysis({
        project_id: resolvedProjectId,
        parent_analysis_id: parentAnalysisId ?? null,
        name,
        ...contextPatch,
      });
      newAid = created.id;
      // Persist the remaining files of a multi-file upload as child rows so the
      // group's datasets[] is recoverable server-side (Recent sidebar file list).
      const groupParentId = isChildUpload ? (motherAnalysisId ?? parentAnalysisId ?? newAid) : newAid;
      const extras = datasets.filter((d) => d.dataset_id && d.dataset_id !== response.dataset_id);
      if (groupParentId && extras.length > 0) {
        await Promise.all(
          extras.map((d) =>
            persistCreateAnalysis({
              project_id: resolvedProjectId,
              parent_analysis_id: groupParentId,
              name: (d.alias || d.filename || d.dataset_id).replace(/\.[^.]+$/, ""),
              dataset_id: d.dataset_id,
              thread_id: batch.thread_id ?? null,
              session_id: batch.session_id ?? null,
              filename: d.filename ?? d.alias ?? null,
              schema: (d as any).schema ?? null,
              samples: (d as any).samples ?? null,
              viz_config: (d as any).visualization_config ?? null,
            }).catch((e) => {
              console.warn("Failed to persist child dataset", e);
              return null;
            }),
          ),
        );
      }
      if (!isChildUpload) {
        await queryClient.invalidateQueries({ queryKey: ["analyses"] });
        await queryClient.invalidateQueries({ queryKey: analysesKey(resolvedProjectId) });
      }
      await queryClient.invalidateQueries({ queryKey: ["analysis-datasets"] });
      if (newAid) {
        await queryClient.invalidateQueries({ queryKey: projectDashboardKey(newAid) });
      }

      try {
        sessionStorage.setItem(
          "analysis:context",
          JSON.stringify({
            name,
            project: projectName ?? "",
            projectId: resolvedProjectId,
            analysisId: isChildUpload ? (motherAnalysisId ?? newAid) : newAid,
          }),
        );
      } catch {}
    } catch (err) {
      console.warn("Failed to persist uploaded analysis", err);
    }

    onOpenChange(false);



    const urlAid = isChildUpload ? (motherAnalysisId ?? parentAnalysisId) : newAid;
    navigate({
      to: "/analysis",
      search:
        urlAid != null
          ? ({
              aid: urlAid,
              ...(isChildUpload && newAid ? { child: newAid } : {}),
            } as never)
          : undefined,
      state: {
        uploadResponse: {
          ...response,
          session_id: batch.session_id,
          thread_id: batch.thread_id,
          datasets,
        },
        filename: primaryName,
        childAnalysisId: newAid,
        newAnalysisId: urlAid,
        showAutoInsights: true,
      } as never,
    });
    reset();
  };

  const ext = file ? fileExtBadge(file.name) : null;
  const showDetails = status === "complete" || status === "uploading";
  const canStart = !uploading && completedFiles.length > 0;
  const totalBytes = selectedFiles.reduce((sum, f) => sum + f.size, 0);
  const summaryLine =
    !uploading && failedFiles.length > 0
      ? `${completedFiles.length} of ${selectedFiles.length} files ready · ` +
        failedFiles
          .map((f, i) => `${f.name} skipped (${errorFor(f, i) ?? "upload failed"})`)
          .join(" · ")
      : null;





  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-h-[90vh] max-w-[560px] gap-0 overflow-y-auto rounded-2xl border border-secondary bg-primary p-6 shadow-2xl [&>button]:hidden">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-[10px] border border-secondary bg-primary shadow-xs">
              <UploadCloud02 className="size-5 text-fg-secondary" />
            </div>
            <DialogTitle className="text-md font-semibold text-primary">Select File &amp; Start Analysis</DialogTitle>
          </div>
          <button
            type="button"
            onClick={() => handleClose(false)}
            aria-label="Close"
            className="rounded-md p-1 text-fg-quaternary outline-focus-ring hover:bg-primary_hover"
          >
            <X className="size-5" />
          </button>
        </div>
        <DialogDescription className="sr-only">
          Upload a CSV, Excel, or JSON file to start an analysis.
        </DialogDescription>

        {/* Dropzone */}
        <div
          onClick={uploading ? undefined : openPicker}
          onDragOver={(e) => {
            e.preventDefault();
            if (!uploading) setDragActive(true);
          }}
          onDragEnter={(e) => {
            e.preventDefault();
            if (!uploading) setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={onDrop}
          className={cx(
            "relative mt-5 rounded-xl border bg-primary px-6 py-5 text-center transition",
            showDetails ? "border-[#1565ef]" : dragActive ? "border-brand" : "border-secondary",
            !uploading && "cursor-pointer hover:border-brand",
          )}
        >

          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPTED.join(",")}
            className="hidden"
            onChange={(e) => addFiles(Array.from(e.target.files ?? []))}
          />

          <div className="flex flex-col items-center">
            <div className="flex size-10 items-center justify-center rounded-[10px] border border-secondary bg-primary shadow-xs">
              <UploadCloud02 className="size-5 text-fg-secondary" />
            </div>
            <p className="mt-3 text-sm">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  openPicker();
                }}
                className="font-semibold text-[#1565ef] hover:underline"
              >
                Select
              </button>{" "}
              <span className="text-tertiary">or drag and drop file</span>
            </p>
            <p className="mt-1 text-xs text-tertiary">CSV, Excel or JSON (max. 100MB each)</p>
          </div>

          {/* Floating file badge (top-right inside dropzone) when uploaded */}
          {showDetails && ext && (
            <div className="pointer-events-none absolute bottom-3 right-4 flex items-end gap-1">
              <div className="relative flex h-12 w-10 items-end justify-center rounded-md border border-secondary bg-white pb-1 shadow-xs">
                <span className={cx("rounded-sm px-1 py-0.5 text-[9px] font-bold text-white", ext.color)}>
                  {ext.label}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Selected files */}
        {selectedFiles.length > 0 && (
          <div className="mt-3">
            <div className="flex items-center justify-between px-1 pb-2">
              <p className="text-xs font-medium text-tertiary">
                {selectedFiles.length} file{selectedFiles.length > 1 ? "s" : ""} selected ·{" "}
                {formatSize(totalBytes)}
              </p>
              {uploading && (
                <span className="inline-flex items-center gap-1 text-xs text-tertiary">
                  <Loading01 className="size-3 animate-spin" /> Uploading…
                </span>
              )}
            </div>
            {/* Single aggregate progress bar — the whole selection goes in one request. */}
            {(uploading || progress > 0) && (
              <div className="mb-3 flex items-center gap-2">
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[#e4e4e7]">
                  <div
                    className="h-full rounded-full bg-[#1565ef] transition-all duration-300"
                    style={{ width: `${Math.min(100, Math.round(progress))}%` }}
                  />
                </div>
                <span className="w-10 text-right text-xs font-medium text-tertiary">
                  {Math.min(100, Math.round(progress))}%
                </span>
              </div>
            )}
            <div className="max-h-56 space-y-2 overflow-y-auto pr-1">
              {selectedFiles.map((f, i) => {
                const key = fileKey(f);
                const ds = datasetFor(f, i);
                const err = errorFor(f, i);
                const b = fileExtBadge(f.name);
                return (
                  <div key={key} className="rounded-xl border border-secondary bg-primary p-3">
                    <div className="flex items-start gap-3">
                      <div className="relative flex size-10 shrink-0 items-end justify-center rounded-md border border-secondary bg-white pb-1">
                        <span className={cx("rounded-sm px-1 py-0.5 text-[9px] font-bold text-white", b.color)}>
                          {b.label}
                        </span>
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-start justify-between gap-2">
                          <p className="truncate text-sm font-semibold text-primary" title={f.name}>
                            {f.name}
                          </p>
                          <button
                            type="button"
                            onClick={() => removeFile(f)}
                            disabled={uploading}
                            aria-label={`Remove ${f.name}`}
                            className="text-fg-quaternary hover:text-fg-secondary disabled:opacity-40"
                          >
                            <X className="size-4" />
                          </button>
                        </div>
                        <div className="mt-0.5 flex items-center gap-2 text-xs text-tertiary">
                          <span>{formatSize(f.size)}</span>
                          <span className="text-fg-quaternary">|</span>
                          {uploading ? (
                            <span className="inline-flex items-center gap-1 text-tertiary">
                              <Loading01 className="size-3 animate-spin" /> Uploading…
                            </span>
                          ) : ds ? (
                            <span className="inline-flex items-center gap-1 font-medium text-[#067647]">
                              <CheckCircle className="size-3" /> Complete
                            </span>
                          ) : err ? (
                            <span className="inline-flex items-center gap-1 font-medium text-error-primary">
                              <AlertTriangle className="size-3" /> {err}
                              <button
                                type="button"
                                onClick={retryAll}
                                className="ml-1 font-semibold text-[#1565ef] hover:underline"
                              >
                                Retry
                              </button>
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1">
                              <File02 className="size-3" /> Ready
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}



        {/* AI Insights — one card per dataset, bound to its own dataset_id */}
        {status === "complete" && datasets.length > 0 && (
          <>
            <div className="mt-5 flex items-center gap-3">
              <div className="h-px flex-1 bg-border-secondary" />
              <span className="text-xs font-medium text-tertiary">Ai Insights</span>
              <div className="h-px flex-1 bg-border-secondary" />
            </div>

            {datasets.map((d) => (
              <DatasetInsightCard key={d.dataset_id} dataset={d} />
            ))}
          </>
        )}


        {/* Security note */}
        <div className="mt-4 rounded-lg border border-[#abefc6] bg-[#ecfdf3] px-3 py-3">
          <p className="text-xs font-medium text-[#067647]">
            avaloka does not capture your data or train on it beyond customization for your needs. Your data remains
            safe with your account
          </p>
        </div>

        {summaryLine && (
          <p className="mt-3 text-sm text-tertiary">
            {summaryLine}
          </p>
        )}

        {/* Actions */}
        <div className="mt-5 grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => handleClose(false)}
            disabled={status === "uploading"}
            className="inline-flex items-center justify-center gap-2 rounded-lg border border-secondary bg-primary px-4 py-2.5 text-sm font-semibold text-primary shadow-xs hover:bg-primary_hover disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Play className="size-4" />
            Skip
          </button>
          <button
            type="button"
            onClick={start}
            disabled={!canStart}
            className={cx(
              "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold shadow-xs transition",
              canStart
                ? "bg-[#1565ef] text-white hover:bg-[#1257d6]"
                : "cursor-not-allowed bg-[#f4f4f5] text-[#a0a0ab]",
            )}
          >
            <CheckCircle className="size-4" />
            Clean &amp; Start
          </button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/** Insights for exactly one dataset — heading, rows/columns and metrics all
 *  come from the same `dataset_id`, so headings can never drift from schemas. */
function DatasetInsightCard({ dataset }: { dataset: NormalizedDataset }) {
  const viz = (dataset as any)?.visualization_config ?? null;
  const vizCols: string[] | undefined = Array.isArray(viz?.columns)
    ? viz.columns.map((c: any) => (typeof c === "string" ? c : c?.name)).filter(Boolean)
    : undefined;
  const vizDatasetCols: string[] | undefined = Array.isArray(viz?.dataset?.columns) ? viz.dataset.columns : undefined;
  const schemaCols: string[] = Array.isArray(dataset.schema)
    ? (dataset.schema as any[]).map((c) => (typeof c === "string" ? c : c?.name)).filter(Boolean)
    : dataset.schema && typeof dataset.schema === "object"
      ? Object.keys(dataset.schema as Record<string, unknown>)
      : [];
  const columnList: string[] = vizCols ?? vizDatasetCols ?? schemaCols;
  const metrics = columnList.slice(0, 4);
  const rowCount = (viz?.dataset?.rows_sampled as number | undefined) ?? dataset.rows_sampled ?? 0;
  const title = (dataset.alias || dataset.filename || dataset.dataset_id).replace(/\.[^.]+$/, "");

  return (
    <div className="mt-4">
      <div className="flex items-center gap-2">
        <h3 className="truncate text-sm font-semibold text-primary" title={title}>
          {title}
        </h3>
        <span className="shrink-0 rounded-md border border-[#b2ceff] bg-[#eff5ff] px-2 py-0.5 text-xs font-medium text-[#175cd3]">
          Dataset
        </span>
      </div>

      <div className="mt-3 rounded-lg border border-secondary bg-primary px-3 py-2.5">
        <p className="text-xs text-secondary">
          {rowCount.toLocaleString()} rows detected across {columnList.length} columns.
        </p>
      </div>

      {metrics.length > 0 && (
        <div className="mt-4">
          <p className="text-sm font-semibold text-primary">Key Metrics Identified</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {metrics.map((m, i) => {
              const c = METRIC_PALETTE[i % METRIC_PALETTE.length];
              return (
                <span
                  key={m}
                  className={cx("rounded-md border px-2 py-0.5 text-xs font-medium", c.bg, c.text, c.border)}
                >
                  {m}
                </span>
              );
            })}
          </div>
        </div>
      )}

      <div className="mt-4">
        <p className="text-sm font-semibold text-primary">Dataset Issues</p>
        <div className="mt-2 space-y-2">
          <IssueRow index="01" label="Duplicate Data Found" />
          <IssueRow index="02" label="Empty cells found" />
        </div>
      </div>
    </div>
  );
}

function IssueRow({ index, label }: { index: string; label: string }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-secondary bg-primary px-3 py-2.5">
      <div className="flex items-center gap-3">
        <span className="rounded-md border border-secondary px-2 py-0.5 text-xs font-medium text-tertiary">
          {index}
        </span>
        <span className="text-sm text-primary">{label}</span>
      </div>
      <div className="flex size-7 items-center justify-center rounded-full bg-[#fef0c7]">
        <AlertTriangle className="size-4 text-[#dc6803]" />
      </div>
    </div>
  );
}
