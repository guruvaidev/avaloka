import { useEffect, useRef, useState, type DragEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { UploadCloud02, X, Play, CheckCircle, Loading01, File02, AlertTriangle, Diamond01 } from "@untitledui/icons";
import { Dialog, DialogContent, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/base/buttons/button";
import { analysesKey } from "@/lib/analyses";
import { createAnalysis as persistCreateAnalysis } from "@/lib/analysis-messages";
import { projectDashboardKey } from "@/lib/api/project-dashboard";
import { backendApi, type UploadLimits } from "@/lib/api/backendApi";
import { setPendingDashboardAnalysis } from "@/lib/pending-dashboard-analysis";
import { cx } from "@/lib/utils/cx";
import { finishAutoInsightsBot, startAutoInsightsBot } from "@/lib/auto-insights-bot";
import { toast } from "sonner";
import { canReviewUpload, canStartSelection, initialSelection, isSkippedSheetNote, readableDatasetName, realColumnNames, reviewDatasetName, reviewIssues, selectionPayload, uploadKind, CLEANING_TOOLTIP } from "@/lib/upload-review";
import { CleaningRules, CompactDatasetRow, DatasetReviewIssues, EditableDatasetName, FileStatusBar, ReviewTooltip, UploadSummaryCard } from "@/components/dashboard/upload/UploadReviewPanels";
import { Button as ReviewButton } from "@/components/ui/button";
import { ConnectCloudModal } from "@/components/database/ConnectCloudModal";

const ACCEPTED = [".csv", ".xlsx", ".xls", ".json"];
/** Fallback limits while /api/limits is loading or unavailable. */
const FALLBACK_MAX_BYTES = 100 * 1024 * 1024;

type Status = "idle" | "selected" | "uploading" | "complete";

type LimitDialog = {
  kind: "upgrade" | "cloud";
  title: string;
  message: string;
  showUpgrade: boolean;
};

const PLAN_LABEL: Record<string, string> = {
  free: "Free",
  professional: "Professional",
  enterprise: "Enterprise",
};
const planLabel = (plan?: string | null) => PLAN_LABEL[String(plan ?? "free").toLowerCase()] ?? "Free";
const toMb = (bytes: number) => Math.round(bytes / (1024 * 1024));

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

/** File TYPE check only — size is decided by checkLimits() from the plan limits. */
function validate(file: File): string | null {
  const ext = "." + (file.name.split(".").pop() ?? "").toLowerCase();
  if (!ACCEPTED.includes(ext)) return "File type not supported. Use CSV, Excel or JSON.";
  return null;
}

function mapError(err: unknown): string {
  if (err instanceof TypeError) return "Connection failed. Check your internet and try again.";
  const e = err as { status?: number; message?: string; detail?: any };
  // Never replace a server-supplied message with generic text.
  if (typeof e?.detail?.message === "string" && e.detail.message) return e.detail.message;
  if (typeof e?.detail === "string" && e.detail) return e.detail;
  if (e?.status === 415) return "File type not supported. Use CSV, Excel or JSON.";
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
  onStandaloneAnalysisSaved?: (analysisId: string) => void;
}

type NormalizedDataset = {
  dataset_id: string;
  filename?: string;
  alias?: string;
  schema: unknown;
  samples: unknown[];
  rows_sampled: number | null;
  total_rows?: number | null;
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
    rows_sampled: d?.visualization_config?.dataset?.rows_sampled ?? d?.sample_statistics?.sample_size ?? null,
    total_rows: d?.total_rows ?? d?.visualization_config?.dataset?.total_rows ?? null,
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
  onStandaloneAnalysisSaved,
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
  const [limitExceeded, setLimitExceeded] = useState<LimitDialog | null>(null);
  const [cloudOpen, setCloudOpen] = useState(false);
  const [uploadPhase, setUploadPhase] = useState<"uploading" | "processing">("uploading");
  const attemptedRef = useRef<string>("");
  const requestGeneration = useRef(0);
  const replaceRef = useRef<File | null>(null);
  const [step, setStep] = useState<1 | 2>(1);
  const [datasetNames, setDatasetNames] = useState<Record<string, string>>({});
  /** Several-files uploads: names of the files the user ticked (nothing pre-selected). */
  const [pickedFiles, setPickedFiles] = useState<string[]>([]);
  const [openRow, setOpenRow] = useState<string | null>(null);

  // Plan upload limits — fetched once when the modal opens and cached.
  const { data: limits } = useQuery<UploadLimits>({
    queryKey: ["upload-limits"],
    queryFn: () => backendApi.getUploadLimits(),
    enabled: open,
    staleTime: 5 * 60 * 1000,
    retry: false,
  });

  /** Effective limits — 100 MB per file / 100 MB total until /api/limits answers. */
  const effLimits = {
    plan: (limits?.plan as string | undefined) ?? "free",
    maxFileBytes: typeof limits?.max_file_bytes === "number" ? limits.max_file_bytes : FALLBACK_MAX_BYTES,
    maxTotalBytes: typeof limits?.max_total_bytes === "number" ? limits.max_total_bytes : FALLBACK_MAX_BYTES,
    maxFiles: typeof limits?.max_files === "number" ? limits.max_files : null,
    batchJobs: limits?.batch_jobs === true,
  };
  const maxFileMb =
    typeof limits?.max_file_mb === "number" ? limits.max_file_mb : toMb(effLimits.maxFileBytes);
  const maxTotalMb =
    typeof limits?.max_total_mb === "number" ? limits.max_total_mb : toMb(effLimits.maxTotalBytes);

  /** Pre-upload plan limit check. Returns the dialog to show, or null. */
  const checkLimits = (files: File[]): LimitDialog | null => {
    const { plan, maxFileBytes, maxTotalBytes, maxFiles, batchJobs } = effLimits;
    const label = planLabel(plan);
    const isFree = String(plan).toLowerCase() === "free" || !batchJobs;
    const upgradeDialog = (message: string): LimitDialog => ({
      kind: "upgrade",
      title: `${label} plan limit reached`,
      message,
      showUpgrade: isFree,
    });
    const cloudDialog: LimitDialog = {
      kind: "cloud",
      title: "Load large files from cloud storage",
      message: `Files larger than ${maxFileMb} MB are loaded from your cloud storage, where they run as background jobs and you'll be notified when the analysis is ready.`,
      showUpgrade: false,
    };

    if (typeof maxFiles === "number" && files.length > maxFiles) {
      return upgradeDialog(
        `You selected ${files.length} files, but the ${label} plan allows up to ${maxFiles} file${maxFiles > 1 ? "s" : ""} per upload.`,
      );
    }
    const tooBig = files.filter((f) => f.size > maxFileBytes);
    if (tooBig.length) {
      if (!isFree) return cloudDialog;
      return upgradeDialog(
        `${tooBig.map((f) => f.name).join(", ")} exceed${tooBig.length > 1 ? "" : "s"} the per-file limit of ${maxFileMb} MB on the ${label} plan.`,
      );
    }
    const total = files.reduce((sum, f) => sum + f.size, 0);
    if (total > maxTotalBytes) {
      if (!isFree) return cloudDialog;
      return upgradeDialog(
        `Your selection totals ${formatSize(total)}, which exceeds the ${label} plan limit of ${maxTotalMb} MB of data per analysis.`,
      );
    }
    return null;
  };

  /**
   * Dataset returned for a given selected file. Excel workbooks come back as
   * one dataset per sheet ("name [sheet].csv"), so also match on
   * `source_workbook` (the original uploaded file name).
   */
  const datasetFor = (f: File, index: number): NormalizedDataset | null => {
    if (!batch) return null;
    const byName = batch.datasets.find(
      (d) => d.filename === f.name || d.alias === f.name || (d as any).source_workbook === f.name,
    );
    if (byName) return byName;
    if (batch.datasets.length === selectedFiles.length) return batch.datasets[index] ?? null;
    return null;
  };

  /** All sheet datasets belonging to an uploaded workbook, in response order. */
  const sheetsFor = (f: File): NormalizedDataset[] =>
    batch?.datasets.filter((d) => (d as any).source_workbook === f.name) ?? [];

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

  // Upload kind decides the selection: one workbook → all sheets; several files → user picks.
  const kind = uploadKind(datasets as any[], selectedFiles.length);
  const datasetsOfFile = (f: File, index: number): NormalizedDataset[] => {
    const sheets = sheetsFor(f);
    if (sheets.length) return sheets;
    const one = datasetFor(f, index);
    return one ? [one] : [];
  };
  const selectedIds =
    kind === "files"
      ? selectedFiles.flatMap((f, i) => (pickedFiles.includes(f.name) ? datasetsOfFile(f, i).map((d) => d.dataset_id) : []))
      : initialSelection(kind, datasets.map((d) => d.dataset_id));

  const errorFor = (f: File, index: number): string | null => {
    if (datasetFor(f, index)) return null;
    if (fileErrors[f.name]) return fileErrors[f.name];
    if (batchError) return batchError;
    if (batch) return "Skipped by the server";
    return null;
  };

  const uploadAll = async (files: File[]) => {
    if (!files.length) return;
    const generation = ++requestGeneration.current;
    setUploading(true);
    setStep(1);
    setPickedFiles([]);
    setOpenRow(null);
    setProgress(0);
    setUploadPhase("uploading");
    setBatchError(null);
    setFileErrors({});
    try {
      // Real byte-level progress from the upload request (bytes sent / total).
      const res = (await backendApi.uploadFiles(files, undefined, undefined, (p) => {
        if (generation !== requestGeneration.current) return;
        setUploadPhase(p.phase);
        setProgress(p.total > 0 ? Math.min(100, (p.loaded / p.total) * 100) : 0);
      })) as any;

      if (generation !== requestGeneration.current) return;
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
        const incomingNames = new Set(good.map((d) => d.source_workbook ?? d.filename));
        for (const d of prev?.datasets ?? []) {
          if (!incomingNames.has(d.source_workbook ?? d.filename)) byId.set(d.dataset_id, d);
        }
        for (const d of good) byId.set(d.dataset_id, d);
        return {
          session_id: res?.session_id ?? prev?.session_id,
          thread_id: res?.thread_id ?? prev?.thread_id,
          datasets: Array.from(byId.values()),
        };
      });
      setProgress(100);
    } catch (err) {
      if (generation !== requestGeneration.current) return;
      const e = err as { status?: number; detail?: unknown };
      if (e?.status === 413) {
        // Plan limit rejected by the server — show detail.message as-is.
        const detail = e.detail as
          | { reason?: string; message?: string; plan?: string; upgrade_available?: boolean }
          | string
          | undefined;
        const obj = typeof detail === "object" && detail ? detail : null;
        const reason = String(obj?.reason ?? "limit_exceeded");
        const label = planLabel(obj?.plan ?? effLimits.plan);
        const fallbackMessages: Record<string, string> = {
          file_count: `You selected more files than the ${label} plan allows per upload.`,
          max_files: `You selected more files than the ${label} plan allows per upload.`,
          file_size: `One or more files exceed the per-file size limit of the ${label} plan.`,
          max_file_bytes: `One or more files exceed the per-file size limit of the ${label} plan.`,
          total_size: `The total size of your selection exceeds the ${label} plan limit.`,
          max_total_bytes: `The total size of your selection exceeds the ${label} plan limit.`,
        };
        const message =
          typeof detail === "string" && detail
            ? detail
            : typeof obj?.message === "string" && obj.message
              ? obj.message
              : fallbackMessages[reason] ?? `The file(s) exceed the ${label} plan limit.`;
        setLimitExceeded({
          kind: "upgrade",
          title: `${label} plan limit reached`,
          message,
          showUpgrade: obj?.upgrade_available === true,
        });
        setProgress(0);
        return;
      }
      const msg = mapError(err);
      setBatchError(msg);
      toast.error(msg);
      setProgress(0);

    } finally {
      if (generation === requestGeneration.current) {
        setUploading(false);
        setUploadPhase("uploading");
      }
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
      if (!accepted.length) return prev;
      const next = [...prev, ...accepted];
      // Plan limit gate: block before any upload happens.
      const violation = checkLimits(next);
      if (violation) {
        setLimitExceeded(violation);
        return prev;
      }
      return next;
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
    attemptedRef.current = selectedFiles.map(fileKey).join("|");
    void uploadAll(selectedFiles);
  };

  useEffect(() => {
    if (!open) return;
    const list = initialFiles?.length ? initialFiles : initialFile ? [initialFile] : [];
    if (list.length) addFiles(list);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, initialFile, initialFiles]);

  useEffect(() => {
    if (open && canReviewUpload(selectedFiles.length, completedFiles.length, progress, uploading)) setStep(2);
  }, [open, selectedFiles.length, completedFiles.length, progress, uploading]);

  const reset = () => {
    requestGeneration.current += 1;
    setUploading(false);
    setStep(1);
    setDatasetNames({});
    replaceRef.current = null;
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
    const removed = new Set([datasetFor(f, selectedFiles.indexOf(f))?.dataset_id, ...sheetsFor(f).map((d) => d.dataset_id)]);
    setBatch((prev) => prev ? { ...prev, datasets: prev.datasets.filter((d) => !removed.has(d.dataset_id)) } : null);
    setSelectedFiles((prev) => prev.filter((x) => fileKey(x) !== key));
  };

  const handleClose = (next: boolean) => {
    if (!next) reset();
    onOpenChange(next);
  };

  const openPicker = (replace?: File) => {
    replaceRef.current = replace ?? null;
    if (uploading) return;
    if (inputRef.current) inputRef.current.value = "";
    inputRef.current?.click();
  };

  const chooseFiles = (incoming: File[]) => {
    const replacing = replaceRef.current;
    replaceRef.current = null;
    if (!replacing) { addFiles(incoming); return; }
    const replacement = incoming[0];
    if (!replacement) return;
    const error = validate(replacement);
    if (error) { toast.error(error); return; }
    const next = selectedFiles.map((f) => f === replacing ? replacement : f);
    const violation = checkLimits(next);
    if (violation) { setLimitExceeded(violation); return; }
    const removed = new Set([datasetFor(replacing, selectedFiles.indexOf(replacing))?.dataset_id, ...sheetsFor(replacing).map((d) => d.dataset_id)]);
    setBatch((prev) => prev ? { ...prev, datasets: prev.datasets.filter((d) => !removed.has(d.dataset_id)) } : null);
    setDatasetNames((prev) => Object.fromEntries(Object.entries(prev).filter(([id]) => !removed.has(id))));
    attemptedRef.current = "";
    setProgress(0);
    setStep(1);
    setSelectedFiles(next);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    if (uploading) return;
    addFiles(Array.from(e.dataTransfer.files ?? []));
  };




  const start = async () => {
    if (!file || !primary || !batch) return;
    if (!canStartSelection(kind, selectedIds)) return;
    // Several files: the first ticked dataset leads; others stay as unselected chips.
    const response = kind === "files" ? (datasets.find((d) => selectedIds.includes(d.dataset_id)) ?? primary) : primary;
    const payload = selectionPayload(kind, selectedIds);
    if (payload && batch.thread_id) {
      try {
        await backendApi.setSelectedDatasets(batch.thread_id, payload.dataset_ids);
      } catch (err) {
        console.warn("[upload] save selected datasets failed", err);
      }
    }
    startAutoInsightsBot();
    // Close instantly so the dataset details are not visible while we persist.
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
          total_rows: response.total_rows ?? (vizConfig as any)?.dataset?.total_rows,
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
    const name = datasetNames[response.dataset_id] ?? reviewDatasetName(response);
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
              name: datasetNames[d.dataset_id] ?? reviewDatasetName(d),
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

    const isStandaloneDashboardUpload = redirectTo === "analysis" && !resolvedProjectId && !isChildUpload && !!newAid;

    if (isStandaloneDashboardUpload && newAid) {
      finishAutoInsightsBot();
      setPendingDashboardAnalysis(newAid);
      onStandaloneAnalysisSaved?.(newAid);
      try {
        sessionStorage.setItem(
          "analysis:context",
          JSON.stringify({
            name,
            project: "",
            projectId: null,
            analysisId: newAid,
          }),
        );
      } catch {}
      toast.success("Analysis saved. Create or select a project to add it.");
      navigate({ to: "/dashboard" });
      reset();
      return;
    }

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
        ...(payload ? { selectedDatasetIds: payload.dataset_ids } : {}),
        childAnalysisId: newAid,
        newAnalysisId: urlAid,
        showAutoInsights: true,
      } as never,
    });
    reset();
  };



  const canStart = canReviewUpload(selectedFiles.length, completedFiles.length, progress, uploading);
  const canFinish = canStart && canStartSelection(kind, selectedIds);
  const multi = datasets.length > 1 || kind === "files";
  const allNotes = datasets.flatMap((d) => datasetNotes(d));
  const skippedNotes = Array.from(new Set(allNotes.filter(isSkippedSheetNote)));
  const totalDatasetRows = datasets.reduce((sum, d) => sum + (Number(d.total_rows ?? (d.visualization_config as any)?.dataset?.total_rows) || 0), 0);
  const summaryTitle =
    kind === "workbook"
      ? `${readableDatasetName(String((datasets[0] as any)?.source_workbook ?? file?.name ?? "Workbook"))}: ${datasets.length} sheets, ${totalDatasetRows.toLocaleString()} rows`
      : `${selectedFiles.length} files`;
  const rowFor = (d: NormalizedDataset, key: string, label: string, checkbox?: { checked: boolean; onChange: (c: boolean) => void; label: string }, extra?: NormalizedDataset[]) => {
    const group = extra ?? [d];
    const rows = group.reduce((sum, x) => sum + (Number(x.total_rows ?? (x.visualization_config as any)?.dataset?.total_rows) || 0), 0);
    const cols = datasetColumns(d).length;
    const issues = group.reduce((sum, x) => sum + reviewIssues(x).length, 0);
    const domain = group.length === 1 ? datasetDomain(d) : `${group.length} sheets`;
    return <CompactDatasetRow key={key} name={label} meta={`${rows.toLocaleString()} rows · ${cols} columns`} domain={domain} issueCount={issues} open={openRow === key} onToggle={() => setOpenRow(openRow === key ? null : key)} checkbox={checkbox}>
      {group.map((x) => <div key={x.dataset_id}>{group.length > 1 && <p className="mt-3 text-xs font-semibold text-secondary">{String((x as any).sheet_name ?? x.filename)}</p>}<DatasetInsightCard dataset={x} name={datasetNames[x.dataset_id] ?? reviewDatasetName(x)} onNameChange={(name) => setDatasetNames((prev) => ({ ...prev, [x.dataset_id]: name }))} hideSkippedNotes compact /></div>)}
    </CompactDatasetRow>;
  };
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
      <DialogContent className="upload-review-modal flex h-[min(720px,calc(100dvh-32px))] w-[calc(100vw-32px)] max-w-[600px] flex-col gap-0 overflow-hidden rounded-xl border border-ai-border bg-primary p-0 shadow-2xl [&>button]:hidden">
        <header className="shrink-0 border-b border-ai-border px-5 py-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-2"><UploadCloud02 className="size-5 shrink-0 text-tertiary" /><DialogTitle className="text-sm font-semibold text-primary">Select File &amp; Start Analysis</DialogTitle></div>
            <ReviewButton variant="ghost" size="icon" aria-label="Close" onClick={() => handleClose(false)}><X className="size-4" /></ReviewButton>
          </div>
          <p className="mt-2 text-xs text-tertiary" aria-label={`Step ${step} of 2`}><span className={step === 1 ? "font-semibold text-brand-secondary" : ""}>1 Upload</span><span className="mx-2">·</span><span className={step === 2 ? "font-semibold text-brand-secondary" : ""}>2 Review</span></p>
          <DialogDescription className="sr-only">Upload your files, then review dataset details and start analysis.</DialogDescription>
        </header>
        <input ref={inputRef} type="file" multiple accept={ACCEPTED.join(",")} className="hidden" onChange={(event) => chooseFiles(Array.from(event.target.files ?? []))} />
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-5 pb-5" data-upload-step={step}>
          {step === 1 ? <>
        {/* Dropzone */}
        <div
          onClick={uploading ? undefined : () => openPicker()}
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
            dragActive ? "border-brand" : "border-secondary",
            !uploading && "cursor-pointer hover:border-brand",
          )}
        >

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
            <p className="mt-1 text-xs text-tertiary">
              {limits
                ? `CSV, Excel or JSON (max. ${maxFileMb} MB per file, ${maxTotalMb} MB total)`
                : "CSV, Excel or JSON (max. 100MB)"}
            </p>
          </div>

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
                  <Loading01 className="size-3 animate-spin" />{" "}
                  {uploadPhase === "processing" ? "Processing your data…" : "Uploading…"}
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
                const sheets = ds ? sheetsFor(f) : [];
                const sheetNames = sheets
                  .map((d) => (d as any).sheet_name)
                  .filter((s): s is string => typeof s === "string" && !!s);
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
                              <Loading01 className="size-3 animate-spin" />{" "}
                              {uploadPhase === "processing" ? "Processing your data…" : "Uploading…"}
                            </span>
                          ) : ds ? (
                            <span className="inline-flex items-center gap-1 font-medium text-[#067647]">
                              <CheckCircle className="size-3" />{" "}
                              {sheets.length > 1 ? `Ready · ${sheets.length} sheets` : "Ready"}
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
                        {sheetNames.length > 1 && (
                          <div className="mt-1.5 flex flex-wrap gap-1">
                            {sheetNames.map((s) => (
                              <span
                                key={s}
                                className="rounded-md border border-secondary bg-secondary px-1.5 py-0.5 text-[10px] font-medium text-tertiary"
                              >
                                {s}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}



          </> : <div className="pt-4">
            <ReviewButton variant="link" className="mb-3 h-auto px-0 text-xs text-brand-secondary" onClick={() => setStep(1)}>Back</ReviewButton>
            <div className="space-y-2">{selectedFiles.map((f, index) => <FileStatusBar key={fileKey(f)} file={f} ready={Boolean(datasetFor(f, index))} onReplace={() => openPicker(f)} />)}</div>
            <ReviewButton variant="link" className="mt-2 h-auto px-0 text-xs text-brand-secondary" onClick={() => openPicker()}>+ Add file</ReviewButton>
            <h2 className="mt-5 border-b border-ai-border pb-2 text-xs font-semibold text-tertiary">AI Insights</h2>
            {multi ? <>
              <UploadSummaryCard title={summaryTitle} notes={skippedNotes} />
              {kind === "workbook" && <p className="mt-3 text-xs font-medium text-secondary">All {datasets.length} sheets will be analysed.</p>}
              <div className="mt-3 space-y-2">
                {kind === "files"
                  ? selectedFiles.map((f, index) => {
                      const group = datasetsOfFile(f, index);
                      if (!group.length) return null;
                      return rowFor(group[0], `file:${f.name}`, f.name, {
                        checked: pickedFiles.includes(f.name),
                        label: `Select ${f.name}`,
                        onChange: (checked) => setPickedFiles((prev) => (checked ? Array.from(new Set([...prev, f.name])) : prev.filter((n) => n !== f.name))),
                      }, group);
                    })
                  : datasets.map((d) => rowFor(d, d.dataset_id, String((d as any).sheet_name ?? d.filename ?? reviewDatasetName(d))))}
              </div>
            </> : datasets.map((d) => <DatasetInsightCard key={d.dataset_id} dataset={d} name={datasetNames[d.dataset_id] ?? reviewDatasetName(d)} onNameChange={(name) => setDatasetNames((prev) => ({ ...prev, [d.dataset_id]: name }))} />)}
            <CleaningRules />
          </div>}
          {summaryLine && <p className="mt-3 text-sm text-tertiary">{summaryLine}</p>}
        </div>
        <footer className="upload-review-footer sticky bottom-0 z-10 shrink-0 border-t border-ai-border bg-primary px-5 py-3">
          <p className="mb-3 text-xs text-tertiary">Avaloka does not capture your data to train on it beyond customization for your needs.</p>
          {step === 2 && canStart && !canFinish && <p className="mb-2 text-xs font-medium text-secondary">Select the file(s) you want to analyse.</p>}
          {step === 1 ? <ReviewButton className="w-full bg-brand-solid text-primary_on-brand hover:bg-brand-solid_hover" disabled={!canStart} onClick={() => setStep(2)}>Continue</ReviewButton> : <div className="grid grid-cols-2 gap-3">
            <ReviewTooltip text="Start the analysis on the data as uploaded."><ReviewButton variant="outline" onClick={start} disabled={!canFinish}><Play className="size-4" />Skip</ReviewButton></ReviewTooltip>
            <ReviewTooltip text={CLEANING_TOOLTIP}><ReviewButton className="bg-brand-solid text-primary_on-brand hover:bg-brand-solid_hover" onClick={start} disabled={!canFinish}><CheckCircle className="size-4" />Clean &amp; Start</ReviewButton></ReviewTooltip>
          </div>}
        </footer>
      </DialogContent>
      {/* Plan limit dialog — its own nested Radix dialog so it stays clickable */}
      <Dialog
        open={!!limitExceeded}
        onOpenChange={(o) => {
          if (!o) setLimitExceeded(null);
        }}
      >
        {limitExceeded && (
          <DialogContent
            className="max-w-[420px] gap-0 rounded-2xl border border-secondary bg-primary p-6 shadow-xl [&>button]:hidden"
            onPointerDownOutside={(e) => e.stopPropagation()}
          >
            <DialogDescription className="sr-only">{limitExceeded.message}</DialogDescription>
            <button
              type="button"
              onClick={() => setLimitExceeded(null)}
              aria-label="Close"
              className="absolute right-4 top-4 rounded-md p-1 text-fg-secondary hover:bg-secondary"
            >
              <X className="size-4" />
            </button>

            <div className="grid size-11 place-items-center rounded-full bg-[#eff5ff] text-[#1565ef]">
              <Diamond01 className="size-5" />
            </div>

            <DialogTitle className="mt-4 text-lg font-semibold text-primary">{limitExceeded.title}</DialogTitle>
            <p className="mt-1.5 text-sm text-tertiary">{limitExceeded.message}</p>

            <div className="mt-6 flex justify-end gap-2">
              {limitExceeded.kind === "cloud" ? (
                <>
                  <Button color="secondary" size="md" onClick={() => setLimitExceeded(null)}>
                    Cancel
                  </Button>
                  <Button
                    color="primary"
                    size="md"
                    onClick={() => {
                      setLimitExceeded(null);
                      handleClose(false);
                      setCloudOpen(true);
                    }}
                  >
                    Load from cloud storage
                  </Button>
                </>
              ) : (
                <>
                  <Button color="secondary" size="md" onClick={() => setLimitExceeded(null)}>
                    Reduce file size
                  </Button>
                  {limitExceeded.showUpgrade && (
                    <Button
                      color="primary"
                      size="md"
                      onClick={() => {
                        setLimitExceeded(null);
                        handleClose(false);
                        navigate({ to: "/settings", search: { tab: "billing" } as any });
                      }}
                    >
                      Upgrade to Professional
                    </Button>
                  )}
                </>
              )}
            </div>
          </DialogContent>
        )}
      </Dialog>
      <ConnectCloudModal open={cloudOpen} onOpenChange={setCloudOpen} />
    </Dialog>
  );
}

/** Insights for exactly one dataset — heading, rows/columns and metrics all
 *  come from the same `dataset_id`, so headings can never drift from schemas. */
/** Column names for one dataset: visualization columns → dataset columns → schema. */
function datasetColumns(dataset: NormalizedDataset): string[] {
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
  return vizCols ?? vizDatasetCols ?? schemaCols;
}

function datasetNotes(dataset: NormalizedDataset): string[] {
  const rawNotes = (dataset as any).notes;
  return Array.isArray(rawNotes)
    ? rawNotes.map((n) => String(n)).filter(Boolean)
    : typeof rawNotes === "string" && rawNotes.trim()
      ? [rawNotes.trim()]
      : [];
}

function datasetDomain(dataset: NormalizedDataset): string | null {
  const viz = (dataset as any)?.visualization_config ?? {};
  const domain = viz.profile_overview?.domain ?? (dataset as any)?.profiling_result?.domain ?? viz.profiling_result?.domain;
  return typeof domain === "string" && domain.trim() ? domain.trim() : null;
}

function DatasetInsightCard({ dataset, name, onNameChange, hideSkippedNotes = false, compact = false }: { dataset: NormalizedDataset; name: string; onNameChange: (name: string) => void; hideSkippedNotes?: boolean; compact?: boolean }) {
  const viz = (dataset as any)?.visualization_config ?? null;
  const columnList: string[] = datasetColumns(dataset);
  // Placeholder names (column_3, Unnamed: 2) are not real column names.
  const metrics = realColumnNames(columnList).slice(0, 4);
  const totalRows = Number(dataset.total_rows ?? viz?.dataset?.total_rows);
  const sampledRows = Number(viz?.dataset?.rows_sampled ?? dataset.rows_sampled);

  const notes: string[] = datasetNotes(dataset).filter((n) => !(hideSkippedNotes && isSkippedSheetNote(n)));

  return (
    <div className={compact ? "mt-3" : "mt-4"}>
      {!compact && <EditableDatasetName name={name} onSave={onNameChange} />}

      <div className="mt-3 rounded-lg border border-secondary bg-primary px-3 py-2.5">
        <p className="text-xs text-secondary">
          {Number.isFinite(totalRows) && totalRows > 0
            ? `${totalRows.toLocaleString()} rows detected across ${columnList.length} columns.`
            : `Data profiled in a sample of ${Number.isFinite(sampledRows) ? sampledRows.toLocaleString() : dataset.samples.length.toLocaleString()} rows across ${columnList.length} columns.`}
        </p>
      </div>

      {notes.length > 0 && (
        <div className="mt-2 space-y-1">
          {notes.map((n, i) => (
            <p key={i} className="text-xs text-tertiary">
              {n}
            </p>
          ))}
        </div>
      )}

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

      <DatasetReviewIssues dataset={dataset} />
    </div>
  );
}
