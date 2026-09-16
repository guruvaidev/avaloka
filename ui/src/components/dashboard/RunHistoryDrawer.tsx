// Right-side drawer showing the full run history of a scheduled task.
// Fetches GET /tasks/{task_id}/runs through the backend proxy (Bearer token
// + X-Avaloka-Session are attached by backendApi).

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, RefreshCcw01, X as XIcon } from "@untitledui/icons";
import { Button } from "@/components/base/buttons/button";
import { ButtonUtility } from "@/components/base/buttons/button-utility";
import { cx } from "@/lib/utils/cx";
import { backendApi } from "@/lib/api/backendApi";

export type TaskRun = {
  run_id?: string;
  run_number?: number;
  status?: string;
  started_at?: string | null;
  finished_at?: string | null;
  duration_ms?: number | null;
  result?: {
    output_json?: any[] | null;
    output_file_data?: { filename?: string; content?: string; size?: number } | null;
    logs?: string[] | string | null;
    message?: string;
    ray_job?: {
      job_id?: string;
      status?: string;
      dashboard_url?: string;
      namespace?: string;
    } | null;
  } | null;
  error?: string | null;
};

type RunStatus = "success" | "failure" | "running" | "pending";

function normalizeRunStatus(s: unknown): RunStatus {
  const v = String(s ?? "").toLowerCase();
  if (v.includes("succ") || v === "ok" || v.includes("complete") || v === "done") return "success";
  if (v.includes("fail") || v.includes("error")) return "failure";
  if (v.includes("run") || v.includes("start")) return "running";
  return "pending";
}

const RUN_STATUS_STYLES: Record<RunStatus, string> = {
  success: "bg-green-50 text-green-700 ring-green-200 dark:bg-green-500/10 dark:text-green-400 dark:ring-green-500/30",
  failure: "bg-red-50 text-red-700 ring-red-200 dark:bg-red-500/10 dark:text-red-400 dark:ring-red-500/30",
  running: "bg-blue-50 text-blue-700 ring-blue-200 dark:bg-blue-500/10 dark:text-blue-400 dark:ring-blue-500/30",
  pending: "bg-gray-100 text-gray-600 ring-gray-200 dark:bg-white/10 dark:text-gray-300 dark:ring-white/15",
};

const RUN_STATUS_LABEL: Record<RunStatus, string> = {
  success: "Success",
  failure: "Failure",
  running: "Running",
  pending: "Pending",
};

function RunStatusBadge({ status }: { status: RunStatus }) {
  return (
    <span
      className={cx(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset",
        RUN_STATUS_STYLES[status],
      )}
    >
      {RUN_STATUS_LABEL[status]}
    </span>
  );
}

function relativeTime(iso?: string | null): string {
  if (!iso) return "—";
  const t = new Date(iso).getTime();
  if (!Number.isFinite(t)) return "—";
  const diff = Date.now() - t;
  const s = Math.round(diff / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

function absoluteTime(iso?: string | null): string {
  if (!iso) return "No start time recorded";
  const d = new Date(iso);
  return Number.isFinite(d.getTime()) ? d.toLocaleString() : String(iso);
}

function formatDuration(ms?: number | null): string {
  if (ms == null || !Number.isFinite(Number(ms))) return "—";
  const n = Number(ms);
  if (n < 1000) return `${Math.round(n)}ms`;
  if (n < 60_000) return `${(n / 1000).toFixed(1)}s`;
  const mins = Math.floor(n / 60_000);
  const secs = Math.round((n % 60_000) / 1000);
  return `${mins}m ${secs}s`;
}

function Spinner() {
  return <span className="inline-block size-3.5 animate-spin rounded-full border-2 border-secondary border-t-[#1565EF]" />;
}

function RunLogs({ run }: { run: TaskRun }) {
  const lines = useMemo(() => {
    const raw = run.result?.logs;
    if (Array.isArray(raw)) {
      const arr = raw.map((l) => (typeof l === "string" ? l : JSON.stringify(l)));
      if (arr.length > 0) return arr;
    } else if (typeof raw === "string" && raw.trim()) {
      return raw.split("\n");
    }
    if (run.error) return String(run.error).split("\n");
    return [];
  }, [run]);

  if (lines.length === 0) {
    return <p className="text-sm text-tertiary">No logs available for this run.</p>;
  }

  return (
    <pre className="max-h-[320px] overflow-y-auto overflow-x-hidden whitespace-pre-wrap break-words rounded-lg border border-secondary bg-gray-50 p-3 font-mono text-xs leading-relaxed text-secondary dark:bg-gray-950 dark:text-gray-200">
      {lines.join("\n")}
    </pre>
  );
}

function RunRow({ run, index }: { run: TaskRun; index: number }) {
  const [open, setOpen] = useState(false);
  const status = normalizeRunStatus(run.status);

  return (
    <div className="overflow-hidden rounded-xl border border-secondary bg-primary">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-primary_hover"
      >
        {open ? <ChevronDown className="size-4 text-tertiary" /> : <ChevronRight className="size-4 text-tertiary" />}
        <span className="text-sm font-semibold text-primary">Run #{run.run_number ?? index + 1}</span>
        <RunStatusBadge status={status} />
        <span className="ml-auto flex items-center gap-3 text-xs text-tertiary">
          <span title={absoluteTime(run.started_at)}>{relativeTime(run.started_at)}</span>
          <span>{formatDuration(run.duration_ms)}</span>
        </span>
      </button>

      {open && (
        <div className="border-t border-secondary px-4 py-3">
          {run.result?.ray_job?.job_id && (
            <dl className="mb-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 rounded-lg border border-secondary bg-secondary/30 p-3 text-xs">
              <dt className="font-semibold text-tertiary">Ray job</dt>
              <dd className="break-all font-mono text-primary">{run.result.ray_job.job_id}</dd>
              <dt className="font-semibold text-tertiary">Ray status</dt>
              <dd className="text-primary">{run.result.ray_job.status || "Unknown"}</dd>
              {run.result.ray_job.namespace && (
                <>
                  <dt className="font-semibold text-tertiary">Namespace</dt>
                  <dd className="font-mono text-primary">{run.result.ray_job.namespace}</dd>
                </>
              )}
            </dl>
          )}
          {(status === "running" || status === "pending") && !run.result?.logs?.length ? (
            <div className="flex items-center gap-2 text-sm text-tertiary">
              <Spinner />
              {RUN_STATUS_LABEL[status]}…
            </div>
          ) : (
            <RunLogs run={run} />
          )}
        </div>
      )}
    </div>
  );
}

export function RunHistoryDrawer({
  taskId,
  sessionId,
  onClose,
}: {
  taskId: string;
  sessionId: string | null;
  onClose: () => void;
}) {
  const [runs, setRuns] = useState<TaskRun[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const reqToken = useRef(0);

  const load = useCallback(
    async (showSpinner = true) => {
      if (!sessionId) {
        setLoading(false);
        setError("No active analysis session. Open Analysis and select a dataset first.");
        return;
      }
      const token = ++reqToken.current;
      if (showSpinner) setLoading(true);
      try {
        const data = await backendApi.getTaskRuns(taskId, sessionId);
        if (token !== reqToken.current) return;
        setRuns(Array.isArray(data) ? data : []);
        setError(null);
      } catch (err: any) {
        if (token !== reqToken.current) return;
        setError(err?.message ?? "Failed to load run history.");
      } finally {
        if (token === reqToken.current) setLoading(false);
      }
    },
    [taskId, sessionId],
  );

  useEffect(() => {
    void load(true);
  }, [load]);

  const hasActive = !!runs?.some((r) => {
    const s = normalizeRunStatus(r.status);
    return s === "running" || s === "pending";
  });

  useEffect(() => {
    if (!hasActive) return;
    const id = window.setInterval(() => void load(false), 5000);
    return () => window.clearInterval(id);
  }, [hasActive, load]);

  return (
    <div className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col border-l border-secondary bg-primary shadow-2xl">
      <div className="flex items-center justify-between gap-3 border-b border-secondary px-5 py-4">
        <div className="min-w-0">
          <h2 className="truncate text-md font-semibold text-primary">
            Run history — Task {taskId.slice(0, 8)}
          </h2>
          <p className="truncate font-mono text-xs text-tertiary">{taskId}</p>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <ButtonUtility
            size="xs"
            color="tertiary"
            icon={RefreshCcw01}
            tooltip="Refresh"
            onClick={() => void load(true)}
          />
          <ButtonUtility size="xs" color="tertiary" icon={XIcon} tooltip="Close" onClick={onClose} />
        </div>
      </div>

      <div className="scrollbar-hide flex-1 space-y-3 overflow-y-auto p-5">
        {loading && !runs ? (
          <>
            {[0, 1, 2].map((i) => (
              <div key={i} className="h-14 animate-pulse rounded-xl border border-secondary bg-secondary/40" />
            ))}
          </>
        ) : error ? (
          <div className="flex flex-col items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400">
            <span>{error}</span>
            <Button size="sm" color="secondary" onClick={() => void load(true)}>
              Retry
            </Button>
          </div>
        ) : !runs || runs.length === 0 ? (
          <div className="rounded-xl border border-dashed border-secondary bg-primary/50 px-6 py-12 text-center text-sm text-tertiary">
            No runs recorded yet.
          </div>
        ) : (
          runs.map((run, i) => <RunRow key={run.run_id ?? `${run.run_number ?? i}`} run={run} index={i} />)
        )}
      </div>
    </div>
  );
}
