// Scheduled Analysis — top-level page. Fully separate from the normal
// one-off analysis page. Uses REST endpoints for list/status/info/cancel
// and reuses the existing chat endpoint (sendScheduledChat) to create new
// scheduled tasks. Output view reuses AnalysisOutputTable.

import { createFileRoute } from "@tanstack/react-router";
import { FC, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Calendar,
  SearchLg,
  Grid01,
  Rows01,
  RefreshCcw01,
  Send01,
  Trash01,
  InfoCircle,
  BarChart04,
  ClockRewind,
} from "@untitledui/icons";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import { Input } from "@/components/base/input/input";
import { Button } from "@/components/base/buttons/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Avatar } from "@/components/base/avatar/avatar";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { useUpgradeGate } from "@/components/dashboard/UpgradeGate";
import { cx } from "@/lib/utils/cx";
import { AnalysisOutputTable } from "@/components/dashboard/AnalysisOutputTable";
import { DynamicChart, normalizeVizConfig } from "@/components/dashboard/dynamicChart";
import { deriveVizFromRows } from "@/lib/derive-viz";
import { backendApi } from "@/lib/api/backendApi";
import { RunHistoryDrawer } from "@/components/dashboard/RunHistoryDrawer";
import { sendScheduledChat, getActiveSession } from "@/lib/api/scheduled-tasks";
import {
  scheduledPromptCache,
  summarizePrompt,
  formatSchedule,
  relativeTime,
} from "@/lib/api/scheduled-prompt-cache";

export const Route = createFileRoute("/scheduled-analysis")({
  head: () => ({
    meta: [
      { title: "Scheduled Analysis · Avaloka AI" },
      { name: "description", content: "Create and manage recurring scheduled analyses." },
    ],
  }),
  component: ScheduledAnalysisPage,
});

type Status =
  | "Pending"
  | "Running"
  | "Active"
  | "Success"
  | "Cancelled"
  | "Failed"
  | "Unknown";

type TaskRow = {
  id: string;
  schedule: any;
  scheduleLabel: string;
  total_run_count: number;
  last_run_at: string | null;
  max_runs: number | null;
  status: Status;
  statusError?: string | null;
  taskType?: string | null;
  taskLabel?: string | null;
  rayJob?: {
    job_id?: string;
    status?: string;
    dashboard_url?: string;
    namespace?: string;
  } | null;
};

const STATUS_STYLES: Record<Status, string> = {
  Pending: "bg-slate-50 text-slate-700 ring-slate-200",
  Active: "bg-blue-50 text-blue-700 ring-blue-200",
  Running: "bg-orange-50 text-orange-700 ring-orange-200",
  Success: "bg-green-50 text-green-700 ring-green-200",
  Cancelled: "bg-gray-100 text-gray-700 ring-gray-200",
  Failed: "bg-red-50 text-red-700 ring-red-200",
  Unknown: "bg-gray-50 text-gray-600 ring-gray-200",
};

const TERMINAL_STATUSES: Status[] = ["Success", "Failed", "Cancelled"];

function StatusBadge({ status, title }: { status: Status; title?: string }) {
  return (
    <span
      title={title}
      className={cx(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset",
        STATUS_STYLES[status],
      )}
    >
      {status}
    </span>
  );
}

function statusFromString(s: string): Status | null {
  if (s.includes("success") || s === "ok" || s === "completed" || s === "done") return "Success";
  if (s.includes("cancel") || s.includes("revok")) return "Cancelled";
  if (s.includes("fail") || s.includes("error")) return "Failed";
  if (s.includes("run") || s.includes("start")) return "Running";
  if (s.includes("pending") || s.includes("queued") || s.includes("retry")) return "Pending";
  if (s.includes("active") || s.includes("scheduled") || s.includes("idle") || s.includes("enable"))
    return "Active";
  return null;
}

const STATUS_KEYS = [
  "status",
  "state",
  "task_status",
  "last_run_status",
  "run_status",
  "current_status",
  "result",
  "task",
  "data",
  "latest_run",
];

/** Resolve a Status from a wide range of backend payload shapes. */
function normalizeStatus(raw: any, depth = 0): Status {
  if (raw == null) return "Unknown";

  if (typeof raw === "string" || typeof raw === "number") {
    return statusFromString(String(raw).toLowerCase()) ?? "Unknown";
  }

  if (typeof raw === "boolean") return raw ? "Active" : "Cancelled";

  if (Array.isArray(raw)) {
    // Prefer the most recent entry when a run history is returned.
    for (let i = raw.length - 1; i >= 0 && depth < 3; i--) {
      const s = normalizeStatus(raw[i], depth + 1);
      if (s !== "Unknown") return s;
    }
    return "Unknown";
  }

  if (typeof raw === "object" && depth < 3) {
    for (const key of STATUS_KEYS) {
      if (raw[key] != null) {
        const s = normalizeStatus(raw[key], depth + 1);
        if (s !== "Unknown") return s;
      }
    }
    // Boolean flags that imply an active schedule.
    if (raw.cancelled === true || raw.canceled === true || raw.revoked === true) return "Cancelled";
    if (raw.failed === true) return "Failed";
    if (raw.running === true || raw.is_running === true) return "Running";
    if (raw.enabled === true || raw.is_active === true || raw.active === true) return "Active";
    if (raw.enabled === false || raw.is_active === false || raw.active === false) return "Cancelled";
    if (raw.next_due_at != null || raw.next_run_at != null) return "Active";
  }

  return "Unknown";
}

function statusFromTaskListItem(task: any): Status {
  const explicitStatus = normalizeStatus(task);
  if (explicitStatus !== "Unknown") return explicitStatus;

  const runCount = Number(task?.total_run_count ?? 0);
  const maxRuns = task?.max_runs == null ? null : Number(task.max_runs);
  if (maxRuns != null && Number.isFinite(maxRuns) && maxRuns > 0 && runCount >= maxRuns) {
    return "Success";
  }

  // The list endpoint only returns registered schedules. If it omits a
  // status, the task remains active until it is cancelled/removed or reaches
  // its configured run limit.
  if (task?.schedule != null) return "Active";

  return "Unknown";
}

function sleep(ms: number) {
  return new Promise<void>((r) => setTimeout(r, ms));
}

function ScheduledAnalysisPage() {
  const { blocked, dialog: upgradeDialog } = useUpgradeGate();
  const [view, setView] = useState<"grid" | "table">("table");
  const [tasks, setTasks] = useState<TaskRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [chatMsg, setChatMsg] = useState<{ kind: "info" | "error" | "ok"; text: string } | null>(null);
  const [checkTaskId, setCheckTaskId] = useState("");

  const [infoTaskId, setInfoTaskId] = useState<string | null>(null);
  const [infoData, setInfoData] = useState<any>(null);
  const [infoLoading, setInfoLoading] = useState(false);
  const [outputTaskId, setOutputTaskId] = useState<string | null>(null);
  const [outputRows, setOutputRows] = useState<any[] | null>(null);
  const [outputLoading, setOutputLoading] = useState(false);
  const [outputError, setOutputError] = useState<string | null>(null);
  const [outputStatus, setOutputStatus] = useState<Status>("Unknown");
  const [outputTraceback, setOutputTraceback] = useState<string | null>(null);
  const [outputProgress, setOutputProgress] = useState<string>("");
  const [outputFileData, setOutputFileData] = useState<{ filename?: string; content?: string } | null>(null);
  const [outputRunIndex, setOutputRunIndex] = useState<number | null>(null);
  const [outputNextDueAt, setOutputNextDueAt] = useState<number | null>(null);
  const [confirmCancelId, setConfirmCancelId] = useState<string | null>(null);
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const [runsTaskId, setRunsTaskId] = useState<string | null>(null);
  const [outputView, setOutputView] = useState<"table" | "chart">("table");
  const menuRef = useRef<HTMLDivElement | null>(null);
  const outputPollTokenRef = useRef(0);

  const session = useMemo(() => getActiveSession(), []);

  const refresh = useCallback(async () => {
    if (!session) {
      setListError(
        "No active analysis session. Open Analysis and upload/select a dataset first, then return here.",
      );
      return;
    }
    setLoading(true);
    setListError(null);
    try {
      const raw = await backendApi.listTasks(session.sessionId);
      const rows: TaskRow[] = (raw ?? []).map((t: any) => ({
        id: String(t.id),
        schedule: t.schedule,
        scheduleLabel: formatSchedule(t.schedule),
        total_run_count: Number(t.total_run_count ?? 0),
        last_run_at: t.last_run_at ?? null,
        max_runs: t.max_runs ?? null,
        // Seed from the list payload so registered schedules do not appear
        // Unknown when the status endpoint omits a state field.
        status: statusFromTaskListItem(t),
        taskType: t.task_type ?? null,
        taskLabel: t.task_label ?? null,
        rayJob: t.latest_run?.result?.ray_job ?? null,
      }));
      setTasks(rows);

      // Lazily enrich each row's status — don't block initial render.
      rows.forEach((row) => {
        backendApi
          .getTaskStatus(row.id, session.sessionId)
          .then((s) => {
            const status = normalizeStatus(s);
            setTasks((cur) =>
              cur.map((r) =>
                r.id === row.id
                  ? { ...r, status: status === "Unknown" ? r.status : status, statusError: null }
                  : r,
              ),
            );
          })
          .catch((err: any) => {
            const msg = err?.message ?? "Status request failed";
            setTasks((cur) => cur.map((r) => (r.id === row.id ? { ...r, statusError: msg } : r)));
          });
      });
    } catch (err: any) {
      setListError(err?.message ?? "Failed to load tasks.");
    } finally {
      setLoading(false);
    }
  }, [session]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Close overflow menu on outside click
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (!menuRef.current) return;
      if (!menuRef.current.contains(e.target as Node)) setMenuOpenId(null);
    };
    if (menuOpenId) document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [menuOpenId]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return tasks;
    return tasks.filter((t) => {
      const cached = scheduledPromptCache.get(t.id) ?? "";
      return (
        t.id.toLowerCase().includes(q) ||
        cached.toLowerCase().includes(q) ||
        t.scheduleLabel.toLowerCase().includes(q)
      );
    });
  }, [tasks, search]);

  const handleSend = useCallback(async () => {
    const prompt = input.trim();
    if (!prompt || sending) return;
    if (blocked("Scheduling analyses")) return;
    setSending(true);
    setChatMsg({ kind: "info", text: "Scheduling…" });
    try {
      const { reply, createdTask } = await sendScheduledChat({ prompt });
      if (createdTask) {
        scheduledPromptCache.set(createdTask.task_id, prompt);
        setChatMsg({
          kind: "ok",
          text: `Scheduled task created (${createdTask.task_id.slice(0, 8)}…). ${createdTask.schedule}.`,
        });
        setInput("");
        void refresh();
      } else {
        setChatMsg({ kind: "info", text: reply.slice(0, 240) });
      }
    } catch (err: any) {
      setChatMsg({ kind: "error", text: err?.message ?? "Failed to send." });
    } finally {
      setSending(false);
    }
  }, [input, sending, refresh, blocked]);

  const openInfo = useCallback(
    async (id: string) => {
      setMenuOpenId(null);
      setInfoTaskId(id);
      setInfoData(null);
      if (!session) return;
      setInfoLoading(true);
      try {
        const data = await backendApi.getTaskInfo(id, session.sessionId);
        setInfoData(data);
      } catch (err: any) {
        setInfoData({ error: err?.message ?? "Failed to load info." });
      } finally {
        setInfoLoading(false);
      }
    },
    [session],
  );

  const openOutput = useCallback(
    async (id: string) => {
      setMenuOpenId(null);
      setOutputTaskId(id);
      setOutputRows(null);
      setOutputError(null);
      setOutputTraceback(null);
      setOutputStatus("Unknown");
      setOutputProgress("");
      setOutputFileData(null);
      setOutputRunIndex(null);
      setOutputNextDueAt(null);
      if (!session) return;

      const token = ++outputPollTokenRef.current;
      setOutputLoading(true);
      const cancelled = () => token !== outputPollTokenRef.current;

      const parseRows = (res: any): any[] | null => {
        const rows =
          (Array.isArray(res?.result?.output_json) && res.result.output_json) ||
          (Array.isArray(res?.output_json) && res.output_json) ||
          null;
        return rows && rows.length > 0 ? rows : null;
      };
      const parseFileData = (res: any) =>
        res?.result?.output_file_data ?? res?.output_file_data ?? null;

      // Fetch info -> derive authoritative latest run index.
      const fetchLatestIndex = async (): Promise<number> => {
        try {
          const info: any = await backendApi.getTaskInfo(id, session.sessionId);
          const total = Number(info?.total_run_count ?? 0);
          return Math.max(0, total - 1);
        } catch {
          const row = tasks.find((t) => t.id === id);
          return Math.max(0, (row?.total_run_count ?? 1) - 1);
        }
      };

      // Try latest result once, poll status until terminal, then re-fetch.
      const loadRun = async (latestIndex: number): Promise<any | null> => {
        let terminalRes: any = null;
        try {
          const first: any = await backendApi.getTaskResult(id, session.sessionId, latestIndex);
          if (cancelled()) return null;
          const firstStatus = normalizeStatus(first?.status);
          setOutputStatus(firstStatus);
          if (TERMINAL_STATUSES.includes(firstStatus)) terminalRes = first;
        } catch {
          /* not ready yet */
        }
        if (!terminalRes) {
          const maxAttempts = 60; // ~120s at 2s
          for (let i = 0; i < maxAttempts; i++) {
            if (cancelled()) return null;
            setOutputProgress(`Waiting for run #${latestIndex + 1} to finish… (${i + 1}/${maxAttempts})`);
            let statusRaw: any = null;
            try {
              statusRaw = await backendApi.getTaskStatus(id, session.sessionId);
            } catch { /* transient */ }
            if (cancelled()) return null;
            const st = normalizeStatus(statusRaw);
            if (st !== "Unknown") setOutputStatus(st);
            if (TERMINAL_STATUSES.includes(st)) {
              try {
                terminalRes = await backendApi.getTaskResult(id, session.sessionId, latestIndex);
              } catch (e: any) {
                if (!cancelled()) setOutputError(e?.message ?? "Failed to fetch result.");
              }
              break;
            }
            await sleep(2000);
          }
        }
        return terminalRes;
      };

      const applyResult = (res: any, index: number) => {
        const finalStatus = normalizeStatus(res?.status);
        setOutputStatus(finalStatus);
        setOutputRunIndex(index);
        setOutputNextDueAt(typeof res?.next_due_at === "number" ? res.next_due_at : null);

        if (finalStatus === "Failed") {
          setOutputTraceback(res?.traceback ?? res?.error ?? null);
          setOutputError("This run failed. See the error details below.");
          setOutputRows(null);
          setOutputFileData(null);
          return;
        }
        if (finalStatus === "Cancelled") {
          setOutputError("This run was cancelled. No output was produced.");
          setOutputRows(null);
          setOutputFileData(null);
          return;
        }
        setOutputError(null);
        setOutputTraceback(null);
        setOutputFileData(parseFileData(res));
        const rows = parseRows(res);
        if (rows) setOutputRows(rows);
        else setOutputError("Run finished successfully but returned no tabular data.");
      };

      try {
        const latestIndex = await fetchLatestIndex();
        const first = await loadRun(latestIndex);
        if (cancelled()) return;
        setOutputProgress("");
        if (!first) {
          setOutputError("Timed out waiting for this run to finish. Try Refresh in a moment.");
          return;
        }
        applyResult(first, latestIndex);
        setOutputLoading(false);

        // Recurring re-poll: after next_due_at, look for a newer run index.
        let lastIndex = latestIndex;
        // eslint-disable-next-line no-constant-condition
        while (true) {
          if (cancelled()) return;
          // Wait until the backend says the next run is due (+ 3s grace).
          const now = Math.floor(Date.now() / 1000);
          const dueIn = Math.max(5, (Number((first as any)?.next_due_at) || now + 60) - now + 3);
          // Poll info periodically until total_run_count advances.
          const deadline = Date.now() + dueIn * 1000 + 90_000; // give a run up to 90s to appear
          let newIndex = lastIndex;
          while (Date.now() < deadline) {
            if (cancelled()) return;
            await sleep(Math.min(5000, Math.max(2000, dueIn * 200)));
            try {
              const info: any = await backendApi.getTaskInfo(id, session.sessionId);
              const total = Number(info?.total_run_count ?? 0);
              if (total - 1 > lastIndex) {
                newIndex = total - 1;
                break;
              }
            } catch { /* transient */ }
          }
          if (newIndex === lastIndex) continue; // still no new run; keep waiting
          const nextRes = await loadRun(newIndex);
          if (cancelled()) return;
          if (nextRes) applyResult(nextRes, newIndex);
          lastIndex = newIndex;
        }
      } catch (err: any) {
        if (!cancelled()) setOutputError(err?.message ?? "Failed to fetch output.");
      } finally {
        if (!cancelled()) setOutputLoading(false);
      }
    },
    [session, tasks],
  );


  const closeOutput = useCallback(() => {
    outputPollTokenRef.current++; // invalidate any in-flight poll
    setOutputTaskId(null);
    setOutputLoading(false);
    setOutputProgress("");
    setOutputFileData(null);
    setOutputRunIndex(null);
    setOutputNextDueAt(null);
    setOutputView("table");
  }, []);


  const doCancel = useCallback(
    async (id: string) => {
      setConfirmCancelId(null);
      setMenuOpenId(null);
      if (!session) return;
      // Optimistic
      setTasks((cur) => cur.map((r) => (r.id === id ? { ...r, status: "Cancelled" } : r)));
      if (outputTaskId === id) {
        outputPollTokenRef.current++; // stop polling
        setOutputStatus("Cancelled");
        setOutputLoading(false);
        setOutputRows(null);
        setOutputError("This run was cancelled. No output was produced.");
      }
      try {
        await backendApi.deleteTask(id, session.sessionId);
        // Remove after short delay so the greyed state is visible briefly.
        setTimeout(() => {
          setTasks((cur) => cur.filter((r) => r.id !== id));
        }, 500);
      } catch (err: any) {
        setChatMsg({ kind: "error", text: `Cancel failed: ${err?.message ?? "unknown"}` });
        void refresh();
      }
    },
    [session, refresh, outputTaskId],
  );

  const titleFor = (id: string) => {
    const cached = scheduledPromptCache.get(id);
    const task = tasks.find((item) => item.id === id);
    return cached ? summarizePrompt(cached) : task?.taskLabel || `Task ${id.slice(0, 8)}`;
  };

  const outputTask = outputTaskId ? tasks.find((t) => t.id === outputTaskId) : null;
  const infoTaskCached = infoTaskId ? scheduledPromptCache.get(infoTaskId) : undefined;

  return (
    <DashboardShell>
      <div className="flex min-h-full flex-1 flex-col gap-4 p-6">
        {/* Header */}
        <div className="flex flex-1 flex-col rounded-2xl border border-secondary bg-primary shadow-xs">
          <div className="flex items-center justify-between border-b border-secondary px-6 py-4">
            <div className="flex items-center gap-2">
              <Calendar className="size-5 text-fg-secondary" />
              <h1 className="text-lg font-semibold text-primary">Scheduled Analysis</h1>
              <span className="ml-2 rounded-full bg-secondary px-2 py-0.5 text-xs text-tertiary">
                {tasks.length} task{tasks.length === 1 ? "" : "s"}
              </span>
            </div>
            <Button
              color="secondary"
              size="md"
              iconLeading={RefreshCcw01}
              onClick={() => void refresh()}
              isLoading={loading}
            >
              Refresh
            </Button>
          </div>

          {/* Prompt input */}
          <div className="flex items-start gap-3 border-b border-secondary bg-secondary/30 px-6 py-4">
            <div className="flex-1">
              <label className="text-xs font-semibold text-tertiary">
                New scheduled analysis
              </label>
              <form
                className="mt-1.5 flex items-center gap-2"
                onSubmit={(e) => {
                  e.preventDefault();
                  void handleSend();
                }}
              >
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder='e.g. "Group by Region then sum Revenue. Schedule this to run every minute."'
                  disabled={sending}
                  className="flex-1 rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-primary outline-none focus:border-brand"
                />
                <Button
                  type="submit"
                  color="primary"
                  size="md"
                  iconLeading={Send01}
                  isDisabled={sending || !input.trim()}
                >
                  Schedule
                </Button>
              </form>
              {chatMsg && (
                <p
                  className={cx(
                    "mt-2 text-xs",
                    chatMsg.kind === "error"
                      ? "text-red-600"
                      : chatMsg.kind === "ok"
                      ? "text-green-700"
                      : "text-tertiary",
                  )}
                >
                  {chatMsg.text}
                </p>
              )}
            </div>
          </div>

          {/* Toolbar */}
          <div className="flex flex-wrap items-center justify-between gap-3 px-6 py-4">
            <div className="w-full max-w-[360px]">
              <Input
                aria-label="Search scheduled tasks"
                placeholder="Search by prompt, schedule or task id"
                size="md"
                icon={SearchLg}
                value={search}
                onChange={(v: any) =>
                  setSearch(typeof v === "string" ? v : v?.target?.value ?? "")
                }
              />
            </div>
            <div className="inline-flex items-center rounded-lg border border-secondary bg-primary p-0.5 shadow-xs">
              <button
                type="button"
                onClick={() => setView("grid")}
                className={cx(
                  "inline-flex h-8 items-center gap-1.5 rounded-md px-3 text-sm font-semibold",
                  view === "grid" ? "bg-secondary text-primary" : "text-tertiary hover:text-primary",
                )}
              >
                <Grid01 className="size-4" />
                Grid
              </button>
              <button
                type="button"
                onClick={() => setView("table")}
                className={cx(
                  "inline-flex h-8 items-center gap-1.5 rounded-md px-3 text-sm font-semibold",
                  view === "table" ? "bg-secondary text-primary" : "text-tertiary hover:text-primary",
                )}
              >
                <Rows01 className="size-4" />
                Table
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 px-6 pb-6">
            {listError ? (
              <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                {listError}
              </div>
            ) : loading && tasks.length === 0 ? (
              <div className="py-16 text-center text-sm text-tertiary">Loading scheduled tasks…</div>
            ) : filtered.length === 0 ? (
              <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-secondary bg-primary/50 px-6 py-16 text-center">
                <Calendar className="size-8 text-tertiary" />
                <p className="text-sm font-semibold text-primary">No scheduled tasks yet</p>
                <p className="max-w-md text-xs text-tertiary">
                  Use the prompt above to create one. Example:{" "}
                  <em>&ldquo;Group by Region then sum Revenue. Schedule this to run every minute.&rdquo;</em>
                </p>
              </div>
            ) : view === "grid" ? (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                {filtered.map((t) => {
                  const cancelled = t.status === "Cancelled";
                  return (
                    <article
                      key={t.id}
                      className={cx(
                        "relative flex flex-col rounded-xl border border-secondary bg-primary shadow-xs transition hover:border-brand",
                        cancelled && "opacity-60",
                      )}
                    >
                      <div className="flex items-start justify-between p-4">
                        <div className="grid size-9 place-items-center rounded-lg bg-secondary text-brand">
                          <Calendar className="size-5" />
                        </div>
                        <div className="flex flex-wrap items-center justify-end gap-2">
                          <StatusBadge status={t.status} title={t.statusError ?? undefined} />

                          <RowActions
                            onInfo={() => void openInfo(t.id)}
                            onOutput={() => void openOutput(t.id)}
                            onRuns={() => setRunsTaskId(t.id)}
                            onCancel={() => {
                              if (blocked("Cancelling scheduled tasks")) return;
                              setConfirmCancelId(t.id);
                            }}
                          />
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => void openOutput(t.id)}
                        className="flex-1 px-4 pb-4 text-left"
                      >
                        <div className="text-md font-semibold text-primary hover:underline">
                          {titleFor(t.id)}
                        </div>
                        <p className="mt-1 text-sm text-tertiary">
                          Scheduled: {t.scheduleLabel}
                        </p>
                        <p className="mt-1 text-xs text-tertiary">
                          <span
                            role="button"
                            tabIndex={0}
                            title="View run history"
                            className="cursor-pointer text-brand hover:underline"
                            onClick={(e) => {
                              e.stopPropagation();
                              e.preventDefault();
                              setRunsTaskId(t.id);
                            }}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" || e.key === " ") {
                                e.stopPropagation();
                                e.preventDefault();
                                setRunsTaskId(t.id);
                              }
                            }}
                          >
                            Ran {t.total_run_count} time{t.total_run_count === 1 ? "" : "s"}
                          </span>
                        </p>
                        <p className="mt-1 truncate text-xs text-tertiary">
                          task_id: <span className="font-mono">{t.id}</span>
                        </p>
                        {t.rayJob?.job_id && (
                          <p className="mt-1 truncate text-xs text-tertiary">
                            Ray job: <span className="font-mono">{t.rayJob.job_id}</span>
                            {t.rayJob.status ? ` · ${t.rayJob.status}` : ""}
                          </p>
                        )}
                      </button>
                      <div className="flex items-center justify-between gap-3 border-t border-secondary px-4 py-3">
                        <div className="flex min-w-0 items-center gap-2">
                          <Avatar size="sm" alt="You" />
                          <span className="truncate text-sm font-semibold text-primary">You</span>
                        </div>
                        <span className="shrink-0 text-xs text-tertiary">
                          Last run: {relativeTime(t.last_run_at)}
                        </span>
                      </div>
                    </article>
                  );
                })}
              </div>
            ) : (
              <div className="rounded-xl border border-secondary">
                <table className="w-full text-left text-sm">
                  <thead className="bg-secondary/40">
                    <tr className="text-xs font-semibold text-tertiary">
                      <th className="px-4 py-3">Status</th>
                      <th className="px-4 py-3">Task</th>
                      <th className="px-4 py-3">Schedule</th>
                      <th className="px-4 py-3">Runs</th>
                      <th className="px-4 py-3">Owner</th>
                      <th className="px-4 py-3">Last Run</th>
                      <th className="px-4 py-3 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((t) => (
                      <tr
                        key={t.id}
                        className={cx(
                          "border-t border-secondary hover:bg-primary_hover",
                          t.status === "Cancelled" && "opacity-60",
                        )}
                      >
                        <td className="px-4 py-3">
                          <StatusBadge status={t.status} title={t.statusError ?? undefined} />
                        </td>
                        <td className="px-4 py-3 text-sm font-medium text-primary">
                          <button
                            className="hover:underline"
                            onClick={() => void openOutput(t.id)}
                          >
                            {titleFor(t.id)}
                          </button>
                          <div className="font-mono text-xs text-tertiary">{t.id}</div>
                          {t.rayJob?.job_id && (
                            <div className="font-mono text-xs text-tertiary">
                              Ray: {t.rayJob.job_id}
                              {t.rayJob.status ? ` · ${t.rayJob.status}` : ""}
                            </div>
                          )}
                        </td>
                        <td className="px-4 py-3 text-sm text-secondary">{t.scheduleLabel}</td>
                        <td className="px-4 py-3 text-sm text-tertiary">
                          <button
                            type="button"
                            className="text-brand hover:underline"
                            onClick={() => setRunsTaskId(t.id)}
                            title="View run history"
                          >
                            Ran {t.total_run_count} time{t.total_run_count === 1 ? "" : "s"}
                          </button>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex min-w-0 items-center gap-2">
                            <Avatar size="sm" alt="You" />
                            <span className="truncate text-sm font-semibold text-primary">You</span>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-sm text-tertiary">
                          {relativeTime(t.last_run_at)}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <RowActions
                            onInfo={() => void openInfo(t.id)}
                            onOutput={() => void openOutput(t.id)}
                            onRuns={() => setRunsTaskId(t.id)}
                            onCancel={() => {
                              if (blocked("Cancelling scheduled tasks")) return;
                              setConfirmCancelId(t.id);
                            }}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Check task status / result */}
            <div className="mt-6 rounded-xl border border-secondary bg-secondary/30 p-4">
              <label className="text-xs font-semibold text-tertiary">
                Check task status & result
              </label>
              <form
                className="mt-1.5 flex items-center gap-2"
                onSubmit={(e) => {
                  e.preventDefault();
                  const id = checkTaskId.trim();
                  if (!id) return;
                  void openOutput(id);
                }}
              >
                <input
                  value={checkTaskId}
                  onChange={(e) => setCheckTaskId(e.target.value)}
                  placeholder="Enter task_id to check status & view result"
                  className="flex-1 rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-primary outline-none focus:border-brand"
                />
                <Button
                  type="submit"
                  color="primary"
                  size="md"
                  iconLeading={Send01}
                  isDisabled={!checkTaskId.trim()}
                >
                  Send
                </Button>
              </form>
              <p className="mt-2 text-xs text-tertiary">
                Paste a task_id from the list above to view its latest status and output.
              </p>
            </div>
          </div>

        </div>
      </div>

      {/* Info modal */}
      <Dialog open={!!infoTaskId} onOpenChange={(o) => !o && setInfoTaskId(null)}>
        <DialogContent className="max-w-2xl">
          {infoTaskId && (
            <div className="flex flex-col gap-3">
              <div>
                <h2 className="text-lg font-semibold text-primary">
                  {titleFor(infoTaskId)}
                </h2>
                <p className="mt-1 font-mono text-xs text-tertiary">{infoTaskId}</p>
              </div>
              {infoTaskCached && (
                <div>
                  <p className="text-xs font-semibold text-tertiary">Prompt (cached)</p>
                  <p className="mt-1 whitespace-pre-wrap rounded-lg bg-secondary p-3 text-sm text-primary">
                    {infoTaskCached}
                  </p>
                </div>
              )}
              <div>
                <p className="text-xs font-semibold text-tertiary">Task info</p>
                <pre className="mt-1 max-h-80 overflow-auto rounded-lg bg-secondary p-3 text-xs text-primary">
                  {infoLoading ? "Loading…" : JSON.stringify(infoData, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Output modal */}
      <Dialog open={!!outputTaskId} onOpenChange={(o) => !o && closeOutput()}>
        <DialogContent className="max-w-4xl">
          {outputTaskId && (
            <div className="flex flex-col gap-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-semibold text-primary">
                      {titleFor(outputTaskId)}
                    </h2>
                    <StatusBadge status={outputStatus} />
                  </div>
                  <p className="mt-1 text-xs text-tertiary">
                    task_id: <span className="font-mono">{outputTaskId}</span>
                    {outputTask && <> · {outputTask.scheduleLabel}</>}
                    {outputRunIndex != null && <> · Run #{outputRunIndex + 1}</>}
                    {outputNextDueAt && outputNextDueAt * 1000 > Date.now() && (
                      <> · Next run {relativeTime(new Date(outputNextDueAt * 1000).toISOString())}</>
                    )}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {outputFileData?.content && (
                    <a
                      href={outputFileData.content}
                      download={outputFileData.filename ?? `task-${outputTaskId.slice(0, 8)}.csv`}
                      className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-secondary bg-primary px-3 text-sm font-semibold text-primary shadow-xs hover:bg-primary_hover"
                    >
                      Download CSV
                    </a>
                  )}
                  {outputStatus !== "Cancelled" && (
                    <Button
                      color="primary-destructive"
                      size="sm"
                      iconLeading={Trash01}
                      onClick={() => setConfirmCancelId(outputTaskId)}
                    >
                      Cancel task
                    </Button>
                  )}
                </div>
              </div>
              {outputLoading ? (
                <div className="flex flex-col items-center gap-2 py-10 text-center text-sm text-tertiary">
                  <span>Loading latest output…</span>
                  {outputProgress && <span className="text-xs">{outputProgress}</span>}
                </div>
              ) : outputError ? (
                <div className="flex flex-col gap-3">
                  <div
                    className={cx(
                      "rounded-lg border p-3 text-sm",
                      outputStatus === "Failed"
                        ? "border-red-200 bg-red-50 text-red-700"
                        : "border-amber-200 bg-amber-50 text-amber-800",
                    )}
                  >
                    {outputError}
                  </div>
                  {outputTraceback && (
                    <div>
                      <p className="text-xs font-semibold text-tertiary">Traceback</p>
                      <pre className="mt-1 max-h-80 overflow-auto rounded-lg bg-secondary p-3 text-xs text-primary">
                        {outputTraceback}
                      </pre>
                    </div>
                  )}
                </div>
              ) : outputRows ? (
                <div className="flex flex-col gap-3">
                  {/* Data table / Chart toggle — default Data table for scheduled task results */}
                  <div className="flex justify-end">
                    <div className="inline-flex items-center rounded-lg border border-secondary bg-primary p-1 shadow-xs">
                      <button
                        type="button"
                        onClick={() => setOutputView("table")}
                        className={cx(
                          "rounded-md px-3 py-1.5 text-sm font-semibold transition",
                          outputView === "table"
                            ? "bg-secondary text-primary"
                            : "text-tertiary hover:text-primary",
                        )}
                      >
                        Data table
                      </button>
                      <button
                        type="button"
                        onClick={() => setOutputView("chart")}
                        className={cx(
                          "rounded-md px-3 py-1.5 text-sm font-semibold transition",
                          outputView === "chart"
                            ? "bg-secondary text-primary"
                            : "text-tertiary hover:text-primary",
                        )}
                      >
                        Chart
                      </button>
                    </div>
                  </div>
                  {outputView === "table" ? (
                    <AnalysisOutputTable
                      rows={outputRows}
                      title="Latest Output"
                      filename={`task-${outputTaskId.slice(0, 8)}.csv`}
                      emptyMessage="No output rows yet for this task."
                    />
                  ) : (
                    <ScheduledOutputChart rows={outputRows} />
                  )}
                </div>
              ) : (
                <div className="rounded-lg border border-dashed border-secondary bg-primary/50 p-6 text-center text-sm text-tertiary">
                  No data returned by this run.
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>


      {/* Cancel confirm */}
      <Dialog open={!!confirmCancelId} onOpenChange={(o) => !o && setConfirmCancelId(null)}>
        <DialogContent className="max-w-md">
          {confirmCancelId && (
            <div className="flex flex-col gap-4">
              <div>
                <h2 className="text-lg font-semibold text-primary">Cancel scheduled task?</h2>
                <p className="mt-1 text-sm text-tertiary">
                  This will stop future runs of{" "}
                  <span className="font-mono text-xs">{confirmCancelId}</span>. This can&rsquo;t be undone.
                </p>
              </div>
              <div className="flex justify-end gap-2">
                <Button color="secondary" size="sm" onClick={() => setConfirmCancelId(null)}>
                  Keep task
                </Button>
                <Button
                  color="primary-destructive"
                  size="sm"
                  onClick={() => void doCancel(confirmCancelId)}
                >
                  Cancel task
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {runsTaskId && (
        <RunHistoryDrawer
          taskId={runsTaskId}
          sessionId={session?.sessionId ?? null}
          onClose={() => setRunsTaskId(null)}
        />
      )}
      {upgradeDialog}
    </DashboardShell>

  );
}

function ActionIconButton({
  icon: Icon,
  tooltip,
  onClick,
  variant,
}: {
  icon: FC<{ className?: string }>;
  tooltip: string;
  onClick: () => void;
  variant: "info" | "output" | "runs" | "cancel";
}) {
  const variantClasses = {
    info: "hover:text-blue-600 hover:bg-blue-50 dark:hover:text-blue-400 dark:hover:bg-blue-950/40",
    output:
      "hover:text-violet-600 hover:bg-violet-50 dark:hover:text-violet-400 dark:hover:bg-violet-950/40",
    runs: "hover:text-teal-600 hover:bg-teal-50 dark:hover:text-teal-400 dark:hover:bg-teal-950/40",
    cancel: "hover:text-red-600 hover:bg-red-50 dark:hover:text-red-400 dark:hover:bg-red-950/40",
  };

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          title={tooltip}
          aria-label={tooltip}
          onClick={onClick}
          className={cx(
            "inline-flex size-8 items-center justify-center rounded-md text-fg-quaternary transition-colors duration-200 outline-focus-ring focus-visible:outline-2 focus-visible:outline-offset-2",
            variantClasses[variant],
          )}
        >
          <Icon className="size-4" />
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        sideOffset={6}
        className="z-[9999] rounded-md border border-gray-800 bg-gray-900 px-2 py-1 text-xs font-medium text-white shadow-lg dark:border-gray-200 dark:bg-gray-100 dark:text-gray-900"
      >
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}


function RowActions({
  onInfo,
  onOutput,
  onRuns,
  onCancel,
  className,
}: {
  onInfo: () => void;
  onOutput: () => void;
  onRuns: () => void;
  onCancel: () => void;
  className?: string;
}) {
  return (
    <TooltipProvider delayDuration={200}>
      <div className={cx("inline-flex flex-wrap items-center justify-end gap-1", className)}>
        <ActionIconButton icon={InfoCircle} tooltip="Info" variant="info" onClick={onInfo} />
        <ActionIconButton icon={BarChart04} tooltip="View Output" variant="output" onClick={onOutput} />
        <ActionIconButton icon={ClockRewind} tooltip="View runs" variant="runs" onClick={onRuns} />
        <ActionIconButton icon={Trash01} tooltip="Cancel task" variant="cancel" onClick={onCancel} />
      </div>
    </TooltipProvider>
  );
}

function ScheduledOutputChart({ rows }: { rows: any[] }) {
  const viz = useMemo(() => deriveVizFromRows(rows), [rows]);
  const slides = useMemo(() => (viz ? normalizeVizConfig(viz, rows) : []), [viz, rows]);
  if (!rows?.length) {
    return (
      <div className="rounded-lg border border-dashed border-secondary bg-primary/50 p-6 text-center text-sm text-tertiary">
        No rows to chart.
      </div>
    );
  }
  if (!slides.length) {
    return (
      <div className="rounded-lg border border-dashed border-secondary bg-primary/50 p-6 text-center text-sm text-tertiary">
        This output can&rsquo;t be visualised as a chart. Switch back to Data table.
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-6">
      {slides.map((slide, i) => (
        <div key={i} className="rounded-xl border border-secondary bg-primary p-4">
          <h3 className="text-sm font-semibold text-primary">{slide.title}</h3>
          {slide.subtitle && <p className="text-xs text-tertiary">{slide.subtitle}</p>}
          <div className="mt-3 h-[320px] w-full">
            <DynamicChart slide={slide} height={320} />
          </div>
        </div>
      ))}
    </div>
  );
}
