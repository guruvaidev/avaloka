// Client-side store + helpers for SCHEDULED analyses. Kept fully separate
// from the normal one-off analysis flow. Backend interactions go through
// the SAME chat endpoint via `backendApi.sendMessage` — the backend decides
// whether a prompt creates/lists/cancels a scheduled task. We use the
// `task_info` field on the response to detect that a schedule was created.

import { backendApi } from "@/lib/api/backendApi";
import { supabase } from "@/integrations/supabase/client";
import { getCachedAuthUser } from "@/lib/auth-user";

export type ScheduledTaskStatus =
  | "Active"
  | "Running"
  | "Paused"
  | "Cancelled"
  | "Failed";

export type ScheduledTask = {
  task_id: string;
  name: string;
  prompt: string;
  schedule: string;
  status: ScheduledTaskStatus;
  next_run?: string | null;
  last_run?: string | null;
  run_count?: number;
  created_at: string;
  owner?: string;
  latest_output?: Record<string, unknown>[] | null;
  output_filename?: string | null;
};

const STORAGE_KEY = "avaloka:scheduled-tasks";

function readAll(): ScheduledTask[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeAll(items: ScheduledTask[]) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  } catch {}
  try {
    window.dispatchEvent(new CustomEvent("scheduled-tasks:changed"));
  } catch {}
}

export const scheduledTaskStore = {
  list(): ScheduledTask[] {
    return readAll().sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
  },
  upsert(task: ScheduledTask) {
    const all = readAll();
    const idx = all.findIndex((t) => t.task_id === task.task_id);
    if (idx >= 0) all[idx] = { ...all[idx], ...task };
    else all.unshift(task);
    writeAll(all);
  },
  patch(task_id: string, patch: Partial<ScheduledTask>) {
    const all = readAll();
    const idx = all.findIndex((t) => t.task_id === task_id);
    if (idx < 0) return;
    all[idx] = { ...all[idx], ...patch };
    writeAll(all);
  },
  remove(task_id: string) {
    writeAll(readAll().filter((t) => t.task_id !== task_id));
  },
  subscribe(cb: () => void): () => void {
    if (typeof window === "undefined") return () => {};
    const handler = () => cb();
    window.addEventListener("scheduled-tasks:changed", handler);
    window.addEventListener("storage", handler);
    return () => {
      window.removeEventListener("scheduled-tasks:changed", handler);
      window.removeEventListener("storage", handler);
    };
  },
};

/** Detect a "schedule" phrase in a prompt (natural language). */
export function extractSchedulePhrase(prompt: string): string | null {
  const m = prompt.match(
    /every\s+(?:(\d+)\s+)?(minute|minutes|hour|hours|day|days|week|weeks|month|months)/i,
  );
  if (m) {
    const n = m[1] ? parseInt(m[1], 10) : 1;
    const unit = m[2].toLowerCase().replace(/s$/, "");
    return n === 1 ? `Every ${unit}` : `Every ${n} ${unit}s`;
  }
  const cron = prompt.match(/cron\s*[:=]?\s*([-*/\d\s,]+)/i);
  if (cron) return `Cron: ${cron[1].trim()}`;
  return null;
}

/** Heuristic: does this prompt intend to CREATE a scheduled task? */
export function isSchedulingPrompt(prompt: string): boolean {
  if (!prompt) return false;
  return (
    /\bschedule\b/i.test(prompt) ||
    /\brecurring\b/i.test(prompt) ||
    /\bevery\s+(\d+\s+)?(minute|hour|day|week|month)/i.test(prompt) ||
    /\bcron\b/i.test(prompt)
  );
}

export type ChatIntent =
  | { kind: "create" }
  | { kind: "list" }
  | { kind: "status"; task_id: string }
  | { kind: "info"; task_id: string }
  | { kind: "cancel"; task_id: string }
  | { kind: "other" };

const TASK_ID_RE = /([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|task_[A-Za-z0-9_-]+|[A-Za-z0-9]{6,})/;

export function classifyIntent(raw: string): ChatIntent {
  const p = raw.trim();
  const lower = p.toLowerCase();
  if (/\b(list|show|what\s+is\s+(?:a\s+)?list)\b.*\btask/.test(lower)) {
    return { kind: "list" };
  }
  if (/\bcancel\b.*\btask\b/.test(lower)) {
    const m = p.match(TASK_ID_RE);
    if (m) return { kind: "cancel", task_id: m[1] };
  }
  if (/\bstatus\b.*\btask\b/.test(lower) || /status\s+of\s+task/i.test(p)) {
    const m = p.match(TASK_ID_RE);
    if (m) return { kind: "status", task_id: m[1] };
  }
  if (/\b(info|information|details)\b.*\btask\b/.test(lower)) {
    const m = p.match(TASK_ID_RE);
    if (m) return { kind: "info", task_id: m[1] };
  }
  if (isSchedulingPrompt(p)) return { kind: "create" };
  return { kind: "other" };
}

/** Return `{ threadId, sessionId }` from the current analysis session in
 *  sessionStorage, or null if the user has not started one. */
export function getActiveSession(): { threadId: string; sessionId: string } | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem("analysis:session");
    if (!raw) return null;
    const s = JSON.parse(raw);
    if (s?.threadId && s?.sessionId) return { threadId: s.threadId, sessionId: s.sessionId };
  } catch {}
  return null;
}

export async function currentUserLabel(): Promise<string> {
  try {
    const data = { user: await getCachedAuthUser() };
    return data.user?.email ?? "You";
  } catch {
    return "You";
  }
}

/** Send a chat prompt SCOPED to the scheduled-analysis phase.
 *  Tags the message so the backend can route it. Extracts `task_info`
 *  from the response and records a new ScheduledTask when present. */
export async function sendScheduledChat(input: {
  prompt: string;
  datasetIds?: string[];
}): Promise<{
  reply: string;
  createdTask?: ScheduledTask;
  raw: any;
}> {
  const session = getActiveSession();
  if (!session) {
    throw new Error(
      "No active analysis session. Open Analysis and upload a dataset first, then schedule from here.",
    );
  }

  const res: any = await backendApi.sendMessage(
    session.threadId,
    session.sessionId,
    input.prompt,
    input.datasetIds && input.datasetIds.length ? input.datasetIds : undefined,
  );

  const assistantMsgs = (res.messages ?? [])
    .filter((m: any) => m && m.role !== "user" && typeof m.content === "string")
    .map((m: any) => m.content as string);
  const reply = assistantMsgs.join("\n\n") || "Done.";

  let createdTask: ScheduledTask | undefined;
  const taskInfo = res.task_info as { task_id?: string; next_due_at?: number } | undefined;
  if (taskInfo?.task_id) {
    const schedule = extractSchedulePhrase(input.prompt) ?? "Scheduled";
    const nextRun = taskInfo.next_due_at
      ? new Date(taskInfo.next_due_at * 1000).toISOString()
      : null;
    const owner = await currentUserLabel();
    createdTask = {
      task_id: taskInfo.task_id,
      name: summarizePrompt(input.prompt),
      prompt: input.prompt,
      schedule,
      status: "Active",
      next_run: nextRun,
      last_run: null,
      run_count: 0,
      created_at: new Date().toISOString(),
      owner,
      latest_output: Array.isArray(res.output_json) ? res.output_json : null,
      output_filename:
        res.output_file_data?.filename ?? null,
    };
    scheduledTaskStore.upsert(createdTask);
    try {
      const { scheduledPromptCache } = await import("@/lib/api/scheduled-prompt-cache");
      scheduledPromptCache.set(taskInfo.task_id, input.prompt);
    } catch {}
  }

  return { reply, createdTask, raw: res };
}

function summarizePrompt(prompt: string): string {
  const first = prompt.split(/[.!?\n]/)[0]?.trim() ?? prompt.trim();
  if (first.length <= 60) return first;
  return first.slice(0, 57) + "…";
}
