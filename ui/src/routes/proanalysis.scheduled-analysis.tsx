import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Calendar,
  SearchLg,
  Grid01,
  Rows01,
  Plus,
  DotsVertical,
  Send01,
} from "@untitledui/icons";
import { DashboardShell } from "@/components/dashboard/DashboardShell";
import type { AnalysisItem } from "@/components/dashboard/AnalysisSidebar";
import { Input } from "@/components/base/input/input";
import { Button } from "@/components/base/buttons/button";
import { ButtonUtility } from "@/components/base/buttons/button-utility";
import { Avatar } from "@/components/base/avatar/avatar";
import { cx } from "@/lib/utils/cx";
import { AnalysisOutputTable } from "@/components/dashboard/AnalysisOutputTable";
import {
  scheduledTaskStore,
  sendScheduledChat,
  classifyIntent,
  type ScheduledTask,
  type ScheduledTaskStatus,
} from "@/lib/api/scheduled-tasks";
import { Dialog, DialogContent } from "@/components/ui/dialog";

export const Route = createFileRoute("/proanalysis/scheduled-analysis")({
  head: () => ({
    meta: [
      { title: "Scheduled Analysis · Avaloka AI" },
      { name: "description", content: "Create and manage recurring scheduled analyses." },
    ],
  }),
  component: ScheduledAnalysisPage,
});

const DUMMY_ANALYSES: AnalysisItem[] = [
  { id: "a1", label: "Analysis I" },
  { id: "a2", label: "Analysis II" },
  { id: "a3", label: "Analysis III" },
  { id: "a4", label: "Analysis IV" },
];

const STATUS_STYLES: Record<ScheduledTaskStatus, string> = {
  Active: "bg-blue-50 text-blue-700 ring-blue-200",
  Running: "bg-orange-50 text-orange-700 ring-orange-200",
  Paused: "bg-yellow-50 text-yellow-700 ring-yellow-200",
  Cancelled: "bg-gray-50 text-gray-700 ring-gray-200",
  Failed: "bg-red-50 text-red-700 ring-red-200",
};

function StatusBadge({ status }: { status: ScheduledTaskStatus }) {
  return (
    <span
      className={cx(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset",
        STATUS_STYLES[status],
      )}
    >
      {status}
    </span>
  );
}

function CsvIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden>
      <path
        d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z"
        stroke="#079455"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path d="M14 2v6h6" stroke="#079455" strokeWidth="1.6" strokeLinejoin="round" />
      <rect x="6.5" y="12.5" width="11" height="6" rx="1" fill="#079455" />
      <text x="12" y="17.1" textAnchor="middle" fontSize="3.6" fontWeight="700" fill="#fff" fontFamily="Inter, sans-serif">
        CSV
      </text>
    </svg>
  );
}

function formatDate(iso?: string | null) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return `${d.toLocaleDateString([], { day: "2-digit", month: "short", year: "numeric" })} · ${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  } catch {
    return iso;
  }
}

type ChatEntry = {
  id: string;
  role: "user" | "assistant";
  text: string;
  time: string;
};

function ScheduledAnalysisPage() {
  const [view, setView] = useState<"grid" | "table">("grid");
  const [tasks, setTasks] = useState<ScheduledTask[]>(() => scheduledTaskStore.list());
  const [chat, setChat] = useState<ChatEntry[]>([
    {
      id: "sys-1",
      role: "assistant",
      text:
        "Hi! Describe an analysis and add a schedule (e.g. \"Group by Region then sum Revenue. Schedule this to run every minute.\"). You can also ask me to list your tasks, check status, get info, or cancel a task by id.",
      time: nowTime(),
    },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [search, setSearch] = useState("");
  const [openTaskId, setOpenTaskId] = useState<string | null>(null);
  const [detailTaskId, setDetailTaskId] = useState<string | null>(null);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    return scheduledTaskStore.subscribe(() => setTasks(scheduledTaskStore.list()));
  }, []);

  useEffect(() => {
    const el = chatScrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [chat]);

  const filteredTasks = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return tasks;
    return tasks.filter(
      (t) =>
        t.name.toLowerCase().includes(q) ||
        t.task_id.toLowerCase().includes(q) ||
        t.prompt.toLowerCase().includes(q),
    );
  }, [tasks, search]);

  const openTask = openTaskId ? tasks.find((t) => t.task_id === openTaskId) ?? null : null;
  const detailTask = detailTaskId ? tasks.find((t) => t.task_id === detailTaskId) ?? null : null;

  const pushChat = useCallback((role: ChatEntry["role"], text: string) => {
    setChat((c) => [
      ...c,
      { id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`, role, text, time: nowTime() },
    ]);
  }, []);

  const handleSend = useCallback(async () => {
    const prompt = input.trim();
    if (!prompt || sending) return;
    setInput("");
    pushChat("user", prompt);
    setSending(true);

    const intent = classifyIntent(prompt);
    try {
      // Local-only optimistic handling for cancel (also fires backend call).
      if (intent.kind === "cancel") {
        scheduledTaskStore.patch(intent.task_id, { status: "Cancelled" });
      }

      const { reply, createdTask } = await sendScheduledChat({ prompt });

      if (createdTask) {
        pushChat(
          "assistant",
          `${reply}\n\nScheduled task created — id: \`${createdTask.task_id}\` · ${createdTask.schedule}.`,
        );
      } else if (intent.kind === "list") {
        pushChat(
          "assistant",
          `${reply}\n\n(${tasks.length} task${tasks.length === 1 ? "" : "s"} currently shown in the grid.)`,
        );
      } else {
        pushChat("assistant", reply);
      }
    } catch (err: any) {
      pushChat("assistant", `⚠️ ${err?.message ?? "Something went wrong."}`);
    } finally {
      setSending(false);
    }
  }, [input, sending, pushChat, tasks.length]);

  return (
    <DashboardShell analyses={DUMMY_ANALYSES}>
      <div className="flex min-h-full flex-1 flex-col gap-4 p-6">
        <div className="flex flex-1 flex-col rounded-2xl border border-secondary bg-primary shadow-xs">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-secondary px-6 py-4">
            <div className="flex items-center gap-2">
              <Calendar className="size-5 text-fg-secondary" />
              <h1 className="text-lg font-semibold text-primary">Scheduled Analysis</h1>
              <span className="ml-2 rounded-full bg-secondary px-2 py-0.5 text-xs text-tertiary">
                {tasks.length} task{tasks.length === 1 ? "" : "s"}
              </span>
            </div>
            <Button
              color="primary"
              size="md"
              iconLeading={Plus}
              onClick={() => {
                const el = document.getElementById("scheduled-chat-input") as HTMLInputElement | null;
                el?.focus();
              }}
            >
              New Scheduled Task
            </Button>
          </div>

          {/* Toolbar */}
          <div className="flex flex-wrap items-center justify-between gap-3 px-6 py-4">
            <div className="flex flex-1 items-center gap-3">
              <div className="w-full max-w-[360px]">
                <Input
                  aria-label="Search scheduled tasks"
                  placeholder="Search by name, prompt or task id"
                  size="md"
                  icon={SearchLg}
                  value={search}
                  onChange={(v) => setSearch(typeof v === "string" ? v : (v as any)?.target?.value ?? "")}
                />
              </div>
            </div>
            <div className="inline-flex items-center rounded-lg border border-secondary bg-primary p-0.5 shadow-xs">
              <button
                type="button"
                onClick={() => setView("grid")}
                className={cx(
                  "inline-flex h-8 items-center gap-1.5 rounded-md px-3 text-sm font-semibold transition-colors",
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
                  "inline-flex h-8 items-center gap-1.5 rounded-md px-3 text-sm font-semibold transition-colors",
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
            {filteredTasks.length === 0 ? (
              <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-secondary bg-primary/50 px-6 py-16 text-center">
                <Calendar className="size-8 text-tertiary" />
                <p className="text-sm font-semibold text-primary">No scheduled tasks yet</p>
                <p className="max-w-md text-xs text-tertiary">
                  Use the chat below to create one. Example: <em>&ldquo;Group by Region then sum
                  Revenue. Schedule this to run every minute.&rdquo;</em>
                </p>
              </div>
            ) : view === "grid" ? (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                {filteredTasks.map((item) => (
                  <article
                    key={item.task_id}
                    className="flex flex-col rounded-xl border border-secondary bg-primary shadow-xs transition hover:border-brand"
                  >
                    <div className="flex items-start justify-between p-4">
                      <CsvIcon className="size-8" />
                      <div className="flex items-center gap-2">
                        <StatusBadge status={item.status} />
                        <ButtonUtility
                          size="xs"
                          color="tertiary"
                          icon={DotsVertical}
                          tooltip="Details"
                          onClick={() => setDetailTaskId(item.task_id)}
                        />
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => setOpenTaskId(item.task_id)}
                      className="flex-1 px-4 pb-4 text-left"
                    >
                      <div className="text-md font-semibold text-primary hover:underline">
                        {item.name}
                      </div>
                      <p className="mt-1 text-sm text-tertiary">
                        Scheduled: {item.schedule}
                      </p>
                      <p className="mt-1 truncate text-xs text-tertiary">
                        task_id: <span className="font-mono">{item.task_id}</span>
                      </p>
                    </button>
                    <div className="flex items-center justify-between gap-3 border-t border-secondary px-4 py-3">
                      <div className="flex min-w-0 items-center gap-2">
                        <Avatar size="sm" alt={item.owner ?? "You"} />
                        <span className="truncate text-sm font-semibold text-primary">
                          {item.owner ?? "You"}
                        </span>
                      </div>
                      <span className="shrink-0 text-xs text-tertiary">
                        {formatDate(item.created_at)}
                      </span>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <div className="overflow-hidden rounded-xl border border-secondary">
                <table className="w-full text-left text-sm">
                  <thead className="bg-secondary/40">
                    <tr className="text-xs font-semibold text-tertiary">
                      <th className="px-4 py-3">Name</th>
                      <th className="px-4 py-3">Task ID</th>
                      <th className="px-4 py-3">Schedule</th>
                      <th className="px-4 py-3">Status</th>
                      <th className="px-4 py-3">Next Run</th>
                      <th className="px-4 py-3">Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredTasks.map((item) => (
                      <tr
                        key={item.task_id}
                        className="cursor-pointer border-t border-secondary hover:bg-primary_hover"
                        onClick={() => setOpenTaskId(item.task_id)}
                      >
                        <td className="px-4 py-3 text-sm font-medium text-primary">{item.name}</td>
                        <td className="px-4 py-3 font-mono text-xs text-tertiary">
                          {item.task_id}
                        </td>
                        <td className="px-4 py-3 text-sm text-secondary">{item.schedule}</td>
                        <td className="px-4 py-3">
                          <StatusBadge status={item.status} />
                        </td>
                        <td className="px-4 py-3 text-sm text-tertiary">
                          {formatDate(item.next_run)}
                        </td>
                        <td className="px-4 py-3 text-sm text-tertiary">
                          {formatDate(item.created_at)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Chat panel (scoped to scheduled context) */}
        <div className="flex flex-col rounded-2xl border border-secondary bg-primary shadow-xs">
          <div className="flex items-center gap-2 border-b border-secondary px-4 py-3">
            <span className="grid size-7 place-items-center rounded-full bg-gradient-to-br from-[#7f56d9] to-[#1565ef] text-xs font-semibold text-white">
              AI
            </span>
            <div>
              <p className="text-sm font-semibold text-primary">Scheduled Analysis Assistant</p>
              <p className="text-xs text-tertiary">
                Create, list, inspect, or cancel scheduled tasks with natural language.
              </p>
            </div>
          </div>
          <div ref={chatScrollRef} className="max-h-72 flex-1 overflow-y-auto px-4 py-3">
            <ul className="flex flex-col gap-3">
              {chat.map((m) => (
                <li
                  key={m.id}
                  className={cx("flex", m.role === "user" ? "justify-end" : "justify-start")}
                >
                  <div
                    className={cx(
                      "max-w-[80%] whitespace-pre-wrap rounded-2xl px-3 py-2 text-sm",
                      m.role === "user"
                        ? "bg-[#1565ef] text-white"
                        : "bg-secondary text-primary",
                    )}
                  >
                    {m.text}
                    <div
                      className={cx(
                        "mt-1 text-[10px]",
                        m.role === "user" ? "text-white/70" : "text-tertiary",
                      )}
                    >
                      {m.time}
                    </div>
                  </div>
                </li>
              ))}
              {sending && (
                <li className="flex justify-start">
                  <div className="rounded-2xl bg-secondary px-3 py-2 text-sm text-tertiary">
                    Thinking…
                  </div>
                </li>
              )}
            </ul>
          </div>
          <form
            className="flex items-center gap-2 border-t border-secondary px-4 py-3"
            onSubmit={(e) => {
              e.preventDefault();
              void handleSend();
            }}
          >
            <input
              id="scheduled-chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Describe an analysis + schedule, or say “list my tasks”…"
              className="flex-1 rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-primary outline-none focus:border-brand"
              disabled={sending}
            />
            <Button
              type="submit"
              color="primary"
              size="md"
              iconLeading={Send01}
              isDisabled={sending || !input.trim()}
            >
              Send
            </Button>
          </form>
        </div>
      </div>

      {/* Output view modal */}
      <Dialog open={!!openTask} onOpenChange={(o) => !o && setOpenTaskId(null)}>
        <DialogContent className="max-w-4xl">
          {openTask && (
            <div className="flex flex-col gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-semibold text-primary">{openTask.name}</h2>
                  <StatusBadge status={openTask.status} />
                </div>
                <p className="mt-1 text-xs text-tertiary">
                  task_id: <span className="font-mono">{openTask.task_id}</span> · {openTask.schedule}
                </p>
              </div>
              <AnalysisOutputTable
                rows={openTask.latest_output ?? null}
                title="Latest Output"
                filename={openTask.output_filename ?? `${openTask.name}.csv`}
                emptyMessage="No output captured yet for this scheduled task. It will appear here after the next run."
              />
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Detail modal */}
      <Dialog open={!!detailTask} onOpenChange={(o) => !o && setDetailTaskId(null)}>
        <DialogContent className="max-w-lg">
          {detailTask && (
            <div className="flex flex-col gap-3">
              <div>
                <h2 className="text-lg font-semibold text-primary">{detailTask.name}</h2>
                <p className="mt-1 text-xs text-tertiary font-mono">{detailTask.task_id}</p>
              </div>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                <dt className="text-tertiary">Status</dt>
                <dd><StatusBadge status={detailTask.status} /></dd>
                <dt className="text-tertiary">Schedule</dt>
                <dd className="text-primary">{detailTask.schedule}</dd>
                <dt className="text-tertiary">Next run</dt>
                <dd className="text-primary">{formatDate(detailTask.next_run)}</dd>
                <dt className="text-tertiary">Last run</dt>
                <dd className="text-primary">{formatDate(detailTask.last_run)}</dd>
                <dt className="text-tertiary">Run count</dt>
                <dd className="text-primary">{detailTask.run_count ?? 0}</dd>
                <dt className="text-tertiary">Created</dt>
                <dd className="text-primary">{formatDate(detailTask.created_at)}</dd>
              </dl>
              <div>
                <p className="text-xs font-semibold text-tertiary">Prompt</p>
                <p className="mt-1 whitespace-pre-wrap rounded-lg bg-secondary p-3 text-sm text-primary">
                  {detailTask.prompt}
                </p>
              </div>
              <div className="flex justify-end gap-2">
                <Button
                  color="secondary"
                  size="sm"
                  onClick={() => {
                    setDetailTaskId(null);
                    setOpenTaskId(detailTask.task_id);
                  }}
                >
                  View latest output
                </Button>
                <Button
                  color="primary-destructive"
                  size="sm"
                  onClick={() => {
                    scheduledTaskStore.patch(detailTask.task_id, { status: "Cancelled" });
                    setInput(`Cancel task ${detailTask.task_id}`);
                    setDetailTaskId(null);
                  }}
                >
                  Cancel task
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </DashboardShell>
  );
}

function nowTime() {
  return new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}
