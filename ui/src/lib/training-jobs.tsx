/**
 * Global, navigation-proof polling for long-running ("deferred") training turns.
 *
 * A job is keyed by `threadId::deferredId` and lives in a module-level
 * singleton, so switching threads / routes never kills an in-flight poll.
 * Exactly one interval exists per key.
 */

import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { useNavigate } from "@tanstack/react-router";
import { toast } from "sonner";
import { backendApi, type PendingTurnResponse } from "@/lib/api/backendApi";

export type TrainingJobStatus = "running" | "done" | "error" | "superseded";

export type TrainingJob = {
  key: string;
  threadId: string;
  deferredId: string | null;
  sessionId: string;
  analysisId: string | null;
  label: string;
  status: TrainingJobStatus;
  message?: string;
  startedAt: number;
};

type InternalJob = TrainingJob & {
  timer: ReturnType<typeof setInterval> | null;
  resolve: (v: PendingTurnResponse) => void;
  promise: Promise<PendingTurnResponse>;
  /** Result waiting to be rendered by whichever view claims it first. */
  unclaimedResult: any | null;
  errors: number;
  polls: number;
};

type CompleteEvent = { job: TrainingJob; result?: any };

const POLL_MS = 3_000;
const MAX_POLLS = 240; // ~12 minutes

const jobs = new Map<string, InternalJob>();
const listeners = new Set<() => void>();
const completeListeners = new Set<(e: CompleteEvent) => void>();

function emit() {
  listeners.forEach((l) => l());
}

function publicJobs(): TrainingJob[] {
  return [...jobs.values()].map(({ timer, resolve, promise, unclaimedResult, errors, polls, ...j }) => j);
}

function stopTimer(job: InternalJob) {
  if (job.timer) clearInterval(job.timer);
  job.timer = null;
}

function finish(job: InternalJob, status: TrainingJobStatus, payload: PendingTurnResponse) {
  stopTimer(job);
  job.status = status;
  job.message = payload.message;
  if (status === "done") job.unclaimedResult = payload.result ?? null;
  job.resolve(payload);
  emit();
  completeListeners.forEach((l) => l({ job: { ...job }, result: payload.result }));
  // Keep "done" jobs around briefly so a late-mounting view can claim the
  // result; running jobs are removed as soon as they resolve.
  setTimeout(() => {
    if (jobs.get(job.key) === job && !job.unclaimedResult) {
      jobs.delete(job.key);
      emit();
    }
  }, status === "done" ? 5 * 60_000 : 5_000);
}

export function ensureNotificationPermission() {
  if (typeof window === "undefined" || !("Notification" in window)) return;
  try {
    if (Notification.permission === "default") void Notification.requestPermission();
  } catch {
    /* ignore */
  }
}

export type StartTrainingJobInput = {
  threadId: string;
  deferredId: string | null;
  sessionId: string;
  analysisId?: string | null;
  label?: string;
};

/**
 * Start (or join) the poll for a deferred turn. Resolves when the turn
 * reaches a terminal state. Safe to call twice — never double-polls.
 */
export function startTrainingJob(input: StartTrainingJobInput): Promise<PendingTurnResponse> {
  const key = `${input.threadId}::${input.deferredId ?? "latest"}`;
  const existing = jobs.get(key);
  if (existing) return existing.promise;

  // A newer training turn on the same thread supersedes older ones.
  for (const [k, j] of jobs) {
    if (j.threadId === input.threadId && j.status === "running" && k !== key) {
      finish(j, "superseded", { status: "superseded" });
    }
  }

  let resolve!: (v: PendingTurnResponse) => void;
  const promise = new Promise<PendingTurnResponse>((r) => (resolve = r));

  const job: InternalJob = {
    key,
    threadId: input.threadId,
    deferredId: input.deferredId ?? null,
    sessionId: input.sessionId,
    analysisId: input.analysisId ?? null,
    label: input.label || "Training model",
    status: "running",
    startedAt: Date.now(),
    timer: null,
    resolve,
    promise,
    unclaimedResult: null,
    errors: 0,
    polls: 0,
  };
  jobs.set(key, job);
  emit();
  ensureNotificationPermission();

  const tick = async () => {
    if (job.status !== "running") return;
    job.polls += 1;
    if (job.polls > MAX_POLLS) {
      finish(job, "error", {
        status: "error",
        message: "Training is taking longer than expected. It may still finish in the background.",
      });
      return;
    }
    try {
      const data = await backendApi.getPendingTurn(job.threadId, job.sessionId, job.deferredId);
      job.errors = 0;
      const status = String(data?.status ?? "").toLowerCase();
      if (status === "done") finish(job, "done", { ...data, status: "done" });
      else if (status === "error") finish(job, "error", { ...data, status: "error" });
      else if (status === "superseded") finish(job, "superseded", { ...data, status: "superseded" });
      // "running" / "none" / "" -> keep polling
    } catch (err: any) {
      job.errors += 1;
      if (job.errors >= 5) {
        finish(job, "error", {
          status: "error",
          message: err?.message || "Lost connection while waiting for the result.",
        });
      }
    }
  };

  job.timer = setInterval(() => void tick(), POLL_MS);
  return promise;
}

/** Claim a finished result for a thread exactly once. */
export function claimTrainingResult(threadId: string): { deferredId: string | null; result: any } | null {
  for (const job of jobs.values()) {
    if (job.threadId === threadId && job.unclaimedResult) {
      const result = job.unclaimedResult;
      job.unclaimedResult = null;
      jobs.delete(job.key);
      emit();
      return { deferredId: job.deferredId, result };
    }
  }
  return null;
}

export function stopThreadTrainingJobs(threadId: string) {
  for (const job of jobs.values()) {
    if (job.threadId === threadId && job.status === "running") {
      finish(job, "superseded", { status: "superseded" });
    }
  }
}

export function stopAllTrainingJobs() {
  for (const job of jobs.values()) stopTimer(job);
  jobs.clear();
  emit();
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

const TrainingJobsContext = createContext<TrainingJob[]>([]);

export function useTrainingJobs() {
  return useContext(TrainingJobsContext);
}

export function useThreadTrainingJob(threadId?: string | null) {
  const all = useTrainingJobs();
  if (!threadId) return null;
  return all.find((j) => j.threadId === threadId && j.status === "running") ?? null;
}

/** Event fired when a background training result lands while a view is mounted. */
export const TRAINING_RESULT_EVENT = "avaloka:training-result";

export function TrainingJobsProvider({ children }: { children: ReactNode }) {
  const [list, setList] = useState<TrainingJob[]>([]);
  const navigate = useNavigate();
  const navigateRef = useRef(navigate);
  navigateRef.current = navigate;

  useEffect(() => {
    const unsub = subscribe(() => setList(publicJobs()));
    return () => {
      unsub();
    };
  }, []);

  const goToThread = useCallback((job: TrainingJob) => {
    navigateRef.current({
      to: "/analysis",
      search: (job.analysisId ? { aid: job.analysisId } : {}) as never,
    });
    // Let the analysis view pick the result up once it is mounted.
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent(TRAINING_RESULT_EVENT, { detail: { threadId: job.threadId } }));
    }, 400);
  }, []);

  useEffect(() => {
    const onComplete = ({ job, result }: CompleteEvent) => {
      if (job.status === "done") {
        toast.success("Model training finished", {
          description: job.label,
          duration: 10_000,
          action: { label: "View results", onClick: () => goToThread(job) },
        });
        try {
          if (typeof Notification !== "undefined" && Notification.permission === "granted") {
            const n = new Notification("Model training finished", {
              body: "Your training run completed. Click to view the results.",
              tag: job.key,
            });
            n.onclick = () => {
              window.focus();
              goToThread(job);
              n.close();
            };
          }
        } catch {
          /* notifications unavailable */
        }
        // Nudge a mounted analysis view for the same thread.
        window.dispatchEvent(
          new CustomEvent(TRAINING_RESULT_EVENT, { detail: { threadId: job.threadId, result } }),
        );
      } else if (job.status === "error") {
        toast.error(job.message || "Training failed. Please try again.");
      }
    };
    completeListeners.add(onComplete);
    return () => {
      completeListeners.delete(onComplete);
    };
  }, [goToThread]);

  useEffect(() => () => stopAllTrainingJobs(), []);

  const running = list.filter((j) => j.status === "running");

  return (
    <TrainingJobsContext.Provider value={list}>
      {children}
      {running.length > 0 && (
        <div className="pointer-events-none fixed bottom-4 left-1/2 z-[60] -translate-x-1/2">
          <button
            type="button"
            onClick={() => goToThread(running[0])}
            className="pointer-events-auto flex items-center gap-2 rounded-full border border-border bg-background/95 px-3 py-1.5 text-xs font-medium text-foreground shadow-lg backdrop-blur"
          >
            <span className="size-3 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            Training model…
            {running.length > 1 && <span className="text-muted-foreground">({running.length})</span>}
          </button>
        </div>
      )}
    </TrainingJobsContext.Provider>
  );
}
